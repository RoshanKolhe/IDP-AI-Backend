"""
Shared OCR enhancement and cache helpers for document DAGs.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from .ocr_service_factory import get_ocr_service


OCR_OUTPUT_DIRNAME = "ocr_output"


def get_process_instance_dir(local_download_dir: str, process_instance_id: int) -> str:
    return os.path.join(local_download_dir, f"process-instance-{process_instance_id}")


def get_ocr_output_dir(process_instance_dir: str) -> str:
    output_dir = os.path.join(process_instance_dir, OCR_OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_ocr_cache_path(process_instance_dir: str, pdf_filename: str) -> str:
    base_filename = os.path.splitext(os.path.basename(pdf_filename))[0]
    return os.path.join(get_ocr_output_dir(process_instance_dir), f"{base_filename}.json")


def get_ocr_pages_dir(process_instance_dir: str, pdf_filename: str) -> str:
    base_filename = os.path.splitext(os.path.basename(pdf_filename))[0]
    pages_dir = os.path.join(get_ocr_output_dir(process_instance_dir), base_filename)
    os.makedirs(pages_dir, exist_ok=True)
    return pages_dir


def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    rgb_image = image.convert("RGB")
    image_array = np.array(rgb_image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Upscale before denoising for damaged/low-resolution scans.
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(upscaled, None, 15, 7, 21)
    contrasted = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)
    sharpened = cv2.GaussianBlur(contrasted, (0, 0), 3)
    sharpened = cv2.addWeighted(contrasted, 1.7, sharpened, -0.7, 0)
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return Image.fromarray(adaptive)


def build_ocr_cache_payload(
    pdf_filename: str,
    ocr_engine: str,
    page_results: List[Dict],
    cleaned_text: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Dict:
    raw_text = "\n".join(page["text"] for page in page_results if page.get("text", "").strip()).strip()
    return {
        "filename": pdf_filename,
        "ocr_engine": ocr_engine,
        "config": config or {},
        "page_count": len(page_results),
        "pages": page_results,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text if cleaned_text and cleaned_text.strip() else raw_text,
        "processed_at": datetime.utcnow().isoformat(),
    }


def save_ocr_cache(cache_path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_page_outputs(process_instance_dir: str, pdf_filename: str, payload: Dict) -> None:
    pages_dir = get_ocr_pages_dir(process_instance_dir, pdf_filename)

    summary_path = os.path.join(pages_dir, "document.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    for page in payload.get("pages", []):
        page_number = int(page.get("page_number", 0))
        page_base_name = f"page_{page_number:03d}"

        with open(os.path.join(pages_dir, f"{page_base_name}.txt"), "w", encoding="utf-8") as file:
            file.write(page.get("text", "") or "")

        with open(os.path.join(pages_dir, f"{page_base_name}.cleaned.txt"), "w", encoding="utf-8") as file:
            file.write(page.get("cleaned_text", "") or "")

        with open(os.path.join(pages_dir, f"{page_base_name}.json"), "w", encoding="utf-8") as file:
            json.dump(page, file, indent=2, ensure_ascii=False)


def load_ocr_cache(cache_path: str) -> Optional[Dict]:
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_cached_document_text(process_instance_dir: str, pdf_filename: str, prefer_cleaned: bool = True) -> Optional[str]:
    cache = load_ocr_cache(get_ocr_cache_path(process_instance_dir, pdf_filename))
    if not cache:
        return None

    if prefer_cleaned and cache.get("cleaned_text", "").strip():
        return cache["cleaned_text"]
    return cache.get("raw_text")


def get_cached_page_texts(process_instance_dir: str, pdf_filename: str, prefer_cleaned: bool = True) -> List[str]:
    cache = load_ocr_cache(get_ocr_cache_path(process_instance_dir, pdf_filename))
    if not cache:
        return []

    key = "cleaned_text" if prefer_cleaned else "text"
    return [page.get(key) or page.get("text", "") for page in cache.get("pages", [])]


def _resolve_page_text(page_result: Dict) -> str:
    return (page_result.get("cleaned_text") or page_result.get("text") or "").strip()


def ensure_ocr_cache(
    pdf_path: str,
    process_instance_dir: str,
    ocr_engine: str = "paddle",
    config: Optional[Dict] = None,
    cleanup_service=None,
    force_refresh: bool = False,
    logger_callback=None,
    fallback_ocr_engine: str = "tesseract",
) -> Dict:
    pdf_filename = os.path.basename(pdf_path)
    cache_path = get_ocr_cache_path(process_instance_dir, pdf_filename)

    if not force_refresh:
        existing = load_ocr_cache(cache_path)
        if existing:
            save_page_outputs(process_instance_dir, pdf_filename, existing)
            if logger_callback:
                logger_callback(
                    "info",
                    f"Using cached OCR for {pdf_filename}: pages={existing.get('page_count', 0)}",
                )
            return existing

    ocr_service = get_ocr_service(ocr_engine)
    fallback_service = None
    if fallback_ocr_engine and fallback_ocr_engine != ocr_engine:
        fallback_service = get_ocr_service(fallback_ocr_engine)
    config = config or {}
    convert_kwargs = {"dpi": config.get("dpi", 300)}
    if config.get("first_page") is not None:
        convert_kwargs["first_page"] = config["first_page"]
    if config.get("last_page") is not None:
        convert_kwargs["last_page"] = config["last_page"]
    if config.get("thread_count") is not None:
        convert_kwargs["thread_count"] = config["thread_count"]

    images = convert_from_path(pdf_path, **convert_kwargs)
    page_results = []

    for page_number, image in enumerate(images, start=1):
        enhanced_image = enhance_image_for_ocr(image)
        temp_image_path = os.path.join(
            get_ocr_output_dir(process_instance_dir),
            f".tmp_{os.path.splitext(pdf_filename)[0]}_{page_number}.png",
        )

        try:
            if logger_callback:
                logger_callback(
                    "info",
                    f"Scanning {pdf_filename} page {page_number}/{len(images)} with {ocr_engine}",
                )

            enhanced_image.save(temp_image_path, format="PNG")
            page_conf = dict(config)
            page_result = ocr_service.extract_text_with_confidence(temp_image_path, page_conf)
            page_text = (page_result.get("text") or "").strip()

            if not page_text and fallback_service:
                fallback_result = fallback_service.extract_text_with_confidence(temp_image_path, page_conf)
                fallback_text = (fallback_result.get("text") or "").strip()
                if fallback_text:
                    page_result = fallback_result
                    page_text = fallback_text
                    if logger_callback:
                        logger_callback(
                            "warning",
                            f"{ocr_engine} returned empty OCR for {pdf_filename} page {page_number}; used {fallback_ocr_engine} fallback",
                        )

            cleaned_page_text = page_text
            if cleanup_service and page_text:
                try:
                    cleaned_page_text = cleanup_service.cleanup_text(page_text)
                except Exception:
                    cleaned_page_text = page_text

            page_results.append(
                {
                    "page_number": page_number,
                    "text": page_text,
                    "cleaned_text": cleaned_page_text,
                    "confidence": page_result.get("confidence", 0.0),
                    "character_count": len(page_text),
                }
            )

            if logger_callback:
                logger_callback(
                    "success" if page_text else "warning",
                    f"OCR finished for {pdf_filename} page {page_number}: chars={len(page_text)}, confidence={round(float(page_result.get('confidence', 0.0)), 2)}",
                )
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    cleaned_text = "\n".join(
        _resolve_page_text(page_result) for page_result in page_results if _resolve_page_text(page_result)
    ).strip()
    payload = build_ocr_cache_payload(
        pdf_filename=pdf_filename,
        ocr_engine=ocr_engine,
        page_results=page_results,
        cleaned_text=cleaned_text,
        config=config,
    )
    save_ocr_cache(cache_path, payload)
    save_page_outputs(process_instance_dir, pdf_filename, payload)
    return payload
