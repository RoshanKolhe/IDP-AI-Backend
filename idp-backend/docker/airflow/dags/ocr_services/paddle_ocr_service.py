"""
Enterprise PaddleOCR Service (CPU Optimized)
Designed for:
- Bad quality scans
- Government docs
- Agreements
- Large PDFs (100+ pages)
- 8 Core CPU / 32GB RAM
"""

from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import gc
import os

import numpy as np
import cv2

from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader

from .base_ocr_service import BaseOCRService


class PaddleOCRService(BaseOCRService):

    SUPPORTED_LANGUAGES = [
        "en",
        "latin",
        "arabic",
        "cyrillic",
        "devanagari",
        "ch",
    ]

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            self._engine_cls = PaddleOCR
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR import failed: {exc}")

        self._ocr_engine = None
        self._engine_config = None

    # ======================================================
    # Language
    # ======================================================

    def _normalize_language(self, language_mode: Optional[str]) -> str:
        if not language_mode or language_mode == "auto":
            return "en"

        mapping = {
            "eng": "en",
            "eng+hin": "en",
            "hin": "devanagari",
        }

        return mapping.get(language_mode.lower().strip(), "en")

    # ======================================================
    # Engine
    # ======================================================

    def _get_engine(self, config=None):
        config = config or {}

        lang = self._normalize_language(config.get("language_mode"))

        engine_config = {
            "lang": lang,
            "use_angle_cls": True,
            "show_log": False,
            "use_gpu": False,
            "cpu_threads": 4,
        }

        if self._ocr_engine is None or self._engine_config != engine_config:
            self._ocr_engine = self._engine_cls(**engine_config)
            self._engine_config = engine_config

        return self._ocr_engine

    # ======================================================
    # Quality Detection
    # ======================================================

    def _is_low_quality(self, gray):
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean = np.mean(gray)

        if blur < 60 or mean < 100:
            return True

        return False

    # ======================================================
    # Deskew
    # ======================================================

    def _deskew(self, gray):
        coords = np.column_stack(np.where(gray < 200))
        if len(coords) == 0:
            return gray

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        h, w = gray.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            gray,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    # ======================================================
    # Preprocess
    # ======================================================

    def _preprocess_image(self, image):
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        low_quality = self._is_low_quality(gray)

        gray = cv2.fastNlMeansDenoising(gray)

        if low_quality:
            gray = cv2.resize(
                gray, None, fx=2, fy=2,
                interpolation=cv2.INTER_CUBIC
            )

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        # Deskew
        gray = self._deskew(gray)

        # Threshold
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )

        # Morph cleanup
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel
        )

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # ======================================================
    # Parse Result
    # ======================================================

    def _parse_result(self, result):
        if not result or not result[0]:
            return {
                "text": "",
                "confidence": 0.0
            }

        lines = []
        confidences = []

        for item in result[0]:
            text = item[1][0].strip()
            conf = float(item[1][1])

            if text:
                lines.append(text)
                confidences.append(conf)

        text = "\n".join(lines)

        avg_conf = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        return {
            "text": text,
            "confidence": avg_conf
        }

    # ======================================================
    # OCR Single Page
    # ======================================================

    def _extract_page_result(self, image, config=None):
        engine = self._get_engine(config)

        processed = self._preprocess_image(image)

        result = engine.ocr(processed)

        parsed = self._parse_result(result)

        # Retry if low confidence
        if parsed["confidence"] < 0.70:
            retry = engine.ocr(np.array(image.convert("RGB")))
            retry_parsed = self._parse_result(retry)

            if retry_parsed["confidence"] > parsed["confidence"]:
                parsed = retry_parsed

        return parsed

    # ======================================================
    # Process Single PDF Page
    # ======================================================

    def _process_pdf_page(self, image_path, page_no, dpi, config):
        try:
            images = convert_from_path(
                image_path,
                dpi=dpi,
                first_page=page_no,
                last_page=page_no
            )

            if not images:
                return page_no, ""

            image = images[0]

            result = self._extract_page_result(image, config)

            image.close()
            del image
            del images
            gc.collect()

            return page_no, result["text"]

        except Exception:
            return page_no, ""

    # ======================================================
    # Extract Text
    # ======================================================

    def extract_text(self, image_path, config=None):
        config = config or {}

        try:
            if image_path.lower().endswith(".pdf"):

                page_count = len(PdfReader(image_path).pages)

                dpi = config.get("dpi", 300)

                workers = config.get("workers", 4)

                results = {}

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = []

                    for page_no in range(1, page_count + 1):
                        futures.append(
                            executor.submit(
                                self._process_pdf_page,
                                image_path,
                                page_no,
                                dpi,
                                config
                            )
                        )

                    for future in as_completed(futures):
                        page_no, text = future.result()
                        results[page_no] = text

                final_text = []

                for page_no in sorted(results.keys()):
                    if results[page_no].strip():
                        final_text.append(results[page_no])

                return "\n\n".join(final_text)

            # Image OCR
            image = Image.open(image_path)
            result = self._extract_page_result(image, config)
            image.close()

            return result["text"]

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")
        

    def extract_text_with_confidence(self, image_path, config=None):
        config = config or {}

        try:
            # ==========================================
            # PDF OCR
            # ==========================================
            if image_path.lower().endswith(".pdf"):

                page_count = len(PdfReader(image_path).pages)

                dpi = config.get("dpi", 300)
                workers = config.get("workers", 4)

                results = {}

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = []

                    for page_no in range(1, page_count + 1):
                        futures.append(
                            executor.submit(
                                self._process_pdf_page_with_confidence,
                                image_path,
                                page_no,
                                dpi,
                                config
                            )
                        )

                    for future in as_completed(futures):
                        page_no, page_result = future.result()
                        results[page_no] = page_result

                pages = []
                texts = []
                confs = []

                for page_no in sorted(results.keys()):
                    page_data = results[page_no]

                    pages.append({
                        "page": page_no,
                        "text": page_data["text"],
                        "confidence": page_data["confidence"]
                    })

                    if page_data["text"].strip():
                        texts.append(page_data["text"])
                        confs.append(page_data["confidence"])

                final_conf = (
                    sum(confs) / len(confs)
                    if confs else 0.0
                )

                return {
                    "text": "\n\n".join(texts),
                    "confidence": round(final_conf, 4),
                    "total_pages": page_count,
                    "processed_pages": len(pages),
                    "pages": pages
                }

            # ==========================================
            # IMAGE OCR
            # ==========================================
            image = Image.open(image_path)

            result = self._extract_page_result(image, config)

            image.close()

            return {
                "text": result["text"],
                "confidence": round(result["confidence"], 4),
                "total_pages": 1,
                "processed_pages": 1,
                "pages": [
                    {
                        "page": 1,
                        "text": result["text"],
                        "confidence": round(result["confidence"], 4)
                    }
                ]
            }

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")


    # =====================================================
    # ADD THIS ALSO INSIDE SAME CLASS
    # Process PDF Page with Confidence
    # =====================================================

    def _process_pdf_page_with_confidence(
        self,
        image_path,
        page_no,
        dpi,
        config
    ):
        try:
            images = convert_from_path(
                image_path,
                dpi=dpi,
                first_page=page_no,
                last_page=page_no
            )

            if not images:
                return page_no, {
                    "text": "",
                    "confidence": 0.0
                }

            image = images[0]

            result = self._extract_page_result(image, config)

            image.close()

            del image
            del images
            gc.collect()

            return page_no, {
                "text": result["text"],
                "confidence": round(result["confidence"], 4)
            }

        except Exception:
            return page_no, {
                "text": "",
                "confidence": 0.0
            }

    # ======================================================
    # Support Language
    # ======================================================

    def supports_language(self, language_code):
        if not language_code or language_code == "auto":
            return True

        return self._normalize_language(language_code) in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self):
        return self.SUPPORTED_LANGUAGES