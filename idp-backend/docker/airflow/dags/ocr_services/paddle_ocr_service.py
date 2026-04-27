"""
PaddleOCR 3.5 Compatible OCR Service
"""

from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from pdf2image import convert_from_path
from PIL import Image

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

    # --------------------------
    # Language
    # --------------------------

    def _normalize_language(self, language_mode: Optional[str]) -> str:
        if not language_mode or language_mode == "auto":
            return "en"

        language_mode = language_mode.lower().strip()

        mapping = {
            "eng": "en",
            "eng+hin": "en",
            "hin": "devanagari",
        }

        return mapping.get(language_mode, language_mode)

    # --------------------------
    # Engine
    # --------------------------

    def _get_engine(self, config=None):
        config = config or {}

        lang = self._normalize_language(
            config.get("language_mode")
        )

        engine_config = {
            "lang": lang
        }

        if self._ocr_engine is None or self._engine_config != engine_config:
            self._ocr_engine = self._engine_cls(**engine_config)
            self._engine_config = engine_config

        return self._ocr_engine

    # --------------------------
    # PDF
    # --------------------------

    def _get_pdf_convert_kwargs(self, config=None):
        config = config or {}

        kwargs = {
            "dpi": config.get("dpi", 300)
        }

        if config.get("first_page"):
            kwargs["first_page"] = config["first_page"]

        if config.get("last_page"):
            kwargs["last_page"] = config["last_page"]

        if config.get("thread_count"):
            kwargs["thread_count"] = config["thread_count"]

        return kwargs

    # --------------------------
    # Preprocess
    # --------------------------

    def _preprocess_image(self, image):
        img = np.array(image.convert("RGB"))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.fastNlMeansDenoising(gray)

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # --------------------------
    # Parse OCR Output
    # --------------------------

    def _parse_result(self, result):
        texts = []
        confidences = []
        lines = []

        if not result:
            return {
                "text": "",
                "confidence": 0.0,
                "lines": []
            }

        for block in result:
            if isinstance(block, dict):
                text = block.get("rec_text", "")
                conf = float(block.get("rec_score", 0.0))
                box = block.get("dt_polys", [])

                if text:
                    texts.append(text)
                    confidences.append(conf)

                    lines.append({
                        "text": text,
                        "confidence": conf,
                        "box": box
                    })

        total_chars = sum(len(t) for t in texts) or 1

        weighted_conf = sum(
            len(texts[i]) * confidences[i]
            for i in range(len(texts))
        ) / total_chars

        return {
            "text": "\n".join(texts),
            "confidence": weighted_conf,
            "lines": lines
        }

    # --------------------------
    # OCR Single Page
    # --------------------------

    def _extract_page_result(self, image, config=None):
        engine = self._get_engine(config)

        processed = self._preprocess_image(image)

        result = engine.ocr(processed)

        return self._parse_result(result)

    # --------------------------
    # Extract Text
    # --------------------------

    def extract_text(self, image_path, config=None):
        config = config or {}

        try:
            if image_path.lower().endswith(".pdf"):

                from PyPDF2 import PdfReader

                page_count = len(PdfReader(image_path).pages)

                all_text = []

                dpi = config.get("dpi", 150)

                for page_no in range(1, page_count + 1):

                    images = convert_from_path(
                        image_path,
                        dpi=dpi,
                        first_page=page_no,
                        last_page=page_no
                    )

                    if not images:
                        continue

                    image = images[0]

                    result = self._extract_page_result(image, config)

                    if result["text"].strip():
                        all_text.append(result["text"])

                    image.close()
                    del image
                    del images

                return "\n\n".join(all_text)

            image = Image.open(image_path)

            result = self._extract_page_result(image, config)

            image.close()

            return result["text"]

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")

    # --------------------------
    # Extract with Confidence
    # --------------------------

    def extract_text_with_confidence(self, image_path, config=None):
        config = config or {}

        try:
            if image_path.lower().endswith(".pdf"):

                from PyPDF2 import PdfReader

                page_count = len(PdfReader(image_path).pages)

                dpi = config.get("dpi", 150)
                max_pages = config.get("max_pages", page_count)

                all_pages = []
                texts = []
                confs = []

                for page_no in range(1, min(page_count, max_pages) + 1):

                    images = convert_from_path(
                        image_path,
                        dpi=dpi,
                        first_page=page_no,
                        last_page=page_no
                    )

                    if not images:
                        continue

                    image = images[0]

                    result = self._extract_page_result(image, config)

                    all_pages.append(result)

                    if result["text"].strip():
                        texts.append(result["text"])
                        confs.append(result["confidence"])

                    image.close()
                    del image
                    del images

                return {
                    "text": "\n\n".join(texts),
                    "confidence": (
                        sum(confs) / len(confs)
                        if confs else 0.0
                    ),
                    "pages": all_pages
                }

            image = Image.open(image_path)

            result = self._extract_page_result(image, config)

            image.close()

            return result

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")

    # --------------------------
    # Language Support
    # --------------------------

    def supports_language(self, language_code):
        if not language_code or language_code == "auto":
            return True

        return self._normalize_language(language_code) in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self):
        return self.SUPPORTED_LANGUAGES