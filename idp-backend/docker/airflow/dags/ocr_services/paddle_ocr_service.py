"""
Paddle OCR Service Implementation
Handles text extraction using PaddleOCR engine
"""

from typing import Dict, Optional

import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from .base_ocr_service import BaseOCRService


class PaddleOCRService(BaseOCRService):
    """PaddleOCR service implementation."""

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
            raise RuntimeError(f"PaddleOCR not available: {exc}")

        self._ocr_engine = None
        self._engine_config = None

    def _normalize_language(self, language_mode: Optional[str]) -> str:
        if not language_mode or language_mode == "auto":
            return "en"

        language_mode = language_mode.lower().strip()
        language_map = {
            "eng": "en",
            "eng+hin": "en",
            "hin": "devanagari",
        }
        return language_map.get(language_mode, language_mode)

    def _get_engine(self, config: Optional[Dict] = None):
        config = config or {}
        lang = self._normalize_language(config.get("language_mode"))
        use_angle_cls = bool(config.get("use_angle_cls", True))

        engine_config = {
            "use_angle_cls": use_angle_cls,
            "lang": lang,
            "show_log": False,
            "use_gpu": False,
        }

        if self._ocr_engine is None or self._engine_config != engine_config:
            self._ocr_engine = self._engine_cls(**engine_config)
            self._engine_config = engine_config

        return self._ocr_engine

    def _get_pdf_convert_kwargs(self, config: Optional[Dict] = None) -> Dict:
        config = config or {}
        convert_kwargs = {
            "dpi": config.get("dpi", 300),
        }

        if config.get("first_page") is not None:
            convert_kwargs["first_page"] = config["first_page"]
        if config.get("last_page") is not None:
            convert_kwargs["last_page"] = config["last_page"]
        if config.get("thread_count") is not None:
            convert_kwargs["thread_count"] = config["thread_count"]

        return convert_kwargs

    def _image_to_array(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    def _extract_page_result(self, image, config: Optional[Dict] = None) -> Dict:
        ocr_engine = self._get_engine(config)
        image_array = np.array(image.convert("RGB"))
        result = ocr_engine.ocr(image_array, cls=bool((config or {}).get("use_angle_cls", True)))
        lines = result[0] if result else []

        texts = []
        confidences = []
        for line in lines or []:
            if not line or len(line) < 2:
                continue
            text = line[1][0] if len(line[1]) > 0 else ""
            confidence = float(line[1][1]) if len(line[1]) > 1 else 0.0
            if text:
                texts.append(text)
                confidences.append(confidence)

        return {
            "text": "\n".join(texts),
            "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        }

    def extract_text(self, image_path: str, config: Optional[Dict] = None) -> str:
        config = config or {}

        if image_path.lower().endswith(".pdf"):
            images = convert_from_path(image_path, **self._get_pdf_convert_kwargs(config))
            page_texts = []
            for image in images:
                page_result = self._extract_page_result(image, config)
                if page_result["text"].strip():
                    page_texts.append(page_result["text"])
            return "\n".join(page_texts)

        image = Image.open(image_path)
        return self._extract_page_result(image, config)["text"]

    def extract_text_with_confidence(self, image_path: str, config: Optional[Dict] = None) -> Dict:
        config = config or {}

        if image_path.lower().endswith(".pdf"):
            images = convert_from_path(image_path, **self._get_pdf_convert_kwargs(config))
            page_results = [self._extract_page_result(image, config) for image in images]
            text = "\n".join(result["text"] for result in page_results if result["text"].strip())
            confidences = [result["confidence"] for result in page_results if result["text"].strip()]
            return {
                "text": text,
                "confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
            }

        image = Image.open(image_path)
        return self._extract_page_result(image, config)

    def supports_language(self, language_code: str) -> bool:
        if not language_code or language_code == "auto":
            return True

        normalized = self._normalize_language(language_code)
        return normalized in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self) -> list:
        return self.SUPPORTED_LANGUAGES
