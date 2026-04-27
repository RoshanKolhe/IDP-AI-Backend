"""
Enhanced Paddle OCR Service
Preserves original logic + improves OCR quality
"""

from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

from pdf2image import convert_from_path
from PIL import Image

from .base_ocr_service import BaseOCRService


class PaddleOCRService(BaseOCRService):
    """Enhanced PaddleOCR service implementation."""

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

    # -------------------------
    # LANGUAGE HELPERS
    # -------------------------

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

    # -------------------------
    # ENGINE CACHE
    # -------------------------

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

    # -------------------------
    # PDF CONFIG
    # -------------------------

    def _get_pdf_convert_kwargs(self, config: Optional[Dict] = None) -> Dict:
        config = config or {}

        kwargs = {
            "dpi": config.get("dpi", 300),
        }

        if config.get("first_page") is not None:
            kwargs["first_page"] = config["first_page"]

        if config.get("last_page") is not None:
            kwargs["last_page"] = config["last_page"]

        if config.get("thread_count") is not None:
            kwargs["thread_count"] = config["thread_count"]

        return kwargs

    # -------------------------
    # IMAGE PREPROCESSING
    # -------------------------

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
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

        final = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        return final

    # -------------------------
    # OCR PAGE EXTRACTION
    # -------------------------

    def _extract_page_result(
        self,
        image: Image.Image,
        config: Optional[Dict] = None
    ) -> Dict:

        config = config or {}

        ocr_engine = self._get_engine(config)

        processed = self._preprocess_image(image)

        result = ocr_engine.ocr(
            processed,
            cls=bool(config.get("use_angle_cls", True))
        )

        lines = result[0] if result else []

        texts = []
        confidences = []
        structured_lines = []

        for line in lines or []:
            if not line or len(line) < 2:
                continue

            box = line[0]
            text = line[1][0] if len(line[1]) > 0 else ""
            conf = float(line[1][1]) if len(line[1]) > 1 else 0.0

            if text:
                texts.append(text)
                confidences.append(conf)

                structured_lines.append({
                    "text": text,
                    "confidence": conf,
                    "box": box
                })

        # weighted confidence by text length
        total_chars = sum(len(t) for t in texts) or 1

        weighted_conf = sum(
            len(texts[i]) * confidences[i]
            for i in range(len(texts))
        ) / total_chars

        return {
            "text": "\n".join(texts),
            "confidence": weighted_conf,
            "lines": structured_lines,
        }

    # -------------------------
    # MAIN TEXT EXTRACTION
    # -------------------------

    def extract_text(
        self,
        image_path: str,
        config: Optional[Dict] = None
    ) -> str:

        config = config or {}

        try:
            if image_path.lower().endswith(".pdf"):

                images = convert_from_path(
                    image_path,
                    **self._get_pdf_convert_kwargs(config)
                )

                workers = config.get("workers", 4)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(
                        executor.map(
                            lambda img: self._extract_page_result(img, config),
                            images
                        )
                    )

                return "\n\n".join(
                    r["text"] for r in results if r["text"].strip()
                )

            image = Image.open(image_path)

            return self._extract_page_result(image, config)["text"]

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")

    # -------------------------
    # TEXT + CONFIDENCE
    # -------------------------

    def extract_text_with_confidence(
        self,
        image_path: str,
        config: Optional[Dict] = None
    ) -> Dict:

        config = config or {}

        try:
            if image_path.lower().endswith(".pdf"):

                images = convert_from_path(
                    image_path,
                    **self._get_pdf_convert_kwargs(config)
                )

                workers = config.get("workers", 4)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(
                        executor.map(
                            lambda img: self._extract_page_result(img, config),
                            images
                        )
                    )

                text = "\n\n".join(
                    r["text"] for r in results if r["text"].strip()
                )

                confs = [
                    r["confidence"]
                    for r in results
                    if r["text"].strip()
                ]

                return {
                    "text": text,
                    "confidence": (
                        sum(confs) / len(confs)
                        if confs else 0.0
                    ),
                    "pages": results,
                }

            image = Image.open(image_path)

            return self._extract_page_result(image, config)

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")

    # -------------------------
    # LANGUAGE SUPPORT
    # -------------------------

    def supports_language(self, language_code: str) -> bool:
        if not language_code or language_code == "auto":
            return True

        normalized = self._normalize_language(language_code)

        return normalized in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES