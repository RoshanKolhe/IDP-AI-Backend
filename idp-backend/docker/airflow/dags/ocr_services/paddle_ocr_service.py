"""
FINAL ENTERPRISE PADDLE OCR SERVICE
CPU Optimized | Multi-pass OCR | Large PDFs | Bad Quality Docs
Designed for:
- Agreements
- Government scans
- Work orders
- Records
- 100+ page PDFs
- 8 Core CPU / 32 GB RAM
"""

from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import re
import threading

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
        self._engine_lock = threading.Lock()
        self._ocr_call_lock = threading.Lock()

    # =====================================================
    # LANGUAGE
    # =====================================================

    def _normalize_language(self, language_mode: Optional[str]) -> str:
        if not language_mode or language_mode == "auto":
            return "en"

        mapping = {
            "eng": "en",
            "eng+hin": "en",
            "hin": "devanagari",
        }

        return mapping.get(language_mode.lower().strip(), "en")

    # =====================================================
    # ENGINE
    # =====================================================

    def _get_engine(self, config=None):
        config = config or {}

        lang = self._normalize_language(
            config.get("language_mode")
        )

        engine_config = {
            "lang": lang,
            "use_angle_cls": True,
            "show_log": False,
            "use_gpu": False,
            "cpu_threads": 2,
        }

        # PaddleOCR init is expensive and not thread-safe during first bootstrap.
        # Guard initialization to avoid duplicate model downloads and memory corruption.
        if self._ocr_engine is None or self._engine_config != engine_config:
            with self._engine_lock:
                if self._ocr_engine is None or self._engine_config != engine_config:
                    self._ocr_engine = self._engine_cls(**engine_config)
                    self._engine_config = engine_config

        return self._ocr_engine

    # =====================================================
    # DESKEW
    # =====================================================

    def _deskew(self, gray):
        coords = np.column_stack(np.where(gray < 220))

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

    # =====================================================
    # MAIN PREPROCESS
    # =====================================================

    def _preprocess_image(self, image):
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape[:2]

        # Upscale if needed
        if w < 2000:
            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

        # Denoise
        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            18,
            7,
            21
        )

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=4.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        # Sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        gray = cv2.filter2D(gray, -1, kernel)

        # Deskew
        gray = self._deskew(gray)

        # Background cleanup
        blur = cv2.GaussianBlur(gray, (0, 0), 25)
        gray = cv2.divide(gray, blur, scale=255)

        # Threshold
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8
        )

        # Morph cleanup
        kernel = np.ones((1, 1), np.uint8)

        gray = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel
        )

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # =====================================================
    # MULTI VARIANTS
    # =====================================================

    def _generate_variants(self, image):
        variants = []

        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Variant 1
        variants.append(self._preprocess_image(image))

        # Variant 2 CLAHE only
        clahe = cv2.createCLAHE(
            clipLimit=4.0,
            tileGridSize=(8, 8)
        )
        v2 = clahe.apply(gray)
        variants.append(cv2.cvtColor(v2, cv2.COLOR_GRAY2RGB))

        # Variant 3 sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        v3 = cv2.filter2D(gray, -1, kernel)
        variants.append(cv2.cvtColor(v3, cv2.COLOR_GRAY2RGB))

        # Variant 4 original
        variants.append(img)

        # Variant 5 softer threshold
        v5 = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            4
        )
        variants.append(cv2.cvtColor(v5, cv2.COLOR_GRAY2RGB))

        return variants

    # =====================================================
    # PARSE RESULT
    # =====================================================

    def _parse_result(self, result):
        if not result or not result[0]:
            return {
                "text": "",
                "confidence": 0.0
            }

        rows = []

        for item in result[0]:
            box = item[0]
            text = item[1][0].strip()
            conf = float(item[1][1])

            if not text:
                continue

            top_y = min(p[1] for p in box)
            left_x = min(p[0] for p in box)

            rows.append({
                "text": text,
                "conf": conf,
                "top_y": top_y,
                "left_x": left_x
            })

        rows.sort(
            key=lambda x: (
                round(x["top_y"] / 10),
                x["left_x"]
            )
        )

        lines = []
        confs = []

        for row in rows:
            lines.append(row["text"])
            confs.append(row["conf"])

        avg_conf = (
            sum(confs) / len(confs)
            if confs else 0.0
        )

        return {
            "text": "\n".join(lines),
            "confidence": avg_conf
        }

    # =====================================================
    # SCORE RESULT
    # =====================================================

    def _score_text(self, parsed):
        text = parsed["text"]
        conf = parsed["confidence"]

        if not text.strip():
            return 0

        alpha = len(
            re.findall(r"[A-Za-z0-9]", text)
        )

        words = len(text.split())

        score = (
            conf * 100
            + alpha * 0.4
            + words * 3
            + len(text) * 0.05
        )

        return score

    # =====================================================
    # PAGE OCR MULTI PASS
    # =====================================================

    def _extract_page_result(self, image, config=None):
        engine = self._get_engine(config)

        variants = self._generate_variants(image)

        best = {
            "text": "",
            "confidence": 0.0
        }

        best_score = -1

        for variant in variants:
            try:
                # Guard native Paddle inference call; concurrent invocations can
                # crash with allocator corruption in some environments.
                with self._ocr_call_lock:
                    result = engine.ocr(variant)
                parsed = self._parse_result(result)

                score = self._score_text(parsed)

                if score > best_score:
                    best_score = score
                    best = parsed

            except Exception:
                continue

        return best

    # =====================================================
    # PROCESS PDF PAGE
    # =====================================================

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

            result = self._extract_page_result(
                image,
                config
            )

            image.close()
            del image
            del images
            gc.collect()

            return page_no, result["text"]

        except Exception:
            return page_no, ""

    # =====================================================
    # EXTRACT TEXT
    # =====================================================

    def extract_text(self, image_path, config=None):
        config = config or {}

        try:
            if image_path.lower().endswith(".pdf"):

                page_count = len(
                    PdfReader(image_path).pages
                )

                dpi = config.get("dpi", 300)
                workers = config.get("workers", 3)

                results = {}

                with ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:

                    futures = []

                    for page_no in range(
                        1,
                        page_count + 1
                    ):
                        futures.append(
                            executor.submit(
                                self._process_pdf_page,
                                image_path,
                                page_no,
                                dpi,
                                config
                            )
                        )

                    for future in as_completed(
                        futures
                    ):
                        page_no, text = future.result()
                        results[page_no] = text

                final_text = []

                for page_no in sorted(results):
                    if results[page_no].strip():
                        final_text.append(
                            results[page_no]
                        )

                return "\n\n".join(final_text)

            image = Image.open(image_path)

            result = self._extract_page_result(
                image,
                config
            )

            image.close()

            return result["text"]

        except Exception as exc:
            raise RuntimeError(
                f"OCR extraction failed: {exc}"
            )
        
    def extract_text_with_confidence(self, image_path, config=None):
        config = config or {}

        try:
            text = self.extract_text(image_path, config)

            return {
                "text": text,
                "confidence": 0.85 if text.strip() else 0.0,
                "total_pages": 1,
                "processed_pages": 1,
                "pages": [
                    {
                        "page": 1,
                        "text": text,
                        "confidence": 0.85 if text.strip() else 0.0
                    }
                ]
            }

        except Exception as exc:
            raise RuntimeError(f"OCR extraction failed: {exc}")

    # =====================================================
    # LANGUAGE SUPPORT
    # =====================================================

    def supports_language(self, language_code):
        if not language_code or language_code == "auto":
            return True

        return (
            self._normalize_language(language_code)
            in self.SUPPORTED_LANGUAGES
        )

    def get_supported_languages(self):
        return self.SUPPORTED_LANGUAGES
