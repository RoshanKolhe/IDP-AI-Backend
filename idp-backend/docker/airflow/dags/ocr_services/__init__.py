"""
OCR Services Module
Exports base class and factory for OCR service implementations
"""

from .base_ocr_service import BaseOCRService
from .paddle_ocr_service import PaddleOCRService
from .tesseract_ocr_service import TesseractOCRService
from .ocr_service_factory import get_ocr_service

__all__ = [
    'BaseOCRService',
    'PaddleOCRService',
    'TesseractOCRService',
    'get_ocr_service'
]

