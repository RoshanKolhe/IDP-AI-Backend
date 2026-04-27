"""
OCR Service Factory
Factory pattern to create and return appropriate OCR service instances
"""

from typing import Optional
from .tesseract_ocr_service import TesseractOCRService
from .paddle_ocr_service import PaddleOCRService
from .base_ocr_service import BaseOCRService


def get_ocr_service(ocr_engine: str) -> BaseOCRService:
    """
    Factory function to get OCR service instance based on engine name
    
    Args:
        ocr_engine: Name of OCR engine ('tesseract', 'easyocr', 'google_vision', etc.)
    
    Returns:
        Instance of OCR service implementing BaseOCRService
    
    Raises:
        ValueError: If OCR engine is not supported
    """
    ocr_engine = ocr_engine.lower().strip()
    
    if ocr_engine == 'tesseract':
        return TesseractOCRService()
    elif ocr_engine == 'paddle':
        return PaddleOCRService()
    # Future implementations:
    # elif ocr_engine == 'easyocr':
    #     return EasyOCRService()
    # elif ocr_engine == 'google_vision':
    #     return GoogleVisionOCRService()
    else:
        raise ValueError(
            f"Unsupported OCR engine: {ocr_engine}. "
            f"Supported engines: 'tesseract', 'paddle'"
        )

