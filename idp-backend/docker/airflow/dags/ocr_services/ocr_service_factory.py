"""
OCR Service Factory
Factory pattern to create and return appropriate OCR service instances
"""

from typing import Optional
from .tesseract_ocr_service import TesseractOCRService
from .paddle_ocr_service import PaddleOCRService
from .optimized_ocr_service import OptimizedOCRService
from .safe_ocr_service import SafeOCRService
from .base_ocr_service import BaseOCRService


def get_ocr_service(ocr_engine: str) -> BaseOCRService:
    """
    Factory function to get OCR service instance based on engine name
    
    Args:
        ocr_engine: Name of OCR engine ('safe', 'tesseract', 'paddle', 'optimized', etc.)
    
    Returns:
        Instance of OCR service implementing BaseOCRService
    
    Raises:
        ValueError: If OCR engine is not supported
    """
    ocr_engine = ocr_engine.lower().strip()
    
    if ocr_engine == 'safe' or ocr_engine == 'safe_tesseract':
        # Production-safe service with Tesseract only
        return SafeOCRService(enable_paddle_fallback=False)
    elif ocr_engine == 'safe_paddle':
        # Safe service with PaddleOCR fallback (use with caution)
        return SafeOCRService(enable_paddle_fallback=True)
    elif ocr_engine == 'tesseract':
        return TesseractOCRService()
    elif ocr_engine == 'paddle':
        return PaddleOCRService()
    elif ocr_engine == 'optimized':
        # Production-grade optimized service with Tesseract primary, Paddle fallback
        return OptimizedOCRService(
            primary_engine='tesseract',
            fallback_engine='paddle',
            max_workers=4,
            enable_performance_logging=True
        )
    elif ocr_engine == 'optimized_paddle':
        # Optimized service with Paddle primary, Tesseract fallback
        return OptimizedOCRService(
            primary_engine='paddle',
            fallback_engine='tesseract',
            max_workers=2,  # Fewer workers for Paddle due to memory usage
            enable_performance_logging=True
        )
    # Future implementations:
    # elif ocr_engine == 'easyocr':
    #     return EasyOCRService()
    # elif ocr_engine == 'google_vision':
    #     return GoogleVisionOCRService()
    else:
        raise ValueError(
            f"Unsupported OCR engine: {ocr_engine}. "
            f"Supported engines: 'safe', 'safe_paddle', 'tesseract', 'paddle', 'optimized', 'optimized_paddle'"
        )

