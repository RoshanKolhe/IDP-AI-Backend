"""
Safe OCR Service - Production-grade with robust error handling
Prioritizes stability over performance for production environments
"""

import os
import gc
import time
import logging
from typing import Dict, Optional, List
from contextlib import contextmanager

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

from .base_ocr_service import BaseOCRService

logger = logging.getLogger(__name__)


class SafeOCRService(BaseOCRService):
    """
    Safe OCR service that prioritizes stability and error handling
    Uses Tesseract as primary with optional lightweight fallbacks
    """
    
    def __init__(self, enable_paddle_fallback: bool = False):
        """
        Initialize safe OCR service
        
        Args:
            enable_paddle_fallback: Whether to enable PaddleOCR as fallback (risky)
        """
        self.enable_paddle_fallback = enable_paddle_fallback
        self.tesseract_available = self._check_tesseract()
        self.paddle_available = False
        
        if enable_paddle_fallback:
            self.paddle_available = self._check_paddle()
        
        # Supported languages
        self.SUPPORTED_LANGUAGES = ['eng', 'hin', 'spa', 'fra', 'deu', 'chi_sim', 'ara']
        
        logger.info(f"SafeOCR initialized: tesseract={self.tesseract_available}, paddle={self.paddle_available}")
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is properly configured"""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            # Check available languages
            try:
                langs = pytesseract.get_languages()
                logger.info(f"Available Tesseract languages: {langs}")
                return 'eng' in langs
            except:
                # If we can't get languages, assume English is available
                return True
                
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            return False
    
    def _check_paddle(self) -> bool:
        """Safely check if PaddleOCR is available"""
        try:
            from paddleocr import PaddleOCR
            # Don't initialize here, just check import
            logger.info("PaddleOCR import successful")
            return True
        except Exception as e:
            logger.warning(f"PaddleOCR not available: {e}")
            return False
    
    def _safe_tesseract_extract(self, image: Image.Image, config: Optional[Dict] = None) -> Dict:
        """Safely extract text using Tesseract with robust error handling"""
        
        if not self.tesseract_available:
            raise RuntimeError("Tesseract not available")
        
        config = config or {}
        
        # Safe language handling
        lang = config.get('language_mode', 'eng')
        if lang == 'auto' or not lang or lang not in self.SUPPORTED_LANGUAGES:
            lang = 'eng'
        
        # Conservative PSM settings
        psm = config.get('psm', 6)  # Uniform text block
        oem = config.get('oem', 3)  # LSTM only
        
        tesseract_config = f'--psm {psm} --oem {oem}'
        
        try:
            # Extract text with timeout protection
            with self._timeout_context(30):  # 30 second timeout
                text = pytesseract.image_to_string(image, lang=lang, config=tesseract_config)
            
            # Get confidence if possible
            confidence = 75  # Default confidence
            try:
                with self._timeout_context(15):  # Shorter timeout for confidence
                    data = pytesseract.image_to_data(
                        image, lang=lang, config=tesseract_config, 
                        output_type=pytesseract.Output.DICT
                    )
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
            except Exception as e:
                logger.warning(f"Could not get Tesseract confidence: {e}")
            
            return {
                'text': text.strip(),
                'confidence': confidence,
                'engine': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise RuntimeError(f"Tesseract processing failed: {e}")
    
    def _safe_paddle_extract(self, image: Image.Image, config: Optional[Dict] = None) -> Dict:
        """Safely extract text using PaddleOCR with extensive error handling"""
        
        if not self.paddle_available:
            raise RuntimeError("PaddleOCR not available")
        
        try:
            from paddleocr import PaddleOCR
            
            # Initialize with safe settings
            ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                show_log=False,
                use_gpu=False,
                enable_mkldnn=False,
                cpu_threads=1,
            )
            
            # Convert image safely
            if isinstance(image, Image.Image):
                image_array = np.array(image.convert('RGB'))
            else:
                image_array = image
            
            # Process with timeout and memory protection
            with self._timeout_context(45):  # Longer timeout for PaddleOCR
                result = ocr.ocr(image_array, cls=False)
            
            # Clean up OCR instance immediately
            del ocr
            gc.collect()
            
            if not result or not result[0]:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'paddle'
                }
            
            texts = []
            confidences = []
            
            for line in result[0]:
                try:
                    text = line[1][0].strip()
                    conf = float(line[1][1])
                    if text:
                        texts.append(text)
                        confidences.append(conf)
                except (IndexError, ValueError, TypeError):
                    continue
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': '\n'.join(texts),
                'confidence': avg_confidence * 100,
                'engine': 'paddle'
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            raise RuntimeError(f"PaddleOCR processing failed: {e}")
    
    @contextmanager
    def _timeout_context(self, seconds: int):
        """Context manager for operation timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Clean up
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Light preprocessing for better OCR results"""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy for processing
            img_array = np.array(image)
            
            # Light enhancement only
            img_array = cv2.convertScaleAbs(img_array, alpha=1.1, beta=5)
            
            # Simple denoising
            img_array = cv2.medianBlur(img_array, 3)
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image
    
    def extract_text(self, image_path: str, config: Optional[Dict] = None) -> str:
        """Extract text with safe fallback strategy"""
        result = self.extract_text_with_confidence(image_path, config)
        return result.get('text', '')
    
    def extract_text_with_confidence(self, image_path: str, config: Optional[Dict] = None) -> Dict:
        """
        Extract text with confidence using safe fallback strategy
        
        Strategy:
        1. Try Tesseract (most stable)
        2. If enabled and Tesseract fails, try PaddleOCR
        3. Return best result or error
        """
        config = config or {}
        
        try:
            # Handle PDF files
            if image_path.lower().endswith('.pdf'):
                return self._process_pdf_safe(image_path, config)
            else:
                return self._process_image_safe(image_path, config)
                
        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'error': str(e)
            }
    
    def _process_pdf_safe(self, pdf_path: str, config: Dict) -> Dict:
        """Process PDF with safe page-by-page approach"""
        
        # Convert PDF to images with conservative settings
        dpi = config.get('dpi', 150)  # Lower DPI for stability
        max_pages = config.get('max_pages', 10)  # Limit pages
        
        try:
            convert_kwargs = {'dpi': dpi}
            if max_pages:
                convert_kwargs['last_page'] = max_pages
            
            images = convert_from_path(pdf_path, **convert_kwargs)
            
            if not images:
                return {'text': '', 'confidence': 0.0, 'engine': 'failed'}
            
            # Process pages sequentially (safer than parallel)
            all_texts = []
            all_confidences = []
            engines_used = []
            
            for i, img in enumerate(images):
                try:
                    logger.info(f"Processing page {i+1}/{len(images)}")
                    
                    # Preprocess image
                    processed_img = self._preprocess_image(img)
                    
                    # Try extraction with fallback
                    page_result = self._extract_with_fallback(processed_img, config)
                    
                    if page_result['text']:
                        all_texts.append(page_result['text'])
                        all_confidences.append(page_result['confidence'])
                        engines_used.append(page_result['engine'])
                    
                    # Clean up
                    img.close()
                    if processed_img != img:
                        processed_img.close()
                    
                except Exception as e:
                    logger.warning(f"Failed to process page {i+1}: {e}")
                    continue
            
            # Clean up images
            for img in images:
                img.close()
            del images
            gc.collect()
            
            # Aggregate results
            final_text = '\n\n'.join(all_texts)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            return {
                'text': final_text,
                'confidence': avg_confidence,
                'pages_processed': len(all_texts),
                'engines_used': engines_used,
                'engine': 'mixed' if len(set(engines_used)) > 1 else engines_used[0] if engines_used else 'failed'
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'error': str(e)
            }
    
    def _process_image_safe(self, image_path: str, config: Dict) -> Dict:
        """Process single image file safely"""
        
        try:
            image = Image.open(image_path)
            processed_image = self._preprocess_image(image)
            
            result = self._extract_with_fallback(processed_image, config)
            
            # Clean up
            image.close()
            if processed_image != image:
                processed_image.close()
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine': 'failed',
                'error': str(e)
            }
    
    def _extract_with_fallback(self, image: Image.Image, config: Dict) -> Dict:
        """Extract text with intelligent fallback"""
        
        # Try Tesseract first (most stable)
        if self.tesseract_available:
            try:
                result = self._safe_tesseract_extract(image, config)
                
                # Check if result is good enough
                if result['text'] and len(result['text'].strip()) > 5 and result['confidence'] > 30:
                    return result
                
                logger.info(f"Tesseract result poor (conf={result['confidence']:.1f}), trying fallback")
                
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        # Try PaddleOCR fallback if enabled and available
        if self.enable_paddle_fallback and self.paddle_available:
            try:
                result = self._safe_paddle_extract(image, config)
                
                if result['text']:
                    logger.info(f"PaddleOCR fallback successful")
                    return result
                
            except Exception as e:
                logger.warning(f"PaddleOCR fallback failed: {e}")
        
        # Return empty result if all methods fail
        return {
            'text': '',
            'confidence': 0.0,
            'engine': 'failed'
        }
    
    # BaseOCRService interface methods
    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported"""
        if not language_code or language_code == 'auto':
            return True
        return language_code in self.SUPPORTED_LANGUAGES
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()