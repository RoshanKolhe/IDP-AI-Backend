"""
Optimized OCR Service - Production-grade implementation
Combines multiple OCR engines with intelligent fallback and performance optimizations
"""

import os
import gc
import time
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import logging

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

from .base_ocr_service import BaseOCRService

logger = logging.getLogger(__name__)


class OptimizedOCRService(BaseOCRService):
    """
    Production-optimized OCR service with:
    - Intelligent engine selection based on document characteristics
    - Parallel processing for multi-page documents
    - Lightweight preprocessing
    - Memory management and cleanup
    - Performance monitoring
    """
    
    def __init__(self, 
                 primary_engine: str = "tesseract",
                 fallback_engine: str = "paddle",
                 max_workers: int = 4,
                 enable_performance_logging: bool = True):
        """
        Initialize optimized OCR service
        
        Args:
            primary_engine: Primary OCR engine ('tesseract', 'paddle', 'easyocr')
            fallback_engine: Fallback OCR engine
            max_workers: Maximum parallel workers for page processing
            enable_performance_logging: Enable performance metrics logging
        """
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.max_workers = max_workers
        self.enable_performance_logging = enable_performance_logging
        
        # Engine instances (lazy loaded)
        self._engines = {}
        self._performance_stats = {
            'total_pages': 0,
            'total_time': 0,
            'engine_usage': {},
            'fallback_rate': 0
        }
        
        # Supported languages
        self.SUPPORTED_LANGUAGES = ['eng', 'hin', 'spa', 'fra', 'deu', 'chi_sim', 'ara']
    
    @contextmanager
    def _performance_timer(self, operation: str):
        """Context manager for performance timing"""
        start_time = time.time()
        try:
            yield
        finally:
            if self.enable_performance_logging:
                elapsed = time.time() - start_time
                logger.info(f"OCR {operation} took {elapsed:.2f}s")
    
    def _get_engine(self, engine_name: str):
        """Lazy load OCR engines to save memory"""
        if engine_name not in self._engines:
            if engine_name == "tesseract":
                self._engines[engine_name] = self._create_tesseract_engine()
            elif engine_name == "paddle":
                self._engines[engine_name] = self._create_paddle_engine()
            elif engine_name == "easyocr":
                self._engines[engine_name] = self._create_easyocr_engine()
            else:
                raise ValueError(f"Unsupported engine: {engine_name}")
        
        return self._engines[engine_name]
    
    def _create_tesseract_engine(self):
        """Create Tesseract engine wrapper"""
        class TesseractWrapper:
            def extract_text(self, image, config=None):
                config = config or {}
                lang = config.get('language_mode', 'eng')
                
                # Fix language mapping - avoid 'auto' which causes issues
                if lang == 'auto' or not lang:
                    lang = 'eng'
                
                psm = config.get('psm', 6)  # Uniform text block
                oem = config.get('oem', 3)  # LSTM only
                
                tesseract_config = f'--psm {psm} --oem {oem}'
                return pytesseract.image_to_string(image, lang=lang, config=tesseract_config)
            
            def extract_with_confidence(self, image, config=None):
                config = config or {}
                lang = config.get('language_mode', 'eng')
                
                # Fix language mapping - avoid 'auto' which causes issues
                if lang == 'auto' or not lang:
                    lang = 'eng'
                
                psm = config.get('psm', 6)
                oem = config.get('oem', 3)
                
                tesseract_config = f'--psm {psm} --oem {oem}'
                
                # Get text
                text = pytesseract.image_to_string(image, lang=lang, config=tesseract_config)
                
                # Get confidence
                try:
                    data = pytesseract.image_to_data(image, lang=lang, config=tesseract_config, 
                                                   output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                except:
                    avg_confidence = 50  # Default confidence
                
                return {'text': text, 'confidence': avg_confidence}
        
        return TesseractWrapper()
    
    def _create_paddle_engine(self):
        """Create lightweight PaddleOCR engine wrapper with error handling"""
        try:
            from paddleocr import PaddleOCR
            
            class PaddleWrapper:
                def __init__(self):
                    # Use safer configuration to avoid segmentation faults
                    try:
                        self.ocr = PaddleOCR(
                            use_angle_cls=False,  # Disable angle classification for stability
                            lang='en',
                            show_log=False,
                            use_gpu=False,  # CPU mode for stability
                            det_model_dir=None,  # Use default lightweight models
                            rec_model_dir=None,
                            cls_model_dir=None,
                            enable_mkldnn=False,  # Disable MKLDNN for compatibility
                            cpu_threads=1,  # Single thread to avoid race conditions
                        )
                        self.initialized = True
                    except Exception as e:
                        logger.error(f"Failed to initialize PaddleOCR: {e}")
                        self.initialized = False
                
                def extract_text(self, image, config=None):
                    if not self.initialized:
                        raise RuntimeError("PaddleOCR not properly initialized")
                    
                    try:
                        # Convert PIL to numpy array safely
                        if isinstance(image, Image.Image):
                            image_array = np.array(image.convert('RGB'))
                        else:
                            image_array = image
                        
                        # Add timeout and error handling
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("PaddleOCR processing timeout")
                        
                        # Set 30 second timeout
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)
                        
                        try:
                            result = self.ocr.ocr(image_array, cls=False)
                        finally:
                            signal.alarm(0)  # Cancel timeout
                        
                        if not result or not result[0]:
                            return ""
                        
                        texts = [line[1][0] for line in result[0] if line[1][0].strip()]
                        return '\n'.join(texts)
                        
                    except (TimeoutError, MemoryError, RuntimeError) as e:
                        logger.warning(f"PaddleOCR extraction failed safely: {e}")
                        raise RuntimeError(f"PaddleOCR processing failed: {e}")
                    except Exception as e:
                        logger.error(f"PaddleOCR extraction failed: {e}")
                        raise RuntimeError(f"PaddleOCR processing failed: {e}")
                
                def extract_with_confidence(self, image, config=None):
                    if not self.initialized:
                        raise RuntimeError("PaddleOCR not properly initialized")
                    
                    try:
                        if isinstance(image, Image.Image):
                            image_array = np.array(image.convert('RGB'))
                        else:
                            image_array = image
                        
                        # Add timeout protection
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise TimeoutError("PaddleOCR processing timeout")
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(30)
                        
                        try:
                            result = self.ocr.ocr(image_array, cls=False)
                        finally:
                            signal.alarm(0)
                        
                        if not result or not result[0]:
                            return {'text': '', 'confidence': 0.0}
                        
                        texts = []
                        confidences = []
                        
                        for line in result[0]:
                            text = line[1][0].strip()
                            conf = float(line[1][1])
                            if text:
                                texts.append(text)
                                confidences.append(conf)
                        
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        return {
                            'text': '\n'.join(texts),
                            'confidence': avg_confidence * 100  # Convert to percentage
                        }
                        
                    except (TimeoutError, MemoryError, RuntimeError) as e:
                        logger.warning(f"PaddleOCR extraction with confidence failed safely: {e}")
                        raise RuntimeError(f"PaddleOCR processing failed: {e}")
                    except Exception as e:
                        logger.error(f"PaddleOCR extraction with confidence failed: {e}")
                        raise RuntimeError(f"PaddleOCR processing failed: {e}")
            
            return PaddleWrapper()
            
        except ImportError:
            logger.warning("PaddleOCR not available, falling back to Tesseract")
            return self._create_tesseract_engine()
        except Exception as e:
            logger.error(f"Failed to create PaddleOCR wrapper: {e}")
            return self._create_tesseract_engine()
    
    def _create_easyocr_engine(self):
        """Create EasyOCR engine wrapper (future implementation)"""
        try:
            import easyocr
            
            class EasyOCRWrapper:
                def __init__(self):
                    self.reader = easyocr.Reader(['en'], gpu=False)
                
                def extract_text(self, image, config=None):
                    try:
                        if isinstance(image, Image.Image):
                            image_array = np.array(image)
                        else:
                            image_array = image
                        
                        results = self.reader.readtext(image_array)
                        texts = [result[1] for result in results if result[1].strip()]
                        return '\n'.join(texts)
                    except Exception as e:
                        logger.warning(f"EasyOCR extraction failed: {e}")
                        return ""
                
                def extract_with_confidence(self, image, config=None):
                    try:
                        if isinstance(image, Image.Image):
                            image_array = np.array(image)
                        else:
                            image_array = image
                        
                        results = self.reader.readtext(image_array)
                        
                        texts = []
                        confidences = []
                        
                        for result in results:
                            text = result[1].strip()
                            conf = float(result[2])
                            if text:
                                texts.append(text)
                                confidences.append(conf)
                        
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        return {
                            'text': '\n'.join(texts),
                            'confidence': avg_confidence * 100
                        }
                    except Exception as e:
                        logger.warning(f"EasyOCR extraction with confidence failed: {e}")
                        return {'text': '', 'confidence': 0.0}
            
            return EasyOCRWrapper()
            
        except ImportError:
            logger.warning("EasyOCR not available, falling back to Tesseract")
            return self._create_tesseract_engine()
    
    def _lightweight_preprocess(self, image: Image.Image) -> Image.Image:
        """
        Lightweight image preprocessing for better OCR results
        Much faster than the original heavy preprocessing
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Simple contrast enhancement (much faster than CLAHE)
            img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
            
            # Light denoising (faster than fastNlMeansDenoising)
            img_array = cv2.medianBlur(img_array, 3)
            
            # Simple thresholding for better text contrast
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def _should_use_fallback(self, text: str, confidence: float) -> bool:
        """
        Intelligent decision on whether to use fallback engine
        
        Args:
            text: Extracted text
            confidence: Confidence score
            
        Returns:
            True if fallback should be used
        """
        # Use fallback if:
        # 1. No text extracted
        # 2. Very low confidence (< 30%)
        # 3. Text is too short and confidence is low (< 50%)
        
        if not text or not text.strip():
            return True
        
        if confidence < 30:
            return True
        
        if len(text.strip()) < 10 and confidence < 50:
            return True
        
        return False
    
    def _process_single_page(self, image: Image.Image, config: Optional[Dict] = None) -> Dict:
        """
        Process a single page with primary and fallback engines
        
        Args:
            image: PIL Image object
            config: OCR configuration
            
        Returns:
            Dictionary with text, confidence, and engine used
        """
        config = config or {}
        
        # Lightweight preprocessing
        processed_image = self._lightweight_preprocess(image)
        
        # Try primary engine
        try:
            primary_engine = self._get_engine(self.primary_engine)
            result = primary_engine.extract_with_confidence(processed_image, config)
            
            text = result.get('text', '').strip()
            confidence = float(result.get('confidence', 0))
            
            # Update stats
            self._performance_stats['engine_usage'][self.primary_engine] = \
                self._performance_stats['engine_usage'].get(self.primary_engine, 0) + 1
            
            # Check if fallback is needed
            if not self._should_use_fallback(text, confidence):
                return {
                    'text': text,
                    'confidence': confidence,
                    'engine_used': self.primary_engine,
                    'fallback_used': False
                }
        
        except Exception as e:
            logger.warning(f"Primary engine {self.primary_engine} failed: {e}")
        
        # Use fallback engine
        try:
            fallback_engine = self._get_engine(self.fallback_engine)
            result = fallback_engine.extract_with_confidence(processed_image, config)
            
            text = result.get('text', '').strip()
            confidence = float(result.get('confidence', 0))
            
            # Update stats
            self._performance_stats['engine_usage'][self.fallback_engine] = \
                self._performance_stats['engine_usage'].get(self.fallback_engine, 0) + 1
            self._performance_stats['fallback_rate'] += 1
            
            return {
                'text': text,
                'confidence': confidence,
                'engine_used': self.fallback_engine,
                'fallback_used': True
            }
            
        except Exception as e:
            logger.error(f"Fallback engine {self.fallback_engine} failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'engine_used': 'none',
                'fallback_used': True
            }
        
        finally:
            # Clean up processed image
            if processed_image != image:
                processed_image.close()
            gc.collect()
    
    def extract_text(self, image_path: str, config: Optional[Dict] = None) -> str:
        """
        Extract text from image or PDF with optimized performance
        
        Args:
            image_path: Path to image or PDF file
            config: Configuration dictionary
            
        Returns:
            Extracted text
        """
        with self._performance_timer("extract_text"):
            result = self.extract_text_with_confidence(image_path, config)
            return result.get('text', '')
    
    def extract_text_with_confidence(self, image_path: str, config: Optional[Dict] = None) -> Dict:
        """
        Extract text with confidence using parallel processing for PDFs
        
        Args:
            image_path: Path to image or PDF file
            config: Configuration dictionary with:
                - dpi: Image resolution (default: 200 for speed)
                - max_pages: Maximum pages to process
                - language_mode: Language code
                - parallel: Enable parallel processing (default: True)
        
        Returns:
            Dictionary with text, confidence, and processing stats
        """
        config = config or {}
        
        # Optimize DPI for speed vs quality balance
        dpi = config.get('dpi', 200)  # Reduced from 300 for better performance
        max_pages = config.get('max_pages', None)
        parallel = config.get('parallel', True)
        
        try:
            if image_path.lower().endswith('.pdf'):
                return self._process_pdf_parallel(image_path, config, dpi, max_pages, parallel)
            else:
                return self._process_single_image(image_path, config)
                
        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'pages_processed': 0,
                'engine_stats': {},
                'processing_time': 0
            }
    
    def _process_pdf_parallel(self, pdf_path: str, config: Dict, dpi: int, max_pages: Optional[int], parallel: bool) -> Dict:
        """Process PDF with parallel page processing"""
        start_time = time.time()
        
        # Convert PDF to images
        convert_kwargs = {'dpi': dpi}
        if max_pages:
            convert_kwargs['last_page'] = max_pages
        
        with self._performance_timer("pdf_conversion"):
            images = convert_from_path(pdf_path, **convert_kwargs)
        
        if not images:
            return {'text': '', 'confidence': 0.0, 'pages_processed': 0}
        
        # Process pages
        if parallel and len(images) > 1:
            results = self._process_pages_parallel(images, config)
        else:
            results = self._process_pages_sequential(images, config)
        
        # Aggregate results
        all_texts = []
        all_confidences = []
        engine_stats = {}
        
        for result in results:
            if result['text']:
                all_texts.append(result['text'])
                all_confidences.append(result['confidence'])
            
            engine = result['engine_used']
            engine_stats[engine] = engine_stats.get(engine, 0) + 1
        
        # Clean up images
        for img in images:
            img.close()
        del images
        gc.collect()
        
        # Calculate final metrics
        final_text = '\n\n'.join(all_texts)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        processing_time = time.time() - start_time
        
        # Update performance stats
        self._performance_stats['total_pages'] += len(results)
        self._performance_stats['total_time'] += processing_time
        
        return {
            'text': final_text,
            'confidence': avg_confidence,
            'pages_processed': len(results),
            'engine_stats': engine_stats,
            'processing_time': processing_time
        }
    
    def _process_pages_parallel(self, images: List[Image.Image], config: Dict) -> List[Dict]:
        """Process pages in parallel using ThreadPoolExecutor"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(self._process_single_page, img, config): i 
                for i, img in enumerate(images)
            }
            
            # Collect results in order
            page_results = [None] * len(images)
            
            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per page
                    page_results[page_idx] = result
                except Exception as e:
                    logger.error(f"Page {page_idx} processing failed: {e}")
                    page_results[page_idx] = {
                        'text': '',
                        'confidence': 0.0,
                        'engine_used': 'failed',
                        'fallback_used': False
                    }
            
            results = [r for r in page_results if r is not None]
        
        return results
    
    def _process_pages_sequential(self, images: List[Image.Image], config: Dict) -> List[Dict]:
        """Process pages sequentially"""
        results = []
        
        for i, img in enumerate(images):
            try:
                result = self._process_single_page(img, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Page {i} processing failed: {e}")
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'engine_used': 'failed',
                    'fallback_used': False
                })
        
        return results
    
    def _process_single_image(self, image_path: str, config: Dict) -> Dict:
        """Process a single image file"""
        start_time = time.time()
        
        try:
            image = Image.open(image_path)
            result = self._process_single_page(image, config)
            image.close()
            
            result['pages_processed'] = 1
            result['engine_stats'] = {result['engine_used']: 1}
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Single image processing failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'pages_processed': 0,
                'engine_stats': {},
                'processing_time': time.time() - start_time
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self._performance_stats.copy()
        
        if stats['total_pages'] > 0:
            stats['avg_time_per_page'] = stats['total_time'] / stats['total_pages']
            stats['fallback_rate_percent'] = (stats['fallback_rate'] / stats['total_pages']) * 100
        else:
            stats['avg_time_per_page'] = 0
            stats['fallback_rate_percent'] = 0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self._performance_stats = {
            'total_pages': 0,
            'total_time': 0,
            'engine_usage': {},
            'fallback_rate': 0
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