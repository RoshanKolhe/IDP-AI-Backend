"""
Optimized OCR Cache Utilities
Enhanced caching with performance monitoring and batch processing
"""

import json
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from .ocr_service_factory import get_ocr_service
from .optimized_ocr_service import OptimizedOCRService

logger = logging.getLogger(__name__)

OCR_OUTPUT_DIRNAME = "ocr_output"


def get_process_instance_dir(local_download_dir: str, process_instance_id: int) -> str:
    return os.path.join(local_download_dir, f"process-instance-{process_instance_id}")


def get_ocr_output_dir(process_instance_dir: str) -> str:
    output_dir = os.path.join(process_instance_dir, OCR_OUTPUT_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_ocr_cache_path(process_instance_dir: str, pdf_filename: str) -> str:
    base_filename = os.path.splitext(os.path.basename(pdf_filename))[0]
    return os.path.join(get_ocr_output_dir(process_instance_dir), f"{base_filename}.json")


def get_config_hash(config: Dict) -> str:
    """Generate hash for OCR configuration to detect changes"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def load_ocr_cache(cache_path: str) -> Optional[Dict]:
    """Load OCR cache with validation"""
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            cache_data = json.load(file)
        
        # Validate cache structure
        required_fields = ['filename', 'ocr_engine', 'page_count', 'pages']
        if not all(field in cache_data for field in required_fields):
            logger.warning(f"Invalid cache structure in {cache_path}")
            return None
        
        return cache_data
    
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load cache from {cache_path}: {e}")
        return None


def is_cache_valid(cache_data: Dict, config: Dict, pdf_path: str) -> bool:
    """
    Check if cached OCR data is still valid
    
    Args:
        cache_data: Cached OCR data
        config: Current OCR configuration
        pdf_path: Path to PDF file
    
    Returns:
        True if cache is valid, False otherwise
    """
    try:
        # Check if file has been modified
        pdf_mtime = os.path.getmtime(pdf_path)
        cache_time = datetime.fromisoformat(cache_data.get('processed_at', '1970-01-01'))
        
        if pdf_mtime > cache_time.timestamp():
            return False
        
        # Check if configuration has changed significantly
        cached_config = cache_data.get('config', {})
        current_config_hash = get_config_hash(config)
        cached_config_hash = get_config_hash(cached_config)
        
        if current_config_hash != cached_config_hash:
            # Allow minor config changes (like thread_count)
            significant_keys = ['ocr_engine', 'language_mode', 'dpi', 'max_pages']
            for key in significant_keys:
                if config.get(key) != cached_config.get(key):
                    return False
        
        return True
    
    except Exception as e:
        logger.warning(f"Cache validation failed: {e}")
        return False


def ensure_optimized_ocr_cache(
    pdf_path: str,
    process_instance_dir: str,
    ocr_engine: str = "optimized",
    config: Optional[Dict] = None,
    cleanup_service=None,
    force_refresh: bool = False,
    logger_callback=None,
) -> Dict:
    """
    Ensure OCR cache exists with optimized processing
    
    Args:
        pdf_path: Path to PDF file
        process_instance_dir: Process instance directory
        ocr_engine: OCR engine to use ('optimized', 'optimized_paddle', etc.)
        config: OCR configuration
        cleanup_service: Optional text cleanup service
        force_refresh: Force cache refresh
        logger_callback: Optional logging callback
    
    Returns:
        OCR cache payload
    """
    pdf_filename = os.path.basename(pdf_path)
    cache_path = get_ocr_cache_path(process_instance_dir, pdf_filename)
    config = config or {}
    
    # Try to load existing cache
    if not force_refresh:
        existing_cache = load_ocr_cache(cache_path)
        if existing_cache and is_cache_valid(existing_cache, config, pdf_path):
            if logger_callback:
                logger_callback(
                    "info",
                    f"Using valid cached OCR for {pdf_filename}: "
                    f"pages={existing_cache.get('page_count', 0)}, "
                    f"engine={existing_cache.get('ocr_engine', 'unknown')}"
                )
            return existing_cache
    
    # Process with optimized OCR service
    start_time = time.time()
    
    if logger_callback:
        logger_callback("info", f"Starting optimized OCR processing for {pdf_filename}")
    
    try:
        # Get optimized OCR service
        ocr_service = get_ocr_service(ocr_engine)
        
        # Handle different service types
        if hasattr(ocr_service, 'extract_text_with_confidence'):
            # For OptimizedOCRService and SafeOCRService
            result = ocr_service.extract_text_with_confidence(pdf_path, optimized_config)
        else:
            # Fallback to regular processing for other services
            return _fallback_to_regular_processing(
                pdf_path, process_instance_dir, ocr_engine, config, 
                cleanup_service, logger_callback
            )
        
        # Get performance stats
        perf_stats = ocr_service.get_performance_stats()
        
        # Build cache payload
        cache_payload = _build_optimized_cache_payload(
            pdf_filename=pdf_filename,
            ocr_engine=ocr_engine,
            result=result,
            config=optimized_config,
            performance_stats=perf_stats,
            cleanup_service=cleanup_service
        )
        
        # Save cache
        save_ocr_cache(cache_path, cache_payload)
        
        processing_time = time.time() - start_time
        
        if logger_callback:
            logger_callback(
                "success",
                f"Optimized OCR completed for {pdf_filename}: "
                f"pages={result.get('pages_processed', 0)}, "
                f"time={processing_time:.2f}s, "
                f"engines={result.get('engine_stats', {})}"
            )
        
        return cache_payload
    
    except Exception as e:
        error_msg = f"Optimized OCR processing failed for {pdf_filename}: {e}"
        logger.error(error_msg)
        
        if logger_callback:
            logger_callback("error", error_msg)
        
        # Try fallback to regular processing
        try:
            return _fallback_to_regular_processing(
                pdf_path, process_instance_dir, "tesseract", config,
                cleanup_service, logger_callback
            )
        except Exception as fallback_error:
            logger.error(f"Fallback processing also failed: {fallback_error}")
            return _create_empty_cache_payload(pdf_filename, ocr_engine, config)


def _prepare_optimized_config(config: Dict) -> Dict:
    """Prepare configuration for optimized OCR processing"""
    optimized_config = config.copy()
    
    # Set performance-optimized defaults
    optimized_config.setdefault('dpi', 200)  # Balanced quality/speed
    optimized_config.setdefault('parallel', True)  # Enable parallel processing
    optimized_config.setdefault('max_pages', 50)  # Reasonable limit
    
    # Adjust thread count based on available cores
    import multiprocessing
    max_workers = min(4, multiprocessing.cpu_count())
    optimized_config.setdefault('max_workers', max_workers)
    
    return optimized_config


def _build_optimized_cache_payload(
    pdf_filename: str,
    ocr_engine: str,
    result: Dict,
    config: Dict,
    performance_stats: Dict,
    cleanup_service=None
) -> Dict:
    """Build cache payload from optimized OCR result"""
    
    raw_text = result.get('text', '')
    cleaned_text = raw_text
    
    # Apply text cleanup if available
    if cleanup_service and raw_text:
        try:
            cleaned_text = cleanup_service.cleanup_text(raw_text)
        except Exception as e:
            logger.warning(f"Text cleanup failed: {e}")
            cleaned_text = raw_text
    
    # Create page results for compatibility
    pages_processed = result.get('pages_processed', 0)
    avg_confidence = result.get('confidence', 0)
    
    page_results = []
    if pages_processed > 0 and raw_text:
        # Split text into approximate pages for compatibility
        text_parts = raw_text.split('\n\n') if '\n\n' in raw_text else [raw_text]
        pages_per_part = max(1, pages_processed // len(text_parts))
        
        for i, text_part in enumerate(text_parts):
            page_results.append({
                'page_number': i + 1,
                'text': text_part,
                'cleaned_text': text_part,  # Will be cleaned later if needed
                'confidence': avg_confidence,
                'character_count': len(text_part)
            })
    
    return {
        'filename': pdf_filename,
        'ocr_engine': ocr_engine,
        'config': config,
        'page_count': pages_processed,
        'pages': page_results,
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'processed_at': datetime.utcnow().isoformat(),
        'performance_stats': performance_stats,
        'processing_time': result.get('processing_time', 0),
        'engine_stats': result.get('engine_stats', {}),
        'optimized': True  # Flag to indicate optimized processing
    }


def _fallback_to_regular_processing(
    pdf_path: str,
    process_instance_dir: str,
    ocr_engine: str,
    config: Dict,
    cleanup_service=None,
    logger_callback=None
) -> Dict:
    """Fallback to regular OCR processing"""
    
    if logger_callback:
        logger_callback("warning", f"Falling back to regular OCR processing")
    
    # Import the original function
    from .ocr_cache_utils import ensure_ocr_cache
    
    return ensure_ocr_cache(
        pdf_path=pdf_path,
        process_instance_dir=process_instance_dir,
        ocr_engine=ocr_engine,
        config=config,
        cleanup_service=cleanup_service,
        force_refresh=False,
        logger_callback=logger_callback
    )


def _create_empty_cache_payload(pdf_filename: str, ocr_engine: str, config: Dict) -> Dict:
    """Create empty cache payload for failed processing"""
    return {
        'filename': pdf_filename,
        'ocr_engine': ocr_engine,
        'config': config,
        'page_count': 0,
        'pages': [],
        'raw_text': '',
        'cleaned_text': '',
        'processed_at': datetime.utcnow().isoformat(),
        'performance_stats': {},
        'processing_time': 0,
        'engine_stats': {},
        'optimized': True,
        'error': 'Processing failed'
    }


def save_ocr_cache(cache_path: str, payload: Dict) -> None:
    """Save OCR cache with error handling"""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Create backup if cache exists
        if os.path.exists(cache_path):
            backup_path = f"{cache_path}.backup"
            os.rename(cache_path, backup_path)
        
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        
        # Remove backup on successful write
        backup_path = f"{cache_path}.backup"
        if os.path.exists(backup_path):
            os.remove(backup_path)
            
    except Exception as e:
        logger.error(f"Failed to save OCR cache to {cache_path}: {e}")
        
        # Restore backup if available
        backup_path = f"{cache_path}.backup"
        if os.path.exists(backup_path):
            os.rename(backup_path, cache_path)


def get_cached_document_text(process_instance_dir: str, pdf_filename: str) -> Optional[str]:
    """Get cached document text with fallback to cleaned text"""
    cache_path = get_ocr_cache_path(process_instance_dir, pdf_filename)
    cache_data = load_ocr_cache(cache_path)
    
    if not cache_data:
        return None
    
    # Prefer cleaned text, fallback to raw text
    return cache_data.get('cleaned_text') or cache_data.get('raw_text') or ''


def get_cached_page_texts(process_instance_dir: str, pdf_filename: str) -> List[str]:
    """Get cached page texts"""
    cache_path = get_ocr_cache_path(process_instance_dir, pdf_filename)
    cache_data = load_ocr_cache(cache_path)
    
    if not cache_data:
        return []
    
    pages = cache_data.get('pages', [])
    return [page.get('cleaned_text') or page.get('text', '') for page in pages]


def get_cache_performance_stats(process_instance_dir: str, pdf_filename: str) -> Dict:
    """Get performance statistics from cache"""
    cache_path = get_ocr_cache_path(process_instance_dir, pdf_filename)
    cache_data = load_ocr_cache(cache_path)
    
    if not cache_data:
        return {}
    
    return cache_data.get('performance_stats', {})