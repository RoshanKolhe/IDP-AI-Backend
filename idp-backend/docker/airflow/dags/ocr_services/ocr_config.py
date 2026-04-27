"""
OCR Configuration Management
Centralized configuration for OCR services with environment-based settings
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class OCRConfig:
    """Centralized OCR configuration management"""
    
    def __init__(self):
        # Environment-based configuration
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        
        # Default OCR engine selection
        self.default_engine = os.getenv("OCR_DEFAULT_ENGINE", "safe")
        
        # Performance settings
        self.max_workers = int(os.getenv("OCR_MAX_WORKERS", "4"))
        self.default_dpi = int(os.getenv("OCR_DEFAULT_DPI", "200"))
        self.max_pages_per_document = int(os.getenv("OCR_MAX_PAGES", "50"))
        
        # Engine-specific settings
        self.engine_configs = {
            "production": {
                "primary_engine": "safe",  # Use safe service for production
                "fallback_engine": None,
                "dpi": 150,  # Lower DPI for stability
                "max_workers": 2,  # Conservative worker count
                "parallel_processing": False,  # Sequential for stability
                "enable_performance_logging": True,
                "cache_enabled": True,
                "ai_cleanup_enabled": False,  # Disable for speed and stability
            },
            "development": {
                "primary_engine": "safe", 
                "fallback_engine": None,
                "dpi": 150,  # Lower DPI for faster development
                "max_workers": 1,  # Single worker for development
                "parallel_processing": False,
                "enable_performance_logging": True,
                "cache_enabled": True,
                "ai_cleanup_enabled": False,  # Keep simple for development
            },
            "testing": {
                "primary_engine": "safe",
                "fallback_engine": None,  # No fallback for faster tests
                "dpi": 100,  # Minimal DPI for testing
                "max_workers": 1,
                "parallel_processing": False,
                "enable_performance_logging": False,
                "cache_enabled": False,
                "ai_cleanup_enabled": False,
            }
        }
    
    def get_config(self, component: str = None, **overrides) -> Dict[str, Any]:
        """
        Get OCR configuration for a specific component or use case
        
        Args:
            component: Component name (e.g., 'classification', 'extraction', 'validation')
            **overrides: Override specific configuration values
        
        Returns:
            Configuration dictionary
        """
        # Start with environment-based base config
        base_config = self.engine_configs.get(self.environment, self.engine_configs["development"]).copy()
        
        # Apply component-specific overrides
        if component:
            component_config = self._get_component_config(component)
            base_config.update(component_config)
        
        # Apply any explicit overrides
        base_config.update({k: v for k, v in overrides.items() if v is not None})
        
        return base_config
    
    def _get_component_config(self, component: str) -> Dict[str, Any]:
        """Get component-specific configuration overrides"""
        
        component_configs = {
            "classification": {
                "dpi": 150,  # Lower DPI for faster classification
                "max_pages": 5,  # Only need first few pages for classification
                "parallel_processing": False,  # Sequential for stability
                "ai_cleanup_enabled": False,  # Speed over accuracy for classification
            },
            "extraction": {
                "dpi": 200,  # Higher DPI for better field extraction accuracy
                "max_pages": 10,  # Reasonable limit for extraction
                "parallel_processing": False,  # Sequential for stability
                "ai_cleanup_enabled": False,  # Keep simple for now
            },
            "validation": {
                "dpi": 150,
                "max_pages": 3,  # Usually validate specific pages
                "parallel_processing": False,  # Sequential for validation
                "ai_cleanup_enabled": False,
            },
            "full_processing": {
                "dpi": 200,  # Balanced quality
                "max_pages": 20,  # Reasonable limit
                "parallel_processing": False,  # Sequential for stability
                "ai_cleanup_enabled": False,
            }
        }
        
        return component_configs.get(component, {})
    
    def get_engine_name(self, component: str = None) -> str:
        """
        Get the appropriate OCR engine name for a component
        
        Args:
            component: Component name
        
        Returns:
            OCR engine name
        """
        config = self.get_config(component)
        
        # Use safe service by default
        primary_engine = config.get("primary_engine", "safe")
        
        # Map to actual engine names
        if primary_engine == "safe":
            return "safe"
        elif primary_engine == "tesseract":
            return "tesseract"
        elif primary_engine == "paddle":
            return "paddle"
        else:
            return self.default_engine
    
    def get_cache_config(self, component: str = None) -> Dict[str, Any]:
        """Get caching configuration"""
        config = self.get_config(component)
        
        return {
            "cache_enabled": config.get("cache_enabled", True),
            "force_refresh": False,
            "validate_cache": True,
        }
    
    def get_performance_config(self, component: str = None) -> Dict[str, Any]:
        """Get performance-related configuration"""
        config = self.get_config(component)
        
        return {
            "dpi": config.get("dpi", self.default_dpi),
            "max_workers": config.get("max_workers", self.max_workers),
            "parallel": config.get("parallel_processing", True),
            "max_pages": config.get("max_pages", self.max_pages_per_document),
            "enable_performance_logging": config.get("enable_performance_logging", True),
        }
    
    def should_use_ai_cleanup(self, component: str = None) -> bool:
        """Check if AI cleanup should be used for a component"""
        config = self.get_config(component)
        return config.get("ai_cleanup_enabled", False)


# Global configuration instance
ocr_config = OCRConfig()


def get_ocr_config(component: str = None, **overrides) -> Dict[str, Any]:
    """
    Convenience function to get OCR configuration
    
    Args:
        component: Component name
        **overrides: Configuration overrides
    
    Returns:
        OCR configuration dictionary
    """
    return ocr_config.get_config(component, **overrides)


def get_ocr_engine_name(component: str = None) -> str:
    """
    Convenience function to get OCR engine name
    
    Args:
        component: Component name
    
    Returns:
        OCR engine name
    """
    return ocr_config.get_engine_name(component)


def get_performance_config(component: str = None) -> Dict[str, Any]:
    """
    Convenience function to get performance configuration
    
    Args:
        component: Component name
    
    Returns:
        Performance configuration dictionary
    """
    return ocr_config.get_performance_config(component)


# Environment variable documentation
"""
Environment Variables for OCR Configuration:

ENVIRONMENT: Environment name (production, development, testing) - default: development
OCR_DEFAULT_ENGINE: Default OCR engine (safe, tesseract, paddle) - default: safe
OCR_MAX_WORKERS: Maximum parallel workers - default: 4
OCR_DEFAULT_DPI: Default image DPI - default: 150
OCR_MAX_PAGES: Maximum pages per document - default: 20

Example .env file for PRODUCTION (recommended):
ENVIRONMENT=production
OCR_DEFAULT_ENGINE=safe
OCR_MAX_WORKERS=2
OCR_DEFAULT_DPI=150
OCR_MAX_PAGES=20

Example .env file for DEVELOPMENT:
ENVIRONMENT=development
OCR_DEFAULT_ENGINE=safe
OCR_MAX_WORKERS=1
OCR_DEFAULT_DPI=150
OCR_MAX_PAGES=10
"""