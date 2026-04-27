#!/usr/bin/env python3
"""
Quick fix for Tesseract language data issues
Run this script to diagnose and fix common Tesseract problems
"""

import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_tesseract_installation():
    """Check if Tesseract is properly installed"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Tesseract version: {result.stdout.split()[1]}")
            return True
        else:
            logger.error(f"Tesseract version check failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Tesseract not found or not working: {e}")
        return False


def check_tessdata_path():
    """Check and fix TESSDATA_PREFIX environment variable"""
    
    # Common tessdata locations
    possible_paths = [
        '/usr/share/tesseract-ocr/5/tessdata',
        '/usr/share/tesseract-ocr/4/tessdata', 
        '/usr/share/tesseract-ocr/tessdata',
        '/usr/local/share/tessdata',
        '/opt/homebrew/share/tessdata',  # macOS Homebrew
    ]
    
    current_tessdata = os.environ.get('TESSDATA_PREFIX')
    logger.info(f"Current TESSDATA_PREFIX: {current_tessdata}")
    
    # Find valid tessdata directory
    valid_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'eng.traineddata')):
            valid_path = path
            logger.info(f"Found valid tessdata at: {path}")
            break
    
    if not valid_path:
        logger.error("No valid tessdata directory found!")
        logger.info("Searching for eng.traineddata files...")
        
        # Search for eng.traineddata
        try:
            result = subprocess.run(['find', '/', '-name', 'eng.traineddata', '-type', 'f'], 
                                  capture_output=True, text=True, timeout=30)
            if result.stdout:
                files = result.stdout.strip().split('\n')
                logger.info(f"Found eng.traineddata files at: {files}")
                if files:
                    valid_path = os.path.dirname(files[0])
        except Exception as e:
            logger.warning(f"Search failed: {e}")
    
    if valid_path:
        logger.info(f"Setting TESSDATA_PREFIX to: {valid_path}")
        os.environ['TESSDATA_PREFIX'] = valid_path
        return valid_path
    else:
        logger.error("Could not find or set valid TESSDATA_PREFIX")
        return None


def check_available_languages():
    """Check what languages are available"""
    try:
        import pytesseract
        
        # Try to get available languages
        try:
            langs = pytesseract.get_languages()
            logger.info(f"Available languages: {langs}")
            
            if 'eng' in langs:
                logger.info("✅ English language data is available")
                return True
            else:
                logger.error("❌ English language data not found")
                return False
                
        except Exception as e:
            logger.error(f"Could not get language list: {e}")
            
            # Try a simple OCR test
            from PIL import Image
            import numpy as np
            
            # Create a simple test image with text
            img_array = np.ones((100, 300, 3), dtype=np.uint8) * 255
            test_img = Image.fromarray(img_array)
            
            try:
                text = pytesseract.image_to_string(test_img, lang='eng')
                logger.info("✅ Basic OCR test passed")
                return True
            except Exception as test_e:
                logger.error(f"❌ Basic OCR test failed: {test_e}")
                return False
                
    except ImportError:
        logger.error("pytesseract not installed")
        return False


def install_language_data():
    """Try to install English language data"""
    
    logger.info("Attempting to install Tesseract English language data...")
    
    # Try different package managers
    install_commands = [
        ['apt-get', 'update', '&&', 'apt-get', 'install', '-y', 'tesseract-ocr-eng'],
        ['yum', 'install', '-y', 'tesseract-langpack-eng'],
        ['dnf', 'install', '-y', 'tesseract-langpack-eng'],
        ['brew', 'install', 'tesseract-lang'],
    ]
    
    for cmd in install_commands:
        try:
            logger.info(f"Trying: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"✅ Installation successful with: {' '.join(cmd)}")
                return True
            else:
                logger.warning(f"Command failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Command failed: {e}")
    
    logger.error("❌ Could not install language data automatically")
    return False


def create_env_file():
    """Create .env file with proper Tesseract configuration"""
    
    tessdata_path = check_tessdata_path()
    
    env_content = f"""# OCR Configuration - Fixed for Tesseract issues
ENVIRONMENT=production
OCR_DEFAULT_ENGINE=safe
OCR_MAX_WORKERS=2
OCR_DEFAULT_DPI=150
OCR_MAX_PAGES=20

# Tesseract Configuration
TESSDATA_PREFIX={tessdata_path or '/usr/share/tesseract-ocr/5/tessdata'}
OMP_THREAD_LIMIT=1

# Disable problematic features
PYTHONFAULTHANDLER=true
"""
    
    env_file_path = '/opt/airflow/.env'
    try:
        with open(env_file_path, 'w') as f:
            f.write(env_content)
        logger.info(f"✅ Created {env_file_path} with Tesseract configuration")
        return True
    except Exception as e:
        logger.error(f"❌ Could not create .env file: {e}")
        return False


def main():
    """Main diagnostic and fix routine"""
    
    logger.info("🔧 Starting Tesseract diagnostic and fix...")
    
    # Step 1: Check Tesseract installation
    if not check_tesseract_installation():
        logger.error("❌ Tesseract is not properly installed")
        return False
    
    # Step 2: Check and fix tessdata path
    tessdata_path = check_tessdata_path()
    if not tessdata_path:
        logger.warning("⚠️ Could not find tessdata directory")
        
        # Try to install language data
        if install_language_data():
            tessdata_path = check_tessdata_path()
    
    # Step 3: Check available languages
    if not check_available_languages():
        logger.error("❌ Language check failed")
        
        # Try to install language data
        if install_language_data():
            check_available_languages()
    
    # Step 4: Create proper configuration
    create_env_file()
    
    # Step 5: Final test
    logger.info("🧪 Running final OCR test...")
    
    try:
        from ocr_services.safe_ocr_service import SafeOCRService
        
        ocr_service = SafeOCRService(enable_paddle_fallback=False)
        
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create test image with text
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a font, fall back to default if not available
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 30), "Hello World Test", fill='black', font=font)
        
        # Test OCR
        result = ocr_service._safe_tesseract_extract(img, {})
        
        if result['text'] and 'hello' in result['text'].lower():
            logger.info("✅ Final OCR test PASSED!")
            logger.info(f"Extracted text: '{result['text'].strip()}'")
            return True
        else:
            logger.warning(f"⚠️ OCR test returned: '{result['text']}'")
            return False
            
    except Exception as e:
        logger.error(f"❌ Final test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Tesseract fix completed successfully!")
        print("You can now run your OCR workflows.")
    else:
        print("\n❌ Some issues remain. Check the logs above.")
        print("You may need to manually install Tesseract language data.")
        
    sys.exit(0 if success else 1)