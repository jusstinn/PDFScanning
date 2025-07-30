#!/usr/bin/env python3
"""
Setup script for PDF Text Scanner
Helps install dependencies and verify system requirements
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies: {e}")
        return False

def check_tesseract():
    """Check if Tesseract is installed."""
    try:
        result = subprocess.run(["tesseract", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Tesseract is installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Tesseract not found")
    return False

def check_poppler():
    """Check if Poppler is installed."""
    try:
        result = subprocess.run(["pdftoppm", "-h"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Poppler is installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Poppler not found")
    return False

def get_system_info():
    """Get system information for installation guidance."""
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    else:
        return "unknown"

def print_installation_instructions(system):
    """Print installation instructions for missing components."""
    print("\n📋 Installation Instructions:")
    
    if system == "windows":
        print("\n🔧 For Windows:")
        print("1. Install Tesseract:")
        print("   - Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Add to PATH: C:\\Program Files\\Tesseract-OCR")
        print("   - Make sure to install Romanian language pack for better results")
        print("\n2. Install Poppler:")
        print("   - Download from: http://blog.alivate.com.au/poppler-windows/")
        print("   - Extract to C:\\poppler")
        print("   - Add to PATH: C:\\poppler\\bin")
        
    elif system == "macos":
        print("\n🔧 For macOS:")
        print("1. Install Homebrew (if not installed):")
        print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("\n2. Install Tesseract:")
        print("   brew install tesseract")
        print("   brew install tesseract-lang  # For additional language packs")
        print("\n3. Install Poppler:")
        print("   brew install poppler")
        
    elif system == "linux":
        print("\n🔧 For Linux (Ubuntu/Debian):")
        print("1. Install Tesseract:")
        print("   sudo apt-get update")
        print("   sudo apt-get install tesseract-ocr")
        print("   sudo apt-get install tesseract-ocr-ron  # Romanian language pack")
        print("\n2. Install Poppler:")
        print("   sudo apt-get install poppler-utils")
        
    print("\n🔄 After installation, restart your terminal and run this setup script again.")

def test_installation():
    """Test if the installation works correctly."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        from PIL import Image
        import pytesseract
        from pdf2image import convert_from_path
        import streamlit
        print("✅ All Python packages imported successfully")
        
        # Test Tesseract
        if check_tesseract():
            print("✅ Tesseract is working")
        else:
            print("❌ Tesseract test failed")
            return False
            
        # Test Poppler
        if check_poppler():
            print("✅ Poppler is working")
        else:
            print("❌ Poppler test failed")
            return False
            
        print("🎉 Installation test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 PDF Text Scanner Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        sys.exit(1)
    
    # Check system components
    system = get_system_info()
    print(f"\n💻 System: {system}")
    
    tesseract_ok = check_tesseract()
    poppler_ok = check_poppler()
    
    if not tesseract_ok or not poppler_ok:
        print_installation_instructions(system)
        print("\n⚠️  Please install the missing components and run this script again.")
        return
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("1. Run the web interface: streamlit run pdf_scanner.py")
        print("2. Or use command line: python cli_scanner.py document.pdf")
        print("\n📚 See README.md for detailed usage instructions")
    else:
        print("\n❌ Setup test failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 