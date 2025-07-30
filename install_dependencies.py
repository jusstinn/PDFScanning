#!/usr/bin/env python3
"""
Simple dependency installer for Document Text Scanner
Helps users install Tesseract OCR and Poppler automatically
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import tempfile
import shutil

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def install_tesseract_windows():
    """Install Tesseract on Windows."""
    print("Installing Tesseract OCR for Windows...")
    
    # Download Tesseract installer
    tesseract_url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe"
    installer_path = "tesseract-installer.exe"
    
    if download_file(tesseract_url, installer_path):
        print("Tesseract installer downloaded successfully!")
        print("Please run the installer manually:")
        print(f"  {os.path.abspath(installer_path)}")
        print("\nInstallation instructions:")
        print("1. Run the installer as Administrator")
        print("2. Install to: C:\\Program Files\\Tesseract-OCR")
        print("3. Make sure to install Romanian language pack")
        print("4. Add to PATH: C:\\Program Files\\Tesseract-OCR")
        return True
    else:
        print("Failed to download Tesseract installer.")
        print("Please download manually from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def install_poppler_windows():
    """Install Poppler on Windows."""
    print("Installing Poppler for Windows...")
    
    # Download Poppler
    poppler_url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.11.0-0/Release-23.11.0-0.zip"
    poppler_zip = "poppler-windows.zip"
    
    if download_file(poppler_url, poppler_zip):
        print("Poppler downloaded successfully!")
        
        # Extract to C:\poppler
        poppler_dir = r"C:\poppler"
        try:
            with zipfile.ZipFile(poppler_zip, 'r') as zip_ref:
                zip_ref.extractall(poppler_dir)
            print(f"Poppler extracted to: {poppler_dir}")
            print("Please add to PATH: C:\\poppler\\bin")
            return True
        except Exception as e:
            print(f"Error extracting Poppler: {e}")
            return False
    else:
        print("Failed to download Poppler.")
        print("Please download manually from: https://github.com/oschwartz10612/poppler-windows/releases")
        return False

def main():
    """Main installation function."""
    print("Document Text Scanner - Dependency Installer")
    print("=" * 50)
    
    system = platform.system().lower()
    
    if system == "windows":
        print("Detected Windows system")
        
        # Install Tesseract
        print("\n1. Installing Tesseract OCR...")
        install_tesseract_windows()
        
        # Install Poppler
        print("\n2. Installing Poppler...")
        install_poppler_windows()
        
        print("\nInstallation completed!")
        print("Please restart your terminal and run the app again.")
        
    elif system == "darwin":  # macOS
        print("Detected macOS system")
        print("Please install dependencies using Homebrew:")
        print("  brew install tesseract tesseract-lang poppler")
        
    elif system == "linux":
        print("Detected Linux system")
        print("Please install dependencies using your package manager:")
        print("  sudo apt-get install tesseract-ocr tesseract-ocr-ron poppler-utils")
        
    else:
        print(f"Unsupported system: {system}")
        print("Please install Tesseract OCR and Poppler manually.")

if __name__ == "__main__":
    main() 