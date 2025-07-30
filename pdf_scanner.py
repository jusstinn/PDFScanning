import os
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
import tempfile
import json
import platform
import subprocess

class DocumentScanner:
    def __init__(self):
        """Initialize the Document Scanner with OCR capabilities."""
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.tesseract_path = self.find_tesseract()
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
    
    def find_tesseract(self) -> str:
        """
        Automatically find Tesseract installation on the system.
        Returns the path to tesseract executable or None if not found.
        """
        system = platform.system().lower()
        
        # Common installation paths
        possible_paths = []
        
        if system == "windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
                r"C:\tesseract\tesseract.exe",
                "tesseract.exe"  # If it's in PATH
            ]
        elif system == "darwin":  # macOS
            possible_paths = [
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
                "/usr/bin/tesseract"
            ]
        else:  # Linux
            possible_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/opt/tesseract/bin/tesseract"
            ]
        
        # Check each possible path
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Test if it's executable
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return path
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
        
        return None
        
    def load_document(self, file_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Load document (PDF or image) and convert to PIL Images.
        
        Args:
            file_path: Path to the file (PDF or image)
            dpi: Resolution for PDF conversion (ignored for images)
            
        Returns:
            List of PIL Image objects
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                # Check if Poppler is available for PDF processing
                if not self.check_poppler():
                    st.error("Poppler is required for PDF processing. Please install Poppler or use image files.")
                    return []
                
                # Convert PDF to images
                images = convert_from_path(file_path, dpi=dpi)
                return images
            else:
                # Load single image
                image = Image.open(file_path)
                return [image]
                
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            return []
    
    def check_poppler(self) -> bool:
        """
        Check if Poppler is available for PDF processing.
        Returns True if Poppler is found, False otherwise.
        """
        try:
            # Try to import pdf2image and check if poppler is available
            from pdf2image.pdf2image import check_poppler_version
            check_poppler_version()
            return True
        except Exception:
            return False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if it's RGB
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_from_image(self, image: Image.Image, lang: str = 'eng') -> str:
        """
        Extract text from a single image using OCR.
        
        Args:
            image: PIL Image to extract text from
            lang: Language for OCR (default: English)
            
        Returns:
            Extracted text as string
        """
        try:
            # Check if Tesseract is available
            if not self.tesseract_path:
                st.error("Tesseract OCR is not installed or not found. Please install Tesseract OCR.")
                return ""
            
            # Preprocess the image
            processed_img = self.preprocess_image(image)
            
            # Configure OCR parameters for better text extraction
            # PSM 6: Assume a uniform block of text
            # OEM 3: Default OCR Engine Mode
            # Add character recognition mode for better results
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-_+=@#$%&*<>/|\\'
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(processed_img, lang=lang, config=custom_config)
            
            # Clean up the extracted text
            cleaned_text = self.clean_extracted_text(text)
            
            return cleaned_text
        except Exception as e:
            st.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean and format extracted text for better readability.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Preserve important whitespace for codes and formatting
            # Replace multiple spaces with single space, but keep structure
            line = ' '.join(line.split())
            
            cleaned_lines.append(line)
        
        # Join lines back together
        cleaned_text = '\n'.join(cleaned_lines)
        
        return cleaned_text.strip()
    
    def process_document(self, file_path: str, lang: str = 'eng') -> Dict[str, any]:
        """
        Process a document (PDF or image) and extract text.
        
        Args:
            file_path: Path to the document file
            lang: Language for OCR
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        results = {
            'filename': os.path.basename(file_path),
            'pages': [],
            'total_pages': 0,
            'full_text': '',
            'processing_time': 0,
            'file_type': os.path.splitext(file_path)[1].lower()
        }
        
        try:
            # Load document (PDF or image)
            images = self.load_document(file_path)
            results['total_pages'] = len(images)
            
            if not images:
                return results
            
            # Process each page/image
            for i, image in enumerate(images):
                page_text = self.extract_text_from_image(image, lang)
                results['pages'].append({
                    'page_number': i + 1,
                    'text': page_text,
                    'word_count': len(page_text.split())
                })
                results['full_text'] += f"\n--- Page {i + 1} ---\n{page_text}"
            
            return results
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return results
    
    def save_results(self, results: Dict[str, any], output_path: str):
        """
        Save extraction results to a file.
        
        Args:
            results: Results dictionary from process_pdf
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving results: {str(e)}")

def main():
    """Main function to run the Document Scanner application."""
    st.set_page_config(
        page_title="Document Text Scanner",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Text Scanner")
    st.markdown("Extract text from scanned PDFs and images using OCR technology")
    
    # Initialize scanner
    scanner = DocumentScanner()
    
    # Show system status
    st.sidebar.header("System Status")
    
    # Check Tesseract status
    if scanner.tesseract_path:
        st.sidebar.success("✅ Tesseract OCR: Available")
    else:
        st.sidebar.error("❌ Tesseract OCR: Not Found")
        st.sidebar.info("Install Tesseract OCR for text extraction")
    
    # Check Poppler status
    if scanner.check_poppler():
        st.sidebar.success("✅ Poppler: Available (PDF support)")
    else:
        st.sidebar.warning("⚠️ Poppler: Not Found (PDF processing disabled)")
        st.sidebar.info("Install Poppler for PDF support")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    language = st.sidebar.selectbox(
        "OCR Language",
        ["eng", "eng+ron", "ron", "fra", "deu", "spa", "ita", "por", "rus", "chi_sim", "jpn"],
        help="Select the language of the text in your PDF (eng for best character recognition)"
    )
    
    dpi = st.sidebar.slider(
        "Image Resolution (DPI)",
        min_value=150,
        max_value=600,
        value=300,
        step=50,
        help="Higher DPI = better quality but slower processing"
    )
    
    # File upload
    st.header("Upload Document")
    
    # Show available file types based on system capabilities
    available_types = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    if scanner.check_poppler():
        available_types.insert(0, 'pdf')
    
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=available_types,
        help=f"Upload a scanned document. Available formats: {', '.join(available_types)}"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Process button
            if st.button("🔍 Extract Text", type="primary"):
                with st.spinner("Processing document..."):
                    # Process the document
                    results = scanner.process_document(file_path, language)
                
                if results['total_pages'] > 0:
                    st.success(f"✅ Successfully processed {results['total_pages']} pages!")
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Summary")
                        st.metric("Total Pages", results['total_pages'])
                        st.metric("Total Words", sum(page['word_count'] for page in results['pages']))
                        
                        # Page-by-page breakdown
                        st.subheader("📄 Page Details")
                        page_data = []
                        for page in results['pages']:
                            page_data.append({
                                'Page': page['page_number'],
                                'Words': page['word_count'],
                                'Characters': len(page['text'])
                            })
                        
                        df = pd.DataFrame(page_data)
                        st.dataframe(df, use_container_width=True)
                    
                    with col2:
                        st.subheader("📝 Extracted Text")
                        
                        # Full text display
                        st.text_area(
                            "Complete Text",
                            value=results['full_text'],
                            height=400,
                            help="Complete extracted text from all pages"
                        )
                        
                        # Download options
                        st.subheader("💾 Download Results")
                        
                        # Download as JSON
                        json_str = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name=f"{results['filename']}_extracted.json",
                            mime="application/json"
                        )
                        
                        # Download as TXT
                        st.download_button(
                            label="Download as TXT",
                            data=results['full_text'],
                            file_name=f"{results['filename']}_extracted.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("❌ No pages were processed. Please check your document file.")
                    
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions:
        1. **Upload a Document**: Click 'Browse files' and select your scanned PDF or image
        2. **Configure Settings**: 
           - Choose the language of your text
           - Adjust DPI if needed (higher = better quality, slower processing)
        3. **Extract Text**: Click the 'Extract Text' button
        4. **Review Results**: View the extracted text and download if needed
        
        ### Supported File Types:
        - **PDF files**: Multi-page documents
        - **Images**: JPG, JPEG, PNG, BMP, TIFF, TIF
        
        ### Tips for Better Results:
        - Ensure your document has good image quality
        - Use higher DPI for better accuracy (PDFs only)
        - Choose the correct language for your text
        - For mixed content (codes + Romanian text), use "eng+ron"
        - Clean, well-lit scans work best
        - The app preserves code formatting and structure
        
        ### Supported Languages:
        - English (eng)
        - Romanian (ron)
        - English + Romanian (eng+ron) - for mixed content
        - French (fra)
        - German (deu)
        - Spanish (spa)
        - Italian (ita)
        - Portuguese (por)
        - Russian (rus)
        - Chinese Simplified (chi_sim)
        - Japanese (jpn)
        """)

if __name__ == "__main__":
    main() 