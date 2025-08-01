import os
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Any
import tempfile
import json
import platform
import subprocess

# Optional OCR engine imports - set to None initially, import only when needed
easyocr = None
keras_ocr = None
TrOCRProcessor = None
VisionEncoderDecoderModel = None

# Cloud OCR services - import only when needed
azure_cv_client = None
google_vision_client = None

class DocumentScanner:
    def __init__(self, default_engine: str = "tesseract", preprocess_settings: Dict[str, Any] | None = None):
        """Initialize the Document Scanner with OCR capabilities.

        Args:
            default_engine: OCR engine to use if none is specified when calling processing functions.
            preprocess_settings: Optional dictionary with preprocessing flags (see ``preprocess_image`` for keys).
        """
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # Default OCR engine
        self.default_engine = default_engine.lower()

        # Tesseract setup (if available)
        self.tesseract_path = self.find_tesseract()
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # Lazy-loaded engine objects (they will be created on first use)
        self._easyocr_reader = None
        self._keras_pipeline = None
        self._trocr_processor = None
        self._trocr_model = None
        
        # Cloud OCR clients (lazy-loaded)
        self._azure_client = None
        self._google_client = None

        # Pre-processing configuration (flags)
        self.preprocess_settings = preprocess_settings or {
            "grayscale": True,
            "resize_factor": 2,
            "denoise": True,
            "thresholding": "otsu",  # 'otsu', 'gaussian', or None
            "clahe": False,
        }
    
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
    
    def preprocess_image(
        self,
        image: Image.Image,
        grayscale: bool | None = None,
        resize_factor: int | float | None = None,
        denoise: bool | None = None,
        thresholding: str | None = None,
        clahe: bool | None = None,
    ) -> np.ndarray:
        """Comprehensive image preprocessing pipeline for OCR.

        All parameters are optional. If a parameter is **None**, the value from
        :pyattr:`self.preprocess_settings` is used. This makes it possible to
        configure defaults at *DocumentScanner* creation and still override
        them per-call if needed.
        
        Args:
            image: Source :class:`PIL.Image.Image`.
            grayscale: Convert to grayscale.
            resize_factor: Scale factor for *cv2.resize* using **INTER_CUBIC**.
            denoise: Apply *medianBlur* (kernel size 3).
            thresholding: "otsu", "gaussian", or ``None``.
            clahe: Apply CLAHE contrast enhancement (useful for handwriting).
            
        Returns:
            Pre-processed image as a NumPy array ready for OCR.
        """

        # Fallback to configured defaults when parameters are omitted
        settings = {
            "grayscale": grayscale if grayscale is not None else self.preprocess_settings.get("grayscale", True),
            "resize_factor": resize_factor if resize_factor is not None else self.preprocess_settings.get("resize_factor", 2),
            "denoise": denoise if denoise is not None else self.preprocess_settings.get("denoise", True),
            "thresholding": thresholding if thresholding is not None else self.preprocess_settings.get("thresholding", "otsu"),
            "clahe": clahe if clahe is not None else self.preprocess_settings.get("clahe", False),
        }

        # Convert PIL image to OpenCV array
        img = np.array(image)

        # 1. Grayscale conversion
        if settings["grayscale"] and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Resize (upsample) – helps with low-res scans / handwriting
        if settings["resize_factor"] and settings["resize_factor"] > 1:
            new_size = (int(img.shape[1] * settings["resize_factor"]), int(img.shape[0] * settings["resize_factor"]))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

        # 3. CLAHE (contrast limited adaptive histogram equalization)
        if settings["clahe"]:
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe_obj.apply(img)

        # 4. Denoise (median blur)
        if settings["denoise"]:
            img = cv2.medianBlur(img, 3)

        # 5. Thresholding – convert to binary
        if settings["thresholding"] == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif settings["thresholding"] == "gaussian":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 2)

        # 6. Morphological close to remove small holes and connect text regions
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return img
    
    def extract_text_from_image(
        self,
        image: Image.Image,
        lang: str = 'eng',
        engine: str | None = None,
    ) -> str:
        """
        Extract text from a single image using OCR.
        
        Args:
            image: PIL Image to extract text from
            lang: Language for OCR (default: English)
            
        Returns:
            Extracted text as string
        """
        try:

            selected_engine = (engine or self.default_engine or "tesseract").lower()

            # Preprocess the image once (some engines work better on RGB so keep original too)
            processed_img = self.preprocess_image(image)

            # -----------------------------
            # 1) Tesseract
            # -----------------------------
            if selected_engine == "tesseract":
                if not self.tesseract_path:
                    st.error("Tesseract OCR is not installed or not found. Please install Tesseract OCR.")
                    return ""

                custom_config = (
                    r"--oem 3 --psm 6 -c preserve_interword_spaces=1 "
                    r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}\"'-_+=@#$%&*<>/|\\"
                )
                text = pytesseract.image_to_string(processed_img, lang=lang, config=custom_config)
            
            # -----------------------------
            # 2) EasyOCR
            # -----------------------------
            elif selected_engine == "easyocr":
                try:
                    global easyocr
                    if easyocr is None:
                        import easyocr
                except ImportError:
                    st.error("EasyOCR is not installed. Install with 'pip install easyocr'.")
                    return ""
                if self._easyocr_reader is None:
                    # Use lang split by + and limit to available easyocr codes
                    langs = [l for l in lang.split("+") if l]
                    self._easyocr_reader = easyocr.Reader(langs, gpu=False)
                result = self._easyocr_reader.readtext(np.array(image))
                text = "\n".join([res[1] for res in result])

            # -----------------------------
            # 3) Keras-OCR
            # -----------------------------
            elif selected_engine == "keras":
                try:
                    global keras_ocr
                    if keras_ocr is None:
                        import keras_ocr
                except ImportError:
                    st.error("Keras-OCR is not installed. Install with 'pip install keras-ocr'.")
                    return ""
                except Exception as e:
                    st.error(f"Keras-OCR compatibility issue: {str(e)}. Try downgrading NumPy: 'pip install numpy<2.0'")
                    return ""
                if self._keras_pipeline is None:
                    self._keras_pipeline = keras_ocr.pipeline.Pipeline()
                prediction_groups = self._keras_pipeline.recognize([np.array(image)])
                text_lines = [" ".join([word[0] for word in line]) for line in prediction_groups[0]]
                text = "\n".join(text_lines)

            # -----------------------------
            # 4) TrOCR (Transformer OCR)
            # -----------------------------
            elif selected_engine == "trocr":
                try:
                    global TrOCRProcessor, VisionEncoderDecoderModel
                    if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
                        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                except ImportError:
                    st.error("TrOCR dependencies not installed. Install with 'pip install transformers sentencepiece'.")
                    return ""
                if self._trocr_model is None:
                    with st.spinner("Loading TrOCR model (first time only)..."):
                        self._trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                        self._trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                pixel_values = self._trocr_processor(images=image, return_tensors="pt").pixel_values
                generated_ids = self._trocr_model.generate(pixel_values)
                text = self._trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # -----------------------------
            # 5) Microsoft Azure Computer Vision OCR
            # -----------------------------
            elif selected_engine == "azure":
                try:
                    global azure_cv_client
                    if azure_cv_client is None:
                        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
                        from msrest.authentication import CognitiveServicesCredentials
                        azure_cv_client = True  # Mark as imported
                except ImportError:
                    st.error("Azure Computer Vision SDK not installed. Install with 'pip install azure-cognitiveservices-vision-computervision'.")
                    return ""
                
                # Check for Azure credentials
                azure_key = os.getenv('AZURE_CV_KEY')
                azure_endpoint = os.getenv('AZURE_CV_ENDPOINT')
                
                if not azure_key or not azure_endpoint:
                    st.error("Azure credentials not found. Please set AZURE_CV_KEY and AZURE_CV_ENDPOINT environment variables.")
                    st.info("Get your credentials from: https://portal.azure.com/ → Create Computer Vision resource")
                    return ""
                
                try:
                    if self._azure_client is None:
                        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                        from msrest.authentication import CognitiveServicesCredentials
                        self._azure_client = ComputerVisionClient(azure_endpoint, CognitiveServicesCredentials(azure_key))
                    
                    # Convert PIL image to bytes
                    import io
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    # Call Azure OCR
                    with st.spinner("Processing with Azure OCR..."):
                        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
                        read_response = self._azure_client.read_in_stream(img_byte_arr, raw=True)
                        operation_id = read_response.headers["Operation-Location"].split("/")[-1]
                        
                        # Wait for result
                        import time
                        while True:
                            read_result = self._azure_client.get_read_result(operation_id)
                            if read_result.status not in ['notStarted', 'running']:
                                break
                            time.sleep(1)
                        
                        # Extract text
                        text_lines = []
                        if read_result.status == OperationStatusCodes.succeeded:
                            for text_result in read_result.analyze_result.read_results:
                                for line in text_result.lines:
                                    text_lines.append(line.text)
                        text = "\n".join(text_lines)
                except Exception as e:
                    st.error(f"Azure OCR error: {str(e)}")
                    return ""

            # -----------------------------
            # 6) Google Cloud Vision OCR
            # -----------------------------
            elif selected_engine == "google":
                try:
                    global google_vision_client
                    if google_vision_client is None:
                        from google.cloud import vision
                        google_vision_client = True  # Mark as imported
                except ImportError:
                    st.error("Google Cloud Vision SDK not installed. Install with 'pip install google-cloud-vision'.")
                    return ""
                
                # Check for Google credentials
                google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if not google_creds:
                    st.error("Google Cloud credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
                    st.info("Get your credentials from: https://console.cloud.google.com/ → Create Vision API service account key")
                    return ""
                
                try:
                    if self._google_client is None:
                        from google.cloud import vision
                        self._google_client = vision.ImageAnnotatorClient()
                    
                    # Convert PIL image to bytes
                    import io
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    # Call Google Vision OCR
                    with st.spinner("Processing with Google Cloud Vision..."):
                        from google.cloud import vision
                        google_image = vision.Image(content=img_byte_arr.getvalue())
                        response = self._google_client.text_detection(image=google_image)
                        
                        if response.error.message:
                            st.error(f"Google Vision API error: {response.error.message}")
                            return ""
                        
                        # Extract text
                        texts = response.text_annotations
                        if texts:
                            text = texts[0].description  # First annotation contains full text
                        else:
                            text = ""
                except Exception as e:
                    st.error(f"Google Cloud Vision error: {str(e)}")
                    return ""

            else:
                st.error(f"Unsupported OCR engine: {selected_engine}")
                return ""

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
    
    def process_document(self, file_path: str, lang: str = 'eng', engine: str | None = None) -> Dict[str, any]:
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
                results_page_text = self.extract_text_from_image(image, lang=lang, engine=engine)
                results['pages'].append({
                    'page_number': i + 1,
                    'text': results_page_text,
                    'word_count': len(results_page_text.split())
                })
                results['full_text'] += f"\n--- Page {i + 1} ---\n{results_page_text}"
            
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

    ocr_engine = st.sidebar.selectbox(
        "OCR Engine",
        ["tesseract", "easyocr", "keras", "trocr", "azure", "google"],
        help="Choose which OCR backend to use (cloud engines require API credentials)"
    )

    language = st.sidebar.selectbox(
        "OCR Language",
        ["eng", "eng+ron", "ron", "fra", "deu", "spa", "ita", "por", "rus", "chi_sim", "jpn"],
        help="Select the language of the text in your PDF (eng for best character recognition)"
    )
    
    # Show cloud engine setup info
    if ocr_engine in ["azure", "google"]:
        st.sidebar.markdown("### Cloud Engine Setup")
        if ocr_engine == "azure":
            st.sidebar.info("""
            **Azure Computer Vision Setup:**
            1. Install: `pip install azure-cognitiveservices-vision-computervision`
            2. Set environment variables:
               - `AZURE_CV_KEY=your_key`
               - `AZURE_CV_ENDPOINT=your_endpoint`
            3. Get credentials from [Azure Portal](https://portal.azure.com/)
            """)
        elif ocr_engine == "google":
            st.sidebar.info("""
            **Google Cloud Vision Setup:**
            1. Install: `pip install google-cloud-vision`
            2. Set environment variable:
               - `GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json`
            3. Get credentials from [Google Cloud Console](https://console.cloud.google.com/)
            """)
    
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
                    results = scanner.process_document(file_path, language, engine=ocr_engine)
                
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