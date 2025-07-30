# Document Text Scanner

A powerful application that extracts text from scanned PDF documents and images using Optical Character Recognition (OCR) technology. This tool can process scanned PDFs and images, converting them into searchable, editable text. Perfect for extracting mixed content including codes, Romanian text, and other languages.

## Features

- 📄 **Document Processing**: Convert scanned PDFs and images to text using OCR
- 🌍 **Multi-language Support**: Supports 10 languages including English, Romanian, French, German, Spanish, Italian, Portuguese, Russian, Chinese, and Japanese
- 🎛️ **Configurable Settings**: Adjustable DPI, language selection, and processing options
- 💻 **Dual Interface**: Web-based Streamlit app and command-line interface
- 📊 **Detailed Results**: Page-by-page breakdown with word counts and statistics
- 💾 **Multiple Export Formats**: Save results as JSON or plain text
- 🖼️ **Image Preprocessing**: Advanced image processing for better OCR accuracy

## Prerequisites

### System Requirements
- Python 3.7 or higher
- Windows, macOS, or Linux

### Required Software
1. **Tesseract OCR Engine**: The core OCR engine
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

2. **Poppler** (for PDF processing):
   - **Windows**: Download from [poppler releases](http://blog.alivate.com.au/poppler-windows/)
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd PdfScans
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Tesseract installation**
   ```bash
   tesseract --version
   ```

## Usage

### Web Interface (Recommended)

1. **Start the Streamlit app**
   ```bash
   streamlit run pdf_scanner.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

3. **Upload your PDF** and configure settings:
   - Select the language of your text
   - Adjust DPI if needed (higher = better quality, slower processing)
   - Click "Extract Text"

4. **Review and download results**

### Command Line Interface

#### Basic Usage
```bash
python cli_scanner.py document.pdf
python cli_scanner.py image.jpg
```

#### Advanced Options
```bash
# Specify language and output file (mixed content)
python cli_scanner.py document.pdf --language eng+ron --output results.txt

# Export as JSON format
python cli_scanner.py document.pdf --format json --output results.json

# Verbose output with custom DPI
python cli_scanner.py document.pdf --language ron --dpi 400 --verbose
```

#### Command Line Options
- `--language, -l`: OCR language (eng, fra, deu, spa, ita, por, rus, chi_sim, jpn)
- `--output, -o`: Output file path
- `--format, -f`: Output format (text or json)
- `--dpi, -d`: Image resolution (150-600, default: 300)
- `--verbose, -v`: Verbose output

## Supported Languages

| Language | Code | Description |
|----------|------|-------------|
| English | `eng` | English (default) |
| Romanian | `ron` | Romanian |
| English + Romanian | `eng+ron` | Mixed content (codes + Romanian text) |
| French | `fra` | French |
| German | `deu` | German |
| Spanish | `spa` | Spanish |
| Italian | `ita` | Italian |
| Portuguese | `por` | Portuguese |
| Russian | `rus` | Russian |
| Chinese Simplified | `chi_sim` | Chinese (Simplified) |
| Japanese | `jpn` | Japanese |

## Tips for Better Results

### Image Quality
- Use high-resolution scans (300 DPI or higher)
- Ensure good lighting and contrast
- Avoid blurry or skewed images
- Clean, well-lit scans work best

### Processing Settings
- **Higher DPI**: Better accuracy but slower processing
- **Correct Language**: Always select the appropriate language
- **Clean Images**: Pre-process images if they're noisy or low quality

### Best Practices
- Test with a single page first
- Use consistent lighting when scanning
- For mixed content (codes + Romanian text), use "eng+ron" language setting
- Avoid handwritten text (designed for printed text)
- Check results and adjust settings if needed
- The app preserves code formatting and structure

## Troubleshooting

### Common Issues

1. **"Tesseract not found" error**
   - Install Tesseract OCR engine
   - Add Tesseract to your system PATH
   - Restart your terminal/command prompt

2. **"Poppler not found" error**
   - Install Poppler utilities
   - Ensure `pdftoppm` is available in your PATH

3. **Poor OCR accuracy**
   - Increase DPI setting
   - Check image quality
   - Verify language selection
   - Try different preprocessing settings

4. **Memory issues with large PDFs**
   - Reduce DPI setting
   - Process pages in smaller batches
   - Close other applications

### Windows-Specific Issues

1. **Tesseract PATH**: Add Tesseract installation directory to system PATH
2. **Poppler PATH**: Add Poppler `bin` directory to system PATH
3. **Permission errors**: Run as administrator if needed

### Performance Optimization

- Use SSD storage for faster processing
- Close unnecessary applications
- Adjust DPI based on your needs (300 DPI is usually sufficient)
- Process large documents in smaller batches

## File Structure

```
PdfScans/
├── pdf_scanner.py      # Main Streamlit application
├── cli_scanner.py      # Command-line interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Supported File Formats

- **PDF files**: Multi-page documents
- **Images**: JPG, JPEG, PNG, BMP, TIFF, TIF

## Output Formats

### Text Format
Plain text output with page separators:
```
--- Page 1 ---
[Extracted text from page 1]

--- Page 2 ---
[Extracted text from page 2]
```

### JSON Format
Structured data with metadata:
```json
{
  "filename": "document.pdf",
  "total_pages": 2,
  "pages": [
    {
      "page_number": 1,
      "text": "Extracted text...",
      "word_count": 150
    }
  ],
  "full_text": "Complete extracted text..."
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

If you encounter issues or have questions:
1. Check the troubleshooting section
2. Verify your installation
3. Test with a simple PDF first
4. Check the console output for error messages

## Acknowledgments

- **Tesseract OCR**: The core OCR engine
- **Streamlit**: Web interface framework
- **OpenCV**: Image processing capabilities
- **Pillow**: Image handling
- **pdf2image**: PDF to image conversion 