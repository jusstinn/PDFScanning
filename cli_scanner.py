#!/usr/bin/env python3
"""
Command-line PDF Text Scanner
Extract text from scanned PDF documents using OCR
"""

import argparse
import os
import sys
import json
import platform
import subprocess
from pdf_scanner import DocumentScanner

def main():
    parser = argparse.ArgumentParser(
        description="Extract text from scanned PDF documents using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_scanner.py document.pdf
  python cli_scanner.py image.jpg --language eng+ron --output results.txt
  python cli_scanner.py document.pdf --language ron --format json
        """
    )
    
    parser.add_argument(
        'document_file',
        help='Path to the document file to process (PDF or image)'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='eng',
        choices=['eng', 'ron', 'eng+ron', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn'],
        help='Language for OCR (default: eng, use eng+ron for mixed content)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: prints to console)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--dpi', '-d',
        type=int,
        default=300,
        help='DPI for image conversion (default: 300)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.document_file):
        print(f"Error: File '{args.document_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Check file extension
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    file_ext = os.path.splitext(args.document_file)[1].lower()
    if file_ext not in supported_extensions:
        print(f"Error: File must be a supported format: {', '.join(supported_extensions)}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize scanner
    scanner = DocumentScanner()
    
    # Check system requirements
    if not scanner.tesseract_path:
        print("Error: Tesseract OCR is not installed or not found.", file=sys.stderr)
        print("Please install Tesseract OCR for text extraction.", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Processing document: {args.document_file}")
        print(f"Language: {args.language}")
        print(f"DPI: {args.dpi}")
        print(f"Tesseract path: {scanner.tesseract_path}")
        if not scanner.check_poppler():
            print("Warning: Poppler not found. PDF processing may not work.")
        print("Starting OCR processing...")
    
    try:
        # Process the document
        results = scanner.process_document(args.document_file, args.language)
        
        if results['total_pages'] == 0:
            print("Error: No pages were processed.", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            print(f"Successfully processed {results['total_pages']} pages")
            print(f"Total words extracted: {sum(page['word_count'] for page in results['pages'])}")
        
        # Prepare output
        if args.format == 'json':
            output_data = json.dumps(results, indent=2, ensure_ascii=False)
        else:
            output_data = results['full_text']
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_data)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        else:
            print(output_data)
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 