#!/bin/bash

echo "Starting PDF Text Scanner Web Application..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Start the Streamlit app
echo "Starting Streamlit application..."
echo "The app will open in your default web browser."
echo
streamlit run pdf_scanner.py 