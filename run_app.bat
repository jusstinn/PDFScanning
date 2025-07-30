@echo off
echo Starting PDF Text Scanner Web Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

REM Start the Streamlit app
echo Starting Streamlit application...
echo The app will open in your default web browser.
echo.
streamlit run pdf_scanner.py

pause 