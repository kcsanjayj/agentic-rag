@echo off
REM Agentic-RAG Startup Script for Windows
REM This script starts the Agentic-RAG application

echo.
echo ========================================
echo    Agentic-RAG Development Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Create necessary directories
echo [INFO] Creating directories...
if not exist "data\uploads" mkdir data\uploads
if not exist "data\chroma_db" mkdir data\chroma_db
if not exist "data\cache" mkdir data\cache
if not exist "logs" mkdir logs

REM Set environment variables
set PYTHONPATH=%CD%
set HOST=0.0.0.0
set PORT=8000

REM Check if .env file exists
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] Creating .env file from example...
        copy .env.example .env >nul
        echo [WARNING] Please edit .env file with your configuration
    ) else (
        echo [WARNING] No .env file found, using default settings
    )
)

echo.
echo [SUCCESS] Starting Agentic-RAG server...
echo.
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the application
python -m uvicorn backend.main:app --host %HOST% --port %PORT% --reload

echo.
echo Server stopped
pause
