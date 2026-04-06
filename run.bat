@echo off
echo Starting Simple Agentic-RAG Application...
echo.
echo  Browser will open automatically at http://localhost:8000
echo  API documentation available at http://localhost:8000/docs
echo.
echo Available utility scripts in scripts/:
echo   - check_collections.py - Check ChromaDB collections
echo   - check_vector_db.py - Test vector DB similarity search
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Run the startup script
python start.py

echo.
echo Application stopped. Press any key to exit...
pause >nul
