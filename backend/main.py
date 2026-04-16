"""
Agentic-RAG Backend - Clean Implementation
Uses proper modular architecture from the project structure
"""

import os
import sys

# Get the absolute path to the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

# Add both backend and project root to Python path for proper imports
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Import new secure settings
from backend.core.settings import Settings

# Import API routes
from backend.api.routes import router

# Setup logging
from backend.core.logger import safe_log, get_logger

logger = get_logger(__name__)


def setup_logging():
    """Setup secure logging configuration"""
    logging.basicConfig(
        level=getattr(logging, Settings.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_app() -> FastAPI:
    """Application factory pattern with security enhancements"""
    setup_logging()
    
    app = FastAPI(
        title="Agentic-RAG",
        version="2.0.0",
        description="Secure enterprise-grade RAG system with agentic processing"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Global exception handler - prevents leaking sensitive data
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error occurred")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": "An unexpected error occurred"},
        )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Static files - serve frontend from project root
    project_root = os.path.dirname(backend_dir)
    frontend_path = os.path.join(project_root, "frontend")
    
    if os.path.exists(frontend_path):
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
        safe_log(logger, "info", "Serving frontend", path=frontend_path)
    else:
        safe_log(logger, "warning", "Frontend not found", path=frontend_path)
    
    return app


app = create_app()


if __name__ == "__main__":
    safe_log(logger, "info", "Starting Agentic-RAG server", host=Settings.HOST, port=Settings.PORT)
    uvicorn.run(
        "main:app",
        host=Settings.HOST,
        port=Settings.PORT,
        reload=Settings.DEBUG,
        log_level=Settings.LOG_LEVEL.lower()
    )
