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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import configuration
from backend.config import settings

# Import API routes
from backend.api.routes import router

# Setup logging
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_app() -> FastAPI:
    """Application factory pattern"""
    app = FastAPI(
        title="Agentic-RAG",
        version="1.0.0",
        description="Enterprise-grade RAG system with agentic processing"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    # Static files - serve frontend from project root
    project_root = os.path.dirname(backend_dir)
    frontend_path = os.path.join(project_root, "frontend")
    
    if os.path.exists(frontend_path):
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
        logger.info(f"Serving frontend from: {frontend_path}")
    else:
        logger.warning(f"Frontend not found at: {frontend_path}")
    
    return app


app = create_app()


if __name__ == "__main__":
    logger.info(f"Starting Agentic-RAG server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
