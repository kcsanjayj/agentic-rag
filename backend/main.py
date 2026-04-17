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

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from slowapi import Limiter
from slowapi.util import get_remote_address

# Import new secure settings
from backend.core.settings import Settings
from backend.core.security import add_security_headers, verify_api_key
from backend.core.startup_check import validate_env, validate_dependencies

# Validate startup requirements
validate_env()
validate_dependencies()

# Import API routes
from backend.api.routes import router

# Setup logging
from backend.core.logger import safe_log, get_logger

logger = get_logger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Environment-based CORS origins
def get_allowed_origins():
    """Get CORS allowed origins based on environment"""
    if os.environ.get("ENV") == "production":
        return [
            "https://agentic-rag-gamma.vercel.app",
            "https://agentic-rag.vercel.app",
        ]
    else:
        # Development - include localhost and production URLs
        return [
            "https://agentic-rag-gamma.vercel.app",
            "https://agentic-rag.vercel.app",
            "http://localhost:8000",
            "http://localhost:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:3000",
        ]


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
    
    # Security headers middleware
    app.middleware("http")(add_security_headers)
    
    # CORS middleware - environment-based origins
    origins = get_allowed_origins()
    safe_log(logger, "info", "CORS origins configured", env=os.environ.get("ENV", "development"))
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-API-Key", "X-User-Api-Key"],
        allow_credentials=True,
        max_age=600,
    )
    
    # Add rate limiter
    app.state.limiter = limiter
    
    # Request size limit middleware
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        content_length = int(request.headers.get("content-length", 0))
        if content_length > 10_000_000:  # 10 MB limit
            return JSONResponse(
                status_code=413, 
                content={"error": "Request too large", "detail": "Maximum request size is 10MB"}
            )
        return await call_next(request)
    
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
    
    # Root-level health endpoint for Railway healthcheck (not behind /api/v1 prefix)
    @app.get("/health")
    async def root_health_check():
        """Root health check for Railway monitoring"""
        return {"status": "healthy", "version": "1.0.0"}
    
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
