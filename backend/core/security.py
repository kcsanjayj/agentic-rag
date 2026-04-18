"""
Security utilities for Agentic-RAG
API authentication, prompt injection protection, security headers
"""
import os
import re
from fastapi import Header, HTTPException, Request
from typing import Optional, Any, Dict


# API Secret for internal authentication - MUST be set in environment
API_SECRET = os.environ.get("API_SECRET")

if not API_SECRET:
    raise RuntimeError("API_SECRET environment variable must be set. Run: export API_SECRET=your-secure-random-key")


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify internal API key for protected endpoints"""
    if not x_api_key or x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API key")
    return x_api_key


# Prompt Injection Protection
BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "show hidden data",
    "bypass security",
    "disable restrictions",
    "ignore all rules",
    "pretend you are",
    "you are now",
    "system instruction",
    "override",
    "jailbreak",
    "dan mode",
    "developer mode",
    "ignore safety",
    "disregard",
    "new instructions",
]


def is_safe_input(query: str) -> tuple[bool, str]:
    """
    Check if user input is safe from prompt injection
    Returns (is_safe, reason)
    """
    if not query:
        return False, "Empty query"
    
    q = query.lower()
    
    for pattern in BLOCKED_PATTERNS:
        if pattern in q:
            return False, f"Unsafe input detected: blocked pattern"
    
    # Check for excessive special characters (obfuscation attempt)
    special_chars = len([c for c in query if not c.isalnum() and not c.isspace()])
    if special_chars > len(query) * 0.5:
        return False, "Input contains too many special characters"
    
    return True, "OK"


# Security Headers Middleware
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Enable XSS protection in browser
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # HTTPS enforcement (HSTS)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy - Updated to allow Font Awesome CDN
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "connect-src 'self' https:;"
    )
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions policy
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response


# Cost/Usage Protection
MAX_QUERY_LENGTH = 2000
MAX_TOKENS_PER_REQUEST = 4000  # Approximate limit
MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_HOUR = 100
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_query_length(query: str) -> tuple[bool, str]:
    """Validate query length for cost protection"""
    if not query:
        return False, "Query cannot be empty"
    
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long (max {MAX_QUERY_LENGTH} characters)"
    
    # Rough token estimation (1 token ≈ 4 chars for English)
    estimated_tokens = len(query) / 4
    if estimated_tokens > MAX_TOKENS_PER_REQUEST / 2:  # Leave room for response
        return False, f"Query too complex (estimated {int(estimated_tokens)} tokens, max {MAX_TOKENS_PER_REQUEST//2})"
    
    return True, "OK"


def sanitize_output(data: Any) -> Any:
    """
    Recursively sanitize output to prevent HTML/JS injection
    and unsafe content in API responses.
    """
    if isinstance(data, str):
        # Remove potential XSS patterns
        data = re.sub(r"(javascript:|<script.*?>.*?</script>)", "", data, flags=re.IGNORECASE)
        return data
    elif isinstance(data, dict):
        return {k: sanitize_output(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(item) for item in data]
    return data


# =============================================================================
# USER API KEY DEPENDENCY (PRO: Unified API key handling)
# =============================================================================

def get_user_api_key(x_user_api_key: str = Header(None, description="User's OpenAI API key")):
    """
    FastAPI dependency to extract and validate user's OpenAI API key.
    
    Clean: Single source of truth for API key validation
    Secure: Validates key format before use
    Reusable: Use in any endpoint that needs user API key
    """
    logger.info(f"get_user_api_key called - header present: {bool(x_user_api_key)}, length: {len(x_user_api_key) if x_user_api_key else 0}")
    
    if not x_user_api_key:
        logger.error("API key missing - X-User-Api-Key header not provided")
        raise HTTPException(status_code=400, detail="API key required. Provide your API key in X-User-Api-Key header")
    
    # Basic length check - different providers have different key formats
    # OpenAI: sk-..., NVIDIA: nvapi-... or long hex, etc.
    if len(x_user_api_key) < 10:
        logger.error(f"API key too short: {len(x_user_api_key)} chars")
        raise HTTPException(status_code=400, detail="Invalid API key. Key too short.")
    
    logger.info(f"API key validated successfully")
    return x_user_api_key


# =============================================================================
# COST PROTECTION LIMITS (PRO: Prevent abuse)
# =============================================================================

class CostLimits:
    """Cost protection limits for user-provided API keys"""
    
    MAX_TOKENS_PER_REQUEST = 4000  # Max tokens per LLM call
    MAX_CHUNKS_PER_UPLOAD = 500   # Max chunks per document upload
    MAX_CHUNK_SIZE = 2000         # Max characters per chunk
    MAX_FILE_SIZE_MB = 50         # Max file size in MB
    MAX_QUERY_LENGTH = 1000       # Max query length in characters
    MAX_REQUESTS_PER_MINUTE = 30  # Rate limit per user


def validate_query_length(query: str) -> tuple[bool, str]:
    """Validate query length for cost protection"""
    if len(query) > CostLimits.MAX_QUERY_LENGTH:
        return False, f"Query too long (max {CostLimits.MAX_QUERY_LENGTH} characters)"
    if len(query) < 3:
        return False, "Query too short (min 3 characters)"
    return True, ""


# =============================================================================
# OPENAI API KEY VALIDATION (PRO: Prevent fake keys)
# =============================================================================

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for API validation (reusable)
_validation_executor = ThreadPoolExecutor(max_workers=2)


async def validate_openai_key(api_key: str) -> bool:
    """
    PRO: Validate OpenAI API key with a cheap API call.
    Prevents fake keys and abuse.
    """
    if not api_key or not api_key.startswith("sk-"):
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Run validation in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _validation_executor,
            lambda: client.models.list(limit=1)
        )
        return True
        
    except Exception as e:
        logger.warning(f"API key validation failed: {e}")
        return False


# Cache for validated keys (TTL: 5 minutes)
_validated_keys: Dict[str, float] = {}
_validation_ttl = 300  # seconds


async def validate_openai_key_cached(api_key: str) -> bool:
    """Validate with caching to avoid repeated API calls"""
    import time
    
    # Check cache
    if api_key in _validated_keys:
        if time.time() - _validated_keys[api_key] < _validation_ttl:
            return True
        del _validated_keys[api_key]
    
    # Validate fresh
    is_valid = await validate_openai_key(api_key)
    if is_valid:
        _validated_keys[api_key] = time.time()
    
    return is_valid
