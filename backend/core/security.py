"""
Security utilities for Agentic-RAG
API authentication, prompt injection protection, security headers
"""
import os
import re
from fastapi import Header, HTTPException, Request
from typing import Optional


# API Secret for internal authentication
API_SECRET = os.environ.get("API_SECRET", "dev-secret-change-in-production")


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


def sanitize_output(text: str) -> str:
    """Sanitize LLM output to prevent XSS and other injection"""
    # Remove any script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove event handlers
    text = re.sub(r'\s*on\w+="[^"]*"', '', text, flags=re.IGNORECASE)
    text = re.sub(r"\s*on\w+='[^']*'", '', text, flags=re.IGNORECASE)
    return text


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
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' https:;"
    
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
