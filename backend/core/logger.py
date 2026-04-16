"""
Safe logging module - prevents secret leakage in logs
"""
import logging
import re
import os

# Setup file logging for error monitoring
def setup_file_logging():
    """Setup file handler for error logging"""
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    file_handler.setLevel(logging.WARNING)  # Only log warnings and errors
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Get root logger and add file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def mask_api_key(key: str) -> str:
    """Mask API key for safe logging - show only first/last 4 chars"""
    if not key or len(key) < 8:
        return "***" if key else ""
    return f"{key[:4]}...{key[-4:]}"


def safe_log(logger: logging.Logger, level: str, message: str, **kwargs):
    """
    Safe logging function that automatically masks sensitive data
    
    Args:
        logger: The logger instance
        level: Log level (debug, info, warning, error)
        message: The log message
        **kwargs: Additional context (will be masked if contains 'key', 'token', 'secret')
    """
    # Mask any potential secrets in kwargs
    safe_kwargs = {}
    for k, v in kwargs.items():
        key_lower = k.lower()
        if any(sensitive in key_lower for sensitive in ['key', 'token', 'secret', 'password', 'api_key']):
            safe_kwargs[k] = mask_api_key(str(v))
        else:
            safe_kwargs[k] = v
    
    # Format message with masked kwargs
    if safe_kwargs:
        safe_message = f"{message} | {safe_kwargs}"
    else:
        safe_message = message
    
    # Get the log method
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(safe_message)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance"""
    logger = logging.getLogger(name)
    return logger


# Common patterns to redact
SENSITIVE_PATTERNS = [
    (r'AIza[\w-]{35}', '***GOOGLE_KEY***'),
    (r'sk-[\w-]{48}', '***OPENAI_KEY***'),
    (r'sk-proj-[\w-]{100,}', '***OPENAI_PROJ_KEY***'),
    (r'gsk_[\w-]{40,}', '***GROQ_KEY***'),
    (r'nvapi-[\w-]{60,}', '***NVIDIA_KEY***'),
    (r'[\w-]{40,}\.[\w-]{20,}\.[\w-]{40,}', '***JWT_TOKEN***'),  # JWT pattern
]


def sanitize_message(message: str) -> str:
    """Sanitize a message by replacing sensitive patterns"""
    sanitized = message
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized
