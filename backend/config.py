"""
Configuration settings for Agentic-RAG
"""

import os
import json
import random
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file (local development)
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Runtime config file path
RUNTIME_CONFIG_FILE = PROJECT_ROOT / "data" / "runtime_config.json"

# Runtime configuration storage (replaces .env file)
_runtime_config: Dict[str, Any] = {}

def _load_runtime_config():
    """Load runtime configuration from file"""
    global _runtime_config
    try:
        if RUNTIME_CONFIG_FILE.exists():
            with open(RUNTIME_CONFIG_FILE, 'r', encoding='utf-8') as f:
                _runtime_config = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load runtime config: {e}")
        _runtime_config = {}

def _save_runtime_config():
    """Save runtime configuration to file"""
    try:
        RUNTIME_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNTIME_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(_runtime_config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save runtime config: {e}")

def set_runtime_config(key: str, value: Any):
    """Set runtime configuration value and save to file"""
    _runtime_config[key] = value
    _save_runtime_config()

def get_runtime_config(key: str, default: Any = None) -> Any:
    """Get runtime configuration value"""
    return _runtime_config.get(key, default)

def update_runtime_config(updates: Dict[str, Any]):
    """Update multiple runtime configuration values and save to file"""
    _runtime_config.update(updates)
    _save_runtime_config()

def clear_runtime_config():
    """Clear all runtime configuration and remove file"""
    global _runtime_config
    _runtime_config = {}
    try:
        if RUNTIME_CONFIG_FILE.exists():
            RUNTIME_CONFIG_FILE.unlink()
    except Exception as e:
        print(f"Warning: Could not remove runtime config file: {e}")

# Load existing runtime config on module import
_load_runtime_config()


class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8000))  # Railway provides PORT env var
    DEBUG: bool = False  # Disable reload for production
    
    # Database settings
    CHROMA_DB_PATH: str = "data/chroma_db"
    CACHE_DIR: str = "data/cache"
    
    # AI Provider Selection
    AI_PROVIDER: str = "gemini"  # Options: openai, gemini, anthropic, local
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_KEY_2: Optional[str] = None
    OPENAI_API_KEY_3: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_BASE_URL: Optional[str] = None
    
    # Gemini Settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_KEY_2: Optional[str] = None
    GEMINI_API_KEY_3: Optional[str] = None
    GEMINI_MODEL: str = ""  # Empty - will be auto-selected from API
    GEMINI_TEMPERATURE: float = 0.2
    
    # Anthropic Settings
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY_2: Optional[str] = None
    ANTHROPIC_API_KEY_3: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    ANTHROPIC_TEMPERATURE: float = 0.1
    
    # Local LLM Settings (Ollama)
    LOCAL_LLM_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "llama2"
    LOCAL_LLM_TEMPERATURE: float = 0.1
    
    # NVIDIA Settings
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_MODEL: str = "meta/llama-3.1-70b-instruct"
    NVIDIA_TEMPERATURE: float = 0.1
    NVIDIA_MAX_TOKENS: int = 4096
    
    # Groq Settings
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama3-8b-8192"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 8192
    
    # Hugging Face Settings
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    HUGGINGFACE_TEMPERATURE: float = 0.1
    HUGGINGFACE_MAX_TOKENS: int = 4096
    
    # Embeddings settings - Use fast model for quick loading
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384d
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_PREFIX_QUERY: str = ""  # MiniLM doesn't need prefixes
    EMBEDDING_PREFIX_DOC: str = ""
    
    # Document processing - Larger chunks for better context
    CHUNK_SIZE: int = 1000  # Larger chunks prevent tiny useless fragments
    CHUNK_OVERLAP: int = 200  # Better overlap for context continuity
    MAX_DOCUMENT_SIZE: int = 200 * 1024 * 1024  # 200MB
    
    # Reranker settings - DISABLED for speed
    USE_RERANKER: bool = False
    TOP_K_RETRIEVAL: int = 8
    TOP_K_FINAL: int = 8  # Increased for better coverage
    SIMILARITY_THRESHOLD: float = 0.7
    VECTOR_WEIGHT: float = 0.7  # Vector 70%
    BM25_WEIGHT: float = 0.3    # BM25 30%
    RERANK_THRESHOLD: float = 0.5  # Self-healing threshold
    
    # Query Rewriting settings - DISABLED for speed
    ENABLE_QUERY_REWRITING: bool = False  # Disabled for fast retrieval
    QUERY_REWRITE_TEMPERATURE: float = 0.1
    MAX_ITERATIONS: int = 3
    TIMEOUT_SECONDS: int = 30
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/agentic-rag.log"
    
    def get_api_keys(self, provider: str) -> List[str]:
        """Get all available API keys for a provider - check runtime config first"""
        # Check runtime config first
        runtime_key = get_runtime_config(f"{provider.upper()}_API_KEY")
        if runtime_key:
            return [runtime_key]
        
        # Fall back to env vars
        keys = []
        if provider.lower() == "openai":
            keys = [self.OPENAI_API_KEY, self.OPENAI_API_KEY_2, self.OPENAI_API_KEY_3]
        elif provider.lower() == "gemini":
            keys = [self.GEMINI_API_KEY, self.GEMINI_API_KEY_2, self.GEMINI_API_KEY_3]
        elif provider.lower() == "anthropic":
            keys = [self.ANTHROPIC_API_KEY, self.ANTHROPIC_API_KEY_2, self.ANTHROPIC_API_KEY_3]
        elif provider.lower() == "nvidia":
            keys = [self.NVIDIA_API_KEY]
        elif provider.lower() == "groq":
            keys = [self.GROQ_API_KEY]
        elif provider.lower() == "huggingface":
            keys = [self.HUGGINGFACE_API_KEY]
        
        # Filter out None/empty keys
        return [key for key in keys if key and key.strip()]
    
    def get_random_api_key(self, provider: str) -> Optional[str]:
        """Get a random API key from available keys for load balancing"""
        keys = self.get_api_keys(provider)
        return random.choice(keys) if keys else None
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers that have API keys configured"""
        providers = ["openai", "gemini", "anthropic", "nvidia", "groq", "huggingface", "local"]
        available = []
        for provider in providers:
            if provider == "local":
                available.append(provider)  # Local doesn't need API key
            elif self.get_api_keys(provider):
                available.append(provider)
        return available
    
    def auto_select_provider(self) -> str:
        """Auto-select the first available provider with valid API key"""
        available = self.get_available_providers()
        
        # Priority order: gemini -> openai -> anthropic -> nvidia -> groq -> huggingface -> local
        priority = ["gemini", "openai", "anthropic", "nvidia", "groq", "huggingface", "local"]
        
        for provider in priority:
            if provider in available:
                return provider
        
        return "local"  # Fallback to local
    
    def get_ai_config(self):
        """Get AI configuration based on selected provider - check runtime config first"""
        # Check runtime config first
        runtime_provider = get_runtime_config("AI_PROVIDER")
        provider = runtime_provider.lower() if runtime_provider else self.AI_PROVIDER.lower()
        
        api_key = self.get_random_api_key(provider)
        
        # AUTO-DETECT: If current provider has no API key, switch to first available
        if not api_key and provider != "local":
            available = self.get_available_providers()
            if available:
                new_provider = available[0]  # Use first available
                print(f"Auto-switching from {provider} to {new_provider} (no API key found)")
                provider = new_provider
                api_key = self.get_random_api_key(provider)
        
        # Get custom model from runtime config if available
        custom_model = get_runtime_config(f"{provider.upper()}_MODEL")
        
        # Use model manager for validation and fallbacks
        try:
            from backend.agents.model_manager import ModelManager
            model_manager = ModelManager()
            
            # Normalize and validate model name
            if custom_model:
                custom_model = model_manager.normalize_model_name(provider, custom_model)
            
            # Get valid model (uses fallback if invalid or empty)
            model = model_manager.get_valid_model(provider, custom_model or getattr(self, f"{provider.upper()}_MODEL", ""))
            
            logger.info(f"Model manager selected model: {model} for provider: {provider}")
            
        except Exception as e:
            logger.warning(f"Model manager not available, using default: {str(e)}")
            # Fallback to old method
            fallback_models = {
                "gemini": "gemini-1.5-flash",
                "openai": "gpt-3.5-turbo",
                "anthropic": "claude-3-haiku-20240307",
                "nvidia": "meta/llama-3.1-70b-instruct",
                "groq": "llama3-8b-8192",
                "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
            }
            model = custom_model or getattr(self, f"{provider.upper()}_MODEL", "") or fallback_models.get(provider, "")
        
        temperature = get_runtime_config(f"{provider.upper()}_TEMPERATURE")
        if temperature is None:
            temperature = getattr(self, f"{provider.upper()}_TEMPERATURE", 0.1)

        if provider == "openai":
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "base_url": self.OPENAI_BASE_URL,
                "available_keys": len(self.get_api_keys("openai"))
            }
        elif provider == "gemini":
            return {
                "provider": "gemini",
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "available_keys": len(self.get_api_keys("gemini"))
            }
        elif provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "available_keys": len(self.get_api_keys("anthropic"))
            }
        elif provider == "local":
            return {
                "provider": "local",
                "url": self.LOCAL_LLM_URL,
                "model": model or "llama2",
                "temperature": temperature,
                "available_keys": 0
            }
        elif provider == "nvidia":
            return {
                "provider": "nvidia",
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "available_keys": len(self.get_api_keys("nvidia"))
            }
        elif provider == "groq":
            return {
                "provider": "groq",
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "available_keys": len(self.get_api_keys("groq"))
            }
        elif provider == "huggingface":
            return {
                "provider": "huggingface",
                "api_key": api_key,
                "model": model,
                "temperature": temperature,
                "available_keys": len(self.get_api_keys("huggingface"))
            }
        else:
            # Default to OpenAI
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": model,
                "temperature": self.OPENAI_TEMPERATURE,
                "base_url": self.OPENAI_BASE_URL,
                "available_keys": len(self.get_api_keys("openai"))
            }
    
    def is_ai_configured(self):
        """Check if the selected AI provider is properly configured"""
        config = self.get_ai_config()
        if config["provider"] == "local":
            return True  # Local models don't need API keys
        return bool(config.get("api_key"))
    
    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields


settings = Settings()
