"""
Configuration settings for Agentic-RAG
"""

import os
import random
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
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
    GEMINI_API_KEY: Optional[str] = "AIzaSyD3TJnyqvnIKr5Kn-IsAyG7gsMndQVxJNI"  # User provided
    GEMINI_API_KEY_2: Optional[str] = None
    GEMINI_API_KEY_3: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"  # Free model
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
    
    # Embeddings settings - Use fast model for quick loading
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384d
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_PREFIX_QUERY: str = ""  # MiniLM doesn't need prefixes
    EMBEDDING_PREFIX_DOC: str = ""
    
    # Document processing - Semantic Chunking (300 tokens, 75 overlap)
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 75
    MAX_DOCUMENT_SIZE: int = 200 * 1024 * 1024  # 200MB
    
    # Reranker settings - Use cross-encoder for 9.5+ quality
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-6-v2"  # Fast cross-encoder
    RERANKER_DEVICE: str = "cpu"
    USE_RERANKER: bool = True  # Enable for 9.5+ quality
    TOP_K_RETRIEVAL: int = 12  # Increased for better reranking
    TOP_K_FINAL: int = 3  # Final selected chunks
    SIMILARITY_THRESHOLD: float = 0.7
    VECTOR_WEIGHT: float = 0.7  # Vector 70%
    BM25_WEIGHT: float = 0.3    # BM25 30%
    RERANK_THRESHOLD: float = 0.5  # Self-healing threshold
    
    # Query Rewriting settings
    ENABLE_QUERY_REWRITING: bool = True
    QUERY_REWRITE_TEMPERATURE: float = 0.1
    MAX_ITERATIONS: int = 3
    TIMEOUT_SECONDS: int = 30
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/agentic-rag.log"
    
    def get_api_keys(self, provider: str) -> List[str]:
        """Get all available API keys for a provider"""
        keys = []
        if provider.lower() == "openai":
            keys = [self.OPENAI_API_KEY, self.OPENAI_API_KEY_2, self.OPENAI_API_KEY_3]
        elif provider.lower() == "gemini":
            keys = [self.GEMINI_API_KEY, self.GEMINI_API_KEY_2, self.GEMINI_API_KEY_3]
        elif provider.lower() == "anthropic":
            keys = [self.ANTHROPIC_API_KEY, self.ANTHROPIC_API_KEY_2, self.ANTHROPIC_API_KEY_3]
        
        # Filter out None/empty keys
        return [key for key in keys if key and key.strip()]
    
    def get_random_api_key(self, provider: str) -> Optional[str]:
        """Get a random API key from available keys for load balancing"""
        keys = self.get_api_keys(provider)
        return random.choice(keys) if keys else None
    
    def get_ai_config(self):
        """Get AI configuration based on selected provider"""
        provider = self.AI_PROVIDER.lower()
        api_key = self.get_random_api_key(provider)
        
        if provider == "openai":
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": self.OPENAI_MODEL,
                "temperature": self.OPENAI_TEMPERATURE,
                "base_url": self.OPENAI_BASE_URL,
                "available_keys": len(self.get_api_keys("openai"))
            }
        elif provider == "gemini":
            return {
                "provider": "gemini",
                "api_key": api_key,
                "model": self.GEMINI_MODEL,
                "temperature": self.GEMINI_TEMPERATURE,
                "available_keys": len(self.get_api_keys("gemini"))
            }
        elif provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": api_key,
                "model": self.ANTHROPIC_MODEL,
                "temperature": self.ANTHROPIC_TEMPERATURE,
                "available_keys": len(self.get_api_keys("anthropic"))
            }
        elif provider == "local":
            return {
                "provider": "local",
                "url": self.LOCAL_LLM_URL,
                "model": self.LOCAL_LLM_MODEL,
                "temperature": self.LOCAL_LLM_TEMPERATURE,
                "available_keys": 0
            }
        else:
            # Default to OpenAI
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": self.OPENAI_MODEL,
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
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields


settings = Settings()
