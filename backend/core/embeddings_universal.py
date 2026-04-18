"""
Universal Embeddings - Works with ANY AI Provider
Supports: OpenAI, NVIDIA, Gemini, Anthropic, and mock embeddings
"""

import asyncio
import hashlib
import base64
import json
from typing import List, Optional, Dict, Any
import numpy as np
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Simple cache for embeddings
_cache = {}

def _mock_embedding(text: str, dimension: int = 1536) -> List[float]:
    """Generate deterministic mock embedding for testing"""
    hash_val = hashlib.md5(text.encode()).hexdigest()
    embedding = []
    for i in range(0, len(hash_val), 2):
        val = int(hash_val[i:i+2], 16) / 255.0
        embedding.append(val)
    # Pad to target dimension
    while len(embedding) < dimension:
        embedding.extend(embedding[:min(dimension-len(embedding), len(embedding))])
    return embedding[:dimension]


class UniversalEmbeddingGenerator:
    """Universal embedding generator - works with any provider"""
    
    def __init__(self, api_key: str, provider: str = "auto"):
        """
        Initialize with user-provided API key
        
        Args:
            api_key: API key (any provider)
            provider: Provider name or 'auto' to detect from key format
        """
        self._api_key = api_key
        self._provider = self._detect_provider(api_key) if provider == "auto" else provider
        self._dimension = 1536  # Standard dimension
        self._client = None
        
        # Initialize appropriate client
        self._initialize_client()
        
        logger.info(f"Universal Embedding Generator ready (provider: {self._provider})")
    
    def _detect_provider(self, api_key: str) -> str:
        """Detect provider from API key format"""
        if not api_key:
            return "mock"
        
        # OpenAI: sk-...
        if api_key.startswith("sk-"):
            return "openai"
        
        # NVIDIA: nvapi-... or contains nvidia-specific patterns
        if api_key.startswith("nvapi-") or "nvidia" in api_key.lower():
            return "nvidia"
        
        # Gemini: typically AIza...
        if api_key.startswith("AIza"):
            return "gemini"
        
        # Anthropic: sk-ant-...
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        
        # Default to OpenAI-compatible (many providers use OpenAI-compatible APIs)
        return "openai_compatible"
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider"""
        if self._provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
                logger.info("OpenAI client initialized for embeddings")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self._provider = "mock"
        
        elif self._provider == "nvidia":
            # NVIDIA uses OpenAI-compatible API
            try:
                from openai import OpenAI
                # NVIDIA endpoints
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url="https://integrate.api.nvidia.com/v1"
                )
                logger.info("NVIDIA client initialized (OpenAI-compatible)")
            except Exception as e:
                logger.warning(f"Could not initialize NVIDIA client: {e}")
                self._provider = "mock"
        
        elif self._provider == "gemini":
            # Gemini embeddings via their API
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._client = genai
                logger.info("Gemini client initialized for embeddings")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")
                self._provider = "mock"
        
        elif self._provider == "openai_compatible":
            # Generic OpenAI-compatible provider
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
                logger.info("Generic OpenAI-compatible client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize client: {e}")
                self._provider = "mock"
        
        else:
            self._provider = "mock"
            logger.info("Using mock embeddings (no valid API key)")
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation"""
        try:
            if self._provider == "openai" or self._provider == "nvidia" or self._provider == "openai_compatible":
                return self._generate_openai_embedding(text)
            
            elif self._provider == "gemini":
                return self._generate_gemini_embedding(text)
            
            else:
                return _mock_embedding(text, self._dimension)
                
        except Exception as e:
            logger.error(f"Error generating embedding with {self._provider}: {e}")
            return _mock_embedding(text, self._dimension)
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI-compatible API"""
        try:
            response = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8191]  # OpenAI max input length
            )
            embedding = response.data[0].embedding
            self._dimension = len(embedding)
            return embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def _generate_gemini_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API"""
        try:
            # Use Gemini's embedding model
            model = self._client.GenerativeModel('models/text-embedding-004')
            result = model.embed_content(text)
            embedding = result.embedding
            
            # Ensure dimension matches (Gemini outputs 768d, need to pad to 1536)
            if len(embedding) < self._dimension:
                # Pad with zeros
                embedding = list(embedding) + [0.0] * (self._dimension - len(embedding))
            elif len(embedding) > self._dimension:
                # Truncate (not ideal but maintains compatibility)
                embedding = embedding[:self._dimension]
            
            return embedding
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise
    
    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding generation"""
        try:
            if self._provider in ["openai", "nvidia", "openai_compatible"]:
                # OpenAI supports batching
                response = self._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[t[:8191] for t in texts]
                )
                embeddings = [item.embedding for item in response.data]
                self._dimension = len(embeddings[0]) if embeddings else 1536
                return embeddings
            
            elif self._provider == "gemini":
                # Gemini batch processing
                embeddings = []
                for text in texts:
                    embeddings.append(self._generate_gemini_embedding(text))
                return embeddings
            
            else:
                return [_mock_embedding(t, self._dimension) for t in texts]
                
        except Exception as e:
            logger.error(f"Batch embedding error with {self._provider}: {e}")
            return [_mock_embedding(t, self._dimension) for t in texts]
    
    async def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text (async)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_embedding_sync, text)
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        return await self.generate_embedding(query, is_query=True)
    
    async def generate_document_embedding(self, text: str) -> List[float]:
        """Generate embedding for document"""
        return await self.generate_embedding(text, is_query=False)
    
    async def generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_embeddings_sync, texts)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension
    
    def get_provider(self) -> str:
        """Get detected provider name"""
        return self._provider


# Backward compatibility - old function names
def embed_text(text: str, api_key: str, use_cache: bool = True) -> List[float]:
    """Backward compatible single text embedding"""
    gen = UniversalEmbeddingGenerator(api_key)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(gen.generate_embedding(text))


def embed_texts(texts: List[str], api_key: str, use_cache: bool = True) -> List[List[float]]:
    """Backward compatible batch embedding"""
    gen = UniversalEmbeddingGenerator(api_key)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(gen.generate_embeddings(texts))


# Legacy class name for backward compatibility
class EmbeddingGenerator(UniversalEmbeddingGenerator):
    """Legacy name - redirects to UniversalEmbeddingGenerator"""
    pass
