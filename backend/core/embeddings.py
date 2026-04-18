"""
PRO RAG Stack: OpenAI Embeddings (lightweight, no heavy ML models)
"""

import asyncio
from typing import List, Optional
import numpy as np
from openai import OpenAI
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Simple cache for embeddings (PRO optimization)
_cache = {}


def get_openai_client(api_key: str) -> OpenAI:
    """Create OpenAI client with user-provided API key"""
    return OpenAI(api_key=api_key)


def _mock_embedding(text: str) -> List[float]:
    """Generate deterministic mock embedding for testing without API key"""
    import hashlib
    hash_val = hashlib.md5(text.encode()).hexdigest()
    embedding = []
    for i in range(0, len(hash_val), 2):
        val = int(hash_val[i:i+2], 16) / 255.0
        embedding.append(val)
    # Pad to 1536 dimensions (OpenAI text-embedding-3-small)
    while len(embedding) < 1536:
        embedding.extend(embedding[:min(1536-len(embedding), len(embedding))])
    return embedding[:1536]


def embed_text(text: str, api_key: str, use_cache: bool = True) -> List[float]:
    """
    Generate embedding for a single text using OpenAI API
    
    Args:
        text: Text to embed
        api_key: User-provided OpenAI API key
        use_cache: Whether to use caching (default: True)
    
    Returns:
        1536-dimensional embedding vector
    """
    # Check cache first (PRO optimization)
    if use_cache and text in _cache:
        return _cache[text]

    client = get_openai_client(api_key)

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # fast + cheap
            input=text
        )
        embedding = response.data[0].embedding

        # Cache the result (PRO optimization)
        if use_cache:
            _cache[text] = embedding

        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}, falling back to mock")
        return _mock_embedding(text)


def embed_texts(texts: List[str], api_key: str, use_cache: bool = True) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using OpenAI API (batch - faster)

    Args:
        texts: List of texts to embed
        api_key: User-provided OpenAI API key
        use_cache: Whether to use caching (default: True)

    Returns:
        List of 1536-dimensional embedding vectors
    """
    # Check cache for each text
    cached_results = {}
    texts_to_embed = []

    if use_cache:
        for text in texts:
            if text in _cache:
                cached_results[text] = _cache[text]
            else:
                texts_to_embed.append(text)
    else:
        texts_to_embed = texts

    # All texts were cached
    if not texts_to_embed:
        return [cached_results[text] for text in texts]

    client = get_openai_client(api_key)

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts_to_embed
        )

        # Map results back
        results = {}
        for i, item in enumerate(response.data):
            text = texts_to_embed[i]
            embedding = item.embedding
            results[text] = embedding

            # Cache the result
            if use_cache:
                _cache[text] = embedding

        # Combine cached + new results in original order
        return [cached_results.get(text, results.get(text, _mock_embedding(text))) for text in texts]

    except Exception as e:
        logger.error(f"Error generating batch embeddings: {str(e)}, falling back to mock")
        return [_mock_embedding(text) for text in texts]


def clear_cache():
    """Clear the embedding cache"""
    global _cache
    _cache = {}
    logger.info("Embedding cache cleared")


class EmbeddingGenerator:
    """PRO RAG: OpenAI Embeddings (lightweight, cloud-based, batch-optimized)"""

    def __init__(self, api_key: str):
        """
        Initialize with user-provided API key.
        
        Args:
            api_key: OpenAI API key (required, user-provided for SaaS billing model)
        """
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Valid OpenAI API key required (must start with 'sk-')")
        
        self._api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.model_name = "text-embedding-3-small"
        logger.info(f"PRO Embedding generator ready (OpenAI API-based)")

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation using pre-initialized client"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}, falling back to mock")
            return _mock_embedding(text)

    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch embedding generation using pre-initialized client"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}, falling back to mock")
            return [_mock_embedding(text) for text in texts]

    async def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text (async wrapper)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_embedding_sync, text)

    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding specifically for queries"""
        return await self.generate_embedding(query, is_query=True)

    async def generate_document_embedding(self, text: str) -> List[float]:
        """Generate embedding specifically for documents"""
        return await self.generate_embedding(text, is_query=False)

    async def generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts (PRO: uses batching)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_embeddings_sync, texts)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (OpenAI text-embedding-3-small is 1536d)"""
        return 1536

    async def similarity(self, text1: str, text2: str, text1_is_query: bool = False) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            embedding1 = await self.generate_embedding(text1, is_query=text1_is_query)
            embedding2 = await self.generate_embedding(text2, is_query=False)
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
