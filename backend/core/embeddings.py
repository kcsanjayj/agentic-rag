"""
Embedding generation for documents and queries
"""

import asyncio
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using sentence transformers with BGE support"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.normalize = getattr(settings, 'EMBEDDING_NORMALIZE', True)
        self.query_prefix = getattr(settings, 'EMBEDDING_PREFIX_QUERY', '')
        self.doc_prefix = getattr(settings, 'EMBEDDING_PREFIX_DOC', '')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded. Dimension: {self.get_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    async def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text with BGE prefix support"""
        try:
            # Add prefix for BGE models (better retrieval quality)
            if is_query and self.query_prefix:
                text = self.query_prefix + text
            elif not is_query and self.doc_prefix:
                text = self.doc_prefix + text
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            if self.normalize:
                embedding = await loop.run_in_executor(
                    None, 
                    lambda: self.model.encode(text, normalize_embeddings=True)
                )
            else:
                embedding = await loop.run_in_executor(
                    None, 
                    self.model.encode,
                    text
                )
            
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding specifically for queries (with prefix)"""
        return await self.generate_embedding(query, is_query=True)
    
    async def generate_document_embedding(self, text: str) -> List[float]:
        """Generate embedding specifically for documents"""
        return await self.generate_embedding(text, is_query=False)
    
    async def generate_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            # Add prefixes if needed
            if is_query and self.query_prefix:
                texts = [self.query_prefix + t for t in texts]
            elif not is_query and self.doc_prefix:
                texts = [self.doc_prefix + t for t in texts]
            
            loop = asyncio.get_event_loop()
            
            if self.normalize:
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(texts, normalize_embeddings=True)
                )
            else:
                embeddings = await loop.run_in_executor(
                    None,
                    self.model.encode,
                    texts
                )
            
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0
    
    async def similarity(self, text1: str, text2: str, text1_is_query: bool = False) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            embedding1 = await self.generate_embedding(text1, is_query=text1_is_query)
            embedding2 = await self.generate_embedding(text2, is_query=False)
            
            # For normalized embeddings, dot product = cosine similarity
            if self.normalize:
                similarity = np.dot(embedding1, embedding2)
            else:
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
