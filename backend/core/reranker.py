"""
Cross-Encoder Reranker - For precise relevance scoring
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)

# Try to import sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("CrossEncoder not available. Install with: pip install sentence-transformers")


class CrossEncoderReranker:
    """Cross-encoder reranker for precise relevance scoring"""
    
    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self.device = settings.RERANKER_DEVICE
        self.enabled = settings.USE_RERANKER and CROSS_ENCODER_AVAILABLE
        self.model = None
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {str(e)}")
            self.enabled = False
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: The search query
            documents: List of document dicts with 'content' field
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents with cross-encoder scores
        """
        if top_k is None:
            top_k = settings.TOP_K_FINAL
        
        if not self.enabled or not documents:
            # Fallback: return documents sorted by original score
            logger.info("Cross-encoder disabled or no documents, using original scores")
            sorted_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
            for doc in sorted_docs:
                doc['rerank_score'] = doc.get('score', 0)
                doc['rerank_method'] = 'original'
            return sorted_docs[:top_k]
        
        try:
            logger.info(f"Reranking {len(documents)} documents with cross-encoder")
            
            # Prepare pairs for cross-encoder
            pairs = [(query, doc['content']) for doc in documents]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Normalize scores to 0-1 range
            if len(scores) > 0:
                min_score = np.min(scores)
                max_score = np.max(scores)
                if max_score > min_score:
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(scores) * 0.5
            else:
                normalized_scores = scores
            
            # Add rerank scores to documents
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(normalized_scores[i])
                doc['cross_encoder_score'] = float(scores[i])
                doc['rerank_method'] = 'cross-encoder'
            
            # Sort by rerank score
            sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranking complete. Top score: {sorted_docs[0]['rerank_score']:.3f}")
            return sorted_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            # Fallback to original scores
            sorted_docs = sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
            for doc in sorted_docs:
                doc['rerank_score'] = doc.get('score', 0)
                doc['rerank_method'] = 'fallback'
            return sorted_docs[:top_k]
    
    def compute_relevance(self, query: str, document: str) -> float:
        """
        Compute relevance score for a single query-document pair
        
        Args:
            query: The search query
            document: The document text
            
        Returns:
            Relevance score between 0 and 1
        """
        if not self.enabled:
            return 0.5  # Default neutral score
        
        try:
            score = self.model.predict([(query, document)])[0]
            # Normalize to 0-1
            return float(np.clip(score, 0, 1))
        except Exception as e:
            logger.error(f"Error computing relevance: {str(e)}")
            return 0.5


class SimpleReranker:
    """Simple reranker using basic heuristics when cross-encoder not available"""
    
    @staticmethod
    def rerank(query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple reranking based on keyword overlap and position"""
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            content = doc.get('content', '').lower()
            doc_words = set(content.split())
            
            # Keyword overlap score
            overlap = len(query_words.intersection(doc_words))
            overlap_score = overlap / len(query_words) if query_words else 0
            
            # Position score (earlier mentions = more relevant)
            position_score = 0
            for word in query_words:
                if word in content:
                    position = content.index(word) / len(content) if len(content) > 0 else 1
                    position_score += (1 - position)  # Earlier = higher score
            
            position_score = position_score / len(query_words) if query_words else 0
            
            # Combine scores
            original_score = doc.get('score', 0)
            combined_score = (original_score * 0.4 + overlap_score * 0.4 + position_score * 0.2)
            
            doc['rerank_score'] = combined_score
            doc['rerank_method'] = 'simple'
            scored_docs.append(doc)
        
        # Sort by combined score
        sorted_docs = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_docs[:top_k]


def get_reranker():
    """Get the appropriate reranker instance"""
    if CROSS_ENCODER_AVAILABLE and settings.USE_RERANKER:
        return CrossEncoderReranker()
    else:
        return SimpleReranker()
