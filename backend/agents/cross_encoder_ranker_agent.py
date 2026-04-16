"""
Cross Encoder Ranker Agent - Reranks retrieved documents using cross-encoder models
Improves retrieval accuracy by re-scoring document-query pairs
"""

from typing import List, Dict, Any
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class CrossEncoderRankerAgent:
    """Reranks retrieved documents using cross-encoder models for better accuracy"""
    
    def __init__(self):
        self.use_mock = True  # Will use mock scoring unless cross-encoder is available
        logger.info("Cross Encoder Ranker Agent initialized (mock mode)")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using cross-encoder scoring
        
        Args:
            query: The search query
            documents: List of retrieved documents with text and metadata
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        try:
            if self.use_mock:
                return self._mock_rerank(query, documents, top_k)
            else:
                return self._cross_encoder_rerank(query, documents, top_k)
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return documents[:top_k]
    
    def _mock_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Mock reranking based on keyword overlap"""
        query_terms = set(query.lower().split())
        
        scored_docs = []
        for doc in documents:
            text = doc.get("text", "").lower()
            doc_terms = set(text.split())
            
            # Calculate overlap score
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            
            scored_docs.append({
                **doc,
                "rerank_score": score
            })
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]
    
    def _cross_encoder_rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Use actual cross-encoder model for reranking"""
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Prepare pairs
            pairs = [[query, doc.get("text", "")] for doc in documents]
            
            # Get scores
            scores = model.predict(pairs)
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # Sort by score and return top_k
            documents.sort(key=lambda x: x["rerank_score"], reverse=True)
            return documents[:top_k]
            
        except Exception as e:
            logger.warning(f"Cross-encoder not available, using mock: {str(e)}")
            self.use_mock = True
            return self._mock_rerank(query, documents, top_k)
    
    def is_available(self) -> bool:
        """Check if cross-encoder is available"""
        return not self.use_mock
