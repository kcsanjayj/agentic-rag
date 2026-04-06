"""
Retrieval Agent - GOLD STANDARD PIPELINE
User Query -> [Query Rewriter] -> [Hybrid Retrieval (top 12)] -> [Cross-Encoder Reranker] -> [Top 3 Selection] -> [Weak Signal Check] -> Final Context
"""

from typing import List, Dict, Any
from collections import Counter
from backend.utils.logger import setup_logger
from backend.core.vector_store import VectorStore, get_vector_store
from backend.core.embeddings import EmbeddingGenerator
from backend.core.reranker import get_reranker, CrossEncoderReranker
from backend.agents.query_rewrite_agent import get_query_rewriter
from backend.config import settings

logger = setup_logger(__name__)


class RetrievalAgent:
    """Agent responsible for document retrieval with GOLD STANDARD pipeline"""
    
    def __init__(self):
        self.vector_store = get_vector_store()  # 🔥 FIX: Use singleton to match routes.py
        self.embedding_generator = EmbeddingGenerator()
        self.reranker = get_reranker()
        self.query_rewriter = get_query_rewriter()
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.vector_weight = settings.VECTOR_WEIGHT
        self.bm25_weight = settings.BM25_WEIGHT
        self.rerank_threshold = settings.RERANK_THRESHOLD
        
    async def retrieve(self, query: str, top_k: int = None, enable_rewrite: bool = True, 
                      filter_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        FAST RETRIEVAL PIPELINE with doc_id filtering:
        1. Single query (no rewrite)
        2. Vector search only (no hybrid)
        3. Return top results directly (no reranking)
        """
        try:
            top_k = top_k or settings.TOP_K_RETRIEVAL
            final_k = settings.TOP_K_FINAL
            logger.info(f"FAST RETRIEVAL: Query='{query}' | Filter={filter_dict}")
            
            # 🔥 FAST: Single query, no rewriting
            query_embedding = await self.embedding_generator.generate_query_embedding(query)
            
            # 🔥 FAST: Single vector search with filter
            candidates = await self.vector_store.similarity_search(
                query_embedding,
                top_k=top_k,
                threshold=0.0,
                filter_dict=filter_dict
            )
            
            if not candidates:
                print("❌ No candidates retrieved from vector store!")
                return {
                    "retrieved_chunks": [],
                    "agent_trace": {},
                    "metadata": {"total_candidates": 0, "selected": 0}
                }
            
            # 🔥 SIMPLE: Just take top candidates (no aggressive filtering)
            selected_chunks = candidates[:final_k]  # 🔥 Simple TOP-K approach
            
            # 🔥 DEBUG
            print(f"\n🔥 RETRIEVED {len(candidates)} CHUNKS, selected top {len(selected_chunks)}")
            for i, c in enumerate(selected_chunks[:3]):
                print(f"Chunk {i+1}: {c.get('content', '')[:100]}...")
            
            return {
                "retrieved_chunks": selected_chunks,
                "agent_trace": {},
                "metadata": {
                    "total_candidates": len(candidates),
                    "selected": len(selected_chunks),
                    "pipeline": "fast"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in FAST retrieval: {str(e)}")
            raise
    
    # ========== Hybrid Scoring & Reranking Methods ==========
    
    async def _hybrid_score(self, query: str, candidates: List[Dict[str, Any]], 
                           query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Apply hybrid scoring: Vector 70% + BM25 30%"""
        scored = []
        query_words = query.lower().split()
        
        for i, candidate in enumerate(candidates):
            # Vector similarity (cosine)
            vector_score = candidate.get("score", 0.0)
            
            # BM25 score
            bm25_score = self._calculate_bm25(query_words, candidate["content"])
            
            # Combined hybrid score
            hybrid_score = (vector_score * self.vector_weight) + (bm25_score * self.bm25_weight)
            
            # Add context-enhanced embedding
            context_prefix = f"Document content: "
            enhanced_text = context_prefix + candidate["content"][:200]
            
            scored.append({
                "chunk_id": candidate.get("id", f"chunk_{i}"),
                "content": candidate["content"],
                "metadata": candidate.get("metadata", {}),
                "vector_score": round(vector_score, 3),
                "bm25_score": round(bm25_score, 3),
                "hybrid_score": round(hybrid_score, 3),
                "enhanced_text": enhanced_text
            })
        
        # Sort by hybrid score
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored
    
    def _calculate_bm25(self, query_words: List[str], document: str, 
                       k1: float = 1.5, b: float = 0.75) -> float:
        """Calculate BM25 score for a document"""
        doc_words = document.lower().split()
        doc_len = len(doc_words)
        avg_doc_len = 300  # Approximate average
        
        if not doc_words:
            return 0.0
        
        word_freq = Counter(doc_words)
        
        score = 0.0
        for word in query_words:
            if word in word_freq:
                tf = word_freq[word]
                idf = 1.0  # Simplified IDF
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf * (numerator / denominator)
        
        return min(score / max(len(query_words), 1), 1.0)
    
    def _rerank_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-encoder style reranking with confidence bonus"""
        reranked = []
        
        for i, candidate in enumerate(candidates):
            position_bonus = max(0, (10 - i) * 0.01)
            confidence_bonus = 0.1 if candidate["hybrid_score"] > 0.7 else \
                              0.05 if candidate["hybrid_score"] > 0.5 else 0
            
            rerank_score = candidate["hybrid_score"] + confidence_bonus + position_bonus
            
            reranked.append({
                **candidate,
                "rerank_score": round(rerank_score, 3),
                "rank": i + 1
            })
        
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        for i, r in enumerate(reranked):
            r["rank"] = i + 1
        
        return reranked
    
    def _select_chunks(self, reranked: List[Dict[str, Any]], 
                      max_select: int = None) -> List[Dict[str, Any]]:
        """Select top chunks with role assignment (primary, supporting)"""
        max_select = max_select or settings.TOP_K_FINAL
        selected = []
        
        for i, chunk in enumerate(reranked[:max_select * 2]):
            if len(selected) >= max_select:
                break
            
            if chunk["rerank_score"] >= 0.7:
                role = "primary" if not selected else "supporting"
                reason = "high confidence match"
            elif chunk["rerank_score"] >= 0.5:
                role = "primary" if not selected else "supporting"
                reason = "good relevance"
            elif chunk["rerank_score"] >= 0.3:
                role = "supporting"
                reason = "moderate relevance"
            else:
                continue
            
            selected.append({
                **chunk,
                "selected": True,
                "role": role,
                "selection_reason": reason
            })
        
        # Guarantee at least one selection
        if not selected and reranked:
            selected.append({
                **reranked[0],
                "selected": True,
                "role": "primary",
                "selection_reason": "best available (low confidence)"
            })
        
        return selected
    
    def _build_trace(self, query: str, candidates: List[Dict[str, Any]], 
                    selected: List[Dict[str, Any]], top_score: float,
                    strategy: str, rewrite_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build comprehensive agent trace"""
        trace = {
            "intent": self._detect_intent(query),
            "query_type": self._classify_query_type(query),
            "doc_type": "General Document",
            "retrieval": {
                "top_k_fetched": len(candidates),
                "reranked": len(candidates),
                "vector_weight": self.vector_weight,
                "bm25_weight": self.bm25_weight
            },
            "selection": {
                "primary": next((c for c in selected if c.get("role") == "primary"), None),
                "supporting": [c for c in selected if c.get("role") == "supporting"]
            },
            "strategy": strategy,
            "top_similarity": round(top_score, 3)
        }
        
        if rewrite_info:
            trace["query_rewrite"] = rewrite_info
            
        return trace
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()
        if any(w in query_lower for w in ["summarize", "summary", "overview"]):
            return "Summarization"
        elif any(w in query_lower for w in ["useful", "worth", "value", "evaluate"]):
            return "Evaluation"
        elif any(w in query_lower for w in ["extract", "data", "get"]):
            return "Extraction"
        else:
            return "Information Seeking"
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        if any(w in query_lower for w in ["what", "how", "why", "where", "when", "who"]):
            return "Question"
        elif any(w in query_lower for w in ["find", "search", "look for"]):
            return "Search"
        else:
            return "General"
