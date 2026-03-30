"""
Retrieval Agent - GOLD STANDARD PIPELINE
User Query -> [Query Rewriter] -> [Hybrid Retrieval (top 12)] -> [Cross-Encoder Reranker] -> [Top 3 Selection] -> [Weak Signal Check] -> Final Context
"""

from typing import List, Dict, Any
from collections import Counter
from backend.utils.logger import setup_logger
from backend.core.vector_store import VectorStore
from backend.core.embeddings import EmbeddingGenerator
from backend.core.reranker import get_reranker, CrossEncoderReranker
from backend.agents.query_rewriting_agent import get_query_rewriter
from backend.config import settings

logger = setup_logger(__name__)


class RetrievalAgent:
    """Agent responsible for document retrieval with GOLD STANDARD pipeline"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_generator = EmbeddingGenerator()
        self.reranker = get_reranker()
        self.query_rewriter = get_query_rewriter()
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.vector_weight = settings.VECTOR_WEIGHT
        self.bm25_weight = settings.BM25_WEIGHT
        self.rerank_threshold = settings.RERANK_THRESHOLD
        
    async def retrieve(self, query: str, top_k: int = None, enable_rewrite: bool = True) -> Dict[str, Any]:
        """
        GOLD STANDARD PIPELINE:
        1. Query Rewriting (optional)
        2. Hybrid Retrieval (top 12)
        3. Cross-Encoder Reranking
        4. Top 3 Selection
        5. Weak Signal Check with Self-Healing
        """
        try:
            top_k = top_k or settings.TOP_K_RETRIEVAL
            final_k = settings.TOP_K_FINAL
            logger.info(f"GOLD STANDARD PIPELINE: Query='{query}'")
            
            # ========== STEP 1: QUERY REWRITING ==========
            rewritten_query = query
            query_variations = [query]
            rewrite_info = None
            
            if enable_rewrite and settings.ENABLE_QUERY_REWRITING:
                rewrite_result = await self.query_rewriter.rewrite_query(query)
                rewritten_query = rewrite_result.get("rewritten", query)
                query_variations = rewrite_result.get("expanded", [query])
                rewrite_info = rewrite_result
                logger.info(f"Query rewritten: '{query}' -> '{rewritten_query}'")
            
            # ========== STEP 2: HYBRID RETRIEVAL (top 12) ==========
            all_candidates = []
            
            # Try multiple query variations for better coverage
            for var_query in query_variations[:3]:  # Limit to top 3 variations
                # Generate query embedding WITH BGE PREFIX
                query_embedding = await self.embedding_generator.generate_query_embedding(var_query)
                
                # Fetch candidates from vector store
                candidates = await self.vector_store.similarity_search(
                    query_embedding,
                    top_k=top_k,
                    threshold=0.0  # Get all candidates, filter later
                )
                
                if candidates:
                    # Apply Hybrid Scoring (Vector 70% + BM25 30%)
                    scored = await self._hybrid_score(var_query, candidates, query_embedding)
                    all_candidates.extend(scored)
            
            # Deduplicate and sort
            seen_ids = set()
            unique_candidates = []
            for c in sorted(all_candidates, key=lambda x: x["hybrid_score"], reverse=True):
                if c["chunk_id"] not in seen_ids:
                    unique_candidates.append(c)
                    seen_ids.add(c["chunk_id"])
            
            if not unique_candidates:
                logger.warning("No candidates retrieved")
                return {
                    "retrieved_chunks": [],
                    "agent_trace": self._build_trace(query, [], [], 0, "no_results", rewrite_info),
                    "metadata": {"total_candidates": 0, "selected": 0, "query_rewritten": enable_rewrite}
                }
            
            logger.info(f"Retrieved {len(unique_candidates)} unique candidates")
            
            # ========== STEP 3: CROSS-ENCODER RERANKING ==========
            # Use top 10 for reranking, then select final 3
            rerank_candidates = unique_candidates[:10]
            reranked = self.reranker.rerank(rewritten_query, rerank_candidates, top_k=len(rerank_candidates))
            
            logger.info(f"Cross-encoder reranking complete. Top score: {reranked[0]['rerank_score']:.3f}")
            
            # ========== STEP 4: TOP 3 SELECTION ==========
            selected_chunks = self._select_chunks(reranked, max_select=final_k)
            
            # ========== STEP 5: WEAK SIGNAL CHECK (Self-Healing) ==========
            top_score = selected_chunks[0]["rerank_score"] if selected_chunks else 0
            retry_performed = False
            expanded_query_used = None
            
            if top_score < self.rerank_threshold:
                logger.info(f"WEAK SIGNAL: {top_score:.3f} < {self.rerank_threshold}, triggering self-healing")
                
                # Expand query with more terms
                expanded_query = f"{query} document content information details purpose"
                expanded_query_used = expanded_query
                
                # Retry with expanded query
                expanded_embedding = await self.embedding_generator.generate_query_embedding(expanded_query)
                retry_candidates = await self.vector_store.similarity_search(
                    expanded_embedding,
                    top_k=top_k,
                    threshold=0.0
                )
                
                if retry_candidates:
                    retry_scored = await self._hybrid_score(expanded_query, retry_candidates, expanded_embedding)
                    retry_reranked = self.reranker.rerank(expanded_query, retry_scored[:10], top_k=10)
                    retry_selected = self._select_chunks(retry_reranked, max_select=final_k)
                    
                    # Use retry results if better
                    if retry_selected and retry_selected[0]["rerank_score"] > top_score:
                        selected_chunks = retry_selected
                        top_score = selected_chunks[0]["rerank_score"]
                        retry_performed = True
                        logger.info(f"SELF-HEALING SUCCESS: Score improved to {top_score:.3f}")
            
            # Build agent trace
            agent_trace = self._build_trace(
                query, unique_candidates, selected_chunks, top_score,
                "self-healing" if retry_performed else "standard",
                rewrite_info
            )
            
            # Add metadata about pipeline
            metadata = {
                "total_candidates": len(unique_candidates),
                "selected": len(selected_chunks),
                "top_similarity": top_score,
                "query_rewritten": enable_rewrite and settings.ENABLE_QUERY_REWRITING,
                "self_healing_performed": retry_performed,
                "expanded_query": expanded_query_used if retry_performed else None,
                "pipeline": "gold_standard",
                "embedding_model": settings.EMBEDDING_MODEL,
                "reranker": "cross-encoder" if isinstance(self.reranker, CrossEncoderReranker) else "simple"
            }
            
            logger.info(f"GOLD STANDARD PIPELINE COMPLETE: {len(selected_chunks)} chunks, top score: {top_score:.3f}")
            
            return {
                "retrieved_chunks": selected_chunks,
                "agent_trace": agent_trace,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in GOLD STANDARD retrieval: {str(e)}")
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
