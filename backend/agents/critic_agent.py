"""
Critic Agent - Self-correction and validation for 10/10 elite RAG system
"""

from typing import Dict, Any, List, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class CriticAgent:
    """Agent for self-correction and validation of generated responses"""
    
    def __init__(self):
        self.criticism_rules = {
            "min_grounding": 0.7,  # At least 70% of answer should be grounded
            "max_hallucination": 0.1,  # Less than 10% hallucination
            "min_completeness": 0.6,  # At least 60% completeness
            "required_citations": 2  # Minimum citations required
        }
    
    async def criticize_answer(self, answer: str, retrieved_docs: List[Dict[str, Any]], 
                              query: str, agent_trace: Dict[str, Any] = None) -> Dict[str, Any]:
        """Criticize the generated answer and determine if retry is needed"""
        
        # 10/10 FEATURE: Comprehensive criticism
        grounding_score = self._check_grounding(answer, retrieved_docs)
        hallucination_score = self._detect_hallucination(answer, retrieved_docs)
        completeness_score = self._check_completeness(answer, query, retrieved_docs)
        citation_score = self._check_citations(answer, retrieved_docs)
        
        # Overall critic decision
        overall_score = (grounding_score * 0.4 + 
                        (1 - hallucination_score) * 0.3 + 
                        completeness_score * 0.2 + 
                        citation_score * 0.1)
        
        # Determine if retry is needed
        retry_needed = (
            grounding_score < self.criticism_rules["min_grounding"] or
            hallucination_score > self.criticism_rules["max_hallucination"] or
            completeness_score < self.criticism_rules["min_completeness"] or
            citation_score < 0.5
        )
        
        # Generate specific feedback
        feedback = self._generate_feedback(
            grounding_score, hallucination_score, completeness_score, citation_score
        )
        
        # Suggest improvements
        improvements = self._suggest_improvements(
            grounding_score, hallucination_score, completeness_score, citation_score
        )
        
        return {
            "decision": "RETRY" if retry_needed else "PASS",
            "overall_score": round(overall_score, 3),
            "grounding_score": round(grounding_score, 3),
            "hallucination_score": round(hallucination_score, 3),
            "completeness_score": round(completeness_score, 3),
            "citation_score": round(citation_score, 3),
            "feedback": feedback,
            "improvements": improvements,
            "retry_needed": retry_needed
        }
    
    def _check_grounding(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Check if answer is grounded in retrieved chunks"""
        if not retrieved_docs:
            return 0.0
        
        # Get content from top chunks
        chunk_contents = [doc.get("content", "").lower() for doc in retrieved_docs[:3]]
        answer_lower = answer.lower()
        
        # Check how much of answer is covered by chunks
        total_answer_words = len(answer_lower.split())
        grounded_words = 0
        
        for word in answer_lower.split():
            for chunk in chunk_contents:
                if word in chunk:
                    grounded_words += 1
                    break
        
        return grounded_words / total_answer_words if total_answer_words > 0 else 0.0
    
    def _detect_hallucination(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Detect hallucinated content in answer"""
        if not retrieved_docs:
            return 1.0  # Assume full hallucination if no docs
        
        # Look for specific claims that aren't in chunks
        chunk_contents = [doc.get("content", "").lower() for doc in retrieved_docs[:3]]
        answer_lower = answer.lower()
        
        # Check for specific numbers, dates, facts
        import re
        answer_facts = re.findall(r'\b\d+\b|\b\d{4}\b', answer_lower)
        
        hallucinated_facts = 0
        for fact in answer_facts:
            found = any(fact in chunk for chunk in chunk_contents)
            if not found:
                hallucinated_facts += 1
        
        return hallucinated_facts / len(answer_facts) if answer_facts else 0.0
    
    def _check_completeness(self, answer: str, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Check if answer completely addresses the query"""
        
        # Extract key concepts from query
        query_concepts = set(query.lower().split())
        answer_concepts = set(answer.lower().split())
        
        # Check how many query concepts are addressed
        addressed_concepts = query_concepts.intersection(answer_concepts)
        
        # Base completeness from concept coverage
        concept_coverage = len(addressed_concepts) / len(query_concepts) if query_concepts else 0.0
        
        # Boost if answer has substantial content
        length_factor = min(len(answer.split()) / 50, 1.0)  # Ideal length ~50 words
        
        return (concept_coverage * 0.7 + length_factor * 0.3)
    
    def _check_citations(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Check if answer includes proper citations"""
        
        # Look for citation patterns
        import re
        citation_patterns = [
            r'\[chunk \d+\]',
            r'\[document \d+\]',
            r'\[source \d+\]',
            r'\(\d+\)'
        ]
        
        found_citations = 0
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer.lower())
            found_citations += len(matches)
        
        # Normalize by expected citations
        expected_citations = min(len(retrieved_docs), 3)
        return min(found_citations / expected_citations, 1.0) if expected_citations > 0 else 0.0
    
    def _generate_feedback(self, grounding: float, hallucination: float, 
                          completeness: float, citation: float) -> str:
        """Generate specific feedback based on scores"""
        
        feedback_parts = []
        
        if grounding < 0.7:
            feedback_parts.append("Answer lacks sufficient grounding in source documents")
        
        if hallucination > 0.1:
            feedback_parts.append("Potential hallucination detected - some claims not supported by sources")
        
        if completeness < 0.6:
            feedback_parts.append("Answer does not fully address the query")
        
        if citation < 0.5:
            feedback_parts.append("Insufficient citations provided")
        
        return "; ".join(feedback_parts) if feedback_parts else "Answer meets quality standards"
    
    def _suggest_improvements(self, grounding: float, hallucination: float, 
                             completeness: float, citation: float) -> List[str]:
        """Suggest specific improvements"""
        
        improvements = []
        
        if grounding < 0.7:
            improvements.append("Increase reliance on retrieved chunks")
        
        if hallucination > 0.1:
            improvements.append("Verify all claims against source documents")
        
        if completeness < 0.6:
            improvements.append("Expand answer to address all aspects of query")
        
        if citation < 0.5:
            improvements.append("Add proper citations for source attribution")
        
        return improvements
    
    def get_critic_status(self) -> Dict[str, Any]:
        """Get critic agent status"""
        return {
            "status": "active",
            "rules": self.criticism_rules,
            "version": "10/10 elite"
        }
