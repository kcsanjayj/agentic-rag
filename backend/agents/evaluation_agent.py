"""
Evaluation Agent - Comprehensive metrics for 10/10 elite RAG system
"""

from typing import Dict, Any, List, Tuple
import time
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationAgent:
    """Agent for comprehensive evaluation metrics and scoring"""
    
    def __init__(self):
        self.evaluation_weights = {
            "retrieval_relevance": 0.3,
            "grounding_score": 0.25,
            "answer_completeness": 0.2,
            "response_quality": 0.15,
            "efficiency": 0.1
        }
        
        self.benchmarks = {
            "retrieval_relevance": 0.8,
            "grounding_score": 0.9,
            "answer_completeness": 0.7,
            "response_quality": 0.8,
            "efficiency": 0.9
        }
    
    async def evaluate_response(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]], 
                              agent_trace: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Comprehensive evaluation of RAG response"""
        
        # Calculate individual metrics
        retrieval_relevance = self._calculate_retrieval_relevance(query, retrieved_docs)
        grounding_score = self._calculate_grounding_score(answer, retrieved_docs)
        answer_completeness = self._calculate_answer_completeness(query, answer, retrieved_docs)
        response_quality = self._calculate_response_quality(answer, agent_trace)
        efficiency_score = self._calculate_efficiency_score(processing_time, len(retrieved_docs))
        
        # Calculate overall score
        overall_score = (
            retrieval_relevance * self.evaluation_weights["retrieval_relevance"] +
            grounding_score * self.evaluation_weights["grounding_score"] +
            answer_completeness * self.evaluation_weights["answer_completeness"] +
            response_quality * self.evaluation_weights["response_quality"] +
            efficiency_score * self.evaluation_weights["efficiency"]
        )
        
        # Generate evaluation summary
        evaluation_summary = self._generate_evaluation_summary(
            retrieval_relevance, grounding_score, answer_completeness, 
            response_quality, efficiency_score, overall_score
        )
        
        # Calculate performance grade
        grade = self._calculate_grade(overall_score)
        
        # Identify areas for improvement
        improvements = self._identify_improvements(
            retrieval_relevance, grounding_score, answer_completeness, 
            response_quality, efficiency_score
        )
        
        return {
            "overall_score": round(overall_score, 3),
            "grade": grade,
            "metrics": {
                "retrieval_relevance": round(retrieval_relevance, 3),
                "grounding_score": round(grounding_score, 3),
                "answer_completeness": round(answer_completeness, 3),
                "response_quality": round(response_quality, 3),
                "efficiency": round(efficiency_score, 3)
            },
            "benchmarks": self.benchmarks,
            "summary": evaluation_summary,
            "improvements": improvements,
            "evaluation_time": time.time()
        }
    
    def _calculate_retrieval_relevance(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate retrieval relevance score"""
        
        if not retrieved_docs:
            return 0.0
        
        query_terms = set(query.lower().split())
        relevance_scores = []
        
        for doc in retrieved_docs[:5]:  # Top 5 docs
            content = doc.get("content", "").lower()
            doc_terms = set(content.split())
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(doc_terms))
            relevance = overlap / len(query_terms) if query_terms else 0.0
            
            # Boost by rerank score if available
            rerank_score = doc.get("rerank_score", doc.get("score", 0.5))
            relevance = relevance * 0.7 + rerank_score * 0.3
            
            relevance_scores.append(relevance)
        
        # Return average of top 3
        top_scores = sorted(relevance_scores, reverse=True)[:3]
        return sum(top_scores) / len(top_scores) if top_scores else 0.0
    
    def _calculate_grounding_score(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate grounding score based on source attribution"""
        
        if not retrieved_docs:
            return 0.0
        
        answer_lower = answer.lower()
        chunk_contents = [doc.get("content", "").lower() for doc in retrieved_docs[:3]]
        
        # Check for citations
        import re
        citation_patterns = [r'\[chunk \d+\]', r'\[source \d+\]', r'\(\d+\)']
        found_citations = 0
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, answer_lower)
            found_citations += len(matches)
        
        # Calculate content overlap
        answer_words = answer_lower.split()
        grounded_words = 0
        
        for word in answer_words:
            for chunk in chunk_contents:
                if word in chunk:
                    grounded_words += 1
                    break
        
        content_grounding = grounded_words / len(answer_words) if answer_words else 0.0
        
        # Combine citation and content grounding
        citation_score = min(found_citations / 2, 1.0)  # Expect at least 2 citations
        grounding_score = content_grounding * 0.6 + citation_score * 0.4
        
        return grounding_score
    
    def _calculate_answer_completeness(self, query: str, answer: str, 
                                     retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate answer completeness score"""
        
        query_concepts = set(query.lower().split())
        answer_concepts = set(answer.lower().split())
        
        # Concept coverage
        covered_concepts = query_concepts.intersection(answer_concepts)
        concept_coverage = len(covered_concepts) / len(query_concepts) if query_concepts else 0.0
        
        # Length factor (answers should be substantial but not too long)
        answer_length = len(answer.split())
        length_factor = min(answer_length / 40, 1.0)  # Ideal ~40 words
        
        # Source utilization
        if retrieved_docs:
            source_words = set()
            for doc in retrieved_docs[:3]:
                source_words.update(doc.get("content", "").lower().split())
            
            utilized_source_words = answer_concepts.intersection(source_words)
            source_utilization = len(utilized_source_words) / len(source_words) if source_words else 0.0
        else:
            source_utilization = 0.0
        
        # Combine factors
        completeness = (concept_coverage * 0.5 + 
                       length_factor * 0.3 + 
                       source_utilization * 0.2)
        
        return completeness
    
    def _calculate_response_quality(self, answer: str, agent_trace: Dict[str, Any]) -> float:
        """Calculate response quality score"""
        
        quality_score = 0.0
        
        # Structure and clarity
        if answer.strip():
            quality_score += 0.3
        
        # No repetition
        sentences = answer.split('.')
        unique_sentences = set(sentences)
        if len(unique_sentences) >= len(sentences) * 0.8:
            quality_score += 0.2
        
        # Professional language
        professional_terms = ["analysis", "evaluation", "assessment", "conclusion", "summary"]
        professional_count = sum(1 for term in professional_terms if term in answer.lower())
        if professional_count >= 1:
            quality_score += 0.2
        
        # Proper formatting
        if any(marker in answer for marker in [":", "-", "•"]):
            quality_score += 0.1
        
        # Agent trace quality
        if agent_trace and len(agent_trace) >= 3:
            quality_score += 0.2
        
        return quality_score
    
    def _calculate_efficiency_score(self, processing_time: float, doc_count: int) -> float:
        """Calculate efficiency score"""
        
        # Time efficiency (under 2 seconds is good)
        time_efficiency = max(0, 1 - (processing_time / 2.0))
        
        # Document processing efficiency
        if doc_count == 0:
            doc_efficiency = 0.0
        else:
            doc_efficiency = min(1.0, 10 / doc_count)  # Expect ~10 docs max
        
        # Combine efficiency metrics
        efficiency = (time_efficiency * 0.6 + doc_efficiency * 0.4)
        
        return efficiency
    
    def _generate_evaluation_summary(self, retrieval_relevance: float, grounding_score: float,
                                   answer_completeness: float, response_quality: float,
                                   efficiency: float, overall_score: float) -> str:
        """Generate evaluation summary"""
        
        summary_parts = []
        
        if retrieval_relevance >= 0.8:
            summary_parts.append("Excellent retrieval relevance")
        elif retrieval_relevance >= 0.6:
            summary_parts.append("Good retrieval relevance")
        else:
            summary_parts.append("Poor retrieval relevance")
        
        if grounding_score >= 0.9:
            summary_parts.append("Strong grounding in sources")
        elif grounding_score >= 0.7:
            summary_parts.append("Adequate grounding")
        else:
            summary_parts.append("Weak grounding")
        
        if answer_completeness >= 0.7:
            summary_parts.append("Complete answer")
        elif answer_completeness >= 0.5:
            summary_parts.append("Partially complete answer")
        else:
            summary_parts.append("Incomplete answer")
        
        if overall_score >= 0.8:
            summary_parts.append("Overall excellent performance")
        elif overall_score >= 0.6:
            summary_parts.append("Overall good performance")
        else:
            summary_parts.append("Overall poor performance")
        
        return "; ".join(summary_parts)
    
    def _calculate_grade(self, overall_score: float) -> str:
        """Calculate performance grade"""
        
        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B"
        elif overall_score >= 0.6:
            return "C"
        elif overall_score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _identify_improvements(self, retrieval_relevance: float, grounding_score: float,
                             answer_completeness: float, response_quality: float,
                             efficiency: float) -> List[str]:
        """Identify areas for improvement"""
        
        improvements = []
        
        if retrieval_relevance < 0.7:
            improvements.append("Improve query understanding and retrieval precision")
        
        if grounding_score < 0.8:
            improvements.append("Enhance source attribution and reduce hallucination")
        
        if answer_completeness < 0.6:
            improvements.append("Provide more comprehensive answers")
        
        if response_quality < 0.7:
            improvements.append("Improve response structure and clarity")
        
        if efficiency < 0.7:
            improvements.append("Optimize processing time and resource usage")
        
        return improvements
    
    def format_evaluation_output(self, evaluation: Dict[str, Any]) -> str:
        """Format evaluation for output (interview-killer format)"""
        
        metrics = evaluation["metrics"]
        
        output_parts = []
        output_parts.append("EVALUATION:")
        output_parts.append(f"- Retrieval Relevance: {metrics['retrieval_relevance']:.2f}")
        output_parts.append(f"- Grounding Score: {metrics['grounding_score']:.2f}")
        output_parts.append(f"- Answer Completeness: {metrics['answer_completeness']:.2f}")
        output_parts.append(f"- Response Quality: {metrics['response_quality']:.2f}")
        output_parts.append(f"- Efficiency: {metrics['efficiency']:.2f}")
        output_parts.append(f"- Overall Score: {evaluation['overall_score']:.2f} ({evaluation['grade']})")
        
        return "\n".join(output_parts)
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """Get evaluation agent status"""
        return {
            "status": "active",
            "weights": self.evaluation_weights,
            "benchmarks": self.benchmarks,
            "version": "10/10 elite"
        }
