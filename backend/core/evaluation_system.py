"""
Evaluation System - Auto-metrics for faithfulness, relevance, and quality
"""

from typing import Dict, Any, List, Optional
import time
import json
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationSystem:
    """Comprehensive evaluation system for 9.5+ quality metrics"""
    
    def __init__(self):
        self.evaluation_history = []
        self.metrics_weights = {
            "faithfulness": 0.30,
            "answer_relevance": 0.25,
            "context_utilization": 0.20,
            "groundedness": 0.15,
            "completeness": 0.10
        }
    
    async def evaluate_response(self, query: str, answer: str, 
                               retrieved_docs: List[Dict[str, Any]],
                               ground_truth: Optional[str] = None,
                               agent_trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive evaluation with auto-metrics"""
        
        evaluation_start = time.time()
        
        # Calculate all metrics
        metrics = {
            "faithfulness": await self._calculate_faithfulness(answer, retrieved_docs),
            "answer_relevance": self._calculate_answer_relevance(query, answer),
            "context_utilization": self._calculate_context_utilization(answer, retrieved_docs),
            "groundedness": self._calculate_groundedness(answer, retrieved_docs),
            "completeness": self._calculate_completeness(query, answer, retrieved_docs)
        }
        
        # Add optional ground truth accuracy
        if ground_truth:
            metrics["accuracy_vs_ground_truth"] = self._calculate_ground_truth_accuracy(answer, ground_truth)
        
        # Calculate overall score
        overall_score = sum(
            metrics[metric] * self.metrics_weights[metric] 
            for metric in self.metrics_weights
        )
        metrics["overall_score"] = round(overall_score, 3)
        
        # Performance metrics
        metrics["evaluation_time"] = time.time() - evaluation_start
        
        # Quality classification
        metrics["quality_tier"] = self._classify_quality(overall_score)
        
        # Detailed analysis
        metrics["detailed_analysis"] = await self._detailed_analysis(
            query, answer, retrieved_docs, metrics
        )
        
        # Store evaluation
        evaluation_record = {
            "timestamp": time.time(),
            "query": query,
            "answer": answer,
            "metrics": metrics,
            "retrieved_docs_count": len(retrieved_docs),
            "agent_trace": agent_trace
        }
        self.evaluation_history.append(evaluation_record)
        
        return metrics
    
    async def _calculate_faithfulness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness score (1.0 = fully supported by sources)"""
        if not retrieved_docs:
            return 0.0
        
        answer_sentences = self._split_into_sentences(answer)
        faithful_sentences = 0
        
        for sentence in answer_sentences:
            if self._is_sentence_supported(sentence, retrieved_docs):
                faithful_sentences += 1
        
        return faithful_sentences / len(answer_sentences) if answer_sentences else 0.0
    
    def _is_sentence_supported(self, sentence: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
        """Check if a sentence is supported by retrieved documents"""
        sentence_words = set(sentence.lower().split())
        
        for doc in retrieved_docs:
            doc_words = set(doc["content"].lower().split())
            overlap = len(sentence_words.intersection(doc_words))
            
            # If 60%+ of sentence words are in a document, consider it supported
            if len(sentence_words) > 0 and overlap / len(sentence_words) >= 0.6:
                return True
        
        return False
    
    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """Calculate answer relevance to query"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(answer_words))
        return overlap / len(query_words)
    
    def _calculate_context_utilization(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well the answer utilizes retrieved context"""
        if not retrieved_docs:
            return 0.0
        
        answer_words = set(answer.lower().split())
        all_doc_words = set()
        
        for doc in retrieved_docs:
            all_doc_words.update(doc["content"].lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words.intersection(all_doc_words))
        return overlap / len(answer_words)
    
    def _calculate_groundedness(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate groundedness (claims backed by sources)"""
        if not retrieved_docs:
            return 0.0
        
        # Check for citations
        has_citations = any(pattern in answer.lower() for pattern in [
            "according to", "source", "document", "based on", "chunk", "as shown"
        ])
        
        # Base groundedness on context utilization
        base_score = self._calculate_context_utilization(answer, retrieved_docs)
        
        # Boost if citations are present
        if has_citations:
            base_score = min(base_score + 0.2, 1.0)
        
        return base_score
    
    def _calculate_completeness(self, query: str, answer: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate answer completeness"""
        # Check if answer addresses different aspects of query
        query_aspects = self._extract_query_aspects(query)
        addressed_aspects = 0
        
        for aspect in query_aspects:
            if aspect.lower() in answer.lower():
                addressed_aspects += 1
        
        aspect_score = addressed_aspects / len(query_aspects) if query_aspects else 0.5
        
        # Length-based completeness (longer answers tend to be more complete)
        length_score = min(len(answer) / 200, 1.0)  # Normalize to 200 chars
        
        return (aspect_score + length_score) / 2
    
    def _extract_query_aspects(self, query: str) -> List[str]:
        """Extract key aspects from query"""
        # Simple aspect extraction - can be enhanced with NLP
        aspects = []
        
        # Look for question words and phrases
        question_indicators = ["what", "how", "why", "where", "when", "who", "which"]
        for indicator in question_indicators:
            if indicator in query.lower():
                aspects.append(indicator)
        
        # Look for key nouns (simplified)
        words = query.lower().split()
        for word in words:
            if len(word) > 4 and word not in question_indicators:
                aspects.append(word)
        
        return aspects[:5]  # Top 5 aspects
    
    def _calculate_ground_truth_accuracy(self, answer: str, ground_truth: str) -> float:
        """Calculate accuracy against ground truth"""
        answer_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if not gt_words:
            return 0.0
        
        overlap = len(answer_words.intersection(gt_words))
        return overlap / len(gt_words)
    
    def _classify_quality(self, overall_score: float) -> str:
        """Classify response quality tier"""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "acceptable"
        elif overall_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    async def _detailed_analysis(self, query: str, answer: str, 
                               retrieved_docs: List[Dict[str, Any]], 
                               metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis of the response"""
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "risk_factors": []
        }
        
        # Analyze strengths
        if metrics["faithfulness"] > 0.8:
            analysis["strengths"].append("High faithfulness to sources")
        if metrics["answer_relevance"] > 0.8:
            analysis["strengths"].append("Highly relevant to query")
        if metrics["groundedness"] > 0.8:
            analysis["strengths"].append("Well-grounded with citations")
        
        # Analyze weaknesses
        if metrics["faithfulness"] < 0.5:
            analysis["weaknesses"].append("Low faithfulness - potential hallucination")
            analysis["risk_factors"].append("hallucination")
        if metrics["answer_relevance"] < 0.5:
            analysis["weaknesses"].append("Low relevance to query")
        if metrics["context_utilization"] < 0.5:
            analysis["weaknesses"].append("Poor use of retrieved context")
        
        # Generate recommendations
        if metrics["faithfulness"] < 0.7:
            analysis["recommendations"].append("Improve source grounding")
        if metrics["answer_relevance"] < 0.7:
            analysis["recommendations"].append("Focus on query intent")
        if metrics["completeness"] < 0.7:
            analysis["recommendations"].append("Provide more comprehensive answer")
        
        return analysis
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_evaluation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent evaluations"""
        recent_evaluations = self.evaluation_history[-last_n:]
        
        if not recent_evaluations:
            return {"message": "No evaluations available"}
        
        # Calculate averages
        avg_metrics = {
            "faithfulness": sum(e["metrics"]["faithfulness"] for e in recent_evaluations) / len(recent_evaluations),
            "answer_relevance": sum(e["metrics"]["answer_relevance"] for e in recent_evaluations) / len(recent_evaluations),
            "context_utilization": sum(e["metrics"]["context_utilization"] for e in recent_evaluations) / len(recent_evaluations),
            "groundedness": sum(e["metrics"]["groundedness"] for e in recent_evaluations) / len(recent_evaluations),
            "overall_score": sum(e["metrics"]["overall_score"] for e in recent_evaluations) / len(recent_evaluations)
        }
        
        # Quality distribution
        quality_distribution = {}
        for evaluation in recent_evaluations:
            tier = evaluation["metrics"]["quality_tier"]
            quality_distribution[tier] = quality_distribution.get(tier, 0) + 1
        
        return {
            "total_evaluations": len(recent_evaluations),
            "average_metrics": {k: round(v, 3) for k, v in avg_metrics.items()},
            "quality_distribution": quality_distribution,
            "trend": "improving" if len(recent_evaluations) > 1 and 
                     recent_evaluations[-1]["metrics"]["overall_score"] > recent_evaluations[0]["metrics"]["overall_score"] 
                     else "stable"
        }


# Global evaluation system instance
evaluation_system = EvaluationSystem()
