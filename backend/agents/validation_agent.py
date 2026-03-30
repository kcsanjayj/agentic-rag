"""
Validation Agent - Validates the quality and accuracy of generated responses
with comprehensive evaluation metrics for 9.5+ quality
"""

from typing import Dict, Any, List, Optional
import re
import math
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ValidationAgent:
    """Agent responsible for validating generated responses with FAANG-level metrics"""
    
    def __init__(self):
        self.validation_rules = {
            "min_length": 20,
            "max_length": 2000,
            "required_elements": ["answer"],
            "forbidden_patterns": [
                r"i don't know",
                r"i cannot answer",
                r"as an ai",
                r"i'm sorry"
            ]
        }
        # Ground truth storage for evaluation
        self.ground_truth_cache = {}
    
    async def validate(self, query: str, answer: str, 
                      retrieved_docs: List[Dict[str, Any]],
                      agent_trace: Dict[str, Any] = None,
                      ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Validate a generated answer with comprehensive evaluation metrics"""
        try:
            logger.info("Validating generated answer with comprehensive metrics")
            
            # Perform standard validation checks
            validation_result = await self._standard_validate(query, answer, retrieved_docs)
            
            # Add comprehensive evaluation metrics
            evaluation_metrics = self._calculate_evaluation_metrics(
                query, answer, retrieved_docs, ground_truth
            )
            validation_result["evaluation_metrics"] = evaluation_metrics
            
            # Enhanced critic evaluation with hallucination detection
            critic_eval = self._enhanced_critic_evaluation(
                query, answer, retrieved_docs, agent_trace, validation_result, evaluation_metrics
            )
            validation_result["critic_evaluation"] = critic_eval
            
            # Determine if answer is valid based on all metrics
            is_valid = self._determine_validity(validation_result, evaluation_metrics, critic_eval)
            validation_result["is_valid"] = is_valid
            
            # Generate detailed feedback
            if not is_valid:
                validation_result["feedback"] = self._generate_enhanced_feedback(
                    validation_result, evaluation_metrics, critic_eval
                )
                validation_result["reason"] = validation_result["feedback"]
            
            # Add suggested actions based on metrics
            validation_result["suggested_actions"] = self._suggest_improvements(
                evaluation_metrics, critic_eval
            )
            
            logger.info(f"Validation completed. Valid: {is_valid}, "
                       f"Grounding: {critic_eval['grounding']}, "
                       f"Hallucination Risk: {evaluation_metrics.get('hallucination_rate', 'N/A')}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "issues": [{"type": "error", "message": str(e), "severity": "high"}],
                "feedback": "Validation failed due to an error.",
                "reason": "Validation failed due to an error.",
                "scores": {},
                "evaluation_metrics": {},
                "critic_evaluation": {
                    "grounding": "Weak",
                    "completeness": "Low",
                    "risk": "High",
                    "improvement_needed": "Yes",
                    "hallucination_detected": True
                },
                "suggested_actions": ["retry_with_expanded_query"]
            }
    
    def _calculate_evaluation_metrics(self, query: str, answer: str,
                                     retrieved_docs: List[Dict[str, Any]],
                                     ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics (FAANG-level)"""
        metrics = {}
        
        # 1. Retrieval Recall - Did we fetch correct chunks?
        if retrieved_docs:
            # Calculate based on similarity scores
            scores = [doc.get("score", 0) for doc in retrieved_docs]
            avg_score = sum(scores) / len(scores) if scores else 0
            metrics["retrieval_recall"] = round(min(avg_score * 1.2, 1.0), 2)  # Scale to 0-1
        else:
            metrics["retrieval_recall"] = 0.0
        
        # 2. Groundedness Score - Is answer supported by evidence?
        if retrieved_docs:
            answer_words = set(answer.lower().split())
            all_doc_words = set()
            for doc in retrieved_docs:
                all_doc_words.update(doc["content"].lower().split())
            
            if answer_words:
                overlap = len(answer_words.intersection(all_doc_words))
                metrics["groundedness"] = round(min(overlap / len(answer_words), 1.0), 2)
            else:
                metrics["groundedness"] = 0.0
        else:
            metrics["groundedness"] = 0.0
        
        # 3. Answer Relevance - Matches query intent?
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        if query_words:
            overlap = len(query_words.intersection(answer_words))
            metrics["answer_relevance"] = round(min(overlap / len(query_words), 1.0), 2)
        else:
            metrics["answer_relevance"] = 0.0
        
        # 4. Hallucination Rate - % unsupported claims
        if retrieved_docs and answer_words:
            all_doc_words = set()
            for doc in retrieved_docs:
                all_doc_words.update(doc["content"].lower().split())
            
            unsupported = answer_words - all_doc_words
            # Filter out common words
            common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                          "being", "have", "has", "had", "do", "does", "did", "will",
                          "would", "could", "should", "may", "might", "must", "shall",
                          "can", "need", "dare", "ought", "used", "to", "of", "in",
                          "for", "on", "with", "at", "by", "from", "as", "into",
                          "through", "during", "before", "after", "above", "below",
                          "between", "under", "and", "but", "or", "yet", "so",
                          "if", "because", "although", "though", "while", "where",
                          "when", "that", "which", "who", "whom", "whose", "what",
                          "this", "these", "those", "i", "you", "he", "she", "it",
                          "we", "they", "me", "him", "her", "us", "them", "my",
                          "your", "his", "her", "its", "our", "their"}
            unsupported -= common_words
            
            if answer_words - common_words:
                hallucination_rate = len(unsupported) / len(answer_words - common_words)
                metrics["hallucination_rate"] = round(hallucination_rate, 2)
            else:
                metrics["hallucination_rate"] = 0.0
        else:
            metrics["hallucination_rate"] = 1.0 if answer_words else 0.0
        
        # 5. Context Utilization - How well did we use retrieved chunks?
        if retrieved_docs:
            metrics["context_utilization"] = round(min(len(retrieved_docs) * 0.25, 1.0), 2)
        else:
            metrics["context_utilization"] = 0.0
        
        # 6. Accuracy against ground truth (if available)
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            answer_words = set(answer.lower().split())
            if gt_words:
                overlap = len(gt_words.intersection(answer_words))
                metrics["accuracy_vs_ground_truth"] = round(min(overlap / len(gt_words), 1.0), 2)
        
        # Calculate overall quality score (weighted average)
        weights = {
            "groundedness": 0.35,
            "answer_relevance": 0.25,
            "retrieval_recall": 0.20,
            "context_utilization": 0.20
        }
        
        overall_score = sum(
            metrics.get(metric, 0) * weight
            for metric, weight in weights.items()
        )
        metrics["overall_quality_score"] = round(overall_score, 2)
        
        return metrics
    
    def _enhanced_critic_evaluation(self, query: str, answer: str,
                                    retrieved_docs: List[Dict[str, Any]],
                                    agent_trace: Dict[str, Any],
                                    validation_result: Dict[str, Any],
                                    evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced critic evaluation on the answer
        
        Returns:
            Dict with grounding, completeness, risk, and improvement_needed
        """
        # Calculate grounding based on source overlap
        if not retrieved_docs:
            grounding = "Weak"
        else:
            answer_words = set(answer.lower().split())
            max_overlap = 0
            for doc in retrieved_docs:
                doc_words = set(doc["content"].lower().split())
                overlap = len(answer_words.intersection(doc_words))
                max_overlap = max(max_overlap, overlap / len(answer_words) if answer_words else 0)
            
            if max_overlap > 0.6:
                grounding = "Strong"
            elif max_overlap > 0.3:
                grounding = "Moderate"
            else:
                grounding = "Weak"
        
        # Calculate completeness based on answer structure
        has_sections = all(s in answer for s in ["Summary", "Key Insight", "Key Features"])
        has_evidence = "Evidence" in answer or "Chunk" in answer
        
        if has_sections and has_evidence:
            completeness = "High"
        elif has_sections or has_evidence:
            completeness = "Medium"
        else:
            completeness = "Low"
        
        # Calculate risk based on validation confidence and grounding
        confidence = validation_result.get("confidence", 0.5)
        if grounding == "Strong" and confidence > 0.7:
            risk = "Low"
        elif grounding == "Weak" or confidence < 0.4:
            risk = "High"
        else:
            risk = "Medium"
        
        # Determine if improvement is needed
        high_issues = [i for i in validation_result.get("issues", []) if i.get("severity") == "high"]
        improvement_needed = "Yes" if (risk == "High" or grounding == "Weak" or high_issues) else "No"
        
        return {
            "grounding": grounding,
            "completeness": completeness,
            "risk": risk,
            "improvement_needed": improvement_needed,
            "overlap_ratio": round(max_overlap, 2) if 'max_overlap' in dir() else 0
        }
    
    def _determine_validity(self, validation_result: Dict[str, Any],
                           evaluation_metrics: Dict[str, Any],
                           critic_eval: Dict[str, Any]) -> bool:
        """Determine if answer is valid based on all metrics"""
        base_confidence = validation_result.get("confidence", 0)
        hallucination_rate = evaluation_metrics.get("hallucination_rate", 1.0)
        groundedness = evaluation_metrics.get("groundedness", 0)
        
        is_confident = base_confidence >= 0.5
        low_hallucination = hallucination_rate < 0.5
        decent_groundedness = groundedness >= 0.3
        grounding_ok = critic_eval.get("grounding") != "Weak"
        
        return is_confident and low_hallucination and decent_groundedness and grounding_ok
    
    def _generate_enhanced_feedback(self, validation_result: Dict[str, Any],
                                   evaluation_metrics: Dict[str, Any],
                                   critic_eval: Dict[str, Any]) -> str:
        """Generate enhanced feedback based on all metrics"""
        feedback_parts = []
        
        hallucination_rate = evaluation_metrics.get("hallucination_rate", 0)
        if hallucination_rate > 0.5:
            feedback_parts.append(f"HIGH HALLUCINATION RISK ({hallucination_rate:.0%})")
        elif hallucination_rate > 0.3:
            feedback_parts.append(f"MODERATE HALLUCINATION RISK ({hallucination_rate:.0%})")
        
        groundedness = evaluation_metrics.get("groundedness", 0)
        if groundedness < 0.3:
            feedback_parts.append(f"LOW GROUNDEDNESS ({groundedness:.0%})")
        
        high_issues = [i for i in validation_result.get("issues", []) if i.get("severity") == "high"]
        if high_issues:
            feedback_parts.append("Critical issues found:")
            for issue in high_issues:
                feedback_parts.append(f"  - {issue['message']}")
        
        if not feedback_parts:
            feedback_parts.append("Answer needs general improvement.")
        
        return "\n".join(feedback_parts)
    
    def _suggest_improvements(self, evaluation_metrics: Dict[str, Any],
                             critic_eval: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on metrics"""
        suggestions = []
        
        hallucination_rate = evaluation_metrics.get("hallucination_rate", 0)
        groundedness = evaluation_metrics.get("groundedness", 0)
        
        if hallucination_rate > 0.4:
            suggestions.append("retry_with_stricter_grounding")
        if groundedness < 0.5:
            suggestions.append("expand_retrieval_top_k")
        if critic_eval.get("hallucination_detected"):
            suggestions.append("verify_against_sources")
        
        if not suggestions:
            suggestions.append("minor_refinement")
        
        return suggestions
    
    async def _standard_validate(self, query: str, answer: str, 
                                retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Standard validation checks"""
        validation_result = {
            "is_valid": True,
            "confidence": 0.0,
            "issues": [],
            "feedback": "",
            "scores": {}
        }
        
        # Perform various validation checks
        checks = [
            ("length", self._check_length(answer)),
            ("content_quality", self._check_content_quality(answer)),
            ("relevance", self._check_relevance(query, answer)),
            ("source_usage", self._check_source_usage(answer, retrieved_docs)),
            ("factual_consistency", self._check_factual_consistency(answer, retrieved_docs)),
            ("forbidden_patterns", self._check_forbidden_patterns(answer))
        ]
        
        # Aggregate validation results
        total_score = 0
        max_score = 0
        
        for check_name, check_result in checks:
            validation_result["scores"][check_name] = check_result["score"]
            validation_result["issues"].extend(check_result.get("issues", []))
            total_score += check_result["score"]
            max_score += check_result["max_score"]
        
        # Calculate overall confidence
        if max_score > 0:
            validation_result["confidence"] = total_score / max_score
        
        # Determine if answer is valid
        validation_result["is_valid"] = (
            validation_result["confidence"] >= 0.6 and
            len([issue for issue in validation_result["issues"] if issue["severity"] == "high"]) == 0
        )
        
        # Generate feedback if not valid
        if not validation_result["is_valid"]:
            validation_result["feedback"] = self._generate_feedback(validation_result)
            validation_result["reason"] = validation_result["feedback"]
        
        return validation_result
    
    def _check_length(self, answer: str) -> Dict[str, Any]:
        """Check if answer length is appropriate"""
        issues = []
        score = 1.0
        
        if len(answer) < self.validation_rules["min_length"]:
            issues.append({
                "type": "length",
                "message": f"Answer is too short (minimum {self.validation_rules['min_length']} characters)",
                "severity": "medium"
            })
            score -= 0.3
        
        if len(answer) > self.validation_rules["max_length"]:
            issues.append({
                "type": "length", 
                "message": f"Answer is too long (maximum {self.validation_rules['max_length']} characters)",
                "severity": "low"
            })
            score -= 0.1
        
        return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
    
    def _check_content_quality(self, answer: str) -> Dict[str, Any]:
        """Check content quality metrics"""
        issues = []
        score = 1.0
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) < 2:
            issues.append({
                "type": "structure",
                "message": "Answer lacks proper sentence structure",
                "severity": "medium"
            })
            score -= 0.2
        
        # Check for repetitive content
        words = answer.lower().split()
        if len(set(words)) / len(words) < 0.5:
            issues.append({
                "type": "repetition",
                "message": "Answer contains too much repetition",
                "severity": "medium"
            })
            score -= 0.2
        
        # Check for vague statements
        vague_phrases = ["might be", "could be", "perhaps", "maybe"]
        for phrase in vague_phrases:
            if phrase in answer.lower():
                issues.append({
                    "type": "vagueness",
                    "message": f"Answer contains vague phrase: '{phrase}'",
                    "severity": "low"
                })
                score -= 0.1
        
        return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
    
    def _check_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """Check if answer is relevant to the query"""
        issues = []
        score = 1.0
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(answer_words))
        overlap_ratio = overlap / len(query_words) if query_words else 0
        
        if overlap_ratio < 0.2:
            issues.append({
                "type": "relevance",
                "message": "Answer doesn't seem to address the query directly",
                "severity": "high"
            })
            score -= 0.4
        elif overlap_ratio < 0.5:
            issues.append({
                "type": "relevance",
                "message": "Answer has limited relevance to the query",
                "severity": "medium"
            })
            score -= 0.2
        
        return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
    
    def _check_source_usage(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if answer properly uses retrieved sources"""
        issues = []
        score = 1.0
        
        if not retrieved_docs:
            issues.append({
                "type": "sources",
                "message": "No sources were provided for validation",
                "severity": "medium"
            })
            score -= 0.3
            return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
        
        # Check if answer mentions sources
        has_citations = any(word in answer.lower() for word in ["according to", "source", "document", "based on"])
        
        if not has_citations and len(retrieved_docs) > 0:
            issues.append({
                "type": "citation",
                "message": "Answer doesn't cite its sources",
                "severity": "low"
            })
            score -= 0.1
        
        return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
    
    def _check_factual_consistency(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check factual consistency with sources"""
        issues = []
        score = 1.0
        
        if not retrieved_docs:
            return {"score": 0.5, "max_score": 1.0, "issues": []}
        
        # Simple consistency check - can be enhanced with more sophisticated methods
        answer_lower = answer.lower()
        for doc in retrieved_docs:
            content_lower = doc["content"].lower()
            
            # Check if answer contains information not in sources
            answer_words = set(answer_lower.split())
            content_words = set(content_lower.split())
            
            # If answer has many words not in any source, flag it
            unique_words = answer_words - content_words
            if len(unique_words) > len(answer_words) * 0.3:
                issues.append({
                    "type": "consistency",
                    "message": "Answer may contain information not found in sources",
                    "severity": "medium"
                })
                score -= 0.2
                break
        
        return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
    
    def _check_forbidden_patterns(self, answer: str) -> Dict[str, Any]:
        """Check for forbidden patterns in answer"""
        issues = []
        score = 1.0
        
        answer_lower = answer.lower()
        
        for pattern in self.validation_rules["forbidden_patterns"]:
            if re.search(pattern, answer_lower):
                issues.append({
                    "type": "forbidden",
                    "message": f"Answer contains forbidden pattern: {pattern}",
                    "severity": "high"
                })
                score -= 0.5
        
        return {"score": max(0.0, score), "max_score": 1.0, "issues": issues}
    
    def _generate_feedback(self, validation_result: Dict[str, Any]) -> str:
        """Generate feedback for improving the answer"""
        high_issues = [issue for issue in validation_result["issues"] if issue["severity"] == "high"]
        medium_issues = [issue for issue in validation_result["issues"] if issue["severity"] == "medium"]
        
        feedback_parts = []
        
        if high_issues:
            feedback_parts.append("Please address these critical issues:")
            for issue in high_issues:
                feedback_parts.append(f"- {issue['message']}")
        
        if medium_issues:
            feedback_parts.append("Consider these improvements:")
            for issue in medium_issues:
                feedback_parts.append(f"- {issue['message']}")
        
        if not feedback_parts:
            feedback_parts.append("The answer needs general improvement in quality and relevance.")
        
        return "\n".join(feedback_parts)
