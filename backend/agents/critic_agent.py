"""
Critic Agent - Self-criticism and validation
Detects weak responses and forces regeneration
"""

import re
from typing import Dict, Any, List
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class CriticAgent:
    """
    🔍 Critic Agent - Self-checks responses for quality
    Forces regeneration for generic/short responses
    """
    
    def __init__(self):
        self.generic_phrases = [
            "i don't have", "no information", "not mentioned", "not specified",
            "cannot determine", "unable to", "no specific", "not provided",
            "no details", "generic response", "template"
        ]
        self.min_response_length = 120
        self.min_summary_length = 50
        self.min_key_points = 2
        
        logger.info("🔍 Critic Agent initialized")
    
    def critique(self, answer: str, context: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critique and validate the generated response
        
        Args:
            answer: Generated response
            context: Original document context
            task: Task information
            
        Returns:
            Critique results with validation score
        """
        try:
            score = 0.0
            issues = []
            
            # 🔥 IMMEDIATE SELF-CRITIC: Generic or short responses
            if "generic" in answer.lower() or len(answer) < 120:
                logger.warning("🔥 SELF-CRITIC: Generic or short response detected!")
                return {
                    "is_valid": False,
                    "score": 0.0,
                    "issues": ["Generic/short response - immediate regeneration needed"],
                    "recommendations": ["REGENERATE_IMMEDIATELY"],
                    "metrics": {"response_length": len(answer), "force_regenerate": True},
                    "force_regenerate": True
                }
            
            # Check for generic filler phrases
            generic_count = 0
            for phrase in self.generic_phrases:
                if phrase.lower() in answer.lower():
                    generic_count += 1
                    issues.append(f"Contains generic phrase: '{phrase}'")
            
            if generic_count == 0:
                score += 0.3
            
            # Check response length
            if len(answer) >= self.min_response_length:
                score += 0.2
            else:
                issues.append(f"Response too short: {len(answer)} chars")
            
            # Check for proper structure
            has_summary = "Summary:" in answer
            has_key_points = "Key Points:" in answer or "•" in answer
            
            if has_summary and has_key_points:
                score += 0.2
            
            # Check grounding
            context_words = set(context.lower().split()[:200])
            answer_words = set(answer.lower().split())
            overlap = len(context_words.intersection(answer_words))
            
            if overlap > 10:
                score += 0.1
            
            # Final validation
            is_valid = score >= 0.6
            
            logger.info(f"🔍 Critic score: {score:.2f}, Valid: {is_valid}")
            
            return {
                "is_valid": is_valid,
                "score": score,
                "issues": issues,
                "recommendations": ["Regenerate" if not is_valid else "Accept"],
                "metrics": {
                    "response_length": len(answer),
                    "grounding_overlap": overlap
                },
                "force_regenerate": not is_valid
            }
            
        except Exception as e:
            logger.error(f"Error in critic: {str(e)}")
            return {
                "is_valid": False,
                "score": 0.0,
                "issues": [f"Critic error: {str(e)}"],
                "recommendations": ["Retry generation"],
                "force_regenerate": True
            }
