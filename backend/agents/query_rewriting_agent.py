"""
Query Rewriting Agent - Expands and rewrites queries for better retrieval
"""

from typing import Dict, Any, Optional
from backend.utils.logger import setup_logger
from backend.config import settings

logger = setup_logger(__name__)


class QueryRewritingAgent:
    """Agent responsible for rewriting queries to improve retrieval quality"""
    
    def __init__(self):
        self.enabled = settings.ENABLE_QUERY_REWRITING
        self.expansion_templates = {
            "summarize": [
                "content and purpose of the document",
                "main topics and key information",
                "document structure and sections"
            ],
            "what is": [
                "definition and explanation",
                "key characteristics and features",
                "purpose and usage"
            ],
            "how to": [
                "step-by-step process",
                "implementation details",
                "practical examples"
            ],
            "explain": [
                "detailed explanation",
                "underlying concepts",
                "practical implications"
            ],
            "compare": [
                "similarities and differences",
                "key distinctions",
                "use case differences"
            ],
            "use cases": [
                "practical applications",
                "real-world scenarios",
                "implementation examples"
            ],
            "features": [
                "key capabilities",
                "functional characteristics",
                "technical specifications"
            ]
        }
    
    async def rewrite_query(self, original_query: str, query_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Rewrite query for better retrieval
        
        Returns:
            Dict with original, rewritten, and expanded queries
        """
        if not self.enabled:
            return {
                "original": original_query,
                "rewritten": original_query,
                "expanded": [original_query],
                "strategy": "none"
            }
        
        try:
            logger.info(f"Rewriting query: {original_query}")
            
            # Detect query intent
            query_lower = original_query.lower()
            intent = self._detect_intent(query_lower)
            
            # Expand query with related terms
            expanded_terms = self._expand_query(query_lower, intent)
            
            # Build rewritten query
            rewritten = self._build_rewritten_query(original_query, expanded_terms, intent)
            
            # Create multiple variations for better coverage
            variations = self._create_variations(original_query, rewritten, expanded_terms, intent)
            
            result = {
                "original": original_query,
                "rewritten": rewritten,
                "expanded": variations,
                "intent": intent,
                "strategy": "expansion",
                "expanded_terms": expanded_terms
            }
            
            logger.info(f"Query rewritten: {original_query} -> {rewritten}")
            return result
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return {
                "original": original_query,
                "rewritten": original_query,
                "expanded": [original_query],
                "strategy": "fallback"
            }
    
    def _detect_intent(self, query_lower: str) -> str:
        """Detect query intent from keywords"""
        if any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return "summarize"
        elif any(word in query_lower for word in ["what is", "what's", "define"]):
            return "what is"
        elif any(word in query_lower for word in ["how to", "how do", "steps"]):
            return "how to"
        elif any(word in query_lower for word in ["explain", "describe", "elaborate"]):
            return "explain"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "compare"
        elif any(word in query_lower for word in ["use cases", "applications", "usage"]):
            return "use cases"
        elif any(word in query_lower for word in ["features", "capabilities", "functions"]):
            return "features"
        else:
            return "general"
    
    def _expand_query(self, query_lower: str, intent: str) -> list:
        """Expand query with related terms based on intent"""
        expansions = []
        
        # Get intent-specific expansions
        if intent in self.expansion_templates:
            expansions.extend(self.expansion_templates[intent])
        
        # Add semantic expansions based on keywords
        if "pdf" in query_lower:
            expansions.extend(["document format", "portable document", "file type"])
        if "test" in query_lower:
            expansions.extend(["testing", "validation", "verification"])
        if "mobile" in query_lower:
            expansions.extend(["smartphone", "tablet", "portable device"])
        if "render" in query_lower:
            expansions.extend(["display", "visualization", "processing"])
        if "download" in query_lower or "size" in query_lower:
            expansions.extend(["file size", "bandwidth", "transfer speed"])
        
        return list(set(expansions))  # Remove duplicates
    
    def _build_rewritten_query(self, original: str, expanded_terms: list, intent: str) -> str:
        """Build the rewritten query"""
        if not expanded_terms:
            return original
        
        # Create a comprehensive query that includes original + expansions
        # Format: "Original query focusing on: expansion1, expansion2, expansion3"
        
        # Limit expansions to avoid too long queries
        selected_terms = expanded_terms[:5]
        
        if intent == "summarize":
            return f"{original} focusing on: {', '.join(selected_terms)}"
        elif intent == "what is":
            return f"{original} including: {', '.join(selected_terms)}"
        elif intent == "use cases":
            return f"{original} and practical applications: {', '.join(selected_terms)}"
        else:
            return f"{original} related to: {', '.join(selected_terms)}"
    
    def _create_variations(self, original: str, rewritten: str, expanded_terms: list, intent: str) -> list:
        """Create multiple query variations for better coverage"""
        variations = [original, rewritten]
        
        # Add intent-specific variations
        if intent == "summarize":
            variations.append(f"main content and key points of {original}")
            variations.append(f"overview and purpose: {original}")
        elif intent == "use cases":
            variations.append(f"practical scenarios and applications for {original}")
            variations.append(f"when and how to use: {original}")
        elif intent == "features":
            variations.append(f"key characteristics and capabilities: {original}")
            variations.append(f"what makes {original} unique and useful")
        
        # Add keyword-focused variations
        if expanded_terms:
            keyword_variation = f"{original} (keywords: {', '.join(expanded_terms[:3])})"
            variations.append(keyword_variation)
        
        return list(set(variations))  # Remove duplicates


# Global instance for reuse
_query_rewriter = None

def get_query_rewriter() -> QueryRewritingAgent:
    """Get or create the global query rewriter instance"""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewritingAgent()
    return _query_rewriter
