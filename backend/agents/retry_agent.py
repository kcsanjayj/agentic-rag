"""
Retry Agent - Handles retries and fallback strategies
"""

from typing import Dict, Any, List, Optional
import time
from backend.utils.logger import setup_logger
from backend.agents.reasoning_agent import ReasoningAgent

logger = setup_logger(__name__)


class RetryAgent:
    """Handles intelligent retries with different strategies"""
    
    def __init__(self):
        self.reasoning_agent = ReasoningAgent()
        self.retry_strategies = [
            "broader_query",
            "full_document",
            "different_prompt",
            "minimal_summary"
        ]
        self.max_retries = 2
    
    async def retry_if_needed(
        self, 
        is_valid: bool, 
        original_query: str,
        context: str, 
        task: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        full_text: Optional[str] = None,
        force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """
        Retry with different strategies if response is invalid
        
        Args:
            is_valid: Whether the original response was valid
            original_query: Original user query
            context: Original context used
            task: Task information
            retrieved_docs: Originally retrieved documents
            full_text: Full document text for fallback
            force_regenerate: Force regeneration from self-critic
            
        Returns:
            Retry results or None if no retry needed
        """
        if is_valid and not force_regenerate:
            return None
        
        logger.info("🔄 Starting retry process...")
        
        # 🔥 RETRY LOGIC: Check for weak context
        if self._is_weak_context(context, retrieved_docs):
            logger.warning("🔥 WEAK CONTEXT DETECTED - Using full document summary")
            original_query = "full document summary"
            if full_text:
                context = full_text[:4000]
        
        for attempt in range(self.max_retries):
            strategy = self.retry_strategies[attempt]
            logger.info(f"🔄 Retry attempt {attempt + 1} with strategy: {strategy}")
            
            try:
                if strategy == "broader_query":
                    # Try with a broader query
                    new_context = await self._retry_with_broader_query(retrieved_docs)
                elif strategy == "full_document" and full_text:
                    # Use full document text
                    new_context = full_text[:4000]
                elif strategy == "different_prompt":
                    # Try with a different prompt approach
                    new_context = context
                    task = self._modify_task_for_retry(task)
                else:  # minimal_summary
                    # Last resort - minimal prompt
                    new_context = context[:2000]
                    task = {"task_type": "general_summary", "strategy": "minimal"}
                
                # Generate new response
                response = await self.reasoning_agent.reason(new_context, task)
                
                # Add retry metadata
                response["agent_trace"]["retry_attempt"] = attempt + 1
                response["agent_trace"]["retry_strategy"] = strategy
                response["agent_trace"]["original_query"] = original_query
                response["agent_trace"]["weak_context_detected"] = self._is_weak_context(context, retrieved_docs)
                
                logger.info(f"🔄 Retry {attempt + 1} completed")
                return response
                
            except Exception as e:
                logger.error(f"🔄 Retry {attempt + 1} failed: {str(e)}")
                continue
        
        # All retries failed
        logger.error("🔄 All retry attempts failed")
        return {
            "answer": "Unable to generate a satisfactory response after multiple attempts.",
            "task_type": task.get("task_type", "unknown"),
            "confidence": 0.1,
            "error": "All retries failed",
            "agent_trace": {
                "retry_attempts": self.max_retries,
                "status": "failed",
                "weak_context_detected": self._is_weak_context(context, retrieved_docs)
            }
        }
    
    def _is_weak_context(self, context: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
        """Check if context is weak and needs full document"""
        # Weak context indicators
        if len(context) < 500:
            return True
        
        if len(retrieved_docs) < 3:
            return True
        
        # Check for generic content
        generic_indicators = ["document content available", "content available", "data extracted"]
        for indicator in generic_indicators:
            if indicator in context.lower():
                return True
        
        return False
    
    async def _retry_with_broader_query(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Retry by using more documents"""
        # Take more documents if available
        more_docs = retrieved_docs[:10] if len(retrieved_docs) > 6 else retrieved_docs
        context = "\n\n".join([doc.get('content', '') for doc in more_docs])
        return context
    
    def _modify_task_for_retry(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Modify task for retry attempt"""
        modified_task = task.copy()
        modified_task["strategy"] = "retry_attempt"
        modified_task["confidence"] = max(0.3, task.get("confidence", 0.5) - 0.2)
        return modified_task
