"""
Memory System - Stores and retrieves conversation context for improved follow-up queries
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConversationMemory:
    """Simple in-memory conversation storage with persistence"""
    
    def __init__(self, max_history: int = 10, ttl_hours: int = 24):
        self.max_history = max_history
        self.ttl = timedelta(hours=ttl_hours)
        self.conversations = defaultdict(list)  # conversation_id -> list of interactions
        self.metadata = {}  # conversation_id -> metadata
        self.cache_dir = "data/memory"
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_persistent_memory()
    
    def add_interaction(self, conversation_id: str, query: str, answer: str, 
                       context: Optional[Dict] = None) -> None:
        """Add an interaction to conversation memory"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:500],  # Truncate for storage
            "context": context or {}
        }
        
        self.conversations[conversation_id].append(interaction)
        
        # Keep only recent history
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
        
        self.metadata[conversation_id] = {
            "last_updated": datetime.now().isoformat(),
            "interaction_count": len(self.conversations[conversation_id])
        }
        
        logger.info(f"Added interaction to memory for conversation {conversation_id}")
    
    def get_context(self, conversation_id: str, current_query: str) -> Dict[str, Any]:
        """Get relevant context from conversation history"""
        history = self.conversations.get(conversation_id, [])
        
        if not history:
            return {"has_context": False, "boost_keywords": [], "previous_topics": []}
        
        # Extract keywords from history
        all_queries = [h["query"].lower() for h in history]
        all_keywords = set()
        for q in all_queries:
            words = q.split()
            all_keywords.update([w for w in words if len(w) > 3])
        
        # Find relevant previous topics
        current_keywords = set(current_query.lower().split())
        relevant_history = []
        
        for interaction in history[-3:]:  # Check last 3 interactions
            past_keywords = set(interaction["query"].lower().split())
            overlap = len(current_keywords.intersection(past_keywords))
            if overlap > 0:
                relevant_history.append(interaction)
        
        # Boost keywords that appeared in previous related queries
        boost_keywords = list(all_keywords.intersection(current_keywords))
        
        return {
            "has_context": len(history) > 0,
            "interaction_count": len(history),
            "boost_keywords": boost_keywords[:10],  # Top 10 relevant keywords
            "previous_topics": self._extract_topics(history),
            "relevant_history": relevant_history[:2],  # Last 2 relevant interactions
            "conversation_age_hours": self._get_conversation_age(conversation_id)
        }
    
    def _extract_topics(self, history: List[Dict]) -> List[str]:
        """Extract main topics from conversation history"""
        topics = []
        for interaction in history:
            query = interaction["query"].lower()
            # Simple topic extraction based on common document-related terms
            if "pdf" in query or "document" in query:
                topics.append("document_analysis")
            if "summary" in query or "about" in query:
                topics.append("summarization")
            if "feature" in query or "what does" in query:
                topics.append("feature_extraction")
        return list(set(topics))
    
    def _get_conversation_age(self, conversation_id: str) -> float:
        """Get conversation age in hours"""
        meta = self.metadata.get(conversation_id)
        if meta and "last_updated" in meta:
            last_updated = datetime.fromisoformat(meta["last_updated"])
            age = datetime.now() - last_updated
            return age.total_seconds() / 3600
        return 0.0
    
    def _load_persistent_memory(self) -> None:
        """Load memory from disk if available"""
        try:
            memory_file = os.path.join(self.cache_dir, "conversations.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    for conv_id, interactions in data.get("conversations", {}).items():
                        self.conversations[conv_id] = interactions
                    self.metadata = data.get("metadata", {})
                logger.info(f"Loaded {len(self.conversations)} conversations from memory")
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
    
    def save_memory(self) -> None:
        """Persist memory to disk"""
        try:
            memory_file = os.path.join(self.cache_dir, "conversations.json")
            data = {
                "conversations": dict(self.conversations),
                "metadata": self.metadata,
                "saved_at": datetime.now().isoformat()
            }
            with open(memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Memory saved to disk")
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
    
    def clear_old_conversations(self, max_age_hours: int = 48) -> int:
        """Clear conversations older than specified hours"""
        cleared = 0
        current_time = datetime.now()
        
        for conv_id in list(self.conversations.keys()):
            meta = self.metadata.get(conv_id, {})
            if "last_updated" in meta:
                last_updated = datetime.fromisoformat(meta["last_updated"])
                age = current_time - last_updated
                if age > timedelta(hours=max_age_hours):
                    del self.conversations[conv_id]
                    del self.metadata[conv_id]
                    cleared += 1
        
        if cleared > 0:
            logger.info(f"Cleared {cleared} old conversations from memory")
        
        return cleared


# Global memory instance
memory_store = ConversationMemory()
