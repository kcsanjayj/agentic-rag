"""
PRO Conversation Memory System
Enables context-aware multi-turn conversations.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Message:
    """Single conversation message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """
    PRO Conversation Memory with context window management.
    Stores conversation history and provides context for agentic responses.
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Message] = []
        self.session_start = datetime.now()
        
        logger.info(f"Memory initialized (max_history={max_history})")
    
    def add_user_message(self, content: str, metadata: Dict = None):
        """Add user message to history"""
        msg = Message(role="user", content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._trim_history()
        logger.debug(f"Added user message ({len(content)} chars)")
    
    def add_assistant_message(self, content: str, metadata: Dict = None):
        """Add assistant message to history"""
        msg = Message(role="assistant", content=content, metadata=metadata or {})
        self.messages.append(msg)
        self._trim_history()
        logger.debug(f"Added assistant message ({len(content)} chars)")
    
    def add(self, user: str, assistant: str, metadata: Dict = None):
        """Convenience: Add both user and assistant message"""
        self.add_user_message(user)
        self.add_assistant_message(assistant, metadata)
    
    def _trim_history(self):
        """Keep only recent messages"""
        if len(self.messages) > self.max_history:
            # Keep system messages and last N messages
            self.messages = self.messages[-self.max_history:]
    
    def get_context(self, last_n: int = 5) -> str:
        """
        Get conversation context as formatted string.
        
        Args:
            last_n: Number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        if not self.messages:
            return ""
        
        recent = self.messages[-last_n:]
        lines = []
        
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")
        
        return "\n\n".join(lines)
    
    def get_context_for_prompt(self, last_n: int = 5) -> str:
        """Get context formatted for LLM prompt injection"""
        context = self.get_context(last_n)
        if not context:
            return ""
        
        return f"""Previous conversation:
{context}

---"""
    
    def get_last_query(self) -> Optional[str]:
        """Get the last user query"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_last_response(self) -> Optional[str]:
        """Get the last assistant response"""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None
    
    def clear(self):
        """Clear all conversation history"""
        self.messages = []
        logger.info("Memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in self.messages if m.role == "assistant"),
            "session_start": self.session_start.isoformat(),
            "max_history": self.max_history
        }
    
    def to_dict(self) -> List[Dict]:
        """Serialize to dict for storage"""
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "metadata": m.metadata
            }
            for m in self.messages
        ]
    
    @classmethod
    def from_dict(cls, data: List[Dict]) -> "ConversationMemory":
        """Deserialize from dict"""
        memory = cls()
        for item in data:
            msg = Message(
                role=item["role"],
                content=item["content"],
                timestamp=item.get("timestamp", time.time()),
                metadata=item.get("metadata", {})
            )
            memory.messages.append(msg)
        return memory


class SessionMemoryManager:
    """
    Manages multiple conversation memories (one per user/session).
    Simple in-memory storage (can be extended to Redis/DB).
    """
    
    def __init__(self):
        self.memories: Dict[str, ConversationMemory] = {}
        logger.info("Session memory manager initialized")
    
    def get_or_create(self, session_id: str) -> ConversationMemory:
        """Get existing memory or create new one"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationMemory()
            logger.info(f"Created new memory for session {session_id[:8]}...")
        return self.memories[session_id]
    
    def get(self, session_id: str) -> Optional[ConversationMemory]:
        """Get memory if exists"""
        return self.memories.get(session_id)
    
    def delete(self, session_id: str):
        """Delete a session's memory"""
        if session_id in self.memories:
            del self.memories[session_id]
            logger.info(f"Deleted memory for session {session_id[:8]}...")
    
    def clear_all(self):
        """Clear all memories"""
        self.memories = {}
        logger.info("All memories cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all sessions"""
        return {
            "total_sessions": len(self.memories),
            "sessions": {sid[:8]: mem.get_stats() for sid, mem in self.memories.items()}
        }


# Global session manager (singleton)
_session_manager: Optional[SessionMemoryManager] = None


def get_session_manager() -> SessionMemoryManager:
    """Get or create global session manager"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionMemoryManager()
    return _session_manager


def get_memory(session_id: str) -> ConversationMemory:
    """Convenience: Get memory for a session"""
    return get_session_manager().get_or_create(session_id)


__all__ = [
    "Message",
    "ConversationMemory",
    "SessionMemoryManager",
    "get_session_manager",
    "get_memory",
]
