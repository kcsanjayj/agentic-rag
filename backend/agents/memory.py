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


# =============================================================================
# PRO: REDIS MEMORY BACKEND (Production Scaling)
# =============================================================================

class RedisMemoryBackend:
    """
    PRO: Redis-backed memory for production.
    Survives restarts, supports multi-instance scaling.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            import redis as redis_lib
            self.client = redis_lib.from_url(self.redis_url, decode_responses=True)
            self.client.ping()
            logger.info(f"Redis connected: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to in-memory.")
            self.client = None
    
    def _key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"agentic_rag:memory:{session_id}"
    
    def save(self, session_id: str, memory: ConversationMemory, ttl: int = 86400):
        """Save memory to Redis with TTL (default: 24 hours)"""
        if not self.client:
            return False
        
        try:
            data = json.dumps(memory.to_dict())
            self.client.setex(self._key(session_id), ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis save failed: {e}")
            return False
    
    def load(self, session_id: str) -> Optional[ConversationMemory]:
        """Load memory from Redis"""
        if not self.client:
            return None
        
        try:
            data = self.client.get(self._key(session_id))
            if data:
                return ConversationMemory.from_dict(json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Redis load failed: {e}")
            return None
    
    def delete(self, session_id: str) -> bool:
        """Delete memory from Redis"""
        if not self.client:
            return False
        
        try:
            return self.client.delete(self._key(session_id)) > 0
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False


# Global Redis backend (lazy initialization)
_redis_backend: Optional[RedisMemoryBackend] = None


def get_redis_backend() -> Optional[RedisMemoryBackend]:
    """Get Redis backend (if available)"""
    global _redis_backend
    if _redis_backend is None:
        _redis_backend = RedisMemoryBackend()
    return _redis_backend if _redis_backend.is_connected() else None


__all__ = [
    "Message",
    "ConversationMemory",
    "SessionMemoryManager",
    "RedisMemoryBackend",
    "get_session_manager",
    "get_memory",
    "get_redis_backend",
]
