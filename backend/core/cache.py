"""
Caching System - Performance optimization with TTL-based cache
"""

from typing import Dict, Any, Optional, Callable
import hashlib
import json
import time
from functools import wraps
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class Cache:
    """Simple TTL-based cache for RAG operations"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl  # seconds
        self.stats = {"hits": 0, "misses": 0, "total_requests": 0}
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            key_data = data
        elif isinstance(data, (dict, list)):
            key_data = json.dumps(data, sort_keys=True)
        else:
            key_data = str(data)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        self.stats["total_requests"] += 1
        
        if key in self.cache:
            entry = self.cache[key]
            if entry["expires_at"] > time.time():
                self.stats["hits"] += 1
                logger.debug(f"Cache HIT: {key[:8]}...")
                return entry["value"]
            else:
                # Expired
                del self.cache[key]
        
        self.stats["misses"] += 1
        logger.debug(f"Cache MISS: {key[:8]}...")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with TTL"""
        expires_at = time.time() + (ttl or self.default_ttl)
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        logger.debug(f"Cache SET: {key[:8]}...")
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats["total_requests"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hit_rate": round(hit_rate, 2),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": total
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry["expires_at"] <= current_time
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)


# Global cache instances
query_cache = Cache(ttl=1800)  # 30 minutes for queries
embedding_cache = Cache(ttl=3600)  # 1 hour for embeddings
retrieval_cache = Cache(ttl=900)  # 15 minutes for retrieval


def cached(cache_instance: Cache, key_func: Optional[Callable] = None):
    """Decorator to cache function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default: hash args and kwargs
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
                key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            cached_value = cache_instance.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute and cache
            result = await func(*args, **kwargs)
            cache_instance.set(key, result)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
                key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            cached_value = cache_instance.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache_instance.set(key, result)
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
