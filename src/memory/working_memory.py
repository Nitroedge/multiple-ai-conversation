"""
Working Memory Manager - Redis-based short-term memory with TTL
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import redis.asyncio as redis

from .models import MemoryItem, ConversationState

logger = logging.getLogger(__name__)


class WorkingMemoryManager:
    """
    Manages short-term memory using Redis with TTL for active conversations
    """

    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl

        # Key prefixes for different data types
        self.MEMORY_PREFIX = "memory:session:"
        self.STATE_PREFIX = "state:session:"
        self.ACTIVE_SESSIONS_KEY = "active_sessions"
        self.SESSION_STATS_PREFIX = "stats:session:"

    async def initialize(self):
        """Initialize the working memory manager"""
        try:
            await self.redis.ping()
            logger.info("Working Memory Manager connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def shutdown(self):
        """Shutdown the working memory manager"""
        if self.redis:
            await self.redis.close()

    async def store_memory(self, memory_item: MemoryItem) -> None:
        """Store a memory item in working memory with TTL"""
        try:
            memory_key = f"{self.MEMORY_PREFIX}{memory_item.session_id}"
            memory_data = memory_item.to_dict()

            # Add memory to session's memory list
            await self.redis.lpush(memory_key, json.dumps(memory_data))
            await self.redis.expire(memory_key, self.default_ttl)

            # Add session to active sessions set
            await self.redis.sadd(self.ACTIVE_SESSIONS_KEY, memory_item.session_id)

            # Update session statistics
            await self._update_session_stats(memory_item.session_id, memory_item)

            logger.debug(f"Stored memory {memory_item.memory_id} in working memory")

        except Exception as e:
            logger.error(f"Error storing memory in working memory: {e}")
            raise

    async def get_recent_memories(
        self,
        session_id: str,
        limit: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[MemoryItem]:
        """Retrieve recent memories for a session"""
        try:
            memory_key = f"{self.MEMORY_PREFIX}{session_id}"

            # Get recent memories from Redis list
            memory_data_list = await self.redis.lrange(memory_key, 0, limit - 1)

            memories = []
            for memory_data in memory_data_list:
                try:
                    memory_dict = json.loads(memory_data)
                    memory_item = MemoryItem.from_dict(memory_dict)

                    # Filter by memory type if specified
                    if memory_types is None or memory_item.memory_type.value in memory_types:
                        memories.append(memory_item)

                except Exception as e:
                    logger.warning(f"Error parsing memory data: {e}")
                    continue

            logger.debug(f"Retrieved {len(memories)} recent memories for session {session_id}")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving recent memories: {e}")
            return []

    async def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """Get current conversation state from Redis"""
        try:
            state_key = f"{self.STATE_PREFIX}{session_id}"
            state_data = await self.redis.get(state_key)

            if state_data:
                state_dict = json.loads(state_data)
                return ConversationState.from_dict(state_dict)

            return None

        except Exception as e:
            logger.error(f"Error retrieving conversation state: {e}")
            return None

    async def update_conversation_state(self, state: ConversationState) -> None:
        """Update conversation state in Redis"""
        try:
            state_key = f"{self.STATE_PREFIX}{state.session_id}"
            state_data = json.dumps(state.to_dict())

            await self.redis.setex(state_key, self.default_ttl, state_data)

            # Add session to active sessions
            await self.redis.sadd(self.ACTIVE_SESSIONS_KEY, state.session_id)

            logger.debug(f"Updated conversation state for session {state.session_id}")

        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")
            raise

    async def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        try:
            session_ids = await self.redis.smembers(self.ACTIVE_SESSIONS_KEY)
            return [session_id.decode() if isinstance(session_id, bytes) else session_id for session_id in session_ids]

        except Exception as e:
            logger.error(f"Error retrieving active sessions: {e}")
            return []

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get working memory summary for a session"""
        try:
            memory_key = f"{self.MEMORY_PREFIX}{session_id}"
            stats_key = f"{self.SESSION_STATS_PREFIX}{session_id}"

            # Get memory count and recent memories
            memory_count = await self.redis.llen(memory_key)
            recent_memories = await self.get_recent_memories(session_id, limit=5)

            # Get session statistics
            stats_data = await self.redis.get(stats_key)
            stats = json.loads(stats_data) if stats_data else {}

            # Get TTL for session data
            ttl = await self.redis.ttl(memory_key)

            summary = {
                'session_id': session_id,
                'memory_count': memory_count,
                'recent_memories_count': len(recent_memories),
                'ttl_seconds': ttl,
                'statistics': stats,
                'last_activity': stats.get('last_activity'),
                'total_importance_score': stats.get('total_importance_score', 0.0)
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {'session_id': session_id, 'error': str(e)}

    async def get_memories_for_consolidation(
        self,
        session_id: str,
        importance_threshold: float = 0.7
    ) -> List[MemoryItem]:
        """Get memories that should be consolidated to long-term storage"""
        try:
            all_memories = await self.get_recent_memories(session_id, limit=100)

            # Filter memories that meet consolidation criteria
            consolidation_candidates = [
                memory for memory in all_memories
                if memory.importance_score >= importance_threshold
            ]

            logger.debug(f"Found {len(consolidation_candidates)} memories for consolidation in session {session_id}")
            return consolidation_candidates

        except Exception as e:
            logger.error(f"Error getting memories for consolidation: {e}")
            return []

    async def clear_session(self, session_id: str) -> None:
        """Clear all working memory data for a session"""
        try:
            memory_key = f"{self.MEMORY_PREFIX}{session_id}"
            state_key = f"{self.STATE_PREFIX}{session_id}"
            stats_key = f"{self.SESSION_STATS_PREFIX}{session_id}"

            # Delete session data
            await self.redis.delete(memory_key, state_key, stats_key)

            # Remove from active sessions
            await self.redis.srem(self.ACTIVE_SESSIONS_KEY, session_id)

            logger.info(f"Cleared working memory for session {session_id}")

        except Exception as e:
            logger.error(f"Error clearing session working memory: {e}")
            raise

    async def extend_session_ttl(self, session_id: str, additional_seconds: int = None) -> None:
        """Extend TTL for session data"""
        try:
            ttl_seconds = additional_seconds or self.default_ttl

            memory_key = f"{self.MEMORY_PREFIX}{session_id}"
            state_key = f"{self.STATE_PREFIX}{session_id}"
            stats_key = f"{self.SESSION_STATS_PREFIX}{session_id}"

            # Extend TTL for all session keys
            await asyncio.gather(
                self.redis.expire(memory_key, ttl_seconds),
                self.redis.expire(state_key, ttl_seconds),
                self.redis.expire(stats_key, ttl_seconds)
            )

            logger.debug(f"Extended TTL for session {session_id} by {ttl_seconds} seconds")

        except Exception as e:
            logger.error(f"Error extending session TTL: {e}")

    async def get_memory_by_id(self, session_id: str, memory_id: str) -> Optional[MemoryItem]:
        """Get a specific memory by ID from working memory"""
        try:
            memories = await self.get_recent_memories(session_id, limit=100)

            for memory in memories:
                if memory.memory_id == memory_id:
                    return memory

            return None

        except Exception as e:
            logger.error(f"Error retrieving memory by ID: {e}")
            return None

    async def _update_session_stats(self, session_id: str, memory_item: MemoryItem) -> None:
        """Update session statistics"""
        try:
            stats_key = f"{self.SESSION_STATS_PREFIX}{session_id}"

            # Get current stats
            stats_data = await self.redis.get(stats_key)
            stats = json.loads(stats_data) if stats_data else {
                'memory_count': 0,
                'total_importance_score': 0.0,
                'last_activity': None,
                'agent_contributions': {}
            }

            # Update statistics
            stats['memory_count'] += 1
            stats['total_importance_score'] += memory_item.importance_score
            stats['last_activity'] = memory_item.timestamp.isoformat()

            # Track agent contributions
            if memory_item.agent_id:
                agent_stats = stats['agent_contributions'].get(memory_item.agent_id, {
                    'count': 0,
                    'total_importance': 0.0
                })
                agent_stats['count'] += 1
                agent_stats['total_importance'] += memory_item.importance_score
                stats['agent_contributions'][memory_item.agent_id] = agent_stats

            # Store updated stats
            await self.redis.setex(stats_key, self.default_ttl, json.dumps(stats))

        except Exception as e:
            logger.warning(f"Error updating session stats: {e}")

    async def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get overall working memory usage statistics"""
        try:
            active_sessions = await self.get_active_sessions()

            total_memories = 0
            total_sessions = len(active_sessions)

            for session_id in active_sessions:
                memory_key = f"{self.MEMORY_PREFIX}{session_id}"
                session_memory_count = await self.redis.llen(memory_key)
                total_memories += session_memory_count

            # Get Redis memory info
            redis_info = await self.redis.info('memory')

            return {
                'active_sessions': total_sessions,
                'total_memories': total_memories,
                'avg_memories_per_session': total_memories / max(1, total_sessions),
                'redis_memory_used': redis_info.get('used_memory_human', 'N/A'),
                'redis_memory_peak': redis_info.get('used_memory_peak_human', 'N/A')
            }

        except Exception as e:
            logger.error(f"Error getting memory usage stats: {e}")
            return {'error': str(e)}