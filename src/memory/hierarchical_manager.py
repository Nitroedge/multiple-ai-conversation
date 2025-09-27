"""
Hierarchical Memory Manager - Orchestrates the entire memory system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .models import MemoryItem, MemoryType, ConversationState, MemoryQuery, MemoryRetrievalResult
from .working_memory import WorkingMemoryManager
from .long_term_memory import LongTermMemoryManager
from .vector_retrieval import VectorMemoryRetrieval
from .consolidation import MemoryConsolidationEngine

logger = logging.getLogger(__name__)


class HierarchicalMemoryManager:
    """
    Orchestrates the hierarchical memory system with working memory, long-term memory,
    vector-based retrieval, and memory consolidation.
    """

    def __init__(
        self,
        redis_client,
        mongodb_client,
        consolidation_threshold: float = 0.7,
        working_memory_ttl: int = 3600  # 1 hour
    ):
        self.consolidation_threshold = consolidation_threshold
        self.working_memory_ttl = working_memory_ttl

        # Initialize memory subsystems
        self.working_memory = WorkingMemoryManager(redis_client, working_memory_ttl)
        self.long_term_memory = LongTermMemoryManager(mongodb_client)
        self.vector_retrieval = VectorMemoryRetrieval(mongodb_client)
        self.consolidation_engine = MemoryConsolidationEngine(
            threshold=consolidation_threshold
        )

        # Background task for memory consolidation
        self._consolidation_task = None
        self._running = False

    async def initialize(self):
        """Initialize the memory manager and start background tasks"""
        await self.working_memory.initialize()
        await self.long_term_memory.initialize()
        await self.vector_retrieval.initialize()

        # Start background consolidation task
        self._running = True
        self._consolidation_task = asyncio.create_task(self._consolidation_worker())

        logger.info("Hierarchical Memory Manager initialized successfully")

    async def shutdown(self):
        """Gracefully shutdown the memory manager"""
        self._running = False
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

        await self.working_memory.shutdown()
        await self.long_term_memory.shutdown()

        logger.info("Hierarchical Memory Manager shutdown complete")

    async def store_memory(self, memory_item: MemoryItem) -> str:
        """
        Store a memory item in the appropriate storage layer
        """
        try:
            # Always store in working memory first for immediate access
            await self.working_memory.store_memory(memory_item)

            # Check if memory should be consolidated to long-term storage
            if memory_item.importance_score >= self.consolidation_threshold:
                await self._consolidate_to_long_term(memory_item)

            logger.debug(f"Stored memory item {memory_item.memory_id} (importance: {memory_item.importance_score})")
            return memory_item.memory_id

        except Exception as e:
            logger.error(f"Error storing memory item: {e}")
            raise

    async def retrieve_memories(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """
        Retrieve memories using multi-layered approach:
        1. Check working memory first
        2. Search long-term memory with vector similarity
        3. Combine and rank results
        """
        start_time = datetime.now()

        try:
            # Get working memory results
            working_memories = await self.working_memory.get_recent_memories(
                query.session_id,
                limit=query.max_results // 2
            )

            # Get long-term memory results using vector search
            vector_results = await self.vector_retrieval.search_similar_memories(
                query.query_text,
                session_id=query.session_id,
                limit=query.max_results,
                memory_types=query.memory_types,
                importance_threshold=query.importance_threshold
            )

            # Combine and deduplicate results
            all_memories = working_memories + vector_results.memories
            unique_memories = self._deduplicate_memories(all_memories)

            # Rank by relevance and recency
            ranked_memories = await self._rank_memories_by_relevance(
                unique_memories,
                query.query_text
            )

            # Limit results
            final_memories = ranked_memories[:query.max_results]

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.debug(f"Retrieved {len(final_memories)} memories for query in {query_time:.2f}ms")

            return MemoryRetrievalResult(
                memories=final_memories,
                total_found=len(unique_memories),
                query_time_ms=query_time,
                relevance_scores=None  # TODO: Add relevance scoring
            )

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            raise

    async def get_conversation_state(self, session_id: str) -> Optional[ConversationState]:
        """Get current conversation state"""
        return await self.working_memory.get_conversation_state(session_id)

    async def update_conversation_state(self, state: ConversationState) -> None:
        """Update conversation state"""
        await self.working_memory.update_conversation_state(state)

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary from all memory layers"""
        try:
            # Get working memory summary
            working_summary = await self.working_memory.get_session_summary(session_id)

            # Get long-term memory statistics
            long_term_stats = await self.long_term_memory.get_session_statistics(session_id)

            # Get conversation state
            conversation_state = await self.get_conversation_state(session_id)

            summary = {
                'session_id': session_id,
                'working_memory': working_summary,
                'long_term_memory': long_term_stats,
                'conversation_state': conversation_state.to_dict() if conversation_state else None,
                'generated_at': datetime.now().isoformat()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            raise

    async def cleanup_session(self, session_id: str, archive: bool = True) -> None:
        """
        Clean up session data from working memory and optionally archive to long-term
        """
        try:
            if archive:
                # Get all working memories for the session
                working_memories = await self.working_memory.get_recent_memories(session_id, limit=1000)

                # Consolidate and archive important memories
                for memory in working_memories:
                    if memory.importance_score >= self.consolidation_threshold:
                        await self._consolidate_to_long_term(memory)

            # Clear working memory
            await self.working_memory.clear_session(session_id)

            logger.info(f"Session {session_id} cleaned up (archived: {archive})")

        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
            raise

    async def _consolidate_to_long_term(self, memory_item: MemoryItem) -> None:
        """Consolidate memory item to long-term storage"""
        try:
            # Generate embedding if not present
            if not memory_item.embedding:
                memory_item.embedding = await self.vector_retrieval.generate_embedding(
                    memory_item.content
                )

            # Store in long-term memory
            await self.long_term_memory.store_memory(memory_item)

            logger.debug(f"Consolidated memory {memory_item.memory_id} to long-term storage")

        except Exception as e:
            logger.error(f"Error consolidating memory to long-term: {e}")
            raise

    async def _consolidation_worker(self):
        """Background worker for periodic memory consolidation"""
        consolidation_interval = 300  # 5 minutes

        while self._running:
            try:
                await asyncio.sleep(consolidation_interval)

                if not self._running:
                    break

                await self._run_consolidation_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consolidation worker: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _run_consolidation_cycle(self):
        """Run a single consolidation cycle"""
        try:
            # Get active sessions from working memory
            active_sessions = await self.working_memory.get_active_sessions()

            for session_id in active_sessions:
                # Get memories that need consolidation
                memories_to_consolidate = await self.working_memory.get_memories_for_consolidation(
                    session_id,
                    importance_threshold=self.consolidation_threshold
                )

                if memories_to_consolidate:
                    # Run consolidation algorithm
                    consolidation_result = await self.consolidation_engine.consolidate_memories(
                        memories_to_consolidate
                    )

                    # Store consolidated memories in long-term storage
                    for consolidated_memory in consolidation_result.consolidated_memories:
                        await self._consolidate_to_long_term(consolidated_memory)

                    logger.info(f"Consolidated {len(consolidation_result.consolidated_memories)} memories for session {session_id}")

        except Exception as e:
            logger.error(f"Error in consolidation cycle: {e}")

    def _deduplicate_memories(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """Remove duplicate memories based on content similarity"""
        if not memories:
            return []

        unique_memories = []
        seen_content = set()

        for memory in memories:
            # Simple deduplication based on content hash
            content_hash = hash(memory.content.lower().strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_memories.append(memory)

        return unique_memories

    async def _rank_memories_by_relevance(
        self,
        memories: List[MemoryItem],
        query_text: str
    ) -> List[MemoryItem]:
        """
        Rank memories by relevance to query, considering:
        - Vector similarity (if embeddings available)
        - Recency
        - Importance score
        """
        if not memories:
            return []

        # For now, simple ranking by importance and recency
        # TODO: Implement proper vector similarity ranking

        def ranking_score(memory: MemoryItem) -> float:
            # Combine importance and recency
            time_weight = max(0.1, 1.0 - (datetime.now() - memory.timestamp).total_seconds() / 86400)
            return memory.importance_score * 0.7 + time_weight * 0.3

        return sorted(memories, key=ranking_score, reverse=True)