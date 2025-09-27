"""
Memory Consolidation Engine - Algorithms for memory importance and merging
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .models import MemoryItem, MemoryType, ConsolidationResult

logger = logging.getLogger(__name__)


class MemoryConsolidationEngine:
    """
    Manages memory consolidation using various algorithms to merge similar memories
    and calculate importance scores based on multiple factors
    """

    def __init__(
        self,
        threshold: float = 0.7,
        similarity_threshold: float = 0.85,
        time_decay_factor: float = 0.1
    ):
        self.importance_threshold = threshold
        self.similarity_threshold = similarity_threshold
        self.time_decay_factor = time_decay_factor

        # Consolidation strategies
        self.strategies = {
            "clustering": self._clustering_consolidation,
            "similarity_merge": self._similarity_merge_consolidation,
            "importance_filter": self._importance_filter_consolidation,
            "temporal_grouping": self._temporal_grouping_consolidation
        }

    async def consolidate_memories(
        self,
        memories: List[MemoryItem],
        strategy: str = "clustering"
    ) -> ConsolidationResult:
        """
        Consolidate a list of memories using the specified strategy
        """
        start_time = datetime.now()

        try:
            if not memories:
                return ConsolidationResult(
                    consolidated_memories=[],
                    original_count=0,
                    consolidated_count=0,
                    processing_time_ms=0,
                    consolidation_strategy=strategy
                )

            # Update importance scores before consolidation
            await self._update_importance_scores(memories)

            # Apply consolidation strategy
            if strategy in self.strategies:
                consolidated_memories = await self.strategies[strategy](memories)
            else:
                logger.warning(f"Unknown consolidation strategy: {strategy}, using clustering")
                consolidated_memories = await self._clustering_consolidation(memories)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(f"Consolidated {len(memories)} memories to {len(consolidated_memories)} using {strategy}")

            return ConsolidationResult(
                consolidated_memories=consolidated_memories,
                original_count=len(memories),
                consolidated_count=len(consolidated_memories),
                processing_time_ms=processing_time,
                consolidation_strategy=strategy
            )

        except Exception as e:
            logger.error(f"Error in memory consolidation: {e}")
            raise

    async def _update_importance_scores(self, memories: List[MemoryItem]) -> None:
        """Update importance scores based on multiple factors"""
        try:
            for memory in memories:
                # Calculate importance based on multiple factors
                importance_score = await self._calculate_importance_score(memory)
                memory.importance_score = min(1.0, max(0.0, importance_score))

        except Exception as e:
            logger.error(f"Error updating importance scores: {e}")

    async def _calculate_importance_score(self, memory: MemoryItem) -> float:
        """
        Calculate importance score based on:
        - Emotional intensity
        - Recency (with decay)
        - Content complexity
        - Agent involvement
        - Context tags
        """
        try:
            score_components = []

            # Emotional intensity factor (0.0 to 1.0)
            if memory.emotions:
                emotion_intensity = sum(abs(value) for value in memory.emotions.values())
                emotion_factor = min(1.0, emotion_intensity / len(memory.emotions))
                score_components.append(('emotion', emotion_factor, 0.3))

            # Recency factor with time decay
            time_since = datetime.utcnow() - memory.timestamp
            hours_since = time_since.total_seconds() / 3600
            recency_factor = max(0.1, np.exp(-self.time_decay_factor * hours_since))
            score_components.append(('recency', recency_factor, 0.2))

            # Content complexity (based on length and structure)
            content_complexity = self._calculate_content_complexity(memory.content)
            score_components.append(('complexity', content_complexity, 0.2))

            # Agent involvement factor
            agent_factor = 1.0 if memory.agent_id else 0.5  # Human messages less important by default
            score_components.append(('agent', agent_factor, 0.1))

            # Context tags factor
            context_factor = min(1.0, len(memory.context_tags or []) / 5.0)  # More tags = more context
            score_components.append(('context', context_factor, 0.1))

            # Memory type factor
            type_factor = self._get_memory_type_importance(memory.memory_type)
            score_components.append(('type', type_factor, 0.1))

            # Calculate weighted average
            weighted_score = sum(score * weight for _, score, weight in score_components)
            total_weight = sum(weight for _, _, weight in score_components)

            final_score = weighted_score / total_weight if total_weight > 0 else 0.5

            logger.debug(f"Calculated importance {final_score:.3f} for memory {memory.memory_id}")
            return final_score

        except Exception as e:
            logger.warning(f"Error calculating importance score: {e}")
            return memory.importance_score  # Return original score if calculation fails

    def _calculate_content_complexity(self, content: str) -> float:
        """Calculate content complexity based on various metrics"""
        try:
            # Basic complexity metrics
            word_count = len(content.split())
            char_count = len(content)
            sentence_count = content.count('.') + content.count('!') + content.count('?')

            # Vocabulary richness (unique words / total words)
            words = content.lower().split()
            unique_words = len(set(words))
            vocab_richness = unique_words / max(1, len(words))

            # Punctuation diversity
            punctuation_count = sum(1 for char in content if not char.isalnum() and not char.isspace())
            punctuation_ratio = punctuation_count / max(1, char_count)

            # Combine metrics
            length_factor = min(1.0, word_count / 100.0)  # Normalize to 100 words
            structure_factor = min(1.0, sentence_count / 5.0)  # Normalize to 5 sentences
            richness_factor = min(1.0, vocab_richness * 2.0)  # Scale vocab richness
            punctuation_factor = min(1.0, punctuation_ratio * 10.0)  # Scale punctuation

            complexity = (length_factor + structure_factor + richness_factor + punctuation_factor) / 4.0
            return complexity

        except Exception as e:
            logger.warning(f"Error calculating content complexity: {e}")
            return 0.5

    def _get_memory_type_importance(self, memory_type: MemoryType) -> float:
        """Get importance factor based on memory type"""
        type_importance = {
            MemoryType.SHARED_EPISODIC: 1.0,  # Highest importance
            MemoryType.EPISODIC: 0.8,
            MemoryType.SEMANTIC: 0.7,
            MemoryType.PROCEDURAL: 0.6
        }
        return type_importance.get(memory_type, 0.5)

    async def _clustering_consolidation(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """Consolidate memories using clustering based on embeddings"""
        try:
            # Filter memories with embeddings
            memories_with_embeddings = [m for m in memories if m.embedding]

            if len(memories_with_embeddings) < 2:
                return memories

            # Extract embeddings
            embeddings = np.array([memory.embedding for memory in memories_with_embeddings])

            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,  # Convert similarity to distance
                min_samples=2,
                metric='cosine'
            )

            cluster_labels = clustering.fit_predict(embeddings)

            # Group memories by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(memories_with_embeddings[i])

            consolidated_memories = []

            # Process each cluster
            for cluster_label, cluster_memories in clusters.items():
                if cluster_label == -1:  # Noise points (not clustered)
                    # Add high-importance noise points individually
                    for memory in cluster_memories:
                        if memory.importance_score >= self.importance_threshold:
                            consolidated_memories.append(memory)
                else:
                    # Merge cluster memories
                    merged_memory = await self._merge_cluster_memories(cluster_memories)
                    if merged_memory:
                        consolidated_memories.append(merged_memory)

            # Add memories without embeddings if they're important
            memories_without_embeddings = [m for m in memories if not m.embedding]
            for memory in memories_without_embeddings:
                if memory.importance_score >= self.importance_threshold:
                    consolidated_memories.append(memory)

            return consolidated_memories

        except Exception as e:
            logger.error(f"Error in clustering consolidation: {e}")
            return memories

    async def _similarity_merge_consolidation(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """Consolidate memories by merging highly similar ones"""
        try:
            consolidated = []
            processed = set()

            for i, memory in enumerate(memories):
                if i in processed:
                    continue

                # Find similar memories
                similar_memories = [memory]
                for j, other_memory in enumerate(memories[i+1:], i+1):
                    if j in processed:
                        continue

                    similarity = await self._calculate_memory_similarity(memory, other_memory)
                    if similarity >= self.similarity_threshold:
                        similar_memories.append(other_memory)
                        processed.add(j)

                # Merge similar memories or keep original
                if len(similar_memories) > 1:
                    merged = await self._merge_similar_memories(similar_memories)
                    if merged:
                        consolidated.append(merged)
                else:
                    if memory.importance_score >= self.importance_threshold:
                        consolidated.append(memory)

                processed.add(i)

            return consolidated

        except Exception as e:
            logger.error(f"Error in similarity merge consolidation: {e}")
            return memories

    async def _importance_filter_consolidation(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """Simple consolidation that filters by importance threshold"""
        try:
            return [memory for memory in memories if memory.importance_score >= self.importance_threshold]

        except Exception as e:
            logger.error(f"Error in importance filter consolidation: {e}")
            return memories

    async def _temporal_grouping_consolidation(self, memories: List[MemoryItem]) -> List[MemoryItem]:
        """Group memories by time periods and consolidate within groups"""
        try:
            # Group memories by time periods (e.g., 1-hour windows)
            time_groups = defaultdict(list)

            for memory in memories:
                # Group by hour
                time_key = memory.timestamp.replace(minute=0, second=0, microsecond=0)
                time_groups[time_key].append(memory)

            consolidated = []

            # Process each time group
            for time_key, group_memories in time_groups.items():
                # Sort by importance within the group
                group_memories.sort(key=lambda m: m.importance_score, reverse=True)

                # Take top memories from each group
                top_memories = [
                    memory for memory in group_memories
                    if memory.importance_score >= self.importance_threshold
                ]

                # If we have many important memories in this time period, merge some
                if len(top_memories) > 5:
                    # Keep top 3 and try to merge the rest
                    consolidated.extend(top_memories[:3])
                    if len(top_memories) > 3:
                        merged = await self._merge_similar_memories(top_memories[3:])
                        if merged:
                            consolidated.append(merged)
                else:
                    consolidated.extend(top_memories)

            return consolidated

        except Exception as e:
            logger.error(f"Error in temporal grouping consolidation: {e}")
            return memories

    async def _merge_cluster_memories(self, cluster_memories: List[MemoryItem]) -> Optional[MemoryItem]:
        """Merge memories from the same cluster"""
        try:
            if not cluster_memories:
                return None

            # Find the most important memory as the base
            base_memory = max(cluster_memories, key=lambda m: m.importance_score)

            # Merge content from all memories
            all_content = [memory.content for memory in cluster_memories]
            merged_content = await self._merge_content(all_content)

            # Combine emotions
            merged_emotions = {}
            for memory in cluster_memories:
                if memory.emotions:
                    for emotion, value in memory.emotions.items():
                        if emotion in merged_emotions:
                            merged_emotions[emotion] = max(merged_emotions[emotion], value)
                        else:
                            merged_emotions[emotion] = value

            # Combine context tags
            all_tags = set()
            for memory in cluster_memories:
                if memory.context_tags:
                    all_tags.update(memory.context_tags)

            # Calculate merged importance (weighted average)
            total_importance = sum(m.importance_score for m in cluster_memories)
            merged_importance = min(1.0, total_importance / len(cluster_memories) * 1.2)  # Slight boost for merging

            # Create merged memory
            merged_memory = MemoryItem(
                content=merged_content,
                timestamp=base_memory.timestamp,
                importance_score=merged_importance,
                memory_type=base_memory.memory_type,
                session_id=base_memory.session_id,
                agent_id=base_memory.agent_id,
                emotions=merged_emotions,
                context_tags=list(all_tags),
                embedding=base_memory.embedding  # Use base memory's embedding
            )

            return merged_memory

        except Exception as e:
            logger.error(f"Error merging cluster memories: {e}")
            return None

    async def _merge_similar_memories(self, memories: List[MemoryItem]) -> Optional[MemoryItem]:
        """Merge a list of similar memories"""
        return await self._merge_cluster_memories(memories)

    async def _merge_content(self, content_list: List[str]) -> str:
        """Merge content from multiple memories intelligently"""
        try:
            if not content_list:
                return ""

            if len(content_list) == 1:
                return content_list[0]

            # For now, use simple concatenation with deduplication
            # TODO: Implement more sophisticated content merging
            unique_content = []
            seen_content = set()

            for content in content_list:
                content_lower = content.lower().strip()
                if content_lower not in seen_content:
                    seen_content.add(content_lower)
                    unique_content.append(content.strip())

            # Join with appropriate separator
            if len(unique_content) <= 2:
                merged = " ".join(unique_content)
            else:
                merged = ". ".join(unique_content)

            return merged

        except Exception as e:
            logger.warning(f"Error merging content: {e}")
            return content_list[0] if content_list else ""

    async def _calculate_memory_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate similarity between two memories"""
        try:
            # If both have embeddings, use cosine similarity
            if memory1.embedding and memory2.embedding:
                embedding1 = np.array(memory1.embedding).reshape(1, -1)
                embedding2 = np.array(memory2.embedding).reshape(1, -1)
                similarity = cosine_similarity(embedding1, embedding2)[0][0]
                return float(similarity)

            # Fallback to simple text similarity
            return await self._text_similarity(memory1.content, memory2.content)

        except Exception as e:
            logger.warning(f"Error calculating memory similarity: {e}")
            return 0.0

    async def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            # Jaccard similarity
            similarity = len(intersection) / len(union)
            return similarity

        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            return 0.0

    async def get_consolidation_recommendations(
        self,
        memories: List[MemoryItem]
    ) -> Dict[str, Any]:
        """Get recommendations for consolidation strategies"""
        try:
            recommendations = {
                "total_memories": len(memories),
                "high_importance_count": sum(1 for m in memories if m.importance_score >= self.importance_threshold),
                "memories_with_embeddings": sum(1 for m in memories if m.embedding),
                "memory_type_distribution": {},
                "recommended_strategy": "clustering",
                "estimated_reduction": 0
            }

            # Calculate memory type distribution
            type_counts = defaultdict(int)
            for memory in memories:
                type_counts[memory.memory_type.value] += 1

            recommendations["memory_type_distribution"] = dict(type_counts)

            # Recommend strategy based on data characteristics
            if recommendations["memories_with_embeddings"] >= len(memories) * 0.8:
                recommendations["recommended_strategy"] = "clustering"
                recommendations["estimated_reduction"] = int(len(memories) * 0.3)
            elif len(memories) > 100:
                recommendations["recommended_strategy"] = "temporal_grouping"
                recommendations["estimated_reduction"] = int(len(memories) * 0.4)
            else:
                recommendations["recommended_strategy"] = "importance_filter"
                recommendations["estimated_reduction"] = len(memories) - recommendations["high_importance_count"]

            return recommendations

        except Exception as e:
            logger.error(f"Error generating consolidation recommendations: {e}")
            return {"error": str(e)}