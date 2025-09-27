"""
Long-term Memory Manager - MongoDB-based persistent memory storage
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import motor.motor_asyncio

from .models import MemoryItem, MemoryType, MemoryQuery, MemoryRetrievalResult

logger = logging.getLogger(__name__)


class LongTermMemoryManager:
    """
    Manages persistent long-term memory using MongoDB with optimized queries and indexing
    """

    def __init__(self, mongodb_client: motor.motor_asyncio.AsyncIOMotorClient):
        self.client = mongodb_client
        self.db = mongodb_client.multi_agent_conversations
        self.collection = self.db.long_term_memory

        # Collection names
        self.MEMORY_COLLECTION = "long_term_memory"
        self.CHARACTER_DEVELOPMENT_COLLECTION = "character_development"
        self.SESSION_ANALYTICS_COLLECTION = "session_analytics"

    async def initialize(self):
        """Initialize the long-term memory manager and ensure indexes"""
        try:
            # Ensure database collections exist and have proper indexes
            await self._ensure_indexes()

            # Test connection
            await self.db.command("ping")
            logger.info("Long-term Memory Manager connected to MongoDB")

        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            raise

    async def shutdown(self):
        """Shutdown the long-term memory manager"""
        if self.client:
            self.client.close()

    async def store_memory(self, memory_item: MemoryItem) -> str:
        """Store a memory item in long-term persistent storage"""
        try:
            memory_doc = {
                'memory_id': memory_item.memory_id,
                'session_id': memory_item.session_id,
                'content': memory_item.content,
                'timestamp': memory_item.timestamp,
                'importance_score': memory_item.importance_score,
                'memory_type': memory_item.memory_type.value,
                'agent_id': memory_item.agent_id,
                'emotions': memory_item.emotions or {},
                'context_tags': memory_item.context_tags or [],
                'embedding': memory_item.embedding,
                'consolidation_date': datetime.utcnow(),
                'access_count': 0,
                'last_accessed': None
            }

            result = await self.collection.insert_one(memory_doc)

            logger.debug(f"Stored memory {memory_item.memory_id} in long-term storage")
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error storing memory in long-term storage: {e}")
            raise

    async def retrieve_memories_by_query(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """Retrieve memories using complex query with multiple filters"""
        start_time = datetime.now()

        try:
            # Build MongoDB query
            mongo_query = {"session_id": query.session_id}

            # Add memory type filter
            if query.memory_types:
                mongo_query["memory_type"] = {"$in": [mt.value for mt in query.memory_types]}

            # Add agent filter
            if query.agent_id:
                mongo_query["agent_id"] = query.agent_id

            # Add importance threshold
            if query.importance_threshold > 0:
                mongo_query["importance_score"] = {"$gte": query.importance_threshold}

            # Add time range filter
            if query.time_range:
                start_time_filter, end_time_filter = query.time_range
                mongo_query["timestamp"] = {
                    "$gte": start_time_filter,
                    "$lte": end_time_filter
                }

            # Add text search if query text provided
            if query.query_text:
                mongo_query["$text"] = {"$search": query.query_text}

            # Execute query with sorting and limiting
            cursor = self.collection.find(mongo_query).sort([
                ("importance_score", -1),
                ("timestamp", -1)
            ]).limit(query.max_results)

            # Convert to MemoryItem objects
            memories = []
            async for doc in cursor:
                try:
                    memory_item = self._doc_to_memory_item(doc)
                    memories.append(memory_item)

                    # Update access tracking
                    await self._update_access_tracking(doc["_id"])

                except Exception as e:
                    logger.warning(f"Error converting document to memory item: {e}")
                    continue

            # Get total count for pagination
            total_count = await self.collection.count_documents(mongo_query)

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.debug(f"Retrieved {len(memories)} memories from long-term storage in {query_time:.2f}ms")

            return MemoryRetrievalResult(
                memories=memories,
                total_found=total_count,
                query_time_ms=query_time
            )

        except Exception as e:
            logger.error(f"Error retrieving memories from long-term storage: {e}")
            raise

    async def get_memories_by_importance(
        self,
        session_id: str,
        min_importance: float = 0.7,
        limit: int = 50
    ) -> List[MemoryItem]:
        """Get memories above a certain importance threshold"""
        try:
            query = {
                "session_id": session_id,
                "importance_score": {"$gte": min_importance}
            }

            cursor = self.collection.find(query).sort("importance_score", -1).limit(limit)

            memories = []
            async for doc in cursor:
                memory_item = self._doc_to_memory_item(doc)
                memories.append(memory_item)

            logger.debug(f"Retrieved {len(memories)} high-importance memories for session {session_id}")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving high-importance memories: {e}")
            return []

    async def get_memories_by_agent(
        self,
        session_id: str,
        agent_id: str,
        limit: int = 50
    ) -> List[MemoryItem]:
        """Get memories associated with a specific agent"""
        try:
            query = {
                "session_id": session_id,
                "agent_id": agent_id
            }

            cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)

            memories = []
            async for doc in cursor:
                memory_item = self._doc_to_memory_item(doc)
                memories.append(memory_item)

            logger.debug(f"Retrieved {len(memories)} memories for agent {agent_id} in session {session_id}")
            return memories

        except Exception as e:
            logger.error(f"Error retrieving agent memories: {e}")
            return []

    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a session's long-term memories"""
        try:
            # Aggregation pipeline for session statistics
            pipeline = [
                {"$match": {"session_id": session_id}},
                {
                    "$group": {
                        "_id": None,
                        "total_memories": {"$sum": 1},
                        "avg_importance": {"$avg": "$importance_score"},
                        "max_importance": {"$max": "$importance_score"},
                        "min_importance": {"$min": "$importance_score"},
                        "earliest_memory": {"$min": "$timestamp"},
                        "latest_memory": {"$max": "$timestamp"},
                        "memory_types": {"$addToSet": "$memory_type"},
                        "involved_agents": {"$addToSet": "$agent_id"},
                        "total_access_count": {"$sum": "$access_count"}
                    }
                }
            ]

            result = await self.collection.aggregate(pipeline).to_list(length=1)

            if result:
                stats = result[0]
                stats.pop("_id", None)  # Remove MongoDB _id field

                # Add memory type distribution
                type_distribution = await self._get_memory_type_distribution(session_id)
                stats["memory_type_distribution"] = type_distribution

                # Add agent contribution statistics
                agent_stats = await self._get_agent_contribution_stats(session_id)
                stats["agent_contributions"] = agent_stats

                return stats
            else:
                return {
                    "total_memories": 0,
                    "session_id": session_id,
                    "message": "No long-term memories found for this session"
                }

        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return {"error": str(e)}

    async def delete_session_memories(self, session_id: str) -> int:
        """Delete all long-term memories for a session"""
        try:
            result = await self.collection.delete_many({"session_id": session_id})
            deleted_count = result.deleted_count

            logger.info(f"Deleted {deleted_count} long-term memories for session {session_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting session memories: {e}")
            raise

    async def update_memory_importance(self, memory_id: str, new_importance: float) -> bool:
        """Update the importance score of a specific memory"""
        try:
            result = await self.collection.update_one(
                {"memory_id": memory_id},
                {
                    "$set": {
                        "importance_score": new_importance,
                        "last_updated": datetime.utcnow()
                    }
                }
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Error updating memory importance: {e}")
            return False

    async def get_memory_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get analytics data for long-term memory usage"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)

            # Aggregation pipeline for analytics
            pipeline = [
                {"$match": {"consolidation_date": {"$gte": start_date}}},
                {
                    "$group": {
                        "_id": {
                            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$consolidation_date"}},
                            "memory_type": "$memory_type"
                        },
                        "count": {"$sum": 1},
                        "avg_importance": {"$avg": "$importance_score"}
                    }
                },
                {"$sort": {"_id.date": 1}}
            ]

            analytics_data = await self.collection.aggregate(pipeline).to_list(length=None)

            # Process analytics data
            daily_stats = {}
            for item in analytics_data:
                date = item["_id"]["date"]
                memory_type = item["_id"]["memory_type"]

                if date not in daily_stats:
                    daily_stats[date] = {"total_memories": 0, "types": {}}

                daily_stats[date]["types"][memory_type] = {
                    "count": item["count"],
                    "avg_importance": item["avg_importance"]
                }
                daily_stats[date]["total_memories"] += item["count"]

            return {
                "period_days": days_back,
                "daily_statistics": daily_stats,
                "generated_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating memory analytics: {e}")
            return {"error": str(e)}

    def _doc_to_memory_item(self, doc: Dict[str, Any]) -> MemoryItem:
        """Convert MongoDB document to MemoryItem object"""
        return MemoryItem(
            memory_id=doc["memory_id"],
            content=doc["content"],
            timestamp=doc["timestamp"],
            importance_score=doc["importance_score"],
            memory_type=MemoryType(doc["memory_type"]),
            session_id=doc["session_id"],
            agent_id=doc.get("agent_id"),
            emotions=doc.get("emotions"),
            context_tags=doc.get("context_tags"),
            embedding=doc.get("embedding")
        )

    async def _update_access_tracking(self, doc_id) -> None:
        """Update access tracking for a memory"""
        try:
            await self.collection.update_one(
                {"_id": doc_id},
                {
                    "$inc": {"access_count": 1},
                    "$set": {"last_accessed": datetime.utcnow()}
                }
            )
        except Exception as e:
            logger.warning(f"Error updating access tracking: {e}")

    async def _get_memory_type_distribution(self, session_id: str) -> Dict[str, int]:
        """Get distribution of memory types for a session"""
        try:
            pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {"_id": "$memory_type", "count": {"$sum": 1}}}
            ]

            result = await self.collection.aggregate(pipeline).to_list(length=None)
            return {item["_id"]: item["count"] for item in result}

        except Exception as e:
            logger.warning(f"Error getting memory type distribution: {e}")
            return {}

    async def _get_agent_contribution_stats(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Get agent contribution statistics for a session"""
        try:
            pipeline = [
                {"$match": {"session_id": session_id, "agent_id": {"$ne": None}}},
                {
                    "$group": {
                        "_id": "$agent_id",
                        "memory_count": {"$sum": 1},
                        "avg_importance": {"$avg": "$importance_score"},
                        "total_importance": {"$sum": "$importance_score"}
                    }
                }
            ]

            result = await self.collection.aggregate(pipeline).to_list(length=None)
            return {
                item["_id"]: {
                    "memory_count": item["memory_count"],
                    "avg_importance": item["avg_importance"],
                    "total_importance": item["total_importance"]
                }
                for item in result
            }

        except Exception as e:
            logger.warning(f"Error getting agent contribution stats: {e}")
            return {}

    async def _ensure_indexes(self):
        """Ensure proper indexes exist for optimal query performance"""
        try:
            # Create indexes for common queries
            await self.collection.create_index([("session_id", 1), ("timestamp", -1)])
            await self.collection.create_index([("session_id", 1), ("importance_score", -1)])
            await self.collection.create_index([("session_id", 1), ("agent_id", 1)])
            await self.collection.create_index([("memory_type", 1)])
            await self.collection.create_index([("consolidation_date", -1)])
            await self.collection.create_index([("memory_id", 1)], unique=True)

            # Create text index for content search
            await self.collection.create_index([("content", "text")])

            # Create compound indexes for complex queries
            await self.collection.create_index([
                ("session_id", 1),
                ("memory_type", 1),
                ("importance_score", -1)
            ])

            logger.info("Long-term memory indexes ensured")

        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")