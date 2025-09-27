"""
Vector Memory Retrieval - Semantic search using embeddings
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import motor.motor_asyncio
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity

from .models import MemoryItem, MemoryType, MemoryRetrievalResult

logger = logging.getLogger(__name__)


class VectorMemoryRetrieval:
    """
    Manages vector-based memory retrieval using embeddings for semantic similarity search
    """

    def __init__(
        self,
        mongodb_client: motor.motor_asyncio.AsyncIOMotorClient,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_embedding_model: str = "text-embedding-ada-002"
    ):
        self.mongodb_client = mongodb_client
        self.db = mongodb_client.multi_agent_conversations
        self.collection = self.db.long_term_memory

        # Embedding models
        self.local_model_name = embedding_model
        self.openai_model_name = openai_embedding_model
        self.local_model = None

        # Check if OpenAI API key is properly configured
        import os
        openai_key = os.getenv('OPENAI_API_KEY', '')
        self.use_openai = openai_key and openai_key != 'your_openai_api_key_here'
        if not self.use_openai:
            logger.info("OpenAI API key not configured, using local embeddings")

        # Vector search configuration
        self.vector_dimension = 1536 if self.use_openai else 384  # OpenAI ada-002: 1536, all-MiniLM-L6-v2: 384
        self.similarity_threshold = 0.7

    async def initialize(self):
        """Initialize the vector retrieval system"""
        try:
            # Initialize local embedding model as fallback
            await asyncio.get_event_loop().run_in_executor(
                None, self._load_local_model
            )

            # Test embedding generation
            test_embedding = await self.generate_embedding("test")
            if test_embedding:
                logger.info(f"Vector Memory Retrieval initialized with {len(test_embedding)}-dimensional embeddings")
            else:
                raise Exception("Failed to generate test embedding")

        except Exception as e:
            logger.error(f"Failed to initialize Vector Memory Retrieval: {e}")
            raise

    def _load_local_model(self):
        """Load local sentence transformer model"""
        try:
            self.local_model = SentenceTransformer(self.local_model_name)
            logger.info(f"Loaded local embedding model: {self.local_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            self.local_model = None

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI or local model"""
        try:
            if self.use_openai:
                return await self._generate_openai_embedding(text)
            else:
                return await self._generate_local_embedding(text)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Try fallback method
            try:
                if self.use_openai:
                    return await self._generate_local_embedding(text)
                else:
                    return await self._generate_openai_embedding(text)
            except Exception as fallback_error:
                logger.error(f"Fallback embedding generation failed: {fallback_error}")
                return None

    async def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        try:
            response = await openai.Embedding.acreate(
                model=self.openai_model_name,
                input=text.replace("\n", " ")
            )

            embedding = response['data'][0]['embedding']
            return embedding

        except Exception as e:
            logger.warning(f"OpenAI embedding generation failed: {e}")
            return None

    async def _generate_local_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using local SentenceTransformer model"""
        try:
            if not self.local_model:
                return None

            # Run in executor to avoid blocking
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.local_model.encode, text
            )

            return embedding.tolist()

        except Exception as e:
            logger.warning(f"Local embedding generation failed: {e}")
            return None

    async def search_similar_memories(
        self,
        query_text: str,
        session_id: str,
        limit: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        importance_threshold: float = 0.0,
        similarity_threshold: float = None
    ) -> MemoryRetrievalResult:
        """Search for memories similar to query using vector similarity"""
        start_time = datetime.now()

        try:
            # Generate embedding for query
            query_embedding = await self.generate_embedding(query_text)
            if not query_embedding:
                logger.warning("Failed to generate query embedding, falling back to text search")
                return await self._fallback_text_search(query_text, session_id, limit)

            # Build MongoDB query
            mongo_query = {"session_id": session_id, "embedding": {"$exists": True}}

            if memory_types:
                mongo_query["memory_type"] = {"$in": [mt.value for mt in memory_types]}

            if importance_threshold > 0:
                mongo_query["importance_score"] = {"$gte": importance_threshold}

            # Get memories with embeddings
            cursor = self.collection.find(mongo_query)
            candidates = await cursor.to_list(length=None)

            if not candidates:
                logger.debug(f"No memories with embeddings found for session {session_id}")
                return MemoryRetrievalResult(memories=[], total_found=0, query_time_ms=0)

            # Calculate similarities
            similarities = await self._calculate_similarities(query_embedding, candidates)

            # Filter by similarity threshold
            threshold = similarity_threshold or self.similarity_threshold
            filtered_results = [
                (memory, similarity) for memory, similarity in similarities
                if similarity >= threshold
            ]

            # Sort by similarity and limit results
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            top_results = filtered_results[:limit]

            # Convert to MemoryItem objects
            memories = []
            relevance_scores = []

            for memory_doc, similarity in top_results:
                try:
                    memory_item = self._doc_to_memory_item(memory_doc)
                    memories.append(memory_item)
                    relevance_scores.append(similarity)
                except Exception as e:
                    logger.warning(f"Error converting memory document: {e}")
                    continue

            query_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.debug(f"Found {len(memories)} similar memories in {query_time:.2f}ms")

            return MemoryRetrievalResult(
                memories=memories,
                total_found=len(candidates),
                query_time_ms=query_time,
                relevance_scores=relevance_scores
            )

        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            raise

    async def _calculate_similarities(
        self,
        query_embedding: List[float],
        candidates: List[Dict]
    ) -> List[Tuple[Dict, float]]:
        """Calculate cosine similarities between query and candidate embeddings"""
        try:
            query_vector = np.array(query_embedding).reshape(1, -1)
            similarities = []

            for candidate in candidates:
                try:
                    candidate_embedding = candidate.get("embedding")
                    if not candidate_embedding:
                        continue

                    candidate_vector = np.array(candidate_embedding).reshape(1, -1)

                    # Ensure vectors have same dimensions
                    if candidate_vector.shape[1] != query_vector.shape[1]:
                        logger.warning(f"Embedding dimension mismatch: query={query_vector.shape[1]}, candidate={candidate_vector.shape[1]}")
                        continue

                    similarity = cosine_similarity(query_vector, candidate_vector)[0][0]
                    similarities.append((candidate, float(similarity)))

                except Exception as e:
                    logger.warning(f"Error calculating similarity for candidate: {e}")
                    continue

            return similarities

        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return []

    async def _fallback_text_search(
        self,
        query_text: str,
        session_id: str,
        limit: int
    ) -> MemoryRetrievalResult:
        """Fallback to MongoDB text search when vector search fails"""
        try:
            mongo_query = {
                "session_id": session_id,
                "$text": {"$search": query_text}
            }

            cursor = self.collection.find(mongo_query).sort([
                ("score", {"$meta": "textScore"}),
                ("importance_score", -1)
            ]).limit(limit)

            memories = []
            async for doc in cursor:
                try:
                    memory_item = self._doc_to_memory_item(doc)
                    memories.append(memory_item)
                except Exception as e:
                    logger.warning(f"Error converting document in fallback search: {e}")
                    continue

            return MemoryRetrievalResult(
                memories=memories,
                total_found=len(memories),
                query_time_ms=0  # Not tracked for fallback
            )

        except Exception as e:
            logger.error(f"Error in fallback text search: {e}")
            return MemoryRetrievalResult(memories=[], total_found=0, query_time_ms=0)

    async def update_memory_embedding(self, memory_id: str, content: str) -> bool:
        """Update embedding for a specific memory"""
        try:
            embedding = await self.generate_embedding(content)
            if not embedding:
                return False

            result = await self.collection.update_one(
                {"memory_id": memory_id},
                {"$set": {"embedding": embedding, "embedding_updated": datetime.utcnow()}}
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Error updating memory embedding: {e}")
            return False

    async def batch_generate_embeddings(
        self,
        session_id: str,
        batch_size: int = 50
    ) -> int:
        """Generate embeddings for memories that don't have them"""
        try:
            # Find memories without embeddings
            query = {
                "session_id": session_id,
                "embedding": {"$exists": False}
            }

            memories_without_embeddings = await self.collection.find(query).to_list(length=None)

            if not memories_without_embeddings:
                logger.debug(f"All memories in session {session_id} already have embeddings")
                return 0

            updated_count = 0

            # Process in batches
            for i in range(0, len(memories_without_embeddings), batch_size):
                batch = memories_without_embeddings[i:i + batch_size]

                # Generate embeddings for batch
                embeddings = await self._generate_batch_embeddings([doc["content"] for doc in batch])

                # Update documents with embeddings
                for doc, embedding in zip(batch, embeddings):
                    if embedding:
                        await self.collection.update_one(
                            {"_id": doc["_id"]},
                            {
                                "$set": {
                                    "embedding": embedding,
                                    "embedding_generated": datetime.utcnow()
                                }
                            }
                        )
                        updated_count += 1

                # Small delay between batches to avoid rate limits
                await asyncio.sleep(0.1)

            logger.info(f"Generated embeddings for {updated_count} memories in session {session_id}")
            return updated_count

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return 0

    async def _generate_batch_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently"""
        try:
            embeddings = []

            if self.use_openai:
                # OpenAI API supports batch processing
                try:
                    response = await openai.Embedding.acreate(
                        model=self.openai_model_name,
                        input=[text.replace("\n", " ") for text in texts]
                    )

                    for item in response['data']:
                        embeddings.append(item['embedding'])

                except Exception as e:
                    logger.warning(f"Batch OpenAI embedding failed: {e}")
                    # Fallback to individual generation
                    for text in texts:
                        embedding = await self._generate_openai_embedding(text)
                        embeddings.append(embedding)
            else:
                # Local model batch processing
                if self.local_model:
                    batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                        None, self.local_model.encode, texts
                    )
                    embeddings = [emb.tolist() for emb in batch_embeddings]
                else:
                    embeddings = [None] * len(texts)

            return embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return [None] * len(texts)

    async def get_similar_memories_by_embedding(
        self,
        target_embedding: List[float],
        session_id: str,
        limit: int = 10,
        exclude_memory_id: Optional[str] = None
    ) -> List[Tuple[MemoryItem, float]]:
        """Find memories similar to a given embedding"""
        try:
            query = {"session_id": session_id, "embedding": {"$exists": True}}

            if exclude_memory_id:
                query["memory_id"] = {"$ne": exclude_memory_id}

            candidates = await self.collection.find(query).to_list(length=None)

            if not candidates:
                return []

            # Calculate similarities
            similarities = await self._calculate_similarities(target_embedding, candidates)

            # Sort and limit
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:limit]

            # Convert to MemoryItem objects
            results = []
            for memory_doc, similarity in top_similarities:
                try:
                    memory_item = self._doc_to_memory_item(memory_doc)
                    results.append((memory_item, similarity))
                except Exception as e:
                    logger.warning(f"Error converting similar memory: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Error finding similar memories by embedding: {e}")
            return []

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

    async def get_embedding_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about embeddings for a session"""
        try:
            total_memories = await self.collection.count_documents({"session_id": session_id})

            memories_with_embeddings = await self.collection.count_documents({
                "session_id": session_id,
                "embedding": {"$exists": True}
            })

            coverage_percentage = (memories_with_embeddings / max(1, total_memories)) * 100

            return {
                "session_id": session_id,
                "total_memories": total_memories,
                "memories_with_embeddings": memories_with_embeddings,
                "coverage_percentage": round(coverage_percentage, 2),
                "embedding_model": self.openai_model_name if self.use_openai else self.local_model_name,
                "vector_dimension": self.vector_dimension
            }

        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {"error": str(e)}