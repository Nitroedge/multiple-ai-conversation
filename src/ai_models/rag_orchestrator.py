"""
RAG (Retrieval-Augmented Generation) Orchestrator
Coordinates retrieval and generation for enhanced AI responses
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .model_interface import ModelRequest, ModelResponse, TaskType
from .vector_enhancer import VectorEnhancer, EmbeddingProvider

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies"""
    SEMANTIC_ONLY = "semantic_only"
    HYBRID_SEARCH = "hybrid_search"
    KEYWORD_ONLY = "keyword_only"
    MULTI_VECTOR = "multi_vector"
    CONVERSATIONAL = "conversational"


class RAGConfig(BaseModel):
    """Configuration for RAG operations"""
    # Retrieval settings
    max_retrieved_docs: int = Field(default=10, description="Maximum documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    retrieval_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.HYBRID_SEARCH)

    # Context settings
    max_context_tokens: int = Field(default=4000, description="Maximum tokens for context")
    context_overlap: int = Field(default=100, description="Overlap between context chunks")

    # Generation settings
    include_citations: bool = Field(default=True, description="Include source citations")
    cite_format: str = Field(default="[{source_id}]", description="Citation format")

    # Quality settings
    relevance_threshold: float = Field(default=0.6, description="Minimum relevance for inclusion")
    diversity_threshold: float = Field(default=0.8, description="Similarity threshold for diversity")

    # Embedding settings
    embedding_provider: Optional[EmbeddingProvider] = Field(None, description="Preferred embedding provider")
    embedding_model: Optional[str] = Field(None, description="Specific embedding model")


class RetrievedDocument(BaseModel):
    """Document retrieved for RAG"""
    doc_id: str
    content: str
    similarity_score: float
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    timestamp: Optional[datetime] = None


class RAGContext(BaseModel):
    """Context built for generation"""
    retrieved_docs: List[RetrievedDocument]
    context_text: str
    total_tokens: int
    retrieval_time_ms: float
    citations: List[Dict[str, str]] = Field(default_factory=list)


class RAGResult(BaseModel):
    """Result from RAG operation"""
    response: ModelResponse
    rag_context: RAGContext
    retrieval_strategy_used: RetrievalStrategy
    total_time_ms: float


class RAGOrchestrator:
    """
    Orchestrates retrieval-augmented generation workflow
    """

    def __init__(
        self,
        vector_enhancer: VectorEnhancer,
        memory_manager,
        model_router
    ):
        self.vector_enhancer = vector_enhancer
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self.default_config = RAGConfig()

        # Strategy-specific configurations
        self.strategy_configs = {
            RetrievalStrategy.SEMANTIC_ONLY: {
                "use_vector_search": True,
                "use_text_search": False,
                "vector_weight": 1.0,
                "text_weight": 0.0
            },
            RetrievalStrategy.HYBRID_SEARCH: {
                "use_vector_search": True,
                "use_text_search": True,
                "vector_weight": 0.7,
                "text_weight": 0.3
            },
            RetrievalStrategy.KEYWORD_ONLY: {
                "use_vector_search": False,
                "use_text_search": True,
                "vector_weight": 0.0,
                "text_weight": 1.0
            },
            RetrievalStrategy.MULTI_VECTOR: {
                "use_multiple_embeddings": True,
                "embedding_providers": [
                    EmbeddingProvider.OPENAI,
                    EmbeddingProvider.SENTENCE_TRANSFORMERS
                ]
            }
        }

    async def process_rag_request(
        self,
        request: ModelRequest,
        config: Optional[RAGConfig] = None
    ) -> RAGResult:
        """Process a request with RAG enhancement"""
        start_time = time.time()
        config = config or self.default_config

        try:
            # Step 1: Retrieve relevant context
            rag_context = await self._retrieve_context(request, config)

            # Step 2: Enhance request with context
            enhanced_request = await self._enhance_request_with_context(
                request, rag_context, config
            )

            # Step 3: Generate response
            response = await self.model_router.execute_request_with_routing(enhanced_request)

            # Step 4: Post-process response with citations
            if config.include_citations:
                response = await self._add_citations(response, rag_context, config)

            total_time = (time.time() - start_time) * 1000

            return RAGResult(
                response=response,
                rag_context=rag_context,
                retrieval_strategy_used=config.retrieval_strategy,
                total_time_ms=total_time
            )

        except Exception as e:
            self.logger.error(f"RAG processing failed: {e}")
            # Fallback to regular processing
            response = await self.model_router.execute_request_with_routing(request)
            return RAGResult(
                response=response,
                rag_context=RAGContext(
                    retrieved_docs=[],
                    context_text="",
                    total_tokens=0,
                    retrieval_time_ms=0
                ),
                retrieval_strategy_used=config.retrieval_strategy,
                total_time_ms=(time.time() - start_time) * 1000
            )

    async def _retrieve_context(
        self,
        request: ModelRequest,
        config: RAGConfig
    ) -> RAGContext:
        """Retrieve relevant context for the request"""
        start_time = time.time()

        try:
            # Determine search strategy
            strategy_config = self.strategy_configs.get(
                config.retrieval_strategy,
                self.strategy_configs[RetrievalStrategy.HYBRID_SEARCH]
            )

            # Perform retrieval based on strategy
            if config.retrieval_strategy == RetrievalStrategy.MULTI_VECTOR:
                retrieved_docs = await self._multi_vector_retrieval(request, config)
            elif config.retrieval_strategy == RetrievalStrategy.CONVERSATIONAL:
                retrieved_docs = await self._conversational_retrieval(request, config)
            else:
                retrieved_docs = await self._standard_retrieval(request, config, strategy_config)

            # Build context from retrieved documents
            context_text, citations = await self._build_context(retrieved_docs, config)

            retrieval_time = (time.time() - start_time) * 1000

            return RAGContext(
                retrieved_docs=retrieved_docs,
                context_text=context_text,
                total_tokens=len(context_text.split()) * 1.33,  # Rough token estimate
                retrieval_time_ms=retrieval_time,
                citations=citations
            )

        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return RAGContext(
                retrieved_docs=[],
                context_text="",
                total_tokens=0,
                retrieval_time_ms=0
            )

    async def _standard_retrieval(
        self,
        request: ModelRequest,
        config: RAGConfig,
        strategy_config: Dict[str, Any]
    ) -> List[RetrievedDocument]:
        """Standard retrieval using vector and/or text search"""
        retrieved_docs = []

        try:
            session_id = request.session_id or "default"

            # Vector search
            if strategy_config.get("use_vector_search", True):
                vector_results = await self._vector_search(
                    request.prompt,
                    session_id,
                    config
                )
                retrieved_docs.extend(vector_results)

            # Text search
            if strategy_config.get("use_text_search", False):
                text_results = await self._text_search(
                    request.prompt,
                    session_id,
                    config
                )
                retrieved_docs.extend(text_results)

            # Combine and rank results
            if strategy_config.get("use_vector_search") and strategy_config.get("use_text_search"):
                retrieved_docs = await self._rank_fusion(
                    retrieved_docs,
                    strategy_config.get("vector_weight", 0.7),
                    strategy_config.get("text_weight", 0.3)
                )

            # Apply diversity and relevance filtering
            retrieved_docs = await self._filter_documents(retrieved_docs, config)

            return retrieved_docs[:config.max_retrieved_docs]

        except Exception as e:
            self.logger.error(f"Standard retrieval failed: {e}")
            return []

    async def _vector_search(
        self,
        query: str,
        session_id: str,
        config: RAGConfig
    ) -> List[RetrievedDocument]:
        """Perform vector similarity search"""
        try:
            # Get best embedding model for the query
            provider, model_name = self.vector_enhancer.get_best_embedding_model(
                task_type="semantic_search",
                text_length=len(query),
                multilingual=False  # Could be determined from query
            )

            # Use configured provider if specified
            if config.embedding_provider:
                provider = config.embedding_provider
            if config.embedding_model:
                model_name = config.embedding_model

            # Search for similar memories
            if hasattr(self.memory_manager, 'vector_retrieval'):
                memory_results = await self.memory_manager.vector_retrieval.search_similar_memories(
                    query_text=query,
                    session_id=session_id,
                    limit=config.max_retrieved_docs * 2,  # Get more for filtering
                    similarity_threshold=config.similarity_threshold
                )

                retrieved_docs = []
                for memory in memory_results.memories:
                    # Calculate relevance score based on multiple factors
                    relevance_score = await self._calculate_relevance_score(
                        memory, query, session_id
                    )

                    if relevance_score >= config.relevance_threshold:
                        retrieved_docs.append(RetrievedDocument(
                            doc_id=memory.memory_id,
                            content=memory.content,
                            similarity_score=relevance_score,  # Use relevance as similarity
                            relevance_score=relevance_score,
                            metadata={
                                "memory_type": memory.memory_type.value,
                                "importance": memory.importance_score,
                                "timestamp": memory.timestamp.isoformat(),
                                "emotions": memory.emotions,
                                "context_tags": memory.context_tags
                            },
                            source=f"memory:{memory.memory_type.value}",
                            timestamp=memory.timestamp
                        ))

                return retrieved_docs

            return []

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

    async def _text_search(
        self,
        query: str,
        session_id: str,
        config: RAGConfig
    ) -> List[RetrievedDocument]:
        """Perform text-based search"""
        try:
            # Implementation depends on available text search capabilities
            # This could use MongoDB full-text search, Elasticsearch, etc.

            if hasattr(self.memory_manager, 'long_term_memory'):
                # Use MongoDB text search as fallback
                collection = self.memory_manager.long_term_memory.collection

                # Build text search query
                search_results = await collection.find({
                    "session_id": session_id,
                    "$text": {"$search": query}
                }).sort([
                    ("score", {"$meta": "textScore"}),
                    ("importance_score", -1)
                ]).limit(config.max_retrieved_docs).to_list(length=None)

                retrieved_docs = []
                for doc in search_results:
                    relevance_score = doc.get("score", 0.5)  # From text search score

                    retrieved_docs.append(RetrievedDocument(
                        doc_id=doc["memory_id"],
                        content=doc["content"],
                        similarity_score=relevance_score,
                        relevance_score=relevance_score,
                        metadata={
                            "memory_type": doc.get("memory_type", "unknown"),
                            "importance": doc.get("importance_score", 0),
                            "timestamp": doc.get("timestamp", datetime.utcnow()).isoformat()
                        },
                        source="text_search",
                        timestamp=doc.get("timestamp", datetime.utcnow())
                    ))

                return retrieved_docs

            return []

        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return []

    async def _multi_vector_retrieval(
        self,
        request: ModelRequest,
        config: RAGConfig
    ) -> List[RetrievedDocument]:
        """Retrieval using multiple embedding models"""
        try:
            strategy_config = self.strategy_configs[RetrievalStrategy.MULTI_VECTOR]
            providers = strategy_config.get("embedding_providers", [])

            all_results = []
            session_id = request.session_id or "default"

            for provider in providers:
                try:
                    # Generate embedding with this provider
                    embedding = await self.vector_enhancer.generate_embedding(
                        request.prompt,
                        provider=provider
                    )

                    if embedding:
                        # Search using this embedding
                        # This would require extended memory system support
                        # For now, use standard vector search
                        results = await self._vector_search(request.prompt, session_id, config)
                        all_results.extend(results)

                except Exception as e:
                    self.logger.warning(f"Multi-vector retrieval failed for {provider}: {e}")

            # Remove duplicates and rank
            unique_results = {}
            for doc in all_results:
                if doc.doc_id not in unique_results:
                    unique_results[doc.doc_id] = doc
                else:
                    # Keep the one with higher relevance score
                    if doc.relevance_score > unique_results[doc.doc_id].relevance_score:
                        unique_results[doc.doc_id] = doc

            return list(unique_results.values())

        except Exception as e:
            self.logger.error(f"Multi-vector retrieval failed: {e}")
            return []

    async def _conversational_retrieval(
        self,
        request: ModelRequest,
        config: RAGConfig
    ) -> List[RetrievedDocument]:
        """Retrieval optimized for conversational context"""
        try:
            # Build enhanced query using conversation history
            query_parts = [request.prompt]

            # Add recent conversation context
            if request.conversation_history:
                recent_messages = request.conversation_history[-3:]  # Last 3 messages
                for msg in recent_messages:
                    content = msg.get("content", "")
                    if content and len(content) > 10:
                        query_parts.append(content)

            enhanced_query = " ".join(query_parts)

            # Use enhanced query for retrieval
            session_id = request.session_id or "default"
            results = await self._vector_search(enhanced_query, session_id, config)

            # Boost relevance for recent memories
            for doc in results:
                if doc.timestamp:
                    # Boost score for recent memories
                    time_diff = (datetime.utcnow() - doc.timestamp).total_seconds()
                    if time_diff < 3600:  # Within last hour
                        doc.relevance_score *= 1.2
                    elif time_diff < 86400:  # Within last day
                        doc.relevance_score *= 1.1

            return sorted(results, key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            self.logger.error(f"Conversational retrieval failed: {e}")
            return []

    async def _calculate_relevance_score(
        self,
        memory,
        query: str,
        session_id: str
    ) -> float:
        """Calculate relevance score for a memory"""
        try:
            base_score = 0.5

            # Factor 1: Importance score
            importance_weight = 0.3
            importance_score = memory.importance_score * importance_weight

            # Factor 2: Recency
            recency_weight = 0.2
            time_diff = (datetime.utcnow() - memory.timestamp).total_seconds()
            recency_score = max(0, 1 - (time_diff / 86400)) * recency_weight  # Decay over days

            # Factor 3: Memory type relevance
            type_weight = 0.2
            type_score = 0.8 if memory.memory_type.value in ["conversation", "insight"] else 0.5
            type_score *= type_weight

            # Factor 4: Emotional relevance
            emotion_weight = 0.1
            emotion_score = 0.0
            if memory.emotions:
                # Boost if emotions are present
                emotion_score = 0.7 * emotion_weight

            # Factor 5: Context tag matching
            context_weight = 0.2
            context_score = 0.0
            if memory.context_tags:
                query_words = set(query.lower().split())
                tag_words = set()
                for tag in memory.context_tags:
                    tag_words.update(tag.lower().split())

                if query_words & tag_words:  # Intersection
                    overlap_ratio = len(query_words & tag_words) / len(query_words)
                    context_score = overlap_ratio * context_weight

            total_score = base_score + importance_score + recency_score + type_score + emotion_score + context_score
            return min(1.0, total_score)

        except Exception as e:
            self.logger.warning(f"Relevance calculation failed: {e}")
            return 0.5

    async def _rank_fusion(
        self,
        documents: List[RetrievedDocument],
        vector_weight: float,
        text_weight: float
    ) -> List[RetrievedDocument]:
        """Combine rankings from multiple retrieval methods"""
        try:
            # Group documents by source
            vector_docs = [d for d in documents if "vector" in d.source or "memory" in d.source]
            text_docs = [d for d in documents if "text" in d.source]

            # Create combined ranking
            doc_scores = {}

            # Rank vector results
            for i, doc in enumerate(sorted(vector_docs, key=lambda x: x.similarity_score, reverse=True)):
                rank_score = 1.0 / (i + 1)  # Reciprocal rank
                doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0) + rank_score * vector_weight

            # Rank text results
            for i, doc in enumerate(sorted(text_docs, key=lambda x: x.relevance_score, reverse=True)):
                rank_score = 1.0 / (i + 1)
                doc_scores[doc.doc_id] = doc_scores.get(doc.doc_id, 0) + rank_score * text_weight

            # Create final ranking
            all_docs = {doc.doc_id: doc for doc in documents}
            ranked_docs = []

            for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
                if doc_id in all_docs:
                    doc = all_docs[doc_id]
                    doc.relevance_score = score  # Update with fused score
                    ranked_docs.append(doc)

            return ranked_docs

        except Exception as e:
            self.logger.error(f"Rank fusion failed: {e}")
            return documents

    async def _filter_documents(
        self,
        documents: List[RetrievedDocument],
        config: RAGConfig
    ) -> List[RetrievedDocument]:
        """Filter documents for relevance and diversity"""
        try:
            # Filter by relevance threshold
            relevant_docs = [
                doc for doc in documents
                if doc.relevance_score >= config.relevance_threshold
            ]

            # Apply diversity filtering
            if len(relevant_docs) <= config.max_retrieved_docs:
                return relevant_docs

            # Select diverse documents
            selected_docs = [relevant_docs[0]]  # Start with highest scored
            remaining_docs = relevant_docs[1:]

            while len(selected_docs) < config.max_retrieved_docs and remaining_docs:
                best_candidate = None
                best_min_similarity = -1

                for candidate in remaining_docs:
                    # Calculate minimum similarity to already selected docs
                    min_similarity = 1.0
                    for selected in selected_docs:
                        # Simple text similarity (could use embeddings)
                        similarity = await self._calculate_text_similarity(
                            candidate.content,
                            selected.content
                        )
                        min_similarity = min(min_similarity, similarity)

                    # Prefer documents that are different from selected ones
                    if min_similarity < config.diversity_threshold and min_similarity > best_min_similarity:
                        best_candidate = candidate
                        best_min_similarity = min_similarity

                if best_candidate:
                    selected_docs.append(best_candidate)
                    remaining_docs.remove(best_candidate)
                else:
                    # If no diverse candidate found, take next highest scored
                    selected_docs.append(remaining_docs[0])
                    remaining_docs = remaining_docs[1:]

            return selected_docs

        except Exception as e:
            self.logger.error(f"Document filtering failed: {e}")
            return documents[:config.max_retrieved_docs]

    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        try:
            # Simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.5

    async def _build_context(
        self,
        documents: List[RetrievedDocument],
        config: RAGConfig
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Build context text from retrieved documents"""
        try:
            context_parts = []
            citations = []
            total_tokens = 0

            for i, doc in enumerate(documents):
                # Estimate tokens
                doc_tokens = len(doc.content.split()) * 1.33

                if total_tokens + doc_tokens > config.max_context_tokens:
                    break

                # Add document to context
                citation_id = f"doc_{i+1}"
                context_parts.append(f"[{citation_id}] {doc.content}")

                # Track citation
                citations.append({
                    "id": citation_id,
                    "source": doc.source,
                    "relevance": str(round(doc.relevance_score, 3)),
                    "doc_id": doc.doc_id
                })

                total_tokens += doc_tokens

            context_text = "\n\n".join(context_parts)
            return context_text, citations

        except Exception as e:
            self.logger.error(f"Context building failed: {e}")
            return "", []

    async def _enhance_request_with_context(
        self,
        request: ModelRequest,
        context: RAGContext,
        config: RAGConfig
    ) -> ModelRequest:
        """Enhance request with retrieved context"""
        try:
            if not context.context_text:
                return request

            # Build enhanced prompt
            enhanced_prompt = f"""Based on the following context information, please answer the question.

Context:
{context.context_text}

Question: {request.prompt}

Please provide a comprehensive answer based on the context provided."""

            if config.include_citations:
                enhanced_prompt += " Include citations using the document references provided in the context."

            # Create enhanced request
            enhanced_request = ModelRequest(
                prompt=enhanced_prompt,
                context=request.context,
                system_prompt=request.system_prompt,
                conversation_history=request.conversation_history,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                task_type=request.task_type,
                required_capabilities=request.required_capabilities,
                session_id=request.session_id,
                user_id=request.user_id,
                priority=request.priority,
                use_rag=True,  # Mark as RAG request
                citation_required=config.include_citations
            )

            return enhanced_request

        except Exception as e:
            self.logger.error(f"Request enhancement failed: {e}")
            return request

    async def _add_citations(
        self,
        response: ModelResponse,
        context: RAGContext,
        config: RAGConfig
    ) -> ModelResponse:
        """Add citations to the response"""
        try:
            # Add citation information to response
            response.citations = context.citations
            response.retrieved_contexts = [
                {
                    "doc_id": doc.doc_id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "relevance_score": doc.relevance_score,
                    "source": doc.source
                }
                for doc in context.retrieved_docs
            ]

            response.context_relevance_scores = [
                doc.relevance_score for doc in context.retrieved_docs
            ]

            return response

        except Exception as e:
            self.logger.error(f"Citation addition failed: {e}")
            return response

    async def get_rag_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get RAG analytics for a session"""
        try:
            # This would track RAG usage and performance
            # Implementation depends on how analytics are stored
            return {
                "session_id": session_id,
                "rag_requests": 0,  # Would be tracked
                "avg_retrieval_time": 0.0,
                "avg_context_length": 0,
                "most_used_strategy": "hybrid_search",
                "retrieval_success_rate": 1.0
            }

        except Exception as e:
            self.logger.error(f"RAG analytics failed: {e}")
            return {"error": str(e)}