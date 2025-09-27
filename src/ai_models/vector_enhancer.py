"""
Vector Memory Enhancement System
Provides advanced vector operations with multi-model embedding support
"""

import asyncio
import logging
import numpy as np
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import httpx

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Available embedding providers"""
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    LOCAL = "local"


class EmbeddingModel(BaseModel):
    """Configuration for embedding models"""
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    max_input_tokens: int
    cost_per_token: float = 0.0
    capabilities: List[str] = Field(default_factory=list)


class VectorSearchResult(BaseModel):
    """Result from vector search"""
    memory_id: str
    content: str
    similarity_score: float
    embedding_provider: EmbeddingProvider
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HybridSearchResult(BaseModel):
    """Result from hybrid search (vector + text)"""
    memory_id: str
    content: str
    vector_score: float
    text_score: float
    combined_score: float
    rank_fusion_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorEnhancer:
    """
    Enhanced vector operations with multi-model embedding support
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Available embedding models
        self.embedding_models = {
            EmbeddingProvider.OPENAI: {
                "text-embedding-ada-002": EmbeddingModel(
                    provider=EmbeddingProvider.OPENAI,
                    model_name="text-embedding-ada-002",
                    dimensions=1536,
                    max_input_tokens=8191,
                    cost_per_token=0.0000001,
                    capabilities=["general", "semantic", "multilingual"]
                ),
                "text-embedding-3-small": EmbeddingModel(
                    provider=EmbeddingProvider.OPENAI,
                    model_name="text-embedding-3-small",
                    dimensions=1536,
                    max_input_tokens=8191,
                    cost_per_token=0.00000002,
                    capabilities=["general", "semantic", "multilingual", "cost_efficient"]
                ),
                "text-embedding-3-large": EmbeddingModel(
                    provider=EmbeddingProvider.OPENAI,
                    model_name="text-embedding-3-large",
                    dimensions=3072,
                    max_input_tokens=8191,
                    cost_per_token=0.00000013,
                    capabilities=["general", "semantic", "multilingual", "high_quality"]
                )
            },
            EmbeddingProvider.COHERE: {
                "embed-english-v3.0": EmbeddingModel(
                    provider=EmbeddingProvider.COHERE,
                    model_name="embed-english-v3.0",
                    dimensions=1024,
                    max_input_tokens=512,
                    cost_per_token=0.0000001,
                    capabilities=["general", "semantic", "classification"]
                ),
                "embed-multilingual-v3.0": EmbeddingModel(
                    provider=EmbeddingProvider.COHERE,
                    model_name="embed-multilingual-v3.0",
                    dimensions=1024,
                    max_input_tokens=512,
                    cost_per_token=0.0000001,
                    capabilities=["multilingual", "semantic", "classification"]
                )
            },
            EmbeddingProvider.SENTENCE_TRANSFORMERS: {
                "all-MiniLM-L6-v2": EmbeddingModel(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                    model_name="all-MiniLM-L6-v2",
                    dimensions=384,
                    max_input_tokens=256,
                    cost_per_token=0.0,
                    capabilities=["general", "fast", "local"]
                ),
                "all-mpnet-base-v2": EmbeddingModel(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                    model_name="all-mpnet-base-v2",
                    dimensions=768,
                    max_input_tokens=384,
                    cost_per_token=0.0,
                    capabilities=["general", "high_quality", "local"]
                ),
                "multi-qa-MiniLM-L6-cos-v1": EmbeddingModel(
                    provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
                    model_name="multi-qa-MiniLM-L6-cos-v1",
                    dimensions=384,
                    max_input_tokens=512,
                    cost_per_token=0.0,
                    capabilities=["question_answering", "semantic_search", "local"]
                )
            }
        }

        # Initialize providers
        self.openai_client = None
        self.cohere_client = None
        self.local_models = {}

        # Search configuration
        self.default_provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
        self.fallback_providers = [
            EmbeddingProvider.SENTENCE_TRANSFORMERS,
            EmbeddingProvider.OPENAI
        ]

        # Hybrid search weights
        self.hybrid_weights = {
            "vector": 0.7,
            "text": 0.3
        }

    async def initialize(self):
        """Initialize embedding providers"""
        try:
            # Initialize OpenAI
            openai_key = self.config.get("openai_api_key")
            if openai_key:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=openai_key)

            # Initialize Cohere
            cohere_key = self.config.get("cohere_api_key")
            if cohere_key:
                self.cohere_client = httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {cohere_key}"}
                )

            # Initialize local models
            await self._initialize_local_models()

            self.logger.info("Vector enhancer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector enhancer: {e}")
            raise

    async def _initialize_local_models(self):
        """Initialize local sentence transformer models"""
        try:
            # Load commonly used models
            models_to_load = [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2"
            ]

            for model_name in models_to_load:
                try:
                    model = await asyncio.get_event_loop().run_in_executor(
                        None, SentenceTransformer, model_name
                    )
                    self.local_models[model_name] = model
                    self.logger.info(f"Loaded local model: {model_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to load local model {model_name}: {e}")

        except Exception as e:
            self.logger.error(f"Error initializing local models: {e}")

    async def generate_embedding(
        self,
        text: str,
        provider: Optional[EmbeddingProvider] = None,
        model_name: Optional[str] = None
    ) -> Optional[List[float]]:
        """Generate embedding using specified or best available provider"""
        try:
            provider = provider or self.default_provider

            if provider == EmbeddingProvider.OPENAI:
                return await self._generate_openai_embedding(text, model_name)
            elif provider == EmbeddingProvider.COHERE:
                return await self._generate_cohere_embedding(text, model_name)
            elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                return await self._generate_local_embedding(text, model_name)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            self.logger.error(f"Embedding generation failed for {provider}: {e}")

            # Try fallback providers
            for fallback_provider in self.fallback_providers:
                if fallback_provider != provider:
                    try:
                        return await self.generate_embedding(text, fallback_provider)
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback provider {fallback_provider} failed: {fallback_error}")

            return None

    async def _generate_openai_embedding(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Optional[List[float]]:
        """Generate embedding using OpenAI"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")

        model_name = model_name or "text-embedding-ada-002"

        try:
            response = await self.openai_client.embeddings.create(
                model=model_name,
                input=text.replace("\n", " ")
            )

            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            raise

    async def _generate_cohere_embedding(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Optional[List[float]]:
        """Generate embedding using Cohere"""
        if not self.cohere_client:
            raise Exception("Cohere client not initialized")

        model_name = model_name or "embed-english-v3.0"

        try:
            payload = {
                "model": model_name,
                "texts": [text],
                "input_type": "search_document"
            }

            response = await self.cohere_client.post(
                "https://api.cohere.ai/v1/embed",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return result["embeddings"][0]
            else:
                raise Exception(f"Cohere API error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Cohere embedding generation failed: {e}")
            raise

    async def _generate_local_embedding(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Optional[List[float]]:
        """Generate embedding using local sentence transformer"""
        model_name = model_name or "all-MiniLM-L6-v2"

        # Load model if not cached
        if model_name not in self.local_models:
            try:
                model = await asyncio.get_event_loop().run_in_executor(
                    None, SentenceTransformer, model_name
                )
                self.local_models[model_name] = model
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                raise

        try:
            model = self.local_models[model_name]
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, model.encode, text
            )
            return embedding.tolist()

        except Exception as e:
            self.logger.error(f"Local embedding generation failed: {e}")
            raise

    async def batch_generate_embeddings(
        self,
        texts: List[str],
        provider: Optional[EmbeddingProvider] = None,
        model_name: Optional[str] = None
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently"""
        try:
            provider = provider or self.default_provider

            if provider == EmbeddingProvider.OPENAI:
                return await self._batch_openai_embeddings(texts, model_name)
            elif provider == EmbeddingProvider.COHERE:
                return await self._batch_cohere_embeddings(texts, model_name)
            elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                return await self._batch_local_embeddings(texts, model_name)
            else:
                # Fallback to individual generation
                embeddings = []
                for text in texts:
                    embedding = await self.generate_embedding(text, provider, model_name)
                    embeddings.append(embedding)
                return embeddings

        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)

    async def _batch_openai_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None
    ) -> List[Optional[List[float]]]:
        """Batch generate OpenAI embeddings"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")

        model_name = model_name or "text-embedding-ada-002"

        try:
            # Clean texts
            cleaned_texts = [text.replace("\n", " ") for text in texts]

            response = await self.openai_client.embeddings.create(
                model=model_name,
                input=cleaned_texts
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            self.logger.error(f"Batch OpenAI embedding generation failed: {e}")
            raise

    async def _batch_cohere_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None
    ) -> List[Optional[List[float]]]:
        """Batch generate Cohere embeddings"""
        if not self.cohere_client:
            raise Exception("Cohere client not initialized")

        model_name = model_name or "embed-english-v3.0"

        try:
            payload = {
                "model": model_name,
                "texts": texts,
                "input_type": "search_document"
            }

            response = await self.cohere_client.post(
                "https://api.cohere.ai/v1/embed",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return result["embeddings"]
            else:
                raise Exception(f"Cohere API error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Batch Cohere embedding generation failed: {e}")
            raise

    async def _batch_local_embeddings(
        self,
        texts: List[str],
        model_name: Optional[str] = None
    ) -> List[Optional[List[float]]]:
        """Batch generate local embeddings"""
        model_name = model_name or "all-MiniLM-L6-v2"

        # Load model if not cached
        if model_name not in self.local_models:
            try:
                model = await asyncio.get_event_loop().run_in_executor(
                    None, SentenceTransformer, model_name
                )
                self.local_models[model_name] = model
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                raise

        try:
            model = self.local_models[model_name]
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, model.encode, texts
            )
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            self.logger.error(f"Batch local embedding generation failed: {e}")
            raise

    async def calculate_similarity_matrix(
        self,
        embeddings: List[List[float]]
    ) -> np.ndarray:
        """Calculate similarity matrix for embeddings"""
        try:
            embedding_matrix = np.array(embeddings)
            # Cosine similarity matrix
            similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T) / (
                np.linalg.norm(embedding_matrix, axis=1, keepdims=True) *
                np.linalg.norm(embedding_matrix, axis=1, keepdims=True).T
            )
            return similarity_matrix

        except Exception as e:
            self.logger.error(f"Similarity matrix calculation failed: {e}")
            return np.array([])

    async def find_semantic_clusters(
        self,
        embeddings: List[List[float]],
        threshold: float = 0.8,
        min_cluster_size: int = 2
    ) -> List[List[int]]:
        """Find semantic clusters in embeddings"""
        try:
            if len(embeddings) < 2:
                return []

            similarity_matrix = await self.calculate_similarity_matrix(embeddings)
            clusters = []
            visited = set()

            for i in range(len(embeddings)):
                if i in visited:
                    continue

                cluster = [i]
                visited.add(i)

                for j in range(i + 1, len(embeddings)):
                    if j not in visited and similarity_matrix[i][j] >= threshold:
                        cluster.append(j)
                        visited.add(j)

                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

            return clusters

        except Exception as e:
            self.logger.error(f"Semantic clustering failed: {e}")
            return []

    def get_best_embedding_model(
        self,
        task_type: str,
        text_length: int,
        multilingual: bool = False
    ) -> Tuple[EmbeddingProvider, str]:
        """Select best embedding model for specific requirements"""
        try:
            # Task-specific model selection
            if task_type == "question_answering":
                if "multi-qa-MiniLM-L6-cos-v1" in self.local_models:
                    return EmbeddingProvider.SENTENCE_TRANSFORMERS, "multi-qa-MiniLM-L6-cos-v1"

            # Multilingual requirements
            if multilingual:
                if self.cohere_client:
                    return EmbeddingProvider.COHERE, "embed-multilingual-v3.0"
                elif self.openai_client:
                    return EmbeddingProvider.OPENAI, "text-embedding-ada-002"

            # High quality requirements
            if text_length > 1000:
                if self.openai_client:
                    return EmbeddingProvider.OPENAI, "text-embedding-3-large"
                elif "all-mpnet-base-v2" in self.local_models:
                    return EmbeddingProvider.SENTENCE_TRANSFORMERS, "all-mpnet-base-v2"

            # Fast/cost-efficient requirements
            if text_length < 500:
                if "all-MiniLM-L6-v2" in self.local_models:
                    return EmbeddingProvider.SENTENCE_TRANSFORMERS, "all-MiniLM-L6-v2"
                elif self.openai_client:
                    return EmbeddingProvider.OPENAI, "text-embedding-3-small"

            # Default fallback
            return self.default_provider, "all-MiniLM-L6-v2"

        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return self.default_provider, "all-MiniLM-L6-v2"

    async def get_provider_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of all embedding providers"""
        capabilities = {}

        for provider, models in self.embedding_models.items():
            provider_capabilities = {
                "available": False,
                "models": {},
                "total_models": len(models)
            }

            # Check availability
            if provider == EmbeddingProvider.OPENAI:
                provider_capabilities["available"] = self.openai_client is not None
            elif provider == EmbeddingProvider.COHERE:
                provider_capabilities["available"] = self.cohere_client is not None
            elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                provider_capabilities["available"] = len(self.local_models) > 0

            # Add model details
            for model_name, model_config in models.items():
                provider_capabilities["models"][model_name] = {
                    "dimensions": model_config.dimensions,
                    "max_tokens": model_config.max_input_tokens,
                    "cost_per_token": model_config.cost_per_token,
                    "capabilities": model_config.capabilities
                }

            capabilities[provider.value] = provider_capabilities

        return capabilities