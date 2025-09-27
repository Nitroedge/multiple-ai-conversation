"""
Multi-Model AI Integration Package
Provides unified interface for multiple AI providers (Claude, GPT-4, Gemini, local models)
"""

from .model_interface import ModelInterface, ModelProvider, ModelRequest, ModelResponse
from .model_router import SmartModelRouter, TaskType, ModelCapability
from .providers import ClaudeProvider, GPT4Provider, GeminiProvider, LocalProvider
from .rag_orchestrator import RAGOrchestrator, RAGConfig
from .vector_enhancer import VectorEnhancer, EmbeddingProvider

__all__ = [
    # Core interfaces
    "ModelInterface",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",

    # Routing and orchestration
    "SmartModelRouter",
    "TaskType",
    "ModelCapability",

    # AI Providers
    "ClaudeProvider",
    "GPT4Provider",
    "GeminiProvider",
    "LocalProvider",

    # RAG and Vector Enhancement
    "RAGOrchestrator",
    "RAGConfig",
    "VectorEnhancer",
    "EmbeddingProvider"
]