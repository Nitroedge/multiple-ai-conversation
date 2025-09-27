"""
Unified AI Model Interface
Provides a consistent interface for interacting with different AI providers
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported AI model providers"""
    CLAUDE = "claude"
    GPT4 = "gpt4"
    GEMINI = "gemini"
    LOCAL = "local"
    AZURE_OPENAI = "azure_openai"


class TaskType(str, Enum):
    """Types of tasks for model routing"""
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "question_answering"
    REASONING = "reasoning"
    MATH = "math"


class ModelCapability(str, Enum):
    """Model capabilities for task matching"""
    TEXT_GENERATION = "text_generation"
    CODE_UNDERSTANDING = "code_understanding"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    CREATIVE_WRITING = "creative_writing"
    MULTILINGUAL = "multilingual"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"
    COST_EFFICIENT = "cost_efficient"


class ModelRequest(BaseModel):
    """Unified request format for all AI models"""
    prompt: str = Field(..., description="Main prompt/question")
    context: Optional[str] = Field(None, description="Additional context")
    system_prompt: Optional[str] = Field(None, description="System prompt/instructions")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

    # Model preferences
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")

    # Task information
    task_type: Optional[TaskType] = Field(None, description="Type of task")
    required_capabilities: Optional[List[ModelCapability]] = Field(default_factory=list)

    # Metadata
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    priority: Optional[int] = Field(5, description="Request priority (1-10)")

    # Advanced options
    use_rag: bool = Field(False, description="Use retrieval-augmented generation")
    stream: bool = Field(False, description="Stream response")
    citation_required: bool = Field(False, description="Include citations in response")


class ModelResponse(BaseModel):
    """Unified response format from all AI models"""
    content: str = Field(..., description="Generated content")
    provider: ModelProvider = Field(..., description="Provider that generated the response")
    model_name: str = Field(..., description="Specific model used")

    # Performance metrics
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")

    # Quality indicators
    confidence_score: Optional[float] = Field(None, description="Model confidence (0-1)")
    safety_score: Optional[float] = Field(None, description="Safety assessment (0-1)")

    # Additional metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    # RAG information
    retrieved_contexts: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    context_relevance_scores: Optional[List[float]] = Field(default_factory=list)

    # Error handling
    error: Optional[str] = Field(None, description="Error message if any")
    warning: Optional[str] = Field(None, description="Warning message if any")


class ModelInterface(ABC):
    """Abstract base class for all AI model providers"""

    def __init__(self, provider: ModelProvider, config: Dict[str, Any]):
        self.provider = provider
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{provider.value}")
        self._performance_metrics = {
            "total_requests": 0,
            "total_response_time": 0.0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "error_count": 0
        }

    @abstractmethod
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for the given request"""
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate the provider configuration"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """Get the capabilities of this model provider"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass

    @abstractmethod
    def estimate_cost(self, request: ModelRequest) -> float:
        """Estimate the cost for processing this request"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the provider"""
        try:
            start_time = time.time()
            test_request = ModelRequest(
                prompt="Test health check",
                max_tokens=10,
                temperature=0.1
            )

            response = await self.generate_response(test_request)
            response_time = (time.time() - start_time) * 1000

            return {
                "provider": self.provider.value,
                "status": "healthy",
                "response_time_ms": response_time,
                "test_successful": response.error is None,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Health check failed for {self.provider.value}: {e}")
            return {
                "provider": self.provider.value,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def update_metrics(self, response: ModelResponse):
        """Update performance metrics"""
        self._performance_metrics["total_requests"] += 1
        self._performance_metrics["total_response_time"] += response.response_time_ms

        if response.tokens_used:
            self._performance_metrics["total_tokens"] += response.tokens_used

        if response.cost_estimate:
            self._performance_metrics["total_cost"] += response.cost_estimate

        if response.error:
            self._performance_metrics["error_count"] += 1

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this provider"""
        metrics = self._performance_metrics.copy()

        if metrics["total_requests"] > 0:
            metrics["avg_response_time"] = metrics["total_response_time"] / metrics["total_requests"]
            metrics["error_rate"] = metrics["error_count"] / metrics["total_requests"]
            metrics["avg_tokens_per_request"] = metrics["total_tokens"] / metrics["total_requests"]
            metrics["avg_cost_per_request"] = metrics["total_cost"] / metrics["total_requests"]
        else:
            metrics.update({
                "avg_response_time": 0.0,
                "error_rate": 0.0,
                "avg_tokens_per_request": 0.0,
                "avg_cost_per_request": 0.0
            })

        return metrics

    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if this provider supports a specific capability"""
        return capability in self.get_capabilities()

    def supports_task_type(self, task_type: TaskType) -> bool:
        """Check if this provider is suitable for a specific task type"""
        # Default implementation - can be overridden by specific providers
        task_capability_mapping = {
            TaskType.CONVERSATION: [ModelCapability.TEXT_GENERATION],
            TaskType.ANALYSIS: [ModelCapability.TEXT_GENERATION],
            TaskType.CREATIVE: [ModelCapability.CREATIVE_WRITING],
            TaskType.TECHNICAL: [ModelCapability.CODE_UNDERSTANDING],
            TaskType.MATH: [ModelCapability.MATHEMATICAL_REASONING],
            TaskType.CODE_GENERATION: [ModelCapability.CODE_UNDERSTANDING],
            TaskType.REASONING: [ModelCapability.TEXT_GENERATION],
        }

        required_capabilities = task_capability_mapping.get(task_type, [])
        return all(self.supports_capability(cap) for cap in required_capabilities)

    async def preprocess_request(self, request: ModelRequest) -> ModelRequest:
        """Preprocess request before sending to the model"""
        # Default implementation - can be overridden
        return request

    async def postprocess_response(self, response: ModelResponse) -> ModelResponse:
        """Postprocess response before returning"""
        # Default implementation - can be overridden
        self.update_metrics(response)
        return response


class ModelManager:
    """Manages multiple model providers"""

    def __init__(self):
        self.providers: Dict[ModelProvider, ModelInterface] = {}
        self.logger = logging.getLogger(__name__)

    def register_provider(self, provider: ModelInterface):
        """Register a model provider"""
        self.providers[provider.provider] = provider
        self.logger.info(f"Registered provider: {provider.provider.value}")

    def get_provider(self, provider_type: ModelProvider) -> Optional[ModelInterface]:
        """Get a specific provider"""
        return self.providers.get(provider_type)

    def get_available_providers(self) -> List[ModelProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())

    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        results = {}

        for provider_type, provider in self.providers.items():
            try:
                results[provider_type.value] = await provider.health_check()
            except Exception as e:
                results[provider_type.value] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }

        return results

    def get_all_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all providers"""
        metrics = {}

        for provider_type, provider in self.providers.items():
            metrics[provider_type.value] = provider.get_performance_metrics()

        return metrics

    def find_providers_for_task(self, task_type: TaskType) -> List[ModelProvider]:
        """Find providers that support a specific task type"""
        suitable_providers = []

        for provider_type, provider in self.providers.items():
            if provider.supports_task_type(task_type):
                suitable_providers.append(provider_type)

        return suitable_providers

    def find_providers_with_capabilities(self, capabilities: List[ModelCapability]) -> List[ModelProvider]:
        """Find providers that have all specified capabilities"""
        suitable_providers = []

        for provider_type, provider in self.providers.items():
            if all(provider.supports_capability(cap) for cap in capabilities):
                suitable_providers.append(provider_type)

        return suitable_providers