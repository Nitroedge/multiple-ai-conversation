"""
Local Model Provider Implementation
Provides interface to locally hosted models (Ollama, Hugging Face Transformers, etc.)
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime

from ..model_interface import ModelInterface, ModelProvider, ModelRequest, ModelResponse, TaskType, ModelCapability

logger = logging.getLogger(__name__)


class LocalProvider(ModelInterface):
    """Local model provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelProvider.LOCAL, config)

        self.backend_type = config.get("backend_type", "ollama")  # ollama, transformers, llamacpp
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.default_model = config.get("default_model", "llama2")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 120)  # Local models can be slower

        # Model registry for different backends
        self.model_registry = {
            "ollama": {
                "models": ["llama2", "mistral", "codellama", "deepseek-coder", "neural-chat"],
                "api_endpoint": "/api/generate"
            },
            "transformers": {
                "models": ["microsoft/DialoGPT-medium", "facebook/blenderbot-400M-distill"],
                "api_endpoint": "/generate"
            },
            "llamacpp": {
                "models": ["llama-2-7b-chat", "llama-2-13b-chat", "code-llama-7b"],
                "api_endpoint": "/completion"
            }
        }

        # Model capabilities based on model type
        self.model_capabilities = {
            "llama2": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.COST_EFFICIENT
            ],
            "mistral": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.FAST_INFERENCE
            ],
            "codellama": [
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.TEXT_GENERATION,
                ModelCapability.FAST_INFERENCE
            ],
            "deepseek-coder": [
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.TEXT_GENERATION
            ]
        }

        # Local models have no API costs
        self.cost_per_token = 0.0

    async def validate_config(self) -> bool:
        """Validate local model configuration"""
        try:
            if self.backend_type == "ollama":
                return await self._validate_ollama()
            elif self.backend_type == "transformers":
                return await self._validate_transformers()
            elif self.backend_type == "llamacpp":
                return await self._validate_llamacpp()
            else:
                self.logger.error(f"Unsupported backend type: {self.backend_type}")
                return False

        except Exception as e:
            self.logger.error(f"Local model validation error: {e}")
            return False

    async def _validate_ollama(self) -> bool:
        """Validate Ollama setup"""
        try:
            async with httpx.AsyncClient() as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags", timeout=5)

                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [model["name"] for model in models_data.get("models", [])]

                    if available_models:
                        self.logger.info(f"Ollama validation successful. Models: {available_models}")
                        return True
                    else:
                        self.logger.warning("Ollama is running but no models are installed")
                        return False
                else:
                    self.logger.error(f"Ollama validation failed: {response.status_code}")
                    return False

        except Exception as e:
            self.logger.error(f"Ollama validation error: {e}")
            return False

    async def _validate_transformers(self) -> bool:
        """Validate Transformers setup"""
        try:
            # Check if transformers API is available
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200

        except Exception:
            self.logger.warning("Transformers API not available")
            return False

    async def _validate_llamacpp(self) -> bool:
        """Validate llama.cpp setup"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200

        except Exception:
            self.logger.warning("llama.cpp API not available")
            return False

    def get_capabilities(self) -> List[ModelCapability]:
        """Get local model capabilities"""
        all_capabilities = set()
        for capabilities in self.model_capabilities.values():
            all_capabilities.update(capabilities)

        # Add local-specific capabilities
        all_capabilities.update([
            ModelCapability.COST_EFFICIENT,
            ModelCapability.FAST_INFERENCE
        ])

        return list(all_capabilities)

    def get_available_models(self) -> List[str]:
        """Get available local models"""
        return self.model_registry.get(self.backend_type, {}).get("models", [])

    def estimate_cost(self, request: ModelRequest) -> float:
        """Estimate cost for local request (always 0)"""
        return 0.0  # Local models have no API costs

    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using local model"""
        start_time = time.time()

        try:
            # Select appropriate model
            model = self._select_model(request)

            # Generate response based on backend type
            if self.backend_type == "ollama":
                response_data = await self._generate_ollama_response(request, model)
            elif self.backend_type == "transformers":
                response_data = await self._generate_transformers_response(request, model)
            elif self.backend_type == "llamacpp":
                response_data = await self._generate_llamacpp_response(request, model)
            else:
                raise Exception(f"Unsupported backend: {self.backend_type}")

            response_time_ms = (time.time() - start_time) * 1000

            return ModelResponse(
                content=response_data.get("content", ""),
                provider=ModelProvider.LOCAL,
                model_name=f"{self.backend_type}:{model}",
                response_time_ms=response_time_ms,
                tokens_used=response_data.get("tokens_used"),
                cost_estimate=0.0,  # Local models are free
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Local model response generation failed: {e}")

            return ModelResponse(
                content="",
                provider=ModelProvider.LOCAL,
                model_name=f"{self.backend_type}:{self.default_model}",
                response_time_ms=response_time_ms,
                error=f"Local model error: {str(e)}"
            )

    def _select_model(self, request: ModelRequest) -> str:
        """Select appropriate local model based on request"""
        available_models = self.get_available_models()

        if not available_models:
            return self.default_model

        # Task-based model selection
        if request.task_type == TaskType.CODE_GENERATION:
            # Prefer code-specialized models
            code_models = ["codellama", "deepseek-coder"]
            for model in code_models:
                if model in available_models:
                    return model

        elif request.task_type == TaskType.CREATIVE:
            # Prefer general language models
            creative_models = ["llama2", "mistral"]
            for model in creative_models:
                if model in available_models:
                    return model

        # Capability-based selection
        if ModelCapability.FAST_INFERENCE in request.required_capabilities:
            fast_models = ["mistral", "codellama"]
            for model in fast_models:
                if model in available_models:
                    return model

        # Default to first available model
        return available_models[0] if available_models else self.default_model

    async def _generate_ollama_response(self, request: ModelRequest, model: str) -> Dict[str, Any]:
        """Generate response using Ollama"""
        # Build prompt for Ollama
        prompt = self._build_prompt(request)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "num_predict": request.max_tokens or 1000
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result.get("response", ""),
                    "tokens_used": result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
                }
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

    async def _generate_transformers_response(self, request: ModelRequest, model: str) -> Dict[str, Any]:
        """Generate response using Transformers"""
        prompt = self._build_prompt(request)

        payload = {
            "model": model,
            "prompt": prompt,
            "max_length": request.max_tokens or 1000,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result.get("generated_text", ""),
                    "tokens_used": result.get("tokens_used")
                }
            else:
                raise Exception(f"Transformers API error: {response.status_code}")

    async def _generate_llamacpp_response(self, request: ModelRequest, model: str) -> Dict[str, Any]:
        """Generate response using llama.cpp"""
        prompt = self._build_prompt(request)

        payload = {
            "prompt": prompt,
            "n_predict": request.max_tokens or 1000,
            "temperature": request.temperature or 0.7,
            "top_p": request.top_p or 0.9,
            "stream": False
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result.get("content", ""),
                    "tokens_used": result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0)
                }
            else:
                raise Exception(f"llama.cpp API error: {response.status_code}")

    def _build_prompt(self, request: ModelRequest) -> str:
        """Build prompt for local model"""
        prompt_parts = []

        # Add system prompt
        if request.system_prompt:
            prompt_parts.append(f"System: {request.system_prompt}")

        # Add conversation history
        for msg in request.conversation_history:
            role = msg.get("role", "user").title()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        # Add context
        if request.context:
            prompt_parts.append(f"Context: {request.context}")

        # Add current request
        prompt_parts.append(f"User: {request.prompt}")
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def supports_task_type(self, task_type: TaskType) -> bool:
        """Check if local models support specific task types"""
        # Local models generally support most text-based tasks
        # but may be weaker on specialized tasks
        strong_tasks = [
            TaskType.CONVERSATION,
            TaskType.CREATIVE,
            TaskType.SUMMARIZATION,
            TaskType.TRANSLATION
        ]

        moderate_tasks = [
            TaskType.TECHNICAL,
            TaskType.CODE_GENERATION,
            TaskType.ANALYSIS
        ]

        return task_type in strong_tasks or task_type in moderate_tasks

    async def preprocess_request(self, request: ModelRequest) -> ModelRequest:
        """Preprocess request for local model optimizations"""
        # Local models often benefit from more explicit instructions
        if request.task_type == TaskType.CODE_GENERATION:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are a helpful programming assistant. Provide clear, working code "
                    "with comments. Explain your approach when helpful."
                )

        elif request.task_type == TaskType.CREATIVE:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are a creative writing assistant. Write engaging, original content "
                    "that captures the reader's interest."
                )

        # Adjust max_tokens for local models (they may be slower)
        if request.max_tokens and request.max_tokens > 2000:
            request.max_tokens = 2000  # Cap for performance

        return request

    async def health_check(self) -> Dict[str, Any]:
        """Health check for local models"""
        try:
            start_time = time.time()

            if self.backend_type == "ollama":
                result = await self._health_check_ollama()
            elif self.backend_type == "transformers":
                result = await self._health_check_transformers()
            elif self.backend_type == "llamacpp":
                result = await self._health_check_llamacpp()
            else:
                raise Exception(f"Unknown backend: {self.backend_type}")

            response_time = (time.time() - start_time) * 1000

            return {
                "provider": self.provider.value,
                "backend": self.backend_type,
                "status": "healthy" if result else "unhealthy",
                "response_time_ms": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "provider": self.provider.value,
                "backend": self.backend_type,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _health_check_ollama(self) -> bool:
        """Health check for Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5)
                return response.status_code == 200
        except Exception:
            return False

    async def _health_check_transformers(self) -> bool:
        """Health check for Transformers"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200
        except Exception:
            return False

    async def _health_check_llamacpp(self) -> bool:
        """Health check for llama.cpp"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200
        except Exception:
            return False