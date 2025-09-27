"""
GPT-4 Provider Implementation
Enhanced version of the existing OpenAI integration with unified interface
"""

import asyncio
import logging
import time
import tiktoken
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

from ..model_interface import ModelInterface, ModelProvider, ModelRequest, ModelResponse, TaskType, ModelCapability

logger = logging.getLogger(__name__)


class GPT4Provider(ModelInterface):
    """GPT-4 provider implementation with enhanced features"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelProvider.GPT4, config)

        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.organization = config.get("organization")
        self.default_model = config.get("default_model", "gpt-4-turbo-preview")
        self.max_retries = config.get("max_retries", 3)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization
        )

        # Available models
        self.available_models = [
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]

        # Model capabilities
        self.model_capabilities = {
            "gpt-4-turbo-preview": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.VISION,
                ModelCapability.LONG_CONTEXT
            ],
            "gpt-4": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.FUNCTION_CALLING
            ],
            "gpt-3.5-turbo": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.FAST_INFERENCE,
                ModelCapability.COST_EFFICIENT
            ]
        }

        # Cost per token (approximate)
        self.cost_per_input_token = {
            "gpt-4-turbo-preview": 0.00001,
            "gpt-4": 0.00003,
            "gpt-4-32k": 0.00006,
            "gpt-3.5-turbo": 0.0000015,
            "gpt-3.5-turbo-16k": 0.000003
        }
        self.cost_per_output_token = {
            "gpt-4-turbo-preview": 0.00003,
            "gpt-4": 0.00006,
            "gpt-4-32k": 0.00012,
            "gpt-3.5-turbo": 0.000002,
            "gpt-3.5-turbo-16k": 0.000004
        }

        # Token encoders for cost calculation
        self.encoders = {}

    async def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        if not self.api_key:
            self.logger.error("OpenAI API key not provided")
            return False

        try:
            # Test API connection
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )

            if response and response.choices:
                self.logger.info("OpenAI API validation successful")
                return True
            else:
                self.logger.error("OpenAI API validation failed")
                return False

        except Exception as e:
            self.logger.error(f"OpenAI API validation error: {e}")
            return False

    def get_capabilities(self) -> List[ModelCapability]:
        """Get GPT-4 capabilities"""
        all_capabilities = set()
        for capabilities in self.model_capabilities.values():
            all_capabilities.update(capabilities)
        return list(all_capabilities)

    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return self.available_models.copy()

    def estimate_cost(self, request: ModelRequest) -> float:
        """Estimate cost for OpenAI request"""
        try:
            model = self._select_model(request)

            # Calculate input tokens
            input_text = self._build_input_text(request)
            input_tokens = self._count_tokens(input_text, model)

            # Estimate output tokens
            output_tokens = request.max_tokens or 1000

            input_cost = input_tokens * self.cost_per_input_token.get(model, 0.00001)
            output_cost = output_tokens * self.cost_per_output_token.get(model, 0.00003)

            return input_cost + output_cost

        except Exception as e:
            self.logger.warning(f"Cost estimation failed: {e}")
            return 0.01

    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using OpenAI"""
        start_time = time.time()

        try:
            # Select appropriate model
            model = self._select_model(request)

            # Build OpenAI messages
            messages = self._build_openai_messages(request)

            # Prepare completion parameters
            completion_params = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens or 4000,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9
            }

            # Add streaming if requested
            if request.stream:
                completion_params["stream"] = True

            # Make API request
            response = await self.client.chat.completions.create(**completion_params)

            # Handle streaming vs non-streaming
            if request.stream:
                content = await self._handle_streaming_response(response)
            else:
                content = response.choices[0].message.content

            response_time_ms = (time.time() - start_time) * 1000

            return ModelResponse(
                content=content,
                provider=ModelProvider.GPT4,
                model_name=model,
                response_time_ms=response_time_ms,
                tokens_used=getattr(response, 'usage', {}).get('total_tokens', 0),
                cost_estimate=self._calculate_actual_cost(response, model),
                request_id=getattr(response, 'id', None),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"OpenAI response generation failed: {e}")

            return ModelResponse(
                content="",
                provider=ModelProvider.GPT4,
                model_name=self.default_model,
                response_time_ms=response_time_ms,
                error=f"OpenAI API error: {str(e)}"
            )

    def _select_model(self, request: ModelRequest) -> str:
        """Select appropriate OpenAI model based on request"""
        # Task-based model selection
        if request.task_type == TaskType.CREATIVE:
            return "gpt-4-turbo-preview"  # Best for creative tasks
        elif request.task_type == TaskType.CODE_GENERATION:
            return "gpt-4"  # Excellent for code
        elif request.task_type == TaskType.MATH:
            return "gpt-4"  # Best mathematical reasoning
        elif request.task_type == TaskType.SUMMARIZATION:
            return "gpt-3.5-turbo"  # Fast and cost-effective

        # Capability-based selection
        if ModelCapability.FAST_INFERENCE in request.required_capabilities:
            return "gpt-3.5-turbo"
        elif ModelCapability.LONG_CONTEXT in request.required_capabilities:
            return "gpt-4-turbo-preview"
        elif ModelCapability.VISION in request.required_capabilities:
            return "gpt-4-turbo-preview"

        # Default based on complexity
        input_length = len(self._build_input_text(request))
        if input_length > 8000:  # Long context
            return "gpt-4-turbo-preview"
        elif input_length > 4000:  # Medium context
            return "gpt-4"
        else:  # Short context
            return self.default_model

    def _build_openai_messages(self, request: ModelRequest) -> List[Dict[str, str]]:
        """Build OpenAI messages format"""
        messages = []

        # Add system prompt
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })

        # Add conversation history
        for msg in request.conversation_history:
            role = msg.get("role", "user")
            if role not in ["system", "user", "assistant"]:
                role = "user"
            messages.append({
                "role": role,
                "content": msg.get("content", "")
            })

        # Add current request
        content = request.prompt
        if request.context:
            content = f"Context: {request.context}\n\nRequest: {content}"

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    async def _handle_streaming_response(self, response) -> str:
        """Handle streaming response from OpenAI"""
        content_parts = []
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                content = getattr(chunk.choices[0].delta, 'content', '')
                if content:
                    content_parts.append(content)

        return ''.join(content_parts)

    def _build_input_text(self, request: ModelRequest) -> str:
        """Build complete input text for token counting"""
        text_parts = []

        if request.system_prompt:
            text_parts.append(request.system_prompt)

        for msg in request.conversation_history:
            text_parts.append(msg.get("content", ""))

        if request.context:
            text_parts.append(request.context)

        text_parts.append(request.prompt)

        return " ".join(text_parts)

    def _count_tokens(self, text: str, model: str) -> int:
        """Count tokens for cost estimation"""
        try:
            # Get or create encoder for model
            if model not in self.encoders:
                try:
                    self.encoders[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback to cl100k_base for newer models
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")

            return len(self.encoders[model].encode(text))

        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}")
            # Rough approximation: 1 token ≈ 0.75 words
            return int(len(text.split()) * 1.33)

    def _calculate_actual_cost(self, response, model: str) -> float:
        """Calculate actual cost based on token usage"""
        try:
            usage = getattr(response, 'usage', None)
            if not usage:
                return 0.0

            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)

            input_cost = input_tokens * self.cost_per_input_token.get(model, 0.00001)
            output_cost = output_tokens * self.cost_per_output_token.get(model, 0.00003)

            return input_cost + output_cost

        except Exception:
            return 0.0

    def supports_task_type(self, task_type: TaskType) -> bool:
        """Check if GPT-4 supports specific task types"""
        # GPT-4 supports all task types well
        return True

    async def preprocess_request(self, request: ModelRequest) -> ModelRequest:
        """Preprocess request for OpenAI-specific optimizations"""
        # Add specific system prompts for different task types
        if request.task_type == TaskType.CODE_GENERATION and not request.system_prompt:
            request.system_prompt = (
                "You are an expert software engineer. Provide clean, efficient, and well-documented code. "
                "Include comments explaining the logic and follow best practices for the given language."
            )

        elif request.task_type == TaskType.MATH and not request.system_prompt:
            request.system_prompt = (
                "You are a mathematics expert. Show your work step-by-step and explain your reasoning clearly. "
                "Double-check your calculations and provide accurate solutions."
            )

        elif request.task_type == TaskType.ANALYSIS and not request.system_prompt:
            request.system_prompt = (
                "You are a thoughtful analyst. Provide comprehensive analysis with clear structure, "
                "evidence-based conclusions, and actionable insights."
            )

        return request

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check for OpenAI"""
        try:
            start_time = time.time()

            # Test multiple aspects
            test_response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use fastest model for health check
                messages=[{"role": "user", "content": "Health check"}],
                max_tokens=5
            )

            response_time = (time.time() - start_time) * 1000

            # Check if we got a valid response
            is_healthy = (
                test_response and
                test_response.choices and
                test_response.choices[0].message.content
            )

            return {
                "provider": self.provider.value,
                "status": "healthy" if is_healthy else "unhealthy",
                "response_time_ms": response_time,
                "test_successful": is_healthy,
                "model_tested": "gpt-3.5-turbo",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "provider": self.provider.value,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }