"""
Claude AI Provider Implementation
Provides interface to Anthropic's Claude models
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


class ClaudeProvider(ModelInterface):
    """Claude AI provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelProvider.CLAUDE, config)

        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.default_model = config.get("default_model", "claude-3-sonnet-20240229")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)

        # Claude-specific configuration
        self.available_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]

        # Model capabilities
        self.model_capabilities = {
            "claude-3-opus-20240229": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.VISION
            ],
            "claude-3-sonnet-20240229": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.VISION
            ],
            "claude-3-haiku-20240307": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.FAST_INFERENCE,
                ModelCapability.COST_EFFICIENT,
                ModelCapability.MULTILINGUAL
            ]
        }

        # Cost per token (approximate)
        self.cost_per_input_token = {
            "claude-3-opus-20240229": 0.000015,
            "claude-3-sonnet-20240229": 0.000003,
            "claude-3-haiku-20240307": 0.00000025
        }
        self.cost_per_output_token = {
            "claude-3-opus-20240229": 0.000075,
            "claude-3-sonnet-20240229": 0.000015,
            "claude-3-haiku-20240307": 0.00000125
        }

    async def validate_config(self) -> bool:
        """Validate Claude configuration"""
        if not self.api_key:
            self.logger.error("Claude API key not provided")
            return False

        try:
            # Test API connection with a simple request
            async with httpx.AsyncClient() as client:
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }

                test_payload = {
                    "model": self.default_model,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Hi"}]
                }

                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=test_payload,
                    timeout=10
                )

                if response.status_code == 200:
                    self.logger.info("Claude API validation successful")
                    return True
                else:
                    self.logger.error(f"Claude API validation failed: {response.status_code}")
                    return False

        except Exception as e:
            self.logger.error(f"Claude API validation error: {e}")
            return False

    def get_capabilities(self) -> List[ModelCapability]:
        """Get Claude capabilities"""
        # Return union of all model capabilities
        all_capabilities = set()
        for capabilities in self.model_capabilities.values():
            all_capabilities.update(capabilities)
        return list(all_capabilities)

    def get_available_models(self) -> List[str]:
        """Get available Claude models"""
        return self.available_models.copy()

    def estimate_cost(self, request: ModelRequest) -> float:
        """Estimate cost for Claude request"""
        try:
            model = self._select_model(request)

            # Estimate input tokens (rough approximation: 1 token ≈ 0.75 words)
            input_text = self._build_input_text(request)
            estimated_input_tokens = len(input_text.split()) * 1.33

            # Estimate output tokens
            estimated_output_tokens = request.max_tokens or 1000

            input_cost = estimated_input_tokens * self.cost_per_input_token.get(model, 0.000003)
            output_cost = estimated_output_tokens * self.cost_per_output_token.get(model, 0.000015)

            return input_cost + output_cost

        except Exception as e:
            self.logger.warning(f"Cost estimation failed: {e}")
            return 0.01  # Default estimate

    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Claude"""
        start_time = time.time()

        try:
            # Select appropriate model
            model = self._select_model(request)

            # Build Claude-specific payload
            payload = self._build_claude_payload(request, model)

            # Make API request with retries
            response_data = await self._make_api_request(payload)

            # Parse response
            response_time_ms = (time.time() - start_time) * 1000

            return ModelResponse(
                content=response_data.get("content", [{}])[0].get("text", ""),
                provider=ModelProvider.CLAUDE,
                model_name=model,
                response_time_ms=response_time_ms,
                tokens_used=response_data.get("usage", {}).get("input_tokens", 0) +
                           response_data.get("usage", {}).get("output_tokens", 0),
                cost_estimate=self._calculate_actual_cost(response_data, model),
                request_id=response_data.get("id"),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Claude response generation failed: {e}")

            return ModelResponse(
                content="",
                provider=ModelProvider.CLAUDE,
                model_name=self.default_model,
                response_time_ms=response_time_ms,
                error=f"Claude API error: {str(e)}"
            )

    def _select_model(self, request: ModelRequest) -> str:
        """Select appropriate Claude model based on request"""
        # Task-based model selection
        if request.task_type == TaskType.CREATIVE:
            return "claude-3-opus-20240229"  # Best for creative tasks
        elif request.task_type == TaskType.TECHNICAL or request.task_type == TaskType.CODE_GENERATION:
            return "claude-3-sonnet-20240229"  # Good balance for technical tasks
        elif request.task_type == TaskType.SUMMARIZATION:
            return "claude-3-haiku-20240307"  # Fast and cost-effective

        # Capability-based selection
        if ModelCapability.FAST_INFERENCE in request.required_capabilities:
            return "claude-3-haiku-20240307"
        elif ModelCapability.CREATIVE_WRITING in request.required_capabilities:
            return "claude-3-opus-20240229"

        # Default to Sonnet for balanced performance
        return self.default_model

    def _build_claude_payload(self, request: ModelRequest, model: str) -> Dict[str, Any]:
        """Build Claude API payload"""
        messages = []

        # Add conversation history
        for msg in request.conversation_history:
            role = "assistant" if msg.get("role") == "assistant" else "user"
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

        payload = {
            "model": model,
            "max_tokens": min(request.max_tokens or 4000, 4000),  # Claude max is 4000
            "messages": messages
        }

        # Add system prompt if provided
        if request.system_prompt:
            payload["system"] = request.system_prompt

        # Add sampling parameters
        if request.temperature is not None:
            payload["temperature"] = min(max(request.temperature, 0.0), 1.0)

        if request.top_p is not None:
            payload["top_p"] = min(max(request.top_p, 0.0), 1.0)

        return payload

    async def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to Claude API with retries"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/v1/messages",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        self.logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_msg = f"Claude API error {response.status_code}: {response.text}"
                        if attempt == self.max_retries - 1:
                            raise Exception(error_msg)
                        else:
                            self.logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                            await asyncio.sleep(1)

            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    raise Exception("Request timeout")
                await asyncio.sleep(1)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1)

        raise Exception("Max retries exceeded")

    def _build_input_text(self, request: ModelRequest) -> str:
        """Build complete input text for cost estimation"""
        text_parts = []

        if request.system_prompt:
            text_parts.append(request.system_prompt)

        for msg in request.conversation_history:
            text_parts.append(msg.get("content", ""))

        if request.context:
            text_parts.append(request.context)

        text_parts.append(request.prompt)

        return " ".join(text_parts)

    def _calculate_actual_cost(self, response_data: Dict[str, Any], model: str) -> float:
        """Calculate actual cost based on token usage"""
        try:
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            input_cost = input_tokens * self.cost_per_input_token.get(model, 0.000003)
            output_cost = output_tokens * self.cost_per_output_token.get(model, 0.000015)

            return input_cost + output_cost

        except Exception:
            return 0.0

    def supports_task_type(self, task_type: TaskType) -> bool:
        """Check if Claude supports specific task types"""
        # Claude supports all task types well
        return True

    async def preprocess_request(self, request: ModelRequest) -> ModelRequest:
        """Preprocess request for Claude-specific optimizations"""
        # Claude works well with clear, structured prompts
        if request.task_type == TaskType.CODE_GENERATION:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are an expert programmer. Provide clean, well-documented code "
                    "with explanations. Follow best practices and include error handling."
                )

        elif request.task_type == TaskType.ANALYSIS:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are a thoughtful analyst. Provide comprehensive, structured analysis "
                    "with clear reasoning and evidence-based conclusions."
                )

        return request