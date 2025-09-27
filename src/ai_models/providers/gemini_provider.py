"""
Google Gemini Provider Implementation
Provides interface to Google's Gemini models
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


class GeminiProvider(ModelInterface):
    """Google Gemini provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(ModelProvider.GEMINI, config)

        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://generativelanguage.googleapis.com")
        self.default_model = config.get("default_model", "gemini-pro")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 60)

        # Available models
        self.available_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-ultra"  # When available
        ]

        # Model capabilities
        self.model_capabilities = {
            "gemini-pro": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.FAST_INFERENCE
            ],
            "gemini-pro-vision": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.VISION,
                ModelCapability.MULTILINGUAL,
                ModelCapability.CODE_UNDERSTANDING
            ],
            "gemini-ultra": [
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.MATHEMATICAL_REASONING,
                ModelCapability.CREATIVE_WRITING,
                ModelCapability.MULTILINGUAL,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.VISION
            ]
        }

        # Cost estimates (Gemini pricing varies by region)
        self.cost_per_input_token = {
            "gemini-pro": 0.000000125,  # Very competitive pricing
            "gemini-pro-vision": 0.00000025,
            "gemini-ultra": 0.000008  # Premium model
        }
        self.cost_per_output_token = {
            "gemini-pro": 0.000000375,
            "gemini-pro-vision": 0.00000075,
            "gemini-ultra": 0.000024
        }

    async def validate_config(self) -> bool:
        """Validate Gemini configuration"""
        if not self.api_key:
            self.logger.error("Gemini API key not provided")
            return False

        try:
            # Test API connection
            test_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Hello"}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 10
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/models/{self.default_model}:generateContent?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=test_payload,
                    timeout=10
                )

                if response.status_code == 200:
                    self.logger.info("Gemini API validation successful")
                    return True
                else:
                    self.logger.error(f"Gemini API validation failed: {response.status_code}")
                    return False

        except Exception as e:
            self.logger.error(f"Gemini API validation error: {e}")
            return False

    def get_capabilities(self) -> List[ModelCapability]:
        """Get Gemini capabilities"""
        all_capabilities = set()
        for capabilities in self.model_capabilities.values():
            all_capabilities.update(capabilities)
        return list(all_capabilities)

    def get_available_models(self) -> List[str]:
        """Get available Gemini models"""
        return self.available_models.copy()

    def estimate_cost(self, request: ModelRequest) -> float:
        """Estimate cost for Gemini request"""
        try:
            model = self._select_model(request)

            # Estimate tokens (Gemini uses similar tokenization to other models)
            input_text = self._build_input_text(request)
            estimated_input_tokens = len(input_text.split()) * 1.33

            estimated_output_tokens = request.max_tokens or 1000

            input_cost = estimated_input_tokens * self.cost_per_input_token.get(model, 0.000000125)
            output_cost = estimated_output_tokens * self.cost_per_output_token.get(model, 0.000000375)

            return input_cost + output_cost

        except Exception as e:
            self.logger.warning(f"Cost estimation failed: {e}")
            return 0.001  # Very low default for Gemini

    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Gemini"""
        start_time = time.time()

        try:
            # Select appropriate model
            model = self._select_model(request)

            # Build Gemini-specific payload
            payload = self._build_gemini_payload(request, model)

            # Make API request
            response_data = await self._make_api_request(payload, model)

            # Parse response
            response_time_ms = (time.time() - start_time) * 1000

            # Extract content from Gemini response format
            content = ""
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    content = " ".join(part.get("text", "") for part in parts)

            return ModelResponse(
                content=content,
                provider=ModelProvider.GEMINI,
                model_name=model,
                response_time_ms=response_time_ms,
                tokens_used=self._extract_token_usage(response_data),
                cost_estimate=self._calculate_actual_cost(response_data, model),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Gemini response generation failed: {e}")

            return ModelResponse(
                content="",
                provider=ModelProvider.GEMINI,
                model_name=self.default_model,
                response_time_ms=response_time_ms,
                error=f"Gemini API error: {str(e)}"
            )

    def _select_model(self, request: ModelRequest) -> str:
        """Select appropriate Gemini model based on request"""
        # Check for vision requirements
        if ModelCapability.VISION in request.required_capabilities:
            return "gemini-pro-vision"

        # Task-based selection
        if request.task_type == TaskType.CREATIVE:
            return "gemini-pro"  # Excellent for creative tasks
        elif request.task_type == TaskType.TECHNICAL or request.task_type == TaskType.CODE_GENERATION:
            return "gemini-pro"  # Strong coding capabilities
        elif request.task_type == TaskType.MATH:
            return "gemini-pro"  # Good mathematical reasoning

        # Check for premium requirements
        if (request.required_capabilities and
            len(request.required_capabilities) > 3):
            # Complex multi-capability requests might benefit from Ultra when available
            if "gemini-ultra" in self.available_models:
                return "gemini-ultra"

        return self.default_model

    def _build_gemini_payload(self, request: ModelRequest, model: str) -> Dict[str, Any]:
        """Build Gemini API payload"""
        contents = []

        # Gemini uses a different format for conversation history
        # Combine system prompt and conversation into contents
        current_text_parts = []

        if request.system_prompt:
            current_text_parts.append(f"System: {request.system_prompt}")

        # Add conversation history
        for msg in request.conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            current_text_parts.append(f"{role.title()}: {content}")

        # Add context if provided
        if request.context:
            current_text_parts.append(f"Context: {request.context}")

        # Add current request
        current_text_parts.append(f"User: {request.prompt}")

        # Combine all parts
        combined_text = "\n\n".join(current_text_parts)

        contents.append({
            "parts": [{"text": combined_text}]
        })

        # Build generation config
        generation_config = {
            "maxOutputTokens": min(request.max_tokens or 2048, 2048),  # Gemini max varies
        }

        if request.temperature is not None:
            generation_config["temperature"] = min(max(request.temperature, 0.0), 1.0)

        if request.top_p is not None:
            generation_config["topP"] = min(max(request.top_p, 0.0), 1.0)

        payload = {
            "contents": contents,
            "generationConfig": generation_config
        }

        return payload

    async def _make_api_request(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Make request to Gemini API with retries"""
        url = f"{self.base_url}/v1/models/{model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url,
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
                        error_msg = f"Gemini API error {response.status_code}: {response.text}"
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

    def _extract_token_usage(self, response_data: Dict[str, Any]) -> Optional[int]:
        """Extract token usage from Gemini response"""
        try:
            # Gemini might include usage metadata in future API versions
            usage_metadata = response_data.get("usageMetadata", {})
            if usage_metadata:
                prompt_tokens = usage_metadata.get("promptTokenCount", 0)
                candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
                return prompt_tokens + candidates_tokens
            return None
        except Exception:
            return None

    def _calculate_actual_cost(self, response_data: Dict[str, Any], model: str) -> float:
        """Calculate actual cost based on token usage"""
        try:
            usage_metadata = response_data.get("usageMetadata", {})
            if not usage_metadata:
                return 0.0

            input_tokens = usage_metadata.get("promptTokenCount", 0)
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)

            input_cost = input_tokens * self.cost_per_input_token.get(model, 0.000000125)
            output_cost = output_tokens * self.cost_per_output_token.get(model, 0.000000375)

            return input_cost + output_cost

        except Exception:
            return 0.0

    def supports_task_type(self, task_type: TaskType) -> bool:
        """Check if Gemini supports specific task types"""
        # Gemini supports most task types well, with some exceptions
        strong_tasks = [
            TaskType.CONVERSATION,
            TaskType.CREATIVE,
            TaskType.TECHNICAL,
            TaskType.CODE_GENERATION,
            TaskType.REASONING,
            TaskType.MATH
        ]
        return task_type in strong_tasks

    async def preprocess_request(self, request: ModelRequest) -> ModelRequest:
        """Preprocess request for Gemini-specific optimizations"""
        # Gemini works well with clear, conversational prompts
        if request.task_type == TaskType.CODE_GENERATION:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are an expert programmer. Write clean, efficient code with clear comments. "
                    "Explain your approach and include error handling where appropriate."
                )

        elif request.task_type == TaskType.CREATIVE:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are a creative writer. Create engaging, original content that captures "
                    "the reader's attention. Be imaginative and expressive in your writing."
                )

        elif request.task_type == TaskType.MATH:
            if not request.system_prompt:
                request.system_prompt = (
                    "You are a mathematics expert. Show step-by-step solutions with clear explanations. "
                    "Verify your work and explain the reasoning behind each step."
                )

        return request

    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check for Gemini"""
        try:
            start_time = time.time()

            test_payload = {
                "contents": [{"parts": [{"text": "Health check test"}]}],
                "generationConfig": {"maxOutputTokens": 5}
            }

            response_data = await self._make_api_request(test_payload, "gemini-pro")
            response_time = (time.time() - start_time) * 1000

            # Check if we got a valid response
            is_healthy = (
                "candidates" in response_data and
                response_data["candidates"] and
                "content" in response_data["candidates"][0]
            )

            return {
                "provider": self.provider.value,
                "status": "healthy" if is_healthy else "unhealthy",
                "response_time_ms": response_time,
                "test_successful": is_healthy,
                "model_tested": "gemini-pro",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                "provider": self.provider.value,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }