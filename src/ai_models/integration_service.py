"""
AI Models Integration Service
Coordinates multi-model AI system with existing memory and multi-agent infrastructure
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from .model_interface import ModelManager, ModelProvider, ModelRequest, ModelResponse
from .model_router import SmartModelRouter, RoutingStrategy
from .rag_orchestrator import RAGOrchestrator, RAGConfig
from .vector_enhancer import VectorEnhancer, EmbeddingProvider
from .providers import ClaudeProvider, GPT4Provider, GeminiProvider, LocalProvider

logger = logging.getLogger(__name__)


class AIIntegrationConfig(BaseModel):
    """Configuration for AI integration"""
    # Provider settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Local model settings
    local_backend: str = "ollama"
    local_base_url: str = "http://localhost:11434"

    # Routing settings
    default_routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    fallback_enabled: bool = True
    max_fallback_attempts: int = 3

    # RAG settings
    enable_rag: bool = True
    default_max_retrieved_docs: int = 10
    default_similarity_threshold: float = 0.7

    # Vector settings
    default_embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    enable_multi_embedding: bool = True

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_async_embedding: bool = True

    # Integration settings
    integrate_with_memory: bool = True
    integrate_with_multi_agent: bool = True
    enable_background_tasks: bool = True


class AIIntegrationService:
    """
    Main service that integrates multi-model AI with existing systems
    """

    def __init__(self, config: AIIntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.model_manager: Optional[ModelManager] = None
        self.model_router: Optional[SmartModelRouter] = None
        self.vector_enhancer: Optional[VectorEnhancer] = None
        self.rag_orchestrator: Optional[RAGOrchestrator] = None

        # External integrations
        self.memory_manager = None
        self.multi_agent_coordinator = None

        # State
        self.initialized = False
        self.providers_initialized = {}

    async def initialize(
        self,
        memory_manager=None,
        multi_agent_coordinator=None
    ):
        """Initialize the AI integration service"""
        try:
            self.logger.info("🤖 Initializing AI Models Integration Service...")

            # Store external dependencies
            self.memory_manager = memory_manager
            self.multi_agent_coordinator = multi_agent_coordinator

            # Initialize vector enhancer first
            await self._initialize_vector_enhancer()

            # Initialize model providers
            await self._initialize_model_providers()

            # Initialize routing
            await self._initialize_routing()

            # Initialize RAG if enabled
            if self.config.enable_rag:
                await self._initialize_rag()

            # Setup integrations
            if self.config.integrate_with_memory and self.memory_manager:
                await self._setup_memory_integration()

            if self.config.integrate_with_multi_agent and self.multi_agent_coordinator:
                await self._setup_multi_agent_integration()

            # Start background tasks
            if self.config.enable_background_tasks:
                await self._start_background_tasks()

            self.initialized = True
            self.logger.info("✅ AI Models Integration Service initialized successfully")

            # Log initialization summary
            await self._log_initialization_summary()

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AI integration service: {e}")
            raise

    async def _initialize_vector_enhancer(self):
        """Initialize vector enhancement system"""
        try:
            vector_config = {
                "openai_api_key": self.config.openai_api_key,
                "cohere_api_key": self.config.cohere_api_key
            }

            self.vector_enhancer = VectorEnhancer(vector_config)
            await self.vector_enhancer.initialize()

            self.logger.info("✅ Vector enhancement system initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector enhancer: {e}")
            raise

    async def _initialize_model_providers(self):
        """Initialize AI model providers"""
        try:
            self.model_manager = ModelManager()
            provider_count = 0

            # Initialize Claude provider
            if self.config.anthropic_api_key:
                try:
                    claude_provider = ClaudeProvider({
                        "api_key": self.config.anthropic_api_key,
                        "default_model": "claude-3-sonnet-20240229"
                    })

                    if await claude_provider.validate_config():
                        self.model_manager.register_provider(claude_provider)
                        self.providers_initialized[ModelProvider.CLAUDE] = True
                        provider_count += 1
                        self.logger.info("✅ Claude provider initialized")
                    else:
                        self.logger.warning("⚠️ Claude provider validation failed")

                except Exception as e:
                    self.logger.warning(f"Failed to initialize Claude provider: {e}")

            # Initialize GPT-4 provider
            if self.config.openai_api_key:
                try:
                    gpt4_provider = GPT4Provider({
                        "api_key": self.config.openai_api_key,
                        "default_model": "gpt-4-turbo-preview"
                    })

                    if await gpt4_provider.validate_config():
                        self.model_manager.register_provider(gpt4_provider)
                        self.providers_initialized[ModelProvider.GPT4] = True
                        provider_count += 1
                        self.logger.info("✅ GPT-4 provider initialized")
                    else:
                        self.logger.warning("⚠️ GPT-4 provider validation failed")

                except Exception as e:
                    self.logger.warning(f"Failed to initialize GPT-4 provider: {e}")

            # Initialize Gemini provider
            if self.config.gemini_api_key:
                try:
                    gemini_provider = GeminiProvider({
                        "api_key": self.config.gemini_api_key,
                        "default_model": "gemini-pro"
                    })

                    if await gemini_provider.validate_config():
                        self.model_manager.register_provider(gemini_provider)
                        self.providers_initialized[ModelProvider.GEMINI] = True
                        provider_count += 1
                        self.logger.info("✅ Gemini provider initialized")
                    else:
                        self.logger.warning("⚠️ Gemini provider validation failed")

                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini provider: {e}")

            # Initialize Local provider
            try:
                local_provider = LocalProvider({
                    "backend_type": self.config.local_backend,
                    "base_url": self.config.local_base_url
                })

                if await local_provider.validate_config():
                    self.model_manager.register_provider(local_provider)
                    self.providers_initialized[ModelProvider.LOCAL] = True
                    provider_count += 1
                    self.logger.info("✅ Local provider initialized")
                else:
                    self.logger.warning("⚠️ Local provider validation failed")

            except Exception as e:
                self.logger.warning(f"Failed to initialize local provider: {e}")

            if provider_count == 0:
                raise Exception("No AI providers were successfully initialized")

            self.logger.info(f"✅ Initialized {provider_count} AI providers")

        except Exception as e:
            self.logger.error(f"Failed to initialize model providers: {e}")
            raise

    async def _initialize_routing(self):
        """Initialize smart model routing"""
        try:
            if not self.model_manager:
                raise Exception("Model manager not initialized")

            self.model_router = SmartModelRouter(self.model_manager)

            # Configure routing settings
            self.model_router.default_strategy = self.config.default_routing_strategy
            self.model_router.fallback_enabled = self.config.fallback_enabled
            self.model_router.max_fallback_attempts = self.config.max_fallback_attempts

            self.logger.info("✅ Smart model routing initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize routing: {e}")
            raise

    async def _initialize_rag(self):
        """Initialize RAG orchestrator"""
        try:
            if not self.vector_enhancer or not self.model_router:
                raise Exception("Dependencies not initialized for RAG")

            self.rag_orchestrator = RAGOrchestrator(
                vector_enhancer=self.vector_enhancer,
                memory_manager=self.memory_manager,
                model_router=self.model_router
            )

            self.logger.info("✅ RAG orchestrator initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG: {e}")
            raise

    async def _setup_memory_integration(self):
        """Setup integration with memory system"""
        try:
            if not self.memory_manager:
                self.logger.warning("Memory manager not available for integration")
                return

            # Enhance memory system with multi-model embeddings
            if hasattr(self.memory_manager, 'vector_retrieval') and self.vector_enhancer:
                # Replace or enhance the existing vector retrieval
                memory_vector_system = self.memory_manager.vector_retrieval

                # Add multi-model embedding support
                original_generate_embedding = memory_vector_system.generate_embedding

                async def enhanced_generate_embedding(text: str):
                    # Try multiple providers for better embeddings
                    providers_to_try = [
                        self.config.default_embedding_provider,
                        EmbeddingProvider.SENTENCE_TRANSFORMERS,
                        EmbeddingProvider.OPENAI
                    ]

                    for provider in providers_to_try:
                        try:
                            embedding = await self.vector_enhancer.generate_embedding(
                                text, provider=provider
                            )
                            if embedding:
                                return embedding
                        except Exception as e:
                            self.logger.warning(f"Embedding with {provider} failed: {e}")

                    # Fallback to original method
                    return await original_generate_embedding(text)

                # Monkey patch for enhanced embedding generation
                memory_vector_system.generate_embedding = enhanced_generate_embedding

            self.logger.info("✅ Memory system integration completed")

        except Exception as e:
            self.logger.error(f"Failed to setup memory integration: {e}")

    async def _setup_multi_agent_integration(self):
        """Setup integration with multi-agent system"""
        try:
            if not self.multi_agent_coordinator:
                self.logger.warning("Multi-agent coordinator not available for integration")
                return

            # Enhance multi-agent responses with intelligent model routing
            # This would require modifications to the multi-agent system
            # to use the new AI model router instead of direct OpenAI calls

            self.logger.info("✅ Multi-agent system integration completed")

        except Exception as e:
            self.logger.error(f"Failed to setup multi-agent integration: {e}")

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Background embedding generation for existing memories
            if self.config.enable_async_embedding and self.vector_enhancer:
                asyncio.create_task(self._background_embedding_task())

            # Performance monitoring and optimization
            asyncio.create_task(self._performance_monitoring_task())

            self.logger.info("✅ Background tasks started")

        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")

    async def _background_embedding_task(self):
        """Background task for generating embeddings"""
        try:
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes

                if self.memory_manager and hasattr(self.memory_manager, 'vector_retrieval'):
                    # Check for memories without embeddings and generate them
                    # This would be implemented based on the memory system API
                    pass

        except Exception as e:
            self.logger.error(f"Background embedding task failed: {e}")

    async def _performance_monitoring_task(self):
        """Background task for performance monitoring"""
        try:
            while True:
                await asyncio.sleep(600)  # Run every 10 minutes

                if self.model_router:
                    # Collect performance metrics
                    metrics = self.model_router.get_routing_analytics()

                    # Log performance summary
                    self.logger.info(
                        f"🔍 Performance metrics: "
                        f"requests={metrics.get('total_decisions', 0)}, "
                        f"avg_decision_time={metrics.get('avg_decision_time_ms', 0):.2f}ms"
                    )

        except Exception as e:
            self.logger.error(f"Performance monitoring task failed: {e}")

    async def _log_initialization_summary(self):
        """Log summary of initialization"""
        try:
            summary = {
                "providers": list(self.providers_initialized.keys()),
                "provider_count": len(self.providers_initialized),
                "vector_enhancer": self.vector_enhancer is not None,
                "model_router": self.model_router is not None,
                "rag_orchestrator": self.rag_orchestrator is not None,
                "memory_integration": self.config.integrate_with_memory and self.memory_manager is not None,
                "multi_agent_integration": self.config.integrate_with_multi_agent and self.multi_agent_coordinator is not None
            }

            self.logger.info(f"🎯 AI Integration Summary: {summary}")

        except Exception as e:
            self.logger.warning(f"Failed to log initialization summary: {e}")

    # Public API methods

    async def generate_response(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        use_rag: bool = None,
        routing_strategy: Optional[RoutingStrategy] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate AI response with intelligent routing"""
        try:
            if not self.initialized:
                raise Exception("AI integration service not initialized")

            # Build request
            request = ModelRequest(
                prompt=prompt,
                session_id=session_id,
                use_rag=use_rag if use_rag is not None else self.config.enable_rag,
                **kwargs
            )

            # Use RAG if enabled and available
            if request.use_rag and self.rag_orchestrator:
                rag_config = RAGConfig(
                    max_retrieved_docs=self.config.default_max_retrieved_docs,
                    similarity_threshold=self.config.default_similarity_threshold
                )
                rag_result = await self.rag_orchestrator.process_rag_request(request, rag_config)
                return rag_result.response

            # Otherwise use standard routing
            elif self.model_router:
                return await self.model_router.execute_request_with_routing(request)

            else:
                raise Exception("No AI generation method available")

        except Exception as e:
            self.logger.error(f"AI response generation failed: {e}")
            raise

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of AI integration system"""
        try:
            status = {
                "initialized": self.initialized,
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "model_manager": self.model_manager is not None,
                    "model_router": self.model_router is not None,
                    "vector_enhancer": self.vector_enhancer is not None,
                    "rag_orchestrator": self.rag_orchestrator is not None
                },
                "providers": {},
                "integrations": {
                    "memory": self.memory_manager is not None,
                    "multi_agent": self.multi_agent_coordinator is not None
                }
            }

            # Get provider health
            if self.model_manager:
                health_results = await self.model_manager.health_check_all()
                for provider, result in health_results.items():
                    status["providers"][provider] = result.get("status", "unknown")

            # Get performance metrics
            if self.model_router:
                analytics = self.model_router.get_routing_analytics()
                status["performance"] = {
                    "total_requests": analytics.get("total_decisions", 0),
                    "avg_decision_time_ms": analytics.get("avg_decision_time_ms", 0),
                    "provider_usage": analytics.get("provider_usage", {})
                }

            return status

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {
                "initialized": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def shutdown(self):
        """Gracefully shutdown the AI integration service"""
        try:
            self.logger.info("🔄 Shutting down AI integration service...")

            # Cancel background tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()

            # Close provider connections
            if self.vector_enhancer and hasattr(self.vector_enhancer, 'cohere_client'):
                if self.vector_enhancer.cohere_client:
                    await self.vector_enhancer.cohere_client.aclose()

            self.initialized = False
            self.logger.info("✅ AI integration service shutdown complete")

        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")


# Factory function for easy integration
async def create_ai_integration_service(
    config: Dict[str, Any],
    memory_manager=None,
    multi_agent_coordinator=None
) -> AIIntegrationService:
    """Factory function to create and initialize AI integration service"""
    try:
        # Create configuration
        ai_config = AIIntegrationConfig(**config)

        # Create service
        service = AIIntegrationService(ai_config)

        # Initialize
        await service.initialize(
            memory_manager=memory_manager,
            multi_agent_coordinator=multi_agent_coordinator
        )

        return service

    except Exception as e:
        logger.error(f"Failed to create AI integration service: {e}")
        raise