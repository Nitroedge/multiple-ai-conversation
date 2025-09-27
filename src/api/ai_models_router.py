"""
AI Models API Router
REST endpoints for multi-model AI integration, routing, and RAG operations
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ..ai_models import (
    ModelInterface, ModelProvider, ModelRequest, ModelResponse,
    SmartModelRouter, TaskType, ModelCapability, RoutingStrategy,
    RAGOrchestrator, RAGConfig, RetrievalStrategy,
    VectorEnhancer, EmbeddingProvider
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-models", tags=["AI Models"])


# Request/Response Models
class ModelSelectionRequest(BaseModel):
    """Request for model selection"""
    prompt: str = Field(..., description="Input prompt")
    task_type: Optional[TaskType] = Field(None, description="Type of task")
    required_capabilities: List[ModelCapability] = Field(default_factory=list)
    strategy: Optional[RoutingStrategy] = Field(RoutingStrategy.BALANCED, description="Routing strategy")
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")


class AIGenerationRequest(BaseModel):
    """Request for AI text generation"""
    prompt: str = Field(..., description="Input prompt")
    context: Optional[str] = Field(None, description="Additional context")
    system_prompt: Optional[str] = Field(None, description="System instructions")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    # Model selection
    preferred_provider: Optional[ModelProvider] = Field(None, description="Preferred AI provider")
    task_type: Optional[TaskType] = Field(None, description="Type of task")
    required_capabilities: List[ModelCapability] = Field(default_factory=list)
    routing_strategy: RoutingStrategy = Field(RoutingStrategy.BALANCED, description="Routing strategy")

    # Generation parameters
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")

    # RAG settings
    use_rag: bool = Field(False, description="Use retrieval-augmented generation")
    max_retrieved_docs: Optional[int] = Field(10, description="Max documents for RAG")
    retrieval_strategy: Optional[RetrievalStrategy] = Field(RetrievalStrategy.HYBRID_SEARCH)

    # Session info
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")


class RAGRequest(BaseModel):
    """Request for RAG-enhanced generation"""
    prompt: str = Field(..., description="Input prompt")
    session_id: str = Field(..., description="Session identifier")

    # RAG configuration
    max_retrieved_docs: int = Field(10, description="Maximum documents to retrieve")
    similarity_threshold: float = Field(0.7, description="Minimum similarity threshold")
    retrieval_strategy: RetrievalStrategy = Field(RetrievalStrategy.HYBRID_SEARCH)
    include_citations: bool = Field(True, description="Include source citations")

    # Generation settings
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens")
    temperature: Optional[float] = Field(0.7, description="Temperature")
    preferred_provider: Optional[ModelProvider] = Field(None, description="Preferred provider")


class EmbeddingRequest(BaseModel):
    """Request for embedding generation"""
    texts: List[str] = Field(..., description="Texts to embed")
    provider: Optional[EmbeddingProvider] = Field(None, description="Embedding provider")
    model_name: Optional[str] = Field(None, description="Specific model")


class ModelHealthResponse(BaseModel):
    """Health status of AI models"""
    provider: str
    status: str
    response_time_ms: float
    timestamp: str
    error: Optional[str] = None


class ProviderCapabilitiesResponse(BaseModel):
    """Provider capabilities information"""
    provider: str
    available: bool
    models: List[str]
    capabilities: List[str]
    cost_efficient: bool
    supports_streaming: bool


class RoutingAnalyticsResponse(BaseModel):
    """Analytics about model routing"""
    total_requests: int
    provider_usage: Dict[str, int]
    strategy_usage: Dict[str, int]
    avg_response_time: float
    success_rate: float


# Dependency injection
class AIModelsService:
    """Service for AI models operations"""

    def __init__(self):
        self.model_router: Optional[SmartModelRouter] = None
        self.rag_orchestrator: Optional[RAGOrchestrator] = None
        self.vector_enhancer: Optional[VectorEnhancer] = None
        self.initialized = False

    async def initialize(self, config: Dict[str, Any]):
        """Initialize the AI models service"""
        if self.initialized:
            return

        try:
            from ..ai_models.model_router import SmartModelRouter
            from ..ai_models.rag_orchestrator import RAGOrchestrator
            from ..ai_models.vector_enhancer import VectorEnhancer
            from ..ai_models.model_interface import ModelManager
            from ..ai_models.providers import (
                ClaudeProvider, GPT4Provider, GeminiProvider, LocalProvider
            )

            # Initialize vector enhancer
            self.vector_enhancer = VectorEnhancer(config)
            await self.vector_enhancer.initialize()

            # Initialize model manager and providers
            model_manager = ModelManager()

            # Register providers based on configuration
            if config.get("openai_api_key"):
                gpt4_provider = GPT4Provider({
                    "api_key": config["openai_api_key"],
                    "default_model": "gpt-4-turbo-preview"
                })
                if await gpt4_provider.validate_config():
                    model_manager.register_provider(gpt4_provider)

            if config.get("anthropic_api_key"):
                claude_provider = ClaudeProvider({
                    "api_key": config["anthropic_api_key"],
                    "default_model": "claude-3-sonnet-20240229"
                })
                if await claude_provider.validate_config():
                    model_manager.register_provider(claude_provider)

            if config.get("gemini_api_key"):
                gemini_provider = GeminiProvider({
                    "api_key": config["gemini_api_key"],
                    "default_model": "gemini-pro"
                })
                if await gemini_provider.validate_config():
                    model_manager.register_provider(gemini_provider)

            # Always add local provider
            local_provider = LocalProvider({
                "backend_type": config.get("local_backend", "ollama"),
                "base_url": config.get("local_base_url", "http://localhost:11434")
            })
            if await local_provider.validate_config():
                model_manager.register_provider(local_provider)

            # Initialize router
            self.model_router = SmartModelRouter(model_manager)

            # Initialize RAG orchestrator (requires memory manager)
            # This would need to be injected from the main application
            # self.rag_orchestrator = RAGOrchestrator(
            #     self.vector_enhancer,
            #     memory_manager,  # Injected
            #     self.model_router
            # )

            self.initialized = True
            logger.info("AI Models service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Models service: {e}")
            raise


# Global service instance
ai_models_service = AIModelsService()


async def get_ai_models_service() -> AIModelsService:
    """Dependency to get AI models service"""
    if not ai_models_service.initialized:
        # Initialize with default config - would be injected in real app
        config = {
            "openai_api_key": "placeholder",
            "anthropic_api_key": "placeholder",
            "gemini_api_key": "placeholder"
        }
        await ai_models_service.initialize(config)

    return ai_models_service


# API Endpoints

@router.post("/generate", response_model=ModelResponse)
async def generate_text(
    request: AIGenerationRequest,
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Generate text using AI models with intelligent routing"""
    try:
        if not service.model_router:
            raise HTTPException(status_code=503, detail="AI models service not available")

        # Build model request
        model_request = ModelRequest(
            prompt=request.prompt,
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
            use_rag=request.use_rag
        )

        # Generate response
        if request.use_rag and service.rag_orchestrator:
            rag_config = RAGConfig(
                max_retrieved_docs=request.max_retrieved_docs or 10,
                retrieval_strategy=request.retrieval_strategy or RetrievalStrategy.HYBRID_SEARCH
            )
            rag_result = await service.rag_orchestrator.process_rag_request(
                model_request, rag_config
            )
            return rag_result.response
        else:
            return await service.model_router.execute_request_with_routing(model_request)

    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/rag-generate", response_model=ModelResponse)
async def rag_generate(
    request: RAGRequest,
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Generate text using RAG (Retrieval-Augmented Generation)"""
    try:
        if not service.rag_orchestrator:
            raise HTTPException(status_code=503, detail="RAG service not available")

        # Build model request
        model_request = ModelRequest(
            prompt=request.prompt,
            session_id=request.session_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_rag=True
        )

        # Build RAG config
        rag_config = RAGConfig(
            max_retrieved_docs=request.max_retrieved_docs,
            similarity_threshold=request.similarity_threshold,
            retrieval_strategy=request.retrieval_strategy,
            include_citations=request.include_citations
        )

        # Process RAG request
        rag_result = await service.rag_orchestrator.process_rag_request(
            model_request, rag_config
        )

        return rag_result.response

    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {str(e)}")


@router.post("/route-selection")
async def get_model_selection(
    request: ModelSelectionRequest,
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Get model routing decision without generating response"""
    try:
        if not service.model_router:
            raise HTTPException(status_code=503, detail="Router service not available")

        model_request = ModelRequest(
            prompt=request.prompt,
            task_type=request.task_type,
            required_capabilities=request.required_capabilities,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        routing_decision = await service.model_router.route_request(
            model_request, request.strategy
        )

        return {
            "selected_provider": routing_decision.selected_provider,
            "fallback_providers": routing_decision.fallback_providers,
            "routing_strategy": routing_decision.routing_strategy,
            "decision_time_ms": routing_decision.decision_time_ms,
            "reasoning": routing_decision.reasoning,
            "score_breakdown": routing_decision.score_breakdown
        }

    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")


@router.post("/embeddings")
async def generate_embeddings(
    request: EmbeddingRequest,
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Generate embeddings for text"""
    try:
        if not service.vector_enhancer:
            raise HTTPException(status_code=503, detail="Vector service not available")

        embeddings = await service.vector_enhancer.batch_generate_embeddings(
            request.texts,
            provider=request.provider,
            model_name=request.model_name
        )

        return {
            "embeddings": embeddings,
            "provider": request.provider or "auto-selected",
            "model": request.model_name or "default",
            "count": len(embeddings)
        }

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@router.get("/health", response_model=List[ModelHealthResponse])
async def check_models_health(
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Check health of all AI model providers"""
    try:
        if not service.model_router:
            raise HTTPException(status_code=503, detail="AI models service not available")

        health_results = await service.model_router.model_manager.health_check_all()

        return [
            ModelHealthResponse(
                provider=provider,
                status=result.get("status", "unknown"),
                response_time_ms=result.get("response_time_ms", 0),
                timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
                error=result.get("error")
            )
            for provider, result in health_results.items()
        ]

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/providers", response_model=List[ProviderCapabilitiesResponse])
async def get_provider_capabilities(
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Get capabilities of all available providers"""
    try:
        if not service.model_router:
            raise HTTPException(status_code=503, detail="AI models service not available")

        providers_info = []

        for provider_type in service.model_router.model_manager.get_available_providers():
            provider = service.model_router.model_manager.get_provider(provider_type)
            if provider:
                capabilities = provider.get_capabilities()
                models = provider.get_available_models()

                providers_info.append(ProviderCapabilitiesResponse(
                    provider=provider_type.value,
                    available=True,
                    models=models,
                    capabilities=[cap.value for cap in capabilities],
                    cost_efficient="cost_efficient" in [cap.value for cap in capabilities],
                    supports_streaming=provider_type in [ModelProvider.GPT4, ModelProvider.CLAUDE]
                ))

        return providers_info

    except Exception as e:
        logger.error(f"Provider capabilities check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capabilities check failed: {str(e)}")


@router.get("/analytics", response_model=RoutingAnalyticsResponse)
async def get_routing_analytics(
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Get analytics about model routing and usage"""
    try:
        if not service.model_router:
            raise HTTPException(status_code=503, detail="Router service not available")

        analytics = service.model_router.get_routing_analytics()

        return RoutingAnalyticsResponse(
            total_requests=analytics.get("total_decisions", 0),
            provider_usage=analytics.get("provider_usage", {}),
            strategy_usage=analytics.get("strategy_usage", {}),
            avg_response_time=analytics.get("avg_decision_time_ms", 0),
            success_rate=1.0  # Would calculate from actual data
        )

    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@router.get("/embedding-providers")
async def get_embedding_providers(
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Get available embedding providers and their capabilities"""
    try:
        if not service.vector_enhancer:
            raise HTTPException(status_code=503, detail="Vector service not available")

        capabilities = await service.vector_enhancer.get_provider_capabilities()
        return capabilities

    except Exception as e:
        logger.error(f"Embedding providers check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Provider check failed: {str(e)}")


@router.post("/optimize-routing")
async def optimize_routing_rules(
    task_performance: Dict[str, Dict[str, float]],
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Optimize routing rules based on performance data"""
    try:
        if not service.model_router:
            raise HTTPException(status_code=503, detail="Router service not available")

        # This would implement routing optimization
        # Based on performance data, adjust routing rules

        return {
            "status": "optimization_completed",
            "rules_updated": 0,
            "performance_improvement": 0.0,
            "message": "Routing optimization feature would be implemented here"
        }

    except Exception as e:
        logger.error(f"Routing optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


# Background task endpoints
@router.post("/background/batch-embed")
async def batch_embed_session_memories(
    session_id: str,
    background_tasks: BackgroundTasks,
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Start background embedding generation for session memories"""
    try:
        async def embed_memories():
            if service.vector_enhancer:
                # This would integrate with memory system
                logger.info(f"Starting background embedding for session {session_id}")
                # Implementation would batch generate embeddings

        background_tasks.add_task(embed_memories)

        return {
            "status": "started",
            "session_id": session_id,
            "message": "Background embedding task started"
        }

    except Exception as e:
        logger.error(f"Background embedding task failed: {e}")
        raise HTTPException(status_code=500, detail=f"Background task failed: {str(e)}")


@router.get("/status")
async def get_service_status(
    service: AIModelsService = Depends(get_ai_models_service)
):
    """Get overall status of AI models service"""
    try:
        return {
            "service": "ai_models",
            "status": "operational" if service.initialized else "initializing",
            "components": {
                "model_router": service.model_router is not None,
                "rag_orchestrator": service.rag_orchestrator is not None,
                "vector_enhancer": service.vector_enhancer is not None
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "service": "ai_models",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }