"""
Multi-Agent Conversation Engine - Main FastAPI Application
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import motor.motor_asyncio
from datetime import datetime
import os
import json

from .memory import HierarchicalMemoryManager
from .webhooks import webhook_router
from .api import api_router
from .api.multi_agent_router import multi_agent_router, initialize_multi_agent_systems
from .api.ai_models_router import router as ai_models_router
from .websockets import ConnectionManager
from .config import get_settings

# Get settings first
settings = get_settings()

# Configure enhanced structured logging
from .logging_config import setup_structured_logging, get_logger, set_correlation_id, generate_correlation_id

# Setup logging based on environment
setup_structured_logging(
    log_level=settings.log_level,
    enable_json_output=settings.environment != "development",
    log_file=None  # Can be configured via environment variable
)

logger = get_logger(__name__)

# Global instances
memory_manager: HierarchicalMemoryManager = None
redis_client = None
mongodb_client = None
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    corr_id = generate_correlation_id()
    set_correlation_id(corr_id)

    logger.info_with_context(
        "Starting Multi-Agent Conversation Engine...",
        component="startup",
        operation="application_start"
    )

    # Track start time for uptime metrics
    app.state.start_time = datetime.utcnow()

    # Initialize database connections
    global redis_client, mongodb_client, memory_manager

    try:
        # Redis connection
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        logger.info_with_context(
            "Redis connection established",
            component="startup",
            operation="redis_connect",
            redis_url=settings.redis_url.split('@')[-1]  # Hide credentials
        )

        # MongoDB connection
        mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
        await mongodb_client.admin.command('ping')
        logger.info_with_context(
            "MongoDB connection established",
            component="startup",
            operation="mongodb_connect"
        )

        # Initialize memory manager
        memory_manager = HierarchicalMemoryManager(
            redis_client=redis_client,
            mongodb_client=mongodb_client,
            consolidation_threshold=settings.memory_consolidation_threshold
        )
        await memory_manager.initialize()
        logger.info_with_context(
            "Memory Manager initialized",
            component="startup",
            operation="memory_init",
            consolidation_threshold=settings.memory_consolidation_threshold
        )

        # Initialize multi-agent coordination systems
        multi_agent_success = await initialize_multi_agent_systems(redis_client)
        if multi_agent_success:
            logger.info("✅ Multi-Agent Coordination Systems initialized")
        else:
            logger.warning("⚠️ Multi-Agent Coordination Systems failed to initialize")

        logger.info("🚀 Application startup complete")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Multi-Agent Conversation Engine...")

    if memory_manager:
        await memory_manager.shutdown()

    if redis_client:
        await redis_client.close()

    if mongodb_client:
        mongodb_client.close()

    logger.info("👋 Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Multi-Agent Conversation Engine",
    description="Next-generation multi-AI conversation system with dynamic personalities and hierarchical memory",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
from starlette.middleware.base import BaseHTTPMiddleware

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation IDs to FastAPI requests"""

    async def dispatch(self, request, call_next):
        # Generate correlation ID for request
        corr_id = generate_correlation_id()
        set_correlation_id(corr_id)

        # Add to request headers
        request.state.correlation_id = corr_id

        # Log request start
        logger.info_with_context(
            f"Request started",
            component="middleware",
            operation="request_start",
            method=request.method,
            path=str(request.url.path),
            correlation_id=corr_id
        )

        response = await call_next(request)

        # Log request end
        logger.info_with_context(
            f"Request completed",
            component="middleware",
            operation="request_end",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            correlation_id=corr_id
        )

        return response

app.add_middleware(CorrelationIdMiddleware)

# Include routers
app.include_router(webhook_router, prefix="/webhook", tags=["webhooks"])
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(multi_agent_router, prefix="/api", tags=["multi-agent"])
app.include_router(ai_models_router, prefix="/api", tags=["ai-models"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent Conversation Engine v2.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    try:
        # Check Redis
        await redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Check MongoDB
        await mongodb_client.admin.command('ping')
        health_status["services"]["mongodb"] = "healthy"
    except Exception as e:
        health_status["services"]["mongodb"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    try:
        # Check Memory Manager
        if memory_manager:
            memory_stats = await memory_manager.working_memory.get_memory_usage_stats()
            health_status["services"]["memory_manager"] = "healthy"
            health_status["memory_stats"] = memory_stats
        else:
            health_status["services"]["memory_manager"] = "not_initialized"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["memory_manager"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/metrics")
async def metrics():
    """System metrics endpoint"""
    try:
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "uptime_seconds": (datetime.utcnow() - app.state.start_time).total_seconds() if hasattr(app.state, 'start_time') else 0,
                "version": "2.0.0",
                "environment": os.getenv("ENVIRONMENT", "unknown")
            },
            "memory": {},
            "multi_agent": {},
            "performance": {
                "active_websocket_connections": connection_manager.get_total_connection_count()
            }
        }

        # Get memory manager metrics
        if memory_manager:
            memory_stats = await memory_manager.working_memory.get_memory_usage_stats()
            metrics_data["memory"] = memory_stats

        # Get multi-agent coordination metrics
        from .api.multi_agent_router import get_coordination_metrics
        try:
            coordination_metrics = await get_coordination_metrics()
            metrics_data["multi_agent"] = coordination_metrics
        except Exception as e:
            logger.warning(f"Could not get coordination metrics: {e}")
            metrics_data["multi_agent"] = {"error": "metrics_unavailable"}

        return metrics_data

    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return JSONResponse(
            content={"error": "Failed to generate metrics", "details": str(e)},
            status_code=500
        )


@app.get("/status")
async def status():
    """Detailed system status endpoint"""
    try:
        status_data = {
            "system": {
                "name": "Multi-Agent Conversation Engine",
                "version": "2.0.0",
                "environment": os.getenv("ENVIRONMENT", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "operational"
            },
            "services": {},
            "features": {
                "multi_agent_coordination": True,
                "hierarchical_memory": True,
                "voice_processing": True,
                "home_automation": True,
                "real_time_websockets": True,
                "vector_embeddings": True
            },
            "statistics": {}
        }

        # Check service statuses
        services_healthy = True

        # Redis status
        try:
            await redis_client.ping()
            redis_info = await redis_client.info()
            status_data["services"]["redis"] = {
                "status": "healthy",
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_human": redis_info.get("used_memory_human", "0B"),
                "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            services_healthy = False
            status_data["services"]["redis"] = {"status": "unhealthy", "error": str(e)}

        # MongoDB status
        try:
            mongo_status = await mongodb_client.admin.command('serverStatus')
            status_data["services"]["mongodb"] = {
                "status": "healthy",
                "connections": mongo_status.get("connections", {}).get("current", 0),
                "uptime": mongo_status.get("uptime", 0),
                "version": mongo_status.get("version", "unknown")
            }
        except Exception as e:
            services_healthy = False
            status_data["services"]["mongodb"] = {"status": "unhealthy", "error": str(e)}

        # Memory Manager status
        try:
            if memory_manager:
                memory_stats = await memory_manager.working_memory.get_memory_usage_stats()
                status_data["services"]["memory_manager"] = {
                    "status": "healthy",
                    "active_sessions": memory_stats.get("active_sessions", 0),
                    "total_memories": memory_stats.get("total_memories", 0)
                }
                status_data["statistics"]["memory"] = memory_stats
            else:
                services_healthy = False
                status_data["services"]["memory_manager"] = {"status": "not_initialized"}
        except Exception as e:
            services_healthy = False
            status_data["services"]["memory_manager"] = {"status": "unhealthy", "error": str(e)}

        # WebSocket connections
        status_data["services"]["websockets"] = {
            "status": "healthy",
            "active_connections": connection_manager.get_total_connection_count(),
            "active_sessions": len(connection_manager.active_connections)
        }

        # Overall system status
        status_data["system"]["status"] = "operational" if services_healthy else "degraded"

        # Return appropriate status code
        status_code = 200 if services_healthy else 503
        return JSONResponse(content=status_data, status_code=status_code)

    except Exception as e:
        logger.error(f"Error generating status: {e}")
        return JSONResponse(
            content={
                "system": {"status": "error", "timestamp": datetime.utcnow().isoformat()},
                "error": "Failed to generate status",
                "details": str(e)
            },
            status_code=500
        )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time conversation"""
    await connection_manager.connect(websocket, session_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Process message through memory system
            await process_websocket_message(session_id, message_data, websocket)

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, session_id)
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await connection_manager.send_error(websocket, str(e))


async def process_websocket_message(session_id: str, message_data: dict, websocket: WebSocket):
    """Process incoming WebSocket message"""
    try:
        message_type = message_data.get("type")

        if message_type == "user_message":
            # Store user message in memory
            from .memory.models import MemoryItem, MemoryType

            user_memory = MemoryItem(
                content=message_data.get("content", ""),
                timestamp=datetime.utcnow(),
                importance_score=0.6,  # User messages have moderate importance
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
                agent_id=None,  # Human user
                context_tags=message_data.get("tags", [])
            )

            await memory_manager.store_memory(user_memory)

            # Broadcast to other connections
            await connection_manager.broadcast_to_session(
                session_id,
                {
                    "type": "new_message",
                    "speaker": "user",
                    "content": message_data.get("content"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        elif message_type == "get_conversation_state":
            # Return current conversation state
            state = await memory_manager.get_conversation_state(session_id)
            await websocket.send_text(json.dumps({
                "type": "conversation_state",
                "state": state.to_dict() if state else None
            }))

        elif message_type == "search_memories":
            # Search memories
            from .memory.models import MemoryQuery

            query = MemoryQuery(
                query_text=message_data.get("query", ""),
                session_id=session_id,
                max_results=message_data.get("limit", 10)
            )

            results = await memory_manager.retrieve_memories(query)
            await websocket.send_text(json.dumps({
                "type": "memory_search_results",
                "results": results.to_dict()
            }))

    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}")
        await connection_manager.send_error(websocket, str(e))


# Make global instances available to other modules
def get_memory_manager():
    """Get global memory manager instance"""
    return memory_manager


def get_redis_client():
    """Get global Redis client instance"""
    return redis_client


def get_mongodb_client():
    """Get global MongoDB client instance"""
    return mongodb_client


def get_connection_manager():
    """Get global connection manager instance"""
    return connection_manager


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )