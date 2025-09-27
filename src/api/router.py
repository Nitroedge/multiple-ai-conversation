"""
API router for REST endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel

from ..config import get_settings
from ..memory.models import MemoryItem, MemoryType, MemoryQuery, ConversationState

logger = logging.getLogger(__name__)
api_router = APIRouter()

settings = get_settings()


# Pydantic models for API requests/responses
class ConversationStartRequest(BaseModel):
    user_id: Optional[str] = None
    initial_message: Optional[str] = None
    agents: Optional[List[str]] = ["OSWALD", "TONY_KING", "VICTORIA"]
    topic: Optional[str] = "general_conversation"


class ConversationStartResponse(BaseModel):
    session_id: str
    status: str
    agents: List[str]
    message: str
    timestamp: str


class MemoryStoreRequest(BaseModel):
    content: str
    session_id: str
    importance_score: float = 0.5
    memory_type: str = "episodic"
    agent_id: Optional[str] = None
    context_tags: Optional[List[str]] = None


class MemorySearchRequest(BaseModel):
    query_text: str
    session_id: str
    agent_id: Optional[str] = None
    memory_types: Optional[List[str]] = None
    importance_threshold: float = 0.0
    max_results: int = 10


class AgentResponseRequest(BaseModel):
    session_id: str
    agent_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_input: Optional[str] = None


@api_router.post("/conversations/start", response_model=ConversationStartResponse)
async def start_conversation(request: ConversationStartRequest):
    """Start a new conversation session"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Generate session ID
        import uuid
        session_id = str(uuid.uuid4())

        # Create initial conversation state
        conversation_state = ConversationState(
            session_id=session_id,
            active_agents=request.agents or ["OSWALD", "TONY_KING", "VICTORIA"],
            conversation_stage="greeting",
            topic_focus=request.topic or "general_conversation",
            emotion_context={},
            last_speaker=None,
            turn_count=0,
            user_preferences={},
            timestamp=datetime.utcnow()
        )

        await memory_manager.update_conversation_state(conversation_state)

        # Store initial message if provided
        if request.initial_message:
            initial_memory = MemoryItem(
                content=request.initial_message,
                timestamp=datetime.utcnow(),
                importance_score=0.6,
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
                agent_id=None,
                context_tags=["initial_message"]
            )
            await memory_manager.store_memory(initial_memory)

        response = ConversationStartResponse(
            session_id=session_id,
            status="started",
            agents=conversation_state.active_agents,
            message="Conversation started successfully",
            timestamp=datetime.utcnow().isoformat()
        )

        logger.info(f"Started new conversation: {session_id}")
        return response

    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/conversations/{session_id}/state")
async def get_conversation_state(session_id: str = Path(..., description="Session ID")):
    """Get current conversation state"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        state = await memory_manager.get_conversation_state(session_id)

        if not state:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "session_id": session_id,
            "state": state.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/conversations/{session_id}/summary")
async def get_conversation_summary(session_id: str = Path(..., description="Session ID")):
    """Get comprehensive conversation summary"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        summary = await memory_manager.get_session_summary(session_id)

        return {
            "session_id": session_id,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    """Store a memory item"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Create memory item
        memory_item = MemoryItem(
            content=request.content,
            timestamp=datetime.utcnow(),
            importance_score=request.importance_score,
            memory_type=MemoryType(request.memory_type),
            session_id=request.session_id,
            agent_id=request.agent_id,
            context_tags=request.context_tags
        )

        memory_id = await memory_manager.store_memory(memory_item)

        return {
            "memory_id": memory_id,
            "status": "stored",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/memory/search")
async def search_memories(request: MemorySearchRequest):
    """Search memories"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Create memory query
        memory_types = None
        if request.memory_types:
            memory_types = [MemoryType(mt) for mt in request.memory_types]

        query = MemoryQuery(
            query_text=request.query_text,
            session_id=request.session_id,
            agent_id=request.agent_id,
            memory_types=memory_types,
            importance_threshold=request.importance_threshold,
            max_results=request.max_results
        )

        results = await memory_manager.retrieve_memories(query)

        return {
            "query": request.query_text,
            "session_id": request.session_id,
            "results": results.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/agents/response")
async def generate_agent_response(request: AgentResponseRequest):
    """Generate response from an agent"""
    try:
        # For now, return mock response
        # TODO: Implement actual agent response generation

        mock_responses = {
            "OSWALD": "OH WOW! That's absolutely fascinating! Tell me more!",
            "TONY_KING": "Listen here, that reminds me of this story from Brooklyn...",
            "VICTORIA": "That raises some interesting philosophical questions about the nature of..."
        }

        agent_id = request.agent_id or "OSWALD"
        response_content = mock_responses.get(agent_id, "I'm thinking about that...")

        return {
            "agent_id": agent_id,
            "response": response_content,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "processing_time_ms": 100,
                "confidence": 0.95,
                "mock_response": True
            }
        }

    except Exception as e:
        logger.error(f"Error generating agent response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/agents")
async def list_agents():
    """List available agents"""
    from ..config import CHARACTER_DEFAULTS

    agents = []
    for agent_name, config in CHARACTER_DEFAULTS.items():
        agents.append({
            "id": agent_name,
            "name": agent_name.replace("_", " ").title(),
            "personality": config["personality"],
            "voice": config["voice"],
            "response_style": config["response_style"]
        })

    return {
        "agents": agents,
        "total": len(agents),
        "timestamp": datetime.utcnow().isoformat()
    }


@api_router.get("/system/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        from ..main import get_memory_manager, get_connection_manager
        memory_manager = get_memory_manager()
        connection_manager = get_connection_manager()

        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational"
        }

        # Memory statistics
        if memory_manager:
            memory_stats = await memory_manager.working_memory.get_memory_usage_stats()
            stats["memory"] = memory_stats
        else:
            stats["memory"] = {"status": "unavailable"}

        # Connection statistics
        if connection_manager:
            connection_stats = connection_manager.get_connection_stats()
            stats["connections"] = connection_stats
        else:
            stats["connections"] = {"status": "unavailable"}

        return stats

    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    try:
        from ..main import get_memory_manager, get_redis_client, get_mongodb_client

        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }

        # Check Redis
        redis_client = get_redis_client()
        if redis_client:
            try:
                await redis_client.ping()
                health["components"]["redis"] = {"status": "healthy", "response_time_ms": 1}
            except Exception as e:
                health["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        else:
            health["components"]["redis"] = {"status": "not_initialized"}
            health["status"] = "degraded"

        # Check MongoDB
        mongodb_client = get_mongodb_client()
        if mongodb_client:
            try:
                await mongodb_client.admin.command('ping')
                health["components"]["mongodb"] = {"status": "healthy", "response_time_ms": 2}
            except Exception as e:
                health["components"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "degraded"
        else:
            health["components"]["mongodb"] = {"status": "not_initialized"}
            health["status"] = "degraded"

        # Check Memory Manager
        memory_manager = get_memory_manager()
        if memory_manager:
            health["components"]["memory_manager"] = {"status": "healthy"}
        else:
            health["components"]["memory_manager"] = {"status": "not_initialized"}
            health["status"] = "degraded"

        return health

    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put("/conversations/{session_id}/state")
async def update_conversation_state(
    session_id: str = Path(..., description="Session ID"),
    state_data: Dict[str, Any] = Body(...)
):
    """Update conversation state"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Get current state
        current_state = await memory_manager.get_conversation_state(session_id)
        if not current_state:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update state based on update_type
        update_type = state_data.get("update_type", "merge")

        if update_type == "replace":
            # Replace entire state data
            current_state.update_from_dict(state_data.get("state_data", {}))
        elif update_type == "merge":
            # Merge new data with existing
            current_state.merge_update(state_data.get("state_data", {}))

        current_state.timestamp = datetime.utcnow()
        await memory_manager.update_conversation_state(current_state)

        return {
            "status": "updated",
            "session_id": session_id,
            "update_type": update_type,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/sessions/{session_id}/save")
async def save_session(
    session_id: str = Path(..., description="Session ID"),
    save_data: Dict[str, Any] = Body(...)
):
    """Save comprehensive session data"""
    try:
        from ..main import get_memory_manager, get_redis_client
        memory_manager = get_memory_manager()
        redis_client = get_redis_client()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        save_scope = save_data.get("save_scope", "full")
        compression = save_data.get("compression", False)

        # Collect session data
        session_data = {
            "session_id": session_id,
            "save_timestamp": datetime.utcnow().isoformat(),
            "save_scope": save_scope,
            "data": {}
        }

        # Get conversation state
        conversation_state = await memory_manager.get_conversation_state(session_id)
        if conversation_state:
            session_data["data"]["conversation_state"] = conversation_state.to_dict()

        # Get working memory from Redis
        if redis_client:
            working_memory_key = f"session:{session_id}:state"
            working_memory = await redis_client.hgetall(working_memory_key)
            session_data["data"]["working_memory"] = working_memory

        # Get recent memories
        if save_scope == "full":
            memories = await memory_manager.working_memory.get_session_memories(session_id)
            session_data["data"]["memories"] = [mem.to_dict() for mem in memories]

        # Save to persistent storage (MongoDB)
        from ..main import get_mongodb_client
        mongodb_client = get_mongodb_client()
        if mongodb_client:
            db = mongodb_client.conversation_engine
            collection = db.session_backups

            await collection.replace_one(
                {"session_id": session_id},
                session_data,
                upsert=True
            )

        return {
            "status": "saved",
            "session_id": session_id,
            "save_scope": save_scope,
            "data_size": len(str(session_data)),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error saving session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/sessions/{session_id}/load")
async def load_session(
    session_id: str = Path(..., description="Session ID"),
    load_data: Dict[str, Any] = Body(...)
):
    """Load comprehensive session data"""
    try:
        from ..main import get_memory_manager, get_mongodb_client
        memory_manager = get_memory_manager()
        mongodb_client = get_mongodb_client()

        if not memory_manager or not mongodb_client:
            raise HTTPException(status_code=503, detail="Required services not available")

        load_type = load_data.get("load_type", "latest")
        include_history = load_data.get("include_history", False)

        # Load from persistent storage
        db = mongodb_client.conversation_engine
        collection = db.session_backups

        session_backup = await collection.find_one({"session_id": session_id})
        if not session_backup:
            raise HTTPException(status_code=404, detail="Session backup not found")

        # Restore conversation state
        if "conversation_state" in session_backup["data"]:
            from ..memory.models import ConversationState
            state_data = session_backup["data"]["conversation_state"]
            conversation_state = ConversationState(**state_data)
            await memory_manager.update_conversation_state(conversation_state)

        # Restore working memory to Redis
        if "working_memory" in session_backup["data"]:
            from ..main import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                working_memory_key = f"session:{session_id}:state"
                for key, value in session_backup["data"]["working_memory"].items():
                    await redis_client.hset(working_memory_key, key, value)

        # Restore memories if requested
        restored_memory_count = 0
        if include_history and "memories" in session_backup["data"]:
            from ..memory.models import MemoryItem, MemoryType
            for mem_data in session_backup["data"]["memories"]:
                memory_item = MemoryItem(
                    content=mem_data["content"],
                    timestamp=datetime.fromisoformat(mem_data["timestamp"]),
                    importance_score=mem_data["importance_score"],
                    memory_type=MemoryType(mem_data["memory_type"]),
                    session_id=mem_data["session_id"],
                    agent_id=mem_data.get("agent_id"),
                    context_tags=mem_data.get("context_tags", [])
                )
                await memory_manager.store_memory(memory_item)
                restored_memory_count += 1

        return {
            "status": "loaded",
            "session_id": session_id,
            "load_type": load_type,
            "restored_memories": restored_memory_count,
            "backup_timestamp": session_backup["save_timestamp"],
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/sessions/{session_id}/cleanup")
async def cleanup_session(
    session_id: str = Path(..., description="Session ID"),
    cleanup_data: Dict[str, Any] = Body(...)
):
    """Cleanup old session data"""
    try:
        from ..main import get_memory_manager, get_redis_client, get_mongodb_client
        memory_manager = get_memory_manager()
        redis_client = get_redis_client()
        mongodb_client = get_mongodb_client()

        cleanup_age = cleanup_data.get("cleanup_age", 72)  # hours
        preserve_important = cleanup_data.get("preserve_important", True)

        cleanup_stats = {
            "session_id": session_id,
            "cleanup_age_hours": cleanup_age,
            "preserve_important": preserve_important,
            "deleted": {
                "redis_keys": 0,
                "mongodb_memories": 0,
                "mongodb_states": 0
            }
        }

        cutoff_time = datetime.utcnow() - timedelta(hours=cleanup_age)

        # Cleanup Redis keys
        if redis_client:
            pattern = f"session:{session_id}:*"
            keys = await redis_client.keys(pattern)
            for key in keys:
                # Check if key should be preserved
                if preserve_important and ("important" in key or "agent_state" in key):
                    continue

                await redis_client.delete(key)
                cleanup_stats["deleted"]["redis_keys"] += 1

        # Cleanup MongoDB memories
        if mongodb_client and memory_manager:
            db = mongodb_client.conversation_engine

            # Delete old memories
            memory_filter = {
                "session_id": session_id,
                "timestamp": {"$lt": cutoff_time}
            }
            if preserve_important:
                memory_filter["importance_score"] = {"$lt": 0.7}

            result = await db.memories.delete_many(memory_filter)
            cleanup_stats["deleted"]["mongodb_memories"] = result.deleted_count

            # Cleanup old conversation states (keep latest 5)
            states = await db.conversation_states.find(
                {"session_id": session_id}
            ).sort("timestamp", -1).skip(5).to_list(None)

            if states:
                state_ids = [state["_id"] for state in states]
                result = await db.conversation_states.delete_many(
                    {"_id": {"$in": state_ids}}
                )
                cleanup_stats["deleted"]["mongodb_states"] = result.deleted_count

        return {
            "status": "cleaned",
            "timestamp": datetime.utcnow().isoformat(),
            **cleanup_stats
        }

    except Exception as e:
        logger.error(f"Error cleaning up session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/websocket/broadcast/{session_id}")
async def broadcast_to_session(
    session_id: str = Path(..., description="Session ID"),
    message_data: Dict[str, Any] = Body(...)
):
    """Broadcast message to WebSocket connections for a session"""
    try:
        from ..main import get_connection_manager
        connection_manager = get_connection_manager()

        if not connection_manager:
            raise HTTPException(status_code=503, detail="Connection manager not available")

        await connection_manager.broadcast_to_session(session_id, message_data)

        return {
            "status": "broadcasted",
            "session_id": session_id,
            "message_type": message_data.get("type", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error broadcasting to session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/state/recovery")
async def execute_state_recovery(recovery_data: Dict[str, Any] = Body(...)):
    """Execute state recovery and conflict resolution"""
    try:
        session_id = recovery_data.get("session_id")
        recovery_action = recovery_data.get("recovery_action")

        if not session_id or not recovery_action:
            raise HTTPException(status_code=400, detail="session_id and recovery_action required")

        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Execute recovery based on action type
        if recovery_action == "conflict_resolution":
            resolution_strategy = recovery_data.get("resolution_strategy", "merge_with_precedence")

            # Get current state
            current_state = await memory_manager.get_conversation_state(session_id)
            if current_state:
                # Apply conflict resolution logic
                current_state.timestamp = datetime.utcnow()
                await memory_manager.update_conversation_state(current_state)

                return {
                    "status": "resolved",
                    "session_id": session_id,
                    "resolution_strategy": resolution_strategy,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail="No state found for recovery")

        return {
            "status": "completed",
            "session_id": session_id,
            "recovery_action": recovery_action,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in state recovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/agents/failover")
async def execute_agent_failover(failover_data: Dict[str, Any] = Body(...)):
    """Execute agent failover for error recovery"""
    try:
        session_id = failover_data.get("session_id")
        current_agent = failover_data.get("current_agent")
        backup_agent = failover_data.get("backup_agent")

        if not all([session_id, current_agent, backup_agent]):
            raise HTTPException(status_code=400, detail="session_id, current_agent, and backup_agent required")

        from ..main import get_memory_manager, get_connection_manager
        memory_manager = get_memory_manager()
        connection_manager = get_connection_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Get conversation state
        conversation_state = await memory_manager.get_conversation_state(session_id)
        if not conversation_state:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update active agents
        if current_agent in conversation_state.active_agents and backup_agent not in conversation_state.active_agents:
            conversation_state.active_agents.remove(current_agent)
            conversation_state.active_agents.append(backup_agent)

        # Update last speaker
        conversation_state.last_speaker = backup_agent
        conversation_state.timestamp = datetime.utcnow()

        await memory_manager.update_conversation_state(conversation_state)

        # Notify via WebSocket
        if connection_manager:
            await connection_manager.broadcast_to_session(session_id, {
                "type": "agent_failover",
                "previous_agent": current_agent,
                "new_agent": backup_agent,
                "reason": failover_data.get("reason", "error_recovery"),
                "timestamp": datetime.utcnow().isoformat()
            })

        return {
            "status": "failover_completed",
            "session_id": session_id,
            "previous_agent": current_agent,
            "new_agent": backup_agent,
            "preserve_context": failover_data.get("preserve_context", True),
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in agent failover: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/memory/cleanup")
async def execute_memory_cleanup(cleanup_data: Dict[str, Any] = Body(...)):
    """Execute memory cleanup for error recovery"""
    try:
        session_id = cleanup_data.get("session_id")
        cleanup_threshold = cleanup_data.get("cleanup_threshold", 0.8)
        preserve_important = cleanup_data.get("preserve_important", True)

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Get current memory usage
        memory_stats = await memory_manager.working_memory.get_memory_usage_stats()

        cleanup_stats = {
            "session_id": session_id,
            "cleanup_threshold": cleanup_threshold,
            "preserve_important": preserve_important,
            "before_cleanup": memory_stats,
            "deleted_items": 0
        }

        # Perform cleanup if threshold exceeded
        if memory_stats.get("usage_percentage", 0) > cleanup_threshold:
            # Get memories for cleanup
            memories = await memory_manager.working_memory.get_session_memories(session_id)

            memories_to_delete = []
            for memory in memories:
                # Skip important memories if preserve_important is True
                if preserve_important and memory.importance_score >= 0.7:
                    continue

                # Skip very recent memories (last 10 minutes)
                if (datetime.utcnow() - memory.timestamp).total_seconds() < 600:
                    continue

                memories_to_delete.append(memory)

            # Delete selected memories
            for memory in memories_to_delete:
                await memory_manager.working_memory.remove_memory(memory.memory_id)
                cleanup_stats["deleted_items"] += 1

            # Update stats
            cleanup_stats["after_cleanup"] = await memory_manager.working_memory.get_memory_usage_stats()

        return {
            "status": "cleanup_completed",
            "timestamp": datetime.utcnow().isoformat(),
            **cleanup_stats
        }

    except Exception as e:
        logger.error(f"Error in memory cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/logs/store")
async def store_error_log(log_data: Dict[str, Any] = Body(...)):
    """Store error and recovery logs"""
    try:
        # Add server timestamp
        log_data["server_timestamp"] = datetime.utcnow().isoformat()

        # Store in MongoDB
        from ..main import get_mongodb_client
        mongodb_client = get_mongodb_client()

        if mongodb_client:
            db = mongodb_client.conversation_engine
            collection = db.error_logs

            result = await collection.insert_one(log_data)

            return {
                "status": "logged",
                "log_id": log_data.get("log_id"),
                "mongodb_id": str(result.inserted_id),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Fallback to file logging
            logger.error(f"Error log (no MongoDB): {log_data}")
            return {
                "status": "logged_to_file",
                "log_id": log_data.get("log_id"),
                "timestamp": datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Error storing log: {e}")
        # Don't raise HTTP error for logging failures
        return {
            "status": "log_failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Voice Processing API Models
class VoiceTranscriptionRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    session_id: str
    user_id: Optional[str] = None
    format: str = "wav"
    language: Optional[str] = None


class VoiceSynthesisRequest(BaseModel):
    text: str
    session_id: str
    agent_source: Optional[str] = None
    response_type: str = "direct_answer"
    voice_config: Optional[Dict[str, Any]] = None


class VoiceStreamRequest(BaseModel):
    stream_id: str
    audio_chunk: str  # Base64 encoded audio chunk
    chunk_index: int = 0
    is_final: bool = False
    session_id: str


class VoiceMetricsRequest(BaseModel):
    processing_id: str
    total_processing_time: float
    stt_latency: float
    tts_latency: float
    transcription_confidence: float
    command_type: str
    success: bool
    timestamp: str


# Voice Processing Endpoints
@api_router.post("/voice/transcribe")
async def transcribe_voice(request: VoiceTranscriptionRequest):
    """Transcribe voice audio to text using STT"""
    try:
        import base64
        from ..voice import WhisperSTTProcessor, STTConfiguration, STTModel, TranscriptionQuality

        # Decode audio data
        audio_bytes = base64.b64decode(request.audio_data)

        # Create STT processor with configuration
        stt_config = STTConfiguration(
            model=STTModel.WHISPER_BASE,
            quality=TranscriptionQuality.BALANCED,
            language=request.language
        )

        stt_processor = WhisperSTTProcessor(stt_config)
        await stt_processor.initialize()

        # Transcribe audio
        result = await stt_processor.transcribe_audio(audio_bytes)

        # Cleanup
        await stt_processor.cleanup()

        return {
            "text": result.text,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "language": result.language,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/voice/synthesize")
async def synthesize_voice(request: VoiceSynthesisRequest):
    """Synthesize text to voice using TTS"""
    try:
        import base64
        from ..voice import ElevenLabsTTSProcessor, TTSConfiguration, TTSProvider, TTSQuality

        # Create TTS processor with configuration
        tts_config = TTSConfiguration(
            provider=TTSProvider.ELEVENLABS,
            quality=TTSQuality.BALANCED,
            api_key=settings.ELEVENLABS_API_KEY if hasattr(settings, 'ELEVENLABS_API_KEY') else None
        )

        tts_processor = ElevenLabsTTSProcessor(tts_config)
        await tts_processor.initialize()

        # Get voice profile (simplified for now)
        voice_profiles = await tts_processor.get_available_voices()
        voice_profile = voice_profiles[0] if voice_profiles else None

        # Synthesize speech
        result = await tts_processor.synthesize_text(
            request.text,
            voice_profile=voice_profile
        )

        # Cleanup
        await tts_processor.cleanup()

        # Encode audio data
        audio_data_b64 = base64.b64encode(result.audio_data).decode()

        return {
            "audio_data": audio_data_b64,
            "format": result.format.value,
            "duration": result.duration,
            "processing_time": result.processing_time,
            "voice_profile": {
                "voice_id": result.voice_profile.voice_id,
                "name": result.voice_profile.name
            },
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/voice/broadcast")
async def broadcast_voice(broadcast_data: Dict[str, Any] = Body(...)):
    """Broadcast voice response to WebSocket connections"""
    try:
        from ..main import get_connection_manager
        connection_manager = get_connection_manager()

        if not connection_manager:
            raise HTTPException(status_code=503, detail="Connection manager not available")

        session_id = broadcast_data.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        # Broadcast voice response
        await connection_manager.broadcast_to_session(session_id, {
            "type": "voice_response",
            "audio_data": broadcast_data.get("audio_data"),
            "response_type": broadcast_data.get("response_type"),
            "metadata": broadcast_data.get("metadata", {}),
            "timestamp": datetime.utcnow().isoformat()
        })

        return {
            "status": "broadcasted",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice broadcast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/voice/stream/process")
async def process_voice_stream(request: VoiceStreamRequest):
    """Process streaming voice audio chunk"""
    try:
        import base64

        # Decode audio chunk
        audio_chunk = base64.b64decode(request.audio_chunk)

        # For now, return placeholder response
        # TODO: Implement actual streaming processing with voice pipeline
        return {
            "stream_id": request.stream_id,
            "chunk_index": request.chunk_index,
            "processed": True,
            "partial_text": "Processing..." if not request.is_final else None,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice stream processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/voice/stream/finalize")
async def finalize_voice_stream(finalize_data: Dict[str, Any] = Body(...)):
    """Finalize voice stream and return complete transcription"""
    try:
        stream_id = finalize_data.get("stream_id")
        session_id = finalize_data.get("session_id")
        final_transcription = finalize_data.get("final_transcription", "")

        # Process final transcription as a complete voice command
        # TODO: Integrate with voice pipeline for command processing

        return {
            "stream_id": stream_id,
            "session_id": session_id,
            "final_transcription": final_transcription,
            "response": "Stream processed successfully",
            "audio_data": None,  # Placeholder for synthesized response
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice stream finalization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/metrics/voice")
async def log_voice_metrics(request: VoiceMetricsRequest):
    """Log voice processing performance metrics"""
    try:
        # Store metrics in MongoDB
        from ..main import get_mongodb_client
        mongodb_client = get_mongodb_client()

        metrics_data = {
            **request.dict(),
            "server_timestamp": datetime.utcnow().isoformat()
        }

        if mongodb_client:
            db = mongodb_client.conversation_engine
            collection = db.voice_metrics

            await collection.insert_one(metrics_data)

        return {
            "status": "logged",
            "processing_id": request.processing_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice metrics logging error: {e}")
        # Don't raise error for metrics logging failures
        return {
            "status": "log_failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@api_router.get("/voice/config")
async def get_voice_configuration():
    """Get current voice processing configuration"""
    try:
        # TODO: Integrate with voice configuration manager
        return {
            "stt_model": "whisper_base",
            "tts_provider": "elevenlabs",
            "streaming_enabled": False,
            "personality_adaptation": True,
            "available_voices": [],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice config retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/voice/config")
async def update_voice_configuration(config_data: Dict[str, Any] = Body(...)):
    """Update voice processing configuration"""
    try:
        # TODO: Integrate with voice configuration manager
        return {
            "status": "updated",
            "updated_settings": list(config_data.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Voice config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Conversation processing endpoint for voice commands
@api_router.post("/conversation/process")
async def process_conversation(conversation_data: Dict[str, Any] = Body(...)):
    """Process conversation text (from voice or text input)"""
    try:
        text = conversation_data.get("text", "")
        session_id = conversation_data.get("session_id")
        user_id = conversation_data.get("user_id")
        agent_target = conversation_data.get("agent_target")
        context = conversation_data.get("context", {})

        if not text or not session_id:
            raise HTTPException(status_code=400, detail="text and session_id required")

        # TODO: Integrate with actual conversation processing pipeline
        # For now, return mock response

        mock_responses = {
            "alice": "That's fascinating! Let me think about that from a creative perspective...",
            "bob": "Interesting point. From my analytical viewpoint, I'd say...",
            "charlie": "Oh, that reminds me of something. Let me share a story...",
            "dana": "I understand how you might feel about that. It's important to consider..."
        }

        response_text = mock_responses.get(
            agent_target,
            f"I understand you said: '{text}'. Let me process that thoughtfully."
        )

        return {
            "response": response_text,
            "agent_name": agent_target or "assistant",
            "session_id": session_id,
            "processing_context": context,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Conversation processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent selection endpoint
@api_router.post("/agents/select")
async def select_agent(selection_data: Dict[str, Any] = Body(...)):
    """Select and activate an agent"""
    try:
        agent_name = selection_data.get("agent_name")
        session_id = selection_data.get("session_id")
        user_id = selection_data.get("user_id")
        switch_reason = selection_data.get("switch_reason", "user_request")

        if not agent_name or not session_id:
            raise HTTPException(status_code=400, detail="agent_name and session_id required")

        # TODO: Integrate with actual agent management system
        return {
            "status": "agent_selected",
            "agent_name": agent_name,
            "session_id": session_id,
            "switch_reason": switch_reason,
            "message": f"Switched to {agent_name}. How can I help you?",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Agent selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System control endpoint
@api_router.post("/system/control")
async def system_control(control_data: Dict[str, Any] = Body(...)):
    """Execute system control commands"""
    try:
        action = control_data.get("action")
        session_id = control_data.get("session_id")
        user_id = control_data.get("user_id")

        if not action or not session_id:
            raise HTTPException(status_code=400, detail="action and session_id required")

        # Execute system control action
        response_messages = {
            "stop": "System paused. Say 'resume' to continue.",
            "pause": "System paused. Say 'resume' to continue.",
            "resume": "System resumed. How can I help you?",
            "restart": "System restarted. Starting fresh conversation."
        }

        response_message = response_messages.get(action, f"Executed action: {action}")

        return {
            "status": "executed",
            "action": action,
            "session_id": session_id,
            "message": response_message,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"System control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Home automation endpoint
@api_router.post("/home/control")
async def home_control(control_data: Dict[str, Any] = Body(...)):
    """Execute home automation commands"""
    try:
        device = control_data.get("device")
        action = control_data.get("action")
        location = control_data.get("location", "living room")
        session_id = control_data.get("session_id")

        if not device or not action:
            raise HTTPException(status_code=400, detail="device and action required")

        # TODO: Integrate with actual home automation system
        message = f"Turned {action} the {device} in the {location}."

        return {
            "status": "executed",
            "device": device,
            "action": action,
            "location": location,
            "session_id": session_id,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Home control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))