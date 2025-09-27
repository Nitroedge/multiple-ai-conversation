"""
FastAPI router for webhook endpoints
"""

import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Form, File, UploadFile
from fastapi.responses import JSONResponse
import httpx
import base64

from ..config import get_settings
from ..memory.models import MemoryItem, MemoryType, ConversationState

logger = logging.getLogger(__name__)
webhook_router = APIRouter()

settings = get_settings()


@webhook_router.post("/voice-command")
async def voice_command_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    audio_data: Optional[str] = Form(None),  # Base64 encoded audio
    audio_file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),  # Direct text input for testing
    user_id: Optional[str] = Form(None)
):
    """
    n8n webhook endpoint for processing voice commands
    Handles both audio files and direct text input
    """
    try:
        logger.info(f"Voice command webhook triggered for session {session_id}")

        # Get memory manager from main app
        from ..main import get_memory_manager, get_connection_manager
        memory_manager = get_memory_manager()
        connection_manager = get_connection_manager()

        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory manager not available")

        # Process input (audio or text)
        input_text = ""
        audio_metadata = {}

        if text_input:
            # Direct text input (for testing)
            input_text = text_input
            logger.info(f"Processing text input: {text_input[:100]}...")

        elif audio_file:
            # Process uploaded audio file
            audio_content = await audio_file.read()
            input_text, audio_metadata = await process_audio_input(audio_content)

        elif audio_data:
            # Process base64 encoded audio
            try:
                audio_bytes = base64.b64decode(audio_data)
                input_text, audio_metadata = await process_audio_input(audio_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="No input provided (audio_file, audio_data, or text_input required)")

        if not input_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from input")

        # Store user input in memory
        user_memory = MemoryItem(
            content=input_text,
            timestamp=datetime.utcnow(),
            importance_score=0.6,  # User input has moderate importance
            memory_type=MemoryType.EPISODIC,
            session_id=session_id,
            agent_id=None,  # Human user
            context_tags=["voice_input", "user_message"]
        )

        await memory_manager.store_memory(user_memory)

        # Update conversation state
        conversation_state = await memory_manager.get_conversation_state(session_id)
        if not conversation_state:
            # Create new conversation state
            conversation_state = ConversationState(
                session_id=session_id,
                active_agents=["OSWALD", "TONY_KING", "VICTORIA"],
                conversation_stage="discussion",
                topic_focus="general_conversation",
                emotion_context={},
                last_speaker="user",
                turn_count=1,
                user_preferences={},
                timestamp=datetime.utcnow()
            )
        else:
            conversation_state.last_speaker = "user"
            conversation_state.turn_count += 1
            conversation_state.timestamp = datetime.utcnow()

        await memory_manager.update_conversation_state(conversation_state)

        # Trigger agent response in background
        background_tasks.add_task(
            trigger_agent_response,
            session_id,
            input_text,
            user_id or "anonymous"
        )

        # Broadcast to WebSocket connections
        if connection_manager:
            await connection_manager.broadcast_to_session(
                session_id,
                {
                    "type": "user_message",
                    "content": input_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "audio_metadata": audio_metadata
                }
            )

        # Return immediate response for n8n
        response_data = {
            "status": "received",
            "session_id": session_id,
            "message": "Voice command processed successfully",
            "input_text": input_text,
            "audio_metadata": audio_metadata,
            "timestamp": datetime.utcnow().isoformat()
        }

        return JSONResponse(content=response_data, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice command webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@webhook_router.post("/agent-coordination")
async def agent_coordination_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    n8n webhook endpoint for coordinating agent responses
    """
    try:
        body = await request.json()
        session_id = body.get("session_id")
        agent_id = body.get("agent_id")
        context = body.get("context", {})

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        logger.info(f"Agent coordination webhook triggered for session {session_id}, agent {agent_id}")

        # Process agent coordination in background
        background_tasks.add_task(
            coordinate_agent_response,
            session_id,
            agent_id,
            context
        )

        return {
            "status": "coordinating",
            "session_id": session_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in agent coordination webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@webhook_router.post("/home-automation")
async def home_automation_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    n8n webhook endpoint for executing home automation commands
    """
    try:
        body = await request.json()
        session_id = body.get("session_id")
        command = body.get("command")
        device_type = body.get("device_type")
        parameters = body.get("parameters", {})

        logger.info(f"Home automation webhook: {command} for {device_type}")

        # Process home automation in background
        background_tasks.add_task(
            execute_home_automation,
            session_id,
            command,
            device_type,
            parameters
        )

        return {
            "status": "executing",
            "command": command,
            "device_type": device_type,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in home automation webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@webhook_router.post("/memory-consolidation")
async def memory_consolidation_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    n8n webhook endpoint for triggering memory consolidation
    """
    try:
        body = await request.json()
        session_ids = body.get("session_ids", [])
        strategy = body.get("strategy", "clustering")

        logger.info(f"Memory consolidation webhook triggered for {len(session_ids)} sessions")

        # Process memory consolidation in background
        background_tasks.add_task(
            consolidate_memories,
            session_ids,
            strategy
        )

        return {
            "status": "consolidating",
            "session_count": len(session_ids),
            "strategy": strategy,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in memory consolidation webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_audio_input(audio_data: bytes) -> tuple[str, dict]:
    """Process audio input through STT"""
    try:
        # For now, mock STT processing
        # TODO: Integrate with OpenAI Whisper or local STT

        # Save audio file temporarily
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            # Mock STT response for development
            if settings.mock_ai_responses:
                text_result = "This is a mock transcription of the audio input."
                confidence = 0.95
            else:
                # TODO: Implement actual STT processing
                # text_result, confidence = await transcribe_audio(temp_path)
                text_result = "STT processing not yet implemented"
                confidence = 0.0

            audio_metadata = {
                "duration_ms": len(audio_data) * 8,  # Rough estimate
                "size_bytes": len(audio_data),
                "confidence": confidence,
                "processing_time_ms": 100  # Mock processing time
            }

            return text_result, audio_metadata

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error processing audio input: {e}")
        return "", {"error": str(e)}


async def trigger_agent_response(session_id: str, user_input: str, user_id: str):
    """Trigger agent response through n8n workflow"""
    try:
        # Prepare payload for n8n agent coordination workflow
        payload = {
            "session_id": session_id,
            "user_input": user_input,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "trigger_source": "voice_command_webhook"
        }

        # Send to n8n agent coordination workflow
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.n8n_webhook_url}/agent-coordination",
                json=payload,
                timeout=30.0
            )

            if response.status_code == 200:
                logger.info(f"Successfully triggered agent response for session {session_id}")
            else:
                logger.warning(f"Agent coordination request failed: {response.status_code}")

    except Exception as e:
        logger.error(f"Error triggering agent response: {e}")


async def coordinate_agent_response(session_id: str, agent_id: Optional[str], context: dict):
    """Coordinate response from a specific agent or select one automatically"""
    try:
        from ..main import get_memory_manager, get_connection_manager
        memory_manager = get_memory_manager()
        connection_manager = get_connection_manager()

        if not memory_manager:
            logger.error("Memory manager not available for agent coordination")
            return

        # Get conversation context
        conversation_state = await memory_manager.get_conversation_state(session_id)
        if not conversation_state:
            logger.warning(f"No conversation state found for session {session_id}")
            return

        # Select agent if not specified
        if not agent_id:
            # Simple round-robin selection for now
            # TODO: Implement intelligent agent selection based on context
            active_agents = conversation_state.active_agents
            last_speaker = conversation_state.last_speaker

            if last_speaker in active_agents:
                current_index = active_agents.index(last_speaker)
                next_index = (current_index + 1) % len(active_agents)
                agent_id = active_agents[next_index]
            else:
                agent_id = active_agents[0] if active_agents else "OSWALD"

        # Generate agent response
        agent_response = await generate_agent_response(agent_id, session_id, context)

        if agent_response:
            # Update conversation state
            conversation_state.last_speaker = agent_id
            conversation_state.turn_count += 1
            conversation_state.timestamp = datetime.utcnow()
            await memory_manager.update_conversation_state(conversation_state)

            # Broadcast response to WebSocket connections
            await connection_manager.broadcast_to_session(
                session_id,
                {
                    "type": "agent_response",
                    "agent_id": agent_id,
                    "content": agent_response,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

    except Exception as e:
        logger.error(f"Error in agent coordination: {e}")


async def generate_agent_response(agent_id: str, session_id: str, context: dict) -> Optional[str]:
    """Generate response from specified agent"""
    try:
        # For now, return mock responses
        # TODO: Implement actual agent response generation with character personalities

        mock_responses = {
            "OSWALD": "OH WOW! That's absolutely fascinating! I'm so excited to explore this topic further!",
            "TONY_KING": "Listen here, pal, that's exactly the kinda thing my cousin Vinny from Brooklyn would say!",
            "VICTORIA": "From a philosophical perspective, this raises some profound questions about the nature of reality."
        }

        response = mock_responses.get(agent_id, "I'm processing your message...")

        # Store agent response in memory
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if memory_manager:
            agent_memory = MemoryItem(
                content=response,
                timestamp=datetime.utcnow(),
                importance_score=0.7,  # Agent responses have higher importance
                memory_type=MemoryType.EPISODIC,
                session_id=session_id,
                agent_id=agent_id,
                context_tags=["agent_response", agent_id.lower()]
            )
            await memory_manager.store_memory(agent_memory)

        return response

    except Exception as e:
        logger.error(f"Error generating agent response: {e}")
        return None


async def execute_home_automation(session_id: str, command: str, device_type: str, parameters: dict):
    """Execute home automation command"""
    try:
        logger.info(f"Executing home automation: {command} on {device_type}")

        # For now, mock home automation execution
        # TODO: Integrate with actual Home Assistant API

        result = {
            "status": "executed",
            "command": command,
            "device_type": device_type,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Store automation result in memory
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if memory_manager:
            automation_memory = MemoryItem(
                content=f"Executed {command} on {device_type}",
                timestamp=datetime.utcnow(),
                importance_score=0.8,  # Home automation has high importance
                memory_type=MemoryType.PROCEDURAL,
                session_id=session_id,
                agent_id="system",
                context_tags=["home_automation", device_type, command]
            )
            await memory_manager.store_memory(automation_memory)

        logger.info(f"Home automation completed: {result}")

    except Exception as e:
        logger.error(f"Error executing home automation: {e}")


async def consolidate_memories(session_ids: list, strategy: str):
    """Consolidate memories for specified sessions"""
    try:
        from ..main import get_memory_manager
        memory_manager = get_memory_manager()

        if not memory_manager:
            logger.error("Memory manager not available for consolidation")
            return

        consolidated_count = 0

        for session_id in session_ids:
            try:
                # Get memories for consolidation
                memories = await memory_manager.working_memory.get_memories_for_consolidation(
                    session_id,
                    importance_threshold=0.7
                )

                if memories:
                    # Run consolidation
                    result = await memory_manager.consolidation_engine.consolidate_memories(
                        memories,
                        strategy=strategy
                    )

                    consolidated_count += result.consolidated_count
                    logger.info(f"Consolidated {result.original_count} to {result.consolidated_count} memories for session {session_id}")

            except Exception as e:
                logger.error(f"Error consolidating memories for session {session_id}: {e}")

        logger.info(f"Memory consolidation completed: {consolidated_count} total memories consolidated")

    except Exception as e:
        logger.error(f"Error in memory consolidation: {e}")