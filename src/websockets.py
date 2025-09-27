"""
WebSocket connection management for real-time conversation updates
"""

import logging
import json
import asyncio
from typing import Dict, List, Set
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time conversation updates"""

    def __init__(self):
        # Map of session_id to list of WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Set of all active WebSocket connections
        self.all_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection and add it to the session"""
        await websocket.accept()

        # Add to session connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []

        self.active_connections[session_id].append(websocket)
        self.all_connections.add(websocket)

        logger.info(f"WebSocket connected for session {session_id}. Total connections: {len(self.all_connections)}")

        # Send welcome message
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to Multi-Agent Conversation Engine"
        })

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection"""
        # Remove from session connections
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

            # Clean up empty session lists
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

        # Remove from all connections
        self.all_connections.discard(websocket)

        logger.info(f"WebSocket disconnected for session {session_id}. Total connections: {len(self.all_connections)}")

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Error sending personal message: {e}")
            # Connection might be closed, remove it
            self.all_connections.discard(websocket)

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast a message to all connections in a session"""
        if session_id not in self.active_connections:
            logger.debug(f"No active connections for session {session_id}")
            return

        connections = self.active_connections[session_id].copy()
        disconnected_connections = []

        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Error broadcasting to session {session_id}: {e}")
                disconnected_connections.append(connection)

        # Clean up disconnected connections
        for connection in disconnected_connections:
            self.disconnect(connection, session_id)

        logger.debug(f"Broadcasted message to {len(connections) - len(disconnected_connections)} connections in session {session_id}")

    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections"""
        if not self.all_connections:
            logger.debug("No active connections for broadcast")
            return

        connections = self.all_connections.copy()
        disconnected_connections = []

        for connection in connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Error broadcasting to all connections: {e}")
                disconnected_connections.append(connection)

        # Clean up disconnected connections
        for connection in disconnected_connections:
            self.all_connections.discard(connection)
            # Remove from session connections too
            for session_id, session_connections in self.active_connections.items():
                if connection in session_connections:
                    session_connections.remove(connection)

        logger.debug(f"Broadcasted message to {len(connections) - len(disconnected_connections)} connections")

    async def send_error(self, websocket: WebSocket, error_message: str):
        """Send an error message to a specific connection"""
        error_data = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_personal_message(websocket, error_data)

    async def send_typing_indicator(self, session_id: str, agent_id: str, is_typing: bool = True):
        """Send typing indicator for an agent"""
        typing_message = {
            "type": "agent_typing",
            "agent_id": agent_id,
            "is_typing": is_typing,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_to_session(session_id, typing_message)

    async def send_conversation_state_update(self, session_id: str, state_data: dict):
        """Send conversation state update to session"""
        state_message = {
            "type": "conversation_state_update",
            "state": state_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_to_session(session_id, state_message)

    async def send_memory_update(self, session_id: str, memory_data: dict):
        """Send memory system update to session"""
        memory_message = {
            "type": "memory_update",
            "memory": memory_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_to_session(session_id, memory_message)

    async def send_agent_response(self, session_id: str, agent_id: str, response: str, metadata: dict = None):
        """Send agent response to session"""
        response_message = {
            "type": "agent_response",
            "agent_id": agent_id,
            "content": response,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_to_session(session_id, response_message)

    async def send_system_notification(self, session_id: str, notification: str, notification_type: str = "info"):
        """Send system notification to session"""
        notification_message = {
            "type": "system_notification",
            "notification_type": notification_type,
            "message": notification,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast_to_session(session_id, notification_message)

    def get_session_connection_count(self, session_id: str) -> int:
        """Get the number of active connections for a session"""
        return len(self.active_connections.get(session_id, []))

    def get_total_connection_count(self) -> int:
        """Get the total number of active connections"""
        return len(self.all_connections)

    def get_active_sessions(self) -> List[str]:
        """Get list of session IDs with active connections"""
        return list(self.active_connections.keys())

    def get_connection_stats(self) -> dict:
        """Get detailed connection statistics"""
        stats = {
            "total_connections": self.get_total_connection_count(),
            "active_sessions": len(self.active_connections),
            "sessions": {}
        }

        for session_id, connections in self.active_connections.items():
            stats["sessions"][session_id] = {
                "connection_count": len(connections),
                "connections": [str(id(conn)) for conn in connections]
            }

        return stats

    async def cleanup_stale_connections(self):
        """Remove stale/closed connections"""
        stale_connections = []

        for connection in self.all_connections.copy():
            try:
                # Try to send a ping message to check if connection is alive
                await connection.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception:
                stale_connections.append(connection)

        # Clean up stale connections
        for connection in stale_connections:
            self.all_connections.discard(connection)

            # Remove from session connections
            for session_id, session_connections in self.active_connections.items():
                if connection in session_connections:
                    session_connections.remove(connection)

        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale WebSocket connections")

    async def start_periodic_cleanup(self, interval_seconds: int = 300):
        """Start periodic cleanup of stale connections"""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self.cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error in periodic connection cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying