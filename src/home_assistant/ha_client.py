"""
Home Assistant client for API communication and device control
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum

import aiohttp
import websockets
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class HAConnectionError(Exception):
    """Home Assistant connection error"""
    pass


class HAAuthenticationError(Exception):
    """Home Assistant authentication error"""
    pass


class HAApiError(Exception):
    """Home Assistant API error"""
    pass


class HAEntityState(str, Enum):
    """Home Assistant entity states"""
    ON = "on"
    OFF = "off"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    IDLE = "idle"
    ACTIVE = "active"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    OPEN = "open"
    CLOSED = "closed"


class HAConfiguration(BaseModel):
    """Configuration for Home Assistant connection"""
    host: str = Field(description="Home Assistant host (e.g., 192.168.1.100)")
    port: int = Field(default=8123, description="Home Assistant port")
    ssl: bool = Field(default=False, description="Use HTTPS")
    token: str = Field(description="Long-lived access token")

    # Connection settings
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")

    # WebSocket settings
    websocket_heartbeat: float = Field(default=30.0, description="WebSocket heartbeat interval")
    reconnect_attempts: int = Field(default=10, description="WebSocket reconnection attempts")

    @validator('host')
    def validate_host(cls, v):
        if not v or not v.strip():
            raise ValueError("Host cannot be empty")
        return v.strip()

    @validator('token')
    def validate_token(cls, v):
        if not v or len(v) < 10:
            raise ValueError("Token must be at least 10 characters long")
        return v

    @property
    def base_url(self) -> str:
        """Get base URL for Home Assistant"""
        protocol = "https" if self.ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL for Home Assistant"""
        protocol = "wss" if self.ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}/api/websocket"


class HomeAssistantClient:
    """Home Assistant API client with WebSocket support"""

    def __init__(self, config: HAConfiguration):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None

        # Connection state
        self.is_connected = False
        self.is_authenticated = False
        self.websocket_id = 1

        # Event handlers
        self.event_handlers: Dict[str, List[callable]] = {}

        # State cache
        self.entity_states: Dict[str, Dict[str, Any]] = {}
        self.last_update = datetime.now()

    async def connect(self) -> None:
        """Connect to Home Assistant"""
        try:
            logger.info(f"Connecting to Home Assistant at {self.config.base_url}")

            # Create HTTP session
            connector = aiohttp.TCPConnector(
                ssl=False if not self.config.ssl else None
            )
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.token}",
                    "Content-Type": "application/json"
                }
            )

            # Test connection
            await self._test_connection()

            # Connect WebSocket
            await self._connect_websocket()

            self.is_connected = True
            logger.info("Successfully connected to Home Assistant")

        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            await self.disconnect()
            raise HAConnectionError(f"Connection failed: {e}")

    async def _test_connection(self) -> None:
        """Test HTTP API connection"""
        try:
            url = f"{self.config.base_url}/api/"
            async with self.session.get(url) as response:
                if response.status == 401:
                    raise HAAuthenticationError("Invalid access token")
                elif response.status != 200:
                    raise HAConnectionError(f"HTTP {response.status}: {await response.text()}")

                data = await response.json()
                logger.info(f"Connected to Home Assistant {data.get('version', 'unknown')}")

        except aiohttp.ClientError as e:
            raise HAConnectionError(f"HTTP connection error: {e}")

    async def _connect_websocket(self) -> None:
        """Connect WebSocket for real-time updates"""
        try:
            logger.info("Connecting WebSocket...")

            self.websocket = await websockets.connect(
                self.config.websocket_url,
                extra_headers={"Authorization": f"Bearer {self.config.token}"}
            )

            # Handle authentication
            auth_message = await self.websocket.recv()
            auth_data = json.loads(auth_message)

            if auth_data.get("type") != "auth_required":
                raise HAAuthenticationError("Unexpected authentication flow")

            # Send authentication
            await self.websocket.send(json.dumps({
                "type": "auth",
                "access_token": self.config.token
            }))

            # Verify authentication
            auth_result = await self.websocket.recv()
            auth_result_data = json.loads(auth_result)

            if auth_result_data.get("type") != "auth_ok":
                raise HAAuthenticationError("WebSocket authentication failed")

            self.is_authenticated = True
            logger.info("WebSocket authenticated successfully")

            # Subscribe to state changes
            await self._subscribe_to_events()

            # Start message handling
            asyncio.create_task(self._handle_websocket_messages())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise HAConnectionError(f"WebSocket error: {e}")

    async def _subscribe_to_events(self) -> None:
        """Subscribe to state change events"""
        try:
            subscribe_message = {
                "id": self._get_next_id(),
                "type": "subscribe_events",
                "event_type": "state_changed"
            }

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to state change events")

        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
            self.is_authenticated = False
        except Exception as e:
            logger.error(f"WebSocket message handling error: {e}")

    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Process individual WebSocket message"""
        message_type = data.get("type")

        if message_type == "event":
            event_data = data.get("event", {})
            event_type = event_data.get("event_type")

            if event_type == "state_changed":
                await self._handle_state_change(event_data.get("data", {}))

        elif message_type == "result":
            # Handle command results
            success = data.get("success", False)
            if not success:
                error = data.get("error", {})
                logger.error(f"Command failed: {error}")

    async def _handle_state_change(self, data: Dict[str, Any]) -> None:
        """Handle entity state change"""
        entity_id = data.get("entity_id")
        new_state = data.get("new_state")

        if entity_id and new_state:
            # Update local cache
            self.entity_states[entity_id] = new_state
            self.last_update = datetime.now()

            # Trigger event handlers
            await self._trigger_event_handlers("state_changed", {
                "entity_id": entity_id,
                "new_state": new_state,
                "old_state": data.get("old_state")
            })

    def _get_next_id(self) -> int:
        """Get next WebSocket message ID"""
        current_id = self.websocket_id
        self.websocket_id += 1
        return current_id

    async def get_states(self, entity_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get entity states"""
        if not self.is_connected:
            raise HAConnectionError("Not connected to Home Assistant")

        try:
            if entity_id:
                url = f"{self.config.base_url}/api/states/{entity_id}"
            else:
                url = f"{self.config.base_url}/api/states"

            async with self.session.get(url) as response:
                if response.status == 404:
                    raise HAApiError(f"Entity not found: {entity_id}")
                elif response.status != 200:
                    raise HAApiError(f"API error: {response.status}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise HAConnectionError(f"Failed to get states: {e}")

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call Home Assistant service"""
        if not self.is_connected:
            raise HAConnectionError("Not connected to Home Assistant")

        try:
            url = f"{self.config.base_url}/api/services/{domain}/{service}"

            payload = {}
            if entity_id:
                payload["entity_id"] = entity_id
            if data:
                payload.update(data)

            async with self.session.post(url, json=payload) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise HAApiError(f"Service call failed: {response.status} - {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise HAConnectionError(f"Service call failed: {e}")

    async def turn_on(self, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Turn on an entity"""
        domain = entity_id.split(".")[0]
        return await self.call_service(domain, "turn_on", entity_id, kwargs)

    async def turn_off(self, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Turn off an entity"""
        domain = entity_id.split(".")[0]
        return await self.call_service(domain, "turn_off", entity_id, kwargs)

    async def toggle(self, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Toggle an entity"""
        domain = entity_id.split(".")[0]
        return await self.call_service(domain, "toggle", entity_id, kwargs)

    async def set_state(
        self,
        entity_id: str,
        state: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Set entity state"""
        if not self.is_connected:
            raise HAConnectionError("Not connected to Home Assistant")

        try:
            url = f"{self.config.base_url}/api/states/{entity_id}"

            payload = {"state": state}
            if attributes:
                payload["attributes"] = attributes

            async with self.session.post(url, json=payload) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise HAApiError(f"Set state failed: {response.status} - {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise HAConnectionError(f"Set state failed: {e}")

    async def get_services(self) -> Dict[str, Any]:
        """Get available services"""
        if not self.is_connected:
            raise HAConnectionError("Not connected to Home Assistant")

        try:
            url = f"{self.config.base_url}/api/services"

            async with self.session.get(url) as response:
                if response.status != 200:
                    raise HAApiError(f"Failed to get services: {response.status}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise HAConnectionError(f"Failed to get services: {e}")

    async def get_config(self) -> Dict[str, Any]:
        """Get Home Assistant configuration"""
        if not self.is_connected:
            raise HAConnectionError("Not connected to Home Assistant")

        try:
            url = f"{self.config.base_url}/api/config"

            async with self.session.get(url) as response:
                if response.status != 200:
                    raise HAApiError(f"Failed to get config: {response.status}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise HAConnectionError(f"Failed to get config: {e}")

    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def unregister_event_handler(self, event_type: str, handler: callable) -> None:
        """Unregister event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def _trigger_event_handlers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger event handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

    async def get_cached_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get cached entity state"""
        return self.entity_states.get(entity_id)

    async def wait_for_state(
        self,
        entity_id: str,
        target_state: str,
        timeout: float = 30.0
    ) -> bool:
        """Wait for entity to reach target state"""
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            current_state = await self.get_cached_state(entity_id)

            if current_state and current_state.get("state") == target_state:
                return True

            await asyncio.sleep(0.5)

        return False

    async def is_entity_available(self, entity_id: str) -> bool:
        """Check if entity is available"""
        try:
            state = await self.get_states(entity_id)
            return state.get("state") != "unavailable"
        except:
            return False

    async def get_entity_attributes(self, entity_id: str) -> Dict[str, Any]:
        """Get entity attributes"""
        try:
            state = await self.get_states(entity_id)
            return state.get("attributes", {})
        except:
            return {}

    async def disconnect(self) -> None:
        """Disconnect from Home Assistant"""
        logger.info("Disconnecting from Home Assistant")

        self.is_connected = False
        self.is_authenticated = False

        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            self.websocket = None

        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")
            self.session = None

        logger.info("Disconnected from Home Assistant")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# Utility functions
def create_ha_client(
    host: str,
    token: str,
    port: int = 8123,
    ssl: bool = False
) -> HomeAssistantClient:
    """Create Home Assistant client with basic configuration"""
    config = HAConfiguration(
        host=host,
        port=port,
        ssl=ssl,
        token=token
    )
    return HomeAssistantClient(config)