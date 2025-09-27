"""
ESP32 hardware interface for direct device control and sensor monitoring
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict

import aiohttp
import asyncio_mqtt
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ESP32ConnectionType(str, Enum):
    """ESP32 connection types"""
    HTTP = "http"
    MQTT = "mqtt"
    WEBSOCKET = "websocket"
    SERIAL = "serial"


class SensorType(str, Enum):
    """Sensor types supported by ESP32"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    AIR_QUALITY = "air_quality"
    PROXIMITY = "proximity"
    VIBRATION = "vibration"
    MAGNETIC = "magnetic"
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


class ActuatorType(str, Enum):
    """Actuator types supported by ESP32"""
    LED = "led"
    RELAY = "relay"
    SERVO = "servo"
    STEPPER = "stepper"
    BUZZER = "buzzer"
    SPEAKER = "speaker"
    DISPLAY = "display"
    FAN = "fan"
    HEATER = "heater"
    MOTOR = "motor"


class CommandType(str, Enum):
    """ESP32 command types"""
    READ_SENSOR = "read_sensor"
    CONTROL_ACTUATOR = "control_actuator"
    SET_CONFIG = "set_config"
    GET_STATUS = "get_status"
    RESTART = "restart"
    UPDATE_FIRMWARE = "update_firmware"
    CALIBRATE = "calibrate"


@dataclass
class SensorData:
    """Sensor reading data"""
    sensor_id: str
    sensor_type: SensorType
    value: Union[float, int, str, bool]
    unit: str
    timestamp: datetime
    device_id: str
    accuracy: Optional[float] = None
    calibration_offset: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "accuracy": self.accuracy,
            "calibration_offset": self.calibration_offset,
            "metadata": self.metadata or {}
        }


@dataclass
class ControlCommand:
    """Control command for ESP32 devices"""
    command_id: str
    device_id: str
    command_type: CommandType
    target: str  # sensor/actuator identifier
    action: str  # specific action to perform
    parameters: Dict[str, Any]
    timestamp: datetime
    timeout: float = 30.0
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "device_id": self.device_id,
            "command_type": self.command_type.value,
            "target": self.target,
            "action": self.action,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }


class ESP32Configuration(BaseModel):
    """Configuration for ESP32 device"""
    device_id: str = Field(description="Unique device identifier")
    name: str = Field(description="Human-readable device name")
    host: str = Field(description="ESP32 IP address or hostname")
    port: int = Field(default=80, description="HTTP port")

    # Connection settings
    connection_type: ESP32ConnectionType = Field(default=ESP32ConnectionType.HTTP)
    timeout: float = Field(default=10.0, description="Request timeout")
    retry_attempts: int = Field(default=3, description="Retry attempts")

    # MQTT settings (if using MQTT)
    mqtt_broker: Optional[str] = Field(default=None, description="MQTT broker address")
    mqtt_port: int = Field(default=1883, description="MQTT port")
    mqtt_username: Optional[str] = Field(default=None, description="MQTT username")
    mqtt_password: Optional[str] = Field(default=None, description="MQTT password")
    mqtt_topic_prefix: str = Field(default="homeassistant", description="MQTT topic prefix")

    # Device capabilities
    sensors: List[Dict[str, Any]] = Field(default_factory=list, description="Available sensors")
    actuators: List[Dict[str, Any]] = Field(default_factory=list, description="Available actuators")

    # Security
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    use_ssl: bool = Field(default=False, description="Use SSL/TLS")

    @validator('host')
    def validate_host(cls, v):
        if not v or not v.strip():
            raise ValueError("Host cannot be empty")
        return v.strip()

    @property
    def base_url(self) -> str:
        """Get base URL for HTTP communication"""
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"


class ESP32Device:
    """Individual ESP32 device interface"""

    def __init__(self, config: ESP32Configuration):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.mqtt_client: Optional[asyncio_mqtt.Client] = None

        # Device state
        self.is_connected = False
        self.last_ping = None
        self.sensor_data_cache: Dict[str, SensorData] = {}
        self.command_history: List[ControlCommand] = []

        # Event handlers
        self.sensor_handlers: Dict[str, List[Callable]] = {}
        self.status_handlers: List[Callable] = []

    async def connect(self) -> None:
        """Connect to ESP32 device"""
        try:
            logger.info(f"Connecting to ESP32 device {self.config.device_id} at {self.config.host}")

            if self.config.connection_type == ESP32ConnectionType.HTTP:
                await self._connect_http()
            elif self.config.connection_type == ESP32ConnectionType.MQTT:
                await self._connect_mqtt()
            else:
                raise ValueError(f"Unsupported connection type: {self.config.connection_type}")

            # Test connection
            await self.ping()

            self.is_connected = True
            logger.info(f"Successfully connected to ESP32 device {self.config.device_id}")

        except Exception as e:
            logger.error(f"Failed to connect to ESP32 device {self.config.device_id}: {e}")
            await self.disconnect()
            raise

    async def _connect_http(self) -> None:
        """Connect using HTTP"""
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers
        )

    async def _connect_mqtt(self) -> None:
        """Connect using MQTT"""
        if not self.config.mqtt_broker:
            raise ValueError("MQTT broker not configured")

        self.mqtt_client = asyncio_mqtt.Client(
            hostname=self.config.mqtt_broker,
            port=self.config.mqtt_port,
            username=self.config.mqtt_username,
            password=self.config.mqtt_password
        )

        await self.mqtt_client.__aenter__()

        # Subscribe to device topics
        device_topic = f"{self.config.mqtt_topic_prefix}/{self.config.device_id}/+"
        await self.mqtt_client.subscribe(device_topic)

        # Start message handler
        asyncio.create_task(self._handle_mqtt_messages())

    async def _handle_mqtt_messages(self) -> None:
        """Handle incoming MQTT messages"""
        try:
            async for message in self.mqtt_client.messages:
                try:
                    topic_parts = message.topic.value.split('/')
                    if len(topic_parts) >= 3:
                        device_id = topic_parts[-2]
                        data_type = topic_parts[-1]

                        if device_id == self.config.device_id:
                            payload = json.loads(message.payload.decode())
                            await self._process_mqtt_message(data_type, payload)

                except Exception as e:
                    logger.error(f"Error processing MQTT message: {e}")

        except Exception as e:
            logger.error(f"MQTT message handling error: {e}")

    async def _process_mqtt_message(self, data_type: str, payload: Dict[str, Any]) -> None:
        """Process MQTT message"""
        if data_type == "sensor":
            await self._handle_sensor_data(payload)
        elif data_type == "status":
            await self._handle_status_update(payload)

    async def ping(self) -> bool:
        """Ping device to check connectivity"""
        try:
            if self.config.connection_type == ESP32ConnectionType.HTTP:
                url = f"{self.config.base_url}/ping"
                async with self.session.get(url) as response:
                    success = response.status == 200
            elif self.config.connection_type == ESP32ConnectionType.MQTT:
                # Send ping via MQTT
                ping_topic = f"{self.config.mqtt_topic_prefix}/{self.config.device_id}/ping"
                await self.mqtt_client.publish(ping_topic, json.dumps({"timestamp": datetime.now().isoformat()}))
                success = True
            else:
                success = False

            if success:
                self.last_ping = datetime.now()

            return success

        except Exception as e:
            logger.error(f"Ping failed for device {self.config.device_id}: {e}")
            return False

    async def read_sensor(self, sensor_id: str) -> Optional[SensorData]:
        """Read sensor data"""
        try:
            command = ControlCommand(
                command_id=f"read_{sensor_id}_{int(datetime.now().timestamp())}",
                device_id=self.config.device_id,
                command_type=CommandType.READ_SENSOR,
                target=sensor_id,
                action="read",
                parameters={},
                timestamp=datetime.now()
            )

            response = await self._send_command(command)

            if response and "sensor_data" in response:
                sensor_data = SensorData(
                    sensor_id=sensor_id,
                    sensor_type=SensorType(response["sensor_data"]["type"]),
                    value=response["sensor_data"]["value"],
                    unit=response["sensor_data"]["unit"],
                    timestamp=datetime.fromisoformat(response["sensor_data"]["timestamp"]),
                    device_id=self.config.device_id,
                    accuracy=response["sensor_data"].get("accuracy"),
                    metadata=response["sensor_data"].get("metadata")
                )

                # Cache sensor data
                self.sensor_data_cache[sensor_id] = sensor_data

                # Trigger handlers
                await self._trigger_sensor_handlers(sensor_id, sensor_data)

                return sensor_data

        except Exception as e:
            logger.error(f"Failed to read sensor {sensor_id}: {e}")

        return None

    async def control_actuator(
        self,
        actuator_id: str,
        action: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Control actuator"""
        try:
            command = ControlCommand(
                command_id=f"control_{actuator_id}_{int(datetime.now().timestamp())}",
                device_id=self.config.device_id,
                command_type=CommandType.CONTROL_ACTUATOR,
                target=actuator_id,
                action=action,
                parameters=parameters or {},
                timestamp=datetime.now()
            )

            response = await self._send_command(command)
            return response and response.get("success", False)

        except Exception as e:
            logger.error(f"Failed to control actuator {actuator_id}: {e}")
            return False

    async def _send_command(self, command: ControlCommand) -> Optional[Dict[str, Any]]:
        """Send command to ESP32 device"""
        try:
            self.command_history.append(command)

            if self.config.connection_type == ESP32ConnectionType.HTTP:
                return await self._send_http_command(command)
            elif self.config.connection_type == ESP32ConnectionType.MQTT:
                return await self._send_mqtt_command(command)

        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return None

    async def _send_http_command(self, command: ControlCommand) -> Optional[Dict[str, Any]]:
        """Send command via HTTP"""
        try:
            url = f"{self.config.base_url}/api/command"
            payload = command.to_dict()

            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HTTP command failed: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"HTTP command error: {e}")
            return None

    async def _send_mqtt_command(self, command: ControlCommand) -> Optional[Dict[str, Any]]:
        """Send command via MQTT"""
        try:
            command_topic = f"{self.config.mqtt_topic_prefix}/{self.config.device_id}/command"
            payload = json.dumps(command.to_dict())

            await self.mqtt_client.publish(command_topic, payload)

            # For MQTT, we'll assume success (actual response would come via separate topic)
            return {"success": True, "command_id": command.command_id}

        except Exception as e:
            logger.error(f"MQTT command error: {e}")
            return None

    async def get_status(self) -> Dict[str, Any]:
        """Get device status"""
        try:
            command = ControlCommand(
                command_id=f"status_{int(datetime.now().timestamp())}",
                device_id=self.config.device_id,
                command_type=CommandType.GET_STATUS,
                target="system",
                action="get_status",
                parameters={},
                timestamp=datetime.now()
            )

            response = await self._send_command(command)
            return response or {}

        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return {}

    async def read_all_sensors(self) -> Dict[str, SensorData]:
        """Read all available sensors"""
        sensor_data = {}

        for sensor_config in self.config.sensors:
            sensor_id = sensor_config["id"]
            data = await self.read_sensor(sensor_id)
            if data:
                sensor_data[sensor_id] = data

        return sensor_data

    async def _handle_sensor_data(self, data: Dict[str, Any]) -> None:
        """Handle incoming sensor data"""
        try:
            sensor_data = SensorData(
                sensor_id=data["sensor_id"],
                sensor_type=SensorType(data["sensor_type"]),
                value=data["value"],
                unit=data["unit"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                device_id=self.config.device_id,
                accuracy=data.get("accuracy"),
                metadata=data.get("metadata")
            )

            self.sensor_data_cache[sensor_data.sensor_id] = sensor_data
            await self._trigger_sensor_handlers(sensor_data.sensor_id, sensor_data)

        except Exception as e:
            logger.error(f"Error handling sensor data: {e}")

    async def _handle_status_update(self, data: Dict[str, Any]) -> None:
        """Handle status update"""
        for handler in self.status_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Status handler error: {e}")

    def register_sensor_handler(self, sensor_id: str, handler: Callable) -> None:
        """Register sensor data handler"""
        if sensor_id not in self.sensor_handlers:
            self.sensor_handlers[sensor_id] = []
        self.sensor_handlers[sensor_id].append(handler)

    def register_status_handler(self, handler: Callable) -> None:
        """Register status update handler"""
        self.status_handlers.append(handler)

    async def _trigger_sensor_handlers(self, sensor_id: str, data: SensorData) -> None:
        """Trigger sensor data handlers"""
        if sensor_id in self.sensor_handlers:
            for handler in self.sensor_handlers[sensor_id]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Sensor handler error: {e}")

    async def disconnect(self) -> None:
        """Disconnect from ESP32 device"""
        logger.info(f"Disconnecting from ESP32 device {self.config.device_id}")

        self.is_connected = False

        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")
            self.session = None

        if self.mqtt_client:
            try:
                await self.mqtt_client.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing MQTT client: {e}")
            self.mqtt_client = None

        logger.info(f"Disconnected from ESP32 device {self.config.device_id}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


class ESP32Interface:
    """Manager for multiple ESP32 devices"""

    def __init__(self):
        self.devices: Dict[str, ESP32Device] = {}
        self.device_configs: Dict[str, ESP32Configuration] = {}

        # Global event handlers
        self.global_sensor_handlers: List[Callable] = []
        self.device_status_handlers: List[Callable] = []

    async def add_device(self, config: ESP32Configuration) -> ESP32Device:
        """Add ESP32 device"""
        try:
            device = ESP32Device(config)
            await device.connect()

            self.devices[config.device_id] = device
            self.device_configs[config.device_id] = config

            # Register global handlers
            device.register_status_handler(self._handle_device_status)

            logger.info(f"Added ESP32 device: {config.device_id}")
            return device

        except Exception as e:
            logger.error(f"Failed to add ESP32 device {config.device_id}: {e}")
            raise

    async def remove_device(self, device_id: str) -> None:
        """Remove ESP32 device"""
        if device_id in self.devices:
            device = self.devices[device_id]
            await device.disconnect()

            del self.devices[device_id]
            del self.device_configs[device_id]

            logger.info(f"Removed ESP32 device: {device_id}")

    def get_device(self, device_id: str) -> Optional[ESP32Device]:
        """Get ESP32 device by ID"""
        return self.devices.get(device_id)

    async def read_sensor_from_device(self, device_id: str, sensor_id: str) -> Optional[SensorData]:
        """Read sensor from specific device"""
        device = self.get_device(device_id)
        if device:
            return await device.read_sensor(sensor_id)
        return None

    async def control_actuator_on_device(
        self,
        device_id: str,
        actuator_id: str,
        action: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Control actuator on specific device"""
        device = self.get_device(device_id)
        if device:
            return await device.control_actuator(actuator_id, action, parameters)
        return False

    async def read_all_sensors(self) -> Dict[str, Dict[str, SensorData]]:
        """Read all sensors from all devices"""
        all_sensor_data = {}

        for device_id, device in self.devices.items():
            try:
                device_data = await device.read_all_sensors()
                if device_data:
                    all_sensor_data[device_id] = device_data
            except Exception as e:
                logger.error(f"Failed to read sensors from device {device_id}: {e}")

        return all_sensor_data

    async def ping_all_devices(self) -> Dict[str, bool]:
        """Ping all devices"""
        ping_results = {}

        for device_id, device in self.devices.items():
            try:
                ping_results[device_id] = await device.ping()
            except Exception as e:
                logger.error(f"Failed to ping device {device_id}: {e}")
                ping_results[device_id] = False

        return ping_results

    def get_device_list(self) -> List[Dict[str, Any]]:
        """Get list of all devices"""
        device_list = []

        for device_id, config in self.device_configs.items():
            device = self.devices.get(device_id)
            device_info = {
                "device_id": device_id,
                "name": config.name,
                "host": config.host,
                "connection_type": config.connection_type.value,
                "is_connected": device.is_connected if device else False,
                "last_ping": device.last_ping.isoformat() if device and device.last_ping else None,
                "sensors": len(config.sensors),
                "actuators": len(config.actuators)
            }
            device_list.append(device_info)

        return device_list

    def register_global_sensor_handler(self, handler: Callable) -> None:
        """Register global sensor data handler"""
        self.global_sensor_handlers.append(handler)

    def register_device_status_handler(self, handler: Callable) -> None:
        """Register device status handler"""
        self.device_status_handlers.append(handler)

    async def _handle_device_status(self, status_data: Dict[str, Any]) -> None:
        """Handle device status updates"""
        for handler in self.device_status_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(status_data)
                else:
                    handler(status_data)
            except Exception as e:
                logger.error(f"Device status handler error: {e}")

    async def shutdown(self) -> None:
        """Shutdown all devices"""
        logger.info("Shutting down ESP32 interface")

        for device_id in list(self.devices.keys()):
            await self.remove_device(device_id)

        logger.info("ESP32 interface shutdown complete")


# Utility functions
def create_esp32_device_config(
    device_id: str,
    name: str,
    host: str,
    sensors: List[Dict[str, Any]] = None,
    actuators: List[Dict[str, Any]] = None,
    **kwargs
) -> ESP32Configuration:
    """Create ESP32 device configuration"""
    return ESP32Configuration(
        device_id=device_id,
        name=name,
        host=host,
        sensors=sensors or [],
        actuators=actuators or [],
        **kwargs
    )