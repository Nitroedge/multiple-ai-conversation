"""
Device management system for Home Assistant integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field

from .ha_client import HomeAssistantClient, HAEntityState
from .esp32_interface import ESP32Interface, ESP32Device, SensorData

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Device types"""
    LIGHT = "light"
    SWITCH = "switch"
    FAN = "fan"
    CLIMATE = "climate"
    SENSOR = "sensor"
    BINARY_SENSOR = "binary_sensor"
    CAMERA = "camera"
    MEDIA_PLAYER = "media_player"
    COVER = "cover"
    LOCK = "lock"
    VACUUM = "vacuum"
    WATER_HEATER = "water_heater"
    ALARM_CONTROL_PANEL = "alarm_control_panel"
    ESP32_DEVICE = "esp32_device"


class DeviceCapability(str, Enum):
    """Device capabilities"""
    ON_OFF = "on_off"
    BRIGHTNESS = "brightness"
    COLOR = "color"
    TEMPERATURE = "temperature"
    POSITION = "position"
    VOLUME = "volume"
    SPEED = "speed"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    MOTION = "motion"
    CONTACT = "contact"
    SMOKE = "smoke"
    BATTERY = "battery"


class DeviceState(str, Enum):
    """Device states"""
    ONLINE = "online"
    OFFLINE = "offline"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class Device:
    """Device representation"""
    entity_id: str
    name: str
    device_type: DeviceType
    capabilities: List[DeviceCapability]
    state: DeviceState
    attributes: Dict[str, Any]
    location: Optional[str] = None
    last_seen: Optional[datetime] = None
    ha_entity: bool = True  # True for HA entities, False for ESP32 devices
    esp32_device_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "device_type": self.device_type.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "state": self.state.value,
            "attributes": self.attributes,
            "location": self.location,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "ha_entity": self.ha_entity,
            "esp32_device_id": self.esp32_device_id,
            "metadata": self.metadata or {}
        }

    def has_capability(self, capability: DeviceCapability) -> bool:
        """Check if device has specific capability"""
        return capability in self.capabilities

    def is_available(self) -> bool:
        """Check if device is available"""
        return self.state in [DeviceState.ONLINE, DeviceState.UNKNOWN]

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get device attribute"""
        return self.attributes.get(key, default)


class DeviceManager:
    """Manages all smart home devices"""

    def __init__(self, ha_client: HomeAssistantClient, esp32_interface: ESP32Interface):
        self.ha_client = ha_client
        self.esp32_interface = esp32_interface

        # Device registry
        self.devices: Dict[str, Device] = {}
        self.device_groups: Dict[str, List[str]] = {}  # Group name -> entity IDs
        self.locations: Dict[str, List[str]] = {}  # Location -> entity IDs

        # Discovery settings
        self.auto_discovery = True
        self.discovery_interval = 300  # 5 minutes

        # State monitoring
        self.state_update_handlers: List[callable] = []
        self.device_discovery_handlers: List[callable] = []

        # Performance tracking
        self.last_discovery = None
        self.discovery_count = 0
        self.update_count = 0

    async def initialize(self) -> None:
        """Initialize device manager"""
        try:
            logger.info("Initializing device manager...")

            # Register HA event handlers
            self.ha_client.register_event_handler("state_changed", self._handle_ha_state_change)

            # Register ESP32 handlers
            self.esp32_interface.register_global_sensor_handler(self._handle_esp32_sensor_data)
            self.esp32_interface.register_device_status_handler(self._handle_esp32_status)

            # Perform initial discovery
            await self.discover_devices()

            # Start periodic discovery if enabled
            if self.auto_discovery:
                asyncio.create_task(self._periodic_discovery())

            logger.info(f"Device manager initialized with {len(self.devices)} devices")

        except Exception as e:
            logger.error(f"Failed to initialize device manager: {e}")
            raise

    async def discover_devices(self) -> int:
        """Discover all available devices"""
        try:
            logger.info("Starting device discovery...")
            start_time = datetime.now()

            discovered_count = 0

            # Discover Home Assistant entities
            ha_count = await self._discover_ha_entities()
            discovered_count += ha_count

            # Discover ESP32 devices
            esp32_count = await self._discover_esp32_devices()
            discovered_count += esp32_count

            # Update groups and locations
            await self._update_device_organization()

            processing_time = (datetime.now() - start_time).total_seconds()
            self.last_discovery = datetime.now()
            self.discovery_count += 1

            logger.info(f"Device discovery completed: {discovered_count} devices found in {processing_time:.2f}s")

            # Trigger discovery handlers
            await self._trigger_discovery_handlers(discovered_count)

            return discovered_count

        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return 0

    async def _discover_ha_entities(self) -> int:
        """Discover Home Assistant entities"""
        try:
            if not self.ha_client.is_connected:
                logger.warning("Home Assistant not connected, skipping HA discovery")
                return 0

            # Get all entity states
            states = await self.ha_client.get_states()
            if not isinstance(states, list):
                logger.warning("Invalid states response from Home Assistant")
                return 0

            discovered_count = 0

            for entity_state in states:
                try:
                    entity_id = entity_state.get("entity_id")
                    if not entity_id:
                        continue

                    device = await self._create_device_from_ha_entity(entity_state)
                    if device:
                        self.devices[entity_id] = device
                        discovered_count += 1

                except Exception as e:
                    logger.error(f"Failed to process HA entity {entity_id}: {e}")

            logger.info(f"Discovered {discovered_count} Home Assistant entities")
            return discovered_count

        except Exception as e:
            logger.error(f"Home Assistant entity discovery failed: {e}")
            return 0

    async def _create_device_from_ha_entity(self, entity_state: Dict[str, Any]) -> Optional[Device]:
        """Create device from Home Assistant entity"""
        try:
            entity_id = entity_state["entity_id"]
            domain = entity_id.split(".")[0]

            # Map HA domain to device type
            device_type_mapping = {
                "light": DeviceType.LIGHT,
                "switch": DeviceType.SWITCH,
                "fan": DeviceType.FAN,
                "climate": DeviceType.CLIMATE,
                "sensor": DeviceType.SENSOR,
                "binary_sensor": DeviceType.BINARY_SENSOR,
                "camera": DeviceType.CAMERA,
                "media_player": DeviceType.MEDIA_PLAYER,
                "cover": DeviceType.COVER,
                "lock": DeviceType.LOCK,
                "vacuum": DeviceType.VACUUM,
                "water_heater": DeviceType.WATER_HEATER,
                "alarm_control_panel": DeviceType.ALARM_CONTROL_PANEL
            }

            device_type = device_type_mapping.get(domain)
            if not device_type:
                return None  # Skip unsupported device types

            # Determine device capabilities
            capabilities = self._determine_ha_capabilities(entity_state, device_type)

            # Determine device state
            device_state = self._map_ha_state_to_device_state(entity_state.get("state"))

            # Extract attributes
            attributes = entity_state.get("attributes", {})

            # Get friendly name
            name = attributes.get("friendly_name", entity_id)

            # Get location/area
            location = attributes.get("area_id") or attributes.get("room")

            device = Device(
                entity_id=entity_id,
                name=name,
                device_type=device_type,
                capabilities=capabilities,
                state=device_state,
                attributes=attributes,
                location=location,
                last_seen=datetime.now(),
                ha_entity=True,
                metadata={
                    "domain": domain,
                    "last_changed": entity_state.get("last_changed"),
                    "last_updated": entity_state.get("last_updated")
                }
            )

            return device

        except Exception as e:
            logger.error(f"Failed to create device from HA entity: {e}")
            return None

    def _determine_ha_capabilities(self, entity_state: Dict[str, Any], device_type: DeviceType) -> List[DeviceCapability]:
        """Determine device capabilities from HA entity"""
        capabilities = []
        attributes = entity_state.get("attributes", {})

        # Common capabilities based on device type
        type_capabilities = {
            DeviceType.LIGHT: [DeviceCapability.ON_OFF],
            DeviceType.SWITCH: [DeviceCapability.ON_OFF],
            DeviceType.FAN: [DeviceCapability.ON_OFF, DeviceCapability.SPEED],
            DeviceType.CLIMATE: [DeviceCapability.TEMPERATURE],
            DeviceType.COVER: [DeviceCapability.POSITION],
            DeviceType.MEDIA_PLAYER: [DeviceCapability.ON_OFF, DeviceCapability.VOLUME],
            DeviceType.LOCK: [DeviceCapability.ON_OFF],
        }

        capabilities.extend(type_capabilities.get(device_type, []))

        # Check for additional capabilities based on attributes
        if "brightness" in attributes:
            capabilities.append(DeviceCapability.BRIGHTNESS)

        if any(attr in attributes for attr in ["rgb_color", "hs_color", "color_temp"]):
            capabilities.append(DeviceCapability.COLOR)

        if "temperature" in attributes:
            capabilities.append(DeviceCapability.TEMPERATURE)

        if "humidity" in attributes:
            capabilities.append(DeviceCapability.HUMIDITY)

        if "battery_level" in attributes:
            capabilities.append(DeviceCapability.BATTERY)

        return list(set(capabilities))  # Remove duplicates

    def _map_ha_state_to_device_state(self, ha_state: str) -> DeviceState:
        """Map Home Assistant state to device state"""
        state_mapping = {
            "unavailable": DeviceState.UNAVAILABLE,
            "unknown": DeviceState.UNKNOWN,
            "on": DeviceState.ONLINE,
            "off": DeviceState.ONLINE,
            "idle": DeviceState.ONLINE,
            "active": DeviceState.ONLINE,
            "locked": DeviceState.ONLINE,
            "unlocked": DeviceState.ONLINE,
            "open": DeviceState.ONLINE,
            "closed": DeviceState.ONLINE
        }

        return state_mapping.get(ha_state, DeviceState.ONLINE)

    async def _discover_esp32_devices(self) -> int:
        """Discover ESP32 devices"""
        try:
            esp32_devices = self.esp32_interface.get_device_list()
            discovered_count = 0

            for device_info in esp32_devices:
                try:
                    device = await self._create_device_from_esp32(device_info)
                    if device:
                        self.devices[device.entity_id] = device
                        discovered_count += 1

                except Exception as e:
                    logger.error(f"Failed to process ESP32 device {device_info.get('device_id')}: {e}")

            logger.info(f"Discovered {discovered_count} ESP32 devices")
            return discovered_count

        except Exception as e:
            logger.error(f"ESP32 device discovery failed: {e}")
            return 0

    async def _create_device_from_esp32(self, device_info: Dict[str, Any]) -> Optional[Device]:
        """Create device from ESP32 device info"""
        try:
            device_id = device_info["device_id"]
            entity_id = f"esp32.{device_id}"

            # Determine capabilities based on sensors and actuators
            capabilities = []
            esp32_device = self.esp32_interface.get_device(device_id)

            if esp32_device:
                config = esp32_device.config

                # Add capabilities based on sensors
                for sensor in config.sensors:
                    sensor_type = sensor.get("type", "").lower()
                    if "temperature" in sensor_type:
                        capabilities.append(DeviceCapability.TEMPERATURE)
                    elif "humidity" in sensor_type:
                        capabilities.append(DeviceCapability.HUMIDITY)
                    elif "pressure" in sensor_type:
                        capabilities.append(DeviceCapability.PRESSURE)
                    elif "motion" in sensor_type:
                        capabilities.append(DeviceCapability.MOTION)

                # Add capabilities based on actuators
                for actuator in config.actuators:
                    actuator_type = actuator.get("type", "").lower()
                    if actuator_type in ["led", "relay", "switch"]:
                        capabilities.append(DeviceCapability.ON_OFF)
                    elif actuator_type in ["servo", "stepper"]:
                        capabilities.append(DeviceCapability.POSITION)

            # Determine device state
            device_state = DeviceState.ONLINE if device_info["is_connected"] else DeviceState.OFFLINE

            device = Device(
                entity_id=entity_id,
                name=device_info["name"],
                device_type=DeviceType.ESP32_DEVICE,
                capabilities=capabilities,
                state=device_state,
                attributes={
                    "host": device_info["host"],
                    "connection_type": device_info["connection_type"],
                    "sensors": device_info["sensors"],
                    "actuators": device_info["actuators"]
                },
                last_seen=datetime.fromisoformat(device_info["last_ping"]) if device_info["last_ping"] else None,
                ha_entity=False,
                esp32_device_id=device_id,
                metadata={
                    "device_type": "esp32",
                    "firmware_version": None  # Could be populated from device status
                }
            )

            return device

        except Exception as e:
            logger.error(f"Failed to create ESP32 device: {e}")
            return None

    async def _update_device_organization(self) -> None:
        """Update device groups and location mappings"""
        # Clear existing mappings
        self.device_groups.clear()
        self.locations.clear()

        for entity_id, device in self.devices.items():
            # Group by device type
            device_type = device.device_type.value
            if device_type not in self.device_groups:
                self.device_groups[device_type] = []
            self.device_groups[device_type].append(entity_id)

            # Group by location
            if device.location:
                if device.location not in self.locations:
                    self.locations[device.location] = []
                self.locations[device.location].append(entity_id)

    async def _periodic_discovery(self) -> None:
        """Periodic device discovery"""
        while self.auto_discovery:
            try:
                await asyncio.sleep(self.discovery_interval)
                await self.discover_devices()
            except Exception as e:
                logger.error(f"Periodic discovery error: {e}")

    async def _handle_ha_state_change(self, event_data: Dict[str, Any]) -> None:
        """Handle Home Assistant state changes"""
        try:
            entity_id = event_data.get("entity_id")
            new_state = event_data.get("new_state")

            if entity_id and new_state and entity_id in self.devices:
                device = self.devices[entity_id]

                # Update device state
                device.state = self._map_ha_state_to_device_state(new_state.get("state"))
                device.attributes = new_state.get("attributes", {})
                device.last_seen = datetime.now()

                self.update_count += 1

                # Trigger state update handlers
                await self._trigger_state_update_handlers(entity_id, device)

        except Exception as e:
            logger.error(f"Error handling HA state change: {e}")

    async def _handle_esp32_sensor_data(self, sensor_data: SensorData) -> None:
        """Handle ESP32 sensor data updates"""
        try:
            entity_id = f"esp32.{sensor_data.device_id}"

            if entity_id in self.devices:
                device = self.devices[entity_id]

                # Update device attributes with sensor data
                sensor_key = f"sensor_{sensor_data.sensor_id}"
                device.attributes[sensor_key] = {
                    "value": sensor_data.value,
                    "unit": sensor_data.unit,
                    "timestamp": sensor_data.timestamp.isoformat()
                }

                device.last_seen = datetime.now()
                device.state = DeviceState.ONLINE

                self.update_count += 1

                # Trigger state update handlers
                await self._trigger_state_update_handlers(entity_id, device)

        except Exception as e:
            logger.error(f"Error handling ESP32 sensor data: {e}")

    async def _handle_esp32_status(self, status_data: Dict[str, Any]) -> None:
        """Handle ESP32 device status updates"""
        try:
            device_id = status_data.get("device_id")
            if device_id:
                entity_id = f"esp32.{device_id}"

                if entity_id in self.devices:
                    device = self.devices[entity_id]
                    device.attributes.update(status_data)
                    device.last_seen = datetime.now()

                    # Update device state based on status
                    if status_data.get("online", True):
                        device.state = DeviceState.ONLINE
                    else:
                        device.state = DeviceState.OFFLINE

                    await self._trigger_state_update_handlers(entity_id, device)

        except Exception as e:
            logger.error(f"Error handling ESP32 status: {e}")

    async def _trigger_state_update_handlers(self, entity_id: str, device: Device) -> None:
        """Trigger state update handlers"""
        for handler in self.state_update_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(entity_id, device)
                else:
                    handler(entity_id, device)
            except Exception as e:
                logger.error(f"State update handler error: {e}")

    async def _trigger_discovery_handlers(self, count: int) -> None:
        """Trigger discovery handlers"""
        for handler in self.device_discovery_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(count)
                else:
                    handler(count)
            except Exception as e:
                logger.error(f"Discovery handler error: {e}")

    def get_device(self, entity_id: str) -> Optional[Device]:
        """Get device by entity ID"""
        return self.devices.get(entity_id)

    def get_devices_by_type(self, device_type: DeviceType) -> List[Device]:
        """Get devices by type"""
        return [device for device in self.devices.values() if device.device_type == device_type]

    def get_devices_by_location(self, location: str) -> List[Device]:
        """Get devices by location"""
        entity_ids = self.locations.get(location, [])
        return [self.devices[entity_id] for entity_id in entity_ids if entity_id in self.devices]

    def get_devices_with_capability(self, capability: DeviceCapability) -> List[Device]:
        """Get devices with specific capability"""
        return [device for device in self.devices.values() if device.has_capability(capability)]

    def get_available_devices(self) -> List[Device]:
        """Get all available devices"""
        return [device for device in self.devices.values() if device.is_available()]

    def get_device_stats(self) -> Dict[str, Any]:
        """Get device statistics"""
        total_devices = len(self.devices)
        available_devices = len(self.get_available_devices())
        ha_devices = len([d for d in self.devices.values() if d.ha_entity])
        esp32_devices = len([d for d in self.devices.values() if not d.ha_entity])

        return {
            "total_devices": total_devices,
            "available_devices": available_devices,
            "unavailable_devices": total_devices - available_devices,
            "ha_devices": ha_devices,
            "esp32_devices": esp32_devices,
            "device_types": {dt: len(devices) for dt, devices in self.device_groups.items()},
            "locations": {loc: len(devices) for loc, devices in self.locations.items()},
            "last_discovery": self.last_discovery.isoformat() if self.last_discovery else None,
            "discovery_count": self.discovery_count,
            "update_count": self.update_count
        }

    async def control_device(
        self,
        entity_id: str,
        action: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Control a device"""
        try:
            device = self.get_device(entity_id)
            if not device:
                logger.error(f"Device not found: {entity_id}")
                return False

            if not device.is_available():
                logger.error(f"Device not available: {entity_id}")
                return False

            if device.ha_entity:
                return await self._control_ha_device(entity_id, action, parameters)
            else:
                return await self._control_esp32_device(device, action, parameters)

        except Exception as e:
            logger.error(f"Failed to control device {entity_id}: {e}")
            return False

    async def _control_ha_device(self, entity_id: str, action: str, parameters: Optional[Dict[str, Any]]) -> bool:
        """Control Home Assistant device"""
        try:
            domain = entity_id.split(".")[0]

            if action == "turn_on":
                await self.ha_client.turn_on(entity_id, **(parameters or {}))
            elif action == "turn_off":
                await self.ha_client.turn_off(entity_id, **(parameters or {}))
            elif action == "toggle":
                await self.ha_client.toggle(entity_id, **(parameters or {}))
            else:
                # Call custom service
                service_parts = action.split(".")
                if len(service_parts) == 2:
                    service_domain, service_name = service_parts
                    await self.ha_client.call_service(service_domain, service_name, entity_id, parameters)
                else:
                    await self.ha_client.call_service(domain, action, entity_id, parameters)

            return True

        except Exception as e:
            logger.error(f"Failed to control HA device {entity_id}: {e}")
            return False

    async def _control_esp32_device(self, device: Device, action: str, parameters: Optional[Dict[str, Any]]) -> bool:
        """Control ESP32 device"""
        try:
            if not device.esp32_device_id:
                return False

            esp32_device = self.esp32_interface.get_device(device.esp32_device_id)
            if not esp32_device:
                return False

            # Map action to actuator control
            target = parameters.get("target") if parameters else None
            if not target:
                # Use first available actuator
                actuators = device.attributes.get("actuators", [])
                if actuators:
                    target = actuators[0].get("id")

            if target:
                return await esp32_device.control_actuator(target, action, parameters)

            return False

        except Exception as e:
            logger.error(f"Failed to control ESP32 device {device.entity_id}: {e}")
            return False

    def register_state_update_handler(self, handler: callable) -> None:
        """Register state update handler"""
        self.state_update_handlers.append(handler)

    def register_discovery_handler(self, handler: callable) -> None:
        """Register device discovery handler"""
        self.device_discovery_handlers.append(handler)

    async def refresh_device(self, entity_id: str) -> bool:
        """Refresh specific device state"""
        try:
            device = self.get_device(entity_id)
            if not device:
                return False

            if device.ha_entity:
                # Get latest state from HA
                state = await self.ha_client.get_states(entity_id)
                if state:
                    device.state = self._map_ha_state_to_device_state(state.get("state"))
                    device.attributes = state.get("attributes", {})
                    device.last_seen = datetime.now()
                    return True
            else:
                # Ping ESP32 device
                if device.esp32_device_id:
                    esp32_device = self.esp32_interface.get_device(device.esp32_device_id)
                    if esp32_device:
                        ping_result = await esp32_device.ping()
                        device.state = DeviceState.ONLINE if ping_result else DeviceState.OFFLINE
                        device.last_seen = datetime.now()
                        return ping_result

            return False

        except Exception as e:
            logger.error(f"Failed to refresh device {entity_id}: {e}")
            return False