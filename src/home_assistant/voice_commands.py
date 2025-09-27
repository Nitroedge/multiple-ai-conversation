"""
Voice command processing for Home Assistant device control
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .device_manager import DeviceManager, Device, DeviceType, DeviceCapability
from .automation_engine import AutomationEngine, AutomationRule
from .environmental_context import EnvironmentalContextManager

logger = logging.getLogger(__name__)


class HomeCommandType(str, Enum):
    """Types of home automation voice commands"""
    DEVICE_CONTROL = "device_control"
    SCENE_ACTIVATION = "scene_activation"
    AUTOMATION_CONTROL = "automation_control"
    STATUS_QUERY = "status_query"
    ENVIRONMENTAL_QUERY = "environmental_query"
    SCHEDULE_CONTROL = "schedule_control"
    SECURITY_CONTROL = "security_control"
    ENERGY_MANAGEMENT = "energy_management"


class CommandAction(str, Enum):
    """Command actions"""
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    TOGGLE = "toggle"
    SET_BRIGHTNESS = "set_brightness"
    SET_COLOR = "set_color"
    SET_TEMPERATURE = "set_temperature"
    SET_SPEED = "set_speed"
    INCREASE = "increase"
    DECREASE = "decrease"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    LOCK = "lock"
    UNLOCK = "unlock"
    OPEN = "open"
    CLOSE = "close"
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"


class DeviceSelector(str, Enum):
    """Device selection patterns"""
    SPECIFIC_DEVICE = "specific_device"
    DEVICE_TYPE = "device_type"
    ROOM_DEVICES = "room_devices"
    ALL_DEVICES = "all_devices"
    DEVICE_GROUP = "device_group"


@dataclass
class HomeCommand:
    """Parsed home automation voice command"""
    command_id: str
    text: str
    command_type: HomeCommandType
    action: CommandAction
    device_selector: DeviceSelector
    target_devices: List[str]
    location: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    timestamp: datetime = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "text": self.text,
            "command_type": self.command_type.value,
            "action": self.action.value,
            "device_selector": self.device_selector.value,
            "target_devices": self.target_devices,
            "location": self.location,
            "parameters": self.parameters or {},
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id
        }


class HomeVoiceCommandProcessor:
    """Processes voice commands for home automation"""

    def __init__(
        self,
        device_manager: DeviceManager,
        automation_engine: AutomationEngine,
        environmental_context: EnvironmentalContextManager
    ):
        self.device_manager = device_manager
        self.automation_engine = automation_engine
        self.environmental_context = environmental_context

        # Command patterns and mappings
        self.command_patterns = self._initialize_command_patterns()
        self.device_aliases = self._initialize_device_aliases()
        self.location_aliases = self._initialize_location_aliases()

        # Command history
        self.command_history: List[HomeCommand] = []
        self.max_history = 100

        # Processing statistics
        self.total_commands = 0
        self.successful_commands = 0
        self.failed_commands = 0

    def _initialize_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize command recognition patterns"""
        return {
            # Device control patterns
            "turn_on": {
                "patterns": [
                    r"turn on (?:the )?(.+)",
                    r"switch on (?:the )?(.+)",
                    r"activate (?:the )?(.+)",
                    r"start (?:the )?(.+)"
                ],
                "action": CommandAction.TURN_ON,
                "command_type": HomeCommandType.DEVICE_CONTROL
            },
            "turn_off": {
                "patterns": [
                    r"turn off (?:the )?(.+)",
                    r"switch off (?:the )?(.+)",
                    r"deactivate (?:the )?(.+)",
                    r"stop (?:the )?(.+)"
                ],
                "action": CommandAction.TURN_OFF,
                "command_type": HomeCommandType.DEVICE_CONTROL
            },
            "toggle": {
                "patterns": [
                    r"toggle (?:the )?(.+)",
                    r"flip (?:the )?(.+)"
                ],
                "action": CommandAction.TOGGLE,
                "command_type": HomeCommandType.DEVICE_CONTROL
            },
            "brightness": {
                "patterns": [
                    r"set (?:the )?(.+) brightness to (\d+)(?:%)?",
                    r"dim (?:the )?(.+) to (\d+)(?:%)?",
                    r"brighten (?:the )?(.+) to (\d+)(?:%)?"
                ],
                "action": CommandAction.SET_BRIGHTNESS,
                "command_type": HomeCommandType.DEVICE_CONTROL
            },
            "temperature": {
                "patterns": [
                    r"set (?:the )?temperature to (\d+)(?:\s?degrees?)?",
                    r"set (?:the )?thermostat to (\d+)(?:\s?degrees?)?",
                    r"make it (\d+)(?:\s?degrees?)?"
                ],
                "action": CommandAction.SET_TEMPERATURE,
                "command_type": HomeCommandType.DEVICE_CONTROL
            },
            "lock": {
                "patterns": [
                    r"lock (?:the )?(.+)",
                    r"secure (?:the )?(.+)"
                ],
                "action": CommandAction.LOCK,
                "command_type": HomeCommandType.SECURITY_CONTROL
            },
            "unlock": {
                "patterns": [
                    r"unlock (?:the )?(.+)",
                    r"open (?:the )?(.+) lock"
                ],
                "action": CommandAction.UNLOCK,
                "command_type": HomeCommandType.SECURITY_CONTROL
            },
            "status_query": {
                "patterns": [
                    r"what(?:'s| is) (?:the )?status of (?:the )?(.+)",
                    r"is (?:the )?(.+) on(?:\?)?",
                    r"is (?:the )?(.+) off(?:\?)?",
                    r"check (?:the )?(.+)"
                ],
                "action": CommandAction.TURN_ON,  # Placeholder
                "command_type": HomeCommandType.STATUS_QUERY
            },
            "environmental_query": {
                "patterns": [
                    r"what(?:'s| is) (?:the )?temperature(?: in (?:the )?(.+))?",
                    r"how warm is it(?: in (?:the )?(.+))?",
                    r"what(?:'s| is) (?:the )?humidity(?: in (?:the )?(.+))?",
                    r"is it comfortable(?: in (?:the )?(.+))?"
                ],
                "action": CommandAction.TURN_ON,  # Placeholder
                "command_type": HomeCommandType.ENVIRONMENTAL_QUERY
            },
            "scene_activation": {
                "patterns": [
                    r"activate (?:the )?(.+) scene",
                    r"set (?:the )?(.+) scene",
                    r"turn on (?:the )?(.+) scene"
                ],
                "action": CommandAction.ACTIVATE,
                "command_type": HomeCommandType.SCENE_ACTIVATION
            },
            "all_lights": {
                "patterns": [
                    r"turn (on|off) all (?:the )?lights",
                    r"(turn on|turn off) all lights",
                    r"lights (on|off) everywhere"
                ],
                "action": None,  # Will be determined by pattern match
                "command_type": HomeCommandType.DEVICE_CONTROL
            }
        }

    def _initialize_device_aliases(self) -> Dict[str, List[str]]:
        """Initialize device name aliases"""
        return {
            "lights": ["light", "lights", "lamp", "lamps", "lighting"],
            "fan": ["fan", "fans", "ceiling fan"],
            "thermostat": ["thermostat", "temperature", "ac", "air conditioning", "heating"],
            "tv": ["tv", "television", "telly"],
            "music": ["music", "speaker", "speakers", "audio", "sound system"],
            "door": ["door", "doors"],
            "window": ["window", "windows"],
            "lock": ["lock", "locks", "door lock"],
            "garage": ["garage", "garage door"],
            "blinds": ["blinds", "curtains", "shades"],
            "vacuum": ["vacuum", "robot vacuum", "roomba"]
        }

    def _initialize_location_aliases(self) -> Dict[str, List[str]]:
        """Initialize location aliases"""
        return {
            "living room": ["living room", "lounge", "sitting room", "front room"],
            "bedroom": ["bedroom", "bed room", "master bedroom"],
            "kitchen": ["kitchen", "dining room"],
            "bathroom": ["bathroom", "bath", "toilet"],
            "office": ["office", "study", "work room"],
            "garage": ["garage"],
            "basement": ["basement", "cellar"],
            "attic": ["attic", "loft"],
            "outside": ["outside", "outdoor", "garden", "yard", "patio", "deck"]
        }

    async def process_voice_command(self, text: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[HomeCommand]:
        """Process voice command text into structured command"""
        try:
            self.total_commands += 1

            # Normalize text
            normalized_text = self._normalize_text(text)

            # Parse command
            command = await self._parse_command(normalized_text, session_id, user_id)

            if command:
                # Store in history
                self.command_history.append(command)
                if len(self.command_history) > self.max_history:
                    self.command_history = self.command_history[-self.max_history:]

                logger.info(f"Parsed voice command: {command.command_type.value} - {command.action.value}")
                return command
            else:
                logger.warning(f"Failed to parse voice command: {text}")
                return None

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Convert to lowercase
        text = text.lower().strip()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    async def _parse_command(self, text: str, session_id: Optional[str], user_id: Optional[str]) -> Optional[HomeCommand]:
        """Parse normalized text into command structure"""
        # Try to match against command patterns
        for pattern_name, pattern_info in self.command_patterns.items():
            for pattern in pattern_info["patterns"]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return await self._create_command_from_match(
                        text, match, pattern_info, session_id, user_id
                    )

        return None

    async def _create_command_from_match(
        self,
        text: str,
        match: re.Match,
        pattern_info: Dict[str, Any],
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> Optional[HomeCommand]:
        """Create command from regex match"""
        try:
            command_id = f"home_cmd_{int(datetime.now().timestamp())}_{hash(text) % 10000}"
            command_type = pattern_info["command_type"]
            action = pattern_info["action"]

            # Extract device/location information from match groups
            groups = match.groups()
            device_info = groups[0] if groups else ""

            # Handle special cases
            if "all_lights" in text or "all lights" in text:
                target_devices = await self._get_all_devices_by_type("light")
                device_selector = DeviceSelector.DEVICE_TYPE
                location = None
                if "on" in text:
                    action = CommandAction.TURN_ON
                elif "off" in text:
                    action = CommandAction.TURN_OFF
            else:
                # Parse device and location information
                target_devices, device_selector, location = await self._parse_device_target(device_info)

            # Extract parameters
            parameters = await self._extract_parameters(text, match, action)

            # Determine confidence based on parsing success
            confidence = 0.9 if target_devices else 0.5

            command = HomeCommand(
                command_id=command_id,
                text=text,
                command_type=command_type,
                action=action,
                device_selector=device_selector,
                target_devices=target_devices,
                location=location,
                parameters=parameters,
                confidence=confidence,
                session_id=session_id,
                user_id=user_id
            )

            return command

        except Exception as e:
            logger.error(f"Error creating command from match: {e}")
            return None

    async def _parse_device_target(self, device_info: str) -> Tuple[List[str], DeviceSelector, Optional[str]]:
        """Parse device target information"""
        if not device_info:
            return [], DeviceSelector.ALL_DEVICES, None

        device_info = device_info.strip()

        # Check for location-specific requests
        location = None
        for loc, aliases in self.location_aliases.items():
            for alias in aliases:
                if f"in the {alias}" in device_info or f"in {alias}" in device_info:
                    location = loc
                    device_info = device_info.replace(f"in the {alias}", "").replace(f"in {alias}", "").strip()
                    break
            if location:
                break

        # Extract device type or specific device
        device_type = None
        specific_device = None

        # Check device aliases
        for device_category, aliases in self.device_aliases.items():
            for alias in aliases:
                if alias in device_info:
                    device_type = device_category
                    break
            if device_type:
                break

        if device_type:
            # Get devices by type and location
            if location:
                devices = await self._get_devices_by_type_and_location(device_type, location)
                selector = DeviceSelector.ROOM_DEVICES
            else:
                devices = await self._get_all_devices_by_type(device_type)
                selector = DeviceSelector.DEVICE_TYPE
        else:
            # Try to find specific device by name
            devices = await self._find_device_by_name(device_info)
            if devices:
                selector = DeviceSelector.SPECIFIC_DEVICE
            else:
                # If location specified, get all devices in location
                if location:
                    devices = await self._get_all_devices_in_location(location)
                    selector = DeviceSelector.ROOM_DEVICES
                else:
                    devices = []
                    selector = DeviceSelector.ALL_DEVICES

        return devices, selector, location

    async def _get_all_devices_by_type(self, device_type: str) -> List[str]:
        """Get all devices of specific type"""
        # Map device aliases to DeviceType enum
        type_mapping = {
            "lights": DeviceType.LIGHT,
            "fan": DeviceType.FAN,
            "thermostat": DeviceType.CLIMATE,
            "tv": DeviceType.MEDIA_PLAYER,
            "lock": DeviceType.LOCK,
            "vacuum": DeviceType.VACUUM
        }

        enum_type = type_mapping.get(device_type)
        if enum_type:
            devices = self.device_manager.get_devices_by_type(enum_type)
            return [device.entity_id for device in devices if device.is_available()]

        return []

    async def _get_devices_by_type_and_location(self, device_type: str, location: str) -> List[str]:
        """Get devices by type in specific location"""
        all_devices = await self._get_all_devices_by_type(device_type)
        location_devices = await self._get_all_devices_in_location(location)

        # Return intersection
        return list(set(all_devices) & set(location_devices))

    async def _get_all_devices_in_location(self, location: str) -> List[str]:
        """Get all devices in specific location"""
        devices = self.device_manager.get_devices_by_location(location)
        return [device.entity_id for device in devices if device.is_available()]

    async def _find_device_by_name(self, name: str) -> List[str]:
        """Find device by name or partial name match"""
        all_devices = self.device_manager.get_available_devices()
        matching_devices = []

        for device in all_devices:
            # Check if name matches device name or entity ID
            if (name.lower() in device.name.lower() or
                name.lower() in device.entity_id.lower()):
                matching_devices.append(device.entity_id)

        return matching_devices

    async def _extract_parameters(self, text: str, match: re.Match, action: CommandAction) -> Dict[str, Any]:
        """Extract command parameters from text"""
        parameters = {}

        try:
            # Extract brightness values
            if action == CommandAction.SET_BRIGHTNESS:
                groups = match.groups()
                if len(groups) >= 2:
                    brightness = int(groups[1])
                    parameters["brightness"] = min(100, max(0, brightness))

            # Extract temperature values
            elif action == CommandAction.SET_TEMPERATURE:
                groups = match.groups()
                if groups:
                    temperature = int(groups[0])
                    parameters["temperature"] = temperature

            # Extract color information
            elif action == CommandAction.SET_COLOR:
                # Simple color extraction (could be enhanced)
                colors = {
                    "red": [255, 0, 0],
                    "green": [0, 255, 0],
                    "blue": [0, 0, 255],
                    "white": [255, 255, 255],
                    "yellow": [255, 255, 0],
                    "purple": [128, 0, 128],
                    "orange": [255, 165, 0]
                }

                for color_name, rgb in colors.items():
                    if color_name in text:
                        parameters["rgb_color"] = rgb
                        break

        except (ValueError, IndexError) as e:
            logger.warning(f"Error extracting parameters: {e}")

        return parameters

    async def execute_command(self, command: HomeCommand) -> Dict[str, Any]:
        """Execute home automation command"""
        try:
            logger.info(f"Executing home command: {command.command_type.value} - {command.action.value}")

            if command.command_type == HomeCommandType.DEVICE_CONTROL:
                result = await self._execute_device_control(command)
            elif command.command_type == HomeCommandType.STATUS_QUERY:
                result = await self._execute_status_query(command)
            elif command.command_type == HomeCommandType.ENVIRONMENTAL_QUERY:
                result = await self._execute_environmental_query(command)
            elif command.command_type == HomeCommandType.SCENE_ACTIVATION:
                result = await self._execute_scene_activation(command)
            elif command.command_type == HomeCommandType.AUTOMATION_CONTROL:
                result = await self._execute_automation_control(command)
            else:
                result = {
                    "success": False,
                    "message": f"Unsupported command type: {command.command_type.value}"
                }

            if result.get("success", False):
                self.successful_commands += 1
            else:
                self.failed_commands += 1

            return result

        except Exception as e:
            logger.error(f"Error executing command: {e}")
            self.failed_commands += 1
            return {
                "success": False,
                "message": f"Execution error: {str(e)}"
            }

    async def _execute_device_control(self, command: HomeCommand) -> Dict[str, Any]:
        """Execute device control command"""
        if not command.target_devices:
            return {
                "success": False,
                "message": "No target devices found"
            }

        results = []
        success_count = 0

        for entity_id in command.target_devices:
            try:
                device = self.device_manager.get_device(entity_id)
                if not device:
                    results.append(f"Device not found: {entity_id}")
                    continue

                if not device.is_available():
                    results.append(f"Device unavailable: {device.name}")
                    continue

                # Map action to device control
                action_map = {
                    CommandAction.TURN_ON: "turn_on",
                    CommandAction.TURN_OFF: "turn_off",
                    CommandAction.TOGGLE: "toggle",
                    CommandAction.LOCK: "lock",
                    CommandAction.UNLOCK: "unlock"
                }

                if command.action in action_map:
                    success = await self.device_manager.control_device(
                        entity_id,
                        action_map[command.action],
                        command.parameters
                    )
                else:
                    # Handle parameter-based commands
                    success = await self._execute_parameter_command(device, command)

                if success:
                    success_count += 1
                    results.append(f"Successfully controlled {device.name}")
                else:
                    results.append(f"Failed to control {device.name}")

            except Exception as e:
                results.append(f"Error controlling {entity_id}: {str(e)}")

        return {
            "success": success_count > 0,
            "message": f"Controlled {success_count}/{len(command.target_devices)} devices",
            "details": results
        }

    async def _execute_parameter_command(self, device: Device, command: HomeCommand) -> bool:
        """Execute parameter-based command (brightness, temperature, etc.)"""
        try:
            if command.action == CommandAction.SET_BRIGHTNESS:
                if device.has_capability(DeviceCapability.BRIGHTNESS):
                    return await self.device_manager.control_device(
                        device.entity_id,
                        "turn_on",
                        {"brightness_pct": command.parameters.get("brightness", 50)}
                    )

            elif command.action == CommandAction.SET_TEMPERATURE:
                if device.has_capability(DeviceCapability.TEMPERATURE):
                    return await self.device_manager.control_device(
                        device.entity_id,
                        "set_temperature",
                        {"temperature": command.parameters.get("temperature", 20)}
                    )

            elif command.action == CommandAction.SET_COLOR:
                if device.has_capability(DeviceCapability.COLOR):
                    return await self.device_manager.control_device(
                        device.entity_id,
                        "turn_on",
                        {"rgb_color": command.parameters.get("rgb_color", [255, 255, 255])}
                    )

            return False

        except Exception as e:
            logger.error(f"Error executing parameter command: {e}")
            return False

    async def _execute_status_query(self, command: HomeCommand) -> Dict[str, Any]:
        """Execute status query command"""
        if not command.target_devices:
            return {
                "success": False,
                "message": "No devices specified for status query"
            }

        status_info = []

        for entity_id in command.target_devices:
            device = self.device_manager.get_device(entity_id)
            if device:
                status_info.append({
                    "device": device.name,
                    "state": device.state.value,
                    "available": device.is_available(),
                    "location": device.location
                })

        return {
            "success": True,
            "message": f"Status for {len(status_info)} devices",
            "status": status_info
        }

    async def _execute_environmental_query(self, command: HomeCommand) -> Dict[str, Any]:
        """Execute environmental query command"""
        location = command.location or "unknown"

        env_data = self.environmental_context.get_environmental_data(location)
        if not env_data:
            return {
                "success": False,
                "message": f"No environmental data available for {location}"
            }

        return {
            "success": True,
            "message": f"Environmental status for {location}",
            "environmental_data": {
                "temperature": env_data.get_value("temperature"),
                "humidity": env_data.get_value("humidity"),
                "comfort_level": env_data.comfort_level.value,
                "occupancy": env_data.occupancy_detected,
                "last_updated": env_data.last_updated.isoformat()
            }
        }

    async def _execute_scene_activation(self, command: HomeCommand) -> Dict[str, Any]:
        """Execute scene activation command"""
        # Scene activation would integrate with Home Assistant scenes
        # For now, return placeholder
        return {
            "success": True,
            "message": f"Scene activation not yet implemented"
        }

    async def _execute_automation_control(self, command: HomeCommand) -> Dict[str, Any]:
        """Execute automation control command"""
        # Automation control would integrate with automation engine
        # For now, return placeholder
        return {
            "success": True,
            "message": f"Automation control not yet implemented"
        }

    def get_command_history(self, limit: int = 10) -> List[HomeCommand]:
        """Get recent command history"""
        return self.command_history[-limit:]

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get command processing statistics"""
        return {
            "total_commands": self.total_commands,
            "successful_commands": self.successful_commands,
            "failed_commands": self.failed_commands,
            "success_rate": self.successful_commands / max(1, self.total_commands),
            "recent_commands": len(self.command_history)
        }

    def get_supported_commands(self) -> List[Dict[str, str]]:
        """Get list of supported command patterns"""
        commands = []
        for pattern_name, pattern_info in self.command_patterns.items():
            commands.append({
                "name": pattern_name,
                "type": pattern_info["command_type"].value,
                "action": pattern_info["action"].value if pattern_info["action"] else "varies",
                "examples": pattern_info["patterns"][:2]  # Show first 2 patterns as examples
            })
        return commands


# Utility functions
def create_device_control_command(
    entity_id: str,
    action: CommandAction,
    parameters: Optional[Dict[str, Any]] = None
) -> HomeCommand:
    """Create device control command programmatically"""
    return HomeCommand(
        command_id=f"prog_cmd_{int(datetime.now().timestamp())}",
        text=f"Programmatic {action.value} for {entity_id}",
        command_type=HomeCommandType.DEVICE_CONTROL,
        action=action,
        device_selector=DeviceSelector.SPECIFIC_DEVICE,
        target_devices=[entity_id],
        parameters=parameters
    )