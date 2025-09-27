"""
Environmental context awareness system for intelligent home automation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import statistics

from pydantic import BaseModel, Field

from .device_manager import DeviceManager, Device, DeviceCapability
from .esp32_interface import ESP32Interface, SensorData, SensorType

logger = logging.getLogger(__name__)


class EnvironmentalParameter(str, Enum):
    """Environmental parameters to monitor"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT_LEVEL = "light_level"
    AIR_QUALITY = "air_quality"
    NOISE_LEVEL = "noise_level"
    MOTION_ACTIVITY = "motion_activity"
    OCCUPANCY = "occupancy"
    ENERGY_USAGE = "energy_usage"
    WEATHER = "weather"


class ContextLevel(str, Enum):
    """Context awareness levels"""
    ROOM = "room"
    ZONE = "zone"
    FLOOR = "floor"
    BUILDING = "building"
    OUTDOOR = "outdoor"


class ContextTriggerType(str, Enum):
    """Context trigger types"""
    THRESHOLD = "threshold"
    TREND = "trend"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"


class ComfortLevel(str, Enum):
    """Comfort levels for environmental conditions"""
    VERY_UNCOMFORTABLE = "very_uncomfortable"
    UNCOMFORTABLE = "uncomfortable"
    NEUTRAL = "neutral"
    COMFORTABLE = "comfortable"
    VERY_COMFORTABLE = "very_comfortable"


@dataclass
class EnvironmentalReading:
    """Single environmental reading"""
    parameter: EnvironmentalParameter
    value: float
    unit: str
    location: str
    timestamp: datetime
    sensor_id: str
    accuracy: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter.value,
            "value": self.value,
            "unit": self.unit,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "accuracy": self.accuracy,
            "metadata": self.metadata or {}
        }


@dataclass
class EnvironmentalData:
    """Aggregated environmental data for a location"""
    location: str
    context_level: ContextLevel
    readings: Dict[EnvironmentalParameter, EnvironmentalReading]
    comfort_level: ComfortLevel
    last_updated: datetime
    occupancy_detected: bool = False
    motion_detected: bool = False
    people_count: int = 0

    def get_reading(self, parameter: EnvironmentalParameter) -> Optional[EnvironmentalReading]:
        """Get specific environmental reading"""
        return self.readings.get(parameter)

    def get_value(self, parameter: EnvironmentalParameter) -> Optional[float]:
        """Get parameter value"""
        reading = self.get_reading(parameter)
        return reading.value if reading else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "context_level": self.context_level.value,
            "readings": {param.value: reading.to_dict() for param, reading in self.readings.items()},
            "comfort_level": self.comfort_level.value,
            "last_updated": self.last_updated.isoformat(),
            "occupancy_detected": self.occupancy_detected,
            "motion_detected": self.motion_detected,
            "people_count": self.people_count
        }


@dataclass
class ContextRule:
    """Environmental context rule"""
    rule_id: str
    name: str
    location: str
    parameter: EnvironmentalParameter
    trigger_type: ContextTriggerType
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    trend_duration: Optional[int] = None  # minutes
    pattern_schedule: Optional[Dict[str, Any]] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextTrigger:
    """Environmental context trigger event"""
    trigger_id: str
    rule_id: str
    location: str
    parameter: EnvironmentalParameter
    trigger_type: ContextTriggerType
    current_value: float
    trigger_reason: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "rule_id": self.rule_id,
            "location": self.location,
            "parameter": self.parameter.value,
            "trigger_type": self.trigger_type.value,
            "current_value": self.current_value,
            "trigger_reason": self.trigger_reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class EnvironmentalContextManager:
    """Manages environmental context awareness"""

    def __init__(self, device_manager: DeviceManager, esp32_interface: ESP32Interface):
        self.device_manager = device_manager
        self.esp32_interface = esp32_interface

        # Environmental data storage
        self.environmental_data: Dict[str, EnvironmentalData] = {}
        self.historical_data: Dict[str, List[EnvironmentalReading]] = {}

        # Context rules and triggers
        self.context_rules: Dict[str, ContextRule] = {}
        self.active_triggers: List[ContextTrigger] = []

        # Event handlers
        self.context_change_handlers: List[Callable] = []
        self.trigger_handlers: List[Callable] = []

        # Configuration
        self.data_retention_hours = 168  # 7 days
        self.update_interval = 60  # seconds
        self.trend_analysis_points = 10

        # Comfort thresholds (can be configured per location)
        self.comfort_thresholds = {
            EnvironmentalParameter.TEMPERATURE: {
                ComfortLevel.VERY_COMFORTABLE: (20.0, 24.0),
                ComfortLevel.COMFORTABLE: (18.0, 26.0),
                ComfortLevel.NEUTRAL: (16.0, 28.0),
                ComfortLevel.UNCOMFORTABLE: (14.0, 30.0),
                ComfortLevel.VERY_UNCOMFORTABLE: (0.0, 100.0)  # Outside other ranges
            },
            EnvironmentalParameter.HUMIDITY: {
                ComfortLevel.VERY_COMFORTABLE: (40.0, 60.0),
                ComfortLevel.COMFORTABLE: (35.0, 65.0),
                ComfortLevel.NEUTRAL: (30.0, 70.0),
                ComfortLevel.UNCOMFORTABLE: (25.0, 75.0),
                ComfortLevel.VERY_UNCOMFORTABLE: (0.0, 100.0)
            }
        }

        # Background tasks
        self.monitoring_task = None
        self.analysis_task = None

    async def initialize(self) -> None:
        """Initialize environmental context manager"""
        try:
            logger.info("Initializing environmental context manager...")

            # Register device update handlers
            self.device_manager.register_state_update_handler(self._handle_device_update)

            # Register ESP32 sensor handlers
            self.esp32_interface.register_global_sensor_handler(self._handle_sensor_reading)

            # Start background monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.analysis_task = asyncio.create_task(self._analysis_loop())

            logger.info("Environmental context manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize environmental context manager: {e}")
            raise

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                await self._update_environmental_data()
                await self._check_context_rules()
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.update_interval)

    async def _analysis_loop(self) -> None:
        """Background analysis loop"""
        while True:
            try:
                await self._analyze_trends()
                await self._detect_patterns()
                await self._cleanup_old_data()
                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(300)

    async def _update_environmental_data(self) -> None:
        """Update environmental data from all sources"""
        try:
            # Get all sensor devices
            sensor_devices = self.device_manager.get_devices_with_capability(DeviceCapability.TEMPERATURE)
            sensor_devices.extend(self.device_manager.get_devices_with_capability(DeviceCapability.HUMIDITY))

            # Update from Home Assistant sensors
            for device in sensor_devices:
                await self._process_ha_sensor(device)

            # Update from ESP32 devices
            esp32_data = await self.esp32_interface.read_all_sensors()
            for device_id, sensor_readings in esp32_data.items():
                for sensor_id, sensor_data in sensor_readings.items():
                    await self._process_esp32_sensor(sensor_data)

            # Update environmental context
            await self._update_context_data()

        except Exception as e:
            logger.error(f"Error updating environmental data: {e}")

    async def _process_ha_sensor(self, device: Device) -> None:
        """Process Home Assistant sensor data"""
        try:
            location = device.location or "unknown"

            # Extract sensor readings from device attributes
            if device.has_capability(DeviceCapability.TEMPERATURE):
                temp_value = device.get_attribute("temperature")
                if temp_value is not None:
                    reading = EnvironmentalReading(
                        parameter=EnvironmentalParameter.TEMPERATURE,
                        value=float(temp_value),
                        unit=device.get_attribute("unit_of_measurement", "°C"),
                        location=location,
                        timestamp=datetime.now(),
                        sensor_id=device.entity_id
                    )
                    await self._store_reading(reading)

            if device.has_capability(DeviceCapability.HUMIDITY):
                humidity_value = device.get_attribute("humidity")
                if humidity_value is not None:
                    reading = EnvironmentalReading(
                        parameter=EnvironmentalParameter.HUMIDITY,
                        value=float(humidity_value),
                        unit=device.get_attribute("unit_of_measurement", "%"),
                        location=location,
                        timestamp=datetime.now(),
                        sensor_id=device.entity_id
                    )
                    await self._store_reading(reading)

        except Exception as e:
            logger.error(f"Error processing HA sensor {device.entity_id}: {e}")

    async def _process_esp32_sensor(self, sensor_data: SensorData) -> None:
        """Process ESP32 sensor data"""
        try:
            # Map sensor types to environmental parameters
            sensor_mapping = {
                SensorType.TEMPERATURE: EnvironmentalParameter.TEMPERATURE,
                SensorType.HUMIDITY: EnvironmentalParameter.HUMIDITY,
                SensorType.PRESSURE: EnvironmentalParameter.PRESSURE,
                SensorType.LIGHT: EnvironmentalParameter.LIGHT_LEVEL,
                SensorType.SOUND: EnvironmentalParameter.NOISE_LEVEL,
                SensorType.AIR_QUALITY: EnvironmentalParameter.AIR_QUALITY,
                SensorType.MOTION: EnvironmentalParameter.MOTION_ACTIVITY
            }

            parameter = sensor_mapping.get(sensor_data.sensor_type)
            if parameter:
                # Get device location (could be configured in ESP32 device metadata)
                esp32_device = self.esp32_interface.get_device(sensor_data.device_id)
                location = "unknown"
                if esp32_device:
                    device_entity = self.device_manager.get_device(f"esp32.{sensor_data.device_id}")
                    if device_entity:
                        location = device_entity.location or "unknown"

                reading = EnvironmentalReading(
                    parameter=parameter,
                    value=float(sensor_data.value),
                    unit=sensor_data.unit,
                    location=location,
                    timestamp=sensor_data.timestamp,
                    sensor_id=sensor_data.sensor_id,
                    accuracy=sensor_data.accuracy
                )
                await self._store_reading(reading)

        except Exception as e:
            logger.error(f"Error processing ESP32 sensor data: {e}")

    async def _store_reading(self, reading: EnvironmentalReading) -> None:
        """Store environmental reading"""
        # Store in historical data
        location_key = f"{reading.location}_{reading.parameter.value}"
        if location_key not in self.historical_data:
            self.historical_data[location_key] = []

        self.historical_data[location_key].append(reading)

        # Limit historical data size
        max_readings = int(self.data_retention_hours * 60 / (self.update_interval / 60))
        if len(self.historical_data[location_key]) > max_readings:
            self.historical_data[location_key] = self.historical_data[location_key][-max_readings:]

    async def _update_context_data(self) -> None:
        """Update aggregated environmental context data"""
        try:
            # Group readings by location
            location_readings = {}

            for location_param_key, readings in self.historical_data.items():
                if not readings:
                    continue

                location, parameter = location_param_key.rsplit('_', 1)
                parameter_enum = EnvironmentalParameter(parameter)

                if location not in location_readings:
                    location_readings[location] = {}

                # Get most recent reading
                latest_reading = max(readings, key=lambda r: r.timestamp)
                location_readings[location][parameter_enum] = latest_reading

            # Create environmental data objects
            for location, readings in location_readings.items():
                comfort_level = self._calculate_comfort_level(readings)

                # Detect occupancy and motion
                occupancy_detected = await self._detect_occupancy(location)
                motion_detected = await self._detect_motion(location)

                env_data = EnvironmentalData(
                    location=location,
                    context_level=ContextLevel.ROOM,  # Could be determined by location hierarchy
                    readings=readings,
                    comfort_level=comfort_level,
                    last_updated=datetime.now(),
                    occupancy_detected=occupancy_detected,
                    motion_detected=motion_detected,
                    people_count=1 if occupancy_detected else 0  # Simple estimation
                )

                # Check if context changed significantly
                old_data = self.environmental_data.get(location)
                if await self._context_changed(old_data, env_data):
                    await self._trigger_context_change_handlers(location, old_data, env_data)

                self.environmental_data[location] = env_data

        except Exception as e:
            logger.error(f"Error updating context data: {e}")

    def _calculate_comfort_level(self, readings: Dict[EnvironmentalParameter, EnvironmentalReading]) -> ComfortLevel:
        """Calculate overall comfort level based on environmental readings"""
        comfort_scores = []

        for parameter, reading in readings.items():
            if parameter in self.comfort_thresholds:
                score = self._get_comfort_score(parameter, reading.value)
                comfort_scores.append(score)

        if not comfort_scores:
            return ComfortLevel.NEUTRAL

        # Use average comfort score
        avg_score = sum(comfort_scores) / len(comfort_scores)

        # Map score to comfort level
        if avg_score >= 4.5:
            return ComfortLevel.VERY_COMFORTABLE
        elif avg_score >= 3.5:
            return ComfortLevel.COMFORTABLE
        elif avg_score >= 2.5:
            return ComfortLevel.NEUTRAL
        elif avg_score >= 1.5:
            return ComfortLevel.UNCOMFORTABLE
        else:
            return ComfortLevel.VERY_UNCOMFORTABLE

    def _get_comfort_score(self, parameter: EnvironmentalParameter, value: float) -> float:
        """Get comfort score for a parameter value (1-5 scale)"""
        thresholds = self.comfort_thresholds.get(parameter, {})

        for comfort_level, (min_val, max_val) in thresholds.items():
            if min_val <= value <= max_val:
                return {
                    ComfortLevel.VERY_COMFORTABLE: 5.0,
                    ComfortLevel.COMFORTABLE: 4.0,
                    ComfortLevel.NEUTRAL: 3.0,
                    ComfortLevel.UNCOMFORTABLE: 2.0,
                    ComfortLevel.VERY_UNCOMFORTABLE: 1.0
                }.get(comfort_level, 3.0)

        return 1.0  # Very uncomfortable if outside all ranges

    async def _detect_occupancy(self, location: str) -> bool:
        """Detect occupancy in location"""
        # Check motion sensors
        motion_devices = [
            device for device in self.device_manager.get_devices_by_location(location)
            if device.has_capability(DeviceCapability.MOTION)
        ]

        for device in motion_devices:
            if device.get_attribute("state") == "on":
                return True

        # Check other occupancy indicators
        # Could include door sensors, light usage, etc.

        return False

    async def _detect_motion(self, location: str) -> bool:
        """Detect recent motion in location"""
        # Check for motion in last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=5)

        location_param_key = f"{location}_{EnvironmentalParameter.MOTION_ACTIVITY.value}"
        readings = self.historical_data.get(location_param_key, [])

        for reading in readings:
            if reading.timestamp > cutoff_time and reading.value > 0:
                return True

        return False

    async def _context_changed(self, old_data: Optional[EnvironmentalData], new_data: EnvironmentalData) -> bool:
        """Check if environmental context changed significantly"""
        if not old_data:
            return True

        # Check for significant changes
        if old_data.comfort_level != new_data.comfort_level:
            return True

        if old_data.occupancy_detected != new_data.occupancy_detected:
            return True

        # Check for significant parameter changes
        for parameter, new_reading in new_data.readings.items():
            old_reading = old_data.get_reading(parameter)
            if old_reading:
                # Define significance thresholds
                thresholds = {
                    EnvironmentalParameter.TEMPERATURE: 1.0,  # 1 degree
                    EnvironmentalParameter.HUMIDITY: 5.0,     # 5%
                    EnvironmentalParameter.LIGHT_LEVEL: 100.0 # 100 lux
                }

                threshold = thresholds.get(parameter, 10.0)
                if abs(new_reading.value - old_reading.value) > threshold:
                    return True

        return False

    async def _check_context_rules(self) -> None:
        """Check context rules and trigger events"""
        for rule in self.context_rules.values():
            if not rule.enabled:
                continue

            try:
                await self._evaluate_context_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating context rule {rule.rule_id}: {e}")

    async def _evaluate_context_rule(self, rule: ContextRule) -> None:
        """Evaluate single context rule"""
        env_data = self.environmental_data.get(rule.location)
        if not env_data:
            return

        reading = env_data.get_reading(rule.parameter)
        if not reading:
            return

        triggered = False
        trigger_reason = ""

        if rule.trigger_type == ContextTriggerType.THRESHOLD:
            if rule.threshold_min is not None and reading.value < rule.threshold_min:
                triggered = True
                trigger_reason = f"Value {reading.value} below minimum threshold {rule.threshold_min}"
            elif rule.threshold_max is not None and reading.value > rule.threshold_max:
                triggered = True
                trigger_reason = f"Value {reading.value} above maximum threshold {rule.threshold_max}"

        elif rule.trigger_type == ContextTriggerType.TREND:
            trend_result = await self._analyze_parameter_trend(rule.location, rule.parameter, rule.trend_duration or 30)
            if trend_result:
                triggered = True
                trigger_reason = f"Trend detected: {trend_result}"

        if triggered:
            trigger = ContextTrigger(
                trigger_id=f"{rule.rule_id}_{int(datetime.now().timestamp())}",
                rule_id=rule.rule_id,
                location=rule.location,
                parameter=rule.parameter,
                trigger_type=rule.trigger_type,
                current_value=reading.value,
                trigger_reason=trigger_reason,
                timestamp=datetime.now()
            )

            self.active_triggers.append(trigger)
            await self._trigger_trigger_handlers(trigger)

    async def _analyze_trends(self) -> None:
        """Analyze environmental trends"""
        for location, env_data in self.environmental_data.items():
            for parameter in env_data.readings.keys():
                try:
                    trend = await self._analyze_parameter_trend(location, parameter, 60)  # 1 hour
                    if trend:
                        logger.debug(f"Trend detected in {location} {parameter.value}: {trend}")
                except Exception as e:
                    logger.error(f"Error analyzing trend for {location} {parameter.value}: {e}")

    async def _analyze_parameter_trend(self, location: str, parameter: EnvironmentalParameter, duration_minutes: int) -> Optional[str]:
        """Analyze trend for specific parameter"""
        location_param_key = f"{location}_{parameter.value}"
        readings = self.historical_data.get(location_param_key, [])

        if len(readings) < self.trend_analysis_points:
            return None

        # Get recent readings
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_readings = [r for r in readings if r.timestamp > cutoff_time]

        if len(recent_readings) < self.trend_analysis_points:
            return None

        # Calculate trend
        values = [r.value for r in recent_readings[-self.trend_analysis_points:]]
        timestamps = [(r.timestamp - recent_readings[0].timestamp).total_seconds() for r in recent_readings[-self.trend_analysis_points:]]

        # Simple linear regression
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)

        if n * sum_x2 - sum_x * sum_x == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Determine trend significance
        if abs(slope) < 0.001:  # Configurable threshold
            return None

        if slope > 0:
            return f"increasing at {slope:.3f} units/second"
        else:
            return f"decreasing at {abs(slope):.3f} units/second"

    async def _detect_patterns(self) -> None:
        """Detect patterns in environmental data"""
        # Could implement pattern detection algorithms
        # For now, placeholder for future implementation
        pass

    async def _cleanup_old_data(self) -> None:
        """Cleanup old historical data"""
        cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)

        for location_param_key in list(self.historical_data.keys()):
            readings = self.historical_data[location_param_key]
            filtered_readings = [r for r in readings if r.timestamp > cutoff_time]
            self.historical_data[location_param_key] = filtered_readings

        # Cleanup old triggers
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_triggers = [t for t in self.active_triggers if t.timestamp > cutoff_time]

    async def _handle_device_update(self, entity_id: str, device: Device) -> None:
        """Handle device state updates"""
        # Environmental data will be updated in the monitoring loop
        pass

    async def _handle_sensor_reading(self, sensor_data: SensorData) -> None:
        """Handle ESP32 sensor readings"""
        await self._process_esp32_sensor(sensor_data)

    async def _trigger_context_change_handlers(
        self,
        location: str,
        old_data: Optional[EnvironmentalData],
        new_data: EnvironmentalData
    ) -> None:
        """Trigger context change handlers"""
        for handler in self.context_change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(location, old_data, new_data)
                else:
                    handler(location, old_data, new_data)
            except Exception as e:
                logger.error(f"Context change handler error: {e}")

    async def _trigger_trigger_handlers(self, trigger: ContextTrigger) -> None:
        """Trigger context trigger handlers"""
        for handler in self.trigger_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(trigger)
                else:
                    handler(trigger)
            except Exception as e:
                logger.error(f"Trigger handler error: {e}")

    def add_context_rule(self, rule: ContextRule) -> bool:
        """Add context rule"""
        try:
            self.context_rules[rule.rule_id] = rule
            logger.info(f"Added context rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add context rule: {e}")
            return False

    def remove_context_rule(self, rule_id: str) -> bool:
        """Remove context rule"""
        if rule_id in self.context_rules:
            del self.context_rules[rule_id]
            logger.info(f"Removed context rule: {rule_id}")
            return True
        return False

    def get_environmental_data(self, location: str) -> Optional[EnvironmentalData]:
        """Get environmental data for location"""
        return self.environmental_data.get(location)

    def get_all_environmental_data(self) -> Dict[str, EnvironmentalData]:
        """Get all environmental data"""
        return self.environmental_data.copy()

    def get_historical_readings(
        self,
        location: str,
        parameter: EnvironmentalParameter,
        hours: int = 24
    ) -> List[EnvironmentalReading]:
        """Get historical readings for location and parameter"""
        location_param_key = f"{location}_{parameter.value}"
        readings = self.historical_data.get(location_param_key, [])

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [r for r in readings if r.timestamp > cutoff_time]

    def get_active_triggers(self) -> List[ContextTrigger]:
        """Get active context triggers"""
        return self.active_triggers.copy()

    def get_context_summary(self) -> Dict[str, Any]:
        """Get environmental context summary"""
        return {
            "total_locations": len(self.environmental_data),
            "monitored_parameters": len(set(
                param for env_data in self.environmental_data.values()
                for param in env_data.readings.keys()
            )),
            "active_rules": len([r for r in self.context_rules.values() if r.enabled]),
            "active_triggers": len(self.active_triggers),
            "locations": {
                location: {
                    "comfort_level": env_data.comfort_level.value,
                    "occupancy": env_data.occupancy_detected,
                    "motion": env_data.motion_detected,
                    "last_updated": env_data.last_updated.isoformat()
                }
                for location, env_data in self.environmental_data.items()
            }
        }

    def register_context_change_handler(self, handler: Callable) -> None:
        """Register context change handler"""
        self.context_change_handlers.append(handler)

    def register_trigger_handler(self, handler: Callable) -> None:
        """Register trigger handler"""
        self.trigger_handlers.append(handler)

    async def shutdown(self) -> None:
        """Shutdown environmental context manager"""
        logger.info("Shutting down environmental context manager...")

        if self.monitoring_task:
            self.monitoring_task.cancel()

        if self.analysis_task:
            self.analysis_task.cancel()

        logger.info("Environmental context manager shutdown complete")


# Utility functions
def create_temperature_rule(
    rule_id: str,
    name: str,
    location: str,
    min_temp: float,
    max_temp: float
) -> ContextRule:
    """Create temperature threshold rule"""
    return ContextRule(
        rule_id=rule_id,
        name=name,
        location=location,
        parameter=EnvironmentalParameter.TEMPERATURE,
        trigger_type=ContextTriggerType.THRESHOLD,
        threshold_min=min_temp,
        threshold_max=max_temp
    )


def create_humidity_rule(
    rule_id: str,
    name: str,
    location: str,
    min_humidity: float,
    max_humidity: float
) -> ContextRule:
    """Create humidity threshold rule"""
    return ContextRule(
        rule_id=rule_id,
        name=name,
        location=location,
        parameter=EnvironmentalParameter.HUMIDITY,
        trigger_type=ContextTriggerType.THRESHOLD,
        threshold_min=min_humidity,
        threshold_max=max_humidity
    )