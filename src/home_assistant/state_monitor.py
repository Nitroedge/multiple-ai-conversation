"""
Advanced device state monitoring system for Home Assistant and ESP32 devices.
Provides real-time state tracking, health monitoring, and performance analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import statistics
import time

logger = logging.getLogger(__name__)

class DeviceHealth(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class DeviceState:
    entity_id: str
    state: str
    attributes: Dict[str, Any]
    last_updated: datetime
    last_changed: datetime
    source: str  # 'home_assistant' or 'esp32'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'state': self.state,
            'attributes': self.attributes,
            'last_updated': self.last_updated.isoformat(),
            'last_changed': self.last_changed.isoformat(),
            'source': self.source
        }

@dataclass
class DeviceMetrics:
    entity_id: str
    response_time_ms: float
    last_seen: datetime
    uptime_percentage: float
    error_count: int
    state_changes_count: int
    avg_response_time: float
    health_status: DeviceHealth

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'response_time_ms': self.response_time_ms,
            'last_seen': self.last_seen.isoformat(),
            'uptime_percentage': self.uptime_percentage,
            'error_count': self.error_count,
            'state_changes_count': self.state_changes_count,
            'avg_response_time': self.avg_response_time,
            'health_status': self.health_status.value
        }

@dataclass
class StateAlert:
    id: str
    entity_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.resolved_at is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'entity_id': self.entity_id,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'triggered_at': self.triggered_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'is_active': self.is_active,
            'metadata': self.metadata
        }

@dataclass
class MonitoringConfig:
    health_check_interval: int = 30  # seconds
    metrics_retention_hours: int = 24
    alert_cooldown_minutes: int = 15
    response_time_threshold_ms: float = 5000.0
    uptime_threshold_percentage: float = 95.0
    max_error_count: int = 5
    enabled_alerts: Set[str] = field(default_factory=lambda: {
        'device_offline', 'slow_response', 'high_error_rate', 'state_stuck'
    })

class DeviceStateMonitor:
    """
    Comprehensive device state monitoring system that tracks:
    - Real-time device states
    - Health and performance metrics
    - Alert generation and management
    - Historical state tracking
    """

    def __init__(self, ha_client=None, esp32_interface=None, config: Optional[MonitoringConfig] = None):
        self.ha_client = ha_client
        self.esp32_interface = esp32_interface
        self.config = config or MonitoringConfig()

        # State tracking
        self.device_states: Dict[str, DeviceState] = {}
        self.device_metrics: Dict[str, DeviceMetrics] = {}
        self.state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Alert management
        self.active_alerts: Dict[str, StateAlert] = {}
        self.alert_history: List[StateAlert] = []
        self.alert_callbacks: List[Callable[[StateAlert], None]] = []
        self.last_alert_times: Dict[str, datetime] = {}

        # Monitoring control
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []

        # Statistics
        self.stats = {
            'total_devices': 0,
            'healthy_devices': 0,
            'warning_devices': 0,
            'critical_devices': 0,
            'offline_devices': 0,
            'total_alerts_today': 0,
            'avg_system_response_time': 0.0
        }

    async def start_monitoring(self) -> None:
        """Start the device monitoring system"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        logger.info("Starting device state monitoring")

        # Start monitoring tasks
        tasks = [
            self._health_check_loop(),
            self._metrics_collection_loop(),
            self._alert_processing_loop(),
            self._cleanup_loop()
        ]

        self.monitoring_tasks = [asyncio.create_task(task) for task in tasks]
        logger.info(f"Started {len(self.monitoring_tasks)} monitoring tasks")

    async def stop_monitoring(self) -> None:
        """Stop the device monitoring system"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        logger.info("Stopping device state monitoring")

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        logger.info("Device state monitoring stopped")

    async def update_device_state(self, entity_id: str, state: str, attributes: Dict[str, Any], source: str) -> None:
        """Update the state of a device"""
        now = datetime.now()

        # Check if state actually changed
        previous_state = self.device_states.get(entity_id)
        state_changed = previous_state is None or previous_state.state != state

        # Create new device state
        device_state = DeviceState(
            entity_id=entity_id,
            state=state,
            attributes=attributes,
            last_updated=now,
            last_changed=now if state_changed else (previous_state.last_changed if previous_state else now),
            source=source
        )

        self.device_states[entity_id] = device_state

        # Add to history
        self.state_history[entity_id].append({
            'timestamp': now,
            'state': state,
            'attributes': attributes,
            'changed': state_changed
        })

        # Update metrics if state changed
        if state_changed and entity_id in self.device_metrics:
            self.device_metrics[entity_id].state_changes_count += 1

        # Check for alerts
        await self._check_device_alerts(entity_id, device_state)

        logger.debug(f"Updated state for {entity_id}: {state} (changed: {state_changed})")

    async def record_response_time(self, entity_id: str, response_time_ms: float) -> None:
        """Record response time for a device"""
        self.response_times[entity_id].append({
            'timestamp': datetime.now(),
            'response_time': response_time_ms
        })

        # Update metrics
        if entity_id in self.device_metrics:
            self.device_metrics[entity_id].response_time_ms = response_time_ms

            # Calculate average response time
            recent_times = [r['response_time'] for r in list(self.response_times[entity_id])[-10:]]
            self.device_metrics[entity_id].avg_response_time = statistics.mean(recent_times) if recent_times else 0.0

    async def get_device_state(self, entity_id: str) -> Optional[DeviceState]:
        """Get current state of a device"""
        return self.device_states.get(entity_id)

    async def get_device_metrics(self, entity_id: str) -> Optional[DeviceMetrics]:
        """Get metrics for a device"""
        return self.device_metrics.get(entity_id)

    async def get_device_history(self, entity_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get state history for a device"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.state_history.get(entity_id, deque())

        return [
            entry for entry in history
            if entry['timestamp'] > cutoff_time
        ]

    async def get_active_alerts(self, entity_id: Optional[str] = None) -> List[StateAlert]:
        """Get active alerts, optionally filtered by entity"""
        alerts = [alert for alert in self.active_alerts.values() if alert.is_active]

        if entity_id:
            alerts = [alert for alert in alerts if alert.entity_id == entity_id]

        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system monitoring statistics"""
        # Update statistics
        total = len(self.device_metrics)
        health_counts = defaultdict(int)

        for metrics in self.device_metrics.values():
            health_counts[metrics.health_status] += 1

        # Calculate average response time
        all_response_times = []
        for times in self.response_times.values():
            if times:
                all_response_times.extend([r['response_time'] for r in times])

        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0.0

        # Count today's alerts
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        alerts_today = len([
            alert for alert in self.alert_history
            if alert.triggered_at >= today_start
        ])

        self.stats.update({
            'total_devices': total,
            'healthy_devices': health_counts[DeviceHealth.HEALTHY],
            'warning_devices': health_counts[DeviceHealth.WARNING],
            'critical_devices': health_counts[DeviceHealth.CRITICAL],
            'offline_devices': health_counts[DeviceHealth.OFFLINE],
            'total_alerts_today': alerts_today,
            'avg_system_response_time': avg_response_time,
            'active_alerts_count': len(self.active_alerts),
            'monitoring_active': self.monitoring_active
        })

        return self.stats.copy()

    def add_alert_callback(self, callback: Callable[[StateAlert], None]) -> None:
        """Add a callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]

            logger.info(f"Resolved alert {alert_id} for {alert.entity_id}")
            return True

        return False

    async def _health_check_loop(self) -> None:
        """Main health checking loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all devices"""
        now = datetime.now()

        # Check Home Assistant devices
        if self.ha_client:
            await self._check_ha_devices(now)

        # Check ESP32 devices
        if self.esp32_interface:
            await self._check_esp32_devices(now)

        # Update overall health metrics
        await self._update_health_metrics()

    async def _check_ha_devices(self, now: datetime) -> None:
        """Check health of Home Assistant devices"""
        if not self.ha_client or not hasattr(self.ha_client, 'get_states'):
            return

        try:
            start_time = time.time()
            states = await self.ha_client.get_states()
            response_time = (time.time() - start_time) * 1000

            for state_data in states:
                entity_id = state_data.get('entity_id')
                if not entity_id:
                    continue

                # Record response time
                await self.record_response_time(entity_id, response_time)

                # Update device state
                await self.update_device_state(
                    entity_id=entity_id,
                    state=state_data.get('state', 'unknown'),
                    attributes=state_data.get('attributes', {}),
                    source='home_assistant'
                )

                # Initialize or update metrics
                await self._update_device_metrics(entity_id, now, response_time)

        except Exception as e:
            logger.error(f"Error checking HA devices: {e}")

    async def _check_esp32_devices(self, now: datetime) -> None:
        """Check health of ESP32 devices"""
        if not self.esp32_interface or not hasattr(self.esp32_interface, 'devices'):
            return

        for device_id, device in self.esp32_interface.devices.items():
            try:
                start_time = time.time()

                # Check device connectivity
                status = await device.get_status()
                response_time = (time.time() - start_time) * 1000

                # Record response time
                await self.record_response_time(device_id, response_time)

                # Update device state
                await self.update_device_state(
                    entity_id=device_id,
                    state='online' if status.get('connected') else 'offline',
                    attributes=status,
                    source='esp32'
                )

                # Initialize or update metrics
                await self._update_device_metrics(device_id, now, response_time)

            except Exception as e:
                logger.error(f"Error checking ESP32 device {device_id}: {e}")

                # Mark as offline
                await self.update_device_state(
                    entity_id=device_id,
                    state='offline',
                    attributes={'error': str(e)},
                    source='esp32'
                )

    async def _update_device_metrics(self, entity_id: str, now: datetime, response_time: float) -> None:
        """Update metrics for a device"""
        if entity_id not in self.device_metrics:
            self.device_metrics[entity_id] = DeviceMetrics(
                entity_id=entity_id,
                response_time_ms=response_time,
                last_seen=now,
                uptime_percentage=100.0,
                error_count=0,
                state_changes_count=0,
                avg_response_time=response_time,
                health_status=DeviceHealth.UNKNOWN
            )
        else:
            metrics = self.device_metrics[entity_id]
            metrics.response_time_ms = response_time
            metrics.last_seen = now

            # Calculate uptime percentage (simplified)
            time_diff = (now - metrics.last_seen).total_seconds()
            if time_diff > 300:  # 5 minutes
                metrics.uptime_percentage = max(0, metrics.uptime_percentage - 1)
            else:
                metrics.uptime_percentage = min(100, metrics.uptime_percentage + 0.1)

    async def _update_health_metrics(self) -> None:
        """Update health status for all devices"""
        now = datetime.now()

        for entity_id, metrics in self.device_metrics.items():
            old_health = metrics.health_status

            # Determine health status
            time_since_seen = (now - metrics.last_seen).total_seconds()

            if time_since_seen > 300:  # 5 minutes offline
                metrics.health_status = DeviceHealth.OFFLINE
            elif metrics.error_count >= self.config.max_error_count:
                metrics.health_status = DeviceHealth.CRITICAL
            elif (metrics.response_time_ms > self.config.response_time_threshold_ms or
                  metrics.uptime_percentage < self.config.uptime_threshold_percentage):
                metrics.health_status = DeviceHealth.WARNING
            else:
                metrics.health_status = DeviceHealth.HEALTHY

            # Log health changes
            if old_health != metrics.health_status:
                logger.info(f"Device {entity_id} health changed: {old_health.value} -> {metrics.health_status.value}")

    async def _check_device_alerts(self, entity_id: str, device_state: DeviceState) -> None:
        """Check if any alerts should be triggered for a device"""
        now = datetime.now()

        # Check alert cooldown
        last_alert_time = self.last_alert_times.get(entity_id)
        if last_alert_time and (now - last_alert_time).total_seconds() < self.config.alert_cooldown_minutes * 60:
            return

        metrics = self.device_metrics.get(entity_id)
        if not metrics:
            return

        # Device offline alert
        if ('device_offline' in self.config.enabled_alerts and
            device_state.state in ['offline', 'unavailable']):
            await self._trigger_alert(
                entity_id=entity_id,
                alert_type='device_offline',
                severity=AlertSeverity.CRITICAL,
                message=f"Device {entity_id} is offline",
                metadata={'last_seen': metrics.last_seen.isoformat()}
            )

        # Slow response alert
        if ('slow_response' in self.config.enabled_alerts and
            metrics.response_time_ms > self.config.response_time_threshold_ms):
            await self._trigger_alert(
                entity_id=entity_id,
                alert_type='slow_response',
                severity=AlertSeverity.WARNING,
                message=f"Device {entity_id} has slow response time: {metrics.response_time_ms:.1f}ms",
                metadata={'response_time': metrics.response_time_ms}
            )

        # High error rate alert
        if ('high_error_rate' in self.config.enabled_alerts and
            metrics.error_count >= self.config.max_error_count):
            await self._trigger_alert(
                entity_id=entity_id,
                alert_type='high_error_rate',
                severity=AlertSeverity.CRITICAL,
                message=f"Device {entity_id} has high error rate: {metrics.error_count} errors",
                metadata={'error_count': metrics.error_count}
            )

        # State stuck alert (no state changes in last hour)
        if 'state_stuck' in self.config.enabled_alerts:
            time_since_change = (now - device_state.last_changed).total_seconds()
            if time_since_change > 3600:  # 1 hour
                await self._trigger_alert(
                    entity_id=entity_id,
                    alert_type='state_stuck',
                    severity=AlertSeverity.WARNING,
                    message=f"Device {entity_id} state hasn't changed for {time_since_change/3600:.1f} hours",
                    metadata={'last_changed': device_state.last_changed.isoformat()}
                )

    async def _trigger_alert(self, entity_id: str, alert_type: str, severity: AlertSeverity,
                           message: str, metadata: Dict[str, Any] = None) -> None:
        """Trigger a new alert"""
        alert_id = f"{entity_id}_{alert_type}_{int(time.time())}"

        alert = StateAlert(
            id=alert_id,
            entity_id=entity_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            triggered_at=datetime.now(),
            metadata=metadata or {}
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[entity_id] = alert.triggered_at

        logger.warning(f"Alert triggered: {alert.message}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def _metrics_collection_loop(self) -> None:
        """Collect and aggregate metrics periodically"""
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect metrics every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(10)

    async def _collect_metrics(self) -> None:
        """Collect system-wide metrics"""
        # This could be expanded to collect more detailed metrics
        # For now, just update the system stats
        await self.get_system_stats()

    async def _alert_processing_loop(self) -> None:
        """Process and manage alerts"""
        while self.monitoring_active:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Process alerts every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(10)

    async def _process_alerts(self) -> None:
        """Process active alerts and auto-resolve if conditions are met"""
        now = datetime.now()
        alerts_to_resolve = []

        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve alerts if conditions are no longer met
            if await self._should_auto_resolve_alert(alert, now):
                alerts_to_resolve.append(alert_id)

        # Resolve alerts
        for alert_id in alerts_to_resolve:
            await self.resolve_alert(alert_id)

    async def _should_auto_resolve_alert(self, alert: StateAlert, now: datetime) -> bool:
        """Check if an alert should be auto-resolved"""
        metrics = self.device_metrics.get(alert.entity_id)
        device_state = self.device_states.get(alert.entity_id)

        if not metrics or not device_state:
            return False

        # Auto-resolve based on alert type
        if alert.alert_type == 'device_offline' and device_state.state not in ['offline', 'unavailable']:
            return True
        elif alert.alert_type == 'slow_response' and metrics.response_time_ms <= self.config.response_time_threshold_ms:
            return True
        elif alert.alert_type == 'high_error_rate' and metrics.error_count < self.config.max_error_count:
            return True

        return False

    async def _cleanup_loop(self) -> None:
        """Clean up old data periodically"""
        while self.monitoring_active:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_data(self) -> None:
        """Clean up old historical data"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.metrics_retention_hours)

        # Clean up old alert history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.triggered_at > cutoff_time
        ]

        # Clean up old response time data
        for entity_id in list(self.response_times.keys()):
            old_times = self.response_times[entity_id]
            new_times = deque([
                entry for entry in old_times
                if entry['timestamp'] > cutoff_time
            ], maxlen=100)
            self.response_times[entity_id] = new_times

        logger.debug("Completed data cleanup")