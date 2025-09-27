"""
Home Assistant Integration Module

This module provides comprehensive integration with Home Assistant and ESP32 devices
for the multi-agent conversation engine. It includes device management, automation,
environmental monitoring, voice control, state monitoring, workflow automation,
and security management.

Key Components:
- HomeAssistantClient: Interface with Home Assistant API
- ESP32Interface: Direct hardware communication with ESP32 devices
- DeviceManager: Unified device discovery and control
- AutomationEngine: Rule-based automation system
- EnvironmentalContextManager: Environmental monitoring and analysis
- HomeVoiceCommandProcessor: Natural language processing for home automation
- DeviceStateMonitor: Real-time device state and health monitoring
- WorkflowEngine: Advanced workflow automation with conditional logic
- SecurityManager: Authentication, authorization, and security monitoring
"""

from .ha_client import (
    HomeAssistantClient,
    HAConfiguration,
    HAConnectionError,
    HAAuthenticationError
)
from .device_manager import (
    DeviceManager,
    Device,
    DeviceType,
    DeviceState,
    DeviceCapability
)
from .automation_engine import (
    AutomationEngine,
    AutomationRule,
    AutomationTrigger,
    AutomationAction,
    AutomationCondition
)
from .esp32_interface import (
    ESP32Interface,
    ESP32Device,
    ESP32Configuration,
    SensorData,
    ControlCommand
)
from .environmental_context import (
    EnvironmentalContextManager,
    EnvironmentalData,
    ContextRule,
    ContextTrigger
)
from .voice_commands import (
    HomeVoiceCommandProcessor,
    HomeCommand,
    HomeCommandType,
    DeviceSelector
)
from .state_monitor import (
    DeviceStateMonitor,
    MonitoringConfig,
    DeviceHealth,
    AlertSeverity
)
from .workflow_engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowStatus,
    StepStatus
)
from .security_manager import (
    SecurityManager,
    SecurityConfig,
    UserRole,
    Permission
)
from .integration_tests import TestRunner

__all__ = [
    # Core Home Assistant Client
    "HomeAssistantClient",
    "HAConfiguration",
    "HAConnectionError",
    "HAAuthenticationError",

    # Device Management
    "DeviceManager",
    "Device",
    "DeviceType",
    "DeviceState",
    "DeviceCapability",

    # Automation Engine
    "AutomationEngine",
    "AutomationRule",
    "AutomationTrigger",
    "AutomationAction",
    "AutomationCondition",

    # ESP32 Hardware Interface
    "ESP32Interface",
    "ESP32Device",
    "ESP32Configuration",
    "SensorData",
    "ControlCommand",

    # Environmental Context
    "EnvironmentalContextManager",
    "EnvironmentalData",
    "ContextRule",
    "ContextTrigger",

    # Voice Command Processing
    "HomeVoiceCommandProcessor",
    "HomeCommand",
    "HomeCommandType",
    "DeviceSelector",

    # State Monitoring
    "DeviceStateMonitor",
    "MonitoringConfig",
    "DeviceHealth",
    "AlertSeverity",

    # Workflow Engine
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStatus",
    "StepStatus",

    # Security Management
    "SecurityManager",
    "SecurityConfig",
    "UserRole",
    "Permission",

    # Testing
    "TestRunner"
]