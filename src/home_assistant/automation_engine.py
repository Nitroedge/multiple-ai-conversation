"""
Automation engine for Home Assistant integration with intelligent rule processing
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta, time
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import re

from pydantic import BaseModel, Field, validator

from .device_manager import DeviceManager, Device, DeviceCapability, DeviceState
from .ha_client import HomeAssistantClient

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Automation trigger types"""
    STATE_CHANGE = "state_change"
    TIME = "time"
    INTERVAL = "interval"
    DEVICE_EVENT = "device_event"
    SENSOR_VALUE = "sensor_value"
    VOICE_COMMAND = "voice_command"
    SCENE_ACTIVATION = "scene_activation"
    WEBHOOK = "webhook"
    MQTT = "mqtt"


class ConditionType(str, Enum):
    """Automation condition types"""
    STATE = "state"
    TIME = "time"
    TEMPLATE = "template"
    DEVICE = "device"
    ZONE = "zone"
    NUMERIC_STATE = "numeric_state"
    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(str, Enum):
    """Automation action types"""
    DEVICE_CONTROL = "device_control"
    SERVICE_CALL = "service_call"
    NOTIFICATION = "notification"
    SCRIPT = "script"
    SCENE = "scene"
    DELAY = "delay"
    WAIT = "wait"
    CONDITION = "condition"
    VOICE_RESPONSE = "voice_response"


class AutomationMode(str, Enum):
    """Automation execution modes"""
    SINGLE = "single"
    RESTART = "restart"
    QUEUED = "queued"
    PARALLEL = "parallel"


@dataclass
class AutomationTrigger:
    """Automation trigger definition"""
    trigger_id: str
    trigger_type: TriggerType
    entity_id: Optional[str] = None
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    at_time: Optional[str] = None  # HH:MM format
    interval: Optional[int] = None  # seconds
    value_template: Optional[str] = None
    above: Optional[float] = None
    below: Optional[float] = None
    webhook_id: Optional[str] = None
    event_type: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutomationCondition:
    """Automation condition definition"""
    condition_id: str
    condition_type: ConditionType
    entity_id: Optional[str] = None
    state: Optional[str] = None
    above: Optional[float] = None
    below: Optional[float] = None
    attribute: Optional[str] = None
    value_template: Optional[str] = None
    after: Optional[str] = None  # HH:MM format
    before: Optional[str] = None  # HH:MM format
    weekday: Optional[List[str]] = None
    conditions: Optional[List['AutomationCondition']] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.conditions:
            data['conditions'] = [c.to_dict() for c in self.conditions]
        return data


@dataclass
class AutomationAction:
    """Automation action definition"""
    action_id: str
    action_type: ActionType
    entity_id: Optional[str] = None
    service: Optional[str] = None
    service_data: Optional[Dict[str, Any]] = None
    delay: Optional[int] = None  # seconds
    message: Optional[str] = None
    title: Optional[str] = None
    target: Optional[str] = None
    wait_template: Optional[str] = None
    timeout: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutomationRule(BaseModel):
    """Complete automation rule"""
    rule_id: str = Field(description="Unique rule identifier")
    name: str = Field(description="Human-readable rule name")
    description: Optional[str] = Field(default=None, description="Rule description")

    # Rule configuration
    triggers: List[AutomationTrigger] = Field(description="Automation triggers")
    conditions: List[AutomationCondition] = Field(default_factory=list, description="Automation conditions")
    actions: List[AutomationAction] = Field(description="Automation actions")

    # Execution settings
    mode: AutomationMode = Field(default=AutomationMode.SINGLE, description="Execution mode")
    max_executions: int = Field(default=10, description="Maximum parallel executions")

    # State and metadata
    enabled: bool = Field(default=True, description="Rule enabled state")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_triggered: Optional[datetime] = Field(default=None, description="Last trigger timestamp")
    trigger_count: int = Field(default=0, description="Total trigger count")

    # Advanced settings
    priority: int = Field(default=0, description="Execution priority (higher = first)")
    timeout: int = Field(default=300, description="Execution timeout in seconds")
    retry_count: int = Field(default=0, description="Retry count on failure")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "triggers": [t.to_dict() for t in self.triggers],
            "conditions": [c.to_dict() for c in self.conditions],
            "actions": [a.to_dict() for a in self.actions],
            "mode": self.mode.value,
            "max_executions": self.max_executions,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
            "priority": self.priority,
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }


class AutomationEngine:
    """Main automation engine"""

    def __init__(self, device_manager: DeviceManager, ha_client: HomeAssistantClient):
        self.device_manager = device_manager
        self.ha_client = ha_client

        # Rule storage
        self.rules: Dict[str, AutomationRule] = {}
        self.active_executions: Dict[str, List[asyncio.Task]] = {}

        # Event handlers
        self.trigger_handlers: List[Callable] = []
        self.execution_handlers: List[Callable] = []

        # State tracking
        self.entity_states: Dict[str, Dict[str, Any]] = {}
        self.last_trigger_times: Dict[str, datetime] = {}

        # Performance metrics
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.average_execution_time = 0.0

        # Time-based triggers
        self.time_triggers: Dict[str, asyncio.Task] = {}

    async def initialize(self) -> None:
        """Initialize automation engine"""
        try:
            logger.info("Initializing automation engine...")

            # Register device state change handler
            self.device_manager.register_state_update_handler(self._handle_device_state_change)

            # Register HA event handlers
            self.ha_client.register_event_handler("state_changed", self._handle_ha_event)

            # Start time-based trigger scheduler
            asyncio.create_task(self._time_trigger_scheduler())

            logger.info("Automation engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize automation engine: {e}")
            raise

    async def add_rule(self, rule: AutomationRule) -> bool:
        """Add automation rule"""
        try:
            # Validate rule
            validation_errors = await self._validate_rule(rule)
            if validation_errors:
                logger.error(f"Rule validation failed: {validation_errors}")
                return False

            # Store rule
            self.rules[rule.rule_id] = rule

            # Initialize active executions tracking
            self.active_executions[rule.rule_id] = []

            # Setup time-based triggers
            await self._setup_time_triggers(rule)

            logger.info(f"Added automation rule: {rule.name} ({rule.rule_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add automation rule: {e}")
            return False

    async def _validate_rule(self, rule: AutomationRule) -> List[str]:
        """Validate automation rule"""
        errors = []

        # Check for empty triggers and actions
        if not rule.triggers:
            errors.append("Rule must have at least one trigger")

        if not rule.actions:
            errors.append("Rule must have at least one action")

        # Validate entity references
        for trigger in rule.triggers:
            if trigger.entity_id and not await self._entity_exists(trigger.entity_id):
                errors.append(f"Trigger entity not found: {trigger.entity_id}")

        for condition in rule.conditions:
            if condition.entity_id and not await self._entity_exists(condition.entity_id):
                errors.append(f"Condition entity not found: {condition.entity_id}")

        for action in rule.actions:
            if action.entity_id and not await self._entity_exists(action.entity_id):
                errors.append(f"Action entity not found: {action.entity_id}")

        return errors

    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if entity exists"""
        return self.device_manager.get_device(entity_id) is not None

    async def _setup_time_triggers(self, rule: AutomationRule) -> None:
        """Setup time-based triggers for rule"""
        for trigger in rule.triggers:
            if trigger.trigger_type == TriggerType.TIME and trigger.at_time:
                trigger_key = f"{rule.rule_id}_{trigger.trigger_id}"
                task = asyncio.create_task(self._schedule_time_trigger(rule, trigger))
                self.time_triggers[trigger_key] = task

            elif trigger.trigger_type == TriggerType.INTERVAL and trigger.interval:
                trigger_key = f"{rule.rule_id}_{trigger.trigger_id}"
                task = asyncio.create_task(self._schedule_interval_trigger(rule, trigger))
                self.time_triggers[trigger_key] = task

    async def _schedule_time_trigger(self, rule: AutomationRule, trigger: AutomationTrigger) -> None:
        """Schedule time-based trigger"""
        try:
            while rule.enabled and trigger.enabled:
                # Parse time
                hour, minute = map(int, trigger.at_time.split(':'))

                # Calculate next trigger time
                now = datetime.now()
                trigger_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

                if trigger_time <= now:
                    trigger_time += timedelta(days=1)

                # Wait until trigger time
                sleep_duration = (trigger_time - now).total_seconds()
                await asyncio.sleep(sleep_duration)

                # Execute rule
                if rule.enabled and trigger.enabled:
                    await self._execute_rule(rule, trigger)

        except asyncio.CancelledError:
            logger.debug(f"Time trigger cancelled for rule {rule.rule_id}")
        except Exception as e:
            logger.error(f"Time trigger error for rule {rule.rule_id}: {e}")

    async def _schedule_interval_trigger(self, rule: AutomationRule, trigger: AutomationTrigger) -> None:
        """Schedule interval-based trigger"""
        try:
            while rule.enabled and trigger.enabled:
                await asyncio.sleep(trigger.interval)

                if rule.enabled and trigger.enabled:
                    await self._execute_rule(rule, trigger)

        except asyncio.CancelledError:
            logger.debug(f"Interval trigger cancelled for rule {rule.rule_id}")
        except Exception as e:
            logger.error(f"Interval trigger error for rule {rule.rule_id}: {e}")

    async def _time_trigger_scheduler(self) -> None:
        """Main scheduler for time-based triggers"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Clean up completed tasks
                for trigger_key, task in list(self.time_triggers.items()):
                    if task.done():
                        del self.time_triggers[trigger_key]

            except Exception as e:
                logger.error(f"Time trigger scheduler error: {e}")

    async def _handle_device_state_change(self, entity_id: str, device: Device) -> None:
        """Handle device state change events"""
        try:
            # Store current state for comparison
            old_state = self.entity_states.get(entity_id, {})
            new_state = {
                "state": device.state.value,
                "attributes": device.attributes.copy()
            }
            self.entity_states[entity_id] = new_state

            # Check triggers for all rules
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                for trigger in rule.triggers:
                    if not trigger.enabled:
                        continue

                    if await self._check_state_trigger(trigger, entity_id, old_state, new_state):
                        await self._execute_rule(rule, trigger)

        except Exception as e:
            logger.error(f"Error handling device state change: {e}")

    async def _handle_ha_event(self, event_data: Dict[str, Any]) -> None:
        """Handle Home Assistant events"""
        try:
            entity_id = event_data.get("entity_id")
            if not entity_id:
                return

            # Process state change for HA entities
            new_state_data = event_data.get("new_state", {})
            old_state_data = event_data.get("old_state", {})

            old_state = {
                "state": old_state_data.get("state") if old_state_data else None,
                "attributes": old_state_data.get("attributes", {}) if old_state_data else {}
            }

            new_state = {
                "state": new_state_data.get("state"),
                "attributes": new_state_data.get("attributes", {})
            }

            self.entity_states[entity_id] = new_state

            # Check triggers
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                for trigger in rule.triggers:
                    if not trigger.enabled:
                        continue

                    if await self._check_state_trigger(trigger, entity_id, old_state, new_state):
                        await self._execute_rule(rule, trigger)

        except Exception as e:
            logger.error(f"Error handling HA event: {e}")

    async def _check_state_trigger(
        self,
        trigger: AutomationTrigger,
        entity_id: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any]
    ) -> bool:
        """Check if state change matches trigger conditions"""
        if trigger.trigger_type != TriggerType.STATE_CHANGE:
            return False

        if trigger.entity_id and trigger.entity_id != entity_id:
            return False

        # Check from_state condition
        if trigger.from_state:
            if old_state.get("state") != trigger.from_state:
                return False

        # Check to_state condition
        if trigger.to_state:
            if new_state.get("state") != trigger.to_state:
                return False

        # Check numeric conditions for sensor values
        if trigger.trigger_type == TriggerType.SENSOR_VALUE:
            try:
                value = float(new_state.get("state", 0))

                if trigger.above is not None and value <= trigger.above:
                    return False

                if trigger.below is not None and value >= trigger.below:
                    return False

            except (ValueError, TypeError):
                return False

        return True

    async def _execute_rule(self, rule: AutomationRule, trigger: AutomationTrigger) -> None:
        """Execute automation rule"""
        try:
            # Check execution mode
            active_count = len(self.active_executions[rule.rule_id])

            if rule.mode == AutomationMode.SINGLE and active_count > 0:
                logger.debug(f"Skipping rule execution (single mode): {rule.name}")
                return

            elif rule.mode == AutomationMode.RESTART and active_count > 0:
                # Cancel existing executions
                for task in self.active_executions[rule.rule_id]:
                    task.cancel()
                self.active_executions[rule.rule_id].clear()

            elif active_count >= rule.max_executions:
                logger.warning(f"Maximum executions reached for rule: {rule.name}")
                return

            # Create execution task
            execution_task = asyncio.create_task(
                self._execute_rule_actions(rule, trigger)
            )
            self.active_executions[rule.rule_id].append(execution_task)

            # Update rule statistics
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1

            logger.info(f"Executing automation rule: {rule.name} (trigger: {trigger.trigger_id})")

        except Exception as e:
            logger.error(f"Failed to execute rule {rule.name}: {e}")

    async def _execute_rule_actions(self, rule: AutomationRule, trigger: AutomationTrigger) -> None:
        """Execute rule actions with conditions check"""
        execution_start = datetime.now()
        success = False

        try:
            # Check conditions
            if rule.conditions:
                conditions_met = await self._check_conditions(rule.conditions)
                if not conditions_met:
                    logger.debug(f"Conditions not met for rule: {rule.name}")
                    return

            # Execute actions
            for action in rule.actions:
                if not action.enabled:
                    continue

                await self._execute_action(action, rule, trigger)

            success = True
            self.success_count += 1

        except Exception as e:
            logger.error(f"Error executing rule actions for {rule.name}: {e}")
            self.error_count += 1

            # Retry if configured
            if rule.retry_count > 0:
                await asyncio.sleep(1)  # Wait before retry
                for retry in range(rule.retry_count):
                    try:
                        for action in rule.actions:
                            if action.enabled:
                                await self._execute_action(action, rule, trigger)
                        success = True
                        break
                    except Exception as retry_error:
                        logger.error(f"Retry {retry + 1} failed for rule {rule.name}: {retry_error}")

        finally:
            # Update metrics
            execution_time = (datetime.now() - execution_start).total_seconds()
            self.execution_count += 1
            self.average_execution_time = (
                (self.average_execution_time * (self.execution_count - 1) + execution_time) /
                self.execution_count
            )

            # Remove from active executions
            active_tasks = self.active_executions[rule.rule_id]
            current_task = asyncio.current_task()
            if current_task in active_tasks:
                active_tasks.remove(current_task)

            # Trigger execution handlers
            await self._trigger_execution_handlers(rule, trigger, success, execution_time)

    async def _check_conditions(self, conditions: List[AutomationCondition]) -> bool:
        """Check if all conditions are met"""
        for condition in conditions:
            if not condition.enabled:
                continue

            if not await self._evaluate_condition(condition):
                return False

        return True

    async def _evaluate_condition(self, condition: AutomationCondition) -> bool:
        """Evaluate single condition"""
        try:
            if condition.condition_type == ConditionType.STATE:
                return await self._check_state_condition(condition)

            elif condition.condition_type == ConditionType.TIME:
                return await self._check_time_condition(condition)

            elif condition.condition_type == ConditionType.NUMERIC_STATE:
                return await self._check_numeric_condition(condition)

            elif condition.condition_type == ConditionType.AND:
                if condition.conditions:
                    return all(await self._evaluate_condition(c) for c in condition.conditions)

            elif condition.condition_type == ConditionType.OR:
                if condition.conditions:
                    return any(await self._evaluate_condition(c) for c in condition.conditions)

            elif condition.condition_type == ConditionType.NOT:
                if condition.conditions and len(condition.conditions) == 1:
                    return not await self._evaluate_condition(condition.conditions[0])

            return True

        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False

    async def _check_state_condition(self, condition: AutomationCondition) -> bool:
        """Check state-based condition"""
        if not condition.entity_id:
            return False

        current_state = self.entity_states.get(condition.entity_id, {})

        if condition.state:
            return current_state.get("state") == condition.state

        if condition.attribute and condition.value_template:
            attribute_value = current_state.get("attributes", {}).get(condition.attribute)
            # Simple template evaluation (could be expanded)
            return str(attribute_value) == condition.value_template

        return True

    async def _check_time_condition(self, condition: AutomationCondition) -> bool:
        """Check time-based condition"""
        now = datetime.now().time()

        if condition.after:
            after_time = time.fromisoformat(condition.after)
            if now < after_time:
                return False

        if condition.before:
            before_time = time.fromisoformat(condition.before)
            if now > before_time:
                return False

        if condition.weekday:
            current_weekday = datetime.now().strftime("%A").lower()
            if current_weekday not in [wd.lower() for wd in condition.weekday]:
                return False

        return True

    async def _check_numeric_condition(self, condition: AutomationCondition) -> bool:
        """Check numeric state condition"""
        if not condition.entity_id:
            return False

        current_state = self.entity_states.get(condition.entity_id, {})

        try:
            if condition.attribute:
                value = float(current_state.get("attributes", {}).get(condition.attribute, 0))
            else:
                value = float(current_state.get("state", 0))

            if condition.above is not None and value <= condition.above:
                return False

            if condition.below is not None and value >= condition.below:
                return False

            return True

        except (ValueError, TypeError):
            return False

    async def _execute_action(self, action: AutomationAction, rule: AutomationRule, trigger: AutomationTrigger) -> None:
        """Execute single action"""
        try:
            if action.action_type == ActionType.DEVICE_CONTROL:
                await self._execute_device_control(action)

            elif action.action_type == ActionType.SERVICE_CALL:
                await self._execute_service_call(action)

            elif action.action_type == ActionType.DELAY:
                if action.delay:
                    await asyncio.sleep(action.delay)

            elif action.action_type == ActionType.NOTIFICATION:
                await self._execute_notification(action)

            elif action.action_type == ActionType.VOICE_RESPONSE:
                await self._execute_voice_response(action)

            # Add more action types as needed

        except Exception as e:
            logger.error(f"Error executing action {action.action_id}: {e}")
            raise

    async def _execute_device_control(self, action: AutomationAction) -> None:
        """Execute device control action"""
        if not action.entity_id:
            return

        success = await self.device_manager.control_device(
            action.entity_id,
            action.service or "toggle",
            action.service_data
        )

        if not success:
            raise Exception(f"Failed to control device: {action.entity_id}")

    async def _execute_service_call(self, action: AutomationAction) -> None:
        """Execute Home Assistant service call"""
        if not action.service:
            return

        service_parts = action.service.split(".")
        if len(service_parts) != 2:
            raise ValueError(f"Invalid service format: {action.service}")

        domain, service = service_parts

        await self.ha_client.call_service(
            domain,
            service,
            action.entity_id,
            action.service_data or {}
        )

    async def _execute_notification(self, action: AutomationAction) -> None:
        """Execute notification action"""
        # Could integrate with notification service
        logger.info(f"Notification: {action.title} - {action.message}")

    async def _execute_voice_response(self, action: AutomationAction) -> None:
        """Execute voice response action"""
        # Could integrate with voice system
        logger.info(f"Voice response: {action.message}")

    async def _trigger_execution_handlers(
        self,
        rule: AutomationRule,
        trigger: AutomationTrigger,
        success: bool,
        execution_time: float
    ) -> None:
        """Trigger execution event handlers"""
        for handler in self.execution_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(rule, trigger, success, execution_time)
                else:
                    handler(rule, trigger, success, execution_time)
            except Exception as e:
                logger.error(f"Execution handler error: {e}")

    def get_rule(self, rule_id: str) -> Optional[AutomationRule]:
        """Get automation rule by ID"""
        return self.rules.get(rule_id)

    def get_all_rules(self) -> List[AutomationRule]:
        """Get all automation rules"""
        return list(self.rules.values())

    def get_enabled_rules(self) -> List[AutomationRule]:
        """Get enabled automation rules"""
        return [rule for rule in self.rules.values() if rule.enabled]

    async def enable_rule(self, rule_id: str) -> bool:
        """Enable automation rule"""
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = True
            await self._setup_time_triggers(rule)
            logger.info(f"Enabled automation rule: {rule.name}")
            return True
        return False

    async def disable_rule(self, rule_id: str) -> bool:
        """Disable automation rule"""
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = False

            # Cancel time triggers
            for trigger_key in list(self.time_triggers.keys()):
                if trigger_key.startswith(rule_id):
                    self.time_triggers[trigger_key].cancel()
                    del self.time_triggers[trigger_key]

            # Cancel active executions
            for task in self.active_executions.get(rule_id, []):
                task.cancel()
            self.active_executions[rule_id] = []

            logger.info(f"Disabled automation rule: {rule.name}")
            return True
        return False

    async def remove_rule(self, rule_id: str) -> bool:
        """Remove automation rule"""
        if rule_id in self.rules:
            await self.disable_rule(rule_id)
            del self.rules[rule_id]
            if rule_id in self.active_executions:
                del self.active_executions[rule_id]
            logger.info(f"Removed automation rule: {rule_id}")
            return True
        return False

    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len(self.get_enabled_rules()),
            "active_executions": sum(len(tasks) for tasks in self.active_executions.values()),
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / max(1, self.execution_count),
            "average_execution_time": self.average_execution_time,
            "time_triggers_active": len(self.time_triggers)
        }

    def register_trigger_handler(self, handler: Callable) -> None:
        """Register trigger event handler"""
        self.trigger_handlers.append(handler)

    def register_execution_handler(self, handler: Callable) -> None:
        """Register execution event handler"""
        self.execution_handlers.append(handler)


# Utility functions for creating common automation patterns
def create_simple_automation(
    rule_id: str,
    name: str,
    trigger_entity: str,
    trigger_state: str,
    action_entity: str,
    action_service: str
) -> AutomationRule:
    """Create simple state-based automation"""
    trigger = AutomationTrigger(
        trigger_id="main_trigger",
        trigger_type=TriggerType.STATE_CHANGE,
        entity_id=trigger_entity,
        to_state=trigger_state
    )

    action = AutomationAction(
        action_id="main_action",
        action_type=ActionType.DEVICE_CONTROL,
        entity_id=action_entity,
        service=action_service
    )

    return AutomationRule(
        rule_id=rule_id,
        name=name,
        triggers=[trigger],
        actions=[action]
    )


def create_time_based_automation(
    rule_id: str,
    name: str,
    trigger_time: str,
    action_entity: str,
    action_service: str
) -> AutomationRule:
    """Create time-based automation"""
    trigger = AutomationTrigger(
        trigger_id="time_trigger",
        trigger_type=TriggerType.TIME,
        at_time=trigger_time
    )

    action = AutomationAction(
        action_id="time_action",
        action_type=ActionType.DEVICE_CONTROL,
        entity_id=action_entity,
        service=action_service
    )

    return AutomationRule(
        rule_id=rule_id,
        name=name,
        triggers=[trigger],
        actions=[action]
    )