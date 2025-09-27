"""
Advanced automation workflow engine for complex home automation scenarios.
Supports sequential workflows, parallel execution, conditional logic, and error handling.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowTriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    STATE_CHANGE = "state_change"
    WEBHOOK = "webhook"
    GEOFENCE = "geofence"
    CONDITION_MET = "condition_met"

@dataclass
class WorkflowContext:
    """Context data that flows through workflow execution"""
    workflow_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None

    def get_variable(self, name: str, default: Any = None) -> Any:
        return self.variables.get(name, default)

    def set_variable(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get_step_result(self, step_id: str, key: str = None, default: Any = None) -> Any:
        if step_id not in self.step_results:
            return default

        if key is None:
            return self.step_results[step_id]

        return self.step_results[step_id].get(key, default)

class WorkflowStep(ABC):
    """Abstract base class for workflow steps"""

    def __init__(self, step_id: str, name: str, description: str = "",
                 retry_count: int = 0, timeout_seconds: int = 30):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.retry_count = retry_count
        self.timeout_seconds = timeout_seconds
        self.status = StepStatus.PENDING
        self.error_message: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute the workflow step and return results"""
        pass

    async def can_execute(self, context: WorkflowContext) -> bool:
        """Check if this step can be executed (pre-conditions)"""
        return True

    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class DeviceControlStep(WorkflowStep):
    """Step to control a device"""

    def __init__(self, step_id: str, entity_id: str, action: str,
                 parameters: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(step_id, f"Control {entity_id}", **kwargs)
        self.entity_id = entity_id
        self.action = action
        self.parameters = parameters or {}

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute device control"""
        # This would integrate with the device manager
        logger.info(f"Controlling device {self.entity_id}: {self.action}")

        # Simulate device control
        await asyncio.sleep(0.1)

        return {
            'entity_id': self.entity_id,
            'action': self.action,
            'parameters': self.parameters,
            'success': True
        }

class DelayStep(WorkflowStep):
    """Step to add delays in workflow"""

    def __init__(self, step_id: str, delay_seconds: float, **kwargs):
        super().__init__(step_id, f"Wait {delay_seconds}s", **kwargs)
        self.delay_seconds = delay_seconds

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute delay"""
        logger.info(f"Delaying for {self.delay_seconds} seconds")
        await asyncio.sleep(self.delay_seconds)

        return {
            'delay_seconds': self.delay_seconds,
            'completed_at': datetime.now().isoformat()
        }

class ConditionStep(WorkflowStep):
    """Step to evaluate conditions"""

    def __init__(self, step_id: str, condition: str, variables: Dict[str, Any] = None, **kwargs):
        super().__init__(step_id, f"Check condition: {condition}", **kwargs)
        self.condition = condition
        self.condition_variables = variables or {}

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Evaluate condition"""
        # Simple condition evaluation (could be extended with a proper expression parser)
        result = await self._evaluate_condition(context)

        return {
            'condition': self.condition,
            'result': result,
            'variables_used': self.condition_variables
        }

    async def _evaluate_condition(self, context: WorkflowContext) -> bool:
        """Evaluate the condition (simplified implementation)"""
        # This is a simplified condition evaluator
        # In production, you'd want a proper expression parser

        # Replace variables in condition
        condition_text = self.condition
        for var_name, var_value in context.variables.items():
            condition_text = condition_text.replace(f"${var_name}", str(var_value))

        # Simple evaluation for basic conditions
        try:
            # WARNING: eval is dangerous in production - use a proper expression parser
            return eval(condition_text)
        except Exception as e:
            logger.error(f"Error evaluating condition '{self.condition}': {e}")
            return False

class NotificationStep(WorkflowStep):
    """Step to send notifications"""

    def __init__(self, step_id: str, message: str, recipient: str = "default",
                 notification_type: str = "info", **kwargs):
        super().__init__(step_id, f"Send notification: {message[:50]}...", **kwargs)
        self.message = message
        self.recipient = recipient
        self.notification_type = notification_type

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Send notification"""
        # Replace variables in message
        formatted_message = self.message
        for var_name, var_value in context.variables.items():
            formatted_message = formatted_message.replace(f"${var_name}", str(var_value))

        logger.info(f"Sending {self.notification_type} notification to {self.recipient}: {formatted_message}")

        return {
            'message': formatted_message,
            'recipient': self.recipient,
            'type': self.notification_type,
            'sent_at': datetime.now().isoformat()
        }

class ScriptStep(WorkflowStep):
    """Step to execute custom scripts"""

    def __init__(self, step_id: str, script_code: str, script_type: str = "python", **kwargs):
        super().__init__(step_id, f"Execute {script_type} script", **kwargs)
        self.script_code = script_code
        self.script_type = script_type

    async def execute(self, context: WorkflowContext) -> Dict[str, Any]:
        """Execute script"""
        logger.info(f"Executing {self.script_type} script")

        # This is a placeholder - in production, you'd want proper sandboxed execution
        # For now, just log the script
        logger.debug(f"Script code: {self.script_code}")

        return {
            'script_type': self.script_type,
            'executed_at': datetime.now().isoformat(),
            'output': "Script executed successfully"
        }

@dataclass
class WorkflowDefinition:
    """Definition of a workflow"""
    workflow_id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    parallel_execution: bool = False
    max_execution_time: int = 3600  # seconds
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow"""
        self.steps.append(step)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

@dataclass
class WorkflowExecution:
    """Runtime instance of a workflow execution"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    context: WorkflowContext
    current_step_index: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    triggered_by: Optional[str] = None

    def get_execution_time(self) -> Optional[float]:
        """Get total execution time in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

class WorkflowEngine:
    """
    Advanced workflow engine for home automation.
    Supports complex workflows with conditions, loops, and error handling.
    """

    def __init__(self, device_manager=None, ha_client=None, esp32_interface=None):
        self.device_manager = device_manager
        self.ha_client = ha_client
        self.esp32_interface = esp32_interface

        # Workflow storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []

        # Execution control
        self.running_executions: Set[str] = set()
        self.paused_executions: Set[str] = set()

        # Event handlers
        self.step_callbacks: List[Callable[[str, WorkflowStep, str], None]] = []
        self.workflow_callbacks: List[Callable[[WorkflowExecution], None]] = []

        # Scheduling
        self.scheduler_task: Optional[asyncio.Task] = None
        self.scheduler_running = False

    async def start(self) -> None:
        """Start the workflow engine"""
        logger.info("Starting workflow engine")

        if not self.scheduler_running:
            self.scheduler_running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the workflow engine"""
        logger.info("Stopping workflow engine")

        self.scheduler_running = False

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel all running executions
        for execution_id in list(self.running_executions):
            await self.cancel_execution(execution_id)

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a new workflow"""
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")

    def unregister_workflow(self, workflow_id: str) -> bool:
        """Unregister a workflow"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Unregistered workflow: {workflow_id}")
            return True
        return False

    async def execute_workflow(self, workflow_id: str, variables: Dict[str, Any] = None,
                             triggered_by: str = "manual") -> str:
        """Execute a workflow and return execution ID"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        if not workflow.enabled:
            raise ValueError(f"Workflow {workflow_id} is disabled")

        # Create execution
        execution_id = str(uuid.uuid4())
        context = WorkflowContext(
            workflow_id=workflow_id,
            variables={**workflow.variables, **(variables or {})},
            start_time=datetime.now()
        )

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            context=context,
            triggered_by=triggered_by
        )

        self.executions[execution_id] = execution
        self.running_executions.add(execution_id)

        logger.info(f"Starting workflow execution {execution_id} for workflow {workflow_id}")

        # Start execution in background
        asyncio.create_task(self._execute_workflow_async(execution_id))

        return execution_id

    async def _execute_workflow_async(self, execution_id: str) -> None:
        """Execute workflow asynchronously"""
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]

        try:
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()

            await self._notify_workflow_callbacks(execution)

            if workflow.parallel_execution:
                await self._execute_parallel(workflow, execution)
            else:
                await self._execute_sequential(workflow, execution)

            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()

            logger.info(f"Workflow execution {execution_id} completed successfully")

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_message = str(e)

            logger.error(f"Workflow execution {execution_id} failed: {e}")

        finally:
            self.running_executions.discard(execution_id)
            self.execution_history.append(execution)
            await self._notify_workflow_callbacks(execution)

    async def _execute_sequential(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute workflow steps sequentially"""
        for i, step in enumerate(workflow.steps):
            if execution.status in [WorkflowStatus.CANCELLED, WorkflowStatus.FAILED]:
                break

            execution.current_step_index = i

            # Check if step can be executed
            if not await step.can_execute(execution.context):
                step.status = StepStatus.SKIPPED
                continue

            await self._execute_step(step, execution)

    async def _execute_parallel(self, workflow: WorkflowDefinition, execution: WorkflowExecution) -> None:
        """Execute workflow steps in parallel"""
        tasks = []

        for step in workflow.steps:
            if await step.can_execute(execution.context):
                task = asyncio.create_task(self._execute_step(step, execution))
                tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution) -> None:
        """Execute a single workflow step"""
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()

        await self._notify_step_callbacks(execution.execution_id, step, "started")

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                step.execute(execution.context),
                timeout=step.timeout_seconds
            )

            # Store step result in context
            execution.context.step_results[step.step_id] = result

            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now()

            logger.debug(f"Step {step.step_id} completed successfully")
            await self._notify_step_callbacks(execution.execution_id, step, "completed")

        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error_message = f"Step timed out after {step.timeout_seconds} seconds"
            step.end_time = datetime.now()

            logger.error(f"Step {step.step_id} timed out")
            await self._notify_step_callbacks(execution.execution_id, step, "failed")

            if step.retry_count > 0:
                await self._retry_step(step, execution)
            else:
                raise Exception(step.error_message)

        except Exception as e:
            step.status = StepStatus.FAILED
            step.error_message = str(e)
            step.end_time = datetime.now()

            logger.error(f"Step {step.step_id} failed: {e}")
            await self._notify_step_callbacks(execution.execution_id, step, "failed")

            if step.retry_count > 0:
                await self._retry_step(step, execution)
            else:
                raise

    async def _retry_step(self, step: WorkflowStep, execution: WorkflowExecution) -> None:
        """Retry a failed step"""
        for retry in range(step.retry_count):
            logger.info(f"Retrying step {step.step_id} (attempt {retry + 1}/{step.retry_count})")

            # Reset step status
            step.status = StepStatus.RUNNING
            step.error_message = None

            try:
                result = await asyncio.wait_for(
                    step.execute(execution.context),
                    timeout=step.timeout_seconds
                )

                execution.context.step_results[step.step_id] = result
                step.status = StepStatus.COMPLETED
                step.end_time = datetime.now()

                logger.info(f"Step {step.step_id} succeeded on retry {retry + 1}")
                return

            except Exception as e:
                step.error_message = str(e)
                if retry == step.retry_count - 1:  # Last retry
                    step.status = StepStatus.FAILED
                    step.end_time = datetime.now()
                    raise

                # Wait before retry
                await asyncio.sleep(2 ** retry)  # Exponential backoff

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running workflow execution"""
        if execution_id in self.running_executions:
            execution = self.executions[execution_id]
            execution.status = WorkflowStatus.PAUSED
            self.paused_executions.add(execution_id)

            logger.info(f"Paused workflow execution {execution_id}")
            return True

        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution"""
        if execution_id in self.paused_executions:
            execution = self.executions[execution_id]
            execution.status = WorkflowStatus.RUNNING
            self.paused_executions.remove(execution_id)

            logger.info(f"Resumed workflow execution {execution_id}")
            return True

        return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution"""
        if execution_id in self.running_executions:
            execution = self.executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.now()

            self.running_executions.discard(execution_id)
            self.paused_executions.discard(execution_id)

            logger.info(f"Cancelled workflow execution {execution_id}")
            return True

        return False

    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get status of a workflow execution"""
        return self.executions.get(execution_id)

    def get_running_executions(self) -> List[WorkflowExecution]:
        """Get all currently running executions"""
        return [
            self.executions[execution_id]
            for execution_id in self.running_executions
            if execution_id in self.executions
        ]

    def get_execution_history(self, workflow_id: Optional[str] = None,
                            limit: int = 100) -> List[WorkflowExecution]:
        """Get execution history"""
        history = self.execution_history

        if workflow_id:
            history = [ex for ex in history if ex.workflow_id == workflow_id]

        return sorted(history, key=lambda x: x.start_time or datetime.min, reverse=True)[:limit]

    def add_step_callback(self, callback: Callable[[str, WorkflowStep, str], None]) -> None:
        """Add callback for step events"""
        self.step_callbacks.append(callback)

    def add_workflow_callback(self, callback: Callable[[WorkflowExecution], None]) -> None:
        """Add callback for workflow events"""
        self.workflow_callbacks.append(callback)

    async def _notify_step_callbacks(self, execution_id: str, step: WorkflowStep, event: str) -> None:
        """Notify step event callbacks"""
        for callback in self.step_callbacks:
            try:
                callback(execution_id, step, event)
            except Exception as e:
                logger.error(f"Error in step callback: {e}")

    async def _notify_workflow_callbacks(self, execution: WorkflowExecution) -> None:
        """Notify workflow event callbacks"""
        for callback in self.workflow_callbacks:
            try:
                callback(execution)
            except Exception as e:
                logger.error(f"Error in workflow callback: {e}")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop for time-based triggers"""
        while self.scheduler_running:
            try:
                await self._check_scheduled_workflows()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(10)

    async def _check_scheduled_workflows(self) -> None:
        """Check for workflows that should be triggered by schedule"""
        now = datetime.now()

        for workflow in self.workflows.values():
            if not workflow.enabled:
                continue

            for trigger in workflow.triggers:
                if trigger.get('type') == 'scheduled':
                    if await self._should_trigger_scheduled(trigger, now):
                        logger.info(f"Triggering scheduled workflow: {workflow.workflow_id}")
                        await self.execute_workflow(
                            workflow.workflow_id,
                            triggered_by=f"schedule:{trigger.get('schedule')}"
                        )

    async def _should_trigger_scheduled(self, trigger: Dict[str, Any], now: datetime) -> bool:
        """Check if a scheduled trigger should fire"""
        # This is a simplified scheduler - in production you'd want a proper cron-like system
        schedule = trigger.get('schedule', '')

        # Example: "daily:08:00" or "hourly:30" or "interval:300"
        if schedule.startswith('daily:'):
            time_str = schedule.split(':', 1)[1]
            try:
                hour, minute = map(int, time_str.split(':'))
                return now.hour == hour and now.minute == minute
            except ValueError:
                return False

        elif schedule.startswith('hourly:'):
            try:
                minute = int(schedule.split(':', 1)[1])
                return now.minute == minute
            except ValueError:
                return False

        elif schedule.startswith('interval:'):
            try:
                interval_seconds = int(schedule.split(':', 1)[1])
                # This would need state tracking for proper interval scheduling
                return now.second == 0 and now.minute % (interval_seconds // 60) == 0
            except ValueError:
                return False

        return False

# Helper functions for creating common workflow patterns

def create_device_sequence_workflow(workflow_id: str, name: str, device_actions: List[Dict[str, Any]],
                                  delays: List[float] = None) -> WorkflowDefinition:
    """Create a workflow that controls devices in sequence"""
    workflow = WorkflowDefinition(workflow_id=workflow_id, name=name)

    delays = delays or [0] * len(device_actions)

    for i, (action, delay) in enumerate(zip(device_actions, delays)):
        if delay > 0:
            workflow.add_step(DelayStep(f"delay_{i}", delay))

        workflow.add_step(DeviceControlStep(
            step_id=f"control_{i}",
            entity_id=action['entity_id'],
            action=action['action'],
            parameters=action.get('parameters', {})
        ))

    return workflow

def create_conditional_workflow(workflow_id: str, name: str, condition: str,
                              true_steps: List[WorkflowStep], false_steps: List[WorkflowStep] = None) -> WorkflowDefinition:
    """Create a workflow with conditional execution"""
    workflow = WorkflowDefinition(workflow_id=workflow_id, name=name)

    # Add condition check
    workflow.add_step(ConditionStep("condition_check", condition))

    # Add conditional execution logic (simplified)
    # In a full implementation, you'd need proper conditional branching
    for step in true_steps:
        workflow.add_step(step)

    return workflow

def create_notification_workflow(workflow_id: str, name: str, message: str,
                               recipients: List[str] = None) -> WorkflowDefinition:
    """Create a workflow that sends notifications"""
    workflow = WorkflowDefinition(workflow_id=workflow_id, name=name)

    recipients = recipients or ["default"]

    for i, recipient in enumerate(recipients):
        workflow.add_step(NotificationStep(
            step_id=f"notify_{i}",
            message=message,
            recipient=recipient
        ))

    return workflow