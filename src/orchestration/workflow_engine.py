"""
Complex Workflow Engine
Advanced orchestration system for multi-step, conditional agent coordination
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Set
from pydantic import BaseModel, Field
from uuid import uuid4

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of workflow steps"""
    AGENT_TASK = "agent_task"
    PARALLEL_TASKS = "parallel_tasks"
    SEQUENTIAL_TASKS = "sequential_tasks"
    CONDITIONAL_BRANCH = "conditional_branch"
    LOOP = "loop"
    WAIT = "wait"
    HUMAN_INPUT = "human_input"
    API_CALL = "api_call"
    DATA_TRANSFORM = "data_transform"
    DECISION_POINT = "decision_point"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"


class ConditionType(str, Enum):
    """Types of workflow conditions"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    REGEX_MATCH = "regex_match"
    CUSTOM_FUNCTION = "custom_function"
    AGENT_CONFIDENCE = "agent_confidence"
    OUTPUT_QUALITY = "output_quality"
    TIME_ELAPSED = "time_elapsed"
    RESOURCE_AVAILABLE = "resource_available"


class WorkflowState(str, Enum):
    """Workflow execution states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_INPUT = "waiting_for_input"


class WorkflowCondition(BaseModel):
    """Condition for workflow branching and control flow"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    condition_type: ConditionType
    field_path: str = Field(description="JSONPath to the field to evaluate")
    expected_value: Any = Field(description="Expected value for comparison")
    operator: str = Field(default="and", description="Logical operator with other conditions")
    custom_function: Optional[str] = Field(None, description="Custom function name for evaluation")
    description: str = Field(default="", description="Human-readable condition description")


class WorkflowStep(BaseModel):
    """Individual step in a workflow"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    step_type: StepType
    description: str = ""

    # Agent assignment
    assigned_agent_id: Optional[str] = None
    required_capabilities: List[str] = Field(default_factory=list)
    preferred_agent_types: List[str] = Field(default_factory=list)

    # Step configuration
    config: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(None, description="Step timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Control flow
    conditions: List[WorkflowCondition] = Field(default_factory=list)
    on_success_step: Optional[str] = Field(None, description="Next step on success")
    on_failure_step: Optional[str] = Field(None, description="Next step on failure")
    parallel_steps: List[str] = Field(default_factory=list, description="Steps to run in parallel")

    # Input/Output
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    output_mapping: Dict[str, str] = Field(default_factory=dict)
    required_inputs: List[str] = Field(default_factory=list)

    # Metadata
    tags: List[str] = Field(default_factory=list)
    priority: int = Field(default=5, description="Step priority (1-10)")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    version: str = "1.0.0"

    # Steps and flow
    steps: List[WorkflowStep]
    start_step_id: str
    end_step_ids: List[str] = Field(default_factory=list)

    # Configuration
    global_timeout_seconds: Optional[int] = Field(None)
    max_concurrent_steps: int = Field(default=10)
    retry_policy: Dict[str, Any] = Field(default_factory=dict)

    # Variables and context
    variables: Dict[str, Any] = Field(default_factory=dict)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    author: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        return next((step for step in self.steps if step.id == step_id), None)

    def validate_workflow(self) -> List[str]:
        """Validate workflow definition and return any errors"""
        errors = []

        # Check start step exists
        if not self.get_step(self.start_step_id):
            errors.append(f"Start step '{self.start_step_id}' not found")

        # Check end steps exist
        for end_step_id in self.end_step_ids:
            if not self.get_step(end_step_id):
                errors.append(f"End step '{end_step_id}' not found")

        # Check step references
        for step in self.steps:
            if step.on_success_step and not self.get_step(step.on_success_step):
                errors.append(f"Step '{step.id}' references non-existent success step '{step.on_success_step}'")

            if step.on_failure_step and not self.get_step(step.on_failure_step):
                errors.append(f"Step '{step.id}' references non-existent failure step '{step.on_failure_step}'")

            for parallel_step_id in step.parallel_steps:
                if not self.get_step(parallel_step_id):
                    errors.append(f"Step '{step.id}' references non-existent parallel step '{parallel_step_id}'")

        return errors


class WorkflowExecution(BaseModel):
    """Runtime execution state of a workflow"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    workflow_definition: WorkflowDefinition

    # Execution state
    state: WorkflowState = WorkflowState.PENDING
    current_step_id: Optional[str] = None
    completed_steps: Set[str] = Field(default_factory=set)
    failed_steps: Set[str] = Field(default_factory=set)

    # Context and data
    context: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    step_outputs: Dict[str, Any] = Field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None

    # Agent assignments
    step_agent_assignments: Dict[str, str] = Field(default_factory=dict)

    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

    # Performance metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class WorkflowExecutor:
    """Executes workflow steps and manages execution flow"""

    def __init__(self, agent_coordinator=None, external_api_manager=None):
        self.agent_coordinator = agent_coordinator
        self.external_api_manager = external_api_manager
        self.logger = logging.getLogger(__name__)

        # Step type handlers
        self.step_handlers = {
            StepType.AGENT_TASK: self._execute_agent_task,
            StepType.PARALLEL_TASKS: self._execute_parallel_tasks,
            StepType.SEQUENTIAL_TASKS: self._execute_sequential_tasks,
            StepType.CONDITIONAL_BRANCH: self._execute_conditional_branch,
            StepType.LOOP: self._execute_loop,
            StepType.WAIT: self._execute_wait,
            StepType.API_CALL: self._execute_api_call,
            StepType.DATA_TRANSFORM: self._execute_data_transform,
            StepType.DECISION_POINT: self._execute_decision_point,
            StepType.AGGREGATION: self._execute_aggregation,
            StepType.VALIDATION: self._execute_validation
        }

    async def execute_step(
        self,
        execution: WorkflowExecution,
        step: WorkflowStep
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            self.logger.info(f"Executing step: {step.name} ({step.step_type})")

            # Check conditions before execution
            if step.conditions and not await self._evaluate_conditions(step.conditions, execution):
                return {"skipped": True, "reason": "Conditions not met"}

            # Get step handler
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise Exception(f"No handler for step type: {step.step_type}")

            # Prepare step inputs
            step_inputs = await self._prepare_step_inputs(step, execution)

            # Execute step with timeout
            if step.timeout_seconds:
                result = await asyncio.wait_for(
                    handler(step, step_inputs, execution),
                    timeout=step.timeout_seconds
                )
            else:
                result = await handler(step, step_inputs, execution)

            # Process step outputs
            await self._process_step_outputs(step, result, execution)

            return result

        except Exception as e:
            self.logger.error(f"Step execution failed: {step.name} - {e}")
            execution.errors.append({
                "step_id": step.id,
                "step_name": step.name,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            raise

    async def _execute_agent_task(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a task using an AI agent"""
        try:
            if not self.agent_coordinator:
                raise Exception("Agent coordinator not available")

            # Select appropriate agent
            agent_id = await self._select_agent_for_step(step, execution)

            # Prepare agent task
            task_config = {
                "prompt": inputs.get("prompt", step.config.get("prompt", "")),
                "context": inputs.get("context", execution.context),
                "required_capabilities": step.required_capabilities,
                "step_id": step.id,
                "workflow_id": execution.workflow_id
            }

            # Execute agent task
            result = await self.agent_coordinator.execute_agent_task(
                agent_id=agent_id,
                task_config=task_config
            )

            # Store agent assignment
            execution.step_agent_assignments[step.id] = agent_id

            return {
                "success": True,
                "agent_id": agent_id,
                "response": result.get("response", ""),
                "confidence": result.get("confidence", 0.0),
                "execution_time_ms": result.get("execution_time_ms", 0)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_parallel_tasks(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute multiple steps in parallel"""
        try:
            parallel_steps = [
                execution.workflow_definition.get_step(step_id)
                for step_id in step.parallel_steps
            ]
            parallel_steps = [s for s in parallel_steps if s is not None]

            if not parallel_steps:
                return {"success": True, "results": []}

            # Execute all parallel steps concurrently
            tasks = [
                self.execute_step(execution, parallel_step)
                for parallel_step in parallel_steps
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            success_count = 0
            parallel_results = {}

            for i, result in enumerate(results):
                step_id = parallel_steps[i].id
                if isinstance(result, Exception):
                    parallel_results[step_id] = {"success": False, "error": str(result)}
                else:
                    parallel_results[step_id] = result
                    if result.get("success", False):
                        success_count += 1

            return {
                "success": success_count > 0,
                "parallel_results": parallel_results,
                "success_count": success_count,
                "total_count": len(parallel_steps)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_sequential_tasks(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute multiple steps sequentially"""
        try:
            sequential_steps = step.config.get("sequential_steps", [])
            results = []

            for step_id in sequential_steps:
                sequential_step = execution.workflow_definition.get_step(step_id)
                if not sequential_step:
                    continue

                result = await self.execute_step(execution, sequential_step)
                results.append({"step_id": step_id, "result": result})

                # Stop on failure if configured
                if not result.get("success", False) and step.config.get("stop_on_failure", True):
                    break

            return {
                "success": all(r["result"].get("success", False) for r in results),
                "sequential_results": results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_conditional_branch(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute conditional branching logic"""
        try:
            conditions = step.conditions
            branch_config = step.config.get("branches", {})

            # Evaluate conditions
            conditions_met = await self._evaluate_conditions(conditions, execution)

            # Determine which branch to take
            if conditions_met:
                branch_step_id = step.config.get("true_branch")
            else:
                branch_step_id = step.config.get("false_branch")

            if not branch_step_id:
                return {"success": True, "branch_taken": "none"}

            # Execute branch step
            branch_step = execution.workflow_definition.get_step(branch_step_id)
            if branch_step:
                result = await self.execute_step(execution, branch_step)
                return {
                    "success": result.get("success", False),
                    "branch_taken": "true" if conditions_met else "false",
                    "branch_result": result
                }

            return {"success": False, "error": f"Branch step not found: {branch_step_id}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_loop(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute loop logic"""
        try:
            loop_config = step.config
            max_iterations = loop_config.get("max_iterations", 10)
            loop_step_id = loop_config.get("loop_step")
            continue_condition = loop_config.get("continue_condition", {})

            if not loop_step_id:
                return {"success": False, "error": "No loop step specified"}

            loop_step = execution.workflow_definition.get_step(loop_step_id)
            if not loop_step:
                return {"success": False, "error": f"Loop step not found: {loop_step_id}"}

            iterations = 0
            results = []

            while iterations < max_iterations:
                # Execute loop step
                result = await self.execute_step(execution, loop_step)
                results.append(result)
                iterations += 1

                # Check continue condition
                if continue_condition:
                    continue_loop = await self._evaluate_single_condition(
                        continue_condition, execution
                    )
                    if not continue_loop:
                        break

                # Stop on failure if configured
                if not result.get("success", False) and loop_config.get("stop_on_failure", True):
                    break

            return {
                "success": True,
                "iterations": iterations,
                "loop_results": results
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_wait(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute wait step"""
        try:
            wait_seconds = step.config.get("wait_seconds", 1)
            await asyncio.sleep(wait_seconds)

            return {
                "success": True,
                "waited_seconds": wait_seconds
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_api_call(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute external API call"""
        try:
            if not self.external_api_manager:
                raise Exception("External API manager not available")

            api_config = step.config
            endpoint_id = api_config.get("endpoint_id")

            if not endpoint_id:
                return {"success": False, "error": "No endpoint specified"}

            # Make API call
            result = await self.external_api_manager.call_endpoint(
                endpoint_id=endpoint_id,
                data=inputs,
                headers=api_config.get("headers", {}),
                params=api_config.get("params", {})
            )

            return {
                "success": True,
                "api_result": result,
                "status_code": result.get("status_code"),
                "response_data": result.get("data")
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_data_transform(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute data transformation"""
        try:
            transform_config = step.config
            transform_type = transform_config.get("type", "mapping")

            if transform_type == "mapping":
                # Simple field mapping
                mapping = transform_config.get("mapping", {})
                transformed_data = {}

                for output_field, input_path in mapping.items():
                    value = self._get_nested_value(inputs, input_path)
                    transformed_data[output_field] = value

                return {
                    "success": True,
                    "transformed_data": transformed_data
                }

            elif transform_type == "aggregation":
                # Data aggregation
                source_field = transform_config.get("source_field")
                operation = transform_config.get("operation", "sum")

                if source_field not in inputs:
                    return {"success": False, "error": f"Source field not found: {source_field}"}

                data = inputs[source_field]
                if not isinstance(data, list):
                    return {"success": False, "error": "Source data must be a list for aggregation"}

                if operation == "sum":
                    result = sum(data)
                elif operation == "avg":
                    result = sum(data) / len(data) if data else 0
                elif operation == "count":
                    result = len(data)
                elif operation == "max":
                    result = max(data) if data else None
                elif operation == "min":
                    result = min(data) if data else None
                else:
                    return {"success": False, "error": f"Unknown operation: {operation}"}

                return {
                    "success": True,
                    "aggregated_value": result
                }

            return {"success": False, "error": f"Unknown transform type: {transform_type}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_decision_point(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute decision point logic"""
        try:
            decision_config = step.config
            decision_rules = decision_config.get("rules", [])

            for rule in decision_rules:
                rule_conditions = rule.get("conditions", [])
                if await self._evaluate_conditions(rule_conditions, execution):
                    return {
                        "success": True,
                        "decision": rule.get("decision"),
                        "rule_matched": rule.get("name", "unnamed"),
                        "next_step": rule.get("next_step")
                    }

            # Default decision if no rules match
            default_decision = decision_config.get("default_decision")
            return {
                "success": True,
                "decision": default_decision,
                "rule_matched": "default",
                "next_step": decision_config.get("default_next_step")
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_aggregation(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute result aggregation"""
        try:
            aggregation_config = step.config
            source_steps = aggregation_config.get("source_steps", [])
            aggregation_method = aggregation_config.get("method", "collect")

            aggregated_results = {}

            for step_id in source_steps:
                if step_id in execution.step_outputs:
                    aggregated_results[step_id] = execution.step_outputs[step_id]

            if aggregation_method == "collect":
                return {
                    "success": True,
                    "aggregated_results": aggregated_results
                }
            elif aggregation_method == "consensus":
                # Simple consensus: majority vote
                decisions = [
                    result.get("decision") for result in aggregated_results.values()
                    if "decision" in result
                ]

                if decisions:
                    consensus = max(set(decisions), key=decisions.count)
                    return {
                        "success": True,
                        "consensus_decision": consensus,
                        "vote_count": decisions.count(consensus),
                        "total_votes": len(decisions)
                    }

            return {"success": True, "aggregated_results": aggregated_results}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_validation(
        self,
        step: WorkflowStep,
        inputs: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute validation logic"""
        try:
            validation_config = step.config
            validation_rules = validation_config.get("rules", [])
            validation_results = []

            for rule in validation_rules:
                rule_name = rule.get("name", "unnamed")
                field_path = rule.get("field_path")
                validation_type = rule.get("type", "required")

                value = self._get_nested_value(inputs, field_path)

                if validation_type == "required":
                    is_valid = value is not None and value != ""
                elif validation_type == "type":
                    expected_type = rule.get("expected_type", "string")
                    is_valid = type(value).__name__ == expected_type
                elif validation_type == "range":
                    min_val = rule.get("min_value")
                    max_val = rule.get("max_value")
                    is_valid = (min_val is None or value >= min_val) and \
                              (max_val is None or value <= max_val)
                else:
                    is_valid = True

                validation_results.append({
                    "rule": rule_name,
                    "field": field_path,
                    "valid": is_valid,
                    "value": value
                })

            all_valid = all(result["valid"] for result in validation_results)

            return {
                "success": all_valid,
                "validation_results": validation_results,
                "all_valid": all_valid
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # Helper methods

    async def _evaluate_conditions(
        self,
        conditions: List[WorkflowCondition],
        execution: WorkflowExecution
    ) -> bool:
        """Evaluate a list of conditions"""
        if not conditions:
            return True

        results = []
        for condition in conditions:
            result = await self._evaluate_single_condition(condition, execution)
            results.append(result)

        # Simple AND logic for now (could be enhanced with complex operators)
        return all(results)

    async def _evaluate_single_condition(
        self,
        condition: Union[WorkflowCondition, Dict[str, Any]],
        execution: WorkflowExecution
    ) -> bool:
        """Evaluate a single condition"""
        try:
            if isinstance(condition, dict):
                condition = WorkflowCondition(**condition)

            # Get value to evaluate
            value = self._get_nested_value(execution.context, condition.field_path)
            expected = condition.expected_value

            # Evaluate based on condition type
            if condition.condition_type == ConditionType.EQUALS:
                return value == expected
            elif condition.condition_type == ConditionType.NOT_EQUALS:
                return value != expected
            elif condition.condition_type == ConditionType.GREATER_THAN:
                return value > expected
            elif condition.condition_type == ConditionType.LESS_THAN:
                return value < expected
            elif condition.condition_type == ConditionType.CONTAINS:
                return expected in str(value)
            elif condition.condition_type == ConditionType.REGEX_MATCH:
                import re
                return bool(re.search(expected, str(value)))
            elif condition.condition_type == ConditionType.CUSTOM_FUNCTION:
                # Custom function evaluation would be implemented here
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Condition evaluation failed: {e}")
            return False

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation path"""
        try:
            value = data
            for key in path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and key.isdigit():
                    value = value[int(key)]
                else:
                    return None
            return value
        except (KeyError, IndexError, ValueError):
            return None

    async def _prepare_step_inputs(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Prepare inputs for step execution"""
        inputs = {}

        # Apply input mapping
        for input_key, source_path in step.input_mapping.items():
            value = self._get_nested_value(execution.context, source_path)
            inputs[input_key] = value

        # Add step configuration
        inputs.update(step.config)

        # Add execution context
        inputs["_execution_context"] = execution.context
        inputs["_workflow_inputs"] = execution.inputs

        return inputs

    async def _process_step_outputs(
        self,
        step: WorkflowStep,
        result: Dict[str, Any],
        execution: WorkflowExecution
    ):
        """Process step outputs and update execution context"""
        # Store step output
        execution.step_outputs[step.id] = result

        # Apply output mapping
        for output_key, target_path in step.output_mapping.items():
            if output_key in result:
                self._set_nested_value(execution.context, target_path, result[output_key])

        # Mark step as completed
        if result.get("success", False):
            execution.completed_steps.add(step.id)
        else:
            execution.failed_steps.add(step.id)

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set nested value using dot notation path"""
        try:
            keys = path.split('.')
            current = data

            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = value
        except Exception as e:
            self.logger.warning(f"Failed to set nested value {path}: {e}")

    async def _select_agent_for_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> str:
        """Select the best agent for a workflow step"""
        if step.assigned_agent_id:
            return step.assigned_agent_id

        if self.agent_coordinator:
            # Use agent coordinator to select best agent
            selection_criteria = {
                "required_capabilities": step.required_capabilities,
                "preferred_types": step.preferred_agent_types,
                "workflow_context": execution.context
            }

            return await self.agent_coordinator.select_agent(selection_criteria)

        # Fallback to default agent
        return "default_agent"


class WorkflowEngine:
    """Main workflow engine that manages workflow execution"""

    def __init__(self, agent_coordinator=None, external_api_manager=None):
        self.agent_coordinator = agent_coordinator
        self.external_api_manager = external_api_manager
        self.executor = WorkflowExecutor(agent_coordinator, external_api_manager)
        self.logger = logging.getLogger(__name__)

        # Active executions
        self.active_executions: Dict[str, WorkflowExecution] = {}

        # Execution history
        self.execution_history: List[WorkflowExecution] = []

    async def execute_workflow(
        self,
        workflow_definition: WorkflowDefinition,
        inputs: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a complete workflow"""
        try:
            # Validate workflow
            validation_errors = workflow_definition.validate_workflow()
            if validation_errors:
                raise Exception(f"Workflow validation failed: {validation_errors}")

            # Create execution
            execution = WorkflowExecution(
                workflow_id=workflow_definition.id,
                workflow_definition=workflow_definition,
                inputs=inputs or {},
                context=context or {}
            )

            # Start execution
            execution.state = WorkflowState.RUNNING
            execution.started_at = datetime.utcnow()
            execution.current_step_id = workflow_definition.start_step_id

            # Register execution
            self.active_executions[execution.id] = execution

            try:
                # Execute workflow steps
                await self._execute_workflow_flow(execution)

                # Complete execution
                execution.state = WorkflowState.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.execution_time_ms = (
                    execution.completed_at - execution.started_at
                ).total_seconds() * 1000

            except Exception as e:
                execution.state = WorkflowState.FAILED
                execution.errors.append({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                raise

            finally:
                # Move to history
                self.execution_history.append(execution)
                if execution.id in self.active_executions:
                    del self.active_executions[execution.id]

            return execution

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise

    async def _execute_workflow_flow(self, execution: WorkflowExecution):
        """Execute the workflow flow from start to end"""
        max_steps = 1000  # Prevent infinite loops
        step_count = 0

        while execution.current_step_id and step_count < max_steps:
            step = execution.workflow_definition.get_step(execution.current_step_id)
            if not step:
                raise Exception(f"Step not found: {execution.current_step_id}")

            # Execute step
            result = await self.executor.execute_step(execution, step)

            # Determine next step
            if result.get("success", False):
                next_step_id = step.on_success_step
            else:
                next_step_id = step.on_failure_step

            # Check if we've reached an end step
            if (execution.current_step_id in execution.workflow_definition.end_step_ids or
                not next_step_id):
                break

            execution.current_step_id = next_step_id
            step_count += 1

        if step_count >= max_steps:
            raise Exception("Workflow exceeded maximum step limit (possible infinite loop)")

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            # Check history
            execution = next(
                (e for e in self.execution_history if e.id == execution_id),
                None
            )

        if not execution:
            return None

        return {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "state": execution.state,
            "current_step_id": execution.current_step_id,
            "completed_steps": list(execution.completed_steps),
            "failed_steps": list(execution.failed_steps),
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "execution_time_ms": execution.execution_time_ms,
            "errors": execution.errors,
            "warnings": execution.warnings
        }

    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running workflow execution"""
        execution = self.active_executions.get(execution_id)
        if execution and execution.state == WorkflowState.RUNNING:
            execution.state = WorkflowState.PAUSED
            return True
        return False

    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution"""
        execution = self.active_executions.get(execution_id)
        if execution and execution.state == WorkflowState.PAUSED:
            execution.state = WorkflowState.RUNNING
            # Continue execution from current step
            await self._execute_workflow_flow(execution)
            return True
        return False

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        execution = self.active_executions.get(execution_id)
        if execution:
            execution.state = WorkflowState.CANCELLED
            execution.completed_at = datetime.utcnow()
            return True
        return False

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get list of active workflow executions"""
        return [
            {
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "state": execution.state,
                "started_at": execution.started_at,
                "current_step_id": execution.current_step_id
            }
            for execution in self.active_executions.values()
        ]

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        total_executions = len(self.execution_history) + len(self.active_executions)

        if not self.execution_history:
            return {"total_executions": total_executions}

        completed = len([e for e in self.execution_history if e.state == WorkflowState.COMPLETED])
        failed = len([e for e in self.execution_history if e.state == WorkflowState.FAILED])

        avg_execution_time = sum(
            e.execution_time_ms for e in self.execution_history
            if e.execution_time_ms
        ) / len([e for e in self.execution_history if e.execution_time_ms])

        return {
            "total_executions": total_executions,
            "active_executions": len(self.active_executions),
            "completed_executions": completed,
            "failed_executions": failed,
            "success_rate": completed / max(1, completed + failed),
            "avg_execution_time_ms": avg_execution_time
        }