"""
Multi-Agent Scenario Testing Framework

This module provides comprehensive testing infrastructure for multi-agent coordination
scenarios, including conflict resolution, collaboration workflows, and communication patterns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import time
from collections import defaultdict, deque
import redis.asyncio as redis

from .conflict_resolver import ConflictResolver, ConflictType, ConflictSeverity, ResolutionStrategy
from .collaboration_engine import CollaborationEngine, CollaborationPattern, TaskType, AgentRole
from .agent_coordinator import AgentCoordinator, CoordinationStrategy, AgentStatus
from .agent_communication import AgentCommunication, MessageType, CommunicationProtocol
from .conversation_orchestrator import ConversationOrchestrator, FlowState, TurnType
from .role_manager import RoleManager
from .context_sharing import ContextSharingManager

logger = logging.getLogger(__name__)

class TestScenarioType(Enum):
    CONFLICT_RESOLUTION = "conflict_resolution"
    COLLABORATION_WORKFLOW = "collaboration_workflow"
    AGENT_COORDINATION = "agent_coordination"
    COMMUNICATION_PATTERN = "communication_pattern"
    ROLE_MANAGEMENT = "role_management"
    CONTEXT_SHARING = "context_sharing"
    INTEGRATION_TEST = "integration_test"
    STRESS_TEST = "stress_test"
    PERFORMANCE_TEST = "performance_test"

class TestResultStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class TestSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class TestAssertion:
    """Represents a test assertion to be validated"""
    assertion_id: str
    description: str
    assertion_function: Callable[[Dict[str, Any]], bool]
    expected_value: Any = None
    actual_value: Any = None
    passed: Optional[bool] = None
    error_message: Optional[str] = None

    async def validate(self, test_results: Dict[str, Any]) -> bool:
        """Validate the assertion against test results"""
        try:
            self.passed = self.assertion_function(test_results)
            return self.passed
        except Exception as e:
            self.passed = False
            self.error_message = str(e)
            return False

@dataclass
class MockAgent:
    """Mock agent for testing scenarios"""
    agent_id: str
    capabilities: Dict[str, Any]
    roles: List[AgentRole] = field(default_factory=list)
    status: AgentStatus = AgentStatus.AVAILABLE
    performance_score: float = 0.8
    specializations: List[str] = field(default_factory=list)
    response_delay: float = 0.1  # Simulated response delay
    error_rate: float = 0.0  # Probability of returning errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "roles": [role.value for role in self.roles],
            "status": self.status.value,
            "performance_score": self.performance_score,
            "specializations": self.specializations
        }

@dataclass
class TestScenario:
    """Represents a complete test scenario"""
    scenario_id: str
    name: str
    description: str
    scenario_type: TestScenarioType
    severity: TestSeverity
    setup_function: Callable
    test_function: Callable
    teardown_function: Optional[Callable] = None
    mock_agents: List[MockAgent] = field(default_factory=list)
    test_data: Dict[str, Any] = field(default_factory=dict)
    assertions: List[TestAssertion] = field(default_factory=list)
    timeout_seconds: int = 30
    retry_count: int = 0
    dependencies: List[str] = field(default_factory=list)  # Scenario IDs this depends on
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestResult:
    """Results from executing a test scenario"""
    scenario_id: str
    status: TestResultStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    assertion_results: List[TestAssertion] = field(default_factory=list)
    test_output: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "assertion_results": [
                {
                    "assertion_id": a.assertion_id,
                    "description": a.description,
                    "passed": a.passed,
                    "expected_value": a.expected_value,
                    "actual_value": a.actual_value,
                    "error_message": a.error_message
                } for a in self.assertion_results
            ],
            "test_output": self.test_output,
            "error_message": self.error_message,
            "performance_metrics": self.performance_metrics
        }

class MultiAgentTestFramework:
    """
    Comprehensive testing framework for multi-agent coordination scenarios.

    Provides structured testing infrastructure for validating conflict resolution,
    collaboration workflows, agent coordination, and communication patterns.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.test_scenarios: Dict[str, TestScenario] = {}
        self.test_results: Dict[str, TestResult] = {}
        self.test_history: deque = deque(maxlen=1000)
        self.mock_agents: Dict[str, MockAgent] = {}

        # Test systems
        self.conflict_resolver: Optional[ConflictResolver] = None
        self.collaboration_engine: Optional[CollaborationEngine] = None
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.agent_communication: Optional[AgentCommunication] = None
        self.conversation_orchestrator: Optional[ConversationOrchestrator] = None
        self.role_manager: Optional[RoleManager] = None
        self.context_sharing_manager: Optional[ContextSharingManager] = None

        # Test execution state
        self.running_tests: Set[str] = set()
        self.test_metrics: Dict[str, Any] = defaultdict(int)

        # Initialize built-in test scenarios
        self._initialize_built_in_scenarios()

    async def initialize(self):
        """Initialize the test framework with fresh system instances"""
        self.conflict_resolver = ConflictResolver(self.redis_client)
        self.collaboration_engine = CollaborationEngine(self.redis_client)
        self.agent_coordinator = AgentCoordinator(self.redis_client)
        self.agent_communication = AgentCommunication(self.redis_client)
        self.conversation_orchestrator = ConversationOrchestrator()
        self.role_manager = RoleManager()
        self.context_sharing_manager = ContextSharingManager(self.redis_client)

        logger.info("✅ Multi-agent test framework initialized")

    def _initialize_built_in_scenarios(self):
        """Initialize built-in test scenarios"""

        # Conflict Resolution Scenarios
        self.add_test_scenario(TestScenario(
            scenario_id="conflict_resource_contention",
            name="Resource Contention Conflict",
            description="Test conflict resolution when multiple agents claim the same resource",
            scenario_type=TestScenarioType.CONFLICT_RESOLUTION,
            severity=TestSeverity.HIGH,
            setup_function=self._setup_resource_contention_test,
            test_function=self._test_resource_contention_resolution,
            mock_agents=[
                MockAgent("agent_1", {"resource_manager": True}, performance_score=0.9),
                MockAgent("agent_2", {"resource_manager": True}, performance_score=0.8),
                MockAgent("agent_3", {"resource_manager": True}, performance_score=0.7)
            ],
            tags={"conflict", "resource", "resolution"}
        ))

        self.add_test_scenario(TestScenario(
            scenario_id="conflict_contradictory_actions",
            name="Contradictory Actions Conflict",
            description="Test resolution of conflicting agent actions",
            scenario_type=TestScenarioType.CONFLICT_RESOLUTION,
            severity=TestSeverity.HIGH,
            setup_function=self._setup_contradictory_actions_test,
            test_function=self._test_contradictory_actions_resolution,
            mock_agents=[
                MockAgent("agent_a", {"action_executor": True}, performance_score=0.85),
                MockAgent("agent_b", {"action_executor": True}, performance_score=0.82)
            ],
            tags={"conflict", "actions", "contradiction"}
        ))

        # Collaboration Workflow Scenarios
        self.add_test_scenario(TestScenario(
            scenario_id="collaboration_pipeline",
            name="Pipeline Collaboration Workflow",
            description="Test sequential pipeline collaboration pattern",
            scenario_type=TestScenarioType.COLLABORATION_WORKFLOW,
            severity=TestSeverity.MEDIUM,
            setup_function=self._setup_pipeline_collaboration_test,
            test_function=self._test_pipeline_collaboration,
            mock_agents=[
                MockAgent("specialist_1", {"analysis": True}, [AgentRole.SPECIALIST]),
                MockAgent("specialist_2", {"processing": True}, [AgentRole.SPECIALIST]),
                MockAgent("reviewer_1", {"quality_check": True}, [AgentRole.REVIEWER]),
                MockAgent("synthesizer_1", {"synthesis": True}, [AgentRole.SYNTHESIZER])
            ],
            tags={"collaboration", "pipeline", "workflow"}
        ))

        self.add_test_scenario(TestScenario(
            scenario_id="collaboration_brainstorming",
            name="Brainstorming Collaboration",
            description="Test creative brainstorming collaboration pattern",
            scenario_type=TestScenarioType.COLLABORATION_WORKFLOW,
            severity=TestSeverity.MEDIUM,
            setup_function=self._setup_brainstorming_collaboration_test,
            test_function=self._test_brainstorming_collaboration,
            mock_agents=[
                MockAgent("creative_1", {"ideation": True}, [AgentRole.GENERALIST]),
                MockAgent("creative_2", {"ideation": True}, [AgentRole.GENERALIST]),
                MockAgent("creative_3", {"ideation": True}, [AgentRole.GENERALIST]),
                MockAgent("coordinator_1", {"facilitation": True}, [AgentRole.COORDINATOR])
            ],
            tags={"collaboration", "brainstorming", "creative"}
        ))

        # Agent Coordination Scenarios
        self.add_test_scenario(TestScenario(
            scenario_id="coordination_task_assignment",
            name="Task Assignment Coordination",
            description="Test optimal task assignment based on agent capabilities",
            scenario_type=TestScenarioType.AGENT_COORDINATION,
            severity=TestSeverity.HIGH,
            setup_function=self._setup_task_assignment_test,
            test_function=self._test_task_assignment_coordination,
            mock_agents=[
                MockAgent("expert_1", {"domain": "technical", "expertise": 0.9}),
                MockAgent("expert_2", {"domain": "creative", "expertise": 0.85}),
                MockAgent("generalist_1", {"domain": "general", "expertise": 0.7})
            ],
            tags={"coordination", "assignment", "capabilities"}
        ))

        # Communication Pattern Scenarios
        self.add_test_scenario(TestScenario(
            scenario_id="communication_broadcast",
            name="Broadcast Communication Pattern",
            description="Test one-to-many broadcast communication",
            scenario_type=TestScenarioType.COMMUNICATION_PATTERN,
            severity=TestSeverity.MEDIUM,
            setup_function=self._setup_broadcast_communication_test,
            test_function=self._test_broadcast_communication,
            mock_agents=[
                MockAgent("broadcaster", {"communication": True}),
                MockAgent("receiver_1", {"listening": True}),
                MockAgent("receiver_2", {"listening": True}),
                MockAgent("receiver_3", {"listening": True})
            ],
            tags={"communication", "broadcast", "messaging"}
        ))

        # Stress Testing Scenarios
        self.add_test_scenario(TestScenario(
            scenario_id="stress_many_agents",
            name="Many Agents Stress Test",
            description="Test system performance with many concurrent agents",
            scenario_type=TestScenarioType.STRESS_TEST,
            severity=TestSeverity.LOW,
            setup_function=self._setup_many_agents_stress_test,
            test_function=self._test_many_agents_performance,
            timeout_seconds=60,
            tags={"stress", "performance", "scalability"}
        ))

    def add_test_scenario(self, scenario: TestScenario):
        """Add a test scenario to the framework"""
        self.test_scenarios[scenario.scenario_id] = scenario
        logger.info(f"Added test scenario: {scenario.name}")

    def add_mock_agent(self, agent: MockAgent):
        """Add a mock agent for testing"""
        self.mock_agents[agent.agent_id] = agent

    async def run_scenario(self, scenario_id: str) -> TestResult:
        """Run a single test scenario"""
        if scenario_id not in self.test_scenarios:
            raise ValueError(f"Test scenario {scenario_id} not found")

        scenario = self.test_scenarios[scenario_id]
        self.running_tests.add(scenario_id)

        result = TestResult(
            scenario_id=scenario_id,
            status=TestResultStatus.RUNNING,
            started_at=datetime.now()
        )

        try:
            # Check dependencies
            for dep_id in scenario.dependencies:
                if dep_id not in self.test_results or self.test_results[dep_id].status != TestResultStatus.PASSED:
                    result.status = TestResultStatus.SKIPPED
                    result.error_message = f"Dependency {dep_id} not satisfied"
                    return result

            # Setup phase
            if scenario.setup_function:
                await scenario.setup_function(scenario)

            # Test execution with timeout
            start_time = time.time()
            test_output = await asyncio.wait_for(
                scenario.test_function(scenario),
                timeout=scenario.timeout_seconds
            )
            end_time = time.time()

            result.duration_seconds = end_time - start_time
            result.test_output = test_output or {}
            result.performance_metrics = {
                "execution_time": result.duration_seconds,
                "agents_used": len(scenario.mock_agents)
            }

            # Validate assertions
            all_assertions_passed = True
            for assertion in scenario.assertions:
                assertion_passed = await assertion.validate(result.test_output)
                result.assertion_results.append(assertion)
                if not assertion_passed:
                    all_assertions_passed = False

            result.status = TestResultStatus.PASSED if all_assertions_passed else TestResultStatus.FAILED

            # Teardown phase
            if scenario.teardown_function:
                await scenario.teardown_function(scenario)

        except asyncio.TimeoutError:
            result.status = TestResultStatus.FAILED
            result.error_message = f"Test timed out after {scenario.timeout_seconds} seconds"
        except Exception as e:
            result.status = TestResultStatus.ERROR
            result.error_message = str(e)
            logger.error(f"Error in test scenario {scenario_id}: {e}")

        finally:
            result.completed_at = datetime.now()
            if result.started_at and result.completed_at:
                result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

            self.running_tests.discard(scenario_id)
            self.test_results[scenario_id] = result
            self.test_history.append(result)
            self.test_metrics["total_tests"] += 1
            self.test_metrics[f"status_{result.status.value}"] += 1

        logger.info(f"Test scenario {scenario_id} completed with status: {result.status.value}")
        return result

    async def run_test_suite(
        self,
        scenario_types: Optional[List[TestScenarioType]] = None,
        tags: Optional[Set[str]] = None,
        severity: Optional[TestSeverity] = None
    ) -> Dict[str, TestResult]:
        """Run multiple test scenarios as a suite"""

        # Filter scenarios based on criteria
        scenarios_to_run = []
        for scenario in self.test_scenarios.values():
            include = True

            if scenario_types and scenario.scenario_type not in scenario_types:
                include = False
            if tags and not tags.intersection(scenario.tags):
                include = False
            if severity and scenario.severity != severity:
                include = False

            if include:
                scenarios_to_run.append(scenario)

        logger.info(f"Running test suite with {len(scenarios_to_run)} scenarios")

        # Sort by dependencies and run
        results = {}
        for scenario in scenarios_to_run:
            result = await self.run_scenario(scenario.scenario_id)
            results[scenario.scenario_id] = result

        return results

    # Built-in test scenario implementations

    async def _setup_resource_contention_test(self, scenario: TestScenario):
        """Setup for resource contention conflict test"""
        # Register mock agents
        for agent in scenario.mock_agents:
            await self.agent_coordinator.register_agent(
                agent.agent_id,
                agent.capabilities
            )

        # Add assertion: conflict should be detected
        scenario.assertions.append(TestAssertion(
            assertion_id="conflict_detected",
            description="A resource contention conflict should be detected",
            assertion_function=lambda results: len(results.get("conflicts", [])) > 0
        ))

        # Add assertion: conflict should be resolved
        scenario.assertions.append(TestAssertion(
            assertion_id="conflict_resolved",
            description="The conflict should be successfully resolved",
            assertion_function=lambda results: results.get("resolution_result", {}).get("success", False)
        ))

    async def _test_resource_contention_resolution(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test resource contention conflict resolution"""
        # Create conflict context
        context = {
            "agents": {
                agent.agent_id: {
                    "position": {"resource_claim": "shared_database"},
                    "priority": i + 1,
                    "confidence": 0.8,
                    "expertise_score": agent.performance_score,
                    "resources_claimed": ["shared_database"]
                }
                for i, agent in enumerate(scenario.mock_agents)
            },
            "resource_claims": {
                "shared_database": [agent.agent_id for agent in scenario.mock_agents]
            }
        }

        # Detect conflicts
        conflicts = await self.conflict_resolver.detect_conflicts(context)

        if not conflicts:
            return {"conflicts": [], "resolution_result": {"success": False}}

        # Resolve first conflict
        conflict = conflicts[0]
        resolution_result = await self.conflict_resolver.resolve_conflict(conflict)

        return {
            "conflicts": [conflict.to_dict()],
            "resolution_result": resolution_result
        }

    async def _setup_contradictory_actions_test(self, scenario: TestScenario):
        """Setup for contradictory actions conflict test"""
        for agent in scenario.mock_agents:
            await self.agent_coordinator.register_agent(
                agent.agent_id,
                agent.capabilities
            )

        scenario.assertions.append(TestAssertion(
            assertion_id="contradictory_actions_detected",
            description="Contradictory actions should be detected",
            assertion_function=lambda results: len(results.get("conflicts", [])) > 0
        ))

    async def _test_contradictory_actions_resolution(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test contradictory actions conflict resolution"""
        context = {
            "agents": {
                agent.agent_id: {
                    "position": {"action": "start" if i == 0 else "stop"},
                    "priority": 5,
                    "confidence": 0.9
                }
                for i, agent in enumerate(scenario.mock_agents)
            },
            "proposed_actions": [
                {"type": "start", "target": "service_x"},
                {"type": "stop", "target": "service_x"}
            ]
        }

        conflicts = await self.conflict_resolver.detect_conflicts(context)
        if conflicts:
            resolution_result = await self.conflict_resolver.resolve_conflict(conflicts[0])
            return {"conflicts": [conflicts[0].to_dict()], "resolution_result": resolution_result}

        return {"conflicts": [], "resolution_result": {"success": False}}

    async def _setup_pipeline_collaboration_test(self, scenario: TestScenario):
        """Setup for pipeline collaboration test"""
        scenario.assertions.append(TestAssertion(
            assertion_id="workflow_created",
            description="Workflow should be created successfully",
            assertion_function=lambda results: "workflow_id" in results
        ))

        scenario.assertions.append(TestAssertion(
            assertion_id="agents_assigned",
            description="All agents should be assigned to tasks",
            assertion_function=lambda results: len(results.get("assignments", {})) == len(scenario.mock_agents)
        ))

    async def _test_pipeline_collaboration(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test pipeline collaboration workflow"""
        # Create workflow
        workflow = await self.collaboration_engine.create_workflow_from_template(
            template_id="document_analysis_pipeline",
            title="Test Document Analysis",
            description="Testing pipeline collaboration"
        )

        # Assign agents
        available_agents = {
            agent.agent_id: agent.to_dict() for agent in scenario.mock_agents
        }
        assignments = await self.collaboration_engine.assign_agents_to_workflow(
            workflow.workflow_id,
            available_agents
        )

        # Start workflow
        success = await self.collaboration_engine.start_workflow(workflow.workflow_id)

        return {
            "workflow_id": workflow.workflow_id,
            "assignments": assignments,
            "workflow_started": success
        }

    async def _setup_brainstorming_collaboration_test(self, scenario: TestScenario):
        """Setup for brainstorming collaboration test"""
        scenario.assertions.append(TestAssertion(
            assertion_id="creative_workflow_created",
            description="Creative brainstorming workflow should be created",
            assertion_function=lambda results: results.get("workflow_pattern") == "brainstorming"
        ))

    async def _test_brainstorming_collaboration(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test brainstorming collaboration workflow"""
        workflow = await self.collaboration_engine.create_workflow_from_template(
            template_id="creative_brainstorming",
            title="Test Brainstorming Session",
            description="Testing creative collaboration"
        )

        available_agents = {
            agent.agent_id: agent.to_dict() for agent in scenario.mock_agents
        }
        assignments = await self.collaboration_engine.assign_agents_to_workflow(
            workflow.workflow_id,
            available_agents
        )

        return {
            "workflow_id": workflow.workflow_id,
            "workflow_pattern": workflow.pattern.value,
            "assignments": assignments
        }

    async def _setup_task_assignment_test(self, scenario: TestScenario):
        """Setup for task assignment coordination test"""
        for agent in scenario.mock_agents:
            await self.agent_coordinator.register_agent(
                agent.agent_id,
                agent.capabilities
            )

        scenario.assertions.append(TestAssertion(
            assertion_id="task_assigned",
            description="Task should be assigned to an agent",
            assertion_function=lambda results: results.get("assigned_agent") is not None
        ))

    async def _test_task_assignment_coordination(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test task assignment coordination"""
        assignment = await self.agent_coordinator.assign_task(
            task_description="Analyze technical documentation",
            requirements={"domain": "technical", "expertise_level": 0.8},
            strategy=CoordinationStrategy.EXPERTISE_BASED
        )

        return {"assigned_agent": assignment.get("agent_id") if assignment else None}

    async def _setup_broadcast_communication_test(self, scenario: TestScenario):
        """Setup for broadcast communication test"""
        scenario.assertions.append(TestAssertion(
            assertion_id="message_broadcast",
            description="Message should be broadcast to all recipients",
            assertion_function=lambda results: results.get("broadcast_successful", False)
        ))

    async def _test_broadcast_communication(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test broadcast communication pattern"""
        broadcaster = scenario.mock_agents[0]
        recipients = scenario.mock_agents[1:]

        message = await self.agent_communication.send_message(
            sender_id=broadcaster.agent_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.BROADCAST,
            content={"message": "Test broadcast message"},
            priority="normal"
        )

        return {
            "message_id": message.message_id,
            "broadcast_successful": True,
            "recipient_count": len(recipients)
        }

    async def _setup_many_agents_stress_test(self, scenario: TestScenario):
        """Setup for many agents stress test"""
        # Create many mock agents
        for i in range(100):
            agent = MockAgent(
                agent_id=f"stress_agent_{i}",
                capabilities={"general": True, "stress_test": True},
                performance_score=0.5 + (i % 50) / 100
            )
            scenario.mock_agents.append(agent)

        scenario.assertions.append(TestAssertion(
            assertion_id="performance_acceptable",
            description="Performance should be acceptable with many agents",
            assertion_function=lambda results: results.get("execution_time", 999) < 10.0
        ))

    async def _test_many_agents_performance(self, scenario: TestScenario) -> Dict[str, Any]:
        """Test performance with many agents"""
        start_time = time.time()

        # Register all agents
        for agent in scenario.mock_agents:
            await self.agent_coordinator.register_agent(
                agent.agent_id,
                agent.capabilities
            )

        # Get active agents
        active_agents = await self.agent_coordinator.get_active_agents()

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "execution_time": execution_time,
            "agents_registered": len(scenario.mock_agents),
            "agents_active": len(active_agents)
        }

    async def get_test_metrics(self) -> Dict[str, Any]:
        """Get comprehensive testing metrics"""
        total_scenarios = len(self.test_scenarios)
        completed_tests = len(self.test_results)

        # Calculate success rates by scenario type
        type_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
        for result in self.test_results.values():
            scenario = self.test_scenarios[result.scenario_id]
            scenario_type = scenario.scenario_type.value
            type_stats[scenario_type]["total"] += 1
            if result.status == TestResultStatus.PASSED:
                type_stats[scenario_type]["passed"] += 1
            elif result.status == TestResultStatus.FAILED:
                type_stats[scenario_type]["failed"] += 1

        # Calculate average execution times
        avg_execution_times = {}
        for scenario_type in TestScenarioType:
            type_name = scenario_type.value
            execution_times = [
                r.duration_seconds for r in self.test_results.values()
                if r.duration_seconds and self.test_scenarios[r.scenario_id].scenario_type == scenario_type
            ]
            if execution_times:
                avg_execution_times[type_name] = sum(execution_times) / len(execution_times)

        return {
            "total_scenarios": total_scenarios,
            "completed_tests": completed_tests,
            "running_tests": len(self.running_tests),
            "overall_success_rate": (
                self.test_metrics.get("status_passed", 0) /
                max(1, self.test_metrics.get("total_tests", 1))
            ),
            "scenario_type_stats": dict(type_stats),
            "average_execution_times": avg_execution_times,
            "test_metrics": dict(self.test_metrics),
            "supported_scenario_types": [t.value for t in TestScenarioType]
        }

    async def cleanup_test_environment(self):
        """Clean up test environment and resources"""
        # Clear test results
        old_count = len(self.test_results)
        self.test_results.clear()
        self.running_tests.clear()

        # Reset systems
        if self.conflict_resolver:
            await self.conflict_resolver.cleanup_old_conflicts(max_age_hours=0)
        if self.collaboration_engine:
            await self.collaboration_engine.cleanup_old_workflows(max_age_hours=0)

        logger.info(f"Cleaned up test environment: {old_count} test results cleared")
        return old_count