"""
Central Agent Coordination System

This module provides the core coordination infrastructure for managing multiple AI agents
in collaborative conversations, including agent lifecycle management, resource allocation,
and coordination strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    PROCESSING = "processing"
    WAITING = "waiting"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class CoordinationStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    EXPERTISE_BASED = "expertise_based"
    WORKLOAD_BALANCED = "workload_balanced"
    PRIORITY_BASED = "priority_based"
    CONSENSUS_DRIVEN = "consensus_driven"
    HIERARCHICAL = "hierarchical"

class AgentPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class AgentCapabilities:
    """Defines an agent's capabilities and specializations"""
    domains: Set[str] = field(default_factory=set)  # e.g., {"technology", "creative", "analysis"}
    skills: Set[str] = field(default_factory=set)    # e.g., {"coding", "writing", "math"}
    languages: Set[str] = field(default_factory=set) # e.g., {"english", "spanish", "python"}
    max_concurrent_tasks: int = 3
    response_time_target: float = 2.0  # seconds
    quality_rating: float = 1.0  # 0.0 to 1.0

    def matches_requirements(self, requirements: Dict[str, Any]) -> float:
        """Calculate how well this agent matches given requirements (0.0 to 1.0)"""
        score = 0.0
        total_weight = 0.0

        # Domain matching
        required_domains = set(requirements.get('domains', []))
        if required_domains:
            domain_overlap = len(self.domains.intersection(required_domains))
            score += (domain_overlap / len(required_domains)) * 0.4
            total_weight += 0.4

        # Skill matching
        required_skills = set(requirements.get('skills', []))
        if required_skills:
            skill_overlap = len(self.skills.intersection(required_skills))
            score += (skill_overlap / len(required_skills)) * 0.4
            total_weight += 0.4

        # Language matching
        required_languages = set(requirements.get('languages', []))
        if required_languages:
            lang_overlap = len(self.languages.intersection(required_languages))
            score += (lang_overlap / len(required_languages)) * 0.2
            total_weight += 0.2

        return score / total_weight if total_weight > 0 else 0.0

@dataclass
class AgentMetrics:
    """Performance and workload metrics for an agent"""
    total_tasks_completed: int = 0
    total_response_time: float = 0.0
    current_workload: int = 0
    error_count: int = 0
    last_activity: Optional[datetime] = None
    success_rate: float = 1.0
    average_response_time: float = 0.0

    def update_response_time(self, response_time: float) -> None:
        """Update average response time with new measurement"""
        self.total_response_time += response_time
        self.total_tasks_completed += 1
        self.average_response_time = self.total_response_time / self.total_tasks_completed

    def calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0.0 to 1.0)"""
        response_score = max(0, 1.0 - (self.average_response_time / 10.0))  # Normalize to 10s max
        workload_score = max(0, 1.0 - (self.current_workload / 10.0))      # Normalize to 10 tasks max
        success_score = self.success_rate

        return (response_score + workload_score + success_score) / 3.0

@dataclass
class ManagedAgent:
    """Represents an agent under coordination management"""
    agent_id: str
    name: str
    agent_type: str
    status: AgentStatus
    capabilities: AgentCapabilities
    metrics: AgentMetrics
    priority: AgentPriority = AgentPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    current_tasks: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return (self.status == AgentStatus.AVAILABLE and
                len(self.current_tasks) < self.capabilities.max_concurrent_tasks)

    def can_handle_task(self, task_requirements: Dict[str, Any]) -> bool:
        """Check if agent can handle a specific task"""
        if not self.is_available():
            return False

        capability_match = self.capabilities.matches_requirements(task_requirements)
        return capability_match >= task_requirements.get('min_capability_match', 0.5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'current_tasks': list(self.current_tasks),
            'capabilities': {
                'domains': list(self.capabilities.domains),
                'skills': list(self.capabilities.skills),
                'languages': list(self.capabilities.languages),
                'max_concurrent_tasks': self.capabilities.max_concurrent_tasks,
                'response_time_target': self.capabilities.response_time_target,
                'quality_rating': self.capabilities.quality_rating
            },
            'metrics': {
                'total_tasks_completed': self.metrics.total_tasks_completed,
                'current_workload': self.metrics.current_workload,
                'error_count': self.metrics.error_count,
                'success_rate': self.metrics.success_rate,
                'average_response_time': self.metrics.average_response_time,
                'efficiency_score': self.metrics.calculate_efficiency_score()
            },
            'metadata': self.metadata
        }

@dataclass
class CoordinationTask:
    """Represents a task to be coordinated among agents"""
    task_id: str
    task_type: str
    requirements: Dict[str, Any]
    priority: AgentPriority
    assigned_agent_id: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class CoordinationConfig:
    """Configuration for agent coordination"""
    strategy: CoordinationStrategy = CoordinationStrategy.EXPERTISE_BASED
    max_agents: int = 10
    heartbeat_interval: int = 30  # seconds
    task_timeout: int = 300  # seconds
    agent_offline_threshold: int = 90  # seconds
    enable_load_balancing: bool = True
    enable_failover: bool = True
    min_agents_for_consensus: int = 3
    consensus_threshold: float = 0.7

class AgentCoordinator:
    """
    Central coordination system for managing multiple AI agents.

    This class orchestrates agent registration, task assignment, load balancing,
    health monitoring, and coordination strategies.
    """

    def __init__(self, config: Optional[CoordinationConfig] = None):
        self.config = config or CoordinationConfig()

        # Agent management
        self.agents: Dict[str, ManagedAgent] = {}
        self.tasks: Dict[str, CoordinationTask] = {}

        # Coordination state
        self.task_queue: deque = deque()
        self.active_conversations: Dict[str, Set[str]] = defaultdict(set)  # conversation_id -> agent_ids

        # Monitoring
        self.coordinator_active = False
        self.monitoring_tasks: List[asyncio.Task] = []

        # Event callbacks
        self.agent_callbacks: List[Callable[[str, ManagedAgent, str], None]] = []
        self.task_callbacks: List[Callable[[str, CoordinationTask, str], None]] = []

        # Statistics
        self.stats = {
            'total_agents_registered': 0,
            'total_tasks_assigned': 0,
            'average_task_completion_time': 0.0,
            'active_conversations': 0,
            'system_efficiency': 0.0
        }

    async def start(self) -> None:
        """Start the coordination system"""
        if self.coordinator_active:
            logger.warning("Coordinator already active")
            return

        self.coordinator_active = True
        logger.info("Starting agent coordinator")

        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._task_processor()),
            asyncio.create_task(self._stats_collector())
        ]

        logger.info("Agent coordinator started successfully")

    async def stop(self) -> None:
        """Stop the coordination system"""
        if not self.coordinator_active:
            return

        self.coordinator_active = False
        logger.info("Stopping agent coordinator")

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()

        logger.info("Agent coordinator stopped")

    async def register_agent(self, agent_id: str, name: str, agent_type: str,
                           capabilities: AgentCapabilities,
                           priority: AgentPriority = AgentPriority.NORMAL,
                           metadata: Dict[str, Any] = None) -> bool:
        """Register a new agent with the coordinator"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered")
            return False

        if len(self.agents) >= self.config.max_agents:
            logger.error(f"Maximum number of agents ({self.config.max_agents}) reached")
            return False

        agent = ManagedAgent(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type,
            status=AgentStatus.AVAILABLE,
            capabilities=capabilities,
            metrics=AgentMetrics(),
            priority=priority,
            last_heartbeat=datetime.now(),
            metadata=metadata or {}
        )

        self.agents[agent_id] = agent
        self.stats['total_agents_registered'] += 1

        logger.info(f"Registered agent {name} ({agent_id}) with capabilities: {capabilities.domains}")
        await self._notify_agent_callbacks(agent_id, agent, "registered")

        return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the coordinator"""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        # Reassign any active tasks
        for task_id in list(agent.current_tasks):
            await self._reassign_task(task_id)

        # Remove from active conversations
        for conversation_id, agent_ids in self.active_conversations.items():
            agent_ids.discard(agent_id)

        del self.agents[agent_id]
        logger.info(f"Unregistered agent {agent.name} ({agent_id})")
        await self._notify_agent_callbacks(agent_id, agent, "unregistered")

        return True

    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update an agent's status"""
        if agent_id not in self.agents:
            return False

        old_status = self.agents[agent_id].status
        self.agents[agent_id].status = status
        self.agents[agent_id].last_heartbeat = datetime.now()

        if old_status != status:
            logger.debug(f"Agent {agent_id} status changed: {old_status.value} -> {status.value}")
            await self._notify_agent_callbacks(agent_id, self.agents[agent_id], "status_changed")

        return True

    async def assign_task(self, task_type: str, requirements: Dict[str, Any],
                         priority: AgentPriority = AgentPriority.NORMAL,
                         conversation_id: Optional[str] = None) -> str:
        """Assign a task to the most suitable agent"""
        task_id = str(uuid.uuid4())

        task = CoordinationTask(
            task_id=task_id,
            task_type=task_type,
            requirements=requirements,
            priority=priority
        )

        # Find best agent for task
        best_agent = await self._find_best_agent(requirements, conversation_id)

        if best_agent:
            await self._assign_task_to_agent(task, best_agent.agent_id)
            if conversation_id:
                self.active_conversations[conversation_id].add(best_agent.agent_id)
        else:
            # Queue task for later assignment
            self.task_queue.append(task)
            logger.warning(f"No suitable agent found for task {task_id}, queued for later")

        self.tasks[task_id] = task
        await self._notify_task_callbacks(task_id, task, "created")

        return task_id

    async def complete_task(self, task_id: str, result: Dict[str, Any],
                          agent_id: str) -> bool:
        """Mark a task as completed"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        agent = self.agents.get(agent_id)

        if not agent or task_id not in agent.current_tasks:
            return False

        # Update task
        task.status = "completed"
        task.completed_at = datetime.now()
        task.result = result

        # Update agent
        agent.current_tasks.remove(task_id)
        agent.metrics.current_workload -= 1

        if task.assigned_at:
            response_time = (task.completed_at - task.assigned_at).total_seconds()
            agent.metrics.update_response_time(response_time)

        agent.status = AgentStatus.AVAILABLE if not agent.current_tasks else AgentStatus.BUSY

        self.stats['total_tasks_assigned'] += 1
        logger.info(f"Task {task_id} completed by agent {agent_id}")
        await self._notify_task_callbacks(task_id, task, "completed")

        return True

    async def fail_task(self, task_id: str, error_message: str, agent_id: str) -> bool:
        """Mark a task as failed"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        agent = self.agents.get(agent_id)

        if not agent or task_id not in agent.current_tasks:
            return False

        # Update task
        task.status = "failed"
        task.completed_at = datetime.now()
        task.error_message = error_message

        # Update agent
        agent.current_tasks.remove(task_id)
        agent.metrics.current_workload -= 1
        agent.metrics.error_count += 1
        agent.status = AgentStatus.AVAILABLE if not agent.current_tasks else AgentStatus.BUSY

        # Update success rate
        total_tasks = agent.metrics.total_tasks_completed + agent.metrics.error_count
        if total_tasks > 0:
            agent.metrics.success_rate = agent.metrics.total_tasks_completed / total_tasks

        logger.error(f"Task {task_id} failed on agent {agent_id}: {error_message}")
        await self._notify_task_callbacks(task_id, task, "failed")

        # Attempt to reassign if failover is enabled
        if self.config.enable_failover:
            await self._reassign_task(task_id)

        return True

    async def get_agents_for_conversation(self, conversation_id: str) -> List[ManagedAgent]:
        """Get all agents participating in a conversation"""
        agent_ids = self.active_conversations.get(conversation_id, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]

    async def get_available_agents(self, requirements: Dict[str, Any] = None) -> List[ManagedAgent]:
        """Get list of available agents, optionally filtered by requirements"""
        available = [agent for agent in self.agents.values() if agent.is_available()]

        if requirements:
            available = [
                agent for agent in available
                if agent.can_handle_task(requirements)
            ]

        return sorted(available, key=lambda a: a.metrics.calculate_efficiency_score(), reverse=True)

    async def get_agent_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a specific agent"""
        agent = self.agents.get(agent_id)
        return agent.to_dict() if agent else None

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_agents = len(self.agents)
        available_agents = len([a for a in self.agents.values() if a.is_available()])
        total_efficiency = sum(a.metrics.calculate_efficiency_score() for a in self.agents.values())

        self.stats.update({
            'total_agents': total_agents,
            'available_agents': available_agents,
            'busy_agents': total_agents - available_agents,
            'pending_tasks': len(self.task_queue),
            'active_conversations': len(self.active_conversations),
            'system_efficiency': total_efficiency / total_agents if total_agents > 0 else 0.0
        })

        return self.stats.copy()

    async def _find_best_agent(self, requirements: Dict[str, Any],
                             conversation_id: Optional[str] = None) -> Optional[ManagedAgent]:
        """Find the best agent for given requirements"""
        candidates = await self.get_available_agents(requirements)

        if not candidates:
            return None

        # If conversation-specific, prefer agents already in the conversation
        if conversation_id and conversation_id in self.active_conversations:
            conversation_agents = [
                agent for agent in candidates
                if agent.agent_id in self.active_conversations[conversation_id]
            ]
            if conversation_agents:
                candidates = conversation_agents

        # Apply coordination strategy
        return await self._apply_coordination_strategy(candidates, requirements)

    async def _apply_coordination_strategy(self, candidates: List[ManagedAgent],
                                         requirements: Dict[str, Any]) -> Optional[ManagedAgent]:
        """Apply the configured coordination strategy to select an agent"""
        if not candidates:
            return None

        strategy = self.config.strategy

        if strategy == CoordinationStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            return min(candidates, key=lambda a: a.metrics.total_tasks_completed)

        elif strategy == CoordinationStrategy.EXPERTISE_BASED:
            # Select based on capability match
            scored_agents = [
                (agent, agent.capabilities.matches_requirements(requirements))
                for agent in candidates
            ]
            return max(scored_agents, key=lambda x: x[1])[0]

        elif strategy == CoordinationStrategy.WORKLOAD_BALANCED:
            # Select agent with lowest current workload
            return min(candidates, key=lambda a: a.metrics.current_workload)

        elif strategy == CoordinationStrategy.PRIORITY_BASED:
            # Select highest priority agent first
            priority_order = [AgentPriority.CRITICAL, AgentPriority.HIGH, AgentPriority.NORMAL, AgentPriority.LOW]
            for priority in priority_order:
                priority_agents = [a for a in candidates if a.priority == priority]
                if priority_agents:
                    return max(priority_agents, key=lambda a: a.metrics.calculate_efficiency_score())

        else:  # Default to efficiency-based selection
            return max(candidates, key=lambda a: a.metrics.calculate_efficiency_score())

    async def _assign_task_to_agent(self, task: CoordinationTask, agent_id: str) -> None:
        """Assign a specific task to a specific agent"""
        agent = self.agents[agent_id]

        task.assigned_agent_id = agent_id
        task.assigned_at = datetime.now()
        task.status = "assigned"

        agent.current_tasks.add(task.task_id)
        agent.metrics.current_workload += 1
        agent.status = AgentStatus.BUSY

        logger.debug(f"Assigned task {task.task_id} to agent {agent_id}")

    async def _reassign_task(self, task_id: str) -> bool:
        """Reassign a failed or orphaned task"""
        task = self.tasks.get(task_id)
        if not task or task.status == "completed":
            return False

        # Reset task state
        if task.assigned_agent_id:
            old_agent = self.agents.get(task.assigned_agent_id)
            if old_agent:
                old_agent.current_tasks.discard(task_id)
                old_agent.metrics.current_workload = max(0, old_agent.metrics.current_workload - 1)

        task.assigned_agent_id = None
        task.assigned_at = None
        task.status = "pending"

        # Find new agent
        best_agent = await self._find_best_agent(task.requirements)
        if best_agent:
            await self._assign_task_to_agent(task, best_agent.agent_id)
            logger.info(f"Reassigned task {task_id} to agent {best_agent.agent_id}")
            return True
        else:
            self.task_queue.append(task)
            logger.warning(f"Could not reassign task {task_id}, queued for later")
            return False

    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats and detect offline agents"""
        while self.coordinator_active:
            try:
                now = datetime.now()
                offline_threshold = timedelta(seconds=self.config.agent_offline_threshold)

                for agent_id, agent in list(self.agents.items()):
                    if agent.last_heartbeat and (now - agent.last_heartbeat) > offline_threshold:
                        logger.warning(f"Agent {agent_id} appears offline, marking as offline")
                        agent.status = AgentStatus.OFFLINE

                        # Reassign tasks from offline agent
                        for task_id in list(agent.current_tasks):
                            await self._reassign_task(task_id)

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)

    async def _task_processor(self) -> None:
        """Process queued tasks"""
        while self.coordinator_active:
            try:
                if self.task_queue:
                    task = self.task_queue.popleft()
                    best_agent = await self._find_best_agent(task.requirements)

                    if best_agent:
                        await self._assign_task_to_agent(task, best_agent.agent_id)
                        await self._notify_task_callbacks(task.task_id, task, "assigned")
                    else:
                        # Put back in queue
                        self.task_queue.append(task)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(5)

    async def _stats_collector(self) -> None:
        """Collect and update system statistics"""
        while self.coordinator_active:
            try:
                await self.get_system_stats()
                await asyncio.sleep(60)  # Update stats every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats collector: {e}")
                await asyncio.sleep(10)

    def add_agent_callback(self, callback: Callable[[str, ManagedAgent, str], None]) -> None:
        """Add callback for agent events"""
        self.agent_callbacks.append(callback)

    def add_task_callback(self, callback: Callable[[str, CoordinationTask, str], None]) -> None:
        """Add callback for task events"""
        self.task_callbacks.append(callback)

    async def _notify_agent_callbacks(self, agent_id: str, agent: ManagedAgent, event: str) -> None:
        """Notify agent event callbacks"""
        for callback in self.agent_callbacks:
            try:
                callback(agent_id, agent, event)
            except Exception as e:
                logger.error(f"Error in agent callback: {e}")

    async def _notify_task_callbacks(self, task_id: str, task: CoordinationTask, event: str) -> None:
        """Notify task event callbacks"""
        for callback in self.task_callbacks:
            try:
                callback(task_id, task, event)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")