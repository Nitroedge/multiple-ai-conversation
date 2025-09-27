"""
Agent Role Management System

This module provides dynamic role assignment, capability management, and role-based
coordination for multi-agent conversations and collaborative tasks.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class RoleType(Enum):
    FACILITATOR = "facilitator"        # Guides conversation flow
    SPECIALIST = "specialist"          # Domain expert
    COORDINATOR = "coordinator"        # Coordinates other agents
    MODERATOR = "moderator"           # Manages discussion quality
    ANALYZER = "analyzer"             # Analyzes information
    SYNTHESIZER = "synthesizer"       # Combines different perspectives
    CRITIC = "critic"                 # Provides critical analysis
    SUPPORTER = "supporter"           # Provides support and encouragement
    RESEARCHER = "researcher"         # Gathers information
    EXECUTOR = "executor"             # Executes tasks and actions

class RoleStatus(Enum):
    AVAILABLE = "available"
    ASSIGNED = "assigned"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"

class AssignmentStrategy(Enum):
    CAPABILITY_BASED = "capability_based"
    WORKLOAD_BALANCED = "workload_balanced"
    PREFERENCE_BASED = "preference_based"
    RANDOM = "random"
    EXPERIENCE_BASED = "experience_based"
    COLLABORATIVE_FIT = "collaborative_fit"

@dataclass
class RoleCapability:
    """Defines a specific capability within a role"""
    capability_id: str
    name: str
    description: str
    required_skills: Set[str] = field(default_factory=set)
    proficiency_level: float = 1.0  # 0.0 to 1.0
    prerequisites: Set[str] = field(default_factory=set)  # Other capability IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'capability_id': self.capability_id,
            'name': self.name,
            'description': self.description,
            'required_skills': list(self.required_skills),
            'proficiency_level': self.proficiency_level,
            'prerequisites': list(self.prerequisites),
            'metadata': self.metadata
        }

@dataclass
class AgentRole:
    """Defines a role that can be assigned to agents"""
    role_id: str
    name: str
    role_type: RoleType
    description: str
    responsibilities: List[str]
    capabilities: List[RoleCapability]
    required_qualifications: Set[str] = field(default_factory=set)
    preferred_qualifications: Set[str] = field(default_factory=set)
    max_concurrent_assignments: int = 1
    priority: int = 0  # Higher numbers = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_capability_score(self, agent_capabilities: Set[str]) -> float:
        """Calculate how well an agent's capabilities match this role"""
        if not self.required_qualifications:
            return 1.0

        required_match = len(self.required_qualifications.intersection(agent_capabilities))
        required_total = len(self.required_qualifications)

        base_score = required_match / required_total if required_total > 0 else 1.0

        # Bonus for preferred qualifications
        preferred_match = len(self.preferred_qualifications.intersection(agent_capabilities))
        preferred_total = len(self.preferred_qualifications)

        bonus = 0.0
        if preferred_total > 0:
            bonus = (preferred_match / preferred_total) * 0.2  # 20% bonus

        return min(1.0, base_score + bonus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role_id': self.role_id,
            'name': self.name,
            'role_type': self.role_type.value,
            'description': self.description,
            'responsibilities': self.responsibilities,
            'capabilities': [cap.to_dict() for cap in self.capabilities],
            'required_qualifications': list(self.required_qualifications),
            'preferred_qualifications': list(self.preferred_qualifications),
            'max_concurrent_assignments': self.max_concurrent_assignments,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class RoleAssignment:
    """Represents an assignment of a role to an agent"""
    assignment_id: str
    role_id: str
    agent_id: str
    conversation_id: Optional[str]
    status: RoleStatus
    assigned_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    performance_score: Optional[float] = None
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_duration(self) -> Optional[timedelta]:
        """Get the duration of this assignment"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.now() - self.started_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'assignment_id': self.assignment_id,
            'role_id': self.role_id,
            'agent_id': self.agent_id,
            'conversation_id': self.conversation_id,
            'status': self.status.value,
            'assigned_at': self.assigned_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'performance_score': self.performance_score,
            'feedback': self.feedback,
            'duration_seconds': self.get_duration().total_seconds() if self.get_duration() else None,
            'metadata': self.metadata
        }

@dataclass
class AgentProfile:
    """Profile information for role assignment"""
    agent_id: str
    capabilities: Set[str]
    skills: Set[str]
    experience_ratings: Dict[str, float] = field(default_factory=dict)  # role_type -> rating
    preferences: Set[RoleType] = field(default_factory=set)
    availability: bool = True
    current_workload: int = 0
    max_concurrent_roles: int = 3
    performance_history: List[float] = field(default_factory=list)
    collaboration_ratings: Dict[str, float] = field(default_factory=dict)  # agent_id -> rating

    def get_average_performance(self) -> float:
        """Get average performance score"""
        if not self.performance_history:
            return 0.5  # Default neutral rating
        return sum(self.performance_history) / len(self.performance_history)

    def get_collaboration_score(self, other_agent_ids: Set[str]) -> float:
        """Get collaboration score with other agents"""
        if not other_agent_ids:
            return 1.0

        scores = [self.collaboration_ratings.get(agent_id, 0.5) for agent_id in other_agent_ids]
        return sum(scores) / len(scores) if scores else 0.5

    def can_take_role(self) -> bool:
        """Check if agent can take on additional roles"""
        return self.availability and self.current_workload < self.max_concurrent_roles

@dataclass
class RoleManagerConfig:
    """Configuration for role management"""
    max_roles_per_agent: int = 3
    role_timeout_hours: int = 24
    enable_role_rotation: bool = True
    rotation_interval_hours: int = 4
    assignment_strategy: AssignmentStrategy = AssignmentStrategy.CAPABILITY_BASED
    enable_performance_tracking: bool = True
    min_performance_threshold: float = 0.3

class RoleManager:
    """
    Manages agent roles, assignments, and role-based coordination in multi-agent systems.

    This class handles dynamic role assignment, capability matching, performance tracking,
    and role-based coordination strategies.
    """

    def __init__(self, config: Optional[RoleManagerConfig] = None):
        self.config = config or RoleManagerConfig()

        # Role definitions
        self.roles: Dict[str, AgentRole] = {}
        self.role_templates: Dict[RoleType, List[AgentRole]] = defaultdict(list)

        # Agent management
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.assignments: Dict[str, RoleAssignment] = {}

        # Assignment tracking
        self.active_assignments: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> assignment_ids
        self.conversation_assignments: Dict[str, Set[str]] = defaultdict(set)  # conversation_id -> assignment_ids

        # System state
        self.role_manager_active = False
        self.background_tasks: List[asyncio.Task] = []

        # Event callbacks
        self.assignment_callbacks: List[Callable[[str, RoleAssignment, str], None]] = []
        self.role_callbacks: List[Callable[[str, AgentRole, str], None]] = []

        # Statistics
        self.stats = {
            'total_roles': 0,
            'total_assignments': 0,
            'active_assignments': 0,
            'average_assignment_duration': 0.0,
            'role_success_rate': 0.0
        }

        # Initialize default roles
        self._initialize_default_roles()

    async def start(self) -> None:
        """Start the role management system"""
        if self.role_manager_active:
            logger.warning("Role manager already active")
            return

        self.role_manager_active = True
        logger.info("Starting role management system")

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._assignment_monitor()),
            asyncio.create_task(self._performance_tracker()),
            asyncio.create_task(self._role_rotator())
        ]

        logger.info("Role management system started")

    async def stop(self) -> None:
        """Stop the role management system"""
        if not self.role_manager_active:
            return

        self.role_manager_active = False
        logger.info("Stopping role management system")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        logger.info("Role management system stopped")

    def register_role(self, role: AgentRole) -> None:
        """Register a new role definition"""
        self.roles[role.role_id] = role
        self.role_templates[role.role_type].append(role)
        self.stats['total_roles'] += 1

        logger.info(f"Registered role: {role.name} ({role.role_id}) of type {role.role_type.value}")

    def register_agent_profile(self, profile: AgentProfile) -> None:
        """Register an agent profile for role assignment"""
        self.agent_profiles[profile.agent_id] = profile
        logger.info(f"Registered agent profile: {profile.agent_id}")

    async def assign_role(self, role_id: str, agent_id: Optional[str] = None,
                         conversation_id: Optional[str] = None,
                         strategy: Optional[AssignmentStrategy] = None) -> Optional[str]:
        """Assign a role to an agent"""
        role = self.roles.get(role_id)
        if not role:
            logger.error(f"Role {role_id} not found")
            return None

        # Find best agent if not specified
        if agent_id is None:
            agent_id = await self._find_best_agent_for_role(
                role, conversation_id, strategy or self.config.assignment_strategy
            )

        if not agent_id:
            logger.warning(f"No suitable agent found for role {role_id}")
            return None

        # Check if agent can take the role
        profile = self.agent_profiles.get(agent_id)
        if not profile or not profile.can_take_role():
            logger.warning(f"Agent {agent_id} cannot take role {role_id}")
            return None

        # Create assignment
        assignment_id = str(uuid.uuid4())
        assignment = RoleAssignment(
            assignment_id=assignment_id,
            role_id=role_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            status=RoleStatus.ASSIGNED
        )

        # Update tracking
        self.assignments[assignment_id] = assignment
        self.active_assignments[agent_id].add(assignment_id)
        if conversation_id:
            self.conversation_assignments[conversation_id].add(assignment_id)

        # Update agent profile
        profile.current_workload += 1

        self.stats['total_assignments'] += 1
        self.stats['active_assignments'] += 1

        logger.info(f"Assigned role {role.name} to agent {agent_id} (assignment {assignment_id})")
        await self._notify_assignment_callbacks(assignment_id, assignment, "assigned")

        return assignment_id

    async def start_role(self, assignment_id: str) -> bool:
        """Start an assigned role"""
        assignment = self.assignments.get(assignment_id)
        if not assignment or assignment.status != RoleStatus.ASSIGNED:
            return False

        assignment.status = RoleStatus.ACTIVE
        assignment.started_at = datetime.now()

        logger.info(f"Started role assignment {assignment_id}")
        await self._notify_assignment_callbacks(assignment_id, assignment, "started")

        return True

    async def complete_role(self, assignment_id: str, performance_score: Optional[float] = None,
                           feedback: Optional[str] = None) -> bool:
        """Complete a role assignment"""
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            return False

        assignment.status = RoleStatus.COMPLETED
        assignment.completed_at = datetime.now()
        assignment.performance_score = performance_score
        assignment.feedback = feedback

        # Update agent profile
        profile = self.agent_profiles.get(assignment.agent_id)
        if profile:
            profile.current_workload = max(0, profile.current_workload - 1)
            if performance_score is not None:
                profile.performance_history.append(performance_score)
                # Keep only recent performance history
                if len(profile.performance_history) > 100:
                    profile.performance_history = profile.performance_history[-50:]

        # Update tracking
        self.active_assignments[assignment.agent_id].discard(assignment_id)
        if assignment.conversation_id:
            self.conversation_assignments[assignment.conversation_id].discard(assignment_id)

        self.stats['active_assignments'] = max(0, self.stats['active_assignments'] - 1)

        logger.info(f"Completed role assignment {assignment_id} with score {performance_score}")
        await self._notify_assignment_callbacks(assignment_id, assignment, "completed")

        return True

    async def suspend_role(self, assignment_id: str, reason: str) -> bool:
        """Suspend a role assignment"""
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            return False

        assignment.status = RoleStatus.SUSPENDED
        assignment.metadata['suspension_reason'] = reason

        logger.info(f"Suspended role assignment {assignment_id}: {reason}")
        await self._notify_assignment_callbacks(assignment_id, assignment, "suspended")

        return True

    async def reassign_role(self, assignment_id: str, new_agent_id: str) -> bool:
        """Reassign a role to a different agent"""
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            return False

        old_agent_id = assignment.agent_id

        # Update agent profiles
        old_profile = self.agent_profiles.get(old_agent_id)
        if old_profile:
            old_profile.current_workload = max(0, old_profile.current_workload - 1)

        new_profile = self.agent_profiles.get(new_agent_id)
        if not new_profile or not new_profile.can_take_role():
            return False

        # Update assignment
        assignment.agent_id = new_agent_id
        assignment.metadata['reassigned_from'] = old_agent_id
        assignment.metadata['reassigned_at'] = datetime.now().isoformat()

        # Update tracking
        self.active_assignments[old_agent_id].discard(assignment_id)
        self.active_assignments[new_agent_id].add(assignment_id)

        new_profile.current_workload += 1

        logger.info(f"Reassigned role {assignment_id} from {old_agent_id} to {new_agent_id}")
        await self._notify_assignment_callbacks(assignment_id, assignment, "reassigned")

        return True

    async def get_agent_roles(self, agent_id: str) -> List[RoleAssignment]:
        """Get all active role assignments for an agent"""
        assignment_ids = self.active_assignments.get(agent_id, set())
        return [self.assignments[aid] for aid in assignment_ids if aid in self.assignments]

    async def get_conversation_roles(self, conversation_id: str) -> List[RoleAssignment]:
        """Get all role assignments for a conversation"""
        assignment_ids = self.conversation_assignments.get(conversation_id, set())
        return [self.assignments[aid] for aid in assignment_ids if aid in self.assignments]

    async def get_role_assignments(self, role_id: str) -> List[RoleAssignment]:
        """Get all assignments for a specific role"""
        return [assignment for assignment in self.assignments.values()
                if assignment.role_id == role_id]

    async def suggest_roles_for_conversation(self, conversation_id: str,
                                           required_roles: Optional[List[RoleType]] = None,
                                           participant_count: int = 3) -> List[AgentRole]:
        """Suggest optimal roles for a conversation"""
        if required_roles is None:
            # Default role suggestions based on conversation size
            if participant_count <= 2:
                required_roles = [RoleType.SPECIALIST, RoleType.ANALYST]
            elif participant_count <= 4:
                required_roles = [RoleType.FACILITATOR, RoleType.SPECIALIST, RoleType.ANALYZER]
            else:
                required_roles = [RoleType.FACILITATOR, RoleType.COORDINATOR,
                                RoleType.SPECIALIST, RoleType.MODERATOR]

        suggested_roles = []
        for role_type in required_roles:
            # Find best role of this type
            candidates = self.role_templates.get(role_type, [])
            if candidates:
                # Sort by priority and take the best
                best_role = max(candidates, key=lambda r: r.priority)
                suggested_roles.append(best_role)

        return suggested_roles

    async def optimize_role_assignments(self, conversation_id: str) -> Dict[str, Any]:
        """Optimize role assignments for better performance"""
        current_assignments = await self.get_conversation_roles(conversation_id)

        optimization_suggestions = {
            'current_assignments': len(current_assignments),
            'performance_issues': [],
            'suggestions': []
        }

        # Analyze current performance
        for assignment in current_assignments:
            if assignment.performance_score and assignment.performance_score < self.config.min_performance_threshold:
                optimization_suggestions['performance_issues'].append({
                    'assignment_id': assignment.assignment_id,
                    'agent_id': assignment.agent_id,
                    'role_id': assignment.role_id,
                    'performance_score': assignment.performance_score
                })

                # Suggest reassignment
                role = self.roles.get(assignment.role_id)
                if role:
                    better_agent = await self._find_best_agent_for_role(role, conversation_id)
                    if better_agent and better_agent != assignment.agent_id:
                        optimization_suggestions['suggestions'].append({
                            'type': 'reassign',
                            'assignment_id': assignment.assignment_id,
                            'current_agent': assignment.agent_id,
                            'suggested_agent': better_agent
                        })

        return optimization_suggestions

    async def _find_best_agent_for_role(self, role: AgentRole, conversation_id: Optional[str] = None,
                                       strategy: AssignmentStrategy = AssignmentStrategy.CAPABILITY_BASED) -> Optional[str]:
        """Find the best agent for a specific role"""
        candidates = []

        for agent_id, profile in self.agent_profiles.items():
            if not profile.can_take_role():
                continue

            # Calculate capability match
            capability_score = role.get_capability_score(profile.capabilities)
            if capability_score < 0.3:  # Minimum capability threshold
                continue

            score = capability_score

            # Apply strategy-specific scoring
            if strategy == AssignmentStrategy.WORKLOAD_BALANCED:
                workload_factor = 1.0 - (profile.current_workload / profile.max_concurrent_roles)
                score = score * 0.7 + workload_factor * 0.3

            elif strategy == AssignmentStrategy.EXPERIENCE_BASED:
                experience_score = profile.experience_ratings.get(role.role_type.value, 0.5)
                score = score * 0.6 + experience_score * 0.4

            elif strategy == AssignmentStrategy.PREFERENCE_BASED:
                preference_bonus = 0.2 if role.role_type in profile.preferences else 0.0
                score += preference_bonus

            elif strategy == AssignmentStrategy.COLLABORATIVE_FIT:
                if conversation_id:
                    other_agents = {a.agent_id for a in await self.get_conversation_roles(conversation_id)}
                    collab_score = profile.get_collaboration_score(other_agents)
                    score = score * 0.7 + collab_score * 0.3

            # Performance factor
            performance_score = profile.get_average_performance()
            score = score * 0.8 + performance_score * 0.2

            candidates.append((agent_id, score))

        if not candidates:
            return None

        # Sort by score and return best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _initialize_default_roles(self) -> None:
        """Initialize default role templates"""
        default_roles = [
            AgentRole(
                role_id="facilitator_001",
                name="Conversation Facilitator",
                role_type=RoleType.FACILITATOR,
                description="Guides conversation flow and ensures productive discussion",
                responsibilities=[
                    "Guide conversation flow",
                    "Ensure all participants contribute",
                    "Manage time and agenda",
                    "Resolve communication issues"
                ],
                capabilities=[
                    RoleCapability("facilitation", "Group Facilitation", "Ability to guide group discussions"),
                    RoleCapability("communication", "Clear Communication", "Excellent communication skills")
                ],
                required_qualifications={"communication", "leadership"},
                preferred_qualifications={"psychology", "group_dynamics"},
                priority=10
            ),
            AgentRole(
                role_id="specialist_001",
                name="Domain Specialist",
                role_type=RoleType.SPECIALIST,
                description="Provides expert knowledge in specific domain",
                responsibilities=[
                    "Provide expert knowledge",
                    "Answer technical questions",
                    "Validate information accuracy",
                    "Share best practices"
                ],
                capabilities=[
                    RoleCapability("expertise", "Domain Expertise", "Deep knowledge in specific field"),
                    RoleCapability("analysis", "Technical Analysis", "Ability to analyze complex problems")
                ],
                required_qualifications={"domain_expertise", "analysis"},
                preferred_qualifications={"teaching", "research"},
                priority=9
            ),
            AgentRole(
                role_id="coordinator_001",
                name="Task Coordinator",
                role_type=RoleType.COORDINATOR,
                description="Coordinates tasks and ensures alignment",
                responsibilities=[
                    "Coordinate team activities",
                    "Track progress and deliverables",
                    "Ensure alignment with objectives",
                    "Manage dependencies"
                ],
                capabilities=[
                    RoleCapability("coordination", "Task Coordination", "Ability to coordinate multiple activities"),
                    RoleCapability("planning", "Strategic Planning", "Planning and organizing skills")
                ],
                required_qualifications={"project_management", "organization"},
                preferred_qualifications={"agile", "leadership"},
                priority=8
            ),
            AgentRole(
                role_id="analyzer_001",
                name="Information Analyzer",
                role_type=RoleType.ANALYZER,
                description="Analyzes information and provides insights",
                responsibilities=[
                    "Analyze complex information",
                    "Identify patterns and trends",
                    "Provide data-driven insights",
                    "Evaluate options and alternatives"
                ],
                capabilities=[
                    RoleCapability("analysis", "Data Analysis", "Ability to analyze complex data"),
                    RoleCapability("critical_thinking", "Critical Thinking", "Strong analytical reasoning")
                ],
                required_qualifications={"analysis", "critical_thinking"},
                preferred_qualifications={"statistics", "research"},
                priority=7
            )
        ]

        for role in default_roles:
            self.register_role(role)

    async def _assignment_monitor(self) -> None:
        """Monitor role assignments for timeouts and issues"""
        while self.role_manager_active:
            try:
                now = datetime.now()
                timeout_threshold = timedelta(hours=self.config.role_timeout_hours)

                # Check for assignment timeouts
                for assignment in list(self.assignments.values()):
                    if (assignment.status == RoleStatus.ACTIVE and
                        assignment.started_at and
                        (now - assignment.started_at) > timeout_threshold):

                        logger.warning(f"Role assignment {assignment.assignment_id} timed out")
                        await self.suspend_role(assignment.assignment_id, "Timeout")

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in assignment monitor: {e}")
                await asyncio.sleep(60)

    async def _performance_tracker(self) -> None:
        """Track and analyze role performance"""
        while self.role_manager_active:
            try:
                if self.config.enable_performance_tracking:
                    # Calculate success rate
                    completed_assignments = [a for a in self.assignments.values()
                                           if a.status == RoleStatus.COMPLETED]

                    if completed_assignments:
                        successful = len([a for a in completed_assignments
                                        if a.performance_score and a.performance_score >= 0.7])
                        self.stats['role_success_rate'] = successful / len(completed_assignments)

                        # Calculate average duration
                        durations = [a.get_duration().total_seconds()
                                   for a in completed_assignments
                                   if a.get_duration()]
                        if durations:
                            self.stats['average_assignment_duration'] = sum(durations) / len(durations)

                await asyncio.sleep(600)  # Update every 10 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance tracker: {e}")
                await asyncio.sleep(60)

    async def _role_rotator(self) -> None:
        """Implement role rotation if enabled"""
        while self.role_manager_active:
            try:
                if self.config.enable_role_rotation:
                    rotation_threshold = timedelta(hours=self.config.rotation_interval_hours)
                    now = datetime.now()

                    for assignment in list(self.assignments.values()):
                        if (assignment.status == RoleStatus.ACTIVE and
                            assignment.started_at and
                            (now - assignment.started_at) > rotation_threshold):

                            # Consider rotation based on performance
                            if (assignment.performance_score and
                                assignment.performance_score < 0.6):

                                role = self.roles.get(assignment.role_id)
                                if role:
                                    better_agent = await self._find_best_agent_for_role(
                                        role, assignment.conversation_id
                                    )
                                    if better_agent and better_agent != assignment.agent_id:
                                        logger.info(f"Rotating role {assignment.assignment_id} due to performance")
                                        await self.reassign_role(assignment.assignment_id, better_agent)

                await asyncio.sleep(3600)  # Check every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in role rotator: {e}")
                await asyncio.sleep(300)

    def add_assignment_callback(self, callback: Callable[[str, RoleAssignment, str], None]) -> None:
        """Add callback for assignment events"""
        self.assignment_callbacks.append(callback)

    def add_role_callback(self, callback: Callable[[str, AgentRole, str], None]) -> None:
        """Add callback for role events"""
        self.role_callbacks.append(callback)

    async def _notify_assignment_callbacks(self, assignment_id: str, assignment: RoleAssignment, event: str) -> None:
        """Notify assignment event callbacks"""
        for callback in self.assignment_callbacks:
            try:
                callback(assignment_id, assignment, event)
            except Exception as e:
                logger.error(f"Error in assignment callback: {e}")

    async def get_role_manager_stats(self) -> Dict[str, Any]:
        """Get role management statistics"""
        self.stats.update({
            'total_roles': len(self.roles),
            'active_assignments': len([a for a in self.assignments.values()
                                     if a.status == RoleStatus.ACTIVE]),
            'total_agents': len(self.agent_profiles),
            'available_agents': len([p for p in self.agent_profiles.values() if p.can_take_role()])
        })

        return self.stats.copy()