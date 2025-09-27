"""
Dynamic Role Assignment System
AI-driven role optimization and intelligent task delegation
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
from pydantic import BaseModel, Field
from uuid import uuid4
import numpy as np

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    """Types of roles in the system"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    ANALYST = "analyst"
    CREATIVE = "creative"
    CRITIC = "critic"
    FACILITATOR = "facilitator"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    MODERATOR = "moderator"
    ADVISOR = "advisor"
    INNOVATOR = "innovator"


class SkillDomain(str, Enum):
    """Skill domains for role assignment"""
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    COMMUNICATION = "communication"
    LEADERSHIP = "leadership"
    PROBLEM_SOLVING = "problem_solving"
    DOMAIN_EXPERTISE = "domain_expertise"
    PROJECT_MANAGEMENT = "project_management"
    QUALITY_ASSURANCE = "quality_assurance"
    RESEARCH = "research"


class AgentCapability(BaseModel):
    """Capability profile for an agent"""
    skill_domain: SkillDomain
    proficiency_level: float = Field(ge=0.0, le=1.0, description="Skill level (0-1)")
    experience_points: int = Field(default=0, description="Experience in this domain")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    avg_response_time_ms: float = Field(default=1000.0)
    confidence_scores: List[float] = Field(default_factory=list)


class AgentProfile(BaseModel):
    """Comprehensive agent profile for role assignment"""
    agent_id: str
    name: str = ""
    agent_type: str = ""

    # Capabilities
    capabilities: Dict[SkillDomain, AgentCapability] = Field(default_factory=dict)
    specialized_domains: List[SkillDomain] = Field(default_factory=list)

    # Performance metrics
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    avg_task_completion_time_ms: float = 1000.0
    overall_success_rate: float = 0.5
    reliability_score: float = 0.5

    # Role history
    role_assignments: List[Dict[str, Any]] = Field(default_factory=list)
    preferred_roles: List[RoleType] = Field(default_factory=list)
    avoided_roles: List[RoleType] = Field(default_factory=list)

    # Collaboration metrics
    collaboration_scores: Dict[str, float] = Field(default_factory=dict)
    team_synergy_ratings: List[float] = Field(default_factory=list)

    # Learning and adaptation
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    adaptation_speed: float = Field(default=0.05, ge=0.0, le=0.5)

    # Status
    availability_status: str = "available"
    current_workload: int = 0
    max_concurrent_tasks: int = 3

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

    def get_capability_score(self, skill_domain: SkillDomain) -> float:
        """Get capability score for a specific skill domain"""
        if skill_domain in self.capabilities:
            capability = self.capabilities[skill_domain]
            # Combine proficiency, experience, and success rate
            base_score = capability.proficiency_level
            experience_bonus = min(0.2, capability.experience_points / 1000)
            success_bonus = capability.success_rate * 0.1
            return min(1.0, base_score + experience_bonus + success_bonus)
        return 0.0

    def update_capability(self, skill_domain: SkillDomain, task_result: Dict[str, Any]):
        """Update capability based on task performance"""
        if skill_domain not in self.capabilities:
            self.capabilities[skill_domain] = AgentCapability(skill_domain=skill_domain)

        capability = self.capabilities[skill_domain]

        # Update experience
        capability.experience_points += 1

        # Update success rate
        was_successful = task_result.get("success", False)
        confidence = task_result.get("confidence", 0.5)
        response_time = task_result.get("execution_time_ms", 1000)

        # Moving average for success rate
        capability.success_rate = (
            capability.success_rate * 0.9 + (1.0 if was_successful else 0.0) * 0.1
        )

        # Update confidence scores
        capability.confidence_scores.append(confidence)
        if len(capability.confidence_scores) > 100:
            capability.confidence_scores = capability.confidence_scores[-100:]

        # Update response time
        capability.avg_response_time_ms = (
            capability.avg_response_time_ms * 0.9 + response_time * 0.1
        )

        # Adaptive learning - improve proficiency based on performance
        if was_successful and confidence > 0.7:
            improvement = self.learning_rate * (1 - capability.proficiency_level) * 0.1
            capability.proficiency_level = min(1.0, capability.proficiency_level + improvement)
        elif not was_successful:
            degradation = self.learning_rate * 0.05
            capability.proficiency_level = max(0.0, capability.proficiency_level - degradation)

        capability.last_updated = datetime.utcnow()


class RoleRequirement(BaseModel):
    """Requirements for a specific role"""
    role_type: RoleType
    required_skills: Dict[SkillDomain, float] = Field(default_factory=dict)
    min_experience_points: int = 0
    min_success_rate: float = 0.5
    max_response_time_ms: float = 5000.0
    required_agent_types: List[str] = Field(default_factory=list)
    excluded_agent_types: List[str] = Field(default_factory=list)
    collaboration_requirements: List[str] = Field(default_factory=list)
    workload_capacity: int = 1
    priority_level: int = Field(default=5, ge=1, le=10)


class RoleAssignment(BaseModel):
    """Assignment of an agent to a specific role"""
    assignment_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    role_type: RoleType
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None

    # Assignment details
    assignment_score: float = Field(description="How well the agent fits the role")
    confidence_level: float = Field(default=0.5)
    expected_performance: float = Field(default=0.5)

    # Status
    status: str = "assigned"  # assigned, active, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Performance tracking
    actual_performance: Optional[float] = None
    task_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_by: str = "system"
    notes: str = ""


class TeamComposition(BaseModel):
    """Composition of a team for complex tasks"""
    team_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = ""

    # Team structure
    role_assignments: List[RoleAssignment] = Field(default_factory=list)
    team_lead_agent_id: Optional[str] = None

    # Team metrics
    synergy_score: float = Field(default=0.5)
    estimated_performance: float = Field(default=0.5)
    skill_coverage: Dict[SkillDomain, float] = Field(default_factory=dict)

    # Status
    status: str = "forming"  # forming, active, completed, disbanded
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceBasedRoleSelection:
    """Role selection based on historical performance data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def calculate_role_fitness(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate how well an agent fits a specific role"""
        try:
            fitness_scores = []

            # 1. Skill alignment (40% weight)
            skill_score = self._calculate_skill_alignment(agent_profile, role_requirement)
            fitness_scores.append(("skill", skill_score, 0.4))

            # 2. Experience factor (20% weight)
            experience_score = self._calculate_experience_factor(agent_profile, role_requirement)
            fitness_scores.append(("experience", experience_score, 0.2))

            # 3. Performance history (25% weight)
            performance_score = self._calculate_performance_score(agent_profile, role_requirement)
            fitness_scores.append(("performance", performance_score, 0.25))

            # 4. Availability and workload (15% weight)
            availability_score = self._calculate_availability_score(agent_profile, role_requirement)
            fitness_scores.append(("availability", availability_score, 0.15))

            # Calculate weighted average
            total_fitness = sum(score * weight for _, score, weight in fitness_scores)

            self.logger.debug(
                f"Role fitness for {agent_profile.agent_id} -> {role_requirement.role_type}: "
                f"{total_fitness:.3f} "
                f"(skill: {skill_score:.3f}, exp: {experience_score:.3f}, "
                f"perf: {performance_score:.3f}, avail: {availability_score:.3f})"
            )

            return total_fitness

        except Exception as e:
            self.logger.error(f"Error calculating role fitness: {e}")
            return 0.0

    def _calculate_skill_alignment(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate skill alignment score"""
        if not role_requirement.required_skills:
            return 0.8  # Default if no specific skills required

        total_weight = sum(role_requirement.required_skills.values())
        if total_weight == 0:
            return 0.8

        alignment_score = 0.0
        for skill_domain, required_level in role_requirement.required_skills.items():
            agent_level = agent_profile.get_capability_score(skill_domain)

            # Score based on how well agent meets requirement
            if agent_level >= required_level:
                skill_score = 1.0  # Meets or exceeds requirement
            else:
                skill_score = agent_level / required_level  # Partial match

            weight = required_level / total_weight
            alignment_score += skill_score * weight

        return min(1.0, alignment_score)

    def _calculate_experience_factor(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate experience factor score"""
        # Overall experience
        total_experience = sum(
            cap.experience_points for cap in agent_profile.capabilities.values()
        )

        experience_score = min(1.0, total_experience / max(1, role_requirement.min_experience_points))

        # Role-specific experience
        role_history_score = 0.0
        role_assignments = [
            assignment for assignment in agent_profile.role_assignments
            if assignment.get("role_type") == role_requirement.role_type
        ]

        if role_assignments:
            successful_assignments = len([
                a for a in role_assignments
                if a.get("success", False)
            ])
            role_history_score = successful_assignments / len(role_assignments)

        # Combine scores
        return (experience_score * 0.7 + role_history_score * 0.3)

    def _calculate_performance_score(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate performance-based score"""
        performance_factors = []

        # Overall success rate
        if agent_profile.overall_success_rate >= role_requirement.min_success_rate:
            success_score = 1.0
        else:
            success_score = agent_profile.overall_success_rate / role_requirement.min_success_rate

        performance_factors.append(success_score)

        # Response time factor
        if agent_profile.avg_task_completion_time_ms <= role_requirement.max_response_time_ms:
            time_score = 1.0
        else:
            time_score = role_requirement.max_response_time_ms / agent_profile.avg_task_completion_time_ms

        performance_factors.append(time_score)

        # Reliability score
        performance_factors.append(agent_profile.reliability_score)

        return sum(performance_factors) / len(performance_factors)

    def _calculate_availability_score(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate availability and workload score"""
        if agent_profile.availability_status != "available":
            return 0.0

        # Workload capacity
        workload_ratio = agent_profile.current_workload / max(1, agent_profile.max_concurrent_tasks)
        if workload_ratio >= 1.0:
            return 0.0  # Fully loaded

        # Higher score for lower workload
        workload_score = 1.0 - workload_ratio

        # Check if agent can handle the required workload
        if (agent_profile.current_workload + role_requirement.workload_capacity >
            agent_profile.max_concurrent_tasks):
            return 0.0

        return workload_score


class SkillBasedRoleSelection:
    """Role selection based on skill matching and development"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def recommend_optimal_roles(
        self,
        agent_profile: AgentProfile,
        available_roles: List[RoleRequirement]
    ) -> List[Tuple[RoleRequirement, float]]:
        """Recommend optimal roles for an agent"""
        try:
            role_recommendations = []

            for role_requirement in available_roles:
                # Calculate base fitness
                fitness_score = await self._calculate_skill_based_fitness(
                    agent_profile, role_requirement
                )

                # Apply growth potential bonus
                growth_bonus = self._calculate_growth_potential(
                    agent_profile, role_requirement
                )

                # Apply agent preferences
                preference_modifier = self._apply_agent_preferences(
                    agent_profile, role_requirement
                )

                final_score = fitness_score + growth_bonus + preference_modifier
                role_recommendations.append((role_requirement, final_score))

            # Sort by score (highest first)
            role_recommendations.sort(key=lambda x: x[1], reverse=True)

            return role_recommendations

        except Exception as e:
            self.logger.error(f"Error generating role recommendations: {e}")
            return []

    async def _calculate_skill_based_fitness(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate fitness based on skill matching"""
        if not role_requirement.required_skills:
            return 0.5

        skill_matches = []
        for skill_domain, required_level in role_requirement.required_skills.items():
            agent_capability = agent_profile.get_capability_score(skill_domain)

            # Calculate match quality
            if agent_capability >= required_level:
                match_score = 1.0
            else:
                # Partial credit for partial skills
                match_score = agent_capability / required_level

            skill_matches.append(match_score)

        # Return average skill match
        return sum(skill_matches) / len(skill_matches) if skill_matches else 0.0

    def _calculate_growth_potential(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Calculate growth potential bonus for skill development"""
        growth_bonus = 0.0

        for skill_domain, required_level in role_requirement.required_skills.items():
            agent_capability = agent_profile.get_capability_score(skill_domain)

            # Bonus for roles that provide growth opportunities
            if agent_capability < required_level:
                # Learning opportunity
                growth_potential = (required_level - agent_capability) * agent_profile.learning_rate
                growth_bonus += growth_potential * 0.1  # 10% bonus for growth
            elif agent_capability < 0.9:
                # Skill improvement opportunity
                improvement_potential = (0.9 - agent_capability) * agent_profile.learning_rate
                growth_bonus += improvement_potential * 0.05  # 5% bonus for improvement

        return min(0.2, growth_bonus)  # Cap at 20% bonus

    def _apply_agent_preferences(
        self,
        agent_profile: AgentProfile,
        role_requirement: RoleRequirement
    ) -> float:
        """Apply agent role preferences"""
        preference_modifier = 0.0

        # Preferred roles get a bonus
        if role_requirement.role_type in agent_profile.preferred_roles:
            preference_modifier += 0.1

        # Avoided roles get a penalty
        if role_requirement.role_type in agent_profile.avoided_roles:
            preference_modifier -= 0.2

        return preference_modifier


class RoleOptimizer:
    """Optimizes role assignments for maximum team performance"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_selector = PerformanceBasedRoleSelection()
        self.skill_selector = SkillBasedRoleSelection()

    async def optimize_team_composition(
        self,
        available_agents: List[AgentProfile],
        required_roles: List[RoleRequirement],
        optimization_strategy: str = "balanced"
    ) -> TeamComposition:
        """Optimize team composition for a set of required roles"""
        try:
            team = TeamComposition(name=f"Optimized Team {datetime.now().strftime('%Y%m%d_%H%M%S')}")

            if optimization_strategy == "greedy":
                assignments = await self._greedy_assignment(available_agents, required_roles)
            elif optimization_strategy == "balanced":
                assignments = await self._balanced_assignment(available_agents, required_roles)
            elif optimization_strategy == "skill_focused":
                assignments = await self._skill_focused_assignment(available_agents, required_roles)
            else:
                assignments = await self._balanced_assignment(available_agents, required_roles)

            team.role_assignments = assignments

            # Calculate team metrics
            await self._calculate_team_metrics(team, available_agents)

            return team

        except Exception as e:
            self.logger.error(f"Error optimizing team composition: {e}")
            return TeamComposition()

    async def _greedy_assignment(
        self,
        available_agents: List[AgentProfile],
        required_roles: List[RoleRequirement]
    ) -> List[RoleAssignment]:
        """Greedy assignment algorithm - assign best agent to each role"""
        assignments = []
        used_agents = set()

        # Sort roles by priority
        sorted_roles = sorted(required_roles, key=lambda r: r.priority_level, reverse=True)

        for role_requirement in sorted_roles:
            best_agent = None
            best_score = -1.0

            for agent in available_agents:
                if agent.agent_id in used_agents:
                    continue

                # Check basic requirements
                if not self._meets_basic_requirements(agent, role_requirement):
                    continue

                # Calculate fitness score
                fitness_score = await self.performance_selector.calculate_role_fitness(
                    agent, role_requirement
                )

                if fitness_score > best_score:
                    best_score = fitness_score
                    best_agent = agent

            if best_agent:
                assignment = RoleAssignment(
                    agent_id=best_agent.agent_id,
                    role_type=role_requirement.role_type,
                    assignment_score=best_score,
                    confidence_level=best_score,
                    expected_performance=best_score
                )
                assignments.append(assignment)
                used_agents.add(best_agent.agent_id)

        return assignments

    async def _balanced_assignment(
        self,
        available_agents: List[AgentProfile],
        required_roles: List[RoleRequirement]
    ) -> List[RoleAssignment]:
        """Balanced assignment considering overall team performance"""
        assignments = []

        # Create assignment matrix
        assignment_matrix = await self._create_assignment_matrix(available_agents, required_roles)

        # Use Hungarian algorithm concept (simplified)
        used_agents = set()
        used_roles = set()

        # Multiple passes to find optimal assignment
        for _ in range(len(required_roles)):
            best_assignment = None
            best_score = -1.0

            for i, agent in enumerate(available_agents):
                if agent.agent_id in used_agents:
                    continue

                for j, role in enumerate(required_roles):
                    if j in used_roles:
                        continue

                    if not self._meets_basic_requirements(agent, role):
                        continue

                    score = assignment_matrix[i][j]

                    # Apply team balance bonus
                    balance_bonus = self._calculate_team_balance_bonus(
                        assignments, agent, role
                    )

                    total_score = score + balance_bonus

                    if total_score > best_score:
                        best_score = total_score
                        best_assignment = (i, j, agent, role, score)

            if best_assignment:
                i, j, agent, role, score = best_assignment
                assignment = RoleAssignment(
                    agent_id=agent.agent_id,
                    role_type=role.role_type,
                    assignment_score=score,
                    confidence_level=score,
                    expected_performance=score
                )
                assignments.append(assignment)
                used_agents.add(agent.agent_id)
                used_roles.add(j)

        return assignments

    async def _skill_focused_assignment(
        self,
        available_agents: List[AgentProfile],
        required_roles: List[RoleRequirement]
    ) -> List[RoleAssignment]:
        """Assignment focused on maximizing skill utilization"""
        assignments = []
        used_agents = set()

        # Group roles by skill requirements
        skill_groups = self._group_roles_by_skills(required_roles)

        for skill_domain, roles_in_group in skill_groups.items():
            # Find agents with highest capability in this skill domain
            skilled_agents = [
                (agent, agent.get_capability_score(skill_domain))
                for agent in available_agents
                if agent.agent_id not in used_agents
            ]
            skilled_agents.sort(key=lambda x: x[1], reverse=True)

            # Assign roles in this skill group
            for role in roles_in_group:
                for agent, skill_score in skilled_agents:
                    if agent.agent_id in used_agents:
                        continue

                    if not self._meets_basic_requirements(agent, role):
                        continue

                    # Calculate full fitness score
                    fitness_score = await self.performance_selector.calculate_role_fitness(
                        agent, role
                    )

                    assignment = RoleAssignment(
                        agent_id=agent.agent_id,
                        role_type=role.role_type,
                        assignment_score=fitness_score,
                        confidence_level=fitness_score,
                        expected_performance=fitness_score
                    )
                    assignments.append(assignment)
                    used_agents.add(agent.agent_id)
                    break

        return assignments

    async def _create_assignment_matrix(
        self,
        agents: List[AgentProfile],
        roles: List[RoleRequirement]
    ) -> List[List[float]]:
        """Create assignment scoring matrix"""
        matrix = []

        for agent in agents:
            agent_scores = []
            for role in roles:
                if self._meets_basic_requirements(agent, role):
                    score = await self.performance_selector.calculate_role_fitness(agent, role)
                else:
                    score = 0.0
                agent_scores.append(score)
            matrix.append(agent_scores)

        return matrix

    def _meets_basic_requirements(
        self,
        agent: AgentProfile,
        role: RoleRequirement
    ) -> bool:
        """Check if agent meets basic requirements for role"""
        # Check agent type requirements
        if role.required_agent_types and agent.agent_type not in role.required_agent_types:
            return False

        # Check excluded agent types
        if role.excluded_agent_types and agent.agent_type in role.excluded_agent_types:
            return False

        # Check availability
        if agent.availability_status != "available":
            return False

        # Check workload capacity
        if (agent.current_workload + role.workload_capacity > agent.max_concurrent_tasks):
            return False

        # Check minimum success rate
        if agent.overall_success_rate < role.min_success_rate:
            return False

        return True

    def _calculate_team_balance_bonus(
        self,
        current_assignments: List[RoleAssignment],
        candidate_agent: AgentProfile,
        candidate_role: RoleRequirement
    ) -> float:
        """Calculate bonus for team balance"""
        bonus = 0.0

        # Skill diversity bonus
        if current_assignments:
            current_skills = set()
            for assignment in current_assignments:
                # Would need to look up agent profiles for assigned agents
                pass

        # Role diversity bonus
        current_roles = {assignment.role_type for assignment in current_assignments}
        if candidate_role.role_type not in current_roles:
            bonus += 0.05  # Bonus for role diversity

        return bonus

    def _group_roles_by_skills(
        self,
        roles: List[RoleRequirement]
    ) -> Dict[SkillDomain, List[RoleRequirement]]:
        """Group roles by their primary skill requirements"""
        skill_groups = {}

        for role in roles:
            if not role.required_skills:
                continue

            # Find primary skill (highest requirement)
            primary_skill = max(role.required_skills.items(), key=lambda x: x[1])[0]

            if primary_skill not in skill_groups:
                skill_groups[primary_skill] = []
            skill_groups[primary_skill].append(role)

        return skill_groups

    async def _calculate_team_metrics(
        self,
        team: TeamComposition,
        all_agents: List[AgentProfile]
    ):
        """Calculate team performance metrics"""
        if not team.role_assignments:
            return

        # Get agent profiles for assigned agents
        assigned_agents = {
            assignment.agent_id: next(
                (agent for agent in all_agents if agent.agent_id == assignment.agent_id),
                None
            )
            for assignment in team.role_assignments
        }

        # Calculate skill coverage
        skill_coverage = {}
        for skill_domain in SkillDomain:
            max_skill_level = 0.0
            for agent in assigned_agents.values():
                if agent:
                    skill_level = agent.get_capability_score(skill_domain)
                    max_skill_level = max(max_skill_level, skill_level)
            skill_coverage[skill_domain] = max_skill_level

        team.skill_coverage = skill_coverage

        # Calculate synergy score
        synergy_scores = []
        for assignment in team.role_assignments:
            synergy_scores.append(assignment.assignment_score)

        if synergy_scores:
            team.synergy_score = sum(synergy_scores) / len(synergy_scores)
            team.estimated_performance = team.synergy_score

        # Identify team lead (highest overall score)
        if team.role_assignments:
            best_assignment = max(team.role_assignments, key=lambda a: a.assignment_score)
            team.team_lead_agent_id = best_assignment.agent_id


class DynamicRoleManager:
    """Main manager for dynamic role assignments"""

    def __init__(self, agent_coordinator=None):
        self.agent_coordinator = agent_coordinator
        self.logger = logging.getLogger(__name__)

        # Core components
        self.role_optimizer = RoleOptimizer()
        self.performance_selector = PerformanceBasedRoleSelection()
        self.skill_selector = SkillBasedRoleSelection()

        # Agent profiles
        self.agent_profiles: Dict[str, AgentProfile] = {}

        # Active assignments
        self.active_assignments: Dict[str, RoleAssignment] = {}
        self.active_teams: Dict[str, TeamComposition] = {}

        # Role definitions
        self.role_definitions: Dict[RoleType, RoleRequirement] = {}

        # Performance tracking
        self.assignment_history: List[RoleAssignment] = []

    async def initialize(self, agent_ids: List[str]):
        """Initialize the dynamic role manager"""
        try:
            # Create agent profiles for all agents
            for agent_id in agent_ids:
                if agent_id not in self.agent_profiles:
                    # Get agent info from coordinator if available
                    agent_info = {}
                    if self.agent_coordinator:
                        agent_info = await self.agent_coordinator.get_agent_info(agent_id)

                    profile = AgentProfile(
                        agent_id=agent_id,
                        name=agent_info.get("name", f"Agent {agent_id}"),
                        agent_type=agent_info.get("type", "general")
                    )

                    # Initialize with basic capabilities
                    for skill_domain in SkillDomain:
                        profile.capabilities[skill_domain] = AgentCapability(
                            skill_domain=skill_domain,
                            proficiency_level=0.5  # Start with neutral proficiency
                        )

                    self.agent_profiles[agent_id] = profile

            # Initialize standard role definitions
            await self._initialize_standard_roles()

            self.logger.info(f"Dynamic Role Manager initialized with {len(self.agent_profiles)} agents")

        except Exception as e:
            self.logger.error(f"Failed to initialize Dynamic Role Manager: {e}")
            raise

    async def assign_role_to_agent(
        self,
        agent_id: str,
        role_type: RoleType,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> Optional[RoleAssignment]:
        """Assign a specific role to an agent"""
        try:
            agent_profile = self.agent_profiles.get(agent_id)
            if not agent_profile:
                raise Exception(f"Agent profile not found: {agent_id}")

            role_requirement = self.role_definitions.get(role_type)
            if not role_requirement:
                raise Exception(f"Role definition not found: {role_type}")

            # Check if agent meets requirements
            if not self.role_optimizer._meets_basic_requirements(agent_profile, role_requirement):
                return None

            # Calculate assignment score
            assignment_score = await self.performance_selector.calculate_role_fitness(
                agent_profile, role_requirement
            )

            # Create assignment
            assignment = RoleAssignment(
                agent_id=agent_id,
                role_type=role_type,
                task_id=task_id,
                workflow_id=workflow_id,
                assignment_score=assignment_score,
                confidence_level=assignment_score,
                expected_performance=assignment_score
            )

            # Update agent workload
            agent_profile.current_workload += role_requirement.workload_capacity

            # Store assignment
            self.active_assignments[assignment.assignment_id] = assignment
            agent_profile.role_assignments.append({
                "assignment_id": assignment.assignment_id,
                "role_type": role_type,
                "assigned_at": assignment.assigned_at,
                "task_id": task_id,
                "workflow_id": workflow_id
            })

            self.logger.info(f"Assigned role {role_type} to agent {agent_id} (score: {assignment_score:.3f})")

            return assignment

        except Exception as e:
            self.logger.error(f"Error assigning role: {e}")
            return None

    async def find_best_agent_for_role(
        self,
        role_type: RoleType,
        exclude_agents: List[str] = None
    ) -> Optional[str]:
        """Find the best available agent for a specific role"""
        try:
            role_requirement = self.role_definitions.get(role_type)
            if not role_requirement:
                return None

            exclude_agents = exclude_agents or []
            available_agents = [
                agent for agent in self.agent_profiles.values()
                if agent.agent_id not in exclude_agents and
                   agent.availability_status == "available"
            ]

            if not available_agents:
                return None

            best_agent = None
            best_score = -1.0

            for agent in available_agents:
                if not self.role_optimizer._meets_basic_requirements(agent, role_requirement):
                    continue

                fitness_score = await self.performance_selector.calculate_role_fitness(
                    agent, role_requirement
                )

                if fitness_score > best_score:
                    best_score = fitness_score
                    best_agent = agent

            return best_agent.agent_id if best_agent else None

        except Exception as e:
            self.logger.error(f"Error finding best agent for role: {e}")
            return None

    async def create_optimal_team(
        self,
        required_roles: List[RoleType],
        strategy: str = "balanced"
    ) -> Optional[TeamComposition]:
        """Create an optimal team composition for required roles"""
        try:
            # Convert role types to requirements
            role_requirements = []
            for role_type in required_roles:
                if role_type in self.role_definitions:
                    role_requirements.append(self.role_definitions[role_type])

            if not role_requirements:
                return None

            # Get available agents
            available_agents = [
                agent for agent in self.agent_profiles.values()
                if agent.availability_status == "available"
            ]

            # Optimize team composition
            team = await self.role_optimizer.optimize_team_composition(
                available_agents, role_requirements, strategy
            )

            if team.role_assignments:
                # Update agent workloads and store assignments
                for assignment in team.role_assignments:
                    agent_profile = self.agent_profiles.get(assignment.agent_id)
                    if agent_profile:
                        role_req = next(
                            (r for r in role_requirements if r.role_type == assignment.role_type),
                            None
                        )
                        if role_req:
                            agent_profile.current_workload += role_req.workload_capacity

                    self.active_assignments[assignment.assignment_id] = assignment

                # Store team
                self.active_teams[team.team_id] = team
                team.status = "active"

                self.logger.info(f"Created optimal team {team.team_id} with {len(team.role_assignments)} members")

            return team

        except Exception as e:
            self.logger.error(f"Error creating optimal team: {e}")
            return None

    async def update_agent_performance(
        self,
        assignment_id: str,
        task_result: Dict[str, Any]
    ):
        """Update agent performance based on task results"""
        try:
            assignment = self.active_assignments.get(assignment_id)
            if not assignment:
                return

            agent_profile = self.agent_profiles.get(assignment.agent_id)
            if not agent_profile:
                return

            # Update assignment performance
            assignment.actual_performance = task_result.get("performance_score", 0.5)
            assignment.task_results.append(task_result)

            if task_result.get("success", False):
                assignment.status = "completed"
            else:
                assignment.status = "failed"

            assignment.completed_at = datetime.utcnow()

            # Update agent profile
            agent_profile.total_tasks_completed += 1 if task_result.get("success", False) else 0
            agent_profile.total_tasks_failed += 0 if task_result.get("success", False) else 1

            # Update overall success rate
            total_tasks = agent_profile.total_tasks_completed + agent_profile.total_tasks_failed
            if total_tasks > 0:
                agent_profile.overall_success_rate = agent_profile.total_tasks_completed / total_tasks

            # Update capabilities based on role and performance
            role_requirement = self.role_definitions.get(assignment.role_type)
            if role_requirement and role_requirement.required_skills:
                for skill_domain in role_requirement.required_skills.keys():
                    agent_profile.update_capability(skill_domain, task_result)

            # Update availability
            role_req = self.role_definitions.get(assignment.role_type)
            if role_req:
                agent_profile.current_workload = max(
                    0, agent_profile.current_workload - role_req.workload_capacity
                )

            agent_profile.last_active = datetime.utcnow()

            # Move to history
            self.assignment_history.append(assignment)
            if assignment_id in self.active_assignments:
                del self.active_assignments[assignment_id]

            self.logger.info(f"Updated performance for agent {assignment.agent_id} in role {assignment.role_type}")

        except Exception as e:
            self.logger.error(f"Error updating agent performance: {e}")

    async def _initialize_standard_roles(self):
        """Initialize standard role definitions"""
        # Coordinator role
        self.role_definitions[RoleType.COORDINATOR] = RoleRequirement(
            role_type=RoleType.COORDINATOR,
            required_skills={
                SkillDomain.LEADERSHIP: 0.7,
                SkillDomain.COMMUNICATION: 0.8,
                SkillDomain.PROJECT_MANAGEMENT: 0.6
            },
            min_experience_points=100,
            min_success_rate=0.7,
            workload_capacity=2
        )

        # Specialist role
        self.role_definitions[RoleType.SPECIALIST] = RoleRequirement(
            role_type=RoleType.SPECIALIST,
            required_skills={
                SkillDomain.DOMAIN_EXPERTISE: 0.8,
                SkillDomain.TECHNICAL: 0.7
            },
            min_experience_points=150,
            min_success_rate=0.8
        )

        # Analyst role
        self.role_definitions[RoleType.ANALYST] = RoleRequirement(
            role_type=RoleType.ANALYST,
            required_skills={
                SkillDomain.ANALYTICAL: 0.8,
                SkillDomain.PROBLEM_SOLVING: 0.7,
                SkillDomain.RESEARCH: 0.6
            },
            min_experience_points=80,
            min_success_rate=0.75
        )

        # Creative role
        self.role_definitions[RoleType.CREATIVE] = RoleRequirement(
            role_type=RoleType.CREATIVE,
            required_skills={
                SkillDomain.CREATIVE: 0.8,
                SkillDomain.COMMUNICATION: 0.6
            },
            min_experience_points=50,
            min_success_rate=0.6
        )

        # Add more standard roles...
        # (Additional role definitions would be added here)

        self.logger.info(f"Initialized {len(self.role_definitions)} standard role definitions")

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile by ID"""
        return self.agent_profiles.get(agent_id)

    def get_active_assignments(self) -> List[RoleAssignment]:
        """Get all active role assignments"""
        return list(self.active_assignments.values())

    def get_team_composition(self, team_id: str) -> Optional[TeamComposition]:
        """Get team composition by ID"""
        return self.active_teams.get(team_id)

    def get_role_assignment_metrics(self) -> Dict[str, Any]:
        """Get metrics about role assignments"""
        try:
            total_assignments = len(self.assignment_history) + len(self.active_assignments)

            if not self.assignment_history:
                return {"total_assignments": total_assignments}

            completed_assignments = [a for a in self.assignment_history if a.status == "completed"]
            failed_assignments = [a for a in self.assignment_history if a.status == "failed"]

            success_rate = len(completed_assignments) / len(self.assignment_history)

            # Average performance
            performance_scores = [
                a.actual_performance for a in self.assignment_history
                if a.actual_performance is not None
            ]
            avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.5

            # Role distribution
            role_distribution = {}
            for assignment in self.assignment_history:
                role = assignment.role_type.value
                role_distribution[role] = role_distribution.get(role, 0) + 1

            return {
                "total_assignments": total_assignments,
                "active_assignments": len(self.active_assignments),
                "completed_assignments": len(completed_assignments),
                "failed_assignments": len(failed_assignments),
                "success_rate": success_rate,
                "avg_performance": avg_performance,
                "role_distribution": role_distribution,
                "active_teams": len(self.active_teams)
            }

        except Exception as e:
            self.logger.error(f"Error calculating assignment metrics: {e}")
            return {"error": str(e)}