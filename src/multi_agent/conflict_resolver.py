"""
Multi-Agent Conflict Resolution System

This module provides sophisticated conflict detection and resolution mechanisms
for managing competing agent actions, resource conflicts, and coordination disputes
in multi-agent collaborative environments.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from collections import defaultdict, deque
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    RESOURCE_CONTENTION = "resource_contention"     # Multiple agents want same resource
    CONTRADICTORY_ACTIONS = "contradictory_actions" # Agents propose conflicting actions
    PRIORITY_CONFLICT = "priority_conflict"         # Priority-based conflicts
    CONSENSUS_DEADLOCK = "consensus_deadlock"       # Cannot reach consensus
    TEMPORAL_CONFLICT = "temporal_conflict"         # Timing/sequencing conflicts
    ROLE_OVERLAP = "role_overlap"                  # Role boundary conflicts
    CONTEXT_DIVERGENCE = "context_divergence"       # Different context interpretations
    CAPABILITY_DISPUTE = "capability_dispute"       # Who should handle what

class ConflictSeverity(Enum):
    CRITICAL = "critical"      # System-breaking conflicts
    HIGH = "high"             # Major functionality impact
    MEDIUM = "medium"         # Moderate impact
    LOW = "low"              # Minor conflicts
    TRIVIAL = "trivial"       # Cosmetic/preference conflicts

class ResolutionStrategy(Enum):
    HIERARCHICAL = "hierarchical"           # Higher priority wins
    CONSENSUS = "consensus"                 # Seek agreement
    EXPERTISE_BASED = "expertise_based"     # Most qualified agent decides
    MAJORITY_VOTE = "majority_vote"         # Democratic resolution
    ROUND_ROBIN = "round_robin"            # Take turns
    COMPROMISE = "compromise"               # Find middle ground
    ESCALATION = "escalation"              # Escalate to human/system
    RANDOM = "random"                      # Random selection
    PERFORMANCE_BASED = "performance_based" # Best performing agent wins

class ConflictStatus(Enum):
    DETECTED = "detected"
    ANALYZING = "analyzing"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"

@dataclass
class ConflictParticipant:
    """Represents an agent involved in a conflict"""
    agent_id: str
    position: Dict[str, Any]  # Agent's stance/proposal
    priority: int = 0
    confidence: float = 0.5
    expertise_score: float = 0.5
    performance_history: float = 0.5
    resources_claimed: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Conflict:
    """Represents a detected conflict between agents"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    participants: List[ConflictParticipant]
    resource_ids: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    status: ConflictStatus = ConflictStatus.DETECTED
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolution_result: Optional[Dict[str, Any]] = None
    escalation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration(self) -> Optional[timedelta]:
        if self.resolved_at:
            return self.resolved_at - self.detected_at
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "participants": [
                {
                    "agent_id": p.agent_id,
                    "position": p.position,
                    "priority": p.priority,
                    "confidence": p.confidence,
                    "expertise_score": p.expertise_score,
                    "performance_history": p.performance_history,
                    "resources_claimed": list(p.resources_claimed),
                    "metadata": p.metadata
                }
                for p in self.participants
            ],
            "resource_ids": list(self.resource_ids),
            "context": self.context,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "status": self.status.value,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "resolution_result": self.resolution_result,
            "escalation_reason": self.escalation_reason,
            "metadata": self.metadata
        }

@dataclass
class ConflictDetectionRule:
    """Defines rules for detecting specific types of conflicts"""
    rule_id: str
    conflict_type: ConflictType
    detection_function: Callable[[Dict[str, Any]], bool]
    severity_function: Callable[[Dict[str, Any]], ConflictSeverity]
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResolutionRule:
    """Defines rules for resolving specific types of conflicts"""
    rule_id: str
    applicable_types: Set[ConflictType]
    strategy: ResolutionStrategy
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    success_rate: float = 0.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConflictResolver:
    """
    Advanced conflict resolution system for multi-agent coordination.

    Provides sophisticated conflict detection, analysis, and resolution
    mechanisms with multiple strategies and escalation paths.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.active_conflicts: Dict[str, Conflict] = {}
        self.conflict_history: deque = deque(maxlen=10000)
        self.detection_rules: Dict[str, ConflictDetectionRule] = {}
        self.resolution_rules: Dict[str, ResolutionRule] = {}
        self.resolution_callbacks: Dict[ResolutionStrategy, Callable] = {}
        self.escalation_callbacks: List[Callable] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(int)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Initialize default detection and resolution rules
        self._initialize_default_rules()
        self._initialize_resolution_strategies()

    def _initialize_default_rules(self):
        """Initialize default conflict detection rules"""

        # Resource contention detection
        def detect_resource_contention(context: Dict[str, Any]) -> bool:
            resource_claims = context.get("resource_claims", {})
            for resource_id, claimants in resource_claims.items():
                if len(claimants) > 1:
                    return True
            return False

        def resource_severity(context: Dict[str, Any]) -> ConflictSeverity:
            max_claimants = max(
                len(claimants) for claimants in context.get("resource_claims", {}).values()
            ) if context.get("resource_claims") else 0
            if max_claimants > 5:
                return ConflictSeverity.CRITICAL
            elif max_claimants > 3:
                return ConflictSeverity.HIGH
            elif max_claimants > 2:
                return ConflictSeverity.MEDIUM
            return ConflictSeverity.LOW

        self.add_detection_rule(ConflictDetectionRule(
            rule_id="resource_contention_basic",
            conflict_type=ConflictType.RESOURCE_CONTENTION,
            detection_function=detect_resource_contention,
            severity_function=resource_severity,
            priority=10
        ))

        # Contradictory actions detection
        def detect_contradictory_actions(context: Dict[str, Any]) -> bool:
            proposed_actions = context.get("proposed_actions", [])
            if len(proposed_actions) < 2:
                return False

            # Check for direct contradictions
            action_types = [action.get("type") for action in proposed_actions]
            contradictory_pairs = [
                ("create", "delete"),
                ("enable", "disable"),
                ("start", "stop"),
                ("increase", "decrease")
            ]

            for action1, action2 in contradictory_pairs:
                if action1 in action_types and action2 in action_types:
                    return True
            return False

        def action_severity(context: Dict[str, Any]) -> ConflictSeverity:
            critical_actions = {"delete", "stop", "disable", "terminate"}
            proposed_actions = context.get("proposed_actions", [])

            for action in proposed_actions:
                if action.get("type") in critical_actions:
                    return ConflictSeverity.HIGH
            return ConflictSeverity.MEDIUM

        self.add_detection_rule(ConflictDetectionRule(
            rule_id="contradictory_actions_basic",
            conflict_type=ConflictType.CONTRADICTORY_ACTIONS,
            detection_function=detect_contradictory_actions,
            severity_function=action_severity,
            priority=9
        ))

        # Priority conflict detection
        def detect_priority_conflict(context: Dict[str, Any]) -> bool:
            agent_priorities = context.get("agent_priorities", {})
            if len(agent_priorities) < 2:
                return False

            # Check if multiple agents have the same high priority
            priorities = list(agent_priorities.values())
            high_priority_count = sum(1 for p in priorities if p >= 8)
            return high_priority_count > 1

        def priority_severity(context: Dict[str, Any]) -> ConflictSeverity:
            agent_priorities = context.get("agent_priorities", {})
            max_priority = max(agent_priorities.values()) if agent_priorities else 0

            if max_priority >= 9:
                return ConflictSeverity.HIGH
            elif max_priority >= 7:
                return ConflictSeverity.MEDIUM
            return ConflictSeverity.LOW

        self.add_detection_rule(ConflictDetectionRule(
            rule_id="priority_conflict_basic",
            conflict_type=ConflictType.PRIORITY_CONFLICT,
            detection_function=detect_priority_conflict,
            severity_function=priority_severity,
            priority=8
        ))

    def _initialize_resolution_strategies(self):
        """Initialize resolution strategy implementations"""

        self.resolution_callbacks[ResolutionStrategy.HIERARCHICAL] = self._resolve_hierarchical
        self.resolution_callbacks[ResolutionStrategy.CONSENSUS] = self._resolve_consensus
        self.resolution_callbacks[ResolutionStrategy.EXPERTISE_BASED] = self._resolve_expertise_based
        self.resolution_callbacks[ResolutionStrategy.MAJORITY_VOTE] = self._resolve_majority_vote
        self.resolution_callbacks[ResolutionStrategy.ROUND_ROBIN] = self._resolve_round_robin
        self.resolution_callbacks[ResolutionStrategy.COMPROMISE] = self._resolve_compromise
        self.resolution_callbacks[ResolutionStrategy.PERFORMANCE_BASED] = self._resolve_performance_based
        self.resolution_callbacks[ResolutionStrategy.RANDOM] = self._resolve_random

        # Add default resolution rules
        self.add_resolution_rule(ResolutionRule(
            rule_id="resource_contention_hierarchical",
            applicable_types={ConflictType.RESOURCE_CONTENTION},
            strategy=ResolutionStrategy.HIERARCHICAL,
            priority=10
        ))

        self.add_resolution_rule(ResolutionRule(
            rule_id="contradictory_actions_expertise",
            applicable_types={ConflictType.CONTRADICTORY_ACTIONS},
            strategy=ResolutionStrategy.EXPERTISE_BASED,
            priority=9
        ))

        self.add_resolution_rule(ResolutionRule(
            rule_id="priority_conflict_consensus",
            applicable_types={ConflictType.PRIORITY_CONFLICT},
            strategy=ResolutionStrategy.CONSENSUS,
            priority=8
        ))

    def add_detection_rule(self, rule: ConflictDetectionRule):
        """Add a conflict detection rule"""
        self.detection_rules[rule.rule_id] = rule
        logger.info(f"Added conflict detection rule: {rule.rule_id}")

    def add_resolution_rule(self, rule: ResolutionRule):
        """Add a conflict resolution rule"""
        self.resolution_rules[rule.rule_id] = rule
        logger.info(f"Added conflict resolution rule: {rule.rule_id}")

    def add_escalation_callback(self, callback: Callable[[Conflict], None]):
        """Add a callback for conflict escalation"""
        self.escalation_callbacks.append(callback)

    async def detect_conflicts(self, context: Dict[str, Any]) -> List[Conflict]:
        """
        Detect conflicts in the given context using all enabled detection rules.

        Args:
            context: Context containing agent actions, resource claims, etc.

        Returns:
            List of detected conflicts
        """
        detected_conflicts = []

        # Sort rules by priority (higher first)
        sorted_rules = sorted(
            [rule for rule in self.detection_rules.values() if rule.enabled],
            key=lambda r: r.priority,
            reverse=True
        )

        for rule in sorted_rules:
            try:
                if rule.detection_function(context):
                    severity = rule.severity_function(context)
                    conflict = await self._create_conflict(
                        conflict_type=rule.conflict_type,
                        severity=severity,
                        context=context
                    )
                    detected_conflicts.append(conflict)
                    logger.info(f"Detected {rule.conflict_type.value} conflict with {severity.value} severity")

            except Exception as e:
                logger.error(f"Error in detection rule {rule.rule_id}: {e}")

        return detected_conflicts

    async def _create_conflict(
        self,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        context: Dict[str, Any]
    ) -> Conflict:
        """Create a conflict object from detection context"""

        conflict_id = str(uuid.uuid4())
        participants = []
        resource_ids = set()

        # Extract participants from context
        agent_data = context.get("agents", {})
        for agent_id, agent_info in agent_data.items():
            participant = ConflictParticipant(
                agent_id=agent_id,
                position=agent_info.get("position", {}),
                priority=agent_info.get("priority", 0),
                confidence=agent_info.get("confidence", 0.5),
                expertise_score=agent_info.get("expertise_score", 0.5),
                performance_history=self.agent_performance.get(agent_id, {}).get("average", 0.5),
                resources_claimed=set(agent_info.get("resources_claimed", []))
            )
            participants.append(participant)
            resource_ids.update(participant.resources_claimed)

        conflict = Conflict(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            severity=severity,
            participants=participants,
            resource_ids=resource_ids,
            context=context
        )

        self.active_conflicts[conflict_id] = conflict
        self.performance_metrics["conflicts_detected"] += 1

        return conflict

    async def resolve_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """
        Resolve a conflict using the most appropriate strategy.

        Args:
            conflict: The conflict to resolve

        Returns:
            Resolution result containing winner, action, and metadata
        """
        try:
            conflict.status = ConflictStatus.ANALYZING

            # Find applicable resolution rules
            applicable_rules = [
                rule for rule in self.resolution_rules.values()
                if conflict.conflict_type in rule.applicable_types and rule.enabled
            ]

            if not applicable_rules:
                logger.warning(f"No resolution rules found for conflict type {conflict.conflict_type}")
                return await self._escalate_conflict(conflict, "No applicable resolution rules")

            # Sort by priority and success rate
            applicable_rules.sort(
                key=lambda r: (r.priority, r.success_rate),
                reverse=True
            )

            conflict.status = ConflictStatus.RESOLVING

            # Try resolution strategies in order
            for rule in applicable_rules:
                try:
                    strategy_func = self.resolution_callbacks.get(rule.strategy)
                    if not strategy_func:
                        continue

                    conflict.resolution_strategy = rule.strategy
                    result = await strategy_func(conflict)

                    if result and result.get("success"):
                        conflict.status = ConflictStatus.RESOLVED
                        conflict.resolved_at = datetime.now()
                        conflict.resolution_result = result

                        # Update performance metrics
                        self.performance_metrics["conflicts_resolved"] += 1
                        self.performance_metrics[f"strategy_{rule.strategy.value}_success"] += 1
                        rule.success_rate = self._calculate_success_rate(rule)

                        # Move to history
                        self.conflict_history.append(conflict)
                        if conflict.conflict_id in self.active_conflicts:
                            del self.active_conflicts[conflict.conflict_id]

                        logger.info(f"Resolved conflict {conflict.conflict_id} using {rule.strategy.value}")
                        return result

                except Exception as e:
                    logger.error(f"Error resolving conflict with strategy {rule.strategy}: {e}")
                    continue

            # If all strategies failed, escalate
            return await self._escalate_conflict(conflict, "All resolution strategies failed")

        except Exception as e:
            logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
            conflict.status = ConflictStatus.FAILED
            return {"success": False, "error": str(e)}

    async def _resolve_hierarchical(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict based on agent hierarchy/priority"""
        if not conflict.participants:
            return {"success": False, "reason": "No participants"}

        # Find participant with highest priority
        winner = max(conflict.participants, key=lambda p: p.priority)

        return {
            "success": True,
            "strategy": "hierarchical",
            "winner_id": winner.agent_id,
            "reason": f"Agent {winner.agent_id} has highest priority ({winner.priority})",
            "action": winner.position,
            "losers": [p.agent_id for p in conflict.participants if p.agent_id != winner.agent_id]
        }

    async def _resolve_expertise_based(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict based on agent expertise"""
        if not conflict.participants:
            return {"success": False, "reason": "No participants"}

        # Find participant with highest expertise score
        winner = max(conflict.participants, key=lambda p: p.expertise_score)

        return {
            "success": True,
            "strategy": "expertise_based",
            "winner_id": winner.agent_id,
            "reason": f"Agent {winner.agent_id} has highest expertise ({winner.expertise_score:.2f})",
            "action": winner.position,
            "losers": [p.agent_id for p in conflict.participants if p.agent_id != winner.agent_id]
        }

    async def _resolve_performance_based(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict based on agent performance history"""
        if not conflict.participants:
            return {"success": False, "reason": "No participants"}

        # Find participant with best performance history
        winner = max(conflict.participants, key=lambda p: p.performance_history)

        return {
            "success": True,
            "strategy": "performance_based",
            "winner_id": winner.agent_id,
            "reason": f"Agent {winner.agent_id} has best performance ({winner.performance_history:.2f})",
            "action": winner.position,
            "losers": [p.agent_id for p in conflict.participants if p.agent_id != winner.agent_id]
        }

    async def _resolve_consensus(self, conflict: Conflict) -> Dict[str, Any]:
        """Attempt to resolve conflict through consensus"""
        if len(conflict.participants) < 2:
            return {"success": False, "reason": "Not enough participants for consensus"}

        # Simple consensus: find common ground in positions
        common_elements = {}
        all_positions = [p.position for p in conflict.participants]

        # Find shared keys and values
        if all_positions:
            first_position = all_positions[0]
            for key, value in first_position.items():
                if all(pos.get(key) == value for pos in all_positions[1:]):
                    common_elements[key] = value

        if common_elements:
            return {
                "success": True,
                "strategy": "consensus",
                "action": common_elements,
                "reason": "Found consensus on common elements",
                "participants": [p.agent_id for p in conflict.participants]
            }

        return {"success": False, "reason": "No consensus reached"}

    async def _resolve_majority_vote(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict through majority voting"""
        if len(conflict.participants) < 3:
            return {"success": False, "reason": "Not enough participants for majority vote"}

        # Group participants by similar positions
        position_groups = defaultdict(list)
        for participant in conflict.participants:
            position_key = json.dumps(participant.position, sort_keys=True)
            position_groups[position_key].append(participant)

        # Find majority group
        majority_group = max(position_groups.values(), key=len)
        majority_size = len(majority_group)
        total_participants = len(conflict.participants)

        if majority_size > total_participants / 2:
            representative = majority_group[0]
            return {
                "success": True,
                "strategy": "majority_vote",
                "winner_ids": [p.agent_id for p in majority_group],
                "action": representative.position,
                "reason": f"Majority vote: {majority_size}/{total_participants} participants",
                "vote_distribution": {k: len(v) for k, v in position_groups.items()}
            }

        return {"success": False, "reason": "No clear majority"}

    async def _resolve_round_robin(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict by round-robin selection"""
        if not conflict.participants:
            return {"success": False, "reason": "No participants"}

        # Use a simple hash of conflict ID to ensure deterministic but fair selection
        selection_index = hash(conflict.conflict_id) % len(conflict.participants)
        winner = conflict.participants[selection_index]

        return {
            "success": True,
            "strategy": "round_robin",
            "winner_id": winner.agent_id,
            "action": winner.position,
            "reason": f"Round-robin selection (index {selection_index})",
            "rotation_order": [p.agent_id for p in conflict.participants]
        }

    async def _resolve_compromise(self, conflict: Conflict) -> Dict[str, Any]:
        """Attempt to find a compromise solution"""
        if len(conflict.participants) < 2:
            return {"success": False, "reason": "Not enough participants for compromise"}

        # Average numerical values, combine sets, etc.
        compromise_position = {}
        all_positions = [p.position for p in conflict.participants]

        # Find all unique keys
        all_keys = set()
        for position in all_positions:
            all_keys.update(position.keys())

        for key in all_keys:
            values = [pos.get(key) for pos in all_positions if key in pos]

            if not values:
                continue

            # Handle different value types
            if all(isinstance(v, (int, float)) for v in values):
                # Average numerical values
                compromise_position[key] = sum(values) / len(values)
            elif all(isinstance(v, bool) for v in values):
                # Majority vote for booleans
                compromise_position[key] = sum(values) > len(values) / 2
            elif all(isinstance(v, str) for v in values):
                # Use most common string
                from collections import Counter
                counter = Counter(values)
                compromise_position[key] = counter.most_common(1)[0][0]
            elif all(isinstance(v, (list, set)) for v in values):
                # Union of collections
                combined = set()
                for v in values:
                    combined.update(v)
                compromise_position[key] = list(combined)

        if compromise_position:
            return {
                "success": True,
                "strategy": "compromise",
                "action": compromise_position,
                "reason": "Compromise solution found",
                "participants": [p.agent_id for p in conflict.participants]
            }

        return {"success": False, "reason": "No compromise possible"}

    async def _resolve_random(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve conflict through random selection"""
        if not conflict.participants:
            return {"success": False, "reason": "No participants"}

        import random
        winner = random.choice(conflict.participants)

        return {
            "success": True,
            "strategy": "random",
            "winner_id": winner.agent_id,
            "action": winner.position,
            "reason": "Random selection",
            "participants": [p.agent_id for p in conflict.participants]
        }

    async def _escalate_conflict(self, conflict: Conflict, reason: str) -> Dict[str, Any]:
        """Escalate conflict to higher-level resolution"""
        conflict.status = ConflictStatus.ESCALATED
        conflict.escalation_reason = reason

        # Notify escalation callbacks
        for callback in self.escalation_callbacks:
            try:
                await callback(conflict) if asyncio.iscoroutinefunction(callback) else callback(conflict)
            except Exception as e:
                logger.error(f"Error in escalation callback: {e}")

        self.performance_metrics["conflicts_escalated"] += 1

        return {
            "success": False,
            "escalated": True,
            "reason": reason,
            "conflict_id": conflict.conflict_id,
            "requires_human_intervention": True
        }

    def _calculate_success_rate(self, rule: ResolutionRule) -> float:
        """Calculate success rate for a resolution rule"""
        strategy_successes = self.performance_metrics.get(f"strategy_{rule.strategy.value}_success", 0)
        strategy_attempts = self.performance_metrics.get(f"strategy_{rule.strategy.value}_attempts", 1)
        return strategy_successes / strategy_attempts

    async def get_conflict_metrics(self) -> Dict[str, Any]:
        """Get comprehensive conflict resolution metrics"""
        active_count = len(self.active_conflicts)
        history_count = len(self.conflict_history)

        # Calculate success rates by strategy
        strategy_stats = {}
        for strategy in ResolutionStrategy:
            successes = self.performance_metrics.get(f"strategy_{strategy.value}_success", 0)
            attempts = self.performance_metrics.get(f"strategy_{strategy.value}_attempts", 0)
            strategy_stats[strategy.value] = {
                "successes": successes,
                "attempts": attempts,
                "success_rate": successes / attempts if attempts > 0 else 0.0
            }

        # Calculate conflict type distribution
        type_distribution = defaultdict(int)
        for conflict in list(self.active_conflicts.values()) + list(self.conflict_history):
            type_distribution[conflict.conflict_type.value] += 1

        # Calculate average resolution time
        resolved_conflicts = [c for c in self.conflict_history if c.resolved_at]
        avg_resolution_time = None
        if resolved_conflicts:
            total_time = sum(c.duration().total_seconds() for c in resolved_conflicts)
            avg_resolution_time = total_time / len(resolved_conflicts)

        return {
            "active_conflicts": active_count,
            "resolved_conflicts": history_count,
            "total_detected": self.performance_metrics.get("conflicts_detected", 0),
            "total_resolved": self.performance_metrics.get("conflicts_resolved", 0),
            "total_escalated": self.performance_metrics.get("conflicts_escalated", 0),
            "success_rate": (
                self.performance_metrics.get("conflicts_resolved", 0) /
                max(1, self.performance_metrics.get("conflicts_detected", 1))
            ),
            "strategy_performance": strategy_stats,
            "conflict_type_distribution": dict(type_distribution),
            "average_resolution_time_seconds": avg_resolution_time,
            "detection_rules_count": len(self.detection_rules),
            "resolution_rules_count": len(self.resolution_rules)
        }

    async def cleanup_old_conflicts(self, max_age_hours: int = 24):
        """Clean up old resolved conflicts from active tracking"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        to_remove = []
        for conflict_id, conflict in self.active_conflicts.items():
            if conflict.resolved_at and conflict.resolved_at < cutoff_time:
                to_remove.append(conflict_id)

        for conflict_id in to_remove:
            conflict = self.active_conflicts.pop(conflict_id)
            self.conflict_history.append(conflict)

        logger.info(f"Cleaned up {len(to_remove)} old conflicts")
        return len(to_remove)