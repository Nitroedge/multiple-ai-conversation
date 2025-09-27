"""
Character-specific memory structures and management
Implements agent-specific memory schemas with personality-aware storage
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid

from ..memory.models import MemoryItem, MemoryType
from ..personality.big_five_model import PersonalityProfile, PersonalityTrait


class CharacterMemoryType(Enum):
    """Extended memory types specific to character development"""
    PERSONALITY_TRAIT = "personality_trait"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    CONVERSATION_STYLE = "conversation_style"
    RELATIONSHIP_MEMORY = "relationship_memory"
    GROWTH_MILESTONE = "growth_milestone"
    PREFERENCE_LEARNED = "preference_learned"
    INTERACTION_PATTERN = "interaction_pattern"
    EMOTIONAL_ASSOCIATION = "emotional_association"


@dataclass
class CharacterTraitMemory:
    """Memory of personality trait manifestation in conversation"""
    trait: PersonalityTrait
    manifestation: str  # How the trait was expressed
    conversation_context: str
    strength_evidence: float  # 0.0 to 1.0
    timestamp: datetime
    session_id: str
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "trait": self.trait.value,
            "manifestation": self.manifestation,
            "conversation_context": self.conversation_context,
            "strength_evidence": self.strength_evidence,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterTraitMemory':
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            trait=PersonalityTrait(data["trait"]),
            manifestation=data["manifestation"],
            conversation_context=data["conversation_context"],
            strength_evidence=data["strength_evidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"]
        )


@dataclass
class RelationshipMemory:
    """Memory of relationships and interactions with users/other agents"""
    relationship_id: str
    relationship_type: str  # "user", "agent", "group"
    relationship_name: str
    relationship_quality: float  # -1.0 (negative) to 1.0 (positive)
    interaction_count: int
    last_interaction: datetime

    # Relationship characteristics
    trust_level: float  # 0.0 to 1.0
    familiarity_level: float  # 0.0 to 1.0
    emotional_bond: float  # 0.0 to 1.0

    # Memory details
    memorable_moments: List[str] = field(default_factory=list)
    shared_interests: List[str] = field(default_factory=list)
    conversation_topics: List[str] = field(default_factory=list)

    # Behavioral patterns in this relationship
    typical_interaction_style: str = ""
    preferred_communication_style: str = ""

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_relationship(self, interaction_quality: float, interaction_context: str) -> None:
        """Update relationship based on new interaction"""
        # Update relationship quality with momentum
        momentum = 0.1
        self.relationship_quality = (
            self.relationship_quality * (1 - momentum) +
            interaction_quality * momentum
        )

        # Increase familiarity
        self.familiarity_level = min(1.0, self.familiarity_level + 0.01)

        # Update trust based on positive interactions
        if interaction_quality > 0.5:
            self.trust_level = min(1.0, self.trust_level + 0.02)
        elif interaction_quality < 0.3:
            self.trust_level = max(0.0, self.trust_level - 0.01)

        # Update emotional bond for very positive interactions
        if interaction_quality > 0.8:
            self.emotional_bond = min(1.0, self.emotional_bond + 0.05)

        self.interaction_count += 1
        self.last_interaction = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Add to memorable moments if significant
        if interaction_quality > 0.8 or interaction_quality < 0.2:
            if len(self.memorable_moments) >= 10:
                self.memorable_moments.pop(0)  # Remove oldest
            self.memorable_moments.append(f"{datetime.utcnow().isoformat()}: {interaction_context}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "relationship_id": self.relationship_id,
            "relationship_type": self.relationship_type,
            "relationship_name": self.relationship_name,
            "relationship_quality": self.relationship_quality,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat(),
            "trust_level": self.trust_level,
            "familiarity_level": self.familiarity_level,
            "emotional_bond": self.emotional_bond,
            "memorable_moments": self.memorable_moments,
            "shared_interests": self.shared_interests,
            "conversation_topics": self.conversation_topics,
            "typical_interaction_style": self.typical_interaction_style,
            "preferred_communication_style": self.preferred_communication_style,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipMemory':
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            relationship_id=data["relationship_id"],
            relationship_type=data["relationship_type"],
            relationship_name=data["relationship_name"],
            relationship_quality=data["relationship_quality"],
            interaction_count=data["interaction_count"],
            last_interaction=datetime.fromisoformat(data["last_interaction"]),
            trust_level=data["trust_level"],
            familiarity_level=data["familiarity_level"],
            emotional_bond=data["emotional_bond"],
            memorable_moments=data.get("memorable_moments", []),
            shared_interests=data.get("shared_interests", []),
            conversation_topics=data.get("conversation_topics", []),
            typical_interaction_style=data.get("typical_interaction_style", ""),
            preferred_communication_style=data.get("preferred_communication_style", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


@dataclass
class BehavioralPattern:
    """Learned behavioral patterns and preferences"""
    pattern_id: str
    pattern_name: str
    pattern_type: str  # "communication", "topic_preference", "interaction_style", etc.
    pattern_description: str

    # Pattern strength and confidence
    occurrence_count: int
    confidence_score: float  # 0.0 to 1.0
    last_observed: datetime

    # Context and triggers
    typical_contexts: List[str] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)

    # Pattern metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def reinforce_pattern(self, context: str = "") -> None:
        """Reinforce this behavioral pattern with new evidence"""
        self.occurrence_count += 1
        self.confidence_score = min(1.0, self.confidence_score + 0.05)
        self.last_observed = datetime.utcnow()

        if context and context not in self.typical_contexts:
            self.typical_contexts.append(context)
            if len(self.typical_contexts) > 5:
                self.typical_contexts.pop(0)  # Keep most recent 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type,
            "pattern_description": self.pattern_description,
            "occurrence_count": self.occurrence_count,
            "confidence_score": self.confidence_score,
            "last_observed": self.last_observed.isoformat(),
            "typical_contexts": self.typical_contexts,
            "trigger_conditions": self.trigger_conditions,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BehavioralPattern':
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            pattern_id=data["pattern_id"],
            pattern_name=data["pattern_name"],
            pattern_type=data["pattern_type"],
            pattern_description=data["pattern_description"],
            occurrence_count=data["occurrence_count"],
            confidence_score=data["confidence_score"],
            last_observed=datetime.fromisoformat(data["last_observed"]),
            typical_contexts=data.get("typical_contexts", []),
            trigger_conditions=data.get("trigger_conditions", []),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class CharacterGrowthMilestone:
    """Significant character development milestones"""
    milestone_id: str
    milestone_type: str  # "personality_shift", "new_skill", "relationship_deepened", etc.
    description: str
    significance_level: float  # 0.0 to 1.0

    # Changes recorded
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]

    # Context
    trigger_event: str
    session_id: str

    timestamp: datetime = field(default_factory=datetime.utcnow)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "milestone_id": self.milestone_id,
            "milestone_type": self.milestone_type,
            "description": self.description,
            "significance_level": self.significance_level,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "trigger_event": self.trigger_event,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterGrowthMilestone':
        return cls(
            memory_id=data.get("memory_id", str(uuid.uuid4())),
            milestone_id=data["milestone_id"],
            milestone_type=data["milestone_type"],
            description=data["description"],
            significance_level=data["significance_level"],
            before_state=data["before_state"],
            after_state=data["after_state"],
            trigger_event=data["trigger_event"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class CharacterMemoryManager:
    """Manages character-specific memories and personality development"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Character memory collections
        self.trait_memories: Dict[str, CharacterTraitMemory] = {}
        self.relationship_memories: Dict[str, RelationshipMemory] = {}
        self.behavioral_patterns: Dict[str, BehavioralPattern] = {}
        self.growth_milestones: Dict[str, CharacterGrowthMilestone] = {}

        # Memory analytics
        self.memory_stats = {
            "total_trait_memories": 0,
            "total_relationships": 0,
            "total_patterns": 0,
            "total_milestones": 0,
            "character_evolution_score": 0.0,
            "last_significant_growth": None
        }

    def store_trait_memory(self, trait_memory: CharacterTraitMemory) -> str:
        """Store a personality trait manifestation memory"""
        self.trait_memories[trait_memory.memory_id] = trait_memory
        self.memory_stats["total_trait_memories"] += 1

        # Check for behavioral pattern
        self._analyze_for_patterns(trait_memory)

        return trait_memory.memory_id

    def store_relationship_memory(self, relationship_memory: RelationshipMemory) -> str:
        """Store or update a relationship memory"""
        existing_rel = self.get_relationship_by_id(relationship_memory.relationship_id)

        if existing_rel:
            # Update existing relationship
            existing_rel.update_relationship(
                relationship_memory.relationship_quality,
                f"Interaction at {relationship_memory.updated_at}"
            )
        else:
            self.relationship_memories[relationship_memory.memory_id] = relationship_memory
            self.memory_stats["total_relationships"] += 1

        return relationship_memory.memory_id

    def store_behavioral_pattern(self, pattern: BehavioralPattern) -> str:
        """Store or reinforce a behavioral pattern"""
        existing_pattern = self.get_pattern_by_id(pattern.pattern_id)

        if existing_pattern:
            existing_pattern.reinforce_pattern()
        else:
            self.behavioral_patterns[pattern.memory_id] = pattern
            self.memory_stats["total_patterns"] += 1

        return pattern.memory_id

    def store_growth_milestone(self, milestone: CharacterGrowthMilestone) -> str:
        """Store a character growth milestone"""
        self.growth_milestones[milestone.memory_id] = milestone
        self.memory_stats["total_milestones"] += 1
        self.memory_stats["last_significant_growth"] = milestone.timestamp.isoformat()

        # Update character evolution score
        self._update_evolution_score(milestone.significance_level)

        return milestone.memory_id

    def get_relationship_by_id(self, relationship_id: str) -> Optional[RelationshipMemory]:
        """Get relationship memory by relationship ID"""
        for memory in self.relationship_memories.values():
            if memory.relationship_id == relationship_id:
                return memory
        return None

    def get_pattern_by_id(self, pattern_id: str) -> Optional[BehavioralPattern]:
        """Get behavioral pattern by pattern ID"""
        for pattern in self.behavioral_patterns.values():
            if pattern.pattern_id == pattern_id:
                return pattern
        return None

    def get_trait_memories_for_trait(self, trait: PersonalityTrait, limit: int = 10) -> List[CharacterTraitMemory]:
        """Get recent trait memories for a specific personality trait"""
        trait_memories = [
            memory for memory in self.trait_memories.values()
            if memory.trait == trait
        ]

        # Sort by timestamp (most recent first)
        trait_memories.sort(key=lambda x: x.timestamp, reverse=True)

        return trait_memories[:limit]

    def get_relationship_summary(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive relationship summary"""
        relationship = self.get_relationship_by_id(relationship_id)
        if not relationship:
            return None

        return {
            "relationship_overview": relationship.to_dict(),
            "relationship_strength": self._calculate_relationship_strength(relationship),
            "interaction_history": self._get_recent_interactions(relationship_id),
            "growth_potential": self._assess_relationship_growth_potential(relationship),
            "recommended_actions": self._generate_relationship_recommendations(relationship)
        }

    def get_behavioral_insights(self) -> Dict[str, Any]:
        """Get insights about learned behavioral patterns"""
        patterns_by_type = {}
        for pattern in self.behavioral_patterns.values():
            if pattern.pattern_type not in patterns_by_type:
                patterns_by_type[pattern.pattern_type] = []
            patterns_by_type[pattern.pattern_type].append(pattern)

        return {
            "patterns_by_type": {
                ptype: [p.to_dict() for p in patterns]
                for ptype, patterns in patterns_by_type.items()
            },
            "most_confident_patterns": self._get_most_confident_patterns(),
            "emerging_patterns": self._get_emerging_patterns(),
            "behavioral_consistency": self._calculate_behavioral_consistency()
        }

    def get_character_development_summary(self) -> Dict[str, Any]:
        """Get comprehensive character development summary"""
        return {
            "development_stats": self.memory_stats,
            "personality_evolution": self._analyze_personality_evolution(),
            "relationship_development": self._analyze_relationship_development(),
            "behavioral_maturity": self._calculate_behavioral_maturity(),
            "growth_trajectory": self._analyze_growth_trajectory(),
            "recent_milestones": self._get_recent_milestones(5)
        }

    def _analyze_for_patterns(self, trait_memory: CharacterTraitMemory) -> None:
        """Analyze trait memory for emerging behavioral patterns"""
        # Simple pattern detection based on repeated trait manifestations
        similar_memories = self.get_trait_memories_for_trait(trait_memory.trait, 5)

        if len(similar_memories) >= 3:
            # Look for consistent manifestation patterns
            manifestations = [m.manifestation for m in similar_memories]

            # Simple similarity check (could be enhanced with NLP)
            common_words = set()
            for manifestation in manifestations:
                words = set(manifestation.lower().split())
                if not common_words:
                    common_words = words
                else:
                    common_words &= words

            if len(common_words) >= 2:  # Found pattern
                pattern_id = f"{trait_memory.trait.value}_pattern_{len(common_words)}"

                if not self.get_pattern_by_id(pattern_id):
                    pattern = BehavioralPattern(
                        pattern_id=pattern_id,
                        pattern_name=f"{trait_memory.trait.value.title()} Expression Pattern",
                        pattern_type="trait_manifestation",
                        pattern_description=f"Consistent way of expressing {trait_memory.trait.value}",
                        occurrence_count=len(similar_memories),
                        confidence_score=min(1.0, len(similar_memories) * 0.2),
                        last_observed=trait_memory.timestamp,
                        typical_contexts=[m.conversation_context for m in similar_memories[-3:]]
                    )
                    self.store_behavioral_pattern(pattern)

    def _update_evolution_score(self, milestone_significance: float) -> None:
        """Update character evolution score based on milestones"""
        current_score = self.memory_stats["character_evolution_score"]

        # Weighted average with decay for older milestones
        decay_factor = 0.9
        new_contribution = milestone_significance * 0.1

        self.memory_stats["character_evolution_score"] = (
            current_score * decay_factor + new_contribution
        )

    def _calculate_relationship_strength(self, relationship: RelationshipMemory) -> float:
        """Calculate overall relationship strength"""
        # Weighted combination of relationship factors
        weights = {
            "quality": 0.3,
            "trust": 0.25,
            "familiarity": 0.2,
            "emotional_bond": 0.15,
            "interaction_frequency": 0.1
        }

        # Calculate interaction frequency score
        days_since_last = (datetime.utcnow() - relationship.last_interaction).days
        frequency_score = max(0.0, 1.0 - (days_since_last / 30.0))  # Decay over 30 days

        strength = (
            relationship.relationship_quality * weights["quality"] +
            relationship.trust_level * weights["trust"] +
            relationship.familiarity_level * weights["familiarity"] +
            relationship.emotional_bond * weights["emotional_bond"] +
            frequency_score * weights["interaction_frequency"]
        )

        return max(0.0, min(1.0, strength))

    def _get_recent_interactions(self, relationship_id: str, limit: int = 5) -> List[str]:
        """Get recent interactions for a relationship"""
        relationship = self.get_relationship_by_id(relationship_id)
        if not relationship:
            return []

        return relationship.memorable_moments[-limit:]

    def _assess_relationship_growth_potential(self, relationship: RelationshipMemory) -> str:
        """Assess potential for relationship growth"""
        if relationship.trust_level > 0.8 and relationship.emotional_bond > 0.7:
            return "high"
        elif relationship.familiarity_level > 0.6 and relationship.relationship_quality > 0.6:
            return "moderate"
        elif relationship.interaction_count < 5:
            return "early_stage"
        else:
            return "limited"

    def _generate_relationship_recommendations(self, relationship: RelationshipMemory) -> List[str]:
        """Generate recommendations for relationship development"""
        recommendations = []

        if relationship.trust_level < 0.5:
            recommendations.append("Focus on building trust through consistent, honest interactions")

        if relationship.emotional_bond < 0.5 and relationship.familiarity_level > 0.6:
            recommendations.append("Explore deeper emotional connections and shared experiences")

        if len(relationship.shared_interests) < 3:
            recommendations.append("Discover and cultivate shared interests")

        if relationship.interaction_count < 10:
            recommendations.append("Increase interaction frequency to build familiarity")

        return recommendations

    def _get_most_confident_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most confident behavioral patterns"""
        patterns = list(self.behavioral_patterns.values())
        patterns.sort(key=lambda x: x.confidence_score, reverse=True)

        return [p.to_dict() for p in patterns[:limit]]

    def _get_emerging_patterns(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get emerging behavioral patterns (recent, low confidence)"""
        recent_threshold = datetime.utcnow() - timedelta(days=7)

        emerging = [
            p for p in self.behavioral_patterns.values()
            if p.last_observed > recent_threshold and p.confidence_score < 0.5
        ]

        emerging.sort(key=lambda x: x.last_observed, reverse=True)

        return [p.to_dict() for p in emerging[:limit]]

    def _calculate_behavioral_consistency(self) -> float:
        """Calculate how consistent the agent's behavior patterns are"""
        if not self.behavioral_patterns:
            return 0.0

        # Calculate average confidence across all patterns
        total_confidence = sum(p.confidence_score for p in self.behavioral_patterns.values())
        average_confidence = total_confidence / len(self.behavioral_patterns)

        # Factor in pattern reinforcement frequency
        total_occurrences = sum(p.occurrence_count for p in self.behavioral_patterns.values())
        pattern_density = total_occurrences / max(len(self.behavioral_patterns), 1)

        # Normalize pattern density (assume good consistency at 10+ occurrences per pattern)
        density_score = min(1.0, pattern_density / 10.0)

        # Weighted combination
        consistency = (average_confidence * 0.7) + (density_score * 0.3)

        return consistency

    def _analyze_personality_evolution(self) -> Dict[str, Any]:
        """Analyze how personality has evolved over time"""
        trait_evolution = {}

        for trait in PersonalityTrait:
            memories = self.get_trait_memories_for_trait(trait, 20)
            if len(memories) >= 3:
                # Analyze trend in trait strength over time
                recent_strength = sum(m.strength_evidence for m in memories[:5]) / 5
                older_strength = sum(m.strength_evidence for m in memories[-5:]) / min(5, len(memories[-5:]))

                trend = recent_strength - older_strength

                trait_evolution[trait.value] = {
                    "trend": "increasing" if trend > 0.1 else "decreasing" if trend < -0.1 else "stable",
                    "trend_magnitude": abs(trend),
                    "recent_average": recent_strength,
                    "sample_size": len(memories)
                }

        return trait_evolution

    def _analyze_relationship_development(self) -> Dict[str, Any]:
        """Analyze relationship development patterns"""
        if not self.relationship_memories:
            return {"status": "no_relationships"}

        total_relationships = len(self.relationship_memories)
        strong_relationships = len([
            r for r in self.relationship_memories.values()
            if self._calculate_relationship_strength(r) > 0.7
        ])

        average_quality = sum(
            r.relationship_quality for r in self.relationship_memories.values()
        ) / total_relationships

        return {
            "total_relationships": total_relationships,
            "strong_relationships": strong_relationships,
            "relationship_strength_ratio": strong_relationships / total_relationships,
            "average_relationship_quality": average_quality,
            "relationship_types": self._count_relationship_types()
        }

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count relationships by type"""
        type_counts = {}
        for relationship in self.relationship_memories.values():
            rtype = relationship.relationship_type
            type_counts[rtype] = type_counts.get(rtype, 0) + 1

        return type_counts

    def _calculate_behavioral_maturity(self) -> float:
        """Calculate behavioral maturity score"""
        if not self.behavioral_patterns:
            return 0.0

        # Factors: pattern diversity, confidence, consistency
        pattern_types = set(p.pattern_type for p in self.behavioral_patterns.values())
        diversity_score = min(1.0, len(pattern_types) / 5.0)  # Assume 5 types is mature

        consistency_score = self._calculate_behavioral_consistency()

        # Average pattern age (older patterns indicate stability)
        pattern_ages = [
            (datetime.utcnow() - p.created_at).days
            for p in self.behavioral_patterns.values()
        ]
        average_age = sum(pattern_ages) / len(pattern_ages) if pattern_ages else 0
        age_score = min(1.0, average_age / 30.0)  # 30 days = mature

        # Weighted combination
        maturity = (
            diversity_score * 0.4 +
            consistency_score * 0.4 +
            age_score * 0.2
        )

        return maturity

    def _analyze_growth_trajectory(self) -> Dict[str, Any]:
        """Analyze character growth trajectory"""
        if not self.growth_milestones:
            return {"status": "no_growth_data"}

        milestones = list(self.growth_milestones.values())
        milestones.sort(key=lambda x: x.timestamp)

        # Calculate growth velocity (milestones per time period)
        if len(milestones) >= 2:
            time_span = (milestones[-1].timestamp - milestones[0].timestamp).days
            growth_velocity = len(milestones) / max(time_span, 1)
        else:
            growth_velocity = 0.0

        # Average significance of milestones
        average_significance = sum(m.significance_level for m in milestones) / len(milestones)

        return {
            "total_milestones": len(milestones),
            "growth_velocity": growth_velocity,  # milestones per day
            "average_significance": average_significance,
            "growth_trend": self._determine_growth_trend(milestones),
            "most_significant_milestone": max(milestones, key=lambda x: x.significance_level).to_dict()
        }

    def _determine_growth_trend(self, milestones: List[CharacterGrowthMilestone]) -> str:
        """Determine overall growth trend"""
        if len(milestones) < 3:
            return "insufficient_data"

        # Compare recent vs older milestone significance
        recent_avg = sum(m.significance_level for m in milestones[-3:]) / 3
        older_avg = sum(m.significance_level for m in milestones[:3]) / 3

        if recent_avg > older_avg + 0.1:
            return "accelerating"
        elif recent_avg < older_avg - 0.1:
            return "decelerating"
        else:
            return "steady"

    def _get_recent_milestones(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent growth milestones"""
        milestones = list(self.growth_milestones.values())
        milestones.sort(key=lambda x: x.timestamp, reverse=True)

        return [m.to_dict() for m in milestones[:limit]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire character memory to dictionary"""
        return {
            "agent_id": self.agent_id,
            "trait_memories": {
                mid: memory.to_dict()
                for mid, memory in self.trait_memories.items()
            },
            "relationship_memories": {
                mid: memory.to_dict()
                for mid, memory in self.relationship_memories.items()
            },
            "behavioral_patterns": {
                mid: pattern.to_dict()
                for mid, pattern in self.behavioral_patterns.items()
            },
            "growth_milestones": {
                mid: milestone.to_dict()
                for mid, milestone in self.growth_milestones.items()
            },
            "memory_stats": self.memory_stats
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterMemoryManager':
        """Create CharacterMemoryManager from dictionary"""
        manager = cls(data["agent_id"])

        # Load trait memories
        for mid, memory_data in data.get("trait_memories", {}).items():
            manager.trait_memories[mid] = CharacterTraitMemory.from_dict(memory_data)

        # Load relationship memories
        for mid, memory_data in data.get("relationship_memories", {}).items():
            manager.relationship_memories[mid] = RelationshipMemory.from_dict(memory_data)

        # Load behavioral patterns
        for mid, pattern_data in data.get("behavioral_patterns", {}).items():
            manager.behavioral_patterns[mid] = BehavioralPattern.from_dict(pattern_data)

        # Load growth milestones
        for mid, milestone_data in data.get("growth_milestones", {}).items():
            manager.growth_milestones[mid] = CharacterGrowthMilestone.from_dict(milestone_data)

        # Load stats
        manager.memory_stats = data.get("memory_stats", manager.memory_stats)

        return manager