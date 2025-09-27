"""
Character adaptation algorithms for dynamic personality evolution
Implements learning algorithms that allow characters to grow and adapt based on interactions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import math
import statistics
from datetime import datetime, timedelta

from ..personality.big_five_model import PersonalityProfile, PersonalityTrait, PersonalityEngine
from ..character.character_memory import CharacterMemoryManager, BehavioralPattern, CharacterGrowthMilestone
from ..emotion.emotion_tracking import EmotionTracker, EmotionType, EmotionalState


class AdaptationStrategy(Enum):
    """Different adaptation strategies for character development"""
    GRADUAL_DRIFT = "gradual_drift"  # Slow, continuous adaptation
    EXPERIENCE_BASED = "experience_based"  # Adaptation based on significant experiences
    SOCIAL_MIRRORING = "social_mirroring"  # Adapt to match interaction partners
    CRISIS_CATALYZED = "crisis_catalyzed"  # Rapid adaptation during emotional crises
    GROWTH_ORIENTED = "growth_oriented"  # Adaptation toward psychological growth
    STABLE_CORE = "stable_core"  # Minimal adaptation, maintain core personality


class AdaptationTrigger(Enum):
    """Events that can trigger personality adaptation"""
    REPEATED_INTERACTION = "repeated_interaction"
    EMOTIONAL_PEAK = "emotional_peak"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    BEHAVIORAL_FEEDBACK = "behavioral_feedback"
    TEMPORAL_DRIFT = "temporal_drift"
    CRISIS_EVENT = "crisis_event"
    LEARNING_MOMENT = "learning_moment"
    SOCIAL_PRESSURE = "social_pressure"


@dataclass
class AdaptationEvent:
    """Individual adaptation event with context and impact"""
    event_id: str
    trigger: AdaptationTrigger
    strategy: AdaptationStrategy
    personality_changes: Dict[PersonalityTrait, float]
    confidence: float  # 0.0 to 1.0
    impact_magnitude: float  # 0.0 to 1.0

    # Context
    session_id: str
    interaction_context: str
    emotional_context: Optional[Dict[str, Any]] = None
    social_context: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    adaptation_rationale: str = ""
    success_probability: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trigger": self.trigger.value,
            "strategy": self.strategy.value,
            "personality_changes": {trait.value: change for trait, change in self.personality_changes.items()},
            "confidence": self.confidence,
            "impact_magnitude": self.impact_magnitude,
            "session_id": self.session_id,
            "interaction_context": self.interaction_context,
            "emotional_context": self.emotional_context,
            "social_context": self.social_context,
            "timestamp": self.timestamp.isoformat(),
            "adaptation_rationale": self.adaptation_rationale,
            "success_probability": self.success_probability
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptationEvent':
        return cls(
            event_id=data["event_id"],
            trigger=AdaptationTrigger(data["trigger"]),
            strategy=AdaptationStrategy(data["strategy"]),
            personality_changes={
                PersonalityTrait(trait): change
                for trait, change in data["personality_changes"].items()
            },
            confidence=data["confidence"],
            impact_magnitude=data["impact_magnitude"],
            session_id=data["session_id"],
            interaction_context=data["interaction_context"],
            emotional_context=data.get("emotional_context"),
            social_context=data.get("social_context"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            adaptation_rationale=data.get("adaptation_rationale", ""),
            success_probability=data.get("success_probability", 0.5)
        )


@dataclass
class AdaptationConfiguration:
    """Configuration for character adaptation behavior"""
    agent_id: str

    # Adaptation sensitivity
    adaptation_rate: float = 0.1  # How quickly to adapt (0.0 to 1.0)
    emotional_sensitivity: float = 0.5  # How much emotions influence adaptation
    social_sensitivity: float = 0.3  # How much social feedback influences adaptation

    # Stability factors
    core_stability: float = 0.8  # How resistant core traits are to change
    temporal_decay: float = 0.05  # How much adaptation effects decay over time

    # Adaptation boundaries
    max_trait_change_per_event: float = 0.05  # Maximum change per single event
    max_cumulative_change: float = 0.3  # Maximum total change from baseline

    # Active strategies
    enabled_strategies: Set[AdaptationStrategy] = field(default_factory=lambda: {
        AdaptationStrategy.GRADUAL_DRIFT,
        AdaptationStrategy.EXPERIENCE_BASED,
        AdaptationStrategy.GROWTH_ORIENTED
    })

    # Adaptation triggers
    trigger_thresholds: Dict[AdaptationTrigger, float] = field(default_factory=lambda: {
        AdaptationTrigger.REPEATED_INTERACTION: 0.7,
        AdaptationTrigger.EMOTIONAL_PEAK: 0.8,
        AdaptationTrigger.RELATIONSHIP_MILESTONE: 0.6,
        AdaptationTrigger.BEHAVIORAL_FEEDBACK: 0.5,
        AdaptationTrigger.TEMPORAL_DRIFT: 0.3,
        AdaptationTrigger.CRISIS_EVENT: 0.9,
        AdaptationTrigger.LEARNING_MOMENT: 0.6
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "adaptation_rate": self.adaptation_rate,
            "emotional_sensitivity": self.emotional_sensitivity,
            "social_sensitivity": self.social_sensitivity,
            "core_stability": self.core_stability,
            "temporal_decay": self.temporal_decay,
            "max_trait_change_per_event": self.max_trait_change_per_event,
            "max_cumulative_change": self.max_cumulative_change,
            "enabled_strategies": [s.value for s in self.enabled_strategies],
            "trigger_thresholds": {t.value: threshold for t, threshold in self.trigger_thresholds.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptationConfiguration':
        return cls(
            agent_id=data["agent_id"],
            adaptation_rate=data.get("adaptation_rate", 0.1),
            emotional_sensitivity=data.get("emotional_sensitivity", 0.5),
            social_sensitivity=data.get("social_sensitivity", 0.3),
            core_stability=data.get("core_stability", 0.8),
            temporal_decay=data.get("temporal_decay", 0.05),
            max_trait_change_per_event=data.get("max_trait_change_per_event", 0.05),
            max_cumulative_change=data.get("max_cumulative_change", 0.3),
            enabled_strategies={
                AdaptationStrategy(s) for s in data.get("enabled_strategies", [])
            },
            trigger_thresholds={
                AdaptationTrigger(t): threshold
                for t, threshold in data.get("trigger_thresholds", {}).items()
            }
        )


class CharacterAdaptationEngine:
    """Main engine for managing character adaptation and personality evolution"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.configuration = AdaptationConfiguration(agent_id)

        # Adaptation history
        self.adaptation_events: List[AdaptationEvent] = []
        self.baseline_personality: Optional[PersonalityProfile] = None
        self.adaptation_statistics = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "average_impact": 0.0,
            "dominant_strategies": [],
            "trait_change_history": {trait: [] for trait in PersonalityTrait}
        }

    def set_baseline_personality(self, personality_profile: PersonalityProfile) -> None:
        """Set the baseline personality profile for adaptation tracking"""
        self.baseline_personality = personality_profile.copy() if hasattr(personality_profile, 'copy') else personality_profile

    def configure_adaptation(self, config: AdaptationConfiguration) -> None:
        """Update adaptation configuration"""
        self.configuration = config

    def evaluate_adaptation_triggers(
        self,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        emotion_tracker: EmotionTracker,
        interaction_context: Dict[str, Any]
    ) -> List[AdaptationEvent]:
        """Evaluate potential adaptation triggers and generate adaptation events"""

        potential_events = []

        # Check each trigger type
        for trigger in AdaptationTrigger:
            if trigger in self.configuration.trigger_thresholds:
                trigger_strength = self._evaluate_trigger_strength(
                    trigger, personality_profile, character_memory, emotion_tracker, interaction_context
                )

                if trigger_strength >= self.configuration.trigger_thresholds[trigger]:
                    # Generate adaptation event
                    event = self._generate_adaptation_event(
                        trigger, trigger_strength, personality_profile,
                        character_memory, emotion_tracker, interaction_context
                    )
                    if event:
                        potential_events.append(event)

        return potential_events

    def apply_adaptation_events(
        self,
        events: List[AdaptationEvent],
        personality_profile: PersonalityProfile
    ) -> PersonalityProfile:
        """Apply adaptation events to personality profile"""

        if not events:
            return personality_profile

        # Sort events by impact magnitude (apply strongest first)
        events.sort(key=lambda e: e.impact_magnitude, reverse=True)

        total_changes = {trait: 0.0 for trait in PersonalityTrait}

        for event in events:
            # Check if this adaptation should be applied
            if self._should_apply_adaptation(event, total_changes):
                # Apply personality changes
                for trait, change in event.personality_changes.items():
                    # Apply bounded change
                    bounded_change = self._bound_trait_change(trait, change, total_changes[trait])

                    if abs(bounded_change) > 0.001:  # Only apply significant changes
                        current_score = personality_profile.get_trait_score(trait)
                        new_score = max(0.0, min(1.0, current_score + bounded_change))

                        # Update personality score
                        personality_profile.traits[trait].update_score(new_score, bounded_change)
                        total_changes[trait] += bounded_change

                        # Record change in statistics
                        self.adaptation_statistics["trait_change_history"][trait].append({
                            "timestamp": event.timestamp.isoformat(),
                            "change": bounded_change,
                            "trigger": event.trigger.value,
                            "strategy": event.strategy.value
                        })

                # Store successful adaptation
                self.adaptation_events.append(event)
                self.adaptation_statistics["successful_adaptations"] += 1

                # Create growth milestone for significant adaptations
                if event.impact_magnitude > 0.7:
                    milestone = self._create_growth_milestone(event, total_changes)
                    # This would be stored in character memory

        # Update adaptation statistics
        self._update_adaptation_statistics()

        return personality_profile

    def get_adaptation_insights(self) -> Dict[str, Any]:
        """Get insights about character adaptation patterns"""
        return {
            "adaptation_statistics": self.adaptation_statistics,
            "configuration": self.configuration.to_dict(),
            "recent_adaptations": [
                event.to_dict() for event in self.adaptation_events[-10:]
            ],
            "trait_evolution_summary": self._analyze_trait_evolution(),
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(),
            "growth_trajectory": self._analyze_growth_trajectory()
        }

    def _evaluate_trigger_strength(
        self,
        trigger: AdaptationTrigger,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        emotion_tracker: EmotionTracker,
        interaction_context: Dict[str, Any]
    ) -> float:
        """Evaluate the strength of a specific adaptation trigger"""

        if trigger == AdaptationTrigger.REPEATED_INTERACTION:
            return self._evaluate_repeated_interaction_trigger(character_memory, interaction_context)

        elif trigger == AdaptationTrigger.EMOTIONAL_PEAK:
            return self._evaluate_emotional_peak_trigger(emotion_tracker)

        elif trigger == AdaptationTrigger.RELATIONSHIP_MILESTONE:
            return self._evaluate_relationship_milestone_trigger(character_memory, interaction_context)

        elif trigger == AdaptationTrigger.BEHAVIORAL_FEEDBACK:
            return self._evaluate_behavioral_feedback_trigger(character_memory, interaction_context)

        elif trigger == AdaptationTrigger.TEMPORAL_DRIFT:
            return self._evaluate_temporal_drift_trigger(personality_profile)

        elif trigger == AdaptationTrigger.CRISIS_EVENT:
            return self._evaluate_crisis_event_trigger(emotion_tracker, interaction_context)

        elif trigger == AdaptationTrigger.LEARNING_MOMENT:
            return self._evaluate_learning_moment_trigger(character_memory, interaction_context)

        else:
            return 0.0

    def _evaluate_repeated_interaction_trigger(
        self,
        character_memory: CharacterMemoryManager,
        interaction_context: Dict[str, Any]
    ) -> float:
        """Evaluate repeated interaction patterns"""
        behavioral_insights = character_memory.get_behavioral_insights()

        # Look for strong, consistent patterns
        confident_patterns = [
            p for p in behavioral_insights.get("most_confident_patterns", [])
            if p.get("confidence_score", 0) > 0.7
        ]

        if confident_patterns:
            # Strong patterns suggest potential for adaptation
            max_confidence = max(p.get("confidence_score", 0) for p in confident_patterns)
            return max_confidence

        return 0.0

    def _evaluate_emotional_peak_trigger(self, emotion_tracker: EmotionTracker) -> float:
        """Evaluate emotional peak events"""
        current_state = emotion_tracker.get_current_emotional_state()

        if not current_state:
            return 0.0

        # High intensity emotions can trigger adaptation
        intensity = current_state.primary_intensity

        # Extreme emotional states (very high or low arousal) can be triggers
        arousal_factor = abs(current_state.arousal)

        # Combine intensity and arousal
        trigger_strength = (intensity + arousal_factor) / 2.0

        return min(1.0, trigger_strength)

    def _evaluate_relationship_milestone_trigger(
        self,
        character_memory: CharacterMemoryManager,
        interaction_context: Dict[str, Any]
    ) -> float:
        """Evaluate relationship milestone events"""
        user_id = interaction_context.get("user_id")
        if not user_id:
            return 0.0

        relationship = character_memory.get_relationship_by_id(user_id)
        if not relationship:
            return 0.0

        # Check for significant relationship developments
        trust_level = relationship.trust_level
        emotional_bond = relationship.emotional_bond
        interaction_count = relationship.interaction_count

        # Milestones: high trust, strong bond, many interactions
        milestone_factors = []

        if trust_level > 0.8:
            milestone_factors.append(trust_level)

        if emotional_bond > 0.7:
            milestone_factors.append(emotional_bond)

        if interaction_count > 50 and interaction_count % 25 == 0:  # Every 25 interactions after 50
            milestone_factors.append(0.6)

        return max(milestone_factors) if milestone_factors else 0.0

    def _evaluate_behavioral_feedback_trigger(
        self,
        character_memory: CharacterMemoryManager,
        interaction_context: Dict[str, Any]
    ) -> float:
        """Evaluate behavioral feedback from interactions"""
        # Look for feedback indicators in interaction context
        user_satisfaction = interaction_context.get("user_satisfaction", 0.5)
        conversation_quality = interaction_context.get("conversation_quality", 0.5)

        # Extreme feedback (very positive or negative) can trigger adaptation
        feedback_extremity = max(
            abs(user_satisfaction - 0.5),
            abs(conversation_quality - 0.5)
        ) * 2  # Scale to 0-1

        return feedback_extremity

    def _evaluate_temporal_drift_trigger(self, personality_profile: PersonalityProfile) -> float:
        """Evaluate natural temporal drift"""
        # Check time since last adaptation
        if not self.adaptation_events:
            return 0.3  # Moderate trigger for initial adaptation

        last_adaptation = max(self.adaptation_events, key=lambda e: e.timestamp)
        time_since_last = datetime.utcnow() - last_adaptation.timestamp

        # Gradual increase in trigger strength over time
        days_since = time_since_last.days
        temporal_strength = min(1.0, days_since / 30.0)  # Peak at 30 days

        return temporal_strength * 0.4  # Moderate strength for temporal drift

    def _evaluate_crisis_event_trigger(
        self,
        emotion_tracker: EmotionTracker,
        interaction_context: Dict[str, Any]
    ) -> float:
        """Evaluate crisis or significant emotional events"""
        current_state = emotion_tracker.get_current_emotional_state()

        if not current_state:
            return 0.0

        # Crisis indicators
        crisis_factors = []

        # Very high intensity negative emotions
        if current_state.primary_intensity > 0.8:
            negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR]
            if current_state.primary_emotion in negative_emotions:
                crisis_factors.append(current_state.primary_intensity)

        # Very low emotional valence (very negative)
        if current_state.valence < -0.7:
            crisis_factors.append(abs(current_state.valence))

        # Context indicators
        crisis_keywords = ["crisis", "emergency", "serious", "urgent", "problem", "trouble"]
        context_text = interaction_context.get("user_input", "").lower()

        if any(keyword in context_text for keyword in crisis_keywords):
            crisis_factors.append(0.7)

        return max(crisis_factors) if crisis_factors else 0.0

    def _evaluate_learning_moment_trigger(
        self,
        character_memory: CharacterMemoryManager,
        interaction_context: Dict[str, Any]
    ) -> float:
        """Evaluate learning moments and insights"""
        # Look for learning indicators
        learning_keywords = [
            "learn", "understand", "realize", "discover", "insight",
            "new", "different", "perspective", "teach", "explain"
        ]

        context_text = interaction_context.get("user_input", "").lower()
        learning_indicators = sum(1 for keyword in learning_keywords if keyword in context_text)

        # Question asking indicates curiosity and learning
        question_count = context_text.count("?")

        # Combine indicators
        learning_strength = min(1.0, (learning_indicators * 0.2) + (question_count * 0.3))

        return learning_strength

    def _generate_adaptation_event(
        self,
        trigger: AdaptationTrigger,
        trigger_strength: float,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        emotion_tracker: EmotionTracker,
        interaction_context: Dict[str, Any]
    ) -> Optional[AdaptationEvent]:
        """Generate an adaptation event based on trigger and context"""

        # Select adaptation strategy
        strategy = self._select_adaptation_strategy(trigger, personality_profile, emotion_tracker)

        if strategy not in self.configuration.enabled_strategies:
            return None

        # Calculate personality changes based on strategy and trigger
        personality_changes = self._calculate_personality_changes(
            strategy, trigger, trigger_strength, personality_profile,
            character_memory, emotion_tracker, interaction_context
        )

        if not personality_changes:
            return None

        # Calculate event impact and confidence
        impact_magnitude = max(abs(change) for change in personality_changes.values())
        confidence = min(1.0, trigger_strength * 0.8 + 0.2)  # Base confidence of 0.2

        # Create adaptation event
        event = AdaptationEvent(
            event_id=f"adapt_{self.agent_id}_{datetime.utcnow().timestamp()}",
            trigger=trigger,
            strategy=strategy,
            personality_changes=personality_changes,
            confidence=confidence,
            impact_magnitude=impact_magnitude,
            session_id=interaction_context.get("session_id", "unknown"),
            interaction_context=interaction_context.get("user_input", "")[:200],
            emotional_context=emotion_tracker.get_current_emotional_state().to_dict() if emotion_tracker.get_current_emotional_state() else None,
            adaptation_rationale=self._generate_adaptation_rationale(strategy, trigger, personality_changes)
        )

        return event

    def _select_adaptation_strategy(
        self,
        trigger: AdaptationTrigger,
        personality_profile: PersonalityProfile,
        emotion_tracker: EmotionTracker
    ) -> AdaptationStrategy:
        """Select appropriate adaptation strategy based on context"""

        # Map triggers to preferred strategies
        trigger_strategy_map = {
            AdaptationTrigger.REPEATED_INTERACTION: AdaptationStrategy.EXPERIENCE_BASED,
            AdaptationTrigger.EMOTIONAL_PEAK: AdaptationStrategy.CRISIS_CATALYZED,
            AdaptationTrigger.RELATIONSHIP_MILESTONE: AdaptationStrategy.SOCIAL_MIRRORING,
            AdaptationTrigger.BEHAVIORAL_FEEDBACK: AdaptationStrategy.EXPERIENCE_BASED,
            AdaptationTrigger.TEMPORAL_DRIFT: AdaptationStrategy.GRADUAL_DRIFT,
            AdaptationTrigger.CRISIS_EVENT: AdaptationStrategy.CRISIS_CATALYZED,
            AdaptationTrigger.LEARNING_MOMENT: AdaptationStrategy.GROWTH_ORIENTED
        }

        preferred_strategy = trigger_strategy_map.get(trigger, AdaptationStrategy.GRADUAL_DRIFT)

        # Check if preferred strategy is enabled
        if preferred_strategy in self.configuration.enabled_strategies:
            return preferred_strategy

        # Fallback to first enabled strategy
        return list(self.configuration.enabled_strategies)[0] if self.configuration.enabled_strategies else AdaptationStrategy.GRADUAL_DRIFT

    def _calculate_personality_changes(
        self,
        strategy: AdaptationStrategy,
        trigger: AdaptationTrigger,
        trigger_strength: float,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        emotion_tracker: EmotionTracker,
        interaction_context: Dict[str, Any]
    ) -> Dict[PersonalityTrait, float]:
        """Calculate specific personality trait changes"""

        changes = {}

        if strategy == AdaptationStrategy.GRADUAL_DRIFT:
            changes = self._calculate_gradual_drift_changes(personality_profile, trigger_strength)

        elif strategy == AdaptationStrategy.EXPERIENCE_BASED:
            changes = self._calculate_experience_based_changes(
                personality_profile, character_memory, trigger_strength
            )

        elif strategy == AdaptationStrategy.SOCIAL_MIRRORING:
            changes = self._calculate_social_mirroring_changes(
                personality_profile, character_memory, interaction_context, trigger_strength
            )

        elif strategy == AdaptationStrategy.CRISIS_CATALYZED:
            changes = self._calculate_crisis_catalyzed_changes(
                personality_profile, emotion_tracker, trigger_strength
            )

        elif strategy == AdaptationStrategy.GROWTH_ORIENTED:
            changes = self._calculate_growth_oriented_changes(
                personality_profile, character_memory, trigger_strength
            )

        # Apply adaptation rate scaling
        scaled_changes = {
            trait: change * self.configuration.adaptation_rate
            for trait, change in changes.items()
        }

        return scaled_changes

    def _calculate_gradual_drift_changes(
        self,
        personality_profile: PersonalityProfile,
        trigger_strength: float
    ) -> Dict[PersonalityTrait, float]:
        """Calculate changes for gradual drift strategy"""
        changes = {}

        # Small random changes toward personality balance
        for trait in PersonalityTrait:
            current_score = personality_profile.get_trait_score(trait)

            # Slight drift toward center (0.5) or toward extremes based on current position
            if current_score < 0.3:
                # Low scores drift slightly upward
                drift = 0.02 * trigger_strength
            elif current_score > 0.7:
                # High scores drift slightly downward
                drift = -0.02 * trigger_strength
            else:
                # Middle scores have minimal drift
                drift = (0.5 - current_score) * 0.01 * trigger_strength

            if abs(drift) > 0.001:
                changes[trait] = drift

        return changes

    def _calculate_experience_based_changes(
        self,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        trigger_strength: float
    ) -> Dict[PersonalityTrait, float]:
        """Calculate changes based on accumulated experiences"""
        changes = {}

        # Analyze behavioral patterns for adaptation cues
        behavioral_insights = character_memory.get_behavioral_insights()

        # Strong behavioral patterns suggest personality reinforcement
        for pattern_data in behavioral_insights.get("most_confident_patterns", []):
            pattern_type = pattern_data.get("pattern_type", "")
            confidence = pattern_data.get("confidence_score", 0)

            if confidence > 0.7:
                if "communication" in pattern_type:
                    # Communication patterns affect Extraversion
                    changes[PersonalityTrait.EXTRAVERSION] = 0.03 * trigger_strength * confidence

                elif "creative" in pattern_type:
                    # Creative patterns affect Openness
                    changes[PersonalityTrait.OPENNESS] = 0.03 * trigger_strength * confidence

                elif "organized" in pattern_type:
                    # Organization patterns affect Conscientiousness
                    changes[PersonalityTrait.CONSCIENTIOUSNESS] = 0.03 * trigger_strength * confidence

        return changes

    def _calculate_social_mirroring_changes(
        self,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        interaction_context: Dict[str, Any],
        trigger_strength: float
    ) -> Dict[PersonalityTrait, float]:
        """Calculate changes based on social mirroring"""
        changes = {}

        user_id = interaction_context.get("user_id")
        if not user_id:
            return changes

        relationship = character_memory.get_relationship_by_id(user_id)
        if not relationship:
            return changes

        # Strong relationships can influence personality
        if relationship.trust_level > 0.7 and relationship.emotional_bond > 0.6:
            # Infer user personality from interaction style
            user_input = interaction_context.get("user_input", "").lower()

            # Simple user trait inference
            if any(word in user_input for word in ["exciting", "amazing", "love", "awesome"]):
                # Enthusiastic user -> increase Extraversion
                changes[PersonalityTrait.EXTRAVERSION] = 0.02 * trigger_strength

            if any(word in user_input for word in ["think", "analyze", "consider", "philosophy"]):
                # Analytical user -> increase Openness
                changes[PersonalityTrait.OPENNESS] = 0.02 * trigger_strength

            if any(word in user_input for word in ["help", "support", "care", "understand"]):
                # Supportive user -> increase Agreeableness
                changes[PersonalityTrait.AGREEABLENESS] = 0.02 * trigger_strength

        return changes

    def _calculate_crisis_catalyzed_changes(
        self,
        personality_profile: PersonalityProfile,
        emotion_tracker: EmotionTracker,
        trigger_strength: float
    ) -> Dict[PersonalityTrait, float]:
        """Calculate changes during emotional crises"""
        changes = {}

        current_state = emotion_tracker.get_current_emotional_state()
        if not current_state:
            return changes

        # Crisis events can cause significant personality shifts
        if current_state.primary_intensity > 0.8:
            # High stress can increase Neuroticism
            if current_state.primary_emotion in [EmotionType.FEAR, EmotionType.ANGER, EmotionType.SADNESS]:
                changes[PersonalityTrait.NEUROTICISM] = 0.05 * trigger_strength

            # But can also build resilience (decrease Neuroticism) over time
            if len(emotion_tracker.emotion_history) > 20:  # If experienced many emotions
                changes[PersonalityTrait.NEUROTICISM] = -0.02 * trigger_strength

        # Positive crises (excitement, joy) can increase Extraversion
        if current_state.primary_emotion == EmotionType.JOY and current_state.primary_intensity > 0.7:
            changes[PersonalityTrait.EXTRAVERSION] = 0.03 * trigger_strength

        return changes

    def _calculate_growth_oriented_changes(
        self,
        personality_profile: PersonalityProfile,
        character_memory: CharacterMemoryManager,
        trigger_strength: float
    ) -> Dict[PersonalityTrait, float]:
        """Calculate changes oriented toward psychological growth"""
        changes = {}

        # Growth typically involves:
        # - Increased Openness (more open to experiences)
        # - Increased Conscientiousness (better self-regulation)
        # - Decreased Neuroticism (better emotional stability)
        # - Balanced Extraversion and Agreeableness

        current_openness = personality_profile.get_trait_score(PersonalityTrait.OPENNESS)
        current_conscientiousness = personality_profile.get_trait_score(PersonalityTrait.CONSCIENTIOUSNESS)
        current_neuroticism = personality_profile.get_trait_score(PersonalityTrait.NEUROTICISM)

        # Encourage growth in positive directions
        if current_openness < 0.7:
            changes[PersonalityTrait.OPENNESS] = 0.02 * trigger_strength

        if current_conscientiousness < 0.7:
            changes[PersonalityTrait.CONSCIENTIOUSNESS] = 0.02 * trigger_strength

        if current_neuroticism > 0.4:
            changes[PersonalityTrait.NEUROTICISM] = -0.02 * trigger_strength

        # Analyze character development for additional growth opportunities
        development_summary = character_memory.get_character_development_summary()
        behavioral_maturity = development_summary.get("behavioral_maturity", 0.5)

        if behavioral_maturity > 0.7:
            # High maturity enables more significant growth
            for trait, change in changes.items():
                changes[trait] = change * 1.5

        return changes

    def _should_apply_adaptation(self, event: AdaptationEvent, total_changes: Dict[PersonalityTrait, float]) -> bool:
        """Determine if an adaptation event should be applied"""

        # Check confidence threshold
        if event.confidence < 0.3:
            return False

        # Check cumulative change limits
        for trait, change in event.personality_changes.items():
            if abs(total_changes.get(trait, 0) + change) > self.configuration.max_cumulative_change:
                return False

        # Check individual change limits
        for trait, change in event.personality_changes.items():
            if abs(change) > self.configuration.max_trait_change_per_event:
                return False

        return True

    def _bound_trait_change(self, trait: PersonalityTrait, change: float, cumulative_change: float) -> float:
        """Apply bounds to trait changes"""

        # Apply per-event limit
        bounded_change = max(
            -self.configuration.max_trait_change_per_event,
            min(self.configuration.max_trait_change_per_event, change)
        )

        # Apply cumulative limit
        if abs(cumulative_change + bounded_change) > self.configuration.max_cumulative_change:
            # Reduce change to stay within cumulative limit
            remaining_budget = self.configuration.max_cumulative_change - abs(cumulative_change)
            bounded_change = math.copysign(remaining_budget, bounded_change)

        # Apply core stability (some traits are more resistant to change)
        stability_factor = self.configuration.core_stability

        # Core traits (based on research) are more stable
        core_traits = {PersonalityTrait.CONSCIENTIOUSNESS, PersonalityTrait.NEUROTICISM}
        if trait in core_traits:
            bounded_change *= (1 - stability_factor)

        return bounded_change

    def _create_growth_milestone(self, event: AdaptationEvent, total_changes: Dict[PersonalityTrait, float]) -> CharacterGrowthMilestone:
        """Create a growth milestone for significant adaptations"""

        # Calculate before and after states
        before_state = {trait.value: 0.0 for trait in PersonalityTrait}  # This would be actual before values
        after_state = {trait.value: change for trait, change in total_changes.items()}

        milestone = CharacterGrowthMilestone(
            milestone_id=f"growth_{event.event_id}",
            milestone_type="personality_adaptation",
            description=f"Significant personality adaptation triggered by {event.trigger.value}",
            significance_level=event.impact_magnitude,
            before_state=before_state,
            after_state=after_state,
            trigger_event=event.adaptation_rationale,
            session_id=event.session_id
        )

        return milestone

    def _generate_adaptation_rationale(
        self,
        strategy: AdaptationStrategy,
        trigger: AdaptationTrigger,
        personality_changes: Dict[PersonalityTrait, float]
    ) -> str:
        """Generate human-readable rationale for adaptation"""

        # Identify the most significant change
        if not personality_changes:
            return "No significant personality changes"

        max_change_trait = max(personality_changes, key=lambda t: abs(personality_changes[t]))
        max_change_value = personality_changes[max_change_trait]

        direction = "increased" if max_change_value > 0 else "decreased"
        magnitude = "slightly" if abs(max_change_value) < 0.02 else "moderately" if abs(max_change_value) < 0.05 else "significantly"

        rationale = f"Character {magnitude} {direction} in {max_change_trait.value} "
        rationale += f"due to {trigger.value} using {strategy.value} adaptation strategy."

        return rationale

    def _update_adaptation_statistics(self) -> None:
        """Update adaptation statistics"""
        if not self.adaptation_events:
            return

        self.adaptation_statistics["total_adaptations"] = len(self.adaptation_events)

        # Calculate average impact
        impacts = [event.impact_magnitude for event in self.adaptation_events]
        self.adaptation_statistics["average_impact"] = statistics.mean(impacts) if impacts else 0.0

        # Find dominant strategies
        strategy_counts = {}
        for event in self.adaptation_events:
            strategy = event.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        self.adaptation_statistics["dominant_strategies"] = [s[0] for s in sorted_strategies[:3]]

    def _analyze_trait_evolution(self) -> Dict[str, Any]:
        """Analyze how traits have evolved over time"""
        evolution_analysis = {}

        for trait in PersonalityTrait:
            trait_history = self.adaptation_statistics["trait_change_history"][trait]

            if not trait_history:
                evolution_analysis[trait.value] = {"status": "no_changes"}
                continue

            # Calculate total change
            total_change = sum(change["change"] for change in trait_history)

            # Calculate trend
            recent_changes = trait_history[-5:]  # Last 5 changes
            recent_trend = sum(change["change"] for change in recent_changes)

            # Determine evolution pattern
            if abs(total_change) < 0.01:
                pattern = "stable"
            elif total_change > 0.05:
                pattern = "increasing"
            elif total_change < -0.05:
                pattern = "decreasing"
            else:
                pattern = "fluctuating"

            evolution_analysis[trait.value] = {
                "total_change": total_change,
                "recent_trend": recent_trend,
                "pattern": pattern,
                "adaptation_count": len(trait_history),
                "most_common_trigger": self._get_most_common_trigger_for_trait(trait_history)
            }

        return evolution_analysis

    def _get_most_common_trigger_for_trait(self, trait_history: List[Dict[str, Any]]) -> str:
        """Get the most common trigger for trait changes"""
        if not trait_history:
            return "none"

        trigger_counts = {}
        for change in trait_history:
            trigger = change.get("trigger", "unknown")
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        return max(trigger_counts, key=trigger_counts.get) if trigger_counts else "unknown"

    def _calculate_adaptation_effectiveness(self) -> Dict[str, Any]:
        """Calculate how effective adaptations have been"""
        if not self.adaptation_events:
            return {"status": "no_adaptations"}

        # Calculate success rate
        successful_adaptations = [e for e in self.adaptation_events if e.confidence > 0.6]
        success_rate = len(successful_adaptations) / len(self.adaptation_events)

        # Calculate average confidence
        avg_confidence = statistics.mean([e.confidence for e in self.adaptation_events])

        # Calculate adaptation frequency
        if len(self.adaptation_events) >= 2:
            time_span = self.adaptation_events[-1].timestamp - self.adaptation_events[0].timestamp
            frequency = len(self.adaptation_events) / max(time_span.days, 1)
        else:
            frequency = 0.0

        return {
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "adaptation_frequency": frequency,  # adaptations per day
            "total_adaptations": len(self.adaptation_events),
            "effectiveness_score": (success_rate + avg_confidence) / 2
        }

    def _analyze_growth_trajectory(self) -> Dict[str, Any]:
        """Analyze overall character growth trajectory"""
        if not self.adaptation_events:
            return {"status": "no_growth_data"}

        # Calculate growth momentum
        recent_events = [e for e in self.adaptation_events if e.timestamp > datetime.utcnow() - timedelta(days=30)]
        growth_momentum = len(recent_events) / 30  # Growth events per day

        # Calculate growth direction (positive vs negative changes)
        positive_changes = 0
        negative_changes = 0

        for event in self.adaptation_events:
            for trait, change in event.personality_changes.items():
                if trait in [PersonalityTrait.OPENNESS, PersonalityTrait.CONSCIENTIOUSNESS, PersonalityTrait.EXTRAVERSION, PersonalityTrait.AGREEABLENESS]:
                    if change > 0:
                        positive_changes += 1
                    elif change < 0:
                        negative_changes += 1
                elif trait == PersonalityTrait.NEUROTICISM:
                    # For Neuroticism, negative changes are positive growth
                    if change < 0:
                        positive_changes += 1
                    elif change > 0:
                        negative_changes += 1

        growth_direction = "positive" if positive_changes > negative_changes else "negative" if negative_changes > positive_changes else "neutral"

        return {
            "growth_momentum": growth_momentum,
            "growth_direction": growth_direction,
            "positive_changes": positive_changes,
            "negative_changes": negative_changes,
            "growth_consistency": positive_changes / max(positive_changes + negative_changes, 1)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "configuration": self.configuration.to_dict(),
            "adaptation_events": [event.to_dict() for event in self.adaptation_events[-50:]],  # Last 50 events
            "baseline_personality": self.baseline_personality.to_dict() if self.baseline_personality else None,
            "adaptation_statistics": self.adaptation_statistics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterAdaptationEngine':
        """Create from dictionary"""
        engine = cls(data["agent_id"])

        if "configuration" in data:
            engine.configuration = AdaptationConfiguration.from_dict(data["configuration"])

        if "adaptation_events" in data:
            engine.adaptation_events = [
                AdaptationEvent.from_dict(event_data)
                for event_data in data["adaptation_events"]
            ]

        if "baseline_personality" in data and data["baseline_personality"]:
            # This would need PersonalityProfile.from_dict implementation
            pass  # engine.baseline_personality = PersonalityProfile.from_dict(data["baseline_personality"])

        if "adaptation_statistics" in data:
            engine.adaptation_statistics = data["adaptation_statistics"]

        return engine