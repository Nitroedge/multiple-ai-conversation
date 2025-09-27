"""
Emotional state tracking and analysis system
Real-time emotion detection, tracking, and influence on personality development
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import re
from datetime import datetime, timedelta
import math

from ..personality.big_five_model import PersonalityTrait, PersonalityProfile


class EmotionType(Enum):
    """Primary emotion types based on psychological models"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

    # Complex emotions
    LOVE = "love"
    GUILT = "guilt"
    SHAME = "shame"
    PRIDE = "pride"
    ENVY = "envy"
    GRATITUDE = "gratitude"
    HOPE = "hope"
    DESPAIR = "despair"
    EXCITEMENT = "excitement"
    CONTENTMENT = "contentment"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    CURIOSITY = "curiosity"
    BOREDOM = "boredom"


class EmotionIntensity(Enum):
    """Emotion intensity levels"""
    SUBTLE = "subtle"  # 0.1-0.3
    MODERATE = "moderate"  # 0.3-0.6
    STRONG = "strong"  # 0.6-0.8
    INTENSE = "intense"  # 0.8-1.0


@dataclass
class EmotionReading:
    """Individual emotion detection reading"""
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # "text_analysis", "user_indication", "context_inference"
    timestamp: datetime
    context: str = ""
    triggers: List[str] = field(default_factory=list)

    def get_intensity_level(self) -> EmotionIntensity:
        """Get intensity level enum"""
        if self.intensity <= 0.3:
            return EmotionIntensity.SUBTLE
        elif self.intensity <= 0.6:
            return EmotionIntensity.MODERATE
        elif self.intensity <= 0.8:
            return EmotionIntensity.STRONG
        else:
            return EmotionIntensity.INTENSE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion.value,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "triggers": self.triggers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionReading':
        return cls(
            emotion=EmotionType(data["emotion"]),
            intensity=data["intensity"],
            confidence=data["confidence"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context=data.get("context", ""),
            triggers=data.get("triggers", [])
        )


@dataclass
class EmotionalState:
    """Current emotional state with multiple emotions"""
    primary_emotion: EmotionType
    primary_intensity: float
    secondary_emotions: Dict[EmotionType, float] = field(default_factory=dict)

    # Emotional dimensions
    valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.0  # -1.0 (calm) to 1.0 (excited)
    dominance: float = 0.0  # -1.0 (submissive) to 1.0 (dominant)

    # Temporal aspects
    stability: float = 0.5  # How stable this emotional state is
    duration: timedelta = field(default_factory=timedelta)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update_emotional_dimensions(self) -> None:
        """Update VAD (Valence-Arousal-Dominance) dimensions based on emotions"""
        emotion_vad_mappings = {
            EmotionType.JOY: (0.8, 0.6, 0.5),
            EmotionType.SADNESS: (-0.6, -0.4, -0.3),
            EmotionType.ANGER: (-0.5, 0.8, 0.7),
            EmotionType.FEAR: (-0.7, 0.6, -0.6),
            EmotionType.SURPRISE: (0.2, 0.8, 0.0),
            EmotionType.DISGUST: (-0.6, 0.3, 0.2),
            EmotionType.TRUST: (0.6, -0.2, 0.3),
            EmotionType.ANTICIPATION: (0.4, 0.5, 0.2),
            EmotionType.LOVE: (0.9, 0.4, 0.3),
            EmotionType.EXCITEMENT: (0.8, 0.9, 0.6),
            EmotionType.CONTENTMENT: (0.7, -0.3, 0.1),
            EmotionType.FRUSTRATION: (-0.4, 0.6, 0.4),
            EmotionType.CURIOSITY: (0.3, 0.4, 0.1)
        }

        # Calculate weighted VAD based on primary and secondary emotions
        total_intensity = self.primary_intensity
        weighted_valence = 0.0
        weighted_arousal = 0.0
        weighted_dominance = 0.0

        # Primary emotion
        if self.primary_emotion in emotion_vad_mappings:
            v, a, d = emotion_vad_mappings[self.primary_emotion]
            weighted_valence += v * self.primary_intensity
            weighted_arousal += a * self.primary_intensity
            weighted_dominance += d * self.primary_intensity

        # Secondary emotions
        for emotion, intensity in self.secondary_emotions.items():
            if emotion in emotion_vad_mappings:
                v, a, d = emotion_vad_mappings[emotion]
                weighted_valence += v * intensity
                weighted_arousal += a * intensity
                weighted_dominance += d * intensity
                total_intensity += intensity

        # Normalize by total intensity
        if total_intensity > 0:
            self.valence = max(-1.0, min(1.0, weighted_valence / total_intensity))
            self.arousal = max(-1.0, min(1.0, weighted_arousal / total_intensity))
            self.dominance = max(-1.0, min(1.0, weighted_dominance / total_intensity))

    def get_emotional_summary(self) -> str:
        """Get human-readable emotional summary"""
        intensity_desc = {
            EmotionIntensity.SUBTLE: "slightly",
            EmotionIntensity.MODERATE: "moderately",
            EmotionIntensity.STRONG: "strongly",
            EmotionIntensity.INTENSE: "intensely"
        }

        primary_reading = EmotionReading(
            self.primary_emotion, self.primary_intensity, 1.0, "state", datetime.utcnow()
        )
        intensity_level = intensity_desc[primary_reading.get_intensity_level()]

        summary = f"{intensity_level} {self.primary_emotion.value}"

        if self.secondary_emotions:
            secondary_emotion = max(self.secondary_emotions, key=self.secondary_emotions.get)
            summary += f" with elements of {secondary_emotion.value}"

        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion.value,
            "primary_intensity": self.primary_intensity,
            "secondary_emotions": {e.value: i for e, i in self.secondary_emotions.items()},
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "stability": self.stability,
            "duration": self.duration.total_seconds(),
            "last_updated": self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        return cls(
            primary_emotion=EmotionType(data["primary_emotion"]),
            primary_intensity=data["primary_intensity"],
            secondary_emotions={
                EmotionType(e): i for e, i in data.get("secondary_emotions", {}).items()
            },
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.0),
            stability=data.get("stability", 0.5),
            duration=timedelta(seconds=data.get("duration", 0)),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )


class EmotionAnalyzer:
    """Analyzes text and context to detect emotions"""

    def __init__(self):
        self.emotion_lexicon = self._build_emotion_lexicon()
        self.contextual_patterns = self._build_contextual_patterns()

    def analyze_text_emotion(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[EmotionReading]:
        """Analyze text for emotional content"""
        readings = []

        # Lexicon-based analysis
        lexicon_emotions = self._lexicon_analysis(text)
        readings.extend(lexicon_emotions)

        # Pattern-based analysis
        pattern_emotions = self._pattern_analysis(text)
        readings.extend(pattern_emotions)

        # Contextual analysis
        if context:
            contextual_emotions = self._contextual_analysis(text, context)
            readings.extend(contextual_emotions)

        # Aggregate and filter readings
        aggregated_readings = self._aggregate_emotion_readings(readings)

        return aggregated_readings

    def _build_emotion_lexicon(self) -> Dict[EmotionType, List[Tuple[str, float]]]:
        """Build emotion word lexicon with intensity weights"""
        return {
            EmotionType.JOY: [
                ("happy", 0.6), ("joyful", 0.8), ("ecstatic", 0.9), ("delighted", 0.7),
                ("cheerful", 0.5), ("pleased", 0.4), ("elated", 0.8), ("thrilled", 0.9),
                ("glad", 0.5), ("content", 0.4), ("satisfied", 0.5), ("excited", 0.7)
            ],
            EmotionType.SADNESS: [
                ("sad", 0.6), ("depressed", 0.8), ("melancholy", 0.7), ("sorrowful", 0.7),
                ("gloomy", 0.6), ("downcast", 0.6), ("despondent", 0.8), ("dejected", 0.7),
                ("mournful", 0.7), ("blue", 0.4), ("unhappy", 0.6), ("miserable", 0.8)
            ],
            EmotionType.ANGER: [
                ("angry", 0.7), ("furious", 0.9), ("enraged", 0.9), ("livid", 0.9),
                ("irritated", 0.5), ("annoyed", 0.4), ("mad", 0.6), ("outraged", 0.8),
                ("irate", 0.8), ("infuriated", 0.9), ("pissed", 0.7), ("frustrated", 0.6)
            ],
            EmotionType.FEAR: [
                ("afraid", 0.7), ("scared", 0.6), ("terrified", 0.9), ("frightened", 0.7),
                ("anxious", 0.6), ("worried", 0.5), ("nervous", 0.5), ("apprehensive", 0.6),
                ("panicked", 0.9), ("alarmed", 0.7), ("intimidated", 0.6), ("concerned", 0.4)
            ],
            EmotionType.SURPRISE: [
                ("surprised", 0.6), ("astonished", 0.8), ("amazed", 0.7), ("shocked", 0.8),
                ("stunned", 0.8), ("bewildered", 0.6), ("astounded", 0.8), ("flabbergasted", 0.9),
                ("startled", 0.7), ("taken aback", 0.6), ("unexpected", 0.4)
            ],
            EmotionType.CURIOSITY: [
                ("curious", 0.6), ("interested", 0.5), ("intrigued", 0.7), ("wondering", 0.4),
                ("fascinated", 0.8), ("inquisitive", 0.6), ("eager", 0.6), ("questioning", 0.4)
            ],
            EmotionType.LOVE: [
                ("love", 0.8), ("adore", 0.8), ("cherish", 0.7), ("treasure", 0.7),
                ("devoted", 0.7), ("passionate", 0.8), ("affectionate", 0.6), ("caring", 0.5)
            ],
            EmotionType.EXCITEMENT: [
                ("excited", 0.7), ("thrilled", 0.8), ("pumped", 0.7), ("exhilarated", 0.8),
                ("energized", 0.6), ("enthusiastic", 0.7), ("hyped", 0.7), ("stoked", 0.6)
            ]
        }

    def _build_contextual_patterns(self) -> Dict[str, Tuple[EmotionType, float]]:
        """Build contextual emotion patterns"""
        return {
            r"\b(can't wait|looking forward)\b": (EmotionType.ANTICIPATION, 0.6),
            r"\b(what\?+|how\?+|why\?+)\b": (EmotionType.SURPRISE, 0.4),
            r"\b(thank you|thanks|grateful)\b": (EmotionType.GRATITUDE, 0.6),
            r"\b(sorry|apologize|my bad)\b": (EmotionType.GUILT, 0.5),
            r"\b(wow|amazing|incredible)\b": (EmotionType.SURPRISE, 0.6),
            r"\b(hate|disgusting|awful)\b": (EmotionType.DISGUST, 0.7),
            r"\?+": (EmotionType.CURIOSITY, 0.3),
            r"!{2,}": (EmotionType.EXCITEMENT, 0.5),
            r"\b(calm|peaceful|relaxed)\b": (EmotionType.CONTENTMENT, 0.5)
        }

    def _lexicon_analysis(self, text: str) -> List[EmotionReading]:
        """Perform lexicon-based emotion analysis"""
        text_lower = text.lower()
        readings = []

        for emotion_type, word_list in self.emotion_lexicon.items():
            total_intensity = 0.0
            found_words = []

            for word, intensity in word_list:
                if word in text_lower:
                    total_intensity += intensity
                    found_words.append(word)

            if total_intensity > 0:
                # Normalize intensity (multiple words can boost intensity)
                normalized_intensity = min(1.0, total_intensity / 2.0)
                confidence = min(1.0, len(found_words) * 0.3)

                reading = EmotionReading(
                    emotion=emotion_type,
                    intensity=normalized_intensity,
                    confidence=confidence,
                    source="text_analysis",
                    timestamp=datetime.utcnow(),
                    context=text[:100],
                    triggers=found_words
                )
                readings.append(reading)

        return readings

    def _pattern_analysis(self, text: str) -> List[EmotionReading]:
        """Perform pattern-based emotion analysis"""
        readings = []

        for pattern, (emotion_type, intensity) in self.contextual_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Multiple matches can increase intensity
                final_intensity = min(1.0, intensity * len(matches))
                confidence = min(1.0, len(matches) * 0.4)

                reading = EmotionReading(
                    emotion=emotion_type,
                    intensity=final_intensity,
                    confidence=confidence,
                    source="pattern_analysis",
                    timestamp=datetime.utcnow(),
                    context=text[:100],
                    triggers=[str(match) for match in matches]
                )
                readings.append(reading)

        return readings

    def _contextual_analysis(self, text: str, context: Dict[str, Any]) -> List[EmotionReading]:
        """Perform contextual emotion analysis"""
        readings = []

        # Analyze conversation stage context
        conversation_stage = context.get("conversation_stage", "")
        if conversation_stage == "greeting":
            # Greetings often have positive valence
            readings.append(EmotionReading(
                emotion=EmotionType.JOY,
                intensity=0.3,
                confidence=0.4,
                source="context_inference",
                timestamp=datetime.utcnow(),
                context="greeting stage"
            ))

        # Analyze topic context
        topic = context.get("topic_focus", "")
        if any(word in topic.lower() for word in ["problem", "issue", "trouble"]):
            readings.append(EmotionReading(
                emotion=EmotionType.CONCERN,
                intensity=0.5,
                confidence=0.5,
                source="context_inference",
                timestamp=datetime.utcnow(),
                context=f"topic: {topic}"
            ))

        # Analyze user emotion context
        if "detected_user_emotion" in context:
            user_emotion = context["detected_user_emotion"]
            # Mirror some of the user's emotion
            if user_emotion in [e.value for e in EmotionType]:
                readings.append(EmotionReading(
                    emotion=EmotionType(user_emotion),
                    intensity=0.3,  # Mild mirroring
                    confidence=0.6,
                    source="context_inference",
                    timestamp=datetime.utcnow(),
                    context=f"user emotion: {user_emotion}"
                ))

        return readings

    def _aggregate_emotion_readings(self, readings: List[EmotionReading]) -> List[EmotionReading]:
        """Aggregate multiple readings for the same emotion"""
        emotion_groups = {}

        for reading in readings:
            if reading.emotion not in emotion_groups:
                emotion_groups[reading.emotion] = []
            emotion_groups[reading.emotion].append(reading)

        aggregated = []
        for emotion_type, group_readings in emotion_groups.items():
            if len(group_readings) == 1:
                aggregated.append(group_readings[0])
            else:
                # Aggregate multiple readings
                total_intensity = sum(r.intensity for r in group_readings)
                avg_confidence = sum(r.confidence for r in group_readings) / len(group_readings)
                all_triggers = []
                all_sources = []

                for r in group_readings:
                    all_triggers.extend(r.triggers)
                    if r.source not in all_sources:
                        all_sources.append(r.source)

                # Create aggregated reading
                aggregated_reading = EmotionReading(
                    emotion=emotion_type,
                    intensity=min(1.0, total_intensity),
                    confidence=min(1.0, avg_confidence * 1.2),  # Boost confidence for multiple sources
                    source="+".join(all_sources),
                    timestamp=datetime.utcnow(),
                    context=group_readings[0].context,
                    triggers=list(set(all_triggers))
                )
                aggregated.append(aggregated_reading)

        # Sort by intensity
        aggregated.sort(key=lambda x: x.intensity, reverse=True)

        return aggregated


class EmotionTracker:
    """Tracks emotional states over time for an agent"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.current_state: Optional[EmotionalState] = None
        self.emotion_history: List[EmotionReading] = []
        self.state_history: List[EmotionalState] = []
        self.analyzer = EmotionAnalyzer()

        # Emotional patterns and tendencies
        self.baseline_emotions: Dict[EmotionType, float] = {}
        self.emotional_volatility = 0.5  # How quickly emotions change
        self.emotional_recovery_rate = 0.1  # How quickly emotions return to baseline

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> EmotionalState:
        """Process a message and update emotional state"""
        # Analyze emotions in the message
        emotion_readings = self.analyzer.analyze_text_emotion(message, context)

        # Store readings in history
        for reading in emotion_readings:
            self.emotion_history.append(reading)

        # Update current emotional state
        new_state = self._update_emotional_state(emotion_readings)

        # Store state in history
        if self.current_state:
            self.state_history.append(self.current_state)

        self.current_state = new_state

        # Trim history to manage memory
        self._trim_history()

        return new_state

    def get_current_emotional_state(self) -> Optional[EmotionalState]:
        """Get current emotional state"""
        return self.current_state

    def get_emotional_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get emotional trends over specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_readings = [
            r for r in self.emotion_history
            if r.timestamp > cutoff_time
        ]

        if not recent_readings:
            return {"status": "no_recent_data"}

        # Calculate emotion frequencies
        emotion_counts = {}
        total_intensity = {}

        for reading in recent_readings:
            emotion = reading.emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_intensity[emotion] = total_intensity.get(emotion, 0) + reading.intensity

        # Calculate averages
        emotion_averages = {
            emotion: total_intensity[emotion] / emotion_counts[emotion]
            for emotion in emotion_counts
        }

        # Determine dominant emotions
        dominant_emotions = sorted(
            emotion_averages.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            "time_period_hours": hours,
            "total_readings": len(recent_readings),
            "dominant_emotions": [
                {"emotion": emotion.value, "average_intensity": avg}
                for emotion, avg in dominant_emotions
            ],
            "emotion_frequency": {
                emotion.value: count for emotion, count in emotion_counts.items()
            },
            "emotional_volatility": self._calculate_volatility(recent_readings)
        }

    def get_personality_emotion_influence(self, personality_profile: PersonalityProfile) -> Dict[PersonalityTrait, float]:
        """Analyze how emotions are influencing personality development"""
        if not self.emotion_history:
            return {}

        # Get recent emotion data
        recent_cutoff = datetime.utcnow() - timedelta(hours=72)
        recent_emotions = [
            r for r in self.emotion_history
            if r.timestamp > recent_cutoff
        ]

        if not recent_emotions:
            return {}

        # Map emotions to personality trait influences
        trait_influences = {trait: 0.0 for trait in PersonalityTrait}

        for reading in recent_emotions:
            influences = self._map_emotion_to_personality_influence(reading)
            for trait, influence in influences.items():
                trait_influences[trait] += influence

        # Normalize by number of readings
        if recent_emotions:
            for trait in trait_influences:
                trait_influences[trait] /= len(recent_emotions)

        return trait_influences

    def _update_emotional_state(self, new_readings: List[EmotionReading]) -> EmotionalState:
        """Update emotional state based on new readings"""
        if not new_readings:
            # No new emotions, decay current state toward baseline
            return self._decay_emotional_state()

        # Determine primary emotion (highest intensity)
        primary_reading = max(new_readings, key=lambda x: x.intensity)

        # Collect secondary emotions
        secondary_emotions = {}
        for reading in new_readings:
            if reading.emotion != primary_reading.emotion and reading.intensity > 0.2:
                secondary_emotions[reading.emotion] = reading.intensity

        # Create new emotional state
        new_state = EmotionalState(
            primary_emotion=primary_reading.emotion,
            primary_intensity=primary_reading.intensity,
            secondary_emotions=secondary_emotions
        )

        # Update VAD dimensions
        new_state.update_emotional_dimensions()

        # Calculate stability based on consistency with previous state
        if self.current_state:
            new_state.stability = self._calculate_emotional_stability(new_state)
            new_state.duration = self.current_state.duration + timedelta(minutes=1)
        else:
            new_state.stability = 0.5
            new_state.duration = timedelta(minutes=1)

        return new_state

    def _decay_emotional_state(self) -> EmotionalState:
        """Decay current emotional state toward baseline"""
        if not self.current_state:
            # Create neutral baseline state
            return EmotionalState(
                primary_emotion=EmotionType.CONTENTMENT,
                primary_intensity=0.3,
                valence=0.0,
                arousal=0.0,
                dominance=0.0
            )

        # Decay intensity toward baseline
        decay_factor = self.emotional_recovery_rate
        new_intensity = max(0.1, self.current_state.primary_intensity * (1 - decay_factor))

        # Create decayed state
        decayed_state = EmotionalState(
            primary_emotion=self.current_state.primary_emotion,
            primary_intensity=new_intensity,
            secondary_emotions={
                e: max(0.05, i * (1 - decay_factor))
                for e, i in self.current_state.secondary_emotions.items()
            },
            valence=self.current_state.valence * (1 - decay_factor * 0.5),
            arousal=self.current_state.arousal * (1 - decay_factor * 0.7),
            dominance=self.current_state.dominance * (1 - decay_factor * 0.3),
            stability=min(1.0, self.current_state.stability + 0.1),
            duration=self.current_state.duration + timedelta(minutes=5)
        )

        return decayed_state

    def _calculate_emotional_stability(self, new_state: EmotionalState) -> float:
        """Calculate emotional stability based on consistency"""
        if not self.current_state:
            return 0.5

        # Compare primary emotions
        primary_consistency = 1.0 if new_state.primary_emotion == self.current_state.primary_emotion else 0.0

        # Compare VAD dimensions
        valence_diff = abs(new_state.valence - self.current_state.valence)
        arousal_diff = abs(new_state.arousal - self.current_state.arousal)
        dominance_diff = abs(new_state.dominance - self.current_state.dominance)

        vad_consistency = 1.0 - ((valence_diff + arousal_diff + dominance_diff) / 6.0)

        # Combine factors
        stability = (primary_consistency * 0.4) + (vad_consistency * 0.6)

        return max(0.0, min(1.0, stability))

    def _calculate_volatility(self, readings: List[EmotionReading]) -> float:
        """Calculate emotional volatility from readings"""
        if len(readings) < 2:
            return 0.0

        # Calculate intensity variance
        intensities = [r.intensity for r in readings]
        mean_intensity = sum(intensities) / len(intensities)
        variance = sum((i - mean_intensity) ** 2 for i in intensities) / len(intensities)

        # Calculate emotion type changes
        emotion_changes = 0
        for i in range(1, len(readings)):
            if readings[i].emotion != readings[i-1].emotion:
                emotion_changes += 1

        change_rate = emotion_changes / len(readings)

        # Combine variance and change rate
        volatility = (math.sqrt(variance) + change_rate) / 2

        return min(1.0, volatility)

    def _map_emotion_to_personality_influence(self, reading: EmotionReading) -> Dict[PersonalityTrait, float]:
        """Map emotion to personality trait influences"""
        # Emotion-personality influence mappings
        influence_map = {
            EmotionType.JOY: {
                PersonalityTrait.EXTRAVERSION: 0.1,
                PersonalityTrait.AGREEABLENESS: 0.05,
                PersonalityTrait.NEUROTICISM: -0.05
            },
            EmotionType.SADNESS: {
                PersonalityTrait.NEUROTICISM: 0.1,
                PersonalityTrait.EXTRAVERSION: -0.05
            },
            EmotionType.ANGER: {
                PersonalityTrait.NEUROTICISM: 0.1,
                PersonalityTrait.AGREEABLENESS: -0.1
            },
            EmotionType.FEAR: {
                PersonalityTrait.NEUROTICISM: 0.15,
                PersonalityTrait.OPENNESS: -0.05
            },
            EmotionType.CURIOSITY: {
                PersonalityTrait.OPENNESS: 0.15,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.05
            },
            EmotionType.EXCITEMENT: {
                PersonalityTrait.EXTRAVERSION: 0.1,
                PersonalityTrait.OPENNESS: 0.05
            }
        }

        base_influences = influence_map.get(reading.emotion, {})

        # Scale by intensity and confidence
        scaling_factor = reading.intensity * reading.confidence

        scaled_influences = {
            trait: influence * scaling_factor
            for trait, influence in base_influences.items()
        }

        return scaled_influences

    def _trim_history(self, max_readings: int = 1000, max_states: int = 100) -> None:
        """Trim history to manage memory usage"""
        if len(self.emotion_history) > max_readings:
            self.emotion_history = self.emotion_history[-max_readings:]

        if len(self.state_history) > max_states:
            self.state_history = self.state_history[-max_states:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "current_state": self.current_state.to_dict() if self.current_state else None,
            "emotion_history": [r.to_dict() for r in self.emotion_history[-50:]],  # Last 50 readings
            "state_history": [s.to_dict() for s in self.state_history[-20:]],  # Last 20 states
            "baseline_emotions": {e.value: v for e, v in self.baseline_emotions.items()},
            "emotional_volatility": self.emotional_volatility,
            "emotional_recovery_rate": self.emotional_recovery_rate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionTracker':
        """Create from dictionary"""
        tracker = cls(data["agent_id"])

        if data.get("current_state"):
            tracker.current_state = EmotionalState.from_dict(data["current_state"])

        tracker.emotion_history = [
            EmotionReading.from_dict(r) for r in data.get("emotion_history", [])
        ]

        tracker.state_history = [
            EmotionalState.from_dict(s) for s in data.get("state_history", [])
        ]

        tracker.baseline_emotions = {
            EmotionType(e): v for e, v in data.get("baseline_emotions", {}).items()
        }

        tracker.emotional_volatility = data.get("emotional_volatility", 0.5)
        tracker.emotional_recovery_rate = data.get("emotional_recovery_rate", 0.1)

        return tracker