"""
Voice personality adaptation system that adapts voice characteristics based on agent personality
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple

from pydantic import BaseModel, Field

from .tts_processor import VoiceProfile, VoiceGender, VoiceAge
from ..personality.big_five_model import BigFivePersonality, PersonalityTrait
from ..emotion.emotion_tracking import EmotionalState, EmotionType

logger = logging.getLogger(__name__)


class VoiceCharacteristic(str, Enum):
    """Voice characteristics that can be adapted"""
    STABILITY = "stability"
    SIMILARITY_BOOST = "similarity_boost"
    STYLE = "style"
    SPEED = "speed"
    PITCH = "pitch"
    VOLUME = "volume"
    ENERGY = "energy"
    WARMTH = "warmth"


class AdaptationIntensity(str, Enum):
    """Intensity levels for personality adaptation"""
    SUBTLE = "subtle"      # 0.2 - minimal adaptation
    MODERATE = "moderate"  # 0.5 - balanced adaptation
    STRONG = "strong"      # 0.8 - noticeable adaptation
    EXTREME = "extreme"    # 1.0 - maximum adaptation


@dataclass
class VoiceAdaptationRule:
    """Rule for adapting voice characteristics based on personality traits"""
    trait: PersonalityTrait
    characteristic: VoiceCharacteristic
    min_value: float
    max_value: float
    curve_type: str = "linear"  # linear, exponential, logarithmic
    invert: bool = False  # Invert the relationship


@dataclass
class EmotionVoiceMapping:
    """Mapping between emotional state and voice characteristics"""
    emotion: EmotionType
    voice_adjustments: Dict[VoiceCharacteristic, float]
    intensity_multiplier: float = 1.0


class PersonalityVoiceProfile(BaseModel):
    """Extended voice profile with personality-based characteristics"""
    base_profile: VoiceProfile
    personality_adjustments: Dict[VoiceCharacteristic, float] = Field(default_factory=dict)
    emotion_adjustments: Dict[VoiceCharacteristic, float] = Field(default_factory=dict)
    adaptation_intensity: AdaptationIntensity = AdaptationIntensity.MODERATE

    # Personality-driven characteristics
    extroversion_influence: float = 0.0
    agreeableness_influence: float = 0.0
    conscientiousness_influence: float = 0.0
    neuroticism_influence: float = 0.0
    openness_influence: float = 0.0

    # Current emotional influence
    current_emotion_influence: float = 0.0
    emotion_decay_rate: float = 0.1  # How quickly emotion influence fades


class VoicePersonalityAdapter:
    """Adapts voice characteristics based on personality and emotional state"""

    def __init__(self):
        self.adaptation_rules = self._initialize_adaptation_rules()
        self.emotion_mappings = self._initialize_emotion_mappings()
        self.voice_profiles: Dict[str, PersonalityVoiceProfile] = {}

    def _initialize_adaptation_rules(self) -> List[VoiceAdaptationRule]:
        """Initialize default personality-to-voice adaptation rules"""
        return [
            # Extroversion affects energy and volume
            VoiceAdaptationRule(
                PersonalityTrait.EXTROVERSION,
                VoiceCharacteristic.ENERGY,
                0.3, 1.0, "exponential"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.EXTROVERSION,
                VoiceCharacteristic.VOLUME,
                0.4, 0.9, "linear"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.EXTROVERSION,
                VoiceCharacteristic.SPEED,
                0.8, 1.2, "linear"
            ),

            # Agreeableness affects warmth and stability
            VoiceAdaptationRule(
                PersonalityTrait.AGREEABLENESS,
                VoiceCharacteristic.WARMTH,
                0.2, 0.9, "logarithmic"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.AGREEABLENESS,
                VoiceCharacteristic.STABILITY,
                0.6, 0.9, "linear"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.AGREEABLENESS,
                VoiceCharacteristic.PITCH,
                0.8, 1.1, "linear"
            ),

            # Conscientiousness affects stability and consistency
            VoiceAdaptationRule(
                PersonalityTrait.CONSCIENTIOUSNESS,
                VoiceCharacteristic.STABILITY,
                0.4, 0.95, "linear"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.CONSCIENTIOUSNESS,
                VoiceCharacteristic.SIMILARITY_BOOST,
                0.6, 0.9, "linear"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.CONSCIENTIOUSNESS,
                VoiceCharacteristic.SPEED,
                0.9, 1.0, "linear"
            ),

            # Neuroticism affects stability and energy (inverted)
            VoiceAdaptationRule(
                PersonalityTrait.NEUROTICISM,
                VoiceCharacteristic.STABILITY,
                0.3, 0.8, "linear", invert=True
            ),
            VoiceAdaptationRule(
                PersonalityTrait.NEUROTICISM,
                VoiceCharacteristic.ENERGY,
                0.2, 0.7, "exponential", invert=True
            ),

            # Openness affects style and expressiveness
            VoiceAdaptationRule(
                PersonalityTrait.OPENNESS,
                VoiceCharacteristic.STYLE,
                0.0, 0.8, "exponential"
            ),
            VoiceAdaptationRule(
                PersonalityTrait.OPENNESS,
                VoiceCharacteristic.ENERGY,
                0.4, 0.9, "linear"
            ),
        ]

    def _initialize_emotion_mappings(self) -> List[EmotionVoiceMapping]:
        """Initialize emotion-to-voice characteristic mappings"""
        return [
            EmotionVoiceMapping(
                EmotionType.JOY,
                {
                    VoiceCharacteristic.ENERGY: 0.3,
                    VoiceCharacteristic.SPEED: 0.1,
                    VoiceCharacteristic.PITCH: 0.15,
                    VoiceCharacteristic.WARMTH: 0.2
                }
            ),
            EmotionVoiceMapping(
                EmotionType.SADNESS,
                {
                    VoiceCharacteristic.ENERGY: -0.4,
                    VoiceCharacteristic.SPEED: -0.2,
                    VoiceCharacteristic.PITCH: -0.1,
                    VoiceCharacteristic.VOLUME: -0.15
                }
            ),
            EmotionVoiceMapping(
                EmotionType.ANGER,
                {
                    VoiceCharacteristic.ENERGY: 0.4,
                    VoiceCharacteristic.STABILITY: -0.2,
                    VoiceCharacteristic.VOLUME: 0.2,
                    VoiceCharacteristic.SPEED: 0.15
                }
            ),
            EmotionVoiceMapping(
                EmotionType.FEAR,
                {
                    VoiceCharacteristic.STABILITY: -0.3,
                    VoiceCharacteristic.ENERGY: -0.2,
                    VoiceCharacteristic.PITCH: 0.2,
                    VoiceCharacteristic.SPEED: 0.1
                }
            ),
            EmotionVoiceMapping(
                EmotionType.SURPRISE,
                {
                    VoiceCharacteristic.PITCH: 0.25,
                    VoiceCharacteristic.ENERGY: 0.2,
                    VoiceCharacteristic.SPEED: 0.1,
                    VoiceCharacteristic.STYLE: 0.15
                }
            ),
            EmotionVoiceMapping(
                EmotionType.DISGUST,
                {
                    VoiceCharacteristic.ENERGY: -0.2,
                    VoiceCharacteristic.WARMTH: -0.3,
                    VoiceCharacteristic.SPEED: -0.1
                }
            ),
            EmotionVoiceMapping(
                EmotionType.NEUTRAL,
                {}  # No adjustments for neutral emotion
            )
        ]

    def create_personality_voice_profile(
        self,
        base_profile: VoiceProfile,
        personality: BigFivePersonality,
        agent_id: str,
        adaptation_intensity: AdaptationIntensity = AdaptationIntensity.MODERATE
    ) -> PersonalityVoiceProfile:
        """Create a personality-adapted voice profile"""
        try:
            # Calculate personality-based adjustments
            personality_adjustments = self._calculate_personality_adjustments(
                personality, adaptation_intensity
            )

            # Create personality voice profile
            personality_profile = PersonalityVoiceProfile(
                base_profile=base_profile,
                personality_adjustments=personality_adjustments,
                adaptation_intensity=adaptation_intensity,
                extroversion_influence=personality.extroversion,
                agreeableness_influence=personality.agreeableness,
                conscientiousness_influence=personality.conscientiousness,
                neuroticism_influence=personality.neuroticism,
                openness_influence=personality.openness
            )

            # Store profile
            self.voice_profiles[agent_id] = personality_profile

            logger.info(f"Created personality voice profile for agent {agent_id}")
            return personality_profile

        except Exception as e:
            logger.error(f"Failed to create personality voice profile: {e}")
            raise

    def _calculate_personality_adjustments(
        self,
        personality: BigFivePersonality,
        intensity: AdaptationIntensity
    ) -> Dict[VoiceCharacteristic, float]:
        """Calculate voice adjustments based on personality traits"""
        adjustments = {}
        intensity_multiplier = self._get_intensity_multiplier(intensity)

        for rule in self.adaptation_rules:
            # Get trait value (0.0 to 1.0)
            trait_value = getattr(personality, rule.trait.value)

            # Apply curve transformation
            if rule.curve_type == "exponential":
                transformed_value = math.pow(trait_value, 2)
            elif rule.curve_type == "logarithmic":
                transformed_value = math.log(trait_value + 0.1) / math.log(1.1)
                transformed_value = max(0.0, min(1.0, transformed_value))
            else:  # linear
                transformed_value = trait_value

            # Invert if specified
            if rule.invert:
                transformed_value = 1.0 - transformed_value

            # Map to characteristic range
            adjustment_value = (
                rule.min_value +
                (rule.max_value - rule.min_value) * transformed_value
            )

            # Apply intensity multiplier
            if rule.characteristic in adjustments:
                # Average multiple rules affecting same characteristic
                existing_value = adjustments[rule.characteristic]
                adjustments[rule.characteristic] = (existing_value + adjustment_value) / 2
            else:
                adjustments[rule.characteristic] = adjustment_value

            # Apply intensity scaling
            base_value = 0.5  # Neutral baseline
            adjustment_from_base = adjustments[rule.characteristic] - base_value
            adjustments[rule.characteristic] = (
                base_value + adjustment_from_base * intensity_multiplier
            )

        return adjustments

    def _get_intensity_multiplier(self, intensity: AdaptationIntensity) -> float:
        """Get multiplier for adaptation intensity"""
        multipliers = {
            AdaptationIntensity.SUBTLE: 0.2,
            AdaptationIntensity.MODERATE: 0.5,
            AdaptationIntensity.STRONG: 0.8,
            AdaptationIntensity.EXTREME: 1.0
        }
        return multipliers.get(intensity, 0.5)

    def adapt_voice_for_emotion(
        self,
        agent_id: str,
        emotional_state: EmotionalState,
        decay_previous: bool = True
    ) -> Optional[PersonalityVoiceProfile]:
        """Adapt voice profile based on current emotional state"""
        if agent_id not in self.voice_profiles:
            logger.warning(f"No voice profile found for agent {agent_id}")
            return None

        try:
            profile = self.voice_profiles[agent_id]

            # Decay previous emotion influence
            if decay_previous:
                for characteristic in profile.emotion_adjustments:
                    profile.emotion_adjustments[characteristic] *= (
                        1.0 - profile.emotion_decay_rate
                    )

            # Find emotion mapping
            emotion_mapping = None
            for mapping in self.emotion_mappings:
                if mapping.emotion == emotional_state.primary_emotion:
                    emotion_mapping = mapping
                    break

            if emotion_mapping:
                # Apply emotion adjustments
                intensity_factor = emotional_state.intensity * emotion_mapping.intensity_multiplier

                for characteristic, adjustment in emotion_mapping.voice_adjustments.items():
                    scaled_adjustment = adjustment * intensity_factor

                    if characteristic in profile.emotion_adjustments:
                        # Blend with existing emotion adjustments
                        profile.emotion_adjustments[characteristic] = (
                            profile.emotion_adjustments[characteristic] * 0.7 +
                            scaled_adjustment * 0.3
                        )
                    else:
                        profile.emotion_adjustments[characteristic] = scaled_adjustment

                profile.current_emotion_influence = emotional_state.intensity

                logger.debug(f"Adapted voice for emotion {emotional_state.primary_emotion} (intensity: {emotional_state.intensity})")

            return profile

        except Exception as e:
            logger.error(f"Failed to adapt voice for emotion: {e}")
            return self.voice_profiles.get(agent_id)

    def generate_adapted_voice_profile(
        self,
        agent_id: str,
        emotional_state: Optional[EmotionalState] = None
    ) -> Optional[VoiceProfile]:
        """Generate final adapted voice profile combining personality and emotion"""
        if agent_id not in self.voice_profiles:
            return None

        try:
            personality_profile = self.voice_profiles[agent_id]

            # Apply emotion adaptation if provided
            if emotional_state:
                personality_profile = self.adapt_voice_for_emotion(
                    agent_id, emotional_state
                ) or personality_profile

            # Create adapted voice profile
            adapted_profile = VoiceProfile(
                voice_id=personality_profile.base_profile.voice_id,
                name=personality_profile.base_profile.name,
                gender=personality_profile.base_profile.gender,
                age=personality_profile.base_profile.age,
                accent=personality_profile.base_profile.accent,
                description=personality_profile.base_profile.description,
                preview_url=personality_profile.base_profile.preview_url,

                # Apply personality and emotion adjustments
                stability=self._apply_adjustments(
                    personality_profile.base_profile.stability,
                    personality_profile.personality_adjustments.get(VoiceCharacteristic.STABILITY, 0.0),
                    personality_profile.emotion_adjustments.get(VoiceCharacteristic.STABILITY, 0.0)
                ),
                similarity_boost=self._apply_adjustments(
                    personality_profile.base_profile.similarity_boost,
                    personality_profile.personality_adjustments.get(VoiceCharacteristic.SIMILARITY_BOOST, 0.0),
                    personality_profile.emotion_adjustments.get(VoiceCharacteristic.SIMILARITY_BOOST, 0.0)
                ),
                style=self._apply_adjustments(
                    personality_profile.base_profile.style,
                    personality_profile.personality_adjustments.get(VoiceCharacteristic.STYLE, 0.0),
                    personality_profile.emotion_adjustments.get(VoiceCharacteristic.STYLE, 0.0)
                ),
                use_speaker_boost=personality_profile.base_profile.use_speaker_boost
            )

            logger.debug(f"Generated adapted voice profile for agent {agent_id}")
            return adapted_profile

        except Exception as e:
            logger.error(f"Failed to generate adapted voice profile: {e}")
            return None

    def _apply_adjustments(
        self,
        base_value: float,
        personality_adjustment: float,
        emotion_adjustment: float,
        blend_ratio: float = 0.7  # 70% personality, 30% emotion
    ) -> float:
        """Apply personality and emotion adjustments to base voice value"""
        # Combine personality and emotion adjustments
        total_adjustment = (
            personality_adjustment * blend_ratio +
            emotion_adjustment * (1.0 - blend_ratio)
        )

        # Apply adjustment to base value
        adjusted_value = base_value + (total_adjustment - 0.5) * 0.4  # Scale to reasonable range

        # Clamp to valid range [0.0, 1.0]
        return max(0.0, min(1.0, adjusted_value))

    def get_voice_characteristics_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of current voice characteristics for an agent"""
        if agent_id not in self.voice_profiles:
            return None

        profile = self.voice_profiles[agent_id]

        return {
            "agent_id": agent_id,
            "base_voice": {
                "voice_id": profile.base_profile.voice_id,
                "name": profile.base_profile.name,
                "gender": profile.base_profile.gender.value,
                "age": profile.base_profile.age.value
            },
            "personality_influence": {
                "extroversion": profile.extroversion_influence,
                "agreeableness": profile.agreeableness_influence,
                "conscientiousness": profile.conscientiousness_influence,
                "neuroticism": profile.neuroticism_influence,
                "openness": profile.openness_influence
            },
            "current_adjustments": {
                "personality": dict(profile.personality_adjustments),
                "emotion": dict(profile.emotion_adjustments)
            },
            "adaptation_intensity": profile.adaptation_intensity.value,
            "emotion_influence": profile.current_emotion_influence
        }

    def update_adaptation_intensity(
        self,
        agent_id: str,
        new_intensity: AdaptationIntensity
    ) -> bool:
        """Update adaptation intensity for an agent"""
        if agent_id not in self.voice_profiles:
            return False

        try:
            profile = self.voice_profiles[agent_id]
            old_intensity = profile.adaptation_intensity

            # Recalculate personality adjustments with new intensity
            if hasattr(profile, '_base_personality'):
                personality_adjustments = self._calculate_personality_adjustments(
                    profile._base_personality, new_intensity
                )
                profile.personality_adjustments = personality_adjustments

            profile.adaptation_intensity = new_intensity

            logger.info(f"Updated adaptation intensity for agent {agent_id}: {old_intensity.value} -> {new_intensity.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update adaptation intensity: {e}")
            return False

    def remove_voice_profile(self, agent_id: str) -> bool:
        """Remove voice profile for an agent"""
        if agent_id in self.voice_profiles:
            del self.voice_profiles[agent_id]
            logger.info(f"Removed voice profile for agent {agent_id}")
            return True
        return False

    def list_voice_profiles(self) -> List[str]:
        """List all agent IDs with voice profiles"""
        return list(self.voice_profiles.keys())


# Factory function
def create_personality_adapter() -> VoicePersonalityAdapter:
    """Create and initialize voice personality adapter"""
    return VoicePersonalityAdapter()