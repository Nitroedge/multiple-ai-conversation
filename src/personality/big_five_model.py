"""
Big Five Personality Model Implementation
Implements the Five-Factor Model (FFM) for dynamic character personalities
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import math
from datetime import datetime, timedelta


class PersonalityTrait(Enum):
    """Big Five personality traits"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class PersonalityFacet(Enum):
    """Sub-facets of Big Five traits for more granular personality modeling"""
    # Openness facets
    IMAGINATION = "imagination"
    ARTISTIC_INTERESTS = "artistic_interests"
    EMOTIONALITY = "emotionality"
    ADVENTUROUSNESS = "adventurousness"
    INTELLECT = "intellect"
    LIBERALISM = "liberalism"

    # Conscientiousness facets
    SELF_EFFICACY = "self_efficacy"
    ORDERLINESS = "orderliness"
    DUTIFULNESS = "dutifulness"
    ACHIEVEMENT_STRIVING = "achievement_striving"
    SELF_DISCIPLINE = "self_discipline"
    CAUTIOUSNESS = "cautiousness"

    # Extraversion facets
    FRIENDLINESS = "friendliness"
    GREGARIOUSNESS = "gregariousness"
    ASSERTIVENESS = "assertiveness"
    ACTIVITY_LEVEL = "activity_level"
    EXCITEMENT_SEEKING = "excitement_seeking"
    CHEERFULNESS = "cheerfulness"

    # Agreeableness facets
    TRUST = "trust"
    MORALITY = "morality"
    ALTRUISM = "altruism"
    COOPERATION = "cooperation"
    MODESTY = "modesty"
    SYMPATHY = "sympathy"

    # Neuroticism facets
    ANXIETY = "anxiety"
    ANGER = "anger"
    DEPRESSION = "depression"
    SELF_CONSCIOUSNESS = "self_consciousness"
    IMMODERATION = "immoderation"
    VULNERABILITY = "vulnerability"


@dataclass
class PersonalityScore:
    """Individual personality trait score with metadata"""
    trait: PersonalityTrait
    score: float  # 0.0 to 1.0 (normalized)
    confidence: float  # 0.0 to 1.0
    last_updated: datetime
    update_count: int = 0
    adaptation_rate: float = 0.1  # How quickly this trait adapts

    def __post_init__(self):
        """Validate score ranges"""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.adaptation_rate = max(0.01, min(0.5, self.adaptation_rate))

    def update_score(self, new_evidence: float, evidence_weight: float = 1.0) -> None:
        """Update personality score based on new behavioral evidence"""
        # Weighted average with existing score
        weight_factor = evidence_weight * self.adaptation_rate
        self.score = (self.score * (1 - weight_factor)) + (new_evidence * weight_factor)

        # Update metadata
        self.confidence = min(1.0, self.confidence + 0.01)  # Gradual confidence increase
        self.last_updated = datetime.utcnow()
        self.update_count += 1

        # Normalize score
        self.score = max(0.0, min(1.0, self.score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trait": self.trait.value,
            "score": self.score,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
            "update_count": self.update_count,
            "adaptation_rate": self.adaptation_rate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityScore':
        """Create from dictionary"""
        return cls(
            trait=PersonalityTrait(data["trait"]),
            score=data["score"],
            confidence=data["confidence"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            update_count=data.get("update_count", 0),
            adaptation_rate=data.get("adaptation_rate", 0.1)
        )


@dataclass
class PersonalityProfile:
    """Complete Big Five personality profile for an agent"""
    agent_id: str
    traits: Dict[PersonalityTrait, PersonalityScore]
    facets: Dict[PersonalityFacet, float] = field(default_factory=dict)
    personality_descriptors: List[str] = field(default_factory=list)
    behavioral_patterns: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_adaptation: datetime = field(default_factory=datetime.utcnow)
    adaptation_enabled: bool = True

    def __post_init__(self):
        """Initialize facets from traits if not provided"""
        if not self.facets:
            self._initialize_facets()
        self._update_descriptors()

    def _initialize_facets(self) -> None:
        """Initialize personality facets based on main trait scores"""
        facet_mappings = {
            PersonalityTrait.OPENNESS: [
                PersonalityFacet.IMAGINATION, PersonalityFacet.ARTISTIC_INTERESTS,
                PersonalityFacet.EMOTIONALITY, PersonalityFacet.ADVENTUROUSNESS,
                PersonalityFacet.INTELLECT, PersonalityFacet.LIBERALISM
            ],
            PersonalityTrait.CONSCIENTIOUSNESS: [
                PersonalityFacet.SELF_EFFICACY, PersonalityFacet.ORDERLINESS,
                PersonalityFacet.DUTIFULNESS, PersonalityFacet.ACHIEVEMENT_STRIVING,
                PersonalityFacet.SELF_DISCIPLINE, PersonalityFacet.CAUTIOUSNESS
            ],
            PersonalityTrait.EXTRAVERSION: [
                PersonalityFacet.FRIENDLINESS, PersonalityFacet.GREGARIOUSNESS,
                PersonalityFacet.ASSERTIVENESS, PersonalityFacet.ACTIVITY_LEVEL,
                PersonalityFacet.EXCITEMENT_SEEKING, PersonalityFacet.CHEERFULNESS
            ],
            PersonalityTrait.AGREEABLENESS: [
                PersonalityFacet.TRUST, PersonalityFacet.MORALITY,
                PersonalityFacet.ALTRUISM, PersonalityFacet.COOPERATION,
                PersonalityFacet.MODESTY, PersonalityFacet.SYMPATHY
            ],
            PersonalityTrait.NEUROTICISM: [
                PersonalityFacet.ANXIETY, PersonalityFacet.ANGER,
                PersonalityFacet.DEPRESSION, PersonalityFacet.SELF_CONSCIOUSNESS,
                PersonalityFacet.IMMODERATION, PersonalityFacet.VULNERABILITY
            ]
        }

        for trait, facet_list in facet_mappings.items():
            if trait in self.traits:
                base_score = self.traits[trait].score
                for facet in facet_list:
                    # Add some variation to facets around the base trait score
                    variation = (hash(f"{self.agent_id}_{facet.value}") % 40 - 20) / 100
                    self.facets[facet] = max(0.0, min(1.0, base_score + variation))

    def _update_descriptors(self) -> None:
        """Update personality descriptors based on current trait scores"""
        self.personality_descriptors = []

        for trait, score_obj in self.traits.items():
            score = score_obj.score

            if trait == PersonalityTrait.OPENNESS:
                if score > 0.7:
                    self.personality_descriptors.extend(["creative", "imaginative", "curious", "open-minded"])
                elif score < 0.3:
                    self.personality_descriptors.extend(["practical", "conventional", "focused", "traditional"])

            elif trait == PersonalityTrait.CONSCIENTIOUSNESS:
                if score > 0.7:
                    self.personality_descriptors.extend(["organized", "disciplined", "reliable", "goal-oriented"])
                elif score < 0.3:
                    self.personality_descriptors.extend(["spontaneous", "flexible", "carefree", "adaptable"])

            elif trait == PersonalityTrait.EXTRAVERSION:
                if score > 0.7:
                    self.personality_descriptors.extend(["outgoing", "energetic", "talkative", "sociable"])
                elif score < 0.3:
                    self.personality_descriptors.extend(["reserved", "quiet", "introspective", "independent"])

            elif trait == PersonalityTrait.AGREEABLENESS:
                if score > 0.7:
                    self.personality_descriptors.extend(["friendly", "compassionate", "cooperative", "trusting"])
                elif score < 0.3:
                    self.personality_descriptors.extend(["competitive", "skeptical", "direct", "challenging"])

            elif trait == PersonalityTrait.NEUROTICISM:
                if score > 0.7:
                    self.personality_descriptors.extend(["sensitive", "emotional", "reactive", "stress-prone"])
                elif score < 0.3:
                    self.personality_descriptors.extend(["calm", "stable", "resilient", "composed"])

    def get_trait_score(self, trait: PersonalityTrait) -> float:
        """Get current score for a specific trait"""
        return self.traits.get(trait, PersonalityScore(trait, 0.5, 0.0, datetime.utcnow())).score

    def get_facet_score(self, facet: PersonalityFacet) -> float:
        """Get current score for a specific facet"""
        return self.facets.get(facet, 0.5)

    def update_from_behavior(self, behavioral_evidence: Dict[PersonalityTrait, float]) -> None:
        """Update personality based on observed behavior"""
        if not self.adaptation_enabled:
            return

        for trait, evidence in behavioral_evidence.items():
            if trait in self.traits:
                self.traits[trait].update_score(evidence)

        self.last_adaptation = datetime.utcnow()
        self._update_descriptors()

    def get_personality_summary(self) -> Dict[str, Any]:
        """Get a comprehensive personality summary"""
        return {
            "agent_id": self.agent_id,
            "traits": {trait.value: score.score for trait, score in self.traits.items()},
            "dominant_traits": self._get_dominant_traits(),
            "descriptors": self.personality_descriptors[:10],  # Top 10 descriptors
            "adaptation_level": self._calculate_adaptation_level(),
            "personality_type": self._classify_personality_type(),
            "last_updated": self.last_adaptation.isoformat()
        }

    def _get_dominant_traits(self) -> List[str]:
        """Get the most prominent personality traits"""
        sorted_traits = sorted(
            self.traits.items(),
            key=lambda x: abs(x[1].score - 0.5),  # Distance from neutral
            reverse=True
        )

        dominant = []
        for trait, score_obj in sorted_traits[:3]:
            if score_obj.score > 0.6:
                dominant.append(f"high_{trait.value}")
            elif score_obj.score < 0.4:
                dominant.append(f"low_{trait.value}")

        return dominant

    def _calculate_adaptation_level(self) -> str:
        """Calculate how much the personality has adapted"""
        total_updates = sum(score.update_count for score in self.traits.values())

        if total_updates < 10:
            return "minimal"
        elif total_updates < 50:
            return "moderate"
        elif total_updates < 100:
            return "significant"
        else:
            return "extensive"

    def _classify_personality_type(self) -> str:
        """Classify personality into a general type"""
        o = self.get_trait_score(PersonalityTrait.OPENNESS)
        c = self.get_trait_score(PersonalityTrait.CONSCIENTIOUSNESS)
        e = self.get_trait_score(PersonalityTrait.EXTRAVERSION)
        a = self.get_trait_score(PersonalityTrait.AGREEABLENESS)
        n = self.get_trait_score(PersonalityTrait.NEUROTICISM)

        # Simplified personality type classification
        if e > 0.6 and a > 0.6:
            return "enthusiast"  # High extraversion + agreeableness
        elif c > 0.6 and n < 0.4:
            return "achiever"  # High conscientiousness + low neuroticism
        elif o > 0.6 and e > 0.5:
            return "innovator"  # High openness + moderate+ extraversion
        elif a > 0.6 and c > 0.6:
            return "supporter"  # High agreeableness + conscientiousness
        elif n > 0.6:
            return "sensitive"  # High neuroticism
        elif e < 0.4 and o > 0.5:
            return "thinker"  # Low extraversion + moderate+ openness
        else:
            return "balanced"  # No dominant pattern

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "traits": {trait.value: score.to_dict() for trait, score in self.traits.items()},
            "facets": {facet.value: score for facet, score in self.facets.items()},
            "personality_descriptors": self.personality_descriptors,
            "behavioral_patterns": self.behavioral_patterns,
            "created_at": self.created_at.isoformat(),
            "last_adaptation": self.last_adaptation.isoformat(),
            "adaptation_enabled": self.adaptation_enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityProfile':
        """Create from dictionary"""
        traits = {
            PersonalityTrait(trait_name): PersonalityScore.from_dict(score_data)
            for trait_name, score_data in data["traits"].items()
        }

        facets = {
            PersonalityFacet(facet_name): score
            for facet_name, score in data.get("facets", {}).items()
        }

        return cls(
            agent_id=data["agent_id"],
            traits=traits,
            facets=facets,
            personality_descriptors=data.get("personality_descriptors", []),
            behavioral_patterns=data.get("behavioral_patterns", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_adaptation=datetime.fromisoformat(data["last_adaptation"]),
            adaptation_enabled=data.get("adaptation_enabled", True)
        )


class PersonalityEngine:
    """Engine for managing and evolving agent personalities"""

    def __init__(self):
        self.agent_profiles: Dict[str, PersonalityProfile] = {}
        self.behavioral_analyzers: Dict[str, callable] = self._initialize_analyzers()

    def _initialize_analyzers(self) -> Dict[str, callable]:
        """Initialize behavioral analysis functions"""
        return {
            "response_length": self._analyze_response_length,
            "emotional_content": self._analyze_emotional_content,
            "question_asking": self._analyze_question_asking,
            "topic_diversity": self._analyze_topic_diversity,
            "social_references": self._analyze_social_references,
            "certainty_confidence": self._analyze_certainty_confidence,
            "creative_language": self._analyze_creative_language,
            "structured_thinking": self._analyze_structured_thinking
        }

    def create_personality_profile(self, agent_id: str, initial_traits: Optional[Dict[str, float]] = None) -> PersonalityProfile:
        """Create a new personality profile for an agent"""
        if initial_traits is None:
            # Default personality traits for different agent types
            agent_defaults = {
                "OSWALD": {
                    PersonalityTrait.OPENNESS: 0.9,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.6,
                    PersonalityTrait.EXTRAVERSION: 0.8,
                    PersonalityTrait.AGREEABLENESS: 0.8,
                    PersonalityTrait.NEUROTICISM: 0.3
                },
                "TONY_KING": {
                    PersonalityTrait.OPENNESS: 0.6,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.5,
                    PersonalityTrait.EXTRAVERSION: 0.9,
                    PersonalityTrait.AGREEABLENESS: 0.7,
                    PersonalityTrait.NEUROTICISM: 0.4
                },
                "VICTORIA": {
                    PersonalityTrait.OPENNESS: 0.8,
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
                    PersonalityTrait.EXTRAVERSION: 0.5,
                    PersonalityTrait.AGREEABLENESS: 0.6,
                    PersonalityTrait.NEUROTICISM: 0.2
                }
            }

            trait_values = agent_defaults.get(agent_id, {
                PersonalityTrait.OPENNESS: 0.6,
                PersonalityTrait.CONSCIENTIOUSNESS: 0.6,
                PersonalityTrait.EXTRAVERSION: 0.6,
                PersonalityTrait.AGREEABLENESS: 0.6,
                PersonalityTrait.NEUROTICISM: 0.4
            })
        else:
            trait_values = {PersonalityTrait(k): v for k, v in initial_traits.items()}

        # Create PersonalityScore objects
        traits = {
            trait: PersonalityScore(
                trait=trait,
                score=score,
                confidence=0.5,  # Initial moderate confidence
                last_updated=datetime.utcnow()
            )
            for trait, score in trait_values.items()
        }

        profile = PersonalityProfile(agent_id=agent_id, traits=traits)
        self.agent_profiles[agent_id] = profile

        return profile

    def get_personality_profile(self, agent_id: str) -> Optional[PersonalityProfile]:
        """Get personality profile for an agent"""
        return self.agent_profiles.get(agent_id)

    def analyze_message_for_personality(self, agent_id: str, message: str, context: Dict[str, Any] = None) -> Dict[PersonalityTrait, float]:
        """Analyze a message to extract personality behavioral evidence"""
        evidence = {}

        for analyzer_name, analyzer_func in self.behavioral_analyzers.items():
            try:
                analysis_result = analyzer_func(message, context or {})

                # Map analysis results to personality traits
                trait_mapping = self._map_analysis_to_traits(analyzer_name, analysis_result)

                for trait, strength in trait_mapping.items():
                    if trait not in evidence:
                        evidence[trait] = []
                    evidence[trait].append(strength)
            except Exception as e:
                # Log error but continue with other analyzers
                continue

        # Average the evidence for each trait
        final_evidence = {}
        for trait, strengths in evidence.items():
            if strengths:
                final_evidence[trait] = sum(strengths) / len(strengths)

        return final_evidence

    def update_personality_from_message(self, agent_id: str, message: str, context: Dict[str, Any] = None) -> None:
        """Update agent personality based on message analysis"""
        profile = self.get_personality_profile(agent_id)
        if not profile:
            profile = self.create_personality_profile(agent_id)

        evidence = self.analyze_message_for_personality(agent_id, message, context)
        if evidence:
            profile.update_from_behavior(evidence)

    def _analyze_response_length(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze response length (relates to extraversion)"""
        word_count = len(message.split())

        if word_count > 50:
            return 0.8  # High extraversion evidence
        elif word_count < 10:
            return 0.2  # Low extraversion evidence
        else:
            return 0.5  # Neutral

    def _analyze_emotional_content(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze emotional language (relates to neuroticism and agreeableness)"""
        emotional_words = [
            "excited", "amazing", "wonderful", "terrible", "awful", "fantastic",
            "worried", "anxious", "happy", "sad", "angry", "frustrated", "love", "hate"
        ]

        message_lower = message.lower()
        emotional_count = sum(1 for word in emotional_words if word in message_lower)

        # Normalize by message length
        emotion_density = emotional_count / max(len(message.split()), 1)

        return min(1.0, emotion_density * 10)  # Scale up

    def _analyze_question_asking(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze question frequency (relates to openness and agreeableness)"""
        question_count = message.count('?')
        message_length = len(message.split())

        if message_length == 0:
            return 0.5

        question_density = question_count / message_length
        return min(1.0, question_density * 20)  # Scale up

    def _analyze_topic_diversity(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze topic diversity (relates to openness)"""
        # Simple heuristic: count unique concepts/topics mentioned
        topics = ["technology", "science", "art", "music", "politics", "philosophy",
                 "travel", "food", "sports", "history", "nature", "culture"]

        message_lower = message.lower()
        topic_count = sum(1 for topic in topics if topic in message_lower)

        return min(1.0, topic_count / 3)  # Normalize

    def _analyze_social_references(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze social references (relates to extraversion and agreeableness)"""
        social_words = ["we", "us", "together", "everyone", "people", "friends",
                       "community", "team", "group", "others", "someone", "anyone"]

        message_lower = message.lower()
        social_count = sum(1 for word in social_words if word in message_lower)

        return min(1.0, social_count / 5)  # Normalize

    def _analyze_certainty_confidence(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze certainty and confidence (relates to neuroticism and conscientiousness)"""
        uncertain_words = ["maybe", "perhaps", "might", "could", "possibly", "uncertain", "unsure"]
        confident_words = ["definitely", "certainly", "absolutely", "sure", "confident", "know"]

        message_lower = message.lower()
        uncertain_count = sum(1 for word in uncertain_words if word in message_lower)
        confident_count = sum(1 for word in confident_words if word in message_lower)

        # Return confidence level (high = low neuroticism)
        if confident_count > uncertain_count:
            return 0.3  # Low neuroticism evidence
        elif uncertain_count > confident_count:
            return 0.7  # High neuroticism evidence
        else:
            return 0.5  # Neutral

    def _analyze_creative_language(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze creative language use (relates to openness)"""
        creative_indicators = ["imagine", "creative", "innovative", "unique", "original",
                              "metaphor", "analogy", "artistic", "inspiration"]

        message_lower = message.lower()
        creative_count = sum(1 for word in creative_indicators if word in message_lower)

        # Also look for metaphorical language patterns
        metaphor_patterns = ["like a", "as if", "reminds me of", "similar to"]
        metaphor_count = sum(1 for pattern in metaphor_patterns if pattern in message_lower)

        total_creative = creative_count + metaphor_count
        return min(1.0, total_creative / 3)  # Normalize

    def _analyze_structured_thinking(self, message: str, context: Dict[str, Any]) -> float:
        """Analyze structured thinking patterns (relates to conscientiousness)"""
        structure_indicators = ["first", "second", "third", "finally", "in conclusion",
                               "step by step", "organize", "plan", "systematic", "method"]

        message_lower = message.lower()
        structure_count = sum(1 for word in structure_indicators if word in message_lower)

        # Look for numbered or bulleted lists
        import re
        list_patterns = re.findall(r'^\d+\.|^\*|^-', message, re.MULTILINE)

        total_structure = structure_count + len(list_patterns)
        return min(1.0, total_structure / 3)  # Normalize

    def _map_analysis_to_traits(self, analyzer_name: str, analysis_result: float) -> Dict[PersonalityTrait, float]:
        """Map analysis results to personality traits"""
        mappings = {
            "response_length": {
                PersonalityTrait.EXTRAVERSION: analysis_result
            },
            "emotional_content": {
                PersonalityTrait.NEUROTICISM: analysis_result,
                PersonalityTrait.AGREEABLENESS: analysis_result * 0.5
            },
            "question_asking": {
                PersonalityTrait.OPENNESS: analysis_result,
                PersonalityTrait.AGREEABLENESS: analysis_result * 0.7
            },
            "topic_diversity": {
                PersonalityTrait.OPENNESS: analysis_result
            },
            "social_references": {
                PersonalityTrait.EXTRAVERSION: analysis_result,
                PersonalityTrait.AGREEABLENESS: analysis_result * 0.8
            },
            "certainty_confidence": {
                PersonalityTrait.NEUROTICISM: analysis_result,
                PersonalityTrait.CONSCIENTIOUSNESS: 1.0 - analysis_result  # Inverse relationship
            },
            "creative_language": {
                PersonalityTrait.OPENNESS: analysis_result
            },
            "structured_thinking": {
                PersonalityTrait.CONSCIENTIOUSNESS: analysis_result
            }
        }

        return mappings.get(analyzer_name, {})

    def get_personality_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive personality insights for an agent"""
        profile = self.get_personality_profile(agent_id)
        if not profile:
            return {"error": "No personality profile found"}

        summary = profile.get_personality_summary()

        # Add additional insights
        insights = {
            **summary,
            "trait_details": {
                trait.value: {
                    "score": score.score,
                    "confidence": score.confidence,
                    "adaptation_count": score.update_count,
                    "interpretation": self._interpret_trait_score(trait, score.score)
                }
                for trait, score in profile.traits.items()
            },
            "behavioral_recommendations": self._generate_behavioral_recommendations(profile),
            "conversation_style": self._determine_conversation_style(profile)
        }

        return insights

    def _interpret_trait_score(self, trait: PersonalityTrait, score: float) -> str:
        """Interpret a trait score in human-readable terms"""
        interpretations = {
            PersonalityTrait.OPENNESS: {
                "high": "Highly creative, curious, and open to new experiences",
                "low": "Prefers familiar experiences and practical approaches",
                "medium": "Balanced between creativity and practicality"
            },
            PersonalityTrait.CONSCIENTIOUSNESS: {
                "high": "Highly organized, disciplined, and goal-oriented",
                "low": "Flexible, spontaneous, and adaptable",
                "medium": "Balanced approach to planning and spontaneity"
            },
            PersonalityTrait.EXTRAVERSION: {
                "high": "Outgoing, energetic, and enjoys social interaction",
                "low": "Reserved, introspective, and prefers smaller groups",
                "medium": "Comfortable in both social and solitary situations"
            },
            PersonalityTrait.AGREEABLENESS: {
                "high": "Cooperative, trusting, and empathetic",
                "low": "Competitive, skeptical, and direct",
                "medium": "Balanced between cooperation and assertiveness"
            },
            PersonalityTrait.NEUROTICISM: {
                "high": "Emotionally sensitive and reactive to stress",
                "low": "Emotionally stable and resilient",
                "medium": "Generally stable with occasional emotional responses"
            }
        }

        if score > 0.7:
            level = "high"
        elif score < 0.3:
            level = "low"
        else:
            level = "medium"

        return interpretations.get(trait, {}).get(level, "Moderate expression of this trait")

    def _generate_behavioral_recommendations(self, profile: PersonalityProfile) -> List[str]:
        """Generate behavioral recommendations based on personality"""
        recommendations = []

        # Based on dominant traits
        dominant_traits = profile._get_dominant_traits()

        for trait_desc in dominant_traits:
            if "high_extraversion" in trait_desc:
                recommendations.append("Engage in group conversations and collaborative activities")
            elif "low_extraversion" in trait_desc:
                recommendations.append("Provide thoughtful, in-depth responses with time to reflect")
            elif "high_openness" in trait_desc:
                recommendations.append("Introduce creative topics and explore new ideas")
            elif "high_conscientiousness" in trait_desc:
                recommendations.append("Provide structured information and clear action steps")
            elif "high_agreeableness" in trait_desc:
                recommendations.append("Use collaborative language and seek consensus")
            elif "high_neuroticism" in trait_desc:
                recommendations.append("Provide reassurance and emotional support")

        return recommendations[:5]  # Limit to top 5

    def _determine_conversation_style(self, profile: PersonalityProfile) -> Dict[str, str]:
        """Determine optimal conversation style based on personality"""
        style = {}

        # Communication pace
        extraversion = profile.get_trait_score(PersonalityTrait.EXTRAVERSION)
        if extraversion > 0.6:
            style["pace"] = "fast-paced and energetic"
        elif extraversion < 0.4:
            style["pace"] = "thoughtful and measured"
        else:
            style["pace"] = "moderate and adaptive"

        # Topic approach
        openness = profile.get_trait_score(PersonalityTrait.OPENNESS)
        if openness > 0.6:
            style["topics"] = "diverse and creative"
        else:
            style["topics"] = "focused and practical"

        # Interaction style
        agreeableness = profile.get_trait_score(PersonalityTrait.AGREEABLENESS)
        if agreeableness > 0.6:
            style["interaction"] = "collaborative and supportive"
        else:
            style["interaction"] = "direct and challenging"

        # Emotional tone
        neuroticism = profile.get_trait_score(PersonalityTrait.NEUROTICISM)
        if neuroticism > 0.6:
            style["tone"] = "emotionally expressive"
        else:
            style["tone"] = "calm and stable"

        return style