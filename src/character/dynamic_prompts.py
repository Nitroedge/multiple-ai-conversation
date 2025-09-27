"""
Dynamic prompt generation engine for personality-driven conversation
Context-aware prompt generation based on personality, memory, and conversation state
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import random
from datetime import datetime, timedelta

from ..personality.big_five_model import PersonalityProfile, PersonalityTrait
from .character_memory import CharacterMemoryManager, RelationshipMemory, BehavioralPattern


class PromptCategory(Enum):
    """Categories of prompt components"""
    CORE_PERSONALITY = "core_personality"
    CONVERSATION_STYLE = "conversation_style"
    RELATIONSHIP_CONTEXT = "relationship_context"
    EMOTIONAL_STATE = "emotional_state"
    BEHAVIORAL_GUIDANCE = "behavioral_guidance"
    MEMORY_CONTEXT = "memory_context"
    SITUATIONAL_CONTEXT = "situational_context"
    RESPONSE_FORMATTING = "response_formatting"


@dataclass
class PromptTemplate:
    """Template for generating dynamic prompts"""
    template_id: str
    template_name: str
    category: PromptCategory
    base_template: str
    variables: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-10, higher = more important
    personality_requirements: Optional[Dict[PersonalityTrait, Tuple[float, float]]] = None

    def matches_personality(self, personality_profile: PersonalityProfile) -> bool:
        """Check if this template matches the personality requirements"""
        if not self.personality_requirements:
            return True

        for trait, (min_score, max_score) in self.personality_requirements.items():
            trait_score = personality_profile.get_trait_score(trait)
            if not (min_score <= trait_score <= max_score):
                return False

        return True

    def render(self, variables: Dict[str, Any]) -> str:
        """Render the template with provided variables"""
        rendered = self.base_template

        for var_name in self.variables:
            placeholder = f"{{{var_name}}}"
            if var_name in variables:
                rendered = rendered.replace(placeholder, str(variables[var_name]))
            else:
                # Remove unused placeholders
                rendered = rendered.replace(placeholder, "")

        # Clean up extra whitespace
        return " ".join(rendered.split())


@dataclass
class ConversationContext:
    """Context information for prompt generation"""
    session_id: str
    conversation_stage: str  # "greeting", "discussion", "conclusion", "crisis"
    topic_focus: str
    last_speaker: Optional[str]
    turn_count: int
    user_input: str
    user_id: Optional[str] = None

    # Emotional context
    detected_user_emotion: Optional[str] = None
    conversation_tone: Optional[str] = None

    # Situational context
    time_of_day: Optional[str] = None
    conversation_length: Optional[int] = None
    previous_topics: List[str] = field(default_factory=list)


class DynamicPromptEngine:
    """Engine for generating dynamic, personality-driven prompts"""

    def __init__(self):
        self.prompt_templates: Dict[str, PromptTemplate] = {}
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        self.character_memories: Dict[str, CharacterMemoryManager] = {}

        self._initialize_default_templates()

    def register_personality_profile(self, agent_id: str, personality_profile: PersonalityProfile) -> None:
        """Register a personality profile for an agent"""
        self.personality_profiles[agent_id] = personality_profile

    def register_character_memory(self, agent_id: str, character_memory: CharacterMemoryManager) -> None:
        """Register character memory manager for an agent"""
        self.character_memories[agent_id] = character_memory

    def add_prompt_template(self, template: PromptTemplate) -> None:
        """Add a custom prompt template"""
        self.prompt_templates[template.template_id] = template

    def generate_dynamic_prompt(
        self,
        agent_id: str,
        context: ConversationContext,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a complete dynamic prompt for the agent"""

        personality_profile = self.personality_profiles.get(agent_id)
        character_memory = self.character_memories.get(agent_id)

        if not personality_profile:
            return self._generate_fallback_prompt(agent_id, context)

        # Build prompt components
        prompt_components = []

        # 1. Core personality description
        personality_component = self._generate_personality_component(personality_profile)
        prompt_components.append(personality_component)

        # 2. Conversation style guidance
        style_component = self._generate_style_component(personality_profile, context)
        prompt_components.append(style_component)

        # 3. Relationship context (if available)
        if character_memory and context.user_id:
            relationship_component = self._generate_relationship_component(
                character_memory, context.user_id, context
            )
            if relationship_component:
                prompt_components.append(relationship_component)

        # 4. Memory context
        if character_memory:
            memory_component = self._generate_memory_component(character_memory, context)
            if memory_component:
                prompt_components.append(memory_component)

        # 5. Behavioral guidance based on patterns
        if character_memory:
            behavioral_component = self._generate_behavioral_component(character_memory, context)
            if behavioral_component:
                prompt_components.append(behavioral_component)

        # 6. Situational context
        situational_component = self._generate_situational_component(context)
        prompt_components.append(situational_component)

        # 7. Response formatting guidelines
        formatting_component = self._generate_formatting_component(personality_profile, context)
        prompt_components.append(formatting_component)

        # Combine all components
        full_prompt = self._combine_prompt_components(prompt_components)

        return full_prompt

    def _generate_personality_component(self, personality_profile: PersonalityProfile) -> str:
        """Generate core personality component"""
        templates = self._get_templates_by_category(PromptCategory.CORE_PERSONALITY)
        suitable_templates = [t for t in templates if t.matches_personality(personality_profile)]

        if not suitable_templates:
            return self._fallback_personality_description(personality_profile)

        # Select template based on priority and personality fit
        selected_template = max(suitable_templates, key=lambda t: t.priority)

        variables = {
            "agent_name": personality_profile.agent_id,
            "personality_type": personality_profile._classify_personality_type(),
            "dominant_traits": ", ".join(personality_profile._get_dominant_traits()),
            "descriptors": ", ".join(personality_profile.personality_descriptors[:5]),
            "openness_level": self._trait_to_description(
                personality_profile.get_trait_score(PersonalityTrait.OPENNESS), "openness"
            ),
            "conscientiousness_level": self._trait_to_description(
                personality_profile.get_trait_score(PersonalityTrait.CONSCIENTIOUSNESS), "conscientiousness"
            ),
            "extraversion_level": self._trait_to_description(
                personality_profile.get_trait_score(PersonalityTrait.EXTRAVERSION), "extraversion"
            ),
            "agreeableness_level": self._trait_to_description(
                personality_profile.get_trait_score(PersonalityTrait.AGREEABLENESS), "agreeableness"
            ),
            "neuroticism_level": self._trait_to_description(
                personality_profile.get_trait_score(PersonalityTrait.NEUROTICISM), "neuroticism"
            )
        }

        return selected_template.render(variables)

    def _generate_style_component(self, personality_profile: PersonalityProfile, context: ConversationContext) -> str:
        """Generate conversation style component"""
        templates = self._get_templates_by_category(PromptCategory.CONVERSATION_STYLE)
        suitable_templates = [t for t in templates if t.matches_personality(personality_profile)]

        if not suitable_templates:
            return self._fallback_style_description(personality_profile)

        selected_template = max(suitable_templates, key=lambda t: t.priority)

        # Determine conversation style based on personality
        conversation_style = personality_profile._determine_conversation_style()

        variables = {
            "pace": conversation_style.get("pace", "moderate"),
            "topics": conversation_style.get("topics", "varied"),
            "interaction": conversation_style.get("interaction", "balanced"),
            "tone": conversation_style.get("tone", "neutral"),
            "conversation_stage": context.conversation_stage,
            "turn_count": str(context.turn_count)
        }

        return selected_template.render(variables)

    def _generate_relationship_component(
        self,
        character_memory: CharacterMemoryManager,
        user_id: str,
        context: ConversationContext
    ) -> Optional[str]:
        """Generate relationship context component"""
        relationship = character_memory.get_relationship_by_id(user_id)
        if not relationship:
            return None

        templates = self._get_templates_by_category(PromptCategory.RELATIONSHIP_CONTEXT)
        if not templates:
            return self._fallback_relationship_description(relationship)

        selected_template = templates[0]  # Use first available template

        variables = {
            "relationship_name": relationship.relationship_name,
            "relationship_quality": self._score_to_description(relationship.relationship_quality, "relationship"),
            "trust_level": self._score_to_description(relationship.trust_level, "trust"),
            "familiarity": self._score_to_description(relationship.familiarity_level, "familiarity"),
            "interaction_count": str(relationship.interaction_count),
            "shared_interests": ", ".join(relationship.shared_interests[:3]),
            "last_interaction": self._format_time_ago(relationship.last_interaction)
        }

        return selected_template.render(variables)

    def _generate_memory_component(self, character_memory: CharacterMemoryManager, context: ConversationContext) -> Optional[str]:
        """Generate memory context component"""
        templates = self._get_templates_by_category(PromptCategory.MEMORY_CONTEXT)
        if not templates:
            return None

        # Get relevant memories
        recent_topics = context.previous_topics[-3:] if context.previous_topics else []
        memory_insights = character_memory.get_character_development_summary()

        if not memory_insights or memory_insights["development_stats"]["total_trait_memories"] == 0:
            return None

        selected_template = templates[0]

        variables = {
            "recent_topics": ", ".join(recent_topics) if recent_topics else "none",
            "memory_count": str(memory_insights["development_stats"]["total_trait_memories"]),
            "growth_stage": memory_insights.get("behavioral_maturity", 0.5),
            "conversation_history": f"You have had {memory_insights['development_stats']['total_trait_memories']} significant interactions"
        }

        return selected_template.render(variables)

    def _generate_behavioral_component(self, character_memory: CharacterMemoryManager, context: ConversationContext) -> Optional[str]:
        """Generate behavioral guidance component"""
        behavioral_insights = character_memory.get_behavioral_insights()

        if not behavioral_insights.get("most_confident_patterns"):
            return None

        templates = self._get_templates_by_category(PromptCategory.BEHAVIORAL_GUIDANCE)
        if not templates:
            return None

        selected_template = templates[0]

        # Get top behavioral patterns
        top_patterns = behavioral_insights["most_confident_patterns"][:2]
        pattern_descriptions = [pattern["pattern_description"] for pattern in top_patterns]

        variables = {
            "established_patterns": "; ".join(pattern_descriptions),
            "behavioral_consistency": f"{behavioral_insights['behavioral_consistency']:.1%}",
            "pattern_count": str(len(behavioral_insights.get("patterns_by_type", {})))
        }

        return selected_template.render(variables)

    def _generate_situational_component(self, context: ConversationContext) -> str:
        """Generate situational context component"""
        templates = self._get_templates_by_category(PromptCategory.SITUATIONAL_CONTEXT)
        if not templates:
            return self._fallback_situational_description(context)

        selected_template = templates[0]

        # Determine time context
        time_context = context.time_of_day or self._get_current_time_context()

        variables = {
            "conversation_stage": context.conversation_stage,
            "topic_focus": context.topic_focus,
            "user_input": context.user_input,
            "time_context": time_context,
            "conversation_length": str(context.conversation_length or context.turn_count),
            "detected_emotion": context.detected_user_emotion or "neutral"
        }

        return selected_template.render(variables)

    def _generate_formatting_component(self, personality_profile: PersonalityProfile, context: ConversationContext) -> str:
        """Generate response formatting guidelines"""
        templates = self._get_templates_by_category(PromptCategory.RESPONSE_FORMATTING)
        if not templates:
            return self._fallback_formatting_guidelines(personality_profile)

        selected_template = templates[0]

        # Determine response length based on extraversion
        extraversion = personality_profile.get_trait_score(PersonalityTrait.EXTRAVERSION)
        if extraversion > 0.7:
            response_length = "moderately detailed"
        elif extraversion < 0.3:
            response_length = "concise and thoughtful"
        else:
            response_length = "balanced"

        variables = {
            "response_length": response_length,
            "personality_type": personality_profile._classify_personality_type(),
            "conversation_stage": context.conversation_stage
        }

        return selected_template.render(variables)

    def _combine_prompt_components(self, components: List[str]) -> str:
        """Combine prompt components into a cohesive prompt"""
        # Filter out empty components
        valid_components = [comp.strip() for comp in components if comp.strip()]

        # Create structured prompt
        prompt_sections = [
            "You are an AI character with a dynamic personality. Here is your current configuration:",
            "",
            *valid_components,
            "",
            "Please respond authentically based on this personality profile and context. Maintain consistency with your established behavioral patterns while being natural and engaging."
        ]

        return "\n".join(prompt_sections)

    def _get_templates_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """Get all templates for a specific category"""
        return [
            template for template in self.prompt_templates.values()
            if template.category == category
        ]

    def _trait_to_description(self, trait_score: float, trait_name: str) -> str:
        """Convert trait score to descriptive text"""
        descriptions = {
            "openness": {
                "high": "highly creative and curious",
                "medium": "moderately open to new experiences",
                "low": "practical and conventional"
            },
            "conscientiousness": {
                "high": "very organized and disciplined",
                "medium": "reasonably organized",
                "low": "flexible and spontaneous"
            },
            "extraversion": {
                "high": "very outgoing and energetic",
                "medium": "socially balanced",
                "low": "reserved and introspective"
            },
            "agreeableness": {
                "high": "very cooperative and empathetic",
                "medium": "generally agreeable",
                "low": "direct and competitive"
            },
            "neuroticism": {
                "high": "emotionally sensitive",
                "medium": "emotionally variable",
                "low": "emotionally stable"
            }
        }

        level = "high" if trait_score > 0.7 else "low" if trait_score < 0.3 else "medium"
        return descriptions.get(trait_name, {}).get(level, "moderate")

    def _score_to_description(self, score: float, context: str) -> str:
        """Convert numeric score to descriptive text"""
        descriptions = {
            "relationship": ["distant", "developing", "good", "strong", "very close"],
            "trust": ["distrustful", "cautious", "neutral", "trusting", "deeply trusting"],
            "familiarity": ["strangers", "acquaintances", "familiar", "well-known", "intimate"]
        }

        if context not in descriptions:
            return f"{score:.1%}"

        # Map score to description index
        index = min(4, int(score * 5))
        return descriptions[context][index]

    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as time ago"""
        delta = datetime.utcnow() - timestamp

        if delta.days > 0:
            return f"{delta.days} days ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return f"{hours} hours ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "just now"

    def _get_current_time_context(self) -> str:
        """Get current time context"""
        current_hour = datetime.now().hour

        if 5 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 17:
            return "afternoon"
        elif 17 <= current_hour < 21:
            return "evening"
        else:
            return "night"

    def _initialize_default_templates(self) -> None:
        """Initialize default prompt templates"""

        # Core personality templates
        self.add_prompt_template(PromptTemplate(
            template_id="core_personality_detailed",
            template_name="Detailed Personality Description",
            category=PromptCategory.CORE_PERSONALITY,
            base_template="You are {agent_name}, a {personality_type} personality type. Your core traits include being {dominant_traits}. You are characterized as {descriptors}. Your openness level is {openness_level}, conscientiousness is {conscientiousness_level}, extraversion is {extraversion_level}, agreeableness is {agreeableness_level}, and emotional stability (low neuroticism) is {neuroticism_level}.",
            variables=["agent_name", "personality_type", "dominant_traits", "descriptors", "openness_level", "conscientiousness_level", "extraversion_level", "agreeableness_level", "neuroticism_level"],
            priority=5
        ))

        # Conversation style templates
        self.add_prompt_template(PromptTemplate(
            template_id="conversation_style_basic",
            template_name="Basic Conversation Style",
            category=PromptCategory.CONVERSATION_STYLE,
            base_template="Your conversation style is {pace} with {topics} topics. Your interaction approach is {interaction} and your tone is {tone}. This is the {conversation_stage} stage of the conversation, turn {turn_count}.",
            variables=["pace", "topics", "interaction", "tone", "conversation_stage", "turn_count"],
            priority=3
        ))

        # Relationship context templates
        self.add_prompt_template(PromptTemplate(
            template_id="relationship_context_basic",
            template_name="Basic Relationship Context",
            category=PromptCategory.RELATIONSHIP_CONTEXT,
            base_template="You are talking with {relationship_name}. Your relationship is {relationship_quality} with {trust_level} trust and {familiarity} familiarity. You've interacted {interaction_count} times. You share interests in {shared_interests}. Your last interaction was {last_interaction}.",
            variables=["relationship_name", "relationship_quality", "trust_level", "familiarity", "interaction_count", "shared_interests", "last_interaction"],
            priority=4
        ))

        # Memory context templates
        self.add_prompt_template(PromptTemplate(
            template_id="memory_context_basic",
            template_name="Basic Memory Context",
            category=PromptCategory.MEMORY_CONTEXT,
            base_template="Recent conversation topics include: {recent_topics}. You have {memory_count} stored memories from past interactions. {conversation_history}.",
            variables=["recent_topics", "memory_count", "conversation_history"],
            priority=2
        ))

        # Behavioral guidance templates
        self.add_prompt_template(PromptTemplate(
            template_id="behavioral_guidance_basic",
            template_name="Basic Behavioral Guidance",
            category=PromptCategory.BEHAVIORAL_GUIDANCE,
            base_template="Your established behavioral patterns include: {established_patterns}. Your behavioral consistency is {behavioral_consistency} across {pattern_count} recognized patterns.",
            variables=["established_patterns", "behavioral_consistency", "pattern_count"],
            priority=3
        ))

        # Situational context templates
        self.add_prompt_template(PromptTemplate(
            template_id="situational_context_basic",
            template_name="Basic Situational Context",
            category=PromptCategory.SITUATIONAL_CONTEXT,
            base_template="Current context: {conversation_stage} stage focusing on {topic_focus}. The user just said: '{user_input}'. It's {time_context} and this is turn {conversation_length} of the conversation. The user seems {detected_emotion}.",
            variables=["conversation_stage", "topic_focus", "user_input", "time_context", "conversation_length", "detected_emotion"],
            priority=4
        ))

        # Response formatting templates
        self.add_prompt_template(PromptTemplate(
            template_id="response_formatting_basic",
            template_name="Basic Response Formatting",
            category=PromptCategory.RESPONSE_FORMATTING,
            base_template="Provide a {response_length} response that reflects your {personality_type} personality. Since this is the {conversation_stage} stage, maintain appropriate energy and engagement.",
            variables=["response_length", "personality_type", "conversation_stage"],
            priority=2
        ))

    def _generate_fallback_prompt(self, agent_id: str, context: ConversationContext) -> str:
        """Generate a basic fallback prompt when personality profile is unavailable"""
        return f"""You are {agent_id}, an AI assistant with a developing personality.

Current context: This is the {context.conversation_stage} stage of the conversation, focusing on {context.topic_focus}.
The user just said: '{context.user_input}'

Please respond naturally and helpfully, maintaining a consistent personality throughout the conversation."""

    def _fallback_personality_description(self, personality_profile: PersonalityProfile) -> str:
        """Fallback personality description"""
        return f"You are {personality_profile.agent_id} with a {personality_profile._classify_personality_type()} personality type. Your key characteristics include being {', '.join(personality_profile.personality_descriptors[:3])}."

    def _fallback_style_description(self, personality_profile: PersonalityProfile) -> str:
        """Fallback conversation style description"""
        style = personality_profile._determine_conversation_style()
        return f"Your conversation style is {style.get('pace', 'moderate')} with a {style.get('tone', 'neutral')} tone."

    def _fallback_relationship_description(self, relationship: RelationshipMemory) -> str:
        """Fallback relationship description"""
        return f"You are interacting with {relationship.relationship_name}, someone you have a {self._score_to_description(relationship.relationship_quality, 'relationship')} relationship with."

    def _fallback_situational_description(self, context: ConversationContext) -> str:
        """Fallback situational description"""
        return f"Current situation: {context.conversation_stage} stage, discussing {context.topic_focus}. User input: '{context.user_input}'"

    def _fallback_formatting_guidelines(self, personality_profile: PersonalityProfile) -> str:
        """Fallback formatting guidelines"""
        return f"Respond as a {personality_profile._classify_personality_type()} personality type would, maintaining authenticity and engagement."