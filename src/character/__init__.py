"""
Character framework for dynamic personality development
"""

from .character_memory import (
    CharacterMemoryType,
    CharacterTraitMemory,
    RelationshipMemory,
    BehavioralPattern,
    CharacterGrowthMilestone,
    CharacterMemoryManager
)

from .dynamic_prompts import (
    PromptCategory,
    PromptTemplate,
    ConversationContext,
    DynamicPromptEngine
)

__all__ = [
    "CharacterMemoryType",
    "CharacterTraitMemory",
    "RelationshipMemory",
    "BehavioralPattern",
    "CharacterGrowthMilestone",
    "CharacterMemoryManager",
    "PromptCategory",
    "PromptTemplate",
    "ConversationContext",
    "DynamicPromptEngine"
]