"""
Multi-Agent Conversation Engine - Memory Management Module

This module implements a hierarchical memory system with:
- Working memory (Redis) for real-time conversation state
- Long-term memory (MongoDB) for persistent conversation history
- Vector-based memory retrieval for contextual responses
- Memory consolidation algorithms for importance-based storage
"""

from .hierarchical_manager import HierarchicalMemoryManager
from .working_memory import WorkingMemoryManager
from .long_term_memory import LongTermMemoryManager
from .vector_retrieval import VectorMemoryRetrieval
from .consolidation import MemoryConsolidationEngine
from .models import MemoryItem, MemoryType, ConversationState

__all__ = [
    "HierarchicalMemoryManager",
    "WorkingMemoryManager",
    "LongTermMemoryManager",
    "VectorMemoryRetrieval",
    "MemoryConsolidationEngine",
    "MemoryItem",
    "MemoryType",
    "ConversationState"
]