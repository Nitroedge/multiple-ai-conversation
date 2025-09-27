"""
Memory system data models and schemas
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid


class MemoryType(Enum):
    """Types of memory in the hierarchical system"""
    EPISODIC = "episodic"  # Specific conversation events
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # How to do things
    SHARED_EPISODIC = "shared_episodic"  # Shared experiences between agents


@dataclass
class MemoryItem:
    """Individual memory item with metadata"""
    content: str
    timestamp: datetime
    importance_score: float  # 0.0 to 1.0
    memory_type: MemoryType
    session_id: str
    agent_id: Optional[str] = None
    emotions: Optional[Dict[str, float]] = None
    context_tags: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'memory_id': self.memory_id,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'importance_score': self.importance_score,
            'memory_type': self.memory_type.value,
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'emotions': self.emotions or {},
            'context_tags': self.context_tags or [],
            'embedding': self.embedding
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create MemoryItem from dictionary"""
        return cls(
            memory_id=data.get('memory_id', str(uuid.uuid4())),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            importance_score=data['importance_score'],
            memory_type=MemoryType(data['memory_type']),
            session_id=data['session_id'],
            agent_id=data.get('agent_id'),
            emotions=data.get('emotions'),
            context_tags=data.get('context_tags'),
            embedding=data.get('embedding')
        )


@dataclass
class ConversationState:
    """Current state of a conversation"""
    session_id: str
    active_agents: List[str]
    conversation_stage: str  # "greeting", "discussion", "conclusion", "paused"
    topic_focus: str
    emotion_context: Dict[str, float]
    last_speaker: Optional[str]
    turn_count: int
    user_preferences: Dict[str, Any]
    timestamp: datetime

    # Memory context
    recent_memories: List[MemoryItem] = field(default_factory=list)
    context_summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return {
            'session_id': self.session_id,
            'active_agents': self.active_agents,
            'conversation_stage': self.conversation_stage,
            'topic_focus': self.topic_focus,
            'emotion_context': self.emotion_context,
            'last_speaker': self.last_speaker,
            'turn_count': self.turn_count,
            'user_preferences': self.user_preferences,
            'timestamp': self.timestamp.isoformat(),
            'recent_memories': [mem.to_dict() for mem in self.recent_memories],
            'context_summary': self.context_summary
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update conversation state from dictionary"""
        if "active_agents" in data:
            self.active_agents = data["active_agents"]
        if "conversation_stage" in data:
            self.conversation_stage = data["conversation_stage"]
        if "topic_focus" in data:
            self.topic_focus = data["topic_focus"]
        if "emotion_context" in data:
            self.emotion_context = data["emotion_context"]
        if "last_speaker" in data:
            self.last_speaker = data["last_speaker"]
        if "turn_count" in data:
            self.turn_count = data["turn_count"]
        if "user_preferences" in data:
            self.user_preferences = data["user_preferences"]

    def merge_update(self, data: Dict[str, Any]) -> None:
        """Merge update data with existing state"""
        if "active_agents" in data:
            # Merge agent lists, avoiding duplicates
            new_agents = set(self.active_agents + data["active_agents"])
            self.active_agents = list(new_agents)

        if "emotion_context" in data:
            # Merge emotion context dictionaries
            self.emotion_context.update(data["emotion_context"])

        if "user_preferences" in data:
            # Merge user preferences dictionaries
            self.user_preferences.update(data["user_preferences"])

        # Direct updates for simple fields
        for field in ["conversation_stage", "topic_focus", "last_speaker", "turn_count"]:
            if field in data:
                setattr(self, field, data[field])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create ConversationState from dictionary"""
        recent_memories = [
            MemoryItem.from_dict(mem_data)
            for mem_data in data.get('recent_memories', [])
        ]

        return cls(
            session_id=data['session_id'],
            active_agents=data['active_agents'],
            conversation_stage=data['conversation_stage'],
            topic_focus=data['topic_focus'],
            emotion_context=data['emotion_context'],
            last_speaker=data.get('last_speaker'),
            turn_count=data['turn_count'],
            user_preferences=data['user_preferences'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            recent_memories=recent_memories,
            context_summary=data.get('context_summary')
        )


@dataclass
class MemoryQuery:
    """Query structure for memory retrieval"""
    query_text: str
    session_id: str
    agent_id: Optional[str] = None
    memory_types: Optional[List[MemoryType]] = None
    time_range: Optional[tuple] = None  # (start_time, end_time)
    importance_threshold: float = 0.0
    max_results: int = 10
    include_embeddings: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing"""
        return {
            'query_text': self.query_text,
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'memory_types': [mt.value for mt in self.memory_types] if self.memory_types else None,
            'time_range': [dt.isoformat() for dt in self.time_range] if self.time_range else None,
            'importance_threshold': self.importance_threshold,
            'max_results': self.max_results,
            'include_embeddings': self.include_embeddings
        }


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval operation"""
    memories: List[MemoryItem]
    total_found: int
    query_time_ms: float
    relevance_scores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'memories': [mem.to_dict() for mem in self.memories],
            'total_found': self.total_found,
            'query_time_ms': self.query_time_ms,
            'relevance_scores': self.relevance_scores
        }


@dataclass
class ConsolidationResult:
    """Result of memory consolidation process"""
    consolidated_memories: List[MemoryItem]
    original_count: int
    consolidated_count: int
    processing_time_ms: float
    consolidation_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'consolidated_memories': [mem.to_dict() for mem in self.consolidated_memories],
            'original_count': self.original_count,
            'consolidated_count': self.consolidated_count,
            'processing_time_ms': self.processing_time_ms,
            'consolidation_strategy': self.consolidation_strategy
        }