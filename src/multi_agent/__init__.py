"""
Multi-Agent Coordination System

This module provides advanced multi-agent conversation coordination, orchestration,
and collaboration capabilities for the conversation engine.

Key Components:
- AgentCoordinator: Central coordination and orchestration system
- ConversationOrchestrator: Multi-agent conversation flow management
- AgentCommunication: Inter-agent communication protocols
- RoleManager: Dynamic agent role assignment and management
- ContextSharing: Shared conversation context across agents
- ConflictResolver: Conflict detection and resolution
- CollaborationPatterns: Predefined collaboration workflows
"""

from .agent_coordinator import (
    AgentCoordinator,
    CoordinationConfig,
    AgentStatus,
    CoordinationStrategy
)
from .conversation_orchestrator import (
    ConversationOrchestrator,
    ConversationFlow,
    FlowState,
    TurnManager
)
from .agent_communication import (
    AgentCommunication,
    CommunicationProtocol,
    CommunicationChannel,
    MessageType
)
from .role_manager import (
    RoleManager,
    AgentRole,
    RoleAssignment,
    RoleCapability
)
from .context_sharing import (
    ContextSharingManager,
    SharedContext,
    ContextScope,
    ContextUpdateType
)
from .conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ResolutionStrategy,
    ConflictSeverity,
    ConflictStatus,
    Conflict,
    ConflictParticipant
)
from .collaboration_engine import (
    CollaborationEngine,
    CollaborationPattern,
    TaskType,
    CollaborationPhase,
    CollaborationWorkflow,
    CollaborationTask,
    CollaborationTemplate
)

__all__ = [
    # Core coordination
    "AgentCoordinator",
    "CoordinationConfig",
    "AgentStatus",
    "CoordinationStrategy",

    # Conversation orchestration
    "ConversationOrchestrator",
    "ConversationFlow",
    "FlowState",
    "TurnManager",

    # Communication
    "AgentCommunication",
    "CommunicationProtocol",
    "CommunicationChannel",
    "MessageType",

    # Role management
    "RoleManager",
    "AgentRole",
    "RoleAssignment",
    "RoleCapability",

    # Context sharing
    "ContextSharingManager",
    "SharedContext",
    "ContextScope",
    "ContextUpdateType",

    # Conflict resolution
    "ConflictResolver",
    "ConflictType",
    "ResolutionStrategy",
    "ConflictSeverity",
    "ConflictStatus",
    "Conflict",
    "ConflictParticipant",

    # Collaboration engine
    "CollaborationEngine",
    "CollaborationPattern",
    "TaskType",
    "CollaborationPhase",
    "CollaborationWorkflow",
    "CollaborationTask",
    "CollaborationTemplate",
]