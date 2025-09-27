"""
Conversation Orchestration Engine

This module manages multi-agent conversation flows, turn management, and conversation
state orchestration for collaborative agent interactions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class FlowState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TurnType(Enum):
    SEQUENTIAL = "sequential"      # One agent at a time
    PARALLEL = "parallel"          # Multiple agents simultaneously
    CONDITIONAL = "conditional"    # Based on conditions
    INTERRUPT = "interrupt"        # High-priority interruption
    COLLABORATIVE = "collaborative" # Joint response

class ConversationMode(Enum):
    STRUCTURED = "structured"      # Predefined flow
    DYNAMIC = "dynamic"           # Adaptive flow
    FREESTYLE = "freestyle"       # No predetermined structure
    FACILITATED = "facilitated"   # Human-guided

@dataclass
class ConversationTurn:
    """Represents a turn in a conversation"""
    turn_id: str
    conversation_id: str
    agent_id: Optional[str]  # None for system turns
    turn_type: TurnType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # Turn IDs this turn depends on
    status: str = "pending"  # pending, active, completed, failed

    def is_completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'conversation_id': self.conversation_id,
            'agent_id': self.agent_id,
            'turn_type': self.turn_type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'metadata': self.metadata,
            'dependencies': list(self.dependencies),
            'status': self.status
        }

@dataclass
class ConversationFlow:
    """Defines the structure and flow of a multi-agent conversation"""
    flow_id: str
    name: str
    description: str
    mode: ConversationMode
    participants: Set[str]  # Agent IDs
    turns: List[ConversationTurn] = field(default_factory=list)
    current_turn_index: int = 0
    state: FlowState = FlowState.INITIALIZING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_current_turn(self) -> Optional[ConversationTurn]:
        if 0 <= self.current_turn_index < len(self.turns):
            return self.turns[self.current_turn_index]
        return None

    def get_next_turn(self) -> Optional[ConversationTurn]:
        next_index = self.current_turn_index + 1
        if next_index < len(self.turns):
            return self.turns[next_index]
        return None

    def advance_turn(self) -> bool:
        """Advance to the next turn"""
        if self.current_turn_index < len(self.turns) - 1:
            self.current_turn_index += 1
            return True
        return False

    def is_completed(self) -> bool:
        return self.state == FlowState.COMPLETED

    def get_progress_percentage(self) -> float:
        if not self.turns:
            return 0.0
        completed_turns = sum(1 for turn in self.turns if turn.is_completed())
        return (completed_turns / len(self.turns)) * 100.0

@dataclass
class TurnRule:
    """Defines rules for turn management"""
    rule_id: str
    condition: str  # Condition expression
    action: str     # Action to take when condition is met
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Shared context for a conversation"""
    conversation_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    turn_history: List[ConversationTurn] = field(default_factory=list)
    active_topics: Set[str] = field(default_factory=set)
    participant_states: Dict[str, str] = field(default_factory=dict)  # agent_id -> state

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def add_to_shared_memory(self, key: str, value: Any) -> None:
        self.shared_memory[key] = value

@dataclass
class OrchestratorConfig:
    """Configuration for conversation orchestrator"""
    max_concurrent_conversations: int = 100
    turn_timeout_seconds: int = 120
    conversation_timeout_hours: int = 24
    enable_turn_interruption: bool = True
    enable_parallel_processing: bool = True
    max_turn_retries: int = 3
    auto_save_interval_seconds: int = 60

class TurnManager:
    """Manages turn-taking in conversations"""

    def __init__(self):
        self.turn_queues: Dict[str, deque] = defaultdict(deque)  # conversation_id -> turn queue
        self.active_turns: Dict[str, ConversationTurn] = {}      # turn_id -> turn
        self.turn_rules: List[TurnRule] = []

    async def schedule_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Schedule a turn for execution"""
        self.turn_queues[conversation_id].append(turn)
        logger.debug(f"Scheduled turn {turn.turn_id} for conversation {conversation_id}")

    async def get_next_turn(self, conversation_id: str) -> Optional[ConversationTurn]:
        """Get the next turn to execute"""
        queue = self.turn_queues.get(conversation_id)
        if queue:
            return queue.popleft()
        return None

    async def start_turn(self, turn: ConversationTurn) -> None:
        """Mark a turn as active"""
        turn.status = "active"
        self.active_turns[turn.turn_id] = turn
        logger.debug(f"Started turn {turn.turn_id}")

    async def complete_turn(self, turn_id: str, result: Dict[str, Any] = None) -> bool:
        """Mark a turn as completed"""
        if turn_id in self.active_turns:
            turn = self.active_turns[turn_id]
            turn.status = "completed"
            turn.duration_seconds = (datetime.now() - turn.timestamp).total_seconds()
            if result:
                turn.metadata['result'] = result

            del self.active_turns[turn_id]
            logger.debug(f"Completed turn {turn_id}")
            return True
        return False

    async def fail_turn(self, turn_id: str, error: str) -> bool:
        """Mark a turn as failed"""
        if turn_id in self.active_turns:
            turn = self.active_turns[turn_id]
            turn.status = "failed"
            turn.metadata['error'] = error

            del self.active_turns[turn_id]
            logger.error(f"Failed turn {turn_id}: {error}")
            return True
        return False

    def add_turn_rule(self, rule: TurnRule) -> None:
        """Add a turn management rule"""
        self.turn_rules.append(rule)
        self.turn_rules.sort(key=lambda r: r.priority, reverse=True)

    async def evaluate_turn_rules(self, context: ConversationContext) -> List[str]:
        """Evaluate turn rules and return applicable actions"""
        actions = []
        for rule in self.turn_rules:
            if await self._evaluate_condition(rule.condition, context):
                actions.append(rule.action)
        return actions

    async def _evaluate_condition(self, condition: str, context: ConversationContext) -> bool:
        """Evaluate a turn rule condition"""
        # Simplified condition evaluation - in production, use a proper expression parser
        try:
            # Create evaluation context
            eval_context = {
                'variables': context.variables,
                'shared_memory': context.shared_memory,
                'active_topics': context.active_topics,
                'participant_count': len(context.participant_states),
                'turn_count': len(context.turn_history)
            }

            # Replace variables in condition
            eval_condition = condition
            for key, value in eval_context.items():
                eval_condition = eval_condition.replace(f"${key}", str(value))

            # WARNING: eval is dangerous in production - use a proper expression parser
            return eval(eval_condition)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

class ConversationOrchestrator:
    """
    Orchestrates multi-agent conversations with sophisticated flow management,
    turn-taking, and conversation state management.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

        # Conversation management
        self.active_conversations: Dict[str, ConversationFlow] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.turn_manager = TurnManager()

        # Flow templates
        self.flow_templates: Dict[str, ConversationFlow] = {}

        # System state
        self.orchestrator_active = False
        self.background_tasks: List[asyncio.Task] = []

        # Event callbacks
        self.conversation_callbacks: List[Callable[[str, ConversationFlow, str], None]] = []
        self.turn_callbacks: List[Callable[[str, ConversationTurn, str], None]] = []

        # Statistics
        self.stats = {
            'total_conversations': 0,
            'active_conversations': 0,
            'completed_conversations': 0,
            'total_turns': 0,
            'average_conversation_duration': 0.0,
            'average_turns_per_conversation': 0.0
        }

    async def start(self) -> None:
        """Start the conversation orchestrator"""
        if self.orchestrator_active:
            logger.warning("Orchestrator already active")
            return

        self.orchestrator_active = True
        logger.info("Starting conversation orchestrator")

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._conversation_processor()),
            asyncio.create_task(self._turn_processor()),
            asyncio.create_task(self._timeout_monitor()),
            asyncio.create_task(self._auto_saver())
        ]

        logger.info("Conversation orchestrator started")

    async def stop(self) -> None:
        """Stop the conversation orchestrator"""
        if not self.orchestrator_active:
            return

        self.orchestrator_active = False
        logger.info("Stopping conversation orchestrator")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        logger.info("Conversation orchestrator stopped")

    async def create_conversation(self, name: str, description: str,
                                participants: Set[str],
                                mode: ConversationMode = ConversationMode.DYNAMIC,
                                template_id: Optional[str] = None) -> str:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())

        if template_id and template_id in self.flow_templates:
            # Use template
            template = self.flow_templates[template_id]
            flow = ConversationFlow(
                flow_id=conversation_id,
                name=name,
                description=description,
                mode=mode,
                participants=participants,
                turns=template.turns.copy(),
                metadata=template.metadata.copy()
            )
        else:
            # Create new flow
            flow = ConversationFlow(
                flow_id=conversation_id,
                name=name,
                description=description,
                mode=mode,
                participants=participants
            )

        # Create conversation context
        context = ConversationContext(
            conversation_id=conversation_id,
            participant_states={agent_id: "available" for agent_id in participants}
        )

        self.active_conversations[conversation_id] = flow
        self.conversation_contexts[conversation_id] = context
        self.stats['total_conversations'] += 1

        logger.info(f"Created conversation {name} ({conversation_id}) with {len(participants)} participants")
        await self._notify_conversation_callbacks(conversation_id, flow, "created")

        return conversation_id

    async def start_conversation(self, conversation_id: str) -> bool:
        """Start an active conversation"""
        flow = self.active_conversations.get(conversation_id)
        if not flow:
            return False

        flow.state = FlowState.ACTIVE
        flow.started_at = datetime.now()

        logger.info(f"Started conversation {conversation_id}")
        await self._notify_conversation_callbacks(conversation_id, flow, "started")

        return True

    async def add_turn(self, conversation_id: str, agent_id: Optional[str],
                      turn_type: TurnType, content: Dict[str, Any],
                      dependencies: Set[str] = None) -> str:
        """Add a turn to a conversation"""
        flow = self.active_conversations.get(conversation_id)
        if not flow:
            raise ValueError(f"Conversation {conversation_id} not found")

        turn_id = str(uuid.uuid4())
        turn = ConversationTurn(
            turn_id=turn_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            turn_type=turn_type,
            content=content,
            dependencies=dependencies or set()
        )

        flow.turns.append(turn)

        # Update context
        context = self.conversation_contexts[conversation_id]
        context.turn_history.append(turn)

        # Schedule turn if dependencies are met
        if await self._are_dependencies_met(turn, context):
            await self.turn_manager.schedule_turn(conversation_id, turn)

        self.stats['total_turns'] += 1
        logger.debug(f"Added turn {turn_id} to conversation {conversation_id}")
        await self._notify_turn_callbacks(turn_id, turn, "added")

        return turn_id

    async def execute_turn(self, turn_id: str) -> bool:
        """Execute a specific turn"""
        # Find the turn
        turn = None
        conversation_id = None

        for conv_id, flow in self.active_conversations.items():
            for t in flow.turns:
                if t.turn_id == turn_id:
                    turn = t
                    conversation_id = conv_id
                    break
            if turn:
                break

        if not turn:
            return False

        context = self.conversation_contexts[conversation_id]

        try:
            # Start turn
            await self.turn_manager.start_turn(turn)
            await self._notify_turn_callbacks(turn_id, turn, "started")

            # Execute turn based on type
            result = await self._execute_turn_logic(turn, context)

            # Complete turn
            await self.turn_manager.complete_turn(turn_id, result)
            await self._notify_turn_callbacks(turn_id, turn, "completed")

            # Check for next turns
            await self._schedule_dependent_turns(turn, context)

            return True

        except Exception as e:
            await self.turn_manager.fail_turn(turn_id, str(e))
            await self._notify_turn_callbacks(turn_id, turn, "failed")
            return False

    async def pause_conversation(self, conversation_id: str) -> bool:
        """Pause a conversation"""
        flow = self.active_conversations.get(conversation_id)
        if not flow:
            return False

        flow.state = FlowState.PAUSED
        logger.info(f"Paused conversation {conversation_id}")
        await self._notify_conversation_callbacks(conversation_id, flow, "paused")

        return True

    async def resume_conversation(self, conversation_id: str) -> bool:
        """Resume a paused conversation"""
        flow = self.active_conversations.get(conversation_id)
        if not flow:
            return False

        flow.state = FlowState.ACTIVE
        logger.info(f"Resumed conversation {conversation_id}")
        await self._notify_conversation_callbacks(conversation_id, flow, "resumed")

        return True

    async def complete_conversation(self, conversation_id: str) -> bool:
        """Mark a conversation as completed"""
        flow = self.active_conversations.get(conversation_id)
        if not flow:
            return False

        flow.state = FlowState.COMPLETED
        flow.completed_at = datetime.now()

        self.stats['completed_conversations'] += 1
        logger.info(f"Completed conversation {conversation_id}")
        await self._notify_conversation_callbacks(conversation_id, flow, "completed")

        return True

    async def get_conversation_status(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a conversation"""
        flow = self.active_conversations.get(conversation_id)
        context = self.conversation_contexts.get(conversation_id)

        if not flow or not context:
            return None

        return {
            'conversation_id': conversation_id,
            'name': flow.name,
            'description': flow.description,
            'mode': flow.mode.value,
            'state': flow.state.value,
            'participants': list(flow.participants),
            'current_turn_index': flow.current_turn_index,
            'total_turns': len(flow.turns),
            'progress_percentage': flow.get_progress_percentage(),
            'created_at': flow.created_at.isoformat(),
            'started_at': flow.started_at.isoformat() if flow.started_at else None,
            'completed_at': flow.completed_at.isoformat() if flow.completed_at else None,
            'participant_states': context.participant_states,
            'active_topics': list(context.active_topics),
            'metadata': flow.metadata
        }

    async def get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get list of active conversations"""
        active = []
        for conversation_id in self.active_conversations:
            status = await self.get_conversation_status(conversation_id)
            if status:
                active.append(status)
        return active

    async def register_flow_template(self, template_id: str, flow: ConversationFlow) -> None:
        """Register a conversation flow template"""
        self.flow_templates[template_id] = flow
        logger.info(f"Registered flow template: {template_id}")

    def add_conversation_callback(self, callback: Callable[[str, ConversationFlow, str], None]) -> None:
        """Add callback for conversation events"""
        self.conversation_callbacks.append(callback)

    def add_turn_callback(self, callback: Callable[[str, ConversationTurn, str], None]) -> None:
        """Add callback for turn events"""
        self.turn_callbacks.append(callback)

    async def _execute_turn_logic(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, Any]:
        """Execute the logic for a specific turn"""
        if turn.turn_type == TurnType.SEQUENTIAL:
            return await self._execute_sequential_turn(turn, context)
        elif turn.turn_type == TurnType.PARALLEL:
            return await self._execute_parallel_turn(turn, context)
        elif turn.turn_type == TurnType.CONDITIONAL:
            return await self._execute_conditional_turn(turn, context)
        elif turn.turn_type == TurnType.COLLABORATIVE:
            return await self._execute_collaborative_turn(turn, context)
        else:
            return await self._execute_default_turn(turn, context)

    async def _execute_sequential_turn(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, Any]:
        """Execute a sequential turn"""
        # Implementation would integrate with agent execution system
        logger.debug(f"Executing sequential turn {turn.turn_id} for agent {turn.agent_id}")

        # Simulate turn execution
        await asyncio.sleep(0.1)

        return {
            'type': 'sequential',
            'agent_id': turn.agent_id,
            'success': True,
            'response': turn.content
        }

    async def _execute_parallel_turn(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, Any]:
        """Execute a parallel turn"""
        logger.debug(f"Executing parallel turn {turn.turn_id}")

        # Simulate parallel execution
        await asyncio.sleep(0.1)

        return {
            'type': 'parallel',
            'participants': list(context.participant_states.keys()),
            'success': True
        }

    async def _execute_conditional_turn(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, Any]:
        """Execute a conditional turn"""
        condition = turn.content.get('condition', 'true')
        result = await self.turn_manager._evaluate_condition(condition, context)

        logger.debug(f"Executing conditional turn {turn.turn_id}, condition result: {result}")

        return {
            'type': 'conditional',
            'condition': condition,
            'condition_result': result,
            'success': True
        }

    async def _execute_collaborative_turn(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, Any]:
        """Execute a collaborative turn"""
        logger.debug(f"Executing collaborative turn {turn.turn_id}")

        # Simulate collaborative execution
        await asyncio.sleep(0.2)

        return {
            'type': 'collaborative',
            'collaborators': list(context.participant_states.keys()),
            'success': True
        }

    async def _execute_default_turn(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, Any]:
        """Execute a default turn"""
        logger.debug(f"Executing default turn {turn.turn_id}")

        await asyncio.sleep(0.1)

        return {
            'type': 'default',
            'agent_id': turn.agent_id,
            'success': True
        }

    async def _are_dependencies_met(self, turn: ConversationTurn, context: ConversationContext) -> bool:
        """Check if turn dependencies are met"""
        if not turn.dependencies:
            return True

        completed_turns = {t.turn_id for t in context.turn_history if t.is_completed()}
        return turn.dependencies.issubset(completed_turns)

    async def _schedule_dependent_turns(self, completed_turn: ConversationTurn, context: ConversationContext) -> None:
        """Schedule turns that depend on the completed turn"""
        conversation_id = completed_turn.conversation_id
        flow = self.active_conversations.get(conversation_id)

        if not flow:
            return

        for turn in flow.turns:
            if (turn.status == "pending" and
                completed_turn.turn_id in turn.dependencies and
                await self._are_dependencies_met(turn, context)):
                await self.turn_manager.schedule_turn(conversation_id, turn)

    async def _conversation_processor(self) -> None:
        """Background task to process conversations"""
        while self.orchestrator_active:
            try:
                # Process active conversations
                for conversation_id, flow in list(self.active_conversations.items()):
                    if flow.state == FlowState.ACTIVE:
                        await self._process_conversation_flow(conversation_id, flow)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in conversation processor: {e}")
                await asyncio.sleep(5)

    async def _turn_processor(self) -> None:
        """Background task to process turns"""
        while self.orchestrator_active:
            try:
                # Process turn queues
                for conversation_id in list(self.turn_manager.turn_queues.keys()):
                    turn = await self.turn_manager.get_next_turn(conversation_id)
                    if turn:
                        await self.execute_turn(turn.turn_id)

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in turn processor: {e}")
                await asyncio.sleep(1)

    async def _timeout_monitor(self) -> None:
        """Monitor for conversation and turn timeouts"""
        while self.orchestrator_active:
            try:
                now = datetime.now()

                # Check conversation timeouts
                timeout_threshold = timedelta(hours=self.config.conversation_timeout_hours)
                for conversation_id, flow in list(self.active_conversations.items()):
                    if (flow.started_at and
                        (now - flow.started_at) > timeout_threshold and
                        flow.state == FlowState.ACTIVE):
                        logger.warning(f"Conversation {conversation_id} timed out")
                        flow.state = FlowState.FAILED
                        await self._notify_conversation_callbacks(conversation_id, flow, "timeout")

                # Check turn timeouts
                turn_timeout = timedelta(seconds=self.config.turn_timeout_seconds)
                for turn_id, turn in list(self.turn_manager.active_turns.items()):
                    if (now - turn.timestamp) > turn_timeout:
                        logger.warning(f"Turn {turn_id} timed out")
                        await self.turn_manager.fail_turn(turn_id, "Turn timeout")

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in timeout monitor: {e}")
                await asyncio.sleep(10)

    async def _auto_saver(self) -> None:
        """Auto-save conversation state"""
        while self.orchestrator_active:
            try:
                # Save conversation states (would implement actual persistence)
                logger.debug("Auto-saving conversation states")
                await asyncio.sleep(self.config.auto_save_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto saver: {e}")
                await asyncio.sleep(10)

    async def _process_conversation_flow(self, conversation_id: str, flow: ConversationFlow) -> None:
        """Process the flow logic for a conversation"""
        # Apply turn rules
        context = self.conversation_contexts[conversation_id]
        actions = await self.turn_manager.evaluate_turn_rules(context)

        for action in actions:
            await self._execute_flow_action(conversation_id, action)

    async def _execute_flow_action(self, conversation_id: str, action: str) -> None:
        """Execute a flow action"""
        # Implementation would handle various flow actions
        logger.debug(f"Executing flow action '{action}' for conversation {conversation_id}")

    async def _notify_conversation_callbacks(self, conversation_id: str, flow: ConversationFlow, event: str) -> None:
        """Notify conversation event callbacks"""
        for callback in self.conversation_callbacks:
            try:
                callback(conversation_id, flow, event)
            except Exception as e:
                logger.error(f"Error in conversation callback: {e}")

    async def _notify_turn_callbacks(self, turn_id: str, turn: ConversationTurn, event: str) -> None:
        """Notify turn event callbacks"""
        for callback in self.turn_callbacks:
            try:
                callback(turn_id, turn, event)
            except Exception as e:
                logger.error(f"Error in turn callback: {e}")

    async def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        self.stats.update({
            'active_conversations': len(self.active_conversations),
            'total_active_turns': len(self.turn_manager.active_turns),
            'queued_turns': sum(len(queue) for queue in self.turn_manager.turn_queues.values())
        })

        return self.stats.copy()