"""
Conversation Context Sharing System

This module provides shared context management for multi-agent conversations,
enabling agents to share knowledge, maintain consistent state, and coordinate
their understanding of the conversation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import hashlib
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ContextScope(Enum):
    GLOBAL = "global"              # Shared across all conversations
    CONVERSATION = "conversation"   # Shared within a conversation
    AGENT_GROUP = "agent_group"    # Shared within an agent group
    PRIVATE = "private"            # Agent-specific context

class ContextUpdateType(Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    SYNC = "sync"

class ContextAccessLevel(Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"

class ContextDataType(Enum):
    FACT = "fact"                  # Factual information
    OPINION = "opinion"            # Subjective information
    DECISION = "decision"          # Decisions made
    TASK = "task"                  # Task information
    RESOURCE = "resource"          # Shared resources
    MEMORY = "memory"              # Episodic memories
    KNOWLEDGE = "knowledge"        # Domain knowledge
    STATE = "state"                # Current state information

@dataclass
class ContextEntry:
    """Represents a single piece of shared context"""
    entry_id: str
    key: str
    value: Any
    data_type: ContextDataType
    scope: ContextScope
    created_by: str  # Agent ID
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    confidence: float = 1.0  # 0.0 to 1.0
    importance: float = 0.5  # 0.0 to 1.0
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if context entry has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert context entry to dictionary"""
        return {
            'entry_id': self.entry_id,
            'key': self.key,
            'value': self.value,
            'data_type': self.data_type.value,
            'scope': self.scope.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'updated_by': self.updated_by,
            'version': self.version,
            'tags': list(self.tags),
            'confidence': self.confidence,
            'importance': self.importance,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextEntry':
        """Create context entry from dictionary"""
        return cls(
            entry_id=data['entry_id'],
            key=data['key'],
            value=data['value'],
            data_type=ContextDataType(data['data_type']),
            scope=ContextScope(data['scope']),
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            updated_by=data.get('updated_by'),
            version=data.get('version', 1),
            tags=set(data.get('tags', [])),
            confidence=data.get('confidence', 1.0),
            importance=data.get('importance', 0.5),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            metadata=data.get('metadata', {})
        )

@dataclass
class SharedContext:
    """Represents a shared context space"""
    context_id: str
    name: str
    scope: ContextScope
    participants: Set[str]  # Agent IDs
    entries: Dict[str, ContextEntry] = field(default_factory=dict)
    access_permissions: Dict[str, ContextAccessLevel] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: Optional[datetime] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entry(self, key: str) -> Optional[ContextEntry]:
        """Get context entry by key"""
        for entry in self.entries.values():
            if entry.key == key and not entry.is_expired():
                return entry
        return None

    def get_entries_by_type(self, data_type: ContextDataType) -> List[ContextEntry]:
        """Get all entries of a specific type"""
        return [entry for entry in self.entries.values()
                if entry.data_type == data_type and not entry.is_expired()]

    def get_entries_by_tags(self, tags: Set[str]) -> List[ContextEntry]:
        """Get entries that match any of the given tags"""
        return [entry for entry in self.entries.values()
                if entry.tags.intersection(tags) and not entry.is_expired()]

    def can_access(self, agent_id: str, access_level: ContextAccessLevel) -> bool:
        """Check if agent has required access level"""
        if agent_id not in self.participants:
            return False

        agent_access = self.access_permissions.get(agent_id, ContextAccessLevel.READ_ONLY)
        access_hierarchy = {
            ContextAccessLevel.READ_ONLY: 1,
            ContextAccessLevel.READ_WRITE: 2,
            ContextAccessLevel.ADMIN: 3
        }

        return access_hierarchy[agent_access] >= access_hierarchy[access_level]

@dataclass
class ContextUpdate:
    """Represents a context update event"""
    update_id: str
    context_id: str
    entry_id: str
    update_type: ContextUpdateType
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextSharingConfig:
    """Configuration for context sharing system"""
    max_contexts_per_agent: int = 50
    max_entries_per_context: int = 1000
    default_ttl_hours: int = 24
    enable_conflict_resolution: bool = True
    enable_version_control: bool = True
    sync_interval_seconds: int = 30
    cleanup_interval_hours: int = 6

class ContextSharingManager:
    """
    Manages shared context across multi-agent conversations.

    This class provides context storage, synchronization, access control,
    and conflict resolution for collaborative agent interactions.
    """

    def __init__(self, config: Optional[ContextSharingConfig] = None):
        self.config = config or ContextSharingConfig()

        # Context storage
        self.contexts: Dict[str, SharedContext] = {}
        self.global_context: SharedContext = self._create_global_context()

        # Update tracking
        self.context_updates: deque = deque(maxlen=10000)
        self.pending_updates: Dict[str, List[ContextUpdate]] = defaultdict(list)

        # Agent subscriptions
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> context_ids
        self.context_subscribers: Dict[str, Set[str]] = defaultdict(set)  # context_id -> agent_ids

        # Conflict resolution
        self.conflict_resolvers: Dict[str, Callable[[ContextEntry, ContextEntry], ContextEntry]] = {}

        # System state
        self.context_manager_active = False
        self.background_tasks: List[asyncio.Task] = []

        # Event callbacks
        self.update_callbacks: List[Callable[[ContextUpdate], None]] = []
        self.conflict_callbacks: List[Callable[[str, List[ContextEntry]], None]] = []

        # Statistics
        self.stats = {
            'total_contexts': 0,
            'total_entries': 0,
            'updates_processed': 0,
            'conflicts_resolved': 0,
            'active_subscriptions': 0
        }

    async def start(self) -> None:
        """Start the context sharing system"""
        if self.context_manager_active:
            logger.warning("Context manager already active")
            return

        self.context_manager_active = True
        logger.info("Starting context sharing system")

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._sync_processor()),
            asyncio.create_task(self._cleanup_processor()),
            asyncio.create_task(self._conflict_monitor())
        ]

        logger.info("Context sharing system started")

    async def stop(self) -> None:
        """Stop the context sharing system"""
        if not self.context_manager_active:
            return

        self.context_manager_active = False
        logger.info("Stopping context sharing system")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        logger.info("Context sharing system stopped")

    async def create_context(self, name: str, scope: ContextScope,
                           participants: Set[str],
                           conversation_id: Optional[str] = None) -> str:
        """Create a new shared context"""
        context_id = str(uuid.uuid4())

        context = SharedContext(
            context_id=context_id,
            name=name,
            scope=scope,
            participants=participants.copy(),
            conversation_id=conversation_id,
            access_permissions={agent_id: ContextAccessLevel.READ_WRITE for agent_id in participants}
        )

        self.contexts[context_id] = context

        # Set up subscriptions
        for agent_id in participants:
            self.agent_subscriptions[agent_id].add(context_id)
            self.context_subscribers[context_id].add(agent_id)

        self.stats['total_contexts'] += 1

        logger.info(f"Created context {name} ({context_id}) with {len(participants)} participants")
        return context_id

    async def add_context_entry(self, context_id: str, key: str, value: Any,
                              data_type: ContextDataType, agent_id: str,
                              tags: Set[str] = None,
                              confidence: float = 1.0,
                              importance: float = 0.5,
                              ttl_hours: Optional[int] = None) -> str:
        """Add an entry to a shared context"""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")

        if not context.can_access(agent_id, ContextAccessLevel.READ_WRITE):
            raise PermissionError(f"Agent {agent_id} cannot write to context {context_id}")

        # Check for existing entry with same key
        existing_entry = context.get_entry(key)
        if existing_entry:
            return await self.update_context_entry(
                context_id, existing_entry.entry_id, value, agent_id,
                tags=tags, confidence=confidence, importance=importance
            )

        # Create new entry
        entry_id = str(uuid.uuid4())
        expires_at = None
        if ttl_hours or self.config.default_ttl_hours:
            hours = ttl_hours or self.config.default_ttl_hours
            expires_at = datetime.now() + timedelta(hours=hours)

        entry = ContextEntry(
            entry_id=entry_id,
            key=key,
            value=value,
            data_type=data_type,
            scope=context.scope,
            created_by=agent_id,
            tags=tags or set(),
            confidence=confidence,
            importance=importance,
            expires_at=expires_at
        )

        context.entries[entry_id] = entry
        context.last_updated = datetime.now()

        # Create update event
        update = ContextUpdate(
            update_id=str(uuid.uuid4()),
            context_id=context_id,
            entry_id=entry_id,
            update_type=ContextUpdateType.CREATE,
            agent_id=agent_id,
            new_value=value
        )

        await self._process_update(update)
        self.stats['total_entries'] += 1

        logger.debug(f"Added context entry {key} to context {context_id}")
        return entry_id

    async def update_context_entry(self, context_id: str, entry_id: str,
                                 new_value: Any, agent_id: str,
                                 tags: Optional[Set[str]] = None,
                                 confidence: Optional[float] = None,
                                 importance: Optional[float] = None) -> str:
        """Update an existing context entry"""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")

        if not context.can_access(agent_id, ContextAccessLevel.READ_WRITE):
            raise PermissionError(f"Agent {agent_id} cannot write to context {context_id}")

        entry = context.entries.get(entry_id)
        if not entry:
            raise ValueError(f"Entry {entry_id} not found in context {context_id}")

        old_value = entry.value

        # Update entry
        entry.value = new_value
        entry.updated_at = datetime.now()
        entry.updated_by = agent_id
        entry.version += 1

        if tags is not None:
            entry.tags = tags
        if confidence is not None:
            entry.confidence = confidence
        if importance is not None:
            entry.importance = importance

        context.last_updated = datetime.now()

        # Create update event
        update = ContextUpdate(
            update_id=str(uuid.uuid4()),
            context_id=context_id,
            entry_id=entry_id,
            update_type=ContextUpdateType.UPDATE,
            agent_id=agent_id,
            old_value=old_value,
            new_value=new_value
        )

        await self._process_update(update)

        logger.debug(f"Updated context entry {entry.key} in context {context_id}")
        return entry_id

    async def get_context_entry(self, context_id: str, key: str, agent_id: str) -> Optional[ContextEntry]:
        """Get a context entry by key"""
        context = self.contexts.get(context_id)
        if not context:
            return None

        if not context.can_access(agent_id, ContextAccessLevel.READ_ONLY):
            return None

        return context.get_entry(key)

    async def search_context_entries(self, context_id: str, agent_id: str,
                                   query: Optional[str] = None,
                                   data_type: Optional[ContextDataType] = None,
                                   tags: Optional[Set[str]] = None,
                                   min_confidence: float = 0.0,
                                   min_importance: float = 0.0) -> List[ContextEntry]:
        """Search for context entries matching criteria"""
        context = self.contexts.get(context_id)
        if not context:
            return []

        if not context.can_access(agent_id, ContextAccessLevel.READ_ONLY):
            return []

        results = []

        for entry in context.entries.values():
            if entry.is_expired():
                continue

            # Apply filters
            if data_type and entry.data_type != data_type:
                continue
            if entry.confidence < min_confidence:
                continue
            if entry.importance < min_importance:
                continue
            if tags and not entry.tags.intersection(tags):
                continue

            # Text search in key and value
            if query:
                search_text = f"{entry.key} {str(entry.value)}".lower()
                if query.lower() not in search_text:
                    continue

            results.append(entry)

        # Sort by importance and confidence
        results.sort(key=lambda e: (e.importance, e.confidence), reverse=True)
        return results

    async def subscribe_to_context(self, agent_id: str, context_id: str) -> bool:
        """Subscribe an agent to context updates"""
        context = self.contexts.get(context_id)
        if not context:
            return False

        if agent_id not in context.participants:
            return False

        self.agent_subscriptions[agent_id].add(context_id)
        self.context_subscribers[context_id].add(agent_id)

        self.stats['active_subscriptions'] += 1
        logger.debug(f"Agent {agent_id} subscribed to context {context_id}")
        return True

    async def unsubscribe_from_context(self, agent_id: str, context_id: str) -> bool:
        """Unsubscribe an agent from context updates"""
        self.agent_subscriptions[agent_id].discard(context_id)
        self.context_subscribers[context_id].discard(agent_id)

        self.stats['active_subscriptions'] = max(0, self.stats['active_subscriptions'] - 1)
        logger.debug(f"Agent {agent_id} unsubscribed from context {context_id}")
        return True

    async def get_agent_contexts(self, agent_id: str) -> List[SharedContext]:
        """Get all contexts accessible to an agent"""
        context_ids = self.agent_subscriptions.get(agent_id, set())
        return [self.contexts[cid] for cid in context_ids if cid in self.contexts]

    async def merge_contexts(self, source_context_id: str, target_context_id: str,
                           agent_id: str, conflict_resolution: str = "latest") -> bool:
        """Merge one context into another"""
        source = self.contexts.get(source_context_id)
        target = self.contexts.get(target_context_id)

        if not source or not target:
            return False

        if not (source.can_access(agent_id, ContextAccessLevel.ADMIN) and
                target.can_access(agent_id, ContextAccessLevel.ADMIN)):
            return False

        # Merge entries
        for entry in source.entries.values():
            existing = target.get_entry(entry.key)

            if existing:
                # Handle conflict
                if conflict_resolution == "latest":
                    if entry.updated_at and existing.updated_at:
                        if entry.updated_at > existing.updated_at:
                            target.entries[entry.entry_id] = entry
                elif conflict_resolution == "highest_confidence":
                    if entry.confidence > existing.confidence:
                        target.entries[entry.entry_id] = entry
                elif conflict_resolution == "most_important":
                    if entry.importance > existing.importance:
                        target.entries[entry.entry_id] = entry
            else:
                target.entries[entry.entry_id] = entry

        target.last_updated = datetime.now()

        # Create merge update
        update = ContextUpdate(
            update_id=str(uuid.uuid4()),
            context_id=target_context_id,
            entry_id="merge_operation",
            update_type=ContextUpdateType.MERGE,
            agent_id=agent_id,
            metadata={'source_context_id': source_context_id}
        )

        await self._process_update(update)

        logger.info(f"Merged context {source_context_id} into {target_context_id}")
        return True

    async def sync_context(self, context_id: str) -> bool:
        """Synchronize context across all subscribers"""
        context = self.contexts.get(context_id)
        if not context:
            return False

        # Create sync update
        update = ContextUpdate(
            update_id=str(uuid.uuid4()),
            context_id=context_id,
            entry_id="sync_operation",
            update_type=ContextUpdateType.SYNC,
            agent_id="system"
        )

        await self._process_update(update)

        logger.debug(f"Synchronized context {context_id}")
        return True

    async def get_context_summary(self, context_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a context"""
        context = self.contexts.get(context_id)
        if not context:
            return None

        if not context.can_access(agent_id, ContextAccessLevel.READ_ONLY):
            return None

        entries_by_type = defaultdict(int)
        total_confidence = 0.0
        total_importance = 0.0
        active_entries = 0

        for entry in context.entries.values():
            if not entry.is_expired():
                entries_by_type[entry.data_type] += 1
                total_confidence += entry.confidence
                total_importance += entry.importance
                active_entries += 1

        return {
            'context_id': context_id,
            'name': context.name,
            'scope': context.scope.value,
            'participants': list(context.participants),
            'total_entries': active_entries,
            'entries_by_type': {dt.value: count for dt, count in entries_by_type.items()},
            'average_confidence': total_confidence / active_entries if active_entries > 0 else 0.0,
            'average_importance': total_importance / active_entries if active_entries > 0 else 0.0,
            'last_updated': context.last_updated.isoformat() if context.last_updated else None,
            'conversation_id': context.conversation_id
        }

    def _create_global_context(self) -> SharedContext:
        """Create the global context"""
        return SharedContext(
            context_id="global",
            name="Global Context",
            scope=ContextScope.GLOBAL,
            participants=set(),
            access_permissions={}
        )

    async def _process_update(self, update: ContextUpdate) -> None:
        """Process a context update"""
        self.context_updates.append(update)
        self.stats['updates_processed'] += 1

        # Notify subscribers
        subscribers = self.context_subscribers.get(update.context_id, set())
        for callback in self.update_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")

        logger.debug(f"Processed context update {update.update_id}")

    async def _detect_conflicts(self, context_id: str) -> List[Tuple[str, List[ContextEntry]]]:
        """Detect conflicts in context entries"""
        context = self.contexts.get(context_id)
        if not context:
            return []

        conflicts = []
        entries_by_key = defaultdict(list)

        # Group entries by key
        for entry in context.entries.values():
            if not entry.is_expired():
                entries_by_key[entry.key].append(entry)

        # Find conflicts (multiple entries with same key)
        for key, entries in entries_by_key.items():
            if len(entries) > 1:
                # Sort by timestamp to identify conflicts
                entries.sort(key=lambda e: e.updated_at or e.created_at)
                conflicts.append((key, entries))

        return conflicts

    async def _resolve_conflict(self, key: str, conflicting_entries: List[ContextEntry]) -> ContextEntry:
        """Resolve conflicts between context entries"""
        if not conflicting_entries:
            raise ValueError("No entries to resolve")

        if len(conflicting_entries) == 1:
            return conflicting_entries[0]

        # Check for custom resolver
        if key in self.conflict_resolvers:
            resolver = self.conflict_resolvers[key]
            result = conflicting_entries[0]
            for entry in conflicting_entries[1:]:
                result = resolver(result, entry)
            return result

        # Default resolution strategies
        if self.config.enable_conflict_resolution:
            # Strategy 1: Highest confidence
            highest_confidence = max(conflicting_entries, key=lambda e: e.confidence)
            if highest_confidence.confidence > 0.8:
                return highest_confidence

            # Strategy 2: Most recent
            most_recent = max(conflicting_entries, key=lambda e: e.updated_at or e.created_at)
            return most_recent

        # Fallback: return first entry
        return conflicting_entries[0]

    async def _sync_processor(self) -> None:
        """Background task to process context synchronization"""
        while self.context_manager_active:
            try:
                # Process pending updates
                for context_id, updates in list(self.pending_updates.items()):
                    if updates:
                        # Process all pending updates for this context
                        for update in updates:
                            await self._process_update(update)
                        self.pending_updates[context_id].clear()

                await asyncio.sleep(self.config.sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync processor: {e}")
                await asyncio.sleep(5)

    async def _cleanup_processor(self) -> None:
        """Background task to clean up expired entries"""
        while self.context_manager_active:
            try:
                cleanup_count = 0

                for context in self.contexts.values():
                    expired_entries = [
                        entry_id for entry_id, entry in context.entries.items()
                        if entry.is_expired()
                    ]

                    for entry_id in expired_entries:
                        del context.entries[entry_id]
                        cleanup_count += 1

                if cleanup_count > 0:
                    logger.debug(f"Cleaned up {cleanup_count} expired context entries")

                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup processor: {e}")
                await asyncio.sleep(300)

    async def _conflict_monitor(self) -> None:
        """Background task to monitor and resolve conflicts"""
        while self.context_manager_active:
            try:
                if self.config.enable_conflict_resolution:
                    for context_id in list(self.contexts.keys()):
                        conflicts = await self._detect_conflicts(context_id)

                        for key, conflicting_entries in conflicts:
                            try:
                                resolved = await self._resolve_conflict(key, conflicting_entries)

                                # Remove other conflicting entries
                                context = self.contexts[context_id]
                                for entry in conflicting_entries:
                                    if entry.entry_id != resolved.entry_id:
                                        context.entries.pop(entry.entry_id, None)

                                self.stats['conflicts_resolved'] += 1

                                # Notify conflict callbacks
                                for callback in self.conflict_callbacks:
                                    try:
                                        callback(key, conflicting_entries)
                                    except Exception as e:
                                        logger.error(f"Error in conflict callback: {e}")

                            except Exception as e:
                                logger.error(f"Error resolving conflict for key '{key}': {e}")

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in conflict monitor: {e}")
                await asyncio.sleep(30)

    def add_update_callback(self, callback: Callable[[ContextUpdate], None]) -> None:
        """Add callback for context updates"""
        self.update_callbacks.append(callback)

    def add_conflict_callback(self, callback: Callable[[str, List[ContextEntry]], None]) -> None:
        """Add callback for conflict resolution"""
        self.conflict_callbacks.append(callback)

    def add_conflict_resolver(self, key: str, resolver: Callable[[ContextEntry, ContextEntry], ContextEntry]) -> None:
        """Add custom conflict resolver for specific keys"""
        self.conflict_resolvers[key] = resolver

    async def get_context_stats(self) -> Dict[str, Any]:
        """Get context sharing statistics"""
        total_entries = sum(len(ctx.entries) for ctx in self.contexts.values())
        active_entries = sum(
            len([e for e in ctx.entries.values() if not e.is_expired()])
            for ctx in self.contexts.values()
        )

        self.stats.update({
            'total_contexts': len(self.contexts),
            'total_entries': total_entries,
            'active_entries': active_entries,
            'active_subscriptions': sum(len(subs) for subs in self.context_subscribers.values())
        })

        return self.stats.copy()