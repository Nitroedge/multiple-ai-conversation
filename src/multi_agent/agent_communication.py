"""
Inter-Agent Communication System

This module provides communication protocols and channels for agents to interact
with each other in collaborative scenarios, including message routing, broadcasting,
and secure communication patterns.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import redis.asyncio as redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MessageType(Enum):
    DIRECT = "direct"           # One-to-one message
    BROADCAST = "broadcast"     # One-to-many message
    MULTICAST = "multicast"     # One-to-group message
    REQUEST = "request"         # Request-response pattern
    RESPONSE = "response"       # Response to a request
    NOTIFICATION = "notification" # Event notification
    COORDINATION = "coordination" # Coordination message
    SYSTEM = "system"          # System-level message

class MessagePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class CommunicationProtocol(Enum):
    ASYNC_QUEUE = "async_queue"     # Async message queues
    PUBSUB = "pubsub"              # Publish-subscribe
    REQUEST_REPLY = "request_reply" # Request-reply pattern
    STREAMING = "streaming"         # Streaming communication
    BROADCAST = "broadcast"         # Broadcast messages

@dataclass
class Message:
    """Represents a message between agents"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast messages
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    conversation_id: Optional[str] = None
    thread_id: Optional[str] = None
    correlation_id: Optional[str] = None  # For request-response matching
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'conversation_id': self.conversation_id,
            'thread_id': self.thread_id,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            recipient_id=data.get('recipient_id'),
            message_type=MessageType(data['message_type']),
            priority=MessagePriority(data['priority']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            conversation_id=data.get('conversation_id'),
            thread_id=data.get('thread_id'),
            correlation_id=data.get('correlation_id'),
            metadata=data.get('metadata', {})
        )

    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at

@dataclass
class CommunicationChannel:
    """Represents a communication channel between agents"""
    channel_id: str
    channel_type: str  # "direct", "group", "topic"
    participants: Set[str]
    protocol: CommunicationProtocol
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    message_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_participant(self, agent_id: str) -> None:
        """Add participant to channel"""
        self.participants.add(agent_id)

    def remove_participant(self, agent_id: str) -> None:
        """Remove participant from channel"""
        self.participants.discard(agent_id)

    def can_access(self, agent_id: str) -> bool:
        """Check if agent can access this channel"""
        return agent_id in self.participants

@dataclass
class MessageFilter:
    """Filter for message routing and processing"""
    sender_ids: Optional[Set[str]] = None
    recipient_ids: Optional[Set[str]] = None
    message_types: Optional[Set[MessageType]] = None
    priorities: Optional[Set[MessagePriority]] = None
    conversation_ids: Optional[Set[str]] = None
    content_keywords: Optional[Set[str]] = None

    def matches(self, message: Message) -> bool:
        """Check if message matches this filter"""
        if self.sender_ids and message.sender_id not in self.sender_ids:
            return False
        if self.recipient_ids and message.recipient_id not in self.recipient_ids:
            return False
        if self.message_types and message.message_type not in self.message_types:
            return False
        if self.priorities and message.priority not in self.priorities:
            return False
        if self.conversation_ids and message.conversation_id not in self.conversation_ids:
            return False

        if self.content_keywords:
            content_text = json.dumps(message.content).lower()
            if not any(keyword.lower() in content_text for keyword in self.content_keywords):
                return False

        return True

@dataclass
class CommunicationConfig:
    """Configuration for agent communication system"""
    redis_url: str = "redis://localhost:6379"
    message_ttl_seconds: int = 3600  # 1 hour default TTL
    max_message_size: int = 1024 * 1024  # 1MB
    max_queue_size: int = 10000
    enable_message_persistence: bool = True
    enable_delivery_confirmation: bool = True
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

class AgentCommunication:
    """
    Inter-agent communication system providing various communication patterns
    and protocols for collaborative agent interactions.
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        self.config = config or CommunicationConfig()

        # Message storage and routing
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue(maxsize=1000))
        self.channels: Dict[str, CommunicationChannel] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> subscriber_ids

        # Message tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}  # correlation_id -> future
        self.message_history: deque = deque(maxlen=10000)
        self.delivery_confirmations: Dict[str, Set[str]] = {}  # message_id -> confirmed_recipients

        # Redis connection for distributed communication
        self.redis_client: Optional[redis.Redis] = None

        # Event handlers
        self.message_handlers: Dict[MessageType, List[Callable[[Message], None]]] = defaultdict(list)
        self.middleware: List[Callable[[Message], Message]] = []

        # System state
        self.communication_active = False
        self.background_tasks: List[asyncio.Task] = []

        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'active_channels': 0,
            'active_subscriptions': 0
        }

    async def start(self) -> None:
        """Start the communication system"""
        if self.communication_active:
            logger.warning("Communication system already active")
            return

        self.communication_active = True
        logger.info("Starting agent communication system")

        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for distributed communication")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Using local communication only")
            self.redis_client = None

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._cleanup_expired_messages()),
            asyncio.create_task(self._stats_collector())
        ]

        if self.redis_client:
            self.background_tasks.append(asyncio.create_task(self._redis_subscriber()))

        logger.info("Agent communication system started")

    async def stop(self) -> None:
        """Stop the communication system"""
        if not self.communication_active:
            return

        self.communication_active = False
        logger.info("Stopping agent communication system")

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Agent communication system stopped")

    async def send_message(self, sender_id: str, recipient_id: Optional[str],
                          message_type: MessageType, content: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL,
                          conversation_id: Optional[str] = None,
                          thread_id: Optional[str] = None,
                          ttl_seconds: Optional[int] = None) -> str:
        """Send a message to another agent or broadcast"""
        message_id = str(uuid.uuid4())

        expires_at = None
        if ttl_seconds or self.config.message_ttl_seconds:
            ttl = ttl_seconds or self.config.message_ttl_seconds
            expires_at = datetime.now() + timedelta(seconds=ttl)

        message = Message(
            message_id=message_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            content=content,
            expires_at=expires_at,
            conversation_id=conversation_id,
            thread_id=thread_id
        )

        # Apply middleware
        for middleware_func in self.middleware:
            try:
                message = middleware_func(message)
            except Exception as e:
                logger.error(f"Error in message middleware: {e}")

        await self._route_message(message)
        self.message_history.append(message)
        self.stats['messages_sent'] += 1

        logger.debug(f"Sent message {message_id} from {sender_id} to {recipient_id or 'broadcast'}")
        return message_id

    async def send_request(self, sender_id: str, recipient_id: str,
                          content: Dict[str, Any], timeout_seconds: float = 30.0,
                          conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a request and wait for response"""
        correlation_id = str(uuid.uuid4())

        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[correlation_id] = response_future

        try:
            # Send request
            await self.send_message(
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=MessageType.REQUEST,
                content=content,
                conversation_id=conversation_id
            )

            # Set correlation ID in the message
            message = self.message_history[-1]  # Get the last sent message
            message.correlation_id = correlation_id

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout_seconds)
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Request {correlation_id} timed out")
            raise
        finally:
            self.pending_requests.pop(correlation_id, None)

    async def send_response(self, sender_id: str, recipient_id: str,
                           content: Dict[str, Any], correlation_id: str,
                           conversation_id: Optional[str] = None) -> str:
        """Send a response to a request"""
        message_id = await self.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.RESPONSE,
            content=content,
            conversation_id=conversation_id
        )

        # Set correlation ID
        message = self.message_history[-1]
        message.correlation_id = correlation_id

        # Complete pending request if exists
        if correlation_id in self.pending_requests:
            future = self.pending_requests.pop(correlation_id)
            if not future.done():
                future.set_result(content)

        return message_id

    async def broadcast_message(self, sender_id: str, content: Dict[str, Any],
                               topic: Optional[str] = None,
                               conversation_id: Optional[str] = None) -> str:
        """Broadcast a message to all agents or topic subscribers"""
        return await self.send_message(
            sender_id=sender_id,
            recipient_id=None,
            message_type=MessageType.BROADCAST,
            content=content,
            conversation_id=conversation_id
        )

    async def create_channel(self, channel_id: str, channel_type: str,
                           participants: Set[str],
                           protocol: CommunicationProtocol = CommunicationProtocol.ASYNC_QUEUE) -> bool:
        """Create a communication channel"""
        if channel_id in self.channels:
            logger.warning(f"Channel {channel_id} already exists")
            return False

        channel = CommunicationChannel(
            channel_id=channel_id,
            channel_type=channel_type,
            participants=participants.copy(),
            protocol=protocol
        )

        self.channels[channel_id] = channel
        self.stats['active_channels'] += 1

        logger.info(f"Created channel {channel_id} with {len(participants)} participants")
        return True

    async def join_channel(self, channel_id: str, agent_id: str) -> bool:
        """Add agent to a channel"""
        channel = self.channels.get(channel_id)
        if not channel:
            return False

        channel.add_participant(agent_id)
        logger.debug(f"Agent {agent_id} joined channel {channel_id}")
        return True

    async def leave_channel(self, channel_id: str, agent_id: str) -> bool:
        """Remove agent from a channel"""
        channel = self.channels.get(channel_id)
        if not channel:
            return False

        channel.remove_participant(agent_id)
        logger.debug(f"Agent {agent_id} left channel {channel_id}")
        return True

    async def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """Subscribe agent to a topic for broadcasts"""
        self.subscriptions[topic].add(agent_id)
        self.stats['active_subscriptions'] += 1
        logger.debug(f"Agent {agent_id} subscribed to topic {topic}")
        return True

    async def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe agent from a topic"""
        self.subscriptions[topic].discard(agent_id)
        if not self.subscriptions[topic]:
            del self.subscriptions[topic]
        self.stats['active_subscriptions'] = max(0, self.stats['active_subscriptions'] - 1)
        logger.debug(f"Agent {agent_id} unsubscribed from topic {topic}")
        return True

    async def receive_messages(self, agent_id: str, message_filter: Optional[MessageFilter] = None,
                             timeout_seconds: Optional[float] = None) -> List[Message]:
        """Receive messages for an agent"""
        messages = []

        try:
            # Get messages from queue
            queue = self.message_queues[agent_id]
            while not queue.empty():
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if not message.is_expired():
                        if message_filter is None or message_filter.matches(message):
                            messages.append(message)
                        queue.task_done()
                except asyncio.TimeoutError:
                    break

            self.stats['messages_received'] += len(messages)
            return messages

        except Exception as e:
            logger.error(f"Error receiving messages for agent {agent_id}: {e}")
            return []

    async def get_message_history(self, agent_id: str, conversation_id: Optional[str] = None,
                                 limit: int = 100) -> List[Message]:
        """Get message history for an agent"""
        history = []

        for message in reversed(self.message_history):
            if len(history) >= limit:
                break

            # Check if message involves this agent
            if (message.sender_id == agent_id or
                message.recipient_id == agent_id or
                message.recipient_id is None):  # Broadcast message

                if conversation_id is None or message.conversation_id == conversation_id:
                    history.append(message)

        return list(reversed(history))

    def add_message_handler(self, message_type: MessageType,
                           handler: Callable[[Message], None]) -> None:
        """Add a message handler for specific message types"""
        self.message_handlers[message_type].append(handler)

    def add_middleware(self, middleware_func: Callable[[Message], Message]) -> None:
        """Add middleware for message processing"""
        self.middleware.append(middleware_func)

    async def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        active_queues = len([q for q in self.message_queues.values() if not q.empty()])

        self.stats.update({
            'active_channels': len(self.channels),
            'active_subscriptions': sum(len(subs) for subs in self.subscriptions.values()),
            'active_message_queues': active_queues,
            'pending_requests': len(self.pending_requests),
            'message_history_size': len(self.message_history)
        })

        return self.stats.copy()

    async def _route_message(self, message: Message) -> None:
        """Route message to appropriate recipients"""
        if message.message_type == MessageType.BROADCAST:
            await self._broadcast_message(message)
        elif message.recipient_id:
            await self._deliver_message(message, message.recipient_id)
        else:
            logger.warning(f"Message {message.message_id} has no valid recipients")

    async def _broadcast_message(self, message: Message) -> None:
        """Broadcast message to all relevant agents"""
        recipients = set()

        # Add all agents in message queues
        recipients.update(self.message_queues.keys())

        # Add topic subscribers if message has topic metadata
        topic = message.metadata.get('topic')
        if topic and topic in self.subscriptions:
            recipients.update(self.subscriptions[topic])

        # Deliver to all recipients except sender
        for recipient_id in recipients:
            if recipient_id != message.sender_id:
                await self._deliver_message(message, recipient_id)

    async def _deliver_message(self, message: Message, recipient_id: str) -> None:
        """Deliver message to a specific recipient"""
        try:
            queue = self.message_queues[recipient_id]

            # Check queue size
            if queue.qsize() >= self.config.max_queue_size:
                logger.warning(f"Message queue for {recipient_id} is full, dropping oldest message")
                try:
                    await asyncio.wait_for(queue.get(), timeout=0.1)
                    queue.task_done()
                except asyncio.TimeoutError:
                    pass

            await queue.put(message)

            # Distributed delivery via Redis
            if self.redis_client:
                await self._publish_to_redis(message, recipient_id)

            self.stats['messages_delivered'] += 1
            logger.debug(f"Delivered message {message.message_id} to {recipient_id}")

        except Exception as e:
            logger.error(f"Failed to deliver message {message.message_id} to {recipient_id}: {e}")
            self.stats['messages_failed'] += 1

    async def _publish_to_redis(self, message: Message, recipient_id: str) -> None:
        """Publish message to Redis for distributed delivery"""
        try:
            channel = f"agent_messages:{recipient_id}"
            message_data = json.dumps(message.to_dict())
            await self.redis_client.publish(channel, message_data)
        except Exception as e:
            logger.error(f"Failed to publish message to Redis: {e}")

    async def _message_processor(self) -> None:
        """Background task to process messages"""
        while self.communication_active:
            try:
                # Process message handlers
                for message in list(self.message_history)[-100:]:  # Process recent messages
                    for handler in self.message_handlers.get(message.message_type, []):
                        try:
                            handler(message)
                        except Exception as e:
                            logger.error(f"Error in message handler: {e}")

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)

    async def _redis_subscriber(self) -> None:
        """Subscribe to Redis channels for distributed messages"""
        if not self.redis_client:
            return

        try:
            pubsub = self.redis_client.pubsub()

            # Subscribe to all agent message channels
            for agent_id in self.message_queues.keys():
                await pubsub.subscribe(f"agent_messages:{agent_id}")

            while self.communication_active:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        await self._handle_redis_message(message)
                except asyncio.TimeoutError:
                    continue

        except Exception as e:
            logger.error(f"Error in Redis subscriber: {e}")

    async def _handle_redis_message(self, redis_message: Dict[str, Any]) -> None:
        """Handle message received from Redis"""
        try:
            message_data = json.loads(redis_message['data'])
            message = Message.from_dict(message_data)

            # Extract recipient from channel name
            channel = redis_message['channel'].decode()
            recipient_id = channel.split(':')[1]

            # Deliver locally
            await self._deliver_message(message, recipient_id)

        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")

    async def _cleanup_expired_messages(self) -> None:
        """Clean up expired messages"""
        while self.communication_active:
            try:
                # Clean up message history
                current_time = datetime.now()
                self.message_history = deque(
                    [msg for msg in self.message_history if not msg.is_expired()],
                    maxlen=10000
                )

                # Clean up pending requests (timeout handling)
                expired_requests = [
                    correlation_id for correlation_id, future in self.pending_requests.items()
                    if future.done()
                ]

                for correlation_id in expired_requests:
                    self.pending_requests.pop(correlation_id, None)

                await asyncio.sleep(60)  # Cleanup every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message cleanup: {e}")
                await asyncio.sleep(10)

    async def _stats_collector(self) -> None:
        """Collect communication statistics"""
        while self.communication_active:
            try:
                await self.get_communication_stats()
                await asyncio.sleep(30)  # Update stats every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats collector: {e}")
                await asyncio.sleep(10)