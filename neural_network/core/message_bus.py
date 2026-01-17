"""
Message Bus - Inter-form communication following existing JSON protocol
Part of the Neural Network module for the Consciousness system.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
import json

from .model_registry import Priority

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the consciousness system."""
    # Core message types
    SENSORY_INPUT = "sensory_input"
    COGNITIVE_UPDATE = "cognitive_update"
    EMOTIONAL_STATE = "emotional_state"
    AROUSAL_UPDATE = "arousal_update"
    WORKSPACE_BROADCAST = "workspace_broadcast"
    ATTENTION_REQUEST = "attention_request"
    MEMORY_QUERY = "memory_query"
    MEMORY_RESPONSE = "memory_response"

    # Control messages
    FORM_STATUS = "form_status"
    RESOURCE_REQUEST = "resource_request"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"

    # IIT and theoretical
    PHI_UPDATE = "phi_update"
    INTEGRATION_SIGNAL = "integration_signal"
    PREDICTION_ERROR = "prediction_error"
    RECURRENT_FEEDBACK = "recurrent_feedback"


@dataclass
class MessageHeader:
    """Header information for a message."""
    message_id: str
    source_form: str
    target_form: Optional[str]  # None for broadcast
    timestamp: datetime
    priority: Priority
    message_type: MessageType
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_ms: int = 5000  # Time to live


@dataclass
class MessageFooter:
    """Footer information for message integrity."""
    checksum: str
    sequence_number: int
    compressed: bool = False
    encrypted: bool = False


@dataclass
class FormMessage:
    """
    Complete message format following existing interface-spec.md.

    Messages are the primary communication mechanism between consciousness forms.
    """
    header: MessageHeader
    body: Dict[str, Any]
    footer: MessageFooter

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'header': {
                'message_id': self.header.message_id,
                'source_form': self.header.source_form,
                'target_form': self.header.target_form,
                'timestamp': self.header.timestamp.isoformat(),
                'priority': self.header.priority.name,
                'message_type': self.header.message_type.value,
                'correlation_id': self.header.correlation_id,
                'reply_to': self.header.reply_to,
                'ttl_ms': self.header.ttl_ms,
            },
            'body': self.body,
            'footer': {
                'checksum': self.footer.checksum,
                'sequence_number': self.footer.sequence_number,
                'compressed': self.footer.compressed,
                'encrypted': self.footer.encrypted,
            },
        }

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FormMessage":
        """Create message from dictionary."""
        header_data = data['header']
        footer_data = data['footer']

        header = MessageHeader(
            message_id=header_data['message_id'],
            source_form=header_data['source_form'],
            target_form=header_data.get('target_form'),
            timestamp=datetime.fromisoformat(header_data['timestamp']),
            priority=Priority[header_data['priority']],
            message_type=MessageType(header_data['message_type']),
            correlation_id=header_data.get('correlation_id'),
            reply_to=header_data.get('reply_to'),
            ttl_ms=header_data.get('ttl_ms', 5000),
        )

        footer = MessageFooter(
            checksum=footer_data['checksum'],
            sequence_number=footer_data['sequence_number'],
            compressed=footer_data.get('compressed', False),
            encrypted=footer_data.get('encrypted', False),
        )

        return cls(header=header, body=data['body'], footer=footer)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        age_ms = (datetime.now(timezone.utc) - self.header.timestamp).total_seconds() * 1000
        return age_ms > self.header.ttl_ms


MessageCallback = Callable[[FormMessage], Awaitable[None]]


class MessageQueue:
    """Priority-aware message queue."""

    def __init__(self, max_size: int = 10000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._sequence = 0

    async def put(self, message: FormMessage) -> None:
        """Add message to queue with priority ordering."""
        # Priority queue uses (priority, sequence, item)
        # Lower values = higher priority, so invert priority
        priority_value = -message.header.priority.value
        await self._queue.put((priority_value, self._sequence, message))
        self._sequence += 1

    async def get(self) -> FormMessage:
        """Get highest priority message from queue."""
        _, _, message = await self._queue.get()
        return message

    def get_nowait(self) -> Optional[FormMessage]:
        """Get message without waiting, returns None if empty."""
        try:
            _, _, message = self._queue.get_nowait()
            return message
        except asyncio.QueueEmpty:
            return None

    @property
    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class MessageBus:
    """
    Inter-form communication following existing JSON protocol.

    The MessageBus handles all communication between consciousness forms,
    implementing priority queues, pub/sub patterns, and global broadcasting.
    """

    # Priority-based latency targets
    LATENCY_TARGETS_MS = {
        Priority.CRITICAL: 10,
        Priority.HIGH: 30,
        Priority.NORMAL: 50,
        Priority.LOW: 200,
    }

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the message bus.

        Args:
            max_queue_size: Maximum messages per queue
        """
        # Separate queues by priority for guaranteed ordering
        self.queues: Dict[Priority, MessageQueue] = {
            priority: MessageQueue(max_queue_size)
            for priority in Priority
        }

        # Subscribers: form_id -> list of callbacks
        self.subscribers: Dict[str, List[MessageCallback]] = {}

        # Channel subscriptions: channel_name -> set of form_ids
        self.channels: Dict[str, Set[str]] = {}

        # Global workspace broadcast channel
        self.channels['global_workspace'] = set()
        self.channels['arousal'] = set()
        self.channels['emergency'] = set()

        # Message tracking
        self._sequence_counter = 0
        self._message_count = 0
        self._dropped_count = 0

        # Processing state
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        self._lock = asyncio.Lock()

    def _generate_message_id(self) -> str:
        """Generate a unique message ID."""
        self._sequence_counter += 1
        timestamp = datetime.now(timezone.utc).timestamp()
        return f"msg_{timestamp:.6f}_{self._sequence_counter}"

    def _compute_checksum(self, body: Dict[str, Any]) -> str:
        """Compute checksum for message body."""
        body_str = json.dumps(body, sort_keys=True)
        return hashlib.md5(body_str.encode()).hexdigest()[:8]

    def create_message(
        self,
        source_form: str,
        target_form: Optional[str],
        message_type: MessageType,
        body: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        ttl_ms: int = 5000,
    ) -> FormMessage:
        """
        Create a new message with proper formatting.

        Args:
            source_form: ID of the sending form
            target_form: ID of the target form (None for broadcast)
            message_type: Type of message
            body: Message payload
            priority: Message priority
            correlation_id: ID for request-response correlation
            reply_to: Form ID to reply to
            ttl_ms: Time to live in milliseconds

        Returns:
            A properly formatted FormMessage
        """
        message_id = self._generate_message_id()
        timestamp = datetime.now(timezone.utc)
        checksum = self._compute_checksum(body)

        header = MessageHeader(
            message_id=message_id,
            source_form=source_form,
            target_form=target_form,
            timestamp=timestamp,
            priority=priority,
            message_type=message_type,
            correlation_id=correlation_id,
            reply_to=reply_to,
            ttl_ms=ttl_ms,
        )

        footer = MessageFooter(
            checksum=checksum,
            sequence_number=self._sequence_counter,
        )

        return FormMessage(header=header, body=body, footer=footer)

    async def publish(self, message: FormMessage) -> None:
        """
        Publish a message to the appropriate queue.

        Args:
            message: The message to publish
        """
        # Validate checksum
        computed = self._compute_checksum(message.body)
        if computed != message.footer.checksum:
            logger.warning(f"Checksum mismatch for message {message.header.message_id}")

        # Check if expired
        if message.is_expired():
            self._dropped_count += 1
            logger.debug(f"Dropped expired message {message.header.message_id}")
            return

        # Add to appropriate priority queue
        queue = self.queues[message.header.priority]
        try:
            await queue.put(message)
            self._message_count += 1

            logger.debug(
                f"Published message {message.header.message_id} "
                f"from {message.header.source_form} "
                f"to {message.header.target_form or 'broadcast'}"
            )

        except asyncio.QueueFull:
            self._dropped_count += 1
            logger.warning(
                f"Queue full, dropped message {message.header.message_id}"
            )

    async def subscribe(
        self,
        form_id: str,
        callback: MessageCallback
    ) -> None:
        """
        Subscribe a form to receive messages.

        Args:
            form_id: The form ID to subscribe
            callback: Async callback to invoke when messages arrive
        """
        async with self._lock:
            if form_id not in self.subscribers:
                self.subscribers[form_id] = []
            self.subscribers[form_id].append(callback)
            logger.debug(f"Form {form_id} subscribed to message bus")

    async def unsubscribe(self, form_id: str) -> None:
        """
        Unsubscribe a form from messages.

        Args:
            form_id: The form ID to unsubscribe
        """
        async with self._lock:
            if form_id in self.subscribers:
                del self.subscribers[form_id]
                logger.debug(f"Form {form_id} unsubscribed from message bus")

    async def subscribe_channel(
        self,
        form_id: str,
        channel: str
    ) -> None:
        """
        Subscribe a form to a broadcast channel.

        Args:
            form_id: The form ID to subscribe
            channel: The channel name (e.g., 'global_workspace', 'arousal')
        """
        async with self._lock:
            if channel not in self.channels:
                self.channels[channel] = set()
            self.channels[channel].add(form_id)
            logger.debug(f"Form {form_id} subscribed to channel {channel}")

    async def unsubscribe_channel(
        self,
        form_id: str,
        channel: str
    ) -> None:
        """Unsubscribe a form from a channel."""
        async with self._lock:
            if channel in self.channels:
                self.channels[channel].discard(form_id)

    async def broadcast(
        self,
        message: FormMessage,
        channel: str
    ) -> None:
        """
        Broadcast a message to a channel (Form 14 integration).

        Args:
            message: The message to broadcast
            channel: The channel to broadcast on
        """
        if channel not in self.channels:
            logger.warning(f"Unknown broadcast channel: {channel}")
            return

        # Deliver to all channel subscribers
        subscribers = self.channels[channel]
        delivery_count = 0

        for form_id in subscribers:
            if form_id in self.subscribers:
                for callback in self.subscribers[form_id]:
                    try:
                        await callback(message)
                        delivery_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error delivering broadcast to {form_id}: {e}"
                        )

        logger.debug(
            f"Broadcast on {channel}: "
            f"delivered to {delivery_count}/{len(subscribers)} forms"
        )

    async def send_to_form(
        self,
        source_form: str,
        target_form: str,
        message_type: MessageType,
        body: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """
        Convenience method to send a message to a specific form.

        Returns:
            The message ID
        """
        message = self.create_message(
            source_form=source_form,
            target_form=target_form,
            message_type=message_type,
            body=body,
            priority=priority,
        )
        await self.publish(message)
        return message.header.message_id

    async def broadcast_to_channel(
        self,
        source_form: str,
        channel: str,
        message_type: MessageType,
        body: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """
        Convenience method to broadcast to a channel.

        Returns:
            The message ID
        """
        message = self.create_message(
            source_form=source_form,
            target_form=None,
            message_type=message_type,
            body=body,
            priority=priority,
        )
        await self.broadcast(message, channel)
        return message.header.message_id

    async def start_processing(self) -> None:
        """Start the message processing loop."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("Message bus processing started")

    async def stop_processing(self) -> None:
        """Stop the message processing loop."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Message bus processing stopped")

    async def _process_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            processed = False

            # Process by priority order
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                queue = self.queues[priority]
                if not queue.empty():
                    message = queue.get_nowait()
                    if message:
                        await self._deliver_message(message)
                        processed = True

            if not processed:
                # Small delay when no messages
                await asyncio.sleep(0.001)

    async def _deliver_message(self, message: FormMessage) -> None:
        """Deliver a message to its target(s)."""
        # Skip expired messages
        if message.is_expired():
            self._dropped_count += 1
            return

        target = message.header.target_form

        if target:
            # Direct message to specific form
            if target in self.subscribers:
                for callback in self.subscribers[target]:
                    try:
                        await callback(message)
                    except Exception as e:
                        logger.error(f"Error delivering to {target}: {e}")
        else:
            # Broadcast to all subscribers
            for form_id, callbacks in self.subscribers.items():
                if form_id != message.header.source_form:
                    for callback in callbacks:
                        try:
                            await callback(message)
                        except Exception as e:
                            logger.error(f"Error delivering to {form_id}: {e}")

    @property
    def queue_depth(self) -> int:
        """Get total messages across all queues."""
        return sum(q.qsize for q in self.queues.values())

    def get_status(self) -> Dict[str, Any]:
        """Get message bus status."""
        return {
            'running': self._running,
            'subscribers': len(self.subscribers),
            'channels': {
                name: len(forms)
                for name, forms in self.channels.items()
            },
            'queues': {
                priority.name: self.queues[priority].qsize
                for priority in Priority
            },
            'total_queue_depth': self.queue_depth,
            'message_count': self._message_count,
            'dropped_count': self._dropped_count,
        }
