"""
Event Bus for inter-agent communication.

This module provides a pub/sub event bus that enables agents to communicate
with each other asynchronously through events.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set, Union
from enum import Enum
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    An event in the pub/sub system.

    Attributes:
        type: Event type identifier
        source: ID of the agent/component that emitted the event
        data: Event payload data
        timestamp: When the event was created
        id: Unique event identifier
        priority: Event priority for ordering
        correlation_id: ID to correlate related events
        reply_to: Event ID this is a reply to
        metadata: Additional event metadata
    """
    type: str
    source: str
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create an Event from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=data["type"],
            source=data["source"],
            data=data.get("data"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            priority=EventPriority(data.get("priority", 1)),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {}),
        )

    def create_reply(self, source: str, data: Any = None) -> "Event":
        """
        Create a reply event.

        Args:
            source: Source of the reply
            data: Reply data

        Returns:
            A new Event configured as a reply
        """
        return Event(
            type=f"{self.type}.reply",
            source=source,
            data=data,
            reply_to=self.id,
            correlation_id=self.correlation_id or self.id,
        )


# Type alias for event handlers
EventHandler = Union[
    Callable[[Event], None],
    Callable[[Event], Awaitable[None]]
]


@dataclass
class Subscription:
    """
    A subscription to events.

    Attributes:
        id: Unique subscription identifier
        event_type: Event type pattern to subscribe to
        handler: Handler function for events
        source_filter: Optional filter by event source
        once: Whether to unsubscribe after first event
        priority: Handler priority (higher runs first)
    """
    id: str
    event_type: str
    handler: EventHandler
    source_filter: Optional[str] = None
    once: bool = False
    priority: int = 0


class EventBus:
    """
    Pub/sub event bus for inter-agent communication.

    Supports:
    - Event type subscriptions with wildcards
    - Async and sync handlers
    - Source filtering
    - One-time subscriptions
    - Request-reply pattern
    - Event history

    Example:
        bus = EventBus()

        # Subscribe to events
        def handle_task_complete(event: Event):
            print(f"Task completed: {event.data}")

        bus.subscribe("task.complete", handle_task_complete)

        # Publish an event
        await bus.publish(Event(
            type="task.complete",
            source="agent-1",
            data={"task_id": "123", "result": "success"}
        ))

        # Request-reply pattern
        responses = await bus.publish_and_wait(
            Event(type="query.status", source="monitor"),
            timeout=5
        )
    """

    def __init__(self, history_size: int = 1000):
        """
        Initialize the event bus.

        Args:
            history_size: Maximum number of events to keep in history
        """
        self.subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._history: List[Event] = []
        self._history_size = history_size
        self._pending_replies: Dict[str, asyncio.Queue[Event]] = {}
        self._subscription_count = 0

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        source_filter: Optional[str] = None,
        once: bool = False,
        priority: int = 0
    ) -> str:
        """
        Subscribe to events of a given type.

        Args:
            event_type: Event type to subscribe to (supports * wildcard)
            handler: Function to call when event is received
            source_filter: Only handle events from this source
            once: Unsubscribe after handling one event
            priority: Handler priority (higher runs first)

        Returns:
            Subscription ID
        """
        self._subscription_count += 1
        subscription_id = f"sub_{self._subscription_count}"

        subscription = Subscription(
            id=subscription_id,
            event_type=event_type,
            handler=handler,
            source_filter=source_filter,
            once=once,
            priority=priority,
        )

        # Insert sorted by priority (descending)
        subs = self.subscribers[event_type]
        insert_idx = 0
        for i, sub in enumerate(subs):
            if sub.priority < priority:
                insert_idx = i
                break
            insert_idx = i + 1
        subs.insert(insert_idx, subscription)

        logger.debug(f"Subscribed to {event_type}: {subscription_id}")
        return subscription_id

    def unsubscribe(
        self,
        event_type: str,
        handler: Optional[EventHandler] = None,
        subscription_id: Optional[str] = None
    ) -> bool:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from
            handler: Specific handler to remove (optional)
            subscription_id: Specific subscription to remove (optional)

        Returns:
            True if any subscriptions were removed
        """
        if event_type not in self.subscribers:
            return False

        original_count = len(self.subscribers[event_type])

        if subscription_id:
            self.subscribers[event_type] = [
                s for s in self.subscribers[event_type]
                if s.id != subscription_id
            ]
        elif handler:
            self.subscribers[event_type] = [
                s for s in self.subscribers[event_type]
                if s.handler != handler
            ]
        else:
            # Remove all subscriptions for this event type
            del self.subscribers[event_type]
            return True

        removed = original_count - len(self.subscribers[event_type])
        if removed:
            logger.debug(f"Unsubscribed {removed} handler(s) from {event_type}")
        return removed > 0

    def unsubscribe_all(self, source: Optional[str] = None) -> int:
        """
        Unsubscribe all handlers.

        Args:
            source: Optionally only unsubscribe handlers filtering this source

        Returns:
            Number of subscriptions removed
        """
        if source is None:
            count = sum(len(subs) for subs in self.subscribers.values())
            self.subscribers.clear()
            return count

        count = 0
        for event_type in list(self.subscribers.keys()):
            original = len(self.subscribers[event_type])
            self.subscribers[event_type] = [
                s for s in self.subscribers[event_type]
                if s.source_filter != source
            ]
            count += original - len(self.subscribers[event_type])
        return count

    async def publish(self, event: Event) -> int:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish

        Returns:
            Number of handlers that were invoked
        """
        # Add to history
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history.pop(0)

        # Check for pending reply listeners
        if event.reply_to and event.reply_to in self._pending_replies:
            await self._pending_replies[event.reply_to].put(event)

        # Find matching subscribers
        handlers_invoked = 0
        subscriptions_to_remove: List[tuple[str, str]] = []

        for event_pattern, subscriptions in self.subscribers.items():
            if not self._matches_pattern(event_pattern, event.type):
                continue

            for subscription in subscriptions:
                # Check source filter
                if subscription.source_filter and subscription.source_filter != event.source:
                    continue

                # Invoke handler
                try:
                    result = subscription.handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                    handlers_invoked += 1
                except Exception as e:
                    logger.error(f"Error in event handler for {event.type}: {e}")

                # Mark for removal if one-time subscription
                if subscription.once:
                    subscriptions_to_remove.append((event_pattern, subscription.id))

        # Remove one-time subscriptions
        for event_pattern, sub_id in subscriptions_to_remove:
            self.unsubscribe(event_pattern, subscription_id=sub_id)

        logger.debug(f"Published event {event.type} from {event.source}, {handlers_invoked} handlers invoked")
        return handlers_invoked

    async def publish_and_wait(
        self,
        event: Event,
        timeout: int = 30,
        expected_replies: int = 1
    ) -> List[Event]:
        """
        Publish an event and wait for replies.

        This implements a request-reply pattern where the publisher
        waits for responses from handlers.

        Args:
            event: The event to publish
            timeout: Timeout in seconds
            expected_replies: Number of replies to wait for

        Returns:
            List of reply events

        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        # Set up reply queue
        reply_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._pending_replies[event.id] = reply_queue

        try:
            # Publish the event
            await self.publish(event)

            # Wait for replies
            replies = []
            deadline = asyncio.get_event_loop().time() + timeout

            while len(replies) < expected_replies:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break

                try:
                    reply = await asyncio.wait_for(
                        reply_queue.get(),
                        timeout=remaining
                    )
                    replies.append(reply)
                except asyncio.TimeoutError:
                    break

            return replies

        finally:
            # Clean up reply queue
            del self._pending_replies[event.id]

    async def request(
        self,
        event_type: str,
        source: str,
        data: Any = None,
        timeout: int = 30
    ) -> Optional[Event]:
        """
        Send a request event and wait for a single reply.

        Convenience method for simple request-reply patterns.

        Args:
            event_type: Type of request event
            source: Source identifier
            data: Request data
            timeout: Timeout in seconds

        Returns:
            Reply event or None if timeout
        """
        event = Event(type=event_type, source=source, data=data)
        replies = await self.publish_and_wait(event, timeout=timeout, expected_replies=1)
        return replies[0] if replies else None

    def on(self, event_type: str) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator for subscribing to events.

        Example:
            @bus.on("task.complete")
            async def handle_task_complete(event: Event):
                print(event.data)

        Args:
            event_type: Event type to subscribe to

        Returns:
            Decorator function
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler)
            return handler
        return decorator

    def once(self, event_type: str) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator for one-time event subscription.

        Args:
            event_type: Event type to subscribe to

        Returns:
            Decorator function
        """
        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler, once=True)
            return handler
        return decorator

    def get_history(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type
            source: Filter by source
            limit: Maximum events to return

        Returns:
            List of matching events (most recent first)
        """
        events = self._history.copy()
        events.reverse()  # Most recent first

        if event_type:
            events = [e for e in events if self._matches_pattern(event_type, e.type)]

        if source:
            events = [e for e in events if e.source == source]

        return events[:limit]

    def clear_history(self) -> None:
        """Clear the event history."""
        self._history.clear()

    def get_subscription_count(self, event_type: Optional[str] = None) -> int:
        """
        Get number of subscriptions.

        Args:
            event_type: Optionally filter by event type

        Returns:
            Number of subscriptions
        """
        if event_type:
            return len(self.subscribers.get(event_type, []))
        return sum(len(subs) for subs in self.subscribers.values())

    def get_subscribed_types(self) -> Set[str]:
        """
        Get all event types with active subscriptions.

        Returns:
            Set of event type patterns
        """
        return set(self.subscribers.keys())

    def _matches_pattern(self, pattern: str, event_type: str) -> bool:
        """
        Check if an event type matches a subscription pattern.

        Supports:
        - Exact match: "task.complete"
        - Wildcard suffix: "task.*"
        - Full wildcard: "*"

        Args:
            pattern: Subscription pattern
            event_type: Event type to check

        Returns:
            True if pattern matches event type
        """
        if pattern == "*":
            return True
        if pattern == event_type:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix + ".") or event_type == prefix
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return event_type.startswith(prefix)
        return False


class EventEmitter:
    """
    Mixin class for objects that emit events.

    Provides convenient methods for emitting events to an event bus.

    Example:
        class MyAgent(EventEmitter):
            def __init__(self, bus: EventBus):
                super().__init__(bus, source_id="my-agent")

            async def do_work(self):
                await self.emit("work.started", {"task": "example"})
                # ... do work ...
                await self.emit("work.completed", {"result": "success"})
    """

    def __init__(self, event_bus: EventBus, source_id: str):
        """
        Initialize the event emitter.

        Args:
            event_bus: Event bus to emit events to
            source_id: Source identifier for emitted events
        """
        self._event_bus = event_bus
        self._source_id = source_id

    async def emit(
        self,
        event_type: str,
        data: Any = None,
        **metadata
    ) -> int:
        """
        Emit an event.

        Args:
            event_type: Type of event
            data: Event data
            **metadata: Additional metadata

        Returns:
            Number of handlers invoked
        """
        event = Event(
            type=event_type,
            source=self._source_id,
            data=data,
            metadata=metadata,
        )
        return await self._event_bus.publish(event)

    async def emit_and_wait(
        self,
        event_type: str,
        data: Any = None,
        timeout: int = 30,
        expected_replies: int = 1,
        **metadata
    ) -> List[Event]:
        """
        Emit an event and wait for replies.

        Args:
            event_type: Type of event
            data: Event data
            timeout: Timeout in seconds
            expected_replies: Number of replies to wait for
            **metadata: Additional metadata

        Returns:
            List of reply events
        """
        event = Event(
            type=event_type,
            source=self._source_id,
            data=data,
            metadata=metadata,
        )
        return await self._event_bus.publish_and_wait(
            event,
            timeout=timeout,
            expected_replies=expected_replies
        )
