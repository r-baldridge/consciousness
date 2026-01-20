"""
Message Handlers - Processing system for inter-form messages
Part of the Neural Network module for the Consciousness system.

This module provides:
- Handler registry for message types
- Handler coordinator for routing messages
- Base handler classes for common patterns
- Form-specific handler implementations
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Set, Type, Union, TYPE_CHECKING
)

from .message_bus import (
    MessageBus, FormMessage, MessageType, MessageHeader, MessageFooter, Priority
)

if TYPE_CHECKING:
    from .nervous_system import NervousSystem
    from ..adapters.base_adapter import FormAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# HANDLER RESULT TYPES
# =============================================================================

class HandlerResult(Enum):
    """Result of message handling."""
    HANDLED = "handled"  # Message was fully processed
    FORWARDED = "forwarded"  # Message was forwarded to another handler
    PENDING = "pending"  # Message requires async completion
    IGNORED = "ignored"  # Message was not relevant to this handler
    ERROR = "error"  # Handler encountered an error


@dataclass
class HandlerResponse:
    """Response from a message handler."""
    result: HandlerResult
    response_body: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    forward_to: Optional[str] = None  # Form ID to forward to
    broadcast_channel: Optional[str] = None  # Channel to broadcast response
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if handling was successful."""
        return self.result in (HandlerResult.HANDLED, HandlerResult.FORWARDED, HandlerResult.PENDING)


# Type for handler functions
MessageHandler = Callable[[FormMessage, "MessageHandlerContext"], Awaitable[HandlerResponse]]


@dataclass
class MessageHandlerContext:
    """Context provided to message handlers."""
    message_bus: MessageBus
    nervous_system: Optional["NervousSystem"]
    adapters: Dict[str, "FormAdapter"]
    handler_id: str
    correlation_data: Dict[str, Any] = field(default_factory=dict)

    async def reply(
        self,
        original_message: FormMessage,
        response_type: MessageType,
        body: Dict[str, Any],
        priority: Optional[Priority] = None
    ) -> str:
        """Send a reply to the original message sender."""
        reply_msg = self.message_bus.create_message(
            source_form=original_message.header.target_form or "system",
            target_form=original_message.header.source_form,
            message_type=response_type,
            body=body,
            priority=priority or original_message.header.priority,
            correlation_id=original_message.header.correlation_id,
            reply_to=original_message.header.message_id,
        )
        await self.message_bus.publish(reply_msg)
        return reply_msg.header.message_id

    async def broadcast(
        self,
        source_form: str,
        channel: str,
        message_type: MessageType,
        body: Dict[str, Any],
        priority: Priority = Priority.NORMAL
    ) -> str:
        """Broadcast a message to a channel."""
        return await self.message_bus.broadcast_to_channel(
            source_form=source_form,
            channel=channel,
            message_type=message_type,
            body=body,
            priority=priority
        )

    def get_adapter(self, form_id: str) -> Optional["FormAdapter"]:
        """Get an adapter by form ID."""
        return self.adapters.get(form_id)


# =============================================================================
# BASE HANDLER CLASSES
# =============================================================================

class BaseMessageHandler(ABC):
    """Base class for message handlers."""

    def __init__(self, handler_id: str, handled_types: List[MessageType]):
        """
        Initialize the handler.

        Args:
            handler_id: Unique identifier for this handler
            handled_types: Message types this handler processes
        """
        self.handler_id = handler_id
        self.handled_types = set(handled_types)
        self._message_count = 0
        self._error_count = 0
        self._last_handled: Optional[datetime] = None

    def can_handle(self, message: FormMessage) -> bool:
        """Check if this handler can process the message."""
        return message.header.message_type in self.handled_types

    @abstractmethod
    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """
        Handle the message.

        Args:
            message: The message to handle
            context: Handler context with bus and adapters

        Returns:
            HandlerResponse indicating result
        """
        pass

    async def __call__(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Allow handler to be called as a function."""
        try:
            result = await self.handle(message, context)
            self._message_count += 1
            self._last_handled = datetime.now(timezone.utc)
            return result
        except Exception as e:
            self._error_count += 1
            logger.error(f"Handler {self.handler_id} error: {e}")
            return HandlerResponse(
                result=HandlerResult.ERROR,
                error_message=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "handler_id": self.handler_id,
            "handled_types": [t.value for t in self.handled_types],
            "message_count": self._message_count,
            "error_count": self._error_count,
            "last_handled": self._last_handled.isoformat() if self._last_handled else None,
        }


class QueryResponseHandler(BaseMessageHandler):
    """
    Handler for query-response patterns.

    Subclasses implement process_query() to handle queries
    and automatically generate responses.
    """

    def __init__(
        self,
        handler_id: str,
        query_type: MessageType,
        response_type: MessageType
    ):
        super().__init__(handler_id, [query_type])
        self.query_type = query_type
        self.response_type = response_type

    @abstractmethod
    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        """
        Process a query and return response body.

        Args:
            query_body: The query message body
            context: Handler context

        Returns:
            Response body dictionary
        """
        pass

    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Handle query by processing and sending response."""
        try:
            response_body = await self.process_query(message.body, context)

            # Send reply
            await context.reply(
                original_message=message,
                response_type=self.response_type,
                body=response_body
            )

            return HandlerResponse(
                result=HandlerResult.HANDLED,
                response_body=response_body
            )

        except Exception as e:
            logger.error(f"Query processing error in {self.handler_id}: {e}")
            return HandlerResponse(
                result=HandlerResult.ERROR,
                error_message=str(e)
            )


class BroadcastHandler(BaseMessageHandler):
    """Handler that broadcasts responses to channels."""

    def __init__(
        self,
        handler_id: str,
        handled_types: List[MessageType],
        broadcast_channel: str
    ):
        super().__init__(handler_id, handled_types)
        self.broadcast_channel = broadcast_channel

    @abstractmethod
    async def process_for_broadcast(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> Optional[Dict[str, Any]]:
        """
        Process message and return broadcast body if applicable.

        Returns:
            Body to broadcast, or None to skip broadcast
        """
        pass

    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Handle by processing and optionally broadcasting."""
        broadcast_body = await self.process_for_broadcast(message, context)

        if broadcast_body:
            await context.broadcast(
                source_form=message.header.source_form,
                channel=self.broadcast_channel,
                message_type=message.header.message_type,
                body=broadcast_body,
                priority=message.header.priority
            )
            return HandlerResponse(
                result=HandlerResult.HANDLED,
                response_body=broadcast_body,
                broadcast_channel=self.broadcast_channel
            )

        return HandlerResponse(result=HandlerResult.IGNORED)


# =============================================================================
# HANDLER REGISTRY
# =============================================================================

class MessageHandlerRegistry:
    """
    Registry for message handlers.

    Maps message types to handlers and manages handler lifecycle.
    """

    def __init__(self):
        # Type -> list of handlers (multiple handlers per type allowed)
        self._handlers: Dict[MessageType, List[BaseMessageHandler]] = {}
        # Handler ID -> handler instance
        self._handlers_by_id: Dict[str, BaseMessageHandler] = {}
        # Form ID -> handlers registered by that form
        self._form_handlers: Dict[str, List[str]] = {}

    def register(
        self,
        handler: BaseMessageHandler,
        form_id: Optional[str] = None
    ) -> None:
        """
        Register a message handler.

        Args:
            handler: The handler to register
            form_id: Optional form ID that owns this handler
        """
        # Register by handler ID
        self._handlers_by_id[handler.handler_id] = handler

        # Register for each handled type
        for msg_type in handler.handled_types:
            if msg_type not in self._handlers:
                self._handlers[msg_type] = []
            self._handlers[msg_type].append(handler)

        # Track form ownership
        if form_id:
            if form_id not in self._form_handlers:
                self._form_handlers[form_id] = []
            self._form_handlers[form_id].append(handler.handler_id)

        logger.info(
            f"Registered handler {handler.handler_id} for types: "
            f"{[t.value for t in handler.handled_types]}"
        )

    def unregister(self, handler_id: str) -> bool:
        """
        Unregister a handler by ID.

        Args:
            handler_id: The handler ID to remove

        Returns:
            True if handler was found and removed
        """
        if handler_id not in self._handlers_by_id:
            return False

        handler = self._handlers_by_id[handler_id]

        # Remove from type mappings
        for msg_type in handler.handled_types:
            if msg_type in self._handlers:
                self._handlers[msg_type] = [
                    h for h in self._handlers[msg_type]
                    if h.handler_id != handler_id
                ]

        # Remove from ID mapping
        del self._handlers_by_id[handler_id]

        # Remove from form mappings
        for form_id, handlers in self._form_handlers.items():
            if handler_id in handlers:
                handlers.remove(handler_id)

        logger.info(f"Unregistered handler {handler_id}")
        return True

    def unregister_form(self, form_id: str) -> int:
        """
        Unregister all handlers for a form.

        Args:
            form_id: The form ID

        Returns:
            Number of handlers removed
        """
        if form_id not in self._form_handlers:
            return 0

        handler_ids = list(self._form_handlers[form_id])
        count = 0
        for handler_id in handler_ids:
            if self.unregister(handler_id):
                count += 1

        del self._form_handlers[form_id]
        return count

    def get_handlers(self, message_type: MessageType) -> List[BaseMessageHandler]:
        """Get all handlers for a message type."""
        return self._handlers.get(message_type, [])

    def get_handler(self, handler_id: str) -> Optional[BaseMessageHandler]:
        """Get a handler by ID."""
        return self._handlers_by_id.get(handler_id)

    def has_handler(self, message_type: MessageType) -> bool:
        """Check if any handler is registered for a type."""
        return message_type in self._handlers and len(self._handlers[message_type]) > 0

    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        return {
            "total_handlers": len(self._handlers_by_id),
            "types_covered": len(self._handlers),
            "handlers_by_type": {
                t.value: len(handlers)
                for t, handlers in self._handlers.items()
            },
            "forms_with_handlers": len(self._form_handlers),
        }


# =============================================================================
# HANDLER COORDINATOR
# =============================================================================

class MessageHandlerCoordinator:
    """
    Coordinates message handling across the system.

    Integrates with MessageBus to process messages through registered handlers.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        nervous_system: Optional["NervousSystem"] = None
    ):
        self.message_bus = message_bus
        self.nervous_system = nervous_system
        self.registry = MessageHandlerRegistry()

        # Processing statistics
        self._processed_count = 0
        self._handled_count = 0
        self._unhandled_count = 0
        self._error_count = 0

        # Adapters reference (populated from NervousSystem or set directly)
        self._adapters: Dict[str, "FormAdapter"] = {}

        # Running state
        self._running = False

    def set_adapters(self, adapters: Dict[str, "FormAdapter"]) -> None:
        """Set the adapters reference."""
        self._adapters = adapters

    def register_handler(
        self,
        handler: BaseMessageHandler,
        form_id: Optional[str] = None
    ) -> None:
        """Register a handler."""
        self.registry.register(handler, form_id)

    def unregister_handler(self, handler_id: str) -> bool:
        """Unregister a handler."""
        return self.registry.unregister(handler_id)

    def create_context(self, handler_id: str) -> MessageHandlerContext:
        """Create a handler context."""
        adapters = self._adapters
        if self.nervous_system:
            adapters = self.nervous_system.adapters

        return MessageHandlerContext(
            message_bus=self.message_bus,
            nervous_system=self.nervous_system,
            adapters=adapters,
            handler_id=handler_id
        )

    async def process_message(self, message: FormMessage) -> List[HandlerResponse]:
        """
        Process a message through registered handlers.

        Args:
            message: The message to process

        Returns:
            List of responses from handlers
        """
        self._processed_count += 1
        responses = []

        handlers = self.registry.get_handlers(message.header.message_type)

        if not handlers:
            self._unhandled_count += 1
            logger.debug(
                f"No handler for message type {message.header.message_type.value}"
            )
            return responses

        for handler in handlers:
            context = self.create_context(handler.handler_id)

            try:
                response = await handler(message, context)
                responses.append(response)

                if response.result == HandlerResult.HANDLED:
                    self._handled_count += 1
                elif response.result == HandlerResult.ERROR:
                    self._error_count += 1

            except Exception as e:
                self._error_count += 1
                logger.error(f"Handler {handler.handler_id} exception: {e}")
                responses.append(HandlerResponse(
                    result=HandlerResult.ERROR,
                    error_message=str(e)
                ))

        return responses

    async def handle_callback(self, message: FormMessage) -> None:
        """
        MessageBus callback that routes to handlers.

        This can be registered with the MessageBus as a subscriber callback.
        """
        await self.process_message(message)

    async def start(self) -> None:
        """Start the coordinator and subscribe to all messages."""
        if self._running:
            return

        self._running = True

        # Subscribe as a global message handler
        await self.message_bus.subscribe("__coordinator__", self.handle_callback)

        logger.info("Message handler coordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        await self.message_bus.unsubscribe("__coordinator__")
        logger.info("Message handler coordinator stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "running": self._running,
            "processed_count": self._processed_count,
            "handled_count": self._handled_count,
            "unhandled_count": self._unhandled_count,
            "error_count": self._error_count,
            "registry": self.registry.get_status(),
        }


# =============================================================================
# CONCRETE HANDLER IMPLEMENTATIONS
# =============================================================================

# ----- Core System Handlers -----

class ArousalUpdateHandler(BroadcastHandler):
    """Handler for arousal updates from Form 08."""

    def __init__(self):
        super().__init__(
            handler_id="arousal_update_handler",
            handled_types=[MessageType.AROUSAL_UPDATE],
            broadcast_channel="arousal"
        )

    async def process_for_broadcast(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> Optional[Dict[str, Any]]:
        """Process arousal update and broadcast to arousal channel."""
        arousal_level = message.body.get("arousal_level", 0.5)
        arousal_state = message.body.get("arousal_state", "alert")
        gating_signals = message.body.get("gating_signals", {})

        # Always broadcast arousal updates
        return {
            "arousal_level": arousal_level,
            "arousal_state": arousal_state,
            "gating_signals": gating_signals,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class WorkspaceBroadcastHandler(BroadcastHandler):
    """Handler for global workspace broadcasts from Form 14."""

    def __init__(self):
        super().__init__(
            handler_id="workspace_broadcast_handler",
            handled_types=[MessageType.WORKSPACE_BROADCAST],
            broadcast_channel="global_workspace"
        )

    async def process_for_broadcast(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> Optional[Dict[str, Any]]:
        """Process workspace broadcast."""
        return {
            "workspace_contents": message.body.get("workspace_contents", []),
            "slot_count": message.body.get("slot_count", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class PhiUpdateHandler(BaseMessageHandler):
    """Handler for phi (IIT) updates from Form 13."""

    def __init__(self):
        super().__init__(
            handler_id="phi_update_handler",
            handled_types=[MessageType.PHI_UPDATE, MessageType.INTEGRATION_SIGNAL]
        )

    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Process phi update and store in nervous system."""
        phi_value = message.body.get("phi_value", 0.0)
        integration_structure = message.body.get("integration_structure", {})

        # Update nervous system if available
        if context.nervous_system:
            context.nervous_system._phi_value = phi_value

        return HandlerResponse(
            result=HandlerResult.HANDLED,
            response_body={"phi_value": phi_value}
        )


class EmergencyHandler(BroadcastHandler):
    """Handler for emergency messages - highest priority broadcast."""

    def __init__(self):
        super().__init__(
            handler_id="emergency_handler",
            handled_types=[MessageType.EMERGENCY],
            broadcast_channel="emergency"
        )

    async def process_for_broadcast(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> Optional[Dict[str, Any]]:
        """Always broadcast emergency messages."""
        logger.warning(f"EMERGENCY from {message.header.source_form}: {message.body}")
        return {
            "emergency_type": message.body.get("type", "unknown"),
            "description": message.body.get("description", ""),
            "source_form": message.header.source_form,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ----- Sensory Handlers -----

class SensoryInputHandler(BaseMessageHandler):
    """Handler for sensory input messages from Forms 01-06."""

    def __init__(self):
        super().__init__(
            handler_id="sensory_input_handler",
            handled_types=[MessageType.SENSORY_INPUT]
        )

    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Process sensory input and potentially submit to global workspace."""
        modality = message.body.get("modality", "unknown")
        salience = message.body.get("salience", 0.5)
        features = message.body.get("features", {})

        # High salience content should compete for workspace
        if salience > 0.7:
            gw_adapter = context.get_adapter("14-global-workspace")
            if gw_adapter and hasattr(gw_adapter, "submit_content"):
                await gw_adapter.submit_content({
                    "type": "sensory",
                    "modality": modality,
                    "salience": salience,
                    "source_form": message.header.source_form,
                })

        return HandlerResponse(
            result=HandlerResult.HANDLED,
            response_body={"processed_modality": modality, "salience": salience}
        )


class AttentionRequestHandler(BaseMessageHandler):
    """Handler for attention requests."""

    def __init__(self):
        super().__init__(
            handler_id="attention_request_handler",
            handled_types=[MessageType.ATTENTION_REQUEST]
        )

    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Process attention request and route to attention adapter."""
        target = message.body.get("target", {})
        urgency = message.body.get("urgency", 0.5)

        # Route to attention/perceptual adapter (Form 09)
        attention_adapter = context.get_adapter("09-perceptual")
        if attention_adapter:
            # Process attention request through adapter
            pass

        return HandlerResponse(
            result=HandlerResult.HANDLED,
            response_body={"attention_allocated": True}
        )


# ----- Memory Handlers -----

class MemoryQueryHandler(QueryResponseHandler):
    """Handler for memory queries."""

    def __init__(self):
        super().__init__(
            handler_id="memory_query_handler",
            query_type=MessageType.MEMORY_QUERY,
            response_type=MessageType.MEMORY_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        """Process memory query through appropriate memory adapter."""
        query_type = query_body.get("query_type", "recall")
        memory_type = query_body.get("memory_type", "ltm")

        # Route to appropriate memory adapter
        if memory_type == "stm":
            adapter_id = "11-meta-consciousness"  # STM is part of meta
        else:
            adapter_id = "12-narrative-consciousness"  # LTM for narrative

        adapter = context.get_adapter(adapter_id)
        if adapter:
            # Would call adapter's memory retrieval here
            pass

        return {
            "query_type": query_type,
            "memory_type": memory_type,
            "results": [],  # Placeholder
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ----- Philosophical Handlers (Form 28) -----

class PhilosophicalQueryHandler(QueryResponseHandler):
    """Handler for philosophical queries to Form 28."""

    def __init__(self):
        super().__init__(
            handler_id="philosophical_query_handler",
            query_type=MessageType.PHILOSOPHICAL_QUERY,
            response_type=MessageType.PHILOSOPHICAL_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        """Process philosophical query through Form 28 adapter."""
        query = query_body.get("query", "")
        tradition = query_body.get("tradition", None)
        domain = query_body.get("domain", None)

        adapter = context.get_adapter("28-philosophy")
        if adapter:
            result = await adapter.inference({
                "query": query,
                "tradition": tradition,
                "domain": domain,
            })
            return result

        return {
            "query": query,
            "response": "Philosophy adapter not available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class WisdomBroadcastHandler(BroadcastHandler):
    """Handler for wisdom broadcasts from philosophical traditions."""

    def __init__(self):
        super().__init__(
            handler_id="wisdom_broadcast_handler",
            handled_types=[MessageType.WISDOM_BROADCAST],
            broadcast_channel="global_workspace"
        )

    async def process_for_broadcast(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> Optional[Dict[str, Any]]:
        """Broadcast philosophical wisdom to global workspace."""
        return {
            "type": "wisdom",
            "tradition": message.body.get("tradition"),
            "teaching": message.body.get("teaching"),
            "salience": 0.8,  # Wisdom gets high salience
        }


# ----- Extended Form Query Handlers -----

class FolkWisdomQueryHandler(QueryResponseHandler):
    """Handler for folk wisdom queries (Form 29)."""

    def __init__(self):
        super().__init__(
            handler_id="folk_wisdom_query_handler",
            query_type=MessageType.FOLK_WISDOM_QUERY,
            response_type=MessageType.FOLK_WISDOM_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("29-folk-wisdom")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Folk wisdom adapter not available"}


class AnimalCognitionQueryHandler(QueryResponseHandler):
    """Handler for animal cognition queries (Form 30)."""

    def __init__(self):
        super().__init__(
            handler_id="animal_cognition_query_handler",
            query_type=MessageType.ANIMAL_COGNITION_QUERY,
            response_type=MessageType.ANIMAL_COGNITION_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("30-animal-cognition")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Animal cognition adapter not available"}


class PlantIntelligenceQueryHandler(QueryResponseHandler):
    """Handler for plant intelligence queries (Form 31)."""

    def __init__(self):
        super().__init__(
            handler_id="plant_intelligence_query_handler",
            query_type=MessageType.PLANT_INTELLIGENCE_QUERY,
            response_type=MessageType.PLANT_INTELLIGENCE_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("31-plant-intelligence")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Plant intelligence adapter not available"}


class FungalNetworkQueryHandler(QueryResponseHandler):
    """Handler for fungal network queries (Form 32)."""

    def __init__(self):
        super().__init__(
            handler_id="fungal_network_query_handler",
            query_type=MessageType.FUNGAL_NETWORK_QUERY,
            response_type=MessageType.FUNGAL_NETWORK_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("32-fungal-intelligence")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Fungal intelligence adapter not available"}


class SwarmIntelligenceQueryHandler(QueryResponseHandler):
    """Handler for swarm intelligence queries (Form 33)."""

    def __init__(self):
        super().__init__(
            handler_id="swarm_intelligence_query_handler",
            query_type=MessageType.SWARM_INTELLIGENCE_QUERY,
            response_type=MessageType.SWARM_INTELLIGENCE_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("33-swarm-intelligence")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Swarm intelligence adapter not available"}


class GaiaSystemQueryHandler(QueryResponseHandler):
    """Handler for Gaia system queries (Form 34)."""

    def __init__(self):
        super().__init__(
            handler_id="gaia_system_query_handler",
            query_type=MessageType.GAIA_SYSTEM_QUERY,
            response_type=MessageType.GAIA_SYSTEM_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("34-gaia-intelligence")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Gaia intelligence adapter not available"}


class DevelopmentalQueryHandler(QueryResponseHandler):
    """Handler for developmental consciousness queries (Form 35)."""

    def __init__(self):
        super().__init__(
            handler_id="developmental_query_handler",
            query_type=MessageType.DEVELOPMENTAL_QUERY,
            response_type=MessageType.DEVELOPMENTAL_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("35-developmental-consciousness")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Developmental consciousness adapter not available"}


class ContemplativeStateQueryHandler(QueryResponseHandler):
    """Handler for contemplative state queries (Form 36)."""

    def __init__(self):
        super().__init__(
            handler_id="contemplative_state_query_handler",
            query_type=MessageType.CONTEMPLATIVE_STATE_QUERY,
            response_type=MessageType.CONTEMPLATIVE_STATE_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("36-contemplative-states")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Contemplative states adapter not available"}


class PsychedelicQueryHandler(QueryResponseHandler):
    """Handler for psychedelic consciousness queries (Form 37)."""

    def __init__(self):
        super().__init__(
            handler_id="psychedelic_query_handler",
            query_type=MessageType.PSYCHEDELIC_QUERY,
            response_type=MessageType.PSYCHEDELIC_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("37-psychedelic-consciousness")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Psychedelic consciousness adapter not available"}


class NeurodivergentQueryHandler(QueryResponseHandler):
    """Handler for neurodivergent consciousness queries (Form 38)."""

    def __init__(self):
        super().__init__(
            handler_id="neurodivergent_query_handler",
            query_type=MessageType.NEURODIVERGENT_QUERY,
            response_type=MessageType.NEURODIVERGENT_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("38-neurodivergent-consciousness")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Neurodivergent consciousness adapter not available"}


class TraumaQueryHandler(QueryResponseHandler):
    """Handler for trauma consciousness queries (Form 39)."""

    def __init__(self):
        super().__init__(
            handler_id="trauma_query_handler",
            query_type=MessageType.TRAUMA_QUERY,
            response_type=MessageType.TRAUMA_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("39-trauma-consciousness")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Trauma consciousness adapter not available"}


class XenoConsciousnessQueryHandler(QueryResponseHandler):
    """Handler for xenoconsciousness queries (Form 40)."""

    def __init__(self):
        super().__init__(
            handler_id="xeno_consciousness_query_handler",
            query_type=MessageType.XENO_CONSCIOUSNESS_QUERY,
            response_type=MessageType.XENO_CONSCIOUSNESS_RESPONSE
        )

    async def process_query(
        self,
        query_body: Dict[str, Any],
        context: MessageHandlerContext
    ) -> Dict[str, Any]:
        adapter = context.get_adapter("40-xenoconsciousness")
        if adapter:
            return await adapter.inference(query_body)
        return {"error": "Xenoconsciousness adapter not available"}


# ----- Cross-Form Integration Handler -----

class CrossFormSynthesisHandler(BaseMessageHandler):
    """Handler for cross-form synthesis requests."""

    def __init__(self):
        super().__init__(
            handler_id="cross_form_synthesis_handler",
            handled_types=[MessageType.CROSS_FORM_SYNTHESIS, MessageType.INDIGENOUS_KNOWLEDGE_LINK]
        )

    async def handle(
        self,
        message: FormMessage,
        context: MessageHandlerContext
    ) -> HandlerResponse:
        """Synthesize knowledge across multiple forms."""
        forms_to_query = message.body.get("forms", [])
        synthesis_query = message.body.get("query", "")

        results = {}
        for form_id in forms_to_query:
            adapter = context.get_adapter(form_id)
            if adapter:
                try:
                    result = await adapter.inference({"query": synthesis_query})
                    results[form_id] = result
                except Exception as e:
                    results[form_id] = {"error": str(e)}

        return HandlerResponse(
            result=HandlerResult.HANDLED,
            response_body={
                "synthesis_query": synthesis_query,
                "form_results": results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


# =============================================================================
# HANDLER FACTORY
# =============================================================================

class MessageHandlerFactory:
    """Factory for creating and registering standard handlers."""

    # Map of handler classes for easy registration
    STANDARD_HANDLERS: List[Type[BaseMessageHandler]] = [
        # Core system
        ArousalUpdateHandler,
        WorkspaceBroadcastHandler,
        PhiUpdateHandler,
        EmergencyHandler,
        # Sensory/cognitive
        SensoryInputHandler,
        AttentionRequestHandler,
        MemoryQueryHandler,
        # Philosophical (Form 28)
        PhilosophicalQueryHandler,
        WisdomBroadcastHandler,
        # Extended forms (29-40)
        FolkWisdomQueryHandler,
        AnimalCognitionQueryHandler,
        PlantIntelligenceQueryHandler,
        FungalNetworkQueryHandler,
        SwarmIntelligenceQueryHandler,
        GaiaSystemQueryHandler,
        DevelopmentalQueryHandler,
        ContemplativeStateQueryHandler,
        PsychedelicQueryHandler,
        NeurodivergentQueryHandler,
        TraumaQueryHandler,
        XenoConsciousnessQueryHandler,
        # Cross-form
        CrossFormSynthesisHandler,
    ]

    @classmethod
    def create_all_handlers(cls) -> List[BaseMessageHandler]:
        """Create instances of all standard handlers."""
        return [handler_class() for handler_class in cls.STANDARD_HANDLERS]

    @classmethod
    def register_all(cls, coordinator: MessageHandlerCoordinator) -> int:
        """
        Register all standard handlers with a coordinator.

        Args:
            coordinator: The coordinator to register with

        Returns:
            Number of handlers registered
        """
        handlers = cls.create_all_handlers()
        for handler in handlers:
            coordinator.register_handler(handler)
        logger.info(f"Registered {len(handlers)} standard message handlers")
        return len(handlers)

    @classmethod
    def create_handler(cls, handler_id: str) -> Optional[BaseMessageHandler]:
        """Create a specific handler by ID."""
        for handler_class in cls.STANDARD_HANDLERS:
            instance = handler_class()
            if instance.handler_id == handler_id:
                return instance
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_coordinator(
    message_bus: MessageBus,
    nervous_system: Optional["NervousSystem"] = None,
    register_standard_handlers: bool = True
) -> MessageHandlerCoordinator:
    """
    Create a message handler coordinator with optional standard handlers.

    Args:
        message_bus: The message bus to coordinate
        nervous_system: Optional nervous system reference
        register_standard_handlers: Whether to register all standard handlers

    Returns:
        Configured MessageHandlerCoordinator
    """
    coordinator = MessageHandlerCoordinator(message_bus, nervous_system)

    if register_standard_handlers:
        MessageHandlerFactory.register_all(coordinator)

    return coordinator


def get_handled_message_types() -> List[str]:
    """Get list of all message types that have handlers."""
    handled = set()
    for handler_class in MessageHandlerFactory.STANDARD_HANDLERS:
        instance = handler_class()
        for msg_type in instance.handled_types:
            handled.add(msg_type.value)
    return sorted(list(handled))
