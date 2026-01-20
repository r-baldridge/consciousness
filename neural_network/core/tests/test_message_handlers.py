"""
Tests for the Message Handlers module.
"""

import asyncio
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Add module to path for imports
module_path = Path(__file__).parent.parent.parent.parent
if str(module_path) not in sys.path:
    sys.path.insert(0, str(module_path))


class TestHandlerResult(unittest.TestCase):
    """Tests for HandlerResult enum."""

    def test_handler_results(self):
        """Test handler result values."""
        from neural_network.core.message_handlers import HandlerResult

        self.assertEqual(HandlerResult.HANDLED.value, "handled")
        self.assertEqual(HandlerResult.FORWARDED.value, "forwarded")
        self.assertEqual(HandlerResult.PENDING.value, "pending")
        self.assertEqual(HandlerResult.IGNORED.value, "ignored")
        self.assertEqual(HandlerResult.ERROR.value, "error")


class TestHandlerResponse(unittest.TestCase):
    """Tests for HandlerResponse dataclass."""

    def test_success_response(self):
        """Test successful handler response."""
        from neural_network.core.message_handlers import HandlerResult, HandlerResponse

        response = HandlerResponse(
            result=HandlerResult.HANDLED,
            response_body={"key": "value"}
        )
        self.assertTrue(response.success)
        self.assertEqual(response.response_body["key"], "value")

    def test_error_response(self):
        """Test error handler response."""
        from neural_network.core.message_handlers import HandlerResult, HandlerResponse

        response = HandlerResponse(
            result=HandlerResult.ERROR,
            error_message="Test error"
        )
        self.assertFalse(response.success)
        self.assertEqual(response.error_message, "Test error")

    def test_forwarded_is_success(self):
        """Test that forwarded is considered success."""
        from neural_network.core.message_handlers import HandlerResult, HandlerResponse

        response = HandlerResponse(
            result=HandlerResult.FORWARDED,
            forward_to="other-form"
        )
        self.assertTrue(response.success)


class TestMessageHandlerRegistry(unittest.TestCase):
    """Tests for MessageHandlerRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.message_handlers import MessageHandlerRegistry
        self.registry = MessageHandlerRegistry()

    def test_register_handler(self):
        """Test registering a handler."""
        from neural_network.core.message_handlers import ArousalUpdateHandler

        handler = ArousalUpdateHandler()
        self.registry.register(handler)

        self.assertIn(handler.handler_id, self.registry._handlers_by_id)

    def test_get_handlers_by_type(self):
        """Test getting handlers by message type."""
        from neural_network.core.message_handlers import ArousalUpdateHandler
        from neural_network.core.message_bus import MessageType

        handler = ArousalUpdateHandler()
        self.registry.register(handler)

        handlers = self.registry.get_handlers(MessageType.AROUSAL_UPDATE)
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].handler_id, "arousal_update_handler")

    def test_has_handler(self):
        """Test checking for handler existence."""
        from neural_network.core.message_handlers import ArousalUpdateHandler
        from neural_network.core.message_bus import MessageType

        self.assertFalse(self.registry.has_handler(MessageType.AROUSAL_UPDATE))

        handler = ArousalUpdateHandler()
        self.registry.register(handler)

        self.assertTrue(self.registry.has_handler(MessageType.AROUSAL_UPDATE))

    def test_unregister_handler(self):
        """Test unregistering a handler."""
        from neural_network.core.message_handlers import ArousalUpdateHandler
        from neural_network.core.message_bus import MessageType

        handler = ArousalUpdateHandler()
        self.registry.register(handler)
        self.assertTrue(self.registry.has_handler(MessageType.AROUSAL_UPDATE))

        result = self.registry.unregister(handler.handler_id)
        self.assertTrue(result)
        self.assertFalse(self.registry.has_handler(MessageType.AROUSAL_UPDATE))

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent handler returns False."""
        result = self.registry.unregister("nonexistent")
        self.assertFalse(result)

    def test_register_with_form_id(self):
        """Test registering handler with form ID ownership."""
        from neural_network.core.message_handlers import ArousalUpdateHandler

        handler = ArousalUpdateHandler()
        self.registry.register(handler, form_id="08-arousal")

        self.assertIn("08-arousal", self.registry._form_handlers)
        self.assertIn(handler.handler_id, self.registry._form_handlers["08-arousal"])

    def test_unregister_form(self):
        """Test unregistering all handlers for a form."""
        from neural_network.core.message_handlers import ArousalUpdateHandler, PhiUpdateHandler

        handler1 = ArousalUpdateHandler()
        handler2 = PhiUpdateHandler()

        self.registry.register(handler1, form_id="test-form")
        self.registry.register(handler2, form_id="test-form")

        count = self.registry.unregister_form("test-form")
        self.assertEqual(count, 2)
        self.assertNotIn("test-form", self.registry._form_handlers)

    def test_get_status(self):
        """Test getting registry status."""
        from neural_network.core.message_handlers import ArousalUpdateHandler

        handler = ArousalUpdateHandler()
        self.registry.register(handler)

        status = self.registry.get_status()
        self.assertEqual(status["total_handlers"], 1)
        self.assertIn("handlers_by_type", status)


class TestMessageHandlerCoordinator(unittest.TestCase):
    """Tests for MessageHandlerCoordinator."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import MessageHandlerCoordinator

        self.bus = MessageBus()
        self.coordinator = MessageHandlerCoordinator(self.bus)

    def test_register_handler(self):
        """Test registering handler with coordinator."""
        from neural_network.core.message_handlers import ArousalUpdateHandler

        handler = ArousalUpdateHandler()
        self.coordinator.register_handler(handler)

        self.assertTrue(
            self.coordinator.registry.has_handler(handler.handled_types.pop())
        )

    def test_create_context(self):
        """Test creating handler context."""
        context = self.coordinator.create_context("test-handler")

        self.assertIsNotNone(context)
        self.assertEqual(context.handler_id, "test-handler")
        self.assertEqual(context.message_bus, self.bus)

    def test_process_message_no_handler(self):
        """Test processing message with no handler."""
        from neural_network.core.message_bus import MessageType

        message = self.bus.create_message(
            source_form="test-source",
            target_form="test-target",
            message_type=MessageType.COORDINATION,
            body={"test": "data"}
        )

        responses = asyncio.run(self.coordinator.process_message(message))
        self.assertEqual(len(responses), 0)
        self.assertEqual(self.coordinator._unhandled_count, 1)

    def test_process_message_with_handler(self):
        """Test processing message with registered handler."""
        from neural_network.core.message_bus import MessageType
        from neural_network.core.message_handlers import ArousalUpdateHandler

        handler = ArousalUpdateHandler()
        self.coordinator.register_handler(handler)

        message = self.bus.create_message(
            source_form="08-arousal",
            target_form=None,
            message_type=MessageType.AROUSAL_UPDATE,
            body={"arousal_level": 0.7, "arousal_state": "focused"}
        )

        responses = asyncio.run(self.coordinator.process_message(message))
        self.assertEqual(len(responses), 1)
        self.assertEqual(self.coordinator._handled_count, 1)

    def test_get_status(self):
        """Test getting coordinator status."""
        status = self.coordinator.get_status()

        self.assertIn("running", status)
        self.assertIn("processed_count", status)
        self.assertIn("handled_count", status)
        self.assertIn("registry", status)


class TestConcreteHandlers(unittest.TestCase):
    """Tests for concrete handler implementations."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import MessageHandlerContext

        self.bus = MessageBus()
        self.context = MessageHandlerContext(
            message_bus=self.bus,
            nervous_system=None,
            adapters={},
            handler_id="test"
        )

    def test_arousal_update_handler(self):
        """Test ArousalUpdateHandler."""
        from neural_network.core.message_handlers import ArousalUpdateHandler
        from neural_network.core.message_bus import MessageType

        handler = ArousalUpdateHandler()
        self.assertIn(MessageType.AROUSAL_UPDATE, handler.handled_types)

        message = self.bus.create_message(
            source_form="08-arousal",
            target_form=None,
            message_type=MessageType.AROUSAL_UPDATE,
            body={"arousal_level": 0.8, "arousal_state": "focused"}
        )

        response = asyncio.run(handler.handle(message, self.context))
        self.assertTrue(response.success)
        self.assertIsNotNone(response.response_body)
        self.assertEqual(response.response_body["arousal_level"], 0.8)

    def test_workspace_broadcast_handler(self):
        """Test WorkspaceBroadcastHandler."""
        from neural_network.core.message_handlers import WorkspaceBroadcastHandler
        from neural_network.core.message_bus import MessageType

        handler = WorkspaceBroadcastHandler()
        self.assertIn(MessageType.WORKSPACE_BROADCAST, handler.handled_types)

        message = self.bus.create_message(
            source_form="14-global-workspace",
            target_form=None,
            message_type=MessageType.WORKSPACE_BROADCAST,
            body={"workspace_contents": ["item1", "item2"], "slot_count": 2}
        )

        response = asyncio.run(handler.handle(message, self.context))
        self.assertTrue(response.success)

    def test_phi_update_handler(self):
        """Test PhiUpdateHandler."""
        from neural_network.core.message_handlers import PhiUpdateHandler
        from neural_network.core.message_bus import MessageType

        handler = PhiUpdateHandler()
        self.assertIn(MessageType.PHI_UPDATE, handler.handled_types)
        self.assertIn(MessageType.INTEGRATION_SIGNAL, handler.handled_types)

        message = self.bus.create_message(
            source_form="13-integrated-information",
            target_form=None,
            message_type=MessageType.PHI_UPDATE,
            body={"phi_value": 2.5, "integration_structure": {}}
        )

        response = asyncio.run(handler.handle(message, self.context))
        self.assertTrue(response.success)
        self.assertEqual(response.response_body["phi_value"], 2.5)

    def test_emergency_handler(self):
        """Test EmergencyHandler."""
        from neural_network.core.message_handlers import EmergencyHandler
        from neural_network.core.message_bus import MessageType

        handler = EmergencyHandler()
        self.assertIn(MessageType.EMERGENCY, handler.handled_types)

        message = self.bus.create_message(
            source_form="test-form",
            target_form=None,
            message_type=MessageType.EMERGENCY,
            body={"type": "test_emergency", "description": "Test emergency"}
        )

        response = asyncio.run(handler.handle(message, self.context))
        self.assertTrue(response.success)

    def test_sensory_input_handler(self):
        """Test SensoryInputHandler."""
        from neural_network.core.message_handlers import SensoryInputHandler
        from neural_network.core.message_bus import MessageType

        handler = SensoryInputHandler()
        self.assertIn(MessageType.SENSORY_INPUT, handler.handled_types)

        message = self.bus.create_message(
            source_form="01-visual",
            target_form=None,
            message_type=MessageType.SENSORY_INPUT,
            body={"modality": "visual", "salience": 0.6, "features": {}}
        )

        response = asyncio.run(handler.handle(message, self.context))
        self.assertTrue(response.success)
        self.assertEqual(response.response_body["processed_modality"], "visual")

    def test_memory_query_handler(self):
        """Test MemoryQueryHandler."""
        from neural_network.core.message_handlers import MemoryQueryHandler
        from neural_network.core.message_bus import MessageType

        handler = MemoryQueryHandler()
        self.assertEqual(handler.query_type, MessageType.MEMORY_QUERY)
        self.assertEqual(handler.response_type, MessageType.MEMORY_RESPONSE)


class TestMessageHandlerFactory(unittest.TestCase):
    """Tests for MessageHandlerFactory."""

    def test_create_all_handlers(self):
        """Test creating all standard handlers."""
        from neural_network.core.message_handlers import MessageHandlerFactory

        handlers = MessageHandlerFactory.create_all_handlers()
        self.assertGreater(len(handlers), 10)

        # Check that handlers are unique
        handler_ids = [h.handler_id for h in handlers]
        self.assertEqual(len(handler_ids), len(set(handler_ids)))

    def test_register_all(self):
        """Test registering all handlers with coordinator."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import (
            MessageHandlerCoordinator, MessageHandlerFactory
        )

        bus = MessageBus()
        coordinator = MessageHandlerCoordinator(bus)

        count = MessageHandlerFactory.register_all(coordinator)
        self.assertGreater(count, 10)
        self.assertEqual(
            coordinator.registry.get_status()["total_handlers"],
            count
        )

    def test_create_specific_handler(self):
        """Test creating a specific handler by ID."""
        from neural_network.core.message_handlers import MessageHandlerFactory

        handler = MessageHandlerFactory.create_handler("arousal_update_handler")
        self.assertIsNotNone(handler)
        self.assertEqual(handler.handler_id, "arousal_update_handler")

        handler = MessageHandlerFactory.create_handler("nonexistent")
        self.assertIsNone(handler)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for module-level convenience functions."""

    def test_create_coordinator(self):
        """Test create_coordinator function."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import create_coordinator

        bus = MessageBus()
        coordinator = create_coordinator(bus, register_standard_handlers=True)

        self.assertIsNotNone(coordinator)
        self.assertGreater(
            coordinator.registry.get_status()["total_handlers"], 10
        )

    def test_create_coordinator_without_handlers(self):
        """Test create_coordinator without standard handlers."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import create_coordinator

        bus = MessageBus()
        coordinator = create_coordinator(bus, register_standard_handlers=False)

        self.assertEqual(
            coordinator.registry.get_status()["total_handlers"], 0
        )

    def test_get_handled_message_types(self):
        """Test get_handled_message_types function."""
        from neural_network.core.message_handlers import get_handled_message_types

        types = get_handled_message_types()
        self.assertIsInstance(types, list)
        self.assertIn("arousal_update", types)
        self.assertIn("workspace_broadcast", types)
        self.assertIn("phi_update", types)
        self.assertIn("emergency", types)


class TestQueryResponseHandlers(unittest.TestCase):
    """Tests for query-response pattern handlers."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import MessageHandlerContext

        self.bus = MessageBus()
        self.context = MessageHandlerContext(
            message_bus=self.bus,
            nervous_system=None,
            adapters={},
            handler_id="test"
        )

    def test_philosophical_query_handler(self):
        """Test PhilosophicalQueryHandler without adapter."""
        from neural_network.core.message_handlers import PhilosophicalQueryHandler
        from neural_network.core.message_bus import MessageType

        handler = PhilosophicalQueryHandler()
        self.assertEqual(handler.query_type, MessageType.PHILOSOPHICAL_QUERY)
        self.assertEqual(handler.response_type, MessageType.PHILOSOPHICAL_RESPONSE)

    def test_all_extended_form_handlers_exist(self):
        """Test that handlers exist for all extended forms (28-40)."""
        from neural_network.core.message_handlers import MessageHandlerFactory

        handlers = MessageHandlerFactory.create_all_handlers()
        handler_ids = [h.handler_id for h in handlers]

        expected_handlers = [
            "philosophical_query_handler",
            "folk_wisdom_query_handler",
            "animal_cognition_query_handler",
            "plant_intelligence_query_handler",
            "fungal_network_query_handler",
            "swarm_intelligence_query_handler",
            "gaia_system_query_handler",
            "developmental_query_handler",
            "contemplative_state_query_handler",
            "psychedelic_query_handler",
            "neurodivergent_query_handler",
            "trauma_query_handler",
            "xeno_consciousness_query_handler",
        ]

        for expected in expected_handlers:
            self.assertIn(expected, handler_ids, f"Missing handler: {expected}")


class TestCrossFormSynthesis(unittest.TestCase):
    """Tests for cross-form synthesis handler."""

    def setUp(self):
        """Set up test fixtures."""
        from neural_network.core.message_bus import MessageBus
        from neural_network.core.message_handlers import MessageHandlerContext

        self.bus = MessageBus()
        self.context = MessageHandlerContext(
            message_bus=self.bus,
            nervous_system=None,
            adapters={},
            handler_id="test"
        )

    def test_cross_form_synthesis_handler(self):
        """Test CrossFormSynthesisHandler."""
        from neural_network.core.message_handlers import CrossFormSynthesisHandler
        from neural_network.core.message_bus import MessageType

        handler = CrossFormSynthesisHandler()
        self.assertIn(MessageType.CROSS_FORM_SYNTHESIS, handler.handled_types)
        self.assertIn(MessageType.INDIGENOUS_KNOWLEDGE_LINK, handler.handled_types)

        message = self.bus.create_message(
            source_form="test-form",
            target_form=None,
            message_type=MessageType.CROSS_FORM_SYNTHESIS,
            body={
                "query": "Test synthesis",
                "forms": ["28-philosophy", "29-folk-wisdom"]
            }
        )

        response = asyncio.run(handler.handle(message, self.context))
        self.assertTrue(response.success)
        self.assertIn("form_results", response.response_body)


class TestHandlerStats(unittest.TestCase):
    """Tests for handler statistics tracking."""

    def test_handler_stats(self):
        """Test that handlers track statistics."""
        from neural_network.core.message_handlers import ArousalUpdateHandler
        from neural_network.core.message_bus import MessageBus, MessageType
        from neural_network.core.message_handlers import MessageHandlerContext

        handler = ArousalUpdateHandler()
        bus = MessageBus()
        context = MessageHandlerContext(
            message_bus=bus,
            nervous_system=None,
            adapters={},
            handler_id="test"
        )

        self.assertEqual(handler._message_count, 0)

        message = bus.create_message(
            source_form="08-arousal",
            target_form=None,
            message_type=MessageType.AROUSAL_UPDATE,
            body={"arousal_level": 0.5}
        )

        # Call handler
        asyncio.run(handler(message, context))

        self.assertEqual(handler._message_count, 1)
        self.assertIsNotNone(handler._last_handled)

        # Check stats output
        stats = handler.get_stats()
        self.assertEqual(stats["message_count"], 1)
        self.assertIsNotNone(stats["last_handled"])


if __name__ == "__main__":
    unittest.main()
