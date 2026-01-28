"""
Tests for the orchestration module.

These tests demonstrate the functionality of the orchestration components.
"""

import asyncio
import pytest
from pathlib import Path
import tempfile
import shutil

from ..base import (
    Task, TaskResult, TaskStatus, TaskPriority,
    AgentBase, AgentCapability, Session
)
from ..gateway import Gateway, GatewayConfig, AgentStatus
from ..agent_router import AgentRouter, RoutingRule
from ..event_bus import EventBus, Event, EventPriority
from ..workspace import Workspace, WorkspaceManager, WorkspaceConfig
from ..channel_adapter import Message, MessageType


# --- Test Agent Implementation ---

class TestAgent(AgentBase):
    """A simple test agent for testing orchestration."""

    def __init__(self, agent_id: str = None, capabilities: list = None):
        super().__init__(agent_id)
        self._initialized = False
        self._shutdown = False
        self.executed_tasks = []

        for cap in (capabilities or [AgentCapability.CODE_GENERATION]):
            self.add_capability(cap)

    async def execute(self, task: Task) -> TaskResult:
        self.executed_tasks.append(task)
        return TaskResult.success_result(
            task_id=task.id,
            output=f"Completed: {task.description}",
            agent_id=self.id,
        )

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._shutdown = True


# --- Gateway Tests ---

class TestGateway:
    """Tests for the Gateway class."""

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test agent registration."""
        gateway = Gateway(GatewayConfig())
        agent = TestAgent("test-agent-1")

        agent_id = await gateway.register_agent(agent)
        assert agent_id == "test-agent-1"
        assert agent._initialized

        await gateway.shutdown()
        assert agent._shutdown

    @pytest.mark.asyncio
    async def test_dispatch_task(self):
        """Test task dispatching."""
        gateway = Gateway(GatewayConfig())
        agent = TestAgent("test-agent-1", [AgentCapability.CODE_GENERATION])
        await gateway.register_agent(agent)

        task = Task(
            type="code_generation",
            description="Generate test code"
        )
        result = await gateway.dispatch(task)

        assert result.success
        assert result.agent_id == "test-agent-1"
        assert len(agent.executed_tasks) == 1

        await gateway.shutdown()

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test status retrieval."""
        gateway = Gateway(GatewayConfig())
        agent = TestAgent("test-agent-1")
        await gateway.register_agent(agent)

        status = await gateway.get_status()
        assert "test-agent-1" in status
        assert status["test-agent-1"].status == "idle"

        await gateway.shutdown()


# --- Router Tests ---

class TestAgentRouter:
    """Tests for the AgentRouter class."""

    @pytest.mark.asyncio
    async def test_register_and_route(self):
        """Test agent registration and routing."""
        router = AgentRouter()
        agent = TestAgent("code-agent", [AgentCapability.CODE_GENERATION])
        router.register(agent, tasks=["code", "generate"])

        task = Task(type="code_review", description="Review code")
        routed_agent = await router.route(task)

        assert routed_agent.id == "code-agent"

    @pytest.mark.asyncio
    async def test_routing_rules(self):
        """Test rule-based routing."""
        router = AgentRouter()
        agent1 = TestAgent("normal-agent")
        agent2 = TestAgent("urgent-agent")

        router.register(agent1)
        router.register(agent2)

        # Add rule for urgent tasks
        router.add_rule(RoutingRule(
            pattern=".*urgent.*",
            agent_id="urgent-agent",
            priority=100,
            match_description=True
        ))

        task = Task(type="process", description="URGENT: Fix bug")
        routed = await router.route(task)
        assert routed.id == "urgent-agent"


# --- Event Bus Tests ---

class TestEventBus:
    """Tests for the EventBus class."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic pub/sub."""
        bus = EventBus()
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        bus.subscribe("test.event", handler)

        event = Event(type="test.event", source="test", data={"key": "value"})
        count = await bus.publish(event)

        assert count == 1
        assert len(received_events) == 1
        assert received_events[0].data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        """Test wildcard event subscriptions."""
        bus = EventBus()
        received = []

        bus.subscribe("task.*", lambda e: received.append(e.type))

        await bus.publish(Event(type="task.started", source="test"))
        await bus.publish(Event(type="task.completed", source="test"))
        await bus.publish(Event(type="other.event", source="test"))

        assert len(received) == 2
        assert "task.started" in received
        assert "task.completed" in received

    @pytest.mark.asyncio
    async def test_publish_and_wait(self):
        """Test request-reply pattern."""
        bus = EventBus()

        async def responder(event: Event):
            if event.type == "query":
                reply = event.create_reply("responder", data="response")
                await bus.publish(reply)

        bus.subscribe("query", responder)

        replies = await bus.publish_and_wait(
            Event(type="query", source="requester"),
            timeout=5,
            expected_replies=1
        )

        assert len(replies) == 1
        assert replies[0].data == "response"


# --- Workspace Tests ---

class TestWorkspace:
    """Tests for the Workspace class."""

    @pytest.mark.asyncio
    async def test_workspace_lifecycle(self):
        """Test workspace creation and cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorkspaceConfig(
                base_path=Path(tmpdir),
                cleanup_on_exit=True
            )
            workspace = Workspace("test-agent", config)
            await workspace.initialize()

            assert workspace.path.exists()
            assert workspace.is_initialized

            await workspace.cleanup()
            assert not workspace.path.exists()

    @pytest.mark.asyncio
    async def test_file_operations(self):
        """Test file read/write operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorkspaceConfig(base_path=Path(tmpdir))
            workspace = Workspace("test-agent", config)
            await workspace.initialize()

            # Write file
            await workspace.write_file("src/main.py", "print('hello')")

            # Read file
            content = await workspace.read_file("src/main.py")
            assert content == "print('hello')"

            # List files
            files = await workspace.list_files_relative()
            assert "src/main.py" in files

            await workspace.cleanup()

    @pytest.mark.asyncio
    async def test_snapshots(self):
        """Test workspace snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorkspaceConfig(base_path=Path(tmpdir))
            workspace = Workspace("test-agent", config)
            await workspace.initialize()

            # Create some files
            await workspace.write_file("file1.txt", "original content")

            # Create snapshot
            snapshot_id = await workspace.snapshot("Initial state")

            # Modify file
            await workspace.write_file("file1.txt", "modified content")

            # Restore snapshot
            await workspace.restore(snapshot_id)

            # Verify original content
            content = await workspace.read_file("file1.txt")
            assert content == "original content"

            await workspace.cleanup()


# --- Task and Message Tests ---

class TestDataClasses:
    """Tests for data classes."""

    def test_task_creation(self):
        """Test Task creation and conversion."""
        task = Task(
            type="code_review",
            description="Review PR #123",
            priority=TaskPriority.HIGH,
            tags={"urgent", "security"}
        )

        assert task.type == "code_review"
        assert task.priority == TaskPriority.HIGH
        assert "urgent" in task.tags

        # Test serialization
        data = task.to_dict()
        restored = Task.from_dict(data)
        assert restored.type == task.type
        assert restored.description == task.description

    def test_message_creation(self):
        """Test Message creation."""
        msg = Message(
            content="Hello, world!",
            type=MessageType.TEXT,
            sender="user-1",
            channel="general"
        )

        assert msg.content == "Hello, world!"
        assert msg.type == MessageType.TEXT

        data = msg.to_dict()
        restored = Message.from_dict(data)
        assert restored.content == msg.content


# --- Integration Test ---

class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow with all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up components
            gateway = Gateway(GatewayConfig())
            router = AgentRouter()
            bus = EventBus()
            ws_manager = WorkspaceManager(WorkspaceConfig(base_path=Path(tmpdir)))

            # Create and register agent
            agent = TestAgent("worker-1", [AgentCapability.CODE_GENERATION])
            await gateway.register_agent(agent)
            router.register(agent, tasks=["code"])

            # Create workspace
            workspace = await ws_manager.create("worker-1")

            # Set up event tracking
            events_received = []
            bus.subscribe("task.*", lambda e: events_received.append(e))

            # Route and execute task
            task = Task(type="code_review", description="Review code")
            routed_agent = await router.route(task)
            result = await gateway.dispatch(task, agent_id=routed_agent.id)

            # Publish completion event
            await bus.publish(Event(
                type="task.completed",
                source=routed_agent.id,
                data={"task_id": task.id, "result": result.output}
            ))

            # Verify
            assert result.success
            assert len(events_received) == 1
            assert events_received[0].type == "task.completed"

            # Cleanup
            await gateway.shutdown()
            await ws_manager.destroy("worker-1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
