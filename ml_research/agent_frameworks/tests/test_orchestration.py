"""
Tests for orchestration components.

Tests cover:
    - Gateway dispatch
    - AgentRouter routing
    - Workspace isolation
    - EventBus pub/sub
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Awaitable, Set
from enum import Enum, auto
from datetime import datetime
import uuid


# ---------------------------------------------------------------------------
# Test Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A task for agent processing."""
    task_id: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """Result of task processing."""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    agent_id: Optional[str] = None


class Gateway:
    """Gateway for dispatching tasks to agents."""

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._dispatch_log: List[Dict[str, Any]] = []

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent."""
        self._agents[agent_id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agents.pop(agent_id, None)

    async def dispatch(self, task: Task, agent_id: Optional[str] = None) -> TaskResult:
        """Dispatch a task to an agent."""
        if agent_id:
            if agent_id not in self._agents:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Agent {agent_id} not found"
                )
            target_agent = self._agents[agent_id]
        else:
            # Use first available agent
            if not self._agents:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error="No agents available"
                )
            agent_id = next(iter(self._agents))
            target_agent = self._agents[agent_id]

        self._dispatch_log.append({
            "task_id": task.task_id,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        })

        # Execute on agent (simulated)
        try:
            if hasattr(target_agent, 'process'):
                output = await target_agent.process(task)
            else:
                output = f"Processed: {task.description}"

            return TaskResult(
                task_id=task.task_id,
                success=True,
                output=output,
                agent_id=agent_id
            )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                agent_id=agent_id
            )

    def get_dispatch_log(self) -> List[Dict[str, Any]]:
        """Get dispatch history."""
        return self._dispatch_log.copy()

    def list_agents(self) -> List[str]:
        """List registered agent IDs."""
        return list(self._agents.keys())


class RoutingStrategy(Enum):
    """Strategies for routing tasks."""
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    CAPABILITY_MATCH = auto()
    PRIORITY = auto()


class AgentRouter:
    """Routes tasks to appropriate agents."""

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._round_robin_index = 0
        self._agent_loads: Dict[str, int] = {}

    def register_agent(
        self,
        agent_id: str,
        agent: Any,
        capabilities: Optional[List[str]] = None
    ) -> None:
        """Register an agent with capabilities."""
        self._agents[agent_id] = {
            "agent": agent,
            "capabilities": capabilities or []
        }
        self._agent_loads[agent_id] = 0

    def select_agent(
        self,
        task: Task,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select an agent for a task."""
        if not self._agents:
            return None

        candidates = list(self._agents.keys())

        # Filter by capabilities if required
        if required_capabilities:
            candidates = [
                agent_id for agent_id in candidates
                if all(
                    cap in self._agents[agent_id]["capabilities"]
                    for cap in required_capabilities
                )
            ]

        if not candidates:
            return None

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            agent_id = candidates[self._round_robin_index % len(candidates)]
            self._round_robin_index += 1
            return agent_id

        elif self.strategy == RoutingStrategy.LEAST_LOADED:
            return min(candidates, key=lambda x: self._agent_loads.get(x, 0))

        else:
            return candidates[0]

    def update_load(self, agent_id: str, delta: int) -> None:
        """Update agent load."""
        if agent_id in self._agent_loads:
            self._agent_loads[agent_id] = max(0, self._agent_loads[agent_id] + delta)


@dataclass
class WorkspaceConfig:
    """Configuration for a workspace."""
    workspace_id: str
    root_path: str
    environment: Dict[str, str] = field(default_factory=dict)
    isolated: bool = True


class Workspace:
    """Isolated workspace for agent execution."""

    def __init__(self, config: WorkspaceConfig):
        self.config = config
        self._active = False
        self._artifacts: Dict[str, Any] = {}

    async def activate(self) -> None:
        """Activate the workspace."""
        self._active = True

    async def deactivate(self) -> None:
        """Deactivate the workspace."""
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def store_artifact(self, name: str, data: Any) -> None:
        """Store an artifact in the workspace."""
        self._artifacts[name] = data

    def get_artifact(self, name: str) -> Optional[Any]:
        """Get an artifact from the workspace."""
        return self._artifacts.get(name)

    def list_artifacts(self) -> List[str]:
        """List artifact names."""
        return list(self._artifacts.keys())

    def get_environment(self) -> Dict[str, str]:
        """Get workspace environment."""
        return self.config.environment.copy()


EventHandler = Callable[[str, Any], Awaitable[None]]


class EventBus:
    """Pub/sub event bus for agent communication."""

    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._event_history: List[Dict[str, Any]] = []
        self._filters: Dict[str, Callable[[Any], bool]] = {}

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        filter_func: Optional[Callable[[Any], bool]] = None
    ) -> str:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(handler)

        subscription_id = f"{event_type}_{len(self._subscribers[event_type])}"
        if filter_func:
            self._filters[subscription_id] = filter_func

        return subscription_id

    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    async def publish(self, event_type: str, data: Any) -> int:
        """Publish an event."""
        self._event_history.append({
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

        handlers = self._subscribers.get(event_type, [])
        notified = 0

        for handler in handlers:
            try:
                await handler(event_type, data)
                notified += 1
            except Exception:
                pass

        return notified

    def get_history(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get event history."""
        if event_type:
            return [e for e in self._event_history if e["event_type"] == event_type]
        return self._event_history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


# ---------------------------------------------------------------------------
# Tests for Gateway
# ---------------------------------------------------------------------------

class TestGateway:
    """Tests for Gateway dispatch."""

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test registering an agent."""
        gateway = Gateway()

        class MockAgent:
            async def process(self, task):
                return "done"

        gateway.register_agent("agent-1", MockAgent())

        assert "agent-1" in gateway.list_agents()

    @pytest.mark.asyncio
    async def test_dispatch_to_specific_agent(self):
        """Test dispatching to a specific agent."""
        gateway = Gateway()

        class MockAgent:
            async def process(self, task):
                return f"Processed by agent: {task.description}"

        gateway.register_agent("agent-1", MockAgent())

        task = Task(task_id="t1", description="Test task")
        result = await gateway.dispatch(task, agent_id="agent-1")

        assert result.success
        assert result.agent_id == "agent-1"
        assert "Processed by agent" in result.output

    @pytest.mark.asyncio
    async def test_dispatch_to_unknown_agent(self):
        """Test dispatching to unknown agent fails."""
        gateway = Gateway()

        task = Task(task_id="t1", description="Test")
        result = await gateway.dispatch(task, agent_id="unknown")

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_no_agents(self):
        """Test dispatching when no agents available."""
        gateway = Gateway()

        task = Task(task_id="t1", description="Test")
        result = await gateway.dispatch(task)

        assert not result.success
        assert "No agents" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_log(self):
        """Test dispatch logging."""
        gateway = Gateway()

        class MockAgent:
            async def process(self, task):
                return "done"

        gateway.register_agent("agent-1", MockAgent())

        task = Task(task_id="t1", description="Test")
        await gateway.dispatch(task)

        log = gateway.get_dispatch_log()
        assert len(log) == 1
        assert log[0]["task_id"] == "t1"


# ---------------------------------------------------------------------------
# Tests for AgentRouter
# ---------------------------------------------------------------------------

class TestAgentRouter:
    """Tests for AgentRouter."""

    def test_register_with_capabilities(self):
        """Test registering agent with capabilities."""
        router = AgentRouter()
        router.register_agent("agent-1", object(), capabilities=["code", "search"])

        assert "agent-1" in router._agents
        assert "code" in router._agents["agent-1"]["capabilities"]

    def test_round_robin_selection(self):
        """Test round-robin agent selection."""
        router = AgentRouter(strategy=RoutingStrategy.ROUND_ROBIN)
        router.register_agent("a1", object())
        router.register_agent("a2", object())
        router.register_agent("a3", object())

        task = Task(task_id="t1", description="Test")

        selected = []
        for _ in range(6):
            selected.append(router.select_agent(task))

        # Should cycle through agents
        assert selected[0] == selected[3]
        assert selected[1] == selected[4]

    def test_least_loaded_selection(self):
        """Test least-loaded agent selection."""
        router = AgentRouter(strategy=RoutingStrategy.LEAST_LOADED)
        router.register_agent("a1", object())
        router.register_agent("a2", object())

        router.update_load("a1", 5)
        router.update_load("a2", 2)

        task = Task(task_id="t1", description="Test")
        selected = router.select_agent(task)

        assert selected == "a2"  # Least loaded

    def test_capability_filtering(self):
        """Test filtering by capabilities."""
        router = AgentRouter()
        router.register_agent("a1", object(), capabilities=["code"])
        router.register_agent("a2", object(), capabilities=["search"])
        router.register_agent("a3", object(), capabilities=["code", "search"])

        task = Task(task_id="t1", description="Test")

        # Only a1 and a3 have "code" capability
        selected = router.select_agent(task, required_capabilities=["code"])
        assert selected in ["a1", "a3"]

        # Only a3 has both
        selected = router.select_agent(task, required_capabilities=["code", "search"])
        assert selected == "a3"


# ---------------------------------------------------------------------------
# Tests for Workspace
# ---------------------------------------------------------------------------

class TestWorkspace:
    """Tests for Workspace isolation."""

    @pytest.mark.asyncio
    async def test_workspace_lifecycle(self):
        """Test workspace activation/deactivation."""
        config = WorkspaceConfig(
            workspace_id="ws1",
            root_path="/tmp/workspace"
        )
        workspace = Workspace(config)

        assert not workspace.is_active
        await workspace.activate()
        assert workspace.is_active
        await workspace.deactivate()
        assert not workspace.is_active

    def test_artifact_storage(self):
        """Test storing and retrieving artifacts."""
        config = WorkspaceConfig(workspace_id="ws1", root_path="/tmp")
        workspace = Workspace(config)

        workspace.store_artifact("result", {"data": [1, 2, 3]})
        workspace.store_artifact("log", "Some log content")

        assert workspace.get_artifact("result") == {"data": [1, 2, 3]}
        assert workspace.get_artifact("log") == "Some log content"
        assert workspace.get_artifact("missing") is None

    def test_list_artifacts(self):
        """Test listing artifacts."""
        config = WorkspaceConfig(workspace_id="ws1", root_path="/tmp")
        workspace = Workspace(config)

        workspace.store_artifact("a", 1)
        workspace.store_artifact("b", 2)
        workspace.store_artifact("c", 3)

        artifacts = workspace.list_artifacts()
        assert set(artifacts) == {"a", "b", "c"}

    def test_workspace_environment(self):
        """Test workspace environment variables."""
        config = WorkspaceConfig(
            workspace_id="ws1",
            root_path="/tmp",
            environment={"API_KEY": "secret", "DEBUG": "true"}
        )
        workspace = Workspace(config)

        env = workspace.get_environment()
        assert env["API_KEY"] == "secret"
        assert env["DEBUG"] == "true"


# ---------------------------------------------------------------------------
# Tests for EventBus
# ---------------------------------------------------------------------------

class TestEventBus:
    """Tests for EventBus pub/sub."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test basic pub/sub."""
        bus = EventBus()
        received = []

        async def handler(event_type, data):
            received.append((event_type, data))

        bus.subscribe("test_event", handler)
        await bus.publish("test_event", {"key": "value"})

        assert len(received) == 1
        assert received[0][0] == "test_event"
        assert received[0][1] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers."""
        bus = EventBus()
        count = {"value": 0}

        async def handler1(event_type, data):
            count["value"] += 1

        async def handler2(event_type, data):
            count["value"] += 10

        bus.subscribe("event", handler1)
        bus.subscribe("event", handler2)

        await bus.publish("event", None)

        assert count["value"] == 11

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing."""
        bus = EventBus()
        received = []

        async def handler(event_type, data):
            received.append(data)

        bus.subscribe("event", handler)
        await bus.publish("event", 1)

        bus.unsubscribe("event", handler)
        await bus.publish("event", 2)

        assert received == [1]

    @pytest.mark.asyncio
    async def test_event_history(self):
        """Test event history tracking."""
        bus = EventBus()

        async def noop(e, d):
            pass

        bus.subscribe("a", noop)
        bus.subscribe("b", noop)

        await bus.publish("a", {"x": 1})
        await bus.publish("b", {"y": 2})
        await bus.publish("a", {"z": 3})

        all_history = bus.get_history()
        assert len(all_history) == 3

        a_history = bus.get_history("a")
        assert len(a_history) == 2

    @pytest.mark.asyncio
    async def test_publish_returns_notify_count(self):
        """Test that publish returns number of notified handlers."""
        bus = EventBus()

        async def handler(e, d):
            pass

        bus.subscribe("event", handler)
        bus.subscribe("event", handler)

        count = await bus.publish("event", None)
        assert count == 2

    @pytest.mark.asyncio
    async def test_handler_exception_isolation(self):
        """Test that handler exceptions don't affect other handlers."""
        bus = EventBus()
        results = []

        async def bad_handler(e, d):
            raise ValueError("oops")

        async def good_handler(e, d):
            results.append(d)

        bus.subscribe("event", bad_handler)
        bus.subscribe("event", good_handler)

        await bus.publish("event", "data")

        assert results == ["data"]
