"""
Tests for core agent abstractions.

Tests cover:
    - AgentBase subclassing and lifecycle
    - MessageTypes serialization
    - StateMachine transitions
    - Agent composition operators (>>, |)
    - AgentRegistry operations
"""

import pytest
import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Test Data Structures (will be replaced by actual imports)
# ---------------------------------------------------------------------------

class MessageRole(Enum):
    """Role of a message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class AgentMessage:
    """A message in agent communication."""
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata"),
        )


class AgentState(Enum):
    """States in the agent state machine."""
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    WAITING_APPROVAL = auto()
    ERROR = auto()
    COMPLETE = auto()


class StateMachine:
    """Simple state machine for agent lifecycle."""

    VALID_TRANSITIONS = {
        AgentState.IDLE: [AgentState.PLANNING, AgentState.EXECUTING],
        AgentState.PLANNING: [AgentState.WAITING_APPROVAL, AgentState.EXECUTING, AgentState.ERROR],
        AgentState.EXECUTING: [AgentState.COMPLETE, AgentState.ERROR, AgentState.WAITING_APPROVAL],
        AgentState.WAITING_APPROVAL: [AgentState.EXECUTING, AgentState.IDLE, AgentState.ERROR],
        AgentState.ERROR: [AgentState.IDLE],
        AgentState.COMPLETE: [AgentState.IDLE],
    }

    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self._state = initial_state
        self._history: List[AgentState] = [initial_state]

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def history(self) -> List[AgentState]:
        return self._history.copy()

    def can_transition(self, to_state: AgentState) -> bool:
        """Check if transition is valid."""
        return to_state in self.VALID_TRANSITIONS.get(self._state, [])

    def transition(self, to_state: AgentState) -> bool:
        """Attempt to transition to new state."""
        if self.can_transition(to_state):
            self._state = to_state
            self._history.append(to_state)
            return True
        return False

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = AgentState.IDLE
        self._history = [AgentState.IDLE]


# ---------------------------------------------------------------------------
# Tests for MessageTypes
# ---------------------------------------------------------------------------

class TestAgentMessage:
    """Tests for AgentMessage serialization."""

    def test_to_dict_basic(self):
        """Test basic message serialization."""
        msg = AgentMessage(
            role=MessageRole.USER,
            content="Hello, world!"
        )
        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == "Hello, world!"
        assert "tool_calls" not in result
        assert "tool_call_id" not in result

    def test_to_dict_with_tool_calls(self):
        """Test message with tool calls serialization."""
        msg = AgentMessage(
            role=MessageRole.ASSISTANT,
            content="Let me search for that.",
            tool_calls=[
                {"id": "call_1", "name": "search", "arguments": {"query": "test"}}
            ]
        )
        result = msg.to_dict()

        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = AgentMessage(
            role=MessageRole.TOOL,
            content="Search results...",
            tool_call_id="call_1",
            metadata={"source": "web"}
        )
        data = original.to_dict()
        restored = AgentMessage.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_call_id == original.tool_call_id

    def test_all_roles(self):
        """Test all message roles."""
        for role in MessageRole:
            msg = AgentMessage(role=role, content="test")
            assert msg.to_dict()["role"] == role.value


# ---------------------------------------------------------------------------
# Tests for StateMachine
# ---------------------------------------------------------------------------

class TestStateMachine:
    """Tests for StateMachine transitions."""

    def test_initial_state(self):
        """Test initial state is IDLE."""
        sm = StateMachine()
        assert sm.state == AgentState.IDLE

    def test_valid_transition(self):
        """Test valid state transition."""
        sm = StateMachine()
        assert sm.can_transition(AgentState.PLANNING)
        assert sm.transition(AgentState.PLANNING)
        assert sm.state == AgentState.PLANNING

    def test_invalid_transition(self):
        """Test invalid state transition is rejected."""
        sm = StateMachine()
        assert not sm.can_transition(AgentState.COMPLETE)
        assert not sm.transition(AgentState.COMPLETE)
        assert sm.state == AgentState.IDLE

    def test_transition_history(self):
        """Test transition history is recorded."""
        sm = StateMachine()
        sm.transition(AgentState.PLANNING)
        sm.transition(AgentState.EXECUTING)
        sm.transition(AgentState.COMPLETE)

        history = sm.history
        assert len(history) == 4
        assert history[0] == AgentState.IDLE
        assert history[-1] == AgentState.COMPLETE

    def test_reset(self):
        """Test state machine reset."""
        sm = StateMachine()
        sm.transition(AgentState.PLANNING)
        sm.transition(AgentState.ERROR)
        sm.reset()

        assert sm.state == AgentState.IDLE
        assert len(sm.history) == 1

    def test_error_recovery(self):
        """Test recovery from error state."""
        sm = StateMachine()
        sm.transition(AgentState.PLANNING)
        sm.transition(AgentState.ERROR)

        # Can only go to IDLE from ERROR
        assert not sm.can_transition(AgentState.EXECUTING)
        assert sm.can_transition(AgentState.IDLE)

    def test_approval_workflow(self):
        """Test approval workflow transitions."""
        sm = StateMachine()
        sm.transition(AgentState.PLANNING)
        sm.transition(AgentState.WAITING_APPROVAL)

        # From approval, can execute, cancel to idle, or error
        assert sm.can_transition(AgentState.EXECUTING)
        assert sm.can_transition(AgentState.IDLE)
        assert sm.can_transition(AgentState.ERROR)


# ---------------------------------------------------------------------------
# Tests for Agent Composition
# ---------------------------------------------------------------------------

class BaseAgent:
    """Minimal agent base for testing composition."""

    def __init__(self, name: str):
        self.name = name

    async def process(self, input_data: Any) -> Any:
        return f"{self.name}: {input_data}"

    def __rshift__(self, other: "BaseAgent") -> "SequentialPipeline":
        """Create sequential pipeline with >> operator."""
        return SequentialPipeline([self, other])

    def __or__(self, other: "BaseAgent") -> "ParallelPipeline":
        """Create parallel pipeline with | operator."""
        return ParallelPipeline([self, other])


class SequentialPipeline:
    """Sequential agent pipeline."""

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    async def process(self, input_data: Any) -> Any:
        result = input_data
        for agent in self.agents:
            result = await agent.process(result)
        return result

    def __rshift__(self, other: BaseAgent) -> "SequentialPipeline":
        return SequentialPipeline(self.agents + [other])


class ParallelPipeline:
    """Parallel agent pipeline."""

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    async def process(self, input_data: Any) -> List[Any]:
        tasks = [agent.process(input_data) for agent in self.agents]
        return await asyncio.gather(*tasks)

    def __or__(self, other: BaseAgent) -> "ParallelPipeline":
        return ParallelPipeline(self.agents + [other])


class TestAgentComposition:
    """Tests for agent composition operators."""

    @pytest.mark.asyncio
    async def test_sequential_pipeline(self):
        """Test >> operator creates sequential pipeline."""
        agent1 = BaseAgent("A")
        agent2 = BaseAgent("B")

        pipeline = agent1 >> agent2
        result = await pipeline.process("input")

        assert "A:" in result
        assert "B:" in result

    @pytest.mark.asyncio
    async def test_parallel_pipeline(self):
        """Test | operator creates parallel pipeline."""
        agent1 = BaseAgent("A")
        agent2 = BaseAgent("B")

        pipeline = agent1 | agent2
        results = await pipeline.process("input")

        assert len(results) == 2
        assert any("A:" in r for r in results)
        assert any("B:" in r for r in results)

    @pytest.mark.asyncio
    async def test_chained_sequential(self):
        """Test chaining multiple sequential operators."""
        a = BaseAgent("A")
        b = BaseAgent("B")
        c = BaseAgent("C")

        pipeline = a >> b >> c
        result = await pipeline.process("start")

        # Result should contain all agent names in order
        assert "A:" in result
        assert "B:" in result
        assert "C:" in result

    @pytest.mark.asyncio
    async def test_chained_parallel(self):
        """Test chaining multiple parallel operators."""
        a = BaseAgent("A")
        b = BaseAgent("B")
        c = BaseAgent("C")

        pipeline = a | b | c
        results = await pipeline.process("input")

        assert len(results) == 3


# ---------------------------------------------------------------------------
# Tests for AgentRegistry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Registry for agent types."""

    _instance = None
    _agents: Dict[str, type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._agents = {}
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the registry."""
        cls._instance = None
        cls._agents = {}

    def register(self, name: str, agent_class: type) -> None:
        """Register an agent class."""
        self._agents[name] = agent_class

    def get(self, name: str) -> Optional[type]:
        """Get an agent class by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._agents


def register(name: str):
    """Decorator to register an agent class."""
    def decorator(cls):
        AgentRegistry().register(name, cls)
        return cls
    return decorator


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        AgentRegistry.reset()

    def test_singleton(self):
        """Test registry is singleton."""
        reg1 = AgentRegistry()
        reg2 = AgentRegistry()
        assert reg1 is reg2

    def test_register_and_get(self):
        """Test registering and retrieving agents."""
        registry = AgentRegistry()

        class TestAgent:
            pass

        registry.register("test", TestAgent)
        assert registry.get("test") is TestAgent

    def test_register_decorator(self):
        """Test @register decorator."""
        @register("decorated")
        class DecoratedAgent:
            pass

        registry = AgentRegistry()
        assert registry.get("decorated") is DecoratedAgent

    def test_list_agents(self):
        """Test listing registered agents."""
        registry = AgentRegistry()

        class Agent1:
            pass

        class Agent2:
            pass

        registry.register("agent1", Agent1)
        registry.register("agent2", Agent2)

        agents = registry.list_agents()
        assert "agent1" in agents
        assert "agent2" in agents

    def test_contains(self):
        """Test 'in' operator."""
        registry = AgentRegistry()

        class TestAgent:
            pass

        registry.register("exists", TestAgent)

        assert "exists" in registry
        assert "missing" not in registry

    def test_get_missing(self):
        """Test getting non-existent agent returns None."""
        registry = AgentRegistry()
        assert registry.get("missing") is None
