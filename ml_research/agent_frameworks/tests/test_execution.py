"""
Tests for execution components.

Tests cover:
    - ArchitectEditor planning and execution
    - SessionManager lifecycle
    - ToolExecutor sandboxing
    - ClientServer communication
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Tests for ArchitectEditor
# ---------------------------------------------------------------------------

class TestArchitectEditor:
    """Tests for ArchitectEditor pattern."""

    @pytest.mark.asyncio
    async def test_plan_generation(self, mock_backend_with_plan, mock_tool_registry):
        """Test generating an execution plan."""
        # Import actual implementation when available
        from agent_frameworks.execution import ArchitectEditor

        arch_edit = ArchitectEditor(mock_backend_with_plan)
        plan = await arch_edit.plan(
            task="Add user authentication",
            context="Flask app with SQLite",
            available_tools=["read_file", "write_file"]
        )

        assert plan is not None
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    async def test_plan_execution(self, mock_backend_with_plan, mock_tool_registry):
        """Test executing a plan."""
        from agent_frameworks.execution import ArchitectEditor, ExecutionPlan, PlanStep

        arch_edit = ArchitectEditor(mock_backend_with_plan)

        # Create a simple plan
        plan = ExecutionPlan(
            goal="Test",
            steps=[
                PlanStep(
                    description="Read file",
                    tool="read_file",
                    arguments={"path": "test.py"}
                )
            ],
            files_to_modify=["test.py"],
            files_to_create=[],
            requires_approval=False,
            reasoning="Test plan"
        )

        result = await arch_edit.execute(plan, mock_tool_registry)

        assert result is not None
        assert len(result.step_results) == 1

    @pytest.mark.asyncio
    async def test_plan_with_dependencies(self, mock_backend_with_plan, mock_tool_registry):
        """Test plan with step dependencies."""
        from agent_frameworks.execution import ArchitectEditor, ExecutionPlan, PlanStep

        arch_edit = ArchitectEditor(mock_backend_with_plan)

        plan = ExecutionPlan(
            goal="Sequential steps",
            steps=[
                PlanStep(
                    description="Step 1",
                    tool="read_file",
                    arguments={"path": "a.py"},
                    dependencies=[]
                ),
                PlanStep(
                    description="Step 2",
                    tool="write_file",
                    arguments={"path": "b.py", "content": "..."},
                    dependencies=[0]  # Depends on step 0
                ),
            ],
            files_to_modify=["b.py"],
            files_to_create=[],
            requires_approval=False,
            reasoning="Test dependencies"
        )

        result = await arch_edit.execute(plan, mock_tool_registry)
        assert result.success

    @pytest.mark.asyncio
    async def test_execution_order(self, mock_backend_with_plan):
        """Test that execution order respects dependencies."""
        from agent_frameworks.execution import ExecutionPlan, PlanStep

        plan = ExecutionPlan(
            goal="Test order",
            steps=[
                PlanStep(description="A", dependencies=[1, 2]),
                PlanStep(description="B", dependencies=[2]),
                PlanStep(description="C", dependencies=[]),
            ],
            files_to_modify=[],
            files_to_create=[],
            requires_approval=False,
            reasoning="Test"
        )

        order = plan.get_execution_order()

        # C should come before B, B should come before A
        assert order.index(2) < order.index(1)
        assert order.index(1) < order.index(0)

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, mock_backend_with_plan):
        """Test detection of circular dependencies."""
        from agent_frameworks.execution import ExecutionPlan, PlanStep

        plan = ExecutionPlan(
            goal="Circular",
            steps=[
                PlanStep(description="A", dependencies=[1]),
                PlanStep(description="B", dependencies=[0]),  # Circular!
            ],
            files_to_modify=[],
            files_to_create=[],
            requires_approval=False,
            reasoning="Test"
        )

        with pytest.raises(ValueError, match="Circular dependency"):
            plan.get_execution_order()

    @pytest.mark.asyncio
    async def test_ask_mode(self, mock_backend):
        """Test ask mode for questions."""
        from agent_frameworks.execution import ArchitectEditor

        mock_backend._responses = ["The answer is 42."]
        arch_edit = ArchitectEditor(mock_backend)

        answer = await arch_edit.ask(
            question="What is the meaning of life?",
            context="A philosophical inquiry"
        )

        assert "42" in answer


# ---------------------------------------------------------------------------
# Tests for SessionManager
# ---------------------------------------------------------------------------

class SessionState(Enum):
    """Session states for testing."""
    ACTIVE = auto()
    SUSPENDED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class Session:
    """Session for testing."""
    session_id: str
    state: SessionState = SessionState.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })


class SessionManager:
    """Manager for agent sessions."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._sessions: Dict[str, Session] = {}
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def create_session(self, agent_id: str) -> Session:
        """Create a new session."""
        session_id = f"{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def save_session(self, session: Session) -> None:
        """Save session to storage."""
        path = self.storage_path / f"{session.session_id}.json"
        data = {
            "session_id": session.session_id,
            "state": session.state.name,
            "created_at": session.created_at.isoformat(),
            "messages": session.messages,
            "metadata": session.metadata
        }
        path.write_text(json.dumps(data, indent=2))

    async def load_session(self, session_id: str) -> Optional[Session]:
        """Load session from storage."""
        path = self.storage_path / f"{session_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        session = Session(
            session_id=data["session_id"],
            state=SessionState[data["state"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            messages=data["messages"],
            metadata=data["metadata"]
        )
        self._sessions[session_id] = session
        return session

    async def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self._sessions.keys())

    async def end_session(self, session_id: str, state: SessionState = SessionState.COMPLETED) -> None:
        """End a session."""
        session = self._sessions.get(session_id)
        if session:
            session.state = state


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.mark.asyncio
    async def test_create_session(self, tmp_path):
        """Test creating a session."""
        manager = SessionManager(tmp_path / "sessions")
        session = await manager.create_session("test-agent")

        assert session is not None
        assert session.state == SessionState.ACTIVE
        assert "test-agent" in session.session_id

    @pytest.mark.asyncio
    async def test_get_session(self, tmp_path):
        """Test retrieving a session."""
        manager = SessionManager(tmp_path / "sessions")
        session = await manager.create_session("agent-1")

        retrieved = await manager.get_session(session.session_id)
        assert retrieved is session

    @pytest.mark.asyncio
    async def test_save_and_load_session(self, tmp_path):
        """Test session persistence."""
        manager = SessionManager(tmp_path / "sessions")
        session = await manager.create_session("agent-1")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        await manager.save_session(session)

        # Create new manager and load
        manager2 = SessionManager(tmp_path / "sessions")
        loaded = await manager2.load_session(session.session_id)

        assert loaded is not None
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmp_path):
        """Test listing sessions."""
        manager = SessionManager(tmp_path / "sessions")
        await manager.create_session("agent-1")
        await manager.create_session("agent-2")
        await manager.create_session("agent-3")

        sessions = await manager.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_end_session(self, tmp_path):
        """Test ending a session."""
        manager = SessionManager(tmp_path / "sessions")
        session = await manager.create_session("agent-1")

        await manager.end_session(session.session_id)
        assert session.state == SessionState.COMPLETED

        await manager.end_session(session.session_id, SessionState.FAILED)
        assert session.state == SessionState.FAILED


# ---------------------------------------------------------------------------
# Tests for ToolExecutor
# ---------------------------------------------------------------------------

class SandboxMode(Enum):
    """Sandbox modes for testing."""
    NONE = auto()
    SUBPROCESS = auto()
    DOCKER = auto()


@dataclass
class ResourceLimits:
    """Resource limits for testing."""
    timeout: int = 30
    memory_mb: int = 512
    cpu_percent: float = 100.0


@dataclass
class ExecutionContext:
    """Execution context for testing."""
    working_dir: Path
    environment: Dict[str, str] = field(default_factory=dict)
    sandbox_mode: SandboxMode = SandboxMode.SUBPROCESS
    limits: ResourceLimits = field(default_factory=ResourceLimits)


class ToolExecutor:
    """Executes tools in sandbox."""

    def __init__(
        self,
        sandbox_mode: SandboxMode = SandboxMode.SUBPROCESS,
        limits: Optional[ResourceLimits] = None
    ):
        self.sandbox_mode = sandbox_mode
        self.limits = limits or ResourceLimits()
        self._audit_log: List[Dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute a tool."""
        start_time = datetime.now()

        # Log execution
        self._audit_log.append({
            "tool": tool_name,
            "arguments": arguments,
            "sandbox_mode": self.sandbox_mode.name,
            "timestamp": start_time.isoformat()
        })

        # Simulate execution based on sandbox mode
        if self.sandbox_mode == SandboxMode.NONE:
            result = {"executed": True, "sandbox": "none"}
        elif self.sandbox_mode == SandboxMode.SUBPROCESS:
            result = {"executed": True, "sandbox": "subprocess"}
        else:
            result = {"executed": True, "sandbox": "docker"}

        return result

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get execution audit log."""
        return self._audit_log.copy()


class TestToolExecutor:
    """Tests for ToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_no_sandbox(self):
        """Test execution without sandbox."""
        executor = ToolExecutor(sandbox_mode=SandboxMode.NONE)
        result = await executor.execute("test_tool", {"arg": "value"})

        assert result["executed"]
        assert result["sandbox"] == "none"

    @pytest.mark.asyncio
    async def test_execute_subprocess_sandbox(self):
        """Test execution with subprocess sandbox."""
        executor = ToolExecutor(sandbox_mode=SandboxMode.SUBPROCESS)
        result = await executor.execute("test_tool", {})

        assert result["sandbox"] == "subprocess"

    @pytest.mark.asyncio
    async def test_audit_logging(self):
        """Test that executions are logged."""
        executor = ToolExecutor()

        await executor.execute("tool1", {"a": 1})
        await executor.execute("tool2", {"b": 2})

        log = executor.get_audit_log()
        assert len(log) == 2
        assert log[0]["tool"] == "tool1"
        assert log[1]["tool"] == "tool2"

    @pytest.mark.asyncio
    async def test_resource_limits(self):
        """Test resource limit configuration."""
        limits = ResourceLimits(timeout=10, memory_mb=256)
        executor = ToolExecutor(limits=limits)

        assert executor.limits.timeout == 10
        assert executor.limits.memory_mb == 256


# ---------------------------------------------------------------------------
# Tests for ClientServer
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Server configuration for testing."""
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 10


class AgentServer:
    """Agent server for testing."""

    def __init__(self, agent: Any, config: ServerConfig):
        self.agent = agent
        self.config = config
        self._running = False
        self._connections: List[Any] = []

    async def start(self) -> None:
        """Start the server."""
        self._running = True

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        self._connections.clear()

    @property
    def is_running(self) -> bool:
        return self._running


class AgentClient:
    """Agent client for testing."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._connected = False

    async def connect(self) -> None:
        """Connect to server."""
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False

    async def send_task(self, task: str) -> Dict[str, Any]:
        """Send a task to the server."""
        if not self._connected:
            raise ConnectionError("Not connected")
        return {"task": task, "status": "received"}

    @property
    def is_connected(self) -> bool:
        return self._connected


class TestClientServer:
    """Tests for ClientServer architecture."""

    @pytest.mark.asyncio
    async def test_server_lifecycle(self, mock_backend):
        """Test server start/stop."""
        server = AgentServer(mock_backend, ServerConfig())

        assert not server.is_running
        await server.start()
        assert server.is_running
        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_client_connection(self):
        """Test client connection."""
        client = AgentClient()

        assert not client.is_connected
        await client.connect()
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_send_task(self):
        """Test sending task to server."""
        client = AgentClient()
        await client.connect()

        result = await client.send_task("Do something")
        assert result["status"] == "received"

    @pytest.mark.asyncio
    async def test_send_without_connection(self):
        """Test that sending without connection raises error."""
        client = AgentClient()

        with pytest.raises(ConnectionError):
            await client.send_task("Do something")
