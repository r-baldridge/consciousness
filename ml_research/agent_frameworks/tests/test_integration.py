"""
End-to-end integration tests.

Tests cover:
    - Complete workflows
    - Framework adapters
    - Multi-component interactions
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Integration Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def complete_setup(tmp_path, mock_backend, mock_tool_registry):
    """Set up a complete test environment."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create sample project structure
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()

    (workspace / "src" / "main.py").write_text('''
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

    (workspace / "src" / "utils.py").write_text('''
def helper(x):
    return x * 2
''')

    return {
        "workspace": workspace,
        "backend": mock_backend,
        "tools": mock_tool_registry
    }


# ---------------------------------------------------------------------------
# End-to-End Workflow Tests
# ---------------------------------------------------------------------------

class TestEndToEndWorkflows:
    """Tests for complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_plan_review_execute_workflow(
        self,
        complete_setup,
        mock_backend_with_plan
    ):
        """Test complete plan -> review -> execute workflow."""
        from agent_frameworks.execution import ArchitectEditor

        arch_edit = ArchitectEditor(mock_backend_with_plan)

        # Step 1: Generate plan
        plan = await arch_edit.plan(
            task="Add logging to main.py",
            context="Python project with main.py entry point"
        )

        assert plan is not None
        assert len(plan.steps) > 0

        # Step 2: Review plan (simulated approval)
        assert plan.goal is not None
        assert plan.reasoning is not None

        # Step 3: Execute plan
        result = await arch_edit.execute(
            plan,
            complete_setup["tools"]
        )

        assert result is not None
        assert result.plan_id == plan.plan_id

    @pytest.mark.asyncio
    async def test_context_aware_task_execution(self, complete_setup):
        """Test task execution with context awareness."""
        from tests.test_context import RepositoryMap, FileSelector

        workspace = complete_setup["workspace"]

        # Build repository map
        repo_map = RepositoryMap(workspace)
        await repo_map.build()

        # Select relevant files
        selector = FileSelector(repo_map)
        relevant = await selector.select_relevant(
            "add logging",
            max_files=5
        )

        # Verify context was built
        assert len(repo_map.list_modules()) > 0

        # Context should include main.py
        context = repo_map.get_context()
        assert "main" in context.lower() or len(relevant) > 0

    @pytest.mark.asyncio
    async def test_multi_step_task_with_dependencies(
        self,
        complete_setup,
        mock_backend_with_plan
    ):
        """Test multi-step task with dependencies between steps."""
        from agent_frameworks.execution import (
            ArchitectEditor, ExecutionPlan, PlanStep
        )

        arch_edit = ArchitectEditor(mock_backend_with_plan)

        # Create plan with dependencies
        plan = ExecutionPlan(
            goal="Refactor and test",
            steps=[
                PlanStep(
                    description="Read existing code",
                    tool="read_file",
                    arguments={"path": "src/main.py"},
                    dependencies=[]
                ),
                PlanStep(
                    description="Analyze structure",
                    tool=None,
                    dependencies=[0]
                ),
                PlanStep(
                    description="Write refactored code",
                    tool="write_file",
                    arguments={"path": "src/main.py", "content": "# Refactored"},
                    dependencies=[1]
                ),
                PlanStep(
                    description="Run tests",
                    tool="bash",
                    arguments={"command": "pytest"},
                    dependencies=[2]
                )
            ],
            files_to_modify=["src/main.py"],
            files_to_create=[],
            requires_approval=False,
            reasoning="Systematic refactoring approach"
        )

        # Verify dependency order
        order = plan.get_execution_order()
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)
        assert order.index(2) < order.index(3)

        # Execute
        result = await arch_edit.execute(plan, complete_setup["tools"])
        assert len(result.step_results) == 4


# ---------------------------------------------------------------------------
# Framework Adapter Tests
# ---------------------------------------------------------------------------

class MockFrameworkAdapter:
    """Mock adapter for testing framework integration."""

    def __init__(self, name: str):
        self.name = name
        self._connected = False

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def execute(self, task: str) -> Dict[str, Any]:
        if not self._connected:
            raise ConnectionError("Not connected")
        return {"adapter": self.name, "task": task, "status": "completed"}


class TestFrameworkAdapters:
    """Tests for framework adapters."""

    @pytest.mark.asyncio
    async def test_adapter_lifecycle(self):
        """Test adapter connection lifecycle."""
        adapter = MockFrameworkAdapter("test")

        assert not adapter._connected
        await adapter.connect()
        assert adapter._connected
        await adapter.disconnect()
        assert not adapter._connected

    @pytest.mark.asyncio
    async def test_adapter_execution(self):
        """Test executing through adapter."""
        adapter = MockFrameworkAdapter("aider")
        await adapter.connect()

        result = await adapter.execute("fix bug in main.py")

        assert result["adapter"] == "aider"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_multiple_adapters(self):
        """Test using multiple adapters."""
        adapters = [
            MockFrameworkAdapter("aider"),
            MockFrameworkAdapter("opencode"),
            MockFrameworkAdapter("humanlayer")
        ]

        for adapter in adapters:
            await adapter.connect()

        results = await asyncio.gather(*[
            adapter.execute("task") for adapter in adapters
        ])

        assert len(results) == 3
        assert all(r["status"] == "completed" for r in results)


# ---------------------------------------------------------------------------
# Component Integration Tests
# ---------------------------------------------------------------------------

class TestComponentIntegration:
    """Tests for integration between components."""

    @pytest.mark.asyncio
    async def test_context_to_execution(self, complete_setup):
        """Test context management feeding into execution."""
        from tests.test_context import RepositoryMap
        from tests.test_execution import SessionManager, Session

        workspace = complete_setup["workspace"]

        # Build context
        repo_map = RepositoryMap(workspace)
        await repo_map.build()
        context = repo_map.get_context()

        # Create session with context
        session_mgr = SessionManager(workspace / "sessions")
        session = await session_mgr.create_session("agent-1")
        session.metadata["context"] = context

        await session_mgr.save_session(session)

        # Verify session has context
        loaded = await session_mgr.load_session(session.session_id)
        assert "context" in loaded.metadata

    @pytest.mark.asyncio
    async def test_memory_with_checkpoints(self, tmp_path):
        """Test memory management with checkpointing."""
        from tests.test_memory import (
            ContextWindow, Message,
            CheckpointManager
        )

        # Create context window with messages
        window = ContextWindow(max_tokens=1000)
        window.add(Message(role="user", content="Hello", tokens=10))
        window.add(Message(role="assistant", content="Hi there!", tokens=15))

        # Create checkpoint
        checkpoint_mgr = CheckpointManager(tmp_path / "checkpoints")
        checkpoint = checkpoint_mgr.save(
            context=[m.to_dict() for m in window.get_messages()],
            state={"tokens": window.get_tokens()}
        )

        # Simulate crash - create new window and restore
        new_window = ContextWindow(max_tokens=1000)
        restored = checkpoint_mgr.restore(checkpoint.checkpoint_id)

        assert restored is not None
        assert len(restored.context) == 2
        assert restored.state["tokens"] == 25

    @pytest.mark.asyncio
    async def test_approval_with_channels(self):
        """Test approval workflow with channel routing."""
        from tests.test_human_loop import (
            ApprovalWorkflow,
            ChannelRouter,
            Channel,
            ChannelType
        )

        # Set up channels
        router = ChannelRouter()
        router.add_channel(Channel("console", ChannelType.CONSOLE))
        router.add_channel(Channel("slack", ChannelType.SLACK))
        router.add_route("git", ["console", "slack"])

        # Create approval request
        workflow = ApprovalWorkflow()
        request = await workflow.request_approval(
            action="git.push",
            description="Push to main branch"
        )

        # Route to channels
        channels = router.get_channels_for_action(request.action)
        results = await router.send_notification(request, [c.name for c in channels])

        assert len(results) == 2
        assert all(v is True for v in results.values())

    @pytest.mark.asyncio
    async def test_orchestration_with_tools(self, mock_tool_registry):
        """Test orchestration dispatching to tools."""
        from tests.test_orchestration import Gateway, Task

        gateway = Gateway()

        class ToolAgent:
            def __init__(self, registry):
                self.registry = registry

            async def process(self, task):
                # Use a tool from registry
                result = await self.registry.execute_tool(
                    "read_file",
                    {"path": task.description}
                )
                return result

        gateway.register_agent("tool_agent", ToolAgent(mock_tool_registry))

        task = Task(task_id="t1", description="test.py")
        result = await gateway.dispatch(task, agent_id="tool_agent")

        assert result.success


# ---------------------------------------------------------------------------
# Error Handling Integration Tests
# ---------------------------------------------------------------------------

class TestErrorHandlingIntegration:
    """Tests for error handling across components."""

    @pytest.mark.asyncio
    async def test_execution_error_recovery(
        self,
        complete_setup,
        mock_backend_with_plan
    ):
        """Test recovering from execution errors."""
        from agent_frameworks.execution import (
            ArchitectEditor, ExecutionPlan, PlanStep
        )
        from tests.conftest import MockTool

        # Set up tool that fails
        complete_setup["tools"]._tools["failing_tool"] = MockTool(
            "failing_tool",
            raises=ValueError("Simulated failure")
        )

        arch_edit = ArchitectEditor(mock_backend_with_plan)

        plan = ExecutionPlan(
            goal="Test error handling",
            steps=[
                PlanStep(
                    description="Will fail",
                    tool="failing_tool",
                    arguments={}
                ),
                PlanStep(
                    description="Should not run",
                    tool="read_file",
                    arguments={"path": "test.py"},
                    dependencies=[0]
                )
            ],
            files_to_modify=[],
            files_to_create=[],
            requires_approval=False,
            reasoning="Test"
        )

        result = await arch_edit.execute(plan, complete_setup["tools"])

        # First step should fail, second shouldn't run
        assert not result.success
        assert not result.step_results[0].success
        # Second step may not have run due to dependency failure

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, complete_setup):
        """Test graceful degradation when components fail."""
        from tests.test_orchestration import Gateway, Task, EventBus

        gateway = Gateway()
        event_bus = EventBus()

        errors = []

        async def error_handler(event_type, data):
            errors.append(data)

        event_bus.subscribe("error", error_handler)

        # Try to dispatch without any agents
        task = Task(task_id="t1", description="Test")
        result = await gateway.dispatch(task)

        # Should fail gracefully
        assert not result.success
        assert "No agents" in result.error


# ---------------------------------------------------------------------------
# Performance Integration Tests
# ---------------------------------------------------------------------------

class TestPerformanceIntegration:
    """Tests for performance under load."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, tmp_path):
        """Test handling concurrent sessions."""
        from tests.test_execution import SessionManager

        manager = SessionManager(tmp_path / "sessions")

        # Create multiple sessions concurrently
        tasks = [
            manager.create_session(f"agent-{i}")
            for i in range(10)
        ]

        sessions = await asyncio.gather(*tasks)

        assert len(sessions) == 10
        assert len(set(s.session_id for s in sessions)) == 10

    @pytest.mark.asyncio
    async def test_event_bus_throughput(self):
        """Test event bus can handle many events."""
        from tests.test_orchestration import EventBus

        bus = EventBus()
        received_count = {"value": 0}

        async def counter(event_type, data):
            received_count["value"] += 1

        bus.subscribe("test", counter)

        # Publish many events
        tasks = [bus.publish("test", i) for i in range(100)]
        await asyncio.gather(*tasks)

        assert received_count["value"] == 100

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, tmp_path):
        """Test that memory is properly cleaned up."""
        from tests.test_memory import CheckpointManager

        manager = CheckpointManager(tmp_path / "checkpoints", max_checkpoints=5)

        # Create many checkpoints
        for i in range(20):
            manager.save(context=[], state={"n": i})

        # Should only keep max_checkpoints
        assert len(manager.list_checkpoints()) == 5

        # Files should be cleaned up
        checkpoint_files = list((tmp_path / "checkpoints").glob("*.json"))
        assert len(checkpoint_files) == 5
