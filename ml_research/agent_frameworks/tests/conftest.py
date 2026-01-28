"""
Shared pytest fixtures for agent_frameworks tests.

This module provides common fixtures used across all test modules,
including mock backends, temporary workspaces, and sample data.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import json
import subprocess


# ---------------------------------------------------------------------------
# Event Loop Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Mock Backend Fixtures
# ---------------------------------------------------------------------------

@dataclass
class MockLLMResponse:
    """Mock response from LLM backend."""
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 100, "output_tokens": 50})
    model: str = "mock-model"
    finish_reason: str = "stop"


class MockLLMBackend:
    """Mock LLM backend for testing without API calls."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        tool_calls: Optional[List[List[Dict[str, Any]]]] = None,
    ):
        """
        Initialize mock backend.

        Args:
            responses: List of text responses to return in order
            tool_calls: List of tool call lists to return in order
        """
        self._responses = responses or ["This is a mock response."]
        self._tool_calls = tool_calls or []
        self._call_count = 0
        self._call_history: List[Dict[str, Any]] = []

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Get history of all calls made to the backend."""
        return self._call_history

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        config: Any
    ) -> MockLLMResponse:
        """Return mock completion."""
        self._call_history.append({
            "type": "complete",
            "messages": messages,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        })

        idx = self._call_count % len(self._responses)
        tool_idx = self._call_count % len(self._tool_calls) if self._tool_calls else -1
        self._call_count += 1

        return MockLLMResponse(
            content=self._responses[idx],
            tool_calls=self._tool_calls[tool_idx] if tool_idx >= 0 else None,
        )

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Return mock generation."""
        self._call_history.append({
            "type": "generate",
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
        })

        idx = self._call_count % len(self._responses)
        self._call_count += 1
        return self._responses[idx]

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        config: Any
    ) -> AsyncIterator[str]:
        """Stream mock response."""
        self._call_history.append({
            "type": "stream",
            "messages": messages,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        })

        idx = self._call_count % len(self._responses)
        self._call_count += 1

        for word in self._responses[idx].split():
            yield word + " "

    async def embed(self, text: str) -> List[float]:
        """Return mock embedding."""
        self._call_history.append({
            "type": "embed",
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })
        # Return a simple deterministic embedding based on text length
        return [float(i) / 100 for i in range(128)]

    async def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def reset(self) -> None:
        """Reset call count and history."""
        self._call_count = 0
        self._call_history = []


@pytest.fixture
def mock_backend():
    """Provide a mock LLM backend for testing."""
    return MockLLMBackend()


@pytest.fixture
def mock_backend_with_plan():
    """Provide a mock backend that returns valid execution plans."""
    plan_response = json.dumps({
        "goal": "Test goal",
        "reasoning": "Test reasoning",
        "risk_assessment": "Low risk",
        "rollback_strategy": "Revert changes",
        "files_to_modify": ["test.py"],
        "files_to_create": [],
        "requires_approval": False,
        "estimated_duration": "1 minute",
        "steps": [
            {
                "description": "Read test file",
                "tool": "read_file",
                "arguments": {"path": "test.py"},
                "dependencies": [],
                "estimated_impact": "low",
                "rationale": "Need to understand current state",
                "expected_output": "File contents"
            },
            {
                "description": "Modify test file",
                "tool": "write_file",
                "arguments": {"path": "test.py", "content": "# Modified"},
                "dependencies": [0],
                "estimated_impact": "medium",
                "rationale": "Apply changes",
                "expected_output": "File written"
            }
        ]
    })
    return MockLLMBackend(responses=[plan_response])


# ---------------------------------------------------------------------------
# Workspace Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create some basic structure
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "docs").mkdir()

    # Create a sample Python file
    (workspace / "src" / "main.py").write_text('''"""Main module."""

def hello():
    """Say hello."""
    return "Hello, World!"

if __name__ == "__main__":
    print(hello())
''')

    # Create a sample test file
    (workspace / "tests" / "test_main.py").write_text('''"""Tests for main module."""

from src.main import hello

def test_hello():
    assert hello() == "Hello, World!"
''')

    yield workspace

    # Cleanup is automatic with tmp_path


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample git repository for testing."""
    repo_path = tmp_path / "sample_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True
    )

    # Create project structure
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()

    # Create files
    (repo_path / "README.md").write_text("# Sample Repository\n\nA test repository.")
    (repo_path / "src" / "__init__.py").write_text("")
    (repo_path / "src" / "app.py").write_text('''"""Application module."""

class App:
    """Main application class."""

    def __init__(self, name: str):
        self.name = name

    def run(self) -> str:
        """Run the application."""
        return f"Running {self.name}"

    def stop(self) -> str:
        """Stop the application."""
        return f"Stopping {self.name}"
''')
    (repo_path / "src" / "utils.py").write_text('''"""Utility functions."""

def format_message(msg: str) -> str:
    """Format a message."""
    return f"[MSG] {msg}"

def parse_config(path: str) -> dict:
    """Parse configuration file."""
    return {"path": path, "loaded": True}
''')

    # Add and commit files
    subprocess.run(
        ["git", "add", "."],
        cwd=repo_path,
        capture_output=True,
        check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True
    )

    yield repo_path


# ---------------------------------------------------------------------------
# Tool Fixtures
# ---------------------------------------------------------------------------

class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, returns: Any = None, raises: Optional[Exception] = None):
        self.name = name
        self._returns = returns or {"success": True}
        self._raises = raises
        self.call_count = 0
        self.call_args: List[Dict[str, Any]] = []

    async def execute(self, **kwargs) -> Any:
        """Execute the mock tool."""
        self.call_count += 1
        self.call_args.append(kwargs)

        if self._raises:
            raise self._raises

        return self._returns


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self, tools: Optional[Dict[str, MockTool]] = None):
        self._tools = tools or {}

    def register(self, tool: MockTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[MockTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        return await tool.execute(**arguments)

    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self._tools.keys())


@pytest.fixture
def mock_tool_registry():
    """Provide a mock tool registry with common tools."""
    registry = MockToolRegistry()

    # Add common mock tools
    registry.register(MockTool("read_file", returns={"content": "file contents"}))
    registry.register(MockTool("write_file", returns={"written": True}))
    registry.register(MockTool("bash", returns={"stdout": "", "stderr": "", "exit_code": 0}))
    registry.register(MockTool("search", returns={"results": []}))

    return registry


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "agent": {
            "default_mode": "architect",
            "max_iterations": 50,
            "timeout": 300,
        },
        "backend": {
            "provider": "mock",
            "model": "mock-model",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "tools": {
            "sandbox": {
                "enabled": True,
                "type": "subprocess",
                "timeout": 30,
            }
        }
    }


# ---------------------------------------------------------------------------
# Message Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_messages():
    """Provide sample message history for testing."""
    return [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Help me write a Python function."},
        {"role": "assistant", "content": "I'd be happy to help. What should the function do?"},
        {"role": "user", "content": "It should sort a list of numbers."},
    ]


# ---------------------------------------------------------------------------
# Async Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def async_timeout():
    """Default timeout for async operations in tests."""
    return 5.0


def run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Cleanup Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield
    # Import and reset singletons if they exist
    try:
        from agent_frameworks.tools.tool_registry import ToolRegistry
        ToolRegistry.reset()
    except ImportError:
        pass
