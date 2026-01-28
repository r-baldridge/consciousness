"""
Execution Module for Agent Frameworks.

This module provides a complete execution infrastructure for AI agents,
inspired by OpenCode's architecture. It separates planning from execution,
manages sessions, provides sandboxed tool execution, and supports
client-server architectures for distributed agent deployments.

Components:
    - ArchitectEditor: Separates planning (architect) from execution (editor)
    - SessionManager: Multi-session support with persistence
    - ToolExecutor: Sandboxed tool execution with resource limits
    - AgentServer/AgentClient: Client-server architecture for remote agents

Example:
    from agent_frameworks.execution import (
        ArchitectEditor, ExecutionMode, ExecutionPlan,
        SessionManager, Session,
        ToolExecutor, SandboxMode,
        AgentServer, AgentClient, ServerConfig
    )

    # Create an architect-editor workflow
    arch_edit = ArchitectEditor(backend)
    plan = await arch_edit.plan("Add user authentication", context)
    result = await arch_edit.execute(plan, tools)

    # Manage sessions
    session_mgr = SessionManager("/path/to/sessions")
    session = await session_mgr.create_session("agent-1")

    # Execute tools in sandbox
    executor = ToolExecutor(sandbox_mode=SandboxMode.SUBPROCESS)
    result = await executor.execute("file_write", {"path": "test.txt", "content": "hello"})

    # Run agent as server
    server = AgentServer(agent, ServerConfig(port=8765))
    await server.start()
"""

from .architect_editor import (
    ExecutionMode,
    PlanStep,
    ExecutionPlan,
    ExecutionResult,
    ArchitectEditor,
)

from .session_manager import (
    SessionState,
    Session,
    SessionManager,
)

from .tool_executor import (
    SandboxMode,
    ResourceLimits,
    ExecutionContext,
    ToolResult,
    AuditLogEntry,
    ToolExecutor,
)

from .client_server import (
    ServerConfig,
    TaskResult,
    AgentServer,
    AgentClient,
)

__all__ = [
    # Architect-Editor
    "ExecutionMode",
    "PlanStep",
    "ExecutionPlan",
    "ExecutionResult",
    "ArchitectEditor",
    # Session Manager
    "SessionState",
    "Session",
    "SessionManager",
    # Tool Executor
    "SandboxMode",
    "ResourceLimits",
    "ExecutionContext",
    "ToolResult",
    "AuditLogEntry",
    "ToolExecutor",
    # Client-Server
    "ServerConfig",
    "TaskResult",
    "AgentServer",
    "AgentClient",
]
