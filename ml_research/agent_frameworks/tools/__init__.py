"""Tools module for the agent framework.

This module provides the complete tool system including:
- Base classes for defining tools (Tool, ToolSchema, ToolResult)
- Permission system for controlling tool access
- Tool registry for dynamic tool management
- Built-in tools for common operations

Usage:
    from agent_frameworks.tools import (
        Tool, ToolSchema, ToolResult, ToolPermission,
        ToolRegistry, get_registry, tool,
        PermissionManager, UserContext
    )

    # Create a tool using the decorator
    @tool(
        name="my_tool",
        permissions=[ToolPermission.READ]
    )
    async def my_tool(path: str) -> ToolResult:
        '''My custom tool.'''
        return ToolResult.ok(f"Processed {path}")

    # Or create a tool class
    class MyTool(Tool):
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(
                name="my_tool",
                description="My custom tool",
                parameters={"path": {"type": "string"}},
                required=["path"]
            )

        async def execute(self, path: str) -> ToolResult:
            return ToolResult.ok(f"Processed {path}")

    # Register and use tools
    registry = get_registry()
    registry.register(MyTool())
    tool = registry.get("my_tool")
    result = await tool(path="/some/path")
"""

# Core tool classes
from .tool_base import (
    Tool,
    ToolSchema,
    ToolResult,
    ToolPermission,
    python_type_to_json_schema,
)

# Registry
from .tool_registry import (
    ToolRegistry,
    ToolNotFoundError,
    ToolAlreadyRegisteredError,
    FunctionTool,
    get_registry,
    tool,
    register_tool_class,
)

# Permissions
from .permissions import (
    PermissionManager,
    PermissionDecision,
    PermissionRequest,
    PermissionGrant,
    PermissionPolicy,
    AllowAllPolicy,
    DenyDangerousPolicy,
    RequireApprovalPolicy,
    PermissionMatchingPolicy,
    CompositePolicy,
    UserContext,
    ApprovalHandler,
    check_and_execute,
)

# Built-in tools
from .builtin import (
    # File tools
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    ListDirectoryTool,
    get_file_tools,

    # Git tools
    GitStatusTool,
    GitDiffTool,
    GitCommitTool,
    GitBranchTool,
    GitLogTool,
    GitAddTool,
    get_git_tools,

    # Shell tools
    BashTool,
    BackgroundProcessTool,
    WorkingDirectoryTool,
    SandboxConfig,
    get_shell_tools,

    # Search tools
    GrepTool,
    RipgrepTool,
    SemanticSearchTool,
    SearchMatch,
    get_search_tools,

    # Web tools
    WebFetchTool,
    WebSearchTool,
    URLParserTool,
    FetchConfig,
    SearchConfig,
    get_web_tools,

    # Convenience
    get_all_builtin_tools,
)

__all__ = [
    # Core
    "Tool",
    "ToolSchema",
    "ToolResult",
    "ToolPermission",
    "python_type_to_json_schema",

    # Registry
    "ToolRegistry",
    "ToolNotFoundError",
    "ToolAlreadyRegisteredError",
    "FunctionTool",
    "get_registry",
    "tool",
    "register_tool_class",

    # Permissions
    "PermissionManager",
    "PermissionDecision",
    "PermissionRequest",
    "PermissionGrant",
    "PermissionPolicy",
    "AllowAllPolicy",
    "DenyDangerousPolicy",
    "RequireApprovalPolicy",
    "PermissionMatchingPolicy",
    "CompositePolicy",
    "UserContext",
    "ApprovalHandler",
    "check_and_execute",

    # File tools
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
    "ListDirectoryTool",
    "get_file_tools",

    # Git tools
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    "GitBranchTool",
    "GitLogTool",
    "GitAddTool",
    "get_git_tools",

    # Shell tools
    "BashTool",
    "BackgroundProcessTool",
    "WorkingDirectoryTool",
    "SandboxConfig",
    "get_shell_tools",

    # Search tools
    "GrepTool",
    "RipgrepTool",
    "SemanticSearchTool",
    "SearchMatch",
    "get_search_tools",

    # Web tools
    "WebFetchTool",
    "WebSearchTool",
    "URLParserTool",
    "FetchConfig",
    "SearchConfig",
    "get_web_tools",

    # Convenience
    "get_all_builtin_tools",
]


def register_all_builtin_tools(**kwargs):
    """Register all built-in tools with the global registry.

    Args:
        **kwargs: Arguments passed to get_all_builtin_tools()

    Returns:
        List of registered tools
    """
    registry = get_registry()
    tools = get_all_builtin_tools(**kwargs)
    for t in tools:
        registry.register(t, override=True)
    return tools
