"""
Tests for tools components.

Tests cover:
    - Tool base class
    - ToolRegistry operations
    - Permission system
    - Builtin tools
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tests for Tool Base Class
# ---------------------------------------------------------------------------

class TestToolBase:
    """Tests for Tool abstract base class."""

    def test_tool_schema_creation(self):
        """Test creating a tool schema."""
        from agent_frameworks.tools.tool_base import ToolSchema, ToolPermission

        schema = ToolSchema(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "path": {"type": "string", "description": "File path"}
            },
            required=["path"],
            permissions=[ToolPermission.READ]
        )

        assert schema.name == "read_file"
        assert "path" in schema.parameters
        assert ToolPermission.READ in schema.permissions

    def test_tool_schema_to_dict(self):
        """Test schema serialization."""
        from agent_frameworks.tools.tool_base import ToolSchema, ToolPermission

        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters={"arg": {"type": "string"}},
            required=["arg"],
            permissions=[ToolPermission.READ, ToolPermission.WRITE]
        )

        data = schema.to_dict()

        assert data["name"] == "test"
        assert "read" in data["permissions"]
        assert "write" in data["permissions"]

    def test_tool_schema_to_json_schema(self):
        """Test conversion to JSON Schema."""
        from agent_frameworks.tools.tool_base import ToolSchema

        schema = ToolSchema(
            name="test",
            description="Test",
            parameters={"x": {"type": "integer"}},
            required=["x"]
        )

        json_schema = schema.to_json_schema()

        assert json_schema["type"] == "object"
        assert "x" in json_schema["properties"]
        assert "x" in json_schema["required"]

    def test_tool_result_ok(self):
        """Test creating successful result."""
        from agent_frameworks.tools.tool_base import ToolResult

        result = ToolResult.ok("file contents", path="/test.txt")

        assert result.success
        assert result.output == "file contents"
        assert result.metadata["path"] == "/test.txt"
        assert result.error is None

    def test_tool_result_fail(self):
        """Test creating failed result."""
        from agent_frameworks.tools.tool_base import ToolResult

        result = ToolResult.fail("File not found", code=404)

        assert not result.success
        assert result.error == "File not found"
        assert result.metadata["code"] == 404


# ---------------------------------------------------------------------------
# Tests for ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Tests for ToolRegistry operations."""

    def setup_method(self):
        """Reset registry before each test."""
        from agent_frameworks.tools.tool_registry import ToolRegistry
        ToolRegistry.reset()

    def test_singleton_pattern(self):
        """Test registry is a singleton."""
        from agent_frameworks.tools.tool_registry import ToolRegistry

        reg1 = ToolRegistry()
        reg2 = ToolRegistry()

        assert reg1 is reg2

    def test_register_tool(self):
        """Test registering a tool."""
        from agent_frameworks.tools.tool_registry import (
            ToolRegistry, FunctionTool
        )
        from agent_frameworks.tools.tool_base import ToolResult

        async def my_tool(x: int) -> ToolResult:
            """A test tool."""
            return ToolResult.ok(x * 2)

        tool = FunctionTool(my_tool, name="my_tool")
        registry = ToolRegistry()
        registry.register(tool)

        assert "my_tool" in registry

    def test_get_tool(self):
        """Test retrieving a tool."""
        from agent_frameworks.tools.tool_registry import (
            ToolRegistry, FunctionTool, ToolNotFoundError
        )
        from agent_frameworks.tools.tool_base import ToolResult

        async def test_fn(x: str) -> ToolResult:
            return ToolResult.ok(x)

        tool = FunctionTool(test_fn, name="test_tool")
        registry = ToolRegistry()
        registry.register(tool)

        retrieved = registry.get("test_tool")
        assert retrieved is tool

        with pytest.raises(ToolNotFoundError):
            registry.get("nonexistent")

    def test_list_tools(self):
        """Test listing all tools."""
        from agent_frameworks.tools.tool_registry import (
            ToolRegistry, FunctionTool
        )
        from agent_frameworks.tools.tool_base import ToolResult

        async def tool1() -> ToolResult:
            return ToolResult.ok(1)

        async def tool2() -> ToolResult:
            return ToolResult.ok(2)

        registry = ToolRegistry()
        registry.register(FunctionTool(tool1, name="tool1"))
        registry.register(FunctionTool(tool2, name="tool2"))

        names = registry.list_names()
        assert "tool1" in names
        assert "tool2" in names

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        from agent_frameworks.tools.tool_registry import (
            ToolRegistry, FunctionTool, ToolNotFoundError
        )
        from agent_frameworks.tools.tool_base import ToolResult

        async def temp_tool() -> ToolResult:
            return ToolResult.ok(None)

        registry = ToolRegistry()
        tool = FunctionTool(temp_tool, name="temp")
        registry.register(tool)

        assert "temp" in registry
        registry.unregister("temp")
        assert "temp" not in registry

    def test_get_by_permission(self):
        """Test filtering tools by permission."""
        from agent_frameworks.tools.tool_registry import (
            ToolRegistry, FunctionTool
        )
        from agent_frameworks.tools.tool_base import ToolResult, ToolPermission

        async def read_tool() -> ToolResult:
            return ToolResult.ok(None)

        async def write_tool() -> ToolResult:
            return ToolResult.ok(None)

        registry = ToolRegistry()
        registry.register(FunctionTool(
            read_tool, name="read",
            permissions=[ToolPermission.READ]
        ))
        registry.register(FunctionTool(
            write_tool, name="write",
            permissions=[ToolPermission.WRITE]
        ))

        read_tools = registry.get_by_permission(ToolPermission.READ)
        write_tools = registry.get_by_permission(ToolPermission.WRITE)

        assert len(read_tools) == 1
        assert read_tools[0].name == "read"
        assert len(write_tools) == 1
        assert write_tools[0].name == "write"

    def test_get_safe_tools(self):
        """Test getting tools without DANGEROUS permission."""
        from agent_frameworks.tools.tool_registry import (
            ToolRegistry, FunctionTool
        )
        from agent_frameworks.tools.tool_base import ToolResult, ToolPermission

        async def safe_tool() -> ToolResult:
            return ToolResult.ok(None)

        async def dangerous_tool() -> ToolResult:
            return ToolResult.ok(None)

        registry = ToolRegistry()
        registry.register(FunctionTool(
            safe_tool, name="safe",
            permissions=[ToolPermission.READ]
        ))
        registry.register(FunctionTool(
            dangerous_tool, name="dangerous",
            permissions=[ToolPermission.DANGEROUS]
        ))

        safe_tools = registry.get_safe_tools()
        assert len(safe_tools) == 1
        assert safe_tools[0].name == "safe"


# ---------------------------------------------------------------------------
# Tests for @tool Decorator
# ---------------------------------------------------------------------------

class TestToolDecorator:
    """Tests for @tool decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        from agent_frameworks.tools.tool_registry import ToolRegistry
        ToolRegistry.reset()

    def test_basic_decorator(self):
        """Test basic tool decoration."""
        from agent_frameworks.tools.tool_registry import tool, get_registry
        from agent_frameworks.tools.tool_base import ToolResult

        @tool(name="decorated_tool")
        async def my_tool(x: int) -> ToolResult:
            """Multiply by 2."""
            return ToolResult.ok(x * 2)

        registry = get_registry()
        assert "decorated_tool" in registry

    def test_decorator_auto_registers(self):
        """Test that decorator auto-registers by default."""
        from agent_frameworks.tools.tool_registry import tool, get_registry
        from agent_frameworks.tools.tool_base import ToolResult

        @tool()
        async def auto_registered() -> ToolResult:
            """Auto registered tool."""
            return ToolResult.ok(None)

        registry = get_registry()
        assert "auto_registered" in registry

    def test_decorator_no_register(self):
        """Test decorator with register=False."""
        from agent_frameworks.tools.tool_registry import tool, get_registry
        from agent_frameworks.tools.tool_base import ToolResult

        @tool(register=False)
        async def not_registered() -> ToolResult:
            return ToolResult.ok(None)

        registry = get_registry()
        assert "not_registered" not in registry

    def test_decorator_with_permissions(self):
        """Test decorator with permissions."""
        from agent_frameworks.tools.tool_registry import tool, get_registry
        from agent_frameworks.tools.tool_base import ToolResult, ToolPermission

        @tool(permissions=[ToolPermission.WRITE, ToolPermission.DANGEROUS])
        async def dangerous_write() -> ToolResult:
            return ToolResult.ok(None)

        registry = get_registry()
        t = registry.get("dangerous_write")
        assert ToolPermission.WRITE in t.permissions
        assert ToolPermission.DANGEROUS in t.permissions

    @pytest.mark.asyncio
    async def test_decorated_tool_execution(self):
        """Test executing a decorated tool."""
        from agent_frameworks.tools.tool_registry import tool
        from agent_frameworks.tools.tool_base import ToolResult

        @tool(name="adder", register=False)
        async def add(a: int, b: int) -> ToolResult:
            """Add two numbers."""
            return ToolResult.ok(a + b)

        result = await add.execute(a=5, b=3)

        assert result.success
        assert result.output == 8


# ---------------------------------------------------------------------------
# Tests for Permission System
# ---------------------------------------------------------------------------

class TestPermissionSystem:
    """Tests for permission management."""

    def test_user_context_creation(self):
        """Test creating user context."""
        from agent_frameworks.tools.permissions import UserContext, ToolPermission

        ctx = UserContext(
            user_id="user123",
            roles={"developer", "reviewer"},
            permissions={ToolPermission.READ, ToolPermission.WRITE}
        )

        assert ctx.user_id == "user123"
        assert ctx.has_role("developer")
        assert ctx.has_permission(ToolPermission.READ)
        assert not ctx.has_permission(ToolPermission.DANGEROUS)

    def test_admin_context(self):
        """Test admin context has all permissions."""
        from agent_frameworks.tools.permissions import UserContext, ToolPermission

        admin = UserContext.admin()

        assert admin.has_role("admin")
        for perm in ToolPermission:
            assert admin.has_permission(perm)

    def test_anonymous_context(self):
        """Test anonymous context has minimal permissions."""
        from agent_frameworks.tools.permissions import UserContext, ToolPermission

        anon = UserContext.anonymous()

        assert anon.user_id == "anonymous"
        assert anon.has_permission(ToolPermission.READ)
        assert not anon.has_permission(ToolPermission.WRITE)

    def test_allow_all_policy(self):
        """Test policy that allows everything."""
        from agent_frameworks.tools.permissions import (
            AllowAllPolicy, PermissionDecision, UserContext
        )
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult, ToolPermission

        async def tool_fn():
            return ToolResult.ok(None)

        tool = FunctionTool(
            tool_fn, name="test",
            permissions=[ToolPermission.DANGEROUS]
        )
        policy = AllowAllPolicy()
        ctx = UserContext.anonymous()

        decision = policy.evaluate(tool, ctx)
        assert decision == PermissionDecision.ALLOW

    def test_deny_dangerous_policy(self):
        """Test policy that denies dangerous tools."""
        from agent_frameworks.tools.permissions import (
            DenyDangerousPolicy, PermissionDecision, UserContext
        )
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult, ToolPermission

        async def tool_fn():
            return ToolResult.ok(None)

        dangerous_tool = FunctionTool(
            tool_fn, name="dangerous",
            permissions=[ToolPermission.DANGEROUS]
        )
        safe_tool = FunctionTool(
            tool_fn, name="safe",
            permissions=[ToolPermission.READ]
        )

        policy = DenyDangerousPolicy()
        user = UserContext(user_id="user1")
        admin = UserContext.admin()

        # Non-admin denied for dangerous tool
        assert policy.evaluate(dangerous_tool, user) == PermissionDecision.DENY

        # Admin allowed for dangerous tool
        assert policy.evaluate(dangerous_tool, admin) == PermissionDecision.ALLOW

        # Everyone allowed for safe tool
        assert policy.evaluate(safe_tool, user) == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_permission_manager_grant(self):
        """Test granting permissions."""
        from agent_frameworks.tools.permissions import PermissionManager
        from datetime import timedelta

        manager = PermissionManager()

        grant = manager.grant_permission(
            tool_name="bash",
            user_id="user1",
            duration=timedelta(hours=1)
        )

        assert grant.tool_name == "bash"
        assert grant.user_id == "user1"
        assert not grant.is_expired

    @pytest.mark.asyncio
    async def test_permission_manager_revoke(self):
        """Test revoking permissions."""
        from agent_frameworks.tools.permissions import PermissionManager

        manager = PermissionManager()
        manager.grant_permission("tool1", "user1")

        result = manager.revoke_permission("tool1", "user1")
        assert result is True

        result = manager.revoke_permission("tool1", "user1")
        assert result is False  # Already revoked


# ---------------------------------------------------------------------------
# Tests for FunctionTool
# ---------------------------------------------------------------------------

class TestFunctionTool:
    """Tests for FunctionTool class."""

    def test_schema_from_function(self):
        """Test schema is built from function signature."""
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult

        async def my_func(name: str, count: int = 1) -> ToolResult:
            """Process items."""
            return ToolResult.ok(None)

        tool = FunctionTool(my_func)

        assert tool.name == "my_func"
        assert "name" in tool.schema.parameters
        assert "count" in tool.schema.parameters
        assert "name" in tool.schema.required
        assert "count" not in tool.schema.required  # Has default

    def test_custom_name_and_description(self):
        """Test overriding name and description."""
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult

        async def func() -> ToolResult:
            """Original docstring."""
            return ToolResult.ok(None)

        tool = FunctionTool(
            func,
            name="custom_name",
            description="Custom description"
        )

        assert tool.name == "custom_name"
        assert tool.description == "Custom description"

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self):
        """Test that execute returns ToolResult."""
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult

        async def echo(msg: str) -> ToolResult:
            return ToolResult.ok(f"Echo: {msg}")

        tool = FunctionTool(echo)
        result = await tool.execute(msg="hello")

        assert isinstance(result, ToolResult)
        assert result.success
        assert result.output == "Echo: hello"

    @pytest.mark.asyncio
    async def test_execute_wraps_non_result(self):
        """Test that plain return values are wrapped."""
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult

        async def returns_string(x: int) -> str:
            return f"Value: {x}"

        tool = FunctionTool(returns_string)
        result = await tool.execute(x=42)

        # Should be wrapped in ToolResult
        assert isinstance(result, ToolResult)
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_catches_exceptions(self):
        """Test that exceptions become failed results."""
        from agent_frameworks.tools.tool_registry import FunctionTool
        from agent_frameworks.tools.tool_base import ToolResult

        async def failing_func() -> ToolResult:
            raise ValueError("Something went wrong")

        tool = FunctionTool(failing_func)
        result = await tool.execute()

        assert isinstance(result, ToolResult)
        assert not result.success
        assert "Something went wrong" in result.error
