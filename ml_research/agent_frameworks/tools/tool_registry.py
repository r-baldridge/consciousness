"""Dynamic tool registration and discovery.

This module provides a registry for managing tools, including:
- Registering and retrieving tools by name
- Filtering tools by permission
- A decorator for creating function-based tools
"""

from __future__ import annotations
import inspect
import functools
from typing import (
    Dict, Any, Optional, List, Callable, Type, TypeVar,
    get_type_hints, Union, Awaitable
)
from dataclasses import dataclass, field

from .tool_base import (
    Tool, ToolSchema, ToolResult, ToolPermission,
    python_type_to_json_schema
)


T = TypeVar('T', bound=Tool)
F = TypeVar('F', bound=Callable[..., Awaitable[ToolResult]])


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found in the registry."""
    pass


class ToolAlreadyRegisteredError(Exception):
    """Raised when attempting to register a tool with a name that's already taken."""
    pass


class ToolRegistry:
    """Singleton registry for managing available tools.

    The registry provides centralized management of tools, allowing
    tools to be registered, retrieved, and filtered by various criteria.
    """

    _instance: Optional[ToolRegistry] = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, Tool] = {}
            cls._instance._initialized = True
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None

    def register(self, tool: Tool, override: bool = False) -> Tool:
        """Register a tool with the registry.

        Args:
            tool: The tool instance to register
            override: If True, allow overriding existing tool with same name

        Returns:
            The registered tool

        Raises:
            ToolAlreadyRegisteredError: If tool name exists and override is False
        """
        name = tool.schema.name
        if name in self._tools and not override:
            raise ToolAlreadyRegisteredError(
                f"Tool '{name}' is already registered. "
                f"Use override=True to replace it."
            )
        self._tools[name] = tool
        return tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Name of the tool to remove

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")
        del self._tools[name]

    def get(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The requested tool

        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not found. "
                f"Available tools: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def get_optional(self, name: str) -> Optional[Tool]:
        """Get a tool by name, returning None if not found.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools.

        Returns:
            List of all registered tools
        """
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List names of all registered tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_by_permission(self, permission: ToolPermission) -> List[Tool]:
        """Get all tools that require a specific permission.

        Args:
            permission: The permission to filter by

        Returns:
            List of tools requiring the specified permission
        """
        return [
            tool for tool in self._tools.values()
            if permission in tool.schema.permissions
        ]

    def get_safe_tools(self) -> List[Tool]:
        """Get all tools that don't require dangerous permission.

        Returns:
            List of tools without DANGEROUS permission
        """
        return [
            tool for tool in self._tools.values()
            if ToolPermission.DANGEROUS not in tool.schema.permissions
        ]

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools.

        Returns:
            List of tool schemas in dictionary format
        """
        return [tool.schema.to_dict() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _registry


class FunctionTool(Tool):
    """A tool created from a decorated function."""

    def __init__(
        self,
        func: Callable[..., Awaitable[ToolResult]],
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[List[ToolPermission]] = None,
        parameter_descriptions: Optional[Dict[str, str]] = None
    ):
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or "No description provided"
        self._permissions = permissions or []
        self._parameter_descriptions = parameter_descriptions or {}
        self._schema = self._build_schema()

    def _build_schema(self) -> ToolSchema:
        """Build the tool schema from the function signature."""
        sig = inspect.signature(self._func)
        hints = get_type_hints(self._func)

        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            # Get type hint
            param_type = hints.get(param_name, Any)
            param_schema = python_type_to_json_schema(param_type)

            # Add description if provided
            if param_name in self._parameter_descriptions:
                param_schema["description"] = self._parameter_descriptions[param_name]

            parameters[param_name] = param_schema

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return ToolSchema(
            name=self._name,
            description=self._description.strip(),
            parameters=parameters,
            required=required,
            permissions=self._permissions
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the wrapped function."""
        try:
            result = await self._func(**kwargs)
            if isinstance(result, ToolResult):
                return result
            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    permissions: Optional[List[ToolPermission]] = None,
    parameter_descriptions: Optional[Dict[str, str]] = None,
    register: bool = True
) -> Callable[[F], FunctionTool]:
    """Decorator to create a tool from an async function.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        permissions: Required permissions
        parameter_descriptions: Descriptions for each parameter
        register: Whether to register the tool globally

    Returns:
        Decorator function that creates a FunctionTool

    Example:
        @tool(
            name="read_file",
            permissions=[ToolPermission.READ],
            parameter_descriptions={"path": "Path to the file to read"}
        )
        async def read_file(path: str) -> ToolResult:
            '''Read contents of a file.'''
            content = Path(path).read_text()
            return ToolResult.ok(content)
    """
    def decorator(func: F) -> FunctionTool:
        tool_instance = FunctionTool(
            func=func,
            name=name,
            description=description,
            permissions=permissions,
            parameter_descriptions=parameter_descriptions
        )

        if register:
            get_registry().register(tool_instance)

        return tool_instance

    return decorator


def register_tool_class(
    tool_class: Type[T],
    *args,
    **kwargs
) -> T:
    """Register a tool class with the global registry.

    Args:
        tool_class: The tool class to instantiate and register
        *args: Arguments to pass to the tool constructor
        **kwargs: Keyword arguments to pass to the tool constructor

    Returns:
        The registered tool instance
    """
    instance = tool_class(*args, **kwargs)
    get_registry().register(instance)
    return instance
