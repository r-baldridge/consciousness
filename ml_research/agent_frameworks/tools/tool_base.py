"""Base classes for tool definitions in the agent framework.

This module provides the foundational abstractions for creating tools
that agents can use to interact with the external world.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Type, get_type_hints
from enum import Enum
import json

# Optional jsonschema support
try:
    import jsonschema
    from jsonschema import validate as jsonschema_validate, ValidationError as JsonSchemaValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    JsonSchemaValidationError = Exception  # Fallback type


class ValidationError(Exception):
    """Raised when tool argument validation fails."""
    pass


class ToolPermission(Enum):
    """Permission levels for tool operations."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DANGEROUS = "dangerous"


@dataclass
class ToolSchema:
    """Schema definition for a tool.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        parameters: JSON Schema defining the tool's input parameters
        required: List of required parameter names
        permissions: List of permissions required to use this tool
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required: List[str] = field(default_factory=list)
    permissions: List[ToolPermission] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "permissions": [p.value for p in self.permissions]
        }

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for validation."""
        return {
            "type": "object",
            "properties": self.parameters,
            "required": self.required
        }


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        output: The output data from the tool
        error: Error message if execution failed
        metadata: Additional metadata about the execution
    """
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def ok(cls, output: Any, **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata) -> "ToolResult":
        """Create a failed result."""
        return cls(success=False, output=None, error=error, metadata=metadata)


class Tool(ABC):
    """Abstract base class for all tools.

    Tools are the primary way agents interact with the external world.
    Each tool must define its schema and implement the execute method.
    """

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the schema for this tool."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments matching the tool's parameter schema

        Returns:
            ToolResult containing the execution outcome
        """
        ...

    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate arguments against the tool's schema.

        Args:
            args: Dictionary of argument name to value

        Returns:
            True if arguments are valid

        Raises:
            ValidationError: If arguments don't match schema
        """
        json_schema = self.schema.to_json_schema()

        if HAS_JSONSCHEMA:
            try:
                jsonschema_validate(instance=args, schema=json_schema)
                return True
            except JsonSchemaValidationError as e:
                raise ValidationError(f"Invalid arguments for tool {self.schema.name}: {e.message}")
        else:
            # Basic validation without jsonschema
            for required_param in self.schema.required:
                if required_param not in args:
                    raise ValidationError(
                        f"Invalid arguments for tool {self.schema.name}: "
                        f"missing required parameter '{required_param}'"
                    )
            return True

    @property
    def name(self) -> str:
        """Convenience property to get tool name."""
        return self.schema.name

    @property
    def description(self) -> str:
        """Convenience property to get tool description."""
        return self.schema.description

    @property
    def permissions(self) -> List[ToolPermission]:
        """Convenience property to get required permissions."""
        return self.schema.permissions

    async def __call__(self, **kwargs) -> ToolResult:
        """Allow calling tool directly as a function."""
        self.validate_args(kwargs)
        return await self.execute(**kwargs)


# Type mapping for schema generation from Python types
PYTHON_TO_JSON_SCHEMA: Dict[Type, Dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    type(None): {"type": "null"},
}


def python_type_to_json_schema(python_type: Type) -> Dict[str, Any]:
    """Convert a Python type hint to JSON Schema.

    Args:
        python_type: A Python type annotation

    Returns:
        JSON Schema dictionary
    """
    # Handle Optional types
    origin = getattr(python_type, "__origin__", None)

    if origin is type(None):
        return {"type": "null"}

    # Handle Optional[X] which is Union[X, None]
    if origin is type(None) or str(origin) == "typing.Union":
        args = getattr(python_type, "__args__", ())
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_json_schema(non_none_args[0])
        # Multiple types - use anyOf
        return {"anyOf": [python_type_to_json_schema(a) for a in args]}

    # Handle List[X]
    if origin is list:
        args = getattr(python_type, "__args__", ())
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle Dict[K, V]
    if origin is dict:
        args = getattr(python_type, "__args__", ())
        if len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": python_type_to_json_schema(args[1])
            }
        return {"type": "object"}

    # Basic types
    if python_type in PYTHON_TO_JSON_SCHEMA:
        return PYTHON_TO_JSON_SCHEMA[python_type]

    # Default to any
    return {}
