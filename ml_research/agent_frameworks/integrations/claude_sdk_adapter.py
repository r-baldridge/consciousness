"""
Claude Agent SDK native integration.

This module provides native integration with the Claude Agent SDK,
allowing seamless interoperability between our framework and Claude's
official agent implementation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
import logging
import asyncio

if TYPE_CHECKING:
    from ..tools.tool_base import Tool, ToolResult, ToolSchema
    from ..backends.backend_base import LLMBackend

logger = logging.getLogger(__name__)

# Try to import Claude SDK components
try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logger.debug("Anthropic SDK not available. Install with: pip install anthropic")


def require_claude_sdk(func: Callable) -> Callable:
    """Decorator to check if Claude SDK is available."""
    def wrapper(*args, **kwargs):
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "Anthropic SDK is not installed. "
                "Install with: pip install anthropic"
            )
        return func(*args, **kwargs)
    return wrapper


@dataclass
class ClaudeAgentConfig:
    """Configuration for Claude agent integration.

    Attributes:
        model: Model to use (default: claude-sonnet-4-20250514)
        max_tokens: Maximum tokens for responses
        system_prompt: Optional system prompt
        tools_enabled: Whether to enable tool use
    """
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    tools_enabled: bool = True


class ClaudeSDKAdapter:
    """Native integration with Claude Agent SDK.

    This class provides methods to convert between our agent format
    and the Claude Agent SDK's native format.

    Example:
        adapter = ClaudeSDKAdapter()

        # Convert our agent to Claude SDK format
        claude_agent = adapter.to_claude_agent(my_agent)

        # Run with Claude SDK
        result = await claude_agent.run("Hello")

        # Convert Claude SDK tool for our framework
        my_tool = adapter.wrap_tool_for_framework(claude_tool)

        # Use our tool with Claude SDK
        claude_tool = adapter.wrap_tool_for_claude(my_tool)
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude SDK adapter.

        Args:
            api_key: Optional Anthropic API key (uses env var if not provided)
        """
        self.api_key = api_key
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create synchronous Anthropic client."""
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError("Anthropic SDK not available")

        if self._client is None:
            if self.api_key:
                self._client = Anthropic(api_key=self.api_key)
            else:
                self._client = Anthropic()

        return self._client

    def _get_async_client(self) -> Any:
        """Get or create asynchronous Anthropic client."""
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError("Anthropic SDK not available")

        if self._async_client is None:
            if self.api_key:
                self._async_client = AsyncAnthropic(api_key=self.api_key)
            else:
                self._async_client = AsyncAnthropic()

        return self._async_client

    @require_claude_sdk
    def to_claude_agent(
        self,
        agent: Any,
        config: Optional[ClaudeAgentConfig] = None
    ) -> "ClaudeAgentWrapper":
        """Convert our agent to a Claude SDK-compatible wrapper.

        Args:
            agent: Our agent instance
            config: Optional configuration

        Returns:
            Claude SDK-compatible agent wrapper
        """
        config = config or ClaudeAgentConfig()

        return ClaudeAgentWrapper(
            agent=agent,
            adapter=self,
            config=config
        )

    @require_claude_sdk
    def from_claude_agent(self, claude_config: Dict[str, Any]) -> Any:
        """Create an agent from Claude SDK configuration.

        Args:
            claude_config: Configuration dict with model, system, etc.

        Returns:
            Our agent format wrapper
        """
        return _OurAgentFromClaude(
            adapter=self,
            config=claude_config
        )

    @require_claude_sdk
    def wrap_tool_for_claude(self, tool: "Tool") -> Dict[str, Any]:
        """Wrap our tool for use with Claude SDK.

        Args:
            tool: Our Tool instance

        Returns:
            Tool definition dict for Claude SDK
        """
        schema = tool.schema

        return {
            "name": schema.name,
            "description": schema.description,
            "input_schema": {
                "type": "object",
                "properties": schema.parameters,
                "required": schema.required
            }
        }

    @require_claude_sdk
    def wrap_tools_for_claude(self, tools: List["Tool"]) -> List[Dict[str, Any]]:
        """Wrap multiple tools for Claude SDK.

        Args:
            tools: List of our Tool instances

        Returns:
            List of tool definition dicts
        """
        return [self.wrap_tool_for_claude(tool) for tool in tools]

    @require_claude_sdk
    def create_tool_from_claude(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> "Tool":
        """Create our Tool from Claude SDK tool definition.

        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema for parameters
            handler: Async function to handle tool calls

        Returns:
            Our Tool instance
        """
        return _ClaudeToolWrapper(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )

    async def execute_tool_call(
        self,
        tool: "Tool",
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool call and format result for Claude.

        Args:
            tool: The tool to execute
            tool_input: Input parameters

        Returns:
            Result formatted for Claude SDK
        """
        try:
            result = await tool.execute(**tool_input)

            if result.success:
                return {
                    "type": "tool_result",
                    "content": str(result.output)
                }
            else:
                return {
                    "type": "tool_result",
                    "content": f"Error: {result.error}",
                    "is_error": True
                }
        except Exception as e:
            return {
                "type": "tool_result",
                "content": f"Exception: {str(e)}",
                "is_error": True
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics.

        Returns:
            Dictionary with adapter stats
        """
        return {
            "claude_sdk_available": CLAUDE_SDK_AVAILABLE,
            "has_client": self._client is not None,
            "has_async_client": self._async_client is not None
        }


class ClaudeAgentWrapper:
    """Wrapper that provides Claude SDK interface for our agents."""

    def __init__(
        self,
        agent: Any,
        adapter: ClaudeSDKAdapter,
        config: ClaudeAgentConfig
    ):
        """Initialize the wrapper.

        Args:
            agent: Our agent instance
            adapter: The ClaudeSDKAdapter
            config: Agent configuration
        """
        self._agent = agent
        self._adapter = adapter
        self._config = config
        self._messages: List[Dict[str, Any]] = []

    @property
    def id(self) -> str:
        return getattr(self._agent, 'id', str(id(self._agent)))

    async def run(self, input_text: str) -> str:
        """Run the agent with Claude SDK backend.

        Args:
            input_text: User input

        Returns:
            Agent response
        """
        client = self._adapter._get_async_client()

        # Add user message
        self._messages.append({
            "role": "user",
            "content": input_text
        })

        # Get tools if agent has them
        tools = []
        if self._config.tools_enabled and hasattr(self._agent, 'tools'):
            for tool in self._agent.tools:
                tools.append(self._adapter.wrap_tool_for_claude(tool))

        # Build request
        request_params = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "messages": self._messages
        }

        if self._config.system_prompt:
            request_params["system"] = self._config.system_prompt
        elif hasattr(self._agent, 'system_prompt'):
            request_params["system"] = self._agent.system_prompt

        if tools:
            request_params["tools"] = tools

        # Make request
        response = await client.messages.create(**request_params)

        # Process response
        result_text = ""
        tool_calls = []

        for block in response.content:
            if hasattr(block, 'text'):
                result_text += block.text
            elif hasattr(block, 'type') and block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        # Handle tool calls
        if tool_calls and hasattr(self._agent, 'tools'):
            tool_results = []
            tool_map = {t.name: t for t in self._agent.tools}

            for call in tool_calls:
                tool = tool_map.get(call["name"])
                if tool:
                    result = await self._adapter.execute_tool_call(
                        tool, call["input"]
                    )
                    result["tool_use_id"] = call["id"]
                    tool_results.append(result)

            # Add assistant message with tool use
            self._messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Add tool results
            self._messages.append({
                "role": "user",
                "content": tool_results
            })

            # Continue conversation
            return await self.run("")  # Recursive call to get final response

        # Add assistant response to history
        self._messages.append({
            "role": "assistant",
            "content": result_text
        })

        return result_text

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def get_state(self) -> Dict[str, Any]:
        return {
            "type": "claude_sdk_agent",
            "model": self._config.model,
            "messages": self._messages.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._messages = state.get("messages", [])


class _OurAgentFromClaude:
    """Wraps Claude SDK configuration in our agent interface."""

    def __init__(self, adapter: ClaudeSDKAdapter, config: Dict[str, Any]):
        self._adapter = adapter
        self._config = config
        self.id = f"claude-{id(config)}"
        self._messages: List[Dict[str, Any]] = []

    async def run(self, input_text: str) -> str:
        """Run with Claude SDK."""
        client = self._adapter._get_async_client()

        self._messages.append({
            "role": "user",
            "content": input_text
        })

        response = await client.messages.create(
            model=self._config.get("model", "claude-sonnet-4-20250514"),
            max_tokens=self._config.get("max_tokens", 4096),
            system=self._config.get("system"),
            messages=self._messages
        )

        result = ""
        for block in response.content:
            if hasattr(block, 'text'):
                result += block.text

        self._messages.append({
            "role": "assistant",
            "content": result
        })

        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "type": "claude_native",
            "config": self._config,
            "messages": self._messages.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._messages = state.get("messages", [])


class _ClaudeToolWrapper:
    """Wraps a Claude SDK tool definition in our Tool interface."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ):
        from ..tools.tool_base import ToolSchema

        self._handler = handler
        self._schema = ToolSchema(
            name=name,
            description=description,
            parameters=parameters.get("properties", {}),
            required=parameters.get("required", [])
        )

    @property
    def schema(self) -> "ToolSchema":
        return self._schema

    @property
    def name(self) -> str:
        return self._schema.name

    @property
    def description(self) -> str:
        return self._schema.description

    async def execute(self, **kwargs) -> "ToolResult":
        """Execute the tool."""
        from ..tools.tool_base import ToolResult

        try:
            if asyncio.iscoroutinefunction(self._handler):
                result = await self._handler(**kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self._handler(**kwargs))

            return ToolResult.ok(result)
        except Exception as e:
            return ToolResult.fail(str(e))
