"""
LangChain integration adapters.

This module provides bidirectional integration with LangChain,
allowing use of our agents in LangChain and vice versa.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
import logging
import asyncio

if TYPE_CHECKING:
    from ..tools.tool_base import Tool, ToolResult, ToolSchema

logger = logging.getLogger(__name__)

# Try to import LangChain components
try:
    from langchain.agents import AgentExecutor
    from langchain.tools import BaseTool as LangChainBaseTool
    from langchain.schema import AgentAction, AgentFinish
    from langchain.callbacks.manager import CallbackManagerForToolRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.debug("LangChain not available. Install with: pip install langchain")


def require_langchain(func: Callable) -> Callable:
    """Decorator to check if LangChain is available."""
    def wrapper(*args, **kwargs):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. "
                "Install with: pip install langchain langchain-community"
            )
        return func(*args, **kwargs)
    return wrapper


class LangChainAgentAdapter:
    """Adapt our agents for use in LangChain.

    This class provides methods to convert our agents and tools to
    LangChain-compatible formats and vice versa.

    Example:
        adapter = LangChainAgentAdapter()

        # Convert our agent to LangChain
        lc_agent = adapter.to_langchain_agent(my_agent)

        # Use in LangChain executor
        executor = AgentExecutor(agent=lc_agent, tools=lc_tools)
        result = executor.run("Do something")

        # Convert LangChain agent to our format
        my_agent = adapter.from_langchain_agent(lc_agent)
    """

    def __init__(self):
        """Initialize the LangChain adapter."""
        self._converted_tools: Dict[str, Any] = {}

    @require_langchain
    def to_langchain_agent(self, agent: Any) -> Any:
        """Convert our agent to a LangChain agent.

        Args:
            agent: Our agent instance

        Returns:
            LangChain-compatible agent
        """
        # Create a wrapper that implements LangChain's agent interface
        return _LangChainAgentWrapper(agent)

    @require_langchain
    def from_langchain_agent(self, lc_agent: Any) -> Any:
        """Convert a LangChain agent to our format.

        Args:
            lc_agent: LangChain agent instance

        Returns:
            Our agent format wrapper
        """
        return _OurAgentFromLangChain(lc_agent)

    @require_langchain
    def to_langchain_tool(self, tool: "Tool") -> Any:
        """Convert our tool to a LangChain tool.

        Args:
            tool: Our Tool instance

        Returns:
            LangChain BaseTool instance
        """
        return _LangChainToolWrapper(tool)

    @require_langchain
    def to_langchain_tools(self, tools: List["Tool"]) -> List[Any]:
        """Convert multiple tools to LangChain format.

        Args:
            tools: List of our Tool instances

        Returns:
            List of LangChain BaseTool instances
        """
        return [self.to_langchain_tool(tool) for tool in tools]


class LangChainToolAdapter:
    """Adapt LangChain tools for use in our framework.

    Example:
        adapter = LangChainToolAdapter()

        # Convert LangChain tool to our format
        my_tool = adapter.from_langchain_tool(lc_tool)

        # Use in our framework
        result = await my_tool.execute(arg1="value")
    """

    @require_langchain
    def from_langchain_tool(self, lc_tool: Any) -> "Tool":
        """Convert a LangChain tool to our Tool format.

        Args:
            lc_tool: LangChain BaseTool instance

        Returns:
            Our Tool instance
        """
        return _OurToolFromLangChain(lc_tool)

    @require_langchain
    def from_langchain_tools(self, lc_tools: List[Any]) -> List["Tool"]:
        """Convert multiple LangChain tools to our format.

        Args:
            lc_tools: List of LangChain BaseTool instances

        Returns:
            List of our Tool instances
        """
        return [self.from_langchain_tool(tool) for tool in lc_tools]


# Implementation classes (only used when LangChain is available)

if LANGCHAIN_AVAILABLE:

    class _LangChainAgentWrapper:
        """Wraps our agent for LangChain compatibility."""

        def __init__(self, agent: Any):
            self.agent = agent

        def plan(self, intermediate_steps, **kwargs):
            """LangChain agent planning interface."""
            # Convert intermediate steps to our format
            # and get agent's next action
            loop = asyncio.get_event_loop()

            async def _plan():
                # Build context from intermediate steps
                context = []
                for action, observation in intermediate_steps:
                    context.append({
                        "action": action.tool,
                        "input": action.tool_input,
                        "output": observation
                    })

                # Get input from kwargs
                input_text = kwargs.get("input", "")

                # Ask agent for next action
                if hasattr(self.agent, 'step'):
                    result = await self.agent.step(input_text, context)
                elif hasattr(self.agent, 'run'):
                    result = await self.agent.run(input_text)
                else:
                    return AgentFinish(
                        return_values={"output": "Agent does not support step interface"},
                        log="No step interface"
                    )

                # Parse result
                if isinstance(result, dict):
                    if "action" in result:
                        return AgentAction(
                            tool=result["action"],
                            tool_input=result.get("action_input", {}),
                            log=result.get("log", "")
                        )
                    else:
                        return AgentFinish(
                            return_values={"output": result.get("output", str(result))},
                            log=result.get("log", "")
                        )
                else:
                    return AgentFinish(
                        return_values={"output": str(result)},
                        log=""
                    )

            return loop.run_until_complete(_plan())

        @property
        def input_keys(self) -> List[str]:
            return ["input"]

        @property
        def return_values(self) -> List[str]:
            return ["output"]


    class _OurAgentFromLangChain:
        """Wraps a LangChain agent in our interface."""

        def __init__(self, lc_agent: Any):
            self.lc_agent = lc_agent
            self.id = f"langchain-{id(lc_agent)}"

        async def run(self, input_text: str) -> str:
            """Run the agent."""
            # LangChain agents are typically sync
            loop = asyncio.get_event_loop()
            if hasattr(self.lc_agent, 'arun'):
                return await self.lc_agent.arun(input_text)
            else:
                return await loop.run_in_executor(
                    None,
                    lambda: self.lc_agent.run(input_text)
                )

        def get_state(self) -> Dict[str, Any]:
            return {"type": "langchain_agent"}

        def set_state(self, state: Dict[str, Any]) -> None:
            pass


    class _LangChainToolWrapper(LangChainBaseTool):
        """Wraps our Tool as a LangChain tool."""

        def __init__(self, tool: "Tool"):
            self._our_tool = tool
            # LangChain BaseTool requires name and description
            super().__init__(
                name=tool.schema.name,
                description=tool.schema.description
            )

        def _run(
            self,
            *args,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs
        ) -> str:
            """Synchronous run."""
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._our_tool.execute(**kwargs))
            return str(result.output) if result.success else f"Error: {result.error}"

        async def _arun(
            self,
            *args,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs
        ) -> str:
            """Asynchronous run."""
            result = await self._our_tool.execute(**kwargs)
            return str(result.output) if result.success else f"Error: {result.error}"

        @property
        def args_schema(self) -> Dict[str, Any]:
            """Return the tool's argument schema."""
            return self._our_tool.schema.to_json_schema()


    class _OurToolFromLangChain:
        """Wraps a LangChain tool in our Tool interface."""

        def __init__(self, lc_tool: LangChainBaseTool):
            self._lc_tool = lc_tool
            self._schema = self._build_schema()

        def _build_schema(self) -> "ToolSchema":
            """Build our ToolSchema from LangChain tool."""
            from ..tools.tool_base import ToolSchema

            # Get parameters from LangChain tool
            parameters = {}
            if hasattr(self._lc_tool, 'args_schema') and self._lc_tool.args_schema:
                if hasattr(self._lc_tool.args_schema, 'schema'):
                    schema = self._lc_tool.args_schema.schema()
                    parameters = schema.get('properties', {})

            return ToolSchema(
                name=self._lc_tool.name,
                description=self._lc_tool.description or "",
                parameters=parameters,
                required=[]
            )

        @property
        def schema(self) -> "ToolSchema":
            return self._schema

        async def execute(self, **kwargs) -> "ToolResult":
            """Execute the LangChain tool."""
            from ..tools.tool_base import ToolResult

            try:
                # Try async first
                if hasattr(self._lc_tool, '_arun'):
                    result = await self._lc_tool._arun(**kwargs)
                else:
                    # Fall back to sync
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._lc_tool._run(**kwargs)
                    )
                return ToolResult.ok(result)
            except Exception as e:
                return ToolResult.fail(str(e))

        @property
        def name(self) -> str:
            return self._schema.name

        @property
        def description(self) -> str:
            return self._schema.description

else:
    # Stub classes when LangChain is not available
    class _LangChainAgentWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain not available")

    class _OurAgentFromLangChain:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain not available")

    class _LangChainToolWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain not available")

    class _OurToolFromLangChain:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain not available")
