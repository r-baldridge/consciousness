"""
Microsoft AutoGen integration adapter.

This module provides integration with Microsoft's AutoGen framework,
enabling multi-agent conversations and group chat capabilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union, TYPE_CHECKING
import logging
import asyncio

if TYPE_CHECKING:
    from ..tools.tool_base import Tool

logger = logging.getLogger(__name__)

# Try to import AutoGen components
try:
    from autogen import (
        AssistantAgent,
        UserProxyAgent,
        GroupChat,
        GroupChatManager,
        ConversableAgent
    )
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    logger.debug("AutoGen not available. Install with: pip install pyautogen")


def require_autogen(func: Callable) -> Callable:
    """Decorator to check if AutoGen is available."""
    def wrapper(*args, **kwargs):
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is not installed. "
                "Install with: pip install pyautogen"
            )
        return func(*args, **kwargs)
    return wrapper


@dataclass
class AutoGenAgentConfig:
    """Configuration for AutoGen agent conversion.

    Attributes:
        name: Agent name
        system_message: System prompt for the agent
        human_input_mode: How to handle human input ("NEVER", "ALWAYS", "TERMINATE")
        max_consecutive_auto_reply: Max auto replies before stopping
        llm_config: LLM configuration dict
    """
    name: str
    system_message: Optional[str] = None
    human_input_mode: str = "NEVER"
    max_consecutive_auto_reply: int = 10
    llm_config: Optional[Dict[str, Any]] = None


class AutoGenAdapter:
    """Integrate with Microsoft AutoGen.

    This class provides methods to convert our agents to AutoGen format
    and create AutoGen group chats from multiple agents.

    Example:
        adapter = AutoGenAdapter()

        # Convert single agent
        ag_agent = adapter.to_autogen_agent(
            my_agent,
            config=AutoGenAgentConfig(name="assistant")
        )

        # Create group chat
        group_chat = adapter.create_group_chat(
            agents=[agent1, agent2, agent3],
            max_round=10
        )

        # Run group chat
        result = await adapter.run_group_chat(
            group_chat,
            "Discuss the architecture"
        )
    """

    def __init__(self, default_llm_config: Optional[Dict[str, Any]] = None):
        """Initialize the AutoGen adapter.

        Args:
            default_llm_config: Default LLM config for converted agents
        """
        self.default_llm_config = default_llm_config or {
            "config_list": [],
            "temperature": 0.7
        }
        self._agent_map: Dict[str, Any] = {}

    @require_autogen
    def to_autogen_agent(
        self,
        agent: Any,
        config: Optional[AutoGenAgentConfig] = None,
        agent_type: str = "assistant"
    ) -> Any:
        """Convert our agent to an AutoGen agent.

        Args:
            agent: Our agent instance
            config: Optional agent configuration
            agent_type: Type of AutoGen agent ("assistant", "user_proxy")

        Returns:
            AutoGen agent instance
        """
        # Build config
        if config is None:
            name = getattr(agent, 'name', None) or f"agent_{id(agent)}"
            system_message = getattr(agent, 'system_prompt', None)
            config = AutoGenAgentConfig(
                name=name,
                system_message=system_message
            )

        # Get LLM config
        llm_config = config.llm_config or self.default_llm_config

        # Create appropriate agent type
        if agent_type == "user_proxy":
            ag_agent = _AutoGenUserProxyWrapper(
                name=config.name,
                human_input_mode=config.human_input_mode,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                our_agent=agent
            )
        else:
            ag_agent = _AutoGenAssistantWrapper(
                name=config.name,
                system_message=config.system_message or "You are a helpful assistant.",
                llm_config=llm_config,
                our_agent=agent
            )

        # Store mapping
        agent_id = getattr(agent, 'id', str(id(agent)))
        self._agent_map[agent_id] = ag_agent

        return ag_agent

    @require_autogen
    def from_autogen_agent(self, ag_agent: Any) -> Any:
        """Convert an AutoGen agent to our format.

        Args:
            ag_agent: AutoGen agent instance

        Returns:
            Our agent format wrapper
        """
        return _OurAgentFromAutoGen(ag_agent)

    @require_autogen
    def create_group_chat(
        self,
        agents: List[Any],
        max_round: int = 10,
        admin_name: str = "Admin",
        speaker_selection_method: str = "auto",
        **kwargs
    ) -> Any:
        """Create an AutoGen group chat.

        Args:
            agents: List of agents (ours or AutoGen)
            max_round: Maximum conversation rounds
            admin_name: Name of the admin
            speaker_selection_method: How to select next speaker
            **kwargs: Additional GroupChat arguments

        Returns:
            AutoGen GroupChat and GroupChatManager tuple
        """
        # Convert agents if needed
        ag_agents = []
        for i, agent in enumerate(agents):
            if hasattr(agent, '_autogen_agent'):
                ag_agents.append(agent._autogen_agent)
            elif isinstance(agent, (AssistantAgent, UserProxyAgent, ConversableAgent)):
                ag_agents.append(agent)
            else:
                # Convert our agent
                config = AutoGenAgentConfig(
                    name=getattr(agent, 'name', f"agent_{i}")
                )
                ag_agents.append(self.to_autogen_agent(agent, config))

        # Create group chat
        group_chat = GroupChat(
            agents=ag_agents,
            messages=[],
            max_round=max_round,
            speaker_selection_method=speaker_selection_method,
            **kwargs
        )

        # Create manager
        manager = GroupChatManager(
            groupchat=group_chat,
            name=admin_name,
            llm_config=self.default_llm_config
        )

        return group_chat, manager

    @require_autogen
    async def run_group_chat(
        self,
        group_chat_tuple: tuple,
        initial_message: str,
        initiator: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Run an AutoGen group chat.

        Args:
            group_chat_tuple: (GroupChat, GroupChatManager) tuple
            initial_message: Message to start the conversation
            initiator: Agent to initiate (uses first agent if not specified)

        Returns:
            List of conversation messages
        """
        group_chat, manager = group_chat_tuple

        # Get initiator
        if initiator is None:
            initiator = group_chat.agents[0]
        elif hasattr(initiator, '_autogen_agent'):
            initiator = initiator._autogen_agent

        # Run chat (AutoGen is synchronous)
        loop = asyncio.get_event_loop()

        def _run():
            initiator.initiate_chat(
                manager,
                message=initial_message
            )
            return group_chat.messages

        messages = await loop.run_in_executor(None, _run)
        return messages

    @require_autogen
    def register_function(
        self,
        agent: Any,
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Register a function with an AutoGen agent.

        Args:
            agent: AutoGen agent or our agent wrapper
            function: The function to register
            name: Function name (derived from function if not provided)
            description: Function description
        """
        # Get the AutoGen agent
        if hasattr(agent, '_autogen_agent'):
            ag_agent = agent._autogen_agent
        else:
            ag_agent = agent

        # Build function definition
        func_name = name or function.__name__
        func_desc = description or function.__doc__ or ""

        # Register with AutoGen
        ag_agent.register_function(
            function_map={func_name: function}
        )

    @require_autogen
    def register_tool(self, agent: Any, tool: "Tool") -> None:
        """Register one of our tools with an AutoGen agent.

        Args:
            agent: AutoGen agent or wrapper
            tool: Our Tool instance
        """
        async def tool_wrapper(**kwargs):
            result = await tool.execute(**kwargs)
            return result.output if result.success else f"Error: {result.error}"

        # Convert to sync for AutoGen
        def sync_wrapper(**kwargs):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(tool_wrapper(**kwargs))

        self.register_function(
            agent,
            sync_wrapper,
            name=tool.name,
            description=tool.description
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics.

        Returns:
            Dictionary with adapter stats
        """
        return {
            "autogen_available": AUTOGEN_AVAILABLE,
            "converted_agents": len(self._agent_map),
            "default_llm_config": self.default_llm_config
        }


if AUTOGEN_AVAILABLE:

    class _AutoGenAssistantWrapper(AssistantAgent):
        """Wrapper bridging our agent to AutoGen AssistantAgent."""

        def __init__(
            self,
            name: str,
            system_message: str,
            llm_config: Dict[str, Any],
            our_agent: Any
        ):
            self._our_agent = our_agent
            super().__init__(
                name=name,
                system_message=system_message,
                llm_config=llm_config
            )

        @property
        def id(self) -> str:
            return getattr(self._our_agent, 'id', str(id(self._our_agent)))


    class _AutoGenUserProxyWrapper(UserProxyAgent):
        """Wrapper bridging our agent to AutoGen UserProxyAgent."""

        def __init__(
            self,
            name: str,
            human_input_mode: str,
            max_consecutive_auto_reply: int,
            our_agent: Any
        ):
            self._our_agent = our_agent
            super().__init__(
                name=name,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply
            )

        @property
        def id(self) -> str:
            return getattr(self._our_agent, 'id', str(id(self._our_agent)))


    class _OurAgentFromAutoGen:
        """Wraps an AutoGen agent in our interface."""

        def __init__(self, ag_agent: Union[AssistantAgent, UserProxyAgent]):
            self._autogen_agent = ag_agent
            self.id = f"autogen-{id(ag_agent)}"
            self.name = ag_agent.name

        async def run(self, input_text: str) -> str:
            """Run the agent."""
            # AutoGen agents typically work in conversations
            # For single-turn, we create a temporary user proxy
            loop = asyncio.get_event_loop()

            def _execute():
                # Create a simple user proxy to chat with this agent
                user = UserProxyAgent(
                    name="user",
                    human_input_mode="NEVER",
                    max_consecutive_auto_reply=0
                )

                user.initiate_chat(
                    self._autogen_agent,
                    message=input_text
                )

                # Get last assistant message
                for msg in reversed(user.chat_messages[self._autogen_agent]):
                    if msg.get("role") == "assistant":
                        return msg.get("content", "")

                return ""

            return await loop.run_in_executor(None, _execute)

        def get_state(self) -> Dict[str, Any]:
            return {
                "type": "autogen_agent",
                "name": self._autogen_agent.name
            }

        def set_state(self, state: Dict[str, Any]) -> None:
            pass

else:
    # Stub classes when AutoGen is not available
    class _AutoGenAssistantWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("AutoGen not available")

    class _AutoGenUserProxyWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("AutoGen not available")

    class _OurAgentFromAutoGen:
        def __init__(self, *args, **kwargs):
            raise ImportError("AutoGen not available")
