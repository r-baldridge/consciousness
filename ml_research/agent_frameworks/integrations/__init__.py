"""
Integrations Module for Agent Frameworks.

This module provides compatibility layers and adapters for integrating
with other popular agent frameworks and tools, including Aider, LangChain,
CrewAI, AutoGen, and the Claude Agent SDK.

Components:
    - AiderCompatAgent: Drop-in replacement for Aider workflows
    - LangChainAgentAdapter: Bidirectional LangChain integration
    - CrewAIAdapter: CrewAI multi-agent framework integration
    - ClaudeSDKAdapter: Native Claude Agent SDK integration
    - AutoGenAdapter: Microsoft AutoGen integration

Example:
    from agent_frameworks.integrations import (
        AiderCompatAgent,
        LangChainAgentAdapter,
        CrewAIAdapter,
        ClaudeSDKAdapter,
        AutoGenAdapter
    )

    # Use Aider-compatible interface
    aider = AiderCompatAgent(backend)
    await aider.add_file("src/main.py")
    result = await aider.code("Add error handling to the main function")

    # Convert to LangChain agent
    lc_adapter = LangChainAgentAdapter()
    lc_agent = lc_adapter.to_langchain_agent(my_agent)

    # Create CrewAI team
    crew_adapter = CrewAIAdapter()
    crew = crew_adapter.create_crew([agent1, agent2], tasks)
"""

from .aider_compat import (
    AiderCompatAgent,
    AiderConfig,
    FileChange,
)

from .langchain_adapter import (
    LangChainAgentAdapter,
    LangChainToolAdapter,
)

from .crewai_adapter import (
    CrewAIAdapter,
)

from .claude_sdk_adapter import (
    ClaudeSDKAdapter,
)

from .autogen_adapter import (
    AutoGenAdapter,
)

__all__ = [
    # Aider compatibility
    "AiderCompatAgent",
    "AiderConfig",
    "FileChange",
    # LangChain
    "LangChainAgentAdapter",
    "LangChainToolAdapter",
    # CrewAI
    "CrewAIAdapter",
    # Claude SDK
    "ClaudeSDKAdapter",
    # AutoGen
    "AutoGenAdapter",
]
