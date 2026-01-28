"""
Code Generation Templates.

This module provides templates for generating various types of
integration code including agents, tools, and framework integrations.
"""

from .agent_template import AgentTemplate, generate_agent
from .tool_template import ToolTemplate, generate_tool
from .integration_template import IntegrationTemplate, generate_integration

__all__ = [
    "AgentTemplate",
    "ToolTemplate",
    "IntegrationTemplate",
    "generate_agent",
    "generate_tool",
    "generate_integration",
]
