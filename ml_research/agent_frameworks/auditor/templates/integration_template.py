"""
Integration Code Generation Template.

This module provides templates and utilities for generating
framework integration code.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from string import Template


@dataclass
class IntegrationTemplate:
    """
    Template configuration for generating a framework integration.

    Attributes:
        source_framework: Framework being integrated
        target_framework: Framework to integrate with
        integration_name: Name for the integration
        adapter_type: Type of adapter (wrapper, bridge, translator)
        components: Components to integrate
        mappings: API mappings between frameworks
        imports: Additional imports needed
    """
    source_framework: str
    target_framework: str = "agent_frameworks"
    integration_name: Optional[str] = None
    adapter_type: str = "wrapper"  # wrapper, bridge, translator
    components: List[str] = field(default_factory=list)
    mappings: Dict[str, str] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate integration name if not provided."""
        if not self.integration_name:
            self.integration_name = f"{self.source_framework}Integration"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_framework": self.source_framework,
            "target_framework": self.target_framework,
            "integration_name": self.integration_name,
            "adapter_type": self.adapter_type,
            "components": self.components,
            "mappings": self.mappings,
            "imports": self.imports,
        }


# Base integration template
INTEGRATION_TEMPLATE = '''"""
${source_framework} Integration for ${target_framework}.

This module provides integration between ${source_framework} and ${target_framework},
allowing ${source_framework} components to be used within the ${target_framework} ecosystem.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
import asyncio
${additional_imports}

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend


@dataclass
class ${integration_name}Config:
    """Configuration for ${source_framework} integration."""
    auto_convert: bool = True
    preserve_metadata: bool = True
    strict_mode: bool = False
${config_fields}


class ${integration_name}:
    """
    Integration adapter for ${source_framework}.

    This adapter allows ${source_framework} components to be used with
    ${target_framework}, handling API translation and data conversion.

    Attributes:
        config: Integration configuration
        backend: Optional LLM backend for the target framework

    Example:
        ```python
        from ${module_path} import ${integration_name}

        integration = ${integration_name}()

        # Convert a ${source_framework} component
        converted = integration.convert_agent(${source_lower}_agent)

        # Use with ${target_framework}
        result = await converted.run("task")
        ```
    """

    def __init__(
        self,
        config: Optional[${integration_name}Config] = None,
        backend: Optional['LLMBackend'] = None,
    ):
        """
        Initialize the ${source_framework} integration.

        Args:
            config: Integration configuration
            backend: Optional LLM backend for the target framework
        """
        self.config = config or ${integration_name}Config()
        self.backend = backend
${init_body}

${agent_conversion}

${tool_conversion}

${chain_conversion}

${utility_methods}

${adapter_methods}
'''


# Agent conversion template
AGENT_CONVERSION = '''
    def convert_agent(
        self,
        agent: Any,
        name: Optional[str] = None,
    ) -> 'AdaptedAgent':
        """
        Convert a ${source_framework} agent to ${target_framework} format.

        Args:
            agent: The ${source_framework} agent to convert
            name: Optional name for the converted agent

        Returns:
            Adapted agent compatible with ${target_framework}
        """
        return AdaptedAgent(
            source_agent=agent,
            name=name or getattr(agent, "name", "${source_framework}Agent"),
            backend=self.backend,
            config=self.config,
        )


class AdaptedAgent:
    """Wrapper that adapts a ${source_framework} agent to ${target_framework} interface."""

    def __init__(
        self,
        source_agent: Any,
        name: str,
        backend: Optional['LLMBackend'] = None,
        config: Optional[${integration_name}Config] = None,
    ):
        self.source_agent = source_agent
        self.name = name
        self.backend = backend
        self.config = config or ${integration_name}Config()

    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute the adapted agent.

        Args:
            task: Task to execute
            context: Optional context

        Returns:
            Agent response
        """
        # Adapt the call to ${source_framework} format
        try:
            # Check if source agent is async
            if asyncio.iscoroutinefunction(getattr(self.source_agent, 'run', None)):
                result = await self.source_agent.run(task)
            elif hasattr(self.source_agent, 'invoke'):
                result = self.source_agent.invoke(task)
            elif hasattr(self.source_agent, 'run'):
                result = self.source_agent.run(task)
            else:
                result = str(self.source_agent(task))

            # Convert result to string if needed
            if hasattr(result, 'content'):
                return result.content
            elif isinstance(result, dict):
                return result.get('output', str(result))
            else:
                return str(result)

        except Exception as e:
            raise RuntimeError(f"Failed to execute adapted agent: {e}")
'''


# Tool conversion template
TOOL_CONVERSION = '''
    def convert_tool(
        self,
        tool: Any,
    ) -> 'AdaptedTool':
        """
        Convert a ${source_framework} tool to ${target_framework} format.

        Args:
            tool: The ${source_framework} tool to convert

        Returns:
            Adapted tool compatible with ${target_framework}
        """
        from ..tools.tool_base import Tool, ToolSchema, ToolResult

        return AdaptedTool(
            source_tool=tool,
            config=self.config,
        )


class AdaptedTool:
    """Wrapper that adapts a ${source_framework} tool to ${target_framework} interface."""

    def __init__(
        self,
        source_tool: Any,
        config: Optional[${integration_name}Config] = None,
    ):
        self.source_tool = source_tool
        self.config = config or ${integration_name}Config()

        # Extract tool information
        self._name = getattr(source_tool, 'name', 'adapted_tool')
        self._description = getattr(source_tool, 'description', '')

    @property
    def schema(self):
        """Return adapted tool schema."""
        from ..tools.tool_base import ToolSchema

        # Try to extract schema from source tool
        if hasattr(self.source_tool, 'args_schema'):
            params = self._convert_pydantic_schema(self.source_tool.args_schema)
        elif hasattr(self.source_tool, 'parameters'):
            params = self.source_tool.parameters
        else:
            params = {}

        return ToolSchema(
            name=self._name,
            description=self._description,
            parameters=params,
        )

    async def execute(self, **kwargs) -> 'ToolResult':
        """Execute the adapted tool."""
        from ..tools.tool_base import ToolResult

        try:
            # Call the source tool
            if asyncio.iscoroutinefunction(getattr(self.source_tool, 'run', None)):
                result = await self.source_tool.run(**kwargs)
            elif hasattr(self.source_tool, 'run'):
                result = self.source_tool.run(**kwargs)
            elif callable(self.source_tool):
                result = self.source_tool(**kwargs)
            else:
                raise ValueError("Source tool is not callable")

            return ToolResult.ok(result)

        except Exception as e:
            return ToolResult.fail(str(e))

    def _convert_pydantic_schema(self, schema: Any) -> Dict[str, Any]:
        """Convert Pydantic schema to JSON schema format."""
        if hasattr(schema, 'model_json_schema'):
            json_schema = schema.model_json_schema()
            return json_schema.get('properties', {})
        elif hasattr(schema, 'schema'):
            return schema.schema().get('properties', {})
        return {}
'''


# Chain/workflow conversion template
CHAIN_CONVERSION = '''
    def convert_chain(
        self,
        chain: Any,
    ) -> 'AdaptedChain':
        """
        Convert a ${source_framework} chain/workflow to ${target_framework} format.

        Args:
            chain: The ${source_framework} chain to convert

        Returns:
            Adapted chain compatible with ${target_framework}
        """
        return AdaptedChain(
            source_chain=chain,
            config=self.config,
        )


class AdaptedChain:
    """Wrapper that adapts a ${source_framework} chain to ${target_framework} interface."""

    def __init__(
        self,
        source_chain: Any,
        config: Optional[${integration_name}Config] = None,
    ):
        self.source_chain = source_chain
        self.config = config or ${integration_name}Config()

    async def run(
        self,
        input_data: Any,
    ) -> Any:
        """
        Execute the adapted chain.

        Args:
            input_data: Input for the chain

        Returns:
            Chain output
        """
        try:
            if asyncio.iscoroutinefunction(getattr(self.source_chain, 'ainvoke', None)):
                result = await self.source_chain.ainvoke(input_data)
            elif hasattr(self.source_chain, 'invoke'):
                result = self.source_chain.invoke(input_data)
            elif hasattr(self.source_chain, 'run'):
                result = self.source_chain.run(input_data)
            else:
                result = self.source_chain(input_data)

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to execute adapted chain: {e}")
'''


# Utility methods template
UTILITY_METHODS = '''
    def list_components(self) -> Dict[str, List[str]]:
        """
        List available components for conversion.

        Returns:
            Dictionary of component types to names
        """
        return {
            "agents": ${agent_list},
            "tools": ${tool_list},
            "chains": ${chain_list},
        }

    def validate_compatibility(
        self,
        component: Any,
    ) -> bool:
        """
        Check if a component is compatible for conversion.

        Args:
            component: Component to validate

        Returns:
            True if component can be converted
        """
        # Check for common interfaces
        return (
            hasattr(component, 'run') or
            hasattr(component, 'invoke') or
            callable(component)
        )
'''


def generate_integration(template: IntegrationTemplate) -> str:
    """
    Generate integration code from a template.

    Args:
        template: Integration template configuration

    Returns:
        Generated Python code
    """
    # Build imports
    additional_imports = ""
    if template.imports:
        additional_imports = "\n".join(f"import {imp}" for imp in template.imports)

    # Build config fields
    config_fields = ""
    for mapping_name in template.mappings.keys():
        config_fields += f"    {mapping_name}_enabled: bool = True\n"

    # Source framework lowercase
    source_lower = template.source_framework.lower()

    # Module path
    module_path = f"agent_frameworks.integrations.{source_lower}"

    # Component lists
    agent_list = str(template.components) if template.components else "[]"
    tool_list = "[]"
    chain_list = "[]"

    # Apply templates
    agent_conversion = Template(AGENT_CONVERSION).substitute(
        source_framework=template.source_framework,
        target_framework=template.target_framework,
        integration_name=template.integration_name,
    )

    tool_conversion = Template(TOOL_CONVERSION).substitute(
        source_framework=template.source_framework,
        target_framework=template.target_framework,
        integration_name=template.integration_name,
    )

    chain_conversion = Template(CHAIN_CONVERSION).substitute(
        source_framework=template.source_framework,
        target_framework=template.target_framework,
        integration_name=template.integration_name,
    )

    utility_methods = Template(UTILITY_METHODS).substitute(
        agent_list=agent_list,
        tool_list=tool_list,
        chain_list=chain_list,
    )

    # Build main code
    code = Template(INTEGRATION_TEMPLATE).substitute(
        source_framework=template.source_framework,
        target_framework=template.target_framework,
        integration_name=template.integration_name,
        source_lower=source_lower,
        additional_imports=additional_imports,
        config_fields=config_fields,
        module_path=module_path,
        init_body="        pass",
        agent_conversion=agent_conversion,
        tool_conversion=tool_conversion,
        chain_conversion=chain_conversion,
        utility_methods=utility_methods,
        adapter_methods="",
    )

    return code
