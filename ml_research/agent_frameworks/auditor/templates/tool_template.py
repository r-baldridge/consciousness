"""
Tool Code Generation Template.

This module provides templates and utilities for generating
new tool implementations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from string import Template


@dataclass
class ToolTemplate:
    """
    Template configuration for generating a tool.

    Attributes:
        name: Tool name (snake_case)
        class_name: Tool class name (PascalCase)
        description: Tool description
        parameters: List of parameter definitions
        return_type: Return type annotation
        is_async: Whether the tool is async
        permissions: Required permissions
        imports: Additional imports needed
    """
    name: str
    class_name: Optional[str] = None
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "Any"
    is_async: bool = True
    permissions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate class name from tool name if not provided."""
        if not self.class_name:
            # Convert snake_case to PascalCase
            self.class_name = "".join(
                word.capitalize() for word in self.name.split("_")
            ) + "Tool"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "class_name": self.class_name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "is_async": self.is_async,
            "permissions": self.permissions,
            "imports": self.imports,
        }


# Base tool template
TOOL_TEMPLATE = '''"""
${description}

This module provides the ${class_name} implementation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
${additional_imports}

from ..tools.tool_base import Tool, ToolSchema, ToolResult, ToolPermission


class ${class_name}(Tool):
    """
    ${description}

    This tool provides ${capability_description}.

    Attributes:
        schema: Tool schema definition

    Example:
        ```python
        from ${module_path} import ${class_name}

        tool = ${class_name}()
        result = ${await_prefix}tool(${example_args})
        print(result.output)
        ```
    """

    def __init__(self${init_params}):
        """
        Initialize the ${class_name}.
${init_param_docs}
        """
${init_body}

    @property
    def schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name="${name}",
            description="${description}",
            parameters={
${parameter_schema}
            },
            required=[${required_params}],
            permissions=[${permissions}],
        )

    ${async_def}execute(
        self,
${execute_params}
    ) -> ToolResult:
        """
        Execute the tool.

        Args:
${execute_param_docs}

        Returns:
            ToolResult with the execution outcome
        """
        try:
${execute_body}

            return ToolResult.ok(result)

        except Exception as e:
            return ToolResult.fail(str(e))

${additional_methods}
'''


# Decorator-based tool template
DECORATOR_TOOL_TEMPLATE = '''"""
${description}

This module provides the ${name} tool as a decorated function.
"""

from typing import ${type_imports}
${additional_imports}

from ..tools.tool_base import ToolResult


${async_def}${name}(
${function_params}
) -> ToolResult:
    """
    ${description}

    Args:
${function_param_docs}

    Returns:
        ToolResult with the execution outcome

    Example:
        ```python
        result = ${await_prefix}${name}(${example_args})
        print(result.output)
        ```
    """
    try:
${execute_body}

        return ToolResult.ok(result)

    except Exception as e:
        return ToolResult.fail(str(e))


# Tool schema for registration
${name}_schema = {
    "name": "${name}",
    "description": "${description}",
    "parameters": {
${parameter_schema}
    },
    "required": [${required_params}],
}
'''


def generate_tool(template: ToolTemplate, use_class: bool = True) -> str:
    """
    Generate tool code from a template.

    Args:
        template: Tool template configuration
        use_class: Whether to generate class-based tool

    Returns:
        Generated Python code
    """
    # Build async keywords
    async_def = "async def " if template.is_async else "def "
    await_prefix = "await " if template.is_async else ""

    # Build imports
    additional_imports = ""
    if template.imports:
        additional_imports = "\n".join(f"import {imp}" for imp in template.imports)

    # Build parameter schema
    parameter_schema_lines = []
    for param in template.parameters:
        param_name = param.get("name", "param")
        param_type = param.get("type", "string")
        param_desc = param.get("description", "")

        schema_entry = f'                "{param_name}": {{"type": "{param_type}", "description": "{param_desc}"}}'
        parameter_schema_lines.append(schema_entry)

    parameter_schema = ",\n".join(parameter_schema_lines)

    # Build required params
    required_params = ", ".join(
        f'"{p["name"]}"' for p in template.parameters if p.get("required", False)
    )

    # Build permissions
    permissions = ", ".join(
        f'ToolPermission.{p.upper()}' for p in template.permissions
    )

    # Build execute params
    execute_params_lines = []
    execute_param_docs_lines = []
    function_params_lines = []
    function_param_docs_lines = []

    for param in template.parameters:
        param_name = param.get("name", "param")
        param_type = param.get("python_type", "Any")
        param_desc = param.get("description", "")
        param_default = param.get("default")

        if param_default is not None:
            execute_params_lines.append(f"        {param_name}: {param_type} = {param_default!r},")
            function_params_lines.append(f"    {param_name}: {param_type} = {param_default!r},")
        else:
            execute_params_lines.append(f"        {param_name}: {param_type},")
            function_params_lines.append(f"    {param_name}: {param_type},")

        execute_param_docs_lines.append(f"            {param_name}: {param_desc}")
        function_param_docs_lines.append(f"        {param_name}: {param_desc}")

    execute_params = "\n".join(execute_params_lines)
    execute_param_docs = "\n".join(execute_param_docs_lines)
    function_params = "\n".join(function_params_lines)
    function_param_docs = "\n".join(function_param_docs_lines)

    # Build example args
    example_args = ", ".join(
        f'{p["name"]}=...' for p in template.parameters[:2]
    )

    # Build execute body (placeholder)
    execute_body = "            # TODO: Implement tool logic\n"
    execute_body += "            result = None"

    # Build type imports
    type_imports = "Any"
    if template.parameters:
        types_used = set()
        for param in template.parameters:
            ptype = param.get("python_type", "Any")
            if ptype.startswith("List"):
                types_used.add("List")
            elif ptype.startswith("Dict"):
                types_used.add("Dict")
            elif ptype.startswith("Optional"):
                types_used.add("Optional")
        if types_used:
            type_imports = ", ".join(sorted(types_used)) + ", Any"

    # Module path
    module_path = "agent_frameworks.tools"

    if use_class:
        # Class-based tool
        code = Template(TOOL_TEMPLATE).substitute(
            name=template.name,
            class_name=template.class_name,
            description=template.description or f"{template.name} tool.",
            capability_description=template.description or "tool functionality",
            additional_imports=additional_imports,
            module_path=module_path,
            await_prefix=await_prefix,
            init_params="",
            init_param_docs="",
            init_body="        pass",
            parameter_schema=parameter_schema,
            required_params=required_params,
            permissions=permissions,
            async_def=async_def,
            execute_params=execute_params,
            execute_param_docs=execute_param_docs,
            execute_body=execute_body,
            example_args=example_args,
            additional_methods="",
        )
    else:
        # Function-based tool
        code = Template(DECORATOR_TOOL_TEMPLATE).substitute(
            name=template.name,
            description=template.description or f"{template.name} tool.",
            type_imports=type_imports,
            additional_imports=additional_imports,
            async_def=async_def,
            await_prefix=await_prefix,
            function_params=function_params,
            function_param_docs=function_param_docs,
            execute_body=execute_body,
            example_args=example_args,
            parameter_schema=parameter_schema,
            required_params=required_params,
        )

    return code
