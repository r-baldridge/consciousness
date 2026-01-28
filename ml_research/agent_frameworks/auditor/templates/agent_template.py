"""
Agent Code Generation Template.

This module provides templates and utilities for generating
new agent implementations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from string import Template


@dataclass
class AgentTemplate:
    """
    Template configuration for generating an agent.

    Attributes:
        name: Agent class name
        description: Agent description for docstring
        capabilities: List of agent capabilities
        tools: List of tool names the agent uses
        has_memory: Whether agent has memory/state
        has_planning: Whether agent has planning capability
        has_reflection: Whether agent has reflection capability
        async_mode: Whether to use async methods
        imports: Additional imports needed
    """
    name: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    has_memory: bool = False
    has_planning: bool = False
    has_reflection: bool = False
    async_mode: bool = True
    imports: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "has_memory": self.has_memory,
            "has_planning": self.has_planning,
            "has_reflection": self.has_reflection,
            "async_mode": self.async_mode,
            "imports": self.imports,
        }


# Base agent template
AGENT_TEMPLATE = '''"""
${description}

This module provides the ${name} agent implementation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
${additional_imports}

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig


@dataclass
class ${name}Config:
    """Configuration for ${name}."""
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: int = 4096
${config_fields}


class ${name}:
    """
    ${description}

    Attributes:
        backend: LLM backend for inference
        config: Agent configuration
${attribute_docs}

    Example:
        ```python
        from agent_frameworks.backends import AnthropicBackend
        from ${module_path} import ${name}

        backend = AnthropicBackend()
        agent = ${name}(backend)

        result = ${await_prefix}agent.run("Your task here")
        print(result)
        ```
    """

    SYSTEM_PROMPT = """You are ${name}, an AI assistant.
${system_prompt_content}
"""

    def __init__(
        self,
        backend: 'LLMBackend',
        config: Optional[${name}Config] = None,
${init_params}
    ):
        """
        Initialize the ${name}.

        Args:
            backend: LLM backend for inference
            config: Agent configuration
${init_param_docs}
        """
        self.backend = backend
        self.config = config or ${name}Config()
${init_body}

${memory_code}

${planning_code}

${reflection_code}

    ${async_def}run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute the agent on a task.

        Args:
            task: The task to execute
            context: Optional additional context

        Returns:
            Agent response
        """
        from ..backends.backend_base import LLMConfig

        messages = [
            {"role": "user", "content": task}
        ]

        if context:
            context_str = "\\n".join(f"{k}: {v}" for k, v in context.items())
            messages[0]["content"] = f"Context:\\n{context_str}\\n\\nTask: {task}"

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            system=self.SYSTEM_PROMPT,
        )

${run_body}

        response = ${await_keyword}self.backend.complete(messages, config)
        return response.content

${tool_code}

${additional_methods}
'''


MEMORY_CODE = '''
    def _init_memory(self) -> None:
        """Initialize agent memory."""
        self._memory: List[Dict[str, Any]] = []
        self._context_window: List[Dict[str, str]] = []

    def remember(self, key: str, value: Any) -> None:
        """Store a value in memory."""
        self._memory.append({"key": key, "value": value})

    def recall(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        for item in reversed(self._memory):
            if item["key"] == key:
                return item["value"]
        return None

    def clear_memory(self) -> None:
        """Clear all memory."""
        self._memory.clear()
        self._context_window.clear()
'''


PLANNING_CODE = '''
    ${async_def}_plan(
        self,
        task: str,
    ) -> List[str]:
        """
        Create a plan for executing the task.

        Args:
            task: The task to plan for

        Returns:
            List of steps to execute
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.3,
            max_tokens=1000,
            system="Break down the task into clear, actionable steps.",
        )

        messages = [{"role": "user", "content": f"Create a step-by-step plan for: {task}"}]
        response = ${await_keyword}self.backend.complete(messages, config)

        # Parse steps from response
        steps = []
        for line in response.content.split("\\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                steps.append(line.lstrip("0123456789.-) "))

        return steps
'''


REFLECTION_CODE = '''
    ${async_def}_reflect(
        self,
        action: str,
        result: str,
    ) -> str:
        """
        Reflect on an action and its result.

        Args:
            action: The action taken
            result: The result of the action

        Returns:
            Reflection insights
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.4,
            max_tokens=500,
            system="Analyze the action and result. Identify learnings and improvements.",
        )

        prompt = f"Action: {action}\\nResult: {result}\\n\\nReflect on this."
        messages = [{"role": "user", "content": prompt}]
        response = ${await_keyword}self.backend.complete(messages, config)

        return response.content
'''


TOOL_CODE = '''
    ${async_def}_execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.tools[tool_name]
        return ${await_keyword}tool(**arguments)
'''


def generate_agent(template: AgentTemplate) -> str:
    """
    Generate agent code from a template.

    Args:
        template: Agent template configuration

    Returns:
        Generated Python code
    """
    # Build async keywords
    async_def = "async def " if template.async_mode else "def "
    await_keyword = "await " if template.async_mode else ""
    await_prefix = "await " if template.async_mode else ""

    # Build imports
    additional_imports = ""
    if template.imports:
        additional_imports = "\n".join(f"import {imp}" for imp in template.imports)

    # Build config fields
    config_fields = ""
    if template.has_memory:
        config_fields += "    memory_size: int = 100\n"
    if template.has_planning:
        config_fields += "    planning_depth: int = 5\n"

    # Build attribute docs
    attribute_docs = ""
    if template.tools:
        attribute_docs += "        tools: Available tools for the agent\n"
    if template.has_memory:
        attribute_docs += "        memory: Agent memory store\n"

    # Build init params
    init_params = ""
    init_param_docs = ""
    init_body = ""

    if template.tools:
        init_params += "        tools: Optional[Dict[str, Any]] = None,\n"
        init_param_docs += "            tools: Optional dictionary of tools\n"
        init_body += "        self.tools = tools or {}\n"

    # Build memory code
    memory_code = ""
    if template.has_memory:
        memory_code = Template(MEMORY_CODE).substitute(
            async_def=async_def,
            await_keyword=await_keyword,
        )
        init_body += "        self._init_memory()\n"

    # Build planning code
    planning_code = ""
    if template.has_planning:
        planning_code = Template(PLANNING_CODE).substitute(
            async_def=async_def,
            await_keyword=await_keyword,
        )

    # Build reflection code
    reflection_code = ""
    if template.has_reflection:
        reflection_code = Template(REFLECTION_CODE).substitute(
            async_def=async_def,
            await_keyword=await_keyword,
        )

    # Build tool code
    tool_code = ""
    if template.tools:
        tool_code = Template(TOOL_CODE).substitute(
            async_def=async_def,
            await_keyword=await_keyword,
        )

    # Build run body
    run_body = ""
    if template.has_planning:
        run_body += f"        steps = {await_keyword}self._plan(task)\n"
        run_body += "        # Execute steps...\n"

    # Build system prompt content
    system_prompt_content = ""
    if template.capabilities:
        system_prompt_content = "Your capabilities include:\n"
        for cap in template.capabilities:
            system_prompt_content += f"- {cap}\n"

    # Determine module path
    module_path = "agent_frameworks.agents"

    # Apply template
    code = Template(AGENT_TEMPLATE).substitute(
        name=template.name,
        description=template.description or f"{template.name} agent implementation.",
        additional_imports=additional_imports,
        config_fields=config_fields,
        attribute_docs=attribute_docs,
        module_path=module_path,
        await_prefix=await_prefix,
        system_prompt_content=system_prompt_content,
        init_params=init_params,
        init_param_docs=init_param_docs,
        init_body=init_body,
        memory_code=memory_code,
        planning_code=planning_code,
        reflection_code=reflection_code,
        async_def=async_def,
        await_keyword=await_keyword,
        run_body=run_body,
        tool_code=tool_code,
        additional_methods="",
    )

    return code
