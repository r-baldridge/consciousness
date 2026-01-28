"""
LLM-driven Code Generation for Integrations.

This module provides tools for generating integration code from
extracted patterns, creating adapters between frameworks, and
generating comprehensive tests.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig
    from .auditor_agent import Pattern


@dataclass
class IntegrationCode:
    """
    Generated integration code with supporting files.

    Attributes:
        filename: Name of the main integration file
        code: The generated code
        dependencies: Required package dependencies
        usage_example: Example of how to use the integration
        tests: Generated test code
        metadata: Additional metadata
    """
    filename: str
    code: str
    dependencies: List[str] = field(default_factory=list)
    usage_example: str = ""
    tests: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def save(
        self,
        path: Path,
        include_tests: bool = True,
    ) -> List[Path]:
        """
        Save the integration code to files.

        Args:
            path: Directory to save files in
            include_tests: Whether to save test file

        Returns:
            List of created file paths
        """
        created_files: List[Path] = []
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save main integration file
        main_file = path / self.filename
        main_file.write_text(self.code)
        created_files.append(main_file)

        # Save usage example
        if self.usage_example:
            example_file = path / f"example_{self.filename}"
            example_file.write_text(self.usage_example)
            created_files.append(example_file)

        # Save tests
        if include_tests and self.tests:
            test_file = path / f"test_{self.filename}"
            test_file.write_text(self.tests)
            created_files.append(test_file)

        # Save requirements
        if self.dependencies:
            req_file = path / "requirements.txt"
            req_file.write_text("\n".join(self.dependencies))
            created_files.append(req_file)

        return created_files

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "code": self.code,
            "dependencies": self.dependencies,
            "usage_example": self.usage_example,
            "tests": self.tests,
            "metadata": self.metadata,
        }


class IntegrationGenerator:
    """
    Generate integration code from patterns using LLM.

    This generator creates production-ready integration code based on
    extracted patterns, including proper error handling, documentation,
    and comprehensive tests.

    Attributes:
        backend: LLM backend for code generation

    Example:
        ```python
        generator = IntegrationGenerator(backend)

        pattern = Pattern(
            name="Tool Registry",
            category="tools",
            description="Registry pattern for managing tools",
            ...
        )

        integration = await generator.generate(pattern, "agent_frameworks")
        await integration.save(Path("./integrations"))
        ```
    """

    GENERATION_SYSTEM_PROMPT = """You are an expert Python developer specializing in agent frameworks.
Generate production-quality code that:
1. Follows Python best practices and PEP 8
2. Uses proper type hints throughout
3. Includes comprehensive docstrings
4. Handles errors gracefully
5. Is well-tested and maintainable
6. Uses async/await for I/O operations

Always include:
- Module-level docstring
- Class/function docstrings
- Inline comments for complex logic
- Type hints for all parameters and returns"""

    ADAPTER_SYSTEM_PROMPT = """You are an expert at creating framework adapters and bridges.
Generate an adapter that:
1. Provides a clean abstraction between frameworks
2. Handles differences in API conventions
3. Preserves functionality without data loss
4. Is easy to extend and maintain
5. Has proper error handling"""

    TEST_SYSTEM_PROMPT = """You are an expert at writing comprehensive tests.
Generate tests that:
1. Cover happy paths and edge cases
2. Use pytest conventions
3. Include fixtures for setup/teardown
4. Mock external dependencies
5. Have clear, descriptive test names
6. Use parameterized tests where appropriate"""

    def __init__(self, backend: 'LLMBackend'):
        """
        Initialize the integration generator.

        Args:
            backend: LLM backend for code generation
        """
        self.backend = backend

    async def generate(
        self,
        pattern: 'Pattern',
        target_framework: str = "agent_frameworks",
    ) -> IntegrationCode:
        """
        Generate integration code for a pattern.

        Creates a complete integration module based on the pattern,
        including the main code, usage example, and tests.

        Args:
            pattern: Pattern to generate integration for
            target_framework: Framework to integrate with

        Returns:
            Complete integration code
        """
        from ..backends.backend_base import LLMConfig

        # Generate main integration code
        code = await self._generate_main_code(pattern, target_framework)

        # Generate usage example
        example = await self._generate_example(pattern, code)

        # Generate tests
        tests = await self.generate_tests_for_code(code, pattern.name)

        # Analyze dependencies
        dependencies = self._extract_dependencies(code)

        # Generate filename from pattern name
        filename = self._pattern_to_filename(pattern.name)

        return IntegrationCode(
            filename=filename,
            code=code,
            dependencies=dependencies,
            usage_example=example,
            tests=tests,
            metadata={
                "source_pattern": pattern.name,
                "source_framework": pattern.source_framework,
                "target_framework": target_framework,
                "generated_at": datetime.now().isoformat(),
            },
        )

    async def generate_adapter(
        self,
        source_framework: str,
        target_framework: str,
        source_patterns: Optional[List['Pattern']] = None,
    ) -> IntegrationCode:
        """
        Generate an adapter between two frameworks.

        Creates code that bridges between the source and target frameworks,
        allowing code written for one to work with the other.

        Args:
            source_framework: Framework to adapt from
            target_framework: Framework to adapt to
            source_patterns: Patterns from source framework

        Returns:
            Adapter code
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=4000,
            system=self.ADAPTER_SYSTEM_PROMPT,
        )

        patterns_context = ""
        if source_patterns:
            patterns_context = "\n\nSource framework patterns:\n"
            for p in source_patterns[:10]:  # Limit to 10 patterns
                patterns_context += f"- {p.name}: {p.description}\n"

        prompt = f"""Generate an adapter from {source_framework} to {target_framework}.

The adapter should:
1. Allow {source_framework} components to work with {target_framework}
2. Translate API conventions between frameworks
3. Handle any data format differences
4. Provide clear error messages for incompatibilities
{patterns_context}

Generate a complete, production-ready adapter module."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        code = self._extract_code(response.content)

        # Generate tests for adapter
        tests = await self.generate_tests_for_code(
            code, f"{source_framework}_to_{target_framework}_adapter"
        )

        return IntegrationCode(
            filename=f"{source_framework.lower()}_adapter.py",
            code=code,
            dependencies=self._extract_dependencies(code),
            usage_example=self._generate_adapter_example(source_framework, target_framework),
            tests=tests,
            metadata={
                "adapter_type": "framework",
                "source": source_framework,
                "target": target_framework,
            },
        )

    async def generate_tests(
        self,
        integration: IntegrationCode,
    ) -> str:
        """
        Generate tests for an integration.

        Creates comprehensive pytest tests for the integration code.

        Args:
            integration: Integration to generate tests for

        Returns:
            Test code as string
        """
        return await self.generate_tests_for_code(
            integration.code,
            integration.filename.replace(".py", ""),
        )

    async def generate_tests_for_code(
        self,
        code: str,
        module_name: str,
    ) -> str:
        """
        Generate tests for arbitrary code.

        Args:
            code: Code to generate tests for
            module_name: Name of the module being tested

        Returns:
            Test code as string
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=3000,
            system=self.TEST_SYSTEM_PROMPT,
        )

        prompt = f"""Generate comprehensive pytest tests for this module:

Module name: {module_name}

```python
{code[:6000]}  # Limit code length
```

Include:
1. Unit tests for each public function/method
2. Integration tests for key workflows
3. Edge case tests
4. Error handling tests
5. Fixtures for common setup

Use pytest conventions and async testing where appropriate."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        return self._extract_code(response.content)

    async def _generate_main_code(
        self,
        pattern: 'Pattern',
        target_framework: str,
    ) -> str:
        """Generate main integration code."""
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=4000,
            system=self.GENERATION_SYSTEM_PROMPT,
        )

        snippets_context = ""
        if pattern.code_snippets:
            snippets_context = "\n\nReference code snippets:\n```python\n"
            snippets_context += "\n\n".join(pattern.code_snippets[:3])
            snippets_context += "\n```"

        prompt = f"""Generate an integration module for the {target_framework} library based on this pattern:

Pattern Name: {pattern.name}
Category: {pattern.category}
Description: {pattern.description}
Implementation Notes: {pattern.implementation_notes}
Source Framework: {pattern.source_framework}
{snippets_context}

Create a complete, production-ready module that:
1. Implements the pattern for {target_framework}
2. Follows the existing codebase conventions
3. Is fully async-compatible
4. Has comprehensive error handling
5. Includes detailed docstrings

Generate only the Python code, no explanations."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        return self._extract_code(response.content)

    async def _generate_example(
        self,
        pattern: 'Pattern',
        code: str,
    ) -> str:
        """Generate usage example."""
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.3,
            max_tokens=1500,
        )

        prompt = f"""Generate a usage example for this integration:

Pattern: {pattern.name}

Integration code (first 2000 chars):
```python
{code[:2000]}
```

Create a clear, runnable example that demonstrates:
1. Basic usage
2. Configuration options
3. Common patterns
4. Error handling

Use async main function if needed."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        return self._extract_code(response.content)

    def _extract_code(self, content: str) -> str:
        """Extract code from LLM response."""
        # Try to find code blocks
        code_block_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)

        if matches:
            # Join all code blocks
            return "\n\n".join(matches)

        # Try to find any code block
        any_block_pattern = r'```\n?(.*?)```'
        matches = re.findall(any_block_pattern, content, re.DOTALL)

        if matches:
            return "\n\n".join(matches)

        # Return content as-is if no code blocks found
        return content

    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract package dependencies from code."""
        dependencies = set()

        # Standard library modules to exclude
        stdlib = {
            "os", "sys", "json", "re", "ast", "abc", "typing", "dataclasses",
            "asyncio", "pathlib", "datetime", "collections", "functools",
            "itertools", "enum", "logging", "unittest", "copy", "uuid",
        }

        # Find imports
        import_pattern = r'^(?:from\s+(\w+)|import\s+(\w+))'

        for line in code.split("\n"):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                if module and module not in stdlib:
                    # Convert module name to package name
                    package = module.split(".")[0]
                    dependencies.add(package)

        # Common package name mappings
        mappings = {
            "anthropic": "anthropic",
            "openai": "openai",
            "langchain": "langchain",
            "pydantic": "pydantic",
            "pytest": "pytest",
            "httpx": "httpx",
            "aiohttp": "aiohttp",
        }

        result = []
        for dep in sorted(dependencies):
            if dep in mappings:
                result.append(mappings[dep])
            else:
                result.append(dep)

        return result

    def _pattern_to_filename(self, pattern_name: str) -> str:
        """Convert pattern name to valid filename."""
        # Remove special characters and convert to snake_case
        name = re.sub(r'[^\w\s]', '', pattern_name)
        name = re.sub(r'\s+', '_', name)
        name = name.lower()

        return f"{name}.py"

    def _generate_adapter_example(
        self,
        source_framework: str,
        target_framework: str,
    ) -> str:
        """Generate a basic adapter usage example."""
        return f'''"""
Example usage of {source_framework} to {target_framework} adapter.
"""

import asyncio
from {source_framework.lower()}_adapter import {source_framework}Adapter


async def main():
    # Create adapter instance
    adapter = {source_framework}Adapter()

    # Use {source_framework} components with {target_framework}
    # ... your code here ...

    print("Adapter example completed!")


if __name__ == "__main__":
    asyncio.run(main())
'''
