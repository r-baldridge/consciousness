"""
LLM-based Pattern Extraction.

This module provides pattern extraction capabilities using LLM analysis
to identify reusable patterns from framework code and documentation.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig
    from .auditor_agent import Pattern, FrameworkAnalysis


class PatternExtractor:
    """
    Extract reusable patterns from framework code using LLM.

    This extractor uses LLM analysis to identify architectural patterns,
    design patterns, and implementation patterns that can be reused
    across different agent framework implementations.

    Attributes:
        backend: LLM backend for analysis
        PATTERN_CATEGORIES: Supported pattern categories

    Example:
        ```python
        extractor = PatternExtractor(backend)
        patterns = await extractor.extract_from_code(
            code="class Agent: ...",
            context="Main agent orchestration"
        )
        ```
    """

    PATTERN_CATEGORIES = [
        "orchestration",  # Agent coordination patterns
        "memory",         # State and memory management
        "tools",          # Tool integration patterns
        "execution",      # Execution flow patterns
        "human_loop",     # Human-in-the-loop patterns
        "context",        # Context management
        "planning",       # Planning and decomposition
        "reflection",     # Self-reflection patterns
        "retrieval",      # RAG and retrieval patterns
        "multi_agent",    # Multi-agent coordination
    ]

    EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying reusable software patterns in agent frameworks.

For each pattern you identify, provide:
1. name: A clear, descriptive name (e.g., "ReACT Loop", "Tool Registry")
2. category: One of [orchestration, memory, tools, execution, human_loop, context, planning, reflection, retrieval, multi_agent]
3. description: What problem this pattern solves
4. implementation_notes: How to implement this pattern
5. code_snippets: Relevant code examples (if any)
6. dependencies: Required dependencies

Return your response as a JSON array of pattern objects."""

    REUSABILITY_PROMPT = """You are an expert at evaluating code reusability.

Given a software pattern, score its reusability from 0.0 to 1.0 based on:
- Generality: Can it be applied to many situations?
- Independence: Does it have minimal dependencies?
- Clarity: Is the implementation clear and well-documented?
- Testability: Can it be easily tested?
- Modularity: Is it self-contained?

Return only a single float number between 0.0 and 1.0."""

    def __init__(self, backend: 'LLMBackend'):
        """
        Initialize the pattern extractor.

        Args:
            backend: LLM backend for analysis
        """
        self.backend = backend

    async def extract_from_code(
        self,
        code: str,
        context: str = "",
    ) -> List['Pattern']:
        """
        Extract patterns from source code.

        Analyzes the provided code and extracts reusable patterns
        using LLM analysis.

        Args:
            code: Source code to analyze
            context: Additional context about the code

        Returns:
            List of extracted patterns
        """
        from ..backends.backend_base import LLMConfig
        from .auditor_agent import Pattern

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=4000,
            system=self.EXTRACTION_SYSTEM_PROMPT,
        )

        prompt = f"""Analyze this code and extract reusable patterns:

Context: {context}

Code:
```python
{code}
```

Return patterns as a JSON array."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        return self._parse_patterns(response.content, "")

    async def extract_from_docs(
        self,
        docs: str,
        framework_name: str = "",
    ) -> List['Pattern']:
        """
        Extract patterns from documentation.

        Analyzes documentation to identify architectural patterns
        and design decisions that can be extracted as reusable patterns.

        Args:
            docs: Documentation text to analyze
            framework_name: Name of the framework

        Returns:
            List of extracted patterns
        """
        from ..backends.backend_base import LLMConfig
        from .auditor_agent import Pattern

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=4000,
            system=self.EXTRACTION_SYSTEM_PROMPT,
        )

        prompt = f"""Analyze this documentation and extract architectural patterns:

Framework: {framework_name}

Documentation:
{docs}

Return patterns as a JSON array."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        return self._parse_patterns(response.content, framework_name)

    async def extract_from_analysis(
        self,
        analysis: 'FrameworkAnalysis',
    ) -> List['Pattern']:
        """
        Extract patterns from a complete framework analysis.

        Uses the comprehensive analysis to identify patterns that
        might span multiple components and aspects of the framework.

        Args:
            analysis: Complete framework analysis

        Returns:
            List of extracted patterns
        """
        from ..backends.backend_base import LLMConfig
        from .auditor_agent import Pattern

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=4000,
            system=self.EXTRACTION_SYSTEM_PROMPT,
        )

        analysis_context = json.dumps(analysis.to_dict(), indent=2)

        prompt = f"""Based on this framework analysis, extract reusable patterns:

{analysis_context}

Focus on patterns that:
1. Solve common agent framework problems
2. Can be adapted to other frameworks
3. Have clear implementation paths
4. Are well-demonstrated in the analyzed code

Return patterns as a JSON array."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        patterns = self._parse_patterns(response.content, analysis.name)

        # Add framework-specific patterns from the analysis
        for pattern_name in analysis.patterns:
            if not any(p.name == pattern_name for p in patterns):
                patterns.append(Pattern(
                    name=pattern_name,
                    category=self._infer_category(pattern_name),
                    description=f"Pattern identified in {analysis.name}",
                    implementation_notes="See framework source for implementation details",
                    source_framework=analysis.name,
                ))

        return patterns

    async def categorize_pattern(
        self,
        pattern: 'Pattern',
    ) -> str:
        """
        Categorize a pattern into one of the predefined categories.

        Uses LLM analysis to determine the most appropriate category
        for a given pattern.

        Args:
            pattern: Pattern to categorize

        Returns:
            Category string
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.1,
            max_tokens=100,
        )

        prompt = f"""Categorize this pattern into one of these categories:
{', '.join(self.PATTERN_CATEGORIES)}

Pattern:
Name: {pattern.name}
Description: {pattern.description}

Return only the category name, nothing else."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        category = response.content.strip().lower()

        # Validate category
        if category in self.PATTERN_CATEGORIES:
            return category

        # Try to match partial category names
        for cat in self.PATTERN_CATEGORIES:
            if cat in category or category in cat:
                return cat

        return "orchestration"  # Default category

    async def score_reusability(
        self,
        pattern: 'Pattern',
    ) -> float:
        """
        Score how reusable a pattern is.

        Evaluates a pattern on multiple criteria to determine
        how easily it can be reused in other projects.

        Args:
            pattern: Pattern to score

        Returns:
            Reusability score from 0.0 to 1.0
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.1,
            max_tokens=50,
            system=self.REUSABILITY_PROMPT,
        )

        pattern_info = json.dumps(pattern.to_dict(), indent=2)

        messages = [{
            "role": "user",
            "content": f"Score the reusability of this pattern:\n{pattern_info}",
        }]

        response = await self.backend.complete(messages, config)

        # Extract float from response
        try:
            # Find any float in the response
            match = re.search(r'(\d+\.?\d*)', response.content)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
        except (ValueError, AttributeError):
            pass

        # Default score based on heuristics
        return self._heuristic_reusability_score(pattern)

    def _parse_patterns(
        self,
        response_content: str,
        framework_name: str,
    ) -> List['Pattern']:
        """Parse patterns from LLM response."""
        from .auditor_agent import Pattern

        patterns = []

        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response_content)
        if json_match:
            try:
                pattern_data = json.loads(json_match.group())
                for item in pattern_data:
                    if isinstance(item, dict):
                        patterns.append(Pattern(
                            name=item.get("name", "Unknown"),
                            category=item.get("category", "orchestration"),
                            description=item.get("description", ""),
                            implementation_notes=item.get("implementation_notes", ""),
                            code_snippets=item.get("code_snippets", []),
                            source_framework=framework_name,
                            dependencies=item.get("dependencies", []),
                        ))
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse structured text
        if not patterns:
            patterns = self._parse_text_patterns(response_content, framework_name)

        return patterns

    def _parse_text_patterns(
        self,
        text: str,
        framework_name: str,
    ) -> List['Pattern']:
        """Parse patterns from structured text when JSON parsing fails."""
        from .auditor_agent import Pattern

        patterns = []
        current_pattern: Dict[str, Any] = {}

        lines = text.split("\n")
        for line in lines:
            line = line.strip()

            # Look for pattern headers
            if line.startswith("# ") or line.startswith("## "):
                if current_pattern.get("name"):
                    patterns.append(Pattern(
                        name=current_pattern.get("name", ""),
                        category=current_pattern.get("category", "orchestration"),
                        description=current_pattern.get("description", ""),
                        implementation_notes=current_pattern.get("implementation_notes", ""),
                        source_framework=framework_name,
                    ))
                current_pattern = {"name": line.lstrip("# ")}

            # Look for key-value pairs
            elif ":" in line and current_pattern:
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if key in ["category", "description", "implementation_notes"]:
                    current_pattern[key] = value

        # Don't forget the last pattern
        if current_pattern.get("name"):
            patterns.append(Pattern(
                name=current_pattern.get("name", ""),
                category=current_pattern.get("category", "orchestration"),
                description=current_pattern.get("description", ""),
                implementation_notes=current_pattern.get("implementation_notes", ""),
                source_framework=framework_name,
            ))

        return patterns

    def _infer_category(self, pattern_name: str) -> str:
        """Infer category from pattern name using keywords."""
        name_lower = pattern_name.lower()

        category_keywords = {
            "orchestration": ["orchestrat", "coordinat", "workflow", "pipeline", "chain"],
            "memory": ["memory", "state", "cache", "store", "persist"],
            "tools": ["tool", "function", "action", "capability"],
            "execution": ["execut", "run", "invoke", "call"],
            "human_loop": ["human", "approval", "confirm", "review"],
            "context": ["context", "window", "token"],
            "planning": ["plan", "decompos", "task", "goal"],
            "reflection": ["reflect", "reason", "think", "analyze"],
            "retrieval": ["retriev", "rag", "search", "embed"],
            "multi_agent": ["multi", "agent", "swarm", "team"],
        }

        for category, keywords in category_keywords.items():
            if any(kw in name_lower for kw in keywords):
                return category

        return "orchestration"

    def _heuristic_reusability_score(self, pattern: 'Pattern') -> float:
        """Calculate a heuristic reusability score."""
        score = 0.5  # Base score

        # Has description
        if pattern.description:
            score += 0.1

        # Has implementation notes
        if pattern.implementation_notes:
            score += 0.1

        # Has code snippets
        if pattern.code_snippets:
            score += 0.1

        # Few dependencies is better
        if len(pattern.dependencies) == 0:
            score += 0.1
        elif len(pattern.dependencies) <= 2:
            score += 0.05

        # Common categories tend to be more reusable
        common_categories = ["tools", "memory", "orchestration"]
        if pattern.category in common_categories:
            score += 0.1

        return min(1.0, score)
