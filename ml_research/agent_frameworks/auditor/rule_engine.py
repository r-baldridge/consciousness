"""
Rule-based AST Pattern Matching.

This module provides rule-based pattern extraction using AST analysis
and configurable rules for identifying common patterns in code.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .auditor_agent import Pattern, FrameworkAnalysis


@dataclass
class ASTRule:
    """
    Rule for AST-based pattern matching.

    Attributes:
        name: Rule name (becomes pattern name)
        pattern: AST pattern to match (decorator name, class pattern, etc.)
        category: Pattern category
        description: What this pattern represents
        extract: Function to extract pattern details from matched nodes
        match_type: Type of AST matching (decorator, class, function, import)
    """
    name: str
    pattern: str  # Regex or AST selector
    category: str
    description: str = ""
    extract: Optional[Callable[[ast.AST], Dict[str, Any]]] = None
    match_type: str = "class"  # class, function, decorator, import

    def matches(self, node: ast.AST) -> bool:
        """Check if this rule matches the given AST node."""
        if self.match_type == "class":
            if isinstance(node, ast.ClassDef):
                return bool(re.search(self.pattern, node.name, re.IGNORECASE))

        elif self.match_type == "function":
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                return bool(re.search(self.pattern, node.name, re.IGNORECASE))

        elif self.match_type == "decorator":
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    dec_name = self._get_decorator_name(decorator)
                    if dec_name and re.search(self.pattern, dec_name, re.IGNORECASE):
                        return True

        elif self.match_type == "import":
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if re.search(self.pattern, alias.name, re.IGNORECASE):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and re.search(self.pattern, node.module, re.IGNORECASE):
                    return True

        elif self.match_type == "inheritance":
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name and re.search(self.pattern, base_name, re.IGNORECASE):
                        return True

        return False

    def _get_decorator_name(self, decorator: ast.AST) -> Optional[str]:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return None

    def _get_name(self, node: ast.AST) -> Optional[str]:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None


class RuleBasedExtractor:
    """
    Rule-based pattern extraction using AST matching.

    This extractor uses configurable rules to identify patterns
    in Python code through AST analysis. It complements the LLM-based
    extraction by providing deterministic pattern matching.

    Attributes:
        rules: List of active extraction rules
        BUILTIN_RULES: Default rules for common patterns

    Example:
        ```python
        extractor = RuleBasedExtractor()

        # Add custom rule
        extractor.add_rule(ASTRule(
            name="Custom Agent Pattern",
            pattern="MyAgent.*",
            category="orchestration",
            match_type="class"
        ))

        patterns = extractor.extract(analysis)
        ```
    """

    # Built-in rules for common agent framework patterns
    BUILTIN_RULES: List[ASTRule] = [
        # Class-based patterns
        ASTRule(
            name="Agent Base Class",
            pattern=r".*Agent.*",
            category="orchestration",
            description="Base agent class pattern for defining agent behavior",
            match_type="class",
        ),
        ASTRule(
            name="Tool Base Class",
            pattern=r".*Tool.*|.*Action.*|.*Capability.*",
            category="tools",
            description="Base class for tool definitions",
            match_type="class",
        ),
        ASTRule(
            name="Memory Store",
            pattern=r".*Memory.*|.*Store.*|.*Cache.*|.*State.*",
            category="memory",
            description="Memory or state storage pattern",
            match_type="class",
        ),
        ASTRule(
            name="Executor Pattern",
            pattern=r".*Executor.*|.*Runner.*|.*Engine.*",
            category="execution",
            description="Execution engine pattern",
            match_type="class",
        ),
        ASTRule(
            name="Chain Pattern",
            pattern=r".*Chain.*|.*Pipeline.*|.*Workflow.*",
            category="orchestration",
            description="Sequential chain or pipeline pattern",
            match_type="class",
        ),
        ASTRule(
            name="Prompt Template",
            pattern=r".*Prompt.*|.*Template.*",
            category="context",
            description="Prompt or template management pattern",
            match_type="class",
        ),
        ASTRule(
            name="Retriever Pattern",
            pattern=r".*Retriever.*|.*Embedder.*|.*VectorStore.*",
            category="retrieval",
            description="RAG/retrieval pattern",
            match_type="class",
        ),
        ASTRule(
            name="Planner Pattern",
            pattern=r".*Planner.*|.*Decomposer.*|.*TaskManager.*",
            category="planning",
            description="Planning and task decomposition pattern",
            match_type="class",
        ),

        # Decorator patterns
        ASTRule(
            name="Tool Decorator",
            pattern=r"tool|action|capability|function_tool",
            category="tools",
            description="Decorator for defining tools",
            match_type="decorator",
        ),
        ASTRule(
            name="Agent Decorator",
            pattern=r"agent|assistant|bot",
            category="orchestration",
            description="Decorator for defining agents",
            match_type="decorator",
        ),
        ASTRule(
            name="Step Decorator",
            pattern=r"step|stage|phase|task",
            category="execution",
            description="Decorator for workflow steps",
            match_type="decorator",
        ),

        # Inheritance patterns
        ASTRule(
            name="ABC Implementation",
            pattern=r"ABC|Abstract.*|Base.*",
            category="orchestration",
            description="Abstract base class implementation",
            match_type="inheritance",
        ),
        ASTRule(
            name="LLM Backend",
            pattern=r".*Backend.*|.*Provider.*|.*Client.*",
            category="execution",
            description="LLM backend/provider pattern",
            match_type="inheritance",
        ),

        # Function patterns
        ASTRule(
            name="Async Execute",
            pattern=r"execute|run|invoke|call|process",
            category="execution",
            description="Async execution function pattern",
            match_type="function",
        ),
        ASTRule(
            name="Hook Function",
            pattern=r"on_.*|before_.*|after_.*|handle_.*",
            category="orchestration",
            description="Hook/callback pattern for extensibility",
            match_type="function",
        ),
    ]

    def __init__(
        self,
        include_builtin: bool = True,
        custom_rules: Optional[List[ASTRule]] = None,
    ):
        """
        Initialize the rule-based extractor.

        Args:
            include_builtin: Whether to include built-in rules
            custom_rules: Additional custom rules to include
        """
        self.rules: List[ASTRule] = []

        if include_builtin:
            self.rules.extend(self.BUILTIN_RULES)

        if custom_rules:
            self.rules.extend(custom_rules)

    def add_rule(self, rule: ASTRule) -> None:
        """
        Add a new extraction rule.

        Args:
            rule: The rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule by name.

        Args:
            rule_name: Name of the rule to remove

        Returns:
            True if rule was found and removed
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False

    def extract(
        self,
        analysis: 'FrameworkAnalysis',
    ) -> List['Pattern']:
        """
        Extract patterns from a framework analysis.

        Uses AST analysis on code in the analysis metadata to
        identify patterns matching the configured rules.

        Args:
            analysis: The framework analysis to process

        Returns:
            List of extracted patterns
        """
        from .auditor_agent import Pattern

        patterns: List[Pattern] = []
        seen_patterns: set = set()

        # Get code content from analysis metadata
        code_files = analysis.metadata.get("code_files", {})

        for file_path, code_content in code_files.items():
            file_patterns = self.match_pattern_in_code(code_content, file_path)

            for pattern in file_patterns:
                if pattern.name not in seen_patterns:
                    pattern.source_framework = analysis.name
                    patterns.append(pattern)
                    seen_patterns.add(pattern.name)

        # Also check components for patterns
        for component in analysis.components:
            comp_name = component.get("name", "")
            comp_code = component.get("code", "")

            if comp_code:
                file_patterns = self.match_pattern_in_code(comp_code, comp_name)
                for pattern in file_patterns:
                    if pattern.name not in seen_patterns:
                        pattern.source_framework = analysis.name
                        patterns.append(pattern)
                        seen_patterns.add(pattern.name)

        return patterns

    def match_pattern(
        self,
        code: str,
        rule: ASTRule,
    ) -> List[Dict[str, Any]]:
        """
        Match a specific rule against code.

        Args:
            code: Source code to analyze
            rule: Rule to match

        Returns:
            List of matches with extracted details
        """
        matches = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return matches

        for node in ast.walk(tree):
            if rule.matches(node):
                match_info = {
                    "rule": rule.name,
                    "category": rule.category,
                    "line": getattr(node, "lineno", 0),
                }

                # Extract node name
                if hasattr(node, "name"):
                    match_info["name"] = node.name

                # Use custom extractor if provided
                if rule.extract:
                    try:
                        extracted = rule.extract(node)
                        match_info.update(extracted)
                    except Exception:
                        pass

                matches.append(match_info)

        return matches

    def match_pattern_in_code(
        self,
        code: str,
        source_file: str = "",
    ) -> List['Pattern']:
        """
        Match all rules against code and return patterns.

        Args:
            code: Source code to analyze
            source_file: Source file path for context

        Returns:
            List of patterns found
        """
        from .auditor_agent import Pattern

        patterns: List[Pattern] = []
        pattern_names: set = set()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return patterns

        for rule in self.rules:
            for node in ast.walk(tree):
                if rule.matches(node):
                    # Create pattern name from rule and matched node
                    node_name = getattr(node, "name", "")
                    pattern_name = f"{rule.name}: {node_name}" if node_name else rule.name

                    if pattern_name in pattern_names:
                        continue
                    pattern_names.add(pattern_name)

                    # Extract code snippet
                    snippet = self._extract_snippet(code, node)

                    patterns.append(Pattern(
                        name=pattern_name,
                        category=rule.category,
                        description=rule.description,
                        implementation_notes=f"Found in {source_file}" if source_file else "",
                        code_snippets=[snippet] if snippet else [],
                    ))

        return patterns

    def _extract_snippet(
        self,
        code: str,
        node: ast.AST,
        context_lines: int = 2,
    ) -> str:
        """Extract a code snippet around an AST node."""
        if not hasattr(node, "lineno"):
            return ""

        lines = code.split("\n")
        start = max(0, node.lineno - 1 - context_lines)
        end = min(len(lines), getattr(node, "end_lineno", node.lineno) + context_lines)

        return "\n".join(lines[start:end])

    def analyze_inheritance(
        self,
        code: str,
    ) -> Dict[str, List[str]]:
        """
        Analyze class inheritance hierarchy in code.

        Args:
            code: Source code to analyze

        Returns:
            Dictionary mapping class names to their base classes
        """
        hierarchy: Dict[str, List[str]] = {}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return hierarchy

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{self._get_full_name(base)}")
                hierarchy[node.name] = bases

        return hierarchy

    def find_decorators(
        self,
        code: str,
    ) -> Dict[str, List[str]]:
        """
        Find all decorated functions and classes.

        Args:
            code: Source code to analyze

        Returns:
            Dictionary mapping decorator names to decorated items
        """
        decorators: Dict[str, List[str]] = {}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return decorators

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    dec_name = self._get_full_name(decorator)
                    if dec_name:
                        if dec_name not in decorators:
                            decorators[dec_name] = []
                        decorators[dec_name].append(node.name)

        return decorators

    def _get_full_name(self, node: ast.AST) -> str:
        """Get full dotted name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_full_name(node.value)
            return f"{value_name}.{node.attr}" if value_name else node.attr
        elif isinstance(node, ast.Call):
            return self._get_full_name(node.func)
        return ""
