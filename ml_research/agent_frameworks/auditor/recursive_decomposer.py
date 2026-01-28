"""
Recursive Decomposition Strategy.

This module provides tools for breaking large codebases into
analyzable chunks that can be processed by LLMs without exceeding
context limits.
"""

import ast
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig
    from .auditor_agent import FrameworkSource, FrameworkAnalysis


@dataclass
class CodeChunk:
    """
    A chunk of code extracted for analysis.

    Attributes:
        content: The code content
        file_path: Path to the source file
        chunk_type: Type of chunk (module, class, function)
        dependencies: List of dependencies (imports, base classes)
        name: Name of the chunk (module/class/function name)
        docstring: Extracted docstring if available
        line_start: Starting line number in original file
        line_end: Ending line number in original file
        metadata: Additional metadata
    """
    content: str
    file_path: str
    chunk_type: str  # module, class, function, method
    dependencies: List[str] = field(default_factory=list)
    name: str = ""
    docstring: str = ""
    line_start: int = 0
    line_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)

    @property
    def is_large(self) -> bool:
        """Check if chunk exceeds typical LLM context limits."""
        return len(self.content) > 4000

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "dependencies": self.dependencies,
            "name": self.name,
            "docstring": self.docstring,
            "line_range": [self.line_start, self.line_end],
            "metadata": self.metadata,
        }


class RecursiveDecomposer:
    """
    Break complex codebases into analyzable chunks.

    This class provides methods for decomposing large codebases into
    smaller chunks that can be individually analyzed by LLMs while
    preserving important context and dependency information.

    Attributes:
        backend: LLM backend for analysis
        max_chunk_size: Maximum size for individual chunks

    Example:
        ```python
        decomposer = RecursiveDecomposer(backend, max_chunk_size=4000)

        source = FrameworkSource(path=Path("./my_framework"))
        chunks = await decomposer.decompose(source)

        analyses = await asyncio.gather(*[
            decomposer.analyze_chunk(chunk)
            for chunk in chunks
        ])

        final_analysis = await decomposer.synthesize(analyses)
        ```
    """

    ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing code.
For the given code chunk, provide a detailed analysis including:
1. Purpose and functionality
2. Key patterns and abstractions used
3. Dependencies and integrations
4. Strengths and potential improvements
5. How it fits into a larger system

Be concise but thorough. Focus on architectural insights."""

    SYNTHESIS_SYSTEM_PROMPT = """You are synthesizing multiple code analyses into a coherent framework analysis.
Given individual chunk analyses, create a unified view that:
1. Identifies the overall architecture
2. Lists major components and their roles
3. Describes the design patterns used
4. Highlights strengths and weaknesses
5. Identifies integration points

Provide a structured, comprehensive analysis."""

    # File extensions to analyze
    CODE_EXTENSIONS = {".py", ".pyi"}

    # Directories to skip
    SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".eggs", "*.egg-info"}

    def __init__(
        self,
        backend: 'LLMBackend',
        max_chunk_size: int = 4000,
    ):
        """
        Initialize the decomposer.

        Args:
            backend: LLM backend for analysis
            max_chunk_size: Maximum character size for chunks
        """
        self.backend = backend
        self.max_chunk_size = max_chunk_size

    async def decompose(
        self,
        source: 'FrameworkSource',
    ) -> List[CodeChunk]:
        """
        Decompose a framework source into analyzable chunks.

        Recursively processes the source directory/files to create
        appropriately-sized chunks while preserving structure.

        Args:
            source: The framework source to decompose

        Returns:
            List of code chunks
        """
        chunks: List[CodeChunk] = []

        if source.path and source.path.exists():
            if source.path.is_file():
                chunks.extend(self._decompose_file(source.path))
            else:
                chunks.extend(self._decompose_directory(source.path))

        return chunks

    async def analyze_chunk(
        self,
        chunk: CodeChunk,
    ) -> Dict[str, Any]:
        """
        Analyze a single code chunk using the LLM.

        Args:
            chunk: The code chunk to analyze

        Returns:
            Analysis results as a dictionary
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=2000,
            system=self.ANALYSIS_SYSTEM_PROMPT,
        )

        context = f"""
File: {chunk.file_path}
Type: {chunk.chunk_type}
Name: {chunk.name}
Dependencies: {', '.join(chunk.dependencies)}
"""

        prompt = f"""Analyze this code chunk:

{context}

```python
{chunk.content}
```

Provide a structured analysis."""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.backend.complete(messages, config)

            return {
                "chunk_name": chunk.name,
                "chunk_type": chunk.chunk_type,
                "file_path": chunk.file_path,
                "analysis": response.content,
                "dependencies": chunk.dependencies,
                "docstring": chunk.docstring,
            }
        except Exception as e:
            return {
                "chunk_name": chunk.name,
                "chunk_type": chunk.chunk_type,
                "file_path": chunk.file_path,
                "analysis": f"Analysis failed: {str(e)}",
                "dependencies": chunk.dependencies,
                "error": str(e),
            }

    async def synthesize(
        self,
        chunk_analyses: List[Dict[str, Any]],
    ) -> 'FrameworkAnalysis':
        """
        Synthesize chunk analyses into a complete framework analysis.

        Combines individual chunk analyses into a unified framework
        analysis that captures the overall architecture and patterns.

        Args:
            chunk_analyses: List of individual chunk analyses

        Returns:
            Complete framework analysis
        """
        from ..backends.backend_base import LLMConfig
        from .auditor_agent import FrameworkAnalysis

        # Prepare summary of analyses
        analysis_summaries = []
        all_dependencies: Set[str] = set()
        code_files: Dict[str, str] = {}

        for analysis in chunk_analyses:
            summary = f"""
### {analysis['chunk_type']}: {analysis['chunk_name']}
File: {analysis['file_path']}
Dependencies: {', '.join(analysis['dependencies'])}

{analysis['analysis'][:1000]}  # Truncate long analyses
"""
            analysis_summaries.append(summary)
            all_dependencies.update(analysis['dependencies'])

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=4000,
            system=self.SYNTHESIS_SYSTEM_PROMPT,
        )

        prompt = f"""Synthesize these individual code analyses into a complete framework analysis:

{chr(10).join(analysis_summaries[:50])}  # Limit to 50 chunks

Provide:
1. Overall architecture summary
2. List of major components with their roles
3. Design patterns used
4. Strengths of the framework
5. Weaknesses or limitations
6. Integration points for extending the framework

Format your response with clear sections."""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.backend.complete(messages, config)

            # Parse the response into structured analysis
            return self._parse_synthesis_response(
                response.content,
                chunk_analyses,
                all_dependencies,
            )
        except Exception as e:
            # Return a basic analysis on error
            return FrameworkAnalysis(
                name="Unknown",
                architecture={"error": str(e)},
                components=[{"name": a["chunk_name"], "role": a.get("docstring", "")}
                           for a in chunk_analyses],
                patterns=[],
                strengths=[],
                weaknesses=[f"Analysis incomplete due to error: {e}"],
                integration_points=[],
                metadata={"chunk_analyses": chunk_analyses},
            )

    def _decompose_directory(
        self,
        directory: Path,
    ) -> List[CodeChunk]:
        """Decompose all Python files in a directory."""
        chunks: List[CodeChunk] = []

        for file_path in self._iter_python_files(directory):
            chunks.extend(self._decompose_file(file_path))

        return chunks

    def _decompose_file(
        self,
        file_path: Path,
    ) -> List[CodeChunk]:
        """Decompose a single Python file into chunks."""
        chunks: List[CodeChunk] = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IOError):
            return chunks

        # Try to parse as AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # If parsing fails, return whole file as a chunk
            return [CodeChunk(
                content=content[:self.max_chunk_size],
                file_path=str(file_path),
                chunk_type="module",
                name=file_path.stem,
            )]

        # Extract module-level docstring and imports
        module_docstring = ast.get_docstring(tree) or ""
        imports = self._extract_imports(tree)

        # If file is small enough, return as single chunk
        if len(content) <= self.max_chunk_size:
            return [CodeChunk(
                content=content,
                file_path=str(file_path),
                chunk_type="module",
                name=file_path.stem,
                docstring=module_docstring,
                dependencies=imports,
            )]

        # Decompose into smaller chunks
        lines = content.split("\n")

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._decompose_class(node, lines, file_path, imports))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._extract_function_chunk(node, lines, file_path, imports)
                if chunk:
                    chunks.append(chunk)

        # If no chunks extracted, return the whole file in parts
        if not chunks:
            chunks = self._split_by_size(content, file_path)

        return chunks

    def _decompose_class(
        self,
        class_node: ast.ClassDef,
        lines: List[str],
        file_path: Path,
        imports: List[str],
    ) -> List[CodeChunk]:
        """Decompose a class into chunks."""
        chunks: List[CodeChunk] = []

        class_content = "\n".join(
            lines[class_node.lineno - 1:class_node.end_lineno]
        )
        docstring = ast.get_docstring(class_node) or ""

        # Extract base classes
        bases = [self._get_name(base) for base in class_node.bases]
        bases = [b for b in bases if b]

        # If class is small enough, return as single chunk
        if len(class_content) <= self.max_chunk_size:
            return [CodeChunk(
                content=class_content,
                file_path=str(file_path),
                chunk_type="class",
                name=class_node.name,
                docstring=docstring,
                dependencies=imports + bases,
                line_start=class_node.lineno,
                line_end=class_node.end_lineno or class_node.lineno,
            )]

        # Decompose class into methods
        class_header = self._extract_class_header(class_node, lines)
        chunks.append(CodeChunk(
            content=class_header,
            file_path=str(file_path),
            chunk_type="class_header",
            name=f"{class_node.name}.__header__",
            docstring=docstring,
            dependencies=imports + bases,
            line_start=class_node.lineno,
        ))

        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_content = "\n".join(
                    lines[node.lineno - 1:node.end_lineno]
                )
                method_docstring = ast.get_docstring(node) or ""

                chunks.append(CodeChunk(
                    content=method_content,
                    file_path=str(file_path),
                    chunk_type="method",
                    name=f"{class_node.name}.{node.name}",
                    docstring=method_docstring,
                    dependencies=imports,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                ))

        return chunks

    def _extract_function_chunk(
        self,
        func_node: ast.FunctionDef,
        lines: List[str],
        file_path: Path,
        imports: List[str],
    ) -> Optional[CodeChunk]:
        """Extract a function as a chunk."""
        func_content = "\n".join(
            lines[func_node.lineno - 1:func_node.end_lineno]
        )
        docstring = ast.get_docstring(func_node) or ""

        if len(func_content) > self.max_chunk_size:
            func_content = func_content[:self.max_chunk_size]

        return CodeChunk(
            content=func_content,
            file_path=str(file_path),
            chunk_type="function",
            name=func_node.name,
            docstring=docstring,
            dependencies=imports,
            line_start=func_node.lineno,
            line_end=func_node.end_lineno or func_node.lineno,
        )

    def _extract_class_header(
        self,
        class_node: ast.ClassDef,
        lines: List[str],
    ) -> str:
        """Extract class definition and docstring."""
        header_lines = []

        # Add decorators
        for decorator in class_node.decorator_list:
            if hasattr(decorator, "lineno"):
                header_lines.append(lines[decorator.lineno - 1])

        # Add class definition line
        header_lines.append(lines[class_node.lineno - 1])

        # Add docstring if present
        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    header_lines.extend(lines[node.lineno - 1:node.end_lineno])
                    break

        # Add class attributes (not methods)
        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                header_lines.extend(lines[node.lineno - 1:node.end_lineno or node.lineno])

        return "\n".join(header_lines)

    def _extract_imports(
        self,
        tree: ast.AST,
    ) -> List[str]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def _split_by_size(
        self,
        content: str,
        file_path: Path,
    ) -> List[CodeChunk]:
        """Split content by size when AST decomposition isn't possible."""
        chunks = []

        for i in range(0, len(content), self.max_chunk_size):
            chunk_content = content[i:i + self.max_chunk_size]
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=str(file_path),
                chunk_type="fragment",
                name=f"{file_path.stem}_part_{i // self.max_chunk_size}",
            ))

        return chunks

    def _iter_python_files(
        self,
        directory: Path,
    ) -> List[Path]:
        """Iterate over Python files in a directory."""
        files = []

        for item in directory.rglob("*"):
            # Skip unwanted directories
            if any(skip in item.parts for skip in self.SKIP_DIRS):
                continue

            if item.is_file() and item.suffix in self.CODE_EXTENSIONS:
                files.append(item)

        return sorted(files)

    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _parse_synthesis_response(
        self,
        content: str,
        chunk_analyses: List[Dict[str, Any]],
        dependencies: Set[str],
    ) -> 'FrameworkAnalysis':
        """Parse synthesis response into FrameworkAnalysis."""
        from .auditor_agent import FrameworkAnalysis

        lines = content.split("\n")

        architecture = {}
        components = []
        patterns = []
        strengths = []
        weaknesses = []
        integration_points = []

        current_section = None

        for line in lines:
            line = line.strip()

            # Detect sections
            line_lower = line.lower()
            if "architecture" in line_lower and line.startswith("#"):
                current_section = "architecture"
            elif "component" in line_lower and line.startswith("#"):
                current_section = "components"
            elif "pattern" in line_lower and line.startswith("#"):
                current_section = "patterns"
            elif "strength" in line_lower and line.startswith("#"):
                current_section = "strengths"
            elif "weakness" in line_lower or "limitation" in line_lower:
                if line.startswith("#"):
                    current_section = "weaknesses"
            elif "integration" in line_lower and line.startswith("#"):
                current_section = "integration_points"
            elif line.startswith("- ") or line.startswith("* "):
                item = line[2:].strip()
                if current_section == "patterns":
                    patterns.append(item)
                elif current_section == "strengths":
                    strengths.append(item)
                elif current_section == "weaknesses":
                    weaknesses.append(item)
                elif current_section == "integration_points":
                    integration_points.append(item)
                elif current_section == "components":
                    components.append({"name": item, "role": ""})

        # Extract components from chunk analyses if none found
        if not components:
            for analysis in chunk_analyses:
                if analysis["chunk_type"] in ("class", "module"):
                    components.append({
                        "name": analysis["chunk_name"],
                        "role": analysis.get("docstring", "")[:100],
                    })

        return FrameworkAnalysis(
            name="",  # Will be set by caller
            architecture=architecture if architecture else {"description": content[:500]},
            components=components,
            patterns=patterns,
            strengths=strengths,
            weaknesses=weaknesses,
            integration_points=integration_points,
            metadata={
                "chunk_count": len(chunk_analyses),
                "dependencies": list(dependencies),
                "code_files": {a["file_path"]: "" for a in chunk_analyses},
            },
        )
