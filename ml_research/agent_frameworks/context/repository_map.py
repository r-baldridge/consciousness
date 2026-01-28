"""
Repository Map - AST-based Repository Context Generation.

Provides intelligent code structure analysis using AST parsing to generate
compressed context strings suitable for LLM consumption. Inspired by Aider's
repository mapping approach.

Supports multiple languages with graceful degradation when parsers are unavailable:
- Python: Uses built-in ast module (no dependencies)
- JavaScript/TypeScript: Uses tree-sitter (optional)
- Go, Rust: Uses tree-sitter (optional)

Example:
    repo_map = RepositoryMap(Path("/path/to/project"))
    await repo_map.build(extensions=[".py", ".js", ".ts"])

    # Get compressed context for LLM
    context = repo_map.get_context(max_tokens=8000)

    # Find related symbols
    related = repo_map.find_related("UserAuthentication")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum
import ast
import asyncio
import logging
import re
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class FunctionSignature:
    """Represents a function or method signature."""
    name: str
    file_path: Path
    line_number: int
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    def to_signature_string(self) -> str:
        """Generate a compact signature string."""
        prefix = "async " if self.is_async else ""
        params = ", ".join(self.parameters)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{prefix}def {self.name}({params}){ret}"

    def to_context_line(self, include_docstring: bool = False) -> str:
        """Generate a context line for LLM consumption."""
        sig = self.to_signature_string()
        if include_docstring and self.docstring:
            # First line of docstring only
            doc_line = self.docstring.split("\n")[0].strip()
            if len(doc_line) > 60:
                doc_line = doc_line[:57] + "..."
            return f"{sig}  # {doc_line}"
        return sig


@dataclass
class ClassDefinition:
    """Represents a class definition with its methods."""
    name: str
    file_path: Path
    line_number: int
    methods: List[FunctionSignature] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    class_variables: List[Tuple[str, Optional[str]]] = field(default_factory=list)

    def to_context_block(self, include_methods: bool = True, max_methods: int = 10) -> str:
        """Generate a context block for LLM consumption."""
        bases = f"({', '.join(self.base_classes)})" if self.base_classes else ""
        lines = [f"class {self.name}{bases}:"]

        if self.docstring:
            doc_line = self.docstring.split("\n")[0].strip()
            if len(doc_line) > 60:
                doc_line = doc_line[:57] + "..."
            lines.append(f'    """{doc_line}"""')

        if include_methods and self.methods:
            for method in self.methods[:max_methods]:
                sig = method.to_signature_string()
                lines.append(f"    {sig}")
            if len(self.methods) > max_methods:
                lines.append(f"    # ... {len(self.methods) - max_methods} more methods")

        return "\n".join(lines)


@dataclass
class ModuleInfo:
    """Represents a module/file with its contents."""
    file_path: Path
    language: Language
    imports: List[str] = field(default_factory=list)
    classes: List[ClassDefinition] = field(default_factory=list)
    functions: List[FunctionSignature] = field(default_factory=list)
    global_variables: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    docstring: Optional[str] = None
    line_count: int = 0

    def to_context_block(self, max_items: int = 20) -> str:
        """Generate a context block for LLM consumption."""
        lines = [f"# {self.file_path}"]

        if self.docstring:
            doc_line = self.docstring.split("\n")[0].strip()
            lines.append(f"# {doc_line}")

        items_shown = 0

        # Show classes
        for cls in self.classes:
            if items_shown >= max_items:
                break
            lines.append(cls.to_context_block(include_methods=True, max_methods=5))
            items_shown += 1 + len(cls.methods[:5])

        # Show standalone functions
        for func in self.functions:
            if items_shown >= max_items:
                break
            lines.append(func.to_context_line())
            items_shown += 1

        remaining = (len(self.classes) + len(self.functions)) - items_shown
        if remaining > 0:
            lines.append(f"# ... {remaining} more items")

        return "\n".join(lines)


class LanguageParser(ABC):
    """Abstract base class for language-specific parsers."""

    @abstractmethod
    async def parse_file(self, file_path: Path) -> ModuleInfo:
        """Parse a source file and extract structure information."""
        pass


class PythonParser(LanguageParser):
    """Python AST parser using the built-in ast module."""

    async def parse_file(self, file_path: Path) -> ModuleInfo:
        """Parse a Python file using the ast module."""
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ModuleInfo(file_path=file_path, language=Language.PYTHON)

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return ModuleInfo(
                file_path=file_path,
                language=Language.PYTHON,
                line_count=len(content.splitlines())
            )

        module_info = ModuleInfo(
            file_path=file_path,
            language=Language.PYTHON,
            line_count=len(content.splitlines())
        )

        # Extract module docstring
        module_info.docstring = ast.get_docstring(tree)

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    module_info.imports.append(f"{module}.{alias.name}")

        # Extract top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_def = self._parse_class(node, file_path)
                module_info.classes.append(cls_def)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_sig = self._parse_function(node, file_path)
                module_info.functions.append(func_sig)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        type_hint = None
                        module_info.global_variables.append((target.id, type_hint))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                type_hint = self._get_annotation_string(node.annotation)
                module_info.global_variables.append((node.target.id, type_hint))

        return module_info

    def _parse_class(self, node: ast.ClassDef, file_path: Path) -> ClassDefinition:
        """Parse a class definition."""
        decorators = [self._get_decorator_string(d) for d in node.decorator_list]
        base_classes = [self._get_annotation_string(base) for base in node.bases]

        cls_def = ClassDefinition(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            docstring=ast.get_docstring(node),
            decorators=decorators
        )

        # Parse methods and class variables
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._parse_function(item, file_path, is_method=True, class_name=node.name)
                cls_def.methods.append(method)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                type_hint = self._get_annotation_string(item.annotation)
                cls_def.class_variables.append((item.target.id, type_hint))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        cls_def.class_variables.append((target.id, None))

        return cls_def

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        is_method: bool = False,
        class_name: Optional[str] = None
    ) -> FunctionSignature:
        """Parse a function or method definition."""
        params = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                param_str += f": {self._get_annotation_string(arg.annotation)}"
            params.append(param_str)

        # Handle *args and **kwargs
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")

        return_type = None
        if node.returns:
            return_type = self._get_annotation_string(node.returns)

        decorators = [self._get_decorator_string(d) for d in node.decorator_list]

        return FunctionSignature(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            parameters=params,
            return_type=return_type,
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            class_name=class_name,
            decorators=decorators
        )

    def _get_annotation_string(self, node: ast.expr) -> str:
        """Convert an annotation AST node to a string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_string(node.value)
            slice_val = self._get_annotation_string(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._get_annotation_string(e) for e in node.elts)
            return elements
        elif isinstance(node, ast.Attribute):
            value = self._get_annotation_string(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._get_annotation_string(node.left)
            right = self._get_annotation_string(node.right)
            return f"{left} | {right}"
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else "..."

    def _get_decorator_string(self, node: ast.expr) -> str:
        """Convert a decorator AST node to a string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            func = self._get_annotation_string(node.func)
            return f"{func}(...)"
        elif isinstance(node, ast.Attribute):
            return self._get_annotation_string(node)
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else "..."


class TreeSitterParser(LanguageParser):
    """Tree-sitter based parser for multiple languages."""

    # Language file extension mappings
    EXTENSION_MAP = {
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".go": Language.GO,
        ".rs": Language.RUST,
    }

    def __init__(self, language: Language):
        self.language = language
        self._parser = None
        self._tree_sitter_available = False
        self._init_parser()

    def _init_parser(self) -> None:
        """Initialize tree-sitter parser if available."""
        try:
            import tree_sitter
            self._tree_sitter_available = True
            # Note: Actual language grammar loading would go here
            # This requires tree-sitter language packages to be installed
            logger.debug(f"Tree-sitter available for {self.language.value}")
        except ImportError:
            logger.debug("Tree-sitter not available, using regex fallback")
            self._tree_sitter_available = False

    async def parse_file(self, file_path: Path) -> ModuleInfo:
        """Parse a file using tree-sitter or fallback to regex."""
        try:
            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ModuleInfo(file_path=file_path, language=self.language)

        if self._tree_sitter_available:
            return await self._parse_with_tree_sitter(file_path, content)
        else:
            return await self._parse_with_regex(file_path, content)

    async def _parse_with_tree_sitter(self, file_path: Path, content: str) -> ModuleInfo:
        """Parse using tree-sitter (when available)."""
        # Placeholder for full tree-sitter implementation
        # For now, fall back to regex
        return await self._parse_with_regex(file_path, content)

    async def _parse_with_regex(self, file_path: Path, content: str) -> ModuleInfo:
        """Fallback regex-based parsing for basic structure extraction."""
        module_info = ModuleInfo(
            file_path=file_path,
            language=self.language,
            line_count=len(content.splitlines())
        )

        lines = content.splitlines()

        # Language-specific patterns
        if self.language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            await self._parse_js_ts(module_info, lines, content)
        elif self.language == Language.GO:
            await self._parse_go(module_info, lines, content)
        elif self.language == Language.RUST:
            await self._parse_rust(module_info, lines, content)

        return module_info

    async def _parse_js_ts(self, module_info: ModuleInfo, lines: List[str], content: str) -> None:
        """Parse JavaScript/TypeScript with regex patterns."""
        # Class pattern
        class_pattern = re.compile(
            r'^\s*(export\s+)?(abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?',
            re.MULTILINE
        )

        # Function patterns
        func_patterns = [
            # Regular function
            re.compile(r'^\s*(export\s+)?(async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?', re.MULTILINE),
            # Arrow function assigned to const/let
            re.compile(r'^\s*(export\s+)?(const|let)\s+(\w+)\s*=\s*(async\s+)?\([^)]*\)\s*(?::\s*([^=]+))?\s*=>', re.MULTILINE),
            # Method in class
            re.compile(r'^\s*(public|private|protected)?\s*(async\s+)?(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?', re.MULTILINE),
        ]

        # Import patterns
        import_pattern = re.compile(r'^\s*import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE)

        for match in import_pattern.finditer(content):
            module_info.imports.append(match.group(1))

        for match in class_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            bases = []
            if match.group(4):
                bases.append(match.group(4))
            if match.group(5):
                bases.extend([b.strip() for b in match.group(5).split(',')])

            cls_def = ClassDefinition(
                name=match.group(3),
                file_path=module_info.file_path,
                line_number=line_num,
                base_classes=bases
            )
            module_info.classes.append(cls_def)

        for pattern in func_patterns[:2]:  # Only top-level functions
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                is_async = bool(match.group(2) if len(match.groups()) >= 2 else False)
                name = match.group(3) if len(match.groups()) >= 3 else ""
                params_str = match.group(4) if len(match.groups()) >= 4 else ""

                if name and not name.startswith('_'):
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                    func_sig = FunctionSignature(
                        name=name,
                        file_path=module_info.file_path,
                        line_number=line_num,
                        parameters=params,
                        is_async=is_async
                    )
                    module_info.functions.append(func_sig)

    async def _parse_go(self, module_info: ModuleInfo, lines: List[str], content: str) -> None:
        """Parse Go with regex patterns."""
        # Struct pattern (Go's equivalent of class)
        struct_pattern = re.compile(r'^\s*type\s+(\w+)\s+struct\s*\{', re.MULTILINE)

        # Interface pattern
        interface_pattern = re.compile(r'^\s*type\s+(\w+)\s+interface\s*\{', re.MULTILINE)

        # Function pattern
        func_pattern = re.compile(
            r'^\s*func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(([^)]*)\)(?:\s*\(([^)]*)\)|\s*(\w+))?',
            re.MULTILINE
        )

        # Import pattern
        import_pattern = re.compile(r'^\s*import\s+(?:\(\s*)?["\']([^"\']+)["\']', re.MULTILINE)

        for match in import_pattern.finditer(content):
            module_info.imports.append(match.group(1))

        # Parse structs as classes
        for match in struct_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            cls_def = ClassDefinition(
                name=match.group(1),
                file_path=module_info.file_path,
                line_number=line_num
            )
            module_info.classes.append(cls_def)

        # Parse interfaces as classes
        for match in interface_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            cls_def = ClassDefinition(
                name=match.group(1),
                file_path=module_info.file_path,
                line_number=line_num,
                decorators=["interface"]
            )
            module_info.classes.append(cls_def)

        # Parse functions
        for match in func_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            receiver_type = match.group(2)
            name = match.group(3)
            params_str = match.group(4) or ""

            params = [p.strip() for p in params_str.split(',') if p.strip()]

            func_sig = FunctionSignature(
                name=name,
                file_path=module_info.file_path,
                line_number=line_num,
                parameters=params,
                is_method=receiver_type is not None,
                class_name=receiver_type
            )

            if receiver_type:
                # Add to corresponding class if exists
                for cls in module_info.classes:
                    if cls.name == receiver_type:
                        cls.methods.append(func_sig)
                        break
                else:
                    module_info.functions.append(func_sig)
            else:
                module_info.functions.append(func_sig)

    async def _parse_rust(self, module_info: ModuleInfo, lines: List[str], content: str) -> None:
        """Parse Rust with regex patterns."""
        # Struct pattern
        struct_pattern = re.compile(r'^\s*(pub\s+)?struct\s+(\w+)(?:<[^>]+>)?', re.MULTILINE)

        # Enum pattern
        enum_pattern = re.compile(r'^\s*(pub\s+)?enum\s+(\w+)(?:<[^>]+>)?', re.MULTILINE)

        # Trait pattern
        trait_pattern = re.compile(r'^\s*(pub\s+)?trait\s+(\w+)(?:<[^>]+>)?', re.MULTILINE)

        # Impl block pattern
        impl_pattern = re.compile(r'^\s*impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)', re.MULTILINE)

        # Function pattern
        func_pattern = re.compile(
            r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]+))?',
            re.MULTILINE
        )

        # Use pattern (imports)
        use_pattern = re.compile(r'^\s*use\s+([^;]+);', re.MULTILINE)

        for match in use_pattern.finditer(content):
            module_info.imports.append(match.group(1))

        # Parse structs
        for match in struct_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            cls_def = ClassDefinition(
                name=match.group(2),
                file_path=module_info.file_path,
                line_number=line_num,
                decorators=["struct"]
            )
            module_info.classes.append(cls_def)

        # Parse enums
        for match in enum_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            cls_def = ClassDefinition(
                name=match.group(2),
                file_path=module_info.file_path,
                line_number=line_num,
                decorators=["enum"]
            )
            module_info.classes.append(cls_def)

        # Parse traits
        for match in trait_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            cls_def = ClassDefinition(
                name=match.group(2),
                file_path=module_info.file_path,
                line_number=line_num,
                decorators=["trait"]
            )
            module_info.classes.append(cls_def)

        # Parse functions
        for match in func_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            is_async = bool(match.group(2))
            name = match.group(3)
            params_str = match.group(4) or ""
            return_type = match.group(5).strip() if match.group(5) else None

            params = [p.strip() for p in params_str.split(',') if p.strip()]

            func_sig = FunctionSignature(
                name=name,
                file_path=module_info.file_path,
                line_number=line_num,
                parameters=params,
                return_type=return_type,
                is_async=is_async
            )
            module_info.functions.append(func_sig)


class RepositoryMap:
    """
    Generate and maintain a map of repository structure using AST parsing.

    Provides intelligent context generation for LLMs by analyzing code
    structure and generating compressed representations suitable for
    inclusion in prompts.

    Example:
        repo_map = RepositoryMap(Path("/path/to/repo"))
        await repo_map.build(extensions=[".py", ".js"])

        # Get context suitable for 8000 tokens
        context = repo_map.get_context(max_tokens=8000)

        # Find symbols related to "Authentication"
        related = repo_map.find_related("Authentication")
    """

    # Extension to language mapping
    EXTENSION_MAP: Dict[str, Language] = {
        ".py": Language.PYTHON,
        ".pyw": Language.PYTHON,
        ".js": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".go": Language.GO,
        ".rs": Language.RUST,
    }

    # Default extensions to parse
    DEFAULT_EXTENSIONS = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"]

    # Directories to ignore
    IGNORE_DIRS = {
        "__pycache__", ".git", ".svn", ".hg", "node_modules", "venv",
        ".venv", "env", ".env", "build", "dist", ".tox", ".pytest_cache",
        ".mypy_cache", "target", "vendor", ".idea", ".vscode"
    }

    def __init__(self, root_path: Path):
        """
        Initialize the repository map.

        Args:
            root_path: Root directory of the repository to map.
        """
        self.root_path = Path(root_path).resolve()
        self.modules: Dict[Path, ModuleInfo] = {}
        self.functions: Dict[str, FunctionSignature] = {}
        self.classes: Dict[str, ClassDefinition] = {}
        self._parsers: Dict[Language, LanguageParser] = {}
        self._built = False

    def _get_parser(self, language: Language) -> LanguageParser:
        """Get or create a parser for the given language."""
        if language not in self._parsers:
            if language == Language.PYTHON:
                self._parsers[language] = PythonParser()
            else:
                self._parsers[language] = TreeSitterParser(language)
        return self._parsers[language]

    async def build(
        self,
        extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        max_file_size: int = 1_000_000  # 1MB
    ) -> None:
        """
        Parse all files and build the repository map.

        Args:
            extensions: File extensions to parse (e.g., [".py", ".js"]).
                       Defaults to common extensions.
            ignore_patterns: Additional glob patterns to ignore.
            max_file_size: Maximum file size in bytes to parse.
        """
        if extensions is None:
            extensions = self.DEFAULT_EXTENSIONS

        # Normalize extensions
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        # Collect all matching files
        files_to_parse: List[Path] = []

        for ext in extensions:
            pattern = f"**/*{ext}"
            for file_path in self.root_path.glob(pattern):
                # Skip ignored directories
                if any(part in self.IGNORE_DIRS for part in file_path.parts):
                    continue

                # Skip files matching ignore patterns
                if ignore_patterns:
                    rel_path = str(file_path.relative_to(self.root_path))
                    if any(re.match(pattern, rel_path) for pattern in ignore_patterns):
                        continue

                # Skip large files
                try:
                    if file_path.stat().st_size > max_file_size:
                        logger.debug(f"Skipping large file: {file_path}")
                        continue
                except OSError:
                    continue

                files_to_parse.append(file_path)

        # Parse all files concurrently
        async def parse_file(file_path: Path) -> Optional[ModuleInfo]:
            ext = file_path.suffix
            language = self.EXTENSION_MAP.get(ext, Language.UNKNOWN)
            if language == Language.UNKNOWN:
                return None

            parser = self._get_parser(language)
            try:
                return await parser.parse_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
                return None

        # Run parsing with concurrency limit
        semaphore = asyncio.Semaphore(50)  # Limit concurrent file operations

        async def parse_with_limit(file_path: Path) -> Optional[ModuleInfo]:
            async with semaphore:
                return await parse_file(file_path)

        results = await asyncio.gather(
            *[parse_with_limit(fp) for fp in files_to_parse],
            return_exceptions=True
        )

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Parse error: {result}")
                continue
            if result is None:
                continue

            module_info: ModuleInfo = result
            self.modules[module_info.file_path] = module_info

            # Index functions and classes
            for func in module_info.functions:
                key = f"{module_info.file_path}:{func.name}"
                self.functions[key] = func

            for cls in module_info.classes:
                key = f"{module_info.file_path}:{cls.name}"
                self.classes[key] = cls

                # Index methods
                for method in cls.methods:
                    method_key = f"{module_info.file_path}:{cls.name}.{method.name}"
                    self.functions[method_key] = method

        self._built = True
        logger.info(
            f"Built repository map: {len(self.modules)} modules, "
            f"{len(self.classes)} classes, {len(self.functions)} functions"
        )

    def get_context(
        self,
        max_tokens: int = 8000,
        include_docstrings: bool = True,
        prioritize_files: Optional[List[Path]] = None,
        chars_per_token: float = 4.0
    ) -> str:
        """
        Generate a context string suitable for LLM consumption.

        Args:
            max_tokens: Maximum approximate token count for output.
            include_docstrings: Whether to include brief docstrings.
            prioritize_files: Files to include first (e.g., recently modified).
            chars_per_token: Estimated characters per token for budgeting.

        Returns:
            A compressed context string representing the repository structure.
        """
        if not self._built:
            return "# Repository map not built. Call build() first."

        max_chars = int(max_tokens * chars_per_token)
        lines = ["# Repository Structure", ""]
        current_chars = len(lines[0]) + len(lines[1])

        # Sort modules by priority
        sorted_modules = list(self.modules.values())
        if prioritize_files:
            priority_set = set(prioritize_files)
            sorted_modules.sort(
                key=lambda m: (0 if m.file_path in priority_set else 1, str(m.file_path))
            )
        else:
            sorted_modules.sort(key=lambda m: str(m.file_path))

        for module in sorted_modules:
            if current_chars >= max_chars:
                lines.append(f"\n# ... {len(sorted_modules) - sorted_modules.index(module)} more files")
                break

            # Generate module context
            rel_path = module.file_path.relative_to(self.root_path)
            module_header = f"\n## {rel_path}"

            # Check budget
            if current_chars + len(module_header) > max_chars:
                lines.append(f"\n# ... more files truncated")
                break

            lines.append(module_header)
            current_chars += len(module_header)

            # Add classes with methods
            for cls in module.classes:
                cls_block = cls.to_context_block(
                    include_methods=True,
                    max_methods=8
                )
                if current_chars + len(cls_block) > max_chars:
                    break
                lines.append(cls_block)
                current_chars += len(cls_block)

            # Add standalone functions
            for func in module.functions:
                func_line = func.to_context_line(include_docstring=include_docstrings)
                if current_chars + len(func_line) > max_chars:
                    break
                lines.append(func_line)
                current_chars += len(func_line)

        return "\n".join(lines)

    def find_related(self, symbol: str, max_results: int = 20) -> List[str]:
        """
        Find symbols related to the given symbol.

        Uses string matching and import analysis to find related code.

        Args:
            symbol: The symbol name to search for (class, function, etc.).
            max_results: Maximum number of results to return.

        Returns:
            List of related symbol identifiers.
        """
        related: List[Tuple[str, float]] = []
        symbol_lower = symbol.lower()

        # Search in class names
        for key, cls in self.classes.items():
            score = self._similarity_score(symbol_lower, cls.name.lower())
            if score > 0:
                related.append((key, score))

            # Check base classes
            for base in cls.base_classes:
                if symbol_lower in base.lower():
                    related.append((key, 0.8))

        # Search in function names
        for key, func in self.functions.items():
            score = self._similarity_score(symbol_lower, func.name.lower())
            if score > 0:
                related.append((key, score))

        # Search in imports
        for module in self.modules.values():
            for imp in module.imports:
                if symbol_lower in imp.lower():
                    for cls in module.classes:
                        key = f"{module.file_path}:{cls.name}"
                        related.append((key, 0.5))
                    for func in module.functions:
                        key = f"{module.file_path}:{func.name}"
                        related.append((key, 0.5))

        # Sort by score and deduplicate
        related.sort(key=lambda x: -x[1])
        seen: Set[str] = set()
        results: List[str] = []

        for key, score in related:
            if key not in seen:
                seen.add(key)
                results.append(key)
                if len(results) >= max_results:
                    break

        return results

    def _similarity_score(self, query: str, target: str) -> float:
        """Calculate a simple similarity score between query and target."""
        if query == target:
            return 1.0
        if query in target:
            return 0.8
        if target in query:
            return 0.6

        # Check for common substrings
        min_len = min(len(query), len(target))
        if min_len >= 3:
            for i in range(min_len, 2, -1):
                for j in range(len(query) - i + 1):
                    if query[j:j+i] in target:
                        return 0.4 * (i / min_len)

        return 0.0

    def get_file_context(self, file_path: Path) -> Optional[str]:
        """
        Get the context for a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            Context string for the file, or None if not found.
        """
        file_path = Path(file_path).resolve()
        if file_path in self.modules:
            return self.modules[file_path].to_context_block()
        return None

    def get_class(self, class_name: str) -> Optional[ClassDefinition]:
        """
        Get a class definition by name.

        Args:
            class_name: The class name to search for.

        Returns:
            The ClassDefinition if found, None otherwise.
        """
        for key, cls in self.classes.items():
            if cls.name == class_name:
                return cls
        return None

    def get_function(self, func_name: str) -> Optional[FunctionSignature]:
        """
        Get a function by name.

        Args:
            func_name: The function name to search for.

        Returns:
            The FunctionSignature if found, None otherwise.
        """
        for key, func in self.functions.items():
            if func.name == func_name:
                return func
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the repository map."""
        return {
            "modules": len(self.modules),
            "classes": len(self.classes),
            "functions": len(self.functions),
            "total_lines": sum(m.line_count for m in self.modules.values()),
            "languages": list(set(m.language.value for m in self.modules.values())),
        }
