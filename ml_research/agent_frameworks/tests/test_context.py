"""
Tests for context management components.

Tests cover:
    - RepositoryMap building and querying
    - FileSelector ranking algorithms
    - DiffTracker change detection
    - SemanticIndex search functionality
"""

import pytest
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum, auto
import tempfile
import os


# ---------------------------------------------------------------------------
# Test Data Structures (will be replaced by actual imports)
# ---------------------------------------------------------------------------

@dataclass
class FunctionSignature:
    """Represents a function signature."""
    name: str
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    line_number: int = 0


@dataclass
class ClassDefinition:
    """Represents a class definition."""
    name: str
    methods: List[FunctionSignature]
    bases: List[str]
    docstring: Optional[str] = None
    line_number: int = 0


@dataclass
class ModuleInfo:
    """Information about a Python module."""
    path: Path
    functions: List[FunctionSignature]
    classes: List[ClassDefinition]
    imports: List[str]


class RepositoryMap:
    """Maps repository structure using AST analysis."""

    def __init__(self, root: Path):
        self.root = root
        self._modules: Dict[str, ModuleInfo] = {}
        self._built = False

    async def build(self) -> None:
        """Build the repository map."""
        for py_file in self.root.rglob("*.py"):
            try:
                self._parse_file(py_file)
            except Exception:
                pass
        self._built = True

    def _parse_file(self, path: Path) -> None:
        """Parse a Python file."""
        content = path.read_text()
        # Simplified parsing for tests
        module = ModuleInfo(
            path=path,
            functions=[],
            classes=[],
            imports=[]
        )

        # Find function definitions (simplified)
        for i, line in enumerate(content.split("\n")):
            if line.strip().startswith("def "):
                name = line.split("def ")[1].split("(")[0]
                module.functions.append(FunctionSignature(
                    name=name,
                    parameters=[],
                    line_number=i + 1
                ))
            elif line.strip().startswith("class "):
                name = line.split("class ")[1].split("(")[0].split(":")[0]
                module.classes.append(ClassDefinition(
                    name=name,
                    methods=[],
                    bases=[],
                    line_number=i + 1
                ))
            elif line.strip().startswith("import ") or line.strip().startswith("from "):
                module.imports.append(line.strip())

        self._modules[str(path)] = module

    def get_module(self, path: str) -> Optional[ModuleInfo]:
        """Get module info by path."""
        return self._modules.get(path)

    def list_modules(self) -> List[str]:
        """List all module paths."""
        return list(self._modules.keys())

    def get_functions(self) -> List[FunctionSignature]:
        """Get all functions across modules."""
        functions = []
        for module in self._modules.values():
            functions.extend(module.functions)
        return functions

    def get_classes(self) -> List[ClassDefinition]:
        """Get all classes across modules."""
        classes = []
        for module in self._modules.values():
            classes.extend(module.classes)
        return classes

    def get_context(self, max_tokens: int = 8000) -> str:
        """Get repository context as string."""
        lines = []
        for path, module in self._modules.items():
            rel_path = Path(path).relative_to(self.root)
            lines.append(f"## {rel_path}")
            for func in module.functions:
                lines.append(f"  - def {func.name}()")
            for cls in module.classes:
                lines.append(f"  - class {cls.name}")
        return "\n".join(lines)


class SelectionStrategy(Enum):
    """Strategy for file selection."""
    RECENCY = auto()
    RELEVANCE = auto()
    HYBRID = auto()


@dataclass
class FileScore:
    """Score for file relevance."""
    path: str
    score: float
    reasons: List[str]


class FileSelector:
    """Selects relevant files for context."""

    def __init__(self, repo_map: RepositoryMap):
        self.repo_map = repo_map
        self.strategy = SelectionStrategy.HYBRID

    async def select_relevant(
        self,
        query: str,
        max_files: int = 10,
        strategy: Optional[SelectionStrategy] = None
    ) -> List[FileScore]:
        """Select files relevant to the query."""
        strategy = strategy or self.strategy
        scores = []

        for path in self.repo_map.list_modules():
            score = self._calculate_score(path, query, strategy)
            scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:max_files]

    def _calculate_score(
        self,
        path: str,
        query: str,
        strategy: SelectionStrategy
    ) -> FileScore:
        """Calculate relevance score for a file."""
        score = 0.0
        reasons = []

        # Simple keyword matching
        query_terms = query.lower().split()
        path_lower = path.lower()

        for term in query_terms:
            if term in path_lower:
                score += 0.5
                reasons.append(f"Path contains '{term}'")

        module = self.repo_map.get_module(path)
        if module:
            for func in module.functions:
                if any(term in func.name.lower() for term in query_terms):
                    score += 0.3
                    reasons.append(f"Function '{func.name}' matches query")
            for cls in module.classes:
                if any(term in cls.name.lower() for term in query_terms):
                    score += 0.3
                    reasons.append(f"Class '{cls.name}' matches query")

        return FileScore(path=path, score=score, reasons=reasons)


class ChangeType(Enum):
    """Type of file change."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class ChangeEvent:
    """A file change event."""
    path: str
    change_type: ChangeType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: Optional[str] = None


class DiffTracker:
    """Tracks changes to files."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._baseline: Dict[str, str] = {}
        self._changes: List[ChangeEvent] = []

    def snapshot(self) -> None:
        """Take a snapshot of current state."""
        self._baseline.clear()
        for path in self.workspace.rglob("*"):
            if path.is_file():
                try:
                    self._baseline[str(path)] = path.read_text()
                except Exception:
                    pass

    def detect_changes(self) -> List[ChangeEvent]:
        """Detect changes since last snapshot."""
        self._changes.clear()
        current_files = set()

        for path in self.workspace.rglob("*"):
            if path.is_file():
                path_str = str(path)
                current_files.add(path_str)

                try:
                    content = path.read_text()
                except Exception:
                    continue

                if path_str not in self._baseline:
                    self._changes.append(ChangeEvent(
                        path=path_str,
                        change_type=ChangeType.ADDED,
                        new_content=content
                    ))
                elif self._baseline[path_str] != content:
                    self._changes.append(ChangeEvent(
                        path=path_str,
                        change_type=ChangeType.MODIFIED,
                        old_content=self._baseline[path_str],
                        new_content=content
                    ))

        # Detect deleted files
        for path_str in self._baseline:
            if path_str not in current_files:
                self._changes.append(ChangeEvent(
                    path=path_str,
                    change_type=ChangeType.DELETED,
                    old_content=self._baseline[path_str]
                ))

        return self._changes

    def get_changes(self) -> List[ChangeEvent]:
        """Get detected changes."""
        return self._changes.copy()


@dataclass
class SearchResult:
    """Result from semantic search."""
    path: str
    chunk: str
    score: float
    line_start: int
    line_end: int


class ChunkingStrategy(Enum):
    """Strategy for chunking documents."""
    FIXED_SIZE = auto()
    SEMANTIC = auto()
    AST_BASED = auto()


class SemanticIndex:
    """Semantic search index using embeddings."""

    def __init__(self, embed_func=None):
        self._embed_func = embed_func or (lambda x: [0.0] * 128)
        self._chunks: List[Dict[str, Any]] = []

    async def index_file(self, path: Path, strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE) -> None:
        """Index a file."""
        content = path.read_text()
        lines = content.split("\n")

        # Simple fixed-size chunking
        chunk_size = 20
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = "\n".join(chunk_lines)

            if chunk_text.strip():
                embedding = self._embed_func(chunk_text)
                self._chunks.append({
                    "path": str(path),
                    "chunk": chunk_text,
                    "line_start": i + 1,
                    "line_end": i + len(chunk_lines),
                    "embedding": embedding
                })

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant chunks."""
        query_embedding = self._embed_func(query)

        # Calculate similarity scores (simplified cosine similarity)
        scored = []
        for chunk in self._chunks:
            score = sum(a * b for a, b in zip(query_embedding, chunk["embedding"]))
            scored.append((score, chunk))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            results.append(SearchResult(
                path=chunk["path"],
                chunk=chunk["chunk"],
                score=score,
                line_start=chunk["line_start"],
                line_end=chunk["line_end"]
            ))

        return results

    def clear(self) -> None:
        """Clear the index."""
        self._chunks.clear()


# ---------------------------------------------------------------------------
# Tests for RepositoryMap
# ---------------------------------------------------------------------------

class TestRepositoryMap:
    """Tests for RepositoryMap."""

    @pytest.mark.asyncio
    async def test_build_repository_map(self, sample_repo):
        """Test building repository map."""
        repo_map = RepositoryMap(sample_repo)
        await repo_map.build()

        modules = repo_map.list_modules()
        assert len(modules) > 0
        assert any("app.py" in m for m in modules)

    @pytest.mark.asyncio
    async def test_get_module(self, sample_repo):
        """Test getting module info."""
        repo_map = RepositoryMap(sample_repo)
        await repo_map.build()

        # Find app.py module
        app_path = None
        for path in repo_map.list_modules():
            if "app.py" in path:
                app_path = path
                break

        assert app_path is not None
        module = repo_map.get_module(app_path)
        assert module is not None
        assert any(c.name == "App" for c in module.classes)

    @pytest.mark.asyncio
    async def test_get_functions(self, sample_repo):
        """Test getting all functions."""
        repo_map = RepositoryMap(sample_repo)
        await repo_map.build()

        functions = repo_map.get_functions()
        function_names = [f.name for f in functions]

        assert "format_message" in function_names or "parse_config" in function_names

    @pytest.mark.asyncio
    async def test_get_context(self, sample_repo):
        """Test getting context string."""
        repo_map = RepositoryMap(sample_repo)
        await repo_map.build()

        context = repo_map.get_context()
        assert "App" in context or "format_message" in context


# ---------------------------------------------------------------------------
# Tests for FileSelector
# ---------------------------------------------------------------------------

class TestFileSelector:
    """Tests for FileSelector."""

    @pytest.mark.asyncio
    async def test_select_relevant(self, sample_repo):
        """Test selecting relevant files."""
        repo_map = RepositoryMap(sample_repo)
        await repo_map.build()

        selector = FileSelector(repo_map)
        results = await selector.select_relevant("app", max_files=5)

        assert len(results) > 0
        # app.py should be ranked highly
        top_paths = [r.path for r in results[:3]]
        assert any("app" in p.lower() for p in top_paths)

    @pytest.mark.asyncio
    async def test_score_reasons(self, sample_repo):
        """Test that scores include reasons."""
        repo_map = RepositoryMap(sample_repo)
        await repo_map.build()

        selector = FileSelector(repo_map)
        results = await selector.select_relevant("utils", max_files=3)

        # Find utils file result
        utils_result = next((r for r in results if "utils" in r.path.lower()), None)
        if utils_result:
            assert utils_result.score > 0
            assert len(utils_result.reasons) > 0


# ---------------------------------------------------------------------------
# Tests for DiffTracker
# ---------------------------------------------------------------------------

class TestDiffTracker:
    """Tests for DiffTracker."""

    def test_detect_added_file(self, temp_workspace):
        """Test detecting added files."""
        tracker = DiffTracker(temp_workspace)
        tracker.snapshot()

        # Add a new file
        new_file = temp_workspace / "new_file.txt"
        new_file.write_text("New content")

        changes = tracker.detect_changes()

        added = [c for c in changes if c.change_type == ChangeType.ADDED]
        assert len(added) == 1
        assert "new_file.txt" in added[0].path

    def test_detect_modified_file(self, temp_workspace):
        """Test detecting modified files."""
        tracker = DiffTracker(temp_workspace)

        # Create initial file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Original content")

        tracker.snapshot()

        # Modify the file
        test_file.write_text("Modified content")

        changes = tracker.detect_changes()

        modified = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        assert len(modified) == 1
        assert modified[0].old_content == "Original content"
        assert modified[0].new_content == "Modified content"

    def test_detect_deleted_file(self, temp_workspace):
        """Test detecting deleted files."""
        tracker = DiffTracker(temp_workspace)

        # Create file
        test_file = temp_workspace / "to_delete.txt"
        test_file.write_text("Will be deleted")

        tracker.snapshot()

        # Delete the file
        test_file.unlink()

        changes = tracker.detect_changes()

        deleted = [c for c in changes if c.change_type == ChangeType.DELETED]
        assert len(deleted) == 1
        assert "to_delete.txt" in deleted[0].path

    def test_no_changes(self, temp_workspace):
        """Test when there are no changes."""
        tracker = DiffTracker(temp_workspace)
        tracker.snapshot()

        changes = tracker.detect_changes()
        assert len(changes) == 0


# ---------------------------------------------------------------------------
# Tests for SemanticIndex
# ---------------------------------------------------------------------------

class TestSemanticIndex:
    """Tests for SemanticIndex."""

    @pytest.mark.asyncio
    async def test_index_file(self, temp_workspace):
        """Test indexing a file."""
        index = SemanticIndex()

        # Create a file with content
        test_file = temp_workspace / "test.py"
        test_file.write_text("""def hello():
    print("Hello")

def world():
    print("World")
""")

        await index.index_file(test_file)

        # Should have indexed chunks
        assert len(index._chunks) > 0

    @pytest.mark.asyncio
    async def test_search(self, temp_workspace):
        """Test searching the index."""
        # Create a simple embedding function that favors matching terms
        def mock_embed(text):
            # Simple embedding based on text length and presence of "hello"
            base = [0.1] * 128
            if "hello" in text.lower():
                base[0] = 1.0
            return base

        index = SemanticIndex(embed_func=mock_embed)

        test_file = temp_workspace / "test.py"
        test_file.write_text("""def hello():
    print("Hello world")

def goodbye():
    print("Goodbye")
""")

        await index.index_file(test_file)
        results = await index.search("hello", top_k=2)

        assert len(results) > 0
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_clear_index(self, temp_workspace):
        """Test clearing the index."""
        index = SemanticIndex()

        test_file = temp_workspace / "test.py"
        test_file.write_text("def test(): pass")

        await index.index_file(test_file)
        assert len(index._chunks) > 0

        index.clear()
        assert len(index._chunks) == 0
