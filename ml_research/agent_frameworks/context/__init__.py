"""
Context Module for Agent Frameworks.

Provides intelligent context management for LLM-powered coding agents,
inspired by Aider's repository mapping and context optimization techniques.

This module includes:
- RepositoryMap: AST-based code structure analysis
- FileSelector: Intelligent file selection for context
- DiffTracker: Change tracking and diff generation
- SemanticIndex: Embedding-based semantic search

Example:
    from agent_frameworks.context import RepositoryMap, FileSelector

    repo_map = RepositoryMap(Path("/path/to/repo"))
    await repo_map.build()
    context = repo_map.get_context(max_tokens=8000)

    selector = FileSelector(repo_map)
    relevant_files = await selector.select_relevant("implement user auth", max_files=10)
"""

from .repository_map import (
    RepositoryMap,
    FunctionSignature,
    ClassDefinition,
    ModuleInfo,
)
from .file_selector import (
    FileSelector,
    FileScore,
    SelectionStrategy,
)
from .diff_tracker import (
    DiffTracker,
    ChangeEvent,
    ChangeType,
)
from .semantic_index import (
    SemanticIndex,
    SearchResult,
    ChunkingStrategy,
)

__all__ = [
    # Repository mapping
    "RepositoryMap",
    "FunctionSignature",
    "ClassDefinition",
    "ModuleInfo",
    # File selection
    "FileSelector",
    "FileScore",
    "SelectionStrategy",
    # Diff tracking
    "DiffTracker",
    "ChangeEvent",
    "ChangeType",
    # Semantic search
    "SemanticIndex",
    "SearchResult",
    "ChunkingStrategy",
]
