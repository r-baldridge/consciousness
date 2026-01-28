"""
File Selector - Intelligent File Selection for Context.

Provides smart file selection for LLM context based on multiple signals:
- File path pattern matching
- Content grep matching
- Semantic similarity (when embeddings available)
- Recency and edit frequency
- Code structure relevance

Example:
    from agent_frameworks.context import FileSelector, RepositoryMap

    repo_map = RepositoryMap(Path("/path/to/repo"))
    await repo_map.build()

    selector = FileSelector(repo_map)
    files = await selector.select_relevant(
        query="implement user authentication with JWT",
        max_files=10
    )
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Callable, Any
from pathlib import Path
from enum import Enum
import asyncio
import re
import logging
import os
from datetime import datetime

from .repository_map import RepositoryMap, ModuleInfo, ClassDefinition, FunctionSignature


logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Strategies for file selection."""
    BALANCED = "balanced"  # Use all signals with balanced weights
    PATH_FOCUSED = "path_focused"  # Prioritize path matching
    CONTENT_FOCUSED = "content_focused"  # Prioritize content matching
    SEMANTIC = "semantic"  # Prioritize semantic similarity
    RECENT = "recent"  # Prioritize recently modified files
    STRUCTURE = "structure"  # Prioritize code structure relevance


@dataclass
class FileScore:
    """Represents a file with its relevance scores."""
    file_path: Path
    total_score: float = 0.0
    path_score: float = 0.0
    content_score: float = 0.0
    semantic_score: float = 0.0
    structure_score: float = 0.0
    recency_score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        """Add a reason for the match."""
        self.match_reasons.append(reason)


@dataclass
class SelectionConfig:
    """Configuration for file selection."""
    # Weight factors for different scoring methods
    path_weight: float = 0.3
    content_weight: float = 0.35
    semantic_weight: float = 0.2
    structure_weight: float = 0.1
    recency_weight: float = 0.05

    # Search parameters
    max_grep_matches: int = 100
    grep_context_lines: int = 2
    max_file_size: int = 500_000  # 500KB

    # Recency scoring
    recency_days: int = 30  # Files modified within this many days get bonus

    # Path matching
    path_boost_patterns: List[str] = field(default_factory=list)
    path_penalty_patterns: List[str] = field(default_factory=list)


class FileSelector:
    """
    Intelligent file selector for LLM context.

    Combines multiple signals to select the most relevant files for a query:
    - Path pattern matching (file names, directory structure)
    - Content grep (keyword occurrence)
    - Semantic similarity (when embeddings available)
    - Code structure (classes, functions, imports)
    - Recency (recently modified files)

    Example:
        selector = FileSelector(repo_map)

        # Basic selection
        files = await selector.select_relevant("user authentication", max_files=10)

        # With custom strategy
        files = await selector.select_relevant(
            "database migrations",
            max_files=5,
            strategy=SelectionStrategy.CONTENT_FOCUSED
        )

        # With custom config
        config = SelectionConfig(content_weight=0.5, path_weight=0.3)
        files = await selector.select_relevant("API endpoints", config=config)
    """

    # Common patterns that indicate test files
    TEST_PATTERNS = [
        r"test[_/]", r"[_/]test", r"tests?\.py$", r"_test\.py$",
        r"spec\.js$", r"\.spec\.", r"\.test\.", r"__tests__"
    ]

    # Patterns that indicate documentation
    DOC_PATTERNS = [
        r"docs?[/\\]", r"readme", r"\.md$", r"\.rst$", r"\.txt$"
    ]

    # Patterns that indicate generated/vendor code
    GENERATED_PATTERNS = [
        r"node_modules", r"vendor[/\\]", r"\.min\.", r"generated",
        r"__pycache__", r"\.pyc$", r"build[/\\]", r"dist[/\\]"
    ]

    def __init__(
        self,
        repository_map: Optional[RepositoryMap] = None,
        root_path: Optional[Path] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize the file selector.

        Args:
            repository_map: Pre-built repository map for structure analysis.
            root_path: Root path of the repository (required if no repo_map).
            embedding_fn: Optional function to compute embeddings for semantic search.
        """
        self.repo_map = repository_map
        self.root_path = root_path or (repository_map.root_path if repository_map else None)
        self.embedding_fn = embedding_fn
        self._file_embeddings: Dict[Path, List[float]] = {}

        if self.root_path is None:
            raise ValueError("Either repository_map or root_path must be provided")

        self.root_path = Path(self.root_path).resolve()

    async def select_relevant(
        self,
        query: str,
        max_files: int = 10,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        config: Optional[SelectionConfig] = None,
        include_tests: bool = False,
        include_docs: bool = False,
        file_filter: Optional[Callable[[Path], bool]] = None
    ) -> List[Path]:
        """
        Select the most relevant files for a query.

        Args:
            query: Natural language query or search terms.
            max_files: Maximum number of files to return.
            strategy: Selection strategy to use.
            config: Custom configuration for selection.
            include_tests: Whether to include test files.
            include_docs: Whether to include documentation files.
            file_filter: Optional custom filter function.

        Returns:
            List of file paths sorted by relevance.
        """
        if config is None:
            config = self._get_config_for_strategy(strategy)

        # Extract keywords from query
        keywords = self._extract_keywords(query)

        # Score all candidate files
        scores: Dict[Path, FileScore] = {}

        # Get candidate files
        candidates = await self._get_candidate_files(include_tests, include_docs, file_filter)

        # Score files using different methods concurrently
        scoring_tasks = []

        for file_path in candidates:
            scores[file_path] = FileScore(file_path=file_path)

        # Path scoring (fast, no I/O)
        for file_path in candidates:
            scores[file_path].path_score = self._score_path(file_path, keywords, config)

        # Content scoring (requires file reading)
        content_scores = await self._score_content_batch(candidates, keywords, config)
        for file_path, (score, reasons) in content_scores.items():
            if file_path in scores:
                scores[file_path].content_score = score
                scores[file_path].match_reasons.extend(reasons)

        # Structure scoring (uses repo map)
        if self.repo_map and self.repo_map._built:
            structure_scores = self._score_structure_batch(candidates, keywords)
            for file_path, (score, reasons) in structure_scores.items():
                if file_path in scores:
                    scores[file_path].structure_score = score
                    scores[file_path].match_reasons.extend(reasons)

        # Semantic scoring (optional, requires embeddings)
        if self.embedding_fn:
            semantic_scores = await self._score_semantic_batch(candidates, query)
            for file_path, score in semantic_scores.items():
                if file_path in scores:
                    scores[file_path].semantic_score = score

        # Recency scoring
        recency_scores = await self._score_recency_batch(candidates, config)
        for file_path, score in recency_scores.items():
            if file_path in scores:
                scores[file_path].recency_score = score

        # Calculate total scores
        for file_score in scores.values():
            file_score.total_score = (
                config.path_weight * file_score.path_score +
                config.content_weight * file_score.content_score +
                config.semantic_weight * file_score.semantic_score +
                config.structure_weight * file_score.structure_score +
                config.recency_weight * file_score.recency_score
            )

        # Sort by total score
        sorted_scores = sorted(
            scores.values(),
            key=lambda x: x.total_score,
            reverse=True
        )

        # Return top files
        return [fs.file_path for fs in sorted_scores[:max_files]]

    async def select_with_scores(
        self,
        query: str,
        max_files: int = 10,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        config: Optional[SelectionConfig] = None
    ) -> List[FileScore]:
        """
        Select files and return detailed scores.

        Same as select_relevant but returns FileScore objects with
        detailed scoring information.
        """
        if config is None:
            config = self._get_config_for_strategy(strategy)

        keywords = self._extract_keywords(query)
        candidates = await self._get_candidate_files()
        scores: Dict[Path, FileScore] = {}

        for file_path in candidates:
            scores[file_path] = FileScore(file_path=file_path)
            scores[file_path].path_score = self._score_path(file_path, keywords, config)

        content_scores = await self._score_content_batch(candidates, keywords, config)
        for file_path, (score, reasons) in content_scores.items():
            if file_path in scores:
                scores[file_path].content_score = score
                scores[file_path].match_reasons.extend(reasons)

        if self.repo_map and self.repo_map._built:
            structure_scores = self._score_structure_batch(candidates, keywords)
            for file_path, (score, reasons) in structure_scores.items():
                if file_path in scores:
                    scores[file_path].structure_score = score
                    scores[file_path].match_reasons.extend(reasons)

        if self.embedding_fn:
            semantic_scores = await self._score_semantic_batch(candidates, query)
            for file_path, score in semantic_scores.items():
                if file_path in scores:
                    scores[file_path].semantic_score = score

        recency_scores = await self._score_recency_batch(candidates, config)
        for file_path, score in recency_scores.items():
            if file_path in scores:
                scores[file_path].recency_score = score

        for file_score in scores.values():
            file_score.total_score = (
                config.path_weight * file_score.path_score +
                config.content_weight * file_score.content_score +
                config.semantic_weight * file_score.semantic_score +
                config.structure_weight * file_score.structure_score +
                config.recency_weight * file_score.recency_score
            )

        sorted_scores = sorted(
            scores.values(),
            key=lambda x: x.total_score,
            reverse=True
        )

        return sorted_scores[:max_files]

    def _get_config_for_strategy(self, strategy: SelectionStrategy) -> SelectionConfig:
        """Get configuration weights for a strategy."""
        if strategy == SelectionStrategy.BALANCED:
            return SelectionConfig()
        elif strategy == SelectionStrategy.PATH_FOCUSED:
            return SelectionConfig(
                path_weight=0.5,
                content_weight=0.3,
                semantic_weight=0.1,
                structure_weight=0.05,
                recency_weight=0.05
            )
        elif strategy == SelectionStrategy.CONTENT_FOCUSED:
            return SelectionConfig(
                path_weight=0.15,
                content_weight=0.55,
                semantic_weight=0.15,
                structure_weight=0.1,
                recency_weight=0.05
            )
        elif strategy == SelectionStrategy.SEMANTIC:
            return SelectionConfig(
                path_weight=0.1,
                content_weight=0.2,
                semantic_weight=0.5,
                structure_weight=0.15,
                recency_weight=0.05
            )
        elif strategy == SelectionStrategy.RECENT:
            return SelectionConfig(
                path_weight=0.2,
                content_weight=0.25,
                semantic_weight=0.1,
                structure_weight=0.1,
                recency_weight=0.35
            )
        elif strategy == SelectionStrategy.STRUCTURE:
            return SelectionConfig(
                path_weight=0.15,
                content_weight=0.25,
                semantic_weight=0.15,
                structure_weight=0.4,
                recency_weight=0.05
            )
        return SelectionConfig()

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract searchable keywords from a query."""
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "or", "if", "because", "until",
            "while", "i", "me", "my", "we", "our", "you", "your", "it", "its",
            "they", "them", "their", "this", "that", "these", "those", "what",
            "which", "who", "whom", "implement", "create", "add", "make", "fix",
            "update", "change", "modify", "code", "file", "function", "class"
        }

        # Tokenize
        tokens = re.findall(r'\b\w+\b', query.lower())

        # Filter stop words and short words
        keywords = [t for t in tokens if t not in stop_words and len(t) > 2]

        # Also extract potential class/function names (CamelCase, snake_case)
        camel_case = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*', query)
        snake_case = re.findall(r'[a-z]+(?:_[a-z]+)+', query)

        keywords.extend([w.lower() for w in camel_case])
        keywords.extend(snake_case)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_keywords: List[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    async def _get_candidate_files(
        self,
        include_tests: bool = False,
        include_docs: bool = False,
        file_filter: Optional[Callable[[Path], bool]] = None
    ) -> List[Path]:
        """Get all candidate files for scoring."""
        candidates: List[Path] = []

        # If we have a repo map, use its modules
        if self.repo_map and self.repo_map._built:
            candidates = list(self.repo_map.modules.keys())
        else:
            # Scan the directory
            for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"]:
                for file_path in self.root_path.glob(f"**/*{ext}"):
                    candidates.append(file_path)

        # Filter candidates
        filtered: List[Path] = []
        for file_path in candidates:
            rel_path = str(file_path.relative_to(self.root_path))

            # Skip generated code
            if any(re.search(p, rel_path, re.IGNORECASE) for p in self.GENERATED_PATTERNS):
                continue

            # Skip tests unless requested
            if not include_tests and any(re.search(p, rel_path, re.IGNORECASE) for p in self.TEST_PATTERNS):
                continue

            # Skip docs unless requested
            if not include_docs and any(re.search(p, rel_path, re.IGNORECASE) for p in self.DOC_PATTERNS):
                continue

            # Apply custom filter
            if file_filter and not file_filter(file_path):
                continue

            filtered.append(file_path)

        return filtered

    def _score_path(self, file_path: Path, keywords: List[str], config: SelectionConfig) -> float:
        """Score a file based on its path."""
        rel_path = str(file_path.relative_to(self.root_path)).lower()
        file_name = file_path.stem.lower()

        score = 0.0
        max_possible = len(keywords) if keywords else 1

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Exact filename match
            if keyword_lower == file_name:
                score += 1.0
            # Filename contains keyword
            elif keyword_lower in file_name:
                score += 0.7
            # Path contains keyword
            elif keyword_lower in rel_path:
                score += 0.4

        # Apply boost patterns
        for pattern in config.path_boost_patterns:
            if re.search(pattern, rel_path, re.IGNORECASE):
                score *= 1.3

        # Apply penalty patterns
        for pattern in config.path_penalty_patterns:
            if re.search(pattern, rel_path, re.IGNORECASE):
                score *= 0.7

        # Normalize to 0-1
        return min(score / max_possible, 1.0) if max_possible > 0 else 0.0

    async def _score_content_batch(
        self,
        file_paths: List[Path],
        keywords: List[str],
        config: SelectionConfig
    ) -> Dict[Path, Tuple[float, List[str]]]:
        """Score files based on content matching."""
        results: Dict[Path, Tuple[float, List[str]]] = {}

        async def score_file(file_path: Path) -> Tuple[Path, float, List[str]]:
            try:
                # Check file size
                size = await asyncio.to_thread(lambda: file_path.stat().st_size)
                if size > config.max_file_size:
                    return (file_path, 0.0, [])

                content = await asyncio.to_thread(
                    lambda: file_path.read_text(encoding="utf-8", errors="ignore")
                )
                content_lower = content.lower()

                score = 0.0
                reasons: List[str] = []
                max_possible = len(keywords) if keywords else 1

                for keyword in keywords:
                    count = content_lower.count(keyword.lower())
                    if count > 0:
                        # Log scale for frequency
                        import math
                        freq_score = min(1.0, math.log1p(count) / 5.0)
                        score += freq_score
                        if count >= 3:
                            reasons.append(f"'{keyword}' x{count}")

                normalized_score = min(score / max_possible, 1.0) if max_possible > 0 else 0.0
                return (file_path, normalized_score, reasons)

            except Exception as e:
                logger.debug(f"Failed to score content for {file_path}: {e}")
                return (file_path, 0.0, [])

        # Process files concurrently with limit
        semaphore = asyncio.Semaphore(20)

        async def score_with_limit(file_path: Path):
            async with semaphore:
                return await score_file(file_path)

        tasks = [score_with_limit(fp) for fp in file_paths]
        scored = await asyncio.gather(*tasks, return_exceptions=True)

        for result in scored:
            if isinstance(result, Exception):
                continue
            file_path, score, reasons = result
            results[file_path] = (score, reasons)

        return results

    def _score_structure_batch(
        self,
        file_paths: List[Path],
        keywords: List[str]
    ) -> Dict[Path, Tuple[float, List[str]]]:
        """Score files based on code structure relevance."""
        results: Dict[Path, Tuple[float, List[str]]] = {}

        if not self.repo_map or not self.repo_map._built:
            return results

        for file_path in file_paths:
            if file_path not in self.repo_map.modules:
                results[file_path] = (0.0, [])
                continue

            module = self.repo_map.modules[file_path]
            score = 0.0
            reasons: List[str] = []
            max_possible = len(keywords) if keywords else 1

            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Check class names
                for cls in module.classes:
                    if keyword_lower in cls.name.lower():
                        score += 1.0
                        reasons.append(f"class '{cls.name}'")
                    # Check method names
                    for method in cls.methods:
                        if keyword_lower in method.name.lower():
                            score += 0.5
                            if len(reasons) < 5:
                                reasons.append(f"method '{cls.name}.{method.name}'")

                # Check function names
                for func in module.functions:
                    if keyword_lower in func.name.lower():
                        score += 0.8
                        reasons.append(f"function '{func.name}'")

                # Check imports
                for imp in module.imports:
                    if keyword_lower in imp.lower():
                        score += 0.3

            normalized_score = min(score / max_possible, 1.0) if max_possible > 0 else 0.0
            results[file_path] = (normalized_score, reasons[:5])

        return results

    async def _score_semantic_batch(
        self,
        file_paths: List[Path],
        query: str
    ) -> Dict[Path, float]:
        """Score files based on semantic similarity."""
        results: Dict[Path, float] = {}

        if not self.embedding_fn:
            return results

        try:
            # Get query embedding
            query_embedding = await asyncio.to_thread(self.embedding_fn, query)

            for file_path in file_paths:
                if file_path in self._file_embeddings:
                    file_embedding = self._file_embeddings[file_path]
                else:
                    # Generate embedding for file content
                    try:
                        content = await asyncio.to_thread(
                            lambda: file_path.read_text(encoding="utf-8", errors="ignore")
                        )
                        # Use first 8000 chars for embedding
                        file_embedding = await asyncio.to_thread(
                            self.embedding_fn, content[:8000]
                        )
                        self._file_embeddings[file_path] = file_embedding
                    except Exception:
                        results[file_path] = 0.0
                        continue

                # Calculate cosine similarity
                score = self._cosine_similarity(query_embedding, file_embedding)
                results[file_path] = max(0.0, score)

        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e}")

        return results

    async def _score_recency_batch(
        self,
        file_paths: List[Path],
        config: SelectionConfig
    ) -> Dict[Path, float]:
        """Score files based on modification recency."""
        results: Dict[Path, float] = {}
        now = datetime.now().timestamp()
        max_age_seconds = config.recency_days * 24 * 60 * 60

        async def get_mtime(file_path: Path) -> Tuple[Path, float]:
            try:
                mtime = await asyncio.to_thread(lambda: file_path.stat().st_mtime)
                age = now - mtime
                if age <= 0:
                    return (file_path, 1.0)
                elif age >= max_age_seconds:
                    return (file_path, 0.0)
                else:
                    # Linear decay
                    return (file_path, 1.0 - (age / max_age_seconds))
            except Exception:
                return (file_path, 0.0)

        tasks = [get_mtime(fp) for fp in file_paths]
        scored = await asyncio.gather(*tasks, return_exceptions=True)

        for result in scored:
            if isinstance(result, Exception):
                continue
            file_path, score = result
            results[file_path] = score

        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def index_embeddings(self, batch_size: int = 50) -> None:
        """
        Pre-compute embeddings for all files.

        Call this to improve semantic search performance.
        """
        if not self.embedding_fn:
            logger.warning("No embedding function provided, skipping indexing")
            return

        candidates = await self._get_candidate_files()

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            for file_path in batch:
                if file_path in self._file_embeddings:
                    continue

                try:
                    content = await asyncio.to_thread(
                        lambda fp=file_path: fp.read_text(encoding="utf-8", errors="ignore")
                    )
                    embedding = await asyncio.to_thread(
                        self.embedding_fn, content[:8000]
                    )
                    self._file_embeddings[file_path] = embedding
                except Exception as e:
                    logger.debug(f"Failed to embed {file_path}: {e}")

            logger.info(f"Indexed {min(i + batch_size, len(candidates))}/{len(candidates)} files")
