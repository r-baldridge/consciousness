"""Search tools for finding content in files and directories.

This module provides tools for searching file contents using various
methods including regex, grep, and semantic search with embeddings.
"""

from __future__ import annotations
import asyncio
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field

from ..tool_base import Tool, ToolSchema, ToolResult, ToolPermission


@dataclass
class SearchMatch:
    """A single search match result.

    Attributes:
        file: Path to the file containing the match
        line_number: Line number of the match (1-indexed)
        line: The full line containing the match
        match_start: Start column of the match
        match_end: End column of the match
        context_before: Lines before the match
        context_after: Lines after the match
    """
    file: str
    line_number: int
    line: str
    match_start: int = 0
    match_end: int = 0
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file": self.file,
            "line_number": self.line_number,
            "line": self.line,
            "match_start": self.match_start,
            "match_end": self.match_end,
            "context_before": self.context_before,
            "context_after": self.context_after
        }


class GrepTool(Tool):
    """Tool for searching file contents with regex patterns."""

    def __init__(
        self,
        base_path: Optional[Path] = None,
        max_results: int = 100,
        max_file_size: int = 10 * 1024 * 1024  # 10MB
    ):
        """Initialize the grep tool.

        Args:
            base_path: Base path to restrict searches to
            max_results: Maximum number of results to return
            max_file_size: Maximum file size to search
        """
        self._base_path = base_path or Path.cwd()
        self._max_results = max_results
        self._max_file_size = max_file_size

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description="Search for patterns in files using regex.",
            parameters={
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in"
                },
                "glob": {
                    "type": "string",
                    "description": "File pattern to filter (e.g., '*.py')"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines to include"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files in search"
                }
            },
            required=["pattern"],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        case_insensitive: bool = False,
        context_lines: int = 0,
        max_results: Optional[int] = None,
        include_hidden: bool = False
    ) -> ToolResult:
        """Search for a pattern in files.

        Args:
            pattern: Regex pattern to search for
            path: Path to search in
            glob: File pattern filter
            case_insensitive: Case-insensitive matching
            context_lines: Context lines to include
            max_results: Maximum results
            include_hidden: Include hidden files

        Returns:
            ToolResult with search matches
        """
        try:
            # Compile regex
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return ToolResult.fail(f"Invalid regex pattern: {e}")

            # Determine search path
            search_path = Path(path).expanduser().resolve() if path else self._base_path

            if not search_path.exists():
                return ToolResult.fail(f"Path not found: {search_path}")

            # Collect files to search
            files_to_search: List[Path] = []

            if search_path.is_file():
                files_to_search = [search_path]
            else:
                # Get matching files
                pattern_glob = glob or "**/*"
                for file_path in search_path.glob(pattern_glob):
                    if not file_path.is_file():
                        continue

                    # Skip hidden files unless requested
                    if not include_hidden:
                        parts = file_path.relative_to(search_path).parts
                        if any(p.startswith('.') for p in parts):
                            continue

                    # Skip large files
                    if file_path.stat().st_size > self._max_file_size:
                        continue

                    files_to_search.append(file_path)

            # Search files
            matches: List[SearchMatch] = []
            result_limit = max_results or self._max_results
            files_searched = 0
            files_with_matches = 0

            for file_path in files_to_search:
                file_matches = await self._search_file(
                    file_path, regex, context_lines
                )

                if file_matches:
                    files_with_matches += 1
                    matches.extend(file_matches)

                files_searched += 1

                if len(matches) >= result_limit:
                    break

            # Truncate results
            truncated = len(matches) > result_limit
            matches = matches[:result_limit]

            return ToolResult.ok(
                [m.to_dict() for m in matches],
                files_searched=files_searched,
                files_with_matches=files_with_matches,
                total_matches=len(matches),
                truncated=truncated,
                pattern=pattern
            )

        except Exception as e:
            return ToolResult.fail(f"Search error: {e}")

    async def _search_file(
        self,
        file_path: Path,
        regex: re.Pattern,
        context_lines: int
    ) -> List[SearchMatch]:
        """Search a single file for matches.

        Args:
            file_path: Path to the file
            regex: Compiled regex pattern
            context_lines: Context lines to include

        Returns:
            List of matches found
        """
        matches = []

        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
            lines = content.split('\n')

            for i, line in enumerate(lines):
                match = regex.search(line)
                if match:
                    # Get context
                    start_ctx = max(0, i - context_lines)
                    end_ctx = min(len(lines), i + context_lines + 1)

                    matches.append(SearchMatch(
                        file=str(file_path),
                        line_number=i + 1,
                        line=line,
                        match_start=match.start(),
                        match_end=match.end(),
                        context_before=lines[start_ctx:i],
                        context_after=lines[i+1:end_ctx]
                    ))

        except Exception:
            pass  # Skip files that can't be read

        return matches


class RipgrepTool(Tool):
    """Tool for fast file search using ripgrep (rg) if available."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the ripgrep tool.

        Args:
            base_path: Base path for searches
        """
        self._base_path = base_path or Path.cwd()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="ripgrep",
            description="Fast file search using ripgrep. Falls back to built-in search if rg not installed.",
            parameters={
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in"
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (e.g., 'py', 'js')"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern for files"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search"
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context (-C)"
                },
                "max_count": {
                    "type": "integer",
                    "description": "Maximum matches per file"
                },
                "hidden": {
                    "type": "boolean",
                    "description": "Search hidden files"
                }
            },
            required=["pattern"],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        type: Optional[str] = None,
        glob: Optional[str] = None,
        case_insensitive: bool = False,
        context: int = 0,
        max_count: Optional[int] = None,
        hidden: bool = False
    ) -> ToolResult:
        """Search using ripgrep.

        Args:
            pattern: Search pattern
            path: Search path
            type: File type filter
            glob: Glob pattern
            case_insensitive: Case-insensitive
            context: Context lines
            max_count: Max matches per file
            hidden: Include hidden files

        Returns:
            ToolResult with search results
        """
        search_path = Path(path).expanduser().resolve() if path else self._base_path

        if not search_path.exists():
            return ToolResult.fail(f"Path not found: {search_path}")

        # Build ripgrep command
        args = ["rg", "--json"]

        if case_insensitive:
            args.append("-i")
        if context > 0:
            args.extend(["-C", str(context)])
        if max_count:
            args.extend(["-m", str(max_count)])
        if hidden:
            args.append("--hidden")
        if type:
            args.extend(["-t", type])
        if glob:
            args.extend(["-g", glob])

        args.append(pattern)
        args.append(str(search_path))

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=60.0
            )

            if proc.returncode not in (0, 1):  # 1 = no matches, which is OK
                return ToolResult.fail(f"ripgrep error: {stderr.decode()}")

            # Parse JSON output
            import json
            matches = []
            for line in stdout.decode().strip().split('\n'):
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "match":
                        data = entry["data"]
                        matches.append({
                            "file": data["path"]["text"],
                            "line_number": data["line_number"],
                            "line": data["lines"]["text"].rstrip('\n')
                        })
                except json.JSONDecodeError:
                    continue

            return ToolResult.ok(
                matches,
                count=len(matches),
                pattern=pattern
            )

        except FileNotFoundError:
            # ripgrep not installed, fall back to GrepTool
            grep_tool = GrepTool(self._base_path)
            return await grep_tool.execute(
                pattern=pattern,
                path=path,
                glob=glob,
                case_insensitive=case_insensitive,
                context_lines=context,
                include_hidden=hidden
            )
        except asyncio.TimeoutError:
            return ToolResult.fail("Search timed out")
        except Exception as e:
            return ToolResult.fail(f"Search error: {e}")


# Type for embedding function
EmbeddingFunc = Callable[[str], Awaitable[List[float]]]


class SemanticSearchTool(Tool):
    """Tool for semantic search using embeddings.

    This tool requires an embedding function to be provided for
    converting text to vector representations.
    """

    def __init__(
        self,
        embedding_func: Optional[EmbeddingFunc] = None,
        base_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize the semantic search tool.

        Args:
            embedding_func: Async function to compute embeddings
            base_path: Base path for file searches
            index_path: Path to store/load the search index
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self._embedding_func = embedding_func
        self._base_path = base_path or Path.cwd()
        self._index_path = index_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # In-memory index
        self._index: Dict[str, Dict[str, Any]] = {}

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="semantic_search",
            description="Search for semantically similar content using embeddings.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Natural language query"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in"
                },
                "glob": {
                    "type": "string",
                    "description": "File pattern to filter"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return"
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum similarity score (0-1)"
                },
                "rebuild_index": {
                    "type": "boolean",
                    "description": "Force rebuild of the search index"
                }
            },
            required=["query"],
            permissions=[ToolPermission.READ, ToolPermission.NETWORK]
        )

    async def execute(
        self,
        query: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.5,
        rebuild_index: bool = False
    ) -> ToolResult:
        """Perform semantic search.

        Args:
            query: Natural language query
            path: Search path
            glob: File pattern
            top_k: Number of results
            threshold: Minimum similarity
            rebuild_index: Force index rebuild

        Returns:
            ToolResult with semantically similar content
        """
        if self._embedding_func is None:
            return ToolResult.fail(
                "Semantic search requires an embedding function. "
                "Initialize the tool with embedding_func parameter."
            )

        search_path = Path(path).expanduser().resolve() if path else self._base_path

        if not search_path.exists():
            return ToolResult.fail(f"Path not found: {search_path}")

        try:
            # Build or load index
            if rebuild_index or not self._index:
                await self._build_index(search_path, glob)

            if not self._index:
                return ToolResult.ok(
                    [],
                    message="No content indexed"
                )

            # Get query embedding
            query_embedding = await self._embedding_func(query)

            # Search index
            results = []
            for chunk_id, chunk_data in self._index.items():
                similarity = self._cosine_similarity(
                    query_embedding,
                    chunk_data["embedding"]
                )

                if similarity >= threshold:
                    results.append({
                        "file": chunk_data["file"],
                        "chunk": chunk_data["text"],
                        "line_start": chunk_data["line_start"],
                        "similarity": similarity
                    })

            # Sort by similarity
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:top_k]

            return ToolResult.ok(
                results,
                query=query,
                indexed_chunks=len(self._index)
            )

        except Exception as e:
            return ToolResult.fail(f"Semantic search error: {e}")

    async def _build_index(
        self,
        search_path: Path,
        glob_pattern: Optional[str] = None
    ) -> None:
        """Build the search index.

        Args:
            search_path: Path to index
            glob_pattern: File pattern filter
        """
        self._index.clear()

        pattern = glob_pattern or "**/*"
        files = list(search_path.glob(pattern))

        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text(encoding='utf-8', errors='replace')
                chunks = self._chunk_text(content)

                for i, (chunk_text, line_start) in enumerate(chunks):
                    chunk_id = f"{file_path}:{i}"
                    embedding = await self._embedding_func(chunk_text)

                    self._index[chunk_id] = {
                        "file": str(file_path),
                        "text": chunk_text,
                        "line_start": line_start,
                        "embedding": embedding
                    }

            except Exception:
                continue

    def _chunk_text(
        self,
        text: str
    ) -> List[tuple[str, int]]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of (chunk_text, starting_line_number) tuples
        """
        lines = text.split('\n')
        chunks = []

        current_chunk = []
        current_length = 0
        chunk_start_line = 1

        for i, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for newline

            if current_length + line_length > self._chunk_size and current_chunk:
                # Save current chunk
                chunks.append(('\n'.join(current_chunk), chunk_start_line))

                # Calculate overlap
                overlap_lines = []
                overlap_length = 0
                for prev_line in reversed(current_chunk):
                    if overlap_length + len(prev_line) + 1 > self._chunk_overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_length += len(prev_line) + 1

                current_chunk = overlap_lines
                current_length = overlap_length
                chunk_start_line = i + 1 - len(overlap_lines)

            current_chunk.append(line)
            current_length += line_length

        # Save final chunk
        if current_chunk:
            chunks.append(('\n'.join(current_chunk), chunk_start_line))

        return chunks

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# Convenience function to get search tools
def get_search_tools(
    base_path: Optional[Path] = None,
    embedding_func: Optional[EmbeddingFunc] = None
) -> List[Tool]:
    """Get all search tools.

    Args:
        base_path: Base path for searches
        embedding_func: Embedding function for semantic search

    Returns:
        List of search tool instances
    """
    tools = [
        GrepTool(base_path=base_path),
        RipgrepTool(base_path=base_path),
    ]

    if embedding_func:
        tools.append(SemanticSearchTool(
            embedding_func=embedding_func,
            base_path=base_path
        ))

    return tools
