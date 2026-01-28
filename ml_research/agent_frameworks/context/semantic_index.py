"""
Semantic Index - LSP-Powered Semantic Search.

Provides semantic search capabilities for code using:
- Embedding-based similarity search
- Optional LSP integration for go-to-definition and find-references
- Smart chunking strategies for large files

Example:
    index = SemanticIndex(Path("/path/to/repo"))

    # Index the repository
    await index.index_directory(Path("src/"))

    # Search for relevant code
    results = await index.search("user authentication logic", limit=10)
    for result in results:
        print(f"{result.file_path}:{result.line_number} - {result.score}")

    # With LSP integration
    index = SemanticIndex(root_path, lsp_client=my_lsp_client)
    definitions = await index.find_definitions("authenticate")
"""

from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Set, Tuple, Callable, Any, Protocol, AsyncIterator
)
from pathlib import Path
from enum import Enum
import asyncio
import logging
import re
import hashlib
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Strategies for chunking code for embeddings."""
    FIXED_SIZE = "fixed_size"  # Fixed character count chunks
    LINE_BASED = "line_based"  # Fixed line count chunks
    SEMANTIC = "semantic"  # Chunk by semantic boundaries (functions, classes)
    SLIDING_WINDOW = "sliding_window"  # Overlapping windows
    HYBRID = "hybrid"  # Combination of semantic + sliding window


@dataclass
class CodeChunk:
    """Represents a chunk of code for indexing."""
    file_path: Path
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "block", "file"
    symbol_name: Optional[str] = None
    embedding: Optional[List[float]] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """Represents a semantic search result."""
    file_path: Path
    line_number: int
    end_line: int
    content: str
    score: float
    chunk_type: str
    symbol_name: Optional[str] = None
    match_reason: str = ""

    def preview(self, max_lines: int = 5) -> str:
        """Get a preview of the matched content."""
        lines = self.content.splitlines()[:max_lines]
        if len(self.content.splitlines()) > max_lines:
            lines.append("...")
        return "\n".join(lines)


@dataclass
class LSPLocation:
    """Represents a location from LSP."""
    file_path: Path
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None


class LSPClient(Protocol):
    """Protocol for LSP client implementations."""

    async def find_definition(self, file_path: Path, line: int, column: int) -> List[LSPLocation]:
        """Find definition at position."""
        ...

    async def find_references(self, file_path: Path, line: int, column: int) -> List[LSPLocation]:
        """Find all references to symbol at position."""
        ...

    async def get_hover(self, file_path: Path, line: int, column: int) -> Optional[str]:
        """Get hover information at position."""
        ...

    async def get_symbols(self, file_path: Path) -> List[Dict[str, Any]]:
        """Get all symbols in a file."""
        ...


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    Simple embedding provider using TF-IDF-like features.

    This is a fallback when no external embedding model is available.
    Not as powerful as neural embeddings but works without dependencies.
    """

    def __init__(self, dimension: int = 256):
        self._dimension = dimension
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> List[float]:
        """Generate a simple embedding using hashed n-grams."""
        return await asyncio.to_thread(self._compute_embedding, text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding using feature hashing."""
        import math

        embedding = [0.0] * self._dimension

        # Tokenize
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Add character n-grams for robustness
        char_ngrams = []
        for token in tokens:
            for n in [2, 3]:
                for i in range(len(token) - n + 1):
                    char_ngrams.append(token[i:i+n])

        all_features = tokens + char_ngrams

        # Feature hashing
        for feature in all_features:
            # Hash to bucket
            h = hash(feature)
            bucket = h % self._dimension
            # Determine sign
            sign = 1 if (h // self._dimension) % 2 == 0 else -1
            embedding[bucket] += sign

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding


class SemanticIndex:
    """
    Semantic search index for code.

    Provides embedding-based semantic search with optional LSP integration
    for code intelligence features like go-to-definition.

    Example:
        # Basic usage
        index = SemanticIndex(Path("/path/to/repo"))
        await index.index_directory(Path("src/"))
        results = await index.search("authentication", limit=5)

        # With custom embedding provider
        index = SemanticIndex(root_path, embedding_provider=my_provider)

        # With LSP
        index = SemanticIndex(root_path, lsp_client=my_lsp)
        defs = await index.find_definitions("UserService")
    """

    # File extensions to index
    SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"}

    # Directories to skip
    SKIP_DIRS = {
        "__pycache__", ".git", "node_modules", "venv", ".venv",
        "build", "dist", ".tox", ".pytest_cache", "target"
    }

    def __init__(
        self,
        root_path: Path,
        embedding_provider: Optional[EmbeddingProvider] = None,
        lsp_client: Optional[LSPClient] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    ):
        """
        Initialize the semantic index.

        Args:
            root_path: Root directory of the codebase.
            embedding_provider: Provider for text embeddings.
            lsp_client: Optional LSP client for code intelligence.
            chunk_size: Target size for code chunks (characters).
            chunk_overlap: Overlap between chunks (for sliding window).
            chunking_strategy: Strategy for splitting code.
        """
        self.root_path = Path(root_path).resolve()
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self.lsp_client = lsp_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        # Index storage
        self._chunks: Dict[str, CodeChunk] = {}  # hash -> chunk
        self._file_chunks: Dict[Path, List[str]] = {}  # file -> chunk hashes
        self._embeddings: Dict[str, List[float]] = {}  # hash -> embedding

        # Symbol index for fast lookup
        self._symbols: Dict[str, List[str]] = {}  # symbol name -> chunk hashes

        self._indexed = False

    async def index_file(self, file_path: Path) -> int:
        """
        Index a single file.

        Args:
            file_path: Path to the file to index.

        Returns:
            Number of chunks indexed.
        """
        file_path = Path(file_path).resolve()

        if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
            return 0

        try:
            content = await asyncio.to_thread(
                file_path.read_text, encoding="utf-8", errors="replace"
            )
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return 0

        # Generate chunks
        chunks = await self._chunk_content(file_path, content)

        # Generate embeddings
        chunk_hashes = []
        for chunk in chunks:
            self._chunks[chunk.content_hash] = chunk
            chunk_hashes.append(chunk.content_hash)

            # Index symbols
            if chunk.symbol_name:
                if chunk.symbol_name not in self._symbols:
                    self._symbols[chunk.symbol_name] = []
                self._symbols[chunk.symbol_name].append(chunk.content_hash)

        self._file_chunks[file_path] = chunk_hashes

        # Batch embed chunks
        contents = [c.content for c in chunks]
        if contents:
            embeddings = await self.embedding_provider.embed_batch(contents)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                self._embeddings[chunk.content_hash] = embedding

        return len(chunks)

    async def index_directory(
        self,
        dir_path: Optional[Path] = None,
        extensions: Optional[Set[str]] = None,
        max_files: int = 10000
    ) -> int:
        """
        Index all files in a directory.

        Args:
            dir_path: Directory to index (defaults to root_path).
            extensions: File extensions to include.
            max_files: Maximum number of files to index.

        Returns:
            Total number of chunks indexed.
        """
        if dir_path is None:
            dir_path = self.root_path
        else:
            dir_path = Path(dir_path).resolve()

        if extensions is None:
            extensions = self.SUPPORTED_EXTENSIONS

        # Collect files
        files_to_index: List[Path] = []

        for ext in extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            for file_path in dir_path.glob(f"**/*{ext}"):
                # Skip ignored directories
                if any(part in self.SKIP_DIRS for part in file_path.parts):
                    continue

                files_to_index.append(file_path)

                if len(files_to_index) >= max_files:
                    break

            if len(files_to_index) >= max_files:
                break

        # Index files concurrently
        semaphore = asyncio.Semaphore(20)

        async def index_with_limit(fp: Path) -> int:
            async with semaphore:
                return await self.index_file(fp)

        results = await asyncio.gather(
            *[index_with_limit(fp) for fp in files_to_index],
            return_exceptions=True
        )

        total_chunks = sum(r for r in results if isinstance(r, int))
        self._indexed = True

        logger.info(
            f"Indexed {len(files_to_index)} files, {total_chunks} chunks"
        )

        return total_chunks

    async def search(
        self,
        query: str,
        limit: int = 10,
        file_filter: Optional[Callable[[Path], bool]] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for relevant code using semantic similarity.

        Args:
            query: Natural language query.
            limit: Maximum number of results.
            file_filter: Optional filter function for files.
            min_score: Minimum similarity score (0-1).

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not self._indexed:
            logger.warning("Index not built. Call index_directory() first.")
            return []

        # Get query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Score all chunks
        scored: List[Tuple[str, float]] = []

        for chunk_hash, embedding in self._embeddings.items():
            chunk = self._chunks[chunk_hash]

            # Apply file filter
            if file_filter and not file_filter(chunk.file_path):
                continue

            score = self._cosine_similarity(query_embedding, embedding)
            if score >= min_score:
                scored.append((chunk_hash, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results: List[SearchResult] = []
        seen_files: Dict[Path, int] = {}  # Limit results per file

        for chunk_hash, score in scored[:limit * 3]:  # Get extra for deduplication
            chunk = self._chunks[chunk_hash]

            # Limit 2 results per file
            file_count = seen_files.get(chunk.file_path, 0)
            if file_count >= 2:
                continue
            seen_files[chunk.file_path] = file_count + 1

            results.append(SearchResult(
                file_path=chunk.file_path,
                line_number=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                score=score,
                chunk_type=chunk.chunk_type,
                symbol_name=chunk.symbol_name,
                match_reason=f"Semantic similarity: {score:.3f}"
            ))

            if len(results) >= limit:
                break

        return results

    async def search_symbol(self, symbol_name: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for a specific symbol by name.

        Args:
            symbol_name: Name of the symbol to find.
            limit: Maximum number of results.

        Returns:
            List of SearchResult objects.
        """
        results: List[SearchResult] = []
        symbol_lower = symbol_name.lower()

        # Exact matches from symbol index
        if symbol_name in self._symbols:
            for chunk_hash in self._symbols[symbol_name][:limit]:
                chunk = self._chunks[chunk_hash]
                results.append(SearchResult(
                    file_path=chunk.file_path,
                    line_number=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=1.0,
                    chunk_type=chunk.chunk_type,
                    symbol_name=chunk.symbol_name,
                    match_reason="Exact symbol match"
                ))

        # Partial matches
        for name, chunk_hashes in self._symbols.items():
            if symbol_lower in name.lower() and name != symbol_name:
                for chunk_hash in chunk_hashes[:2]:
                    if len(results) >= limit:
                        break
                    chunk = self._chunks[chunk_hash]
                    results.append(SearchResult(
                        file_path=chunk.file_path,
                        line_number=chunk.start_line,
                        end_line=chunk.end_line,
                        content=chunk.content,
                        score=0.8,
                        chunk_type=chunk.chunk_type,
                        symbol_name=chunk.symbol_name,
                        match_reason=f"Partial match: {name}"
                    ))

        return results[:limit]

    async def find_definitions(self, symbol: str) -> List[LSPLocation]:
        """
        Find definitions of a symbol using LSP.

        Args:
            symbol: Symbol name to find.

        Returns:
            List of LSPLocation objects.
        """
        if not self.lsp_client:
            logger.warning("No LSP client configured")
            return []

        # Find symbol in index
        symbol_results = await self.search_symbol(symbol, limit=5)

        locations: List[LSPLocation] = []

        for result in symbol_results:
            try:
                defs = await self.lsp_client.find_definition(
                    result.file_path,
                    result.line_number,
                    0
                )
                locations.extend(defs)
            except Exception as e:
                logger.debug(f"LSP definition lookup failed: {e}")

        return locations

    async def find_references(self, symbol: str) -> List[LSPLocation]:
        """
        Find all references to a symbol using LSP.

        Args:
            symbol: Symbol name to find references for.

        Returns:
            List of LSPLocation objects.
        """
        if not self.lsp_client:
            logger.warning("No LSP client configured")
            return []

        # Find symbol in index
        symbol_results = await self.search_symbol(symbol, limit=1)

        if not symbol_results:
            return []

        result = symbol_results[0]

        try:
            return await self.lsp_client.find_references(
                result.file_path,
                result.line_number,
                0
            )
        except Exception as e:
            logger.warning(f"LSP reference lookup failed: {e}")
            return []

    async def _chunk_content(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk file content based on the configured strategy."""
        if self.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            return await self._chunk_fixed_size(file_path, content)
        elif self.chunking_strategy == ChunkingStrategy.LINE_BASED:
            return await self._chunk_by_lines(file_path, content)
        elif self.chunking_strategy == ChunkingStrategy.SEMANTIC:
            return await self._chunk_semantic(file_path, content)
        elif self.chunking_strategy == ChunkingStrategy.SLIDING_WINDOW:
            return await self._chunk_sliding_window(file_path, content)
        elif self.chunking_strategy == ChunkingStrategy.HYBRID:
            return await self._chunk_hybrid(file_path, content)
        else:
            return await self._chunk_fixed_size(file_path, content)

    async def _chunk_fixed_size(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk by fixed character count."""
        chunks: List[CodeChunk] = []
        lines = content.splitlines(keepends=True)

        current_chunk = ""
        current_start = 1
        current_line = 1

        for i, line in enumerate(lines, 1):
            if len(current_chunk) + len(line) > self.chunk_size and current_chunk:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=current_chunk,
                    start_line=current_start,
                    end_line=current_line - 1,
                    chunk_type="block"
                ))
                current_chunk = ""
                current_start = i

            current_chunk += line
            current_line = i

        if current_chunk:
            chunks.append(CodeChunk(
                file_path=file_path,
                content=current_chunk,
                start_line=current_start,
                end_line=current_line,
                chunk_type="block"
            ))

        return chunks

    async def _chunk_by_lines(
        self,
        file_path: Path,
        content: str,
        lines_per_chunk: int = 50
    ) -> List[CodeChunk]:
        """Chunk by line count."""
        chunks: List[CodeChunk] = []
        lines = content.splitlines(keepends=True)

        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunks.append(CodeChunk(
                file_path=file_path,
                content="".join(chunk_lines),
                start_line=i + 1,
                end_line=min(i + lines_per_chunk, len(lines)),
                chunk_type="block"
            ))

        return chunks

    async def _chunk_semantic(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk by semantic boundaries (functions, classes)."""
        chunks: List[CodeChunk] = []
        ext = file_path.suffix

        if ext == ".py":
            chunks = await self._chunk_python_semantic(file_path, content)
        elif ext in (".js", ".jsx", ".ts", ".tsx"):
            chunks = await self._chunk_js_semantic(file_path, content)
        else:
            # Fallback to line-based
            chunks = await self._chunk_by_lines(file_path, content)

        return chunks

    async def _chunk_python_semantic(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk Python code by semantic units."""
        import ast

        chunks: List[CodeChunk] = []
        lines = content.splitlines(keepends=True)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return await self._chunk_by_lines(file_path, content)

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = node.end_lineno or start

                chunk_content = "".join(lines[start - 1:end])
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=start,
                    end_line=end,
                    chunk_type="function",
                    symbol_name=node.name
                ))

            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or start

                chunk_content = "".join(lines[start - 1:end])
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=start,
                    end_line=end,
                    chunk_type="class",
                    symbol_name=node.name
                ))

        # If no semantic chunks found, fall back
        if not chunks:
            return await self._chunk_by_lines(file_path, content)

        return chunks

    async def _chunk_js_semantic(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk JavaScript/TypeScript code by semantic units."""
        chunks: List[CodeChunk] = []
        lines = content.splitlines(keepends=True)

        # Regex patterns for functions and classes
        patterns = [
            # Function declarations
            (r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)', "function"),
            # Arrow functions
            (r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', "function"),
            # Class declarations
            (r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', "class"),
            # Method definitions (rough)
            (r'^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', "method"),
        ]

        current_chunk_start = 0
        current_chunk_type = "block"
        current_symbol = None
        brace_depth = 0

        for i, line in enumerate(lines):
            # Check for pattern matches
            for pattern, chunk_type in patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous chunk if exists
                    if i > current_chunk_start:
                        chunk_content = "".join(lines[current_chunk_start:i])
                        if chunk_content.strip():
                            chunks.append(CodeChunk(
                                file_path=file_path,
                                content=chunk_content,
                                start_line=current_chunk_start + 1,
                                end_line=i,
                                chunk_type=current_chunk_type,
                                symbol_name=current_symbol
                            ))

                    current_chunk_start = i
                    current_chunk_type = chunk_type
                    current_symbol = match.group(1)
                    break

            # Track brace depth for scope
            brace_depth += line.count('{') - line.count('}')

        # Add final chunk
        if current_chunk_start < len(lines):
            chunk_content = "".join(lines[current_chunk_start:])
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=current_chunk_start + 1,
                    end_line=len(lines),
                    chunk_type=current_chunk_type,
                    symbol_name=current_symbol
                ))

        return chunks if chunks else await self._chunk_by_lines(file_path, content)

    async def _chunk_sliding_window(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk with overlapping sliding windows."""
        chunks: List[CodeChunk] = []
        lines = content.splitlines(keepends=True)

        window_lines = self.chunk_size // 50  # Approximate lines
        overlap_lines = self.chunk_overlap // 50

        i = 0
        while i < len(lines):
            end = min(i + window_lines, len(lines))
            chunk_content = "".join(lines[i:end])

            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=i + 1,
                end_line=end,
                chunk_type="block"
            ))

            i += window_lines - overlap_lines
            if i + overlap_lines >= len(lines):
                break

        return chunks

    async def _chunk_hybrid(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Hybrid chunking: semantic for small units, sliding window for large."""
        # First try semantic
        semantic_chunks = await self._chunk_semantic(file_path, content)

        final_chunks: List[CodeChunk] = []

        for chunk in semantic_chunks:
            if len(chunk.content) <= self.chunk_size * 1.5:
                final_chunks.append(chunk)
            else:
                # Split large chunks with sliding window
                sub_chunks = await self._chunk_sliding_window(
                    file_path, chunk.content
                )
                # Adjust line numbers
                for sub in sub_chunks:
                    sub.start_line += chunk.start_line - 1
                    sub.end_line += chunk.start_line - 1
                    sub.symbol_name = chunk.symbol_name
                final_chunks.extend(sub_chunks)

        return final_chunks

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

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_chunks": len(self._chunks),
            "total_files": len(self._file_chunks),
            "total_symbols": len(self._symbols),
            "indexed": self._indexed,
            "embedding_dimension": self.embedding_provider.dimension,
            "chunking_strategy": self.chunking_strategy.value,
        }

    async def remove_file(self, file_path: Path) -> None:
        """Remove a file from the index."""
        file_path = Path(file_path).resolve()

        if file_path not in self._file_chunks:
            return

        # Remove chunks
        for chunk_hash in self._file_chunks[file_path]:
            chunk = self._chunks.pop(chunk_hash, None)
            self._embeddings.pop(chunk_hash, None)

            # Remove from symbol index
            if chunk and chunk.symbol_name:
                if chunk.symbol_name in self._symbols:
                    self._symbols[chunk.symbol_name] = [
                        h for h in self._symbols[chunk.symbol_name]
                        if h != chunk_hash
                    ]

        del self._file_chunks[file_path]

    async def update_file(self, file_path: Path) -> int:
        """Update the index for a file."""
        await self.remove_file(file_path)
        return await self.index_file(file_path)

    def clear(self) -> None:
        """Clear the entire index."""
        self._chunks.clear()
        self._file_chunks.clear()
        self._embeddings.clear()
        self._symbols.clear()
        self._indexed = False
