"""
Vector-based semantic memory for retrieval.

This module provides semantic memory storage using vector embeddings,
enabling similarity-based retrieval of past information.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json
import uuid
import math
import logging
import asyncio

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single entry in semantic memory.

    Attributes:
        id: Unique identifier for the entry
        content: The text content of the memory
        embedding: Vector embedding of the content
        metadata: Additional data about the entry
        timestamp: When the entry was created
    """
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create entry from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )


@dataclass
class SearchResult:
    """Result from a semantic search.

    Attributes:
        entry: The matching memory entry
        score: Similarity score (0-1, higher is better)
        distance: Raw distance metric
    """
    entry: MemoryEntry
    score: float
    distance: float


class SemanticMemory:
    """Vector-based semantic memory for retrieval.

    This class stores text entries with their vector embeddings and
    provides similarity-based search capabilities.

    Example:
        memory = SemanticMemory(backend, dimension=1536)

        # Add memories
        await memory.add("User prefers Python over JavaScript")
        await memory.add("Project uses PostgreSQL database", {"type": "architecture"})

        # Search for relevant memories
        results = await memory.search("What programming language?", limit=3)
        for result in results:
            print(f"{result.score:.2f}: {result.entry.content}")
    """

    def __init__(
        self,
        backend: Optional["LLMBackend"] = None,
        dimension: int = 1536
    ):
        """Initialize semantic memory.

        Args:
            backend: LLM backend for generating embeddings
            dimension: Embedding vector dimension
        """
        self.backend = backend
        self.dimension = dimension
        self.entries: List[MemoryEntry] = []
        self._id_index: Dict[str, int] = {}  # id -> index mapping

    async def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        entry_id: Optional[str] = None
    ) -> MemoryEntry:
        """Add a new entry to semantic memory.

        Args:
            content: The text content to store
            metadata: Optional metadata about the entry
            entry_id: Optional custom ID (generated if not provided)

        Returns:
            The created MemoryEntry
        """
        entry_id = entry_id or str(uuid.uuid4())

        # Generate embedding
        embedding = await self._get_embedding(content)

        entry = MemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now()
        )

        # Update index
        index = len(self.entries)
        self.entries.append(entry)
        self._id_index[entry_id] = index

        logger.debug(f"Added memory entry: {entry_id}")
        return entry

    async def add_many(
        self,
        contents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[MemoryEntry]:
        """Add multiple entries efficiently.

        Args:
            contents: List of text contents to store
            metadata_list: Optional list of metadata dicts

        Returns:
            List of created MemoryEntry objects
        """
        if metadata_list is None:
            metadata_list = [{}] * len(contents)

        entries = []
        for content, metadata in zip(contents, metadata_list):
            entry = await self.add(content, metadata)
            entries.append(entry)

        return entries

    async def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for relevant memories.

        Args:
            query: The search query
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)
            metadata_filter: Optional filter on metadata fields

        Returns:
            List of SearchResult objects, ordered by relevance
        """
        if not self.entries:
            return []

        # Get query embedding
        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            # Fallback to text search
            return self._text_search(query, limit, metadata_filter)

        # Calculate similarities
        results: List[Tuple[float, MemoryEntry]] = []

        for entry in self.entries:
            # Apply metadata filter
            if metadata_filter:
                if not all(entry.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            if entry.embedding:
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity >= threshold:
                    results.append((similarity, entry))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[0], reverse=True)

        # Convert to SearchResult objects
        return [
            SearchResult(
                entry=entry,
                score=score,
                distance=1.0 - score
            )
            for score, entry in results[:limit]
        ]

    def _text_search(
        self,
        query: str,
        limit: int,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Fallback text-based search when embeddings unavailable.

        Args:
            query: Search query
            limit: Maximum results
            metadata_filter: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        query_lower = query.lower()
        results: List[Tuple[float, MemoryEntry]] = []

        for entry in self.entries:
            # Apply metadata filter
            if metadata_filter:
                if not all(entry.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            # Simple text matching score
            content_lower = entry.content.lower()
            score = 0.0

            if query_lower in content_lower:
                score = 0.8
            else:
                # Word-level matching
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = 0.3 * (overlap / len(query_words))

            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(entry=entry, score=score, distance=1.0 - score)
            for score, entry in results[:limit]
        ]

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID.

        Args:
            entry_id: The ID of the entry

        Returns:
            The MemoryEntry if found, None otherwise
        """
        index = self._id_index.get(entry_id)
        if index is not None:
            return self.entries[index]
        return None

    async def update(
        self,
        entry_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MemoryEntry]:
        """Update an existing entry.

        Args:
            entry_id: The ID of the entry to update
            content: New content (if provided, re-embeds)
            metadata: New metadata (merged with existing)

        Returns:
            Updated MemoryEntry or None if not found
        """
        index = self._id_index.get(entry_id)
        if index is None:
            return None

        entry = self.entries[index]

        if content is not None:
            entry.content = content
            entry.embedding = await self._get_embedding(content)

        if metadata is not None:
            entry.metadata.update(metadata)

        entry.timestamp = datetime.now()
        return entry

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID.

        Args:
            entry_id: The ID of the entry to delete

        Returns:
            True if deleted, False if not found
        """
        index = self._id_index.get(entry_id)
        if index is None:
            return False

        # Remove entry
        del self.entries[index]
        del self._id_index[entry_id]

        # Rebuild index
        self._id_index = {e.id: i for i, e in enumerate(self.entries)}

        return True

    async def clear(self) -> None:
        """Clear all entries from memory."""
        self.entries.clear()
        self._id_index.clear()

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if unavailable
        """
        if not self.backend:
            return None

        try:
            embedding = await self.backend.embed(text)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def save_to_disk(self, path: Path) -> None:
        """Save memory to disk.

        Args:
            path: Path to save to (JSON file)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dimension": self.dimension,
            "entries": [entry.to_dict() for entry in self.entries]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.entries)} entries to {path}")

    async def load_from_disk(self, path: Path) -> None:
        """Load memory from disk.

        Args:
            path: Path to load from (JSON file)
        """
        if not path.exists():
            logger.warning(f"Memory file not found: {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.dimension = data.get("dimension", self.dimension)
        self.entries = [
            MemoryEntry.from_dict(entry_data)
            for entry_data in data.get("entries", [])
        ]
        self._id_index = {e.id: i for i, e in enumerate(self.entries)}

        logger.info(f"Loaded {len(self.entries)} entries from {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about semantic memory.

        Returns:
            Dictionary with memory statistics
        """
        entries_with_embedding = sum(1 for e in self.entries if e.embedding)

        return {
            "entry_count": len(self.entries),
            "dimension": self.dimension,
            "entries_with_embedding": entries_with_embedding,
            "entries_without_embedding": len(self.entries) - entries_with_embedding,
            "has_backend": self.backend is not None
        }

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.entries)
