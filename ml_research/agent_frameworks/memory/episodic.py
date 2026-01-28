"""
Episodic memory for conversation and session storage.

This module provides storage and retrieval of conversation episodes,
enabling agents to recall and learn from past interactions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json
import uuid
import logging
import asyncio

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend
    from .context_window import AgentMessage

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a conversation episode.

    An episode is a coherent segment of conversation, typically a single
    session or task completion.

    Attributes:
        id: Unique identifier for the episode
        messages: List of messages in the episode
        summary: Optional summarized version of the episode
        metadata: Additional data (tags, task info, etc.)
        created_at: When the episode was created
        updated_at: When the episode was last modified
    """
    id: str
    messages: List["AgentMessage"]
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary format."""
        return {
            "id": self.id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if hasattr(m, 'timestamp') else None,
                    "metadata": getattr(m, 'metadata', {})
                }
                for m in self.messages
            ],
            "summary": self.summary,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create episode from dictionary."""
        from .context_window import AgentMessage

        messages = []
        for msg_data in data.get("messages", []):
            msg = AgentMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]) if msg_data.get("timestamp") else datetime.now(),
                metadata=msg_data.get("metadata", {})
            )
            messages.append(msg)

        return cls(
            id=data["id"],
            messages=messages,
            summary=data.get("summary"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        )

    @property
    def message_count(self) -> int:
        """Number of messages in the episode."""
        return len(self.messages)

    @property
    def duration(self) -> Optional[float]:
        """Duration of the episode in seconds, if timestamps available."""
        if len(self.messages) < 2:
            return None
        try:
            first = self.messages[0].timestamp
            last = self.messages[-1].timestamp
            return (last - first).total_seconds()
        except (AttributeError, TypeError):
            return None


class EpisodicMemory:
    """Stores and retrieves conversation episodes.

    This class provides persistent storage for conversation episodes with
    search and retrieval capabilities.

    Example:
        memory = EpisodicMemory(Path("./episodes"), backend)

        # Save an episode
        episode = Episode(
            id=str(uuid.uuid4()),
            messages=conversation_messages,
            metadata={"task": "code_review"}
        )
        await memory.save_episode(episode)

        # Search for relevant episodes
        results = await memory.search_episodes("authentication bug", limit=5)
    """

    SUMMARY_PROMPT = """Summarize this conversation episode concisely. Include:
1. Main topic or task discussed
2. Key decisions or outcomes
3. Any unresolved items
4. Important facts learned

Conversation:
{conversation}

Summary:"""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        backend: Optional["LLMBackend"] = None
    ):
        """Initialize episodic memory.

        Args:
            storage_path: Path for persistent storage (None for in-memory only)
            backend: Optional LLM backend for summarization and search
        """
        self.storage_path = storage_path
        self.backend = backend
        self._episodes: Dict[str, Episode] = {}
        self._index: List[str] = []  # Ordered list of episode IDs by recency

        # Load existing episodes if storage path exists
        if storage_path and storage_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load episodes from disk storage."""
        if not self.storage_path:
            return

        index_file = self.storage_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load episode index: {e}")
                self._index = []

        episodes_dir = self.storage_path / "episodes"
        if episodes_dir.exists():
            for episode_file in episodes_dir.glob("*.json"):
                try:
                    with open(episode_file, 'r') as f:
                        data = json.load(f)
                    episode = Episode.from_dict(data)
                    self._episodes[episode.id] = episode
                except Exception as e:
                    logger.warning(f"Failed to load episode {episode_file}: {e}")

    async def _save_to_disk(self) -> None:
        """Save episodes to disk storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        episodes_dir = self.storage_path / "episodes"
        episodes_dir.mkdir(exist_ok=True)

        # Save index
        index_file = self.storage_path / "index.json"
        with open(index_file, 'w') as f:
            json.dump(self._index, f)

    async def save_episode(self, episode: Episode) -> None:
        """Save an episode to memory.

        Args:
            episode: The episode to save
        """
        episode.updated_at = datetime.now()

        # Generate summary if not present and backend available
        if not episode.summary and self.backend and episode.messages:
            episode.summary = await self.summarize_episode(episode)

        self._episodes[episode.id] = episode

        # Update index
        if episode.id in self._index:
            self._index.remove(episode.id)
        self._index.insert(0, episode.id)

        # Persist to disk
        if self.storage_path:
            episodes_dir = self.storage_path / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)

            episode_file = episodes_dir / f"{episode.id}.json"
            with open(episode_file, 'w') as f:
                json.dump(episode.to_dict(), f, indent=2)

            await self._save_to_disk()

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID.

        Args:
            episode_id: The ID of the episode to retrieve

        Returns:
            The episode if found, None otherwise
        """
        return self._episodes.get(episode_id)

    async def search_episodes(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Episode]:
        """Search for relevant episodes.

        Args:
            query: Search query
            limit: Maximum number of results
            metadata_filter: Optional filter on metadata fields

        Returns:
            List of matching episodes, ordered by relevance
        """
        # Apply metadata filter first
        candidates = list(self._episodes.values())

        if metadata_filter:
            candidates = [
                ep for ep in candidates
                if all(ep.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        if not candidates:
            return []

        # Score episodes based on query
        scored = []
        query_lower = query.lower()

        for episode in candidates:
            score = 0.0

            # Check summary
            if episode.summary:
                if query_lower in episode.summary.lower():
                    score += 3.0
                # Partial word matching
                words = query_lower.split()
                matches = sum(1 for w in words if w in episode.summary.lower())
                score += matches * 0.5

            # Check messages
            for msg in episode.messages:
                if query_lower in msg.content.lower():
                    score += 1.0
                    break  # Only count once per episode

            # Check metadata
            for key, value in episode.metadata.items():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 0.5

            if score > 0:
                scored.append((score, episode))

        # Sort by score and return top results
        scored.sort(key=lambda x: (-x[0], x[1].updated_at), reverse=False)
        return [ep for _, ep in scored[:limit]]

    async def get_recent(self, limit: int = 10) -> List[Episode]:
        """Get the most recent episodes.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of recent episodes, most recent first
        """
        episode_ids = self._index[:limit]
        return [self._episodes[eid] for eid in episode_ids if eid in self._episodes]

    async def summarize_episode(self, episode: Episode) -> str:
        """Generate a summary for an episode.

        Args:
            episode: The episode to summarize

        Returns:
            Summary text
        """
        if not self.backend:
            return self._simple_summary(episode)

        # Format conversation
        conversation = "\n".join(
            f"{m.role}: {m.content}" for m in episode.messages
        )

        prompt = self.SUMMARY_PROMPT.format(conversation=conversation)

        try:
            from ..backends.backend_base import LLMConfig

            response = await self.backend.complete(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(
                    model=self.backend.default_model,
                    max_tokens=500,
                    temperature=0.3
                )
            )
            return response.content
        except Exception as e:
            logger.warning(f"Episode summarization failed: {e}")
            return self._simple_summary(episode)

    def _simple_summary(self, episode: Episode) -> str:
        """Create a simple summary without LLM.

        Args:
            episode: The episode to summarize

        Returns:
            Simple summary
        """
        if not episode.messages:
            return "Empty episode"

        # Take first user message as topic
        topic = "Unknown topic"
        for msg in episode.messages:
            if msg.role == "user":
                topic = msg.content[:100]
                if len(msg.content) > 100:
                    topic += "..."
                break

        return f"Episode about: {topic} ({len(episode.messages)} messages)"

    async def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode.

        Args:
            episode_id: The ID of the episode to delete

        Returns:
            True if deleted, False if not found
        """
        if episode_id not in self._episodes:
            return False

        del self._episodes[episode_id]

        if episode_id in self._index:
            self._index.remove(episode_id)

        # Remove from disk
        if self.storage_path:
            episode_file = self.storage_path / "episodes" / f"{episode_id}.json"
            if episode_file.exists():
                episode_file.unlink()
            await self._save_to_disk()

        return True

    async def clear(self) -> None:
        """Clear all episodes from memory."""
        self._episodes.clear()
        self._index.clear()

        if self.storage_path:
            episodes_dir = self.storage_path / "episodes"
            if episodes_dir.exists():
                for episode_file in episodes_dir.glob("*.json"):
                    episode_file.unlink()
            await self._save_to_disk()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory.

        Returns:
            Dictionary with memory statistics
        """
        total_messages = sum(len(ep.messages) for ep in self._episodes.values())

        return {
            "episode_count": len(self._episodes),
            "total_messages": total_messages,
            "with_summary": sum(1 for ep in self._episodes.values() if ep.summary),
            "storage_path": str(self.storage_path) if self.storage_path else None
        }

    @classmethod
    def create_episode(
        cls,
        messages: List["AgentMessage"],
        metadata: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None
    ) -> Episode:
        """Helper to create a new episode.

        Args:
            messages: Messages for the episode
            metadata: Optional metadata
            summary: Optional pre-computed summary

        Returns:
            New Episode instance
        """
        return Episode(
            id=str(uuid.uuid4()),
            messages=messages,
            summary=summary,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
