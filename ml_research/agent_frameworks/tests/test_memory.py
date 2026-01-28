"""
Tests for memory management components.

Tests cover:
    - ContextWindow sliding window
    - EpisodicMemory storage and retrieval
    - SemanticMemory search
    - CheckpointManager save/restore
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


# ---------------------------------------------------------------------------
# Test Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A message for context."""
    role: str
    content: str
    tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "timestamp": self.timestamp.isoformat()
        }


class ContextWindow:
    """Sliding context window for managing message history."""

    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self._messages: List[Message] = []
        self._total_tokens = 0

    def add(self, message: Message) -> None:
        """Add a message to the context."""
        self._messages.append(message)
        self._total_tokens += message.tokens
        self._trim()

    def _trim(self) -> None:
        """Remove oldest messages to fit within token limit."""
        while self._total_tokens > self.max_tokens and self._messages:
            removed = self._messages.pop(0)
            self._total_tokens -= removed.tokens

    def get_messages(self) -> List[Message]:
        """Get all messages in context."""
        return self._messages.copy()

    def get_tokens(self) -> int:
        """Get total token count."""
        return self._total_tokens

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._total_tokens = 0

    def __len__(self) -> int:
        return len(self._messages)


@dataclass
class Episode:
    """An episode in episodic memory."""
    episode_id: str
    task: str
    messages: List[Dict[str, Any]]
    outcome: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Stores and retrieves past interaction episodes."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._episodes: Dict[str, Episode] = {}

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

    def store(self, episode: Episode) -> str:
        """Store an episode."""
        self._episodes[episode.episode_id] = episode

        if self.storage_path:
            self._save_episode(episode)

        return episode.episode_id

    def retrieve(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        return self._episodes.get(episode_id)

    def search(
        self,
        query: str,
        limit: int = 5
    ) -> List[Episode]:
        """Search episodes by task similarity."""
        results = []
        query_lower = query.lower()

        for episode in self._episodes.values():
            # Simple keyword matching
            if query_lower in episode.task.lower():
                results.append(episode)

        return results[:limit]

    def list_recent(self, limit: int = 10) -> List[Episode]:
        """List most recent episodes."""
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.created_at,
            reverse=True
        )
        return sorted_episodes[:limit]

    def delete(self, episode_id: str) -> bool:
        """Delete an episode."""
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            if self.storage_path:
                path = self.storage_path / f"{episode_id}.json"
                if path.exists():
                    path.unlink()
            return True
        return False

    def _save_episode(self, episode: Episode) -> None:
        """Save episode to disk."""
        if not self.storage_path:
            return

        path = self.storage_path / f"{episode.episode_id}.json"
        data = {
            "episode_id": episode.episode_id,
            "task": episode.task,
            "messages": episode.messages,
            "outcome": episode.outcome,
            "created_at": episode.created_at.isoformat(),
            "metadata": episode.metadata
        }
        path.write_text(json.dumps(data, indent=2))


@dataclass
class MemoryItem:
    """An item in semantic memory."""
    item_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticMemory:
    """Semantic memory with embedding-based search."""

    def __init__(self, embed_func=None):
        self._embed_func = embed_func or (lambda x: [0.0] * 128)
        self._items: Dict[str, MemoryItem] = {}

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store content with its embedding."""
        item_id = hashlib.md5(content.encode()).hexdigest()[:12]
        embedding = self._embed_func(content)

        item = MemoryItem(
            item_id=item_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self._items[item_id] = item
        return item_id

    def search(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search for similar content."""
        query_embedding = self._embed_func(query)

        # Calculate similarity scores
        scored = []
        for item in self._items.values():
            # Cosine similarity (simplified)
            score = sum(a * b for a, b in zip(query_embedding, item.embedding))
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get item by ID."""
        return self._items.get(item_id)

    def delete(self, item_id: str) -> bool:
        """Delete an item."""
        if item_id in self._items:
            del self._items[item_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class Checkpoint:
    """A checkpoint of agent state."""
    checkpoint_id: str
    context: List[Dict[str, Any]]
    state: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


class CheckpointManager:
    """Manages checkpoints for state recovery."""

    def __init__(self, storage_path: Path, max_checkpoints: int = 10):
        self.storage_path = storage_path
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Checkpoint] = []

        storage_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        context: List[Dict[str, Any]],
        state: Dict[str, Any]
    ) -> Checkpoint:
        """Save a checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            context=context,
            state=state
        )

        self._checkpoints.append(checkpoint)
        self._prune()
        self._persist(checkpoint)

        return checkpoint

    def restore(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Restore from a checkpoint."""
        for cp in self._checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                return cp

        # Try loading from disk
        return self._load(checkpoint_id)

    def get_latest(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        if self._checkpoints:
            return self._checkpoints[-1]
        return None

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs."""
        return [cp.checkpoint_id for cp in self._checkpoints]

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        for i, cp in enumerate(self._checkpoints):
            if cp.checkpoint_id == checkpoint_id:
                self._checkpoints.pop(i)
                path = self.storage_path / f"{checkpoint_id}.json"
                if path.exists():
                    path.unlink()
                return True
        return False

    def _prune(self) -> None:
        """Remove old checkpoints beyond max limit."""
        while len(self._checkpoints) > self.max_checkpoints:
            old = self._checkpoints.pop(0)
            path = self.storage_path / f"{old.checkpoint_id}.json"
            if path.exists():
                path.unlink()

    def _persist(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk."""
        path = self.storage_path / f"{checkpoint.checkpoint_id}.json"
        data = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "context": checkpoint.context,
            "state": checkpoint.state,
            "created_at": checkpoint.created_at.isoformat()
        }
        path.write_text(json.dumps(data, indent=2))

    def _load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from disk."""
        path = self.storage_path / f"{checkpoint_id}.json"
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            context=data["context"],
            state=data["state"],
            created_at=datetime.fromisoformat(data["created_at"])
        )


# ---------------------------------------------------------------------------
# Tests for ContextWindow
# ---------------------------------------------------------------------------

class TestContextWindow:
    """Tests for ContextWindow sliding window."""

    def test_add_message(self):
        """Test adding messages."""
        window = ContextWindow(max_tokens=1000)

        window.add(Message(role="user", content="Hello", tokens=10))
        window.add(Message(role="assistant", content="Hi!", tokens=5))

        assert len(window) == 2
        assert window.get_tokens() == 15

    def test_sliding_window_trim(self):
        """Test that old messages are trimmed."""
        window = ContextWindow(max_tokens=100)

        # Add messages that exceed limit
        for i in range(20):
            window.add(Message(role="user", content=f"Message {i}", tokens=10))

        # Should have trimmed to fit within 100 tokens
        assert window.get_tokens() <= 100
        assert len(window) <= 10

    def test_trim_preserves_newest(self):
        """Test that trimming preserves newest messages."""
        window = ContextWindow(max_tokens=30)

        window.add(Message(role="user", content="Old", tokens=10))
        window.add(Message(role="user", content="Middle", tokens=10))
        window.add(Message(role="user", content="New", tokens=10))
        window.add(Message(role="user", content="Newest", tokens=10))

        messages = window.get_messages()
        contents = [m.content for m in messages]

        # Old should be removed, Newest should be present
        assert "Old" not in contents
        assert "Newest" in contents

    def test_clear(self):
        """Test clearing the window."""
        window = ContextWindow()
        window.add(Message(role="user", content="Test", tokens=10))
        window.add(Message(role="user", content="Test2", tokens=10))

        window.clear()

        assert len(window) == 0
        assert window.get_tokens() == 0


# ---------------------------------------------------------------------------
# Tests for EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    """Tests for EpisodicMemory."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving episodes."""
        memory = EpisodicMemory()

        episode = Episode(
            episode_id="ep1",
            task="Fix bug in auth module",
            messages=[{"role": "user", "content": "Fix the login bug"}],
            outcome="success"
        )

        memory.store(episode)
        retrieved = memory.retrieve("ep1")

        assert retrieved is not None
        assert retrieved.task == "Fix bug in auth module"

    def test_search_by_task(self):
        """Test searching episodes by task."""
        memory = EpisodicMemory()

        memory.store(Episode(
            episode_id="ep1",
            task="Fix authentication bug",
            messages=[],
            outcome="success"
        ))
        memory.store(Episode(
            episode_id="ep2",
            task="Add new feature",
            messages=[],
            outcome="success"
        ))
        memory.store(Episode(
            episode_id="ep3",
            task="Fix database bug",
            messages=[],
            outcome="failure"
        ))

        results = memory.search("bug")

        assert len(results) == 2
        assert all("bug" in r.task.lower() for r in results)

    def test_list_recent(self):
        """Test listing recent episodes."""
        memory = EpisodicMemory()

        for i in range(5):
            memory.store(Episode(
                episode_id=f"ep{i}",
                task=f"Task {i}",
                messages=[],
                outcome="success"
            ))

        recent = memory.list_recent(limit=3)
        assert len(recent) == 3

    def test_delete_episode(self):
        """Test deleting an episode."""
        memory = EpisodicMemory()

        memory.store(Episode(
            episode_id="ep1",
            task="Test",
            messages=[],
            outcome="success"
        ))

        assert memory.retrieve("ep1") is not None
        assert memory.delete("ep1")
        assert memory.retrieve("ep1") is None

    def test_persistence(self, tmp_path):
        """Test episode persistence to disk."""
        storage_path = tmp_path / "episodes"
        memory = EpisodicMemory(storage_path=storage_path)

        episode = Episode(
            episode_id="ep1",
            task="Persisted task",
            messages=[{"role": "user", "content": "Hello"}],
            outcome="success"
        )
        memory.store(episode)

        # Check file was created
        assert (storage_path / "ep1.json").exists()


# ---------------------------------------------------------------------------
# Tests for SemanticMemory
# ---------------------------------------------------------------------------

class TestSemanticMemory:
    """Tests for SemanticMemory search."""

    def test_store_content(self):
        """Test storing content."""
        memory = SemanticMemory()

        item_id = memory.store("The quick brown fox")

        assert item_id is not None
        assert len(memory) == 1

    def test_retrieve_by_id(self):
        """Test retrieving by ID."""
        memory = SemanticMemory()

        item_id = memory.store("Test content", metadata={"source": "test"})
        item = memory.get(item_id)

        assert item is not None
        assert item.content == "Test content"
        assert item.metadata["source"] == "test"

    def test_search_returns_results(self):
        """Test that search returns results."""
        # Create embedding function that returns similar vectors
        def mock_embed(text):
            # Create embeddings based on text length
            return [len(text) / 100.0] * 128

        memory = SemanticMemory(embed_func=mock_embed)

        memory.store("Short text")
        memory.store("A much longer piece of text here")
        memory.store("Medium length text")

        results = memory.search("query", top_k=2)

        assert len(results) == 2

    def test_delete_item(self):
        """Test deleting an item."""
        memory = SemanticMemory()

        item_id = memory.store("To be deleted")

        assert len(memory) == 1
        assert memory.delete(item_id)
        assert len(memory) == 0
        assert memory.get(item_id) is None


# ---------------------------------------------------------------------------
# Tests for CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        checkpoint = manager.save(
            context=[{"role": "user", "content": "Hello"}],
            state={"iteration": 5}
        )

        assert checkpoint is not None
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.state["iteration"] == 5

    def test_restore_checkpoint(self, tmp_path):
        """Test restoring a checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        saved = manager.save(
            context=[{"role": "user", "content": "Test"}],
            state={"value": 42}
        )

        restored = manager.restore(saved.checkpoint_id)

        assert restored is not None
        assert restored.state["value"] == 42
        assert len(restored.context) == 1

    def test_get_latest(self, tmp_path):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        manager.save(context=[], state={"n": 1})
        manager.save(context=[], state={"n": 2})
        manager.save(context=[], state={"n": 3})

        latest = manager.get_latest()

        assert latest is not None
        assert latest.state["n"] == 3

    def test_max_checkpoints_limit(self, tmp_path):
        """Test that old checkpoints are pruned."""
        manager = CheckpointManager(tmp_path / "checkpoints", max_checkpoints=3)

        for i in range(5):
            manager.save(context=[], state={"n": i})

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3

    def test_delete_checkpoint(self, tmp_path):
        """Test deleting a checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        cp = manager.save(context=[], state={})

        assert len(manager.list_checkpoints()) == 1
        assert manager.delete(cp.checkpoint_id)
        assert len(manager.list_checkpoints()) == 0

    def test_persistence(self, tmp_path):
        """Test checkpoint persistence."""
        storage = tmp_path / "checkpoints"
        manager = CheckpointManager(storage)

        cp = manager.save(
            context=[{"msg": "test"}],
            state={"persisted": True}
        )

        # File should exist
        assert (storage / f"{cp.checkpoint_id}.json").exists()

        # Create new manager and load
        manager2 = CheckpointManager(storage)
        loaded = manager2._load(cp.checkpoint_id)

        assert loaded is not None
        assert loaded.state["persisted"] is True
