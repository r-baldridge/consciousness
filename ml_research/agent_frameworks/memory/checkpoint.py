"""
State serialization and checkpoint management for agents.

This module provides the ability to save and restore agent state,
enabling pause/resume functionality and state recovery.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Callable
from datetime import datetime
from pathlib import Path
import json
import uuid
import logging
import shutil
import asyncio

if TYPE_CHECKING:
    from .context_window import AgentMessage

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Represents a saved agent state.

    Attributes:
        id: Unique identifier for the checkpoint
        agent_id: ID of the agent that created this checkpoint
        state: Serialized agent state
        messages: Conversation history at checkpoint time
        metadata: Additional checkpoint data
        created_at: When the checkpoint was created
        name: Optional human-readable name
    """
    id: str
    agent_id: str
    state: Dict[str, Any]
    messages: List["AgentMessage"]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary format."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "state": self.state,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if hasattr(m, 'timestamp') else None,
                    "metadata": getattr(m, 'metadata', {})
                }
                for m in self.messages
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "name": self.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
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
            agent_id=data["agent_id"],
            state=data.get("state", {}),
            messages=messages,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            name=data.get("name")
        )

    @property
    def display_name(self) -> str:
        """Human-readable name for the checkpoint."""
        if self.name:
            return self.name
        return f"checkpoint-{self.created_at.strftime('%Y%m%d-%H%M%S')}"


class AgentStateProtocol:
    """Protocol defining what an agent must implement for checkpointing.

    Agents should implement these methods to support checkpointing.
    This is a duck-typed protocol - classes don't need to explicitly
    inherit from this.
    """

    @property
    def id(self) -> str:
        """Return the agent's unique identifier."""
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        """Return the agent's serializable state."""
        raise NotImplementedError

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from dictionary."""
        raise NotImplementedError

    def get_messages(self) -> List["AgentMessage"]:
        """Return the agent's message history."""
        raise NotImplementedError

    def set_messages(self, messages: List["AgentMessage"]) -> None:
        """Restore agent's message history."""
        raise NotImplementedError


class CheckpointManager:
    """Save and restore agent state.

    This class manages checkpoints for agents, providing save, restore,
    and listing capabilities with optional persistent storage.

    Example:
        manager = CheckpointManager(Path("./checkpoints"))

        # Save agent state
        checkpoint = await manager.save(agent, name="before_refactor")
        print(f"Saved checkpoint: {checkpoint.id}")

        # Later, restore agent state
        restored_agent = await manager.restore(checkpoint.id)

        # List available checkpoints
        checkpoints = await manager.list_checkpoints(agent_id=agent.id)
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_checkpoints_per_agent: int = 10,
        agent_factory: Optional[Callable[[str], Any]] = None
    ):
        """Initialize the checkpoint manager.

        Args:
            storage_path: Path for persistent storage (None for in-memory only)
            max_checkpoints_per_agent: Maximum checkpoints to keep per agent
            agent_factory: Optional factory function to create agents for restoration
        """
        self.storage_path = storage_path
        self.max_checkpoints_per_agent = max_checkpoints_per_agent
        self.agent_factory = agent_factory
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._agent_checkpoints: Dict[str, List[str]] = {}  # agent_id -> [checkpoint_ids]

        # Load existing checkpoints if storage path exists
        if storage_path and storage_path.exists():
            self._load_index()

    def _load_index(self) -> None:
        """Load checkpoint index from disk."""
        if not self.storage_path:
            return

        index_file = self.storage_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                self._agent_checkpoints = data.get("agent_checkpoints", {})

                # Load checkpoint metadata (not full state)
                for checkpoint_ids in self._agent_checkpoints.values():
                    for cp_id in checkpoint_ids:
                        cp_file = self.storage_path / "checkpoints" / f"{cp_id}.json"
                        if cp_file.exists():
                            with open(cp_file, 'r') as f:
                                cp_data = json.load(f)
                            self._checkpoints[cp_id] = Checkpoint.from_dict(cp_data)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint index: {e}")

    async def _save_index(self) -> None:
        """Save checkpoint index to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        index_file = self.storage_path / "index.json"
        with open(index_file, 'w') as f:
            json.dump({
                "agent_checkpoints": self._agent_checkpoints
            }, f, indent=2)

    async def save(
        self,
        agent: Any,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """Save an agent's current state.

        Args:
            agent: The agent to checkpoint (must implement state methods)
            name: Optional human-readable name for the checkpoint
            metadata: Optional additional metadata

        Returns:
            The created Checkpoint
        """
        # Get agent ID
        agent_id = getattr(agent, 'id', None) or str(id(agent))

        # Get state
        if hasattr(agent, 'get_state'):
            state = agent.get_state()
        elif hasattr(agent, 'to_dict'):
            state = agent.to_dict()
        else:
            state = {}
            logger.warning(f"Agent {agent_id} doesn't implement get_state or to_dict")

        # Get messages
        if hasattr(agent, 'get_messages'):
            messages = agent.get_messages()
        elif hasattr(agent, 'messages'):
            messages = agent.messages
        elif hasattr(agent, 'context') and hasattr(agent.context, 'messages'):
            messages = agent.context.messages
        else:
            messages = []

        # Create checkpoint
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            state=state,
            messages=messages,
            metadata=metadata or {},
            created_at=datetime.now(),
            name=name
        )

        # Store checkpoint
        self._checkpoints[checkpoint.id] = checkpoint

        # Update agent index
        if agent_id not in self._agent_checkpoints:
            self._agent_checkpoints[agent_id] = []
        self._agent_checkpoints[agent_id].insert(0, checkpoint.id)

        # Enforce max checkpoints limit
        await self._enforce_checkpoint_limit(agent_id)

        # Persist to disk
        if self.storage_path:
            checkpoints_dir = self.storage_path / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            cp_file = checkpoints_dir / f"{checkpoint.id}.json"
            with open(cp_file, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            await self._save_index()

        logger.info(f"Created checkpoint {checkpoint.id} for agent {agent_id}")
        return checkpoint

    async def restore(
        self,
        checkpoint_id: str,
        agent: Optional[Any] = None
    ) -> Any:
        """Restore agent state from a checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to restore
            agent: Optional existing agent to restore into

        Returns:
            The restored agent

        Raises:
            KeyError: If checkpoint not found
            ValueError: If no agent provided and no factory available
        """
        checkpoint = await self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        # Get or create agent
        if agent is None:
            if self.agent_factory:
                agent = self.agent_factory(checkpoint.agent_id)
            else:
                raise ValueError(
                    "No agent provided and no agent_factory configured. "
                    "Either pass an agent to restore into, or configure an agent_factory."
                )

        # Restore state
        if hasattr(agent, 'set_state'):
            agent.set_state(checkpoint.state)
        elif hasattr(agent, 'from_dict'):
            agent = agent.from_dict(checkpoint.state)
        else:
            # Try to set attributes directly
            for key, value in checkpoint.state.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)

        # Restore messages
        if hasattr(agent, 'set_messages'):
            agent.set_messages(checkpoint.messages)
        elif hasattr(agent, 'messages'):
            agent.messages = checkpoint.messages
        elif hasattr(agent, 'context') and hasattr(agent.context, 'messages'):
            agent.context.messages = checkpoint.messages

        logger.info(f"Restored agent {checkpoint.agent_id} from checkpoint {checkpoint_id}")
        return agent

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID.

        Args:
            checkpoint_id: The ID of the checkpoint

        Returns:
            The Checkpoint if found, None otherwise
        """
        # Check in-memory cache
        if checkpoint_id in self._checkpoints:
            return self._checkpoints[checkpoint_id]

        # Try loading from disk
        if self.storage_path:
            cp_file = self.storage_path / "checkpoints" / f"{checkpoint_id}.json"
            if cp_file.exists():
                try:
                    with open(cp_file, 'r') as f:
                        data = json.load(f)
                    checkpoint = Checkpoint.from_dict(data)
                    self._checkpoints[checkpoint_id] = checkpoint
                    return checkpoint
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint {checkpoint_id}: {e}")

        return None

    async def list_checkpoints(
        self,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Checkpoint]:
        """List available checkpoints.

        Args:
            agent_id: Optional filter by agent ID
            limit: Optional maximum number of results

        Returns:
            List of Checkpoint objects, most recent first
        """
        if agent_id:
            checkpoint_ids = self._agent_checkpoints.get(agent_id, [])
        else:
            # All checkpoints, sorted by date
            all_checkpoints = list(self._checkpoints.values())
            all_checkpoints.sort(key=lambda c: c.created_at, reverse=True)
            if limit:
                all_checkpoints = all_checkpoints[:limit]
            return all_checkpoints

        checkpoints = []
        for cp_id in checkpoint_ids:
            cp = await self.get_checkpoint(cp_id)
            if cp:
                checkpoints.append(cp)

        if limit:
            checkpoints = checkpoints[:limit]

        return checkpoints

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False

        # Remove from memory
        del self._checkpoints[checkpoint_id]

        # Remove from agent index
        if checkpoint.agent_id in self._agent_checkpoints:
            if checkpoint_id in self._agent_checkpoints[checkpoint.agent_id]:
                self._agent_checkpoints[checkpoint.agent_id].remove(checkpoint_id)

        # Remove from disk
        if self.storage_path:
            cp_file = self.storage_path / "checkpoints" / f"{checkpoint_id}.json"
            if cp_file.exists():
                cp_file.unlink()
            await self._save_index()

        logger.info(f"Deleted checkpoint {checkpoint_id}")
        return True

    async def delete_all_for_agent(self, agent_id: str) -> int:
        """Delete all checkpoints for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            Number of checkpoints deleted
        """
        checkpoint_ids = self._agent_checkpoints.get(agent_id, []).copy()
        deleted = 0

        for cp_id in checkpoint_ids:
            if await self.delete(cp_id):
                deleted += 1

        return deleted

    async def _enforce_checkpoint_limit(self, agent_id: str) -> None:
        """Enforce maximum checkpoint limit for an agent.

        Args:
            agent_id: The agent ID to check
        """
        checkpoint_ids = self._agent_checkpoints.get(agent_id, [])

        while len(checkpoint_ids) > self.max_checkpoints_per_agent:
            # Remove oldest checkpoint
            oldest_id = checkpoint_ids[-1]
            await self.delete(oldest_id)
            checkpoint_ids = self._agent_checkpoints.get(agent_id, [])

    async def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()
        self._agent_checkpoints.clear()

        if self.storage_path:
            checkpoints_dir = self.storage_path / "checkpoints"
            if checkpoints_dir.exists():
                shutil.rmtree(checkpoints_dir)
            await self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints.

        Returns:
            Dictionary with checkpoint statistics
        """
        return {
            "total_checkpoints": len(self._checkpoints),
            "agents_with_checkpoints": len(self._agent_checkpoints),
            "checkpoints_per_agent": {
                agent_id: len(cp_ids)
                for agent_id, cp_ids in self._agent_checkpoints.items()
            },
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "max_per_agent": self.max_checkpoints_per_agent
        }
