"""
Memory Module for Agent Frameworks.

This module provides comprehensive memory management for AI agents,
including context window management, episodic memory, semantic retrieval,
and checkpoint/state persistence.

Components:
    - ContextWindow: Sliding context management with automatic summarization
    - EpisodicMemory: Conversation and session memory storage
    - SemanticMemory: Vector-based semantic retrieval
    - CheckpointManager: State serialization and restoration

Example:
    from agent_frameworks.memory import (
        ContextWindow, ContextConfig,
        EpisodicMemory, Episode,
        SemanticMemory, MemoryEntry,
        CheckpointManager, Checkpoint
    )

    # Manage context window
    config = ContextConfig(max_tokens=100000, reserve_tokens=4000)
    context = ContextWindow(config, backend)
    await context.add(message)

    # Store episodes
    episodic = EpisodicMemory(storage_path)
    await episodic.save_episode(episode)
    results = await episodic.search_episodes("user authentication")

    # Semantic search
    semantic = SemanticMemory(backend, dimension=1536)
    await semantic.add("Important fact about the user", {"source": "conversation"})
    relevant = await semantic.search("user preferences")

    # Checkpoint agent state
    checkpoint_mgr = CheckpointManager(storage_path)
    checkpoint = await checkpoint_mgr.save(agent, name="before_refactor")
    restored_agent = await checkpoint_mgr.restore(checkpoint.id)
"""

from .context_window import (
    ContextConfig,
    ContextWindow,
)

from .episodic import (
    Episode,
    EpisodicMemory,
)

from .semantic import (
    MemoryEntry,
    SemanticMemory,
)

from .checkpoint import (
    Checkpoint,
    CheckpointManager,
)

__all__ = [
    # Context Window
    "ContextConfig",
    "ContextWindow",
    # Episodic Memory
    "Episode",
    "EpisodicMemory",
    # Semantic Memory
    "MemoryEntry",
    "SemanticMemory",
    # Checkpoint
    "Checkpoint",
    "CheckpointManager",
]
