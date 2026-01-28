"""
Session Manager for Multi-Agent Execution.

Provides persistent session management for AI agents, supporting:
    - Multiple concurrent sessions
    - Session persistence to disk (JSON/pickle)
    - Session state management
    - History tracking
    - Multi-agent collaboration on the same project

Example:
    manager = SessionManager("/path/to/sessions")

    # Create a new session
    session = await manager.create_session("agent-1")

    # Add to session history
    await manager.add_to_history(session.id, {"role": "user", "content": "Hello"})

    # Persist session
    await manager.save_session(session.id)

    # Later, restore session
    session = await manager.get_session(session.id)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
import asyncio
import json
import pickle
import os
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Possible states for a session."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class SessionMessage:
    """A message in the session history."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """
    Represents an agent execution session.

    A session tracks the complete state and history of an agent's
    interaction, including messages, tool executions, and any
    persistent state needed for the agent's operation.

    Attributes:
        id: Unique session identifier
        agent_id: ID of the agent owning this session
        state: Current session state
        history: List of messages in the session
        created_at: When the session was created
        updated_at: When the session was last updated
        metadata: Additional session metadata
        context: Persistent context data for the agent
    """

    id: str
    agent_id: str
    state: SessionState = SessionState.CREATED
    history: List[SessionMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    parent_session_id: Optional[str] = None
    child_session_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure proper initialization."""
        if not self.id:
            self.id = str(uuid.uuid4())

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionMessage:
        """
        Add a message to the session history.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            metadata: Optional additional metadata

        Returns:
            The created SessionMessage
        """
        message = SessionMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.history.append(message)
        self.updated_at = datetime.now()
        return message

    def get_last_message(self, role: Optional[str] = None) -> Optional[SessionMessage]:
        """
        Get the last message, optionally filtered by role.

        Args:
            role: Optional role to filter by

        Returns:
            The last matching message or None
        """
        if role:
            for msg in reversed(self.history):
                if msg.role == role:
                    return msg
            return None
        return self.history[-1] if self.history else None

    def get_messages_by_role(self, role: str) -> List[SessionMessage]:
        """Get all messages with the specified role."""
        return [msg for msg in self.history if msg.role == role]

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "history": [msg.to_dict() for msg in self.history],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "context": self.context,
            "parent_session_id": self.parent_session_id,
            "child_session_ids": self.child_session_ids,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            state=SessionState(data["state"]),
            history=[SessionMessage.from_dict(m) for m in data.get("history", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            context=data.get("context", {}),
            parent_session_id=data.get("parent_session_id"),
            child_session_ids=data.get("child_session_ids", []),
            tags=data.get("tags", []),
        )

    def to_pickle(self) -> bytes:
        """Serialize session to pickle bytes."""
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, data: bytes) -> "Session":
        """Deserialize session from pickle bytes."""
        return pickle.loads(data)


class SessionManager:
    """
    Manages multiple agent sessions with persistence.

    Provides CRUD operations for sessions, persistence to disk,
    and support for multiple agents working on the same project.

    Attributes:
        storage_path: Path to session storage directory
        sessions: In-memory cache of active sessions
        use_pickle: Use pickle (True) or JSON (False) for persistence
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        use_pickle: bool = False,
        auto_save: bool = True,
        auto_save_interval: int = 60,
    ):
        """
        Initialize the session manager.

        Args:
            storage_path: Directory for session persistence
            use_pickle: Use pickle format (default: JSON for portability)
            auto_save: Automatically save sessions periodically
            auto_save_interval: Seconds between auto-saves (default 60)
        """
        self.storage_path = Path(storage_path)
        self.use_pickle = use_pickle
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._auto_save_task: Optional[asyncio.Task] = None

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the session manager (loads existing sessions, starts auto-save)."""
        await self._load_all_sessions()
        if self.auto_save:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"SessionManager started with {len(self._sessions)} sessions")

    async def stop(self) -> None:
        """Stop the session manager (saves all sessions, stops auto-save)."""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
        await self.save_all_sessions()
        logger.info("SessionManager stopped")

    async def create_session(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        parent_session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Session:
        """
        Create a new session for an agent.

        Args:
            agent_id: ID of the agent
            metadata: Optional session metadata
            context: Optional initial context
            parent_session_id: Optional parent session for hierarchies
            tags: Optional tags for organization

        Returns:
            The newly created Session
        """
        async with self._lock:
            session = Session(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                metadata=metadata or {},
                context=context or {},
                parent_session_id=parent_session_id,
                tags=tags or [],
            )

            # Link to parent if specified
            if parent_session_id and parent_session_id in self._sessions:
                self._sessions[parent_session_id].child_session_ids.append(session.id)

            self._sessions[session.id] = session
            logger.info(f"Created session {session.id} for agent {agent_id}")

            # Persist immediately
            await self._save_session_internal(session)

            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.

        Args:
            session_id: The session ID

        Returns:
            The Session if found, None otherwise
        """
        async with self._lock:
            return self._sessions.get(session_id)

    async def list_sessions(
        self,
        agent_id: Optional[str] = None,
        state: Optional[SessionState] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Session]:
        """
        List sessions with optional filters.

        Args:
            agent_id: Filter by agent ID
            state: Filter by session state
            tags: Filter by tags (session must have all specified tags)

        Returns:
            List of matching sessions
        """
        async with self._lock:
            sessions = list(self._sessions.values())

            if agent_id:
                sessions = [s for s in sessions if s.agent_id == agent_id]

            if state:
                sessions = [s for s in sessions if s.state == state]

            if tags:
                sessions = [
                    s for s in sessions if all(tag in s.tags for tag in tags)
                ]

            return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    async def update_session(
        self,
        session_id: str,
        state: Optional[SessionState] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Session]:
        """
        Update session properties.

        Args:
            session_id: Session to update
            state: New state (if provided)
            metadata: Metadata to merge (if provided)
            context: Context to merge (if provided)
            tags: New tags list (if provided)

        Returns:
            Updated session or None if not found
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            if state is not None:
                session.state = state

            if metadata is not None:
                session.metadata.update(metadata)

            if context is not None:
                session.context.update(context)

            if tags is not None:
                session.tags = tags

            session.updated_at = datetime.now()
            return session

    async def add_to_history(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionMessage]:
        """
        Add a message to session history.

        Args:
            session_id: Target session
            role: Message role
            content: Message content
            metadata: Optional metadata

        Returns:
            The created message or None if session not found
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            return session.add_message(role, content, metadata)

    async def close_session(
        self,
        session_id: str,
        final_state: SessionState = SessionState.COMPLETED,
    ) -> bool:
        """
        Close a session with a final state.

        Args:
            session_id: Session to close
            final_state: Final state (default COMPLETED)

        Returns:
            True if session was closed, False if not found
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            session.state = final_state
            session.updated_at = datetime.now()
            await self._save_session_internal(session)

            logger.info(f"Closed session {session_id} with state {final_state.value}")
            return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its persisted data.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if session_id not in self._sessions:
                return False

            session = self._sessions[session_id]

            # Update parent's child list
            if session.parent_session_id:
                parent = self._sessions.get(session.parent_session_id)
                if parent and session_id in parent.child_session_ids:
                    parent.child_session_ids.remove(session_id)

            # Remove from memory
            del self._sessions[session_id]

            # Remove from disk
            file_path = self._get_session_path(session_id)
            if file_path.exists():
                file_path.unlink()

            logger.info(f"Deleted session {session_id}")
            return True

    async def save_session(self, session_id: str) -> bool:
        """
        Save a specific session to disk.

        Args:
            session_id: Session to save

        Returns:
            True if saved, False if not found
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            await self._save_session_internal(session)
            return True

    async def save_all_sessions(self) -> int:
        """
        Save all sessions to disk.

        Returns:
            Number of sessions saved
        """
        async with self._lock:
            count = 0
            for session in self._sessions.values():
                await self._save_session_internal(session)
                count += 1
            return count

    async def fork_session(
        self,
        session_id: str,
        new_agent_id: Optional[str] = None,
    ) -> Optional[Session]:
        """
        Fork a session, creating a copy with new ID.

        Useful for branching conversation or sharing context.

        Args:
            session_id: Session to fork
            new_agent_id: Agent ID for new session (defaults to same agent)

        Returns:
            The forked session or None if source not found
        """
        async with self._lock:
            source = self._sessions.get(session_id)
            if not source:
                return None

            # Create deep copy
            forked = Session(
                id=str(uuid.uuid4()),
                agent_id=new_agent_id or source.agent_id,
                state=SessionState.CREATED,
                history=[
                    SessionMessage(
                        role=m.role,
                        content=m.content,
                        timestamp=m.timestamp,
                        metadata=m.metadata.copy(),
                    )
                    for m in source.history
                ],
                metadata={**source.metadata, "forked_from": session_id},
                context=source.context.copy(),
                parent_session_id=session_id,
                tags=source.tags.copy(),
            )

            source.child_session_ids.append(forked.id)
            self._sessions[forked.id] = forked

            await self._save_session_internal(forked)
            await self._save_session_internal(source)

            logger.info(f"Forked session {session_id} to {forked.id}")
            return forked

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        ext = ".pkl" if self.use_pickle else ".json"
        return self.storage_path / f"{session_id}{ext}"

    async def _save_session_internal(self, session: Session) -> None:
        """Internal method to save a session to disk."""
        file_path = self._get_session_path(session.id)

        try:
            if self.use_pickle:
                data = session.to_pickle()
                await asyncio.to_thread(file_path.write_bytes, data)
            else:
                data = json.dumps(session.to_dict(), indent=2, default=str)
                await asyncio.to_thread(file_path.write_text, data)
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")
            raise

    async def _load_session(self, file_path: Path) -> Optional[Session]:
        """Load a session from disk."""
        try:
            if file_path.suffix == ".pkl":
                data = await asyncio.to_thread(file_path.read_bytes)
                return Session.from_pickle(data)
            else:
                content = await asyncio.to_thread(file_path.read_text)
                data = json.loads(content)
                return Session.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session from {file_path}: {e}")
            return None

    async def _load_all_sessions(self) -> None:
        """Load all sessions from storage directory."""
        patterns = ["*.json", "*.pkl"] if not self.use_pickle else ["*.pkl"]
        if self.use_pickle:
            patterns = ["*.pkl"]
        else:
            patterns = ["*.json"]

        for pattern in patterns:
            for file_path in self.storage_path.glob(pattern):
                session = await self._load_session(file_path)
                if session:
                    self._sessions[session.id] = session

    async def _auto_save_loop(self) -> None:
        """Background loop for auto-saving sessions."""
        while True:
            try:
                await asyncio.sleep(self.auto_save_interval)
                await self.save_all_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-save error: {e}")

    async def get_sessions_for_project(
        self,
        project_id: str,
    ) -> List[Session]:
        """
        Get all sessions associated with a project.

        Uses metadata.project_id to filter sessions.

        Args:
            project_id: Project identifier

        Returns:
            List of sessions for the project
        """
        async with self._lock:
            return [
                s
                for s in self._sessions.values()
                if s.metadata.get("project_id") == project_id
            ]

    async def get_agent_sessions(
        self,
        agent_id: str,
        active_only: bool = False,
    ) -> List[Session]:
        """
        Get all sessions for a specific agent.

        Args:
            agent_id: Agent identifier
            active_only: Only return active sessions

        Returns:
            List of sessions for the agent
        """
        sessions = await self.list_sessions(agent_id=agent_id)
        if active_only:
            sessions = [s for s in sessions if s.state == SessionState.ACTIVE]
        return sessions

    def count(self) -> int:
        """Get total number of sessions."""
        return len(self._sessions)

    async def cleanup_old_sessions(
        self,
        max_age_days: int = 30,
        states: Optional[List[SessionState]] = None,
    ) -> int:
        """
        Remove old sessions based on age and state.

        Args:
            max_age_days: Maximum age in days
            states: Only clean sessions in these states (default: COMPLETED, TERMINATED)

        Returns:
            Number of sessions removed
        """
        if states is None:
            states = [SessionState.COMPLETED, SessionState.TERMINATED]

        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        removed = 0

        async with self._lock:
            to_remove = [
                sid
                for sid, session in self._sessions.items()
                if session.state in states
                and session.updated_at.timestamp() < cutoff
            ]

            for sid in to_remove:
                del self._sessions[sid]
                file_path = self._get_session_path(sid)
                if file_path.exists():
                    file_path.unlink()
                removed += 1

        logger.info(f"Cleaned up {removed} old sessions")
        return removed
