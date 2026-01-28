"""
Workspace management for isolated agent environments.

This module provides per-agent isolated workspaces with file management,
snapshots, and resource limits.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    """
    Configuration for workspaces.

    Attributes:
        base_path: Base directory for all workspaces
        max_size_mb: Maximum workspace size in megabytes
        cleanup_on_exit: Whether to delete workspace on cleanup
        max_files: Maximum number of files allowed
        snapshot_dir: Directory to store snapshots (relative to base_path)
        allowed_extensions: List of allowed file extensions (None = all)
        preserve_snapshots: Number of snapshots to preserve per workspace
    """
    base_path: Path
    max_size_mb: int = 100
    cleanup_on_exit: bool = True
    max_files: int = 10000
    snapshot_dir: str = ".snapshots"
    allowed_extensions: Optional[List[str]] = None
    preserve_snapshots: int = 5


@dataclass
class WorkspaceStats:
    """
    Statistics about a workspace.

    Attributes:
        size_bytes: Total size in bytes
        file_count: Number of files
        directory_count: Number of directories
        last_modified: Last modification time
        snapshot_count: Number of snapshots
    """
    size_bytes: int
    file_count: int
    directory_count: int
    last_modified: datetime
    snapshot_count: int

    @property
    def size_mb(self) -> float:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024)


@dataclass
class Snapshot:
    """
    A workspace snapshot.

    Attributes:
        id: Unique snapshot identifier
        workspace_id: ID of the workspace
        created_at: When the snapshot was created
        description: Optional description
        file_count: Number of files in snapshot
        size_bytes: Total size of snapshot
        checksum: Checksum of snapshot contents
    """
    id: str
    workspace_id: str
    created_at: datetime
    description: Optional[str]
    file_count: int
    size_bytes: int
    checksum: str


class Workspace:
    """
    Isolated workspace for an agent.

    Provides file management, resource tracking, and snapshot capabilities
    within an isolated directory.

    Example:
        config = WorkspaceConfig(base_path=Path("/tmp/workspaces"))
        workspace = Workspace("agent-1", config)
        await workspace.initialize()

        # File operations
        path = await workspace.get_file("src/main.py")
        await workspace.write_file("src/main.py", "print('hello')")
        files = await workspace.list_files()

        # Snapshots
        snapshot_id = await workspace.snapshot()
        await workspace.restore(snapshot_id)

        # Cleanup
        await workspace.cleanup()
    """

    def __init__(self, agent_id: str, config: WorkspaceConfig):
        """
        Initialize the workspace.

        Args:
            agent_id: ID of the agent this workspace belongs to
            config: Workspace configuration
        """
        self.agent_id = agent_id
        self.config = config
        self.path = config.base_path / agent_id
        self._snapshot_path = self.path / config.snapshot_dir
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Check if workspace is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """
        Initialize the workspace directory.

        Creates the workspace directory if it doesn't exist.
        """
        async with self._lock:
            if self._initialized:
                return

            # Create workspace directory
            self.path.mkdir(parents=True, exist_ok=True)
            self._snapshot_path.mkdir(parents=True, exist_ok=True)

            # Create metadata file
            metadata = {
                "agent_id": self.agent_id,
                "created_at": datetime.now().isoformat(),
                "config": {
                    "max_size_mb": self.config.max_size_mb,
                    "max_files": self.config.max_files,
                }
            }
            await self._write_metadata(metadata)

            self._initialized = True
            logger.info(f"Initialized workspace at {self.path}")

    async def cleanup(self) -> None:
        """
        Clean up the workspace.

        If cleanup_on_exit is True, deletes the entire workspace directory.
        """
        async with self._lock:
            if self.config.cleanup_on_exit and self.path.exists():
                shutil.rmtree(self.path)
                logger.info(f"Cleaned up workspace at {self.path}")
            self._initialized = False

    async def get_file(self, relative_path: str) -> Path:
        """
        Get the full path to a file in the workspace.

        Args:
            relative_path: Path relative to workspace root

        Returns:
            Full Path object

        Raises:
            ValueError: If path would escape workspace
        """
        self._ensure_initialized()
        full_path = (self.path / relative_path).resolve()

        # Security check: ensure path is within workspace
        if not str(full_path).startswith(str(self.path.resolve())):
            raise ValueError(f"Path {relative_path} would escape workspace")

        return full_path

    async def read_file(self, relative_path: str) -> str:
        """
        Read a file from the workspace.

        Args:
            relative_path: Path relative to workspace root

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = await self.get_file(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")

        return await asyncio.get_event_loop().run_in_executor(
            None, path.read_text
        )

    async def read_file_bytes(self, relative_path: str) -> bytes:
        """
        Read a file as bytes from the workspace.

        Args:
            relative_path: Path relative to workspace root

        Returns:
            File contents as bytes
        """
        path = await self.get_file(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")

        return await asyncio.get_event_loop().run_in_executor(
            None, path.read_bytes
        )

    async def write_file(
        self,
        relative_path: str,
        content: str,
        create_dirs: bool = True
    ) -> Path:
        """
        Write a file to the workspace.

        Args:
            relative_path: Path relative to workspace root
            content: Content to write
            create_dirs: Whether to create parent directories

        Returns:
            Full path to the written file

        Raises:
            ValueError: If file extension not allowed or size limit exceeded
        """
        self._ensure_initialized()
        await self._check_extension(relative_path)
        await self._check_size_limit(len(content.encode()))

        path = await self.get_file(relative_path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.get_event_loop().run_in_executor(
            None, path.write_text, content
        )

        logger.debug(f"Wrote file: {relative_path}")
        return path

    async def write_file_bytes(
        self,
        relative_path: str,
        content: bytes,
        create_dirs: bool = True
    ) -> Path:
        """
        Write binary content to a file.

        Args:
            relative_path: Path relative to workspace root
            content: Binary content to write
            create_dirs: Whether to create parent directories

        Returns:
            Full path to the written file
        """
        self._ensure_initialized()
        await self._check_extension(relative_path)
        await self._check_size_limit(len(content))

        path = await self.get_file(relative_path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.get_event_loop().run_in_executor(
            None, path.write_bytes, content
        )

        return path

    async def delete_file(self, relative_path: str) -> bool:
        """
        Delete a file from the workspace.

        Args:
            relative_path: Path relative to workspace root

        Returns:
            True if file was deleted, False if it didn't exist
        """
        path = await self.get_file(relative_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted file: {relative_path}")
            return True
        return False

    async def file_exists(self, relative_path: str) -> bool:
        """
        Check if a file exists in the workspace.

        Args:
            relative_path: Path relative to workspace root

        Returns:
            True if file exists
        """
        path = await self.get_file(relative_path)
        return path.exists()

    async def list_files(
        self,
        pattern: str = "**/*",
        include_dirs: bool = False
    ) -> List[Path]:
        """
        List files in the workspace.

        Args:
            pattern: Glob pattern to match
            include_dirs: Whether to include directories

        Returns:
            List of paths matching the pattern
        """
        self._ensure_initialized()

        def _list():
            files = []
            for path in self.path.glob(pattern):
                # Skip snapshot directory
                if self._snapshot_path in path.parents or path == self._snapshot_path:
                    continue
                # Skip metadata
                if path.name == ".workspace_metadata.json":
                    continue
                if include_dirs or path.is_file():
                    files.append(path)
            return sorted(files)

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def list_files_relative(
        self,
        pattern: str = "**/*",
        include_dirs: bool = False
    ) -> List[str]:
        """
        List files as relative paths.

        Args:
            pattern: Glob pattern to match
            include_dirs: Whether to include directories

        Returns:
            List of relative path strings
        """
        files = await self.list_files(pattern, include_dirs)
        return [str(f.relative_to(self.path)) for f in files]

    async def get_stats(self) -> WorkspaceStats:
        """
        Get statistics about the workspace.

        Returns:
            WorkspaceStats with size, file counts, etc.
        """
        self._ensure_initialized()

        def _calculate():
            total_size = 0
            file_count = 0
            dir_count = 0
            latest_mtime = 0

            for root, dirs, files in os.walk(self.path):
                # Skip snapshot directory
                if self.config.snapshot_dir in root:
                    continue
                for name in files:
                    if name == ".workspace_metadata.json":
                        continue
                    path = Path(root) / name
                    try:
                        stat = path.stat()
                        total_size += stat.st_size
                        latest_mtime = max(latest_mtime, stat.st_mtime)
                        file_count += 1
                    except OSError:
                        pass
                for name in dirs:
                    if name != self.config.snapshot_dir:
                        dir_count += 1

            return total_size, file_count, dir_count, latest_mtime

        size, files, dirs, mtime = await asyncio.get_event_loop().run_in_executor(
            None, _calculate
        )

        # Count snapshots
        snapshot_count = len(list(self._snapshot_path.glob("*"))) if self._snapshot_path.exists() else 0

        return WorkspaceStats(
            size_bytes=size,
            file_count=files,
            directory_count=dirs,
            last_modified=datetime.fromtimestamp(mtime) if mtime else datetime.now(),
            snapshot_count=snapshot_count,
        )

    async def snapshot(self, description: Optional[str] = None) -> str:
        """
        Create a snapshot of the current workspace state.

        Args:
            description: Optional description for the snapshot

        Returns:
            Snapshot ID
        """
        self._ensure_initialized()

        snapshot_id = str(uuid.uuid4())[:8]
        snapshot_dir = self._snapshot_path / snapshot_id

        def _create_snapshot():
            # Copy all files to snapshot directory
            shutil.copytree(
                self.path,
                snapshot_dir,
                ignore=shutil.ignore_patterns(self.config.snapshot_dir, ".workspace_metadata.json")
            )

            # Calculate checksum
            hasher = hashlib.sha256()
            file_count = 0
            total_size = 0
            for path in snapshot_dir.rglob("*"):
                if path.is_file():
                    file_count += 1
                    total_size += path.stat().st_size
                    hasher.update(path.read_bytes())

            return file_count, total_size, hasher.hexdigest()

        file_count, size, checksum = await asyncio.get_event_loop().run_in_executor(
            None, _create_snapshot
        )

        # Save snapshot metadata
        snapshot = Snapshot(
            id=snapshot_id,
            workspace_id=self.agent_id,
            created_at=datetime.now(),
            description=description,
            file_count=file_count,
            size_bytes=size,
            checksum=checksum,
        )

        metadata_path = snapshot_dir / ".snapshot_metadata.json"
        metadata_path.write_text(json.dumps({
            "id": snapshot.id,
            "workspace_id": snapshot.workspace_id,
            "created_at": snapshot.created_at.isoformat(),
            "description": snapshot.description,
            "file_count": snapshot.file_count,
            "size_bytes": snapshot.size_bytes,
            "checksum": snapshot.checksum,
        }, indent=2))

        # Cleanup old snapshots
        await self._cleanup_old_snapshots()

        logger.info(f"Created snapshot {snapshot_id} for workspace {self.agent_id}")
        return snapshot_id

    async def restore(self, snapshot_id: str) -> None:
        """
        Restore workspace from a snapshot.

        Args:
            snapshot_id: ID of the snapshot to restore

        Raises:
            ValueError: If snapshot doesn't exist
        """
        self._ensure_initialized()

        snapshot_dir = self._snapshot_path / snapshot_id
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")

        def _restore():
            # Remove current files (but not snapshots or metadata)
            for item in self.path.iterdir():
                if item.name in (self.config.snapshot_dir, ".workspace_metadata.json"):
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

            # Copy snapshot files
            for item in snapshot_dir.iterdir():
                if item.name == ".snapshot_metadata.json":
                    continue
                if item.is_dir():
                    shutil.copytree(item, self.path / item.name)
                else:
                    shutil.copy2(item, self.path / item.name)

        await asyncio.get_event_loop().run_in_executor(None, _restore)
        logger.info(f"Restored workspace {self.agent_id} from snapshot {snapshot_id}")

    async def list_snapshots(self) -> List[Snapshot]:
        """
        List all snapshots for this workspace.

        Returns:
            List of Snapshot objects
        """
        self._ensure_initialized()

        snapshots = []
        if not self._snapshot_path.exists():
            return snapshots

        for snapshot_dir in sorted(self._snapshot_path.iterdir()):
            if not snapshot_dir.is_dir():
                continue
            metadata_path = snapshot_dir / ".snapshot_metadata.json"
            if metadata_path.exists():
                data = json.loads(metadata_path.read_text())
                snapshots.append(Snapshot(
                    id=data["id"],
                    workspace_id=data["workspace_id"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    description=data.get("description"),
                    file_count=data["file_count"],
                    size_bytes=data["size_bytes"],
                    checksum=data["checksum"],
                ))

        return snapshots

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.

        Args:
            snapshot_id: ID of the snapshot to delete

        Returns:
            True if snapshot was deleted
        """
        snapshot_dir = self._snapshot_path / snapshot_id
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
            logger.info(f"Deleted snapshot {snapshot_id}")
            return True
        return False

    async def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond the preserve limit."""
        snapshots = await self.list_snapshots()
        if len(snapshots) > self.config.preserve_snapshots:
            # Sort by created_at and remove oldest
            snapshots.sort(key=lambda s: s.created_at)
            for snapshot in snapshots[:-self.config.preserve_snapshots]:
                await self.delete_snapshot(snapshot.id)

    async def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write workspace metadata file."""
        path = self.path / ".workspace_metadata.json"
        path.write_text(json.dumps(metadata, indent=2))

    async def _check_extension(self, relative_path: str) -> None:
        """Check if file extension is allowed."""
        if self.config.allowed_extensions is None:
            return

        ext = Path(relative_path).suffix.lower()
        if ext and ext not in self.config.allowed_extensions:
            raise ValueError(f"File extension {ext} not allowed")

    async def _check_size_limit(self, additional_bytes: int) -> None:
        """Check if adding bytes would exceed size limit."""
        stats = await self.get_stats()
        max_bytes = self.config.max_size_mb * 1024 * 1024
        if stats.size_bytes + additional_bytes > max_bytes:
            raise ValueError(
                f"Would exceed workspace size limit of {self.config.max_size_mb}MB"
            )

    def _ensure_initialized(self) -> None:
        """Ensure workspace is initialized."""
        if not self._initialized:
            raise RuntimeError("Workspace not initialized. Call initialize() first.")


class WorkspaceManager:
    """
    Manages workspaces for all agents.

    Provides centralized creation, retrieval, and cleanup of agent workspaces.

    Example:
        config = WorkspaceConfig(base_path=Path("/tmp/workspaces"))
        manager = WorkspaceManager(config)

        # Create workspace
        workspace = await manager.create("agent-1")

        # Get existing workspace
        workspace = await manager.get("agent-1")

        # Destroy workspace
        await manager.destroy("agent-1")
    """

    def __init__(self, config: WorkspaceConfig):
        """
        Initialize the workspace manager.

        Args:
            config: Configuration for workspaces
        """
        self.config = config
        self._workspaces: Dict[str, Workspace] = {}
        self._lock = asyncio.Lock()

        # Ensure base path exists
        config.base_path.mkdir(parents=True, exist_ok=True)

    async def create(
        self,
        agent_id: str,
        config_override: Optional[WorkspaceConfig] = None
    ) -> Workspace:
        """
        Create a new workspace for an agent.

        Args:
            agent_id: ID of the agent
            config_override: Optional config override for this workspace

        Returns:
            The created Workspace

        Raises:
            ValueError: If workspace already exists
        """
        async with self._lock:
            if agent_id in self._workspaces:
                raise ValueError(f"Workspace for {agent_id} already exists")

            config = config_override or self.config
            workspace = Workspace(agent_id, config)
            await workspace.initialize()

            self._workspaces[agent_id] = workspace
            logger.info(f"Created workspace for agent {agent_id}")

            return workspace

    async def get(self, agent_id: str) -> Optional[Workspace]:
        """
        Get an existing workspace.

        Args:
            agent_id: ID of the agent

        Returns:
            The Workspace or None if not found
        """
        workspace = self._workspaces.get(agent_id)
        if workspace:
            return workspace

        # Check if workspace exists on disk
        workspace_path = self.config.base_path / agent_id
        if workspace_path.exists():
            workspace = Workspace(agent_id, self.config)
            await workspace.initialize()
            self._workspaces[agent_id] = workspace
            return workspace

        return None

    async def get_or_create(self, agent_id: str) -> Workspace:
        """
        Get an existing workspace or create a new one.

        Args:
            agent_id: ID of the agent

        Returns:
            The Workspace
        """
        workspace = await self.get(agent_id)
        if workspace:
            return workspace
        return await self.create(agent_id)

    async def destroy(self, agent_id: str) -> bool:
        """
        Destroy a workspace.

        Args:
            agent_id: ID of the agent

        Returns:
            True if workspace was destroyed
        """
        async with self._lock:
            workspace = self._workspaces.pop(agent_id, None)
            if workspace:
                await workspace.cleanup()
                logger.info(f"Destroyed workspace for agent {agent_id}")
                return True

            # Also check for orphaned workspace on disk
            workspace_path = self.config.base_path / agent_id
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
                logger.info(f"Cleaned up orphaned workspace for agent {agent_id}")
                return True

            return False

    async def list_workspaces(self) -> List[str]:
        """
        List all workspace agent IDs.

        Returns:
            List of agent IDs with workspaces
        """
        # Include both in-memory and on-disk workspaces
        disk_workspaces = {
            p.name for p in self.config.base_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        }
        memory_workspaces = set(self._workspaces.keys())
        return sorted(disk_workspaces | memory_workspaces)

    async def get_total_size(self) -> int:
        """
        Get total size of all workspaces in bytes.

        Returns:
            Total size in bytes
        """
        total = 0
        for agent_id in await self.list_workspaces():
            workspace = await self.get(agent_id)
            if workspace:
                stats = await workspace.get_stats()
                total += stats.size_bytes
        return total

    async def cleanup_all(self) -> int:
        """
        Cleanup all workspaces.

        Returns:
            Number of workspaces cleaned up
        """
        count = 0
        for agent_id in await self.list_workspaces():
            if await self.destroy(agent_id):
                count += 1
        return count

    async def __aenter__(self) -> "WorkspaceManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Don't cleanup by default - let explicit destroy handle it
        pass
