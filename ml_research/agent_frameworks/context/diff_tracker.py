"""
Diff Tracker - Change Tracking and Diff Generation.

Provides comprehensive change tracking for files with support for:
- In-memory tracking of uncommitted changes
- Git integration for committed changes
- Unified diff generation
- Change event streaming

Example:
    tracker = DiffTracker(Path("/path/to/repo"))

    # Start tracking a file
    await tracker.track_file(Path("src/main.py"))

    # ... file is modified ...

    # Get the diff
    diff = await tracker.get_diff(Path("src/main.py"))
    print(diff)

    # Get all changes
    changes = await tracker.get_all_changes()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Callable, Any, AsyncIterator
from pathlib import Path
from enum import Enum
import asyncio
import difflib
import logging
import hashlib
from datetime import datetime
import subprocess


logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of file changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class ChangeEvent:
    """Represents a single change event."""
    file_path: Path
    change_type: ChangeType
    timestamp: datetime
    diff: Optional[str] = None
    old_path: Optional[Path] = None  # For renames
    lines_added: int = 0
    lines_removed: int = 0
    hunks: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a brief summary of the change."""
        if self.change_type == ChangeType.ADDED:
            return f"Added {self.file_path} (+{self.lines_added})"
        elif self.change_type == ChangeType.DELETED:
            return f"Deleted {self.file_path} (-{self.lines_removed})"
        elif self.change_type == ChangeType.MODIFIED:
            return f"Modified {self.file_path} (+{self.lines_added}/-{self.lines_removed})"
        elif self.change_type == ChangeType.RENAMED:
            return f"Renamed {self.old_path} -> {self.file_path}"
        return f"{self.change_type.value}: {self.file_path}"


@dataclass
class FileSnapshot:
    """Snapshot of a file's state at a point in time."""
    file_path: Path
    content: str
    content_hash: str
    timestamp: datetime
    exists: bool = True

    @classmethod
    def from_file(cls, file_path: Path) -> "FileSnapshot":
        """Create a snapshot from a file on disk."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            return cls(
                file_path=file_path,
                content=content,
                content_hash=content_hash,
                timestamp=datetime.now(),
                exists=True
            )
        except FileNotFoundError:
            return cls(
                file_path=file_path,
                content="",
                content_hash="",
                timestamp=datetime.now(),
                exists=False
            )
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return cls(
                file_path=file_path,
                content="",
                content_hash="",
                timestamp=datetime.now(),
                exists=False
            )

    @classmethod
    async def from_file_async(cls, file_path: Path) -> "FileSnapshot":
        """Create a snapshot asynchronously."""
        return await asyncio.to_thread(cls.from_file, file_path)


class DiffTracker:
    """
    Track changes to files and generate diffs.

    Supports both in-memory change tracking for uncommitted changes
    and git integration for committed changes.

    Example:
        tracker = DiffTracker(Path("/path/to/repo"))

        # Track specific files
        await tracker.track_file(Path("src/main.py"))
        await tracker.track_file(Path("src/utils.py"))

        # Track entire directory
        await tracker.track_directory(Path("src/"))

        # Get changes
        changes = await tracker.get_all_changes()
        for change in changes:
            print(change.summary())
            if change.diff:
                print(change.diff)
    """

    def __init__(
        self,
        root_path: Path,
        use_git: bool = True,
        on_change: Optional[Callable[[ChangeEvent], None]] = None
    ):
        """
        Initialize the diff tracker.

        Args:
            root_path: Root directory for tracking.
            use_git: Whether to use git for change detection.
            on_change: Optional callback for change events.
        """
        self.root_path = Path(root_path).resolve()
        self.use_git = use_git
        self.on_change = on_change

        # In-memory snapshots
        self._snapshots: Dict[Path, FileSnapshot] = {}
        self._tracked_files: Set[Path] = set()
        self._tracked_dirs: Set[Path] = set()

        # Git info
        self._is_git_repo = False
        self._git_root: Optional[Path] = None
        self._check_git()

    def _check_git(self) -> None:
        """Check if the root path is in a git repository."""
        if not self.use_git:
            return

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.root_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self._is_git_repo = True
                self._git_root = Path(result.stdout.strip())
                logger.debug(f"Git repository detected at {self._git_root}")
        except FileNotFoundError:
            logger.debug("Git not available")
        except Exception as e:
            logger.debug(f"Git check failed: {e}")

    async def track_file(self, file_path: Path) -> None:
        """
        Start tracking a file.

        Takes an initial snapshot of the file's content for later comparison.

        Args:
            file_path: Path to the file to track.
        """
        file_path = Path(file_path).resolve()
        self._tracked_files.add(file_path)
        snapshot = await FileSnapshot.from_file_async(file_path)
        self._snapshots[file_path] = snapshot
        logger.debug(f"Tracking {file_path}")

    async def track_directory(
        self,
        dir_path: Path,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> None:
        """
        Start tracking all files in a directory.

        Args:
            dir_path: Directory to track.
            extensions: File extensions to track (e.g., [".py", ".js"]).
            recursive: Whether to track subdirectories.
        """
        dir_path = Path(dir_path).resolve()
        self._tracked_dirs.add(dir_path)

        if extensions is None:
            extensions = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"]

        pattern = "**/*" if recursive else "*"

        for ext in extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    await self.track_file(file_path)

    async def untrack_file(self, file_path: Path) -> None:
        """Stop tracking a file."""
        file_path = Path(file_path).resolve()
        self._tracked_files.discard(file_path)
        self._snapshots.pop(file_path, None)

    async def get_diff(
        self,
        file_path: Path,
        context_lines: int = 3
    ) -> Optional[str]:
        """
        Get the unified diff for a tracked file.

        Args:
            file_path: Path to the file.
            context_lines: Number of context lines in the diff.

        Returns:
            Unified diff string, or None if no changes.
        """
        file_path = Path(file_path).resolve()

        if file_path not in self._snapshots:
            logger.warning(f"File not tracked: {file_path}")
            return None

        original = self._snapshots[file_path]
        current = await FileSnapshot.from_file_async(file_path)

        if original.content_hash == current.content_hash:
            return None

        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            original.content.splitlines(keepends=True),
            current.content.splitlines(keepends=True),
            fromfile=f"a/{file_path.name}",
            tofile=f"b/{file_path.name}",
            n=context_lines
        ))

        if not diff_lines:
            return None

        return "".join(diff_lines)

    async def get_change_event(
        self,
        file_path: Path,
        context_lines: int = 3
    ) -> Optional[ChangeEvent]:
        """
        Get a change event for a tracked file.

        Args:
            file_path: Path to the file.
            context_lines: Number of context lines in the diff.

        Returns:
            ChangeEvent if the file has changed, None otherwise.
        """
        file_path = Path(file_path).resolve()

        if file_path not in self._snapshots:
            return None

        original = self._snapshots[file_path]
        current = await FileSnapshot.from_file_async(file_path)

        # Determine change type
        if not original.exists and current.exists:
            change_type = ChangeType.ADDED
        elif original.exists and not current.exists:
            change_type = ChangeType.DELETED
        elif original.content_hash != current.content_hash:
            change_type = ChangeType.MODIFIED
        else:
            return None

        # Generate diff
        diff = await self.get_diff(file_path, context_lines)

        # Count lines
        lines_added = 0
        lines_removed = 0
        if diff:
            for line in diff.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    lines_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    lines_removed += 1

        # Parse hunks
        hunks = self._parse_hunks(diff) if diff else []

        return ChangeEvent(
            file_path=file_path,
            change_type=change_type,
            timestamp=datetime.now(),
            diff=diff,
            lines_added=lines_added,
            lines_removed=lines_removed,
            hunks=hunks
        )

    async def get_all_changes(self, context_lines: int = 3) -> List[ChangeEvent]:
        """
        Get all changes for tracked files.

        Returns:
            List of ChangeEvent objects for all changed files.
        """
        changes: List[ChangeEvent] = []

        for file_path in self._tracked_files:
            event = await self.get_change_event(file_path, context_lines)
            if event:
                changes.append(event)

        return changes

    async def refresh_snapshot(self, file_path: Path) -> None:
        """
        Refresh the snapshot for a file.

        Call this after changes have been committed or saved.

        Args:
            file_path: Path to the file.
        """
        file_path = Path(file_path).resolve()
        if file_path in self._tracked_files:
            snapshot = await FileSnapshot.from_file_async(file_path)
            self._snapshots[file_path] = snapshot

    async def refresh_all_snapshots(self) -> None:
        """Refresh snapshots for all tracked files."""
        for file_path in self._tracked_files:
            await self.refresh_snapshot(file_path)

    async def get_git_diff(
        self,
        staged: bool = False,
        commit: Optional[str] = None,
        file_paths: Optional[List[Path]] = None
    ) -> Optional[str]:
        """
        Get git diff for the repository.

        Args:
            staged: If True, show staged changes only.
            commit: Compare against a specific commit.
            file_paths: Limit to specific files.

        Returns:
            Git diff output, or None if not a git repo.
        """
        if not self._is_git_repo:
            return None

        cmd = ["git", "diff"]

        if staged:
            cmd.append("--cached")
        elif commit:
            cmd.append(commit)

        if file_paths:
            cmd.append("--")
            cmd.extend([str(fp) for fp in file_paths])

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True
            )
            return result.stdout if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"Git diff failed: {e}")
            return None

    async def get_git_status(self) -> Dict[str, List[Path]]:
        """
        Get git status for the repository.

        Returns:
            Dictionary with keys 'staged', 'unstaged', 'untracked'.
        """
        result = {
            "staged": [],
            "unstaged": [],
            "untracked": []
        }

        if not self._is_git_repo:
            return result

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "status", "--porcelain"],
                cwd=self.root_path,
                capture_output=True,
                text=True
            )

            if proc.returncode != 0:
                return result

            for line in proc.stdout.splitlines():
                if len(line) < 4:
                    continue

                index_status = line[0]
                worktree_status = line[1]
                file_path = self.root_path / line[3:]

                if index_status in "MADRC":
                    result["staged"].append(file_path)
                if worktree_status in "MADRC":
                    result["unstaged"].append(file_path)
                if index_status == "?" and worktree_status == "?":
                    result["untracked"].append(file_path)

        except Exception as e:
            logger.warning(f"Git status failed: {e}")

        return result

    async def get_git_log(
        self,
        max_commits: int = 10,
        file_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Get git log for the repository.

        Args:
            max_commits: Maximum number of commits to return.
            file_path: Limit to a specific file.

        Returns:
            List of commit dictionaries.
        """
        if not self._is_git_repo:
            return []

        cmd = [
            "git", "log",
            f"-{max_commits}",
            "--format=%H|%an|%ae|%at|%s"
        ]

        if file_path:
            cmd.extend(["--", str(file_path)])

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True
            )

            if proc.returncode != 0:
                return []

            commits = []
            for line in proc.stdout.splitlines():
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commits.append({
                        "hash": parts[0],
                        "author_name": parts[1],
                        "author_email": parts[2],
                        "timestamp": datetime.fromtimestamp(int(parts[3])),
                        "message": parts[4]
                    })

            return commits

        except Exception as e:
            logger.warning(f"Git log failed: {e}")
            return []

    async def get_commit_diff(
        self,
        commit_hash: str,
        file_paths: Optional[List[Path]] = None
    ) -> Optional[str]:
        """
        Get the diff for a specific commit.

        Args:
            commit_hash: The commit hash.
            file_paths: Limit to specific files.

        Returns:
            Diff string for the commit.
        """
        if not self._is_git_repo:
            return None

        cmd = ["git", "show", "--format=", commit_hash]

        if file_paths:
            cmd.append("--")
            cmd.extend([str(fp) for fp in file_paths])

        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True
            )

            return proc.stdout if proc.returncode == 0 else None

        except Exception as e:
            logger.warning(f"Git show failed: {e}")
            return None

    def _parse_hunks(self, diff: str) -> List[Dict[str, Any]]:
        """Parse diff hunks from a unified diff."""
        hunks = []
        current_hunk: Optional[Dict[str, Any]] = None

        for line in diff.splitlines():
            if line.startswith("@@"):
                # Parse hunk header: @@ -start,count +start,count @@
                import re
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)?", line)
                if match:
                    if current_hunk:
                        hunks.append(current_hunk)

                    current_hunk = {
                        "old_start": int(match.group(1)),
                        "old_count": int(match.group(2) or 1),
                        "new_start": int(match.group(3)),
                        "new_count": int(match.group(4) or 1),
                        "header": match.group(5) or "",
                        "lines": []
                    }
            elif current_hunk is not None:
                if line.startswith("+") and not line.startswith("+++"):
                    current_hunk["lines"].append({"type": "add", "content": line[1:]})
                elif line.startswith("-") and not line.startswith("---"):
                    current_hunk["lines"].append({"type": "remove", "content": line[1:]})
                elif line.startswith(" "):
                    current_hunk["lines"].append({"type": "context", "content": line[1:]})

        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    async def watch_changes(
        self,
        poll_interval: float = 1.0
    ) -> AsyncIterator[ChangeEvent]:
        """
        Watch for changes to tracked files.

        Yields ChangeEvent objects as changes are detected.

        Args:
            poll_interval: Seconds between checks.

        Yields:
            ChangeEvent for each detected change.
        """
        while True:
            for file_path in list(self._tracked_files):
                event = await self.get_change_event(file_path)
                if event:
                    # Update snapshot after detecting change
                    await self.refresh_snapshot(file_path)

                    # Notify callback
                    if self.on_change:
                        try:
                            self.on_change(event)
                        except Exception as e:
                            logger.warning(f"Change callback failed: {e}")

                    yield event

            await asyncio.sleep(poll_interval)

    def get_tracked_files(self) -> List[Path]:
        """Get list of all tracked files."""
        return list(self._tracked_files)

    def is_tracking(self, file_path: Path) -> bool:
        """Check if a file is being tracked."""
        return Path(file_path).resolve() in self._tracked_files

    async def get_combined_diff(self, context_lines: int = 3) -> str:
        """
        Get a combined diff of all changes.

        Returns:
            Combined unified diff string.
        """
        changes = await self.get_all_changes(context_lines)
        diffs = [c.diff for c in changes if c.diff]
        return "\n".join(diffs)

    def clear(self) -> None:
        """Clear all tracked files and snapshots."""
        self._snapshots.clear()
        self._tracked_files.clear()
        self._tracked_dirs.clear()
