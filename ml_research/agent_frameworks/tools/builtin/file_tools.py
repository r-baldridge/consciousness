"""File system tools for reading, writing, and manipulating files.

This module provides tools for common file operations with proper
error handling and permission controls.
"""

from __future__ import annotations
import os
import fnmatch
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..tool_base import Tool, ToolSchema, ToolResult, ToolPermission


class ReadFileTool(Tool):
    """Tool for reading file contents with optional line range."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the read file tool.

        Args:
            base_path: Optional base path to restrict file access to
        """
        self._base_path = base_path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read the contents of a file. Supports optional line range.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, inclusive)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (1-indexed, inclusive)"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)"
                }
            },
            required=["path"],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        encoding: str = "utf-8"
    ) -> ToolResult:
        """Read file contents.

        Args:
            path: Path to the file
            start_line: Starting line (1-indexed)
            end_line: Ending line (1-indexed)
            encoding: File encoding

        Returns:
            ToolResult with file contents or error
        """
        try:
            file_path = Path(path).expanduser().resolve()

            # Check base path restriction
            if self._base_path:
                if not str(file_path).startswith(str(self._base_path.resolve())):
                    return ToolResult.fail(
                        f"Access denied: path outside allowed directory"
                    )

            if not file_path.exists():
                return ToolResult.fail(f"File not found: {path}")

            if not file_path.is_file():
                return ToolResult.fail(f"Not a file: {path}")

            content = file_path.read_text(encoding=encoding)

            # Handle line range
            if start_line is not None or end_line is not None:
                lines = content.splitlines(keepends=True)
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)

                # Validate bounds
                if start_idx < 0:
                    start_idx = 0
                if end_idx > len(lines):
                    end_idx = len(lines)

                content = "".join(lines[start_idx:end_idx])

            return ToolResult.ok(
                content,
                path=str(file_path),
                size=file_path.stat().st_size,
                lines=content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            )

        except UnicodeDecodeError as e:
            return ToolResult.fail(f"Encoding error: {e}")
        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error reading file: {e}")


class WriteFileTool(Tool):
    """Tool for writing content to files."""

    def __init__(
        self,
        base_path: Optional[Path] = None,
        create_dirs: bool = True
    ):
        """Initialize the write file tool.

        Args:
            base_path: Optional base path to restrict file access to
            create_dirs: Whether to create parent directories if they don't exist
        """
        self._base_path = base_path
        self._create_dirs = create_dirs

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write content to a file. Creates the file if it doesn't exist.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)"
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to file instead of overwriting"
                }
            },
            required=["path", "content"],
            permissions=[ToolPermission.WRITE]
        )

    async def execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        append: bool = False
    ) -> ToolResult:
        """Write content to a file.

        Args:
            path: Path to the file
            content: Content to write
            encoding: File encoding
            append: Whether to append instead of overwrite

        Returns:
            ToolResult indicating success or failure
        """
        try:
            file_path = Path(path).expanduser().resolve()

            # Check base path restriction
            if self._base_path:
                if not str(file_path).startswith(str(self._base_path.resolve())):
                    return ToolResult.fail(
                        f"Access denied: path outside allowed directory"
                    )

            # Create parent directories if needed
            if self._create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write or append
            mode = "a" if append else "w"
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)

            return ToolResult.ok(
                f"Successfully wrote to {path}",
                path=str(file_path),
                bytes_written=len(content.encode(encoding)),
                mode="append" if append else "write"
            )

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error writing file: {e}")


class EditFileTool(Tool):
    """Tool for editing files using search/replace operations."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the edit file tool.

        Args:
            base_path: Optional base path to restrict file access to
        """
        self._base_path = base_path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit_file",
            description="Edit a file by replacing text. Supports regex patterns.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to search for"
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace with"
                },
                "use_regex": {
                    "type": "boolean",
                    "description": "Treat old_text as a regex pattern"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: first only)"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)"
                }
            },
            required=["path", "old_text", "new_text"],
            permissions=[ToolPermission.WRITE]
        )

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        use_regex: bool = False,
        replace_all: bool = False,
        encoding: str = "utf-8"
    ) -> ToolResult:
        """Edit a file by replacing text.

        Args:
            path: Path to the file
            old_text: Text to find
            new_text: Replacement text
            use_regex: Whether to use regex matching
            replace_all: Whether to replace all occurrences
            encoding: File encoding

        Returns:
            ToolResult with edit details or error
        """
        try:
            file_path = Path(path).expanduser().resolve()

            # Check base path restriction
            if self._base_path:
                if not str(file_path).startswith(str(self._base_path.resolve())):
                    return ToolResult.fail(
                        f"Access denied: path outside allowed directory"
                    )

            if not file_path.exists():
                return ToolResult.fail(f"File not found: {path}")

            content = file_path.read_text(encoding=encoding)
            original_content = content

            if use_regex:
                count = replace_all and 0 or 1
                if replace_all:
                    new_content = re.sub(old_text, new_text, content)
                    matches = len(re.findall(old_text, content))
                else:
                    new_content = re.sub(old_text, new_text, content, count=1)
                    matches = 1 if re.search(old_text, content) else 0
            else:
                if replace_all:
                    matches = content.count(old_text)
                    new_content = content.replace(old_text, new_text)
                else:
                    matches = 1 if old_text in content else 0
                    new_content = content.replace(old_text, new_text, 1)

            if matches == 0:
                return ToolResult.fail(f"Text not found in file: {old_text[:50]}...")

            if new_content == original_content:
                return ToolResult.ok(
                    "No changes made",
                    path=str(file_path),
                    replacements=0
                )

            file_path.write_text(new_content, encoding=encoding)

            return ToolResult.ok(
                f"Successfully edited {path}",
                path=str(file_path),
                replacements=matches,
                bytes_changed=len(new_content.encode(encoding)) - len(original_content.encode(encoding))
            )

        except re.error as e:
            return ToolResult.fail(f"Invalid regex pattern: {e}")
        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error editing file: {e}")


class GlobTool(Tool):
    """Tool for finding files by glob pattern."""

    def __init__(
        self,
        base_path: Optional[Path] = None,
        max_results: int = 1000
    ):
        """Initialize the glob tool.

        Args:
            base_path: Base path for searches (default: current directory)
            max_results: Maximum number of results to return
        """
        self._base_path = base_path or Path.cwd()
        self._max_results = max_results

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description="Find files matching a glob pattern (e.g., **/*.py).",
            parameters={
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files"
                },
                "path": {
                    "type": "string",
                    "description": "Base path to search in (default: current directory)"
                },
                "include_dirs": {
                    "type": "boolean",
                    "description": "Include directories in results"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files/directories"
                }
            },
            required=["pattern"],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        include_dirs: bool = False,
        include_hidden: bool = False
    ) -> ToolResult:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py")
            path: Base path to search in
            include_dirs: Whether to include directories
            include_hidden: Whether to include hidden files

        Returns:
            ToolResult with list of matching paths
        """
        try:
            base = Path(path).expanduser().resolve() if path else self._base_path

            if not base.exists():
                return ToolResult.fail(f"Path not found: {base}")

            if not base.is_dir():
                return ToolResult.fail(f"Not a directory: {base}")

            matches = []
            for match in base.glob(pattern):
                # Skip hidden files if not included
                if not include_hidden:
                    parts = match.relative_to(base).parts
                    if any(p.startswith('.') for p in parts):
                        continue

                # Skip directories if not included
                if match.is_dir() and not include_dirs:
                    continue

                matches.append(str(match))

                if len(matches) >= self._max_results:
                    break

            # Sort by modification time (newest first)
            matches.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)

            return ToolResult.ok(
                matches,
                count=len(matches),
                base_path=str(base),
                pattern=pattern,
                truncated=len(matches) >= self._max_results
            )

        except Exception as e:
            return ToolResult.fail(f"Error in glob search: {e}")


class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the list directory tool.

        Args:
            base_path: Optional base path to restrict access to
        """
        self._base_path = base_path

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="List contents of a directory with file information.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files/directories"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively (single level deep)"
                }
            },
            required=["path"],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        path: str,
        show_hidden: bool = False,
        recursive: bool = False
    ) -> ToolResult:
        """List directory contents.

        Args:
            path: Directory path to list
            show_hidden: Whether to show hidden files
            recursive: Whether to list one level of subdirectories

        Returns:
            ToolResult with directory listing
        """
        try:
            dir_path = Path(path).expanduser().resolve()

            if self._base_path:
                if not str(dir_path).startswith(str(self._base_path.resolve())):
                    return ToolResult.fail(
                        f"Access denied: path outside allowed directory"
                    )

            if not dir_path.exists():
                return ToolResult.fail(f"Directory not found: {path}")

            if not dir_path.is_dir():
                return ToolResult.fail(f"Not a directory: {path}")

            entries = []
            for entry in dir_path.iterdir():
                if not show_hidden and entry.name.startswith('.'):
                    continue

                stat = entry.stat()
                entry_info = {
                    "name": entry.name,
                    "path": str(entry),
                    "type": "directory" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else None,
                    "modified": stat.st_mtime
                }
                entries.append(entry_info)

                # Add one level of subdirectory contents if recursive
                if recursive and entry.is_dir():
                    try:
                        for subentry in entry.iterdir():
                            if not show_hidden and subentry.name.startswith('.'):
                                continue
                            substat = subentry.stat()
                            entries.append({
                                "name": subentry.name,
                                "path": str(subentry),
                                "type": "directory" if subentry.is_dir() else "file",
                                "size": substat.st_size if subentry.is_file() else None,
                                "modified": substat.st_mtime
                            })
                    except PermissionError:
                        pass

            # Sort: directories first, then by name
            entries.sort(key=lambda e: (e["type"] != "directory", e["name"].lower()))

            return ToolResult.ok(
                entries,
                path=str(dir_path),
                count=len(entries)
            )

        except PermissionError:
            return ToolResult.fail(f"Permission denied: {path}")
        except Exception as e:
            return ToolResult.fail(f"Error listing directory: {e}")


# Convenience function to get all file tools
def get_file_tools(base_path: Optional[Path] = None) -> List[Tool]:
    """Get all file tools with optional base path restriction.

    Args:
        base_path: Optional base path to restrict all file operations to

    Returns:
        List of file tool instances
    """
    return [
        ReadFileTool(base_path=base_path),
        WriteFileTool(base_path=base_path),
        EditFileTool(base_path=base_path),
        GlobTool(base_path=base_path),
        ListDirectoryTool(base_path=base_path)
    ]
