"""
Aider compatibility layer.

This module provides a drop-in replacement for Aider-style workflows,
implementing Aider's key commands and features while using our backend.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import subprocess
import json
import re
import logging
import asyncio

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class AiderConfig:
    """Configuration for Aider-compatible agent.

    Attributes:
        repo_path: Path to the repository
        auto_commit: Whether to auto-commit changes
        commit_prefix: Prefix for auto-commit messages
        edit_format: Format for edits (diff, whole, etc.)
        map_tokens: Max tokens for repository map
        verbose: Enable verbose output
    """
    repo_path: Path = field(default_factory=Path.cwd)
    auto_commit: bool = True
    commit_prefix: str = "aider: "
    edit_format: str = "diff"
    map_tokens: int = 4000
    verbose: bool = False


@dataclass
class FileChange:
    """Represents a change to a file.

    Attributes:
        path: Path to the file
        original: Original content
        modified: Modified content
        change_type: Type of change (create, modify, delete)
    """
    path: Path
    original: Optional[str]
    modified: Optional[str]
    change_type: str  # "create", "modify", "delete"

    @property
    def diff(self) -> str:
        """Generate a simple diff representation."""
        if self.change_type == "create":
            return f"+++ {self.path}\n@@ new file @@\n{self.modified}"
        elif self.change_type == "delete":
            return f"--- {self.path}\n@@ deleted @@"
        else:
            return f"--- {self.path}\n+++ {self.path}\n(modified)"


class AiderCompatAgent:
    """Drop-in replacement for Aider workflows.

    This class implements Aider's key commands (/add, /drop, /ask, /code)
    and features like repository mapping and auto-commit.

    Example:
        agent = AiderCompatAgent(backend, config)

        # Add files to context
        await agent.add_file("src/main.py")
        await agent.add_file("src/utils.py")

        # Ask about the code
        answer = await agent.ask("What does the main function do?")

        # Make changes
        modified = await agent.code("Add logging to all functions")
        print(f"Modified {len(modified)} files")

        # Drop files from context
        await agent.drop_file("src/utils.py")
    """

    SYSTEM_PROMPT = """You are an expert coding assistant. You help users understand and modify their codebase.

When making code changes, follow these rules:
1. Only modify the files that have been added to the chat
2. Preserve existing code style and conventions
3. Make minimal, focused changes
4. Explain your changes clearly

Current files in context:
{file_list}

Repository structure:
{repo_map}
"""

    EDIT_PROMPT = """Based on the following instruction, make the necessary code changes.

Instruction: {instruction}

Current file contents:
{file_contents}

Respond with the complete modified file contents for each file that needs changes.
Use the following format for each file:

```path/to/file.py
<complete file contents>
```

Only include files that need to be changed."""

    def __init__(
        self,
        backend: Optional["LLMBackend"] = None,
        config: Optional[AiderConfig] = None
    ):
        """Initialize the Aider-compatible agent.

        Args:
            backend: LLM backend for generation
            config: Aider configuration
        """
        self.backend = backend
        self.config = config or AiderConfig()
        self._files: Dict[str, str] = {}  # path -> content
        self._repo_map: Optional[str] = None
        self._conversation: List[Dict[str, str]] = []

    async def add_file(self, path: str) -> bool:
        """Add a file to the context.

        Equivalent to Aider's /add command.

        Args:
            path: Path to the file (relative to repo)

        Returns:
            True if file was added successfully
        """
        full_path = self.config.repo_path / path

        if not full_path.exists():
            logger.warning(f"File not found: {path}")
            return False

        try:
            content = full_path.read_text()
            self._files[path] = content
            logger.info(f"Added file to context: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to add file {path}: {e}")
            return False

    async def add_files(self, paths: List[str]) -> int:
        """Add multiple files to context.

        Args:
            paths: List of file paths

        Returns:
            Number of files successfully added
        """
        added = 0
        for path in paths:
            if await self.add_file(path):
                added += 1
        return added

    async def drop_file(self, path: str) -> bool:
        """Remove a file from the context.

        Equivalent to Aider's /drop command.

        Args:
            path: Path to the file

        Returns:
            True if file was removed
        """
        if path in self._files:
            del self._files[path]
            logger.info(f"Dropped file from context: {path}")
            return True
        return False

    async def drop_all(self) -> None:
        """Remove all files from context."""
        self._files.clear()
        logger.info("Dropped all files from context")

    def get_files(self) -> List[str]:
        """Get list of files currently in context.

        Returns:
            List of file paths
        """
        return list(self._files.keys())

    async def ask(self, question: str) -> str:
        """Ask a question about the code.

        Equivalent to Aider's /ask command. Does not modify files.

        Args:
            question: The question to ask

        Returns:
            The assistant's response
        """
        if not self.backend:
            return "Error: No LLM backend configured"

        # Build context
        system = self._build_system_prompt()
        file_contents = self._format_file_contents()

        prompt = f"""Question about the code:

{question}

Current files:
{file_contents}

Provide a helpful answer. Do not suggest code changes unless asked."""

        self._conversation.append({"role": "user", "content": prompt})

        try:
            from ..backends.backend_base import LLMConfig

            messages = [{"role": "system", "content": system}] + self._conversation

            response = await self.backend.complete(
                messages=messages,
                config=LLMConfig(
                    model=self.backend.default_model,
                    max_tokens=4000,
                    temperature=0.3
                )
            )

            answer = response.content
            self._conversation.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            logger.error(f"Ask failed: {e}")
            return f"Error: {str(e)}"

    async def code(self, instruction: str) -> List[str]:
        """Make code changes based on instruction.

        Equivalent to Aider's /code command.

        Args:
            instruction: What changes to make

        Returns:
            List of modified file paths
        """
        if not self.backend:
            logger.error("No LLM backend configured")
            return []

        if not self._files:
            logger.warning("No files in context. Use add_file() first.")
            return []

        # Build context
        system = self._build_system_prompt()
        file_contents = self._format_file_contents()

        prompt = self.EDIT_PROMPT.format(
            instruction=instruction,
            file_contents=file_contents
        )

        try:
            from ..backends.backend_base import LLMConfig

            response = await self.backend.complete(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                config=LLMConfig(
                    model=self.backend.default_model,
                    max_tokens=8000,
                    temperature=0.2
                )
            )

            # Parse response and apply changes
            changes = self._parse_changes(response.content)
            modified_files = await self._apply_changes(changes)

            # Auto-commit if enabled
            if self.config.auto_commit and modified_files:
                await self._auto_commit(instruction, modified_files)

            return modified_files

        except Exception as e:
            logger.error(f"Code change failed: {e}")
            return []

    async def run(self, command: str) -> str:
        """Run a shell command.

        Equivalent to Aider's /run command.

        Args:
            command: Shell command to run

        Returns:
            Command output
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def commit(self, message: str) -> bool:
        """Create a git commit.

        Args:
            message: Commit message

        Returns:
            True if commit was successful
        """
        try:
            # Stage all modified files
            for path in self._files.keys():
                subprocess.run(
                    ["git", "add", path],
                    cwd=self.config.repo_path,
                    check=True,
                    capture_output=True
                )

            # Create commit
            full_message = f"{self.config.commit_prefix}{message}"
            subprocess.run(
                ["git", "commit", "-m", full_message],
                cwd=self.config.repo_path,
                check=True,
                capture_output=True
            )

            logger.info(f"Created commit: {message}")
            return True

        except subprocess.CalledProcessError as e:
            logger.warning(f"Commit failed: {e}")
            return False

    async def _auto_commit(self, instruction: str, modified_files: List[str]) -> bool:
        """Auto-commit changes with generated message.

        Args:
            instruction: The instruction that caused changes
            modified_files: List of modified file paths

        Returns:
            True if commit was successful
        """
        # Generate commit message
        files_str = ", ".join(modified_files[:3])
        if len(modified_files) > 3:
            files_str += f" (+{len(modified_files) - 3} more)"

        message = f"{instruction[:50]}"
        if len(instruction) > 50:
            message += "..."

        return await self.commit(message)

    def _build_system_prompt(self) -> str:
        """Build system prompt with context."""
        file_list = "\n".join(f"- {path}" for path in self._files.keys())

        return self.SYSTEM_PROMPT.format(
            file_list=file_list or "(no files added)",
            repo_map=self._repo_map or "(repository map not generated)"
        )

    def _format_file_contents(self) -> str:
        """Format all file contents for prompt."""
        parts = []
        for path, content in self._files.items():
            parts.append(f"```{path}\n{content}\n```")
        return "\n\n".join(parts)

    def _parse_changes(self, response: str) -> List[FileChange]:
        """Parse LLM response to extract file changes.

        Args:
            response: The LLM's response

        Returns:
            List of FileChange objects
        """
        changes = []

        # Match code blocks with file paths
        pattern = r"```(\S+)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        for path, content in matches:
            # Clean up path (remove language hints)
            if '/' not in path and '\\' not in path:
                # This is probably a language hint, not a path
                continue

            path = path.strip()
            content = content.strip()

            original = self._files.get(path)

            if original is None:
                change_type = "create"
            elif content != original:
                change_type = "modify"
            else:
                continue  # No change

            changes.append(FileChange(
                path=Path(path),
                original=original,
                modified=content,
                change_type=change_type
            ))

        return changes

    async def _apply_changes(self, changes: List[FileChange]) -> List[str]:
        """Apply file changes to disk.

        Args:
            changes: List of changes to apply

        Returns:
            List of successfully modified file paths
        """
        modified = []

        for change in changes:
            full_path = self.config.repo_path / change.path

            try:
                if change.change_type == "delete":
                    if full_path.exists():
                        full_path.unlink()
                        modified.append(str(change.path))
                else:
                    # Create parent directories if needed
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write content
                    if change.modified is not None:
                        full_path.write_text(change.modified)
                        self._files[str(change.path)] = change.modified
                        modified.append(str(change.path))

                logger.info(f"Applied {change.change_type}: {change.path}")

            except Exception as e:
                logger.error(f"Failed to apply change to {change.path}: {e}")

        return modified

    async def build_repo_map(self, max_tokens: Optional[int] = None) -> str:
        """Build a repository map for context.

        Similar to Aider's repository mapping feature.

        Args:
            max_tokens: Maximum tokens for the map

        Returns:
            Repository map string
        """
        max_tokens = max_tokens or self.config.map_tokens

        repo_path = self.config.repo_path
        map_lines = []

        # Get file structure
        for path in repo_path.rglob("*"):
            if path.is_file():
                # Skip common non-code files
                if any(part.startswith('.') for part in path.parts):
                    continue
                if path.suffix in ['.pyc', '.pyo', '.so', '.o', '.a']:
                    continue

                rel_path = path.relative_to(repo_path)
                map_lines.append(str(rel_path))

        # Sort and truncate
        map_lines.sort()

        # Simple token estimation
        map_text = "\n".join(map_lines)
        while len(map_text) // 4 > max_tokens and map_lines:
            map_lines = map_lines[::2]  # Keep every other line
            map_text = "\n".join(map_lines)

        self._repo_map = map_text
        return map_text

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent state.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "files_in_context": len(self._files),
            "file_paths": list(self._files.keys()),
            "total_content_size": sum(len(c) for c in self._files.values()),
            "conversation_turns": len(self._conversation),
            "has_repo_map": self._repo_map is not None,
            "auto_commit": self.config.auto_commit,
            "repo_path": str(self.config.repo_path)
        }
