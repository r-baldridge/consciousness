"""Git tools for version control operations.

This module provides tools for interacting with Git repositories,
including status, diff, commit, branch, and log operations.
"""

from __future__ import annotations
import asyncio
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..tool_base import Tool, ToolSchema, ToolResult, ToolPermission


async def run_git_command(
    args: List[str],
    cwd: Optional[Path] = None,
    timeout: float = 30.0
) -> tuple[int, str, str]:
    """Run a git command and return the result.

    Args:
        args: Git command arguments (without 'git')
        cwd: Working directory for the command
        timeout: Command timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["git"] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
        return (
            proc.returncode or 0,
            stdout.decode('utf-8', errors='replace'),
            stderr.decode('utf-8', errors='replace')
        )
    except asyncio.TimeoutError:
        proc.kill()
        return (-1, "", "Command timed out")
    except FileNotFoundError:
        return (-1, "", "Git is not installed")
    except Exception as e:
        return (-1, "", str(e))


class GitStatusTool(Tool):
    """Tool for getting Git repository status."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_status",
            description="Get the status of a Git repository, showing staged, modified, and untracked files.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository (default: current directory)"
                },
                "short": {
                    "type": "boolean",
                    "description": "Use short status format"
                },
                "show_branch": {
                    "type": "boolean",
                    "description": "Show branch information"
                }
            },
            required=[],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        path: Optional[str] = None,
        short: bool = False,
        show_branch: bool = True
    ) -> ToolResult:
        """Get Git repository status.

        Args:
            path: Repository path
            short: Use short format
            show_branch: Include branch info

        Returns:
            ToolResult with status information
        """
        cwd = Path(path).expanduser().resolve() if path else Path.cwd()

        args = ["status"]
        if short:
            args.append("--short")
        if show_branch:
            args.append("--branch")
        args.append("--porcelain=2")

        returncode, stdout, stderr = await run_git_command(args, cwd)

        if returncode != 0:
            return ToolResult.fail(f"Git status failed: {stderr}")

        # Parse porcelain v2 output
        status = {
            "branch": None,
            "upstream": None,
            "ahead": 0,
            "behind": 0,
            "staged": [],
            "modified": [],
            "untracked": [],
            "conflicted": []
        }

        for line in stdout.strip().split('\n'):
            if not line:
                continue

            if line.startswith('# branch.head '):
                status["branch"] = line.split(' ', 2)[2]
            elif line.startswith('# branch.upstream '):
                status["upstream"] = line.split(' ', 2)[2]
            elif line.startswith('# branch.ab '):
                parts = line.split(' ')
                status["ahead"] = int(parts[2][1:])
                status["behind"] = int(parts[3][1:])
            elif line.startswith('1 ') or line.startswith('2 '):
                # Tracked entry
                parts = line.split('\t')
                info = parts[0].split(' ')
                xy = info[1]
                file_path = parts[-1]

                if xy[0] != '.':
                    status["staged"].append({"file": file_path, "status": xy[0]})
                if xy[1] != '.':
                    status["modified"].append({"file": file_path, "status": xy[1]})
            elif line.startswith('u '):
                # Unmerged entry
                parts = line.split('\t')
                status["conflicted"].append(parts[-1])
            elif line.startswith('? '):
                # Untracked
                status["untracked"].append(line[2:])

        return ToolResult.ok(
            status,
            raw_output=stdout,
            repository=str(cwd)
        )


class GitDiffTool(Tool):
    """Tool for viewing Git diffs."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_diff",
            description="Show changes between commits, commit and working tree, etc.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "file": {
                    "type": "string",
                    "description": "Specific file to diff"
                },
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes (--cached)"
                },
                "commit": {
                    "type": "string",
                    "description": "Compare with specific commit"
                },
                "stat": {
                    "type": "boolean",
                    "description": "Show diffstat summary"
                }
            },
            required=[],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        path: Optional[str] = None,
        file: Optional[str] = None,
        staged: bool = False,
        commit: Optional[str] = None,
        stat: bool = False
    ) -> ToolResult:
        """Get Git diff.

        Args:
            path: Repository path
            file: Specific file to diff
            staged: Show staged changes
            commit: Compare with specific commit
            stat: Show statistics

        Returns:
            ToolResult with diff output
        """
        cwd = Path(path).expanduser().resolve() if path else Path.cwd()

        args = ["diff"]
        if staged:
            args.append("--cached")
        if commit:
            args.append(commit)
        if stat:
            args.append("--stat")
        if file:
            args.extend(["--", file])

        returncode, stdout, stderr = await run_git_command(args, cwd)

        if returncode != 0:
            return ToolResult.fail(f"Git diff failed: {stderr}")

        # Also get stat if not already requested
        stats = None
        if not stat:
            stat_args = args + ["--stat"]
            _, stat_stdout, _ = await run_git_command(stat_args, cwd)
            stats = stat_stdout.strip()

        return ToolResult.ok(
            stdout,
            stats=stats,
            repository=str(cwd),
            staged=staged,
            commit=commit
        )


class GitCommitTool(Tool):
    """Tool for creating Git commits."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_commit",
            description="Create a Git commit with staged changes.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage and commit (optional)"
                },
                "all": {
                    "type": "boolean",
                    "description": "Stage all modified tracked files (-a)"
                },
                "amend": {
                    "type": "boolean",
                    "description": "Amend the previous commit"
                }
            },
            required=["message"],
            permissions=[ToolPermission.WRITE]
        )

    async def execute(
        self,
        message: str,
        path: Optional[str] = None,
        files: Optional[List[str]] = None,
        all: bool = False,
        amend: bool = False
    ) -> ToolResult:
        """Create a Git commit.

        Args:
            message: Commit message
            path: Repository path
            files: Files to stage
            all: Stage all modified files
            amend: Amend previous commit

        Returns:
            ToolResult with commit information
        """
        cwd = Path(path).expanduser().resolve() if path else Path.cwd()

        # Stage specific files if provided
        if files:
            add_args = ["add", "--"] + files
            returncode, _, stderr = await run_git_command(add_args, cwd)
            if returncode != 0:
                return ToolResult.fail(f"Failed to stage files: {stderr}")

        # Build commit command
        args = ["commit", "-m", message]
        if all:
            args.append("-a")
        if amend:
            args.append("--amend")

        returncode, stdout, stderr = await run_git_command(args, cwd)

        if returncode != 0:
            return ToolResult.fail(f"Git commit failed: {stderr}")

        # Get the commit hash
        _, hash_out, _ = await run_git_command(["rev-parse", "HEAD"], cwd)
        commit_hash = hash_out.strip()

        return ToolResult.ok(
            {
                "hash": commit_hash,
                "message": message,
                "output": stdout
            },
            repository=str(cwd),
            amended=amend
        )


class GitBranchTool(Tool):
    """Tool for managing Git branches."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_branch",
            description="List, create, or delete Git branches.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "name": {
                    "type": "string",
                    "description": "Branch name for create/delete operations"
                },
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "delete", "checkout"],
                    "description": "Action to perform (default: list)"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force delete (-D) or force checkout"
                },
                "all": {
                    "type": "boolean",
                    "description": "Show all branches including remotes"
                }
            },
            required=[],
            permissions=[ToolPermission.WRITE]
        )

    async def execute(
        self,
        path: Optional[str] = None,
        name: Optional[str] = None,
        action: str = "list",
        force: bool = False,
        all: bool = False
    ) -> ToolResult:
        """Manage Git branches.

        Args:
            path: Repository path
            name: Branch name
            action: Action to perform
            force: Force the operation
            all: Include remote branches

        Returns:
            ToolResult with branch information
        """
        cwd = Path(path).expanduser().resolve() if path else Path.cwd()

        if action == "list":
            args = ["branch", "--format=%(refname:short) %(objectname:short) %(upstream:short)"]
            if all:
                args.append("-a")

            returncode, stdout, stderr = await run_git_command(args, cwd)
            if returncode != 0:
                return ToolResult.fail(f"Git branch list failed: {stderr}")

            branches = []
            for line in stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    branches.append({
                        "name": parts[0],
                        "commit": parts[1] if len(parts) > 1 else None,
                        "upstream": parts[2] if len(parts) > 2 else None
                    })

            # Get current branch
            _, current, _ = await run_git_command(["branch", "--show-current"], cwd)

            return ToolResult.ok(
                branches,
                current=current.strip(),
                repository=str(cwd)
            )

        elif action == "create":
            if not name:
                return ToolResult.fail("Branch name required for create")

            args = ["branch", name]
            returncode, stdout, stderr = await run_git_command(args, cwd)

            if returncode != 0:
                return ToolResult.fail(f"Failed to create branch: {stderr}")

            return ToolResult.ok(
                f"Created branch '{name}'",
                branch=name,
                repository=str(cwd)
            )

        elif action == "delete":
            if not name:
                return ToolResult.fail("Branch name required for delete")

            args = ["branch", "-D" if force else "-d", name]
            returncode, stdout, stderr = await run_git_command(args, cwd)

            if returncode != 0:
                return ToolResult.fail(f"Failed to delete branch: {stderr}")

            return ToolResult.ok(
                f"Deleted branch '{name}'",
                branch=name,
                repository=str(cwd)
            )

        elif action == "checkout":
            if not name:
                return ToolResult.fail("Branch name required for checkout")

            args = ["checkout"]
            if force:
                args.append("-f")
            args.append(name)

            returncode, stdout, stderr = await run_git_command(args, cwd)

            if returncode != 0:
                return ToolResult.fail(f"Failed to checkout branch: {stderr}")

            return ToolResult.ok(
                f"Switched to branch '{name}'",
                branch=name,
                repository=str(cwd)
            )

        else:
            return ToolResult.fail(f"Unknown action: {action}")


class GitLogTool(Tool):
    """Tool for viewing Git commit history."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_log",
            description="Show commit history with optional filtering.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of commits to show (default: 10)"
                },
                "oneline": {
                    "type": "boolean",
                    "description": "Show commits in one-line format"
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author"
                },
                "since": {
                    "type": "string",
                    "description": "Show commits after date (e.g., '2 weeks ago')"
                },
                "file": {
                    "type": "string",
                    "description": "Show commits affecting specific file"
                },
                "format": {
                    "type": "string",
                    "description": "Custom format string"
                }
            },
            required=[],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        path: Optional[str] = None,
        n: int = 10,
        oneline: bool = False,
        author: Optional[str] = None,
        since: Optional[str] = None,
        file: Optional[str] = None,
        format: Optional[str] = None
    ) -> ToolResult:
        """Get Git commit history.

        Args:
            path: Repository path
            n: Number of commits
            oneline: One-line format
            author: Author filter
            since: Date filter
            file: File filter
            format: Custom format

        Returns:
            ToolResult with commit history
        """
        cwd = Path(path).expanduser().resolve() if path else Path.cwd()

        args = ["log", f"-{n}"]

        if format:
            args.append(f"--format={format}")
        elif oneline:
            args.append("--oneline")
        else:
            # Use a structured format for parsing
            args.append("--format=%H%n%h%n%an%n%ae%n%at%n%s%n%b%n---COMMIT_END---")

        if author:
            args.append(f"--author={author}")
        if since:
            args.append(f"--since={since}")
        if file:
            args.extend(["--", file])

        returncode, stdout, stderr = await run_git_command(args, cwd)

        if returncode != 0:
            return ToolResult.fail(f"Git log failed: {stderr}")

        # Parse structured output
        if not (format or oneline):
            commits = []
            for block in stdout.split("---COMMIT_END---"):
                block = block.strip()
                if not block:
                    continue
                lines = block.split('\n')
                if len(lines) >= 6:
                    commits.append({
                        "hash": lines[0],
                        "short_hash": lines[1],
                        "author_name": lines[2],
                        "author_email": lines[3],
                        "timestamp": int(lines[4]),
                        "subject": lines[5],
                        "body": '\n'.join(lines[6:]).strip()
                    })

            return ToolResult.ok(
                commits,
                count=len(commits),
                repository=str(cwd)
            )

        return ToolResult.ok(
            stdout,
            repository=str(cwd)
        )


class GitAddTool(Tool):
    """Tool for staging files in Git."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_add",
            description="Stage files for commit.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the Git repository"
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage"
                },
                "all": {
                    "type": "boolean",
                    "description": "Stage all changes (-A)"
                },
                "update": {
                    "type": "boolean",
                    "description": "Stage only tracked files (-u)"
                }
            },
            required=[],
            permissions=[ToolPermission.WRITE]
        )

    async def execute(
        self,
        path: Optional[str] = None,
        files: Optional[List[str]] = None,
        all: bool = False,
        update: bool = False
    ) -> ToolResult:
        """Stage files for commit.

        Args:
            path: Repository path
            files: Specific files to stage
            all: Stage all changes
            update: Stage only tracked files

        Returns:
            ToolResult indicating success
        """
        cwd = Path(path).expanduser().resolve() if path else Path.cwd()

        if not files and not all and not update:
            return ToolResult.fail("Must specify files, --all, or --update")

        args = ["add"]
        if all:
            args.append("-A")
        elif update:
            args.append("-u")

        if files:
            args.extend(["--"] + files)

        returncode, stdout, stderr = await run_git_command(args, cwd)

        if returncode != 0:
            return ToolResult.fail(f"Git add failed: {stderr}")

        return ToolResult.ok(
            "Files staged successfully",
            files=files,
            all=all,
            update=update,
            repository=str(cwd)
        )


# Convenience function to get all git tools
def get_git_tools() -> List[Tool]:
    """Get all git tools.

    Returns:
        List of git tool instances
    """
    return [
        GitStatusTool(),
        GitDiffTool(),
        GitCommitTool(),
        GitBranchTool(),
        GitLogTool(),
        GitAddTool()
    ]
