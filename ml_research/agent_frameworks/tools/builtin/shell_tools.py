"""Shell tools for executing system commands.

This module provides tools for running shell commands with proper
sandboxing, timeout handling, and working directory management.
"""

from __future__ import annotations
import asyncio
import os
import re
import shlex
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

from ..tool_base import Tool, ToolSchema, ToolResult, ToolPermission


# Commands that are considered dangerous and may be blocked
DANGEROUS_COMMANDS = {
    "rm", "rmdir", "del", "format", "fdisk", "mkfs",
    "dd", "shred", "wipefs",
    "shutdown", "reboot", "halt", "poweroff",
    "passwd", "useradd", "userdel", "usermod",
    "chown", "chmod",
    "kill", "killall", "pkill",
    ">", ">>",  # Redirects can overwrite files
}

# Patterns that indicate dangerous operations
DANGEROUS_PATTERNS = [
    r"rm\s+-rf?\s+/",  # rm -rf /
    r"rm\s+-rf?\s+\*",  # rm -rf *
    r">\s*/dev/",  # Redirect to device
    r"chmod\s+777",  # Overly permissive
    r"curl.*\|\s*(bash|sh)",  # Pipe to shell
    r"wget.*\|\s*(bash|sh)",
    r"eval\s+",  # eval commands
    r"\$\(",  # Command substitution (can be dangerous)
]


@dataclass
class SandboxConfig:
    """Configuration for command sandboxing.

    Attributes:
        enabled: Whether sandboxing is enabled
        blocked_commands: Set of blocked command names
        blocked_patterns: List of regex patterns to block
        allowed_paths: Paths the command can access
        max_output_size: Maximum output size in bytes
        allow_network: Whether to allow network access
        allow_env_vars: Whether to pass environment variables
    """
    enabled: bool = True
    blocked_commands: Set[str] = field(default_factory=lambda: DANGEROUS_COMMANDS.copy())
    blocked_patterns: List[str] = field(default_factory=lambda: DANGEROUS_PATTERNS.copy())
    allowed_paths: Optional[List[Path]] = None
    max_output_size: int = 1024 * 1024  # 1MB
    allow_network: bool = True
    allow_env_vars: bool = True

    def is_command_allowed(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if a command is allowed by the sandbox.

        Args:
            command: The command to check

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        if not self.enabled:
            return True, None

        # Extract the base command
        parts = shlex.split(command)
        if not parts:
            return False, "Empty command"

        base_cmd = parts[0].split('/')[-1]  # Handle full paths

        # Check blocked commands
        if base_cmd in self.blocked_commands:
            return False, f"Command '{base_cmd}' is blocked"

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, command):
                return False, f"Command matches blocked pattern: {pattern}"

        return True, None


class BashTool(Tool):
    """Tool for executing shell commands with timeout and sandboxing."""

    def __init__(
        self,
        sandbox: Optional[SandboxConfig] = None,
        default_timeout: float = 120.0,
        working_dir: Optional[Path] = None,
        shell: str = "/bin/bash"
    ):
        """Initialize the bash tool.

        Args:
            sandbox: Sandbox configuration (None for default)
            default_timeout: Default command timeout in seconds
            working_dir: Default working directory
            shell: Shell to use for execution
        """
        self._sandbox = sandbox or SandboxConfig()
        self._default_timeout = default_timeout
        self._working_dir = working_dir or Path.cwd()
        self._shell = shell

    @property
    def schema(self) -> ToolSchema:
        permissions = [ToolPermission.EXECUTE]
        if not self._sandbox.enabled:
            permissions.append(ToolPermission.DANGEROUS)

        return ToolSchema(
            name="bash",
            description="Execute a shell command with optional timeout.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": f"Command timeout in seconds (default: {self._default_timeout})"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command"
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables"
                }
            },
            required=["command"],
            permissions=permissions
        )

    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> ToolResult:
        """Execute a shell command.

        Args:
            command: The command to execute
            timeout: Command timeout in seconds
            cwd: Working directory
            env: Additional environment variables

        Returns:
            ToolResult with command output
        """
        # Check sandbox
        allowed, reason = self._sandbox.is_command_allowed(command)
        if not allowed:
            return ToolResult.fail(f"Command blocked by sandbox: {reason}")

        # Determine working directory
        work_dir = Path(cwd).expanduser().resolve() if cwd else self._working_dir
        if not work_dir.exists():
            return ToolResult.fail(f"Working directory not found: {work_dir}")

        # Check path restrictions
        if self._sandbox.allowed_paths:
            path_allowed = any(
                str(work_dir).startswith(str(p.resolve()))
                for p in self._sandbox.allowed_paths
            )
            if not path_allowed:
                return ToolResult.fail(
                    f"Working directory outside allowed paths"
                )

        # Prepare environment
        cmd_env = os.environ.copy() if self._sandbox.allow_env_vars else {}
        if env:
            cmd_env.update(env)

        # Execute command
        effective_timeout = timeout or self._default_timeout
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=cmd_env,
                shell=True
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                return ToolResult.fail(
                    f"Command timed out after {effective_timeout} seconds",
                    timeout=True,
                    command=command
                )

            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')

            # Truncate if necessary
            if len(stdout_str) > self._sandbox.max_output_size:
                stdout_str = stdout_str[:self._sandbox.max_output_size] + "\n... (output truncated)"
            if len(stderr_str) > self._sandbox.max_output_size:
                stderr_str = stderr_str[:self._sandbox.max_output_size] + "\n... (output truncated)"

            # Build result
            if proc.returncode == 0:
                return ToolResult.ok(
                    stdout_str,
                    stderr=stderr_str,
                    return_code=proc.returncode,
                    command=command,
                    cwd=str(work_dir)
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout_str,
                    error=stderr_str or f"Command exited with code {proc.returncode}",
                    metadata={
                        "return_code": proc.returncode,
                        "command": command,
                        "cwd": str(work_dir)
                    }
                )

        except Exception as e:
            return ToolResult.fail(
                f"Error executing command: {e}",
                command=command
            )


class BackgroundProcessTool(Tool):
    """Tool for running commands in the background."""

    def __init__(
        self,
        sandbox: Optional[SandboxConfig] = None,
        working_dir: Optional[Path] = None
    ):
        """Initialize the background process tool.

        Args:
            sandbox: Sandbox configuration
            working_dir: Default working directory
        """
        self._sandbox = sandbox or SandboxConfig()
        self._working_dir = working_dir or Path.cwd()
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._counter = 0

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="background_process",
            description="Start a command in the background and get a process ID.",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The command to run in background"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command"
                }
            },
            required=["command"],
            permissions=[ToolPermission.EXECUTE]
        )

    async def execute(
        self,
        command: str,
        cwd: Optional[str] = None
    ) -> ToolResult:
        """Start a background process.

        Args:
            command: Command to run
            cwd: Working directory

        Returns:
            ToolResult with process ID
        """
        # Check sandbox
        allowed, reason = self._sandbox.is_command_allowed(command)
        if not allowed:
            return ToolResult.fail(f"Command blocked: {reason}")

        work_dir = Path(cwd).expanduser().resolve() if cwd else self._working_dir

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir
            )

            self._counter += 1
            process_id = f"bg_{self._counter}"
            self._processes[process_id] = proc

            return ToolResult.ok(
                {
                    "process_id": process_id,
                    "pid": proc.pid,
                    "command": command
                },
                pid=proc.pid
            )

        except Exception as e:
            return ToolResult.fail(f"Failed to start process: {e}")

    async def get_output(
        self,
        process_id: str,
        wait: bool = False
    ) -> ToolResult:
        """Get output from a background process.

        Args:
            process_id: The process identifier
            wait: Whether to wait for the process to complete

        Returns:
            ToolResult with process output
        """
        if process_id not in self._processes:
            return ToolResult.fail(f"Process not found: {process_id}")

        proc = self._processes[process_id]

        if wait:
            stdout, stderr = await proc.communicate()
            del self._processes[process_id]
            return ToolResult.ok(
                stdout.decode('utf-8', errors='replace'),
                stderr=stderr.decode('utf-8', errors='replace'),
                return_code=proc.returncode,
                completed=True
            )
        else:
            if proc.returncode is not None:
                # Process already completed
                return ToolResult.ok(
                    None,
                    return_code=proc.returncode,
                    completed=True
                )
            return ToolResult.ok(
                None,
                pid=proc.pid,
                completed=False
            )

    async def kill_process(self, process_id: str) -> ToolResult:
        """Kill a background process.

        Args:
            process_id: The process identifier

        Returns:
            ToolResult indicating success
        """
        if process_id not in self._processes:
            return ToolResult.fail(f"Process not found: {process_id}")

        proc = self._processes[process_id]
        try:
            proc.kill()
            await proc.wait()
            del self._processes[process_id]
            return ToolResult.ok(f"Process {process_id} killed")
        except Exception as e:
            return ToolResult.fail(f"Failed to kill process: {e}")


class WorkingDirectoryTool(Tool):
    """Tool for managing the working directory."""

    def __init__(self):
        """Initialize the working directory tool."""
        self._cwd = Path.cwd()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="working_directory",
            description="Get or change the current working directory.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "New working directory (omit to just get current)"
                }
            },
            required=[],
            permissions=[ToolPermission.READ]
        )

    async def execute(
        self,
        path: Optional[str] = None
    ) -> ToolResult:
        """Get or change working directory.

        Args:
            path: New directory path (None to just get current)

        Returns:
            ToolResult with current directory
        """
        if path is None:
            return ToolResult.ok(
                str(self._cwd),
                path=str(self._cwd)
            )

        new_path = Path(path).expanduser().resolve()

        if not new_path.exists():
            return ToolResult.fail(f"Directory not found: {path}")

        if not new_path.is_dir():
            return ToolResult.fail(f"Not a directory: {path}")

        self._cwd = new_path
        return ToolResult.ok(
            str(self._cwd),
            path=str(self._cwd),
            changed=True
        )

    @property
    def cwd(self) -> Path:
        """Get current working directory."""
        return self._cwd


# Convenience function to get shell tools
def get_shell_tools(
    sandbox: Optional[SandboxConfig] = None,
    working_dir: Optional[Path] = None
) -> List[Tool]:
    """Get all shell tools.

    Args:
        sandbox: Sandbox configuration
        working_dir: Default working directory

    Returns:
        List of shell tool instances
    """
    return [
        BashTool(sandbox=sandbox, working_dir=working_dir),
        WorkingDirectoryTool()
    ]
