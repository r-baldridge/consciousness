"""
Sandboxed Tool Executor for Agent Frameworks.

Provides secure and controlled execution of agent tools with:
    - Multiple sandboxing options (none, subprocess, docker)
    - Timeout handling
    - Resource limits (memory, CPU)
    - Execution context management
    - Comprehensive audit logging

Example:
    executor = ToolExecutor(
        sandbox_mode=SandboxMode.SUBPROCESS,
        default_timeout=30.0,
        resource_limits=ResourceLimits(max_memory_mb=512),
    )

    result = await executor.execute(
        tool_name="file_write",
        arguments={"path": "/tmp/test.txt", "content": "hello"},
        context=ExecutionContext(working_directory="/tmp"),
    )

    # Check audit log
    for entry in executor.get_audit_log():
        print(f"{entry.timestamp}: {entry.tool_name} - {entry.status}")
"""

from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    Dict,
    Any,
    Callable,
    Awaitable,
    Protocol,
    Union,
)
from enum import Enum
from datetime import datetime
import asyncio
import subprocess
import os
import json
import tempfile
import shutil
import logging
import traceback
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class SandboxMode(Enum):
    """Sandboxing modes for tool execution."""

    NONE = "none"  # Direct execution in current process
    SUBPROCESS = "subprocess"  # Execute in subprocess with restrictions
    DOCKER = "docker"  # Execute in Docker container


@dataclass
class ResourceLimits:
    """Resource limits for tool execution."""

    max_memory_mb: int = 512  # Maximum memory in MB
    max_cpu_percent: float = 100.0  # Maximum CPU percentage
    max_file_size_mb: int = 100  # Maximum file size for I/O operations
    max_open_files: int = 100  # Maximum number of open files
    max_processes: int = 10  # Maximum spawned processes
    network_enabled: bool = True  # Allow network access
    filesystem_readonly: bool = False  # Read-only filesystem access

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_file_size_mb": self.max_file_size_mb,
            "max_open_files": self.max_open_files,
            "max_processes": self.max_processes,
            "network_enabled": self.network_enabled,
            "filesystem_readonly": self.filesystem_readonly,
        }


@dataclass
class ExecutionContext:
    """Context for tool execution."""

    working_directory: str = "."
    environment_variables: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timeout: Optional[float] = None
    resource_limits: Optional[ResourceLimits] = None
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "working_directory": self.working_directory,
            "environment_variables": self.environment_variables,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timeout": self.timeout,
            "resource_limits": (
                self.resource_limits.to_dict() if self.resource_limits else None
            ),
            "allowed_paths": self.allowed_paths,
            "blocked_paths": self.blocked_paths,
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    exit_code: int = 0
    duration_ms: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "resource_usage": self.resource_usage,
            "metadata": self.metadata,
        }


@dataclass
class AuditLogEntry:
    """Entry in the audit log for tool executions."""

    id: str
    timestamp: datetime
    tool_name: str
    arguments: Dict[str, Any]
    context: Dict[str, Any]
    status: str  # "started", "completed", "failed", "timeout"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: int = 0
    sandbox_mode: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "context": self.context,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "sandbox_mode": self.sandbox_mode,
        }


class Tool(Protocol):
    """Protocol for tool implementations."""

    name: str

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: ExecutionContext,
    ) -> Any:
        """Execute the tool with given arguments."""
        ...


class ToolExecutor:
    """
    Executes agent tools with sandboxing, timeouts, and audit logging.

    Supports multiple sandboxing modes:
        - NONE: Direct execution (fast but no isolation)
        - SUBPROCESS: Execution in isolated subprocess
        - DOCKER: Full container isolation

    Attributes:
        sandbox_mode: Default sandboxing mode
        default_timeout: Default timeout for tool execution
        resource_limits: Default resource limits
        audit_log: List of audit log entries
    """

    def __init__(
        self,
        sandbox_mode: SandboxMode = SandboxMode.NONE,
        default_timeout: float = 30.0,
        resource_limits: Optional[ResourceLimits] = None,
        audit_log_path: Optional[Union[str, Path]] = None,
        max_audit_entries: int = 10000,
        docker_image: str = "python:3.11-slim",
    ):
        """
        Initialize the tool executor.

        Args:
            sandbox_mode: Default sandboxing mode
            default_timeout: Default timeout in seconds
            resource_limits: Default resource limits
            audit_log_path: Optional path to persist audit log
            max_audit_entries: Maximum audit log entries to keep in memory
            docker_image: Docker image for container sandbox
        """
        self.sandbox_mode = sandbox_mode
        self.default_timeout = default_timeout
        self.resource_limits = resource_limits or ResourceLimits()
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        self.max_audit_entries = max_audit_entries
        self.docker_image = docker_image

        self._tools: Dict[str, Tool] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._lock = asyncio.Lock()

        # Built-in tool handlers
        self._builtin_handlers: Dict[str, Callable] = {
            "shell": self._execute_shell,
            "file_read": self._execute_file_read,
            "file_write": self._execute_file_write,
            "http_request": self._execute_http_request,
        }

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool for execution.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def list_tools(self) -> List[str]:
        """List registered tool names."""
        return list(self._tools.keys()) + list(self._builtin_handlers.keys())

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
        sandbox_mode: Optional[SandboxMode] = None,
    ) -> ToolResult:
        """
        Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Execution context (uses defaults if not provided)
            sandbox_mode: Override default sandbox mode

        Returns:
            ToolResult with output or error
        """
        effective_context = context or ExecutionContext()
        effective_sandbox = sandbox_mode or self.sandbox_mode
        effective_timeout = effective_context.timeout or self.default_timeout
        effective_limits = effective_context.resource_limits or self.resource_limits

        # Create audit entry
        audit_id = str(uuid.uuid4())
        start_time = datetime.now()

        audit_entry = AuditLogEntry(
            id=audit_id,
            timestamp=start_time,
            tool_name=tool_name,
            arguments=self._sanitize_for_log(arguments),
            context=effective_context.to_dict(),
            status="started",
            sandbox_mode=effective_sandbox.value,
        )

        await self._add_audit_entry(audit_entry)

        try:
            # Dispatch to appropriate execution method
            if effective_sandbox == SandboxMode.NONE:
                result = await self._execute_direct(
                    tool_name, arguments, effective_context, effective_timeout
                )
            elif effective_sandbox == SandboxMode.SUBPROCESS:
                result = await self._execute_subprocess(
                    tool_name,
                    arguments,
                    effective_context,
                    effective_timeout,
                    effective_limits,
                )
            elif effective_sandbox == SandboxMode.DOCKER:
                result = await self._execute_docker(
                    tool_name,
                    arguments,
                    effective_context,
                    effective_timeout,
                    effective_limits,
                )
            else:
                result = ToolResult(
                    success=False,
                    error=f"Unknown sandbox mode: {effective_sandbox}",
                )

            # Update audit entry
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            audit_entry.status = "completed" if result.success else "failed"
            audit_entry.result = result.to_dict()
            audit_entry.duration_ms = duration
            audit_entry.error = result.error

            result.duration_ms = duration

        except asyncio.TimeoutError:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            result = ToolResult(
                success=False,
                error=f"Tool execution timed out after {effective_timeout}s",
                duration_ms=duration,
            )
            audit_entry.status = "timeout"
            audit_entry.error = result.error
            audit_entry.duration_ms = duration

        except Exception as e:
            duration = int((datetime.now() - start_time).total_seconds() * 1000)
            result = ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}",
                duration_ms=duration,
            )
            audit_entry.status = "failed"
            audit_entry.error = str(e)
            audit_entry.duration_ms = duration
            logger.error(f"Tool {tool_name} failed: {e}\n{traceback.format_exc()}")

        await self._update_audit_entry(audit_entry)
        return result

    async def _execute_direct(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ExecutionContext,
        timeout: float,
    ) -> ToolResult:
        """Execute tool directly in current process."""
        # Check for registered tool
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            output = await asyncio.wait_for(
                tool.execute(arguments, context),
                timeout=timeout,
            )
            return ToolResult(success=True, output=output)

        # Check for builtin handler
        if tool_name in self._builtin_handlers:
            handler = self._builtin_handlers[tool_name]
            output = await asyncio.wait_for(
                handler(arguments, context),
                timeout=timeout,
            )
            return ToolResult(success=True, output=output)

        return ToolResult(
            success=False,
            error=f"Unknown tool: {tool_name}",
        )

    async def _execute_subprocess(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ExecutionContext,
        timeout: float,
        limits: ResourceLimits,
    ) -> ToolResult:
        """Execute tool in isolated subprocess."""
        # Create temporary script
        script_content = self._generate_subprocess_script(
            tool_name, arguments, context
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(context.environment_variables)

            # Build command with resource limits
            cmd = ["python", script_path]

            # On Unix, we can use ulimit for some limits
            if os.name != "nt":
                # Memory limit (soft)
                mem_bytes = limits.max_memory_mb * 1024 * 1024
                preexec_fn = lambda: self._set_subprocess_limits(limits)
            else:
                preexec_fn = None

            # Run subprocess
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=context.working_directory,
                preexec_fn=preexec_fn,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            if proc.returncode == 0:
                try:
                    output = json.loads(stdout.decode())
                except json.JSONDecodeError:
                    output = stdout.decode()
                return ToolResult(
                    success=True,
                    output=output,
                    exit_code=proc.returncode,
                )
            else:
                return ToolResult(
                    success=False,
                    error=stderr.decode() or f"Exit code: {proc.returncode}",
                    exit_code=proc.returncode,
                )

        finally:
            # Clean up script
            try:
                os.unlink(script_path)
            except Exception:
                pass

    async def _execute_docker(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ExecutionContext,
        timeout: float,
        limits: ResourceLimits,
    ) -> ToolResult:
        """Execute tool in Docker container."""
        # Check if Docker is available
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    error="Docker is not available",
                )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Docker is not installed",
            )

        # Create temporary directory for mounting
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write script
            script_content = self._generate_subprocess_script(
                tool_name, arguments, context
            )
            script_path = Path(tmpdir) / "tool_script.py"
            script_path.write_text(script_content)

            # Build docker command
            cmd = [
                "docker", "run",
                "--rm",
                f"--memory={limits.max_memory_mb}m",
                f"--cpus={limits.max_cpu_percent / 100}",
                f"--pids-limit={limits.max_processes}",
                "-v", f"{tmpdir}:/workspace:rw",
                "-w", "/workspace",
            ]

            # Network settings
            if not limits.network_enabled:
                cmd.extend(["--network", "none"])

            # Read-only filesystem
            if limits.filesystem_readonly:
                cmd.append("--read-only")

            # Environment variables
            for key, value in context.environment_variables.items():
                cmd.extend(["-e", f"{key}={value}"])

            cmd.extend([self.docker_image, "python", "/workspace/tool_script.py"])

            # Run container
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill container
                await asyncio.create_subprocess_exec(
                    "docker", "kill", f"tool-{id(proc)}",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                raise

            if proc.returncode == 0:
                try:
                    output = json.loads(stdout.decode())
                except json.JSONDecodeError:
                    output = stdout.decode()
                return ToolResult(
                    success=True,
                    output=output,
                    exit_code=proc.returncode,
                )
            else:
                return ToolResult(
                    success=False,
                    error=stderr.decode() or f"Exit code: {proc.returncode}",
                    exit_code=proc.returncode,
                )

    def _generate_subprocess_script(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ExecutionContext,
    ) -> str:
        """Generate Python script for subprocess execution."""
        args_json = json.dumps(arguments)
        context_json = json.dumps(context.to_dict())

        return f'''
import json
import sys
import os

TOOL_NAME = {json.dumps(tool_name)}
ARGUMENTS = json.loads({json.dumps(args_json)})
CONTEXT = json.loads({json.dumps(context_json)})

def execute_shell(args, ctx):
    import subprocess
    result = subprocess.run(
        args.get("command", "echo 'no command'"),
        shell=True,
        capture_output=True,
        text=True,
        cwd=ctx.get("working_directory", "."),
        timeout=ctx.get("timeout", 30),
    )
    return {{"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}}

def execute_file_read(args, ctx):
    path = args.get("path", "")
    with open(path, "r") as f:
        return {{"content": f.read()}}

def execute_file_write(args, ctx):
    path = args.get("path", "")
    content = args.get("content", "")
    with open(path, "w") as f:
        f.write(content)
    return {{"written": len(content), "path": path}}

def execute_http_request(args, ctx):
    import urllib.request
    url = args.get("url", "")
    method = args.get("method", "GET")
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=ctx.get("timeout", 30)) as resp:
        return {{"status": resp.status, "body": resp.read().decode()}}

HANDLERS = {{
    "shell": execute_shell,
    "file_read": execute_file_read,
    "file_write": execute_file_write,
    "http_request": execute_http_request,
}}

if __name__ == "__main__":
    try:
        if TOOL_NAME in HANDLERS:
            result = HANDLERS[TOOL_NAME](ARGUMENTS, CONTEXT)
            print(json.dumps(result))
        else:
            print(json.dumps({{"error": f"Unknown tool: {{TOOL_NAME}}"}}))
            sys.exit(1)
    except Exception as e:
        print(json.dumps({{"error": str(e)}}), file=sys.stderr)
        sys.exit(1)
'''

    def _set_subprocess_limits(self, limits: ResourceLimits) -> None:
        """Set resource limits for subprocess (Unix only)."""
        try:
            import resource

            # Memory limit
            mem_bytes = limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (mem_bytes, mem_bytes),
            )

            # File descriptor limit
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (limits.max_open_files, limits.max_open_files),
            )

            # Process limit
            resource.setrlimit(
                resource.RLIMIT_NPROC,
                (limits.max_processes, limits.max_processes),
            )
        except (ImportError, ValueError, OSError):
            pass  # Resource module not available or limits can't be set

    async def _execute_shell(
        self,
        arguments: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Built-in shell command execution."""
        command = arguments.get("command", "")
        timeout = context.timeout or self.default_timeout

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=context.working_directory,
            env={**os.environ, **context.environment_variables},
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "returncode": proc.returncode,
        }

    async def _execute_file_read(
        self,
        arguments: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Built-in file read operation."""
        path = arguments.get("path", "")

        # Security check
        if not self._is_path_allowed(path, context):
            raise PermissionError(f"Path not allowed: {path}")

        content = await asyncio.to_thread(Path(path).read_text)
        return {"content": content, "path": path}

    async def _execute_file_write(
        self,
        arguments: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Built-in file write operation."""
        path = arguments.get("path", "")
        content = arguments.get("content", "")

        # Security check
        if not self._is_path_allowed(path, context):
            raise PermissionError(f"Path not allowed: {path}")

        # Check file size limit
        limits = context.resource_limits or self.resource_limits
        if len(content) > limits.max_file_size_mb * 1024 * 1024:
            raise ValueError(
                f"Content exceeds max file size ({limits.max_file_size_mb}MB)"
            )

        await asyncio.to_thread(Path(path).write_text, content)
        return {"written": len(content), "path": path}

    async def _execute_http_request(
        self,
        arguments: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Built-in HTTP request execution."""
        import urllib.request

        url = arguments.get("url", "")
        method = arguments.get("method", "GET")
        headers = arguments.get("headers", {})
        body = arguments.get("body")

        # Check network permission
        limits = context.resource_limits or self.resource_limits
        if not limits.network_enabled:
            raise PermissionError("Network access is disabled")

        req = urllib.request.Request(url, method=method)
        for key, value in headers.items():
            req.add_header(key, value)

        if body:
            req.data = body.encode() if isinstance(body, str) else body

        timeout = context.timeout or self.default_timeout

        def do_request():
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return {
                    "status": resp.status,
                    "headers": dict(resp.headers),
                    "body": resp.read().decode(),
                }

        return await asyncio.to_thread(do_request)

    def _is_path_allowed(self, path: str, context: ExecutionContext) -> bool:
        """Check if a path is allowed based on context."""
        resolved = Path(path).resolve()

        # Check blocked paths
        for blocked in context.blocked_paths:
            if str(resolved).startswith(str(Path(blocked).resolve())):
                return False

        # If allowed paths specified, check whitelist
        if context.allowed_paths:
            for allowed in context.allowed_paths:
                if str(resolved).startswith(str(Path(allowed).resolve())):
                    return True
            return False

        return True

    def _sanitize_for_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data before logging."""
        sensitive_keys = {"password", "secret", "token", "key", "auth", "credential"}
        sanitized = {}

        for key, value in data.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_for_log(value)
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = f"{value[:100]}... [truncated {len(value)} chars]"
            else:
                sanitized[key] = value

        return sanitized

    async def _add_audit_entry(self, entry: AuditLogEntry) -> None:
        """Add an entry to the audit log."""
        async with self._lock:
            self._audit_log.append(entry)

            # Trim if too many entries
            if len(self._audit_log) > self.max_audit_entries:
                self._audit_log = self._audit_log[-self.max_audit_entries:]

            # Persist if path configured
            if self.audit_log_path:
                await self._persist_audit_entry(entry)

    async def _update_audit_entry(self, entry: AuditLogEntry) -> None:
        """Update an existing audit entry."""
        async with self._lock:
            for i, e in enumerate(self._audit_log):
                if e.id == entry.id:
                    self._audit_log[i] = entry
                    break

            if self.audit_log_path:
                await self._persist_audit_entry(entry)

    async def _persist_audit_entry(self, entry: AuditLogEntry) -> None:
        """Persist audit entry to disk."""
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to JSONL file
            line = json.dumps(entry.to_dict()) + "\n"
            await asyncio.to_thread(
                lambda: self.audit_log_path.open("a").write(line)
            )
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")

    def get_audit_log(
        self,
        tool_name: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Get audit log entries with optional filters.

        Args:
            tool_name: Filter by tool name
            status: Filter by status
            since: Only entries after this time
            limit: Maximum entries to return

        Returns:
            List of matching audit log entries
        """
        entries = self._audit_log.copy()

        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]

        if status:
            entries = [e for e in entries if e.status == status]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    async def clear_audit_log(self) -> None:
        """Clear the in-memory audit log."""
        async with self._lock:
            self._audit_log.clear()
