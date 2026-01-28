"""
Client-Server Architecture for Agent Frameworks.

Provides decoupled client-server communication for running agents
remotely. Supports:
    - WebSocket-based real-time communication
    - HTTP fallback for simple request/response
    - Authentication via tokens
    - Streaming responses
    - Multiple concurrent clients

Example Server:
    agent = MyAgent(backend)
    server = AgentServer(agent, ServerConfig(port=8765, auth_token="secret"))
    await server.start()

Example Client:
    client = AgentClient("ws://localhost:8765", auth_token="secret")
    await client.connect()
    result = await client.send_task("Write a function to sort a list")

    # Or stream response
    async for chunk in client.stream_response("Explain quicksort"):
        print(chunk, end="")
"""

from dataclasses import dataclass, field
from typing import (
    Optional,
    Dict,
    Any,
    AsyncIterator,
    Callable,
    Awaitable,
    List,
    Protocol,
    Union,
)
from enum import Enum
from datetime import datetime
import asyncio
import json
import uuid
import logging
import hashlib
import hmac
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the protocol."""

    # Client -> Server
    AUTH = "auth"
    TASK = "task"
    CANCEL = "cancel"
    PING = "ping"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Server -> Client
    AUTH_RESPONSE = "auth_response"
    TASK_RESPONSE = "task_response"
    TASK_STREAM = "task_stream"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    PONG = "pong"
    EVENT = "event"
    ERROR = "error"


@dataclass
class ServerConfig:
    """Configuration for the agent server."""

    host: str = "localhost"
    port: int = 8765
    auth_token: Optional[str] = None
    max_clients: int = 100
    request_timeout: float = 300.0  # 5 minutes
    heartbeat_interval: float = 30.0
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100  # per minute per client
    max_message_size: int = 10 * 1024 * 1024  # 10MB

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "max_clients": self.max_clients,
            "request_timeout": self.request_timeout,
            "heartbeat_interval": self.heartbeat_interval,
            "cors_origins": self.cors_origins,
            "rate_limit_requests": self.rate_limit_requests,
            "max_message_size": self.max_message_size,
        }


@dataclass
class TaskResult:
    """Result of a task execution."""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            success=data["success"],
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ClientInfo:
    """Information about a connected client."""

    client_id: str
    connected_at: datetime
    last_activity: datetime
    authenticated: bool = False
    subscriptions: List[str] = field(default_factory=list)
    request_count: int = 0
    rate_limit_window_start: float = field(default_factory=time.time)


class AgentProtocol(Protocol):
    """Protocol for agents that can be served."""

    async def process_task(self, task: str, context: Optional[Dict] = None) -> Any:
        """Process a task and return result."""
        ...

    async def stream_task(
        self, task: str, context: Optional[Dict] = None
    ) -> AsyncIterator[str]:
        """Stream task processing results."""
        ...


class AgentServer:
    """
    Server that hosts agent execution over WebSocket/HTTP.

    Provides a networked interface to an agent, allowing remote
    clients to submit tasks and receive results. Supports both
    synchronous request/response and streaming modes.

    Attributes:
        agent: The agent instance to serve
        config: Server configuration
    """

    def __init__(self, agent: AgentProtocol, config: Optional[ServerConfig] = None):
        """
        Initialize the agent server.

        Args:
            agent: Agent instance to serve
            config: Server configuration (uses defaults if not provided)
        """
        self.agent = agent
        self.config = config or ServerConfig()
        self._clients: Dict[str, ClientInfo] = {}
        self._websockets: Dict[str, Any] = {}  # client_id -> websocket
        self._running = False
        self._server = None
        self._http_server = None
        self._tasks: Dict[str, asyncio.Task] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def start(self) -> None:
        """Start the server."""
        try:
            import websockets
            import websockets.server
        except ImportError:
            logger.warning(
                "websockets library not installed. "
                "Install with: pip install websockets"
            )
            # Fall back to HTTP-only mode
            await self._start_http_only()
            return

        self._running = True

        # SSL context if configured
        ssl_context = None
        if self.config.ssl_cert and self.config.ssl_key:
            import ssl

            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.config.ssl_cert, self.config.ssl_key)

        # Start WebSocket server
        self._server = await websockets.server.serve(
            self._handle_websocket,
            self.config.host,
            self.config.port,
            ssl=ssl_context,
            max_size=self.config.max_message_size,
        )

        logger.info(
            f"AgentServer started on "
            f"{'wss' if ssl_context else 'ws'}://{self.config.host}:{self.config.port}"
        )

    async def _start_http_only(self) -> None:
        """Start HTTP-only server (fallback)."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error(
                "Neither websockets nor aiohttp installed. "
                "Install with: pip install websockets aiohttp"
            )
            raise RuntimeError("No web framework available")

        self._running = True

        app = web.Application()
        app.router.add_post("/task", self._handle_http_task)
        app.router.add_get("/health", self._handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        self._http_server = web.TCPSite(
            runner, self.config.host, self.config.port
        )
        await self._http_server.start()

        logger.info(
            f"AgentServer (HTTP-only) started on "
            f"http://{self.config.host}:{self.config.port}"
        )

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False

        # Cancel pending tasks
        for task_id, task in self._tasks.items():
            task.cancel()

        # Close WebSocket connections
        for ws in self._websockets.values():
            await ws.close()

        # Stop servers
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        if self._http_server:
            await self._http_server.stop()

        logger.info("AgentServer stopped")

    async def _handle_websocket(self, websocket, path: str = "/") -> None:
        """Handle WebSocket connection."""
        client_id = str(uuid.uuid4())
        client_info = ClientInfo(
            client_id=client_id,
            connected_at=datetime.now(),
            last_activity=datetime.now(),
        )

        self._clients[client_id] = client_info
        self._websockets[client_id] = websocket

        logger.info(f"Client {client_id} connected")

        try:
            async for message in websocket:
                try:
                    await self._process_message(client_id, message, websocket)
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await self._send_error(websocket, str(e))
        except Exception as e:
            logger.info(f"Client {client_id} disconnected: {e}")
        finally:
            del self._clients[client_id]
            del self._websockets[client_id]

    async def _process_message(
        self, client_id: str, raw_message: str, websocket
    ) -> None:
        """Process incoming message."""
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON")
            return

        msg_type = message.get("type")
        client = self._clients[client_id]
        client.last_activity = datetime.now()

        # Rate limiting
        if not self._check_rate_limit(client):
            await self._send_error(websocket, "Rate limit exceeded")
            return

        # Handle message types
        if msg_type == MessageType.AUTH.value:
            await self._handle_auth(client, message, websocket)

        elif msg_type == MessageType.PING.value:
            await self._send_message(websocket, {"type": MessageType.PONG.value})

        elif msg_type == MessageType.TASK.value:
            # Require authentication if token is configured
            if self.config.auth_token and not client.authenticated:
                await self._send_error(websocket, "Authentication required")
                return
            await self._handle_task(client_id, message, websocket)

        elif msg_type == MessageType.CANCEL.value:
            await self._handle_cancel(message)

        elif msg_type == MessageType.SUBSCRIBE.value:
            client.subscriptions.append(message.get("channel", ""))

        elif msg_type == MessageType.UNSUBSCRIBE.value:
            channel = message.get("channel", "")
            if channel in client.subscriptions:
                client.subscriptions.remove(channel)

        else:
            await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def _handle_auth(
        self, client: ClientInfo, message: Dict, websocket
    ) -> None:
        """Handle authentication."""
        token = message.get("token", "")

        if self.config.auth_token:
            # Simple token comparison (use constant-time comparison)
            if hmac.compare_digest(token, self.config.auth_token):
                client.authenticated = True
                await self._send_message(
                    websocket,
                    {
                        "type": MessageType.AUTH_RESPONSE.value,
                        "success": True,
                        "client_id": client.client_id,
                    },
                )
            else:
                await self._send_message(
                    websocket,
                    {
                        "type": MessageType.AUTH_RESPONSE.value,
                        "success": False,
                        "error": "Invalid token",
                    },
                )
        else:
            # No auth required
            client.authenticated = True
            await self._send_message(
                websocket,
                {
                    "type": MessageType.AUTH_RESPONSE.value,
                    "success": True,
                    "client_id": client.client_id,
                },
            )

    async def _handle_task(
        self, client_id: str, message: Dict, websocket
    ) -> None:
        """Handle task request."""
        task_id = message.get("task_id", str(uuid.uuid4()))
        task_content = message.get("task", "")
        context = message.get("context", {})
        stream = message.get("stream", False)

        start_time = datetime.now()

        # Create async task for execution
        async def execute_task():
            try:
                if stream:
                    # Streaming response
                    async for chunk in self.agent.stream_task(task_content, context):
                        await self._send_message(
                            websocket,
                            {
                                "type": MessageType.TASK_STREAM.value,
                                "task_id": task_id,
                                "chunk": chunk,
                            },
                        )

                    duration = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )
                    await self._send_message(
                        websocket,
                        {
                            "type": MessageType.TASK_COMPLETE.value,
                            "task_id": task_id,
                            "duration_ms": duration,
                        },
                    )
                else:
                    # Single response
                    result = await self.agent.process_task(task_content, context)
                    duration = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )

                    await self._send_message(
                        websocket,
                        {
                            "type": MessageType.TASK_RESPONSE.value,
                            "task_id": task_id,
                            "success": True,
                            "result": result,
                            "duration_ms": duration,
                        },
                    )

            except asyncio.CancelledError:
                await self._send_message(
                    websocket,
                    {
                        "type": MessageType.TASK_ERROR.value,
                        "task_id": task_id,
                        "error": "Task cancelled",
                    },
                )
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                await self._send_message(
                    websocket,
                    {
                        "type": MessageType.TASK_ERROR.value,
                        "task_id": task_id,
                        "error": str(e),
                    },
                )
            finally:
                if task_id in self._tasks:
                    del self._tasks[task_id]

        task = asyncio.create_task(execute_task())
        self._tasks[task_id] = task

    async def _handle_cancel(self, message: Dict) -> None:
        """Handle task cancellation."""
        task_id = message.get("task_id", "")
        if task_id in self._tasks:
            self._tasks[task_id].cancel()
            logger.info(f"Cancelled task {task_id}")

    async def _handle_http_task(self, request) -> Any:
        """Handle HTTP POST task request (aiohttp)."""
        from aiohttp import web

        # Check auth
        if self.config.auth_token:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return web.json_response(
                    {"error": "Authentication required"}, status=401
                )
            token = auth_header[7:]
            if not hmac.compare_digest(token, self.config.auth_token):
                return web.json_response({"error": "Invalid token"}, status=401)

        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        task_content = data.get("task", "")
        context = data.get("context", {})

        start_time = datetime.now()
        try:
            result = await asyncio.wait_for(
                self.agent.process_task(task_content, context),
                timeout=self.config.request_timeout,
            )
            duration = int((datetime.now() - start_time).total_seconds() * 1000)

            return web.json_response(
                {
                    "success": True,
                    "result": result,
                    "duration_ms": duration,
                }
            )
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": "Request timeout"}, status=504
            )
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, status=500
            )

    async def _handle_health(self, request) -> Any:
        """Health check endpoint."""
        from aiohttp import web

        return web.json_response(
            {
                "status": "healthy",
                "clients": len(self._clients),
                "pending_tasks": len(self._tasks),
            }
        )

    def _check_rate_limit(self, client: ClientInfo) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        window_duration = 60.0  # 1 minute

        # Reset window if needed
        if now - client.rate_limit_window_start > window_duration:
            client.rate_limit_window_start = now
            client.request_count = 0

        client.request_count += 1
        return client.request_count <= self.config.rate_limit_requests

    async def _send_message(self, websocket, message: Dict) -> None:
        """Send a message to a WebSocket client."""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def _send_error(self, websocket, error: str) -> None:
        """Send an error message."""
        await self._send_message(
            websocket, {"type": MessageType.ERROR.value, "error": error}
        )

    async def broadcast(self, channel: str, data: Any) -> None:
        """Broadcast a message to all subscribed clients."""
        message = {
            "type": MessageType.EVENT.value,
            "channel": channel,
            "data": data,
        }

        for client_id, client in self._clients.items():
            if channel in client.subscriptions:
                websocket = self._websockets.get(client_id)
                if websocket:
                    await self._send_message(websocket, message)

    def on_event(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)


class AgentClient:
    """
    Client that connects to remote agent server.

    Provides methods for sending tasks and receiving responses,
    including streaming support.

    Attributes:
        server_url: URL of the agent server
        auth_token: Optional authentication token
    """

    def __init__(
        self,
        server_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 300.0,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize the agent client.

        Args:
            server_url: WebSocket URL of the server (ws:// or wss://)
            auth_token: Optional authentication token
            timeout: Default timeout for operations
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.server_url = server_url
        self.auth_token = auth_token
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self._websocket = None
        self._client_id: Optional[str] = None
        self._connected = False
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._subscriptions: List[str] = []
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def connect(self) -> bool:
        """
        Connect to the agent server.

        Returns:
            True if connected and authenticated successfully
        """
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets library not installed. "
                "Install with: pip install websockets"
            )
            return False

        for attempt in range(self.reconnect_attempts):
            try:
                self._websocket = await websockets.connect(
                    self.server_url,
                    close_timeout=10,
                )
                self._connected = True

                # Start receive loop
                self._receive_task = asyncio.create_task(self._receive_loop())

                # Authenticate if token provided
                if self.auth_token:
                    auth_result = await self._authenticate()
                    if not auth_result:
                        await self.disconnect()
                        return False

                logger.info(f"Connected to {self.server_url}")
                return True

            except Exception as e:
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))

        return False

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        logger.info("Disconnected from server")

    async def send_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> TaskResult:
        """
        Send a task and wait for response.

        Args:
            task: Task description/content
            context: Optional context data
            timeout: Optional timeout override

        Returns:
            TaskResult with the response
        """
        if not self._connected:
            return TaskResult(
                task_id="",
                success=False,
                error="Not connected to server",
            )

        task_id = str(uuid.uuid4())
        effective_timeout = timeout or self.timeout

        # Create future for response
        response_future: asyncio.Future = asyncio.Future()
        self._pending_responses[task_id] = response_future

        try:
            # Send task message
            await self._send_message(
                {
                    "type": MessageType.TASK.value,
                    "task_id": task_id,
                    "task": task,
                    "context": context or {},
                    "stream": False,
                }
            )

            # Wait for response
            result = await asyncio.wait_for(
                response_future,
                timeout=effective_timeout,
            )
            return result

        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task_id,
                success=False,
                error=f"Request timed out after {effective_timeout}s",
            )
        finally:
            self._pending_responses.pop(task_id, None)

    async def stream_response(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        Send a task and stream the response.

        Args:
            task: Task description/content
            context: Optional context data
            timeout: Optional timeout override

        Yields:
            Response chunks as they arrive
        """
        if not self._connected:
            raise ConnectionError("Not connected to server")

        task_id = str(uuid.uuid4())
        effective_timeout = timeout or self.timeout

        # Create queue for stream chunks
        stream_queue: asyncio.Queue = asyncio.Queue()
        self._stream_queues[task_id] = stream_queue

        try:
            # Send task message
            await self._send_message(
                {
                    "type": MessageType.TASK.value,
                    "task_id": task_id,
                    "task": task,
                    "context": context or {},
                    "stream": True,
                }
            )

            # Yield chunks from queue
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_queue.get(),
                        timeout=effective_timeout,
                    )

                    if chunk is None:  # End of stream
                        break

                    if isinstance(chunk, Exception):
                        raise chunk

                    yield chunk

                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Stream timed out after {effective_timeout}s"
                    )

        finally:
            self._stream_queues.pop(task_id, None)

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a pending task."""
        await self._send_message(
            {
                "type": MessageType.CANCEL.value,
                "task_id": task_id,
            }
        )

    async def ping(self) -> bool:
        """Send ping and wait for pong."""
        if not self._connected:
            return False

        ping_id = str(uuid.uuid4())
        response_future: asyncio.Future = asyncio.Future()
        self._pending_responses[f"ping_{ping_id}"] = response_future

        try:
            await self._send_message(
                {"type": MessageType.PING.value, "id": ping_id}
            )
            await asyncio.wait_for(response_future, timeout=5.0)
            return True
        except (asyncio.TimeoutError, Exception):
            return False
        finally:
            self._pending_responses.pop(f"ping_{ping_id}", None)

    async def subscribe(self, channel: str) -> None:
        """Subscribe to an event channel."""
        await self._send_message(
            {"type": MessageType.SUBSCRIBE.value, "channel": channel}
        )
        self._subscriptions.append(channel)

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from an event channel."""
        await self._send_message(
            {"type": MessageType.UNSUBSCRIBE.value, "channel": channel}
        )
        if channel in self._subscriptions:
            self._subscriptions.remove(channel)

    def on_event(self, channel: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """Register an event handler for a channel."""
        if channel not in self._event_handlers:
            self._event_handlers[channel] = []
        self._event_handlers[channel].append(handler)

    async def _authenticate(self) -> bool:
        """Perform authentication."""
        response_future: asyncio.Future = asyncio.Future()
        self._pending_responses["auth"] = response_future

        try:
            await self._send_message(
                {
                    "type": MessageType.AUTH.value,
                    "token": self.auth_token,
                }
            )

            result = await asyncio.wait_for(response_future, timeout=10.0)
            if result.get("success"):
                self._client_id = result.get("client_id")
                return True
            else:
                logger.error(f"Authentication failed: {result.get('error')}")
                return False

        except asyncio.TimeoutError:
            logger.error("Authentication timed out")
            return False
        finally:
            self._pending_responses.pop("auth", None)

    async def _send_message(self, message: Dict) -> None:
        """Send a message to the server."""
        if self._websocket:
            await self._websocket.send(json.dumps(message))

    async def _receive_loop(self) -> None:
        """Background loop to receive messages."""
        try:
            async for raw_message in self._websocket:
                try:
                    message = json.loads(raw_message)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {raw_message[:100]}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            self._connected = False

    async def _handle_message(self, message: Dict) -> None:
        """Handle an incoming message."""
        msg_type = message.get("type")

        if msg_type == MessageType.AUTH_RESPONSE.value:
            future = self._pending_responses.get("auth")
            if future and not future.done():
                future.set_result(message)

        elif msg_type == MessageType.PONG.value:
            ping_id = message.get("id", "")
            future = self._pending_responses.get(f"ping_{ping_id}")
            if future and not future.done():
                future.set_result(message)

        elif msg_type == MessageType.TASK_RESPONSE.value:
            task_id = message.get("task_id")
            future = self._pending_responses.get(task_id)
            if future and not future.done():
                result = TaskResult(
                    task_id=task_id,
                    success=message.get("success", False),
                    result=message.get("result"),
                    duration_ms=message.get("duration_ms", 0),
                )
                future.set_result(result)

        elif msg_type == MessageType.TASK_STREAM.value:
            task_id = message.get("task_id")
            queue = self._stream_queues.get(task_id)
            if queue:
                await queue.put(message.get("chunk", ""))

        elif msg_type == MessageType.TASK_COMPLETE.value:
            task_id = message.get("task_id")
            queue = self._stream_queues.get(task_id)
            if queue:
                await queue.put(None)  # Signal end of stream

        elif msg_type == MessageType.TASK_ERROR.value:
            task_id = message.get("task_id")
            error = message.get("error", "Unknown error")

            # Handle for both regular and streaming tasks
            future = self._pending_responses.get(task_id)
            if future and not future.done():
                result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=error,
                )
                future.set_result(result)

            queue = self._stream_queues.get(task_id)
            if queue:
                await queue.put(Exception(error))

        elif msg_type == MessageType.EVENT.value:
            channel = message.get("channel", "")
            data = message.get("data")
            handlers = self._event_handlers.get(channel, [])
            for handler in handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

        elif msg_type == MessageType.ERROR.value:
            logger.error(f"Server error: {message.get('error')}")

    @property
    def connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    @property
    def client_id(self) -> Optional[str]:
        """Get the client ID assigned by the server."""
        return self._client_id


# HTTP Client for simpler use cases
class AgentHttpClient:
    """
    Simple HTTP client for agent server.

    Use this when you don't need streaming or WebSocket features.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize HTTP client.

        Args:
            server_url: HTTP URL of the server
            auth_token: Optional authentication token
            timeout: Request timeout
        """
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

    async def send_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Send a task via HTTP POST.

        Args:
            task: Task content
            context: Optional context

        Returns:
            TaskResult
        """
        try:
            import aiohttp
        except ImportError:
            return TaskResult(
                task_id="",
                success=False,
                error="aiohttp not installed. Install with: pip install aiohttp",
            )

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        payload = {
            "task": task,
            "context": context or {},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/task",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return TaskResult(
                            task_id=data.get("task_id", ""),
                            success=data.get("success", False),
                            result=data.get("result"),
                            duration_ms=data.get("duration_ms", 0),
                        )
                    else:
                        return TaskResult(
                            task_id="",
                            success=False,
                            error=data.get("error", f"HTTP {response.status}"),
                        )

        except Exception as e:
            return TaskResult(
                task_id="",
                success=False,
                error=str(e),
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return await response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
