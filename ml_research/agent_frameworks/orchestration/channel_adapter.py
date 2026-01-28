"""
Channel Adapters for multi-channel communication.

This module provides adapters for various communication channels
(Slack, Discord, Telegram, Webhooks, WebSockets) along with a
MultiChannelManager for unified channel management.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Type of message."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    COMMAND = "command"
    REACTION = "reaction"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class Message:
    """
    A message in the communication system.

    Attributes:
        id: Unique message identifier
        content: Message content
        type: Type of message
        sender: Sender identifier
        channel: Channel the message was received on
        timestamp: When the message was created
        metadata: Additional message metadata
        reply_to: ID of message this is replying to
        attachments: List of attachment data
    """
    content: str
    type: MessageType = MessageType.TEXT
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: Optional[str] = None
    channel: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "sender": self.sender,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "reply_to": self.reply_to,
            "attachments": self.attachments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data["content"],
            type=MessageType(data.get("type", "text")),
            sender=data.get("sender"),
            channel=data.get("channel"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
            reply_to=data.get("reply_to"),
            attachments=data.get("attachments", []),
        )


class ChannelAdapter(ABC):
    """
    Abstract base class for communication channel adapters.

    Adapters handle the specifics of connecting to and communicating
    through different platforms (Slack, Discord, etc.).
    """

    def __init__(self, channel_name: str):
        """
        Initialize the adapter.

        Args:
            channel_name: Name identifier for this channel
        """
        self.channel_name = channel_name
        self._connected = False
        self._message_handlers: List[Callable[[Message], Awaitable[None]]] = []

    @property
    def is_connected(self) -> bool:
        """Check if the adapter is connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the channel.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the channel."""
        ...

    @abstractmethod
    async def receive_message(self) -> Message:
        """
        Receive the next message from the channel.

        Returns:
            The received message

        Raises:
            ConnectionError: If not connected
        """
        ...

    @abstractmethod
    async def send_message(self, message: Message) -> None:
        """
        Send a message to the channel.

        Args:
            message: The message to send

        Raises:
            ConnectionError: If not connected
        """
        ...

    def add_message_handler(
        self,
        handler: Callable[[Message], Awaitable[None]]
    ) -> None:
        """
        Add a handler for incoming messages.

        Args:
            handler: Async callable to handle messages
        """
        self._message_handlers.append(handler)

    def remove_message_handler(
        self,
        handler: Callable[[Message], Awaitable[None]]
    ) -> None:
        """
        Remove a message handler.

        Args:
            handler: The handler to remove
        """
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)

    async def _dispatch_message(self, message: Message) -> None:
        """
        Dispatch a message to all handlers.

        Args:
            message: The message to dispatch
        """
        for handler in self._message_handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")


class SlackAdapter(ChannelAdapter):
    """
    Adapter for Slack communication.

    Uses the Slack Web API and Socket Mode for real-time messaging.

    Example:
        adapter = SlackAdapter(
            token="xoxb-...",
            app_token="xapp-...",
            default_channel="#general"
        )
        await adapter.connect()
        await adapter.send_message(Message(content="Hello, Slack!"))
    """

    def __init__(
        self,
        token: str,
        app_token: Optional[str] = None,
        default_channel: Optional[str] = None,
        channel_name: str = "slack"
    ):
        """
        Initialize the Slack adapter.

        Args:
            token: Slack bot token (xoxb-...)
            app_token: Slack app token for Socket Mode (xapp-...)
            default_channel: Default channel to send messages to
            channel_name: Name identifier for this channel
        """
        super().__init__(channel_name)
        self.token = token
        self.app_token = app_token
        self.default_channel = default_channel
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._client = None
        self._socket_client = None

    async def connect(self) -> None:
        """Connect to Slack."""
        try:
            # In a real implementation, we would initialize the Slack client here
            # from slack_sdk.web.async_client import AsyncWebClient
            # from slack_sdk.socket_mode.aiohttp import SocketModeClient
            # self._client = AsyncWebClient(token=self.token)
            # self._socket_client = SocketModeClient(app_token=self.app_token, web_client=self._client)

            self._connected = True
            logger.info(f"Connected to Slack channel: {self.channel_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Slack: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        if self._socket_client:
            # await self._socket_client.close()
            pass
        self._connected = False
        logger.info(f"Disconnected from Slack channel: {self.channel_name}")

    async def receive_message(self) -> Message:
        """Receive a message from Slack."""
        if not self._connected:
            raise ConnectionError("Not connected to Slack")
        return await self._message_queue.get()

    async def send_message(self, message: Message) -> None:
        """Send a message to Slack."""
        if not self._connected:
            raise ConnectionError("Not connected to Slack")

        channel = message.metadata.get("channel") or self.default_channel
        if not channel:
            raise ValueError("No channel specified and no default channel set")

        # In a real implementation:
        # await self._client.chat_postMessage(
        #     channel=channel,
        #     text=message.content,
        #     thread_ts=message.metadata.get("thread_ts")
        # )

        logger.info(f"Sent message to Slack channel {channel}: {message.content[:50]}...")


class DiscordAdapter(ChannelAdapter):
    """
    Adapter for Discord communication.

    Uses the Discord API for messaging.

    Example:
        adapter = DiscordAdapter(
            token="...",
            default_guild_id="123456789",
            default_channel_id="987654321"
        )
        await adapter.connect()
        await adapter.send_message(Message(content="Hello, Discord!"))
    """

    def __init__(
        self,
        token: str,
        default_guild_id: Optional[str] = None,
        default_channel_id: Optional[str] = None,
        channel_name: str = "discord"
    ):
        """
        Initialize the Discord adapter.

        Args:
            token: Discord bot token
            default_guild_id: Default guild/server ID
            default_channel_id: Default channel ID
            channel_name: Name identifier for this channel
        """
        super().__init__(channel_name)
        self.token = token
        self.default_guild_id = default_guild_id
        self.default_channel_id = default_channel_id
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._client = None

    async def connect(self) -> None:
        """Connect to Discord."""
        try:
            # In a real implementation:
            # import discord
            # self._client = discord.Client()
            # await self._client.start(self.token)

            self._connected = True
            logger.info(f"Connected to Discord channel: {self.channel_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Discord: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        if self._client:
            # await self._client.close()
            pass
        self._connected = False
        logger.info(f"Disconnected from Discord channel: {self.channel_name}")

    async def receive_message(self) -> Message:
        """Receive a message from Discord."""
        if not self._connected:
            raise ConnectionError("Not connected to Discord")
        return await self._message_queue.get()

    async def send_message(self, message: Message) -> None:
        """Send a message to Discord."""
        if not self._connected:
            raise ConnectionError("Not connected to Discord")

        channel_id = message.metadata.get("channel_id") or self.default_channel_id
        if not channel_id:
            raise ValueError("No channel_id specified and no default channel set")

        # In a real implementation:
        # channel = self._client.get_channel(int(channel_id))
        # await channel.send(message.content)

        logger.info(f"Sent message to Discord channel {channel_id}: {message.content[:50]}...")


class TelegramAdapter(ChannelAdapter):
    """
    Adapter for Telegram communication.

    Uses the Telegram Bot API.

    Example:
        adapter = TelegramAdapter(
            token="123456789:ABC...",
            default_chat_id="987654321"
        )
        await adapter.connect()
        await adapter.send_message(Message(content="Hello, Telegram!"))
    """

    def __init__(
        self,
        token: str,
        default_chat_id: Optional[str] = None,
        channel_name: str = "telegram"
    ):
        """
        Initialize the Telegram adapter.

        Args:
            token: Telegram bot token
            default_chat_id: Default chat ID
            channel_name: Name identifier for this channel
        """
        super().__init__(channel_name)
        self.token = token
        self.default_chat_id = default_chat_id
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._bot = None
        self._polling_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to Telegram."""
        try:
            # In a real implementation:
            # from telegram import Bot
            # from telegram.ext import Application
            # self._bot = Bot(token=self.token)
            # Start polling for updates

            self._connected = True
            logger.info(f"Connected to Telegram channel: {self.channel_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Telegram: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Telegram."""
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        self._connected = False
        logger.info(f"Disconnected from Telegram channel: {self.channel_name}")

    async def receive_message(self) -> Message:
        """Receive a message from Telegram."""
        if not self._connected:
            raise ConnectionError("Not connected to Telegram")
        return await self._message_queue.get()

    async def send_message(self, message: Message) -> None:
        """Send a message to Telegram."""
        if not self._connected:
            raise ConnectionError("Not connected to Telegram")

        chat_id = message.metadata.get("chat_id") or self.default_chat_id
        if not chat_id:
            raise ValueError("No chat_id specified and no default chat set")

        # In a real implementation:
        # await self._bot.send_message(chat_id=chat_id, text=message.content)

        logger.info(f"Sent message to Telegram chat {chat_id}: {message.content[:50]}...")


class WebhookAdapter(ChannelAdapter):
    """
    Adapter for webhook-based communication.

    Supports both sending webhooks and receiving via an HTTP server.

    Example:
        adapter = WebhookAdapter(
            outgoing_url="https://example.com/webhook",
            listen_port=8080
        )
        await adapter.connect()
        await adapter.send_message(Message(content="Webhook payload"))
    """

    def __init__(
        self,
        outgoing_url: Optional[str] = None,
        listen_port: int = 8080,
        listen_host: str = "0.0.0.0",
        secret: Optional[str] = None,
        channel_name: str = "webhook"
    ):
        """
        Initialize the webhook adapter.

        Args:
            outgoing_url: URL to send webhooks to
            listen_port: Port to listen for incoming webhooks
            listen_host: Host to bind the listener to
            secret: Secret for webhook verification
            channel_name: Name identifier for this channel
        """
        super().__init__(channel_name)
        self.outgoing_url = outgoing_url
        self.listen_port = listen_port
        self.listen_host = listen_host
        self.secret = secret
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._server = None
        self._server_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Start the webhook server."""
        try:
            # In a real implementation, we'd start an aiohttp server:
            # from aiohttp import web
            # app = web.Application()
            # app.router.add_post("/webhook", self._handle_webhook)
            # runner = web.AppRunner(app)
            # await runner.setup()
            # self._server = web.TCPSite(runner, self.listen_host, self.listen_port)
            # await self._server.start()

            self._connected = True
            logger.info(f"Webhook adapter listening on {self.listen_host}:{self.listen_port}")
        except Exception as e:
            raise ConnectionError(f"Failed to start webhook server: {e}")

    async def disconnect(self) -> None:
        """Stop the webhook server."""
        if self._server:
            # await self._server.stop()
            pass
        self._connected = False
        logger.info(f"Webhook adapter stopped")

    async def receive_message(self) -> Message:
        """Receive a webhook message."""
        if not self._connected:
            raise ConnectionError("Webhook server not running")
        return await self._message_queue.get()

    async def send_message(self, message: Message) -> None:
        """Send a webhook."""
        if not self.outgoing_url:
            raise ValueError("No outgoing URL configured")

        # In a real implementation:
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     payload = message.to_dict()
        #     headers = {"Content-Type": "application/json"}
        #     if self.secret:
        #         headers["X-Webhook-Secret"] = self.secret
        #     async with session.post(self.outgoing_url, json=payload, headers=headers) as resp:
        #         resp.raise_for_status()

        logger.info(f"Sent webhook to {self.outgoing_url}: {message.content[:50]}...")


class WebSocketAdapter(ChannelAdapter):
    """
    Adapter for WebSocket-based communication.

    Supports both client and server modes.

    Example:
        # Client mode
        adapter = WebSocketAdapter(url="wss://example.com/ws")
        await adapter.connect()

        # Server mode
        adapter = WebSocketAdapter(listen_port=8765)
        await adapter.connect()
    """

    def __init__(
        self,
        url: Optional[str] = None,
        listen_port: Optional[int] = None,
        listen_host: str = "0.0.0.0",
        channel_name: str = "websocket"
    ):
        """
        Initialize the WebSocket adapter.

        Args:
            url: URL to connect to (client mode)
            listen_port: Port to listen on (server mode)
            listen_host: Host to bind to (server mode)
            channel_name: Name identifier for this channel
        """
        super().__init__(channel_name)
        self.url = url
        self.listen_port = listen_port
        self.listen_host = listen_host
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._websocket = None
        self._server = None
        self._clients: Dict[str, Any] = {}
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def is_server_mode(self) -> bool:
        """Check if running in server mode."""
        return self.listen_port is not None

    async def connect(self) -> None:
        """Connect/start the WebSocket."""
        try:
            if self.is_server_mode:
                await self._start_server()
            else:
                await self._connect_client()
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect WebSocket: {e}")

    async def _start_server(self) -> None:
        """Start WebSocket server."""
        # In a real implementation:
        # import websockets
        # self._server = await websockets.serve(
        #     self._handle_connection,
        #     self.listen_host,
        #     self.listen_port
        # )
        logger.info(f"WebSocket server listening on {self.listen_host}:{self.listen_port}")

    async def _connect_client(self) -> None:
        """Connect as WebSocket client."""
        if not self.url:
            raise ValueError("No WebSocket URL configured")
        # In a real implementation:
        # import websockets
        # self._websocket = await websockets.connect(self.url)
        # self._receive_task = asyncio.create_task(self._receive_loop())
        logger.info(f"Connected to WebSocket: {self.url}")

    async def disconnect(self) -> None:
        """Disconnect/stop the WebSocket."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            # await self._websocket.close()
            pass

        if self._server:
            # self._server.close()
            # await self._server.wait_closed()
            pass

        self._connected = False
        logger.info("WebSocket disconnected")

    async def receive_message(self) -> Message:
        """Receive a WebSocket message."""
        if not self._connected:
            raise ConnectionError("WebSocket not connected")
        return await self._message_queue.get()

    async def send_message(self, message: Message) -> None:
        """Send a WebSocket message."""
        if not self._connected:
            raise ConnectionError("WebSocket not connected")

        payload = json.dumps(message.to_dict())

        if self.is_server_mode:
            # Broadcast to all connected clients
            client_id = message.metadata.get("client_id")
            if client_id and client_id in self._clients:
                # Send to specific client
                # await self._clients[client_id].send(payload)
                pass
            else:
                # Broadcast
                # for client in self._clients.values():
                #     await client.send(payload)
                pass
        else:
            # Send via client connection
            # await self._websocket.send(payload)
            pass

        logger.info(f"Sent WebSocket message: {message.content[:50]}...")


class MultiChannelManager:
    """
    Manages multiple channel adapters.

    Provides unified interface for receiving from any channel,
    broadcasting to all channels, and managing channel lifecycle.

    Example:
        manager = MultiChannelManager()
        await manager.add_channel("slack", SlackAdapter(token="..."))
        await manager.add_channel("discord", DiscordAdapter(token="..."))

        # Receive from any channel
        channel_name, message = await manager.receive_any()

        # Broadcast to all
        await manager.broadcast(Message(content="Hello, everyone!"))
    """

    def __init__(self):
        """Initialize the manager."""
        self.channels: Dict[str, ChannelAdapter] = {}
        self._receive_queues: Dict[str, asyncio.Queue[Message]] = {}
        self._unified_queue: asyncio.Queue[Tuple[str, Message]] = asyncio.Queue()
        self._receive_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def add_channel(
        self,
        name: str,
        adapter: ChannelAdapter,
        auto_connect: bool = True
    ) -> None:
        """
        Add a channel adapter.

        Args:
            name: Name for this channel
            adapter: The channel adapter
            auto_connect: Whether to connect automatically
        """
        async with self._lock:
            if name in self.channels:
                raise ValueError(f"Channel {name} already exists")

            self.channels[name] = adapter
            self._receive_queues[name] = asyncio.Queue()

            if auto_connect:
                await adapter.connect()

            # Start receive task
            self._receive_tasks[name] = asyncio.create_task(
                self._channel_receive_loop(name, adapter)
            )

            logger.info(f"Added channel: {name}")

    async def remove_channel(self, name: str) -> None:
        """
        Remove a channel adapter.

        Args:
            name: Name of the channel to remove
        """
        async with self._lock:
            if name not in self.channels:
                return

            # Cancel receive task
            if name in self._receive_tasks:
                self._receive_tasks[name].cancel()
                try:
                    await self._receive_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self._receive_tasks[name]

            # Disconnect adapter
            adapter = self.channels[name]
            if adapter.is_connected:
                await adapter.disconnect()

            del self.channels[name]
            del self._receive_queues[name]

            logger.info(f"Removed channel: {name}")

    async def get_channel(self, name: str) -> Optional[ChannelAdapter]:
        """
        Get a channel adapter by name.

        Args:
            name: Name of the channel

        Returns:
            The channel adapter or None
        """
        return self.channels.get(name)

    async def broadcast(
        self,
        message: Message,
        exclude: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast a message to all connected channels.

        Args:
            message: The message to broadcast
            exclude: List of channel names to exclude

        Returns:
            Dict mapping channel names to success status
        """
        exclude = exclude or []
        results = {}

        for name, adapter in self.channels.items():
            if name in exclude:
                continue

            if not adapter.is_connected:
                results[name] = False
                continue

            try:
                await adapter.send_message(message)
                results[name] = True
            except Exception as e:
                logger.error(f"Failed to broadcast to {name}: {e}")
                results[name] = False

        return results

    async def send_to(self, channel_name: str, message: Message) -> None:
        """
        Send a message to a specific channel.

        Args:
            channel_name: Name of the channel
            message: The message to send

        Raises:
            ValueError: If channel doesn't exist
            ConnectionError: If channel is not connected
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel {channel_name} not found")

        adapter = self.channels[channel_name]
        await adapter.send_message(message)

    async def receive_any(self, timeout: Optional[float] = None) -> Tuple[str, Message]:
        """
        Receive a message from any channel.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Tuple of (channel_name, message)

        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        if timeout:
            return await asyncio.wait_for(
                self._unified_queue.get(),
                timeout=timeout
            )
        return await self._unified_queue.get()

    async def receive_from(
        self,
        channel_name: str,
        timeout: Optional[float] = None
    ) -> Message:
        """
        Receive a message from a specific channel.

        Args:
            channel_name: Name of the channel
            timeout: Optional timeout in seconds

        Returns:
            The received message

        Raises:
            ValueError: If channel doesn't exist
            asyncio.TimeoutError: If timeout is reached
        """
        if channel_name not in self._receive_queues:
            raise ValueError(f"Channel {channel_name} not found")

        queue = self._receive_queues[channel_name]
        if timeout:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        return await queue.get()

    async def _channel_receive_loop(
        self,
        name: str,
        adapter: ChannelAdapter
    ) -> None:
        """
        Background loop to receive messages from a channel.

        Args:
            name: Channel name
            adapter: The channel adapter
        """
        while True:
            try:
                message = await adapter.receive_message()
                message.channel = name

                # Put in channel-specific queue
                await self._receive_queues[name].put(message)

                # Put in unified queue
                await self._unified_queue.put((name, message))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving from channel {name}: {e}")
                await asyncio.sleep(1)

    def get_channel_names(self) -> List[str]:
        """Get list of all channel names."""
        return list(self.channels.keys())

    def get_connected_channels(self) -> List[str]:
        """Get list of connected channel names."""
        return [
            name for name, adapter in self.channels.items()
            if adapter.is_connected
        ]

    async def disconnect_all(self) -> None:
        """Disconnect all channels."""
        for name in list(self.channels.keys()):
            await self.remove_channel(name)

    async def __aenter__(self) -> "MultiChannelManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect_all()
