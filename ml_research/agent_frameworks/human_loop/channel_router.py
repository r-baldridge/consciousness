"""Channel routing for human-in-the-loop communications.

This module provides a flexible channel abstraction for routing
approval requests and human queries to various communication platforms
including Slack, email, Discord, webhooks, and console.

Each channel implements the send/receive pattern for bidirectional
communication with humans.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import logging
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .approval import ApprovalRequest, ApprovalResult

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Types of communication channels."""
    SLACK = "slack"
    EMAIL = "email"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    CUSTOM = "custom"


class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ChannelMessage:
    """A message to be sent through a channel.

    Attributes:
        content: The message content
        metadata: Additional metadata (request_id, action, etc.)
        priority: Message priority
        thread_id: Optional thread/conversation ID
        attachments: Optional list of attachments
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    thread_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "thread_id": self.thread_id,
            "attachments": self.attachments,
        }


@dataclass
class ChannelResponse:
    """A response received from a channel.

    Attributes:
        content: The response content
        responder: Identifier of the responder
        timestamp: When the response was received
        metadata: Additional metadata
    """
    content: str
    responder: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChannelError(Exception):
    """Base exception for channel errors."""
    pass


class ChannelSendError(ChannelError):
    """Error sending through a channel."""
    pass


class ChannelReceiveError(ChannelError):
    """Error receiving from a channel."""
    pass


class ChannelTimeoutError(ChannelError):
    """Timeout waiting for channel response."""
    pass


class Channel(ABC):
    """Abstract base class for communication channels.

    Channels provide bidirectional communication with humans for
    approval requests and queries.
    """

    @property
    @abstractmethod
    def channel_type(self) -> ChannelType:
        """Return the type of this channel."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the channel is connected and ready."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the channel."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the channel."""
        ...

    @abstractmethod
    async def send(self, message: ChannelMessage) -> str:
        """Send a message through the channel.

        Args:
            message: The message to send

        Returns:
            Message ID or thread ID for tracking

        Raises:
            ChannelSendError: If sending fails
        """
        ...

    @abstractmethod
    async def receive(self, timeout: int = 300) -> Optional[ChannelResponse]:
        """Receive a response from the channel.

        Args:
            timeout: Timeout in seconds

        Returns:
            The response, or None if timeout

        Raises:
            ChannelReceiveError: If receiving fails
            ChannelTimeoutError: If timeout exceeded
        """
        ...

    async def send_and_wait(
        self,
        message: ChannelMessage,
        timeout: int = 300
    ) -> ChannelResponse:
        """Send a message and wait for a response.

        Args:
            message: The message to send
            timeout: Timeout in seconds

        Returns:
            The response

        Raises:
            ChannelTimeoutError: If no response within timeout
        """
        message_id = await self.send(message)
        message.metadata["message_id"] = message_id

        response = await self.receive(timeout)
        if response is None:
            raise ChannelTimeoutError(f"No response within {timeout} seconds")

        return response

    def format_approval_request(self, request: ApprovalRequest) -> ChannelMessage:
        """Format an approval request as a channel message.

        Args:
            request: The approval request

        Returns:
            Formatted channel message
        """
        content = f"""
**Approval Request**

**Action:** {request.action}
**Description:** {request.description}

**Arguments:**
```json
{json.dumps(request.arguments, indent=2)}
```

**Requester:** {request.requester}
**Request ID:** {request.id}
**Timeout:** {request.timeout.total_seconds()} seconds

Reply with `approve {request.id}` or `deny {request.id} <reason>`
"""
        return ChannelMessage(
            content=content.strip(),
            metadata={
                "request_id": request.id,
                "action": request.action,
                "type": "approval_request"
            },
            priority=MessagePriority.HIGH
        )


class ConsoleChannel(Channel):
    """Console-based channel for testing and CLI usage.

    This channel uses stdin/stdout for communication, making it
    useful for testing and command-line interfaces.
    """

    def __init__(self, prompt: str = "> "):
        """Initialize the console channel.

        Args:
            prompt: Input prompt string
        """
        self._prompt = prompt
        self._connected = False
        self._response_queue: asyncio.Queue[ChannelResponse] = asyncio.Queue()

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.CONSOLE

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True
        logger.info("Console channel connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("Console channel disconnected")

    async def send(self, message: ChannelMessage) -> str:
        """Print message to console."""
        if not self._connected:
            raise ChannelSendError("Console channel not connected")

        print("\n" + "=" * 60)
        print(message.content)
        print("=" * 60)

        message_id = message.metadata.get("request_id", "console_msg")
        return message_id

    async def receive(self, timeout: int = 300) -> Optional[ChannelResponse]:
        """Read input from console."""
        if not self._connected:
            raise ChannelReceiveError("Console channel not connected")

        try:
            loop = asyncio.get_event_loop()
            response_text = await asyncio.wait_for(
                loop.run_in_executor(None, input, self._prompt),
                timeout=timeout
            )

            return ChannelResponse(
                content=response_text.strip(),
                responder="console_user"
            )
        except asyncio.TimeoutError:
            return None


class SlackChannel(Channel):
    """Slack channel for approval requests and queries.

    Uses the Slack Web API for sending messages and receiving
    responses via webhooks or polling.
    """

    def __init__(
        self,
        token: str,
        channel_id: str,
        webhook_url: Optional[str] = None,
        poll_interval: float = 2.0
    ):
        """Initialize the Slack channel.

        Args:
            token: Slack bot token
            channel_id: Default channel ID to send messages to
            webhook_url: Optional incoming webhook URL
            poll_interval: Interval for polling responses (seconds)
        """
        self._token = token
        self._channel_id = channel_id
        self._webhook_url = webhook_url
        self._poll_interval = poll_interval
        self._connected = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._pending_messages: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, ChannelResponse] = {}

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.SLACK

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """Connect to Slack API."""
        self._session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json"
        })

        # Test connection
        try:
            async with self._session.post(
                "https://slack.com/api/auth.test"
            ) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise ChannelError(f"Slack auth failed: {data.get('error')}")

            self._connected = True
            logger.info("Slack channel connected")
        except Exception as e:
            await self._session.close()
            self._session = None
            raise ChannelError(f"Failed to connect to Slack: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Slack API."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Slack channel disconnected")

    async def send(self, message: ChannelMessage) -> str:
        """Send a message to Slack."""
        if not self.is_connected:
            raise ChannelSendError("Slack channel not connected")

        # Build Slack message blocks
        blocks = self._build_blocks(message)

        payload = {
            "channel": self._channel_id,
            "text": message.content,
            "blocks": blocks
        }

        if message.thread_id:
            payload["thread_ts"] = message.thread_id

        try:
            async with self._session.post(
                "https://slack.com/api/chat.postMessage",
                json=payload
            ) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise ChannelSendError(f"Slack send failed: {data.get('error')}")

                message_ts = data.get("ts")
                logger.info(f"Sent Slack message: {message_ts}")
                return message_ts

        except aiohttp.ClientError as e:
            raise ChannelSendError(f"Slack API error: {e}")

    def _build_blocks(self, message: ChannelMessage) -> List[Dict[str, Any]]:
        """Build Slack Block Kit blocks from message."""
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message.content
                }
            }
        ]

        # Add action buttons for approval requests
        if message.metadata.get("type") == "approval_request":
            request_id = message.metadata.get("request_id")
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "value": f"approve_{request_id}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "value": f"deny_{request_id}"
                    }
                ]
            })

        return blocks

    async def receive(self, timeout: int = 300) -> Optional[ChannelResponse]:
        """Receive a response from Slack.

        This implementation uses polling. For production, consider
        using Slack's Events API or Socket Mode.
        """
        if not self.is_connected:
            raise ChannelReceiveError("Slack channel not connected")

        start_time = asyncio.get_event_loop().time()
        last_ts = "0"

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return None

            try:
                # Poll for new messages
                async with self._session.get(
                    "https://slack.com/api/conversations.history",
                    params={
                        "channel": self._channel_id,
                        "oldest": last_ts,
                        "limit": 10
                    }
                ) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        messages = data.get("messages", [])
                        for msg in reversed(messages):
                            # Skip bot messages
                            if msg.get("bot_id"):
                                continue

                            # Check for approval commands
                            text = msg.get("text", "")
                            if text.startswith("approve ") or text.startswith("deny "):
                                return ChannelResponse(
                                    content=text,
                                    responder=msg.get("user", "unknown"),
                                    metadata={"ts": msg.get("ts")}
                                )

                            last_ts = msg.get("ts", last_ts)

            except aiohttp.ClientError as e:
                logger.error(f"Error polling Slack: {e}")

            await asyncio.sleep(self._poll_interval)


class EmailChannel(Channel):
    """Email channel for approval requests.

    Uses SMTP for sending and IMAP for receiving responses.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_address: str,
        to_addresses: List[str],
        use_tls: bool = True
    ):
        """Initialize the email channel.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_address: Sender email address
            to_addresses: List of recipient addresses
            use_tls: Whether to use TLS
        """
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._from_address = from_address
        self._to_addresses = to_addresses
        self._use_tls = use_tls
        self._connected = False

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.EMAIL

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Verify SMTP connection."""
        try:
            # Test SMTP connection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._test_smtp)
            self._connected = True
            logger.info("Email channel connected")
        except Exception as e:
            raise ChannelError(f"Failed to connect to email server: {e}")

    def _test_smtp(self) -> None:
        """Test SMTP connection synchronously."""
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            if self._use_tls:
                server.starttls()
            server.login(self._username, self._password)

    async def disconnect(self) -> None:
        """Disconnect from email server."""
        self._connected = False
        logger.info("Email channel disconnected")

    async def send(self, message: ChannelMessage) -> str:
        """Send an email."""
        if not self._connected:
            raise ChannelSendError("Email channel not connected")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = self._get_subject(message)
        msg["From"] = self._from_address
        msg["To"] = ", ".join(self._to_addresses)

        # Add plain text and HTML versions
        text_part = MIMEText(message.content, "plain")
        html_part = MIMEText(self._to_html(message.content), "html")

        msg.attach(text_part)
        msg.attach(html_part)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email,
                msg
            )

            message_id = msg["Message-ID"] or message.metadata.get("request_id", "email")
            logger.info(f"Sent email: {message_id}")
            return message_id

        except Exception as e:
            raise ChannelSendError(f"Failed to send email: {e}")

    def _send_email(self, msg: MIMEMultipart) -> None:
        """Send email synchronously."""
        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            if self._use_tls:
                server.starttls()
            server.login(self._username, self._password)
            server.send_message(msg)

    def _get_subject(self, message: ChannelMessage) -> str:
        """Generate email subject from message."""
        if message.metadata.get("type") == "approval_request":
            action = message.metadata.get("action", "Action")
            return f"[APPROVAL REQUIRED] {action}"
        return "Agent Notification"

    def _to_html(self, content: str) -> str:
        """Convert markdown-like content to HTML."""
        import re

        html = content
        # Bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        # Code blocks
        html = re.sub(r"```(\w+)?\n(.+?)```", r"<pre><code>\2</code></pre>", html, flags=re.DOTALL)
        # Inline code
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
        # Line breaks
        html = html.replace("\n", "<br>")

        return f"<html><body>{html}</body></html>"

    async def receive(self, timeout: int = 300) -> Optional[ChannelResponse]:
        """Receive an email response.

        Note: Email receiving requires IMAP configuration which is
        not implemented in this basic version. For production use,
        integrate with an email service that supports webhooks.
        """
        logger.warning("Email receive not fully implemented - use webhooks")
        # In production, this would poll IMAP or use webhooks
        await asyncio.sleep(timeout)
        return None


class WebhookChannel(Channel):
    """Webhook channel for custom integrations.

    Sends messages via HTTP POST and receives responses via
    a callback endpoint.
    """

    def __init__(
        self,
        send_url: str,
        receive_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True
    ):
        """Initialize the webhook channel.

        Args:
            send_url: URL to POST messages to
            receive_url: Optional URL to poll for responses
            headers: Optional headers to include in requests
            verify_ssl: Whether to verify SSL certificates
        """
        self._send_url = send_url
        self._receive_url = receive_url
        self._headers = headers or {}
        self._verify_ssl = verify_ssl
        self._connected = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._response_queue: asyncio.Queue[ChannelResponse] = asyncio.Queue()

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WEBHOOK

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """Initialize HTTP session."""
        connector = aiohttp.TCPConnector(ssl=self._verify_ssl)
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            connector=connector
        )
        self._connected = True
        logger.info("Webhook channel connected")

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Webhook channel disconnected")

    async def send(self, message: ChannelMessage) -> str:
        """Send a message via webhook."""
        if not self.is_connected:
            raise ChannelSendError("Webhook channel not connected")

        payload = message.to_dict()

        try:
            async with self._session.post(
                self._send_url,
                json=payload
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise ChannelSendError(
                        f"Webhook returned {resp.status}: {text}"
                    )

                data = await resp.json()
                message_id = data.get("id", message.metadata.get("request_id", "webhook"))
                logger.info(f"Sent webhook message: {message_id}")
                return message_id

        except aiohttp.ClientError as e:
            raise ChannelSendError(f"Webhook error: {e}")

    async def receive(self, timeout: int = 300) -> Optional[ChannelResponse]:
        """Receive a response from the webhook callback.

        If a receive_url is configured, polls that URL.
        Otherwise, waits for responses to be pushed to the queue.
        """
        if not self.is_connected:
            raise ChannelReceiveError("Webhook channel not connected")

        try:
            response = await asyncio.wait_for(
                self._response_queue.get(),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            return None

    async def push_response(self, response: ChannelResponse) -> None:
        """Push a response to the queue (for callback handling)."""
        await self._response_queue.put(response)


class DiscordChannel(Channel):
    """Discord channel for approval requests.

    Uses Discord webhooks for sending and bot API for receiving.
    """

    def __init__(
        self,
        webhook_url: str,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None
    ):
        """Initialize the Discord channel.

        Args:
            webhook_url: Discord webhook URL for sending
            bot_token: Optional bot token for receiving
            channel_id: Optional channel ID for receiving
        """
        self._webhook_url = webhook_url
        self._bot_token = bot_token
        self._channel_id = channel_id
        self._connected = False
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.DISCORD

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> None:
        """Initialize Discord connection."""
        self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("Discord channel connected")

    async def disconnect(self) -> None:
        """Close Discord connection."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Discord channel disconnected")

    async def send(self, message: ChannelMessage) -> str:
        """Send a message via Discord webhook."""
        if not self.is_connected:
            raise ChannelSendError("Discord channel not connected")

        # Build Discord embed
        embed = {
            "title": "Agent Notification",
            "description": message.content,
            "color": self._get_color(message.priority)
        }

        if message.metadata.get("type") == "approval_request":
            embed["title"] = "Approval Required"
            embed["fields"] = [
                {"name": "Request ID", "value": message.metadata.get("request_id", ""), "inline": True},
                {"name": "Action", "value": message.metadata.get("action", ""), "inline": True}
            ]

        payload = {
            "embeds": [embed]
        }

        try:
            async with self._session.post(
                self._webhook_url,
                json=payload
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise ChannelSendError(
                        f"Discord webhook returned {resp.status}: {text}"
                    )

                message_id = message.metadata.get("request_id", "discord")
                logger.info(f"Sent Discord message: {message_id}")
                return message_id

        except aiohttp.ClientError as e:
            raise ChannelSendError(f"Discord error: {e}")

    def _get_color(self, priority: MessagePriority) -> int:
        """Get embed color based on priority."""
        colors = {
            MessagePriority.LOW: 0x808080,      # Gray
            MessagePriority.NORMAL: 0x0099FF,   # Blue
            MessagePriority.HIGH: 0xFF9900,     # Orange
            MessagePriority.URGENT: 0xFF0000,   # Red
        }
        return colors.get(priority, 0x0099FF)

    async def receive(self, timeout: int = 300) -> Optional[ChannelResponse]:
        """Receive a response from Discord.

        Note: Full Discord bot integration required for receiving.
        """
        logger.warning("Discord receive requires bot integration")
        await asyncio.sleep(timeout)
        return None


class ChannelRouter:
    """Routes messages to appropriate channels based on configuration.

    The ChannelRouter manages multiple channels and routes approval
    requests and queries to the appropriate channel based on
    configuration and request properties.
    """

    def __init__(self, default_channel: str = "console"):
        """Initialize the channel router.

        Args:
            default_channel: Name of the default channel
        """
        self._channels: Dict[str, Channel] = {}
        self._default_channel = default_channel
        self._routing_rules: List[Callable[[ApprovalRequest], Optional[str]]] = []

    def register_channel(self, name: str, channel: Channel) -> None:
        """Register a channel with the router.

        Args:
            name: Name to register the channel under
            channel: The channel instance
        """
        self._channels[name] = channel
        logger.info(f"Registered channel: {name} ({channel.channel_type.value})")

    def unregister_channel(self, name: str) -> None:
        """Unregister a channel.

        Args:
            name: Name of the channel to unregister
        """
        if name in self._channels:
            del self._channels[name]
            logger.info(f"Unregistered channel: {name}")

    def get_channel(self, name: str) -> Optional[Channel]:
        """Get a channel by name.

        Args:
            name: Name of the channel

        Returns:
            The channel, or None if not found
        """
        return self._channels.get(name)

    def add_routing_rule(
        self,
        rule: Callable[[ApprovalRequest], Optional[str]]
    ) -> None:
        """Add a routing rule.

        Rules are functions that take an ApprovalRequest and return
        a channel name or None. Rules are evaluated in order until
        one returns a non-None value.

        Args:
            rule: Routing rule function
        """
        self._routing_rules.append(rule)

    def determine_channel(self, request: ApprovalRequest) -> str:
        """Determine which channel to use for a request.

        Args:
            request: The approval request

        Returns:
            Name of the channel to use
        """
        # Check explicit channel on request
        if request.channel and request.channel in self._channels:
            return request.channel

        # Apply routing rules
        for rule in self._routing_rules:
            channel_name = rule(request)
            if channel_name and channel_name in self._channels:
                return channel_name

        # Fall back to default
        return self._default_channel

    async def route(
        self,
        request: ApprovalRequest,
        channel_name: Optional[str] = None
    ) -> ApprovalResult:
        """Route an approval request to the appropriate channel.

        Args:
            request: The approval request
            channel_name: Optional channel override

        Returns:
            The approval result
        """
        name = channel_name or self.determine_channel(request)
        channel = self._channels.get(name)

        if not channel:
            raise ChannelError(f"Channel not found: {name}")

        if not channel.is_connected:
            await channel.connect()

        # Format and send request
        message = channel.format_approval_request(request)

        try:
            await channel.send(message)

            # Wait for response
            response = await channel.receive(
                timeout=int(request.timeout.total_seconds())
            )

            if response is None:
                return ApprovalResult.TIMEOUT

            # Parse response
            return self._parse_approval_response(response, request)

        except ChannelTimeoutError:
            return ApprovalResult.TIMEOUT

    def _parse_approval_response(
        self,
        response: ChannelResponse,
        request: ApprovalRequest
    ) -> ApprovalResult:
        """Parse a channel response into an approval result."""
        content = response.content.lower().strip()

        if content.startswith("approve"):
            return ApprovalResult.APPROVED
        elif content.startswith("deny"):
            return ApprovalResult.DENIED
        else:
            # Treat unrecognized responses as pending/timeout
            logger.warning(f"Unrecognized response: {content}")
            return ApprovalResult.TIMEOUT

    async def send_approval_request(self, request: ApprovalRequest) -> str:
        """Send an approval request to its designated channel.

        Args:
            request: The approval request

        Returns:
            Message ID
        """
        channel_name = self.determine_channel(request)
        channel = self._channels.get(channel_name)

        if not channel:
            raise ChannelError(f"Channel not found: {channel_name}")

        if not channel.is_connected:
            await channel.connect()

        message = channel.format_approval_request(request)
        return await channel.send(message)

    async def connect_all(self) -> None:
        """Connect all registered channels."""
        for name, channel in self._channels.items():
            try:
                if not channel.is_connected:
                    await channel.connect()
            except Exception as e:
                logger.error(f"Failed to connect channel {name}: {e}")

    async def disconnect_all(self) -> None:
        """Disconnect all registered channels."""
        for name, channel in self._channels.items():
            try:
                if channel.is_connected:
                    await channel.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect channel {name}: {e}")
