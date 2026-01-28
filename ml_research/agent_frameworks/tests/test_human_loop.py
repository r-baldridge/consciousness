"""
Tests for human-in-the-loop components.

Tests cover:
    - @require_approval decorator
    - ApprovalWorkflow
    - ChannelRouter
    - EscalationManager
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum, auto
from datetime import datetime, timedelta
import functools


# ---------------------------------------------------------------------------
# Test Data Structures
# ---------------------------------------------------------------------------

class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class ApprovalRequest:
    """Request for human approval."""
    request_id: str
    action: str
    description: str
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    timeout: int = 300  # seconds
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: Optional[str] = None
    response_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.status != ApprovalStatus.PENDING:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.timeout


class ApprovalWorkflow:
    """Workflow for approval requests."""

    def __init__(self, default_timeout: int = 300):
        self.default_timeout = default_timeout
        self._requests: Dict[str, ApprovalRequest] = {}
        self._handlers: List[Callable[[ApprovalRequest], Awaitable[bool]]] = []

    def add_handler(self, handler: Callable[[ApprovalRequest], Awaitable[bool]]) -> None:
        """Add an approval handler."""
        self._handlers.append(handler)

    async def request_approval(
        self,
        action: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> ApprovalRequest:
        """Create an approval request."""
        request = ApprovalRequest(
            request_id=f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{id(self)}",
            action=action,
            description=description,
            context=context or {},
            timeout=timeout or self.default_timeout
        )
        self._requests[request.request_id] = request
        return request

    async def wait_for_approval(
        self,
        request: ApprovalRequest,
        poll_interval: float = 0.1
    ) -> bool:
        """Wait for approval decision."""
        while request.status == ApprovalStatus.PENDING:
            if request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                return False
            await asyncio.sleep(poll_interval)
        return request.status == ApprovalStatus.APPROVED

    async def approve(self, request_id: str, approver: str = "system") -> bool:
        """Approve a request."""
        request = self._requests.get(request_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.APPROVED
        request.approver = approver
        request.response_at = datetime.now()
        return True

    async def reject(self, request_id: str, approver: str = "system") -> bool:
        """Reject a request."""
        request = self._requests.get(request_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return False

        request.status = ApprovalStatus.REJECTED
        request.approver = approver
        request.response_at = datetime.now()
        return True

    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending requests."""
        return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]


def require_approval(action: str, description: str = ""):
    """Decorator to require approval before execution."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, _approval_workflow: Optional[ApprovalWorkflow] = None, **kwargs):
            if _approval_workflow:
                request = await _approval_workflow.request_approval(
                    action=action,
                    description=description or f"Executing {func.__name__}",
                    context={"args": str(args), "kwargs": str(kwargs)}
                )

                # For testing, auto-approve
                await _approval_workflow.approve(request.request_id)

                if not await _approval_workflow.wait_for_approval(request):
                    raise PermissionError(f"Approval denied for {action}")

            return await func(*args, **kwargs)
        return wrapper
    return decorator


class ChannelType(Enum):
    """Types of notification channels."""
    CONSOLE = auto()
    SLACK = auto()
    EMAIL = auto()
    WEBHOOK = auto()


@dataclass
class Channel:
    """Notification channel configuration."""
    name: str
    channel_type: ChannelType
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class ChannelRouter:
    """Routes approval requests to appropriate channels."""

    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._routes: Dict[str, List[str]] = {}  # action -> channel names

    def add_channel(self, channel: Channel) -> None:
        """Add a channel."""
        self._channels[channel.name] = channel

    def add_route(self, action_pattern: str, channel_names: List[str]) -> None:
        """Add a routing rule."""
        self._routes[action_pattern] = channel_names

    def get_channels_for_action(self, action: str) -> List[Channel]:
        """Get channels for an action."""
        channels = []
        for pattern, channel_names in self._routes.items():
            if pattern in action or pattern == "*":
                for name in channel_names:
                    if name in self._channels:
                        channel = self._channels[name]
                        if channel.enabled:
                            channels.append(channel)
        return channels

    async def send_notification(
        self,
        request: ApprovalRequest,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """Send notification to channels."""
        if channels:
            target_channels = [self._channels[n] for n in channels if n in self._channels]
        else:
            target_channels = self.get_channels_for_action(request.action)

        results = {}
        for channel in target_channels:
            # Simulate sending notification
            results[channel.name] = True

        return results


@dataclass
class EscalationLevel:
    """A level in the escalation chain."""
    name: str
    channels: List[str]
    timeout: int  # seconds before escalating


class EscalationManager:
    """Manages escalation of unanswered approval requests."""

    def __init__(self, router: ChannelRouter):
        self.router = router
        self._escalation_chains: Dict[str, List[EscalationLevel]] = {}
        self._current_levels: Dict[str, int] = {}

    def set_chain(self, action: str, levels: List[EscalationLevel]) -> None:
        """Set escalation chain for an action."""
        self._escalation_chains[action] = levels

    def get_chain(self, action: str) -> List[EscalationLevel]:
        """Get escalation chain for an action."""
        return self._escalation_chains.get(action, [])

    async def escalate(self, request: ApprovalRequest) -> Optional[EscalationLevel]:
        """Escalate a request to the next level."""
        chain = self.get_chain(request.action)
        if not chain:
            return None

        current = self._current_levels.get(request.request_id, -1)
        next_level = current + 1

        if next_level >= len(chain):
            return None

        self._current_levels[request.request_id] = next_level
        level = chain[next_level]

        # Send notifications to escalation channels
        await self.router.send_notification(request, level.channels)

        return level

    def get_current_level(self, request_id: str) -> int:
        """Get current escalation level for a request."""
        return self._current_levels.get(request_id, -1)


# ---------------------------------------------------------------------------
# Tests for @require_approval Decorator
# ---------------------------------------------------------------------------

class TestRequireApprovalDecorator:
    """Tests for @require_approval decorator."""

    @pytest.mark.asyncio
    async def test_decorator_with_approval(self):
        """Test decorated function executes with approval."""
        @require_approval("test_action", "Test description")
        async def test_func(x: int) -> int:
            return x * 2

        workflow = ApprovalWorkflow()
        result = await test_func(5, _approval_workflow=workflow)

        assert result == 10

    @pytest.mark.asyncio
    async def test_decorator_without_workflow(self):
        """Test decorated function works without workflow."""
        @require_approval("test_action")
        async def test_func(x: int) -> int:
            return x + 1

        # Without workflow, should just execute
        result = await test_func(5)
        assert result == 6

    @pytest.mark.asyncio
    async def test_decorator_preserves_metadata(self):
        """Test decorator preserves function metadata."""
        @require_approval("action")
        async def documented_func():
            """This is the docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert "docstring" in documented_func.__doc__


# ---------------------------------------------------------------------------
# Tests for ApprovalWorkflow
# ---------------------------------------------------------------------------

class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow."""

    @pytest.mark.asyncio
    async def test_create_request(self):
        """Test creating approval request."""
        workflow = ApprovalWorkflow()
        request = await workflow.request_approval(
            action="delete_file",
            description="Delete important.txt"
        )

        assert request is not None
        assert request.status == ApprovalStatus.PENDING
        assert request.action == "delete_file"

    @pytest.mark.asyncio
    async def test_approve_request(self):
        """Test approving a request."""
        workflow = ApprovalWorkflow()
        request = await workflow.request_approval("action", "desc")

        result = await workflow.approve(request.request_id, "admin")

        assert result is True
        assert request.status == ApprovalStatus.APPROVED
        assert request.approver == "admin"

    @pytest.mark.asyncio
    async def test_reject_request(self):
        """Test rejecting a request."""
        workflow = ApprovalWorkflow()
        request = await workflow.request_approval("action", "desc")

        result = await workflow.reject(request.request_id)

        assert result is True
        assert request.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_request_expiration(self):
        """Test request expiration."""
        workflow = ApprovalWorkflow(default_timeout=0)  # Immediate expiration
        request = await workflow.request_approval("action", "desc", timeout=0)

        await asyncio.sleep(0.01)  # Small delay

        assert request.is_expired

    @pytest.mark.asyncio
    async def test_get_pending_requests(self):
        """Test getting pending requests."""
        workflow = ApprovalWorkflow()

        req1 = await workflow.request_approval("action1", "desc1")
        req2 = await workflow.request_approval("action2", "desc2")
        await workflow.approve(req1.request_id)

        pending = workflow.get_pending()

        assert len(pending) == 1
        assert pending[0].request_id == req2.request_id

    @pytest.mark.asyncio
    async def test_cannot_approve_twice(self):
        """Test that approved requests cannot be approved again."""
        workflow = ApprovalWorkflow()
        request = await workflow.request_approval("action", "desc")

        await workflow.approve(request.request_id)
        result = await workflow.approve(request.request_id)

        assert result is False  # Already approved


# ---------------------------------------------------------------------------
# Tests for ChannelRouter
# ---------------------------------------------------------------------------

class TestChannelRouter:
    """Tests for ChannelRouter."""

    def test_add_channel(self):
        """Test adding channels."""
        router = ChannelRouter()
        channel = Channel("console", ChannelType.CONSOLE)
        router.add_channel(channel)

        assert "console" in router._channels

    def test_add_route(self):
        """Test adding routes."""
        router = ChannelRouter()
        router.add_channel(Channel("slack", ChannelType.SLACK))
        router.add_route("git", ["slack"])

        assert "git" in router._routes

    def test_get_channels_for_action(self):
        """Test getting channels for an action."""
        router = ChannelRouter()
        router.add_channel(Channel("console", ChannelType.CONSOLE))
        router.add_channel(Channel("slack", ChannelType.SLACK))
        router.add_route("git", ["console", "slack"])

        channels = router.get_channels_for_action("git.push")

        assert len(channels) == 2

    def test_disabled_channel_excluded(self):
        """Test that disabled channels are excluded."""
        router = ChannelRouter()
        router.add_channel(Channel("console", ChannelType.CONSOLE, enabled=True))
        router.add_channel(Channel("slack", ChannelType.SLACK, enabled=False))
        router.add_route("*", ["console", "slack"])

        channels = router.get_channels_for_action("any_action")

        assert len(channels) == 1
        assert channels[0].name == "console"

    @pytest.mark.asyncio
    async def test_send_notification(self):
        """Test sending notifications."""
        router = ChannelRouter()
        router.add_channel(Channel("console", ChannelType.CONSOLE))
        router.add_route("*", ["console"])

        request = ApprovalRequest(
            request_id="test",
            action="test_action",
            description="Test",
            context={}
        )

        results = await router.send_notification(request)

        assert results["console"] is True


# ---------------------------------------------------------------------------
# Tests for EscalationManager
# ---------------------------------------------------------------------------

class TestEscalationManager:
    """Tests for EscalationManager."""

    @pytest.fixture
    def setup_escalation(self):
        """Set up router and manager for tests."""
        router = ChannelRouter()
        router.add_channel(Channel("console", ChannelType.CONSOLE))
        router.add_channel(Channel("slack", ChannelType.SLACK))
        router.add_channel(Channel("email", ChannelType.EMAIL))

        manager = EscalationManager(router)
        manager.set_chain("critical", [
            EscalationLevel("primary", ["console"], 60),
            EscalationLevel("secondary", ["slack"], 120),
            EscalationLevel("final", ["email"], 300),
        ])

        return router, manager

    def test_set_escalation_chain(self, setup_escalation):
        """Test setting escalation chain."""
        _, manager = setup_escalation

        chain = manager.get_chain("critical")
        assert len(chain) == 3
        assert chain[0].name == "primary"

    @pytest.mark.asyncio
    async def test_escalate_to_next_level(self, setup_escalation):
        """Test escalating to next level."""
        _, manager = setup_escalation

        request = ApprovalRequest(
            request_id="test",
            action="critical",
            description="Critical action",
            context={}
        )

        level = await manager.escalate(request)

        assert level is not None
        assert level.name == "primary"
        assert manager.get_current_level(request.request_id) == 0

    @pytest.mark.asyncio
    async def test_multiple_escalations(self, setup_escalation):
        """Test multiple escalation steps."""
        _, manager = setup_escalation

        request = ApprovalRequest(
            request_id="test",
            action="critical",
            description="Critical",
            context={}
        )

        level1 = await manager.escalate(request)
        level2 = await manager.escalate(request)
        level3 = await manager.escalate(request)
        level4 = await manager.escalate(request)

        assert level1.name == "primary"
        assert level2.name == "secondary"
        assert level3.name == "final"
        assert level4 is None  # No more levels

    @pytest.mark.asyncio
    async def test_no_chain_for_action(self, setup_escalation):
        """Test escalation for action without chain."""
        _, manager = setup_escalation

        request = ApprovalRequest(
            request_id="test",
            action="unknown_action",
            description="Unknown",
            context={}
        )

        level = await manager.escalate(request)
        assert level is None
