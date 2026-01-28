"""Human approval workflow for agent actions.

This module provides the core approval infrastructure for human-in-the-loop
agent workflows. It includes the @require_approval decorator and ApprovalWorkflow
class for managing approval requests and their lifecycle.

Inspired by HumanLayer's approach to human oversight of AI agents.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any, Dict, TypeVar, ParamSpec
from enum import Enum
from datetime import datetime, timedelta
import functools
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class ApprovalResult(Enum):
    """Result of an approval request."""
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """A request for human approval of an agent action.

    Attributes:
        id: Unique identifier for the request
        action: Name of the action requiring approval
        description: Human-readable description of what the action will do
        arguments: Arguments that will be passed to the action
        requester: Identifier of the agent/system requesting approval
        channel: Channel through which to request approval
        timeout: How long to wait for approval before timing out
        created_at: When the request was created
        result: The approval decision (None if pending)
        reviewer: Who reviewed the request
        denial_reason: If denied, the reason provided
        metadata: Additional context about the request
    """
    id: str
    action: str
    description: str
    arguments: Dict[str, Any]
    requester: str
    channel: str
    timeout: timedelta
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[ApprovalResult] = None
    reviewer: Optional[str] = None
    denial_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    completed_at: Optional[datetime] = None

    @property
    def is_pending(self) -> bool:
        """Check if the request is still pending."""
        return self.status == ApprovalStatus.PENDING and self.result is None

    @property
    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.result is not None:
            return False
        return datetime.now() > self.created_at + self.timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary format."""
        return {
            "id": self.id,
            "action": self.action,
            "description": self.description,
            "arguments": self.arguments,
            "requester": self.requester,
            "channel": self.channel,
            "timeout_seconds": self.timeout.total_seconds(),
            "created_at": self.created_at.isoformat(),
            "result": self.result.value if self.result else None,
            "reviewer": self.reviewer,
            "denial_reason": self.denial_reason,
            "metadata": self.metadata,
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ApprovalDeniedError(Exception):
    """Raised when an action is denied approval."""

    def __init__(self, request: ApprovalRequest, reason: Optional[str] = None):
        self.request = request
        self.reason = reason or request.denial_reason
        message = f"Approval denied for action '{request.action}'"
        if self.reason:
            message += f": {self.reason}"
        super().__init__(message)


class ApprovalTimeoutError(Exception):
    """Raised when an approval request times out."""

    def __init__(self, request: ApprovalRequest):
        self.request = request
        super().__init__(
            f"Approval request for action '{request.action}' timed out "
            f"after {request.timeout.total_seconds()} seconds"
        )


class ApprovalWorkflow:
    """Manages approval requests and their lifecycle.

    The ApprovalWorkflow handles creation, tracking, and resolution of
    approval requests. It maintains a registry of pending requests and
    coordinates with channels for notification delivery.

    Attributes:
        pending: Dictionary of pending approval requests by ID
        completed: List of completed approval requests
        channel_router: Optional router for sending requests to channels
    """

    def __init__(self, channel_router: Optional[Any] = None):
        """Initialize the approval workflow.

        Args:
            channel_router: Optional ChannelRouter for sending notifications
        """
        self.pending: Dict[str, ApprovalRequest] = {}
        self.completed: List[ApprovalRequest] = []
        self.channel_router = channel_router
        self._events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def request_approval(
        self,
        request: ApprovalRequest,
        wait: bool = True
    ) -> ApprovalResult:
        """Submit a request and optionally wait for approval.

        Args:
            request: The approval request to submit
            wait: Whether to wait for the approval decision

        Returns:
            The approval result

        Raises:
            ApprovalDeniedError: If the request is denied
            ApprovalTimeoutError: If the request times out
        """
        async with self._lock:
            self.pending[request.id] = request
            self._events[request.id] = asyncio.Event()

        logger.info(
            f"Approval request submitted: {request.id} for action '{request.action}'"
        )

        # Route to channel if router available
        if self.channel_router:
            try:
                await self.channel_router.send_approval_request(request)
            except Exception as e:
                logger.error(f"Failed to route approval request: {e}")

        if not wait:
            return ApprovalResult.APPROVED  # Will be updated when resolved

        # Wait for approval with timeout
        try:
            event = self._events[request.id]
            await asyncio.wait_for(
                event.wait(),
                timeout=request.timeout.total_seconds()
            )
        except asyncio.TimeoutError:
            await self._handle_timeout(request.id)
            raise ApprovalTimeoutError(request)

        # Get the updated request
        async with self._lock:
            if request.id in self.pending:
                updated_request = self.pending[request.id]
            else:
                # Already moved to completed
                updated_request = next(
                    (r for r in self.completed if r.id == request.id),
                    request
                )

        if updated_request.result == ApprovalResult.DENIED:
            raise ApprovalDeniedError(updated_request)

        return updated_request.result or ApprovalResult.TIMEOUT

    async def approve(
        self,
        request_id: str,
        reviewer: str,
        comment: Optional[str] = None
    ) -> None:
        """Approve a pending request.

        Args:
            request_id: ID of the request to approve
            reviewer: Identifier of the reviewer approving
            comment: Optional comment from the reviewer
        """
        async with self._lock:
            if request_id not in self.pending:
                raise ValueError(f"No pending request with ID: {request_id}")

            request = self.pending[request_id]
            request.result = ApprovalResult.APPROVED
            request.reviewer = reviewer
            request.status = ApprovalStatus.COMPLETED
            request.completed_at = datetime.now()
            if comment:
                request.metadata["approval_comment"] = comment

            self.completed.append(request)
            del self.pending[request_id]

        logger.info(f"Request {request_id} approved by {reviewer}")

        # Signal waiting coroutines
        if request_id in self._events:
            self._events[request_id].set()

    async def deny(
        self,
        request_id: str,
        reviewer: str,
        reason: str
    ) -> None:
        """Deny a pending request.

        Args:
            request_id: ID of the request to deny
            reviewer: Identifier of the reviewer denying
            reason: Reason for the denial
        """
        async with self._lock:
            if request_id not in self.pending:
                raise ValueError(f"No pending request with ID: {request_id}")

            request = self.pending[request_id]
            request.result = ApprovalResult.DENIED
            request.reviewer = reviewer
            request.denial_reason = reason
            request.status = ApprovalStatus.COMPLETED
            request.completed_at = datetime.now()

            self.completed.append(request)
            del self.pending[request_id]

        logger.info(f"Request {request_id} denied by {reviewer}: {reason}")

        # Signal waiting coroutines
        if request_id in self._events:
            self._events[request_id].set()

    async def escalate(
        self,
        request_id: str,
        new_channel: str,
        reason: Optional[str] = None
    ) -> None:
        """Escalate a request to a different channel.

        Args:
            request_id: ID of the request to escalate
            new_channel: New channel to route the request to
            reason: Reason for escalation
        """
        async with self._lock:
            if request_id not in self.pending:
                raise ValueError(f"No pending request with ID: {request_id}")

            request = self.pending[request_id]
            old_channel = request.channel
            request.channel = new_channel
            request.result = ApprovalResult.ESCALATED
            request.metadata["escalation_history"] = request.metadata.get(
                "escalation_history", []
            )
            request.metadata["escalation_history"].append({
                "from": old_channel,
                "to": new_channel,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })
            # Reset result for new channel to handle
            request.result = None

        logger.info(
            f"Request {request_id} escalated from {old_channel} to {new_channel}"
        )

        # Re-route to new channel
        if self.channel_router:
            await self.channel_router.send_approval_request(request)

    async def cancel(self, request_id: str, reason: str = "Cancelled") -> None:
        """Cancel a pending request.

        Args:
            request_id: ID of the request to cancel
            reason: Reason for cancellation
        """
        async with self._lock:
            if request_id not in self.pending:
                raise ValueError(f"No pending request with ID: {request_id}")

            request = self.pending[request_id]
            request.status = ApprovalStatus.CANCELLED
            request.metadata["cancellation_reason"] = reason
            request.completed_at = datetime.now()

            self.completed.append(request)
            del self.pending[request_id]

        logger.info(f"Request {request_id} cancelled: {reason}")

        # Signal waiting coroutines
        if request_id in self._events:
            self._events[request_id].set()

    async def list_pending(
        self,
        channel: Optional[str] = None,
        requester: Optional[str] = None
    ) -> List[ApprovalRequest]:
        """List all pending approval requests.

        Args:
            channel: Optional filter by channel
            requester: Optional filter by requester

        Returns:
            List of pending approval requests
        """
        async with self._lock:
            requests = list(self.pending.values())

        if channel:
            requests = [r for r in requests if r.channel == channel]
        if requester:
            requests = [r for r in requests if r.requester == requester]

        return requests

    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific request by ID.

        Args:
            request_id: ID of the request to retrieve

        Returns:
            The approval request, or None if not found
        """
        async with self._lock:
            if request_id in self.pending:
                return self.pending[request_id]

        # Check completed
        for request in self.completed:
            if request.id == request_id:
                return request

        return None

    async def _handle_timeout(self, request_id: str) -> None:
        """Handle a request that has timed out."""
        async with self._lock:
            if request_id not in self.pending:
                return

            request = self.pending[request_id]
            request.result = ApprovalResult.TIMEOUT
            request.status = ApprovalStatus.EXPIRED
            request.completed_at = datetime.now()

            self.completed.append(request)
            del self.pending[request_id]

        logger.info(f"Request {request_id} timed out")

    async def cleanup_expired(self) -> List[ApprovalRequest]:
        """Clean up expired requests.

        Returns:
            List of requests that were expired
        """
        expired = []
        async with self._lock:
            for request_id, request in list(self.pending.items()):
                if request.is_expired:
                    request.result = ApprovalResult.TIMEOUT
                    request.status = ApprovalStatus.EXPIRED
                    request.completed_at = datetime.now()
                    self.completed.append(request)
                    expired.append(request)
                    del self.pending[request_id]
                    if request_id in self._events:
                        self._events[request_id].set()

        return expired


# Global workflow instance for decorator use
_default_workflow: Optional[ApprovalWorkflow] = None


def get_default_workflow() -> ApprovalWorkflow:
    """Get or create the default approval workflow."""
    global _default_workflow
    if _default_workflow is None:
        _default_workflow = ApprovalWorkflow()
    return _default_workflow


def set_default_workflow(workflow: ApprovalWorkflow) -> None:
    """Set the default approval workflow."""
    global _default_workflow
    _default_workflow = workflow


def require_approval(
    channel: str = "default",
    timeout: int = 300,
    description: Optional[str] = None,
    auto_approve_if: Optional[Callable[..., bool]] = None,
    requester: str = "agent",
    include_args: bool = True,
    workflow: Optional[ApprovalWorkflow] = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that requires human approval before function execution.

    This decorator wraps a function to require human approval before
    execution. The approval request includes the function name, description,
    and optionally the arguments being passed.

    Args:
        channel: The channel to request approval through
        timeout: Timeout in seconds to wait for approval
        description: Custom description for the approval request
        auto_approve_if: Optional callable that takes the same arguments
            as the decorated function and returns True if auto-approval
            should be granted
        requester: Identifier for the requester
        include_args: Whether to include function arguments in the request
        workflow: Optional workflow to use (defaults to global workflow)

    Returns:
        Decorated function that requires approval

    Example:
        @require_approval(channel="slack", timeout=600)
        async def delete_user(user_id: str):
            # Will only execute after human approval
            await db.delete_user(user_id)

        @require_approval(auto_approve_if=lambda amount: amount < 100)
        async def transfer_funds(amount: float):
            # Auto-approved if amount < 100
            await bank.transfer(amount)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Check for auto-approval
            if auto_approve_if is not None:
                try:
                    if auto_approve_if(*args, **kwargs):
                        logger.info(
                            f"Auto-approving {func.__name__} based on condition"
                        )
                        return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Auto-approve check failed: {e}")

            # Build approval request
            func_description = description or func.__doc__ or f"Execute {func.__name__}"

            # Build arguments dict
            arguments: Dict[str, Any] = {}
            if include_args:
                # Get function signature
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                # Map positional args
                for i, arg in enumerate(args):
                    if i < len(params):
                        arguments[params[i]] = _serialize_arg(arg)

                # Add keyword args
                for key, value in kwargs.items():
                    arguments[key] = _serialize_arg(value)

            request = ApprovalRequest(
                id=str(uuid.uuid4()),
                action=func.__name__,
                description=func_description,
                arguments=arguments,
                requester=requester,
                channel=channel,
                timeout=timedelta(seconds=timeout),
                metadata={
                    "module": func.__module__,
                    "qualname": func.__qualname__,
                }
            )

            # Get workflow
            wf = workflow or get_default_workflow()

            # Request approval
            result = await wf.request_approval(request)

            if result == ApprovalResult.APPROVED:
                return await func(*args, **kwargs)
            elif result == ApprovalResult.DENIED:
                raise ApprovalDeniedError(request)
            else:
                raise ApprovalTimeoutError(request)

        return wrapper
    return decorator


def _serialize_arg(arg: Any) -> Any:
    """Serialize an argument for inclusion in approval request."""
    if arg is None or isinstance(arg, (str, int, float, bool)):
        return arg
    elif isinstance(arg, (list, tuple)):
        return [_serialize_arg(item) for item in arg]
    elif isinstance(arg, dict):
        return {str(k): _serialize_arg(v) for k, v in arg.items()}
    elif hasattr(arg, "to_dict"):
        return arg.to_dict()
    elif hasattr(arg, "__dict__"):
        return {k: _serialize_arg(v) for k, v in arg.__dict__.items()
                if not k.startswith("_")}
    else:
        return str(arg)
