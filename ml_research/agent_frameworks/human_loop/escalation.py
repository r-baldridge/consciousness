"""Escalation management for approval requests.

This module provides escalation chains and management for approval
requests that are not answered within their timeout period. Escalation
allows requests to be routed to increasingly senior reviewers or
alternative channels.

Inspired by incident management best practices and HumanLayer's
escalation patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging

from .approval import ApprovalRequest, ApprovalResult, ApprovalStatus
from .channel_router import ChannelRouter, ChannelMessage, MessagePriority

logger = logging.getLogger(__name__)


class EscalationTrigger(Enum):
    """What triggers an escalation."""
    TIMEOUT = "timeout"
    MANUAL = "manual"
    CONDITION = "condition"
    SCHEDULE = "schedule"


class EscalationAction(Enum):
    """Actions to take during escalation."""
    NOTIFY = "notify"         # Just notify additional people
    TRANSFER = "transfer"     # Transfer ownership
    BROADCAST = "broadcast"   # Send to all channels in level
    AUTO_APPROVE = "auto_approve"  # Auto-approve after exhausting chain
    AUTO_DENY = "auto_deny"   # Auto-deny after exhausting chain


@dataclass
class EscalationLevel:
    """A level in an escalation chain.

    Attributes:
        name: Human-readable name for this level
        channels: List of channel names to notify at this level
        timeout: How long to wait at this level before escalating
        notifyees: List of user/group identifiers to notify
        action: What action to take at this level
        message_template: Optional custom message template
        conditions: Optional conditions that must be met to use this level
    """
    name: str
    channels: List[str]
    timeout: timedelta
    notifyees: List[str] = field(default_factory=list)
    action: EscalationAction = EscalationAction.NOTIFY
    message_template: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "channels": self.channels,
            "timeout_seconds": self.timeout.total_seconds(),
            "notifyees": self.notifyees,
            "action": self.action.value,
            "message_template": self.message_template,
            "conditions": self.conditions,
        }


@dataclass
class EscalationChain:
    """A chain of escalation levels.

    Attributes:
        name: Name of this escalation chain
        levels: Ordered list of escalation levels
        final_action: Action to take if chain is exhausted
        metadata: Additional metadata about the chain
    """
    name: str
    levels: List[EscalationLevel]
    final_action: EscalationAction = EscalationAction.AUTO_DENY
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the chain."""
        if not self.levels:
            raise ValueError("Escalation chain must have at least one level")

    @property
    def total_timeout(self) -> timedelta:
        """Calculate total timeout across all levels."""
        return sum(
            (level.timeout for level in self.levels),
            timedelta()
        )

    def get_level(self, index: int) -> Optional[EscalationLevel]:
        """Get a level by index.

        Args:
            index: Level index (0-based)

        Returns:
            The level, or None if index out of range
        """
        if 0 <= index < len(self.levels):
            return self.levels[index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "levels": [level.to_dict() for level in self.levels],
            "final_action": self.final_action.value,
            "total_timeout_seconds": self.total_timeout.total_seconds(),
            "metadata": self.metadata,
        }


@dataclass
class EscalationState:
    """Tracks the state of an escalating request.

    Attributes:
        request: The approval request being escalated
        chain: The escalation chain being used
        current_level: Current level index (0-based)
        started_at: When escalation started
        level_started_at: When current level started
        history: History of escalation events
    """
    request: ApprovalRequest
    chain: EscalationChain
    current_level: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    level_started_at: datetime = field(default_factory=datetime.now)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_escalation_level(self) -> Optional[EscalationLevel]:
        """Get the current escalation level."""
        return self.chain.get_level(self.current_level)

    @property
    def is_exhausted(self) -> bool:
        """Check if all levels have been exhausted."""
        return self.current_level >= len(self.chain.levels)

    @property
    def time_at_level(self) -> timedelta:
        """Calculate time spent at current level."""
        return datetime.now() - self.level_started_at

    @property
    def should_escalate(self) -> bool:
        """Check if we should escalate to next level."""
        if self.is_exhausted:
            return False
        level = self.current_escalation_level
        if level is None:
            return False
        return self.time_at_level >= level.timeout

    def advance_level(self) -> Optional[EscalationLevel]:
        """Advance to the next escalation level.

        Returns:
            The new level, or None if chain exhausted
        """
        self.history.append({
            "event": "level_completed",
            "level": self.current_level,
            "level_name": self.current_escalation_level.name if self.current_escalation_level else None,
            "duration_seconds": self.time_at_level.total_seconds(),
            "timestamp": datetime.now().isoformat()
        })

        self.current_level += 1
        self.level_started_at = datetime.now()

        return self.current_escalation_level

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "request_id": self.request.id,
            "chain_name": self.chain.name,
            "current_level": self.current_level,
            "current_level_name": self.current_escalation_level.name if self.current_escalation_level else None,
            "started_at": self.started_at.isoformat(),
            "level_started_at": self.level_started_at.isoformat(),
            "is_exhausted": self.is_exhausted,
            "history": self.history,
        }


# Type for escalation event handlers
EscalationHandler = Callable[[EscalationState, EscalationLevel], Awaitable[None]]


class EscalationManager:
    """Manages escalation of unanswered requests.

    The EscalationManager handles the progression of approval requests
    through escalation chains, sending notifications and tracking state.
    """

    def __init__(
        self,
        channel_router: Optional[ChannelRouter] = None,
        default_chain: Optional[EscalationChain] = None
    ):
        """Initialize the escalation manager.

        Args:
            channel_router: Router for sending escalation notifications
            default_chain: Default escalation chain to use
        """
        self._router = channel_router
        self._default_chain = default_chain
        self._chains: Dict[str, EscalationChain] = {}
        self._active_escalations: Dict[str, EscalationState] = {}
        self._handlers: List[EscalationHandler] = []
        self._lock = asyncio.Lock()

        if default_chain:
            self._chains[default_chain.name] = default_chain

    def register_chain(self, chain: EscalationChain) -> None:
        """Register an escalation chain.

        Args:
            chain: The chain to register
        """
        self._chains[chain.name] = chain
        logger.info(f"Registered escalation chain: {chain.name}")

    def get_chain(self, name: str) -> Optional[EscalationChain]:
        """Get a chain by name.

        Args:
            name: Name of the chain

        Returns:
            The chain, or None if not found
        """
        return self._chains.get(name)

    def add_handler(self, handler: EscalationHandler) -> None:
        """Add an escalation event handler.

        Args:
            handler: Async function called on escalation events
        """
        self._handlers.append(handler)

    async def escalate(
        self,
        request: ApprovalRequest,
        chain: Optional[EscalationChain] = None
    ) -> ApprovalResult:
        """Escalate a request through an escalation chain.

        This method progresses through escalation levels until:
        - The request is approved/denied
        - The chain is exhausted
        - An error occurs

        Args:
            request: The approval request to escalate
            chain: Optional chain to use (defaults to default_chain)

        Returns:
            The final approval result
        """
        escalation_chain = chain or self._default_chain
        if not escalation_chain:
            raise ValueError("No escalation chain provided or configured")

        # Create escalation state
        state = EscalationState(
            request=request,
            chain=escalation_chain
        )

        async with self._lock:
            self._active_escalations[request.id] = state

        logger.info(
            f"Starting escalation for request {request.id} "
            f"using chain '{escalation_chain.name}'"
        )

        try:
            return await self._run_escalation(state)
        finally:
            async with self._lock:
                if request.id in self._active_escalations:
                    del self._active_escalations[request.id]

    async def _run_escalation(self, state: EscalationState) -> ApprovalResult:
        """Run the escalation process.

        Args:
            state: The escalation state

        Returns:
            The approval result
        """
        while not state.is_exhausted:
            level = state.current_escalation_level
            if level is None:
                break

            logger.info(
                f"Escalation level {state.current_level}: {level.name} "
                f"(timeout: {level.timeout.total_seconds()}s)"
            )

            # Send notifications for this level
            await self._notify_level(state, level)

            # Call handlers
            for handler in self._handlers:
                try:
                    await handler(state, level)
                except Exception as e:
                    logger.error(f"Escalation handler error: {e}")

            # Wait for response or timeout
            result = await self._wait_for_level(state, level)

            if result in (ApprovalResult.APPROVED, ApprovalResult.DENIED):
                state.history.append({
                    "event": "resolved",
                    "result": result.value,
                    "level": state.current_level,
                    "timestamp": datetime.now().isoformat()
                })
                return result

            # Move to next level
            state.advance_level()

        # Chain exhausted - apply final action
        return await self._apply_final_action(state)

    async def _notify_level(
        self,
        state: EscalationState,
        level: EscalationLevel
    ) -> None:
        """Send notifications for an escalation level.

        Args:
            state: The escalation state
            level: The current level
        """
        if not self._router:
            logger.warning("No channel router configured for escalation notifications")
            return

        # Build escalation message
        message_content = self._build_escalation_message(state, level)

        for channel_name in level.channels:
            try:
                channel = self._router.get_channel(channel_name)
                if channel and channel.is_connected:
                    message = ChannelMessage(
                        content=message_content,
                        metadata={
                            "type": "escalation",
                            "request_id": state.request.id,
                            "escalation_level": state.current_level,
                            "level_name": level.name,
                        },
                        priority=MessagePriority.URGENT
                    )
                    await channel.send(message)
                    logger.info(f"Sent escalation to channel: {channel_name}")
            except Exception as e:
                logger.error(f"Failed to send escalation to {channel_name}: {e}")

    def _build_escalation_message(
        self,
        state: EscalationState,
        level: EscalationLevel
    ) -> str:
        """Build the escalation notification message."""
        if level.message_template:
            return level.message_template.format(
                request=state.request,
                level=level,
                state=state
            )

        return f"""
**ESCALATION - Level {state.current_level + 1}: {level.name}**

An approval request requires attention.

**Request ID:** {state.request.id}
**Action:** {state.request.action}
**Description:** {state.request.description}

**Escalation Info:**
- Started: {state.started_at.strftime('%Y-%m-%d %H:%M:%S')}
- Time at level: {state.time_at_level.total_seconds():.0f}s
- Notifying: {', '.join(level.notifyees) if level.notifyees else 'channel members'}

Reply with `approve {state.request.id}` or `deny {state.request.id} <reason>`
"""

    async def _wait_for_level(
        self,
        state: EscalationState,
        level: EscalationLevel
    ) -> ApprovalResult:
        """Wait for a response at the current level.

        Args:
            state: The escalation state
            level: The current level

        Returns:
            The result (TIMEOUT if level times out)
        """
        timeout_seconds = level.timeout.total_seconds()
        poll_interval = min(5.0, timeout_seconds / 10)

        start = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed >= timeout_seconds:
                return ApprovalResult.TIMEOUT

            # Check if request has been resolved
            if state.request.result in (ApprovalResult.APPROVED, ApprovalResult.DENIED):
                return state.request.result

            # Check if request status changed
            if state.request.status == ApprovalStatus.COMPLETED:
                return state.request.result or ApprovalResult.TIMEOUT

            await asyncio.sleep(poll_interval)

    async def _apply_final_action(self, state: EscalationState) -> ApprovalResult:
        """Apply the final action when chain is exhausted.

        Args:
            state: The escalation state

        Returns:
            The final approval result
        """
        final_action = state.chain.final_action

        logger.info(
            f"Escalation chain exhausted for {state.request.id}, "
            f"applying final action: {final_action.value}"
        )

        state.history.append({
            "event": "chain_exhausted",
            "final_action": final_action.value,
            "timestamp": datetime.now().isoformat()
        })

        if final_action == EscalationAction.AUTO_APPROVE:
            state.request.result = ApprovalResult.APPROVED
            state.request.metadata["auto_approved"] = True
            state.request.metadata["auto_approved_reason"] = "escalation_chain_exhausted"
            return ApprovalResult.APPROVED

        elif final_action == EscalationAction.AUTO_DENY:
            state.request.result = ApprovalResult.DENIED
            state.request.denial_reason = "Escalation chain exhausted without response"
            return ApprovalResult.DENIED

        else:
            # NOTIFY, TRANSFER, BROADCAST - just timeout
            return ApprovalResult.TIMEOUT

    async def get_active_escalations(self) -> List[EscalationState]:
        """Get all active escalations.

        Returns:
            List of active escalation states
        """
        async with self._lock:
            return list(self._active_escalations.values())

    async def cancel_escalation(self, request_id: str) -> bool:
        """Cancel an active escalation.

        Args:
            request_id: ID of the request to cancel

        Returns:
            True if cancelled, False if not found
        """
        async with self._lock:
            if request_id in self._active_escalations:
                state = self._active_escalations[request_id]
                state.history.append({
                    "event": "cancelled",
                    "timestamp": datetime.now().isoformat()
                })
                del self._active_escalations[request_id]
                logger.info(f"Cancelled escalation for request {request_id}")
                return True
            return False

    async def resolve_escalation(
        self,
        request_id: str,
        result: ApprovalResult,
        reviewer: str,
        reason: Optional[str] = None
    ) -> bool:
        """Resolve an active escalation.

        Args:
            request_id: ID of the request
            result: The approval result
            reviewer: Who resolved the request
            reason: Optional reason (for denials)

        Returns:
            True if resolved, False if not found
        """
        async with self._lock:
            if request_id not in self._active_escalations:
                return False

            state = self._active_escalations[request_id]
            state.request.result = result
            state.request.reviewer = reviewer
            state.request.status = ApprovalStatus.COMPLETED

            if reason and result == ApprovalResult.DENIED:
                state.request.denial_reason = reason

            state.history.append({
                "event": "resolved_during_escalation",
                "result": result.value,
                "reviewer": reviewer,
                "reason": reason,
                "level": state.current_level,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(
                f"Escalation for {request_id} resolved: {result.value} by {reviewer}"
            )
            return True


# Predefined escalation chains

def create_standard_chain(
    name: str = "standard",
    initial_timeout: int = 300,
    escalation_timeout: int = 600,
    final_timeout: int = 900
) -> EscalationChain:
    """Create a standard 3-level escalation chain.

    Args:
        name: Name for the chain
        initial_timeout: Timeout for first level (seconds)
        escalation_timeout: Timeout for second level (seconds)
        final_timeout: Timeout for final level (seconds)

    Returns:
        Configured escalation chain
    """
    return EscalationChain(
        name=name,
        levels=[
            EscalationLevel(
                name="Initial",
                channels=["default"],
                timeout=timedelta(seconds=initial_timeout),
                action=EscalationAction.NOTIFY
            ),
            EscalationLevel(
                name="Escalated",
                channels=["default", "slack"],
                timeout=timedelta(seconds=escalation_timeout),
                action=EscalationAction.BROADCAST
            ),
            EscalationLevel(
                name="Urgent",
                channels=["default", "slack", "email"],
                timeout=timedelta(seconds=final_timeout),
                action=EscalationAction.BROADCAST
            ),
        ],
        final_action=EscalationAction.AUTO_DENY
    )


def create_urgent_chain(name: str = "urgent") -> EscalationChain:
    """Create an urgent escalation chain with short timeouts.

    Args:
        name: Name for the chain

    Returns:
        Configured escalation chain
    """
    return EscalationChain(
        name=name,
        levels=[
            EscalationLevel(
                name="Immediate",
                channels=["slack", "discord"],
                timeout=timedelta(seconds=60),
                action=EscalationAction.BROADCAST
            ),
            EscalationLevel(
                name="Critical",
                channels=["slack", "discord", "email"],
                timeout=timedelta(seconds=120),
                notifyees=["oncall", "team-lead"],
                action=EscalationAction.BROADCAST
            ),
        ],
        final_action=EscalationAction.AUTO_DENY
    )


def create_lenient_chain(name: str = "lenient") -> EscalationChain:
    """Create a lenient chain that auto-approves if exhausted.

    Args:
        name: Name for the chain

    Returns:
        Configured escalation chain
    """
    return EscalationChain(
        name=name,
        levels=[
            EscalationLevel(
                name="Request",
                channels=["default"],
                timeout=timedelta(seconds=600),
                action=EscalationAction.NOTIFY
            ),
            EscalationLevel(
                name="Reminder",
                channels=["default"],
                timeout=timedelta(seconds=600),
                action=EscalationAction.NOTIFY
            ),
        ],
        final_action=EscalationAction.AUTO_APPROVE
    )
