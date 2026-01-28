"""Permission management system for tools.

This module provides a comprehensive permission system for controlling
tool access based on user context and security policies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Callable, Awaitable
from enum import Enum, auto
import asyncio
from datetime import datetime, timedelta

from .tool_base import Tool, ToolPermission, ToolResult


class PermissionDecision(Enum):
    """Decision outcomes for permission checks."""
    ALLOW = auto()
    DENY = auto()
    REQUIRE_APPROVAL = auto()


@dataclass
class UserContext:
    """Context information about the user requesting tool access.

    Attributes:
        user_id: Unique identifier for the user
        roles: Set of roles assigned to the user
        permissions: Set of explicit permissions granted
        metadata: Additional context information
    """
    user_id: str
    roles: Set[str] = field(default_factory=set)
    permissions: Set[ToolPermission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: ToolPermission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    @classmethod
    def admin(cls, user_id: str = "admin") -> UserContext:
        """Create an admin context with all permissions."""
        return cls(
            user_id=user_id,
            roles={"admin"},
            permissions=set(ToolPermission)
        )

    @classmethod
    def anonymous(cls) -> UserContext:
        """Create an anonymous context with minimal permissions."""
        return cls(
            user_id="anonymous",
            roles=set(),
            permissions={ToolPermission.READ}
        )


@dataclass
class PermissionRequest:
    """A request for permission to use a tool.

    Attributes:
        tool: The tool being requested
        user_context: Context of the requesting user
        reason: Explanation for why the tool is needed
        arguments: The arguments that will be passed to the tool
        timestamp: When the request was made
    """
    tool: Tool
    user_context: UserContext
    reason: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary format."""
        return {
            "tool_name": self.tool.name,
            "user_id": self.user_context.user_id,
            "reason": self.reason,
            "arguments": self.arguments,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PermissionGrant:
    """A granted permission with optional expiration.

    Attributes:
        tool_name: Name of the tool granted access to
        user_id: User who was granted access
        granted_at: When the permission was granted
        expires_at: When the permission expires (None for permanent)
        scope: Optional restrictions on the grant
    """
    tool_name: str
    user_id: str
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    scope: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if this grant has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @classmethod
    def temporary(
        cls,
        tool_name: str,
        user_id: str,
        duration: timedelta = timedelta(hours=1)
    ) -> PermissionGrant:
        """Create a temporary permission grant."""
        now = datetime.now()
        return cls(
            tool_name=tool_name,
            user_id=user_id,
            granted_at=now,
            expires_at=now + duration
        )


class PermissionPolicy(ABC):
    """Abstract base class for permission policies.

    Policies define rules for deciding whether to allow tool access.
    """

    @abstractmethod
    def evaluate(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        """Evaluate whether to allow tool access.

        Args:
            tool: The tool being accessed
            user_context: Context of the requesting user

        Returns:
            PermissionDecision indicating allow/deny/require_approval
        """
        ...


class AllowAllPolicy(PermissionPolicy):
    """Policy that allows all tool access."""

    def evaluate(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        return PermissionDecision.ALLOW


class DenyDangerousPolicy(PermissionPolicy):
    """Policy that denies tools with DANGEROUS permission unless user is admin."""

    def evaluate(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        if ToolPermission.DANGEROUS in tool.permissions:
            if "admin" in user_context.roles:
                return PermissionDecision.ALLOW
            return PermissionDecision.DENY
        return PermissionDecision.ALLOW


class RequireApprovalPolicy(PermissionPolicy):
    """Policy that requires approval for certain tools or permissions."""

    def __init__(
        self,
        require_approval_for: Optional[Set[ToolPermission]] = None,
        auto_approve_roles: Optional[Set[str]] = None
    ):
        self.require_approval_for = require_approval_for or {
            ToolPermission.WRITE,
            ToolPermission.EXECUTE,
            ToolPermission.DANGEROUS
        }
        self.auto_approve_roles = auto_approve_roles or {"admin"}

    def evaluate(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        # Auto-approve for certain roles
        if user_context.roles & self.auto_approve_roles:
            return PermissionDecision.ALLOW

        # Check if any tool permissions require approval
        tool_perms = set(tool.permissions)
        if tool_perms & self.require_approval_for:
            return PermissionDecision.REQUIRE_APPROVAL

        return PermissionDecision.ALLOW


class PermissionMatchingPolicy(PermissionPolicy):
    """Policy that requires user to have all permissions the tool needs."""

    def evaluate(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        required_perms = set(tool.permissions)
        user_perms = user_context.permissions

        if required_perms <= user_perms:
            return PermissionDecision.ALLOW

        return PermissionDecision.DENY


class CompositePolicy(PermissionPolicy):
    """Policy that combines multiple policies.

    All policies must allow for the final decision to be ALLOW.
    If any policy denies, the result is DENY.
    If any policy requires approval (and none deny), result is REQUIRE_APPROVAL.
    """

    def __init__(self, policies: List[PermissionPolicy]):
        self.policies = policies

    def evaluate(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        decisions = [p.evaluate(tool, user_context) for p in self.policies]

        # Any DENY means overall DENY
        if PermissionDecision.DENY in decisions:
            return PermissionDecision.DENY

        # Any REQUIRE_APPROVAL means overall REQUIRE_APPROVAL
        if PermissionDecision.REQUIRE_APPROVAL in decisions:
            return PermissionDecision.REQUIRE_APPROVAL

        return PermissionDecision.ALLOW


# Type alias for approval handlers
ApprovalHandler = Callable[[PermissionRequest], Awaitable[bool]]


class PermissionManager:
    """Central manager for tool permissions.

    Handles permission checks, approval requests, and grant management.
    """

    def __init__(
        self,
        policy: Optional[PermissionPolicy] = None,
        approval_handler: Optional[ApprovalHandler] = None
    ):
        """Initialize the permission manager.

        Args:
            policy: The policy to use for permission decisions
            approval_handler: Async function to handle approval requests
        """
        self._policy = policy or RequireApprovalPolicy()
        self._approval_handler = approval_handler or self._default_approval_handler
        self._grants: Dict[str, List[PermissionGrant]] = {}
        self._denied_tools: Set[str] = set()
        self._audit_log: List[Dict[str, Any]] = []

    async def _default_approval_handler(
        self,
        request: PermissionRequest
    ) -> bool:
        """Default approval handler that denies all requests."""
        return False

    def set_policy(self, policy: PermissionPolicy) -> None:
        """Set the permission policy."""
        self._policy = policy

    def set_approval_handler(self, handler: ApprovalHandler) -> None:
        """Set the approval handler function."""
        self._approval_handler = handler

    def check_permission(
        self,
        tool: Tool,
        user_context: UserContext
    ) -> PermissionDecision:
        """Check if a user has permission to use a tool.

        Args:
            tool: The tool to check access for
            user_context: Context of the requesting user

        Returns:
            PermissionDecision for the access request
        """
        # Check if tool is explicitly denied
        if tool.name in self._denied_tools:
            return PermissionDecision.DENY

        # Check for existing valid grant
        if self._has_valid_grant(tool.name, user_context.user_id):
            return PermissionDecision.ALLOW

        # Evaluate policy
        return self._policy.evaluate(tool, user_context)

    def _has_valid_grant(self, tool_name: str, user_id: str) -> bool:
        """Check if user has a valid (non-expired) grant for the tool."""
        grants = self._grants.get(user_id, [])
        for grant in grants:
            if grant.tool_name == tool_name and not grant.is_expired:
                return True
        return False

    async def request_permission(
        self,
        tool: Tool,
        user_context: UserContext,
        reason: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Request permission to use a tool.

        This method is called when a permission check returns REQUIRE_APPROVAL.

        Args:
            tool: The tool to request access for
            user_context: Context of the requesting user
            reason: Explanation for why access is needed
            arguments: Optional arguments that will be used

        Returns:
            True if permission is granted, False otherwise
        """
        request = PermissionRequest(
            tool=tool,
            user_context=user_context,
            reason=reason,
            arguments=arguments or {}
        )

        # Log the request
        self._audit_log.append({
            "event": "permission_request",
            **request.to_dict()
        })

        # Call the approval handler
        approved = await self._approval_handler(request)

        if approved:
            # Create a temporary grant
            self.grant_permission(
                tool_name=tool.name,
                user_id=user_context.user_id,
                duration=timedelta(hours=1)
            )

        # Log the decision
        self._audit_log.append({
            "event": "permission_decision",
            "tool_name": tool.name,
            "user_id": user_context.user_id,
            "approved": approved,
            "timestamp": datetime.now().isoformat()
        })

        return approved

    def grant_permission(
        self,
        tool_name: str,
        user_id: str,
        duration: Optional[timedelta] = None,
        scope: Optional[Dict[str, Any]] = None
    ) -> PermissionGrant:
        """Grant permission to a user for a tool.

        Args:
            tool_name: Name of the tool to grant access to
            user_id: User to grant access to
            duration: Optional duration for the grant (None for permanent)
            scope: Optional restrictions on the grant

        Returns:
            The created PermissionGrant
        """
        if duration:
            grant = PermissionGrant.temporary(tool_name, user_id, duration)
        else:
            grant = PermissionGrant(
                tool_name=tool_name,
                user_id=user_id,
                scope=scope or {}
            )

        if user_id not in self._grants:
            self._grants[user_id] = []
        self._grants[user_id].append(grant)

        return grant

    def revoke_permission(
        self,
        tool_name: str,
        user_id: str
    ) -> bool:
        """Revoke a user's permission for a tool.

        Args:
            tool_name: Name of the tool
            user_id: User to revoke access from

        Returns:
            True if a grant was revoked, False if none existed
        """
        if user_id not in self._grants:
            return False

        original_count = len(self._grants[user_id])
        self._grants[user_id] = [
            g for g in self._grants[user_id]
            if g.tool_name != tool_name
        ]

        return len(self._grants[user_id]) < original_count

    def deny_tool(self, tool_name: str) -> None:
        """Explicitly deny access to a tool for all users."""
        self._denied_tools.add(tool_name)

    def allow_tool(self, tool_name: str) -> None:
        """Remove explicit denial for a tool."""
        self._denied_tools.discard(tool_name)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log of permission events."""
        return self._audit_log.copy()

    def clear_expired_grants(self) -> int:
        """Remove all expired grants.

        Returns:
            Number of grants removed
        """
        removed = 0
        for user_id in self._grants:
            original_count = len(self._grants[user_id])
            self._grants[user_id] = [
                g for g in self._grants[user_id]
                if not g.is_expired
            ]
            removed += original_count - len(self._grants[user_id])
        return removed


async def check_and_execute(
    tool: Tool,
    permission_manager: PermissionManager,
    user_context: UserContext,
    arguments: Dict[str, Any],
    reason: str = "Tool execution requested"
) -> ToolResult:
    """Helper function to check permission and execute a tool.

    Args:
        tool: The tool to execute
        permission_manager: Permission manager to use
        user_context: Context of the requesting user
        arguments: Arguments to pass to the tool
        reason: Reason for the request (used if approval is needed)

    Returns:
        ToolResult from the tool execution, or error if permission denied
    """
    decision = permission_manager.check_permission(tool, user_context)

    if decision == PermissionDecision.DENY:
        return ToolResult.fail(
            f"Permission denied for tool '{tool.name}'",
            permission_decision="denied"
        )

    if decision == PermissionDecision.REQUIRE_APPROVAL:
        approved = await permission_manager.request_permission(
            tool, user_context, reason, arguments
        )
        if not approved:
            return ToolResult.fail(
                f"Permission request for tool '{tool.name}' was not approved",
                permission_decision="not_approved"
            )

    return await tool.execute(**arguments)
