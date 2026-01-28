"""
Orchestration Module for Agent Frameworks.

This module provides a complete orchestration infrastructure for managing
multiple AI agents, inspired by Clawdbot's architecture. It includes a
central control plane, intelligent routing, multi-channel communication,
isolated workspaces, and inter-agent messaging.

Components:
    - Gateway: Central control plane for managing multiple agents
    - AgentRouter: Routes tasks to appropriate agents based on rules/capabilities
    - ChannelAdapter: Multi-channel adapters (Slack, Discord, Telegram, etc.)
    - Workspace: Per-agent isolated workspaces with snapshots
    - EventBus: Pub/sub event bus for inter-agent communication

Example:
    from agent_frameworks.orchestration import (
        Gateway, GatewayConfig, AgentStatus,
        AgentRouter, RoutingRule,
        MultiChannelManager, SlackAdapter, DiscordAdapter,
        Workspace, WorkspaceManager, WorkspaceConfig,
        EventBus, Event
    )

    # Create and configure gateway
    config = GatewayConfig(max_concurrent_agents=10)
    gateway = Gateway(config)

    # Register agents
    await gateway.register_agent(my_agent)

    # Set up routing
    router = AgentRouter()
    router.register(code_agent, tasks=["code_review", "refactor"])
    router.add_rule(RoutingRule(pattern=".*bug.*", agent_id="debug_agent"))

    # Multi-channel support
    channel_mgr = MultiChannelManager()
    await channel_mgr.add_channel("slack", SlackAdapter(token="..."))
    await channel_mgr.add_channel("discord", DiscordAdapter(token="..."))

    # Workspace isolation
    ws_mgr = WorkspaceManager(WorkspaceConfig(base_path=Path("/tmp/workspaces")))
    workspace = await ws_mgr.create("agent-1")

    # Inter-agent events
    event_bus = EventBus()
    event_bus.subscribe("task_complete", handle_completion)
    await event_bus.publish(Event(type="task_complete", source="agent-1", data=result))
"""

from .gateway import (
    GatewayConfig,
    AgentStatus,
    Gateway,
)

from .agent_router import (
    RoutingRule,
    AgentRouter,
)

from .channel_adapter import (
    Message,
    MessageType,
    ChannelAdapter,
    SlackAdapter,
    DiscordAdapter,
    TelegramAdapter,
    WebhookAdapter,
    WebSocketAdapter,
    MultiChannelManager,
)

from .workspace import (
    WorkspaceConfig,
    Workspace,
    WorkspaceManager,
)

from .event_bus import (
    Event,
    EventBus,
)

from .base import (
    Task,
    TaskResult,
    TaskStatus,
    TaskPriority,
    AgentBase,
    AgentCapability,
    Session,
)

__all__ = [
    # Gateway
    "GatewayConfig",
    "AgentStatus",
    "Gateway",
    # Agent Router
    "RoutingRule",
    "AgentRouter",
    # Channel Adapters
    "Message",
    "MessageType",
    "ChannelAdapter",
    "SlackAdapter",
    "DiscordAdapter",
    "TelegramAdapter",
    "WebhookAdapter",
    "WebSocketAdapter",
    "MultiChannelManager",
    # Workspace
    "WorkspaceConfig",
    "Workspace",
    "WorkspaceManager",
    # Event Bus
    "Event",
    "EventBus",
    # Base Types
    "Task",
    "TaskResult",
    "TaskStatus",
    "TaskPriority",
    "AgentBase",
    "AgentCapability",
    "Session",
]
