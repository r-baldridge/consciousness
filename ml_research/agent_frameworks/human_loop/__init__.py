"""
Human-in-the-Loop Module for Agent Frameworks.

This module provides a comprehensive human-in-the-loop infrastructure
for AI agents, inspired by HumanLayer's approach to human oversight.
It enables agents to request human approval, query humans mid-execution,
and learn from feedback.

Components:
    - ApprovalWorkflow: Manages approval requests and their lifecycle
    - require_approval: Decorator for requiring human approval
    - HumanAsTool: Tool for querying humans during execution
    - ChannelRouter: Routes messages to Slack/email/Discord/webhook
    - EscalationManager: Handles timeout escalations
    - FeedbackCollector: Collects denial feedback for improvement

Example:
    from agent_frameworks.human_loop import (
        ApprovalWorkflow, ApprovalRequest, ApprovalResult,
        require_approval,
        HumanAsTool, HumanQuery,
        ChannelRouter, SlackChannel,
        EscalationManager, EscalationChain,
        FeedbackCollector, DenialFeedback
    )

    # Create an approval workflow
    workflow = ApprovalWorkflow()

    # Use the @require_approval decorator
    @require_approval(channel="slack", timeout=300)
    async def delete_database(db_name: str):
        # This will require human approval before execution
        await db.drop(db_name)

    # Query humans mid-execution
    human = HumanAsTool()
    response = await human.ask_multiple_choice(
        "Which database should I use?",
        options=["postgres", "mysql", "sqlite"]
    )

    # Route to different channels
    router = ChannelRouter()
    router.register_channel("slack", SlackChannel(token="..."))

    # Handle escalations
    escalation = EscalationManager(router)
    result = await escalation.escalate(request, chain)

    # Collect feedback for improvement
    collector = FeedbackCollector()
    await collector.collect(DenialFeedback(
        request_id="...",
        reason="Action too risky",
        suggested_alternative="Use soft delete instead"
    ))
"""

# Approval workflow
from .approval import (
    ApprovalResult,
    ApprovalStatus,
    ApprovalRequest,
    ApprovalDeniedError,
    ApprovalTimeoutError,
    ApprovalWorkflow,
    require_approval,
    get_default_workflow,
    set_default_workflow,
)

# Human as tool
from .human_as_tool import (
    QueryType,
    QueryStatus,
    HumanQuery,
    HumanResponse,
    HumanQueryTimeoutError,
    InvalidResponseError,
    HumanQueryManager,
    HumanAsTool,
    ask_human,
)

# Channel routing
from .channel_router import (
    ChannelType,
    MessagePriority,
    ChannelMessage,
    ChannelResponse,
    ChannelError,
    ChannelSendError,
    ChannelReceiveError,
    ChannelTimeoutError,
    Channel,
    ConsoleChannel,
    SlackChannel,
    EmailChannel,
    WebhookChannel,
    DiscordChannel,
    ChannelRouter,
)

# Escalation management
from .escalation import (
    EscalationTrigger,
    EscalationAction,
    EscalationLevel,
    EscalationChain,
    EscalationState,
    EscalationManager,
    create_standard_chain,
    create_urgent_chain,
    create_lenient_chain,
)

# Feedback collection
from .feedback_collector import (
    FeedbackSeverity,
    FeedbackCategory,
    DenialFeedback,
    FeedbackPattern,
    TrainingExample,
    FeedbackCollector,
)

__all__ = [
    # Approval workflow
    "ApprovalResult",
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalDeniedError",
    "ApprovalTimeoutError",
    "ApprovalWorkflow",
    "require_approval",
    "get_default_workflow",
    "set_default_workflow",
    # Human as tool
    "QueryType",
    "QueryStatus",
    "HumanQuery",
    "HumanResponse",
    "HumanQueryTimeoutError",
    "InvalidResponseError",
    "HumanQueryManager",
    "HumanAsTool",
    "ask_human",
    # Channel routing
    "ChannelType",
    "MessagePriority",
    "ChannelMessage",
    "ChannelResponse",
    "ChannelError",
    "ChannelSendError",
    "ChannelReceiveError",
    "ChannelTimeoutError",
    "Channel",
    "ConsoleChannel",
    "SlackChannel",
    "EmailChannel",
    "WebhookChannel",
    "DiscordChannel",
    "ChannelRouter",
    # Escalation management
    "EscalationTrigger",
    "EscalationAction",
    "EscalationLevel",
    "EscalationChain",
    "EscalationState",
    "EscalationManager",
    "create_standard_chain",
    "create_urgent_chain",
    "create_lenient_chain",
    # Feedback collection
    "FeedbackSeverity",
    "FeedbackCategory",
    "DenialFeedback",
    "FeedbackPattern",
    "TrainingExample",
    "FeedbackCollector",
]
