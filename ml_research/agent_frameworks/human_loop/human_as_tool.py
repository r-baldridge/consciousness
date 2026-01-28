"""Human-as-a-Tool for agent queries during execution.

This module provides tools that allow agents to query humans during
execution, treating human input as just another tool in the agent's
toolkit. This enables agents to request clarification, confirmation,
or additional information when needed.

Inspired by HumanLayer's approach to interactive human-agent collaboration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid
import logging

# Import from sibling tool module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.tool_base import Tool, ToolSchema, ToolResult, ToolPermission

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Type of human query."""
    FREE_FORM = "free_form"
    MULTIPLE_CHOICE = "multiple_choice"
    CONFIRMATION = "confirmation"
    NUMERIC = "numeric"
    DATE = "date"


class QueryStatus(Enum):
    """Status of a human query."""
    PENDING = "pending"
    ANSWERED = "answered"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class HumanQuery:
    """A query to be answered by a human.

    Attributes:
        question: The question to ask the human
        context: Additional context to help the human answer
        options: For multiple choice, the available options
        query_type: The type of query
        timeout: Timeout in seconds for the response
        id: Unique identifier for the query
        created_at: When the query was created
        priority: Priority level (higher = more urgent)
        metadata: Additional metadata about the query
    """
    question: str
    context: str = ""
    options: Optional[List[str]] = None
    query_type: QueryType = QueryType.FREE_FORM
    timeout: int = 300
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set query type based on options if not explicitly set."""
        if self.options is not None and self.query_type == QueryType.FREE_FORM:
            self.query_type = QueryType.MULTIPLE_CHOICE

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary format."""
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "options": self.options,
            "query_type": self.query_type.value,
            "timeout": self.timeout,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority,
            "metadata": self.metadata,
        }

    def format_for_display(self) -> str:
        """Format the query for human display."""
        lines = [f"Question: {self.question}"]

        if self.context:
            lines.append(f"\nContext: {self.context}")

        if self.options:
            lines.append("\nOptions:")
            for i, option in enumerate(self.options, 1):
                lines.append(f"  {i}. {option}")

        return "\n".join(lines)


@dataclass
class HumanResponse:
    """Response from a human to a query.

    Attributes:
        query_id: ID of the query being answered
        response: The human's response
        responder: Identifier of the human responding
        timestamp: When the response was given
        confidence: Self-reported confidence (0-1)
        notes: Additional notes from the responder
    """
    query_id: str
    response: str
    responder: str = "human"
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "query_id": self.query_id,
            "response": self.response,
            "responder": self.responder,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "notes": self.notes,
        }


class HumanQueryTimeoutError(Exception):
    """Raised when a human query times out."""

    def __init__(self, query: HumanQuery):
        self.query = query
        super().__init__(
            f"Human query timed out after {query.timeout} seconds: {query.question}"
        )


class InvalidResponseError(Exception):
    """Raised when a human response is invalid."""

    def __init__(self, query: HumanQuery, response: str, reason: str):
        self.query = query
        self.response = response
        self.reason = reason
        super().__init__(f"Invalid response for query: {reason}")


# Type for response handler callbacks
ResponseHandler = Callable[[HumanQuery], Awaitable[HumanResponse]]


class HumanQueryManager:
    """Manages human queries and responses.

    The HumanQueryManager handles the lifecycle of human queries,
    including submission, waiting for responses, and validation.
    """

    def __init__(self, response_handler: Optional[ResponseHandler] = None):
        """Initialize the query manager.

        Args:
            response_handler: Async function to handle query delivery and
                response collection. If None, uses console-based handler.
        """
        self.pending: Dict[str, HumanQuery] = {}
        self.responses: Dict[str, HumanResponse] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._response_handler = response_handler or self._console_handler

    async def submit_query(self, query: HumanQuery) -> HumanResponse:
        """Submit a query and wait for a response.

        Args:
            query: The query to submit

        Returns:
            The human's response

        Raises:
            HumanQueryTimeoutError: If no response within timeout
            InvalidResponseError: If response is invalid for query type
        """
        async with self._lock:
            self.pending[query.id] = query
            self._events[query.id] = asyncio.Event()

        logger.info(f"Submitting human query: {query.id}")

        try:
            # Use response handler to get response
            response = await asyncio.wait_for(
                self._response_handler(query),
                timeout=query.timeout
            )

            # Validate response
            self._validate_response(query, response)

            # Store response
            async with self._lock:
                self.responses[query.id] = response
                del self.pending[query.id]

            logger.info(f"Received response for query {query.id}")
            return response

        except asyncio.TimeoutError:
            async with self._lock:
                if query.id in self.pending:
                    del self.pending[query.id]
            raise HumanQueryTimeoutError(query)

    def _validate_response(self, query: HumanQuery, response: HumanResponse) -> None:
        """Validate a response matches the query requirements."""
        if query.query_type == QueryType.MULTIPLE_CHOICE:
            if query.options:
                valid_responses = [str(i) for i in range(1, len(query.options) + 1)]
                valid_responses.extend([opt.lower() for opt in query.options])

                if response.response.lower() not in valid_responses:
                    raise InvalidResponseError(
                        query, response.response,
                        f"Response must be one of: {query.options}"
                    )

        elif query.query_type == QueryType.CONFIRMATION:
            valid = {"yes", "no", "y", "n", "true", "false", "1", "0"}
            if response.response.lower() not in valid:
                raise InvalidResponseError(
                    query, response.response,
                    "Response must be yes/no"
                )

        elif query.query_type == QueryType.NUMERIC:
            try:
                float(response.response)
            except ValueError:
                raise InvalidResponseError(
                    query, response.response,
                    "Response must be a number"
                )

    async def _console_handler(self, query: HumanQuery) -> HumanResponse:
        """Default console-based response handler."""
        print("\n" + "=" * 60)
        print("HUMAN INPUT REQUIRED")
        print("=" * 60)
        print(query.format_for_display())
        print("-" * 60)

        # Use asyncio to read input without blocking
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, input, "Your response: ")

        return HumanResponse(
            query_id=query.id,
            response=response_text.strip(),
            responder="console_user"
        )

    async def provide_response(
        self,
        query_id: str,
        response: str,
        responder: str = "human"
    ) -> None:
        """Programmatically provide a response to a pending query.

        This is useful for testing or when responses come through
        channels that don't use the response_handler pattern.

        Args:
            query_id: ID of the query to respond to
            response: The response text
            responder: Identifier of the responder
        """
        async with self._lock:
            if query_id not in self.pending:
                raise ValueError(f"No pending query with ID: {query_id}")

            query = self.pending[query_id]
            human_response = HumanResponse(
                query_id=query_id,
                response=response,
                responder=responder
            )

            self._validate_response(query, human_response)
            self.responses[query_id] = human_response
            del self.pending[query_id]

        if query_id in self._events:
            self._events[query_id].set()


class HumanAsTool(Tool):
    """Tool that allows agents to query humans during execution.

    This tool provides a structured way for agents to request human
    input, supporting various query types including free-form questions,
    multiple choice, and confirmations.

    Example:
        human_tool = HumanAsTool()
        result = await human_tool.ask(HumanQuery(
            question="What should be the default timeout?",
            context="We're configuring the API client",
            options=["30 seconds", "60 seconds", "120 seconds"]
        ))
    """

    def __init__(
        self,
        query_manager: Optional[HumanQueryManager] = None,
        default_timeout: int = 300,
        channel: Optional[str] = None
    ):
        """Initialize the HumanAsTool.

        Args:
            query_manager: Optional query manager (creates new if None)
            default_timeout: Default timeout for queries in seconds
            channel: Optional channel for routing queries
        """
        self._query_manager = query_manager or HumanQueryManager()
        self._default_timeout = default_timeout
        self._channel = channel

    @property
    def schema(self) -> ToolSchema:
        """Return the schema for this tool."""
        return ToolSchema(
            name="human_query",
            description=(
                "Ask a human a question and wait for their response. "
                "Use this when you need clarification, confirmation, "
                "or additional information that only a human can provide."
            ),
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to ask the human"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context to help the human answer"
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of choices for multiple choice"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 300)"
                }
            },
            required=["question"],
            permissions=[ToolPermission.NETWORK]  # May route through channels
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the human query tool.

        Args:
            **kwargs: Arguments matching the tool's parameter schema

        Returns:
            ToolResult containing the human's response
        """
        question = kwargs.get("question", "")
        context = kwargs.get("context", "")
        options = kwargs.get("options")
        timeout = kwargs.get("timeout", self._default_timeout)

        query = HumanQuery(
            question=question,
            context=context,
            options=options,
            timeout=timeout,
            metadata={"channel": self._channel} if self._channel else {}
        )

        try:
            response = await self._query_manager.submit_query(query)
            return ToolResult.ok(
                response.response,
                query_id=query.id,
                responder=response.responder,
                confidence=response.confidence
            )
        except HumanQueryTimeoutError as e:
            return ToolResult.fail(
                f"Human query timed out: {e.query.question}",
                query_id=query.id,
                timeout=True
            )
        except InvalidResponseError as e:
            return ToolResult.fail(
                f"Invalid response: {e.reason}",
                query_id=query.id,
                response=e.response
            )

    async def ask(self, query: HumanQuery) -> str:
        """Ask a human and return their response.

        Args:
            query: The query to ask

        Returns:
            The human's response text

        Raises:
            HumanQueryTimeoutError: If no response within timeout
        """
        response = await self._query_manager.submit_query(query)
        return response.response

    async def ask_question(
        self,
        question: str,
        context: str = "",
        timeout: Optional[int] = None
    ) -> str:
        """Ask a free-form question.

        Args:
            question: The question to ask
            context: Optional context
            timeout: Optional timeout override

        Returns:
            The human's response
        """
        query = HumanQuery(
            question=question,
            context=context,
            query_type=QueryType.FREE_FORM,
            timeout=timeout or self._default_timeout
        )
        return await self.ask(query)

    async def ask_multiple_choice(
        self,
        question: str,
        options: List[str],
        context: str = "",
        timeout: Optional[int] = None
    ) -> str:
        """Ask a multiple choice question.

        Args:
            question: The question to ask
            options: List of available options
            context: Optional context
            timeout: Optional timeout override

        Returns:
            The selected option
        """
        query = HumanQuery(
            question=question,
            context=context,
            options=options,
            query_type=QueryType.MULTIPLE_CHOICE,
            timeout=timeout or self._default_timeout
        )
        response = await self.ask(query)

        # Normalize response to option value
        try:
            # Check if response is a number
            index = int(response) - 1
            if 0 <= index < len(options):
                return options[index]
        except ValueError:
            pass

        # Check if response matches an option
        for option in options:
            if option.lower() == response.lower():
                return option

        return response

    async def ask_confirmation(
        self,
        action: str,
        context: str = "",
        timeout: Optional[int] = None
    ) -> bool:
        """Ask for confirmation before an action.

        Args:
            action: Description of the action to confirm
            context: Optional context
            timeout: Optional timeout override

        Returns:
            True if confirmed, False otherwise
        """
        query = HumanQuery(
            question=f"Do you approve this action: {action}?",
            context=context,
            query_type=QueryType.CONFIRMATION,
            timeout=timeout or self._default_timeout
        )
        response = await self.ask(query)
        return response.lower() in {"yes", "y", "true", "1"}

    async def ask_number(
        self,
        question: str,
        context: str = "",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        timeout: Optional[int] = None
    ) -> float:
        """Ask for a numeric value.

        Args:
            question: The question to ask
            context: Optional context
            min_value: Optional minimum value
            max_value: Optional maximum value
            timeout: Optional timeout override

        Returns:
            The numeric value
        """
        constraint_info = ""
        if min_value is not None and max_value is not None:
            constraint_info = f" (between {min_value} and {max_value})"
        elif min_value is not None:
            constraint_info = f" (minimum: {min_value})"
        elif max_value is not None:
            constraint_info = f" (maximum: {max_value})"

        query = HumanQuery(
            question=f"{question}{constraint_info}",
            context=context,
            query_type=QueryType.NUMERIC,
            timeout=timeout or self._default_timeout,
            metadata={"min_value": min_value, "max_value": max_value}
        )

        response = await self.ask(query)
        value = float(response)

        if min_value is not None and value < min_value:
            raise InvalidResponseError(
                query, response, f"Value must be at least {min_value}"
            )
        if max_value is not None and value > max_value:
            raise InvalidResponseError(
                query, response, f"Value must be at most {max_value}"
            )

        return value


# Convenience function for quick queries
async def ask_human(
    question: str,
    context: str = "",
    options: Optional[List[str]] = None,
    timeout: int = 300
) -> str:
    """Quick helper to ask a human a question.

    Args:
        question: The question to ask
        context: Optional context
        options: Optional list of choices
        timeout: Timeout in seconds

    Returns:
        The human's response
    """
    tool = HumanAsTool()
    if options:
        return await tool.ask_multiple_choice(question, options, context, timeout)
    return await tool.ask_question(question, context, timeout)
