"""
Agent state management using finite state machines.

This module provides a flexible state machine implementation for managing
agent execution states, with support for transitions, guards, and hooks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Set, Callable, Awaitable,
    Any, TypeVar, Generic, Union
)
import asyncio


class AgentState(Enum):
    """Standard states for agent execution lifecycle."""
    IDLE = auto()              # Agent is not doing anything
    PLANNING = auto()          # Agent is creating a plan
    EXECUTING = auto()         # Agent is executing a plan
    WAITING_APPROVAL = auto()  # Agent is waiting for human approval
    COMPLETED = auto()         # Agent has completed successfully
    FAILED = auto()            # Agent has failed
    PAUSED = auto()            # Agent is paused (can be resumed)
    CANCELLED = auto()         # Agent execution was cancelled

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED)

    @property
    def is_active(self) -> bool:
        """Check if this is an active (running) state."""
        return self in (AgentState.PLANNING, AgentState.EXECUTING)

    @property
    def can_be_cancelled(self) -> bool:
        """Check if the agent can be cancelled from this state."""
        return not self.is_terminal


@dataclass
class StateTransition:
    """Record of a state transition.

    Attributes:
        from_state: The state before transition
        to_state: The state after transition
        trigger: What triggered the transition
        timestamp: When the transition occurred
        metadata: Additional transition data
    """
    from_state: AgentState
    to_state: AgentState
    trigger: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "from_state": self.from_state.name,
            "to_state": self.to_state.name,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


# Type for state transition callbacks
StateCallback = Callable[[AgentState, AgentState, str], Awaitable[None]]
# Type for transition guards
TransitionGuard = Callable[[AgentState, AgentState], Awaitable[bool]]


@dataclass
class TransitionRule:
    """Defines a valid state transition.

    Attributes:
        from_state: The source state (or None for any state)
        to_state: The destination state
        triggers: List of valid trigger names for this transition
        guard: Optional async function that must return True to allow transition
    """
    from_state: Optional[AgentState]
    to_state: AgentState
    triggers: List[str] = field(default_factory=list)
    guard: Optional[TransitionGuard] = None

    def matches(self, from_state: AgentState, trigger: str) -> bool:
        """Check if this rule matches the given transition."""
        if self.from_state is not None and self.from_state != from_state:
            return False
        if self.triggers and trigger not in self.triggers:
            return False
        return True


class StateMachine:
    """Finite state machine for managing agent execution states.

    This class provides a flexible state machine with support for:
    - Configurable valid transitions
    - Transition guards (conditions)
    - On-enter and on-exit hooks
    - Transition history tracking
    - Async callbacks

    Usage:
        sm = StateMachine(initial_state=AgentState.IDLE)
        sm.add_transition(AgentState.IDLE, AgentState.PLANNING, ["start"])
        sm.on_enter(AgentState.PLANNING, my_callback)
        await sm.transition(AgentState.PLANNING, trigger="start")
    """

    def __init__(
        self,
        initial_state: AgentState = AgentState.IDLE,
        track_history: bool = True,
        max_history: Optional[int] = 1000
    ):
        """Initialize the state machine.

        Args:
            initial_state: The starting state
            track_history: Whether to track transition history
            max_history: Maximum history entries to keep (None for unlimited)
        """
        self._current_state: AgentState = initial_state
        self._track_history: bool = track_history
        self._max_history: Optional[int] = max_history
        self._history: List[StateTransition] = []
        self._transitions: List[TransitionRule] = []
        self._on_enter_callbacks: Dict[AgentState, List[StateCallback]] = {}
        self._on_exit_callbacks: Dict[AgentState, List[StateCallback]] = {}
        self._global_callbacks: List[StateCallback] = []
        self._lock: asyncio.Lock = asyncio.Lock()

        # Set up default transitions
        self._setup_default_transitions()

    def _setup_default_transitions(self) -> None:
        """Set up standard agent state transitions."""
        # From IDLE
        self.add_transition(AgentState.IDLE, AgentState.PLANNING, ["start", "plan"])
        self.add_transition(AgentState.IDLE, AgentState.CANCELLED, ["cancel"])

        # From PLANNING
        self.add_transition(AgentState.PLANNING, AgentState.EXECUTING, ["execute", "run"])
        self.add_transition(AgentState.PLANNING, AgentState.WAITING_APPROVAL, ["request_approval"])
        self.add_transition(AgentState.PLANNING, AgentState.FAILED, ["fail", "error"])
        self.add_transition(AgentState.PLANNING, AgentState.CANCELLED, ["cancel"])

        # From EXECUTING
        self.add_transition(AgentState.EXECUTING, AgentState.COMPLETED, ["complete", "success"])
        self.add_transition(AgentState.EXECUTING, AgentState.FAILED, ["fail", "error"])
        self.add_transition(AgentState.EXECUTING, AgentState.WAITING_APPROVAL, ["request_approval"])
        self.add_transition(AgentState.EXECUTING, AgentState.PAUSED, ["pause"])
        self.add_transition(AgentState.EXECUTING, AgentState.CANCELLED, ["cancel"])

        # From WAITING_APPROVAL
        self.add_transition(AgentState.WAITING_APPROVAL, AgentState.EXECUTING, ["approve", "continue"])
        self.add_transition(AgentState.WAITING_APPROVAL, AgentState.FAILED, ["deny", "reject"])
        self.add_transition(AgentState.WAITING_APPROVAL, AgentState.CANCELLED, ["cancel", "timeout"])

        # From PAUSED
        self.add_transition(AgentState.PAUSED, AgentState.EXECUTING, ["resume"])
        self.add_transition(AgentState.PAUSED, AgentState.CANCELLED, ["cancel"])

    @property
    def current_state(self) -> AgentState:
        """Get the current state."""
        return self._current_state

    @property
    def history(self) -> List[StateTransition]:
        """Get the transition history."""
        return self._history.copy()

    @property
    def is_terminal(self) -> bool:
        """Check if the current state is terminal."""
        return self._current_state.is_terminal

    @property
    def is_active(self) -> bool:
        """Check if the current state is active."""
        return self._current_state.is_active

    def add_transition(
        self,
        from_state: Optional[AgentState],
        to_state: AgentState,
        triggers: Optional[List[str]] = None,
        guard: Optional[TransitionGuard] = None
    ) -> "StateMachine":
        """Add a valid transition rule.

        Args:
            from_state: Source state (None for any state)
            to_state: Destination state
            triggers: List of trigger names (None for any trigger)
            guard: Optional guard function

        Returns:
            self for method chaining
        """
        rule = TransitionRule(
            from_state=from_state,
            to_state=to_state,
            triggers=triggers or [],
            guard=guard
        )
        self._transitions.append(rule)
        return self

    def on_enter(
        self,
        state: AgentState,
        callback: StateCallback
    ) -> "StateMachine":
        """Register a callback to run when entering a state.

        Args:
            state: The state to watch
            callback: Async function to call

        Returns:
            self for method chaining
        """
        if state not in self._on_enter_callbacks:
            self._on_enter_callbacks[state] = []
        self._on_enter_callbacks[state].append(callback)
        return self

    def on_exit(
        self,
        state: AgentState,
        callback: StateCallback
    ) -> "StateMachine":
        """Register a callback to run when exiting a state.

        Args:
            state: The state to watch
            callback: Async function to call

        Returns:
            self for method chaining
        """
        if state not in self._on_exit_callbacks:
            self._on_exit_callbacks[state] = []
        self._on_exit_callbacks[state].append(callback)
        return self

    def on_transition(self, callback: StateCallback) -> "StateMachine":
        """Register a callback to run on any state transition.

        Args:
            callback: Async function to call

        Returns:
            self for method chaining
        """
        self._global_callbacks.append(callback)
        return self

    def can_transition(
        self,
        to_state: AgentState,
        trigger: str = ""
    ) -> bool:
        """Check if a transition is valid (synchronously, without guards).

        Args:
            to_state: The target state
            trigger: The trigger name

        Returns:
            True if the transition is structurally valid
        """
        for rule in self._transitions:
            if rule.matches(self._current_state, trigger) and rule.to_state == to_state:
                return True
        return False

    async def can_transition_async(
        self,
        to_state: AgentState,
        trigger: str = ""
    ) -> bool:
        """Check if a transition is valid (with guard evaluation).

        Args:
            to_state: The target state
            trigger: The trigger name

        Returns:
            True if the transition is valid and guards pass
        """
        for rule in self._transitions:
            if rule.matches(self._current_state, trigger) and rule.to_state == to_state:
                if rule.guard is not None:
                    try:
                        if not await rule.guard(self._current_state, to_state):
                            continue
                    except Exception:
                        continue
                return True
        return False

    def get_valid_transitions(self) -> Set[AgentState]:
        """Get all states that can be transitioned to from current state.

        Returns:
            Set of valid destination states
        """
        valid = set()
        for rule in self._transitions:
            if rule.from_state is None or rule.from_state == self._current_state:
                valid.add(rule.to_state)
        return valid

    async def transition(
        self,
        to_state: AgentState,
        trigger: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateTransition:
        """Perform a state transition.

        Args:
            to_state: The target state
            trigger: The trigger name
            metadata: Additional transition metadata

        Returns:
            The StateTransition record

        Raises:
            ValueError: If the transition is not valid
        """
        async with self._lock:
            # Find matching rule
            matching_rule: Optional[TransitionRule] = None
            for rule in self._transitions:
                if rule.matches(self._current_state, trigger) and rule.to_state == to_state:
                    if rule.guard is not None:
                        try:
                            if not await rule.guard(self._current_state, to_state):
                                continue
                        except Exception:
                            continue
                    matching_rule = rule
                    break

            if matching_rule is None:
                raise ValueError(
                    f"Invalid transition from {self._current_state} to {to_state} "
                    f"with trigger '{trigger}'"
                )

            from_state = self._current_state

            # Run exit callbacks
            if from_state in self._on_exit_callbacks:
                for callback in self._on_exit_callbacks[from_state]:
                    try:
                        await callback(from_state, to_state, trigger)
                    except Exception:
                        pass  # Don't fail transition on callback errors

            # Update state
            self._current_state = to_state

            # Create transition record
            transition = StateTransition(
                from_state=from_state,
                to_state=to_state,
                trigger=trigger,
                metadata=metadata or {}
            )

            # Track history
            if self._track_history:
                self._history.append(transition)
                if self._max_history and len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

            # Run enter callbacks
            if to_state in self._on_enter_callbacks:
                for callback in self._on_enter_callbacks[to_state]:
                    try:
                        await callback(from_state, to_state, trigger)
                    except Exception:
                        pass

            # Run global callbacks
            for callback in self._global_callbacks:
                try:
                    await callback(from_state, to_state, trigger)
                except Exception:
                    pass

            return transition

    async def trigger(
        self,
        trigger_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[StateTransition]:
        """Trigger a transition by name.

        Finds the first valid transition for the given trigger from the current
        state and performs it.

        Args:
            trigger_name: The trigger name
            metadata: Additional transition metadata

        Returns:
            The StateTransition if successful, None if no valid transition found
        """
        for rule in self._transitions:
            if rule.matches(self._current_state, trigger_name):
                if rule.guard is not None:
                    try:
                        if not await rule.guard(self._current_state, rule.to_state):
                            continue
                    except Exception:
                        continue
                return await self.transition(
                    to_state=rule.to_state,
                    trigger=trigger_name,
                    metadata=metadata
                )
        return None

    def reset(self, state: AgentState = AgentState.IDLE) -> None:
        """Reset the state machine.

        Args:
            state: The state to reset to
        """
        self._current_state = state
        self._history.clear()

    def get_time_in_state(self) -> Optional[float]:
        """Get the time spent in the current state in seconds.

        Returns:
            Time in seconds, or None if no history
        """
        if not self._history:
            return None

        last_transition = self._history[-1]
        return (datetime.now() - last_transition.timestamp).total_seconds()

    def __repr__(self) -> str:
        return f"<StateMachine state={self._current_state.name} history_len={len(self._history)}>"


class StateMachineBuilder:
    """Builder pattern for creating configured state machines.

    Usage:
        sm = (StateMachineBuilder()
            .initial_state(AgentState.IDLE)
            .transition(AgentState.IDLE, AgentState.PLANNING)
            .on_enter(AgentState.PLANNING, my_callback)
            .build())
    """

    def __init__(self):
        """Initialize the builder."""
        self._initial_state: AgentState = AgentState.IDLE
        self._track_history: bool = True
        self._max_history: Optional[int] = 1000
        self._transitions: List[TransitionRule] = []
        self._on_enter: Dict[AgentState, List[StateCallback]] = {}
        self._on_exit: Dict[AgentState, List[StateCallback]] = {}
        self._on_transition: List[StateCallback] = []
        self._use_defaults: bool = True

    def initial_state(self, state: AgentState) -> "StateMachineBuilder":
        """Set the initial state."""
        self._initial_state = state
        return self

    def track_history(self, enabled: bool = True, max_entries: Optional[int] = 1000) -> "StateMachineBuilder":
        """Configure history tracking."""
        self._track_history = enabled
        self._max_history = max_entries
        return self

    def no_defaults(self) -> "StateMachineBuilder":
        """Don't set up default transitions."""
        self._use_defaults = False
        return self

    def transition(
        self,
        from_state: Optional[AgentState],
        to_state: AgentState,
        triggers: Optional[List[str]] = None,
        guard: Optional[TransitionGuard] = None
    ) -> "StateMachineBuilder":
        """Add a transition rule."""
        self._transitions.append(TransitionRule(
            from_state=from_state,
            to_state=to_state,
            triggers=triggers or [],
            guard=guard
        ))
        return self

    def on_enter(self, state: AgentState, callback: StateCallback) -> "StateMachineBuilder":
        """Add an on-enter callback."""
        if state not in self._on_enter:
            self._on_enter[state] = []
        self._on_enter[state].append(callback)
        return self

    def on_exit(self, state: AgentState, callback: StateCallback) -> "StateMachineBuilder":
        """Add an on-exit callback."""
        if state not in self._on_exit:
            self._on_exit[state] = []
        self._on_exit[state].append(callback)
        return self

    def on_any_transition(self, callback: StateCallback) -> "StateMachineBuilder":
        """Add a global transition callback."""
        self._on_transition.append(callback)
        return self

    def build(self) -> StateMachine:
        """Build the configured state machine."""
        sm = StateMachine(
            initial_state=self._initial_state,
            track_history=self._track_history,
            max_history=self._max_history
        )

        if not self._use_defaults:
            sm._transitions.clear()

        # Add custom transitions
        for rule in self._transitions:
            sm.add_transition(
                from_state=rule.from_state,
                to_state=rule.to_state,
                triggers=rule.triggers,
                guard=rule.guard
            )

        # Add callbacks
        for state, callbacks in self._on_enter.items():
            for callback in callbacks:
                sm.on_enter(state, callback)

        for state, callbacks in self._on_exit.items():
            for callback in callbacks:
                sm.on_exit(state, callback)

        for callback in self._on_transition:
            sm.on_transition(callback)

        return sm
