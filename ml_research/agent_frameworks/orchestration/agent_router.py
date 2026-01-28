"""
Agent Router for task-based routing.

The AgentRouter routes tasks to appropriate agents based on configurable
rules, patterns, and agent capabilities.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Awaitable, Set

from .base import Task, AgentBase, AgentCapability


logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """
    A rule for routing tasks to agents.

    Attributes:
        pattern: Regex pattern to match against task type/description
        agent_id: ID of the agent to route to
        priority: Higher priority rules are evaluated first
        conditions: Optional callable for additional routing conditions
        match_description: Whether to match against task description (default: type only)
        tags: Optional tags that must be present on the task
    """
    pattern: str
    agent_id: str
    priority: int = 0
    conditions: Optional[Callable[[Task], bool]] = None
    match_description: bool = False
    tags: Set[str] = field(default_factory=set)

    _compiled_pattern: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Compile the regex pattern."""
        self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, task: Task) -> bool:
        """
        Check if this rule matches a task.

        Args:
            task: The task to check

        Returns:
            True if the rule matches
        """
        # Check pattern match
        pattern = self._compiled_pattern or re.compile(self.pattern, re.IGNORECASE)

        # Match against task type
        if pattern.search(task.type):
            match = True
        # Optionally match against description
        elif self.match_description and pattern.search(task.description):
            match = True
        else:
            match = False

        if not match:
            return False

        # Check required tags
        if self.tags and not self.tags.issubset(task.tags):
            return False

        # Check additional conditions
        if self.conditions:
            try:
                if not self.conditions(task):
                    return False
            except Exception as e:
                logger.warning(f"Routing condition raised exception: {e}")
                return False

        return True


@dataclass
class AgentInfo:
    """
    Information about a registered agent for routing purposes.

    Attributes:
        agent: The agent instance
        task_types: List of task types this agent can handle
        capabilities: Set of agent capabilities
        weight: Routing weight for load balancing
        active: Whether the agent is currently active
    """
    agent: AgentBase
    task_types: List[str] = field(default_factory=list)
    capabilities: Set[AgentCapability] = field(default_factory=set)
    weight: float = 1.0
    active: bool = True


class AgentRouter:
    """
    Routes tasks to appropriate agents based on rules and capabilities.

    The router supports:
    - Rule-based routing with regex patterns
    - Capability-based routing
    - Priority ordering
    - Fallback routing
    - Load balancing

    Example:
        router = AgentRouter()

        # Register agents
        router.register(code_agent, tasks=["code_review", "refactor"])
        router.register(test_agent, tasks=["test", "coverage"])

        # Add custom rules
        router.add_rule(RoutingRule(
            pattern=".*urgent.*",
            agent_id="priority_agent",
            priority=100
        ))

        # Route a task
        task = Task(type="code_review", description="Review PR #123")
        agent = await router.route(task)
    """

    def __init__(self):
        """Initialize the router."""
        self.rules: List[RoutingRule] = []
        self.agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

    def register(
        self,
        agent: AgentBase,
        tasks: Optional[List[str]] = None,
        capabilities: Optional[Set[AgentCapability]] = None,
        weight: float = 1.0
    ) -> None:
        """
        Register an agent with optional task type filters.

        Args:
            agent: The agent to register
            tasks: List of task types this agent can handle
            capabilities: Set of capabilities (uses agent's capabilities if not provided)
            weight: Routing weight for load balancing
        """
        self.agents[agent.id] = AgentInfo(
            agent=agent,
            task_types=tasks or [],
            capabilities=capabilities or agent.capabilities,
            weight=weight,
            active=True,
        )
        logger.info(f"Registered agent {agent.id} with router")

    def unregister(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id} from router")

    def add_rule(self, rule: RoutingRule) -> None:
        """
        Add a routing rule.

        Rules are kept sorted by priority (highest first).

        Args:
            rule: The routing rule to add
        """
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule: pattern='{rule.pattern}' -> agent={rule.agent_id}")

    def remove_rule(self, pattern: str, agent_id: Optional[str] = None) -> bool:
        """
        Remove routing rules matching the pattern.

        Args:
            pattern: The pattern to match
            agent_id: Optionally filter by agent ID as well

        Returns:
            True if any rules were removed
        """
        original_count = len(self.rules)
        self.rules = [
            r for r in self.rules
            if r.pattern != pattern or (agent_id and r.agent_id != agent_id)
        ]
        removed = original_count - len(self.rules)
        if removed:
            logger.info(f"Removed {removed} routing rule(s) with pattern '{pattern}'")
        return removed > 0

    def clear_rules(self) -> None:
        """Remove all routing rules."""
        self.rules.clear()
        logger.info("Cleared all routing rules")

    def set_agent_active(self, agent_id: str, active: bool) -> None:
        """
        Set whether an agent is active for routing.

        Args:
            agent_id: ID of the agent
            active: Whether the agent should receive tasks
        """
        if agent_id in self.agents:
            self.agents[agent_id].active = active
            logger.info(f"Set agent {agent_id} active={active}")

    async def route(self, task: Task) -> AgentBase:
        """
        Find the best agent for a task.

        Args:
            task: The task to route

        Returns:
            The selected agent

        Raises:
            ValueError: If no suitable agent is found
        """
        agent = await self._find_agent(task)
        if agent is None:
            raise ValueError(f"No suitable agent found for task type: {task.type}")
        return agent

    async def route_with_fallback(
        self,
        task: Task,
        fallback: AgentBase
    ) -> AgentBase:
        """
        Find the best agent for a task, falling back to a default.

        Args:
            task: The task to route
            fallback: Agent to use if no better match is found

        Returns:
            The selected agent or fallback
        """
        agent = await self._find_agent(task)
        if agent is None:
            logger.info(f"No agent found for task {task.id}, using fallback")
            return fallback
        return agent

    async def route_all(self, task: Task) -> List[AgentBase]:
        """
        Find all agents that can handle a task.

        Useful for broadcasting or redundant execution.

        Args:
            task: The task to route

        Returns:
            List of agents that can handle the task
        """
        candidates = []

        # Check rules first
        rule_agent_ids = set()
        for rule in self.rules:
            if rule.matches(task):
                if rule.agent_id in self.agents:
                    info = self.agents[rule.agent_id]
                    if info.active:
                        candidates.append(info.agent)
                        rule_agent_ids.add(rule.agent_id)

        # Check agents by task type and capability
        for agent_id, info in self.agents.items():
            if not info.active or agent_id in rule_agent_ids:
                continue

            if self._agent_matches_task(info, task):
                candidates.append(info.agent)

        return candidates

    async def _find_agent(self, task: Task) -> Optional[AgentBase]:
        """
        Internal method to find an agent for a task.

        Args:
            task: The task to route

        Returns:
            Selected agent or None
        """
        # Check rules first (already sorted by priority)
        for rule in self.rules:
            if rule.matches(task):
                if rule.agent_id in self.agents:
                    info = self.agents[rule.agent_id]
                    if info.active:
                        logger.info(
                            f"Routing task {task.id} to {rule.agent_id} via rule"
                        )
                        return info.agent
                else:
                    logger.warning(
                        f"Rule references unknown agent {rule.agent_id}"
                    )

        # Fall back to task type / capability matching
        candidates = []
        for agent_id, info in self.agents.items():
            if not info.active:
                continue

            if self._agent_matches_task(info, task):
                candidates.append((info.weight, info.agent))

        if not candidates:
            return None

        # Weight-based selection (for load balancing)
        # For now, just select highest weight
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]

        logger.info(f"Routing task {task.id} to {selected.id} via capability match")
        return selected

    def _agent_matches_task(self, info: AgentInfo, task: Task) -> bool:
        """
        Check if an agent can handle a task.

        Args:
            info: Agent info
            task: The task to check

        Returns:
            True if the agent can handle the task
        """
        # Check explicit task types
        if info.task_types:
            task_type_lower = task.type.lower()
            for registered_type in info.task_types:
                if registered_type.lower() in task_type_lower or task_type_lower in registered_type.lower():
                    return True

        # Check capabilities
        task_type_lower = task.type.lower()
        for cap in info.capabilities:
            if cap.value.lower() in task_type_lower or task_type_lower in cap.value.lower():
                return True

        # Check agent's own can_handle method
        return info.agent.can_handle(task)

    def get_agents(self) -> List[AgentBase]:
        """Get all registered agents."""
        return [info.agent for info in self.agents.values()]

    def get_active_agents(self) -> List[AgentBase]:
        """Get all active agents."""
        return [info.agent for info in self.agents.values() if info.active]

    def get_rules(self) -> List[RoutingRule]:
        """Get all routing rules."""
        return self.rules.copy()


class ConditionalRouter:
    """
    A router that supports async conditions and complex routing logic.

    This extends AgentRouter with support for async condition evaluation
    and more sophisticated routing strategies.
    """

    def __init__(self, base_router: Optional[AgentRouter] = None):
        """
        Initialize the conditional router.

        Args:
            base_router: Base router to use (creates new if not provided)
        """
        self.base_router = base_router or AgentRouter()
        self.async_conditions: List[tuple[
            Callable[[Task], Awaitable[bool]],
            str,  # agent_id
            int,  # priority
        ]] = []

    def add_async_condition(
        self,
        condition: Callable[[Task], Awaitable[bool]],
        agent_id: str,
        priority: int = 0
    ) -> None:
        """
        Add an async routing condition.

        Args:
            condition: Async callable that returns True if agent should handle task
            agent_id: ID of the agent
            priority: Priority for this condition
        """
        self.async_conditions.append((condition, agent_id, priority))
        self.async_conditions.sort(key=lambda x: x[2], reverse=True)

    async def route(self, task: Task) -> AgentBase:
        """
        Route a task with async condition support.

        Args:
            task: The task to route

        Returns:
            The selected agent

        Raises:
            ValueError: If no suitable agent is found
        """
        # Check async conditions first
        for condition, agent_id, _ in self.async_conditions:
            try:
                if await condition(task):
                    if agent_id in self.base_router.agents:
                        info = self.base_router.agents[agent_id]
                        if info.active:
                            return info.agent
            except Exception as e:
                logger.warning(f"Async condition raised exception: {e}")

        # Fall back to base router
        return await self.base_router.route(task)
