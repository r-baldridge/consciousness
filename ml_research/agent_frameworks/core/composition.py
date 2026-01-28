"""
Agent composition operators for building complex agent workflows.

This module provides composable building blocks for creating agent pipelines,
including sequential execution, parallel execution, approval gates, and retry logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    List, Optional, Dict, Any, Union, Callable,
    Awaitable, TypeVar, Generic, Tuple
)
import asyncio
import uuid

from .base_agent import AgentBase, Task, TaskResult, Plan, ExecutionResult
from .message_types import ApprovalStatus


class ComposableAgent(AgentBase):
    """Base class for composable agent wrappers.

    This provides the foundation for pipeline components that wrap
    or combine other agents.
    """

    AGENT_ID: str = "composable"

    def __init__(self, agents: List[AgentBase]):
        """Initialize with a list of agents to compose.

        Args:
            agents: List of agents to compose
        """
        super().__init__()
        self._agents = agents

    @property
    def agents(self) -> List[AgentBase]:
        """Get the composed agents."""
        return self._agents

    async def plan(self, task: Task) -> Plan:
        """Composable agents delegate planning to their components."""
        raise NotImplementedError("Composable agents don't create their own plans")

    async def execute(self, plan: Plan) -> List[ExecutionResult]:
        """Composable agents delegate execution to their components."""
        raise NotImplementedError("Composable agents don't execute plans directly")


class SequentialPipeline(ComposableAgent):
    """Run agents in sequence, feeding output of one into the next.

    The output of each agent is transformed into a Task for the next agent.
    The final output is the result of the last agent.

    Usage:
        pipeline = agent1 >> agent2 >> agent3
        result = await pipeline.run("initial task")

        # Or explicitly:
        pipeline = SequentialPipeline([agent1, agent2, agent3])
    """

    AGENT_ID: str = "sequential_pipeline"

    def __init__(
        self,
        agents: List[AgentBase],
        output_transformer: Optional[Callable[[TaskResult], Task]] = None
    ):
        """Initialize the sequential pipeline.

        Args:
            agents: List of agents to run in sequence
            output_transformer: Optional function to transform agent output to next task
        """
        super().__init__(agents)
        self._output_transformer = output_transformer or self._default_transformer

    @staticmethod
    def _default_transformer(result: TaskResult) -> Task:
        """Default transformer: use output as next task description."""
        if result.success and result.output:
            if isinstance(result.output, str):
                return Task.create(result.output)
            else:
                return Task.create(str(result.output))
        else:
            return Task.create(f"Continue after: {result.error or 'unknown'}")

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute agents sequentially.

        Args:
            task: The initial task

        Returns:
            The result of the final agent
        """
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()
        current_task = task
        all_results: List[ExecutionResult] = []

        for i, agent in enumerate(self._agents):
            result = await agent.run(current_task)

            # Collect execution results
            all_results.extend(result.execution_results)

            if not result.success:
                # Pipeline fails if any agent fails
                duration = int((datetime.now() - start_time).total_seconds() * 1000)
                return TaskResult.error_result(
                    task_id=task.id,
                    error=f"Pipeline failed at agent {i} ({agent.AGENT_ID}): {result.error}",
                    execution_results=all_results,
                    duration_ms=duration
                )

            # Transform output for next agent (if not last)
            if i < len(self._agents) - 1:
                current_task = self._output_transformer(result)

        duration = int((datetime.now() - start_time).total_seconds() * 1000)
        return TaskResult.success_result(
            task_id=task.id,
            output=result.output,  # Output of last agent
            execution_results=all_results,
            duration_ms=duration,
            metadata={"pipeline_length": len(self._agents)}
        )

    def __rshift__(self, other: AgentBase) -> "SequentialPipeline":
        """Add another agent to the sequence: pipeline >> agent."""
        if isinstance(other, SequentialPipeline):
            return SequentialPipeline(
                self._agents + other._agents,
                self._output_transformer
            )
        return SequentialPipeline(
            self._agents + [other],
            self._output_transformer
        )

    def __repr__(self) -> str:
        agent_ids = " >> ".join(a.AGENT_ID for a in self._agents)
        return f"<SequentialPipeline [{agent_ids}]>"


class ParallelPipeline(ComposableAgent):
    """Run agents in parallel and collect all results.

    All agents receive the same input task and run concurrently.
    Results are collected into a list.

    Usage:
        pipeline = agent1 | agent2 | agent3
        result = await pipeline.run("task for all agents")

        # Or explicitly:
        pipeline = ParallelPipeline([agent1, agent2, agent3])
    """

    AGENT_ID: str = "parallel_pipeline"

    def __init__(
        self,
        agents: List[AgentBase],
        fail_fast: bool = False,
        result_aggregator: Optional[Callable[[List[TaskResult]], Any]] = None
    ):
        """Initialize the parallel pipeline.

        Args:
            agents: List of agents to run in parallel
            fail_fast: If True, cancel remaining agents when one fails
            result_aggregator: Optional function to aggregate results
        """
        super().__init__(agents)
        self._fail_fast = fail_fast
        self._result_aggregator = result_aggregator

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute agents in parallel.

        Args:
            task: The task to run (sent to all agents)

        Returns:
            Aggregated result from all agents
        """
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()

        # Create tasks for all agents
        async def run_agent(agent: AgentBase) -> TaskResult:
            return await agent.run(task)

        if self._fail_fast:
            # Use gather with return_exceptions=False to fail fast
            try:
                results = await asyncio.gather(
                    *[run_agent(agent) for agent in self._agents],
                    return_exceptions=False
                )
            except Exception as e:
                duration = int((datetime.now() - start_time).total_seconds() * 1000)
                return TaskResult.error_result(
                    task_id=task.id,
                    error=f"Parallel execution failed: {str(e)}",
                    duration_ms=duration
                )
        else:
            # Run all agents, collect results even if some fail
            results = await asyncio.gather(
                *[run_agent(agent) for agent in self._agents],
                return_exceptions=True
            )
            # Convert exceptions to error results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(TaskResult.error_result(
                        task_id=task.id,
                        error=f"Agent {self._agents[i].AGENT_ID}: {str(result)}"
                    ))
                else:
                    processed_results.append(result)
            results = processed_results

        duration = int((datetime.now() - start_time).total_seconds() * 1000)

        # Aggregate results
        all_execution_results = []
        for result in results:
            all_execution_results.extend(result.execution_results)

        all_succeeded = all(r.success for r in results)

        if self._result_aggregator:
            aggregated_output = self._result_aggregator(results)
        else:
            aggregated_output = [r.output for r in results]

        if all_succeeded:
            return TaskResult.success_result(
                task_id=task.id,
                output=aggregated_output,
                execution_results=all_execution_results,
                duration_ms=duration,
                metadata={"parallel_count": len(self._agents)}
            )
        else:
            errors = [r.error for r in results if not r.success and r.error]
            return TaskResult(
                task_id=task.id,
                success=False,
                output=aggregated_output,
                execution_results=all_execution_results,
                error="; ".join(errors),
                duration_ms=duration,
                metadata={"parallel_count": len(self._agents), "partial_success": True}
            )

    def __or__(self, other: AgentBase) -> "ParallelPipeline":
        """Add another agent to run in parallel: pipeline | agent."""
        if isinstance(other, ParallelPipeline):
            return ParallelPipeline(
                self._agents + other._agents,
                self._fail_fast,
                self._result_aggregator
            )
        return ParallelPipeline(
            self._agents + [other],
            self._fail_fast,
            self._result_aggregator
        )

    def __repr__(self) -> str:
        agent_ids = " | ".join(a.AGENT_ID for a in self._agents)
        return f"<ParallelPipeline [{agent_ids}]>"


@dataclass
class ApprovalRequest:
    """A request for human approval.

    Attributes:
        id: Unique identifier for this request
        message: Description of what needs approval
        context: Additional context for the approver
        input_task: The task being processed
        intermediate_result: Result so far (if any)
        created_at: When the request was created
    """
    id: str
    message: str
    context: Dict[str, Any]
    input_task: Task
    intermediate_result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, message: str, task: Task, **kwargs) -> "ApprovalRequest":
        """Factory method to create an ApprovalRequest."""
        return cls(
            id=str(uuid.uuid4()),
            message=message,
            context=kwargs.get("context", {}),
            input_task=task,
            intermediate_result=kwargs.get("intermediate_result")
        )


class ApprovalGate(ComposableAgent):
    """Insert a human approval checkpoint between agents.

    This wrapper pauses execution and requests human approval before
    continuing to the next stage.

    Usage:
        pipeline = agent1 >> ApprovalGate() >> agent2
        # Execution will pause after agent1 and request approval
    """

    AGENT_ID: str = "approval_gate"
    REQUIRES_APPROVAL: bool = True

    def __init__(
        self,
        message: str = "Approval required to continue",
        timeout: Optional[float] = None,
        on_deny: Optional[Callable[[ApprovalRequest], Awaitable[TaskResult]]] = None
    ):
        """Initialize the approval gate.

        Args:
            message: Message to show when requesting approval
            timeout: Optional timeout in seconds
            on_deny: Optional handler for denied approvals
        """
        super().__init__([])
        self._message = message
        self._timeout = timeout
        self._on_deny = on_deny

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Request approval and either continue or fail.

        Args:
            task: The task (typically output from previous agent)

        Returns:
            The input task result if approved, error result if denied
        """
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()

        # Create approval request
        request = ApprovalRequest.create(
            message=self._message,
            task=task,
            context={"task_description": task.description}
        )

        # Request approval
        status = await self.request_approval(
            message=self._message,
            context=request.context,
            timeout=self._timeout
        )

        duration = int((datetime.now() - start_time).total_seconds() * 1000)

        if status == ApprovalStatus.APPROVED:
            return TaskResult.success_result(
                task_id=task.id,
                output=task.description,  # Pass through
                duration_ms=duration,
                metadata={"approval_status": "approved"}
            )
        elif status == ApprovalStatus.DENIED:
            if self._on_deny:
                return await self._on_deny(request)
            return TaskResult.error_result(
                task_id=task.id,
                error="Approval denied",
                duration_ms=duration,
                metadata={"approval_status": "denied"}
            )
        elif status == ApprovalStatus.TIMEOUT:
            return TaskResult.error_result(
                task_id=task.id,
                error="Approval timed out",
                duration_ms=duration,
                metadata={"approval_status": "timeout"}
            )
        else:
            return TaskResult.error_result(
                task_id=task.id,
                error=f"Unknown approval status: {status}",
                duration_ms=duration
            )

    def __repr__(self) -> str:
        return f"<ApprovalGate message='{self._message[:30]}...'>"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        retry_on: List of error patterns to retry on (None for all errors)
    """
    max_attempts: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retry_on: Optional[List[str]] = None


class Retry(ComposableAgent):
    """Wrap an agent with retry logic.

    Automatically retries failed executions with configurable
    backoff and retry conditions.

    Usage:
        robust_agent = Retry(my_agent, max_attempts=3)
        result = await robust_agent.run("task")

        # With custom config:
        config = RetryConfig(max_attempts=5, delay=2.0)
        robust_agent = Retry(my_agent, config=config)
    """

    AGENT_ID: str = "retry"

    def __init__(
        self,
        agent: AgentBase,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        config: Optional[RetryConfig] = None
    ):
        """Initialize the retry wrapper.

        Args:
            agent: The agent to wrap
            max_attempts: Maximum retry attempts (used if config not provided)
            delay: Initial delay in seconds (used if config not provided)
            backoff_factor: Backoff multiplier (used if config not provided)
            config: Optional full RetryConfig
        """
        super().__init__([agent])
        self._agent = agent

        if config:
            self._config = config
        else:
            self._config = RetryConfig(
                max_attempts=max_attempts,
                delay=delay,
                backoff_factor=backoff_factor
            )

    def _should_retry(self, result: TaskResult, attempt: int) -> bool:
        """Determine if we should retry based on the result."""
        if attempt >= self._config.max_attempts:
            return False

        if result.success:
            return False

        if self._config.retry_on is None:
            return True

        error = result.error or ""
        return any(pattern in error for pattern in self._config.retry_on)

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Execute with retry logic.

        Args:
            task: The task to execute

        Returns:
            The result (successful or final failure)
        """
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()
        all_execution_results: List[ExecutionResult] = []
        current_delay = self._config.delay
        last_result: Optional[TaskResult] = None

        for attempt in range(1, self._config.max_attempts + 1):
            result = await self._agent.run(task)
            last_result = result
            all_execution_results.extend(result.execution_results)

            if result.success:
                duration = int((datetime.now() - start_time).total_seconds() * 1000)
                result.metadata["retry_attempts"] = attempt
                result.duration_ms = duration
                result.execution_results = all_execution_results
                return result

            if not self._should_retry(result, attempt):
                break

            # Wait before retry
            await asyncio.sleep(current_delay)
            current_delay = min(
                current_delay * self._config.backoff_factor,
                self._config.max_delay
            )

        # All attempts failed
        duration = int((datetime.now() - start_time).total_seconds() * 1000)
        return TaskResult.error_result(
            task_id=task.id,
            error=f"Failed after {self._config.max_attempts} attempts: {last_result.error if last_result else 'unknown'}",
            execution_results=all_execution_results,
            duration_ms=duration,
            metadata={"retry_attempts": self._config.max_attempts}
        )

    def __repr__(self) -> str:
        return f"<Retry agent={self._agent.AGENT_ID} max_attempts={self._config.max_attempts}>"


class ConditionalPipeline(ComposableAgent):
    """Route to different agents based on a condition.

    Usage:
        pipeline = ConditionalPipeline(
            condition=lambda task: "code" in task.description,
            if_true=code_agent,
            if_false=text_agent
        )
    """

    AGENT_ID: str = "conditional"

    def __init__(
        self,
        condition: Callable[[Task], Union[bool, Awaitable[bool]]],
        if_true: AgentBase,
        if_false: AgentBase
    ):
        """Initialize the conditional pipeline.

        Args:
            condition: Function that determines which branch to take
            if_true: Agent to run if condition is True
            if_false: Agent to run if condition is False
        """
        super().__init__([if_true, if_false])
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Evaluate condition and run appropriate agent.

        Args:
            task: The task to execute

        Returns:
            Result from the selected agent
        """
        if isinstance(task, str):
            task = Task.create(task)

        # Evaluate condition
        result = self._condition(task)
        if asyncio.iscoroutine(result):
            condition_result = await result
        else:
            condition_result = result

        # Run appropriate agent
        if condition_result:
            return await self._if_true.run(task)
        else:
            return await self._if_false.run(task)

    def __repr__(self) -> str:
        return f"<ConditionalPipeline if_true={self._if_true.AGENT_ID} if_false={self._if_false.AGENT_ID}>"


class FanOut(ComposableAgent):
    """Split a task into multiple sub-tasks and run agents in parallel.

    Usage:
        fan_out = FanOut(
            splitter=lambda task: [Task.create(f"Part {i}") for i in range(3)],
            agents=[agent1, agent2, agent3]
        )
    """

    AGENT_ID: str = "fan_out"

    def __init__(
        self,
        splitter: Callable[[Task], Union[List[Task], Awaitable[List[Task]]]],
        agents: List[AgentBase],
        aggregator: Optional[Callable[[List[TaskResult]], Any]] = None
    ):
        """Initialize the fan-out pipeline.

        Args:
            splitter: Function that splits a task into multiple tasks
            agents: List of agents (one per split task, or reused if fewer)
            aggregator: Optional function to aggregate results
        """
        super().__init__(agents)
        self._splitter = splitter
        self._aggregator = aggregator

    async def run(self, task: Union[str, Task]) -> TaskResult:
        """Split task and run agents in parallel.

        Args:
            task: The task to split and execute

        Returns:
            Aggregated results from all sub-tasks
        """
        if isinstance(task, str):
            task = Task.create(task)

        start_time = datetime.now()

        # Split the task
        split_result = self._splitter(task)
        if asyncio.iscoroutine(split_result):
            sub_tasks = await split_result
        else:
            sub_tasks = split_result

        # Match agents to tasks (cycle if needed)
        async def run_sub_task(idx: int, sub_task: Task) -> TaskResult:
            agent = self._agents[idx % len(self._agents)]
            return await agent.run(sub_task)

        # Run in parallel
        results = await asyncio.gather(
            *[run_sub_task(i, t) for i, t in enumerate(sub_tasks)],
            return_exceptions=True
        )

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult.error_result(
                    task_id=sub_tasks[i].id,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        duration = int((datetime.now() - start_time).total_seconds() * 1000)

        all_execution_results = []
        for result in processed_results:
            all_execution_results.extend(result.execution_results)

        all_succeeded = all(r.success for r in processed_results)

        if self._aggregator:
            aggregated_output = self._aggregator(processed_results)
        else:
            aggregated_output = [r.output for r in processed_results]

        if all_succeeded:
            return TaskResult.success_result(
                task_id=task.id,
                output=aggregated_output,
                execution_results=all_execution_results,
                duration_ms=duration,
                metadata={"fan_out_count": len(sub_tasks)}
            )
        else:
            errors = [r.error for r in processed_results if not r.success and r.error]
            return TaskResult(
                task_id=task.id,
                success=False,
                output=aggregated_output,
                execution_results=all_execution_results,
                error="; ".join(errors),
                duration_ms=duration,
                metadata={"fan_out_count": len(sub_tasks)}
            )

    def __repr__(self) -> str:
        return f"<FanOut agents={len(self._agents)}>"
