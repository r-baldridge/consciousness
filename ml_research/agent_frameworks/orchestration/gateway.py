"""
Central Gateway for agent orchestration.

The Gateway serves as the central control plane for managing multiple agents,
dispatching tasks, and coordinating the overall orchestration system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from .base import Task, TaskResult, TaskStatus, AgentBase, Session


logger = logging.getLogger(__name__)


class AgentState(Enum):
    """State of an agent in the gateway."""
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class GatewayConfig:
    """
    Configuration for the Gateway.

    Attributes:
        max_concurrent_agents: Maximum number of agents that can run concurrently
        default_timeout: Default timeout for task execution in seconds
        enable_logging: Whether to enable detailed logging
        retry_failed_tasks: Whether to automatically retry failed tasks
        max_retries: Maximum number of retries for failed tasks
        health_check_interval: Interval between health checks in seconds
        queue_size: Maximum size of the task queue
    """
    max_concurrent_agents: int = 10
    default_timeout: int = 300
    enable_logging: bool = True
    retry_failed_tasks: bool = False
    max_retries: int = 3
    health_check_interval: int = 60
    queue_size: int = 1000


@dataclass
class AgentStatus:
    """
    Status information for an agent.

    Attributes:
        agent_id: Unique identifier of the agent
        status: Current state of the agent
        current_task: ID of the currently executing task, if any
        last_activity: Timestamp of last activity
        tasks_completed: Number of tasks completed
        tasks_failed: Number of tasks that failed
        error_message: Last error message, if any
    """
    agent_id: str
    status: str  # idle, running, stopped, error
    current_task: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)
    tasks_completed: int = 0
    tasks_failed: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "current_task": self.current_task,
            "last_activity": self.last_activity.isoformat(),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "error_message": self.error_message,
        }


class Gateway:
    """
    Central control plane for managing multiple agents.

    The Gateway handles agent registration, task dispatching, session management,
    and overall coordination of the orchestration system.

    Example:
        config = GatewayConfig(max_concurrent_agents=10)
        gateway = Gateway(config)

        # Register an agent
        agent_id = await gateway.register_agent(my_agent)

        # Dispatch a task
        task = Task(type="code_review", description="Review PR #123")
        result = await gateway.dispatch(task)

        # Get status
        statuses = await gateway.get_status()
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        """
        Initialize the Gateway.

        Args:
            config: Gateway configuration (uses defaults if not provided)
        """
        self.config = config or GatewayConfig()
        self.agents: Dict[str, AgentBase] = {}
        self.sessions: Dict[str, Session] = {}
        self._agent_statuses: Dict[str, AgentStatus] = {}
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue(maxsize=self.config.queue_size)
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._is_running = False

        if self.config.enable_logging:
            logging.basicConfig(level=logging.INFO)

    async def start(self) -> None:
        """
        Start the gateway and background tasks.

        This starts the task dispatcher and health check loops.
        """
        if self._is_running:
            logger.warning("Gateway is already running")
            return

        self._is_running = True
        self._shutdown_event.clear()

        # Start background tasks
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Gateway started")

    async def stop(self) -> None:
        """
        Stop the gateway gracefully.

        This signals shutdown and waits for background tasks to complete.
        """
        if not self._is_running:
            return

        logger.info("Stopping gateway...")
        self._shutdown_event.set()
        self._is_running = False

        # Cancel background tasks
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Wait for running tasks with timeout
        if self._running_tasks:
            logger.info(f"Waiting for {len(self._running_tasks)} running tasks...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._running_tasks.values(), return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for tasks, cancelling...")
                for task in self._running_tasks.values():
                    task.cancel()

        logger.info("Gateway stopped")

    async def register_agent(self, agent: AgentBase) -> str:
        """
        Register an agent with the gateway.

        Args:
            agent: The agent to register

        Returns:
            The agent's ID

        Raises:
            ValueError: If agent is already registered or max agents reached
        """
        async with self._lock:
            if agent.id in self.agents:
                raise ValueError(f"Agent {agent.id} is already registered")

            if len(self.agents) >= self.config.max_concurrent_agents:
                raise ValueError(
                    f"Maximum number of agents ({self.config.max_concurrent_agents}) reached"
                )

            # Initialize the agent
            logger.info(f"Registering agent {agent.id}...")
            self._agent_statuses[agent.id] = AgentStatus(
                agent_id=agent.id,
                status=AgentState.INITIALIZING.value,
            )

            try:
                await agent.initialize()
                self.agents[agent.id] = agent
                self._agent_statuses[agent.id].status = AgentState.IDLE.value
                logger.info(f"Agent {agent.id} registered successfully")
                return agent.id

            except Exception as e:
                self._agent_statuses[agent.id].status = AgentState.ERROR.value
                self._agent_statuses[agent.id].error_message = str(e)
                logger.error(f"Failed to initialize agent {agent.id}: {e}")
                raise

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the gateway.

        Args:
            agent_id: ID of the agent to unregister

        Raises:
            ValueError: If agent is not registered
        """
        async with self._lock:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} is not registered")

            agent = self.agents[agent_id]

            # Update status
            self._agent_statuses[agent_id].status = AgentState.SHUTTING_DOWN.value

            # Cancel any running task for this agent
            for task_id, running_task in list(self._running_tasks.items()):
                result = self._results.get(task_id)
                if result and result.agent_id == agent_id:
                    running_task.cancel()
                    del self._running_tasks[task_id]

            # Shutdown the agent
            try:
                await agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent {agent_id}: {e}")

            # Remove from registry
            del self.agents[agent_id]
            self._agent_statuses[agent_id].status = AgentState.STOPPED.value

            # Clean up sessions for this agent
            sessions_to_remove = [
                sid for sid, session in self.sessions.items()
                if session.agent_id == agent_id
            ]
            for sid in sessions_to_remove:
                del self.sessions[sid]

            logger.info(f"Agent {agent_id} unregistered")

    async def dispatch(
        self,
        task: Task,
        agent_id: Optional[str] = None,
        wait: bool = True
    ) -> TaskResult:
        """
        Dispatch a task to an appropriate agent.

        Args:
            task: The task to dispatch
            agent_id: Specific agent to handle the task (optional)
            wait: Whether to wait for the task to complete

        Returns:
            TaskResult with the execution outcome

        Raises:
            ValueError: If no suitable agent is available
        """
        # Find or validate agent
        if agent_id:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} is not registered")
            selected_agent = self.agents[agent_id]
        else:
            selected_agent = await self._select_agent(task)
            if not selected_agent:
                raise ValueError(f"No suitable agent found for task type: {task.type}")

        logger.info(f"Dispatching task {task.id} to agent {selected_agent.id}")

        # Execute the task
        if wait:
            return await self._execute_task(task, selected_agent)
        else:
            # Queue for async execution
            async_task = asyncio.create_task(self._execute_task(task, selected_agent))
            self._running_tasks[task.id] = async_task
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.QUEUED,
                agent_id=selected_agent.id,
            )

    async def dispatch_batch(
        self,
        tasks: List[Task],
        agent_id: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[TaskResult]:
        """
        Dispatch multiple tasks concurrently.

        Args:
            tasks: List of tasks to dispatch
            agent_id: Specific agent to handle all tasks (optional)
            max_concurrent: Maximum concurrent task executions

        Returns:
            List of TaskResults in the same order as input tasks
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def dispatch_with_semaphore(task: Task) -> TaskResult:
            async with semaphore:
                return await self.dispatch(task, agent_id=agent_id)

        results = await asyncio.gather(
            *[dispatch_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

        # Convert exceptions to failure results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TaskResult.failure_result(
                    task_id=tasks[i].id,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    async def get_status(self) -> Dict[str, AgentStatus]:
        """
        Get status of all registered agents.

        Returns:
            Dictionary mapping agent IDs to their status
        """
        return {
            agent_id: status
            for agent_id, status in self._agent_statuses.items()
            if agent_id in self.agents
        }

    async def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """
        Get status of a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            AgentStatus or None if not found
        """
        return self._agent_statuses.get(agent_id)

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get the result of a previously dispatched task.

        Args:
            task_id: ID of the task

        Returns:
            TaskResult or None if not found
        """
        return self._results.get(task_id)

    async def create_session(self, agent_id: str, session_id: Optional[str] = None) -> Session:
        """
        Create a new session with an agent.

        Args:
            agent_id: ID of the agent
            session_id: Optional specific session ID

        Returns:
            The created Session

        Raises:
            ValueError: If agent is not registered
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} is not registered")

        import uuid
        sid = session_id or str(uuid.uuid4())

        session = Session(id=sid, agent_id=agent_id)
        self.sessions[sid] = session

        logger.info(f"Created session {sid} for agent {agent_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: ID of the session

        Returns:
            Session or None if not found
        """
        return self.sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """
        Close and remove a session.

        Args:
            session_id: ID of the session
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Closed session {session_id}")

    async def shutdown(self) -> None:
        """
        Shutdown the gateway and all agents.

        This unregisters all agents and cleans up resources.
        """
        logger.info("Shutting down gateway...")

        # Stop the gateway
        await self.stop()

        # Unregister all agents
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            try:
                await self.unregister_agent(agent_id)
            except Exception as e:
                logger.error(f"Error unregistering agent {agent_id}: {e}")

        # Clear all state
        self.sessions.clear()
        self._results.clear()
        self._agent_statuses.clear()

        logger.info("Gateway shutdown complete")

    async def _select_agent(self, task: Task) -> Optional[AgentBase]:
        """
        Select the best agent for a task.

        Args:
            task: The task to find an agent for

        Returns:
            Selected agent or None if no suitable agent found
        """
        # Find idle agents that can handle the task
        candidates = []
        for agent_id, agent in self.agents.items():
            status = self._agent_statuses.get(agent_id)
            if status and status.status == AgentState.IDLE.value:
                if agent.can_handle(task):
                    candidates.append(agent)

        if not candidates:
            # Try agents that are currently busy but can handle the task
            for agent_id, agent in self.agents.items():
                status = self._agent_statuses.get(agent_id)
                if status and status.status != AgentState.ERROR.value:
                    if agent.can_handle(task):
                        candidates.append(agent)

        if not candidates:
            return None

        # For now, return the first candidate
        # Could be extended with more sophisticated selection logic
        return candidates[0]

    async def _execute_task(self, task: Task, agent: AgentBase) -> TaskResult:
        """
        Execute a task on a specific agent.

        Args:
            task: The task to execute
            agent: The agent to execute the task

        Returns:
            TaskResult with the execution outcome
        """
        start_time = datetime.now()
        status = self._agent_statuses.get(agent.id)

        if status:
            status.status = AgentState.RUNNING.value
            status.current_task = task.id
            status.last_activity = datetime.now()

        try:
            # Execute with timeout
            timeout = task.timeout or self.config.default_timeout

            result = await asyncio.wait_for(
                agent.execute(task),
                timeout=timeout
            )

            # Update stats
            if status:
                status.tasks_completed += 1
                status.current_task = None
                status.status = AgentState.IDLE.value
                status.last_activity = datetime.now()

            # Store result
            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.agent_id = agent.id
            self._results[task.id] = result

            logger.info(
                f"Task {task.id} completed by {agent.id} in {result.execution_time:.2f}s"
            )

            return result

        except asyncio.TimeoutError:
            if status:
                status.tasks_failed += 1
                status.current_task = None
                status.status = AgentState.IDLE.value
                status.error_message = f"Task {task.id} timed out"

            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.TIMEOUT,
                error=f"Task timed out after {timeout}s",
                agent_id=agent.id,
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
            self._results[task.id] = result

            logger.warning(f"Task {task.id} timed out on agent {agent.id}")
            return result

        except Exception as e:
            if status:
                status.tasks_failed += 1
                status.current_task = None
                status.status = AgentState.IDLE.value
                status.error_message = str(e)

            result = TaskResult.failure_result(
                task_id=task.id,
                error=str(e),
                agent_id=agent.id,
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
            self._results[task.id] = result

            logger.error(f"Task {task.id} failed on agent {agent.id}: {e}")
            return result

    async def _dispatcher_loop(self) -> None:
        """Background loop for dispatching queued tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Check for tasks in queue
                try:
                    task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0
                    )
                    # Dispatch the task
                    await self.dispatch(task, wait=False)
                except asyncio.TimeoutError:
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dispatcher loop: {e}")
                await asyncio.sleep(1)

    async def _health_check_loop(self) -> None:
        """Background loop for agent health checks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)

                for agent_id, agent in list(self.agents.items()):
                    try:
                        is_healthy = await agent.health_check()
                        if not is_healthy:
                            status = self._agent_statuses.get(agent_id)
                            if status:
                                status.status = AgentState.ERROR.value
                                status.error_message = "Health check failed"
                            logger.warning(f"Agent {agent_id} failed health check")
                    except Exception as e:
                        logger.error(f"Error checking health of agent {agent_id}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def queue_task(self, task: Task) -> None:
        """
        Add a task to the queue for async processing.

        Args:
            task: The task to queue
        """
        await self._task_queue.put(task)
        logger.info(f"Task {task.id} queued")

    def get_queue_size(self) -> int:
        """Get the current number of tasks in the queue."""
        return self._task_queue.qsize()

    def get_running_task_count(self) -> int:
        """Get the number of currently running tasks."""
        return len(self._running_tasks)

    async def __aenter__(self) -> "Gateway":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
