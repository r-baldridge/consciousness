"""
CrewAI integration adapter.

This module provides integration with CrewAI's multi-agent framework,
allowing use of our agents in CrewAI crews and vice versa.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
import logging
import asyncio

if TYPE_CHECKING:
    from ..tools.tool_base import Tool

logger = logging.getLogger(__name__)

# Try to import CrewAI components
try:
    from crewai import Agent as CrewAgent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.debug("CrewAI not available. Install with: pip install crewai")


def require_crewai(func: Callable) -> Callable:
    """Decorator to check if CrewAI is available."""
    def wrapper(*args, **kwargs):
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. "
                "Install with: pip install crewai"
            )
        return func(*args, **kwargs)
    return wrapper


@dataclass
class TaskDefinition:
    """Definition for a CrewAI task.

    Attributes:
        description: What the task should accomplish
        expected_output: Description of expected output
        agent_id: Optional ID of agent to assign task to
    """
    description: str
    expected_output: str
    agent_id: Optional[str] = None


class CrewAIAdapter:
    """Integrate with CrewAI multi-agent framework.

    This class provides methods to convert our agents to CrewAI format
    and create CrewAI crews from our agents.

    Example:
        adapter = CrewAIAdapter()

        # Convert single agent
        crew_agent = adapter.to_crew_agent(my_agent, role="researcher")

        # Create a full crew
        tasks = [
            TaskDefinition("Research the topic", "Detailed research notes"),
            TaskDefinition("Write article", "Complete article draft")
        ]
        crew = adapter.create_crew(
            agents=[agent1, agent2],
            tasks=tasks,
            process="sequential"
        )

        # Run the crew
        result = await adapter.run_crew(crew)
    """

    def __init__(self):
        """Initialize the CrewAI adapter."""
        self._agent_map: Dict[str, Any] = {}

    @require_crewai
    def to_crew_agent(
        self,
        agent: Any,
        role: str,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
        tools: Optional[List["Tool"]] = None,
        verbose: bool = False
    ) -> Any:
        """Convert our agent to a CrewAI agent.

        Args:
            agent: Our agent instance
            role: Role description for the agent
            goal: Agent's goal (derived from agent if not provided)
            backstory: Agent's backstory
            tools: Optional tools to provide
            verbose: Enable verbose output

        Returns:
            CrewAI Agent instance
        """
        # Extract goal from agent if not provided
        if goal is None:
            if hasattr(agent, 'goal'):
                goal = agent.goal
            elif hasattr(agent, 'system_prompt'):
                goal = agent.system_prompt[:200]
            else:
                goal = f"Accomplish tasks as {role}"

        # Extract backstory if not provided
        if backstory is None:
            if hasattr(agent, 'backstory'):
                backstory = agent.backstory
            else:
                backstory = f"An AI agent specialized in {role}"

        # Convert tools
        crew_tools = []
        if tools:
            from .langchain_adapter import LangChainToolAdapter
            lc_adapter = LangChainToolAdapter()
            # CrewAI uses LangChain tools internally
            for tool in tools:
                try:
                    from .langchain_adapter import _LangChainToolWrapper
                    crew_tools.append(_LangChainToolWrapper(tool))
                except Exception as e:
                    logger.warning(f"Could not convert tool {tool.name}: {e}")

        # Create wrapper that uses our agent's LLM
        crew_agent = _CrewAgentWrapper(
            agent=agent,
            role=role,
            goal=goal,
            backstory=backstory,
            tools=crew_tools,
            verbose=verbose
        )

        # Store mapping
        agent_id = getattr(agent, 'id', str(id(agent)))
        self._agent_map[agent_id] = crew_agent

        return crew_agent

    @require_crewai
    def from_crew_agent(self, crew_agent: Any) -> Any:
        """Convert a CrewAI agent to our format.

        Args:
            crew_agent: CrewAI Agent instance

        Returns:
            Our agent format wrapper
        """
        return _OurAgentFromCrew(crew_agent)

    @require_crewai
    def create_task(
        self,
        definition: TaskDefinition,
        agent: Optional[Any] = None
    ) -> Any:
        """Create a CrewAI task.

        Args:
            definition: Task definition
            agent: Optional agent to assign

        Returns:
            CrewAI Task instance
        """
        return Task(
            description=definition.description,
            expected_output=definition.expected_output,
            agent=agent
        )

    @require_crewai
    def create_crew(
        self,
        agents: List[Any],
        tasks: List[TaskDefinition],
        process: str = "sequential",
        verbose: bool = False,
        **kwargs
    ) -> Any:
        """Create a CrewAI crew from agents and tasks.

        Args:
            agents: List of agents (ours or already CrewAI)
            tasks: List of task definitions
            process: Execution process ("sequential" or "hierarchical")
            verbose: Enable verbose output
            **kwargs: Additional Crew arguments

        Returns:
            CrewAI Crew instance
        """
        # Convert agents if needed
        crew_agents = []
        for i, agent in enumerate(agents):
            if hasattr(agent, '_crewai_agent'):
                # Already a wrapper
                crew_agents.append(agent._crewai_agent)
            elif isinstance(agent, CrewAgent):
                crew_agents.append(agent)
            else:
                # Convert our agent
                role = f"Agent {i+1}"
                if hasattr(agent, 'role'):
                    role = agent.role
                crew_agents.append(self.to_crew_agent(agent, role=role))

        # Create tasks
        crew_tasks = []
        for i, task_def in enumerate(tasks):
            # Assign to agent if specified
            task_agent = None
            if task_def.agent_id:
                for agent in crew_agents:
                    agent_id = getattr(agent, 'id', str(id(agent)))
                    if agent_id == task_def.agent_id:
                        task_agent = agent
                        break

            # Default to agent by index
            if task_agent is None and i < len(crew_agents):
                task_agent = crew_agents[i]

            crew_tasks.append(Task(
                description=task_def.description,
                expected_output=task_def.expected_output,
                agent=task_agent
            ))

        # Determine process
        if process == "hierarchical":
            crew_process = Process.hierarchical
        else:
            crew_process = Process.sequential

        return Crew(
            agents=crew_agents,
            tasks=crew_tasks,
            process=crew_process,
            verbose=verbose,
            **kwargs
        )

    @require_crewai
    async def run_crew(self, crew: Any) -> str:
        """Run a CrewAI crew asynchronously.

        Args:
            crew: CrewAI Crew instance

        Returns:
            Crew execution result
        """
        # CrewAI's kickoff is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)
        return str(result)

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics.

        Returns:
            Dictionary with adapter stats
        """
        return {
            "crewai_available": CREWAI_AVAILABLE,
            "converted_agents": len(self._agent_map)
        }


if CREWAI_AVAILABLE:

    class _CrewAgentWrapper(CrewAgent):
        """Wrapper that bridges our agent to CrewAI."""

        def __init__(
            self,
            agent: Any,
            role: str,
            goal: str,
            backstory: str,
            tools: List[Any],
            verbose: bool
        ):
            self._our_agent = agent

            # Initialize CrewAI agent
            super().__init__(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools,
                verbose=verbose,
                allow_delegation=False
            )

        @property
        def id(self) -> str:
            return getattr(self._our_agent, 'id', str(id(self._our_agent)))


    class _OurAgentFromCrew:
        """Wraps a CrewAI agent in our interface."""

        def __init__(self, crew_agent: CrewAgent):
            self._crewai_agent = crew_agent
            self.id = f"crew-{id(crew_agent)}"
            self.role = crew_agent.role
            self.goal = crew_agent.goal

        async def run(self, input_text: str) -> str:
            """Run the agent on input."""
            # CrewAI agents are typically run through tasks/crews
            # This provides a simple interface
            loop = asyncio.get_event_loop()

            def _execute():
                # Create a simple task and execute
                task = Task(
                    description=input_text,
                    expected_output="Response to the input",
                    agent=self._crewai_agent
                )
                crew = Crew(
                    agents=[self._crewai_agent],
                    tasks=[task],
                    verbose=False
                )
                return crew.kickoff()

            result = await loop.run_in_executor(None, _execute)
            return str(result)

        def get_state(self) -> Dict[str, Any]:
            return {
                "type": "crewai_agent",
                "role": self._crewai_agent.role,
                "goal": self._crewai_agent.goal
            }

        def set_state(self, state: Dict[str, Any]) -> None:
            pass

else:
    # Stub classes when CrewAI is not available
    class _CrewAgentWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("CrewAI not available")

    class _OurAgentFromCrew:
        def __init__(self, *args, **kwargs):
            raise ImportError("CrewAI not available")
