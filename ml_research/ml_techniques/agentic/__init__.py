"""
Agentic Techniques

Patterns for autonomous task execution including tool calling,
planning, reflection, and multi-agent collaboration.

=============================================================================
TECHNIQUES
=============================================================================

1. ReAct (Reasoning + Acting)
   - Interleaved thought-action-observation loops
   - Combines reasoning with grounded tool use

2. ToolCalling
   - Structured function/tool invocation
   - Parallel and sequential execution

3. Planning
   - Separate planning from execution
   - DEPS, Plan-and-Execute patterns

4. Reflexion
   - Learn from failures through reflection
   - Maintain memory of past attempts

5. MultiAgent
   - Collaboration between specialized agents
   - Debate, division of labor, peer review

=============================================================================
TOOL SPECIFICATION FORMAT
=============================================================================

Tools are defined as:
    {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "required": True},
            "num_results": {"type": "int", "default": 5},
        },
        "returns": {"type": "list", "items": "SearchResult"},
        "execute": callable_function,
    }
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from enum import Enum
import time
from abc import abstractmethod

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory

# Backend imports
if TYPE_CHECKING:
    from ...backends import LLMBackend

try:
    from ...backends import get_backend, LLMBackend as LLMBackendClass
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    LLMBackendClass = None


def _resolve_backend(backend: Optional[Any], model: Optional[Any]) -> Optional[Any]:
    """Resolve backend from various input types."""
    if backend is not None:
        if isinstance(backend, str):
            if BACKENDS_AVAILABLE:
                return get_backend(backend)
            else:
                raise ImportError("Backends module not available")
        return backend

    if model is not None:
        return model

    if BACKENDS_AVAILABLE:
        try:
            return get_backend()
        except (ValueError, KeyError):
            pass

    return None


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    param_type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolSpec:
    """Specification for a callable tool."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    returns_type: str = "Any"
    execute: Optional[Callable] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format (OpenAI function calling style)."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.param_type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolCall:
    """A call to a tool with arguments."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    call_id: Optional[str]
    success: bool
    result: Any
    error: Optional[str] = None


# =============================================================================
# REACT
# =============================================================================

class ReAct(TechniqueBase):
    """
    ReAct: Synergizing Reasoning and Acting in Language Models.

    Paper: https://arxiv.org/abs/2210.03629

    Interleaves reasoning traces with actions:
        Thought: I need to find information about X
        Action: search("X")
        Observation: [search results]
        Thought: Based on the results, I should...
        Action: ...
        ...

    Configuration:
        backend: LLMBackend instance or name for generation
        tools: List of available tools
        max_iterations: Maximum thought-action cycles
        observation_format: Format string for observations
        thought_prefix: Prefix for thought generation
        stop_sequences: Sequences that indicate completion

    Usage:
        from ml_research.backends import MockBackend

        react = ReAct(
            backend=MockBackend(),
            tools=[search_tool, calculator_tool],
            max_iterations=10,
        )
        result = react.run("What is the population of Tokyo divided by 1000?")
    """

    TECHNIQUE_ID = "react"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        tools: Optional[List[ToolSpec]] = None,
        max_iterations: int = 10,
        thought_prefix: str = "Thought:",
        action_prefix: str = "Action:",
        observation_prefix: str = "Observation:",
        final_answer_prefix: str = "Final Answer:",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend, model)
        self.model = model  # Keep for backward compatibility
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.thought_prefix = thought_prefix
        self.action_prefix = action_prefix
        self.observation_prefix = observation_prefix
        self.final_answer_prefix = final_answer_prefix

    def _parse_action(self, text: str) -> Optional[ToolCall]:
        """Parse action from model output."""
        # Look for Action: tool_name(args)
        import re
        pattern = r'Action:\s*(\w+)\s*\(([^)]*)\)'
        match = re.search(pattern, text)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            # Simple arg parsing (real impl would be more robust)
            args = {"query": args_str.strip('"').strip("'")} if args_str else {}
            return ToolCall(tool_name=tool_name, arguments=args)
        return None

    def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        if call.tool_name not in self.tools:
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                result=None,
                error=f"Unknown tool: {call.tool_name}",
            )

        tool = self.tools[call.tool_name]
        if tool.execute is None:
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                result=None,
                error="Tool has no execute function",
            )

        try:
            result = tool.execute(**call.arguments)
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=True,
                result=result,
            )
        except Exception as e:
            return ToolResult(
                tool_name=call.tool_name,
                call_id=call.call_id,
                success=False,
                result=None,
                error=str(e),
            )

    def _generate_thought_action(
        self,
        task: str,
        history: List[Dict[str, str]],
    ) -> str:
        """Generate next thought and action using backend."""
        # Build prompt
        prompt = self._build_react_prompt(task, history)

        # Try to use backend if available
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(
                    prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stop_sequences=[self.observation_prefix],
                )
            elif callable(self.backend):
                return self.backend(prompt)

        # Placeholder response
        if not history:
            return f"{self.thought_prefix} I need to solve: {task}\n{self.action_prefix} search(\"{task}\")"
        return f"{self.final_answer_prefix} [Placeholder answer]"

    def _build_react_prompt(self, task: str, history: List[Dict[str, str]]) -> str:
        """Build the ReAct-style prompt with history."""
        prompt_parts = [f"Task: {task}\n"]

        for entry in history:
            prompt_parts.append(entry.get("thought_action", ""))
            if "observation" in entry:
                prompt_parts.append(f"{self.observation_prefix} {entry['observation']}\n")

        prompt_parts.append(f"{self.thought_prefix}")
        return "\n".join(prompt_parts)

    def _is_final_answer(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if output contains final answer."""
        if self.final_answer_prefix in text:
            answer = text.split(self.final_answer_prefix)[-1].strip()
            return True, answer
        return False, None

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []
        history: List[Dict[str, str]] = []

        self._call_hooks("pre_run", task=task)

        for iteration in range(self.max_iterations):
            # Generate thought and action
            output = self._generate_thought_action(task, history)
            trace.append({
                "action": "generate",
                "iteration": iteration,
                "output": output[:200],
            })

            # Check for final answer
            is_final, answer = self._is_final_answer(output)
            if is_final:
                self._call_hooks("post_run", answer=answer)
                return TechniqueResult(
                    success=True,
                    output=answer,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    metadata={"iterations": iteration + 1},
                )

            # Parse and execute action
            tool_call = self._parse_action(output)
            if tool_call:
                self._call_hooks("on_step", tool_call=tool_call)
                result = self._execute_tool(tool_call)
                trace.append({
                    "action": "tool_call",
                    "tool": tool_call.tool_name,
                    "success": result.success,
                })

                history.append({
                    "thought_action": output,
                    "observation": str(result.result) if result.success else result.error,
                })

        # Max iterations reached
        return TechniqueResult(
            success=False,
            output=None,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error=f"Max iterations ({self.max_iterations}) reached",
        )


# =============================================================================
# TOOL CALLING
# =============================================================================

class ToolCalling(TechniqueBase):
    """
    Structured tool/function calling.

    Supports:
        - Sequential tool calls
        - Parallel tool calls
        - Tool call chains (output of one feeds another)
        - Error handling and retries

    Configuration:
        tools: Available tools
        parallel_calls: Allow parallel execution
        max_calls_per_turn: Limit calls per iteration
        retry_on_error: Retry failed calls

    Usage:
        tc = ToolCalling(
            model=my_model,
            tools=[calculator, search, code_exec],
            parallel_calls=True,
        )
        result = tc.run({
            "task": "Calculate 15% tip on $47.50",
            "tools_hint": ["calculator"],
        })
    """

    TECHNIQUE_ID = "tool_calling"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        model: Optional[Any] = None,
        tools: Optional[List[ToolSpec]] = None,
        parallel_calls: bool = False,
        max_calls_per_turn: int = 5,
        retry_on_error: bool = True,
        max_retries: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.tools = {t.name: t for t in (tools or [])}
        self.parallel_calls = parallel_calls
        self.max_calls_per_turn = max_calls_per_turn
        self.retry_on_error = retry_on_error
        self.max_retries = max_retries

    def _decide_tool_calls(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> List[ToolCall]:
        """Decide which tools to call (placeholder)."""
        # Real implementation uses LLM
        return []

    def _execute_calls(
        self,
        calls: List[ToolCall],
    ) -> List[ToolResult]:
        """Execute tool calls (potentially in parallel)."""
        results = []

        if self.parallel_calls:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._execute_single, call): call
                    for call in calls
                }
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
        else:
            for call in calls:
                results.append(self._execute_single(call))

        return results

    def _execute_single(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call with retry logic."""
        for attempt in range(self.max_retries + 1):
            tool = self.tools.get(call.tool_name)
            if not tool:
                return ToolResult(
                    tool_name=call.tool_name,
                    call_id=call.call_id,
                    success=False,
                    result=None,
                    error=f"Unknown tool: {call.tool_name}",
                )

            if tool.execute is None:
                return ToolResult(
                    tool_name=call.tool_name,
                    call_id=call.call_id,
                    success=False,
                    result=None,
                    error="Tool has no execute function",
                )

            try:
                result = tool.execute(**call.arguments)
                return ToolResult(
                    tool_name=call.tool_name,
                    call_id=call.call_id,
                    success=True,
                    result=result,
                )
            except Exception as e:
                if attempt == self.max_retries or not self.retry_on_error:
                    return ToolResult(
                        tool_name=call.tool_name,
                        call_id=call.call_id,
                        success=False,
                        result=None,
                        error=str(e),
                    )
                # Retry

        return ToolResult(
            tool_name=call.tool_name,
            call_id=call.call_id,
            success=False,
            result=None,
            error="Max retries exceeded",
        )

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data.get("task", str(input_data)) if isinstance(input_data, dict) else str(input_data)
        context = context or {}
        trace: List[Dict] = []

        # Decide which tools to call
        calls = self._decide_tool_calls(task, context)
        trace.append({
            "action": "decide_calls",
            "num_calls": len(calls),
            "tools": [c.tool_name for c in calls],
        })

        if not calls:
            return TechniqueResult(
                success=True,
                output={"message": "No tool calls needed"},
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
            )

        # Execute calls
        results = self._execute_calls(calls[:self.max_calls_per_turn])
        trace.append({
            "action": "execute_calls",
            "results": [{"tool": r.tool_name, "success": r.success} for r in results],
        })

        # Aggregate results
        all_success = all(r.success for r in results)
        output = {
            "results": [
                {"tool": r.tool_name, "result": r.result, "error": r.error}
                for r in results
            ]
        }

        return TechniqueResult(
            success=all_success,
            output=output,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# PLANNING
# =============================================================================

class PlanStep:
    """A step in an execution plan."""
    def __init__(
        self,
        step_id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        tool: Optional[str] = None,
        estimated_difficulty: float = 0.5,
    ):
        self.step_id = step_id
        self.description = description
        self.dependencies = dependencies or []
        self.tool = tool
        self.estimated_difficulty = estimated_difficulty
        self.status = "pending"
        self.result = None


class Planning(TechniqueBase):
    """
    Plan-and-Execute: Separate planning from execution.

    Variants:
        - Simple: Generate plan, execute sequentially
        - DEPS: Describe, Explain, Plan, Select
        - Hierarchical: Multi-level planning

    Configuration:
        planning_strategy: Which planning approach to use
        allow_replan: Whether to replan on failures
        max_plan_steps: Maximum steps in a plan

    Usage:
        planner = Planning(
            model=my_model,
            planning_strategy="deps",
            allow_replan=True,
        )
        result = planner.run("Build a web scraper for news articles")
    """

    TECHNIQUE_ID = "planning"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        model: Optional[Any] = None,
        planning_strategy: str = "simple",
        allow_replan: bool = True,
        max_plan_steps: int = 10,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.planning_strategy = planning_strategy
        self.allow_replan = allow_replan
        self.max_plan_steps = max_plan_steps
        self.tools = tools or []

    def _generate_plan(self, task: str, context: Dict[str, Any]) -> List[PlanStep]:
        """Generate execution plan (placeholder)."""
        # Real implementation uses LLM
        return [
            PlanStep("1", f"Understand the task: {task[:50]}"),
            PlanStep("2", "Gather required information", dependencies=["1"]),
            PlanStep("3", "Execute main action", dependencies=["2"]),
            PlanStep("4", "Verify results", dependencies=["3"]),
        ]

    def _execute_step(
        self,
        step: PlanStep,
        context: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """Execute a single plan step (placeholder)."""
        # Real implementation uses LLM or tools
        return True, f"[Result of: {step.description}]"

    def _replan(
        self,
        original_plan: List[PlanStep],
        failed_step: PlanStep,
        error: str,
        context: Dict[str, Any],
    ) -> List[PlanStep]:
        """Generate new plan after failure (placeholder)."""
        # Real implementation uses LLM
        return original_plan

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        context = context or {}
        trace: List[Dict] = []

        # Generate plan
        plan = self._generate_plan(task, context)
        trace.append({
            "action": "generate_plan",
            "num_steps": len(plan),
            "steps": [s.description[:50] for s in plan],
        })

        # Execute plan
        results = []
        for step in plan:
            # Check dependencies
            deps_met = all(
                any(r["step_id"] == dep and r["success"] for r in results)
                for dep in step.dependencies
            )

            if not deps_met:
                trace.append({
                    "action": "skip_step",
                    "step_id": step.step_id,
                    "reason": "dependencies not met",
                })
                continue

            self._call_hooks("on_step", step=step)

            success, result = self._execute_step(step, context)
            step.status = "completed" if success else "failed"
            step.result = result

            trace.append({
                "action": "execute_step",
                "step_id": step.step_id,
                "success": success,
            })

            results.append({
                "step_id": step.step_id,
                "success": success,
                "result": result,
            })

            # Replan on failure if enabled
            if not success and self.allow_replan:
                plan = self._replan(plan, step, str(result), context)
                trace.append({"action": "replan"})

        # Aggregate
        all_success = all(r["success"] for r in results)

        return TechniqueResult(
            success=all_success,
            output={"plan": [s.description for s in plan], "results": results},
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# REFLEXION
# =============================================================================

class Reflexion(TechniqueBase):
    """
    Reflexion: Learning from failures through verbal reinforcement.

    Paper: https://arxiv.org/abs/2303.11366

    Algorithm:
        1. Attempt task
        2. If failure, reflect on what went wrong
        3. Store reflection in memory
        4. Retry with reflections as context
        5. Repeat until success or max trials

    Configuration:
        max_trials: Maximum attempt cycles
        memory_type: How to store reflections
        reflection_prompt: Prompt for generating reflections
    """

    TECHNIQUE_ID = "reflexion"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        model: Optional[Any] = None,
        max_trials: int = 3,
        memory_type: str = "full",  # full, summarized, sliding
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.max_trials = max_trials
        self.memory_type = memory_type
        self.reflections: List[str] = []

    def _attempt_task(self, task: str, reflections: List[str]) -> Tuple[bool, Any]:
        """Attempt the task with reflection context (placeholder)."""
        # Real implementation uses LLM
        return False, "Attempt failed"

    def _generate_reflection(self, task: str, attempt_result: Any) -> str:
        """Generate reflection on failure (placeholder)."""
        # Real implementation uses LLM
        return f"Reflection: The attempt failed because..."

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        for trial in range(self.max_trials):
            # Attempt with current reflections
            success, result = self._attempt_task(task, self.reflections)
            trace.append({
                "action": "attempt",
                "trial": trial,
                "success": success,
            })

            if success:
                return TechniqueResult(
                    success=True,
                    output=result,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    metadata={"trials": trial + 1, "reflections": len(self.reflections)},
                )

            # Generate and store reflection
            reflection = self._generate_reflection(task, result)
            self.reflections.append(reflection)
            trace.append({
                "action": "reflect",
                "reflection": reflection[:100],
            })

        # Max trials reached
        return TechniqueResult(
            success=False,
            output=None,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error=f"Max trials ({self.max_trials}) reached",
            metadata={"reflections": self.reflections},
        )


# =============================================================================
# MULTI-AGENT
# =============================================================================

class CoordinationPattern(Enum):
    """Coordination patterns for multi-agent systems."""
    DEBATE = "debate"           # Agents argue positions, judge decides
    DIVISION = "division"       # Coordinator splits work among specialists
    HIERARCHICAL = "hierarchical"  # Manager directs workers
    PEER_REVIEW = "peer_review"    # Agents review each other's work


class MessageType(Enum):
    """Types of messages in multi-agent communication."""
    PROPOSAL = "proposal"       # Initial proposal/answer
    CRITIQUE = "critique"       # Criticism of proposal
    REBUTTAL = "rebuttal"       # Response to critique
    JUDGMENT = "judgment"       # Final decision
    TASK = "task"               # Task assignment
    RESULT = "result"           # Task result
    REVIEW = "review"           # Peer review
    REVISION = "revision"       # Revised work


@dataclass
class AgentMessage:
    """A message passed between agents."""
    sender_id: str
    receiver_id: str
    content: str
    message_type: MessageType
    round_num: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    """Specification for an agent in multi-agent system."""
    agent_id: str
    role: str
    description: str
    model: Optional[Any] = None
    tools: List[ToolSpec] = field(default_factory=list)
    system_prompt: str = ""
    capabilities: List[str] = field(default_factory=list)


@dataclass
class ConversationHistory:
    """Tracks all messages in a multi-agent conversation."""
    messages: List[AgentMessage] = field(default_factory=list)

    def add(self, message: AgentMessage) -> None:
        """Add a message to history."""
        self.messages.append(message)

    def get_by_sender(self, sender_id: str) -> List[AgentMessage]:
        """Get all messages from a specific sender."""
        return [m for m in self.messages if m.sender_id == sender_id]

    def get_by_type(self, msg_type: MessageType) -> List[AgentMessage]:
        """Get all messages of a specific type."""
        return [m for m in self.messages if m.message_type == msg_type]

    def get_round(self, round_num: int) -> List[AgentMessage]:
        """Get all messages from a specific round."""
        return [m for m in self.messages if m.round_num == round_num]

    def format_for_context(self, max_messages: int = 20) -> str:
        """Format recent history for inclusion in prompts."""
        recent = self.messages[-max_messages:]
        lines = []
        for msg in recent:
            lines.append(f"[{msg.sender_id} -> {msg.receiver_id}] ({msg.message_type.value}): {msg.content[:200]}")
        return "\n".join(lines)


class ConsensusProtocol:
    """Protocols for reaching consensus among agents."""

    @staticmethod
    def majority_vote(responses: List[str]) -> Tuple[str, float]:
        """Return the most common response and its confidence."""
        from collections import Counter
        if not responses:
            return "", 0.0
        counts = Counter(responses)
        winner, count = counts.most_common(1)[0]
        confidence = count / len(responses)
        return winner, confidence

    @staticmethod
    def unanimous(responses: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if all responses agree."""
        if not responses:
            return False, None
        unique = set(responses)
        if len(unique) == 1:
            return True, responses[0]
        return False, None

    @staticmethod
    def weighted_average(scores: List[Tuple[str, float]]) -> str:
        """Select response with highest weight/confidence."""
        if not scores:
            return ""
        return max(scores, key=lambda x: x[1])[0]

    @staticmethod
    def merge_responses(responses: List[str], separator: str = "\n\n") -> str:
        """Merge multiple responses into one."""
        return separator.join(responses)


class MultiAgent(TechniqueBase):
    """
    Multi-agent collaboration patterns.

    Paper references:
        - "Improving Factuality and Reasoning in LMs through Multiagent Debate"
          (Du et al., 2023) https://arxiv.org/abs/2305.14325
        - "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate"
          (Chan et al., 2023) https://arxiv.org/abs/2308.07201

    Patterns:
        DEBATE: Multiple agents argue positions, judge decides
            - Proposer generates initial answer
            - Critic finds flaws and counterarguments
            - Proposer refines based on critique
            - Judge evaluates final arguments

        DIVISION: Coordinator splits work among specialists
            - Coordinator decomposes task into subtasks
            - Workers are assigned subtasks based on capabilities
            - Results are aggregated by coordinator

        HIERARCHICAL: Manager directs workers
            - Manager creates strategy/plan
            - Workers receive instructions and execute
            - Workers report results
            - Manager reviews and adjusts

        PEER_REVIEW: Agents review each other's work
            - Worker A produces initial output
            - Worker B reviews and critiques
            - Worker A revises based on feedback
            - Iterate until consensus

    Configuration:
        agents: List of agent specifications
        coordination: Coordination pattern (debate/division/hierarchical/peer_review)
        max_rounds: Maximum interaction rounds
        consensus_threshold: Confidence threshold for consensus (0.0-1.0)

    Usage:
        agents = [
            AgentSpec("proposer", "proposer", "Generates initial proposals"),
            AgentSpec("critic", "critic", "Critiques proposals"),
            AgentSpec("judge", "judge", "Makes final decisions"),
        ]
        ma = MultiAgent(
            agents=agents,
            coordination="debate",
            max_rounds=3,
        )
        result = ma.run("Should we use microservices or monolith?")
    """

    TECHNIQUE_ID = "multi_agent"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        backend: Optional[Any] = None,
        agents: Optional[List[AgentSpec]] = None,
        coordination: str = "hierarchical",
        max_rounds: int = 5,
        consensus_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = backend
        self.agents = {a.agent_id: a for a in (agents or [])}
        self.coordination = coordination
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.history = ConversationHistory()

    def _get_agent_response(
        self,
        agent: AgentSpec,
        prompt: str,
        context: str = "",
    ) -> str:
        """Get a response from an agent (uses backend or placeholder)."""
        full_prompt = f"{agent.system_prompt}\n\n{context}\n\n{prompt}" if context else f"{agent.system_prompt}\n\n{prompt}"

        if self.backend:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(full_prompt)
            elif callable(self.backend):
                return self.backend(full_prompt)

        # Placeholder responses based on role
        role_responses = {
            "proposer": f"I propose the following solution: Based on the requirements, we should...",
            "critic": f"I see several issues with this proposal: First, the approach may not scale well...",
            "judge": f"After reviewing both arguments, I conclude that the proposal has merit but needs refinement...",
            "manager": f"Here is the plan: We will divide this into subtasks and assign them to workers...",
            "worker": f"Task completed. Here are the results: I have analyzed the assigned portion...",
            "reviewer": f"Review feedback: The work is generally good but I suggest these improvements...",
            "coordinator": f"I will decompose this task into the following subtasks...",
        }
        return role_responses.get(agent.role, f"Response from {agent.role}: Processing the request...")

    def _ensure_default_agents(self, pattern: str) -> None:
        """Ensure required agents exist for the pattern."""
        if pattern == "debate":
            defaults = [
                AgentSpec("proposer", "proposer", "Generates proposals", system_prompt="You propose solutions and defend them."),
                AgentSpec("critic", "critic", "Critiques proposals", system_prompt="You find flaws in proposals."),
                AgentSpec("judge", "judge", "Makes final decisions", system_prompt="You evaluate arguments fairly."),
            ]
        elif pattern == "division":
            defaults = [
                AgentSpec("coordinator", "coordinator", "Decomposes and assigns tasks", system_prompt="You break down tasks and coordinate workers."),
                AgentSpec("worker_1", "worker", "Executes subtasks", system_prompt="You complete assigned subtasks."),
                AgentSpec("worker_2", "worker", "Executes subtasks", system_prompt="You complete assigned subtasks."),
            ]
        elif pattern == "hierarchical":
            defaults = [
                AgentSpec("manager", "manager", "Plans and reviews", system_prompt="You create plans and review work."),
                AgentSpec("worker_1", "worker", "Executes instructions", system_prompt="You follow manager instructions."),
                AgentSpec("worker_2", "worker", "Executes instructions", system_prompt="You follow manager instructions."),
            ]
        elif pattern == "peer_review":
            defaults = [
                AgentSpec("author", "author", "Produces initial work", system_prompt="You create initial content."),
                AgentSpec("reviewer", "reviewer", "Reviews and suggests", system_prompt="You review and provide feedback."),
            ]
        else:
            defaults = []

        for agent in defaults:
            if agent.agent_id not in self.agents:
                self.agents[agent.agent_id] = agent

    def _run_debate(
        self,
        task: str,
        trace: List[Dict],
    ) -> Dict[str, Any]:
        """Execute debate coordination pattern."""
        self._ensure_default_agents("debate")

        proposer = self.agents.get("proposer")
        critic = self.agents.get("critic")
        judge = self.agents.get("judge")

        if not all([proposer, critic, judge]):
            return {"error": "Debate requires proposer, critic, and judge agents"}

        current_proposal = ""

        for round_num in range(self.max_rounds):
            # Proposer generates or refines proposal
            if round_num == 0:
                proposal_prompt = f"Task: {task}\n\nGenerate an initial proposal to address this task."
            else:
                proposal_prompt = f"Task: {task}\n\nPrevious proposal: {current_proposal}\n\nCritique received: {critique}\n\nRefine your proposal based on the critique."

            proposal = self._get_agent_response(proposer, proposal_prompt, self.history.format_for_context())
            current_proposal = proposal

            self.history.add(AgentMessage(
                sender_id="proposer",
                receiver_id="critic",
                content=proposal,
                message_type=MessageType.PROPOSAL,
                round_num=round_num,
            ))

            trace.append({
                "action": "proposal",
                "round": round_num,
                "content_preview": proposal[:100],
            })

            # Critic critiques the proposal
            critique_prompt = f"Task: {task}\n\nProposal to critique:\n{proposal}\n\nProvide constructive criticism. Identify flaws, gaps, and suggest improvements."
            critique = self._get_agent_response(critic, critique_prompt, self.history.format_for_context())

            self.history.add(AgentMessage(
                sender_id="critic",
                receiver_id="proposer",
                content=critique,
                message_type=MessageType.CRITIQUE,
                round_num=round_num,
            ))

            trace.append({
                "action": "critique",
                "round": round_num,
                "content_preview": critique[:100],
            })

        # Judge makes final decision
        judge_prompt = f"""Task: {task}

Final proposal:
{current_proposal}

Debate history:
{self.history.format_for_context()}

As the judge, evaluate the final proposal. Consider:
1. Does it address the original task?
2. How well did it respond to critiques?
3. What is your final verdict?

Provide your judgment."""

        judgment = self._get_agent_response(judge, judge_prompt)

        self.history.add(AgentMessage(
            sender_id="judge",
            receiver_id="all",
            content=judgment,
            message_type=MessageType.JUDGMENT,
            round_num=self.max_rounds,
        ))

        trace.append({
            "action": "judgment",
            "content_preview": judgment[:100],
        })

        return {
            "final_proposal": current_proposal,
            "judgment": judgment,
            "rounds_completed": self.max_rounds,
            "message_count": len(self.history.messages),
        }

    def _run_division(
        self,
        task: str,
        trace: List[Dict],
    ) -> Dict[str, Any]:
        """Execute division coordination pattern."""
        self._ensure_default_agents("division")

        coordinator = self.agents.get("coordinator")
        workers = [a for a in self.agents.values() if a.role == "worker"]

        if not coordinator or not workers:
            return {"error": "Division requires coordinator and worker agents"}

        # Coordinator decomposes task
        decompose_prompt = f"""Task to decompose: {task}

You have {len(workers)} workers available. Break this task into subtasks that can be done in parallel.
List each subtask clearly, one per line, prefixed with "SUBTASK:"."""

        decomposition = self._get_agent_response(coordinator, decompose_prompt)

        # Parse subtasks (simple extraction)
        subtasks = []
        for line in decomposition.split('\n'):
            if 'SUBTASK:' in line.upper():
                subtask = line.split(':', 1)[-1].strip()
                if subtask:
                    subtasks.append(subtask)

        # If no structured subtasks found, create generic ones
        if not subtasks:
            subtasks = [f"Part {i+1} of the task" for i in range(len(workers))]

        trace.append({
            "action": "decompose",
            "subtask_count": len(subtasks),
            "subtasks": subtasks[:5],
        })

        # Assign subtasks to workers
        results = []
        for i, (worker, subtask) in enumerate(zip(workers, subtasks[:len(workers)])):
            self.history.add(AgentMessage(
                sender_id="coordinator",
                receiver_id=worker.agent_id,
                content=subtask,
                message_type=MessageType.TASK,
                round_num=0,
            ))

            worker_prompt = f"Original task: {task}\n\nYour assigned subtask: {subtask}\n\nComplete this subtask and provide your results."
            result = self._get_agent_response(worker, worker_prompt)

            results.append({
                "worker": worker.agent_id,
                "subtask": subtask,
                "result": result,
            })

            self.history.add(AgentMessage(
                sender_id=worker.agent_id,
                receiver_id="coordinator",
                content=result,
                message_type=MessageType.RESULT,
                round_num=0,
            ))

            trace.append({
                "action": "worker_result",
                "worker": worker.agent_id,
                "subtask_preview": subtask[:50],
            })

        # Coordinator aggregates results
        aggregate_prompt = f"""Original task: {task}

Worker results:
{chr(10).join(f"- {r['worker']}: {r['result'][:200]}" for r in results)}

Aggregate these results into a coherent final answer."""

        final_result = self._get_agent_response(coordinator, aggregate_prompt)

        trace.append({
            "action": "aggregate",
            "content_preview": final_result[:100],
        })

        return {
            "subtasks": subtasks,
            "worker_results": results,
            "aggregated_result": final_result,
            "workers_used": len(results),
        }

    def _run_hierarchical(
        self,
        task: str,
        trace: List[Dict],
    ) -> Dict[str, Any]:
        """Execute hierarchical coordination pattern."""
        self._ensure_default_agents("hierarchical")

        manager = self.agents.get("manager")
        workers = [a for a in self.agents.values() if a.role == "worker"]

        if not manager or not workers:
            return {"error": "Hierarchical requires manager and worker agents"}

        results_by_round = []

        for round_num in range(self.max_rounds):
            # Manager creates plan/instructions
            if round_num == 0:
                plan_prompt = f"Task: {task}\n\nCreate a plan and provide specific instructions for {len(workers)} workers."
            else:
                prev_results = results_by_round[-1] if results_by_round else []
                plan_prompt = f"""Task: {task}

Previous round results:
{chr(10).join(f"- {r['worker']}: {r['result'][:100]}" for r in prev_results)}

Review the results and provide updated instructions for the next round, or indicate if the task is complete."""

            plan = self._get_agent_response(manager, plan_prompt, self.history.format_for_context())

            trace.append({
                "action": "manager_plan",
                "round": round_num,
                "content_preview": plan[:100],
            })

            # Check if manager indicates completion
            if "complete" in plan.lower() and "task" in plan.lower():
                break

            # Workers execute instructions
            round_results = []
            for worker in workers:
                self.history.add(AgentMessage(
                    sender_id="manager",
                    receiver_id=worker.agent_id,
                    content=plan,
                    message_type=MessageType.TASK,
                    round_num=round_num,
                ))

                worker_prompt = f"Manager instructions:\n{plan}\n\nExecute your part of these instructions."
                result = self._get_agent_response(worker, worker_prompt)

                round_results.append({
                    "worker": worker.agent_id,
                    "result": result,
                })

                self.history.add(AgentMessage(
                    sender_id=worker.agent_id,
                    receiver_id="manager",
                    content=result,
                    message_type=MessageType.RESULT,
                    round_num=round_num,
                ))

            results_by_round.append(round_results)

            trace.append({
                "action": "workers_execute",
                "round": round_num,
                "workers_count": len(round_results),
            })

        # Manager provides final summary
        summary_prompt = f"""Task: {task}

All round results:
{chr(10).join(f"Round {i}: {[r['result'][:50] for r in results]}" for i, results in enumerate(results_by_round))}

Provide a final summary and conclusion."""

        final_summary = self._get_agent_response(manager, summary_prompt)

        return {
            "rounds_completed": len(results_by_round),
            "results_by_round": results_by_round,
            "final_summary": final_summary,
        }

    def _run_peer_review(
        self,
        task: str,
        trace: List[Dict],
    ) -> Dict[str, Any]:
        """Execute peer review coordination pattern."""
        self._ensure_default_agents("peer_review")

        author = self.agents.get("author")
        reviewer = self.agents.get("reviewer")

        if not author or not reviewer:
            return {"error": "Peer review requires author and reviewer agents"}

        # Author produces initial work
        author_prompt = f"Task: {task}\n\nProduce your initial work addressing this task."
        current_work = self._get_agent_response(author, author_prompt)

        self.history.add(AgentMessage(
            sender_id="author",
            receiver_id="reviewer",
            content=current_work,
            message_type=MessageType.PROPOSAL,
            round_num=0,
        ))

        trace.append({
            "action": "initial_work",
            "content_preview": current_work[:100],
        })

        review_history = []

        for round_num in range(self.max_rounds):
            # Reviewer reviews
            review_prompt = f"""Task: {task}

Work to review:
{current_work}

Provide detailed feedback. Be constructive and specific. If the work is satisfactory, say "APPROVED"."""

            review = self._get_agent_response(reviewer, review_prompt, self.history.format_for_context())

            self.history.add(AgentMessage(
                sender_id="reviewer",
                receiver_id="author",
                content=review,
                message_type=MessageType.REVIEW,
                round_num=round_num,
            ))

            review_history.append({
                "round": round_num,
                "review": review,
            })

            trace.append({
                "action": "review",
                "round": round_num,
                "content_preview": review[:100],
            })

            # Check if approved
            if "APPROVED" in review.upper():
                break

            # Author revises
            revision_prompt = f"""Task: {task}

Your previous work:
{current_work}

Reviewer feedback:
{review}

Revise your work based on the feedback."""

            revised_work = self._get_agent_response(author, revision_prompt, self.history.format_for_context())
            current_work = revised_work

            self.history.add(AgentMessage(
                sender_id="author",
                receiver_id="reviewer",
                content=revised_work,
                message_type=MessageType.REVISION,
                round_num=round_num,
            ))

            trace.append({
                "action": "revision",
                "round": round_num,
                "content_preview": revised_work[:100],
            })

        return {
            "final_work": current_work,
            "review_rounds": len(review_history),
            "review_history": review_history,
            "approved": "APPROVED" in review_history[-1]["review"].upper() if review_history else False,
        }

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        # Reset history for new run
        self.history = ConversationHistory()

        self._call_hooks("pre_run", task=task)

        trace.append({
            "action": "start",
            "coordination": self.coordination,
            "agents": list(self.agents.keys()),
            "max_rounds": self.max_rounds,
        })

        try:
            # Dispatch to appropriate pattern
            if self.coordination == "debate":
                result = self._run_debate(task, trace)
            elif self.coordination == "division":
                result = self._run_division(task, trace)
            elif self.coordination == "hierarchical":
                result = self._run_hierarchical(task, trace)
            elif self.coordination == "peer_review":
                result = self._run_peer_review(task, trace)
            else:
                return TechniqueResult(
                    success=False,
                    output=None,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    error=f"Unknown coordination pattern: {self.coordination}",
                )

            if "error" in result:
                return TechniqueResult(
                    success=False,
                    output=result,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    error=result["error"],
                )

            self._call_hooks("post_run", result=result)

            return TechniqueResult(
                success=True,
                output=result,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "coordination_pattern": self.coordination,
                    "message_count": len(self.history.messages),
                    "agents_used": list(self.agents.keys()),
                },
            )

        except Exception as e:
            self._call_hooks("on_error", error=e)
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# =============================================================================
# LATS (Language Agent Tree Search)
# =============================================================================

@dataclass
class TreeNode:
    """Node in the LATS search tree."""
    state: str
    action: Optional[str] = None
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    is_terminal: bool = False


class LATS(TechniqueBase):
    """
    LATS: Language Agent Tree Search.

    Paper: https://arxiv.org/abs/2310.04406 (2023)

    Applies Monte Carlo Tree Search (MCTS) to language agent reasoning.
    Balances exploration of new reasoning paths with exploitation of
    promising ones using UCB1 selection.

    Algorithm:
        1. Selection: Traverse tree using UCB1 to find promising leaf
        2. Expansion: Generate new child nodes (possible actions/thoughts)
        3. Evaluation: Estimate value of expanded node
        4. Backpropagation: Update values along path to root

    Configuration:
        backend: LLMBackend instance or name for generation
        num_simulations: Number of MCTS iterations
        exploration_weight: UCB1 exploration constant (higher = more exploration)
        max_depth: Maximum tree depth
        value_function: Optional custom value estimation function

    Usage:
        from ml_research.backends import MockBackend

        lats = LATS(
            backend=MockBackend(),
            num_simulations=50,
            exploration_weight=1.4,
            max_depth=10,
        )
        result = lats.run("Solve this complex reasoning problem...")
    """

    TECHNIQUE_ID = "lats"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        num_simulations: int = 50,
        exploration_weight: float = 1.4,
        max_depth: int = 10,
        value_function: Optional[Callable[[str], float]] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend, model)
        self.model = model  # Backward compatibility
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.value_function = value_function
        self.tools = {t.name: t for t in (tools or [])}

    def _ucb1_score(self, node: TreeNode, parent_visits: int) -> float:
        """Calculate UCB1 score for node selection."""
        import math
        if node.visits == 0:
            return float('inf')
        exploitation = node.value / node.visits
        exploration = self.exploration_weight * math.sqrt(
            math.log(parent_visits) / node.visits
        )
        return exploitation + exploration

    def _select(self, root: TreeNode) -> TreeNode:
        """Select a promising leaf node using UCB1."""
        node = root
        while node.children and not node.is_terminal:
            # Select child with highest UCB1 score
            node = max(
                node.children,
                key=lambda c: self._ucb1_score(c, node.visits)
            )
        return node

    def _expand(self, node: TreeNode, task: str) -> List[TreeNode]:
        """Expand node by generating possible actions/thoughts."""
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return []

        # Generate possible continuations using backend
        actions = self._generate_actions(node.state, task)
        children = []

        for action in actions:
            child = TreeNode(
                state=f"{node.state}\n{action}",
                action=action,
                parent=node,
                depth=node.depth + 1,
            )
            children.append(child)
            node.children.append(child)

        return children

    def _generate_actions(self, state: str, task: str) -> List[str]:
        """Generate possible actions from current state."""
        if self.backend is not None:
            prompt = f"Task: {task}\n\nCurrent state:\n{state}\n\nGenerate 3 possible next actions or thoughts, one per line:"
            if hasattr(self.backend, 'generate'):
                response = self.backend.generate(prompt, max_tokens=256, temperature=0.8)
                return [line.strip() for line in response.split('\n') if line.strip()][:3]
            elif callable(self.backend):
                response = self.backend(prompt)
                return [line.strip() for line in response.split('\n') if line.strip()][:3]

        # Placeholder actions
        return [
            f"Think about approach {node_id}" for node_id in range(1, 4)
        ]

    def _evaluate(self, node: TreeNode, task: str) -> float:
        """Evaluate the value of a node."""
        if self.value_function:
            return self.value_function(node.state)

        # Use backend for value estimation
        if self.backend is not None:
            prompt = f"Task: {task}\n\nReasoning so far:\n{node.state}\n\nRate progress toward solution (0.0 to 1.0):"
            if hasattr(self.backend, 'generate'):
                response = self.backend.generate(prompt, max_tokens=32, temperature=0.3)
                try:
                    return float(response.strip().split()[0])
                except (ValueError, IndexError):
                    return 0.5

        return 0.5  # Default value

    def _backpropagate(self, node: TreeNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _is_solution(self, node: TreeNode, task: str) -> Tuple[bool, Optional[str]]:
        """Check if node represents a solution."""
        if "Final Answer:" in node.state:
            answer = node.state.split("Final Answer:")[-1].strip()
            return True, answer
        return False, None

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", task=task)

        # Initialize root
        root = TreeNode(state=f"Task: {task}", depth=0)
        best_solution = None
        best_value = -float('inf')

        for sim in range(self.num_simulations):
            # Selection
            leaf = self._select(root)

            # Check if solution found
            is_solution, answer = self._is_solution(leaf, task)
            if is_solution:
                value = self._evaluate(leaf, task)
                if value > best_value:
                    best_value = value
                    best_solution = answer
                self._backpropagate(leaf, value)
                trace.append({
                    "action": "found_solution",
                    "simulation": sim,
                    "value": value,
                })
                continue

            # Expansion
            if not leaf.is_terminal:
                children = self._expand(leaf, task)
                if children:
                    leaf = children[0]  # Evaluate first child

            # Evaluation
            value = self._evaluate(leaf, task)

            # Backpropagation
            self._backpropagate(leaf, value)

            trace.append({
                "action": "simulation",
                "simulation": sim,
                "depth": leaf.depth,
                "value": value,
            })

            self._call_hooks("on_step", simulation=sim, value=value)

        self._call_hooks("post_run", solution=best_solution)

        if best_solution:
            return TechniqueResult(
                success=True,
                output=best_solution,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "simulations": self.num_simulations,
                    "best_value": best_value,
                    "tree_size": root.visits,
                },
            )

        # Return best path if no explicit solution
        best_leaf = self._select(root)
        return TechniqueResult(
            success=False,
            output=best_leaf.state,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error="No explicit solution found",
            metadata={"simulations": self.num_simulations},
        )


# =============================================================================
# ReWOO (Reasoning WithOut Observation)
# =============================================================================

@dataclass
class ReWOOPlan:
    """A plan generated by the ReWOO planner."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)


class ReWOO(TechniqueBase):
    """
    ReWOO: Reasoning WithOut Observation.

    Paper: https://arxiv.org/abs/2305.18323 (2023)

    Plans all tool calls upfront before execution, reducing token usage
    by not interleaving reasoning with observations. Uses a three-stage
    Planner -> Worker -> Solver pattern.

    Algorithm:
        1. Planner: Generate complete plan with tool calls and variable references
        2. Worker: Execute all tool calls, filling in variable values
        3. Solver: Synthesize final answer from plan and results

    Configuration:
        backend: LLMBackend instance or name for generation
        planner_prompt: System prompt for planning stage
        worker_prompt: System prompt for execution stage
        solver_prompt: System prompt for synthesis stage
        tools: Available tools for the worker

    Usage:
        from ml_research.backends import MockBackend

        rewoo = ReWOO(
            backend=MockBackend(),
            tools=[search_tool, calculator_tool],
        )
        result = rewoo.run("What is the population of Tokyo divided by the area of Japan?")
    """

    TECHNIQUE_ID = "rewoo"
    CATEGORY = TechniqueCategory.AGENTIC

    DEFAULT_PLANNER_PROMPT = """You are a planning agent. Given a task, create a step-by-step plan.
For each step, specify:
- The tool to use (or "think" for reasoning)
- The input to the tool (use #E[n] to reference previous step results)
- A brief description

Format each step as:
Plan: <description>
#E[n] = Tool[input]
"""

    DEFAULT_SOLVER_PROMPT = """You are a solver agent. Given the original task and the results
of executing a plan, synthesize the final answer."""

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        planner_prompt: Optional[str] = None,
        worker_prompt: Optional[str] = None,
        solver_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend, model)
        self.model = model
        self.planner_prompt = planner_prompt or self.DEFAULT_PLANNER_PROMPT
        self.worker_prompt = worker_prompt or ""
        self.solver_prompt = solver_prompt or self.DEFAULT_SOLVER_PROMPT
        self.tools = {t.name: t for t in (tools or [])}

    def _plan(self, task: str) -> ReWOOPlan:
        """Generate execution plan upfront."""
        import re

        prompt = f"{self.planner_prompt}\n\nTask: {task}\n\nAvailable tools: {list(self.tools.keys())}\n\nPlan:"

        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                response = self.backend.generate(prompt, max_tokens=1024, temperature=0.7)
            elif callable(self.backend):
                response = self.backend(prompt)
            else:
                response = ""
        else:
            response = ""

        # Parse plan steps
        plan = ReWOOPlan()
        pattern = r'#E\[(\d+)\]\s*=\s*(\w+)\[([^\]]*)\]'
        for match in re.finditer(pattern, response):
            step_num, tool_name, tool_input = match.groups()
            plan.steps.append({
                "step_id": f"E{step_num}",
                "tool": tool_name,
                "input": tool_input,
                "result": None,
            })

        # If no steps parsed, create placeholder
        if not plan.steps:
            plan.steps = [
                {"step_id": "E1", "tool": "think", "input": task, "result": None}
            ]

        return plan

    def _execute_plan(self, plan: ReWOOPlan) -> ReWOOPlan:
        """Execute all planned tool calls (Worker stage)."""
        import re

        for step in plan.steps:
            # Substitute variable references
            tool_input = step["input"]
            for var_name, var_value in plan.variables.items():
                tool_input = tool_input.replace(f"#{var_name}", str(var_value))

            # Execute tool
            tool_name = step["tool"].lower()
            if tool_name == "think":
                # Reasoning step - use backend
                if self.backend is not None and hasattr(self.backend, 'generate'):
                    result = self.backend.generate(f"Think about: {tool_input}", max_tokens=256)
                else:
                    result = f"[Thinking about: {tool_input}]"
            elif tool_name in self.tools:
                tool = self.tools[tool_name]
                if tool.execute:
                    try:
                        result = tool.execute(query=tool_input)
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"[{tool_name} result for: {tool_input}]"
            else:
                result = f"[Unknown tool: {tool_name}]"

            step["result"] = result
            plan.variables[step["step_id"]] = result

        return plan

    def _solve(self, task: str, plan: ReWOOPlan) -> str:
        """Synthesize final answer from plan results (Solver stage)."""
        # Build context from plan execution
        context_parts = [f"Task: {task}\n\nPlan execution results:"]
        for step in plan.steps:
            context_parts.append(f"\n{step['step_id']}: {step['tool']}[{step['input']}]")
            context_parts.append(f"Result: {step['result']}")

        prompt = f"{self.solver_prompt}\n\n{''.join(context_parts)}\n\nFinal Answer:"

        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=512, temperature=0.5)
            elif callable(self.backend):
                return self.backend(prompt)

        # Aggregate results as fallback
        return f"Based on plan execution: {plan.steps[-1]['result'] if plan.steps else 'No results'}"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", task=task)

        # Stage 1: Planning
        plan = self._plan(task)
        trace.append({
            "action": "plan",
            "num_steps": len(plan.steps),
            "steps": [f"{s['step_id']}: {s['tool']}" for s in plan.steps],
        })

        self._call_hooks("on_step", stage="plan", plan=plan)

        # Stage 2: Worker (execution)
        plan = self._execute_plan(plan)
        trace.append({
            "action": "execute",
            "results": [{"step": s["step_id"], "result": str(s["result"])[:100]} for s in plan.steps],
        })

        self._call_hooks("on_step", stage="worker", plan=plan)

        # Stage 3: Solver
        answer = self._solve(task, plan)
        trace.append({
            "action": "solve",
            "answer_preview": answer[:200] if answer else None,
        })

        self._call_hooks("post_run", answer=answer)

        return TechniqueResult(
            success=True,
            output=answer,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            metadata={
                "plan_steps": len(plan.steps),
                "variables": list(plan.variables.keys()),
            },
        )


# =============================================================================
# InnerMonologue
# =============================================================================

@dataclass
class MonologueEntry:
    """An entry in the inner monologue trace."""
    thought: str
    feedback_source: Optional[str] = None
    feedback: Optional[str] = None
    action: Optional[str] = None
    timestamp: float = 0.0


class InnerMonologue(TechniqueBase):
    """
    Inner Monologue: Embodied Reasoning through Planning with Language Models.

    Paper: https://arxiv.org/abs/2207.05608 (2022)

    Maintains an internal reasoning trace for embodied agents, incorporating
    feedback from the environment (success detection, scene description,
    human feedback) into the reasoning process.

    Originally developed for robotics and embodied AI, this pattern is
    useful for any agent that needs to reason about environmental feedback.

    Configuration:
        backend: LLMBackend instance or name for generation
        feedback_sources: List of feedback source names (e.g., ["success_detector", "scene_description", "human"])
        monologue_prompt: System prompt for inner monologue generation
        max_steps: Maximum reasoning steps

    Usage:
        from ml_research.backends import MockBackend

        inner = InnerMonologue(
            backend=MockBackend(),
            feedback_sources=["success_detector", "scene_description"],
        )
        result = inner.run({
            "task": "Pick up the red cup",
            "environment": mock_environment,
        })
    """

    TECHNIQUE_ID = "inner_monologue"
    CATEGORY = TechniqueCategory.AGENTIC

    DEFAULT_MONOLOGUE_PROMPT = """You are an embodied agent reasoning about how to complete a task.
Maintain an inner monologue that:
1. Observes the current state
2. Reasons about what to do next
3. Incorporates feedback from the environment
4. Decides on the next action

Always format your response as:
Thought: <your reasoning>
Action: <your next action>
"""

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        feedback_sources: Optional[List[str]] = None,
        monologue_prompt: Optional[str] = None,
        max_steps: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend, model)
        self.model = model
        self.feedback_sources = feedback_sources or ["success_detector", "scene_description"]
        self.monologue_prompt = monologue_prompt or self.DEFAULT_MONOLOGUE_PROMPT
        self.max_steps = max_steps

    def _get_feedback(
        self,
        source: str,
        action: str,
        environment: Optional[Any],
    ) -> Optional[str]:
        """Get feedback from a specific source."""
        if environment is None:
            return f"[{source}: No environment provided]"

        # Try to get feedback from environment
        if hasattr(environment, 'get_feedback'):
            return environment.get_feedback(source, action)
        elif hasattr(environment, source):
            feedback_fn = getattr(environment, source)
            if callable(feedback_fn):
                return feedback_fn(action)

        return f"[{source}: Feedback unavailable]"

    def _generate_monologue(
        self,
        task: str,
        history: List[MonologueEntry],
        current_feedback: Dict[str, str],
    ) -> Tuple[str, str]:
        """Generate next thought and action."""
        # Build prompt with history and feedback
        prompt_parts = [self.monologue_prompt, f"\nTask: {task}\n"]

        for entry in history[-5:]:  # Last 5 entries for context
            prompt_parts.append(f"\nThought: {entry.thought}")
            if entry.action:
                prompt_parts.append(f"Action: {entry.action}")
            if entry.feedback:
                prompt_parts.append(f"Feedback ({entry.feedback_source}): {entry.feedback}")

        if current_feedback:
            prompt_parts.append("\nCurrent feedback:")
            for source, feedback in current_feedback.items():
                prompt_parts.append(f"  {source}: {feedback}")

        prompt_parts.append("\nThought:")
        prompt = "\n".join(prompt_parts)

        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                response = self.backend.generate(prompt, max_tokens=256, temperature=0.7)
            elif callable(self.backend):
                response = self.backend(prompt)
            else:
                response = "I need to complete the task.\nAction: proceed"
        else:
            response = "I need to complete the task.\nAction: proceed"

        # Parse thought and action
        thought = response
        action = None
        if "Action:" in response:
            parts = response.split("Action:")
            thought = parts[0].strip()
            action = parts[1].strip().split("\n")[0]

        return thought, action

    def _check_success(self, environment: Optional[Any], task: str) -> bool:
        """Check if task is completed."""
        if environment is None:
            return False
        if hasattr(environment, 'is_task_complete'):
            return environment.is_task_complete(task)
        return False

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        # Parse input
        if isinstance(input_data, dict):
            task = input_data.get("task", str(input_data))
            environment = input_data.get("environment")
        else:
            task = str(input_data)
            environment = None

        trace: List[Dict] = []
        history: List[MonologueEntry] = []

        self._call_hooks("pre_run", task=task)

        for step in range(self.max_steps):
            # Get feedback from all sources (based on last action)
            last_action = history[-1].action if history else None
            current_feedback = {}
            for source in self.feedback_sources:
                feedback = self._get_feedback(source, last_action, environment)
                if feedback:
                    current_feedback[source] = feedback

            # Generate next thought and action
            thought, action = self._generate_monologue(task, history, current_feedback)

            # Create monologue entry
            entry = MonologueEntry(
                thought=thought,
                action=action,
                feedback_source=list(current_feedback.keys())[0] if current_feedback else None,
                feedback=list(current_feedback.values())[0] if current_feedback else None,
                timestamp=time.time() - start,
            )
            history.append(entry)

            trace.append({
                "action": "monologue_step",
                "step": step,
                "thought": thought[:100],
                "action": action,
                "feedback": current_feedback,
            })

            self._call_hooks("on_step", entry=entry)

            # Check for task completion
            if self._check_success(environment, task):
                self._call_hooks("post_run", success=True)
                return TechniqueResult(
                    success=True,
                    output={
                        "final_thought": thought,
                        "final_action": action,
                        "monologue": [{"thought": e.thought, "action": e.action} for e in history],
                    },
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    metadata={"steps": step + 1},
                )

            # Check for explicit completion
            if action and "done" in action.lower():
                return TechniqueResult(
                    success=True,
                    output={
                        "final_thought": thought,
                        "monologue": [{"thought": e.thought, "action": e.action} for e in history],
                    },
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    metadata={"steps": step + 1},
                )

        # Max steps reached
        return TechniqueResult(
            success=False,
            output={
                "monologue": [{"thought": e.thought, "action": e.action} for e in history],
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            error=f"Max steps ({self.max_steps}) reached",
        )


# =============================================================================
# Toolformer
# =============================================================================

@dataclass
class ToolToken:
    """A special token for tool invocation."""
    tool_name: str
    start_token: str
    end_token: str
    args_separator: str = ","


class Toolformer(TechniqueBase):
    """
    Toolformer: Language Models Can Teach Themselves to Use Tools.

    Paper: https://arxiv.org/abs/2302.04761 (2023)

    Models learn when and how to call APIs/tools through self-supervised
    learning. The model learns to insert API calls into text where they
    would improve prediction.

    Key insight: The model decides when tool use would be beneficial
    based on whether the tool result reduces perplexity.

    Configuration:
        backend: LLMBackend instance or name for generation
        available_tools: Tools the model can learn to use
        tool_tokens: Special tokens for each tool (e.g., "[Calculator(", ")]")
        insertion_threshold: Perplexity reduction threshold for keeping tool calls

    Usage:
        from ml_research.backends import MockBackend

        toolformer = Toolformer(
            backend=MockBackend(),
            available_tools=[calculator_tool, qa_tool],
            insertion_threshold=0.5,
        )
        result = toolformer.run("The population of France is")
    """

    TECHNIQUE_ID = "toolformer"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        available_tools: Optional[List[ToolSpec]] = None,
        tool_tokens: Optional[Dict[str, ToolToken]] = None,
        insertion_threshold: float = 0.5,
        max_tool_calls: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend, model)
        self.model = model
        self.available_tools = {t.name: t for t in (available_tools or [])}
        self.insertion_threshold = insertion_threshold
        self.max_tool_calls = max_tool_calls

        # Set up tool tokens
        self.tool_tokens = tool_tokens or {}
        for tool_name in self.available_tools:
            if tool_name not in self.tool_tokens:
                self.tool_tokens[tool_name] = ToolToken(
                    tool_name=tool_name,
                    start_token=f"[{tool_name}(",
                    end_token=")]",
                )

    def _find_tool_positions(self, text: str) -> List[Dict[str, Any]]:
        """Find positions where tool calls might be beneficial."""
        import re

        positions = []

        # Look for patterns that suggest tool use
        # Numbers/calculations -> calculator
        if "calculator" in self.available_tools:
            for match in re.finditer(r'\d+\s*[\+\-\*\/]\s*\d+', text):
                positions.append({
                    "position": match.start(),
                    "tool": "calculator",
                    "context": match.group(),
                })

        # Questions -> QA
        if "qa" in self.available_tools:
            for match in re.finditer(r'\?\s*$', text):
                positions.append({
                    "position": match.start(),
                    "tool": "qa",
                    "context": text[:match.start()],
                })

        # Factual queries -> search
        if "search" in self.available_tools:
            factual_patterns = [
                r'the (population|capital|president|CEO) of',
                r'when was .+ (born|founded|created)',
                r'who (is|was|invented|discovered)',
            ]
            for pattern in factual_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    positions.append({
                        "position": match.start(),
                        "tool": "search",
                        "context": match.group(),
                    })

        return positions

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """Execute a tool call and return result."""
        if tool_name not in self.available_tools:
            return f"[Unknown tool: {tool_name}]"

        tool = self.available_tools[tool_name]
        if tool.execute is None:
            return f"[{tool_name} result for: {args}]"

        try:
            # Simple argument parsing
            result = tool.execute(query=args)
            return str(result)
        except Exception as e:
            return f"[Error: {e}]"

    def _insert_tool_call(
        self,
        text: str,
        position: int,
        tool_name: str,
        args: str,
        result: str,
    ) -> str:
        """Insert tool call and result into text."""
        token = self.tool_tokens[tool_name]
        tool_call = f"{token.start_token}{args}{token.end_token} -> {result}"
        return text[:position] + tool_call + " " + text[position:]

    def _evaluate_insertion(
        self,
        original_text: str,
        augmented_text: str,
    ) -> float:
        """Evaluate if tool insertion improves the text (simplified)."""
        # In real Toolformer, this computes perplexity reduction
        # Simplified: check if result adds useful information
        if self.backend is not None and hasattr(self.backend, 'generate'):
            # Generate continuation for both and compare
            orig_cont = self.backend.generate(original_text, max_tokens=50)
            aug_cont = self.backend.generate(augmented_text, max_tokens=50)
            # Simplified scoring
            return 0.6 if len(aug_cont) >= len(orig_cont) else 0.4

        return 0.5  # Neutral

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        text = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", text=text)

        # Find potential tool insertion points
        positions = self._find_tool_positions(text)
        trace.append({
            "action": "find_positions",
            "num_positions": len(positions),
            "positions": [p["tool"] for p in positions],
        })

        # Process each position (limited by max_tool_calls)
        augmented_text = text
        insertions = []
        offset = 0  # Track text length changes

        for pos_info in positions[:self.max_tool_calls]:
            tool_name = pos_info["tool"]
            context_str = pos_info["context"]
            position = pos_info["position"] + offset

            # Execute tool
            result = self._execute_tool(tool_name, context_str)

            # Create augmented text
            test_text = self._insert_tool_call(
                augmented_text, position, tool_name, context_str, result
            )

            # Evaluate if insertion is beneficial
            score = self._evaluate_insertion(augmented_text, test_text)

            trace.append({
                "action": "evaluate_insertion",
                "tool": tool_name,
                "score": score,
                "threshold": self.insertion_threshold,
                "kept": score >= self.insertion_threshold,
            })

            self._call_hooks("on_step", tool=tool_name, score=score)

            if score >= self.insertion_threshold:
                augmented_text = test_text
                token = self.tool_tokens[tool_name]
                insertion_len = len(f"{token.start_token}{context_str}{token.end_token} -> {result} ")
                offset += insertion_len
                insertions.append({
                    "tool": tool_name,
                    "args": context_str,
                    "result": result,
                    "score": score,
                })

        self._call_hooks("post_run", augmented_text=augmented_text)

        return TechniqueResult(
            success=True,
            output=augmented_text,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            metadata={
                "num_insertions": len(insertions),
                "insertions": insertions,
                "original_text": text,
            },
        )


# =============================================================================
# CRITIC
# =============================================================================

@dataclass
class CritiqueResult:
    """Result of a critique step."""
    original_output: str
    critique: str
    issues_found: List[str]
    suggested_corrections: List[str]
    verification_results: Dict[str, Any] = field(default_factory=dict)


class CRITIC(TechniqueBase):
    """
    CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.

    Paper: https://arxiv.org/abs/2305.11738 (2023)

    Interleaves generation and critique, allowing the model to self-correct
    through tool-based verification. After initial generation, the model
    critiques its output using external tools, then refines based on feedback.

    Algorithm:
        1. Generate initial response
        2. Critique response using verification tools
        3. If issues found, generate corrected response
        4. Repeat until no issues or max iterations

    Configuration:
        backend: LLMBackend instance or name for generation
        critique_prompt: System prompt for critique generation
        max_iterations: Maximum critique-correction cycles
        tools_for_verification: Tools for verifying claims (calculator, search, code_exec)

    Usage:
        from ml_research.backends import MockBackend

        critic = CRITIC(
            backend=MockBackend(),
            tools_for_verification=[calculator_tool, search_tool],
            max_iterations=3,
        )
        result = critic.run("What is 15% of 847?")
    """

    TECHNIQUE_ID = "critic"
    CATEGORY = TechniqueCategory.AGENTIC

    DEFAULT_CRITIQUE_PROMPT = """Review the following response for accuracy and completeness.
Identify any:
1. Factual errors
2. Calculation mistakes
3. Logical inconsistencies
4. Missing information

For each issue, suggest a correction. Use available tools to verify claims.

Format:
Issue: <description>
Verification: <tool to use and how>
Correction: <suggested fix>
"""

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        critique_prompt: Optional[str] = None,
        max_iterations: int = 3,
        tools_for_verification: Optional[List[ToolSpec]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend, model)
        self.model = model
        self.critique_prompt = critique_prompt or self.DEFAULT_CRITIQUE_PROMPT
        self.max_iterations = max_iterations
        self.verification_tools = {t.name: t for t in (tools_for_verification or [])}

    def _generate_initial(self, task: str) -> str:
        """Generate initial response."""
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(task, max_tokens=512, temperature=0.7)
            elif callable(self.backend):
                return self.backend(task)

        return f"[Initial response for: {task}]"

    def _critique(self, task: str, response: str) -> CritiqueResult:
        """Generate critique of response."""
        import re

        prompt = f"{self.critique_prompt}\n\nTask: {task}\n\nResponse to critique:\n{response}\n\nCritique:"

        if self.backend is not None and hasattr(self.backend, 'generate'):
            critique_text = self.backend.generate(prompt, max_tokens=512, temperature=0.5)
        else:
            critique_text = "No issues found."

        # Parse critique
        issues = re.findall(r'Issue:\s*(.+?)(?=\n|$)', critique_text)
        corrections = re.findall(r'Correction:\s*(.+?)(?=\n|$)', critique_text)
        verifications = re.findall(r'Verification:\s*(.+?)(?=\n|$)', critique_text)

        # Execute verifications
        verification_results = {}
        for verification in verifications:
            # Try to identify which tool to use
            for tool_name, tool in self.verification_tools.items():
                if tool_name.lower() in verification.lower():
                    if tool.execute:
                        try:
                            result = tool.execute(query=verification)
                            verification_results[verification] = result
                        except Exception as e:
                            verification_results[verification] = f"Error: {e}"
                    break

        return CritiqueResult(
            original_output=response,
            critique=critique_text,
            issues_found=issues,
            suggested_corrections=corrections,
            verification_results=verification_results,
        )

    def _refine(
        self,
        task: str,
        previous_response: str,
        critique: CritiqueResult,
    ) -> str:
        """Generate refined response based on critique."""
        prompt = f"""Task: {task}

Previous response:
{previous_response}

Critique found these issues:
{chr(10).join(f'- {issue}' for issue in critique.issues_found)}

Suggested corrections:
{chr(10).join(f'- {corr}' for corr in critique.suggested_corrections)}

Verification results:
{chr(10).join(f'- {k}: {v}' for k, v in critique.verification_results.items())}

Generate an improved response that addresses these issues:"""

        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=512, temperature=0.5)
            elif callable(self.backend):
                return self.backend(prompt)

        return f"[Refined response incorporating: {critique.suggested_corrections}]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", task=task)

        # Generate initial response
        response = self._generate_initial(task)
        trace.append({
            "action": "generate_initial",
            "response_preview": response[:200],
        })

        critiques: List[CritiqueResult] = []

        for iteration in range(self.max_iterations):
            # Critique current response
            critique = self._critique(task, response)
            critiques.append(critique)

            trace.append({
                "action": "critique",
                "iteration": iteration,
                "num_issues": len(critique.issues_found),
                "issues": critique.issues_found[:3],  # First 3
            })

            self._call_hooks("on_step", iteration=iteration, critique=critique)

            # If no issues, we're done
            if not critique.issues_found:
                trace.append({
                    "action": "no_issues_found",
                    "iteration": iteration,
                })
                break

            # Refine response
            response = self._refine(task, response, critique)
            trace.append({
                "action": "refine",
                "iteration": iteration,
                "response_preview": response[:200],
            })

        self._call_hooks("post_run", final_response=response)

        return TechniqueResult(
            success=True,
            output=response,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            metadata={
                "iterations": len(critiques),
                "total_issues_found": sum(len(c.issues_found) for c in critiques),
                "critiques": [
                    {"issues": c.issues_found, "corrections": c.suggested_corrections}
                    for c in critiques
                ],
            },
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CoordinationPattern",
    "MessageType",
    # Data classes
    "ToolParameter",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "PlanStep",
    "AgentSpec",
    "AgentMessage",
    "ConversationHistory",
    "TreeNode",
    "ReWOOPlan",
    "MonologueEntry",
    "ToolToken",
    "CritiqueResult",
    # Utility classes
    "ConsensusProtocol",
    # Techniques
    "ReAct",
    "ToolCalling",
    "Planning",
    "Reflexion",
    "MultiAgent",
    "LATS",
    "ReWOO",
    "InnerMonologue",
    "Toolformer",
    "CRITIC",
]
