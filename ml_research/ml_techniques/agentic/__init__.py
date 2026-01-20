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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import abstractmethod

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


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
        tools: List of available tools
        max_iterations: Maximum thought-action cycles
        observation_format: Format string for observations
        thought_prefix: Prefix for thought generation
        stop_sequences: Sequences that indicate completion

    Usage:
        react = ReAct(
            model=my_model,
            tools=[search_tool, calculator_tool],
            max_iterations=10,
        )
        result = react.run("What is the population of Tokyo divided by 1000?")
    """

    TECHNIQUE_ID = "react"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        model: Optional[Any] = None,
        tools: Optional[List[ToolSpec]] = None,
        max_iterations: int = 10,
        thought_prefix: str = "Thought:",
        action_prefix: str = "Action:",
        observation_prefix: str = "Observation:",
        final_answer_prefix: str = "Final Answer:",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
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
        """Generate next thought and action (placeholder)."""
        # Real implementation uses LLM
        if not history:
            return f"{self.thought_prefix} I need to solve: {task}\n{self.action_prefix} search(\"{task}\")"
        return f"{self.final_answer_prefix} [Placeholder answer]"

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

@dataclass
class AgentSpec:
    """Specification for an agent in multi-agent system."""
    agent_id: str
    role: str
    description: str
    model: Optional[Any] = None
    tools: List[ToolSpec] = field(default_factory=list)
    system_prompt: str = ""


class MultiAgent(TechniqueBase):
    """
    Multi-agent collaboration patterns.

    Patterns:
        - Debate: Agents argue positions, judge decides
        - Division: Specialized agents handle subtasks
        - Hierarchical: Manager coordinates workers
        - Peer review: Agents check each other's work

    Configuration:
        agents: List of agent specifications
        coordination: Coordination pattern
        max_rounds: Maximum interaction rounds
    """

    TECHNIQUE_ID = "multi_agent"
    CATEGORY = TechniqueCategory.AGENTIC

    def __init__(
        self,
        agents: Optional[List[AgentSpec]] = None,
        coordination: str = "hierarchical",  # hierarchical, debate, peer, division
        max_rounds: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agents = {a.agent_id: a for a in (agents or [])}
        self.coordination = coordination
        self.max_rounds = max_rounds

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        # Placeholder implementation
        return TechniqueResult(
            success=True,
            output={"placeholder": "MultiAgent not yet fully implemented"},
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=[],
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "ToolParameter",
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "PlanStep",
    "AgentSpec",
    # Techniques
    "ReAct",
    "ToolCalling",
    "Planning",
    "Reflexion",
    "MultiAgent",
]
