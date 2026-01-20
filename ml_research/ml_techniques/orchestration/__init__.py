"""
Orchestration Techniques

Patterns for coordinating multiple components, routing tasks,
and managing execution flow with intervention points (hooks).

=============================================================================
TECHNIQUES
=============================================================================

1. HooksSystem
   - Intervention points in execution
   - Pre/post/error/conditional hooks
   - Middleware-style processing

2. TaskRouting
   - Route tasks to appropriate handlers
   - Classifier, embedding, rule-based, LLM-based routing

3. Ensemble
   - Combine multiple model/technique outputs
   - Voting, averaging, stacking, mixture

4. ConditionalExecution
   - Branch execution based on conditions
   - Guard clauses, feature flags, A/B testing

=============================================================================
HOOKS SYSTEM
=============================================================================

Hook Types:
    pre_run      - Before execution starts
    post_run     - After successful execution
    on_error     - When an error occurs
    on_step      - At each intermediate step
    transform_input  - Modify input before processing
    transform_output - Modify output before returning
    conditional  - Conditionally branch execution

Hook Registration:
    technique.add_hook("pre_run", my_hook_fn)
    technique.add_hook("transform_output", lambda x: x.upper())

Hook Signature:
    def my_hook(*, input=None, output=None, error=None, step=None, **kwargs):
        # Return modified value or None to pass through
        return modified_value
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from enum import Enum
from abc import abstractmethod
import time

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


# =============================================================================
# HOOK TYPES
# =============================================================================

class HookType(Enum):
    """Types of hooks available in the system."""
    PRE_RUN = "pre_run"
    POST_RUN = "post_run"
    ON_ERROR = "on_error"
    ON_STEP = "on_step"
    TRANSFORM_INPUT = "transform_input"
    TRANSFORM_OUTPUT = "transform_output"
    CONDITIONAL = "conditional"
    ON_RETRY = "on_retry"
    ON_TIMEOUT = "on_timeout"


@dataclass
class Hook:
    """A hook definition with metadata."""
    hook_type: HookType
    function: Callable
    name: str = ""
    priority: int = 0  # Higher = runs first
    enabled: bool = True
    condition: Optional[Callable[..., bool]] = None  # Only run if condition returns True


@dataclass
class HookContext:
    """Context passed to hook functions."""
    technique_id: str
    hook_type: HookType
    input_data: Any = None
    output_data: Any = None
    error: Optional[Exception] = None
    step_info: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


# =============================================================================
# HOOKS SYSTEM
# =============================================================================

class HooksSystem(TechniqueBase):
    """
    Comprehensive hooks system for execution intervention.

    Provides middleware-style processing with multiple hook points,
    priorities, conditions, and error handling.

    Hook Types:
        - pre_run: Before main execution
        - post_run: After successful execution
        - on_error: When an error occurs
        - on_step: At each intermediate step
        - transform_input: Modify input
        - transform_output: Modify output
        - conditional: Branch based on conditions
        - on_retry: Before retry attempts
        - on_timeout: When timeout occurs

    Configuration:
        hooks: List of Hook definitions
        async_hooks: Whether to run hooks asynchronously
        continue_on_hook_error: Continue if a hook fails
        timeout_ms: Timeout for hook execution

    Usage:
        system = HooksSystem()

        # Add logging hook
        system.add_hook(Hook(
            hook_type=HookType.PRE_RUN,
            function=lambda ctx: print(f"Starting: {ctx.input_data}"),
            name="logger",
        ))

        # Add input transformation
        system.add_hook(Hook(
            hook_type=HookType.TRANSFORM_INPUT,
            function=lambda ctx: ctx.input_data.upper(),
            name="uppercase",
        ))

        # Add conditional hook
        system.add_hook(Hook(
            hook_type=HookType.CONDITIONAL,
            function=lambda ctx: "route_a" if "urgent" in str(ctx.input_data) else "route_b",
            name="router",
        ))

        result = system.run("process this input", inner_technique=my_technique)
    """

    TECHNIQUE_ID = "hooks_system"
    CATEGORY = TechniqueCategory.ORCHESTRATION

    def __init__(
        self,
        hooks: Optional[List[Hook]] = None,
        async_hooks: bool = False,
        continue_on_hook_error: bool = True,
        timeout_ms: int = 5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hooks: Dict[HookType, List[Hook]] = {ht: [] for ht in HookType}
        self.async_hooks = async_hooks
        self.continue_on_hook_error = continue_on_hook_error
        self.timeout_ms = timeout_ms

        # Register initial hooks
        for hook in (hooks or []):
            self.add_hook(hook)

    def add_hook(self, hook: Hook) -> None:
        """Add a hook to the system."""
        self._hooks[hook.hook_type].append(hook)
        # Sort by priority (higher first)
        self._hooks[hook.hook_type].sort(key=lambda h: -h.priority)

    def remove_hook(self, name: str) -> bool:
        """Remove a hook by name."""
        for hook_type in self._hooks:
            self._hooks[hook_type] = [
                h for h in self._hooks[hook_type] if h.name != name
            ]
        return True

    def enable_hook(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a hook by name."""
        for hook_list in self._hooks.values():
            for hook in hook_list:
                if hook.name == name:
                    hook.enabled = enabled

    def _execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> Tuple[Any, List[Dict]]:
        """Execute all hooks of a given type."""
        results = []
        current_value = context.input_data if hook_type == HookType.TRANSFORM_INPUT else context.output_data

        for hook in self._hooks[hook_type]:
            if not hook.enabled:
                continue

            # Check condition
            if hook.condition and not hook.condition(context):
                continue

            try:
                start = time.time()
                result = hook.function(context)
                elapsed = (time.time() - start) * 1000

                results.append({
                    "hook": hook.name or hook.hook_type.value,
                    "success": True,
                    "time_ms": elapsed,
                })

                # Transform hooks update the value
                if hook_type in (HookType.TRANSFORM_INPUT, HookType.TRANSFORM_OUTPUT):
                    if result is not None:
                        current_value = result

            except Exception as e:
                results.append({
                    "hook": hook.name or hook.hook_type.value,
                    "success": False,
                    "error": str(e),
                })

                if not self.continue_on_hook_error:
                    raise

        return current_value, results

    def run(
        self,
        input_data: Any,
        context: Optional[Dict] = None,
        inner_technique: Optional[TechniqueBase] = None,
    ) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []
        hook_context = HookContext(
            technique_id=self.TECHNIQUE_ID,
            hook_type=HookType.PRE_RUN,
            input_data=input_data,
        )

        try:
            # Pre-run hooks
            _, pre_results = self._execute_hooks(HookType.PRE_RUN, hook_context)
            trace.append({"action": "pre_run_hooks", "results": pre_results})

            # Transform input
            hook_context.hook_type = HookType.TRANSFORM_INPUT
            transformed_input, transform_results = self._execute_hooks(
                HookType.TRANSFORM_INPUT, hook_context
            )
            trace.append({"action": "transform_input", "results": transform_results})

            # Execute inner technique if provided
            if inner_technique:
                result = inner_technique.run(transformed_input, context)
                output = result.output
                trace.extend(result.intermediate_steps)
            else:
                output = transformed_input  # Pass-through

            # Transform output
            hook_context.hook_type = HookType.TRANSFORM_OUTPUT
            hook_context.output_data = output
            transformed_output, output_results = self._execute_hooks(
                HookType.TRANSFORM_OUTPUT, hook_context
            )
            trace.append({"action": "transform_output", "results": output_results})

            # Post-run hooks
            hook_context.hook_type = HookType.POST_RUN
            hook_context.output_data = transformed_output
            _, post_results = self._execute_hooks(HookType.POST_RUN, hook_context)
            trace.append({"action": "post_run_hooks", "results": post_results})

            return TechniqueResult(
                success=True,
                output=transformed_output,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
            )

        except Exception as e:
            # Error hooks
            hook_context.hook_type = HookType.ON_ERROR
            hook_context.error = e
            self._execute_hooks(HookType.ON_ERROR, hook_context)

            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# =============================================================================
# TASK ROUTING
# =============================================================================

@dataclass
class Route:
    """A routing rule mapping conditions to handlers."""
    route_id: str
    condition: Union[str, Callable[[Any], bool]]  # String pattern or callable
    handler: Union[str, TechniqueBase, Callable]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskRouting(TechniqueBase):
    """
    Route tasks to appropriate handlers based on content or metadata.

    Routing Strategies:
        - classifier: ML classifier predicts route
        - embedding: Similarity to route descriptions
        - rules: Pattern matching rules
        - llm: LLM decides routing

    Configuration:
        routes: List of Route definitions
        strategy: Routing strategy to use
        fallback: Default handler if no route matches
        multi_route: Allow routing to multiple handlers

    Usage:
        router = TaskRouting(
            routes=[
                Route("math", lambda x: "calculate" in x.lower(), math_handler),
                Route("search", lambda x: "find" in x.lower(), search_handler),
                Route("code", lambda x: "write" in x.lower(), code_handler),
            ],
            fallback=general_handler,
        )
        result = router.run("Calculate 15% of 200")  # Routes to math_handler
    """

    TECHNIQUE_ID = "task_routing"
    CATEGORY = TechniqueCategory.ORCHESTRATION

    def __init__(
        self,
        routes: Optional[List[Route]] = None,
        strategy: str = "rules",  # rules, classifier, embedding, llm
        fallback: Optional[Union[TechniqueBase, Callable]] = None,
        multi_route: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.routes = sorted(routes or [], key=lambda r: -r.priority)
        self.strategy = strategy
        self.fallback = fallback
        self.multi_route = multi_route

    def _match_route(self, input_data: Any) -> List[Route]:
        """Find matching routes for input."""
        matches = []

        for route in self.routes:
            if callable(route.condition):
                if route.condition(input_data):
                    matches.append(route)
            elif isinstance(route.condition, str):
                # Pattern matching
                if route.condition.lower() in str(input_data).lower():
                    matches.append(route)

            if matches and not self.multi_route:
                break

        return matches

    def _execute_handler(
        self,
        handler: Union[TechniqueBase, Callable],
        input_data: Any,
        context: Optional[Dict],
    ) -> TechniqueResult:
        """Execute a route handler."""
        if isinstance(handler, TechniqueBase):
            return handler.run(input_data, context)
        elif callable(handler):
            result = handler(input_data)
            return TechniqueResult(
                success=True,
                output=result,
                technique_id="callable_handler",
                execution_time_ms=0,
            )
        else:
            return TechniqueResult(
                success=False,
                output=None,
                technique_id="unknown",
                execution_time_ms=0,
                error=f"Unknown handler type: {type(handler)}",
            )

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Find matching routes
        matches = self._match_route(input_data)
        trace.append({
            "action": "match_routes",
            "matches": [r.route_id for r in matches],
        })

        if not matches:
            if self.fallback:
                trace.append({"action": "fallback"})
                result = self._execute_handler(self.fallback, input_data, context)
                result.intermediate_steps = trace + result.intermediate_steps
                return result
            else:
                return TechniqueResult(
                    success=False,
                    output=None,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    error="No matching route found",
                )

        # Execute handler(s)
        if self.multi_route:
            results = []
            for route in matches:
                result = self._execute_handler(route.handler, input_data, context)
                results.append({"route": route.route_id, "result": result})
                trace.append({
                    "action": "execute_route",
                    "route": route.route_id,
                    "success": result.success,
                })

            return TechniqueResult(
                success=all(r["result"].success for r in results),
                output=[r["result"].output for r in results],
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
            )
        else:
            route = matches[0]
            trace.append({"action": "execute_route", "route": route.route_id})
            result = self._execute_handler(route.handler, input_data, context)
            result.intermediate_steps = trace + result.intermediate_steps
            result.technique_id = self.TECHNIQUE_ID
            return result


# =============================================================================
# ENSEMBLE
# =============================================================================

class AggregationStrategy(Enum):
    """Strategies for combining ensemble outputs."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    MIN = "min"
    CONCATENATE = "concatenate"
    BEST = "best"  # Select output with highest confidence
    STACK = "stack"  # Meta-learner combines outputs


@dataclass
class EnsembleMember:
    """A member of an ensemble."""
    member_id: str
    technique: Union[TechniqueBase, Callable]
    weight: float = 1.0
    enabled: bool = True


class Ensemble(TechniqueBase):
    """
    Combine outputs from multiple techniques/models.

    Aggregation Strategies:
        - majority_vote: Most common output wins
        - weighted_vote: Votes weighted by member weight
        - average: Average numeric outputs
        - weighted_average: Weighted average
        - max/min: Max or min output
        - concatenate: Combine all outputs
        - best: Select highest confidence
        - stack: Meta-learner combines

    Configuration:
        members: List of ensemble members
        strategy: Aggregation strategy
        parallel: Execute members in parallel
        min_agreement: Minimum agreement threshold

    Usage:
        ensemble = Ensemble(
            members=[
                EnsembleMember("gpt4", gpt4_technique, weight=1.5),
                EnsembleMember("claude", claude_technique, weight=1.0),
                EnsembleMember("llama", llama_technique, weight=0.8),
            ],
            strategy=AggregationStrategy.WEIGHTED_VOTE,
        )
        result = ensemble.run("What is 2+2?")
    """

    TECHNIQUE_ID = "ensemble"
    CATEGORY = TechniqueCategory.ORCHESTRATION

    def __init__(
        self,
        members: Optional[List[EnsembleMember]] = None,
        strategy: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE,
        parallel: bool = True,
        min_agreement: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.members = members or []
        self.strategy = strategy
        self.parallel = parallel
        self.min_agreement = min_agreement

    def _execute_member(
        self,
        member: EnsembleMember,
        input_data: Any,
        context: Optional[Dict],
    ) -> Tuple[str, TechniqueResult]:
        """Execute a single ensemble member."""
        if isinstance(member.technique, TechniqueBase):
            result = member.technique.run(input_data, context)
        elif callable(member.technique):
            output = member.technique(input_data)
            result = TechniqueResult(
                success=True,
                output=output,
                technique_id=member.member_id,
                execution_time_ms=0,
            )
        else:
            result = TechniqueResult(
                success=False,
                output=None,
                technique_id=member.member_id,
                execution_time_ms=0,
                error="Unknown technique type",
            )
        return member.member_id, result

    def _aggregate(
        self,
        results: List[Tuple[str, TechniqueResult, float]],  # (id, result, weight)
    ) -> Any:
        """Aggregate results based on strategy."""
        outputs = [(r[1].output, r[2]) for r in results if r[1].success]

        if not outputs:
            return None

        if self.strategy == AggregationStrategy.MAJORITY_VOTE:
            # Count votes
            from collections import Counter
            votes = Counter(str(o[0]) for o in outputs)
            return votes.most_common(1)[0][0]

        elif self.strategy == AggregationStrategy.WEIGHTED_VOTE:
            # Weighted vote count
            from collections import defaultdict
            weighted_votes: Dict[str, float] = defaultdict(float)
            for output, weight in outputs:
                weighted_votes[str(output)] += weight
            return max(weighted_votes.items(), key=lambda x: x[1])[0]

        elif self.strategy == AggregationStrategy.AVERAGE:
            # Average numeric outputs
            try:
                return sum(float(o[0]) for o in outputs) / len(outputs)
            except (TypeError, ValueError):
                return outputs[0][0]

        elif self.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            try:
                total_weight = sum(o[1] for o in outputs)
                return sum(float(o[0]) * o[1] for o in outputs) / total_weight
            except (TypeError, ValueError):
                return outputs[0][0]

        elif self.strategy == AggregationStrategy.CONCATENATE:
            return [o[0] for o in outputs]

        elif self.strategy == AggregationStrategy.MAX:
            try:
                return max(float(o[0]) for o in outputs)
            except (TypeError, ValueError):
                return outputs[0][0]

        elif self.strategy == AggregationStrategy.MIN:
            try:
                return min(float(o[0]) for o in outputs)
            except (TypeError, ValueError):
                return outputs[0][0]

        else:
            return outputs[0][0]

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        active_members = [m for m in self.members if m.enabled]

        if not active_members:
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=0,
                error="No active ensemble members",
            )

        # Execute members
        results: List[Tuple[str, TechniqueResult, float]] = []

        if self.parallel:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._execute_member, m, input_data, context
                    ): m for m in active_members
                }
                for future in concurrent.futures.as_completed(futures):
                    member = futures[future]
                    member_id, result = future.result()
                    results.append((member_id, result, member.weight))
        else:
            for member in active_members:
                member_id, result = self._execute_member(member, input_data, context)
                results.append((member_id, result, member.weight))

        trace.append({
            "action": "execute_members",
            "results": [
                {"member": r[0], "success": r[1].success}
                for r in results
            ],
        })

        # Aggregate
        aggregated = self._aggregate(results)
        trace.append({
            "action": "aggregate",
            "strategy": self.strategy.value,
        })

        success_rate = sum(1 for r in results if r[1].success) / len(results)

        return TechniqueResult(
            success=success_rate >= self.min_agreement,
            output=aggregated,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
            metadata={
                "num_members": len(active_members),
                "success_rate": success_rate,
                "individual_outputs": [
                    {"member": r[0], "output": r[1].output}
                    for r in results
                ],
            },
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HookType",
    "AggregationStrategy",
    # Data classes
    "Hook",
    "HookContext",
    "Route",
    "EnsembleMember",
    # Techniques
    "HooksSystem",
    "TaskRouting",
    "Ensemble",
]
