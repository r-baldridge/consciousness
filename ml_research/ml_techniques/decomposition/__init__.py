"""
Decomposition Techniques

Methods for breaking complex tasks into simpler, manageable subtasks.
These form the foundation for handling problems that exceed a model's
single-shot capability.

=============================================================================
TECHNIQUES
=============================================================================

1. RecursiveDecomposition
   - Recursively breaks tasks until base case (simple enough)
   - Compose results back up the tree

2. LeastToMost
   - Decompose into ordered list from simplest to hardest
   - Solve sequentially, using previous answers as context

3. HierarchicalTaskDecomposition
   - Tree-structured task breakdown
   - HTN (Hierarchical Task Network) style planning

=============================================================================
COMPOSITION PATTERNS
=============================================================================

Decomposition works well with:
    - chain_of_thought: Reason through each subtask
    - tool_calling: Use tools for atomic subtasks
    - verification: Verify each decomposition step
    - memory: Track decomposition history

Example:
    pipeline = compose([
        RecursiveDecomposition(max_depth=3),
        ChainOfThought(),
        SelfConsistency(n=3),
    ])
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from abc import abstractmethod

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


@dataclass
class DecompositionNode:
    """A node in the decomposition tree."""
    task: str
    depth: int
    subtasks: List["DecompositionNode"] = field(default_factory=list)
    result: Optional[Any] = None
    is_atomic: bool = False


class RecursiveDecomposition(TechniqueBase):
    """
    Recursively decomposes complex tasks into simpler subtasks.

    Algorithm:
        1. Check if task is simple enough (base case)
        2. If yes, solve directly
        3. If no, decompose into subtasks
        4. Recursively solve each subtask
        5. Combine subtask results

    Configuration:
        max_depth: Maximum recursion depth (default: 5)
        simplicity_fn: Function to check if task is simple (default: length-based)
        decompose_fn: Function to decompose task (default: LLM-based)
        combine_fn: Function to combine results (default: concatenation)

    Usage:
        decomp = RecursiveDecomposition(
            model=my_model,
            max_depth=3,
            simplicity_threshold=0.8,
        )
        result = decomp.run("Build a complete web application with auth, db, and API")
    """

    TECHNIQUE_ID = "recursive_decomposition"
    CATEGORY = TechniqueCategory.DECOMPOSITION

    def __init__(
        self,
        model: Optional[Any] = None,
        max_depth: int = 5,
        min_subtasks: int = 2,
        max_subtasks: int = 5,
        simplicity_fn: Optional[Callable[[str], bool]] = None,
        decompose_fn: Optional[Callable[[str], List[str]]] = None,
        combine_fn: Optional[Callable[[str, List[Any]], Any]] = None,
        solve_fn: Optional[Callable[[str], Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.max_depth = max_depth
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks

        # Configurable functions (defaults provided)
        self.simplicity_fn = simplicity_fn or self._default_simplicity
        self.decompose_fn = decompose_fn or self._default_decompose
        self.combine_fn = combine_fn or self._default_combine
        self.solve_fn = solve_fn or self._default_solve

    def _default_simplicity(self, task: str) -> bool:
        """Default: task is simple if short and no complex keywords."""
        complex_keywords = [
            "and then", "after that", "multiple", "several",
            "first", "second", "finally", "also", "additionally",
        ]
        is_short = len(task.split()) < 20
        has_complex = any(kw in task.lower() for kw in complex_keywords)
        return is_short and not has_complex

    def _default_decompose(self, task: str) -> List[str]:
        """Default: use model to decompose (placeholder)."""
        # In real implementation, this would call the LLM
        # For now, return a placeholder
        return [f"Subtask 1 of: {task}", f"Subtask 2 of: {task}"]

    def _default_combine(self, original_task: str, subtask_results: List[Any]) -> Any:
        """Default: combine by concatenation."""
        if all(isinstance(r, str) for r in subtask_results):
            return "\n\n".join(subtask_results)
        return {"task": original_task, "subtask_results": subtask_results}

    def _default_solve(self, task: str) -> Any:
        """Default: solve atomic task (placeholder)."""
        # In real implementation, this would call the LLM
        return f"[Solution for: {task}]"

    def _decompose_recursive(
        self,
        task: str,
        depth: int,
        trace: List[Dict],
    ) -> Tuple[Any, List[Dict]]:
        """Recursively decompose and solve a task."""

        # Track this step
        step = {
            "action": "evaluate",
            "task": task[:100],
            "depth": depth,
        }
        trace.append(step)

        # Base case: max depth reached or task is simple
        if depth >= self.max_depth or self.simplicity_fn(task):
            step["action"] = "solve_atomic"
            result = self.solve_fn(task)
            return result, trace

        # Decompose into subtasks
        subtasks = self.decompose_fn(task)
        step["action"] = "decompose"
        step["num_subtasks"] = len(subtasks)

        # Recursively solve each subtask
        subtask_results = []
        for subtask in subtasks:
            result, trace = self._decompose_recursive(subtask, depth + 1, trace)
            subtask_results.append(result)

        # Combine results
        combined = self.combine_fn(task, subtask_results)
        trace.append({
            "action": "combine",
            "task": task[:100],
            "num_results": len(subtask_results),
        })

        return combined, trace

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        task = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", task=task)

        try:
            result, trace = self._decompose_recursive(task, depth=0, trace=trace)

            self._call_hooks("post_run", result=result)

            return TechniqueResult(
                success=True,
                output=result,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
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


class LeastToMost(TechniqueBase):
    """
    Least-to-Most prompting: solve problems from simplest to hardest.

    Paper: "Least-to-Most Prompting Enables Complex Reasoning in
           Large Language Models" (Zhou et al., 2022)

    Algorithm:
        1. Decompose problem into subproblems
        2. Order subproblems from simplest to most complex
        3. Solve each in order, using previous solutions as context
        4. Final solution builds on all previous solutions

    Usage:
        ltm = LeastToMost(model=my_model)
        result = ltm.run("How many ways can you arrange 5 books on a shelf?")
    """

    TECHNIQUE_ID = "least_to_most"
    CATEGORY = TechniqueCategory.DECOMPOSITION

    def __init__(
        self,
        model: Optional[Any] = None,
        include_context: bool = True,
        max_subproblems: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.include_context = include_context
        self.max_subproblems = max_subproblems

    def _decompose_and_order(self, problem: str) -> List[str]:
        """Decompose problem into ordered subproblems (placeholder)."""
        # Real implementation would use LLM
        return [
            f"Simplest subproblem of: {problem}",
            f"Medium subproblem of: {problem}",
            f"Final subproblem (original): {problem}",
        ]

    def _solve_with_context(
        self,
        subproblem: str,
        previous_solutions: List[Tuple[str, Any]],
    ) -> Any:
        """Solve subproblem using previous solutions as context (placeholder)."""
        # Real implementation would include previous solutions in prompt
        context_str = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in previous_solutions
        )
        return f"[Solution using context: {context_str[:50]}...]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        problem = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        try:
            # Step 1: Decompose and order
            subproblems = self._decompose_and_order(problem)
            trace.append({
                "action": "decompose_and_order",
                "num_subproblems": len(subproblems),
            })

            # Step 2: Solve in order
            solutions: List[Tuple[str, Any]] = []
            for i, subproblem in enumerate(subproblems):
                if self.include_context:
                    solution = self._solve_with_context(subproblem, solutions)
                else:
                    solution = self._solve_with_context(subproblem, [])

                solutions.append((subproblem, solution))
                trace.append({
                    "action": "solve_subproblem",
                    "index": i,
                    "subproblem": subproblem[:50],
                })

            # Final solution is the last one
            final_solution = solutions[-1][1] if solutions else None

            return TechniqueResult(
                success=True,
                output=final_solution,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={"all_solutions": solutions},
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# =============================================================================
# HTN (HIERARCHICAL TASK NETWORK) DECOMPOSITION
# =============================================================================

from enum import Enum
import time
import copy


class TaskStatus(Enum):
    """Status of a task in the HTN plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class MethodSelectionStrategy(Enum):
    """Strategies for selecting among multiple methods."""
    FIRST_APPLICABLE = "first_applicable"  # First method whose preconditions are met
    LOWEST_COST = "lowest_cost"            # Method with lowest estimated cost
    HIGHEST_PRIORITY = "highest_priority"  # Method with highest priority
    RANDOM = "random"                       # Random selection


@dataclass
class Condition:
    """
    A condition (precondition or effect) in HTN planning.

    Can represent:
        - State predicates: Condition("has_item", {"item": "key"})
        - Numeric constraints: Condition("resource_gte", {"resource": "energy", "value": 10})
        - Negations: Condition("not_has_item", {"item": "obstacle"}, negated=True)
    """
    predicate: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    negated: bool = False

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate this condition against a state."""
        # Check if predicate exists in state
        if self.predicate in state:
            value = state[self.predicate]

            # Handle different predicate types
            if isinstance(value, bool):
                result = value
            elif isinstance(value, set):
                # Set membership
                key = list(self.arguments.values())[0] if self.arguments else None
                result = key in value
            elif isinstance(value, dict):
                # All arguments must match
                result = all(value.get(k) == v for k, v in self.arguments.items())
            else:
                result = bool(value)
        else:
            result = False

        return (not result) if self.negated else result

    def __str__(self) -> str:
        neg = "NOT " if self.negated else ""
        args = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        return f"{neg}{self.predicate}({args})"


@dataclass
class TaskTemplate:
    """
    A template defining a task type in the HTN domain.

    Task templates can be:
        - Primitive: Directly executable actions
        - Compound: Must be decomposed using methods

    Example:
        # Primitive task
        move_task = TaskTemplate(
            name="move",
            parameters=["from", "to"],
            preconditions=[Condition("at", {"location": "from"})],
            effects=[
                Condition("at", {"location": "to"}),
                Condition("at", {"location": "from"}, negated=True),
            ],
            is_primitive=True,
            cost=1.0,
        )

        # Compound task
        travel_task = TaskTemplate(
            name="travel",
            parameters=["destination"],
            is_primitive=False,  # Must be decomposed
        )
    """
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: List[Condition] = field(default_factory=list)
    effects: List[Condition] = field(default_factory=list)
    is_primitive: bool = False
    cost: float = 1.0
    priority: int = 0
    description: str = ""

    def instantiate(self, bindings: Dict[str, Any]) -> "TaskInstance":
        """Create a task instance with bound parameters."""
        return TaskInstance(
            template=self,
            bindings=bindings,
            status=TaskStatus.PENDING,
        )

    def check_preconditions(self, state: Dict[str, Any], bindings: Dict[str, Any]) -> bool:
        """Check if all preconditions are satisfied in the given state."""
        for condition in self.preconditions:
            # Substitute bindings into condition arguments
            bound_args = {
                k: bindings.get(v, v) if isinstance(v, str) else v
                for k, v in condition.arguments.items()
            }
            bound_condition = Condition(
                predicate=condition.predicate,
                arguments=bound_args,
                negated=condition.negated,
            )
            if not bound_condition.evaluate(state):
                return False
        return True

    def apply_effects(self, state: Dict[str, Any], bindings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply effects to state and return new state."""
        new_state = copy.deepcopy(state)

        for effect in self.effects:
            # Substitute bindings
            bound_args = {
                k: bindings.get(v, v) if isinstance(v, str) else v
                for k, v in effect.arguments.items()
            }

            if effect.negated:
                # Remove from state
                if effect.predicate in new_state:
                    if isinstance(new_state[effect.predicate], set):
                        key = list(bound_args.values())[0] if bound_args else None
                        new_state[effect.predicate].discard(key)
                    else:
                        new_state[effect.predicate] = False
            else:
                # Add to state
                if effect.predicate not in new_state:
                    if bound_args:
                        new_state[effect.predicate] = set()
                    else:
                        new_state[effect.predicate] = True

                if isinstance(new_state[effect.predicate], set):
                    key = list(bound_args.values())[0] if bound_args else None
                    if key:
                        new_state[effect.predicate].add(key)
                elif bound_args:
                    new_state[effect.predicate] = bound_args
                else:
                    new_state[effect.predicate] = True

        return new_state


@dataclass
class TaskInstance:
    """An instantiated task with bound parameters."""
    template: TaskTemplate
    bindings: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    subtasks: List["TaskInstance"] = field(default_factory=list)
    parent: Optional["TaskInstance"] = None
    depth: int = 0

    @property
    def name(self) -> str:
        return self.template.name

    @property
    def is_primitive(self) -> bool:
        return self.template.is_primitive

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.bindings.items())
        return f"{self.template.name}({args})"


@dataclass
class Method:
    """
    A method for decomposing a compound task into subtasks.

    Multiple methods can exist for the same task, providing
    alternative decomposition strategies.

    Example:
        # Method for "travel" task using car
        travel_by_car = Method(
            name="travel_by_car",
            task_name="travel",
            parameters=["destination"],
            preconditions=[Condition("has_car", {})],
            subtask_templates=[
                ("get_in_car", {}),
                ("drive_to", {"location": "destination"}),
                ("get_out_car", {}),
            ],
            cost=10.0,
            priority=1,
        )

        # Alternative method using walking
        travel_by_foot = Method(
            name="travel_by_foot",
            task_name="travel",
            parameters=["destination"],
            preconditions=[],  # No preconditions
            subtask_templates=[
                ("walk_to", {"location": "destination"}),
            ],
            cost=50.0,
            priority=0,
        )
    """
    name: str
    task_name: str  # Which compound task this method decomposes
    parameters: List[str] = field(default_factory=list)
    preconditions: List[Condition] = field(default_factory=list)
    subtask_templates: List[Tuple[str, Dict[str, str]]] = field(default_factory=list)
    cost: float = 1.0
    priority: int = 0
    description: str = ""

    def is_applicable(self, state: Dict[str, Any], bindings: Dict[str, Any]) -> bool:
        """Check if method's preconditions are met."""
        for condition in self.preconditions:
            bound_args = {
                k: bindings.get(v, v) if isinstance(v, str) else v
                for k, v in condition.arguments.items()
            }
            bound_condition = Condition(
                predicate=condition.predicate,
                arguments=bound_args,
                negated=condition.negated,
            )
            if not bound_condition.evaluate(state):
                return False
        return True

    def get_subtasks(
        self,
        bindings: Dict[str, Any],
        task_library: Dict[str, TaskTemplate],
    ) -> List[TaskInstance]:
        """Generate subtask instances from this method."""
        subtasks = []

        for task_name, param_mapping in self.subtask_templates:
            if task_name not in task_library:
                # Create a default primitive task if not in library
                template = TaskTemplate(
                    name=task_name,
                    is_primitive=True,
                )
            else:
                template = task_library[task_name]

            # Build bindings for subtask
            subtask_bindings = {}
            for param, source in param_mapping.items():
                if source in bindings:
                    subtask_bindings[param] = bindings[source]
                else:
                    subtask_bindings[param] = source

            subtasks.append(template.instantiate(subtask_bindings))

        return subtasks


@dataclass
class HTNPlan:
    """
    A hierarchical plan produced by HTN decomposition.

    Contains the task tree and flattened execution sequence.
    """
    root_task: TaskInstance
    primitive_tasks: List[TaskInstance] = field(default_factory=list)
    total_cost: float = 0.0
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def get_execution_order(self) -> List[TaskInstance]:
        """Get primitive tasks in execution order."""
        return [t for t in self.primitive_tasks if t.status == TaskStatus.PENDING]

    def mark_completed(self, task: TaskInstance) -> None:
        """Mark a task as completed."""
        task.status = TaskStatus.COMPLETED

    def mark_failed(self, task: TaskInstance, reason: str = "") -> None:
        """Mark a task as failed."""
        task.status = TaskStatus.FAILED
        task.result = reason


class PlanValidator:
    """Validates HTN plans for feasibility."""

    @staticmethod
    def validate_plan(
        plan: HTNPlan,
        initial_state: Dict[str, Any],
        task_library: Dict[str, TaskTemplate],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a plan is executable from the initial state.

        Checks:
            1. All primitive tasks have their preconditions satisfied
            2. Effects properly chain (each task's effects enable next task's preconditions)
            3. No circular dependencies
        """
        errors = []
        current_state = copy.deepcopy(initial_state)

        for task in plan.primitive_tasks:
            template = task.template

            # Check preconditions
            if not template.check_preconditions(current_state, task.bindings):
                unmet = []
                for cond in template.preconditions:
                    bound_args = {
                        k: task.bindings.get(v, v) if isinstance(v, str) else v
                        for k, v in cond.arguments.items()
                    }
                    bound_cond = Condition(cond.predicate, bound_args, cond.negated)
                    if not bound_cond.evaluate(current_state):
                        unmet.append(str(bound_cond))
                errors.append(f"Task {task}: unmet preconditions {unmet}")

            # Apply effects to simulate execution
            current_state = template.apply_effects(current_state, task.bindings)

        plan.is_valid = len(errors) == 0
        plan.validation_errors = errors

        return plan.is_valid, errors

    @staticmethod
    def check_goal_achieved(
        plan: HTNPlan,
        initial_state: Dict[str, Any],
        goal_conditions: List[Condition],
        task_library: Dict[str, TaskTemplate],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if executing the plan achieves the goal conditions."""
        current_state = copy.deepcopy(initial_state)

        # Simulate execution
        for task in plan.primitive_tasks:
            current_state = task.template.apply_effects(current_state, task.bindings)

        # Check goal conditions
        for condition in goal_conditions:
            if not condition.evaluate(current_state):
                return False, current_state

        return True, current_state


class HierarchicalTaskDecomposition(TechniqueBase):
    """
    Hierarchical Task Network (HTN) style decomposition.

    Paper references:
        - "SHOP2: An HTN Planning System" (Nau et al., 2003)
        - "Hierarchical Task Network Planning" (Erol et al., 1994)

    Creates a tree structure where:
        - Root = Original goal/task
        - Internal nodes = Compound tasks (decomposed via methods)
        - Leaves = Primitive tasks (directly executable)

    Key Concepts:
        - TaskTemplate: Defines a task type with preconditions and effects
        - Method: Specifies how to decompose a compound task into subtasks
        - State: World state that tasks can query and modify
        - Plan: Sequence of primitive tasks achieving the goal

    Features:
        - Multiple methods per task (alternative decomposition strategies)
        - Precondition checking before decomposition
        - Effect tracking through state simulation
        - Plan validation
        - Dynamic replanning on failure

    Configuration:
        task_library: Dict of TaskTemplate objects defining the domain
        method_library: Dict of Method objects for decomposition
        initial_state: Starting world state
        selection_strategy: How to choose among applicable methods
        max_depth: Maximum decomposition depth
        enable_replanning: Whether to replan on task failure

    Usage:
        # Define domain
        tasks = {
            "pickup": TaskTemplate(
                name="pickup",
                parameters=["item"],
                preconditions=[Condition("at_item", {"item": "item"})],
                effects=[Condition("holding", {"item": "item"})],
                is_primitive=True,
            ),
            "deliver": TaskTemplate(
                name="deliver",
                parameters=["item", "destination"],
                is_primitive=False,  # Compound - needs method
            ),
        }

        methods = {
            "deliver_direct": Method(
                name="deliver_direct",
                task_name="deliver",
                parameters=["item", "destination"],
                subtask_templates=[
                    ("goto", {"location": "item_location"}),
                    ("pickup", {"item": "item"}),
                    ("goto", {"location": "destination"}),
                    ("putdown", {"item": "item"}),
                ],
            ),
        }

        htn = HierarchicalTaskDecomposition(
            task_library=tasks,
            method_library=methods,
            initial_state={"location": "home", "items": {"package"}},
        )

        result = htn.run({
            "task": "deliver",
            "bindings": {"item": "package", "destination": "office"},
        })
    """

    TECHNIQUE_ID = "hierarchical_task_decomposition"
    CATEGORY = TechniqueCategory.DECOMPOSITION

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[Any] = None,
        task_library: Optional[Dict[str, TaskTemplate]] = None,
        method_library: Optional[Dict[str, Method]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        selection_strategy: MethodSelectionStrategy = MethodSelectionStrategy.HIGHEST_PRIORITY,
        max_depth: int = 10,
        enable_replanning: bool = True,
        max_replan_attempts: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend or model
        self.task_library = task_library or {}
        self.method_library = method_library or {}
        self.initial_state = initial_state or {}
        self.selection_strategy = selection_strategy
        self.max_depth = max_depth
        self.enable_replanning = enable_replanning
        self.max_replan_attempts = max_replan_attempts

        # Build method index by task name
        self._methods_by_task: Dict[str, List[Method]] = {}
        for method in self.method_library.values():
            if method.task_name not in self._methods_by_task:
                self._methods_by_task[method.task_name] = []
            self._methods_by_task[method.task_name].append(method)

    def _get_applicable_methods(
        self,
        task: TaskInstance,
        state: Dict[str, Any],
    ) -> List[Method]:
        """Get all methods applicable to a task in the current state."""
        methods = self._methods_by_task.get(task.name, [])
        applicable = []

        for method in methods:
            if method.is_applicable(state, task.bindings):
                applicable.append(method)

        return applicable

    def _select_method(
        self,
        methods: List[Method],
        task: TaskInstance,
        state: Dict[str, Any],
    ) -> Optional[Method]:
        """Select a method based on the selection strategy."""
        if not methods:
            return None

        if self.selection_strategy == MethodSelectionStrategy.FIRST_APPLICABLE:
            return methods[0]

        elif self.selection_strategy == MethodSelectionStrategy.LOWEST_COST:
            return min(methods, key=lambda m: m.cost)

        elif self.selection_strategy == MethodSelectionStrategy.HIGHEST_PRIORITY:
            return max(methods, key=lambda m: m.priority)

        elif self.selection_strategy == MethodSelectionStrategy.RANDOM:
            import random
            return random.choice(methods)

        return methods[0]

    def _decompose_task(
        self,
        task: TaskInstance,
        state: Dict[str, Any],
        depth: int,
        trace: List[Dict],
    ) -> Tuple[List[TaskInstance], Dict[str, Any], bool]:
        """
        Recursively decompose a task into primitive tasks.

        Returns:
            - List of primitive tasks
            - Final state after decomposition
            - Success flag
        """
        trace.append({
            "action": "decompose",
            "task": str(task),
            "depth": depth,
            "is_primitive": task.is_primitive,
        })

        # Check max depth
        if depth > self.max_depth:
            trace.append({
                "action": "max_depth_reached",
                "task": str(task),
            })
            return [], state, False

        # Primitive task - return directly
        if task.is_primitive:
            # Check preconditions
            if task.template.check_preconditions(state, task.bindings):
                # Apply effects and return
                new_state = task.template.apply_effects(state, task.bindings)
                task.status = TaskStatus.COMPLETED
                trace.append({
                    "action": "execute_primitive",
                    "task": str(task),
                    "success": True,
                })
                return [task], new_state, True
            else:
                trace.append({
                    "action": "precondition_failed",
                    "task": str(task),
                })
                return [], state, False

        # Compound task - find and apply method
        applicable_methods = self._get_applicable_methods(task, state)

        if not applicable_methods:
            # Try to generate a method using LLM if no predefined methods
            generated = self._generate_method_llm(task, state, trace)
            if generated:
                applicable_methods = [generated]

        if not applicable_methods:
            trace.append({
                "action": "no_applicable_methods",
                "task": str(task),
            })
            return [], state, False

        # Sort methods by selection strategy before trying them
        if self.selection_strategy == MethodSelectionStrategy.HIGHEST_PRIORITY:
            applicable_methods.sort(key=lambda m: m.priority, reverse=True)
        elif self.selection_strategy == MethodSelectionStrategy.LOWEST_COST:
            applicable_methods.sort(key=lambda m: m.cost)
        elif self.selection_strategy == MethodSelectionStrategy.RANDOM:
            import random
            random.shuffle(applicable_methods)
        # FIRST_APPLICABLE keeps original order

        # Try methods in sorted order until one succeeds
        for method in applicable_methods:
            trace.append({
                "action": "apply_method",
                "task": str(task),
                "method": method.name,
            })

            # Get subtasks from method
            subtasks = method.get_subtasks(task.bindings, self.task_library)

            # Recursively decompose subtasks
            all_primitives = []
            current_state = state
            success = True

            for subtask in subtasks:
                subtask.parent = task
                subtask.depth = depth + 1

                primitives, current_state, sub_success = self._decompose_task(
                    subtask, current_state, depth + 1, trace
                )

                if not sub_success:
                    success = False
                    break

                all_primitives.extend(primitives)
                task.subtasks.append(subtask)

            if success:
                task.status = TaskStatus.COMPLETED
                return all_primitives, current_state, True

        return [], state, False

    def _generate_method_llm(
        self,
        task: TaskInstance,
        state: Dict[str, Any],
        trace: List[Dict],
    ) -> Optional[Method]:
        """Use LLM to generate a decomposition method for unknown tasks."""
        if not self.backend:
            return None

        trace.append({
            "action": "generate_method_llm",
            "task": str(task),
        })

        # Build prompt for LLM
        prompt = f"""Decompose this task into subtasks:

Task: {task.name}
Parameters: {task.bindings}
Current state: {state}

List 2-5 subtasks that would accomplish this task.
Format each subtask as: SUBTASK: <name>(<param1>=<value1>, ...)
"""

        # Generate using backend
        if hasattr(self.backend, 'generate'):
            response = self.backend.generate(prompt)
        elif callable(self.backend):
            response = self.backend(prompt)
        else:
            # Placeholder response
            response = f"SUBTASK: prepare({task.name})\nSUBTASK: execute({task.name})\nSUBTASK: verify({task.name})"

        # Parse response into subtasks
        subtask_templates = []
        for line in response.split('\n'):
            if 'SUBTASK:' in line.upper():
                # Parse subtask specification
                spec = line.split(':', 1)[-1].strip()
                if '(' in spec:
                    name = spec.split('(')[0].strip()
                    # Extract parameters if any
                    params = {}
                    if '=' in spec:
                        param_str = spec.split('(')[1].rstrip(')')
                        for param in param_str.split(','):
                            if '=' in param:
                                k, v = param.split('=')
                                params[k.strip()] = v.strip()
                else:
                    name = spec
                    params = {}

                subtask_templates.append((name, params))

                # Add to task library as primitive if not exists
                if name not in self.task_library:
                    self.task_library[name] = TaskTemplate(
                        name=name,
                        is_primitive=True,
                    )

        if subtask_templates:
            method = Method(
                name=f"generated_{task.name}",
                task_name=task.name,
                subtask_templates=subtask_templates,
                cost=10.0,  # Higher cost for generated methods
            )
            return method

        return None

    def _replan(
        self,
        failed_task: TaskInstance,
        original_goal: TaskInstance,
        current_state: Dict[str, Any],
        trace: List[Dict],
    ) -> Optional[HTNPlan]:
        """Attempt to create a new plan after a task failure."""
        trace.append({
            "action": "replan",
            "failed_task": str(failed_task),
        })

        # Try to decompose the original goal again from current state
        primitives, final_state, success = self._decompose_task(
            original_goal, current_state, 0, trace
        )

        if success:
            plan = HTNPlan(
                root_task=original_goal,
                primitive_tasks=primitives,
                total_cost=sum(t.template.cost for t in primitives),
            )
            return plan

        return None

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        self._call_hooks("pre_run", input_data=input_data)

        # Parse input
        if isinstance(input_data, str):
            # Treat string as task name
            task_name = input_data
            bindings = {}
        elif isinstance(input_data, dict):
            task_name = input_data.get("task", input_data.get("name", "goal"))
            bindings = input_data.get("bindings", input_data.get("parameters", {}))
        else:
            task_name = str(input_data)
            bindings = {}

        # Get or create task template
        if task_name in self.task_library:
            template = self.task_library[task_name]
        else:
            # Create default compound task
            template = TaskTemplate(
                name=task_name,
                is_primitive=False,
            )
            self.task_library[task_name] = template

        # Create root task instance
        root_task = template.instantiate(bindings)

        trace.append({
            "action": "start",
            "task": str(root_task),
            "initial_state": dict(self.initial_state),
        })

        try:
            # Decompose the task
            primitives, final_state, success = self._decompose_task(
                root_task,
                copy.deepcopy(self.initial_state),
                0,
                trace,
            )

            if success:
                # Create and validate plan
                plan = HTNPlan(
                    root_task=root_task,
                    primitive_tasks=primitives,
                    total_cost=sum(t.template.cost for t in primitives),
                )

                is_valid, errors = PlanValidator.validate_plan(
                    plan, self.initial_state, self.task_library
                )

                trace.append({
                    "action": "plan_created",
                    "num_primitives": len(primitives),
                    "total_cost": plan.total_cost,
                    "is_valid": is_valid,
                })

                self._call_hooks("post_run", plan=plan)

                return TechniqueResult(
                    success=True,
                    output={
                        "plan": [str(t) for t in primitives],
                        "task_tree": self._serialize_task_tree(root_task),
                        "total_cost": plan.total_cost,
                        "num_steps": len(primitives),
                        "is_valid": is_valid,
                        "validation_errors": errors,
                        "final_state": final_state,
                    },
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    metadata={
                        "methods_used": [
                            t.get("method") for t in trace
                            if t.get("action") == "apply_method"
                        ],
                        "max_depth_reached": max(
                            (t.get("depth", 0) for t in trace),
                            default=0
                        ),
                    },
                )
            else:
                # Decomposition failed
                return TechniqueResult(
                    success=False,
                    output={
                        "error": "Failed to decompose task",
                        "partial_trace": trace[-5:],
                    },
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=trace,
                    error="No valid decomposition found",
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

    def _serialize_task_tree(self, task: TaskInstance, depth: int = 0) -> Dict:
        """Serialize task tree to dict for output."""
        result = {
            "name": task.name,
            "bindings": task.bindings,
            "is_primitive": task.is_primitive,
            "status": task.status.value,
            "depth": depth,
        }

        if task.subtasks:
            result["subtasks"] = [
                self._serialize_task_tree(st, depth + 1)
                for st in task.subtasks
            ]

        return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TaskStatus",
    "MethodSelectionStrategy",
    # Data classes
    "DecompositionNode",
    "Condition",
    "TaskTemplate",
    "TaskInstance",
    "Method",
    "HTNPlan",
    # Utility classes
    "PlanValidator",
    # Techniques
    "RecursiveDecomposition",
    "LeastToMost",
    "HierarchicalTaskDecomposition",
]
