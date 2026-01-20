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


class HierarchicalTaskDecomposition(TechniqueBase):
    """
    Hierarchical Task Network (HTN) style decomposition.

    Creates a tree structure where:
        - Root = Original goal
        - Internal nodes = Composite tasks
        - Leaves = Primitive/atomic tasks

    Supports:
        - Task templates with preconditions/effects
        - Method ordering (different ways to achieve same goal)
        - Partial-order planning
    """

    TECHNIQUE_ID = "hierarchical_task_decomposition"
    CATEGORY = TechniqueCategory.DECOMPOSITION

    def __init__(
        self,
        model: Optional[Any] = None,
        task_library: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.task_library = task_library or {}
        self.max_depth = max_depth

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        # Placeholder implementation
        return TechniqueResult(
            success=True,
            output={"placeholder": "HTN decomposition not yet implemented"},
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=[],
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DecompositionNode",
    "RecursiveDecomposition",
    "LeastToMost",
    "HierarchicalTaskDecomposition",
]
