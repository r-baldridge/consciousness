"""
Prompting Techniques

Structured approaches for formulating inputs to language models
to elicit better reasoning, accuracy, and task performance.

=============================================================================
TECHNIQUES
=============================================================================

1. ChainOfThought (CoT)
   - Step-by-step reasoning before final answer
   - Zero-shot ("Let's think step by step") or few-shot (with examples)

2. TreeOfThought (ToT)
   - Explore multiple reasoning paths as a tree
   - Search (BFS/DFS/beam) for best path

3. GraphOfThought (GoT)
   - Generalize ToT to arbitrary graph structures
   - Thoughts can merge, loop, form complex patterns

4. SelfConsistency
   - Sample multiple reasoning paths
   - Majority vote on final answer

5. FewShot
   - Provide examples before the query
   - In-context learning without fine-tuning

=============================================================================
PROMPTING PATTERNS
=============================================================================

Zero-Shot CoT:
    "Question: {question}
     Let's think step by step."

Few-Shot CoT:
    "Question: {example_q1}
     Let's think step by step.
     {reasoning_1}
     Answer: {answer_1}

     Question: {actual_question}
     Let's think step by step."

Tree-of-Thought:
    Generate -> Evaluate -> Search -> Select

Self-Consistency:
    Sample N paths -> Extract answers -> Vote
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from enum import Enum
from abc import abstractmethod
import random

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory

# Backend imports - use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ...backends import LLMBackend

# Import backend utilities
try:
    from ...backends import get_backend, LLMBackend as LLMBackendClass
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    LLMBackendClass = None


# =============================================================================
# CHAIN OF THOUGHT
# =============================================================================

class CoTMode(Enum):
    """Chain-of-thought prompting modes."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    AUTO = "auto"  # Automatically generate examples


@dataclass
class CoTExample:
    """An example for few-shot chain-of-thought."""
    question: str
    reasoning: str
    answer: str


class ChainOfThought(TechniqueBase):
    """
    Chain-of-Thought Prompting.

    Paper: "Chain-of-Thought Prompting Elicits Reasoning in Large
           Language Models" (Wei et al., 2022)
    https://arxiv.org/abs/2201.11903

    Prompts the model to show reasoning steps before giving the final answer.
    Dramatically improves performance on arithmetic, commonsense, and
    symbolic reasoning tasks.

    Configuration:
        backend: LLMBackend instance or name (uses global registry if string)
        model: (Deprecated) Use backend instead
        mode: zero_shot, few_shot, or auto
        trigger: Phrase to trigger reasoning (default: "Let's think step by step.")
        examples: List of CoTExample for few-shot mode
        extract_answer: Whether to extract final answer from reasoning
        answer_prefix: Prefix marking the final answer

    Usage:
        # Using a backend
        from ml_research.backends import MockBackend
        cot = ChainOfThought(
            backend=MockBackend(),
            mode=CoTMode.ZERO_SHOT,
        )
        result = cot.run("What is 23 * 47?")

        # Using backend name from registry
        cot = ChainOfThought(backend="mock")

        # Legacy model parameter still supported
        cot = ChainOfThought(model=my_model)
    """

    TECHNIQUE_ID = "chain_of_thought"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        mode: CoTMode = CoTMode.ZERO_SHOT,
        trigger: str = "Let's think step by step.",
        examples: Optional[List[CoTExample]] = None,
        extract_answer: bool = True,
        answer_prefix: str = "Therefore, the answer is",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle backend resolution
        self.backend = self._resolve_backend(backend, model)
        self.model = model  # Keep for backward compatibility

        self.mode = mode
        self.trigger = trigger
        self.examples = examples or []
        self.extract_answer = extract_answer
        self.answer_prefix = answer_prefix

    def _resolve_backend(self, backend: Optional[Any], model: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        # If backend is provided
        if backend is not None:
            # If it's a string, try to get from registry
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            # If it's already a backend, use directly
            return backend

        # If legacy model is provided, use it directly
        if model is not None:
            return model

        # Try to get default backend
        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _build_prompt(self, question: str) -> str:
        """Build the CoT prompt."""
        if self.mode == CoTMode.ZERO_SHOT:
            return f"Question: {question}\n{self.trigger}\n"

        elif self.mode == CoTMode.FEW_SHOT:
            prompt_parts = []
            for ex in self.examples:
                prompt_parts.append(
                    f"Question: {ex.question}\n"
                    f"{self.trigger}\n"
                    f"{ex.reasoning}\n"
                    f"{self.answer_prefix} {ex.answer}\n"
                )
            prompt_parts.append(f"Question: {question}\n{self.trigger}\n")
            return "\n".join(prompt_parts)

        else:  # AUTO
            return f"Question: {question}\n{self.trigger}\n"

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract the final answer from the response."""
        if self.answer_prefix in response:
            answer_part = response.split(self.answer_prefix)[-1]
            # Take first line or up to period
            answer = answer_part.strip().split("\n")[0].strip()
            if answer.endswith("."):
                answer = answer[:-1]
            return answer
        return None

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        # Try to use backend if available
        if self.backend is not None:
            # Check if it's an LLMBackend with generate method
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(
                    prompt,
                    max_tokens=1024,
                    temperature=0.7,
                )
            # Legacy model support - try calling directly
            elif callable(self.backend):
                return self.backend(prompt)

        # Placeholder response for when no backend is available
        return f"Step 1: Analyze the question\nStep 2: Apply reasoning\n{self.answer_prefix} [answer]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", question=question)

        try:
            # Build prompt
            prompt = self._build_prompt(question)
            trace.append({
                "action": "build_prompt",
                "mode": self.mode.value,
                "num_examples": len(self.examples),
            })

            # Generate response
            response = self._generate_response(prompt)
            trace.append({
                "action": "generate",
                "response_length": len(response),
            })

            # Extract answer if requested
            answer = None
            if self.extract_answer:
                answer = self._extract_answer(response)
                trace.append({
                    "action": "extract_answer",
                    "found": answer is not None,
                })

            self._call_hooks("post_run", response=response, answer=answer)

            return TechniqueResult(
                success=True,
                output={
                    "reasoning": response,
                    "answer": answer,
                    "full_response": response,
                },
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


# =============================================================================
# TREE OF THOUGHT
# =============================================================================

class SearchAlgorithm(Enum):
    """Search algorithms for Tree-of-Thought."""
    BFS = "bfs"  # Breadth-first search
    DFS = "dfs"  # Depth-first search
    BEAM = "beam"  # Beam search


class EvaluationStrategy(Enum):
    """Strategies for evaluating thought nodes."""
    VALUE = "value"  # Score each node
    VOTE = "vote"  # Vote on best node
    COMPARE = "compare"  # Pairwise comparison


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    thought: str
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    score: float = 0.0
    depth: int = 0
    is_solution: bool = False


class TreeOfThought(TechniqueBase):
    """
    Tree-of-Thought Prompting.

    Paper: "Tree of Thoughts: Deliberate Problem Solving with Large
           Language Models" (Yao et al., 2023)
    https://arxiv.org/abs/2305.10601

    Extends CoT by exploring multiple reasoning paths as a tree,
    using search algorithms to find the best path. Enables deliberate
    problem-solving with backtracking.

    Configuration:
        branching_factor: Number of children per node (default: 3)
        max_depth: Maximum tree depth (default: 5)
        search_algorithm: BFS, DFS, or beam search
        evaluation_strategy: How to score nodes
        beam_width: Width for beam search (default: 3)

    Usage:
        tot = TreeOfThought(
            model=my_model,
            branching_factor=3,
            max_depth=4,
            search_algorithm=SearchAlgorithm.BEAM,
        )
        result = tot.run("Solve this puzzle: ...")
    """

    TECHNIQUE_ID = "tree_of_thought"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        model: Optional[Any] = None,
        branching_factor: int = 3,
        max_depth: int = 5,
        search_algorithm: SearchAlgorithm = SearchAlgorithm.BFS,
        evaluation_strategy: EvaluationStrategy = EvaluationStrategy.VALUE,
        beam_width: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.search_algorithm = search_algorithm
        self.evaluation_strategy = evaluation_strategy
        self.beam_width = beam_width

    def _generate_thoughts(
        self,
        problem: str,
        parent: Optional[ThoughtNode],
    ) -> List[str]:
        """Generate candidate thoughts/next steps (placeholder)."""
        # Real implementation uses LLM
        context = parent.thought if parent else problem
        return [
            f"Thought {i+1}: Consider approach {i+1} for {context[:30]}..."
            for i in range(self.branching_factor)
        ]

    def _evaluate_thought(self, problem: str, thought: str) -> float:
        """Evaluate a thought node (placeholder)."""
        # Real implementation uses LLM to score
        return random.random()

    def _is_solution(self, thought: str) -> bool:
        """Check if thought represents a solution (placeholder)."""
        return "final answer" in thought.lower() or "solution" in thought.lower()

    def _bfs_search(self, problem: str, root: ThoughtNode) -> Optional[ThoughtNode]:
        """Breadth-first search through thought tree."""
        queue = [root]
        best_solution = None
        best_score = -float("inf")

        while queue:
            node = queue.pop(0)

            if node.depth >= self.max_depth:
                continue

            # Generate and evaluate children
            thoughts = self._generate_thoughts(problem, node)
            for thought in thoughts:
                child = ThoughtNode(
                    thought=thought,
                    parent=node,
                    depth=node.depth + 1,
                )
                child.score = self._evaluate_thought(problem, thought)
                child.is_solution = self._is_solution(thought)
                node.children.append(child)

                if child.is_solution and child.score > best_score:
                    best_solution = child
                    best_score = child.score

                queue.append(child)

        return best_solution

    def _beam_search(self, problem: str, root: ThoughtNode) -> Optional[ThoughtNode]:
        """Beam search through thought tree."""
        beam = [root]
        best_solution = None
        best_score = -float("inf")

        for depth in range(self.max_depth):
            candidates = []

            for node in beam:
                thoughts = self._generate_thoughts(problem, node)
                for thought in thoughts:
                    child = ThoughtNode(
                        thought=thought,
                        parent=node,
                        depth=depth + 1,
                    )
                    child.score = self._evaluate_thought(problem, thought)
                    child.is_solution = self._is_solution(thought)
                    node.children.append(child)
                    candidates.append(child)

                    if child.is_solution and child.score > best_score:
                        best_solution = child
                        best_score = child.score

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: -x.score)
            beam = candidates[:self.beam_width]

            if not beam:
                break

        return best_solution

    def _get_path(self, node: Optional[ThoughtNode]) -> List[str]:
        """Get the path from root to node."""
        if node is None:
            return []
        path = []
        current = node
        while current:
            path.append(current.thought)
            current = current.parent
        return list(reversed(path))

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        problem = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        try:
            # Create root node
            root = ThoughtNode(thought=f"Problem: {problem}", depth=0)
            trace.append({"action": "create_root"})

            # Search
            if self.search_algorithm == SearchAlgorithm.BFS:
                solution = self._bfs_search(problem, root)
            elif self.search_algorithm == SearchAlgorithm.BEAM:
                solution = self._beam_search(problem, root)
            else:  # DFS
                solution = self._bfs_search(problem, root)  # Placeholder

            trace.append({
                "action": "search",
                "algorithm": self.search_algorithm.value,
                "found_solution": solution is not None,
            })

            # Get solution path
            path = self._get_path(solution)

            return TechniqueResult(
                success=solution is not None,
                output={
                    "solution": solution.thought if solution else None,
                    "path": path,
                    "score": solution.score if solution else 0,
                },
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
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
# GRAPH OF THOUGHT
# =============================================================================

@dataclass
class GraphNode:
    """A node in the thought graph."""
    node_id: str
    thought: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphOfThought(TechniqueBase):
    """
    Graph-of-Thought Prompting.

    Paper: "Graph of Thoughts: Solving Elaborate Problems with Large
           Language Models" (Besta et al., 2023)
    https://arxiv.org/abs/2308.09687

    Extends ToT to arbitrary graph structures where thoughts can:
    - Have multiple parents (merge)
    - Loop back (refinement)
    - Form complex reasoning patterns

    Operations:
        - Generate: Create new thoughts
        - Aggregate: Combine multiple thoughts
        - Refine: Improve existing thoughts
        - Score: Evaluate thoughts

    Configuration:
        max_nodes: Maximum nodes in graph
        allow_cycles: Whether to allow cycles
        aggregation_fn: How to combine thoughts
    """

    TECHNIQUE_ID = "graph_of_thought"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        model: Optional[Any] = None,
        max_nodes: int = 50,
        allow_cycles: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.max_nodes = max_nodes
        self.allow_cycles = allow_cycles
        self.graph: Dict[str, GraphNode] = {}

    def _generate_thought(self, problem: str, parents: List[GraphNode]) -> str:
        """Generate a new thought based on parents (placeholder)."""
        parent_context = " + ".join(p.thought[:50] for p in parents)
        return f"New thought combining: {parent_context}"

    def _aggregate_thoughts(self, thoughts: List[GraphNode]) -> str:
        """Aggregate multiple thoughts into one (placeholder)."""
        return f"Aggregated: {' | '.join(t.thought[:30] for t in thoughts)}"

    def _refine_thought(self, thought: GraphNode) -> str:
        """Refine an existing thought (placeholder)."""
        return f"Refined: {thought.thought}"

    def _score_thought(self, thought: str) -> float:
        """Score a thought (placeholder)."""
        return random.random()

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        problem = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        # Create root node
        root_id = "root"
        self.graph[root_id] = GraphNode(
            node_id=root_id,
            thought=f"Problem: {problem}",
        )
        trace.append({"action": "create_root"})

        # Simple expansion (placeholder for full GoT algorithm)
        node_count = 1
        current_nodes = [root_id]

        while node_count < self.max_nodes and current_nodes:
            node_id = current_nodes.pop(0)
            parent = self.graph[node_id]

            # Generate children
            for i in range(2):
                child_id = f"node_{node_count}"
                thought = self._generate_thought(problem, [parent])
                score = self._score_thought(thought)

                self.graph[child_id] = GraphNode(
                    node_id=child_id,
                    thought=thought,
                    parents=[node_id],
                    score=score,
                )
                parent.children.append(child_id)
                current_nodes.append(child_id)
                node_count += 1

                if node_count >= self.max_nodes:
                    break

            trace.append({
                "action": "expand",
                "node": node_id,
                "children": len(parent.children),
            })

        # Find best node
        best_node = max(self.graph.values(), key=lambda n: n.score)

        return TechniqueResult(
            success=True,
            output={
                "best_thought": best_node.thought,
                "best_score": best_node.score,
                "graph_size": len(self.graph),
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# SELF-CONSISTENCY
# =============================================================================

class AggregationMethod(Enum):
    """Methods for aggregating multiple answers."""
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"


class SelfConsistency(TechniqueBase):
    """
    Self-Consistency Decoding.

    Paper: "Self-Consistency Improves Chain of Thought Reasoning in
           Language Models" (Wang et al., 2022)
    https://arxiv.org/abs/2203.11171

    Samples multiple reasoning paths (using temperature > 0) and takes
    a majority vote on the final answer. Based on the intuition that
    correct reasoning paths are more likely to arrive at the same answer.

    Configuration:
        num_samples: Number of reasoning paths to sample (default: 5)
        temperature: Sampling temperature (default: 0.7)
        aggregation: How to combine answers (majority, weighted, unanimous)
        inner_technique: Technique to use for each sample (default: CoT)

    Usage:
        sc = SelfConsistency(
            model=my_model,
            num_samples=5,
            temperature=0.7,
        )
        result = sc.run("What is 23 * 47?")
        # Samples 5 CoT reasoning paths and votes on answer
    """

    TECHNIQUE_ID = "self_consistency"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        model: Optional[Any] = None,
        num_samples: int = 5,
        temperature: float = 0.7,
        aggregation: AggregationMethod = AggregationMethod.MAJORITY,
        inner_technique: Optional[TechniqueBase] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.num_samples = num_samples
        self.temperature = temperature
        self.aggregation = aggregation
        self.inner_technique = inner_technique or ChainOfThought(model=model)

    def _sample_reasoning(self, question: str) -> Tuple[str, Optional[str]]:
        """Sample one reasoning path and extract answer (placeholder)."""
        result = self.inner_technique.run(question)
        if result.success and isinstance(result.output, dict):
            return result.output.get("reasoning", ""), result.output.get("answer")
        return "", None

    def _aggregate_answers(
        self,
        answers: List[Optional[str]],
    ) -> Tuple[Optional[str], float]:
        """Aggregate answers and return (answer, confidence)."""
        # Filter out None answers
        valid_answers = [a for a in answers if a is not None]

        if not valid_answers:
            return None, 0.0

        from collections import Counter
        counts = Counter(valid_answers)

        if self.aggregation == AggregationMethod.MAJORITY:
            winner, count = counts.most_common(1)[0]
            confidence = count / len(answers)
            return winner, confidence

        elif self.aggregation == AggregationMethod.UNANIMOUS:
            if len(counts) == 1:
                return valid_answers[0], 1.0
            return None, 0.0

        else:  # WEIGHTED (placeholder - would use model confidence)
            winner, count = counts.most_common(1)[0]
            confidence = count / len(answers)
            return winner, confidence

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        try:
            # Sample multiple reasoning paths
            samples = []
            answers = []

            for i in range(self.num_samples):
                reasoning, answer = self._sample_reasoning(question)
                samples.append({"reasoning": reasoning, "answer": answer})
                answers.append(answer)
                trace.append({
                    "action": "sample",
                    "index": i,
                    "answer": answer,
                })

            # Aggregate
            final_answer, confidence = self._aggregate_answers(answers)
            trace.append({
                "action": "aggregate",
                "method": self.aggregation.value,
                "answer": final_answer,
                "confidence": confidence,
            })

            return TechniqueResult(
                success=final_answer is not None,
                output={
                    "answer": final_answer,
                    "confidence": confidence,
                    "samples": samples,
                    "vote_distribution": dict(Counter(a for a in answers if a)),
                },
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
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
# FEW-SHOT PROMPTING
# =============================================================================

class ExampleSelectionStrategy(Enum):
    """Strategies for selecting few-shot examples."""
    RANDOM = "random"
    SIMILAR = "similar"  # Most similar to query
    DIVERSE = "diverse"  # Maximize diversity
    CURRICULUM = "curriculum"  # Easy to hard


@dataclass
class FewShotExample:
    """An example for few-shot prompting."""
    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class FewShot(TechniqueBase):
    """
    Few-Shot Prompting.

    Provides examples before the query to enable in-context learning
    without fine-tuning. The model learns the task pattern from examples.

    Configuration:
        examples: Pool of available examples
        num_examples: How many examples to include (default: 5)
        selection_strategy: How to select examples
        example_format: Template for formatting examples
        separator: Separator between examples

    Usage:
        fs = FewShot(
            examples=[
                FewShotExample("2+2", "4"),
                FewShotExample("3*4", "12"),
                # ...
            ],
            num_examples=3,
            selection_strategy=ExampleSelectionStrategy.SIMILAR,
        )
        result = fs.run("5+7")
    """

    TECHNIQUE_ID = "few_shot"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        model: Optional[Any] = None,
        examples: Optional[List[FewShotExample]] = None,
        num_examples: int = 5,
        selection_strategy: ExampleSelectionStrategy = ExampleSelectionStrategy.RANDOM,
        example_format: str = "Input: {input}\nOutput: {output}",
        separator: str = "\n\n",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.examples = examples or []
        self.num_examples = num_examples
        self.selection_strategy = selection_strategy
        self.example_format = example_format
        self.separator = separator

    def _select_examples(
        self,
        query: str,
        n: int,
    ) -> List[FewShotExample]:
        """Select examples based on strategy."""
        if not self.examples:
            return []

        n = min(n, len(self.examples))

        if self.selection_strategy == ExampleSelectionStrategy.RANDOM:
            return random.sample(self.examples, n)

        elif self.selection_strategy == ExampleSelectionStrategy.SIMILAR:
            # Placeholder - real implementation uses embeddings
            # For now, simple string similarity
            def similarity(ex: FewShotExample) -> float:
                common = set(query.lower().split()) & set(ex.input.lower().split())
                return len(common)

            sorted_examples = sorted(self.examples, key=similarity, reverse=True)
            return sorted_examples[:n]

        elif self.selection_strategy == ExampleSelectionStrategy.DIVERSE:
            # Placeholder - select diverse examples
            return self.examples[:n]

        else:  # CURRICULUM
            # Placeholder - sort by difficulty
            return self.examples[:n]

    def _format_prompt(
        self,
        examples: List[FewShotExample],
        query: str,
    ) -> str:
        """Format the few-shot prompt."""
        formatted_examples = [
            self.example_format.format(input=ex.input, output=ex.output)
            for ex in examples
        ]
        examples_str = self.separator.join(formatted_examples)
        query_str = f"Input: {query}\nOutput:"

        return f"{examples_str}{self.separator}{query_str}"

    def _generate_response(self, prompt: str) -> str:
        """Generate response from model (placeholder)."""
        # Real implementation calls the LLM
        return "[Generated output based on examples]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        query = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        try:
            # Select examples
            selected = self._select_examples(query, self.num_examples)
            trace.append({
                "action": "select_examples",
                "strategy": self.selection_strategy.value,
                "num_selected": len(selected),
            })

            # Format prompt
            prompt = self._format_prompt(selected, query)
            trace.append({
                "action": "format_prompt",
                "prompt_length": len(prompt),
            })

            # Generate
            response = self._generate_response(prompt)
            trace.append({
                "action": "generate",
                "response_length": len(response),
            })

            return TechniqueResult(
                success=True,
                output={
                    "response": response,
                    "examples_used": [ex.input for ex in selected],
                    "prompt": prompt,
                },
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
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
# SKELETON OF THOUGHT
# =============================================================================

@dataclass
class SkeletonPoint:
    """A point in the answer skeleton."""
    index: int
    title: str
    content: Optional[str] = None
    expanded: bool = False


class SkeletonOfThought(TechniqueBase):
    """
    Skeleton-of-Thought Prompting.

    Paper: "Skeleton-of-Thought: Large Language Models Can Do Parallel
           Decoding" (Ning et al., 2023)
    https://arxiv.org/abs/2307.15337

    Generates an answer skeleton/outline first, then fills in each point
    in parallel. This reduces latency through parallelization while
    maintaining coherent structure.

    Process:
        1. Generate skeleton (outline) of the answer
        2. Expand each skeleton point independently (parallelizable)
        3. Combine expanded points into final answer

    Configuration:
        backend: LLMBackend instance or name (uses global registry if string)
        skeleton_prompt: Prompt template for generating skeleton
        parallel_expansion: Whether to expand points in parallel
        expansion_prompt: Prompt template for expanding each point
        max_points: Maximum skeleton points (default: 10)

    Usage:
        from ml_research.backends import MockBackend
        sot = SkeletonOfThought(
            backend=MockBackend(),
            parallel_expansion=True,
        )
        result = sot.run("Explain how neural networks learn")
    """

    TECHNIQUE_ID = "skeleton_of_thought"
    CATEGORY = TechniqueCategory.PROMPTING

    DEFAULT_SKELETON_PROMPT = """Given the question, provide a skeleton outline of the answer.
List the main points that should be covered, numbered 1, 2, 3, etc.
Only provide the skeleton, not the full answer.

Question: {question}

Skeleton:"""

    DEFAULT_EXPANSION_PROMPT = """Expand on this point in detail as part of answering the question.

Original question: {question}
Point to expand: {point_title}

Expansion:"""

    def __init__(
        self,
        backend: Optional[Any] = None,
        skeleton_prompt: Optional[str] = None,
        parallel_expansion: bool = True,
        expansion_prompt: Optional[str] = None,
        max_points: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = self._resolve_backend(backend)
        self.skeleton_prompt = skeleton_prompt or self.DEFAULT_SKELETON_PROMPT
        self.parallel_expansion = parallel_expansion
        self.expansion_prompt = expansion_prompt or self.DEFAULT_EXPANSION_PROMPT
        self.max_points = max_points

    def _resolve_backend(self, backend: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        if backend is not None:
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            return backend

        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=1024, temperature=0.7)
            elif callable(self.backend):
                return self.backend(prompt)
        return "[Placeholder response]"

    def _generate_skeleton(self, question: str) -> List[SkeletonPoint]:
        """Generate the answer skeleton."""
        prompt = self.skeleton_prompt.format(question=question)
        response = self._generate_response(prompt)

        # Parse skeleton points from response
        points = []
        lines = response.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets
                title = line.lstrip('0123456789.-) ').strip()
                if title:
                    points.append(SkeletonPoint(index=i, title=title))
                    if len(points) >= self.max_points:
                        break

        return points if points else [SkeletonPoint(index=0, title="Main point")]

    def _expand_point(self, question: str, point: SkeletonPoint) -> SkeletonPoint:
        """Expand a single skeleton point."""
        prompt = self.expansion_prompt.format(
            question=question,
            point_title=point.title,
        )
        content = self._generate_response(prompt)
        return SkeletonPoint(
            index=point.index,
            title=point.title,
            content=content.strip(),
            expanded=True,
        )

    def _combine_expansions(self, points: List[SkeletonPoint]) -> str:
        """Combine expanded points into final answer."""
        parts = []
        for point in sorted(points, key=lambda p: p.index):
            if point.content:
                parts.append(f"**{point.title}**\n{point.content}")
        return "\n\n".join(parts)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", question=question)

        try:
            # Step 1: Generate skeleton
            skeleton = self._generate_skeleton(question)
            trace.append({
                "action": "generate_skeleton",
                "num_points": len(skeleton),
                "points": [p.title for p in skeleton],
            })

            # Step 2: Expand each point
            expanded_points = []
            if self.parallel_expansion:
                # In a real implementation, this would use async/parallel execution
                # For now, sequential but marked as parallel-capable
                for point in skeleton:
                    expanded = self._expand_point(question, point)
                    expanded_points.append(expanded)
                trace.append({
                    "action": "expand_parallel",
                    "num_expanded": len(expanded_points),
                })
            else:
                for point in skeleton:
                    expanded = self._expand_point(question, point)
                    expanded_points.append(expanded)
                    trace.append({
                        "action": "expand_sequential",
                        "point": point.title,
                    })

            # Step 3: Combine into final answer
            final_answer = self._combine_expansions(expanded_points)
            trace.append({
                "action": "combine",
                "answer_length": len(final_answer),
            })

            self._call_hooks("post_run", answer=final_answer)

            return TechniqueResult(
                success=True,
                output={
                    "answer": final_answer,
                    "skeleton": [p.title for p in skeleton],
                    "expansions": {p.title: p.content for p in expanded_points},
                },
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


# =============================================================================
# CHAIN OF ABSTRACTION
# =============================================================================

@dataclass
class AbstractionLevel:
    """A level in the abstraction hierarchy."""
    level: int
    name: str
    description: str
    content: Optional[str] = None


class ChainOfAbstraction(TechniqueBase):
    """
    Chain-of-Abstraction Prompting.

    Paper: "Chain-of-Abstraction Reasoning for Reliable LLM Responses"
           (Gao et al., 2024)
    https://arxiv.org/abs/2401.17464

    Performs abstract reasoning before moving to concrete details,
    using a multi-level abstraction hierarchy. This helps the model
    maintain logical consistency and avoid hallucinations.

    Process:
        1. Start at highest abstraction level (general principles)
        2. Progressively concretize through each level
        3. Ground final answer in specific details

    Configuration:
        backend: LLMBackend instance or name
        abstraction_levels: List of AbstractionLevel defining the hierarchy
        abstraction_prompts: Dict mapping level names to prompts
        default_levels: Use default 3-level hierarchy if none provided

    Usage:
        coa = ChainOfAbstraction(
            backend=my_backend,
            abstraction_levels=[
                AbstractionLevel(0, "principle", "Core principles"),
                AbstractionLevel(1, "strategy", "General strategies"),
                AbstractionLevel(2, "implementation", "Specific implementation"),
            ],
        )
        result = coa.run("How should I design a caching system?")
    """

    TECHNIQUE_ID = "chain_of_abstraction"
    CATEGORY = TechniqueCategory.PROMPTING

    DEFAULT_LEVELS = [
        AbstractionLevel(0, "abstract", "High-level concepts and principles"),
        AbstractionLevel(1, "intermediate", "General approaches and strategies"),
        AbstractionLevel(2, "concrete", "Specific details and implementation"),
    ]

    DEFAULT_PROMPTS = {
        "abstract": """Consider this question at the highest level of abstraction.
What are the fundamental principles, concepts, or theoretical foundations involved?

Question: {question}
{previous_context}

Abstract reasoning:""",

        "intermediate": """Now consider the general approaches and strategies.
Based on the abstract principles, what are the main methods or approaches?

Question: {question}
Abstract foundation: {previous_level}

Strategic reasoning:""",

        "concrete": """Finally, provide specific, concrete details.
Based on the strategies above, what are the specific steps or implementation details?

Question: {question}
Strategic approach: {previous_level}

Concrete answer:""",
    }

    def __init__(
        self,
        backend: Optional[Any] = None,
        abstraction_levels: Optional[List[AbstractionLevel]] = None,
        abstraction_prompts: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = self._resolve_backend(backend)
        self.abstraction_levels = abstraction_levels or self.DEFAULT_LEVELS.copy()
        self.abstraction_prompts = abstraction_prompts or self.DEFAULT_PROMPTS.copy()

    def _resolve_backend(self, backend: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        if backend is not None:
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            return backend

        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=1024, temperature=0.7)
            elif callable(self.backend):
                return self.backend(prompt)
        return "[Placeholder response]"

    def _reason_at_level(
        self,
        question: str,
        level: AbstractionLevel,
        previous_content: Optional[str],
    ) -> str:
        """Perform reasoning at a specific abstraction level."""
        prompt_template = self.abstraction_prompts.get(
            level.name,
            "Reason about: {question}\nContext: {previous_level}"
        )

        prompt = prompt_template.format(
            question=question,
            previous_level=previous_content or "",
            previous_context=f"\nPrevious reasoning: {previous_content}" if previous_content else "",
        )

        return self._generate_response(prompt)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", question=question)

        try:
            # Sort levels by level number
            sorted_levels = sorted(self.abstraction_levels, key=lambda l: l.level)

            # Process each level
            previous_content = None
            level_outputs = {}

            for level in sorted_levels:
                content = self._reason_at_level(question, level, previous_content)
                level.content = content
                level_outputs[level.name] = content
                previous_content = content

                trace.append({
                    "action": f"reason_level_{level.level}",
                    "level_name": level.name,
                    "content_length": len(content),
                })

            # Final answer is the most concrete level
            final_answer = previous_content

            self._call_hooks("post_run", answer=final_answer)

            return TechniqueResult(
                success=True,
                output={
                    "answer": final_answer,
                    "abstraction_chain": level_outputs,
                    "levels_processed": len(sorted_levels),
                },
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


# =============================================================================
# STEP-BACK PROMPTING
# =============================================================================

class StepBackPrompting(TechniqueBase):
    """
    Step-Back Prompting.

    Paper: "Take a Step Back: Evoking Reasoning via Abstraction in
           Large Language Models" (Zheng et al., 2023)
    https://arxiv.org/abs/2310.06117

    First asks a higher-level, more abstract version of the question,
    then uses that abstraction to answer the specific question.
    Helps models retrieve relevant background knowledge.

    Process:
        1. Generate a "step-back" question (more abstract/general)
        2. Answer the step-back question to get principles/background
        3. Use the background to answer the original specific question

    Configuration:
        backend: LLMBackend instance or name
        stepback_prompt_template: Template for generating step-back question
        answer_prompt_template: Template for final answer using background

    Usage:
        sbp = StepBackPrompting(backend=my_backend)
        result = sbp.run("What happens to entropy when ice melts?")
        # Step-back: "What are the principles of entropy and phase transitions?"
    """

    TECHNIQUE_ID = "step_back_prompting"
    CATEGORY = TechniqueCategory.PROMPTING

    DEFAULT_STEPBACK_TEMPLATE = """Given the following question, generate a more abstract,
higher-level question that would help answer the original question.
The step-back question should ask about underlying principles, concepts, or general knowledge.

Original question: {question}

Step-back question:"""

    DEFAULT_ANSWER_TEMPLATE = """Use the following background knowledge to answer the specific question.

Background (from step-back reasoning):
{background}

Specific question: {question}

Answer:"""

    def __init__(
        self,
        backend: Optional[Any] = None,
        stepback_prompt_template: Optional[str] = None,
        answer_prompt_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = self._resolve_backend(backend)
        self.stepback_prompt_template = stepback_prompt_template or self.DEFAULT_STEPBACK_TEMPLATE
        self.answer_prompt_template = answer_prompt_template or self.DEFAULT_ANSWER_TEMPLATE

    def _resolve_backend(self, backend: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        if backend is not None:
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            return backend

        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=1024, temperature=0.7)
            elif callable(self.backend):
                return self.backend(prompt)
        return "[Placeholder response]"

    def _generate_stepback_question(self, question: str) -> str:
        """Generate the step-back question."""
        prompt = self.stepback_prompt_template.format(question=question)
        return self._generate_response(prompt).strip()

    def _answer_stepback(self, stepback_question: str) -> str:
        """Answer the step-back question to get background knowledge."""
        prompt = f"Question: {stepback_question}\n\nAnswer with relevant principles and concepts:"
        return self._generate_response(prompt)

    def _answer_original(self, question: str, background: str) -> str:
        """Answer the original question using the background."""
        prompt = self.answer_prompt_template.format(
            question=question,
            background=background,
        )
        return self._generate_response(prompt)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", question=question)

        try:
            # Step 1: Generate step-back question
            stepback_question = self._generate_stepback_question(question)
            trace.append({
                "action": "generate_stepback",
                "stepback_question": stepback_question,
            })

            # Step 2: Answer step-back question
            background = self._answer_stepback(stepback_question)
            trace.append({
                "action": "answer_stepback",
                "background_length": len(background),
            })

            # Step 3: Answer original question with background
            final_answer = self._answer_original(question, background)
            trace.append({
                "action": "answer_original",
                "answer_length": len(final_answer),
            })

            self._call_hooks("post_run", answer=final_answer)

            return TechniqueResult(
                success=True,
                output={
                    "answer": final_answer,
                    "stepback_question": stepback_question,
                    "background": background,
                },
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


# =============================================================================
# ANALOGICAL PROMPTING
# =============================================================================

@dataclass
class Analogy:
    """An analogy used for reasoning."""
    source_domain: str
    target_domain: str
    mapping: str
    insight: str


class AnalogicalPrompting(TechniqueBase):
    """
    Analogical Prompting.

    Paper: "Large Language Models as Analogical Reasoners" (Yasunaga et al., 2023)
    https://arxiv.org/abs/2310.01714

    Self-generates relevant analogies/examples before solving a problem,
    then uses those analogies to guide reasoning. Leverages the model's
    broad knowledge to find similar problems and transfer solutions.

    Process:
        1. Generate relevant analogies from other domains
        2. Analyze how the analogies relate to the current problem
        3. Use insights from analogies to solve the problem

    Configuration:
        backend: LLMBackend instance or name
        analogy_generation_prompt: Template for generating analogies
        num_analogies: Number of analogies to generate (default: 3)
        use_analogy_prompt: Template for using analogies in reasoning

    Usage:
        ap = AnalogicalPrompting(
            backend=my_backend,
            num_analogies=3,
        )
        result = ap.run("How can we reduce traffic congestion in cities?")
        # Might generate analogies from: blood flow, packet routing, ant colonies
    """

    TECHNIQUE_ID = "analogical_prompting"
    CATEGORY = TechniqueCategory.PROMPTING

    DEFAULT_ANALOGY_PROMPT = """Generate {num_analogies} relevant analogies for the following problem.
For each analogy, identify:
1. A similar problem from a different domain
2. How it relates to the current problem
3. What solution or insight it suggests

Problem: {question}

Analogies:"""

    DEFAULT_USE_ANALOGY_PROMPT = """Use the following analogies to help solve the problem.
Draw insights from each analogy and synthesize them into a solution.

Problem: {question}

Analogies and insights:
{analogies}

Solution based on analogical reasoning:"""

    def __init__(
        self,
        backend: Optional[Any] = None,
        analogy_generation_prompt: Optional[str] = None,
        num_analogies: int = 3,
        use_analogy_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = self._resolve_backend(backend)
        self.analogy_generation_prompt = analogy_generation_prompt or self.DEFAULT_ANALOGY_PROMPT
        self.num_analogies = num_analogies
        self.use_analogy_prompt = use_analogy_prompt or self.DEFAULT_USE_ANALOGY_PROMPT

    def _resolve_backend(self, backend: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        if backend is not None:
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            return backend

        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=1024, temperature=0.7)
            elif callable(self.backend):
                return self.backend(prompt)
        return "[Placeholder response]"

    def _generate_analogies(self, question: str) -> Tuple[str, List[Analogy]]:
        """Generate analogies for the problem."""
        prompt = self.analogy_generation_prompt.format(
            question=question,
            num_analogies=self.num_analogies,
        )
        response = self._generate_response(prompt)

        # Parse analogies (simplified - real implementation would be more robust)
        analogies = []
        sections = response.split('\n\n')
        for i, section in enumerate(sections[:self.num_analogies]):
            if section.strip():
                analogies.append(Analogy(
                    source_domain=f"Domain {i+1}",
                    target_domain="Current problem",
                    mapping=section.strip()[:100],
                    insight=section.strip(),
                ))

        return response, analogies

    def _reason_with_analogies(self, question: str, analogies_text: str) -> str:
        """Use analogies to solve the problem."""
        prompt = self.use_analogy_prompt.format(
            question=question,
            analogies=analogies_text,
        )
        return self._generate_response(prompt)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", question=question)

        try:
            # Step 1: Generate analogies
            analogies_text, analogies = self._generate_analogies(question)
            trace.append({
                "action": "generate_analogies",
                "num_analogies": len(analogies),
                "analogies": [a.insight[:50] + "..." for a in analogies],
            })

            # Step 2: Use analogies to solve
            final_answer = self._reason_with_analogies(question, analogies_text)
            trace.append({
                "action": "reason_with_analogies",
                "answer_length": len(final_answer),
            })

            self._call_hooks("post_run", answer=final_answer)

            return TechniqueResult(
                success=True,
                output={
                    "answer": final_answer,
                    "analogies": [
                        {"source": a.source_domain, "insight": a.insight}
                        for a in analogies
                    ],
                    "analogies_text": analogies_text,
                },
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


# =============================================================================
# CHAIN OF SYMBOL
# =============================================================================

@dataclass
class SymbolDefinition:
    """Definition of a symbol in the symbolic reasoning system."""
    symbol: str
    meaning: str
    category: str = "general"


class ChainOfSymbol(TechniqueBase):
    """
    Chain-of-Symbol Prompting.

    Paper: "Chain-of-Symbol Prompting Elicits Planning in Large Language
           Models" (Hu et al., 2023)
    https://arxiv.org/abs/2305.10276

    Uses symbolic representations during reasoning, particularly effective
    for spatial and logical reasoning tasks. Converts natural language
    to symbols, reasons symbolically, then converts back.

    Process:
        1. Convert problem to symbolic representation
        2. Perform symbolic reasoning/manipulation
        3. Convert symbolic result back to natural language

    Configuration:
        backend: LLMBackend instance or name
        symbol_set: List of SymbolDefinition for the task
        symbol_rules: Rules for symbolic manipulation
        symbolize_prompt: Template for converting to symbols
        reason_prompt: Template for symbolic reasoning
        desymbolize_prompt: Template for converting back

    Usage:
        cos = ChainOfSymbol(
            backend=my_backend,
            symbol_set=[
                SymbolDefinition("->", "moves to", "spatial"),
                SymbolDefinition("@", "is at location", "spatial"),
                SymbolDefinition("^", "is above", "spatial"),
            ],
        )
        result = cos.run("The ball is on the table. The cat pushes the ball. Where is the ball?")
    """

    TECHNIQUE_ID = "chain_of_symbol"
    CATEGORY = TechniqueCategory.PROMPTING

    DEFAULT_SYMBOL_SET = [
        SymbolDefinition("->", "leads to / causes", "causal"),
        SymbolDefinition("<->", "is equivalent to", "logical"),
        SymbolDefinition("@", "is at / located at", "spatial"),
        SymbolDefinition("^", "is above / on top of", "spatial"),
        SymbolDefinition("v", "is below / under", "spatial"),
        SymbolDefinition("&", "and / together with", "logical"),
        SymbolDefinition("|", "or / alternative", "logical"),
        SymbolDefinition("!", "not / negation", "logical"),
        SymbolDefinition("=>", "implies / therefore", "logical"),
        SymbolDefinition("[X]", "entity X", "entity"),
    ]

    DEFAULT_SYMBOLIZE_PROMPT = """Convert the following problem into symbolic representation.
Use these symbols:
{symbol_definitions}

Problem: {question}

Symbolic representation:"""

    DEFAULT_REASON_PROMPT = """Given the symbolic representation, perform step-by-step symbolic reasoning.
Apply the rules and manipulate the symbols to reach the solution.

Symbols:
{symbol_definitions}

Symbolic problem:
{symbolic_problem}

Symbolic reasoning (step by step):"""

    DEFAULT_DESYMBOLIZE_PROMPT = """Convert the symbolic solution back to natural language.

Symbols used:
{symbol_definitions}

Symbolic solution:
{symbolic_solution}

Natural language answer:"""

    def __init__(
        self,
        backend: Optional[Any] = None,
        symbol_set: Optional[List[SymbolDefinition]] = None,
        symbol_rules: Optional[List[str]] = None,
        symbolize_prompt: Optional[str] = None,
        reason_prompt: Optional[str] = None,
        desymbolize_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backend = self._resolve_backend(backend)
        self.symbol_set = symbol_set or self.DEFAULT_SYMBOL_SET.copy()
        self.symbol_rules = symbol_rules or []
        self.symbolize_prompt = symbolize_prompt or self.DEFAULT_SYMBOLIZE_PROMPT
        self.reason_prompt = reason_prompt or self.DEFAULT_REASON_PROMPT
        self.desymbolize_prompt = desymbolize_prompt or self.DEFAULT_DESYMBOLIZE_PROMPT

    def _resolve_backend(self, backend: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        if backend is not None:
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            return backend

        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(prompt, max_tokens=1024, temperature=0.7)
            elif callable(self.backend):
                return self.backend(prompt)
        return "[Placeholder response]"

    def _format_symbol_definitions(self) -> str:
        """Format symbol definitions for prompts."""
        lines = []
        for sym in self.symbol_set:
            lines.append(f"  {sym.symbol} : {sym.meaning} ({sym.category})")
        return "\n".join(lines)

    def _symbolize(self, question: str) -> str:
        """Convert problem to symbolic representation."""
        prompt = self.symbolize_prompt.format(
            question=question,
            symbol_definitions=self._format_symbol_definitions(),
        )
        return self._generate_response(prompt)

    def _symbolic_reasoning(self, symbolic_problem: str) -> str:
        """Perform symbolic reasoning."""
        prompt = self.reason_prompt.format(
            symbolic_problem=symbolic_problem,
            symbol_definitions=self._format_symbol_definitions(),
        )
        return self._generate_response(prompt)

    def _desymbolize(self, symbolic_solution: str) -> str:
        """Convert symbolic solution back to natural language."""
        prompt = self.desymbolize_prompt.format(
            symbolic_solution=symbolic_solution,
            symbol_definitions=self._format_symbol_definitions(),
        )
        return self._generate_response(prompt)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        question = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        self._call_hooks("pre_run", question=question)

        try:
            # Step 1: Convert to symbolic representation
            symbolic_problem = self._symbolize(question)
            trace.append({
                "action": "symbolize",
                "symbolic_problem": symbolic_problem[:200],
            })

            # Step 2: Perform symbolic reasoning
            symbolic_solution = self._symbolic_reasoning(symbolic_problem)
            trace.append({
                "action": "symbolic_reasoning",
                "symbolic_solution": symbolic_solution[:200],
            })

            # Step 3: Convert back to natural language
            final_answer = self._desymbolize(symbolic_solution)
            trace.append({
                "action": "desymbolize",
                "answer_length": len(final_answer),
            })

            self._call_hooks("post_run", answer=final_answer)

            return TechniqueResult(
                success=True,
                output={
                    "answer": final_answer,
                    "symbolic_problem": symbolic_problem,
                    "symbolic_solution": symbolic_solution,
                    "symbols_used": [s.symbol for s in self.symbol_set],
                },
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


# =============================================================================
# PROMPTING TECHNIQUES REGISTRY
# =============================================================================

PROMPTING_TECHNIQUES = [
    ChainOfThought,
    TreeOfThought,
    GraphOfThought,
    SelfConsistency,
    FewShot,
    SkeletonOfThought,
    ChainOfAbstraction,
    StepBackPrompting,
    AnalogicalPrompting,
    ChainOfSymbol,
]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CoTMode",
    "SearchAlgorithm",
    "EvaluationStrategy",
    "AggregationMethod",
    "ExampleSelectionStrategy",
    # Data classes
    "CoTExample",
    "ThoughtNode",
    "GraphNode",
    "FewShotExample",
    "SkeletonPoint",
    "AbstractionLevel",
    "Analogy",
    "SymbolDefinition",
    # Techniques
    "ChainOfThought",
    "TreeOfThought",
    "GraphOfThought",
    "SelfConsistency",
    "FewShot",
    "SkeletonOfThought",
    "ChainOfAbstraction",
    "StepBackPrompting",
    "AnalogicalPrompting",
    "ChainOfSymbol",
    # Registry
    "PROMPTING_TECHNIQUES",
]
