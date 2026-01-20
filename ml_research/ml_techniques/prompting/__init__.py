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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import abstractmethod
import random

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


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
        mode: zero_shot, few_shot, or auto
        trigger: Phrase to trigger reasoning (default: "Let's think step by step.")
        examples: List of CoTExample for few-shot mode
        extract_answer: Whether to extract final answer from reasoning
        answer_prefix: Prefix marking the final answer

    Usage:
        cot = ChainOfThought(
            model=my_model,
            mode=CoTMode.ZERO_SHOT,
        )
        result = cot.run("What is 23 * 47?")
        # Output includes reasoning steps and final answer
    """

    TECHNIQUE_ID = "chain_of_thought"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        model: Optional[Any] = None,
        mode: CoTMode = CoTMode.ZERO_SHOT,
        trigger: str = "Let's think step by step.",
        examples: Optional[List[CoTExample]] = None,
        extract_answer: bool = True,
        answer_prefix: str = "Therefore, the answer is",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.mode = mode
        self.trigger = trigger
        self.examples = examples or []
        self.extract_answer = extract_answer
        self.answer_prefix = answer_prefix

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
        """Generate response from model (placeholder)."""
        # Real implementation calls the LLM
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
    # Techniques
    "ChainOfThought",
    "TreeOfThought",
    "GraphOfThought",
    "SelfConsistency",
    "FewShot",
]
