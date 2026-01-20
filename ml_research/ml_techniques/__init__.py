"""
ML Application Techniques Module

A modular, composable library of techniques for applying ML models effectively.
These are not architectures but APPLICATION PATTERNS - ways to use models to
solve complex tasks through structured approaches.

=============================================================================
TECHNIQUE CATEGORIES
=============================================================================

1. DECOMPOSITION - Breaking complex tasks into manageable parts
   - Recursive decomposition
   - Least-to-most prompting
   - Hierarchical task decomposition
   - Divide-and-conquer patterns

2. PROMPTING - Structured input formulation
   - Chain-of-thought (CoT)
   - Tree-of-thought (ToT)
   - Graph-of-thought (GoT)
   - Self-consistency
   - Few-shot / zero-shot patterns

3. AGENTIC - Autonomous task execution patterns
   - ReAct (Reasoning + Acting)
   - Tool calling / function calling
   - Multi-agent collaboration
   - Reflection and self-correction
   - Planning (DEPS, Plan-and-Execute)

4. MEMORY - Context and knowledge management
   - RAG (Retrieval Augmented Generation)
   - Episodic memory
   - Working memory patterns
   - Context compression
   - Long-term memory architectures

5. CODE_SYNTHESIS - Code generation and execution
   - Program synthesis
   - RLM (Recursive Language Model)
   - Self-debugging
   - Code-as-intermediate-representation
   - Execution feedback loops

6. ORCHESTRATION - Multi-component coordination
   - Task routing
   - Ensemble methods
   - Hierarchical task networks
   - Mixture of agents
   - Pipeline composition

7. VERIFICATION - Output validation and correction
   - Self-evaluation
   - Chain-of-verification
   - Constitutional methods
   - Debate/adversarial checking
   - Consistency enforcement

8. OPTIMIZATION - Improving technique performance
   - Prompt optimization (DSPy)
   - Automatic prompt engineering
   - Few-shot example selection
   - Temperature/sampling tuning

=============================================================================
DESIGN PHILOSOPHY
=============================================================================

Each technique is:
    - MODULAR: Can be used independently
    - COMPOSABLE: Can be combined with other techniques
    - CONFIGURABLE: Behavior controlled via config files
    - PLUGGABLE: Works with any compatible model backend

Composition Example:
    pipeline = compose([
        Decomposition.recursive(),
        Prompting.chain_of_thought(),
        Agentic.tool_calling(tools=[...]),
        Verification.self_consistency(n=5),
    ])
    result = pipeline.run(task="Complex multi-step problem...")

=============================================================================
USAGE
=============================================================================

# Single technique
from ml_techniques.prompting import ChainOfThought
cot = ChainOfThought(model=my_model)
result = cot.run("What is 23 * 47?")

# Composed pipeline
from ml_techniques import Pipeline, compose
from ml_techniques.decomposition import RecursiveDecomposition
from ml_techniques.agentic import ToolCalling
from ml_techniques.verification import SelfConsistency

pipeline = compose([
    RecursiveDecomposition(max_depth=3),
    ToolCalling(tools=[calculator, search]),
    SelfConsistency(samples=5),
])
result = pipeline.run(complex_task)

# Configuration-driven
from ml_techniques import load_pipeline
pipeline = load_pipeline("configs/research_assistant.yaml")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from abc import ABC, abstractmethod


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class TechniqueCategory(Enum):
    """Categories of ML application techniques."""
    DECOMPOSITION = "decomposition"
    PROMPTING = "prompting"
    AGENTIC = "agentic"
    MEMORY = "memory"
    CODE_SYNTHESIS = "code_synthesis"
    ORCHESTRATION = "orchestration"
    VERIFICATION = "verification"
    OPTIMIZATION = "optimization"


class CompositionMode(Enum):
    """How techniques combine in a pipeline."""
    SEQUENTIAL = "sequential"      # Output of one feeds into next
    PARALLEL = "parallel"          # Run simultaneously, merge results
    CONDITIONAL = "conditional"    # Branch based on conditions
    ITERATIVE = "iterative"        # Loop until condition met


class ExecutionStatus(Enum):
    """Status of technique execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


# =============================================================================
# CORE DATA CLASSES
# =============================================================================

@dataclass
class TechniqueResult:
    """Result from executing a technique."""
    success: bool
    output: Any
    technique_id: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def get_trace(self) -> List[str]:
        """Get execution trace for debugging."""
        return [step.get("action", "unknown") for step in self.intermediate_steps]


@dataclass
class TechniqueConfig:
    """Configuration for a technique instance."""
    technique_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_backend: Optional[str] = None
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2.0,
    })


@dataclass
class TechniqueIndex:
    """Index entry for a technique."""
    id: str
    name: str
    category: TechniqueCategory
    description: str
    paper_url: Optional[str] = None
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    composable_with: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# =============================================================================
# BASE CLASSES
# =============================================================================

class TechniqueBase(ABC):
    """
    Abstract base class for all ML application techniques.

    Inherit from this class to implement a new technique.
    """

    TECHNIQUE_ID: str = "base"
    CATEGORY: TechniqueCategory = TechniqueCategory.PROMPTING

    def __init__(self, config: Optional[TechniqueConfig] = None, **kwargs):
        self.config = config or TechniqueConfig(
            technique_id=self.TECHNIQUE_ID,
            parameters=kwargs,
        )
        self._hooks: Dict[str, List[Callable]] = {
            "pre_run": [],
            "post_run": [],
            "on_error": [],
            "on_step": [],
        }
        # Register config hooks
        for hook_name, hook_list in self.config.hooks.items():
            if hook_name in self._hooks:
                self._hooks[hook_name].extend(hook_list)

    @abstractmethod
    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        """Execute the technique on input data."""
        pass

    def add_hook(self, hook_type: str, hook_fn: Callable) -> None:
        """Add a hook function."""
        if hook_type in self._hooks:
            self._hooks[hook_type].append(hook_fn)

    def _call_hooks(self, hook_type: str, **kwargs) -> None:
        """Call all hooks of a given type."""
        for hook in self._hooks.get(hook_type, []):
            try:
                hook(**kwargs)
            except Exception as e:
                # Log but don't fail on hook errors
                pass

    def __rshift__(self, other: "TechniqueBase") -> "Pipeline":
        """Enable technique >> technique syntax for composition."""
        return Pipeline([self, other])

    def __or__(self, other: "TechniqueBase") -> "ParallelComposition":
        """Enable technique | technique syntax for parallel execution."""
        return ParallelComposition([self, other])


class Pipeline(TechniqueBase):
    """Sequential composition of techniques."""

    TECHNIQUE_ID = "pipeline"
    CATEGORY = TechniqueCategory.ORCHESTRATION

    def __init__(self, techniques: List[TechniqueBase], **kwargs):
        super().__init__(**kwargs)
        self.techniques = techniques

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        start = time.time()

        context = context or {}
        current_input = input_data
        all_steps = []

        for technique in self.techniques:
            self._call_hooks("on_step", technique=technique, input=current_input)

            result = technique.run(current_input, context)
            all_steps.extend(result.intermediate_steps)
            all_steps.append({
                "technique": technique.TECHNIQUE_ID,
                "success": result.success,
            })

            if not result.success:
                return TechniqueResult(
                    success=False,
                    output=None,
                    technique_id=self.TECHNIQUE_ID,
                    execution_time_ms=(time.time() - start) * 1000,
                    intermediate_steps=all_steps,
                    error=f"Pipeline failed at {technique.TECHNIQUE_ID}: {result.error}",
                )

            current_input = result.output
            context["previous_result"] = result

        return TechniqueResult(
            success=True,
            output=current_input,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=all_steps,
        )

    def __rshift__(self, other: TechniqueBase) -> "Pipeline":
        """Extend pipeline with another technique."""
        return Pipeline(self.techniques + [other])


class ParallelComposition(TechniqueBase):
    """Parallel execution of techniques with result merging."""

    TECHNIQUE_ID = "parallel"
    CATEGORY = TechniqueCategory.ORCHESTRATION

    def __init__(
        self,
        techniques: List[TechniqueBase],
        merge_fn: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.techniques = techniques
        self.merge_fn = merge_fn or self._default_merge

    def _default_merge(self, results: List[TechniqueResult]) -> Any:
        """Default merge: return list of outputs."""
        return [r.output for r in results if r.success]

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        import time
        import concurrent.futures

        start = time.time()
        context = context or {}
        results = []

        # Run techniques in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(t.run, input_data, context.copy()): t
                for t in self.techniques
            }

            for future in concurrent.futures.as_completed(futures):
                technique = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(TechniqueResult(
                        success=False,
                        output=None,
                        technique_id=technique.TECHNIQUE_ID,
                        execution_time_ms=0,
                        error=str(e),
                    ))

        # Merge results
        merged_output = self.merge_fn(results)
        all_succeeded = all(r.success for r in results)

        return TechniqueResult(
            success=all_succeeded,
            output=merged_output,
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=[
                {"technique": r.technique_id, "success": r.success}
                for r in results
            ],
        )

    def __or__(self, other: TechniqueBase) -> "ParallelComposition":
        """Extend parallel composition."""
        return ParallelComposition(
            self.techniques + [other],
            merge_fn=self.merge_fn,
        )


# =============================================================================
# TECHNIQUE REGISTRY
# =============================================================================

TECHNIQUES: List[TechniqueIndex] = [
    # =========================================================================
    # DECOMPOSITION TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="recursive_decomposition",
        name="Recursive Decomposition",
        category=TechniqueCategory.DECOMPOSITION,
        description="""
        Recursively breaks down complex tasks into simpler subtasks until
        each subtask is simple enough to solve directly. Results are then
        composed back up to form the final answer.

        Pattern:
            1. Check if task is simple enough (base case)
            2. If not, decompose into subtasks
            3. Recursively solve each subtask
            4. Combine subtask results
        """,
        year=2022,
        composable_with=["chain_of_thought", "tool_calling", "self_consistency"],
        config_schema={
            "max_depth": {"type": "int", "default": 5},
            "simplicity_threshold": {"type": "float", "default": 0.8},
            "decomposition_strategy": {"type": "str", "enum": ["binary", "multi", "adaptive"]},
        },
        tags=["decomposition", "recursive", "divide-and-conquer"],
    ),
    TechniqueIndex(
        id="least_to_most",
        name="Least-to-Most Prompting",
        category=TechniqueCategory.DECOMPOSITION,
        description="""
        Decomposes complex problems into simpler subproblems, then solves
        them in order from simplest to most complex, using previous solutions
        as context for harder problems.

        Paper: "Least-to-Most Prompting Enables Complex Reasoning in Large
               Language Models" (Zhou et al., 2022)
        """,
        paper_url="https://arxiv.org/abs/2205.10625",
        year=2022,
        authors=["Denny Zhou", "Nathanael Schärli", "Le Hou", "Jason Wei"],
        composable_with=["chain_of_thought", "few_shot"],
        tags=["decomposition", "prompting", "reasoning"],
    ),
    TechniqueIndex(
        id="hierarchical_task_decomposition",
        name="Hierarchical Task Decomposition",
        category=TechniqueCategory.DECOMPOSITION,
        description="""
        Organizes tasks into a hierarchy where high-level goals are broken
        down into increasingly specific subgoals. Often used in planning
        and robotics (Hierarchical Task Networks - HTN).
        """,
        composable_with=["planning", "tool_calling"],
        tags=["decomposition", "planning", "hierarchical"],
    ),

    # =========================================================================
    # PROMPTING TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="chain_of_thought",
        name="Chain-of-Thought (CoT)",
        category=TechniqueCategory.PROMPTING,
        description="""
        Prompts the model to show its reasoning step-by-step before giving
        a final answer. Dramatically improves performance on reasoning tasks.

        Paper: "Chain-of-Thought Prompting Elicits Reasoning in Large
               Language Models" (Wei et al., 2022)

        Variants:
            - Zero-shot CoT: "Let's think step by step"
            - Few-shot CoT: Provide examples with reasoning
            - Auto-CoT: Automatically generate chain-of-thought examples
        """,
        paper_url="https://arxiv.org/abs/2201.11903",
        year=2022,
        authors=["Jason Wei", "Xuezhi Wang", "Dale Schuurmans"],
        composable_with=["self_consistency", "few_shot", "verification"],
        config_schema={
            "cot_trigger": {"type": "str", "default": "Let's think step by step."},
            "extract_answer": {"type": "bool", "default": True},
        },
        tags=["prompting", "reasoning", "chain-of-thought"],
    ),
    TechniqueIndex(
        id="tree_of_thought",
        name="Tree-of-Thought (ToT)",
        category=TechniqueCategory.PROMPTING,
        description="""
        Extends chain-of-thought by exploring multiple reasoning paths
        as a tree, using search algorithms (BFS/DFS) to find the best path.
        Enables deliberate problem-solving with backtracking.

        Paper: "Tree of Thoughts: Deliberate Problem Solving with Large
               Language Models" (Yao et al., 2023)
        """,
        paper_url="https://arxiv.org/abs/2305.10601",
        year=2023,
        authors=["Shunyu Yao", "Dian Yu", "Jeffrey Zhao", "Izhak Shafran"],
        composable_with=["self_evaluation", "chain_of_thought"],
        config_schema={
            "branching_factor": {"type": "int", "default": 3},
            "max_depth": {"type": "int", "default": 5},
            "search_algorithm": {"type": "str", "enum": ["bfs", "dfs", "beam"]},
            "evaluation_strategy": {"type": "str", "enum": ["value", "vote"]},
        },
        tags=["prompting", "reasoning", "search", "tree"],
    ),
    TechniqueIndex(
        id="graph_of_thought",
        name="Graph-of-Thought (GoT)",
        category=TechniqueCategory.PROMPTING,
        description="""
        Extends tree-of-thought to allow arbitrary graph structures,
        enabling thoughts to merge, loop, and form complex reasoning patterns.

        Paper: "Graph of Thoughts: Solving Elaborate Problems with Large
               Language Models" (Besta et al., 2023)
        """,
        paper_url="https://arxiv.org/abs/2308.09687",
        year=2023,
        composable_with=["tree_of_thought", "self_evaluation"],
        tags=["prompting", "reasoning", "graph"],
    ),
    TechniqueIndex(
        id="self_consistency",
        name="Self-Consistency",
        category=TechniqueCategory.PROMPTING,
        description="""
        Samples multiple reasoning paths and takes a majority vote on the
        final answer. Based on the intuition that correct reasoning paths
        are more likely to arrive at the same answer.

        Paper: "Self-Consistency Improves Chain of Thought Reasoning in
               Language Models" (Wang et al., 2022)
        """,
        paper_url="https://arxiv.org/abs/2203.11171",
        year=2022,
        composable_with=["chain_of_thought", "tree_of_thought"],
        config_schema={
            "num_samples": {"type": "int", "default": 5},
            "temperature": {"type": "float", "default": 0.7},
            "aggregation": {"type": "str", "enum": ["majority", "weighted", "unanimous"]},
        },
        tags=["prompting", "ensemble", "voting"],
    ),
    TechniqueIndex(
        id="few_shot",
        name="Few-Shot Prompting",
        category=TechniqueCategory.PROMPTING,
        description="""
        Provides a few examples of input-output pairs before the actual
        query, enabling in-context learning without fine-tuning.

        Key considerations:
            - Example selection (similarity, diversity)
            - Example ordering
            - Number of examples (typically 3-8)
        """,
        year=2020,
        composable_with=["chain_of_thought", "tool_calling"],
        config_schema={
            "num_examples": {"type": "int", "default": 5},
            "selection_strategy": {"type": "str", "enum": ["random", "similar", "diverse"]},
        },
        tags=["prompting", "in-context-learning"],
    ),

    # =========================================================================
    # AGENTIC TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="react",
        name="ReAct (Reasoning + Acting)",
        category=TechniqueCategory.AGENTIC,
        description="""
        Interleaves reasoning traces with actions in a loop:
        Thought -> Action -> Observation -> Thought -> ...

        Combines chain-of-thought reasoning with tool use for grounded,
        multi-step problem solving.

        Paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
               (Yao et al., 2022)
        """,
        paper_url="https://arxiv.org/abs/2210.03629",
        year=2022,
        authors=["Shunyu Yao", "Jeffrey Zhao", "Dian Yu"],
        composable_with=["tool_calling", "chain_of_thought", "memory"],
        config_schema={
            "max_iterations": {"type": "int", "default": 10},
            "tools": {"type": "list", "items": "tool_spec"},
            "observation_format": {"type": "str", "default": "Observation: {result}"},
        },
        tags=["agentic", "reasoning", "tool-use", "react"],
    ),
    TechniqueIndex(
        id="tool_calling",
        name="Tool Calling / Function Calling",
        category=TechniqueCategory.AGENTIC,
        description="""
        Enables models to call external tools/functions to extend capabilities:
        - Search engines, databases
        - Calculators, code execution
        - APIs, file systems

        Implementation patterns:
            - JSON function schemas
            - ReAct-style interleaving
            - Parallel tool calls
        """,
        year=2023,
        composable_with=["react", "planning", "verification"],
        config_schema={
            "tools": {"type": "list", "items": "tool_spec"},
            "parallel_calls": {"type": "bool", "default": False},
            "max_calls_per_turn": {"type": "int", "default": 5},
        },
        tags=["agentic", "tools", "function-calling"],
    ),
    TechniqueIndex(
        id="reflexion",
        name="Reflexion",
        category=TechniqueCategory.AGENTIC,
        description="""
        Agent reflects on failures to improve future attempts. Maintains
        a memory of past reflections to avoid repeating mistakes.

        Paper: "Reflexion: Language Agents with Verbal Reinforcement Learning"
               (Shinn et al., 2023)
        """,
        paper_url="https://arxiv.org/abs/2303.11366",
        year=2023,
        composable_with=["react", "memory", "self_evaluation"],
        config_schema={
            "max_trials": {"type": "int", "default": 3},
            "reflection_prompt": {"type": "str"},
            "memory_type": {"type": "str", "enum": ["sliding", "summarized", "full"]},
        },
        tags=["agentic", "reflection", "self-improvement"],
    ),
    TechniqueIndex(
        id="planning",
        name="Planning (Plan-and-Execute)",
        category=TechniqueCategory.AGENTIC,
        description="""
        Separates planning from execution:
        1. Generate a plan (sequence of steps)
        2. Execute each step
        3. Optionally replan based on results

        Variants:
            - DEPS: Describe, Explain, Plan, Select
            - Plan-and-Solve
            - Least-to-Most planning
        """,
        composable_with=["tool_calling", "decomposition", "verification"],
        config_schema={
            "planning_strategy": {"type": "str", "enum": ["deps", "simple", "hierarchical"]},
            "allow_replan": {"type": "bool", "default": True},
            "max_plan_steps": {"type": "int", "default": 10},
        },
        tags=["agentic", "planning", "execution"],
    ),
    TechniqueIndex(
        id="multi_agent",
        name="Multi-Agent Collaboration",
        category=TechniqueCategory.AGENTIC,
        description="""
        Multiple specialized agents collaborate to solve complex tasks.

        Patterns:
            - Debate: Agents argue different positions
            - Division of labor: Agents handle different subtasks
            - Hierarchical: Manager agent coordinates workers
            - Peer review: Agents check each other's work
        """,
        composable_with=["planning", "verification", "tool_calling"],
        config_schema={
            "agents": {"type": "list", "items": "agent_spec"},
            "coordination": {"type": "str", "enum": ["hierarchical", "peer", "debate"]},
            "communication_protocol": {"type": "str"},
        },
        tags=["agentic", "multi-agent", "collaboration"],
    ),

    # =========================================================================
    # MEMORY TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="rag",
        name="Retrieval-Augmented Generation (RAG)",
        category=TechniqueCategory.MEMORY,
        description="""
        Retrieves relevant documents/passages from a knowledge base and
        includes them in the prompt context for grounded generation.

        Components:
            - Document store (vector DB, BM25, hybrid)
            - Retriever (dense, sparse, reranking)
            - Generator (LLM with retrieved context)

        Paper: "Retrieval-Augmented Generation for Knowledge-Intensive
               NLP Tasks" (Lewis et al., 2020)
        """,
        paper_url="https://arxiv.org/abs/2005.11401",
        year=2020,
        composable_with=["chain_of_thought", "tool_calling", "verification"],
        config_schema={
            "retriever_type": {"type": "str", "enum": ["dense", "sparse", "hybrid"]},
            "top_k": {"type": "int", "default": 5},
            "rerank": {"type": "bool", "default": False},
            "chunk_size": {"type": "int", "default": 512},
        },
        tags=["memory", "retrieval", "rag", "grounding"],
    ),
    TechniqueIndex(
        id="memory_bank",
        name="Memory Bank / Episodic Memory",
        category=TechniqueCategory.MEMORY,
        description="""
        Maintains a persistent memory of past interactions, experiences,
        or facts that can be queried and updated over time.

        Types:
            - Episodic: Specific past experiences
            - Semantic: General knowledge/facts
            - Working: Current task context
        """,
        composable_with=["rag", "reflexion", "planning"],
        config_schema={
            "memory_type": {"type": "str", "enum": ["episodic", "semantic", "hybrid"]},
            "max_memories": {"type": "int", "default": 1000},
            "retrieval_strategy": {"type": "str"},
        },
        tags=["memory", "long-term", "episodic"],
    ),
    TechniqueIndex(
        id="context_compression",
        name="Context Compression",
        category=TechniqueCategory.MEMORY,
        description="""
        Compresses long contexts to fit within model limits while
        preserving essential information.

        Methods:
            - Summarization
            - Selective attention
            - Learned compression (AutoCompressors)
            - Token pruning
        """,
        composable_with=["rag", "chain_of_thought"],
        config_schema={
            "compression_ratio": {"type": "float", "default": 0.5},
            "method": {"type": "str", "enum": ["summarize", "selective", "prune"]},
        },
        tags=["memory", "compression", "context"],
    ),

    # =========================================================================
    # CODE SYNTHESIS TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="rlm",
        name="RLM (Recursive Language Model)",
        category=TechniqueCategory.CODE_SYNTHESIS,
        description="""
        Algorithmic prompt dissection into code variables. Transforms
        natural language specifications into executable code through
        recursive decomposition and variable extraction.

        Process:
            1. Parse natural language into structured components
            2. Identify variables, constraints, relationships
            3. Generate code that captures the specification
            4. Recursively refine until executable
        """,
        composable_with=["recursive_decomposition", "self_debugging", "verification"],
        config_schema={
            "language": {"type": "str", "default": "python"},
            "max_recursion": {"type": "int", "default": 5},
            "variable_extraction": {"type": "str", "enum": ["explicit", "inferred", "hybrid"]},
        },
        tags=["code", "synthesis", "recursive", "rlm"],
    ),
    TechniqueIndex(
        id="program_synthesis",
        name="Program Synthesis",
        category=TechniqueCategory.CODE_SYNTHESIS,
        description="""
        Generates programs from specifications (natural language, examples,
        or formal specs). Often uses search + verification.

        Approaches:
            - Neural program synthesis
            - Inductive synthesis (from I/O examples)
            - Deductive synthesis (from specs)
        """,
        composable_with=["verification", "self_debugging"],
        tags=["code", "synthesis", "program"],
    ),
    TechniqueIndex(
        id="self_debugging",
        name="Self-Debugging",
        category=TechniqueCategory.CODE_SYNTHESIS,
        description="""
        Model debugs its own generated code by:
            1. Running code and observing errors
            2. Analyzing error messages
            3. Generating fixes
            4. Iterating until correct

        Paper: "Self-Debugging: Teaching LLMs to Debug Their Own Code"
        """,
        paper_url="https://arxiv.org/abs/2304.05128",
        year=2023,
        composable_with=["program_synthesis", "tool_calling"],
        config_schema={
            "max_debug_iterations": {"type": "int", "default": 5},
            "execution_timeout": {"type": "int", "default": 30},
            "include_traceback": {"type": "bool", "default": True},
        },
        tags=["code", "debugging", "self-correction"],
    ),
    TechniqueIndex(
        id="code_as_policy",
        name="Code-as-Policy / Code-as-Actions",
        category=TechniqueCategory.CODE_SYNTHESIS,
        description="""
        Uses code as an intermediate representation for complex actions.
        Instead of outputting actions directly, generates code that
        executes to produce actions.

        Benefits:
            - Composable (functions, loops)
            - Precise (exact parameters)
            - Verifiable (can inspect before running)
        """,
        composable_with=["tool_calling", "planning"],
        tags=["code", "policy", "actions", "robotics"],
    ),

    # =========================================================================
    # ORCHESTRATION TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="task_routing",
        name="Task Routing",
        category=TechniqueCategory.ORCHESTRATION,
        description="""
        Routes incoming tasks to appropriate handlers/models based on
        task characteristics.

        Approaches:
            - Classifier-based routing
            - Embedding similarity
            - Rule-based routing
            - LLM-based routing
        """,
        composable_with=["multi_agent", "ensemble"],
        config_schema={
            "routing_strategy": {"type": "str", "enum": ["classifier", "embedding", "rules", "llm"]},
            "routes": {"type": "list", "items": "route_spec"},
        },
        tags=["orchestration", "routing"],
    ),
    TechniqueIndex(
        id="ensemble",
        name="Ensemble Methods",
        category=TechniqueCategory.ORCHESTRATION,
        description="""
        Combines outputs from multiple models/techniques for better
        results.

        Strategies:
            - Voting (majority, weighted)
            - Averaging (for continuous outputs)
            - Stacking (meta-learner)
            - Mixture of experts
        """,
        composable_with=["self_consistency", "multi_agent"],
        config_schema={
            "models": {"type": "list", "items": "model_spec"},
            "aggregation": {"type": "str", "enum": ["vote", "average", "stack", "best"]},
        },
        tags=["orchestration", "ensemble"],
    ),
    TechniqueIndex(
        id="hooks_system",
        name="Hooks System",
        category=TechniqueCategory.ORCHESTRATION,
        description="""
        Intervention points in model execution for monitoring, modification,
        or control flow changes.

        Hook types:
            - Pre-execution hooks (modify input)
            - Post-execution hooks (modify output)
            - Error hooks (handle failures)
            - Step hooks (monitor progress)
            - Conditional hooks (branch execution)
        """,
        composable_with=["all"],
        config_schema={
            "hooks": {"type": "dict", "items": "hook_spec"},
            "async_hooks": {"type": "bool", "default": False},
        },
        tags=["orchestration", "hooks", "middleware"],
    ),

    # =========================================================================
    # VERIFICATION TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="self_evaluation",
        name="Self-Evaluation",
        category=TechniqueCategory.VERIFICATION,
        description="""
        Model evaluates its own outputs for correctness, quality, or
        adherence to requirements.

        Used in:
            - Tree-of-thought (evaluating paths)
            - Reflexion (identifying failures)
            - Constitutional AI (checking principles)
        """,
        composable_with=["tree_of_thought", "reflexion", "chain_of_thought"],
        config_schema={
            "evaluation_criteria": {"type": "list", "items": "str"},
            "scoring_scale": {"type": "str", "enum": ["binary", "likert", "numeric"]},
        },
        tags=["verification", "self-evaluation"],
    ),
    TechniqueIndex(
        id="chain_of_verification",
        name="Chain-of-Verification (CoVe)",
        category=TechniqueCategory.VERIFICATION,
        description="""
        Generates verification questions about its own response, answers
        them, and revises the response based on any inconsistencies found.

        Paper: "Chain-of-Verification Reduces Hallucination in Large
               Language Models" (Dhuliawala et al., 2023)
        """,
        paper_url="https://arxiv.org/abs/2309.11495",
        year=2023,
        composable_with=["chain_of_thought", "self_evaluation"],
        tags=["verification", "hallucination", "fact-checking"],
    ),
    TechniqueIndex(
        id="constitutional",
        name="Constitutional Methods",
        category=TechniqueCategory.VERIFICATION,
        description="""
        Checks outputs against a set of principles/rules (constitution).
        Used for safety, alignment, and quality control.

        Paper: "Constitutional AI: Harmlessness from AI Feedback"
               (Bai et al., 2022)
        """,
        paper_url="https://arxiv.org/abs/2212.08073",
        year=2022,
        authors=["Yuntao Bai", "Saurav Kadavath", "Sandipan Kundu"],
        composable_with=["self_evaluation", "reflexion"],
        config_schema={
            "principles": {"type": "list", "items": "str"},
            "enforcement": {"type": "str", "enum": ["soft", "hard"]},
        },
        tags=["verification", "safety", "constitutional"],
    ),
    TechniqueIndex(
        id="debate",
        name="Debate / Adversarial Verification",
        category=TechniqueCategory.VERIFICATION,
        description="""
        Multiple agents argue different positions, with a judge determining
        the correct answer. Helps surface errors and edge cases.

        Paper: "AI Safety via Debate" (Irving et al., 2018)
        """,
        paper_url="https://arxiv.org/abs/1805.00899",
        year=2018,
        composable_with=["multi_agent", "self_evaluation"],
        tags=["verification", "debate", "adversarial"],
    ),

    # =========================================================================
    # OPTIMIZATION TECHNIQUES
    # =========================================================================
    TechniqueIndex(
        id="dspy",
        name="DSPy Compilation",
        category=TechniqueCategory.OPTIMIZATION,
        description="""
        Declarative framework for optimizing LM pipelines. Defines
        signatures and modules, then compiles to optimized prompts.

        Paper: "DSPy: Compiling Declarative Language Model Calls into
               Self-Improving Pipelines" (Khattab et al., 2023)

        Components:
            - Signatures: Input/output specs
            - Modules: Composable operations
            - Teleprompters: Optimization algorithms
        """,
        paper_url="https://arxiv.org/abs/2310.03714",
        year=2023,
        authors=["Omar Khattab", "Arnav Singhvi", "Paridhi Maheshwari"],
        composable_with=["few_shot", "chain_of_thought"],
        config_schema={
            "optimizer": {"type": "str", "enum": ["bootstrap", "mipro", "copro"]},
            "metric": {"type": "callable"},
            "num_candidates": {"type": "int", "default": 10},
        },
        tags=["optimization", "compilation", "dspy"],
    ),
    TechniqueIndex(
        id="automatic_prompt_engineering",
        name="Automatic Prompt Engineering (APE)",
        category=TechniqueCategory.OPTIMIZATION,
        description="""
        Automatically generates and optimizes prompts using search or
        learning algorithms.

        Paper: "Large Language Models Are Human-Level Prompt Engineers"
               (Zhou et al., 2022)
        """,
        paper_url="https://arxiv.org/abs/2211.01910",
        year=2022,
        composable_with=["few_shot", "chain_of_thought"],
        config_schema={
            "search_algorithm": {"type": "str", "enum": ["beam", "monte_carlo", "evolutionary"]},
            "num_iterations": {"type": "int", "default": 50},
        },
        tags=["optimization", "prompts", "automatic"],
    ),
]

# Create lookup dictionaries
TECHNIQUE_BY_ID: Dict[str, TechniqueIndex] = {t.id: t for t in TECHNIQUES}
TECHNIQUES_BY_CATEGORY: Dict[TechniqueCategory, List[TechniqueIndex]] = {}
for technique in TECHNIQUES:
    if technique.category not in TECHNIQUES_BY_CATEGORY:
        TECHNIQUES_BY_CATEGORY[technique.category] = []
    TECHNIQUES_BY_CATEGORY[technique.category].append(technique)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_technique_info(technique_id: str) -> Optional[TechniqueIndex]:
    """Get information about a technique by ID."""
    return TECHNIQUE_BY_ID.get(technique_id)


def list_techniques(category: Optional[TechniqueCategory] = None) -> List[TechniqueIndex]:
    """List all techniques, optionally filtered by category."""
    if category:
        return TECHNIQUES_BY_CATEGORY.get(category, [])
    return TECHNIQUES


def get_composable_with(technique_id: str) -> List[TechniqueIndex]:
    """Get techniques that compose well with the given technique."""
    technique = TECHNIQUE_BY_ID.get(technique_id)
    if not technique:
        return []
    return [TECHNIQUE_BY_ID[tid] for tid in technique.composable_with if tid in TECHNIQUE_BY_ID]


def compose(techniques: List[TechniqueBase]) -> Pipeline:
    """Compose multiple techniques into a pipeline."""
    return Pipeline(techniques)


def load_config(config_path: str) -> TechniqueConfig:
    """Load technique configuration from YAML file."""
    import yaml
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return TechniqueConfig(**data)


# =============================================================================
# EXPORTS
# =============================================================================

__version__ = "0.1.0"
__status__ = "indexed"

__all__ = [
    # Enums
    "TechniqueCategory",
    "CompositionMode",
    "ExecutionStatus",
    # Data classes
    "TechniqueResult",
    "TechniqueConfig",
    "TechniqueIndex",
    # Base classes
    "TechniqueBase",
    "Pipeline",
    "ParallelComposition",
    # Registry
    "TECHNIQUES",
    "TECHNIQUE_BY_ID",
    "TECHNIQUES_BY_CATEGORY",
    # Functions
    "get_technique_info",
    "list_techniques",
    "get_composable_with",
    "compose",
    "load_config",
]
