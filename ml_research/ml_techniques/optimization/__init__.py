"""
Optimization Techniques

Methods for optimizing LM pipelines, prompts, and configurations
to improve performance on specific tasks.

=============================================================================
TECHNIQUES
=============================================================================

1. DSPy
   - Declarative framework for LM programs
   - Signatures define input/output specs
   - Teleprompters optimize prompts

2. AutomaticPromptEngineering (APE)
   - Search for optimal prompts
   - Generate, evaluate, iterate

3. FewShotOptimizer
   - Optimize few-shot example selection
   - Find best examples for task

4. HyperparameterTuning
   - Tune temperature, top_p, etc.
   - Grid search, random search, Bayesian

=============================================================================
OPTIMIZATION PIPELINE
=============================================================================

    Define Task → Generate Candidates → Evaluate → Select Best → Refine
         │              │                  │           │          │
         ▼              ▼                  ▼           ▼          ▼
    [Signature]   [Prompt/Config]    [Metric]    [Winner]   [Iterate]

DSPy Workflow:
    1. Define Signature (inputs → outputs)
    2. Create Module (ChainOfThought, etc.)
    3. Define metric function
    4. Run Teleprompter to optimize
    5. Deploy optimized module
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from enum import Enum
from abc import abstractmethod
import time
import random

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


# =============================================================================
# DSPY-STYLE OPTIMIZATION
# =============================================================================

@dataclass
class Signature:
    """
    A signature defines the input/output specification for an LM call.

    Example:
        sig = Signature(
            inputs=["question"],
            outputs=["answer"],
            instructions="Answer the question concisely.",
        )
    """
    inputs: List[str]
    outputs: List[str]
    instructions: str = ""
    input_descriptions: Dict[str, str] = field(default_factory=dict)
    output_descriptions: Dict[str, str] = field(default_factory=dict)

    def to_prompt_template(self) -> str:
        """Convert signature to a prompt template."""
        parts = []

        if self.instructions:
            parts.append(self.instructions)
            parts.append("")

        for inp in self.inputs:
            desc = self.input_descriptions.get(inp, inp)
            parts.append(f"{desc}: {{{inp}}}")

        parts.append("")

        for out in self.outputs:
            desc = self.output_descriptions.get(out, out)
            parts.append(f"{desc}:")

        return "\n".join(parts)


@dataclass
class Example:
    """A training/evaluation example for optimization."""
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizerType(Enum):
    """Types of DSPy-style optimizers."""
    BOOTSTRAP = "bootstrap"          # Bootstrap few-shot examples
    MIPRO = "mipro"                  # Multi-stage instruction optimization
    COPRO = "copro"                  # Coordinate prompt optimization
    RANDOM = "random"                # Random search baseline


@dataclass
class OptimizedModule:
    """Result of optimization - an optimized LM module."""
    signature: Signature
    optimized_prompt: str
    few_shot_examples: List[Example] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metric_score: float = 0.0
    optimization_trace: List[Dict] = field(default_factory=list)


class DSPy(TechniqueBase):
    """
    DSPy-style declarative optimization for LM programs.

    Paper: "DSPy: Compiling Declarative Language Model Calls into
           Self-Improving Pipelines" (Khattab et al., 2023)
    https://arxiv.org/abs/2310.03714

    Components:
        - Signatures: Define input/output specs
        - Modules: Composable LM operations
        - Teleprompters: Optimization algorithms
        - Metrics: Evaluation functions

    Configuration:
        signature: Input/output specification
        optimizer: Optimization algorithm
        metric: Function to evaluate outputs
        train_examples: Examples for optimization
        num_candidates: Number of prompt candidates

    Usage:
        # Define signature
        sig = Signature(
            inputs=["question"],
            outputs=["answer"],
            instructions="Answer the math question step by step.",
        )

        # Create optimizer
        dspy = DSPy(
            model=my_model,
            signature=sig,
            optimizer=OptimizerType.BOOTSTRAP,
            metric=lambda pred, gold: pred == gold,
            train_examples=train_data,
        )

        # Optimize
        result = dspy.run()
        optimized_module = result.output["module"]
    """

    TECHNIQUE_ID = "dspy"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        model: Optional[Any] = None,
        signature: Optional[Signature] = None,
        optimizer: OptimizerType = OptimizerType.BOOTSTRAP,
        metric: Optional[Callable[[Any, Any], float]] = None,
        train_examples: Optional[List[Example]] = None,
        num_candidates: int = 10,
        num_iterations: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.signature = signature or Signature(
            inputs=["input"],
            outputs=["output"],
        )
        self.optimizer = optimizer
        self.metric = metric or (lambda pred, gold: float(pred == gold))
        self.train_examples = train_examples or []
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations

    def _generate_prompt_candidate(
        self,
        base_prompt: str,
        iteration: int,
    ) -> str:
        """Generate a prompt candidate (placeholder)."""
        # Real implementation uses LLM to generate variations
        variations = [
            f"{base_prompt}\nThink carefully.",
            f"{base_prompt}\nLet's work through this step by step.",
            f"{base_prompt}\nProvide a detailed answer.",
            f"Instructions: {base_prompt}",
            f"{base_prompt}\nBe precise and accurate.",
        ]
        return random.choice(variations)

    def _select_demonstrations(
        self,
        examples: List[Example],
        k: int,
    ) -> List[Example]:
        """Select demonstrations for few-shot (placeholder)."""
        # Real implementation uses more sophisticated selection
        if len(examples) <= k:
            return examples
        return random.sample(examples, k)

    def _evaluate_candidate(
        self,
        prompt: str,
        demonstrations: List[Example],
        eval_examples: List[Example],
    ) -> float:
        """Evaluate a prompt candidate on examples (placeholder)."""
        # Real implementation runs the prompt and evaluates
        scores = []
        for ex in eval_examples[:5]:  # Limit for efficiency
            # Placeholder: random score
            score = random.uniform(0.5, 1.0)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _bootstrap_optimize(
        self,
        trace: List[Dict],
    ) -> OptimizedModule:
        """Bootstrap few-shot optimization."""
        base_prompt = self.signature.to_prompt_template()
        best_prompt = base_prompt
        best_demos: List[Example] = []
        best_score = 0.0

        for iteration in range(self.num_iterations):
            # Generate candidates
            candidates = [
                self._generate_prompt_candidate(base_prompt, iteration)
                for _ in range(self.num_candidates)
            ]

            # Evaluate each candidate with different demo selections
            for prompt in candidates:
                demos = self._select_demonstrations(self.train_examples, k=3)
                score = self._evaluate_candidate(
                    prompt, demos, self.train_examples
                )

                if score > best_score:
                    best_score = score
                    best_prompt = prompt
                    best_demos = demos

            trace.append({
                "action": "iteration",
                "iteration": iteration,
                "best_score": best_score,
            })

        return OptimizedModule(
            signature=self.signature,
            optimized_prompt=best_prompt,
            few_shot_examples=best_demos,
            metric_score=best_score,
            optimization_trace=trace,
        )

    def _mipro_optimize(
        self,
        trace: List[Dict],
    ) -> OptimizedModule:
        """Multi-stage instruction optimization (placeholder)."""
        # Real MIPRO has multiple stages:
        # 1. Generate instruction candidates
        # 2. Optimize demonstrations
        # 3. Combine and refine
        return self._bootstrap_optimize(trace)

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        try:
            # Run optimization
            if self.optimizer == OptimizerType.BOOTSTRAP:
                module = self._bootstrap_optimize(trace)
            elif self.optimizer == OptimizerType.MIPRO:
                module = self._mipro_optimize(trace)
            else:
                module = self._bootstrap_optimize(trace)

            return TechniqueResult(
                success=True,
                output={
                    "module": module,
                    "optimized_prompt": module.optimized_prompt,
                    "num_demonstrations": len(module.few_shot_examples),
                    "metric_score": module.metric_score,
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
# AUTOMATIC PROMPT ENGINEERING
# =============================================================================

class SearchAlgorithm(Enum):
    """Search algorithms for prompt optimization."""
    BEAM = "beam"
    MONTE_CARLO = "monte_carlo"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_FREE = "gradient_free"


@dataclass
class PromptCandidate:
    """A candidate prompt during optimization."""
    prompt: str
    score: float
    generation: int = 0
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutomaticPromptEngineering(TechniqueBase):
    """
    Automatic Prompt Engineering (APE).

    Paper: "Large Language Models Are Human-Level Prompt Engineers"
           (Zhou et al., 2022)
    https://arxiv.org/abs/2211.01910

    Automatically generates and optimizes prompts using search.

    Process:
        1. Generate initial prompt candidates
        2. Evaluate on validation examples
        3. Select best and generate variations
        4. Iterate until convergence

    Configuration:
        search_algorithm: beam, monte_carlo, evolutionary
        num_candidates: Candidates per generation
        num_generations: Number of generations
        metric: Evaluation function
        examples: Validation examples

    Usage:
        ape = AutomaticPromptEngineering(
            model=my_model,
            task_description="Classify sentiment",
            search_algorithm=SearchAlgorithm.EVOLUTIONARY,
            examples=validation_data,
        )
        result = ape.run()
        best_prompt = result.output["best_prompt"]
    """

    TECHNIQUE_ID = "automatic_prompt_engineering"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        model: Optional[Any] = None,
        task_description: str = "",
        search_algorithm: SearchAlgorithm = SearchAlgorithm.BEAM,
        num_candidates: int = 10,
        num_generations: int = 5,
        beam_width: int = 3,
        metric: Optional[Callable[[str, Any], float]] = None,
        examples: Optional[List[Example]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.task_description = task_description
        self.search_algorithm = search_algorithm
        self.num_candidates = num_candidates
        self.num_generations = num_generations
        self.beam_width = beam_width
        self.metric = metric or (lambda output, expected: float(output == expected))
        self.examples = examples or []

    def _generate_initial_prompts(self) -> List[str]:
        """Generate initial prompt candidates (placeholder)."""
        # Real implementation uses LLM to generate diverse prompts
        templates = [
            f"Task: {self.task_description}\nInput: {{input}}\nOutput:",
            f"Please {self.task_description.lower()}.\n\nInput: {{input}}\nAnswer:",
            f"You are an expert at {self.task_description.lower()}.\n\n{{input}}\n\nResponse:",
            f"Given the following input, {self.task_description.lower()}:\n\n{{input}}",
            f"Instructions: {self.task_description}\n\nText: {{input}}\n\nResult:",
        ]
        return templates[:self.num_candidates]

    def _generate_variations(
        self,
        prompt: str,
        num_variations: int,
    ) -> List[str]:
        """Generate variations of a prompt (placeholder)."""
        # Real implementation uses LLM to create variations
        variations = []
        modifiers = [
            "Think step by step. ",
            "Be concise. ",
            "Provide a detailed response. ",
            "Consider all aspects. ",
            "",
        ]
        for mod in modifiers[:num_variations]:
            variations.append(mod + prompt)
        return variations

    def _evaluate_prompt(
        self,
        prompt: str,
        examples: List[Example],
    ) -> float:
        """Evaluate a prompt on examples (placeholder)."""
        # Real implementation runs the prompt and evaluates
        scores = []
        for ex in examples[:10]:  # Limit for efficiency
            # Placeholder: random score with bias
            score = random.uniform(0.4, 1.0)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _beam_search(
        self,
        trace: List[Dict],
    ) -> List[PromptCandidate]:
        """Beam search for best prompt."""
        # Initialize with candidates
        initial_prompts = self._generate_initial_prompts()
        candidates = [
            PromptCandidate(
                prompt=p,
                score=self._evaluate_prompt(p, self.examples),
                generation=0,
            )
            for p in initial_prompts
        ]

        # Sort and keep top beam_width
        candidates.sort(key=lambda x: -x.score)
        beam = candidates[:self.beam_width]

        trace.append({
            "action": "initialize",
            "num_candidates": len(candidates),
            "best_score": beam[0].score if beam else 0,
        })

        # Iterate
        for gen in range(1, self.num_generations):
            new_candidates = []

            for parent in beam:
                variations = self._generate_variations(
                    parent.prompt,
                    self.num_candidates // self.beam_width,
                )
                for var in variations:
                    score = self._evaluate_prompt(var, self.examples)
                    new_candidates.append(PromptCandidate(
                        prompt=var,
                        score=score,
                        generation=gen,
                        parent=parent.prompt[:50],
                    ))

            # Combine with previous beam and select top
            all_candidates = beam + new_candidates
            all_candidates.sort(key=lambda x: -x.score)
            beam = all_candidates[:self.beam_width]

            trace.append({
                "action": "generation",
                "generation": gen,
                "best_score": beam[0].score if beam else 0,
            })

        return beam

    def _evolutionary_search(
        self,
        trace: List[Dict],
    ) -> List[PromptCandidate]:
        """Evolutionary search for best prompt (placeholder)."""
        # Simplified evolutionary approach
        return self._beam_search(trace)

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        try:
            # Run search
            if self.search_algorithm == SearchAlgorithm.BEAM:
                candidates = self._beam_search(trace)
            elif self.search_algorithm == SearchAlgorithm.EVOLUTIONARY:
                candidates = self._evolutionary_search(trace)
            else:
                candidates = self._beam_search(trace)

            best = candidates[0] if candidates else None

            return TechniqueResult(
                success=best is not None,
                output={
                    "best_prompt": best.prompt if best else None,
                    "best_score": best.score if best else 0,
                    "top_candidates": [
                        {"prompt": c.prompt, "score": c.score}
                        for c in candidates[:5]
                    ],
                    "generations": self.num_generations,
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
# FEW-SHOT OPTIMIZER
# =============================================================================

class SelectionStrategy(Enum):
    """Strategies for few-shot example selection."""
    RANDOM = "random"
    SIMILARITY = "similarity"  # Most similar to query
    DIVERSITY = "diversity"    # Maximize coverage
    CURRICULUM = "curriculum"  # Easy to hard
    INFLUENCE = "influence"    # Most influential on performance


class FewShotOptimizer(TechniqueBase):
    """
    Optimize few-shot example selection for a task.

    Finds the best set of examples to include in prompts
    for optimal performance.

    Configuration:
        examples: Pool of available examples
        num_shots: Number of examples to select
        strategy: Selection strategy
        metric: Evaluation function

    Usage:
        optimizer = FewShotOptimizer(
            examples=example_pool,
            num_shots=5,
            strategy=SelectionStrategy.INFLUENCE,
        )
        result = optimizer.run(task="sentiment classification")
    """

    TECHNIQUE_ID = "few_shot_optimizer"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        model: Optional[Any] = None,
        examples: Optional[List[Example]] = None,
        num_shots: int = 5,
        strategy: SelectionStrategy = SelectionStrategy.DIVERSITY,
        metric: Optional[Callable] = None,
        num_trials: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.examples = examples or []
        self.num_shots = num_shots
        self.strategy = strategy
        self.metric = metric
        self.num_trials = num_trials

    def _select_random(self, k: int) -> List[Example]:
        """Random selection."""
        if len(self.examples) <= k:
            return self.examples
        return random.sample(self.examples, k)

    def _select_diverse(self, k: int) -> List[Example]:
        """Select diverse examples (placeholder)."""
        # Real implementation uses clustering or MMR
        return self._select_random(k)

    def _evaluate_selection(
        self,
        selected: List[Example],
        eval_examples: List[Example],
    ) -> float:
        """Evaluate a selection of examples (placeholder)."""
        # Real implementation runs with these few-shot examples
        return random.uniform(0.5, 1.0)

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        best_selection: List[Example] = []
        best_score = 0.0

        # Try multiple selections
        for trial in range(self.num_trials):
            if self.strategy == SelectionStrategy.RANDOM:
                selection = self._select_random(self.num_shots)
            elif self.strategy == SelectionStrategy.DIVERSITY:
                selection = self._select_diverse(self.num_shots)
            else:
                selection = self._select_random(self.num_shots)

            score = self._evaluate_selection(selection, self.examples)

            if score > best_score:
                best_score = score
                best_selection = selection

            trace.append({
                "action": "trial",
                "trial": trial,
                "score": score,
            })

        return TechniqueResult(
            success=True,
            output={
                "selected_examples": [
                    {"inputs": ex.inputs, "outputs": ex.outputs}
                    for ex in best_selection
                ],
                "score": best_score,
                "num_trials": self.num_trials,
                "strategy": self.strategy.value,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: str  # float, int, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False


class TuningMethod(Enum):
    """Methods for hyperparameter tuning."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


class HyperparameterTuning(TechniqueBase):
    """
    Hyperparameter tuning for LM inference.

    Tunes parameters like:
        - temperature
        - top_p / top_k
        - frequency_penalty
        - presence_penalty
        - max_tokens

    Configuration:
        search_space: Hyperparameters to tune
        method: grid, random, or bayesian
        num_trials: Number of configurations to try
        metric: Evaluation function

    Usage:
        tuner = HyperparameterTuning(
            search_space=[
                HyperparameterSpace("temperature", "float", 0.0, 1.5),
                HyperparameterSpace("top_p", "float", 0.5, 1.0),
            ],
            method=TuningMethod.RANDOM,
            num_trials=50,
        )
        result = tuner.run(eval_fn=my_eval_function)
    """

    TECHNIQUE_ID = "hyperparameter_tuning"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        model: Optional[Any] = None,
        search_space: Optional[List[HyperparameterSpace]] = None,
        method: TuningMethod = TuningMethod.RANDOM,
        num_trials: int = 50,
        metric: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.search_space = search_space or [
            HyperparameterSpace("temperature", "float", 0.0, 1.5),
            HyperparameterSpace("top_p", "float", 0.5, 1.0),
        ]
        self.method = method
        self.num_trials = num_trials
        self.metric = metric

    def _sample_config(self) -> Dict[str, Any]:
        """Sample a hyperparameter configuration."""
        config = {}
        for hp in self.search_space:
            if hp.param_type == "float":
                if hp.log_scale:
                    import math
                    log_low = math.log(hp.low or 0.001)
                    log_high = math.log(hp.high or 1.0)
                    config[hp.name] = math.exp(random.uniform(log_low, log_high))
                else:
                    config[hp.name] = random.uniform(hp.low or 0, hp.high or 1)
            elif hp.param_type == "int":
                config[hp.name] = random.randint(int(hp.low or 0), int(hp.high or 10))
            elif hp.param_type == "categorical":
                config[hp.name] = random.choice(hp.choices or [])
        return config

    def _evaluate_config(
        self,
        config: Dict[str, Any],
    ) -> float:
        """Evaluate a hyperparameter configuration (placeholder)."""
        # Real implementation runs with these hyperparameters
        return random.uniform(0.5, 1.0)

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        best_config: Dict[str, Any] = {}
        best_score = 0.0
        all_results = []

        for trial in range(self.num_trials):
            config = self._sample_config()
            score = self._evaluate_config(config)

            all_results.append({"config": config, "score": score})

            if score > best_score:
                best_score = score
                best_config = config

            trace.append({
                "action": "trial",
                "trial": trial,
                "config": config,
                "score": score,
            })

        return TechniqueResult(
            success=True,
            output={
                "best_config": best_config,
                "best_score": best_score,
                "all_results": sorted(all_results, key=lambda x: -x["score"])[:10],
                "num_trials": self.num_trials,
                "method": self.method.value,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# OPRO (OPTIMIZATION BY PROMPTING)
# =============================================================================

@dataclass
class OPROCandidate:
    """
    A candidate prompt in the OPRO optimization process.

    Attributes:
        prompt: The candidate prompt text
        score: Performance score on evaluation
        iteration: Which iteration this was generated in
        reasoning: LLM's reasoning for this prompt (if available)
    """
    prompt: str
    score: float
    iteration: int = 0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OPRO(TechniqueBase):
    """
    Optimization by PROmpting (OPRO).

    Paper: "Large Language Models as Optimizers"
           (Yang et al., 2023)
    https://arxiv.org/abs/2309.03409

    Uses an LLM to iteratively optimize prompts by describing the
    optimization task in a meta-prompt and having the LLM propose
    improved prompts based on previous scores.

    Key Concepts:
        - Meta-prompt: Describes the optimization task to the LLM
        - Solution-score pairs: History of previous attempts and their scores
        - Iterative refinement: LLM generates new prompts based on history

    Configuration:
        meta_prompt: Template describing optimization task
        num_iterations: Number of optimization iterations
        scoring_function: Function to evaluate prompt quality
        population_size: Number of candidates per iteration

    Usage:
        opro = OPRO(
            backend=my_backend,
            meta_prompt="Generate a prompt for sentiment classification...",
            scoring_function=eval_sentiment,
            num_iterations=10,
            population_size=5,
        )
        result = opro.run(task_description="Classify movie reviews")
    """

    TECHNIQUE_ID = "opro"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        backend: Optional[Any] = None,
        meta_prompt: Optional[str] = None,
        num_iterations: int = 10,
        scoring_function: Optional[Callable[[str, Any], float]] = None,
        population_size: int = 5,
        examples: Optional[List[Example]] = None,
        keep_top_k: int = 20,
        **kwargs,
    ):
        """
        Initialize OPRO optimizer.

        Args:
            backend: LLM backend for generating prompt candidates
            meta_prompt: Template for the optimization meta-prompt
            num_iterations: Number of optimization iterations
            scoring_function: Function(prompt, examples) -> score
            population_size: Candidates to generate per iteration
            examples: Evaluation examples
            keep_top_k: Number of top solutions to keep in history
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.meta_prompt = meta_prompt or self._default_meta_prompt()
        self.num_iterations = num_iterations
        self.scoring_function = scoring_function or (lambda p, e: random.uniform(0.3, 1.0))
        self.population_size = population_size
        self.examples = examples or []
        self.keep_top_k = keep_top_k

    def _default_meta_prompt(self) -> str:
        """Default meta-prompt template for OPRO."""
        return """You are an expert prompt engineer. Your task is to generate better prompts.

TASK DESCRIPTION:
{task_description}

PREVIOUS PROMPTS AND THEIR SCORES (higher is better):
{solution_history}

Based on the patterns in successful prompts, generate {num_candidates} new prompt candidates
that might score even higher. Be creative but learn from what worked.

Generate each prompt on a new line, prefixed with "PROMPT:"
"""

    def _format_solution_history(
        self,
        history: List[OPROCandidate],
    ) -> str:
        """Format solution history for meta-prompt."""
        if not history:
            return "No previous attempts yet."

        # Sort by score and take top examples
        sorted_history = sorted(history, key=lambda x: -x.score)[:self.keep_top_k]

        lines = []
        for i, candidate in enumerate(sorted_history, 1):
            lines.append(f"{i}. Score: {candidate.score:.3f}")
            lines.append(f"   Prompt: {candidate.prompt[:200]}...")
            lines.append("")

        return "\n".join(lines)

    def _generate_candidates(
        self,
        task_description: str,
        history: List[OPROCandidate],
        iteration: int,
    ) -> List[OPROCandidate]:
        """Generate new candidate prompts using the LLM."""
        # Format the meta-prompt
        formatted_meta = self.meta_prompt.format(
            task_description=task_description,
            solution_history=self._format_solution_history(history),
            num_candidates=self.population_size,
        )

        # Placeholder: In real implementation, call backend
        # response = self.backend.generate(formatted_meta)
        # Parse response to extract prompts

        # Simulated generation for demonstration
        candidates = []
        base_variations = [
            f"Please {task_description.lower()}. Think step by step.",
            f"You are an expert. {task_description}. Be precise.",
            f"Task: {task_description}\nAnalyze carefully and respond.",
            f"Consider the following task: {task_description}\nProvide a detailed answer.",
            f"{task_description}\n\nLet me think through this systematically.",
        ]

        for i in range(self.population_size):
            prompt = random.choice(base_variations)
            # Add iteration-based variations
            if iteration > 3:
                prompt = "Important: " + prompt
            if iteration > 6:
                prompt += "\n\nDouble-check your answer."

            candidates.append(OPROCandidate(
                prompt=prompt,
                score=0.0,  # Will be evaluated
                iteration=iteration,
            ))

        return candidates

    def _evaluate_candidate(
        self,
        candidate: OPROCandidate,
    ) -> float:
        """Evaluate a candidate prompt."""
        return self.scoring_function(candidate.prompt, self.examples)

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        """
        Run OPRO optimization.

        Args:
            input_data: Task description or initial prompt
            context: Additional context (e.g., examples)

        Returns:
            TechniqueResult with optimized prompt and history
        """
        start = time.time()
        trace: List[Dict] = []

        task_description = input_data or "Complete the given task"
        history: List[OPROCandidate] = []
        best_candidate: Optional[OPROCandidate] = None

        try:
            for iteration in range(self.num_iterations):
                # Generate new candidates
                candidates = self._generate_candidates(
                    task_description, history, iteration
                )

                # Evaluate each candidate
                for candidate in candidates:
                    candidate.score = self._evaluate_candidate(candidate)
                    history.append(candidate)

                    if best_candidate is None or candidate.score > best_candidate.score:
                        best_candidate = candidate

                trace.append({
                    "action": "iteration",
                    "iteration": iteration,
                    "num_candidates": len(candidates),
                    "best_score_this_iter": max(c.score for c in candidates),
                    "overall_best_score": best_candidate.score if best_candidate else 0,
                })

            return TechniqueResult(
                success=best_candidate is not None,
                output={
                    "best_prompt": best_candidate.prompt if best_candidate else None,
                    "best_score": best_candidate.score if best_candidate else 0,
                    "optimization_history": [
                        {"prompt": c.prompt[:100], "score": c.score, "iteration": c.iteration}
                        for c in sorted(history, key=lambda x: -x.score)[:10]
                    ],
                    "total_candidates_evaluated": len(history),
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
# META-PROMPTING
# =============================================================================

class ExpertType(Enum):
    """Types of expert LLMs that can be orchestrated."""
    REASONING = "reasoning"
    CODE = "code"
    MATH = "math"
    WRITING = "writing"
    ANALYSIS = "analysis"
    FACTUAL = "factual"
    CREATIVE = "creative"


class AggregationMethod(Enum):
    """Methods for aggregating expert responses."""
    VOTE = "vote"               # Majority voting
    BEST = "best"               # Select highest confidence
    MERGE = "merge"             # Merge responses
    SEQUENTIAL = "sequential"   # Chain expert outputs


@dataclass
class ExpertResponse:
    """Response from an expert LLM."""
    expert_type: ExpertType
    response: str
    confidence: float = 0.0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetaPrompting(TechniqueBase):
    """
    Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding.

    Paper: "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding"
           (Suzgun & Kalai, 2024)
    https://arxiv.org/abs/2401.12954

    A scaffolding LLM (Meta Model) dynamically generates prompts to orchestrate
    multiple expert LLMs, each specialized for different aspects of the task.

    Key Concepts:
        - Scaffolding LLM: Orchestrates the overall problem-solving process
        - Expert LLMs: Specialized models for subtasks (reasoning, coding, etc.)
        - Dynamic prompt generation: Scaffolding creates prompts for experts
        - Response aggregation: Combines expert outputs into final answer

    Configuration:
        scaffolding_prompt: Template for the meta/scaffolding model
        expert_types: List of expert types to use
        aggregation_method: How to combine expert responses

    Usage:
        meta = MetaPrompting(
            backend=my_backend,
            expert_types=[ExpertType.REASONING, ExpertType.CODE],
            aggregation_method=AggregationMethod.MERGE,
        )
        result = meta.run(input_data="Solve this complex problem...")
    """

    TECHNIQUE_ID = "meta_prompting"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        backend: Optional[Any] = None,
        scaffolding_prompt: Optional[str] = None,
        expert_types: Optional[List[ExpertType]] = None,
        aggregation_method: AggregationMethod = AggregationMethod.MERGE,
        max_expert_calls: int = 5,
        **kwargs,
    ):
        """
        Initialize Meta-Prompting.

        Args:
            backend: LLM backend for both scaffolding and experts
            scaffolding_prompt: Template for the meta model
            expert_types: List of expert types to potentially use
            aggregation_method: Method to combine expert outputs
            max_expert_calls: Maximum expert invocations per run
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.scaffolding_prompt = scaffolding_prompt or self._default_scaffolding_prompt()
        self.expert_types = expert_types or [
            ExpertType.REASONING,
            ExpertType.ANALYSIS,
        ]
        self.aggregation_method = aggregation_method
        self.max_expert_calls = max_expert_calls

    def _default_scaffolding_prompt(self) -> str:
        """Default scaffolding prompt template."""
        return """You are a Meta Model that orchestrates expert models to solve complex tasks.

AVAILABLE EXPERTS:
{available_experts}

TASK:
{task}

Analyze the task and determine which expert(s) to consult. For each expert you want to use,
provide:
1. Expert type
2. The specific sub-question or subtask for that expert
3. How their response will contribute to the final answer

Format your plan as:
EXPERT: <type>
SUBTASK: <what to ask them>
PURPOSE: <how this helps>
---

After receiving expert responses, synthesize them into a final answer.
"""

    def _get_expert_prompt(self, expert_type: ExpertType) -> str:
        """Get the system prompt for an expert type."""
        expert_prompts = {
            ExpertType.REASONING: "You are an expert in logical reasoning and step-by-step problem solving.",
            ExpertType.CODE: "You are an expert programmer. Write clean, efficient code.",
            ExpertType.MATH: "You are a mathematics expert. Show your work clearly.",
            ExpertType.WRITING: "You are an expert writer. Craft clear, engaging prose.",
            ExpertType.ANALYSIS: "You are an expert analyst. Provide thorough, balanced analysis.",
            ExpertType.FACTUAL: "You are a factual expert. Provide accurate, verified information.",
            ExpertType.CREATIVE: "You are a creative expert. Generate innovative ideas.",
        }
        return expert_prompts.get(expert_type, "You are a helpful expert.")

    def _call_expert(
        self,
        expert_type: ExpertType,
        subtask: str,
    ) -> ExpertResponse:
        """Call an expert LLM with a subtask."""
        # Placeholder: In real implementation, call backend with expert prompt
        # expert_prompt = self._get_expert_prompt(expert_type)
        # response = self.backend.generate(expert_prompt + "\n\n" + subtask)

        # Simulated response
        responses = {
            ExpertType.REASONING: "Let me analyze this step by step...",
            ExpertType.CODE: "Here's a solution in Python...",
            ExpertType.MATH: "Applying the relevant formulas...",
            ExpertType.ANALYSIS: "Considering multiple perspectives...",
        }

        return ExpertResponse(
            expert_type=expert_type,
            response=responses.get(expert_type, "I'll help with this task..."),
            confidence=random.uniform(0.7, 0.95),
        )

    def _plan_experts(
        self,
        task: str,
    ) -> List[Tuple[ExpertType, str]]:
        """Use scaffolding model to plan which experts to consult."""
        # Placeholder: In real implementation, call backend with scaffolding prompt
        # Plan which experts to use based on task analysis

        # Simple heuristic planning
        plans = []
        for expert in self.expert_types[:self.max_expert_calls]:
            subtask = f"Help with: {task[:100]}"
            plans.append((expert, subtask))

        return plans

    def _aggregate_responses(
        self,
        responses: List[ExpertResponse],
        task: str,
    ) -> str:
        """Aggregate expert responses into final answer."""
        if not responses:
            return "No expert responses available."

        if self.aggregation_method == AggregationMethod.BEST:
            best = max(responses, key=lambda x: x.confidence)
            return best.response

        elif self.aggregation_method == AggregationMethod.VOTE:
            # For voting, would need comparable/classifiable responses
            return responses[0].response

        elif self.aggregation_method == AggregationMethod.MERGE:
            # Merge all responses
            merged = f"Synthesized answer for: {task[:50]}...\n\n"
            for resp in responses:
                merged += f"From {resp.expert_type.value} expert:\n{resp.response}\n\n"
            return merged

        elif self.aggregation_method == AggregationMethod.SEQUENTIAL:
            # Chain responses
            return " -> ".join(r.response for r in responses)

        return responses[0].response

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        """
        Run Meta-Prompting.

        Args:
            input_data: The task/question to solve
            context: Additional context

        Returns:
            TechniqueResult with aggregated expert response
        """
        start = time.time()
        trace: List[Dict] = []

        task = str(input_data) if input_data else "Complete the given task"

        try:
            # Step 1: Plan which experts to consult
            expert_plans = self._plan_experts(task)
            trace.append({
                "action": "planning",
                "experts_planned": [e.value for e, _ in expert_plans],
            })

            # Step 2: Call each expert
            responses: List[ExpertResponse] = []
            for expert_type, subtask in expert_plans:
                response = self._call_expert(expert_type, subtask)
                responses.append(response)
                trace.append({
                    "action": "expert_call",
                    "expert": expert_type.value,
                    "subtask": subtask[:100],
                    "confidence": response.confidence,
                })

            # Step 3: Aggregate responses
            final_answer = self._aggregate_responses(responses, task)
            trace.append({
                "action": "aggregation",
                "method": self.aggregation_method.value,
                "num_responses": len(responses),
            })

            return TechniqueResult(
                success=True,
                output={
                    "answer": final_answer,
                    "experts_consulted": [r.expert_type.value for r in responses],
                    "expert_responses": [
                        {
                            "expert": r.expert_type.value,
                            "response": r.response[:200],
                            "confidence": r.confidence,
                        }
                        for r in responses
                    ],
                    "aggregation_method": self.aggregation_method.value,
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
# ACTIVE PROMPTING
# =============================================================================

class UncertaintyMetric(Enum):
    """Metrics for measuring example uncertainty."""
    ENTROPY = "entropy"                    # Prediction entropy
    DISAGREEMENT = "disagreement"          # Model disagreement
    MARGIN = "margin"                      # Margin between top predictions
    LEAST_CONFIDENCE = "least_confidence"  # 1 - max probability
    VARIATION_RATIO = "variation_ratio"    # Frequency of non-modal prediction


class AnnotationStrategy(Enum):
    """Strategies for selecting examples to annotate."""
    MOST_UNCERTAIN = "most_uncertain"      # Highest uncertainty first
    DIVERSE_UNCERTAIN = "diverse_uncertain"  # Diverse and uncertain
    REPRESENTATIVE = "representative"       # Representative of clusters


@dataclass
class UncertainExample:
    """An example with computed uncertainty."""
    example: Example
    uncertainty_score: float
    predictions: List[str] = field(default_factory=list)
    is_annotated: bool = False
    annotation: Optional[str] = None


class ActivePrompting(TechniqueBase):
    """
    Active Prompting with Chain-of-Thought.

    Paper: "Active Prompting with Chain-of-Thought for Large Language Models"
           (Diao et al., 2023)
    https://arxiv.org/abs/2302.12246

    Dynamically selects the most informative few-shot examples by:
    1. Computing uncertainty for unlabeled examples
    2. Selecting most uncertain examples for human annotation
    3. Using annotated examples as few-shot demonstrations

    Key Concepts:
        - Uncertainty estimation: Identify examples where model is unsure
        - Active selection: Prioritize uncertain examples for annotation
        - Efficient annotation: Maximize information from limited labels

    Configuration:
        uncertainty_metric: How to measure example uncertainty
        annotation_budget: Maximum examples to annotate
        selection_strategy: How to select examples from uncertain pool

    Usage:
        active = ActivePrompting(
            backend=my_backend,
            uncertainty_metric=UncertaintyMetric.ENTROPY,
            annotation_budget=10,
        )
        result = active.run(unlabeled_examples=my_data)
    """

    TECHNIQUE_ID = "active_prompting"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        backend: Optional[Any] = None,
        uncertainty_metric: UncertaintyMetric = UncertaintyMetric.ENTROPY,
        annotation_budget: int = 10,
        selection_strategy: AnnotationStrategy = AnnotationStrategy.MOST_UNCERTAIN,
        num_inference_samples: int = 5,
        initial_prompt: str = "",
        **kwargs,
    ):
        """
        Initialize Active Prompting.

        Args:
            backend: LLM backend for inference
            uncertainty_metric: Metric for measuring uncertainty
            annotation_budget: Max examples to select for annotation
            selection_strategy: Strategy for selecting uncertain examples
            num_inference_samples: Number of samples for uncertainty estimation
            initial_prompt: Base prompt for inference
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.uncertainty_metric = uncertainty_metric
        self.annotation_budget = annotation_budget
        self.selection_strategy = selection_strategy
        self.num_inference_samples = num_inference_samples
        self.initial_prompt = initial_prompt

    def _compute_uncertainty(
        self,
        example: Example,
        predictions: List[str],
    ) -> float:
        """Compute uncertainty for an example based on predictions."""
        if not predictions:
            return 1.0  # Maximum uncertainty if no predictions

        if self.uncertainty_metric == UncertaintyMetric.ENTROPY:
            # Compute entropy from prediction distribution
            from collections import Counter
            counts = Counter(predictions)
            total = len(predictions)
            entropy = 0.0
            for count in counts.values():
                p = count / total
                if p > 0:
                    import math
                    entropy -= p * math.log2(p)
            # Normalize by max entropy
            max_entropy = math.log2(len(predictions)) if len(predictions) > 1 else 1
            return entropy / max_entropy if max_entropy > 0 else 0.0

        elif self.uncertainty_metric == UncertaintyMetric.DISAGREEMENT:
            # Fraction of predictions that differ from mode
            from collections import Counter
            counts = Counter(predictions)
            mode_count = counts.most_common(1)[0][1]
            return 1.0 - (mode_count / len(predictions))

        elif self.uncertainty_metric == UncertaintyMetric.LEAST_CONFIDENCE:
            # 1 - probability of most common prediction
            from collections import Counter
            counts = Counter(predictions)
            max_prob = counts.most_common(1)[0][1] / len(predictions)
            return 1.0 - max_prob

        else:
            # Default: random uncertainty
            return random.uniform(0, 1)

    def _generate_predictions(
        self,
        example: Example,
    ) -> List[str]:
        """Generate multiple predictions for uncertainty estimation."""
        # Placeholder: In real implementation, call backend multiple times
        # with temperature > 0 to get diverse predictions

        # Simulated predictions with some variation
        base_answers = ["A", "B", "C", "D"]
        predictions = []
        for _ in range(self.num_inference_samples):
            if random.random() < 0.6:  # 60% consistent
                predictions.append(base_answers[0])
            else:
                predictions.append(random.choice(base_answers))

        return predictions

    def _select_for_annotation(
        self,
        uncertain_examples: List[UncertainExample],
    ) -> List[UncertainExample]:
        """Select examples for annotation based on strategy."""
        if self.selection_strategy == AnnotationStrategy.MOST_UNCERTAIN:
            # Simply take most uncertain
            sorted_examples = sorted(
                uncertain_examples,
                key=lambda x: -x.uncertainty_score
            )
            return sorted_examples[:self.annotation_budget]

        elif self.selection_strategy == AnnotationStrategy.DIVERSE_UNCERTAIN:
            # Balance uncertainty with diversity (placeholder)
            sorted_examples = sorted(
                uncertain_examples,
                key=lambda x: -x.uncertainty_score
            )
            # Take top 2x budget, then sample for diversity
            top_uncertain = sorted_examples[:self.annotation_budget * 2]
            if len(top_uncertain) <= self.annotation_budget:
                return top_uncertain
            return random.sample(top_uncertain, self.annotation_budget)

        else:
            # Default: most uncertain
            return sorted(
                uncertain_examples,
                key=lambda x: -x.uncertainty_score
            )[:self.annotation_budget]

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        """
        Run Active Prompting.

        Args:
            input_data: List of unlabeled examples or dict with 'unlabeled_examples'
            context: Additional context

        Returns:
            TechniqueResult with selected examples for annotation
        """
        start = time.time()
        trace: List[Dict] = []

        # Extract unlabeled examples
        if isinstance(input_data, list):
            unlabeled = input_data
        elif isinstance(input_data, dict):
            unlabeled = input_data.get("unlabeled_examples", [])
        else:
            unlabeled = []

        # Convert to Example objects if needed
        examples = []
        for item in unlabeled:
            if isinstance(item, Example):
                examples.append(item)
            elif isinstance(item, dict):
                examples.append(Example(
                    inputs=item.get("inputs", item),
                    outputs=item.get("outputs", {}),
                ))

        try:
            # Step 1: Generate predictions and compute uncertainty for each example
            uncertain_examples: List[UncertainExample] = []

            for example in examples:
                predictions = self._generate_predictions(example)
                uncertainty = self._compute_uncertainty(example, predictions)

                uncertain_examples.append(UncertainExample(
                    example=example,
                    uncertainty_score=uncertainty,
                    predictions=predictions,
                ))

            trace.append({
                "action": "compute_uncertainty",
                "num_examples": len(examples),
                "metric": self.uncertainty_metric.value,
                "avg_uncertainty": sum(e.uncertainty_score for e in uncertain_examples) / len(uncertain_examples) if uncertain_examples else 0,
            })

            # Step 2: Select examples for annotation
            selected = self._select_for_annotation(uncertain_examples)

            trace.append({
                "action": "select_for_annotation",
                "strategy": self.selection_strategy.value,
                "num_selected": len(selected),
                "budget": self.annotation_budget,
            })

            return TechniqueResult(
                success=True,
                output={
                    "selected_for_annotation": [
                        {
                            "inputs": s.example.inputs,
                            "uncertainty_score": s.uncertainty_score,
                            "predictions": s.predictions,
                        }
                        for s in selected
                    ],
                    "total_examples": len(examples),
                    "annotation_budget": self.annotation_budget,
                    "uncertainty_distribution": {
                        "min": min(e.uncertainty_score for e in uncertain_examples) if uncertain_examples else 0,
                        "max": max(e.uncertainty_score for e in uncertain_examples) if uncertain_examples else 0,
                        "mean": sum(e.uncertainty_score for e in uncertain_examples) / len(uncertain_examples) if uncertain_examples else 0,
                    },
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
# PROMPTBREEDER
# =============================================================================

@dataclass
class PromptIndividual:
    """
    An individual in the PromptBreeder population.

    Contains both a task prompt and a mutation prompt that describes
    how to mutate the task prompt.
    """
    task_prompt: str
    mutation_prompt: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBreederPopulation:
    """Population of prompt individuals."""
    individuals: List[PromptIndividual]
    generation: int = 0
    best_fitness_history: List[float] = field(default_factory=list)


class PromptBreeder(TechniqueBase):
    """
    PromptBreeder: Self-Referential Self-Improvement via Prompt Evolution.

    Paper: "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution"
           (Fernando et al., 2023)
    https://arxiv.org/abs/2309.16797

    Evolves both task prompts AND mutation prompts using an LLM.
    Key innovation: mutation prompts themselves evolve, enabling
    self-referential self-improvement.

    Key Concepts:
        - Task prompts: Prompts for the actual task
        - Mutation prompts: Prompts that describe how to mutate task prompts
        - Co-evolution: Both evolve together
        - Self-referential: Mutation prompts can mutate themselves

    Mutation Operators:
        - Direct mutation: LLM mutates task prompt
        - EDA mutation: Estimate distribution and sample
        - Hypermutation: Mutate the mutation prompt
        - Crossover: Combine prompts from parents

    Configuration:
        population_size: Number of individuals in population
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        fitness_function: Function to evaluate prompt quality

    Usage:
        pb = PromptBreeder(
            backend=my_backend,
            population_size=20,
            fitness_function=eval_task,
        )
        result = pb.run(task_description="Solve math problems")
    """

    TECHNIQUE_ID = "promptbreeder"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        backend: Optional[Any] = None,
        population_size: int = 20,
        mutation_rate: float = 0.8,
        crossover_rate: float = 0.2,
        fitness_function: Optional[Callable[[str, Any], float]] = None,
        num_generations: int = 10,
        tournament_size: int = 3,
        examples: Optional[List[Example]] = None,
        **kwargs,
    ):
        """
        Initialize PromptBreeder.

        Args:
            backend: LLM backend for mutations and evaluation
            population_size: Size of the prompt population
            mutation_rate: Probability of applying mutation
            crossover_rate: Probability of applying crossover
            fitness_function: Function(prompt, examples) -> fitness
            num_generations: Number of generations to evolve
            tournament_size: Size of tournament for selection
            examples: Evaluation examples
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_function = fitness_function or (lambda p, e: random.uniform(0.3, 1.0))
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.examples = examples or []

    def _initialize_population(
        self,
        task_description: str,
    ) -> PromptBreederPopulation:
        """Initialize the population with diverse prompts."""
        individuals = []

        # Diverse initial task prompts
        task_templates = [
            f"Task: {task_description}\nInput: {{input}}\nOutput:",
            f"You are an expert. {task_description}\n\n{{input}}",
            f"Please {task_description.lower()} for the following:\n{{input}}",
            f"Consider this task: {task_description}\nAnalyze: {{input}}",
            f"Instructions: {task_description}\nProblem: {{input}}\nSolution:",
        ]

        # Diverse initial mutation prompts
        mutation_templates = [
            "Improve this prompt to be clearer and more effective.",
            "Make this prompt more specific and detailed.",
            "Simplify this prompt while maintaining its intent.",
            "Add step-by-step reasoning instructions to this prompt.",
            "Make this prompt more engaging and precise.",
        ]

        for i in range(self.population_size):
            task_prompt = random.choice(task_templates)
            mutation_prompt = random.choice(mutation_templates)

            individuals.append(PromptIndividual(
                task_prompt=task_prompt,
                mutation_prompt=mutation_prompt,
                generation=0,
            ))

        return PromptBreederPopulation(individuals=individuals, generation=0)

    def _evaluate_individual(
        self,
        individual: PromptIndividual,
    ) -> float:
        """Evaluate fitness of an individual."""
        return self.fitness_function(individual.task_prompt, self.examples)

    def _tournament_select(
        self,
        population: PromptBreederPopulation,
    ) -> PromptIndividual:
        """Select individual using tournament selection."""
        tournament = random.sample(
            population.individuals,
            min(self.tournament_size, len(population.individuals))
        )
        return max(tournament, key=lambda x: x.fitness)

    def _mutate_prompt(
        self,
        task_prompt: str,
        mutation_prompt: str,
    ) -> str:
        """Mutate a task prompt using the mutation prompt."""
        # Placeholder: In real implementation, use LLM
        # prompt = f"{mutation_prompt}\n\nOriginal prompt:\n{task_prompt}\n\nMutated prompt:"
        # mutated = self.backend.generate(prompt)

        # Simulated mutation
        mutations = [
            task_prompt + " Think step by step.",
            "Important: " + task_prompt,
            task_prompt.replace("Output:", "Answer:"),
            task_prompt + " Be precise.",
            "Carefully " + task_prompt.lower() if not task_prompt.startswith("Carefully") else task_prompt,
        ]
        return random.choice(mutations)

    def _hypermutate(
        self,
        mutation_prompt: str,
    ) -> str:
        """Mutate the mutation prompt itself (self-referential)."""
        # Placeholder: In real implementation, use LLM
        # prompt = f"Improve this instruction for modifying prompts:\n{mutation_prompt}"
        # new_mutation = self.backend.generate(prompt)

        # Simulated hypermutation
        hypermutations = [
            mutation_prompt + " Focus on clarity.",
            "Thoroughly " + mutation_prompt.lower(),
            mutation_prompt.replace("prompt", "instruction"),
            mutation_prompt + " Add examples if helpful.",
            mutation_prompt,  # Sometimes keep original
        ]
        return random.choice(hypermutations)

    def _crossover(
        self,
        parent1: PromptIndividual,
        parent2: PromptIndividual,
        generation: int,
    ) -> PromptIndividual:
        """Combine two parents to create offspring."""
        # Simple crossover: take parts from each parent
        if random.random() < 0.5:
            task_prompt = parent1.task_prompt
            mutation_prompt = parent2.mutation_prompt
        else:
            task_prompt = parent2.task_prompt
            mutation_prompt = parent1.mutation_prompt

        return PromptIndividual(
            task_prompt=task_prompt,
            mutation_prompt=mutation_prompt,
            generation=generation,
            parent_ids=[id(parent1), id(parent2)],
        )

    def _evolve_generation(
        self,
        population: PromptBreederPopulation,
    ) -> PromptBreederPopulation:
        """Evolve population for one generation."""
        new_individuals = []
        next_gen = population.generation + 1

        # Elitism: keep best individual
        best = max(population.individuals, key=lambda x: x.fitness)
        new_individuals.append(PromptIndividual(
            task_prompt=best.task_prompt,
            mutation_prompt=best.mutation_prompt,
            fitness=best.fitness,
            generation=next_gen,
        ))

        # Create rest of population
        while len(new_individuals) < self.population_size:
            parent = self._tournament_select(population)

            if random.random() < self.crossover_rate:
                # Crossover
                parent2 = self._tournament_select(population)
                child = self._crossover(parent, parent2, next_gen)
            else:
                # Clone
                child = PromptIndividual(
                    task_prompt=parent.task_prompt,
                    mutation_prompt=parent.mutation_prompt,
                    generation=next_gen,
                )

            # Mutation
            if random.random() < self.mutation_rate:
                child.task_prompt = self._mutate_prompt(
                    child.task_prompt,
                    child.mutation_prompt
                )

                # Occasionally hypermutate
                if random.random() < 0.2:
                    child.mutation_prompt = self._hypermutate(child.mutation_prompt)

            new_individuals.append(child)

        # Evaluate all individuals
        for ind in new_individuals:
            if ind.fitness == 0:
                ind.fitness = self._evaluate_individual(ind)

        return PromptBreederPopulation(
            individuals=new_individuals,
            generation=next_gen,
            best_fitness_history=population.best_fitness_history + [best.fitness],
        )

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        """
        Run PromptBreeder evolution.

        Args:
            input_data: Task description
            context: Additional context

        Returns:
            TechniqueResult with evolved prompts
        """
        start = time.time()
        trace: List[Dict] = []

        task_description = str(input_data) if input_data else "Complete the task"

        try:
            # Initialize population
            population = self._initialize_population(task_description)

            # Evaluate initial population
            for ind in population.individuals:
                ind.fitness = self._evaluate_individual(ind)

            best_initial = max(population.individuals, key=lambda x: x.fitness)
            trace.append({
                "action": "initialize",
                "population_size": len(population.individuals),
                "best_fitness": best_initial.fitness,
            })

            # Evolve
            for gen in range(self.num_generations):
                population = self._evolve_generation(population)
                best = max(population.individuals, key=lambda x: x.fitness)

                trace.append({
                    "action": "generation",
                    "generation": gen + 1,
                    "best_fitness": best.fitness,
                    "avg_fitness": sum(i.fitness for i in population.individuals) / len(population.individuals),
                })

            # Get final best
            best_final = max(population.individuals, key=lambda x: x.fitness)

            return TechniqueResult(
                success=True,
                output={
                    "best_prompt": best_final.task_prompt,
                    "best_mutation_prompt": best_final.mutation_prompt,
                    "best_fitness": best_final.fitness,
                    "fitness_history": population.best_fitness_history,
                    "top_prompts": [
                        {
                            "task_prompt": ind.task_prompt,
                            "fitness": ind.fitness,
                        }
                        for ind in sorted(population.individuals, key=lambda x: -x.fitness)[:5]
                    ],
                    "generations_evolved": self.num_generations,
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
# EVOPROMPT
# =============================================================================

@dataclass
class EvoPromptCandidate:
    """A candidate in the EvoPrompt evolution."""
    prompt: str
    fitness: float = 0.0
    generation: int = 0
    evolution_history: List[str] = field(default_factory=list)


class EvoPrompt(TechniqueBase):
    """
    EvoPrompt: Connecting Large Language Models with Evolutionary Algorithms.

    Paper: "Connecting Large Language Models with Evolutionary Algorithms
           Yields Powerful Prompt Optimizers"
           (Guo et al., 2023)
    https://arxiv.org/abs/2309.08532

    Uses evolutionary algorithms (GA/DE) with LLM-based operators
    for discrete prompt optimization.

    Key Concepts:
        - LLM as evolutionary operator: Uses LLM for mutation/crossover
        - Discrete optimization: Works directly with text prompts
        - Evolution prompt: Meta-prompt that guides evolution
        - Population-based: Maintains diverse prompt population

    Evolution Strategies:
        - GA-style: Tournament selection, crossover, mutation
        - DE-style: Differential evolution adapted for text

    Configuration:
        num_generations: Number of generations to evolve
        selection_pressure: Tournament size / selection stringency
        evolution_prompt: Meta-prompt guiding the evolution

    Usage:
        evo = EvoPrompt(
            backend=my_backend,
            num_generations=15,
            selection_pressure=3,
        )
        result = evo.run(task="sentiment classification", examples=data)
    """

    TECHNIQUE_ID = "evoprompt"
    CATEGORY = TechniqueCategory.OPTIMIZATION

    def __init__(
        self,
        backend: Optional[Any] = None,
        num_generations: int = 15,
        selection_pressure: int = 3,
        evolution_prompt: Optional[str] = None,
        population_size: int = 10,
        fitness_function: Optional[Callable[[str, Any], float]] = None,
        examples: Optional[List[Example]] = None,
        mutation_rate: float = 0.5,
        **kwargs,
    ):
        """
        Initialize EvoPrompt.

        Args:
            backend: LLM backend for evolutionary operators
            num_generations: Number of generations to evolve
            selection_pressure: Tournament size for selection
            evolution_prompt: Meta-prompt for guiding evolution
            population_size: Number of prompts in population
            fitness_function: Function(prompt, examples) -> fitness
            examples: Evaluation examples
            mutation_rate: Rate of mutation vs crossover
        """
        super().__init__(**kwargs)
        self.backend = backend
        self.num_generations = num_generations
        self.selection_pressure = selection_pressure
        self.evolution_prompt = evolution_prompt or self._default_evolution_prompt()
        self.population_size = population_size
        self.fitness_function = fitness_function or (lambda p, e: random.uniform(0.3, 1.0))
        self.examples = examples or []
        self.mutation_rate = mutation_rate

    def _default_evolution_prompt(self) -> str:
        """Default evolution prompt for EvoPrompt."""
        return """You are evolving prompts for a task. Given parent prompt(s), create an improved offspring.

TASK: {task}

{parents_section}

Create a new prompt that combines the best aspects of the parent(s) while potentially
introducing beneficial variations. The new prompt should be clear, effective, and
appropriate for the task.

Output only the new prompt, nothing else.
"""

    def _initialize_population(
        self,
        task: str,
    ) -> List[EvoPromptCandidate]:
        """Initialize diverse population of prompts."""
        templates = [
            f"Perform {task}. Input: {{input}}",
            f"Task: {task}\nText: {{input}}\nOutput:",
            f"You are an expert at {task}. Analyze: {{input}}",
            f"Please {task.lower()} for: {{input}}",
            f"Given the input, {task.lower()}.\n\n{{input}}",
            f"Instructions: {task}\nInput: {{input}}\nResult:",
            f"Consider the following and {task.lower()}:\n{{input}}",
            f"{task}. Think step by step.\n\n{{input}}",
            f"Carefully {task.lower()} this: {{input}}",
            f"Your task is to {task.lower()}.\nInput: {{input}}\nOutput:",
        ]

        candidates = []
        for i, template in enumerate(templates[:self.population_size]):
            candidates.append(EvoPromptCandidate(
                prompt=template,
                generation=0,
                evolution_history=["initial"],
            ))

        return candidates

    def _evaluate_candidate(
        self,
        candidate: EvoPromptCandidate,
    ) -> float:
        """Evaluate fitness of a candidate."""
        return self.fitness_function(candidate.prompt, self.examples)

    def _tournament_select(
        self,
        population: List[EvoPromptCandidate],
    ) -> EvoPromptCandidate:
        """Tournament selection."""
        tournament = random.sample(
            population,
            min(self.selection_pressure, len(population))
        )
        return max(tournament, key=lambda x: x.fitness)

    def _llm_mutate(
        self,
        parent: EvoPromptCandidate,
        task: str,
        generation: int,
    ) -> EvoPromptCandidate:
        """Use LLM to mutate a prompt."""
        # Placeholder: In real implementation, call backend
        parents_section = f"PARENT PROMPT:\n{parent.prompt}"
        # evolution_call = self.evolution_prompt.format(task=task, parents_section=parents_section)
        # new_prompt = self.backend.generate(evolution_call)

        # Simulated mutation
        mutations = [
            parent.prompt + " Be precise.",
            "Important: " + parent.prompt,
            parent.prompt.replace(":", ":\n"),
            parent.prompt + " Consider all aspects.",
            parent.prompt,  # Sometimes keep same
        ]
        new_prompt = random.choice(mutations)

        return EvoPromptCandidate(
            prompt=new_prompt,
            generation=generation,
            evolution_history=parent.evolution_history + ["mutate"],
        )

    def _llm_crossover(
        self,
        parent1: EvoPromptCandidate,
        parent2: EvoPromptCandidate,
        task: str,
        generation: int,
    ) -> EvoPromptCandidate:
        """Use LLM to crossover two prompts."""
        # Placeholder: In real implementation, call backend
        parents_section = f"PARENT 1:\n{parent1.prompt}\n\nPARENT 2:\n{parent2.prompt}"
        # evolution_call = self.evolution_prompt.format(task=task, parents_section=parents_section)
        # new_prompt = self.backend.generate(evolution_call)

        # Simulated crossover - combine elements from both
        p1_parts = parent1.prompt.split()
        p2_parts = parent2.prompt.split()

        # Take beginning from p1, end from p2
        midpoint = len(p1_parts) // 2
        new_parts = p1_parts[:midpoint] + p2_parts[midpoint:]
        new_prompt = " ".join(new_parts)

        return EvoPromptCandidate(
            prompt=new_prompt,
            generation=generation,
            evolution_history=parent1.evolution_history + ["crossover"],
        )

    def _evolve_generation(
        self,
        population: List[EvoPromptCandidate],
        task: str,
        generation: int,
    ) -> List[EvoPromptCandidate]:
        """Evolve population for one generation."""
        new_population = []

        # Elitism: keep best
        best = max(population, key=lambda x: x.fitness)
        elite = EvoPromptCandidate(
            prompt=best.prompt,
            fitness=best.fitness,
            generation=generation,
            evolution_history=best.evolution_history + ["elite"],
        )
        new_population.append(elite)

        # Create rest through evolution
        while len(new_population) < self.population_size:
            if random.random() < self.mutation_rate:
                # Mutation
                parent = self._tournament_select(population)
                child = self._llm_mutate(parent, task, generation)
            else:
                # Crossover
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child = self._llm_crossover(parent1, parent2, task, generation)

            new_population.append(child)

        # Evaluate
        for candidate in new_population:
            if candidate.fitness == 0:
                candidate.fitness = self._evaluate_candidate(candidate)

        return new_population

    def run(self, input_data: Any = None, context: Optional[Dict] = None) -> TechniqueResult:
        """
        Run EvoPrompt evolution.

        Args:
            input_data: Task description or dict with 'task' key
            context: Additional context

        Returns:
            TechniqueResult with evolved prompts
        """
        start = time.time()
        trace: List[Dict] = []

        # Extract task
        if isinstance(input_data, dict):
            task = input_data.get("task", "Complete the task")
        else:
            task = str(input_data) if input_data else "Complete the task"

        try:
            # Initialize
            population = self._initialize_population(task)

            # Evaluate initial
            for candidate in population:
                candidate.fitness = self._evaluate_candidate(candidate)

            best_initial = max(population, key=lambda x: x.fitness)
            trace.append({
                "action": "initialize",
                "population_size": len(population),
                "best_fitness": best_initial.fitness,
            })

            fitness_history = [best_initial.fitness]

            # Evolve
            for gen in range(1, self.num_generations + 1):
                population = self._evolve_generation(population, task, gen)
                best = max(population, key=lambda x: x.fitness)
                fitness_history.append(best.fitness)

                trace.append({
                    "action": "generation",
                    "generation": gen,
                    "best_fitness": best.fitness,
                    "avg_fitness": sum(c.fitness for c in population) / len(population),
                })

            # Final best
            best_final = max(population, key=lambda x: x.fitness)

            return TechniqueResult(
                success=True,
                output={
                    "best_prompt": best_final.prompt,
                    "best_fitness": best_final.fitness,
                    "evolution_history": best_final.evolution_history,
                    "fitness_history": fitness_history,
                    "improvement": fitness_history[-1] - fitness_history[0],
                    "top_prompts": [
                        {
                            "prompt": c.prompt,
                            "fitness": c.fitness,
                        }
                        for c in sorted(population, key=lambda x: -x.fitness)[:5]
                    ],
                    "generations_evolved": self.num_generations,
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
    "OptimizerType",
    "SearchAlgorithm",
    "SelectionStrategy",
    "TuningMethod",
    "ExpertType",
    "AggregationMethod",
    "UncertaintyMetric",
    "AnnotationStrategy",
    # Data classes
    "Signature",
    "Example",
    "OptimizedModule",
    "PromptCandidate",
    "HyperparameterSpace",
    "OPROCandidate",
    "ExpertResponse",
    "UncertainExample",
    "PromptIndividual",
    "PromptBreederPopulation",
    "EvoPromptCandidate",
    # Techniques
    "DSPy",
    "AutomaticPromptEngineering",
    "FewShotOptimizer",
    "HyperparameterTuning",
    "OPRO",
    "MetaPrompting",
    "ActivePrompting",
    "PromptBreeder",
    "EvoPrompt",
]
