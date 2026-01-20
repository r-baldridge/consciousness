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
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "OptimizerType",
    "SearchAlgorithm",
    "SelectionStrategy",
    "TuningMethod",
    # Data classes
    "Signature",
    "Example",
    "OptimizedModule",
    "PromptCandidate",
    "HyperparameterSpace",
    # Techniques
    "DSPy",
    "AutomaticPromptEngineering",
    "FewShotOptimizer",
    "HyperparameterTuning",
]
