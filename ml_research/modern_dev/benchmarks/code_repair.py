"""
Code Repair Accuracy Benchmarks (INTEG-003)

This module provides comprehensive benchmarks for evaluating code repair model accuracy.
Includes dataset management, metrics computation, and baseline comparisons.

Key Components:
    - BenchmarkDataset: Container for benchmark samples
    - CodeRepairBenchmark: Main benchmark runner
    - BaselineComparison: Compare against naive approaches

Metrics:
    - Accuracy: Percentage of correctly repaired samples
    - Exact Match: Percentage of outputs matching target exactly
    - Token Error Rate: Average edit distance as percentage of tokens
    - Category Breakdown: Performance by bug type
    - Difficulty Analysis: Performance by sample difficulty

Usage:
    benchmark = CodeRepairBenchmark(pipeline)
    results = benchmark.run(max_samples=100)
    benchmark.export_results(results, "results.json")
"""

from __future__ import annotations

import json
import csv
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CodeRepairSample:
    """A single code repair sample for benchmarking.

    Attributes:
        id: Unique identifier for the sample.
        buggy_code: The code containing a bug.
        fixed_code: The correct/repaired code.
        bug_type: Category of the bug (e.g., "variable_typo", "off_by_one").
        difficulty: Difficulty score from 0.0 (easy) to 1.0 (hard).
        language: Programming language (default: "python").
        metadata: Additional sample metadata.
    """
    id: str
    buggy_code: str
    fixed_code: str
    bug_type: str
    difficulty: float = 0.5
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate sample data."""
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"Difficulty must be in [0, 1], got {self.difficulty}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeRepairSample":
        """Create sample from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkDataset:
    """Dataset for benchmarking code repair models.

    Attributes:
        name: Name of the dataset.
        samples: List of code repair samples.
        categories: Count of samples per bug type.
        description: Optional dataset description.
        version: Dataset version string.
    """
    name: str
    samples: List[CodeRepairSample]
    categories: Dict[str, int] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0.0"

    def __post_init__(self):
        """Compute category counts if not provided."""
        if not self.categories and self.samples:
            self.categories = {}
            for sample in self.samples:
                self.categories[sample.bug_type] = (
                    self.categories.get(sample.bug_type, 0) + 1
                )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> CodeRepairSample:
        """Get sample by index."""
        return self.samples[idx]

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)

    def filter_by_category(self, category: str) -> "BenchmarkDataset":
        """Return a new dataset with only samples of the given category."""
        filtered = [s for s in self.samples if s.bug_type == category]
        return BenchmarkDataset(
            name=f"{self.name}_{category}",
            samples=filtered,
            description=f"Filtered by category: {category}",
        )

    def filter_by_difficulty(
        self,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
    ) -> "BenchmarkDataset":
        """Return a new dataset with samples in the given difficulty range."""
        filtered = [
            s for s in self.samples
            if min_difficulty <= s.difficulty <= max_difficulty
        ]
        return BenchmarkDataset(
            name=f"{self.name}_diff_{min_difficulty:.1f}_{max_difficulty:.1f}",
            samples=filtered,
            description=f"Filtered by difficulty: [{min_difficulty}, {max_difficulty}]",
        )

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of samples by difficulty level."""
        distribution = {
            "easy": 0,      # 0.0 - 0.33
            "medium": 0,    # 0.33 - 0.66
            "hard": 0,      # 0.66 - 1.0
        }
        for sample in self.samples:
            if sample.difficulty < 0.33:
                distribution["easy"] += 1
            elif sample.difficulty < 0.66:
                distribution["medium"] += 1
            else:
                distribution["hard"] += 1
        return distribution

    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "samples": [s.to_dict() for s in self.samples],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BenchmarkDataset":
        """Load dataset from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        samples = [CodeRepairSample.from_dict(s) for s in data["samples"]]
        return cls(
            name=data["name"],
            samples=samples,
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
        )


@dataclass
class SampleResult:
    """Result of evaluating a single sample.

    Attributes:
        sample_id: ID of the evaluated sample.
        prediction: Model's predicted repair.
        target: Ground truth repair.
        is_correct: Whether prediction matches target.
        is_exact_match: Whether prediction exactly matches target.
        token_error_rate: Edit distance as percentage of tokens.
        latency_ms: Inference time in milliseconds.
        confidence: Model confidence score (if available).
    """
    sample_id: str
    prediction: str
    target: str
    is_correct: bool
    is_exact_match: bool
    token_error_rate: float
    latency_ms: float
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class CategoryResult:
    """Results aggregated by category.

    Attributes:
        category: Bug type category.
        num_samples: Number of samples in category.
        accuracy: Accuracy for this category.
        exact_match: Exact match rate.
        avg_token_error_rate: Average token error rate.
        avg_latency_ms: Average latency in milliseconds.
    """
    category: str
    num_samples: int
    accuracy: float
    exact_match: float
    avg_token_error_rate: float
    avg_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run.

    Attributes:
        accuracy: Overall accuracy (correct / total).
        exact_match: Percentage of exact matches.
        token_error_rate: Average token error rate.
        by_category: Results broken down by bug type.
        by_difficulty: Accuracy by difficulty level.
        total_samples: Total number of samples evaluated.
        successful_repairs: Number of successful repairs.
        total_time_ms: Total benchmark time.
        avg_latency_ms: Average per-sample latency.
        sample_results: Individual sample results.
        timestamp: Benchmark timestamp.
        metadata: Additional benchmark metadata.
    """
    accuracy: float
    exact_match: float
    token_error_rate: float
    by_category: Dict[str, CategoryResult]
    by_difficulty: Dict[str, float]
    total_samples: int
    successful_repairs: int
    total_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    sample_results: List[SampleResult] = field(default_factory=list)
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "exact_match": self.exact_match,
            "token_error_rate": self.token_error_rate,
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "by_difficulty": self.by_difficulty,
            "total_samples": self.total_samples,
            "successful_repairs": self.successful_repairs,
            "total_time_ms": self.total_time_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "sample_results": [r.to_dict() for r in self.sample_results],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=" * 60,
            "Code Repair Benchmark Results",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Total Samples: {self.total_samples}",
            f"Successful Repairs: {self.successful_repairs}",
            "",
            "Overall Metrics:",
            f"  Accuracy: {self.accuracy:.2%}",
            f"  Exact Match: {self.exact_match:.2%}",
            f"  Token Error Rate: {self.token_error_rate:.2%}",
            f"  Avg Latency: {self.avg_latency_ms:.2f} ms",
            "",
            "By Difficulty:",
        ]
        for level, acc in self.by_difficulty.items():
            lines.append(f"  {level.capitalize()}: {acc:.2%}")
        lines.append("")
        lines.append("By Category:")
        for cat, result in self.by_category.items():
            lines.append(f"  {cat}: {result.accuracy:.2%} ({result.num_samples} samples)")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Pipeline Protocol
# =============================================================================


class CodeRepairPipeline(Protocol):
    """Protocol for code repair pipelines.

    Any pipeline must implement the repair method.
    """

    def repair(
        self,
        buggy_code: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Repair buggy code.

        Args:
            buggy_code: The code containing a bug.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing at least:
                - "repaired_code": The repaired code string.
                - "confidence" (optional): Model confidence score.
        """
        ...


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def tokenize_code(code: str) -> List[str]:
    """Tokenize code into tokens for comparison.

    Uses a simple regex-based tokenizer that handles:
    - Identifiers
    - Numbers
    - Operators
    - Punctuation
    - Whitespace (preserved for structure)
    """
    # Pattern for tokenization
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+(?:\.[0-9]+)?|[+\-*/=<>!&|^%]+|[(){}\[\],;:.]|\s+'
    tokens = re.findall(pattern, code)
    return tokens


def compute_token_error_rate(prediction: str, target: str) -> float:
    """Compute token-level error rate.

    Returns the edit distance divided by the maximum length.
    """
    pred_tokens = tokenize_code(prediction)
    target_tokens = tokenize_code(target)

    # Token-level edit distance
    distance = compute_edit_distance(
        " ".join(pred_tokens),
        " ".join(target_tokens)
    )

    max_len = max(len(pred_tokens), len(target_tokens), 1)
    return distance / max_len


def normalize_code(code: str) -> str:
    """Normalize code for comparison.

    - Strips leading/trailing whitespace
    - Normalizes line endings
    - Removes trailing whitespace from lines
    """
    lines = code.strip().split("\n")
    lines = [line.rstrip() for line in lines]
    return "\n".join(lines)


def is_semantically_equivalent(prediction: str, target: str) -> bool:
    """Check if two code snippets are semantically equivalent.

    Currently uses normalized string comparison.
    Future: Could use AST comparison for Python code.
    """
    return normalize_code(prediction) == normalize_code(target)


# =============================================================================
# Code Repair Benchmark
# =============================================================================


class CodeRepairBenchmark:
    """Benchmark suite for code repair accuracy.

    Evaluates a code repair pipeline on a dataset of buggy/fixed code pairs.
    Computes various accuracy metrics and provides detailed breakdowns.

    Attributes:
        pipeline: The code repair pipeline to benchmark.
        dataset: The benchmark dataset.
        device: Device to run on (cpu/cuda).
        verbose: Whether to print progress.

    Example:
        >>> pipeline = CodeRepairPipeline(...)
        >>> benchmark = CodeRepairBenchmark(pipeline)
        >>> results = benchmark.run(max_samples=100)
        >>> print(results.summary())
    """

    def __init__(
        self,
        pipeline: CodeRepairPipeline,
        dataset: Optional[BenchmarkDataset] = None,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """Initialize the benchmark.

        Args:
            pipeline: Code repair pipeline to evaluate.
            dataset: Benchmark dataset (loads default if None).
            device: Device to run on.
            verbose: Whether to print progress.
        """
        self.pipeline = pipeline
        self.dataset = dataset or self.load_dataset("simple_bugs")
        self.device = device
        self.verbose = verbose

    def run(
        self,
        max_samples: Optional[int] = None,
        categories: Optional[List[str]] = None,
        difficulty_range: Optional[Tuple[float, float]] = None,
    ) -> BenchmarkResult:
        """Run the full benchmark.

        Args:
            max_samples: Maximum number of samples to evaluate.
            categories: Filter by bug type categories.
            difficulty_range: Filter by difficulty (min, max).

        Returns:
            BenchmarkResult with all metrics.
        """
        # Apply filters
        dataset = self.dataset
        if categories:
            samples = [s for s in dataset.samples if s.bug_type in categories]
            dataset = BenchmarkDataset(
                name=f"{dataset.name}_filtered",
                samples=samples,
            )
        if difficulty_range:
            dataset = dataset.filter_by_difficulty(*difficulty_range)

        # Limit samples
        samples = dataset.samples[:max_samples] if max_samples else dataset.samples

        if self.verbose:
            print(f"Running benchmark on {len(samples)} samples...")

        # Evaluate all samples
        sample_results = []
        start_time = time.perf_counter()

        for i, sample in enumerate(samples):
            result = self.evaluate_sample(sample)
            sample_results.append(result)

            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(samples)} samples...")

        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Compute aggregate metrics
        return self._aggregate_results(sample_results, total_time_ms)

    def evaluate_sample(self, sample: CodeRepairSample) -> SampleResult:
        """Evaluate a single sample.

        Args:
            sample: The code repair sample to evaluate.

        Returns:
            SampleResult with evaluation metrics.
        """
        start_time = time.perf_counter()

        # Run pipeline
        try:
            result = self.pipeline.repair(sample.buggy_code)
            prediction = result.get("repaired_code", "")
            confidence = result.get("confidence", None)
        except Exception as e:
            # Handle pipeline errors
            prediction = ""
            confidence = None
            if self.verbose:
                print(f"  Warning: Pipeline error on sample {sample.id}: {e}")

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Compute metrics
        target = sample.fixed_code
        is_exact_match = normalize_code(prediction) == normalize_code(target)
        is_correct = is_semantically_equivalent(prediction, target)
        token_error_rate = compute_token_error_rate(prediction, target)

        return SampleResult(
            sample_id=sample.id,
            prediction=prediction,
            target=target,
            is_correct=is_correct,
            is_exact_match=is_exact_match,
            token_error_rate=token_error_rate,
            latency_ms=latency_ms,
            confidence=confidence,
        )

    def compute_metrics(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> Dict[str, float]:
        """Compute accuracy metrics for predictions vs targets.

        Args:
            predictions: List of predicted code strings.
            targets: List of target code strings.

        Returns:
            Dictionary with accuracy metrics.
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        n = len(predictions)
        if n == 0:
            return {
                "accuracy": 0.0,
                "exact_match": 0.0,
                "token_error_rate": 0.0,
            }

        exact_matches = sum(
            normalize_code(p) == normalize_code(t)
            for p, t in zip(predictions, targets)
        )
        correct = sum(
            is_semantically_equivalent(p, t)
            for p, t in zip(predictions, targets)
        )
        total_ter = sum(
            compute_token_error_rate(p, t)
            for p, t in zip(predictions, targets)
        )

        return {
            "accuracy": correct / n,
            "exact_match": exact_matches / n,
            "token_error_rate": total_ter / n,
        }

    def _aggregate_results(
        self,
        sample_results: List[SampleResult],
        total_time_ms: float,
    ) -> BenchmarkResult:
        """Aggregate individual sample results into benchmark result."""
        n = len(sample_results)
        if n == 0:
            return BenchmarkResult(
                accuracy=0.0,
                exact_match=0.0,
                token_error_rate=0.0,
                by_category={},
                by_difficulty={},
                total_samples=0,
                successful_repairs=0,
            )

        # Overall metrics
        accuracy = sum(r.is_correct for r in sample_results) / n
        exact_match = sum(r.is_exact_match for r in sample_results) / n
        token_error_rate = sum(r.token_error_rate for r in sample_results) / n
        successful_repairs = sum(r.is_correct for r in sample_results)
        avg_latency_ms = sum(r.latency_ms for r in sample_results) / n

        # By category
        by_category = self._compute_category_results(sample_results)

        # By difficulty
        by_difficulty = self._compute_difficulty_results(sample_results)

        return BenchmarkResult(
            accuracy=accuracy,
            exact_match=exact_match,
            token_error_rate=token_error_rate,
            by_category=by_category,
            by_difficulty=by_difficulty,
            total_samples=n,
            successful_repairs=successful_repairs,
            total_time_ms=total_time_ms,
            avg_latency_ms=avg_latency_ms,
            sample_results=sample_results,
        )

    def _compute_category_results(
        self,
        sample_results: List[SampleResult],
    ) -> Dict[str, CategoryResult]:
        """Compute results by category."""
        # Group by category
        by_cat: Dict[str, List[SampleResult]] = {}
        for result in sample_results:
            # Get category from dataset
            sample = next(
                (s for s in self.dataset.samples if s.id == result.sample_id),
                None
            )
            if sample:
                cat = sample.bug_type
                if cat not in by_cat:
                    by_cat[cat] = []
                by_cat[cat].append(result)

        # Compute category results
        category_results = {}
        for cat, results in by_cat.items():
            n = len(results)
            category_results[cat] = CategoryResult(
                category=cat,
                num_samples=n,
                accuracy=sum(r.is_correct for r in results) / n,
                exact_match=sum(r.is_exact_match for r in results) / n,
                avg_token_error_rate=sum(r.token_error_rate for r in results) / n,
                avg_latency_ms=sum(r.latency_ms for r in results) / n,
            )

        return category_results

    def _compute_difficulty_results(
        self,
        sample_results: List[SampleResult],
    ) -> Dict[str, float]:
        """Compute accuracy by difficulty level."""
        # Group by difficulty
        easy_correct = easy_total = 0
        medium_correct = medium_total = 0
        hard_correct = hard_total = 0

        for result in sample_results:
            sample = next(
                (s for s in self.dataset.samples if s.id == result.sample_id),
                None
            )
            if sample:
                if sample.difficulty < 0.33:
                    easy_correct += result.is_correct
                    easy_total += 1
                elif sample.difficulty < 0.66:
                    medium_correct += result.is_correct
                    medium_total += 1
                else:
                    hard_correct += result.is_correct
                    hard_total += 1

        return {
            "easy": easy_correct / easy_total if easy_total > 0 else 0.0,
            "medium": medium_correct / medium_total if medium_total > 0 else 0.0,
            "hard": hard_correct / hard_total if hard_total > 0 else 0.0,
        }

    @staticmethod
    def load_dataset(name: str) -> BenchmarkDataset:
        """Load a built-in benchmark dataset.

        Available datasets:
            - "simple_bugs": Simple, common bugs (typos, off-by-one, etc.)
            - "complex_bugs": More complex logical bugs
            - "real_world": Curated real-world bug samples

        Args:
            name: Name of the dataset to load.

        Returns:
            BenchmarkDataset instance.

        Raises:
            ValueError: If dataset name is not recognized.
        """
        datasets = {
            "simple_bugs": CodeRepairBenchmark._create_simple_bugs_dataset(),
            "complex_bugs": CodeRepairBenchmark._create_complex_bugs_dataset(),
            "real_world": CodeRepairBenchmark._create_real_world_dataset(),
        }

        if name not in datasets:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(datasets.keys())}"
            )

        return datasets[name]

    @staticmethod
    def _create_simple_bugs_dataset() -> BenchmarkDataset:
        """Create simple bugs dataset."""
        samples = [
            CodeRepairSample(
                id="simple_001",
                buggy_code="def add(a, b): return a + c",
                fixed_code="def add(a, b): return a + b",
                bug_type="variable_typo",
                difficulty=0.1,
            ),
            CodeRepairSample(
                id="simple_002",
                buggy_code="def get_last(lst): return lst[len(lst)]",
                fixed_code="def get_last(lst): return lst[len(lst) - 1]",
                bug_type="off_by_one",
                difficulty=0.3,
            ),
            CodeRepairSample(
                id="simple_003",
                buggy_code="def is_even(n): return n % 2 == 1",
                fixed_code="def is_even(n): return n % 2 == 0",
                bug_type="logic_error",
                difficulty=0.2,
            ),
            CodeRepairSample(
                id="simple_004",
                buggy_code="def factorial(n):\n    if n <= 1:\n        return 1\n    factorial(n - 1) * n",
                fixed_code="def factorial(n):\n    if n <= 1:\n        return 1\n    return factorial(n - 1) * n",
                bug_type="missing_return",
                difficulty=0.3,
            ),
            CodeRepairSample(
                id="simple_005",
                buggy_code="def greet(name):\n    return 'Hello, ' + name + 1",
                fixed_code="def greet(name):\n    return 'Hello, ' + name + '!'",
                bug_type="type_error",
                difficulty=0.3,
            ),
            CodeRepairSample(
                id="simple_006",
                buggy_code="def double(x): return x * x",
                fixed_code="def double(x): return x * 2",
                bug_type="operator_error",
                difficulty=0.2,
            ),
            CodeRepairSample(
                id="simple_007",
                buggy_code="def negate(x): return x",
                fixed_code="def negate(x): return -x",
                bug_type="missing_operator",
                difficulty=0.1,
            ),
            CodeRepairSample(
                id="simple_008",
                buggy_code="def square(x): return x ^ 2",
                fixed_code="def square(x): return x ** 2",
                bug_type="operator_error",
                difficulty=0.2,
            ),
        ]
        return BenchmarkDataset(
            name="simple_bugs",
            samples=samples,
            description="Simple, common programming bugs",
            version="1.0.0",
        )

    @staticmethod
    def _create_complex_bugs_dataset() -> BenchmarkDataset:
        """Create complex bugs dataset."""
        samples = [
            CodeRepairSample(
                id="complex_001",
                buggy_code="""def safe_divide(a, b):
    return a / b""",
                fixed_code="""def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b""",
                bug_type="missing_guard",
                difficulty=0.5,
            ),
            CodeRepairSample(
                id="complex_002",
                buggy_code="""def find_max(lst):
    max_val = 0
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val""",
                fixed_code="""def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val""",
                bug_type="edge_case",
                difficulty=0.6,
            ),
            CodeRepairSample(
                id="complex_003",
                buggy_code="""def count_vowels(s):
    vowels = 'aeiou'
    count = 0
    for c in s:
        if c in vowels:
            count += 1
        return count""",
                fixed_code="""def count_vowels(s):
    vowels = 'aeiou'
    count = 0
    for c in s:
        if c in vowels:
            count += 1
    return count""",
                bug_type="indentation",
                difficulty=0.4,
            ),
            CodeRepairSample(
                id="complex_004",
                buggy_code="""def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1""",
                fixed_code="""def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
                bug_type="algorithm_error",
                difficulty=0.8,
            ),
            CodeRepairSample(
                id="complex_005",
                buggy_code="""def reverse_list(lst):
    result = []
    for i in range(len(lst)):
        result.append(lst[i])
    return result""",
                fixed_code="""def reverse_list(lst):
    result = []
    for i in range(len(lst) - 1, -1, -1):
        result.append(lst[i])
    return result""",
                bug_type="algorithm_error",
                difficulty=0.5,
            ),
        ]
        return BenchmarkDataset(
            name="complex_bugs",
            samples=samples,
            description="Complex logical and algorithmic bugs",
            version="1.0.0",
        )

    @staticmethod
    def _create_real_world_dataset() -> BenchmarkDataset:
        """Create real-world bugs dataset (curated samples)."""
        samples = [
            CodeRepairSample(
                id="real_001",
                buggy_code="""def parse_int(s):
    try:
        return int(s)
    except:
        return None""",
                fixed_code="""def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return None""",
                bug_type="exception_handling",
                difficulty=0.4,
            ),
            CodeRepairSample(
                id="real_002",
                buggy_code="""def read_file(path):
    f = open(path, 'r')
    content = f.read()
    return content""",
                fixed_code="""def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content""",
                bug_type="resource_leak",
                difficulty=0.5,
            ),
            CodeRepairSample(
                id="real_003",
                buggy_code="""def get_user_age(users, name):
    for user in users:
        if user['name'] == name:
            return user['age']""",
                fixed_code="""def get_user_age(users, name):
    for user in users:
        if user['name'] == name:
            return user['age']
    return None""",
                bug_type="missing_return",
                difficulty=0.3,
            ),
            CodeRepairSample(
                id="real_004",
                buggy_code="""def merge_dicts(d1, d2):
    result = d1
    for key in d2:
        result[key] = d2[key]
    return result""",
                fixed_code="""def merge_dicts(d1, d2):
    result = d1.copy()
    for key in d2:
        result[key] = d2[key]
    return result""",
                bug_type="mutation_bug",
                difficulty=0.6,
            ),
        ]
        return BenchmarkDataset(
            name="real_world",
            samples=samples,
            description="Curated real-world programming bugs",
            version="1.0.0",
        )

    def export_results(
        self,
        results: BenchmarkResult,
        path: Union[str, Path],
        format: str = "json",
    ) -> None:
        """Export results to file.

        Args:
            results: Benchmark results to export.
            path: Output file path.
            format: Output format ("json" or "csv").
        """
        path = Path(path)

        if format == "json":
            with open(path, "w") as f:
                json.dump(results.to_dict(), f, indent=2)
        elif format == "csv":
            # Export sample results as CSV
            with open(path, "w", newline="") as f:
                if results.sample_results:
                    writer = csv.DictWriter(f, fieldnames=results.sample_results[0].to_dict().keys())
                    writer.writeheader()
                    for sample_result in results.sample_results:
                        writer.writerow(sample_result.to_dict())
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'.")


# =============================================================================
# Baseline Comparison
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of comparing a model against baselines.

    Attributes:
        model_accuracy: Model's accuracy.
        baseline_accuracies: Dictionary of baseline name -> accuracy.
        improvement_over_best_baseline: Improvement vs. best baseline.
        model_results: Full benchmark results for model.
        baseline_results: Full results for each baseline.
    """
    model_accuracy: float
    baseline_accuracies: Dict[str, float]
    improvement_over_best_baseline: float
    model_results: BenchmarkResult
    baseline_results: Dict[str, BenchmarkResult]

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=" * 60,
            "Model vs Baseline Comparison",
            "=" * 60,
            f"Model Accuracy: {self.model_accuracy:.2%}",
            "",
            "Baseline Accuracies:",
        ]
        for name, acc in sorted(self.baseline_accuracies.items()):
            lines.append(f"  {name}: {acc:.2%}")
        lines.append("")
        best_baseline = max(self.baseline_accuracies.values()) if self.baseline_accuracies else 0
        lines.append(f"Best Baseline: {best_baseline:.2%}")
        lines.append(f"Improvement: {self.improvement_over_best_baseline:+.2%}")
        lines.append("=" * 60)
        return "\n".join(lines)


class BaselineComparison:
    """Compare a code repair model against baseline approaches.

    Built-in baselines:
        - naive_copy: Just return input (0% accuracy baseline)
        - simple_heuristic: Apply simple pattern-based fixes
        - template_matching: Match against common bug templates

    Example:
        >>> comparison = BaselineComparison()
        >>> result = comparison.compare_to_baselines(pipeline, dataset)
        >>> print(result.summary())
    """

    def __init__(self, verbose: bool = True):
        """Initialize with baseline methods.

        Args:
            verbose: Whether to print progress.
        """
        self.verbose = verbose
        self.baselines = {
            "naive_copy": self._naive_copy,
            "simple_heuristic": self._simple_heuristic,
            "template_matching": self._template_matching,
        }

    def compare_to_baselines(
        self,
        pipeline: CodeRepairPipeline,
        dataset: BenchmarkDataset,
        max_samples: Optional[int] = None,
    ) -> ComparisonResult:
        """Compare pipeline against all baselines.

        Args:
            pipeline: The model pipeline to evaluate.
            dataset: Benchmark dataset.
            max_samples: Maximum samples to evaluate.

        Returns:
            ComparisonResult with all comparisons.
        """
        # Wrap baselines as pipelines
        class BaselinePipeline:
            def __init__(self, fn):
                self.fn = fn

            def repair(self, buggy_code: str, **kwargs) -> Dict[str, Any]:
                return {"repaired_code": self.fn(buggy_code)}

        # Run model benchmark
        if self.verbose:
            print("Evaluating model...")
        model_benchmark = CodeRepairBenchmark(
            pipeline, dataset, verbose=self.verbose
        )
        model_results = model_benchmark.run(max_samples=max_samples)

        # Run baseline benchmarks
        baseline_results = {}
        baseline_accuracies = {}

        for name, fn in self.baselines.items():
            if self.verbose:
                print(f"Evaluating baseline: {name}...")
            baseline_pipeline = BaselinePipeline(fn)
            baseline_benchmark = CodeRepairBenchmark(
                baseline_pipeline, dataset, verbose=False
            )
            results = baseline_benchmark.run(max_samples=max_samples)
            baseline_results[name] = results
            baseline_accuracies[name] = results.accuracy

        # Compute improvement
        best_baseline = max(baseline_accuracies.values()) if baseline_accuracies else 0
        improvement = model_results.accuracy - best_baseline

        return ComparisonResult(
            model_accuracy=model_results.accuracy,
            baseline_accuracies=baseline_accuracies,
            improvement_over_best_baseline=improvement,
            model_results=model_results,
            baseline_results=baseline_results,
        )

    def _naive_copy(self, buggy_code: str) -> str:
        """Baseline: just copy input (0% accuracy)."""
        return buggy_code

    def _simple_heuristic(self, buggy_code: str) -> str:
        """Baseline: apply simple heuristic fixes.

        Applies common fixes like:
        - Fix common typos
        - Add missing returns
        """
        fixed = buggy_code

        # Common typo fixes
        common_typos = {
            "retrun": "return",
            "pritn": "print",
            "Fasle": "False",
            "Ture": "True",
            "lenght": "length",
        }
        for typo, correct in common_typos.items():
            fixed = fixed.replace(typo, correct)

        # Try to add missing return for single expression functions
        lines = fixed.strip().split("\n")
        if len(lines) >= 2:
            last_line = lines[-1].strip()
            # Check if last line is an expression without return
            if not last_line.startswith("return ") and not last_line.endswith(":"):
                if any(c in last_line for c in "+-*/"):
                    indent = len(lines[-1]) - len(lines[-1].lstrip())
                    lines[-1] = " " * indent + "return " + last_line
                    fixed = "\n".join(lines)

        return fixed

    def _template_matching(self, buggy_code: str) -> str:
        """Baseline: template matching for common bugs.

        Matches against templates for common bug patterns.
        """
        fixed = buggy_code

        # Off-by-one patterns
        # lst[len(lst)] -> lst[len(lst) - 1]
        fixed = re.sub(
            r'\[len\((\w+)\)\]',
            r'[len(\1) - 1]',
            fixed
        )

        # range(len(x), 0) -> range(len(x) - 1, -1, -1)
        fixed = re.sub(
            r'range\(len\((\w+)\), 0\)',
            r'range(len(\1) - 1, -1, -1)',
            fixed
        )

        # x % 2 == 1 in is_even -> x % 2 == 0
        if "is_even" in fixed:
            fixed = fixed.replace("% 2 == 1", "% 2 == 0")

        # x * x for double -> x * 2
        if "double" in fixed and "x * x" in fixed:
            fixed = fixed.replace("x * x", "x * 2")

        # x ^ 2 -> x ** 2 (Python)
        fixed = re.sub(r'(\w+) \^ 2', r'\1 ** 2', fixed)

        return fixed

    def add_baseline(
        self,
        name: str,
        fn: Callable[[str], str],
    ) -> None:
        """Add a custom baseline.

        Args:
            name: Name of the baseline.
            fn: Function that takes buggy code and returns fixed code.
        """
        self.baselines[name] = fn


# =============================================================================
# CLI and Testing
# =============================================================================


def run_sample_benchmark():
    """Run a sample benchmark for testing."""
    print("=" * 60)
    print("Code Repair Benchmark - Sample Run")
    print("=" * 60)

    # Create a mock pipeline
    class MockPipeline:
        def repair(self, buggy_code: str, **kwargs) -> Dict[str, Any]:
            # Simple mock that applies template matching
            fixed = buggy_code
            # Fix some common patterns
            fixed = fixed.replace("return a + c", "return a + b")
            fixed = fixed.replace("% 2 == 1", "% 2 == 0")
            fixed = fixed.replace("x * x", "x * 2")
            return {"repaired_code": fixed, "confidence": 0.8}

    # Run benchmark
    pipeline = MockPipeline()
    dataset = CodeRepairBenchmark.load_dataset("simple_bugs")
    benchmark = CodeRepairBenchmark(pipeline, dataset)
    results = benchmark.run()

    # Print results
    print(results.summary())

    # Run baseline comparison
    print("\nRunning baseline comparison...")
    comparison = BaselineComparison()
    comp_result = comparison.compare_to_baselines(pipeline, dataset)
    print(comp_result.summary())


if __name__ == "__main__":
    run_sample_benchmark()
