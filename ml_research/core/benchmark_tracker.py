"""
Benchmark Tracker module for ML Research.

Tracks state-of-the-art results on ML benchmarks, maintaining
historical records and enabling comparison across methods.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .taxonomy import Benchmark


class BenchmarkTracker:
    """
    Tracks state-of-the-art results on ML benchmarks.

    Manages benchmark definitions, current SOTA results, and
    historical progress over time.

    Example:
        tracker = BenchmarkTracker()
        tracker.register_benchmark(imagenet_benchmark)
        tracker.update_sota("imagenet_top1", "vit_huge", 90.94)
        history = tracker.get_history("imagenet_top1")
    """

    def __init__(self) -> None:
        """Initialize an empty benchmark tracker."""
        self._benchmarks: Dict[str, Benchmark] = {}

    def register_benchmark(self, benchmark: Benchmark) -> None:
        """
        Register a new benchmark.

        Args:
            benchmark: The Benchmark instance to register.

        Raises:
            ValueError: If a benchmark with the same benchmark_id already exists.
        """
        if benchmark.benchmark_id in self._benchmarks:
            raise ValueError(
                f"Benchmark '{benchmark.benchmark_id}' is already registered"
            )
        self._benchmarks[benchmark.benchmark_id] = benchmark

    def update_sota(
        self,
        benchmark_id: str,
        method_id: str,
        score: float,
        *,
        date: Optional[str] = None
    ) -> bool:
        """
        Update the state-of-the-art for a benchmark.

        Only updates if the new score is better than the current SOTA.
        For most metrics, higher is better. Use negative scores for
        metrics where lower is better (e.g., perplexity, error rate).

        Args:
            benchmark_id: The benchmark to update.
            method_id: The method achieving the new result.
            score: The score achieved.
            date: Optional date string (defaults to current date).

        Returns:
            True if SOTA was updated, False if not (including not found).
        """
        if benchmark_id not in self._benchmarks:
            return False

        benchmark = self._benchmarks[benchmark_id]

        # Use current date if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Check if this is a new SOTA (assuming higher is better)
        is_improvement = score > benchmark.sota_score

        if is_improvement:
            # Add previous SOTA to history
            benchmark.history.append((
                date,
                benchmark.sota_method,
                benchmark.sota_score
            ))

            # Update current SOTA
            benchmark.sota_method = method_id
            benchmark.sota_score = score

            return True

        return False

    def get_benchmark(self, benchmark_id: str) -> Optional[Benchmark]:
        """
        Retrieve a benchmark by its unique identifier.

        Args:
            benchmark_id: The unique identifier of the benchmark.

        Returns:
            The Benchmark if found, None otherwise.
        """
        return self._benchmarks.get(benchmark_id)

    def get_history(
        self,
        benchmark_id: str,
        *,
        limit: Optional[int] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Get the historical SOTA progression for a benchmark.

        Args:
            benchmark_id: The benchmark to get history for.
            limit: Optional limit on number of entries (most recent first).

        Returns:
            List of (date, method_id, score) tuples. Empty list if not found.
        """
        if benchmark_id not in self._benchmarks:
            return []

        benchmark = self._benchmarks[benchmark_id]

        # Include current SOTA in history view
        current = (
            datetime.now().strftime("%Y-%m-%d"),
            benchmark.sota_method,
            benchmark.sota_score
        )

        full_history = benchmark.history + [current]

        # Sort by date descending (most recent first)
        full_history.sort(key=lambda x: x[0], reverse=True)

        if limit:
            return full_history[:limit]
        return full_history

    def get_all_benchmarks(self) -> List[Benchmark]:
        """
        Get all registered benchmarks.

        Returns:
            List of all Benchmark instances.
        """
        return list(self._benchmarks.values())

    def get_benchmarks_by_domain(self, domain: str) -> List[Benchmark]:
        """
        Get all benchmarks for a specific domain.

        Args:
            domain: The domain to filter by (e.g., "image_classification").

        Returns:
            List of Benchmark instances for that domain.
        """
        return [
            b for b in self._benchmarks.values()
            if b.domain.lower() == domain.lower()
        ]

    def get_benchmarks_by_dataset(self, dataset: str) -> List[Benchmark]:
        """
        Get all benchmarks using a specific dataset.

        Args:
            dataset: The dataset name to filter by.

        Returns:
            List of Benchmark instances using that dataset.
        """
        dataset_lower = dataset.lower()
        return [
            b for b in self._benchmarks.values()
            if dataset_lower in b.dataset.lower()
        ]

    def get_method_results(
        self,
        method_id: str
    ) -> Dict[str, float]:
        """
        Get all benchmark results for a specific method.

        Args:
            method_id: The method to get results for.

        Returns:
            Dictionary mapping benchmark_id to score for current SOTAs.
        """
        results: Dict[str, float] = {}
        for benchmark_id, benchmark in self._benchmarks.items():
            if benchmark.sota_method == method_id:
                results[benchmark_id] = benchmark.sota_score
        return results

    def compare_methods(
        self,
        method_ids: List[str],
        benchmark_id: str
    ) -> Dict[str, Optional[float]]:
        """
        Compare multiple methods on a single benchmark.

        Searches history for each method's best result on the benchmark.

        Args:
            method_ids: List of methods to compare.
            benchmark_id: The benchmark to compare on.

        Returns:
            Dictionary mapping method_id to score (None if no result).
        """
        if benchmark_id not in self._benchmarks:
            return {mid: None for mid in method_ids}

        benchmark = self._benchmarks[benchmark_id]
        results: Dict[str, Optional[float]] = {}

        for method_id in method_ids:
            # Check if current SOTA
            if benchmark.sota_method == method_id:
                results[method_id] = benchmark.sota_score
            else:
                # Search history
                found = None
                for _, hist_method, hist_score in benchmark.history:
                    if hist_method == method_id:
                        if found is None or hist_score > found:
                            found = hist_score
                results[method_id] = found

        return results

    def unregister_benchmark(self, benchmark_id: str) -> bool:
        """
        Remove a benchmark from the tracker.

        Args:
            benchmark_id: The unique identifier of the benchmark to remove.

        Returns:
            True if removed, False if not found.
        """
        if benchmark_id in self._benchmarks:
            del self._benchmarks[benchmark_id]
            return True
        return False

    def count(self) -> int:
        """
        Get the total number of registered benchmarks.

        Returns:
            The count of benchmarks.
        """
        return len(self._benchmarks)

    def clear(self) -> None:
        """Remove all benchmarks from the tracker."""
        self._benchmarks.clear()

    def get_domains(self) -> List[str]:
        """
        Get a list of all unique domains.

        Returns:
            List of domain names.
        """
        return list(set(b.domain for b in self._benchmarks.values()))

    def get_datasets(self) -> List[str]:
        """
        Get a list of all unique datasets.

        Returns:
            List of dataset names.
        """
        return list(set(b.dataset for b in self._benchmarks.values()))

    def get_progress_summary(
        self,
        benchmark_id: str
    ) -> Optional[Dict[str, any]]:
        """
        Get a summary of progress on a benchmark.

        Args:
            benchmark_id: The benchmark to summarize.

        Returns:
            Dictionary with progress information, or None if not found:
                - benchmark_name: Name of the benchmark
                - metric: What is being measured
                - current_sota: Current best score
                - current_method: Method achieving SOTA
                - total_improvements: Number of SOTA changes
                - first_recorded: Earliest score in history
                - improvement_percentage: Progress from first to current
        """
        if benchmark_id not in self._benchmarks:
            return None

        benchmark = self._benchmarks[benchmark_id]

        history = benchmark.history
        if not history:
            first_score = benchmark.sota_score
        else:
            # Get earliest recorded score
            sorted_history = sorted(history, key=lambda x: x[0])
            first_score = sorted_history[0][2]

        improvement = (
            ((benchmark.sota_score - first_score) / abs(first_score) * 100)
            if first_score != 0 else 0
        )

        return {
            "benchmark_name": benchmark.name,
            "metric": benchmark.metric,
            "current_sota": benchmark.sota_score,
            "current_method": benchmark.sota_method,
            "total_improvements": len(history),
            "first_recorded": first_score,
            "improvement_percentage": round(improvement, 2)
        }
