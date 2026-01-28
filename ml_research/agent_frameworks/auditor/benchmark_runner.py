"""
Framework Capability Benchmarking.

This module provides tools for benchmarking and comparing the
capabilities of different agent frameworks.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig
    from .auditor_agent import ComparisonReport


@dataclass
class BenchmarkResult:
    """
    Result of a single benchmark run.

    Attributes:
        framework: Name of the framework tested
        capability: The capability being benchmarked
        score: Score from 0.0 to 1.0
        notes: Additional notes about the result
        duration_ms: Time taken to complete benchmark
        metadata: Additional benchmark metadata
    """
    framework: str
    capability: str
    score: float
    notes: str = ""
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "capability": self.capability,
            "score": self.score,
            "notes": self.notes,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """
    A suite of benchmarks for comprehensive testing.

    Attributes:
        name: Suite name
        benchmarks: List of benchmark configurations
        description: Suite description
    """
    name: str
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""

    @classmethod
    def default_suite(cls) -> 'BenchmarkSuite':
        """Create the default benchmark suite."""
        return cls(
            name="default",
            description="Standard agent framework capability benchmarks",
            benchmarks=[
                {"capability": "tool_execution", "weight": 1.0},
                {"capability": "multi_agent", "weight": 1.0},
                {"capability": "memory", "weight": 1.0},
                {"capability": "human_loop", "weight": 0.8},
                {"capability": "context_management", "weight": 1.0},
                {"capability": "streaming", "weight": 0.7},
                {"capability": "error_handling", "weight": 0.9},
                {"capability": "extensibility", "weight": 0.8},
            ],
        )


class BenchmarkRunner:
    """
    Run benchmarks to compare framework capabilities.

    This runner evaluates frameworks against a set of capability
    benchmarks, providing scores and detailed notes for each.

    Attributes:
        backend: LLM backend for analysis
        CAPABILITIES: List of supported capabilities

    Example:
        ```python
        runner = BenchmarkRunner(backend)

        # Run single benchmark
        result = await runner.run_benchmark("langchain", "tool_execution")
        print(f"Score: {result.score}")

        # Compare all capabilities
        comparison = await runner.compare_all(["langchain", "crewai", "autogen"])
        print(comparison.to_markdown())
        ```
    """

    CAPABILITIES = [
        "tool_execution",
        "multi_agent",
        "memory",
        "human_loop",
        "context_management",
        "streaming",
        "error_handling",
        "extensibility",
        "documentation",
        "type_safety",
    ]

    CAPABILITY_DESCRIPTIONS = {
        "tool_execution": "Ability to define and execute external tools/functions",
        "multi_agent": "Support for multi-agent orchestration and collaboration",
        "memory": "State persistence and memory management capabilities",
        "human_loop": "Human-in-the-loop interaction patterns",
        "context_management": "Context window and token management",
        "streaming": "Streaming response support",
        "error_handling": "Error handling and recovery mechanisms",
        "extensibility": "Ease of extending and customizing",
        "documentation": "Quality of documentation and examples",
        "type_safety": "Type hints and runtime type checking",
    }

    BENCHMARK_SYSTEM_PROMPT = """You are an expert at evaluating agent frameworks.
Given a framework name and a capability, provide:
1. A score from 0.0 to 1.0 based on the framework's implementation
2. Specific notes about the implementation quality
3. Key features that support this capability
4. Any limitations or concerns

Base your evaluation on your knowledge of the framework's design and features.
Be objective and specific in your assessment."""

    def __init__(
        self,
        backend: 'LLMBackend',
        suite: Optional[BenchmarkSuite] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            backend: LLM backend for analysis
            suite: Benchmark suite to use (defaults to standard suite)
        """
        self.backend = backend
        self.suite = suite or BenchmarkSuite.default_suite()

        # Cache for benchmark results
        self._cache: Dict[str, BenchmarkResult] = {}

    async def run_benchmark(
        self,
        framework: str,
        capability: str,
        use_cache: bool = True,
    ) -> BenchmarkResult:
        """
        Run a single benchmark for a framework capability.

        Args:
            framework: Name of the framework
            capability: Capability to benchmark
            use_cache: Whether to use cached results

        Returns:
            Benchmark result
        """
        cache_key = f"{framework}:{capability}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.time()

        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.2,
            max_tokens=1500,
            system=self.BENCHMARK_SYSTEM_PROMPT,
        )

        capability_desc = self.CAPABILITY_DESCRIPTIONS.get(
            capability, f"The {capability} capability"
        )

        prompt = f"""Evaluate the {framework} framework for: {capability}

Capability Description: {capability_desc}

Provide:
1. SCORE: A number from 0.0 to 1.0
2. NOTES: Specific observations about this capability
3. FEATURES: Key features supporting this capability
4. LIMITATIONS: Any concerns or limitations

Format:
SCORE: [number]
NOTES: [your notes]
FEATURES: [comma-separated list]
LIMITATIONS: [comma-separated list]"""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.backend.complete(messages, config)
            result = self._parse_benchmark_response(
                response.content, framework, capability
            )
        except Exception as e:
            result = BenchmarkResult(
                framework=framework,
                capability=capability,
                score=0.0,
                notes=f"Benchmark failed: {str(e)}",
            )

        end_time = time.time()
        result.duration_ms = (end_time - start_time) * 1000

        self._cache[cache_key] = result

        return result

    async def run_suite(
        self,
        framework: str,
    ) -> List[BenchmarkResult]:
        """
        Run the complete benchmark suite for a framework.

        Args:
            framework: Framework to benchmark

        Returns:
            List of benchmark results
        """
        results = []

        # Run benchmarks concurrently for speed
        tasks = [
            self.run_benchmark(framework, b["capability"])
            for b in self.suite.benchmarks
        ]

        results = await asyncio.gather(*tasks)

        return results

    async def compare_all(
        self,
        frameworks: List[str],
    ) -> 'ComparisonReport':
        """
        Compare all capabilities across multiple frameworks.

        Runs the benchmark suite for each framework and generates
        a comprehensive comparison report.

        Args:
            frameworks: List of framework names to compare

        Returns:
            Comparison report
        """
        from .auditor_agent import ComparisonReport

        all_results: Dict[str, List[BenchmarkResult]] = {}

        # Run benchmarks for all frameworks
        for framework in frameworks:
            results = await self.run_suite(framework)
            all_results[framework] = results

        # Build comparison matrix
        comparison_matrix: Dict[str, Dict[str, Any]] = {}

        for framework, results in all_results.items():
            comparison_matrix[framework] = {}
            for result in results:
                comparison_matrix[framework][result.capability] = round(result.score, 2)

            # Calculate overall score
            total_score = sum(r.score for r in results)
            comparison_matrix[framework]["overall"] = round(
                total_score / len(results), 2
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            frameworks, comparison_matrix, all_results
        )

        # Determine best for each use case
        best_for = self._determine_best_for(frameworks, comparison_matrix)

        return ComparisonReport(
            frameworks=frameworks,
            comparison_matrix=comparison_matrix,
            recommendations=recommendations,
            best_for=best_for,
        )

    async def quick_compare(
        self,
        frameworks: List[str],
        capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Quick comparison of specific capabilities.

        Args:
            frameworks: Frameworks to compare
            capabilities: Capabilities to check (defaults to all)

        Returns:
            Simple comparison matrix
        """
        capabilities = capabilities or self.CAPABILITIES

        comparison = {}

        for framework in frameworks:
            comparison[framework] = {}

            tasks = [
                self.run_benchmark(framework, cap)
                for cap in capabilities
            ]

            results = await asyncio.gather(*tasks)

            for result in results:
                comparison[framework][result.capability] = result.score

        return comparison

    def _parse_benchmark_response(
        self,
        content: str,
        framework: str,
        capability: str,
    ) -> BenchmarkResult:
        """Parse benchmark response from LLM."""
        score = 0.5  # Default score
        notes = ""
        features = []
        limitations = []

        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            if line.upper().startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Extract number from string
                    import re
                    match = re.search(r'(\d+\.?\d*)', score_str)
                    if match:
                        score = float(match.group(1))
                        score = min(1.0, max(0.0, score))
                except (ValueError, IndexError):
                    pass

            elif line.upper().startswith("NOTES:"):
                notes = line.split(":", 1)[1].strip()

            elif line.upper().startswith("FEATURES:"):
                features_str = line.split(":", 1)[1].strip()
                features = [f.strip() for f in features_str.split(",")]

            elif line.upper().startswith("LIMITATIONS:"):
                lim_str = line.split(":", 1)[1].strip()
                limitations = [l.strip() for l in lim_str.split(",")]

        # Build comprehensive notes if not extracted
        if not notes and (features or limitations):
            notes_parts = []
            if features:
                notes_parts.append(f"Features: {', '.join(features[:3])}")
            if limitations:
                notes_parts.append(f"Limitations: {', '.join(limitations[:3])}")
            notes = ". ".join(notes_parts)

        return BenchmarkResult(
            framework=framework,
            capability=capability,
            score=score,
            notes=notes,
            metadata={
                "features": features,
                "limitations": limitations,
            },
        )

    def _generate_recommendations(
        self,
        frameworks: List[str],
        matrix: Dict[str, Dict[str, Any]],
        all_results: Dict[str, List[BenchmarkResult]],
    ) -> List[str]:
        """Generate recommendations from comparison results."""
        recommendations = []

        # Find framework with highest overall score
        overall_scores = {fw: matrix[fw].get("overall", 0) for fw in frameworks}
        best_overall = max(overall_scores, key=overall_scores.get)
        recommendations.append(
            f"{best_overall} has the highest overall score ({overall_scores[best_overall]:.2f})"
        )

        # Find standout capabilities for each framework
        for framework in frameworks:
            results = all_results[framework]
            best_caps = sorted(results, key=lambda r: r.score, reverse=True)[:2]

            if best_caps and best_caps[0].score >= 0.8:
                recommendations.append(
                    f"{framework} excels at {best_caps[0].capability} "
                    f"({best_caps[0].score:.2f})"
                )

        # Identify gaps
        for framework in frameworks:
            results = all_results[framework]
            weak_caps = [r for r in results if r.score < 0.5]

            if weak_caps:
                cap_names = ", ".join(r.capability for r in weak_caps[:2])
                recommendations.append(
                    f"{framework} could improve: {cap_names}"
                )

        return recommendations

    def _determine_best_for(
        self,
        frameworks: List[str],
        matrix: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """Determine best framework for each use case."""
        use_cases = {
            "Simple tool agents": "tool_execution",
            "Multi-agent systems": "multi_agent",
            "Stateful applications": "memory",
            "Production systems": "error_handling",
            "RAG applications": "context_management",
            "Interactive applications": "streaming",
        }

        best_for = {}

        for use_case, capability in use_cases.items():
            scores = {
                fw: matrix[fw].get(capability, 0)
                for fw in frameworks
            }
            best_framework = max(scores, key=scores.get)
            best_for[use_case] = best_framework

        return best_for

    def clear_cache(self) -> None:
        """Clear the benchmark cache."""
        self._cache.clear()
