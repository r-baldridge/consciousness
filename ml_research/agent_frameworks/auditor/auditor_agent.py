"""
Main Architecture Auditor Agent.

This module provides the primary LLM-centric auditor that analyzes agent
framework architectures, extracts patterns, and generates integration code.
It orchestrates the pattern extraction, rule-based matching, and recursive
decomposition components.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig


@dataclass
class FrameworkSource:
    """
    Source location for a framework to analyze.

    Attributes:
        url: GitHub repository URL
        path: Local filesystem path
        docs_url: URL to framework documentation
        name: Human-readable name for the framework
    """
    url: Optional[str] = None
    path: Optional[Path] = None
    docs_url: Optional[str] = None
    name: str = ""

    def __post_init__(self):
        """Validate that at least one source is provided."""
        if not self.url and not self.path:
            raise ValueError("Must provide either url or path for FrameworkSource")
        if self.path:
            self.path = Path(self.path)
        if not self.name:
            if self.url:
                # Extract name from URL
                self.name = self.url.rstrip('/').split('/')[-1]
            elif self.path:
                self.name = self.path.name


@dataclass
class FrameworkAnalysis:
    """
    Complete analysis of a framework's architecture.

    Attributes:
        name: Framework name
        architecture: Detected architecture patterns and structure
        components: Major components and their roles
        patterns: Design patterns identified in the code
        strengths: Notable strengths of the framework
        weaknesses: Identified weaknesses or limitations
        integration_points: Points where external code can integrate
        source: The original source that was analyzed
        analyzed_at: Timestamp of the analysis
        metadata: Additional analysis metadata
    """
    name: str
    architecture: Dict[str, Any] = field(default_factory=dict)
    components: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    source: Optional[FrameworkSource] = None
    analyzed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary format."""
        return {
            "name": self.name,
            "architecture": self.architecture,
            "components": self.components,
            "patterns": self.patterns,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "integration_points": self.integration_points,
            "analyzed_at": self.analyzed_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Generate a markdown report of the analysis."""
        lines = [
            f"# Framework Analysis: {self.name}",
            f"\n*Analyzed at: {self.analyzed_at.isoformat()}*\n",
            "## Architecture",
        ]

        for key, value in self.architecture.items():
            lines.append(f"- **{key}**: {value}")

        lines.append("\n## Components")
        for comp in self.components:
            name = comp.get("name", "Unknown")
            role = comp.get("role", "")
            lines.append(f"- **{name}**: {role}")

        lines.append("\n## Design Patterns")
        for pattern in self.patterns:
            lines.append(f"- {pattern}")

        lines.append("\n## Strengths")
        for strength in self.strengths:
            lines.append(f"- {strength}")

        lines.append("\n## Weaknesses")
        for weakness in self.weaknesses:
            lines.append(f"- {weakness}")

        lines.append("\n## Integration Points")
        for point in self.integration_points:
            lines.append(f"- {point}")

        return "\n".join(lines)


@dataclass
class Pattern:
    """
    A reusable pattern extracted from framework analysis.

    Attributes:
        name: Pattern name
        category: Category (orchestration, tool, memory, etc.)
        description: Detailed description of the pattern
        implementation_notes: Notes on how to implement
        code_snippets: Example code snippets
        source_framework: Framework this was extracted from
        reusability_score: How reusable this pattern is (0-1)
        dependencies: Required dependencies for this pattern
    """
    name: str
    category: str
    description: str
    implementation_notes: str
    code_snippets: List[str] = field(default_factory=list)
    source_framework: str = ""
    reusability_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary format."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "implementation_notes": self.implementation_notes,
            "code_snippets": self.code_snippets,
            "source_framework": self.source_framework,
            "reusability_score": self.reusability_score,
            "dependencies": self.dependencies,
        }


@dataclass
class ComparisonReport:
    """
    Report comparing multiple frameworks.

    Attributes:
        frameworks: List of frameworks compared
        comparison_matrix: Feature comparison matrix
        recommendations: Recommendations based on comparison
        best_for: Best framework for each use case
        generated_at: Timestamp of report generation
    """
    frameworks: List[str] = field(default_factory=list)
    comparison_matrix: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    best_for: Dict[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "frameworks": self.frameworks,
            "comparison_matrix": self.comparison_matrix,
            "recommendations": self.recommendations,
            "best_for": self.best_for,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate a markdown comparison report."""
        lines = [
            "# Framework Comparison Report",
            f"\n*Generated at: {self.generated_at.isoformat()}*\n",
            f"## Frameworks Compared: {', '.join(self.frameworks)}\n",
            "## Comparison Matrix\n",
            "| Capability | " + " | ".join(self.frameworks) + " |",
            "|" + "---|" * (len(self.frameworks) + 1),
        ]

        # Get all capabilities
        capabilities = set()
        for framework_data in self.comparison_matrix.values():
            capabilities.update(framework_data.keys())

        for cap in sorted(capabilities):
            row = f"| {cap} |"
            for fw in self.frameworks:
                value = self.comparison_matrix.get(fw, {}).get(cap, "N/A")
                row += f" {value} |"
            lines.append(row)

        lines.append("\n## Recommendations")
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        lines.append("\n## Best For Each Use Case")
        for use_case, framework in self.best_for.items():
            lines.append(f"- **{use_case}**: {framework}")

        return "\n".join(lines)


@dataclass
class EvolutionReport:
    """
    Report on industry evolution and trends.

    Attributes:
        trends: Current industry trends
        new_frameworks: Newly discovered frameworks
        deprecated_patterns: Patterns being deprecated
        recommendations: Strategic recommendations
        generated_at: Timestamp of report generation
    """
    trends: List[str] = field(default_factory=list)
    new_frameworks: List[Dict[str, Any]] = field(default_factory=list)
    deprecated_patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "trends": self.trends,
            "new_frameworks": self.new_frameworks,
            "deprecated_patterns": self.deprecated_patterns,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


class AuditorAgent:
    """
    Reviews agent architectures and generates integration patterns.

    This is the main orchestration agent that combines LLM analysis with
    rule-based extraction to provide comprehensive framework auditing.

    Attributes:
        backend: The LLM backend for analysis
        rule_engine: Optional rule-based extractor
        decomposer: Recursive decomposer for large codebases

    Example:
        ```python
        from agent_frameworks.backends import AnthropicBackend
        from agent_frameworks.auditor import AuditorAgent, FrameworkSource

        backend = AnthropicBackend()
        auditor = AuditorAgent(backend)

        source = FrameworkSource(url="https://github.com/langchain-ai/langchain")
        analysis = await auditor.analyze_framework(source)
        patterns = await auditor.extract_patterns(analysis)
        ```
    """

    # System prompts for different analysis tasks
    ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect specializing in AI agent frameworks.
Your task is to analyze framework code and documentation to understand:
1. Core architecture patterns and design decisions
2. Major components and their responsibilities
3. Design patterns used (Factory, Observer, Strategy, etc.)
4. Integration points and extensibility mechanisms
5. Strengths and weaknesses of the design

Provide detailed, actionable insights that can help developers understand and integrate with the framework."""

    PATTERN_EXTRACTION_PROMPT = """You are an expert at identifying reusable software patterns.
Given framework code and analysis, extract patterns that can be reused in other projects.
For each pattern:
1. Give it a clear, descriptive name
2. Categorize it (orchestration, memory, tools, execution, human_loop)
3. Describe what problem it solves
4. Explain how to implement it
5. Provide code snippets when helpful
6. Assess its reusability (0-1 score)"""

    COMPARISON_PROMPT = """You are an expert at comparing software frameworks.
Compare the given frameworks across multiple dimensions:
1. Feature completeness
2. Architecture quality
3. Extensibility
4. Documentation
5. Community support
6. Performance characteristics

Provide clear recommendations for different use cases."""

    def __init__(
        self,
        backend: 'LLMBackend',
        use_rules: bool = True,
        max_chunk_size: int = 4000,
    ):
        """
        Initialize the Auditor Agent.

        Args:
            backend: LLM backend for analysis
            use_rules: Whether to use rule-based extraction
            max_chunk_size: Maximum size for code chunks
        """
        self.backend = backend

        # Import here to avoid circular imports
        from .rule_engine import RuleBasedExtractor
        from .recursive_decomposer import RecursiveDecomposer
        from .pattern_extractor import PatternExtractor
        from .integration_generator import IntegrationGenerator
        from .benchmark_runner import BenchmarkRunner
        from .evolution_tracker import EvolutionTracker
        from .framework_fetcher import FrameworkFetcher

        self.rule_engine = RuleBasedExtractor() if use_rules else None
        self.decomposer = RecursiveDecomposer(backend, max_chunk_size)
        self.pattern_extractor = PatternExtractor(backend)
        self.integration_generator = IntegrationGenerator(backend)
        self.benchmark_runner = BenchmarkRunner(backend)
        self.evolution_tracker = EvolutionTracker(backend)
        self.framework_fetcher = FrameworkFetcher()

        # Cache for analyzed frameworks
        self._analysis_cache: Dict[str, FrameworkAnalysis] = {}

    async def analyze_framework(
        self,
        source: FrameworkSource,
        use_cache: bool = True,
    ) -> FrameworkAnalysis:
        """
        Analyze a new agent framework using recursive decomposition.

        This method:
        1. Fetches the framework source code
        2. Decomposes it into analyzable chunks
        3. Analyzes each chunk with the LLM
        4. Synthesizes results into a complete analysis
        5. Applies rule-based pattern extraction

        Args:
            source: The framework source to analyze
            use_cache: Whether to use cached analysis

        Returns:
            Complete framework analysis
        """
        # Check cache
        cache_key = source.name or str(source.url or source.path)
        if use_cache and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        # Fetch source if needed
        if source.url and not source.path:
            source.path = await self.framework_fetcher.fetch_github(source.url)

        # Decompose into chunks
        chunks = await self.decomposer.decompose(source)

        # Analyze each chunk
        chunk_analyses = await asyncio.gather(*[
            self.decomposer.analyze_chunk(chunk)
            for chunk in chunks
        ])

        # Synthesize into complete analysis
        analysis = await self.decomposer.synthesize(chunk_analyses)
        analysis.source = source
        analysis.name = source.name

        # Apply rule-based extraction if enabled
        if self.rule_engine:
            rule_patterns = self.rule_engine.extract(analysis)
            # Merge rule-based patterns with LLM patterns
            existing_patterns = set(analysis.patterns)
            for pattern in rule_patterns:
                if pattern.name not in existing_patterns:
                    analysis.patterns.append(pattern.name)

        # Cache the result
        self._analysis_cache[cache_key] = analysis

        return analysis

    async def extract_patterns(
        self,
        analysis: FrameworkAnalysis,
    ) -> List[Pattern]:
        """
        Extract reusable patterns from analysis.

        Combines LLM-based pattern extraction with rule-based matching
        to identify patterns that can be reused in other projects.

        Args:
            analysis: The framework analysis to extract patterns from

        Returns:
            List of extracted patterns
        """
        patterns = []

        # LLM-based extraction
        llm_patterns = await self.pattern_extractor.extract_from_analysis(analysis)
        patterns.extend(llm_patterns)

        # Rule-based extraction
        if self.rule_engine:
            rule_patterns = self.rule_engine.extract(analysis)
            # Merge avoiding duplicates
            existing_names = {p.name for p in patterns}
            for pattern in rule_patterns:
                if pattern.name not in existing_names:
                    patterns.append(pattern)

        # Score patterns for reusability
        for pattern in patterns:
            if pattern.reusability_score == 0:
                pattern.reusability_score = await self.pattern_extractor.score_reusability(pattern)

        # Sort by reusability score
        patterns.sort(key=lambda p: p.reusability_score, reverse=True)

        return patterns

    async def generate_integration(
        self,
        pattern: Pattern,
        target_framework: str = "agent_frameworks",
    ) -> 'IntegrationCode':
        """
        Generate integration code for a pattern via LLM.

        Creates a complete, usable integration module based on an
        extracted pattern, ready to use with the target framework.

        Args:
            pattern: The pattern to generate integration for
            target_framework: The framework to integrate with

        Returns:
            Complete integration code with tests
        """
        from .integration_generator import IntegrationCode

        return await self.integration_generator.generate(pattern, target_framework)

    async def compare_frameworks(
        self,
        frameworks: List[str],
    ) -> ComparisonReport:
        """
        Compare capabilities across frameworks.

        Analyzes multiple frameworks and generates a comparison report
        with recommendations for different use cases.

        Args:
            frameworks: List of framework names or URLs to compare

        Returns:
            Detailed comparison report
        """
        # Analyze each framework
        analyses = []
        for fw in frameworks:
            # Check if it's a URL or name
            if fw.startswith("http"):
                source = FrameworkSource(url=fw)
            else:
                # Assume it's cached or a known framework
                if fw in self._analysis_cache:
                    analyses.append(self._analysis_cache[fw])
                    continue
                # Try as a GitHub URL
                source = FrameworkSource(url=f"https://github.com/{fw}")

            analysis = await self.analyze_framework(source)
            analyses.append(analysis)

        # Run benchmarks
        benchmark_results = await self.benchmark_runner.compare_all(
            [a.name for a in analyses]
        )

        # Build comparison matrix
        comparison_matrix = {}
        for analysis in analyses:
            comparison_matrix[analysis.name] = {
                "patterns": len(analysis.patterns),
                "components": len(analysis.components),
                "integration_points": len(analysis.integration_points),
                "strengths": len(analysis.strengths),
            }

            # Add benchmark results
            for result in benchmark_results:
                if result.framework == analysis.name:
                    comparison_matrix[analysis.name][result.capability] = result.score

        # Generate recommendations via LLM
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.3,
            max_tokens=2000,
            system=self.COMPARISON_PROMPT,
        )

        comparison_context = json.dumps({
            "frameworks": [a.to_dict() for a in analyses],
            "comparison_matrix": comparison_matrix,
        }, indent=2)

        messages = [{
            "role": "user",
            "content": f"Compare these frameworks and provide recommendations:\n{comparison_context}",
        }]

        response = await self.backend.complete(messages, config)

        # Parse recommendations from response
        recommendations = []
        best_for = {}

        lines = response.content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                recommendations.append(line[2:])
            if "best for" in line.lower():
                # Try to extract use case -> framework mapping
                parts = line.split(":")
                if len(parts) == 2:
                    use_case = parts[0].replace("Best for", "").strip()
                    framework = parts[1].strip()
                    best_for[use_case] = framework

        return ComparisonReport(
            frameworks=[a.name for a in analyses],
            comparison_matrix=comparison_matrix,
            recommendations=recommendations,
            best_for=best_for,
        )

    async def track_evolution(self) -> EvolutionReport:
        """
        Track industry trends via web search + LLM.

        Monitors the agent framework ecosystem for new developments,
        emerging patterns, and deprecated approaches.

        Returns:
            Evolution report with trends and recommendations
        """
        return await self.evolution_tracker.generate_report()

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._analysis_cache.clear()

    def get_cached_analysis(self, name: str) -> Optional[FrameworkAnalysis]:
        """Get a cached analysis by framework name."""
        return self._analysis_cache.get(name)
