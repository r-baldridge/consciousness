"""
Industry Evolution Tracking.

This module provides tools for tracking the evolution of agent
frameworks and industry trends using LLM analysis.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..backends.backend_base import LLMBackend, LLMConfig
    from .auditor_agent import EvolutionReport


class EvolutionTracker:
    """
    Track industry evolution via LLM analysis.

    This tracker monitors the agent framework ecosystem for new
    developments, emerging patterns, and deprecated approaches
    using LLM-based analysis of trends and frameworks.

    Attributes:
        backend: LLM backend for analysis

    Example:
        ```python
        tracker = EvolutionTracker(backend)

        trends = await tracker.scan_trends()
        new_frameworks = await tracker.find_new_frameworks()
        report = await tracker.generate_report()

        print(report.to_dict())
        ```
    """

    TRENDS_SYSTEM_PROMPT = """You are an expert analyst tracking AI agent framework evolution.
Based on your knowledge of the AI/ML ecosystem, identify:
1. Current trends in agent framework development
2. Emerging patterns and architectures
3. Technologies gaining or losing adoption
4. Key innovations and breakthroughs

Focus on practical, actionable insights for framework developers."""

    FRAMEWORKS_SYSTEM_PROMPT = """You are an expert at discovering and evaluating AI agent frameworks.
Identify notable frameworks in the ecosystem, including:
1. Name and primary purpose
2. Key distinguishing features
3. Technology stack and language
4. Community size and activity level
5. Maturity and production readiness

Focus on frameworks with significant adoption or innovative approaches."""

    DEPRECATION_SYSTEM_PROMPT = """You are an expert at identifying deprecated patterns in AI agent frameworks.
Analyze current trends to identify:
1. Patterns that are being deprecated
2. Approaches that are falling out of favor
3. Technologies being replaced
4. Anti-patterns to avoid

Provide specific, actionable guidance for framework developers."""

    RECOMMENDATIONS_SYSTEM_PROMPT = """You are a strategic advisor for AI agent framework development.
Based on current trends and evolution, provide:
1. Strategic recommendations for framework development
2. Technologies and patterns to invest in
3. Risks to mitigate
4. Opportunities to explore

Be specific and actionable in your recommendations."""

    # Known major frameworks for reference
    KNOWN_FRAMEWORKS = [
        "LangChain",
        "LlamaIndex",
        "AutoGen",
        "CrewAI",
        "Semantic Kernel",
        "Haystack",
        "Guidance",
        "DSPy",
        "Agency Swarm",
        "LangGraph",
        "Pydantic AI",
        "Marvin",
        "Instructor",
        "Outlines",
    ]

    def __init__(self, backend: 'LLMBackend'):
        """
        Initialize the evolution tracker.

        Args:
            backend: LLM backend for analysis
        """
        self.backend = backend

        # Cache for reports
        self._last_report: Optional['EvolutionReport'] = None
        self._last_report_time: Optional[datetime] = None

    async def scan_trends(self) -> List[str]:
        """
        Scan for current industry trends.

        Uses LLM analysis to identify current trends in the
        agent framework ecosystem.

        Returns:
            List of identified trends
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.3,
            max_tokens=2000,
            system=self.TRENDS_SYSTEM_PROMPT,
        )

        prompt = f"""Analyze current trends in the AI agent framework ecosystem.

Known frameworks for context: {', '.join(self.KNOWN_FRAMEWORKS)}

Identify the top 10 trends, considering:
1. Architecture patterns (e.g., graph-based, reactive)
2. Capability trends (e.g., tool use, multi-modal)
3. Integration patterns (e.g., MCP, standardization)
4. Deployment trends (e.g., edge, serverless)
5. Developer experience trends

Format each trend as a single line starting with a bullet point (-)."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        # Parse trends from response
        trends = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                trend = line[1:].strip()
                if trend:
                    trends.append(trend)
            elif line and not line.startswith("#") and len(line) > 20:
                # Also capture trends not in list format
                trends.append(line)

        return trends[:15]  # Limit to top 15 trends

    async def find_new_frameworks(self) -> List[Dict[str, Any]]:
        """
        Find new and emerging frameworks.

        Identifies frameworks that are gaining traction or
        represent innovative approaches.

        Returns:
            List of framework information dictionaries
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.3,
            max_tokens=2500,
            system=self.FRAMEWORKS_SYSTEM_PROMPT,
        )

        prompt = f"""Identify notable AI agent frameworks, especially:
1. Frameworks gaining significant traction
2. Innovative new frameworks
3. Frameworks with unique approaches

Known established frameworks: {', '.join(self.KNOWN_FRAMEWORKS)}

For each framework, provide:
- Name
- Description (1-2 sentences)
- Key features (2-3 points)
- Language/stack
- Maturity level (experimental/beta/stable/mature)
- Notable for (what makes it stand out)

Format as structured entries with clear labels."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        # Parse frameworks from response
        frameworks = []
        current_framework: Dict[str, Any] = {}

        for line in response.content.split("\n"):
            line = line.strip()

            # Detect framework name headers
            if line.startswith("#") or (line and line[0].isdigit() and "." in line):
                if current_framework.get("name"):
                    frameworks.append(current_framework)
                current_framework = {
                    "name": line.lstrip("#0123456789. ").split(":")[0].strip()
                }

            # Parse key-value pairs
            elif ":" in line and current_framework:
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                if key in ["description", "language", "stack", "maturity", "maturity_level", "notable_for"]:
                    current_framework[key] = value
                elif key in ["key_features", "features"]:
                    current_framework["features"] = [f.strip() for f in value.split(",")]

            # Parse bullet points as features
            elif line.startswith("-") and current_framework:
                if "features" not in current_framework:
                    current_framework["features"] = []
                current_framework["features"].append(line[1:].strip())

        # Don't forget the last framework
        if current_framework.get("name"):
            frameworks.append(current_framework)

        return frameworks

    async def find_deprecated_patterns(self) -> List[str]:
        """
        Identify deprecated patterns and approaches.

        Analyzes trends to find patterns that are being
        phased out or considered anti-patterns.

        Returns:
            List of deprecated patterns
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.3,
            max_tokens=1500,
            system=self.DEPRECATION_SYSTEM_PROMPT,
        )

        prompt = """Identify patterns being deprecated in AI agent frameworks:

1. Architecture anti-patterns
2. Deprecated API patterns
3. Superseded approaches
4. Patterns with known issues

For each, explain:
- What the pattern is
- Why it's being deprecated
- What to use instead

Format each as a bullet point with the pattern name followed by explanation."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        # Parse deprecated patterns
        patterns = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                pattern = line[1:].strip()
                if pattern and len(pattern) > 10:
                    patterns.append(pattern)

        return patterns

    async def get_recommendations(self) -> List[str]:
        """
        Get strategic recommendations.

        Provides actionable recommendations for framework
        development based on current trends.

        Returns:
            List of recommendations
        """
        from ..backends.backend_base import LLMConfig

        config = LLMConfig(
            model=self.backend.default_model,
            temperature=0.4,
            max_tokens=2000,
            system=self.RECOMMENDATIONS_SYSTEM_PROMPT,
        )

        prompt = """Provide strategic recommendations for AI agent framework development:

Consider:
1. Technology investments
2. Architecture decisions
3. Integration priorities
4. Developer experience
5. Production readiness

Provide 8-12 specific, actionable recommendations.
Format each as a bullet point starting with an action verb."""

        messages = [{"role": "user", "content": prompt}]
        response = await self.backend.complete(messages, config)

        # Parse recommendations
        recommendations = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*") or (
                len(line) > 5 and line[0].isdigit() and "." in line[:3]
            ):
                rec = line.lstrip("-*0123456789. ").strip()
                if rec and len(rec) > 15:
                    recommendations.append(rec)

        return recommendations

    async def generate_report(
        self,
        use_cache: bool = True,
        cache_hours: int = 24,
    ) -> 'EvolutionReport':
        """
        Generate a comprehensive evolution report.

        Combines trends, frameworks, deprecated patterns, and
        recommendations into a complete report.

        Args:
            use_cache: Whether to use cached report
            cache_hours: Hours to cache report for

        Returns:
            Complete evolution report
        """
        from .auditor_agent import EvolutionReport

        # Check cache
        if (
            use_cache
            and self._last_report
            and self._last_report_time
            and (datetime.now() - self._last_report_time).total_seconds() < cache_hours * 3600
        ):
            return self._last_report

        # Run all analyses concurrently
        trends_task = self.scan_trends()
        frameworks_task = self.find_new_frameworks()
        deprecated_task = self.find_deprecated_patterns()
        recommendations_task = self.get_recommendations()

        trends, frameworks, deprecated, recommendations = await asyncio.gather(
            trends_task,
            frameworks_task,
            deprecated_task,
            recommendations_task,
        )

        report = EvolutionReport(
            trends=trends,
            new_frameworks=frameworks,
            deprecated_patterns=deprecated,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )

        # Cache the report
        self._last_report = report
        self._last_report_time = datetime.now()

        return report

    async def compare_to_previous(
        self,
        previous_report: 'EvolutionReport',
    ) -> Dict[str, Any]:
        """
        Compare current state to a previous report.

        Identifies changes and evolution since the previous report.

        Args:
            previous_report: Previous report to compare against

        Returns:
            Dictionary of changes and new developments
        """
        current_report = await self.generate_report(use_cache=False)

        # Find new trends
        previous_trends = set(previous_report.trends)
        new_trends = [t for t in current_report.trends if t not in previous_trends]

        # Find new frameworks
        previous_frameworks = {f.get("name", "") for f in previous_report.new_frameworks}
        new_frameworks = [
            f for f in current_report.new_frameworks
            if f.get("name", "") not in previous_frameworks
        ]

        # Find newly deprecated patterns
        previous_deprecated = set(previous_report.deprecated_patterns)
        new_deprecated = [
            p for p in current_report.deprecated_patterns
            if p not in previous_deprecated
        ]

        return {
            "new_trends": new_trends,
            "new_frameworks": new_frameworks,
            "new_deprecated_patterns": new_deprecated,
            "previous_report_date": previous_report.generated_at.isoformat(),
            "current_report_date": current_report.generated_at.isoformat(),
            "days_between": (
                current_report.generated_at - previous_report.generated_at
            ).days,
        }

    def clear_cache(self) -> None:
        """Clear the report cache."""
        self._last_report = None
        self._last_report_time = None
