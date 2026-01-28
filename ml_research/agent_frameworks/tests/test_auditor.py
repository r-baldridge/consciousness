"""
Tests for auditor components.

Tests cover:
    - AuditorAgent analysis
    - PatternExtractor
    - IntegrationGenerator
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum, auto
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Test Data Structures
# ---------------------------------------------------------------------------

class PatternCategory(Enum):
    """Categories of patterns."""
    ARCHITECTURE = auto()
    CONTEXT_MANAGEMENT = auto()
    TOOL_USE = auto()
    HUMAN_LOOP = auto()
    ERROR_HANDLING = auto()
    ORCHESTRATION = auto()


@dataclass
class Pattern:
    """A pattern extracted from a framework."""
    pattern_id: str
    name: str
    category: PatternCategory
    description: str
    source_framework: str
    implementation_notes: str = ""
    code_example: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)


@dataclass
class FrameworkSource:
    """Source information for a framework."""
    name: str
    repo_url: str
    version: str = ""
    local_path: Optional[Path] = None
    analysis_date: Optional[datetime] = None


@dataclass
class FrameworkAnalysis:
    """Analysis results for a framework."""
    source: FrameworkSource
    patterns: List[Pattern]
    architecture_summary: str
    strengths: List[str]
    weaknesses: List[str]
    integration_opportunities: List[str]


class PatternExtractor:
    """Extracts patterns from framework source code."""

    def __init__(self, backend=None):
        self._backend = backend
        self._patterns: Dict[str, Pattern] = {}

    async def analyze_framework(
        self,
        source: FrameworkSource,
        focus_areas: Optional[List[PatternCategory]] = None
    ) -> FrameworkAnalysis:
        """Analyze a framework and extract patterns."""
        patterns = []

        # Simulate pattern extraction
        if source.name.lower() == "aider":
            patterns.extend([
                Pattern(
                    pattern_id="aider_repo_map",
                    name="Repository Map",
                    category=PatternCategory.CONTEXT_MANAGEMENT,
                    description="AST-based repository structure analysis",
                    source_framework="Aider"
                ),
                Pattern(
                    pattern_id="aider_file_select",
                    name="File Selection",
                    category=PatternCategory.CONTEXT_MANAGEMENT,
                    description="Intelligent file selection for context",
                    source_framework="Aider"
                )
            ])
        elif source.name.lower() == "humanlayer":
            patterns.extend([
                Pattern(
                    pattern_id="hl_approval",
                    name="Approval Workflow",
                    category=PatternCategory.HUMAN_LOOP,
                    description="Human approval for critical operations",
                    source_framework="HumanLayer"
                )
            ])

        for p in patterns:
            self._patterns[p.pattern_id] = p

        return FrameworkAnalysis(
            source=source,
            patterns=patterns,
            architecture_summary=f"Analysis of {source.name}",
            strengths=["Well-designed API"],
            weaknesses=["Limited documentation"],
            integration_opportunities=["Can integrate context management"]
        )

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def list_patterns(
        self,
        category: Optional[PatternCategory] = None
    ) -> List[Pattern]:
        """List all patterns, optionally filtered by category."""
        patterns = list(self._patterns.values())
        if category:
            patterns = [p for p in patterns if p.category == category]
        return patterns

    def find_related(self, pattern: Pattern, limit: int = 5) -> List[Pattern]:
        """Find patterns related to a given pattern."""
        related = []
        for p in self._patterns.values():
            if p.pattern_id != pattern.pattern_id:
                if p.category == pattern.category:
                    related.append(p)
                elif pattern.tags & p.tags:
                    related.append(p)
        return related[:limit]


@dataclass
class IntegrationSpec:
    """Specification for integrating patterns."""
    name: str
    patterns: List[str]  # Pattern IDs
    architecture: str
    interfaces: Dict[str, str]
    implementation_plan: List[str]


class IntegrationGenerator:
    """Generates integration code from patterns."""

    def __init__(self, extractor: PatternExtractor):
        self._extractor = extractor
        self._generated_specs: Dict[str, IntegrationSpec] = {}

    async def generate_integration(
        self,
        pattern_ids: List[str],
        target_architecture: str = "modular"
    ) -> IntegrationSpec:
        """Generate an integration specification."""
        patterns = [
            self._extractor.get_pattern(pid)
            for pid in pattern_ids
            if self._extractor.get_pattern(pid)
        ]

        spec = IntegrationSpec(
            name=f"Integration_{len(self._generated_specs) + 1}",
            patterns=pattern_ids,
            architecture=target_architecture,
            interfaces={p.name: f"I{p.name}" for p in patterns},
            implementation_plan=[
                f"Implement {p.name} interface" for p in patterns
            ]
        )

        self._generated_specs[spec.name] = spec
        return spec

    async def generate_code(
        self,
        spec: IntegrationSpec,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate code for an integration specification."""
        code_lines = [
            '"""',
            f'Integration: {spec.name}',
            f'Architecture: {spec.architecture}',
            '"""',
            '',
            'from abc import ABC, abstractmethod',
            ''
        ]

        # Generate interfaces
        for pattern_id in spec.patterns:
            pattern = self._extractor.get_pattern(pattern_id)
            if pattern:
                code_lines.extend([
                    f'class I{pattern.name.replace(" ", "")}(ABC):',
                    f'    """{pattern.description}"""',
                    '',
                    '    @abstractmethod',
                    '    def execute(self, *args, **kwargs):',
                    '        pass',
                    ''
                ])

        code = '\n'.join(code_lines)

        if output_path:
            output_path.write_text(code)

        return code

    def list_integrations(self) -> List[IntegrationSpec]:
        """List all generated integrations."""
        return list(self._generated_specs.values())


class AuditorAgent:
    """Agent for auditing and analyzing frameworks."""

    def __init__(self, backend=None):
        self._backend = backend
        self._extractor = PatternExtractor(backend)
        self._generator = IntegrationGenerator(self._extractor)
        self._analyses: Dict[str, FrameworkAnalysis] = {}

    async def audit_framework(
        self,
        source: FrameworkSource
    ) -> FrameworkAnalysis:
        """Audit a single framework."""
        analysis = await self._extractor.analyze_framework(source)
        self._analyses[source.name] = analysis
        return analysis

    async def compare_frameworks(
        self,
        sources: List[FrameworkSource]
    ) -> Dict[str, Any]:
        """Compare multiple frameworks."""
        analyses = []
        for source in sources:
            analysis = await self.audit_framework(source)
            analyses.append(analysis)

        # Create comparison report
        comparison = {
            "frameworks": [a.source.name for a in analyses],
            "pattern_counts": {
                a.source.name: len(a.patterns) for a in analyses
            },
            "categories_covered": {
                a.source.name: list(set(p.category.name for p in a.patterns))
                for a in analyses
            }
        }

        return comparison

    async def generate_synthesis(
        self,
        pattern_ids: List[str]
    ) -> IntegrationSpec:
        """Generate a synthesis of selected patterns."""
        return await self._generator.generate_integration(pattern_ids)

    def get_analysis(self, framework_name: str) -> Optional[FrameworkAnalysis]:
        """Get analysis for a framework."""
        return self._analyses.get(framework_name)

    @property
    def extractor(self) -> PatternExtractor:
        return self._extractor

    @property
    def generator(self) -> IntegrationGenerator:
        return self._generator


# ---------------------------------------------------------------------------
# Tests for PatternExtractor
# ---------------------------------------------------------------------------

class TestPatternExtractor:
    """Tests for PatternExtractor."""

    @pytest.mark.asyncio
    async def test_analyze_framework(self):
        """Test analyzing a framework."""
        extractor = PatternExtractor()

        source = FrameworkSource(
            name="Aider",
            repo_url="https://github.com/paul-gauthier/aider",
            version="0.40.0"
        )

        analysis = await extractor.analyze_framework(source)

        assert analysis is not None
        assert len(analysis.patterns) > 0
        assert analysis.source.name == "Aider"

    @pytest.mark.asyncio
    async def test_list_patterns_by_category(self):
        """Test listing patterns by category."""
        extractor = PatternExtractor()

        source = FrameworkSource(name="Aider", repo_url="https://example.com")
        await extractor.analyze_framework(source)

        context_patterns = extractor.list_patterns(
            category=PatternCategory.CONTEXT_MANAGEMENT
        )

        assert len(context_patterns) > 0
        assert all(
            p.category == PatternCategory.CONTEXT_MANAGEMENT
            for p in context_patterns
        )

    @pytest.mark.asyncio
    async def test_get_pattern_by_id(self):
        """Test getting pattern by ID."""
        extractor = PatternExtractor()

        source = FrameworkSource(name="Aider", repo_url="https://example.com")
        await extractor.analyze_framework(source)

        pattern = extractor.get_pattern("aider_repo_map")

        assert pattern is not None
        assert pattern.name == "Repository Map"

    @pytest.mark.asyncio
    async def test_find_related_patterns(self):
        """Test finding related patterns."""
        extractor = PatternExtractor()

        # Analyze multiple frameworks
        await extractor.analyze_framework(
            FrameworkSource(name="Aider", repo_url="https://example.com")
        )
        await extractor.analyze_framework(
            FrameworkSource(name="HumanLayer", repo_url="https://example.com")
        )

        pattern = extractor.get_pattern("aider_repo_map")
        related = extractor.find_related(pattern)

        # Should find other context management pattern
        assert len(related) >= 1


# ---------------------------------------------------------------------------
# Tests for IntegrationGenerator
# ---------------------------------------------------------------------------

class TestIntegrationGenerator:
    """Tests for IntegrationGenerator."""

    @pytest.mark.asyncio
    async def test_generate_integration(self):
        """Test generating an integration spec."""
        extractor = PatternExtractor()
        await extractor.analyze_framework(
            FrameworkSource(name="Aider", repo_url="https://example.com")
        )

        generator = IntegrationGenerator(extractor)
        spec = await generator.generate_integration(
            pattern_ids=["aider_repo_map", "aider_file_select"]
        )

        assert spec is not None
        assert len(spec.patterns) == 2
        assert len(spec.interfaces) == 2

    @pytest.mark.asyncio
    async def test_generate_code(self, tmp_path):
        """Test generating code from spec."""
        extractor = PatternExtractor()
        await extractor.analyze_framework(
            FrameworkSource(name="Aider", repo_url="https://example.com")
        )

        generator = IntegrationGenerator(extractor)
        spec = await generator.generate_integration(["aider_repo_map"])

        code = await generator.generate_code(spec)

        assert "class I" in code
        assert "ABC" in code
        assert "abstractmethod" in code

    @pytest.mark.asyncio
    async def test_generate_code_to_file(self, tmp_path):
        """Test writing generated code to file."""
        extractor = PatternExtractor()
        await extractor.analyze_framework(
            FrameworkSource(name="Aider", repo_url="https://example.com")
        )

        generator = IntegrationGenerator(extractor)
        spec = await generator.generate_integration(["aider_repo_map"])

        output_file = tmp_path / "integration.py"
        await generator.generate_code(spec, output_path=output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Integration" in content

    def test_list_integrations(self):
        """Test listing all integrations."""
        extractor = PatternExtractor()
        generator = IntegrationGenerator(extractor)

        # Should be empty initially
        assert len(generator.list_integrations()) == 0


# ---------------------------------------------------------------------------
# Tests for AuditorAgent
# ---------------------------------------------------------------------------

class TestAuditorAgent:
    """Tests for AuditorAgent."""

    @pytest.mark.asyncio
    async def test_audit_framework(self):
        """Test auditing a framework."""
        auditor = AuditorAgent()

        source = FrameworkSource(
            name="Aider",
            repo_url="https://github.com/paul-gauthier/aider"
        )

        analysis = await auditor.audit_framework(source)

        assert analysis is not None
        assert auditor.get_analysis("Aider") is analysis

    @pytest.mark.asyncio
    async def test_compare_frameworks(self):
        """Test comparing multiple frameworks."""
        auditor = AuditorAgent()

        sources = [
            FrameworkSource(name="Aider", repo_url="https://example.com"),
            FrameworkSource(name="HumanLayer", repo_url="https://example.com")
        ]

        comparison = await auditor.compare_frameworks(sources)

        assert "frameworks" in comparison
        assert len(comparison["frameworks"]) == 2
        assert "pattern_counts" in comparison

    @pytest.mark.asyncio
    async def test_generate_synthesis(self):
        """Test generating a synthesis."""
        auditor = AuditorAgent()

        await auditor.audit_framework(
            FrameworkSource(name="Aider", repo_url="https://example.com")
        )

        spec = await auditor.generate_synthesis(
            pattern_ids=["aider_repo_map"]
        )

        assert spec is not None
        assert len(spec.patterns) > 0

    def test_access_extractor_and_generator(self):
        """Test accessing extractor and generator."""
        auditor = AuditorAgent()

        assert auditor.extractor is not None
        assert auditor.generator is not None
        assert isinstance(auditor.extractor, PatternExtractor)
        assert isinstance(auditor.generator, IntegrationGenerator)
