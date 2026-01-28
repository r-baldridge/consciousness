"""
Architecture Auditor Module for Agent Frameworks.

This module provides tools for analyzing agent framework architectures,
extracting reusable patterns, generating integration code, and tracking
industry evolution. It combines LLM-powered analysis with rule-based
AST pattern matching for comprehensive framework auditing.

Key Components:
- AuditorAgent: Main LLM-centric auditor for framework analysis
- PatternExtractor: LLM-based pattern extraction from code
- RuleBasedExtractor: AST-based rule matching for patterns
- RecursiveDecomposer: Break codebases into analyzable chunks
- IntegrationGenerator: Generate integration code from patterns
- BenchmarkRunner: Compare framework capabilities
- EvolutionTracker: Track industry trends and evolution
- FrameworkFetcher: Fetch code from GitHub/PyPI/URLs
"""

from .auditor_agent import (
    AuditorAgent,
    FrameworkSource,
    FrameworkAnalysis,
    Pattern,
    ComparisonReport,
    EvolutionReport,
)
from .pattern_extractor import PatternExtractor
from .rule_engine import RuleBasedExtractor, ASTRule
from .recursive_decomposer import RecursiveDecomposer, CodeChunk
from .integration_generator import IntegrationGenerator, IntegrationCode
from .benchmark_runner import BenchmarkRunner, BenchmarkResult
from .evolution_tracker import EvolutionTracker
from .framework_fetcher import FrameworkFetcher

__all__ = [
    # Main agent
    "AuditorAgent",
    # Data classes
    "FrameworkSource",
    "FrameworkAnalysis",
    "Pattern",
    "ComparisonReport",
    "EvolutionReport",
    "CodeChunk",
    "IntegrationCode",
    "BenchmarkResult",
    "ASTRule",
    # Extractors
    "PatternExtractor",
    "RuleBasedExtractor",
    # Utilities
    "RecursiveDecomposer",
    "IntegrationGenerator",
    "BenchmarkRunner",
    "EvolutionTracker",
    "FrameworkFetcher",
]
