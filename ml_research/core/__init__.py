"""
ML Research Core Module.

This module provides the core infrastructure for tracking, documenting,
and analyzing machine learning methods, papers, and benchmarks.

Components:
    - taxonomy: Enums and dataclasses for categorizing ML methods
    - method_registry: Central registry for all ML methods
    - timeline: Historical timeline of ML development
    - lineage_tracker: Tracks evolutionary relationships between methods
    - paper_index: Database of key ML papers
    - benchmark_tracker: SOTA tracking on standard benchmarks

Example:
    from ml_research.core import (
        MLMethod, MethodEra, MethodCategory, MethodLineage,
        MethodRegistry, Timeline, LineageTracker
    )

    # Create a method
    method = MLMethod(
        method_id="transformer_2017",
        name="Transformer",
        year=2017,
        era=MethodEra.ATTENTION,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.ATTENTION_LINE],
        authors=["Vaswani", "Shazeer", "Parmar", ...],
        paper_title="Attention Is All You Need",
        paper_url="https://arxiv.org/abs/1706.03762",
        key_innovation="Self-attention replacing recurrence entirely",
        mathematical_formulation="Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V"
    )

    # Register it
    registry = MethodRegistry()
    registry.register(method)
"""

# Taxonomy - Enums and Dataclasses
from .taxonomy import (
    MethodEra,
    MethodCategory,
    MethodLineage,
    MLMethod,
    Paper,
    Benchmark,
)

# Method Registry
from .method_registry import (
    MethodRegistry,
    get_global_registry,
)

# Timeline
from .timeline import (
    Timeline,
    TimelineEntry,
)

# Lineage Tracker
from .lineage_tracker import (
    LineageTracker,
    LineageNode,
    LineageInfo,
)

# Paper Index
from .paper_index import (
    PaperIndex,
)

# Benchmark Tracker
from .benchmark_tracker import (
    BenchmarkTracker,
)

__all__ = [
    # Taxonomy
    "MethodEra",
    "MethodCategory",
    "MethodLineage",
    "MLMethod",
    "Paper",
    "Benchmark",
    # Method Registry
    "MethodRegistry",
    "get_global_registry",
    # Timeline
    "Timeline",
    "TimelineEntry",
    # Lineage Tracker
    "LineageTracker",
    "LineageNode",
    "LineageInfo",
    # Paper Index
    "PaperIndex",
    # Benchmark Tracker
    "BenchmarkTracker",
]

__version__ = "0.1.0"
