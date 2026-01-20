"""
ML Research Module - Comprehensive Machine Learning History and Taxonomy

This module indexes the complete history of machine learning methods from 1943 to present,
organized by era, category, and lineage. It serves as a research encyclopedia, not a
collection of runnable implementations.

Coverage: 200+ methods spanning:
- Era 1: Foundational (1943-1980) - McCulloch-Pitts to Hopfield
- Era 2: Classical (1980-2006) - MLP, CNN, LSTM, SVM, Ensemble methods
- Era 3: Deep Learning (2006-2017) - AlexNet, ResNet, GAN, VAE
- Era 4: Attention (2017-present) - Transformer, BERT, GPT, LLaMA
- Era 5: Novel (2023+) - Mamba, MoE, KAN, World Models

Key Features:
- Method Registry: Central index of all methods
- Timeline: Historical progression tracking
- Lineage Tracker: Evolution and influence relationships
- Paper Index: Landmark papers database
- Benchmark Tracker: SOTA tracking across domains

Example Usage:
    from consciousness.ml_research import MethodRegistry, Timeline, LineageTracker

    # Get all methods
    all_methods = MethodRegistry.get_all()

    # Get methods by era
    attention_methods = MethodRegistry.get_by_era(MethodEra.ATTENTION)

    # Show method lineage
    LineageTracker.show_lineage('transformer')

    # Get timeline for an era
    Timeline.show_era('deep_learning')
"""

__version__ = "1.0.0"
__author__ = "ML Research Index"

# Core components
from .core.taxonomy import (
    MethodEra,
    MethodCategory,
    MethodLineage,
    MLMethod,
    Paper,
    Benchmark,
)

from .core.method_registry import MethodRegistry
from .core.timeline import Timeline
from .core.lineage_tracker import LineageTracker
from .core.paper_index import PaperIndex
from .core.benchmark_tracker import BenchmarkTracker

# Era modules (lazy imports available)
from . import foundations
from . import classical
from . import deep_learning
from . import attention
from . import novel
from . import reinforcement
from . import optimization

__all__ = [
    # Version
    "__version__",

    # Core taxonomy
    "MethodEra",
    "MethodCategory",
    "MethodLineage",
    "MLMethod",
    "Paper",
    "Benchmark",

    # Core utilities
    "MethodRegistry",
    "Timeline",
    "LineageTracker",
    "PaperIndex",
    "BenchmarkTracker",

    # Era modules
    "foundations",
    "classical",
    "deep_learning",
    "attention",
    "novel",
    "reinforcement",
    "optimization",
]


def get_method_count() -> int:
    """Return the total number of indexed methods."""
    return len(MethodRegistry.get_all())


def get_era_summary() -> dict:
    """Return summary statistics by era."""
    return {
        era.value: len(MethodRegistry.get_by_era(era))
        for era in MethodEra
    }


def search(query: str) -> list:
    """Search for methods by name, author, or keyword."""
    return MethodRegistry.search(query)
