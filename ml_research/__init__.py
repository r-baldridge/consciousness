"""
ML Research Module - Comprehensive Machine Learning History, Architectures, and Techniques

This module provides a unified interface for:
1. Historical ML methods (200+ from 1943-present)
2. Modern architectures (12 state-of-the-art implementations)
3. Application techniques (50+ composable patterns)
4. Shared infrastructure (unified data pipeline)

=============================================================================
COVERAGE
=============================================================================

Historical Methods (200+):
- Era 1: Foundational (1943-1980) - McCulloch-Pitts to Hopfield
- Era 2: Classical (1980-2006) - MLP, CNN, LSTM, SVM, Ensemble methods
- Era 3: Deep Learning (2006-2017) - AlexNet, ResNet, GAN, VAE
- Era 4: Attention (2017-present) - Transformer, BERT, GPT, LLaMA
- Era 5: Novel (2023+) - Mamba, MoE, KAN, World Models

Modern Architectures (12):
- Tier 1: CTM, JEPA, xLSTM, RWKV, Griffin, Mamba
- Tier 2: TTT, Hyena, Consistency Models, Ring Attention, Flow Matching, Titans

Application Techniques (50+):
- Decomposition, Prompting, Agentic, Memory, Code Synthesis
- Orchestration, Verification, Optimization

=============================================================================
KEY FEATURES
=============================================================================

- Unified Index: Cross-references historical → modern → techniques
- Method Registry: Central index of all historical methods
- Architecture Registry: Modern implementations with capabilities
- Technique Registry: Composable application patterns
- Orchestrator: Task routing across architectures
- Timeline: Historical progression tracking
- Lineage Tracker: Evolution and influence relationships
- Data Pipeline: Unified format for training data

=============================================================================
EXAMPLE USAGE
=============================================================================

    from consciousness.ml_research import (
        # Historical
        MethodRegistry, Timeline, LineageTracker,
        # Modern
        Orchestrator, ARCHITECTURES, run_task,
        # Techniques
        TECHNIQUE_INDEX, compose,
        # Unified
        UnifiedIndex, get_compatible_techniques,
    )

    # Get modern implementations of a historical method
    modern = UnifiedIndex.get_modern_implementations('transformer')

    # Get compatible techniques for an architecture
    techniques = get_compatible_techniques('ctm')

    # Run a task with automatic architecture selection
    result = run_task(TaskSpec(task_type=TaskType.CODE_REPAIR, ...))
"""

__version__ = "1.1.0"
__author__ = "ML Research Index"

# =============================================================================
# CORE COMPONENTS (Historical Research Index)
# =============================================================================

from .core.taxonomy import (
    MethodEra,
    MethodCategory,
    MethodLineage,
    MLMethod,
    Paper,
    Benchmark,
)

from .core.method_registry import MethodRegistry, get_global_registry
from .core.timeline import Timeline
from .core.lineage_tracker import LineageTracker
from .core.paper_index import PaperIndex
from .core.benchmark_tracker import BenchmarkTracker

# Unified Index (cross-references all components)
from .core.unified_index import (
    UnifiedIndex,
    ComponentType,
    CrossReference,
    get_modern_implementations,
    get_compatible_techniques,
    get_architectures_for_technique,
    is_compatible,
    trace_lineage,
)

# =============================================================================
# ERA MODULES (Historical Methods)
# =============================================================================

from . import foundations
from . import classical
from . import deep_learning
from . import attention
from . import novel
from . import reinforcement
from . import optimization

# =============================================================================
# MODERN ARCHITECTURES (2023+)
# =============================================================================

from . import modern_dev
from .modern_dev import (
    # Enums
    DevelopmentStatus,
    ImplementationTier,
    TaskType,
    # Data classes
    ArchitectureIndex,
    TaskSpec,
    TaskResult,
    # Base classes
    ArchitectureBase,
    StubArchitecture,
    # Orchestrator
    Orchestrator,
    ARCHITECTURE_CAPABILITIES,
    # Registry
    ARCHITECTURES,
    ARCHITECTURE_BY_ID,
    # Functions
    get_architecture_info,
    list_architectures,
    get_by_status,
    run_task,
)

# =============================================================================
# APPLICATION TECHNIQUES (50+)
# =============================================================================

from . import ml_techniques
from .ml_techniques import (
    # Enums
    TechniqueCategory,
    CompositionMode,
    ExecutionStatus,
    # Data classes
    TechniqueResult,
    TechniqueConfig,
    TechniqueIndex,
    # Base classes
    TechniqueBase,
    Pipeline,
    ParallelComposition,
    # Registry
    TECHNIQUES as TECHNIQUE_INDEX,  # Alias for consistency
    TECHNIQUE_BY_ID,
    TECHNIQUES_BY_CATEGORY,
    # Functions
    get_technique_info,
    list_techniques,
    get_composable_with,
    compose,
    load_config,
)

# =============================================================================
# SHARED INFRASTRUCTURE (from modern_dev.shared)
# =============================================================================

from .modern_dev import shared
from .modern_dev.shared import (
    # Data schema
    CanonicalCodeSample,
    QualityTier,
    # Data loaders
    TRMDataLoader,
    TRMDataset,
    TRMSample,
    CTMDataLoader,
    CTMDataset,
    CTMSample,
    # Bug taxonomy
    BugType,
    BugCategory,
    BugInfo,
    BUG_TAXONOMY,
)
from .modern_dev.shared.data import BaseDataLoader, CURRICULUM_STAGES
from .modern_dev.shared.data.loaders.trm import GridEncoder
from .modern_dev.shared.taxonomy.bug_types import (
    get_bug_info,
    get_bugs_by_category,
    get_bugs_by_difficulty,
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # -------------------------------------------------------------------------
    # Core Taxonomy (Historical)
    # -------------------------------------------------------------------------
    "MethodEra",
    "MethodCategory",
    "MethodLineage",
    "MLMethod",
    "Paper",
    "Benchmark",

    # Core Utilities (Historical)
    "MethodRegistry",
    "get_global_registry",
    "Timeline",
    "LineageTracker",
    "PaperIndex",
    "BenchmarkTracker",

    # Era Modules
    "foundations",
    "classical",
    "deep_learning",
    "attention",
    "novel",
    "reinforcement",
    "optimization",

    # -------------------------------------------------------------------------
    # Unified Index (Cross-references)
    # -------------------------------------------------------------------------
    "UnifiedIndex",
    "ComponentType",
    "CrossReference",
    "get_modern_implementations",
    "get_compatible_techniques",
    "get_architectures_for_technique",
    "is_compatible",
    "trace_lineage",

    # -------------------------------------------------------------------------
    # Modern Architectures
    # -------------------------------------------------------------------------
    "modern_dev",
    "DevelopmentStatus",
    "ImplementationTier",
    "TaskType",
    "ArchitectureIndex",
    "TaskSpec",
    "TaskResult",
    "ArchitectureBase",
    "StubArchitecture",
    "Orchestrator",
    "ARCHITECTURE_CAPABILITIES",
    "ARCHITECTURES",
    "ARCHITECTURE_BY_ID",
    "get_architecture_info",
    "list_architectures",
    "get_by_status",
    "run_task",

    # -------------------------------------------------------------------------
    # Application Techniques
    # -------------------------------------------------------------------------
    "ml_techniques",
    "TechniqueCategory",
    "CompositionMode",
    "ExecutionStatus",
    "TechniqueResult",
    "TechniqueConfig",
    "TechniqueIndex",
    "TechniqueBase",
    "Pipeline",
    "ParallelComposition",
    "TECHNIQUE_INDEX",
    "TECHNIQUE_BY_ID",
    "TECHNIQUES_BY_CATEGORY",
    "get_technique_info",
    "list_techniques",
    "get_composable_with",
    "compose",
    "load_config",

    # -------------------------------------------------------------------------
    # Shared Infrastructure (from modern_dev.shared)
    # -------------------------------------------------------------------------
    "shared",
    "CanonicalCodeSample",
    "QualityTier",
    "BaseDataLoader",
    "TRMDataLoader",
    "TRMDataset",
    "CTMDataLoader",
    "CTMDataset",
    "TRMSample",
    "CTMSample",
    "GridEncoder",
    "CURRICULUM_STAGES",
    "BugType",
    "BugCategory",
    "BugInfo",
    "BUG_TAXONOMY",
    "get_bug_info",
    "get_bugs_by_category",
    "get_bugs_by_difficulty",
]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_method_count() -> int:
    """Return the total number of indexed historical methods."""
    registry = get_global_registry()
    return len(registry.get_all())


def get_architecture_count() -> int:
    """Return the number of modern architectures indexed."""
    return len(ARCHITECTURES)


def get_technique_count() -> int:
    """Return the number of application techniques indexed."""
    return len(TECHNIQUE_INDEX)


def get_era_summary() -> dict:
    """Return summary statistics by era."""
    registry = get_global_registry()
    return {
        era.value: len(registry.get_by_era(era))
        for era in MethodEra
    }


def get_full_summary() -> dict:
    """Return complete summary of all indexed components."""
    return {
        'historical_methods': get_method_count(),
        'modern_architectures': get_architecture_count(),
        'application_techniques': get_technique_count(),
        'era_breakdown': get_era_summary(),
        'unified_index': UnifiedIndex.get_summary(),
    }


def search(query: str) -> list:
    """Search for methods by name, author, or keyword."""
    registry = get_global_registry()
    return registry.search(query)
