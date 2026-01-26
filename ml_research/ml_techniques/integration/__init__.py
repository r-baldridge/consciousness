"""
Architecture-Technique Integration Layer

Provides a unified interface for connecting ml_techniques to modern_dev architectures.
This module enables techniques to leverage architecture-specific capabilities while
maintaining fallback behavior when architectures are unavailable.

=============================================================================
ARCHITECTURE-TECHNIQUE MAPPING
=============================================================================

TRM (Tiny Recursive Model):
    - recursive_decomposition: TRM iterations for decomposition steps
    - chain_of_thought: Recursive reasoning through TRM layers
    - code_repair: TRM's code fixing capabilities

CTM (Continuous Thought Machine):
    - temporal_reasoning: Neural synchronization for time-aware reasoning
    - verification: Confidence scoring via activity patterns
    - memory_patterns: Neuron dynamics for memory

Mamba (Selective SSM):
    - long_context_rag: O(N) attention for large context retrieval
    - streaming_inference: Token-by-token processing
    - compression: State-space context compression

=============================================================================
USAGE
=============================================================================

# Get compatible techniques for an architecture
from ml_techniques.integration import get_techniques_for_architecture
techniques = get_techniques_for_architecture('trm')

# Get the best architecture for a technique
from ml_techniques.integration import get_architecture_for_technique
arch = get_architecture_for_technique('recursive_decomposition')

# Create an integrated technique
from ml_techniques.integration import create_integrated_technique
technique = create_integrated_technique('trm_decomposer', 'trm')
result = technique.run(input_data)

# Import specific integrated techniques
from ml_techniques.integration.trm_techniques import TRMDecomposer
from ml_techniques.integration.ctm_techniques import CTMTemporalReasoning
from ml_techniques.integration.mamba_techniques import MambaRAG
"""

from typing import Any, Dict, List, Optional, Type

from .. import TechniqueBase, TechniqueCategory


# =============================================================================
# ARCHITECTURE-TECHNIQUE REGISTRY
# =============================================================================

ARCHITECTURE_TECHNIQUE_MAP: Dict[str, List[str]] = {
    'trm': [
        'recursive_decomposition',
        'chain_of_thought',
        'code_repair',
        'self_debugging',
        'iterative_refinement',
    ],
    'ctm': [
        'temporal_reasoning',
        'verification',
        'memory_patterns',
        'neural_synchronization',
        'adaptive_computation',
        'self_evaluation',
    ],
    'mamba': [
        'long_context_rag',
        'streaming_inference',
        'compression',
        'genomics',
        'audio_processing',
        'selective_memory',
    ],
}

# Reverse mapping: technique -> preferred architecture
TECHNIQUE_ARCHITECTURE_MAP: Dict[str, str] = {}
for arch, techniques in ARCHITECTURE_TECHNIQUE_MAP.items():
    for technique in techniques:
        # First architecture wins as "preferred"
        if technique not in TECHNIQUE_ARCHITECTURE_MAP:
            TECHNIQUE_ARCHITECTURE_MAP[technique] = arch

# Technique class registry (populated by submodule imports)
_INTEGRATED_TECHNIQUES: Dict[str, Type[TechniqueBase]] = {}


# =============================================================================
# REGISTRY FUNCTIONS
# =============================================================================

def get_techniques_for_architecture(arch_id: str) -> List[str]:
    """
    Get the list of techniques compatible with an architecture.

    Args:
        arch_id: Architecture identifier ('trm', 'ctm', 'mamba')

    Returns:
        List of technique IDs compatible with this architecture

    Example:
        >>> get_techniques_for_architecture('trm')
        ['recursive_decomposition', 'chain_of_thought', 'code_repair', ...]
    """
    return ARCHITECTURE_TECHNIQUE_MAP.get(arch_id, [])


def get_architecture_for_technique(technique_id: str) -> Optional[str]:
    """
    Get the preferred architecture for a technique.

    Args:
        technique_id: Technique identifier

    Returns:
        Architecture ID that best supports this technique, or None

    Example:
        >>> get_architecture_for_technique('recursive_decomposition')
        'trm'
    """
    return TECHNIQUE_ARCHITECTURE_MAP.get(technique_id)


def get_all_architectures() -> List[str]:
    """
    Get all architecture IDs with registered techniques.

    Returns:
        List of architecture IDs
    """
    return list(ARCHITECTURE_TECHNIQUE_MAP.keys())


def get_all_integrated_techniques() -> List[str]:
    """
    Get all technique IDs that have architecture integrations.

    Returns:
        List of technique IDs
    """
    return list(TECHNIQUE_ARCHITECTURE_MAP.keys())


def is_architecture_compatible(arch_id: str, technique_id: str) -> bool:
    """
    Check if an architecture supports a technique.

    Args:
        arch_id: Architecture identifier
        technique_id: Technique identifier

    Returns:
        True if compatible, False otherwise
    """
    return technique_id in ARCHITECTURE_TECHNIQUE_MAP.get(arch_id, [])


# =============================================================================
# TECHNIQUE CREATION
# =============================================================================

def register_integrated_technique(
    technique_id: str,
    technique_class: Type[TechniqueBase],
) -> None:
    """
    Register an integrated technique class.

    Args:
        technique_id: Unique identifier for the technique
        technique_class: The technique class to register
    """
    _INTEGRATED_TECHNIQUES[technique_id] = technique_class


def get_integrated_technique_class(technique_id: str) -> Optional[Type[TechniqueBase]]:
    """
    Get the class for an integrated technique.

    Args:
        technique_id: Technique identifier

    Returns:
        Technique class or None if not found
    """
    return _INTEGRATED_TECHNIQUES.get(technique_id)


def create_integrated_technique(
    technique_id: str,
    arch_id: str,
    **kwargs: Any,
) -> TechniqueBase:
    """
    Create an integrated technique instance.

    This factory function creates a technique that leverages a specific
    architecture's capabilities. If the architecture is unavailable,
    the technique will fall back to base implementation.

    Args:
        technique_id: Identifier for the integrated technique
        arch_id: Architecture to integrate with
        **kwargs: Additional arguments passed to the technique constructor

    Returns:
        An instance of the integrated technique

    Raises:
        ValueError: If technique_id is not registered

    Example:
        >>> technique = create_integrated_technique('trm_decomposer', 'trm', max_iterations=10)
        >>> result = technique.run("Complex task to decompose")
    """
    technique_class = _INTEGRATED_TECHNIQUES.get(technique_id)

    if technique_class is None:
        raise ValueError(
            f"Unknown integrated technique: '{technique_id}'. "
            f"Available techniques: {list(_INTEGRATED_TECHNIQUES.keys())}"
        )

    return technique_class(**kwargs)


def list_integrated_techniques() -> Dict[str, Dict[str, Any]]:
    """
    List all registered integrated techniques with their metadata.

    Returns:
        Dictionary mapping technique_id to metadata
    """
    result = {}
    for technique_id, technique_class in _INTEGRATED_TECHNIQUES.items():
        result[technique_id] = {
            "class": technique_class.__name__,
            "technique_id": getattr(technique_class, 'TECHNIQUE_ID', technique_id),
            "category": getattr(technique_class, 'CATEGORY', TechniqueCategory.ORCHESTRATION),
            "architecture": get_architecture_for_technique(technique_id),
        }
    return result


# =============================================================================
# ARCHITECTURE AVAILABILITY CHECK
# =============================================================================

def check_architecture_available(arch_id: str) -> bool:
    """
    Check if an architecture module is available for import.

    Args:
        arch_id: Architecture identifier

    Returns:
        True if architecture can be imported, False otherwise
    """
    arch_modules = {
        'trm': 'modern_dev.trm',
        'ctm': 'modern_dev.ctm',
        'mamba': 'modern_dev.mamba_impl',
    }

    module_name = arch_modules.get(arch_id)
    if not module_name:
        return False

    try:
        __import__(module_name, fromlist=[''])
        return True
    except ImportError:
        return False


def get_architecture_status() -> Dict[str, bool]:
    """
    Get availability status for all architectures.

    Returns:
        Dictionary mapping arch_id to availability boolean
    """
    return {
        arch_id: check_architecture_available(arch_id)
        for arch_id in ARCHITECTURE_TECHNIQUE_MAP.keys()
    }


# =============================================================================
# IMPORTS - Load submodules and register techniques
# =============================================================================

# Import technique modules to register their classes
try:
    from . import trm_techniques
    from .trm_techniques import (
        TRMDecomposer,
        TRMChainOfThought,
        TRMCodeRepair,
    )
except ImportError:
    TRMDecomposer = None
    TRMChainOfThought = None
    TRMCodeRepair = None

try:
    from . import ctm_techniques
    from .ctm_techniques import (
        CTMTemporalReasoning,
        CTMMemory,
        CTMVerification,
    )
except ImportError:
    CTMTemporalReasoning = None
    CTMMemory = None
    CTMVerification = None

try:
    from . import mamba_techniques
    from .mamba_techniques import (
        MambaRAG,
        MambaStreaming,
        MambaCompression,
    )
except ImportError:
    MambaRAG = None
    MambaStreaming = None
    MambaCompression = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Registry data
    "ARCHITECTURE_TECHNIQUE_MAP",
    "TECHNIQUE_ARCHITECTURE_MAP",
    # Registry functions
    "get_techniques_for_architecture",
    "get_architecture_for_technique",
    "get_all_architectures",
    "get_all_integrated_techniques",
    "is_architecture_compatible",
    # Technique creation
    "register_integrated_technique",
    "get_integrated_technique_class",
    "create_integrated_technique",
    "list_integrated_techniques",
    # Architecture availability
    "check_architecture_available",
    "get_architecture_status",
    # TRM techniques
    "TRMDecomposer",
    "TRMChainOfThought",
    "TRMCodeRepair",
    # CTM techniques
    "CTMTemporalReasoning",
    "CTMMemory",
    "CTMVerification",
    # Mamba techniques
    "MambaRAG",
    "MambaStreaming",
    "MambaCompression",
]
