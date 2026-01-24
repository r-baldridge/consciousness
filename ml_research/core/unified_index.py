"""
Unified Index for ML Research Module

Cross-references historical methods, modern architectures, application techniques,
and data loaders into a single queryable interface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Any, Callable
from enum import Enum


class ComponentType(Enum):
    """Types of components in the unified index."""
    HISTORICAL_METHOD = "historical_method"
    MODERN_ARCHITECTURE = "modern_architecture"
    TECHNIQUE = "technique"
    DATA_LOADER = "data_loader"


@dataclass
class CrossReference:
    """A cross-reference between components."""
    source_type: ComponentType
    source_id: str
    target_type: ComponentType
    target_id: str
    relationship: str  # 'evolution', 'implementation', 'compatible', 'requires'
    strength: float = 1.0  # 0-1, how strong the relationship is
    notes: Optional[str] = None


@dataclass
class UnifiedEntry:
    """A unified entry that can represent any component."""
    component_type: ComponentType
    component_id: str
    name: str
    year: Optional[int] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedIndex:
    """
    Central access point for all ml_research components.

    Provides cross-referencing between:
    - Historical methods (200+ from 1943-present)
    - Modern architectures (12 state-of-the-art)
    - Application techniques (50+)
    - Data loaders (TRM, CTM, etc.)
    """

    # Singleton instance
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if UnifiedIndex._initialized:
            return

        # Entries by type
        self._entries: Dict[ComponentType, Dict[str, UnifiedEntry]] = {
            ct: {} for ct in ComponentType
        }

        # Cross-references
        self._cross_refs: List[CrossReference] = []

        # Lookup caches
        self._lineage_cache: Dict[str, List[str]] = {}
        self._technique_cache: Dict[str, List[str]] = {}

        # Data loader registry
        self._data_loaders: Dict[str, Type] = {}

        # Initialize mappings
        self._init_lineage_mappings()
        self._init_technique_mappings()
        self._init_loader_mappings()

        UnifiedIndex._initialized = True

    # =========================================================================
    # LINEAGE MAPPINGS: Historical → Modern
    # =========================================================================

    # Historical methods that evolved into modern architectures
    LINEAGE_TO_ARCHITECTURE: Dict[str, List[str]] = {
        # Transformer lineage
        'transformer': ['ctm', 'titans', 'ring_attention', 'flash_attention'],
        'attention': ['ctm', 'titans', 'ring_attention', 'flash_attention'],
        'self_attention': ['ctm', 'titans', 'ring_attention'],

        # RNN/LSTM lineage
        'lstm': ['xlstm', 'rwkv', 'griffin'],
        'gru': ['xlstm', 'rwkv'],
        'rnn': ['xlstm', 'rwkv', 'griffin', 'mamba_impl'],

        # State space lineage
        's4': ['mamba_impl', 'hyena'],
        'state_space': ['mamba_impl', 'hyena', 'griffin'],

        # Generative lineage
        'vae': ['flow_matching', 'consistency_models'],
        'diffusion': ['flow_matching', 'consistency_models'],
        'gan': ['flow_matching'],

        # Representation learning
        'autoencoder': ['jepa', 'mae'],
        'contrastive_learning': ['jepa', 'clip'],

        # Memory networks
        'memory_networks': ['titans', 'ttt'],
        'neural_turing_machine': ['titans'],
    }

    # Reverse mapping: Architecture → Historical predecessors
    ARCHITECTURE_PREDECESSORS: Dict[str, List[str]] = {
        'ctm': ['transformer', 'attention', 'neural_ode'],
        'xlstm': ['lstm', 'gru', 'highway_networks'],
        'rwkv': ['transformer', 'rnn', 'linear_attention'],
        'griffin': ['gated_linear_unit', 'lstm', 'local_attention'],
        'mamba_impl': ['s4', 'state_space', 'selective_ssm'],
        'hyena': ['s4', 'long_convolutions', 'gating'],
        'titans': ['transformer', 'memory_networks', 'ttt'],
        'ttt': ['transformer', 'meta_learning', 'online_learning'],
        'jepa': ['mae', 'contrastive_learning', 'world_models'],
        'flow_matching': ['normalizing_flows', 'diffusion', 'optimal_transport'],
        'consistency_models': ['diffusion', 'score_matching'],
        'ring_attention': ['transformer', 'flash_attention', 'sequence_parallelism'],
    }

    # =========================================================================
    # TECHNIQUE MAPPINGS: Architecture ↔ Technique
    # =========================================================================

    # Which techniques work well with which architectures
    ARCHITECTURE_TECHNIQUES: Dict[str, List[str]] = {
        'ctm': [
            'chain_of_thought', 'tree_of_thought', 'recursive_decomposition',
            'self_reflection', 'iterative_refinement', 'adaptive_computation',
        ],
        'xlstm': [
            'sequential_reasoning', 'time_series', 'streaming_inference',
            'online_learning', 'state_tracking',
        ],
        'rwkv': [
            'streaming_inference', 'long_context', 'efficient_generation',
            'continuous_learning', 'state_caching',
        ],
        'griffin': [
            'long_context', 'local_attention', 'efficient_inference',
            'document_processing', 'sliding_window',
        ],
        'mamba_impl': [
            'streaming_inference', 'long_sequences', 'efficient_training',
            'genomics', 'audio_processing', 'selective_memory',
        ],
        'hyena': [
            'very_long_sequences', 'subquadratic', 'scientific_computing',
            'geometric_deep_learning',
        ],
        'titans': [
            'continual_learning', 'test_time_adaptation', 'memory_augmented',
            'in_context_learning', 'meta_learning',
        ],
        'ttt': [
            'test_time_training', 'adaptive_inference', 'meta_learning',
            'long_context_reasoning', 'self_improvement',
        ],
        'jepa': [
            'self_supervised_learning', 'representation_learning',
            'video_understanding', 'world_modeling', 'multimodal',
        ],
        'flow_matching': [
            'fast_generation', 'image_synthesis', 'video_generation',
            'optimal_transport', 'latent_interpolation',
        ],
        'consistency_models': [
            'single_step_generation', 'real_time_synthesis',
            'interactive_generation', 'fast_sampling',
        ],
        'ring_attention': [
            'infinite_context', 'distributed_inference', 'document_qa',
            'long_form_generation', 'multi_gpu_scaling',
        ],
    }

    # Techniques and their requirements
    TECHNIQUE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
        'chain_of_thought': {
            'capabilities': ['reasoning', 'text_generation'],
            'min_context_length': 2048,
            'structured_output': False,
        },
        'tree_of_thought': {
            'capabilities': ['reasoning', 'branching', 'backtracking'],
            'min_context_length': 4096,
            'requires_sampling': True,
        },
        'tool_calling': {
            'capabilities': ['function_calling', 'structured_output'],
            'requires_schema': True,
        },
        'rag': {
            'capabilities': ['retrieval', 'context_integration'],
            'requires_embeddings': True,
            'min_context_length': 8192,
        },
        'self_reflection': {
            'capabilities': ['reasoning', 'self_evaluation'],
            'min_context_length': 2048,
        },
        'streaming_inference': {
            'capabilities': ['incremental_output', 'state_caching'],
            'latency_sensitive': True,
        },
        'long_context': {
            'min_context_length': 32768,
            'memory_efficient': True,
        },
    }

    # =========================================================================
    # DATA LOADER MAPPINGS
    # =========================================================================

    LOADER_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
        'trm': {
            'grid_dimensions': (64, 48),
            'vocab_size': 32768,
            'special_fields': ['buggy_grid', 'fixed_grid', 'diff_mask', 'buggy_mask'],
            'supports_curriculum': True,
            'config_class': 'TRM_CONFIG',
        },
        'ctm': {
            'grid_dimensions': (64, 48),
            'vocab_size': 32768,
            'special_fields': [
                'buggy_grid', 'fixed_grid', 'diff_mask', 'buggy_mask',
                'positions', 'bug_location', 'bug_location_mask',
            ],
            'supports_curriculum': True,
            'config_class': 'CTM_CONFIG',
            'additional_features': ['2D_positions', 'bug_localization'],
        },
    }

    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================

    def _init_lineage_mappings(self):
        """Initialize lineage cross-references."""
        for method, architectures in self.LINEAGE_TO_ARCHITECTURE.items():
            for arch in architectures:
                self._cross_refs.append(CrossReference(
                    source_type=ComponentType.HISTORICAL_METHOD,
                    source_id=method,
                    target_type=ComponentType.MODERN_ARCHITECTURE,
                    target_id=arch,
                    relationship='evolution',
                ))

        for arch, predecessors in self.ARCHITECTURE_PREDECESSORS.items():
            for pred in predecessors:
                self._cross_refs.append(CrossReference(
                    source_type=ComponentType.MODERN_ARCHITECTURE,
                    source_id=arch,
                    target_type=ComponentType.HISTORICAL_METHOD,
                    target_id=pred,
                    relationship='predecessor',
                ))

    def _init_technique_mappings(self):
        """Initialize technique compatibility cross-references."""
        for arch, techniques in self.ARCHITECTURE_TECHNIQUES.items():
            for technique in techniques:
                self._cross_refs.append(CrossReference(
                    source_type=ComponentType.MODERN_ARCHITECTURE,
                    source_id=arch,
                    target_type=ComponentType.TECHNIQUE,
                    target_id=technique,
                    relationship='compatible',
                ))

    def _init_loader_mappings(self):
        """Initialize data loader cross-references."""
        for loader_id, requirements in self.LOADER_REQUIREMENTS.items():
            self._cross_refs.append(CrossReference(
                source_type=ComponentType.MODERN_ARCHITECTURE,
                source_id=loader_id,
                target_type=ComponentType.DATA_LOADER,
                target_id=f"{loader_id}_loader",
                relationship='uses',
            ))

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    @classmethod
    def get_modern_implementations(cls, historical_method: str) -> List[str]:
        """
        Get modern architectures that implement or evolve from a historical method.

        Args:
            historical_method: ID of a historical method (e.g., 'transformer', 'lstm')

        Returns:
            List of architecture IDs that evolved from this method
        """
        return cls.LINEAGE_TO_ARCHITECTURE.get(historical_method, [])

    @classmethod
    def get_predecessors(cls, architecture: str) -> List[str]:
        """
        Get historical methods that an architecture builds upon.

        Args:
            architecture: ID of a modern architecture (e.g., 'ctm', 'mamba_impl')

        Returns:
            List of historical method IDs that this architecture evolved from
        """
        return cls.ARCHITECTURE_PREDECESSORS.get(architecture, [])

    @classmethod
    def get_compatible_techniques(cls, architecture: str) -> List[str]:
        """
        Get techniques that work well with an architecture.

        Args:
            architecture: ID of a modern architecture

        Returns:
            List of technique IDs compatible with this architecture
        """
        return cls.ARCHITECTURE_TECHNIQUES.get(architecture, [])

    @classmethod
    def get_architectures_for_technique(cls, technique: str) -> List[str]:
        """
        Get architectures that support a specific technique.

        Args:
            technique: ID of a technique

        Returns:
            List of architecture IDs that support this technique
        """
        result = []
        for arch, techniques in cls.ARCHITECTURE_TECHNIQUES.items():
            if technique in techniques:
                result.append(arch)
        return result

    @classmethod
    def is_compatible(cls, architecture: str, technique: str) -> bool:
        """
        Check if an architecture is compatible with a technique.

        Args:
            architecture: ID of a modern architecture
            technique: ID of a technique

        Returns:
            True if compatible, False otherwise
        """
        compatible = cls.ARCHITECTURE_TECHNIQUES.get(architecture, [])
        return technique in compatible

    @classmethod
    def get_technique_requirements(cls, technique: str) -> Dict[str, Any]:
        """
        Get requirements for a technique.

        Args:
            technique: ID of a technique

        Returns:
            Dict of requirements (capabilities, context length, etc.)
        """
        return cls.TECHNIQUE_REQUIREMENTS.get(technique, {})

    @classmethod
    def get_loader_requirements(cls, architecture: str) -> Dict[str, Any]:
        """
        Get data loader requirements for an architecture.

        Args:
            architecture: ID of a modern architecture

        Returns:
            Dict of loader requirements (grid size, special fields, etc.)
        """
        return cls.LOADER_REQUIREMENTS.get(architecture, {})

    @classmethod
    def filter_compatible(
        cls,
        architecture: str,
        techniques: List[str]
    ) -> List[str]:
        """
        Filter techniques to only those compatible with an architecture.

        Args:
            architecture: ID of a modern architecture
            techniques: List of technique IDs to filter

        Returns:
            Filtered list of compatible technique IDs
        """
        compatible = set(cls.ARCHITECTURE_TECHNIQUES.get(architecture, []))
        return [t for t in techniques if t in compatible]

    @classmethod
    def trace_lineage(
        cls,
        from_method: str,
        to_architecture: str
    ) -> Optional[List[str]]:
        """
        Trace the evolution path from a historical method to a modern architecture.

        Args:
            from_method: ID of a historical method
            to_architecture: ID of a modern architecture

        Returns:
            List of method/architecture IDs forming the path, or None if no path
        """
        # Direct connection
        if to_architecture in cls.LINEAGE_TO_ARCHITECTURE.get(from_method, []):
            return [from_method, to_architecture]

        # Check through predecessors
        predecessors = cls.ARCHITECTURE_PREDECESSORS.get(to_architecture, [])
        if from_method in predecessors:
            return [from_method, to_architecture]

        # Simple path finding (could be enhanced with BFS for longer paths)
        for pred in predecessors:
            if pred in cls.LINEAGE_TO_ARCHITECTURE:
                for intermediate in cls.LINEAGE_TO_ARCHITECTURE[pred]:
                    if intermediate == to_architecture:
                        return [from_method, pred, to_architecture]

        return None

    # =========================================================================
    # DATA LOADER REGISTRY
    # =========================================================================

    def register_loader(self, architecture: str, loader_class: Type):
        """
        Register a data loader for an architecture.

        Args:
            architecture: ID of the architecture
            loader_class: The data loader class
        """
        self._data_loaders[architecture] = loader_class

    def get_data_loader(self, architecture: str) -> Optional[Type]:
        """
        Get the data loader class for an architecture.

        Args:
            architecture: ID of the architecture

        Returns:
            Data loader class, or None if not registered
        """
        return self._data_loaders.get(architecture)

    # =========================================================================
    # SUMMARY METHODS
    # =========================================================================

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get a summary of all indexed components."""
        return {
            'lineage_mappings': len(cls.LINEAGE_TO_ARCHITECTURE),
            'architectures_with_predecessors': len(cls.ARCHITECTURE_PREDECESSORS),
            'architecture_technique_mappings': len(cls.ARCHITECTURE_TECHNIQUES),
            'technique_requirements': len(cls.TECHNIQUE_REQUIREMENTS),
            'loader_requirements': len(cls.LOADER_REQUIREMENTS),
            'total_architectures': len(set(cls.ARCHITECTURE_TECHNIQUES.keys())),
            'unique_techniques': len(set(
                t for techniques in cls.ARCHITECTURE_TECHNIQUES.values()
                for t in techniques
            )),
        }

    @classmethod
    def list_all_architectures(cls) -> List[str]:
        """List all modern architectures in the index."""
        return list(cls.ARCHITECTURE_TECHNIQUES.keys())

    @classmethod
    def list_all_techniques(cls) -> List[str]:
        """List all unique techniques in the index."""
        techniques = set()
        for tech_list in cls.ARCHITECTURE_TECHNIQUES.values():
            techniques.update(tech_list)
        return sorted(list(techniques))


# Module-level convenience functions
def get_modern_implementations(method: str) -> List[str]:
    """Get modern architectures implementing a historical method."""
    return UnifiedIndex.get_modern_implementations(method)


def get_compatible_techniques(architecture: str) -> List[str]:
    """Get techniques compatible with an architecture."""
    return UnifiedIndex.get_compatible_techniques(architecture)


def get_architectures_for_technique(technique: str) -> List[str]:
    """Get architectures that support a technique."""
    return UnifiedIndex.get_architectures_for_technique(technique)


def is_compatible(architecture: str, technique: str) -> bool:
    """Check architecture-technique compatibility."""
    return UnifiedIndex.is_compatible(architecture, technique)


def trace_lineage(from_method: str, to_architecture: str) -> Optional[List[str]]:
    """Trace evolution from historical method to modern architecture."""
    return UnifiedIndex.trace_lineage(from_method, to_architecture)
