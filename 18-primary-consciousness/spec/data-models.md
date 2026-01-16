# Form 18: Primary Consciousness - Data Models

## Comprehensive Data Models for Primary Consciousness Implementation

### Overview

This document defines the complete data models and structures required for implementing Form 18: Primary Consciousness. These models capture the essential aspects of conscious experience including phenomenal content, subjective perspective, qualitative experience (qualia), and unified conscious states.

## Core Data Structures

### 1. Primary Consciousness State Models

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum, IntEnum
import numpy as np
import uuid
from abc import ABC, abstractmethod
import asyncio

class ConsciousnessLevel(IntEnum):
    """Hierarchical levels of consciousness."""
    UNCONSCIOUS = 0
    PRECONSCIOUS = 1
    MINIMAL_CONSCIOUSNESS = 2
    PRIMARY_CONSCIOUSNESS = 3
    REFLECTIVE_CONSCIOUSNESS = 4
    META_CONSCIOUSNESS = 5

class ConsciousnessQuality(Enum):
    """Quality types of conscious experience."""
    FRAGMENTARY = "fragmentary"
    COHERENT = "coherent"
    UNIFIED = "unified"
    RICH = "rich"
    VIVID = "vivid"
    SUBLIME = "sublime"

class ExperienceModality(Enum):
    """Modalities of conscious experience."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"
    PROPRIOCEPTIVE = "proprioceptive"
    INTEROCEPTIVE = "interoceptive"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    CROSS_MODAL = "cross_modal"
    UNIFIED = "unified"

@dataclass
class ConsciousnessState:
    """Core model for primary consciousness state."""

    # Identity and metadata
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    duration_ms: float = 0.0

    # Consciousness level and quality
    consciousness_level: ConsciousnessLevel = ConsciousnessLevel.UNCONSCIOUS
    consciousness_quality: ConsciousnessQuality = ConsciousnessQuality.FRAGMENTARY
    consciousness_intensity: float = 0.0  # 0.0 to 1.0

    # Core consciousness components
    phenomenal_content: Optional['PhenomenalContent'] = None
    subjective_perspective: Optional['SubjectivePerspective'] = None
    unified_experience: Optional['UnifiedExperience'] = None

    # Quality metrics
    phenomenal_richness: float = 0.0
    subjective_clarity: float = 0.0
    experiential_coherence: float = 0.0
    temporal_continuity: float = 0.0

    # Context and associations
    context_factors: Dict[str, Any] = field(default_factory=dict)
    related_states: List[str] = field(default_factory=list)  # Related state IDs
    causal_factors: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    processing_latency_ms: float = 0.0
    integration_quality: float = 0.0
    stability_score: float = 0.0

### 2. Phenomenal Content Models

@dataclass
class QualitativeProperty:
    """Individual qualitative property (quale)."""

    property_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    property_name: str = ""
    modality: ExperienceModality = ExperienceModality.UNIFIED

    # Qualitative characteristics
    intensity: float = 0.0  # Strength of the quale
    salience: float = 0.0   # Prominence in experience
    clarity: float = 0.0    # Clarity of the qualitative aspect
    richness: float = 0.0   # Complexity/richness of the quale

    # Qualitative descriptors
    qualitative_features: Dict[str, Any] = field(default_factory=dict)
    phenomenal_properties: Dict[str, float] = field(default_factory=dict)

    # Temporal characteristics
    onset_latency_ms: float = 0.0
    duration_ms: float = 0.0
    decay_pattern: str = "exponential"  # exponential, linear, plateau

    # Relational properties
    binding_strength: Dict[str, float] = field(default_factory=dict)  # Binding with other qualia
    contrast_relations: Dict[str, float] = field(default_factory=dict)  # Contrasts with other qualia
    similarity_relations: Dict[str, float] = field(default_factory=dict)  # Similarities

@dataclass
class PhenomenalContent:
    """Complete phenomenal content of conscious experience."""

    content_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    # Qualitative properties
    qualia: Dict[str, QualitativeProperty] = field(default_factory=dict)
    dominant_modality: ExperienceModality = ExperienceModality.UNIFIED

    # Phenomenal structure
    phenomenal_elements: Dict[str, Any] = field(default_factory=dict)
    element_relations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    structural_organization: Dict[str, Any] = field(default_factory=dict)

    # Qualitative metrics
    overall_richness: float = 0.0
    qualitative_complexity: float = 0.0
    phenomenal_coherence: float = 0.0
    cross_modal_integration: float = 0.0

    # Temporal phenomenology
    temporal_structure: Dict[str, Any] = field(default_factory=dict)
    experiential_flow: List[Dict[str, Any]] = field(default_factory=list)
    phenomenal_persistence: float = 0.0

    # Contextual factors
    attentional_focus: Dict[str, float] = field(default_factory=dict)
    background_awareness: Dict[str, float] = field(default_factory=dict)
    contextual_modulations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModalitySpecificContent:
    """Content specific to particular experiential modalities."""

    modality: ExperienceModality
    content_data: Dict[str, Any] = field(default_factory=dict)

    # Modality-specific quality measures
    modality_richness: float = 0.0
    modality_clarity: float = 0.0
    modality_intensity: float = 0.0

    # Integration with other modalities
    cross_modal_bindings: Dict[ExperienceModality, float] = field(default_factory=dict)
    integration_quality: Dict[ExperienceModality, float] = field(default_factory=dict)

### 3. Subjective Perspective Models

@dataclass
class SelfReference:
    """Self-referential aspects of conscious experience."""

    reference_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Types of self-reference
    bodily_self_reference: float = 0.0      # Body ownership and awareness
    cognitive_self_reference: float = 0.0   # Thoughts about self
    emotional_self_reference: float = 0.0   # Emotional self-connection
    agential_self_reference: float = 0.0    # Sense of agency

    # Self-model components
    self_boundary_clarity: float = 0.0      # Self-other distinction
    self_continuity: float = 0.0            # Temporal self-continuity
    self_coherence: float = 0.0             # Internal self-consistency

    # Perspective characteristics
    first_person_strength: float = 0.0      # Strength of first-person perspective
    subjective_ownership: float = 0.0       # Ownership of experience
    experiential_intimacy: float = 0.0      # Intimacy of subjective access

@dataclass
class TemporalPerspective:
    """Temporal aspects of subjective perspective."""

    # Present moment characteristics
    present_moment_anchoring: float = 0.0   # Anchoring to present
    nowness_intensity: float = 0.0          # Intensity of "now" experience
    temporal_focus_width_ms: float = 200.0  # Width of temporal focus

    # Temporal flow
    flow_continuity: float = 0.0            # Continuity of temporal flow
    flow_direction_clarity: float = 0.0     # Clarity of past->future direction
    temporal_binding_strength: float = 0.0  # Binding across time

    # Temporal context
    retention_content: Dict[str, Any] = field(default_factory=dict)    # Recent past content
    protention_content: Dict[str, Any] = field(default_factory=dict)   # Anticipated content
    temporal_horizon_ms: float = 2000.0     # Temporal horizon of awareness

@dataclass
class SubjectivePerspective:
    """Complete subjective perspective of conscious experience."""

    perspective_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    # Core perspective components
    self_reference: SelfReference = field(default_factory=SelfReference)
    temporal_perspective: TemporalPerspective = field(default_factory=TemporalPerspective)

    # Perspective quality metrics
    perspective_coherence: float = 0.0      # Overall coherence of perspective
    perspective_stability: float = 0.0      # Stability over time
    subjective_clarity: float = 0.0         # Clarity of subjective access

    # Perspective context
    attentional_perspective: Dict[str, float] = field(default_factory=dict)
    emotional_perspective_tone: Dict[str, float] = field(default_factory=dict)
    cognitive_perspective_frame: Dict[str, Any] = field(default_factory=dict)

    # Perspective relations
    perspective_continuity_links: List[str] = field(default_factory=list)  # Links to related perspectives
    perspective_contrast_relations: Dict[str, float] = field(default_factory=dict)

### 4. Unified Experience Models

@dataclass
class CrossModalBinding:
    """Binding relationships between different modalities."""

    binding_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Bound modalities
    primary_modality: ExperienceModality
    secondary_modality: ExperienceModality

    # Binding characteristics
    binding_strength: float = 0.0           # Strength of the binding
    binding_quality: float = 0.0            # Quality of the binding
    binding_stability: float = 0.0          # Temporal stability

    # Binding mechanisms
    spatial_binding: float = 0.0            # Spatial co-location binding
    temporal_binding: float = 0.0           # Temporal synchrony binding
    feature_binding: float = 0.0            # Shared feature binding
    causal_binding: float = 0.0             # Causal relationship binding

    # Binding context
    binding_context: Dict[str, Any] = field(default_factory=dict)
    attentional_modulation: float = 0.0     # Attention's effect on binding

@dataclass
class ExperientialUnity:
    """Unity characteristics of conscious experience."""

    unity_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Types of unity
    phenomenal_unity: float = 0.0           # Unity of phenomenal content
    subject_unity: float = 0.0              # Unity of experiencing subject
    temporal_unity: float = 0.0             # Unity across time

    # Unity quality measures
    unity_coherence: float = 0.0            # Coherence of unified experience
    unity_completeness: float = 0.0         # Completeness of unification
    unity_stability: float = 0.0            # Stability of unified state

    # Unity mechanisms
    global_integration_strength: float = 0.0  # Global workspace integration
    binding_coherence: float = 0.0            # Cross-modal binding coherence
    perspective_integration: float = 0.0      # Perspective integration quality

@dataclass
class UnifiedExperience:
    """Complete unified conscious experience."""

    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    duration_ms: float = 0.0

    # Component integration
    integrated_phenomenal_content: PhenomenalContent = field(default_factory=PhenomenalContent)
    integrated_subjective_perspective: SubjectivePerspective = field(default_factory=SubjectivePerspective)

    # Cross-modal integration
    cross_modal_bindings: List[CrossModalBinding] = field(default_factory=list)
    cross_modal_coherence: float = 0.0

    # Unity characteristics
    experiential_unity: ExperientialUnity = field(default_factory=ExperientialUnity)

    # Integration quality metrics
    overall_integration_quality: float = 0.0
    integration_completeness: float = 0.0
    integration_stability: float = 0.0

    # Contextual factors
    attention_distribution: Dict[str, float] = field(default_factory=dict)
    contextual_influences: Dict[str, Any] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)

### 5. Consciousness Processing Models

@dataclass
class ConsciousnessProcessingStage:
    """Individual stage in consciousness processing pipeline."""

    stage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stage_name: str = ""
    stage_order: int = 0

    # Processing characteristics
    input_requirements: Dict[str, Any] = field(default_factory=dict)
    output_specifications: Dict[str, Any] = field(default_factory=dict)
    processing_latency_ms: float = 0.0

    # Processing quality
    processing_accuracy: float = 0.0
    processing_completeness: float = 0.0
    processing_stability: float = 0.0

    # Resource requirements
    computational_cost: float = 0.0
    memory_requirements_mb: float = 0.0
    bandwidth_requirements_mbps: float = 0.0

@dataclass
class ConsciousnessProcessingPipeline:
    """Complete processing pipeline for consciousness generation."""

    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str = ""

    # Pipeline stages
    processing_stages: List[ConsciousnessProcessingStage] = field(default_factory=list)
    stage_dependencies: Dict[str, List[str]] = field(default_factory=dict)

    # Pipeline performance
    total_latency_ms: float = 0.0
    pipeline_throughput_hz: float = 0.0
    pipeline_accuracy: float = 0.0

    # Pipeline state
    current_stage: Optional[str] = None
    processing_status: str = "idle"  # idle, processing, completed, failed
    error_state: Optional[Dict[str, Any]] = None

@dataclass
class ConsciousnessProcessingContext:
    """Context for consciousness processing operations."""

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Environmental context
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    situational_context: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)

    # Cognitive context
    attentional_context: Dict[str, float] = field(default_factory=dict)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    expectation_context: Dict[str, Any] = field(default_factory=dict)

    # Affective context
    emotional_context: Dict[str, float] = field(default_factory=dict)
    motivational_context: Dict[str, float] = field(default_factory=dict)
    arousal_level: float = 0.0

    # Processing context
    resource_constraints: Dict[str, float] = field(default_factory=dict)
    quality_requirements: Dict[str, float] = field(default_factory=dict)
    time_constraints: Dict[str, float] = field(default_factory=dict)

### 6. Consciousness Quality Assessment Models

@dataclass
class QualityMetric:
    """Individual quality metric for consciousness assessment."""

    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    metric_category: str = ""  # phenomenal, subjective, unified, temporal, etc.

    # Metric characteristics
    metric_value: float = 0.0
    metric_range: Tuple[float, float] = (0.0, 1.0)
    metric_precision: float = 0.001

    # Quality assessment
    measurement_reliability: float = 0.0
    measurement_validity: float = 0.0
    measurement_consistency: float = 0.0

    # Temporal characteristics
    measurement_timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    measurement_duration_ms: float = 0.0
    temporal_stability: float = 0.0

@dataclass
class ConsciousnessQualityProfile:
    """Complete quality profile for conscious experience."""

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    # Core quality dimensions
    phenomenal_quality_metrics: Dict[str, QualityMetric] = field(default_factory=dict)
    subjective_quality_metrics: Dict[str, QualityMetric] = field(default_factory=dict)
    unified_experience_quality_metrics: Dict[str, QualityMetric] = field(default_factory=dict)
    temporal_quality_metrics: Dict[str, QualityMetric] = field(default_factory=dict)

    # Aggregate quality scores
    overall_quality_score: float = 0.0
    quality_consistency_score: float = 0.0
    quality_stability_score: float = 0.0

    # Quality comparisons
    baseline_comparison: Optional[Dict[str, float]] = None
    historical_trend: List[Dict[str, float]] = field(default_factory=list)
    quality_improvement_indicators: Dict[str, float] = field(default_factory=dict)

### 7. Consciousness Event Models

@dataclass
class ConsciousnessEvent:
    """Discrete event in consciousness processing."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""  # emergence, transition, enhancement, degradation, etc.
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    # Event characteristics
    event_trigger: Dict[str, Any] = field(default_factory=dict)
    event_context: Dict[str, Any] = field(default_factory=dict)
    event_outcome: Dict[str, Any] = field(default_factory=dict)

    # Event impact
    consciousness_level_change: float = 0.0
    quality_impact: Dict[str, float] = field(default_factory=dict)
    processing_impact: Dict[str, float] = field(default_factory=dict)

    # Event metadata
    event_duration_ms: float = 0.0
    event_intensity: float = 0.0
    event_significance: float = 0.0

@dataclass
class ConsciousnessEventSequence:
    """Sequence of related consciousness events."""

    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_type: str = ""  # emergence_sequence, transition_sequence, etc.

    # Event sequence
    events: List[ConsciousnessEvent] = field(default_factory=list)
    event_relationships: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Sequence characteristics
    sequence_coherence: float = 0.0
    sequence_completeness: float = 0.0
    sequence_predictability: float = 0.0

    # Sequence outcomes
    final_consciousness_state: Optional[ConsciousnessState] = None
    sequence_effectiveness: float = 0.0
    sequence_efficiency: float = 0.0

### 8. Integration and Communication Models

@dataclass
class ConsciousnessMessage:
    """Message for inter-system consciousness communication."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    # Message routing
    source_system: str = ""
    target_system: str = ""
    message_priority: int = 1  # 1 = highest

    # Message content
    message_type: str = ""
    consciousness_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Quality assurance
    message_integrity_hash: str = ""
    expected_processing_time_ms: float = 0.0
    quality_requirements: Dict[str, float] = field(default_factory=dict)

@dataclass
class ConsciousnessIntegrationInterface:
    """Interface for integrating with other consciousness systems."""

    interface_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interface_name: str = ""
    interface_version: str = "1.0"

    # Interface capabilities
    supported_message_types: List[str] = field(default_factory=list)
    supported_data_formats: List[str] = field(default_factory=list)
    integration_protocols: List[str] = field(default_factory=list)

    # Performance characteristics
    max_throughput_messages_per_second: float = 1000.0
    max_latency_ms: float = 50.0
    reliability_percentage: float = 99.9

    # Quality assurance
    data_validation_enabled: bool = True
    error_recovery_enabled: bool = True
    monitoring_enabled: bool = True

### 9. Historical and Temporal Models

@dataclass
class ConsciousnessHistoryEntry:
    """Entry in consciousness state history."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    # Historical state
    consciousness_state: ConsciousnessState
    processing_context: ConsciousnessProcessingContext
    quality_assessment: ConsciousnessQualityProfile

    # Historical metadata
    state_duration_ms: float = 0.0
    transition_from_previous: Dict[str, float] = field(default_factory=dict)
    significant_events: List[str] = field(default_factory=list)  # Event IDs

@dataclass
class ConsciousnessHistoryManager:
    """Manager for consciousness state history."""

    manager_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # History storage
    history_entries: List[ConsciousnessHistoryEntry] = field(default_factory=list)
    max_history_length: int = 10000

    # History analysis
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    quality_trends: Dict[str, List[float]] = field(default_factory=dict)
    event_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Performance tracking
    history_maintenance_cost_ms: float = 0.0
    analysis_accuracy: float = 0.0
    prediction_capability: float = 0.0

## Data Model Relationships

### Primary Relationships

1. **ConsciousnessState** contains:
   - PhenomenalContent (phenomenal aspects)
   - SubjectivePerspective (subjective aspects)
   - UnifiedExperience (integration aspects)

2. **PhenomenalContent** contains:
   - Multiple QualitativeProperty instances (qualia)
   - ModalitySpecificContent for each modality

3. **UnifiedExperience** integrates:
   - PhenomenalContent and SubjectivePerspective
   - Multiple CrossModalBinding instances
   - ExperientialUnity characteristics

4. **ConsciousnessQualityProfile** assesses:
   - Quality across all consciousness components
   - Temporal stability and consistency
   - Comparative quality metrics

### Integration Patterns

- **Temporal Integration**: Consciousness states link through time via history management
- **Cross-Modal Integration**: Different modalities bind through CrossModalBinding
- **Quality Integration**: Multiple quality metrics combine into overall assessments
- **Event Integration**: Discrete events compose into meaningful sequences

This comprehensive data model provides the structured foundation for implementing sophisticated primary consciousness with rich phenomenal content, coherent subjective perspective, and unified conscious experience.