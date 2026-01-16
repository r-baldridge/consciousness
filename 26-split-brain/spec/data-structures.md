# Form 26: Split-brain Consciousness - Data Structures

## Core Data Models

### Hemispheric State Structures

**HemisphericState**
```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import time
import uuid

@dataclass
class HemisphericState:
    """Complete state representation for a single hemisphere."""

    hemisphere_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hemisphere_type: HemisphereType = None
    timestamp: float = field(default_factory=time.time)

    # Processing state
    active_processes: List[str] = field(default_factory=list)
    processing_load: float = 0.0
    available_resources: Dict[str, float] = field(default_factory=dict)

    # Cognitive state
    attention_state: Optional['AttentionState'] = None
    memory_state: Optional['MemoryState'] = None
    emotional_state: Optional['EmotionalState'] = None

    # Specialization state
    active_specializations: Set[str] = field(default_factory=set)
    specialization_weights: Dict[str, float] = field(default_factory=dict)

    # Communication state
    outgoing_messages: List['CommunicationMessage'] = field(default_factory=list)
    incoming_messages: List['CommunicationMessage'] = field(default_factory=list)

    # Performance metrics
    response_time: float = 0.0
    accuracy_score: float = 1.0
    confidence_level: float = 1.0

    def __post_init__(self):
        if self.hemisphere_type is None:
            raise ValueError("hemisphere_type must be specified")
```

**LeftHemisphereState**
```python
@dataclass
class LeftHemisphereState(HemisphericState):
    """Specialized state for left hemisphere processing."""

    # Language processing state
    language_context: Dict[str, Any] = field(default_factory=dict)
    active_linguistic_features: Set[str] = field(default_factory=set)
    semantic_networks: Dict[str, List[str]] = field(default_factory=dict)

    # Analytical processing state
    logical_reasoning_stack: List[Dict[str, Any]] = field(default_factory=list)
    sequential_processing_queue: List[Any] = field(default_factory=list)
    symbolic_representations: Dict[str, Any] = field(default_factory=dict)

    # Verbal output state
    pending_verbal_responses: List[str] = field(default_factory=list)
    speech_production_state: Dict[str, Any] = field(default_factory=dict)
    narrative_construction_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.hemisphere_type = HemisphereType.LEFT
        if not self.active_specializations:
            self.active_specializations = {
                "language_processing", "sequential_analysis",
                "logical_reasoning", "verbal_expression"
            }
```

**RightHemisphereState**
```python
@dataclass
class RightHemisphereState(HemisphericState):
    """Specialized state for right hemisphere processing."""

    # Spatial processing state
    spatial_maps: Dict[str, Any] = field(default_factory=dict)
    visual_processing_buffers: List[Any] = field(default_factory=list)
    spatial_attention_grid: Optional[Any] = None

    # Pattern recognition state
    pattern_templates: Dict[str, Any] = field(default_factory=dict)
    recognition_confidence_maps: Dict[str, float] = field(default_factory=dict)
    holistic_processing_context: Dict[str, Any] = field(default_factory=dict)

    # Emotional processing state
    emotional_valence_map: Dict[str, float] = field(default_factory=dict)
    affective_context: Dict[str, Any] = field(default_factory=dict)
    emotional_memory_associations: List[Dict[str, Any]] = field(default_factory=list)

    # Creative processing state
    creative_associations: List[Dict[str, Any]] = field(default_factory=list)
    artistic_processing_state: Dict[str, Any] = field(default_factory=dict)
    intuitive_insights: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.hemisphere_type = HemisphereType.RIGHT
        if not self.active_specializations:
            self.active_specializations = {
                "spatial_processing", "pattern_recognition",
                "emotional_processing", "creative_thinking"
            }
```

### Communication Data Structures

**CommunicationMessage**
```python
@dataclass
class CommunicationMessage:
    """Message structure for inter-hemispheric communication."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: HemisphereType = None
    receiver: HemisphereType = None
    channel: CommunicationChannel = None

    # Content
    content: Any = None
    content_type: str = "generic"
    encoding: str = "raw"

    # Metadata
    priority: int = 1  # 1 (low) to 10 (high)
    timestamp: float = field(default_factory=time.time)
    expiration_time: Optional[float] = None

    # Transmission properties
    size_bytes: int = 0
    compression_ratio: float = 1.0
    encryption_level: str = "none"

    # Status tracking
    status: str = "pending"  # pending, transmitted, received, failed
    retry_count: int = 0
    delivery_confirmation: bool = False

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expiration_time is None:
            return False
        return time.time() > self.expiration_time

    def calculate_latency(self, delivery_time: float) -> float:
        """Calculate message delivery latency."""
        return delivery_time - self.timestamp
```

**CommunicationChannel**
```python
@dataclass
class CommunicationChannelState:
    """State information for communication channels."""

    channel_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_type: CommunicationChannel = None

    # Capacity and performance
    bandwidth_bps: float = 1000000.0  # bits per second
    latency_ms: float = 10.0  # milliseconds
    packet_loss_rate: float = 0.0  # 0.0 to 1.0

    # Current status
    status: CommunicationStatus = CommunicationStatus.CONNECTED
    utilization: float = 0.0  # 0.0 to 1.0
    queue_length: int = 0

    # Quality metrics
    error_rate: float = 0.0
    throughput_bps: float = 0.0
    jitter_ms: float = 0.0

    # Configuration
    compression_enabled: bool = False
    encryption_enabled: bool = False
    priority_queuing: bool = True

    # Statistics
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_transferred: int = 0
    uptime_seconds: float = 0.0

    def calculate_effective_bandwidth(self) -> float:
        """Calculate effective bandwidth considering packet loss."""
        return self.bandwidth_bps * (1.0 - self.packet_loss_rate)
```

### Conflict Management Structures

**ConflictEvent**
```python
@dataclass
class ConflictEvent:
    """Detailed conflict event representation."""

    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = None
    severity: float = 0.0  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)

    # Conflict participants
    left_hemisphere_data: Any = None
    right_hemisphere_data: Any = None
    conflicting_aspects: List[str] = field(default_factory=list)

    # Context information
    task_context: Optional['ProcessingContext'] = None
    environmental_factors: Dict[str, Any] = field(default_factory=dict)

    # Conflict analysis
    similarity_score: float = 0.0
    incompatibility_score: float = 0.0
    resolution_complexity: float = 0.0

    # Resolution tracking
    resolution_attempted: bool = False
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolution_result: Optional['ResolutionResult'] = None
    resolution_time: Optional[float] = None

    # Learning data
    previous_similar_conflicts: List[str] = field(default_factory=list)
    success_probability: float = 0.0
    recommended_strategy: Optional[ResolutionStrategy] = None
```

**ResolutionResult**
```python
@dataclass
class ResolutionResult:
    """Result of conflict resolution attempt."""

    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_id: str = None
    timestamp: float = field(default_factory=time.time)

    # Resolution outcome
    success: bool = False
    strategy_used: ResolutionStrategy = None
    resolved_output: Any = None

    # Performance metrics
    resolution_time_ms: float = 0.0
    computational_cost: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

    # Quality assessment
    coherence_score: float = 0.0
    satisfaction_score: float = 0.0
    stability_score: float = 0.0

    # Side effects
    hemispheric_satisfaction: Dict[HemisphereType, float] = field(default_factory=dict)
    unresolved_tensions: List[str] = field(default_factory=list)
    adaptive_changes: List[str] = field(default_factory=list)

    # Learning feedback
    effectiveness_rating: float = 0.0
    improvement_suggestions: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)
```

### Memory Structures

**MemoryItem**
```python
@dataclass
class MemoryItem:
    """Individual memory item with hemispheric association."""

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = None

    # Hemispheric association
    hemisphere: HemisphereType = None
    accessibility: Dict[HemisphereType, float] = field(default_factory=dict)

    # Temporal information
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0

    # Memory strength and quality
    strength: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    vividness: float = 1.0  # 0.0 to 1.0

    # Associations
    semantic_associations: List[str] = field(default_factory=list)
    episodic_associations: List[str] = field(default_factory=list)
    emotional_associations: Dict[str, float] = field(default_factory=dict)

    # Encoding information
    encoding_context: Dict[str, Any] = field(default_factory=dict)
    encoding_strategies: List[str] = field(default_factory=list)
    consolidation_status: str = "unconsolidated"

    def decay_strength(self, decay_rate: float = 0.01):
        """Apply memory decay based on time and access patterns."""
        time_since_access = time.time() - self.last_access_time
        decay_factor = decay_rate * time_since_access / 86400  # daily decay
        self.strength = max(0.0, self.strength - decay_factor)
```

**MemoryState**
```python
@dataclass
class MemoryState:
    """Current memory system state for a hemisphere."""

    hemisphere: HemisphereType = None
    timestamp: float = field(default_factory=time.time)

    # Memory pools
    working_memory: List[MemoryItem] = field(default_factory=list)
    short_term_memory: List[MemoryItem] = field(default_factory=list)
    long_term_memory: Dict[str, MemoryItem] = field(default_factory=dict)

    # Capacity and utilization
    working_memory_capacity: int = 7
    working_memory_utilization: float = 0.0
    total_memory_items: int = 0

    # Performance metrics
    retrieval_latency_ms: float = 100.0
    storage_success_rate: float = 1.0
    forgetting_rate: float = 0.01

    # Active processes
    active_retrievals: List[str] = field(default_factory=list)
    pending_consolidations: List[str] = field(default_factory=list)

    def add_to_working_memory(self, item: MemoryItem) -> bool:
        """Add item to working memory with capacity checking."""
        if len(self.working_memory) >= self.working_memory_capacity:
            # Remove oldest item
            self.working_memory.pop(0)

        self.working_memory.append(item)
        self.working_memory_utilization = len(self.working_memory) / self.working_memory_capacity
        return True
```

### Attention Structures

**AttentionState**
```python
@dataclass
class AttentionState:
    """Attention state for a hemisphere."""

    hemisphere: HemisphereType = None
    timestamp: float = field(default_factory=time.time)

    # Attention allocation
    attention_targets: List[Dict[str, Any]] = field(default_factory=list)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    total_attention_capacity: float = 1.0
    available_attention: float = 1.0

    # Attention types
    focused_attention: Optional[Dict[str, Any]] = None
    divided_attention_targets: List[Dict[str, Any]] = field(default_factory=list)
    sustained_attention_duration: float = 0.0

    # Dynamic properties
    attention_switching_rate: float = 0.1  # switches per second
    attention_stability: float = 1.0  # 0.0 to 1.0
    distraction_resistance: float = 0.8  # 0.0 to 1.0

    # Performance metrics
    attention_efficiency: float = 1.0
    target_detection_accuracy: float = 1.0
    response_time_ms: float = 200.0

    def allocate_attention(self, target: Dict[str, Any], weight: float) -> bool:
        """Allocate attention to a specific target."""
        required_attention = weight

        if required_attention > self.available_attention:
            # Need to redistribute or reject
            return False

        target_id = target.get('id', str(uuid.uuid4()))
        self.attention_targets.append(target)
        self.attention_weights[target_id] = weight
        self.available_attention -= weight

        return True
```

### Integration and Unity Structures

**UnityState**
```python
@dataclass
class UnityState:
    """State of consciousness unity/division."""

    timestamp: float = field(default_factory=time.time)
    unity_mode: UnityMode = UnityMode.NATURAL_UNITY

    # Unity metrics
    integration_strength: float = 1.0  # 0.0 to 1.0
    coherence_level: float = 1.0  # 0.0 to 1.0
    behavioral_consistency: float = 1.0  # 0.0 to 1.0

    # Hemispheric coordination
    left_right_synchronization: float = 1.0
    communication_efficiency: float = 1.0
    conflict_frequency: float = 0.0

    # Simulation state
    unity_simulation_active: bool = False
    simulation_quality: float = 1.0
    simulation_computational_cost: float = 0.0

    # Historical tracking
    unity_history: List[float] = field(default_factory=list)
    stability_trend: str = "stable"  # improving, stable, declining

    def update_unity_metrics(self, left_state: HemisphericState, right_state: HemisphericState):
        """Update unity metrics based on hemispheric states."""
        # Calculate integration strength based on communication
        comm_messages = len(left_state.outgoing_messages) + len(right_state.outgoing_messages)
        self.integration_strength = min(1.0, comm_messages / 10.0)

        # Calculate coherence based on conflicting outputs
        # (Implementation would analyze actual outputs)

        # Update history
        self.unity_history.append(self.coherence_level)
        if len(self.unity_history) > 100:  # Keep last 100 measurements
            self.unity_history.pop(0)
```

**SplitBrainSystemState**
```python
@dataclass
class SplitBrainSystemState:
    """Complete system state for split-brain consciousness."""

    system_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Hemispheric states
    left_hemisphere: LeftHemisphereState = field(default_factory=LeftHemisphereState)
    right_hemisphere: RightHemisphereState = field(default_factory=RightHemisphereState)

    # Communication system
    communication_channels: Dict[CommunicationChannel, CommunicationChannelState] = field(default_factory=dict)
    message_queue: List[CommunicationMessage] = field(default_factory=list)

    # Conflict management
    active_conflicts: List[ConflictEvent] = field(default_factory=list)
    recent_resolutions: List[ResolutionResult] = field(default_factory=list)

    # Unity and integration
    unity_state: UnityState = field(default_factory=UnityState)
    integration_state: Dict[str, Any] = field(default_factory=dict)

    # Performance and monitoring
    system_performance: Dict[str, float] = field(default_factory=dict)
    health_metrics: Dict[str, float] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    disconnection_level: float = 0.0  # 0.0 (connected) to 1.0 (disconnected)
    compensation_strategies: List[CompensationType] = field(default_factory=list)

    def get_overall_coherence(self) -> float:
        """Calculate overall system coherence."""
        hemisphere_coherence = (
            self.left_hemisphere.confidence_level +
            self.right_hemisphere.confidence_level
        ) / 2.0

        communication_quality = sum(
            channel.throughput_bps / channel.bandwidth_bps
            for channel in self.communication_channels.values()
        ) / max(1, len(self.communication_channels))

        conflict_impact = 1.0 - (len(self.active_conflicts) * 0.1)

        return (hemisphere_coherence + communication_quality + conflict_impact) / 3.0
```

### Configuration and Metadata Structures

**SystemConfiguration**
```python
@dataclass
class SystemConfiguration:
    """Configuration parameters for split-brain consciousness system."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0.0"
    timestamp: float = field(default_factory=time.time)

    # Core settings
    disconnection_level: float = 0.0
    compensation_enabled: bool = True
    unity_simulation_mode: UnityMode = UnityMode.NATURAL_UNITY

    # Communication settings
    default_bandwidth: float = 1000000.0
    communication_timeout: float = 1.0
    message_retry_limit: int = 3

    # Conflict resolution
    default_resolution_strategy: ResolutionStrategy = ResolutionStrategy.INTEGRATION
    conflict_detection_threshold: float = 0.3
    resolution_timeout: float = 5.0

    # Performance settings
    max_processing_latency: float = 100.0  # milliseconds
    memory_limit_per_hemisphere: int = 1000000  # items
    attention_update_frequency: float = 10.0  # Hz

    # Monitoring settings
    metrics_collection_interval: float = 1.0  # seconds
    health_check_interval: float = 5.0  # seconds
    log_retention_days: int = 30

    # Hemispheric specializations
    left_specializations: Set[str] = field(default_factory=lambda: {
        "language_processing", "sequential_analysis", "logical_reasoning",
        "verbal_expression", "mathematical_processing"
    })

    right_specializations: Set[str] = field(default_factory=lambda: {
        "spatial_processing", "pattern_recognition", "emotional_processing",
        "creative_thinking", "artistic_processing"
    })

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if not 0.0 <= self.disconnection_level <= 1.0:
            errors.append("disconnection_level must be between 0.0 and 1.0")

        if self.default_bandwidth <= 0:
            errors.append("default_bandwidth must be positive")

        if not 0.0 <= self.conflict_detection_threshold <= 1.0:
            errors.append("conflict_detection_threshold must be between 0.0 and 1.0")

        return errors
```

These data structures provide a comprehensive foundation for representing all aspects of split-brain consciousness, from basic hemispheric states to complex integration and conflict resolution mechanisms. The structures are designed to be extensible, type-safe, and efficient for real-time consciousness simulation and analysis.