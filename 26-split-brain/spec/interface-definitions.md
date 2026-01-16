# Form 26: Split-brain Consciousness - Interface Definitions

## Core System Interfaces

### Hemispheric Processing Interface

**IHemisphericProcessor**
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class HemisphereType(Enum):
    LEFT = "left"
    RIGHT = "right"

class ProcessingMode(Enum):
    ISOLATED = "isolated"
    COLLABORATIVE = "collaborative"
    CONFLICTED = "conflicted"

@dataclass
class ProcessingContext:
    hemisphere: HemisphereType
    task_type: str
    priority: int
    timestamp: float
    session_id: str

class IHemisphericProcessor(ABC):
    """Interface for independent hemispheric processing units."""

    @abstractmethod
    def process_input(self, input_data: Any, context: ProcessingContext) -> Any:
        """Process input data within hemispheric specialization."""
        pass

    @abstractmethod
    def get_specializations(self) -> List[str]:
        """Return list of cognitive specializations for this hemisphere."""
        pass

    @abstractmethod
    def set_processing_mode(self, mode: ProcessingMode) -> None:
        """Configure processing mode for hemisphere."""
        pass

    @abstractmethod
    def get_processing_state(self) -> Dict[str, Any]:
        """Return current processing state and status."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset hemisphere to initial state."""
        pass
```

**ILeftHemisphere**
```python
class LanguageProcessingResult:
    def __init__(self, text: str, confidence: float, linguistic_features: Dict):
        self.text = text
        self.confidence = confidence
        self.linguistic_features = linguistic_features

class ILeftHemisphere(IHemisphericProcessor):
    """Specialized interface for left hemisphere processing."""

    @abstractmethod
    def process_language(self, text: str, context: ProcessingContext) -> LanguageProcessingResult:
        """Process linguistic input and generate language responses."""
        pass

    @abstractmethod
    def perform_sequential_analysis(self, data: List[Any], context: ProcessingContext) -> Any:
        """Perform sequential, analytical processing."""
        pass

    @abstractmethod
    def execute_logical_reasoning(self, premises: List[str], context: ProcessingContext) -> Any:
        """Execute logical reasoning and problem solving."""
        pass

    @abstractmethod
    def generate_verbal_response(self, internal_state: Any, context: ProcessingContext) -> str:
        """Generate verbal responses and explanations."""
        pass
```

**IRightHemisphere**
```python
class SpatialProcessingResult:
    def __init__(self, spatial_map: Any, confidence: float, visual_features: Dict):
        self.spatial_map = spatial_map
        self.confidence = confidence
        self.visual_features = visual_features

class IRightHemisphere(IHemisphericProcessor):
    """Specialized interface for right hemisphere processing."""

    @abstractmethod
    def process_spatial_information(self, spatial_data: Any, context: ProcessingContext) -> SpatialProcessingResult:
        """Process spatial and visual-spatial information."""
        pass

    @abstractmethod
    def perform_pattern_recognition(self, data: Any, context: ProcessingContext) -> Any:
        """Perform holistic pattern recognition and analysis."""
        pass

    @abstractmethod
    def process_emotional_content(self, emotional_data: Any, context: ProcessingContext) -> Any:
        """Process emotional and affective information."""
        pass

    @abstractmethod
    def generate_nonverbal_response(self, internal_state: Any, context: ProcessingContext) -> Any:
        """Generate non-verbal responses and behaviors."""
        pass
```

### Inter-hemispheric Communication Interface

**IInterhemisphericCommunication**
```python
class CommunicationChannel(Enum):
    CALLOSAL = "callosal"
    SUBCORTICAL = "subcortical"
    EXTERNAL = "external"
    SENSORY_CROSSOVER = "sensory_crossover"

@dataclass
class CommunicationMessage:
    sender: HemisphereType
    receiver: HemisphereType
    content: Any
    channel: CommunicationChannel
    priority: int
    timestamp: float
    message_id: str

class CommunicationStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    BLOCKED = "blocked"

class IInterhemisphericCommunication(ABC):
    """Interface for managing communication between hemispheres."""

    @abstractmethod
    def send_message(self, message: CommunicationMessage) -> bool:
        """Send message between hemispheres."""
        pass

    @abstractmethod
    def receive_message(self, hemisphere: HemisphereType, timeout: float = 1.0) -> Optional[CommunicationMessage]:
        """Receive message for specified hemisphere."""
        pass

    @abstractmethod
    def set_channel_status(self, channel: CommunicationChannel, status: CommunicationStatus) -> None:
        """Configure communication channel status."""
        pass

    @abstractmethod
    def get_channel_status(self, channel: CommunicationChannel) -> CommunicationStatus:
        """Get current status of communication channel."""
        pass

    @abstractmethod
    def set_bandwidth_limit(self, channel: CommunicationChannel, bandwidth: float) -> None:
        """Set bandwidth limitation for communication channel."""
        pass

    @abstractmethod
    def get_communication_history(self, limit: int = 100) -> List[CommunicationMessage]:
        """Retrieve recent communication history."""
        pass
```

### Conflict Detection and Resolution Interface

**IConflictDetector**
```python
class ConflictType(Enum):
    RESPONSE_CONFLICT = "response_conflict"
    GOAL_CONFLICT = "goal_conflict"
    PREFERENCE_CONFLICT = "preference_conflict"
    ATTENTION_CONFLICT = "attention_conflict"
    MEMORY_CONFLICT = "memory_conflict"

@dataclass
class ConflictEvent:
    conflict_type: ConflictType
    left_response: Any
    right_response: Any
    severity: float
    context: ProcessingContext
    timestamp: float
    conflict_id: str

class IConflictDetector(ABC):
    """Interface for detecting conflicts between hemispheres."""

    @abstractmethod
    def detect_conflicts(self, left_output: Any, right_output: Any, context: ProcessingContext) -> List[ConflictEvent]:
        """Detect conflicts between hemispheric outputs."""
        pass

    @abstractmethod
    def assess_conflict_severity(self, conflict: ConflictEvent) -> float:
        """Assess the severity of a detected conflict."""
        pass

    @abstractmethod
    def classify_conflict_type(self, conflict_data: Any) -> ConflictType:
        """Classify the type of conflict detected."""
        pass

    @abstractmethod
    def set_conflict_threshold(self, conflict_type: ConflictType, threshold: float) -> None:
        """Set threshold for conflict detection sensitivity."""
        pass
```

**IConflictResolver**
```python
class ResolutionStrategy(Enum):
    LEFT_DOMINANCE = "left_dominance"
    RIGHT_DOMINANCE = "right_dominance"
    INTEGRATION = "integration"
    ALTERNATION = "alternation"
    EXTERNAL_ARBITRATION = "external_arbitration"

@dataclass
class ResolutionResult:
    strategy_used: ResolutionStrategy
    resolved_response: Any
    confidence: float
    resolution_time: float
    success: bool

class IConflictResolver(ABC):
    """Interface for resolving conflicts between hemispheres."""

    @abstractmethod
    def resolve_conflict(self, conflict: ConflictEvent) -> ResolutionResult:
        """Resolve detected conflict between hemispheres."""
        pass

    @abstractmethod
    def select_resolution_strategy(self, conflict: ConflictEvent) -> ResolutionStrategy:
        """Select appropriate resolution strategy for conflict."""
        pass

    @abstractmethod
    def apply_resolution_strategy(self, conflict: ConflictEvent, strategy: ResolutionStrategy) -> ResolutionResult:
        """Apply selected resolution strategy to conflict."""
        pass

    @abstractmethod
    def learn_from_resolution(self, conflict: ConflictEvent, result: ResolutionResult) -> None:
        """Learn from resolution outcomes to improve future decisions."""
        pass
```

### Compensation Mechanism Interface

**ICompensationManager**
```python
class CompensationType(Enum):
    CROSS_CUING = "cross_cuing"
    SUBCORTICAL_ROUTING = "subcortical_routing"
    EXTERNAL_FEEDBACK = "external_feedback"
    BEHAVIORAL_ADAPTATION = "behavioral_adaptation"

@dataclass
class CompensationStrategy:
    compensation_type: CompensationType
    effectiveness: float
    resource_cost: float
    learning_required: bool
    adaptation_time: float

class ICompensationManager(ABC):
    """Interface for managing compensation mechanisms."""

    @abstractmethod
    def develop_compensation_strategy(self, communication_status: Dict[CommunicationChannel, CommunicationStatus]) -> CompensationStrategy:
        """Develop compensation strategy based on communication status."""
        pass

    @abstractmethod
    def implement_compensation(self, strategy: CompensationStrategy, context: ProcessingContext) -> bool:
        """Implement compensation mechanism."""
        pass

    @abstractmethod
    def evaluate_compensation_effectiveness(self, strategy: CompensationStrategy) -> float:
        """Evaluate effectiveness of compensation mechanism."""
        pass

    @abstractmethod
    def adapt_compensation_strategy(self, strategy: CompensationStrategy, feedback: Any) -> CompensationStrategy:
        """Adapt compensation strategy based on feedback."""
        pass
```

### Unity Simulation Interface

**IUnitySimulator**
```python
class UnityMode(Enum):
    NATURAL_UNITY = "natural_unity"
    SIMULATED_UNITY = "simulated_unity"
    APPARENT_UNITY = "apparent_unity"
    DIVIDED_AWARENESS = "divided_awareness"

@dataclass
class UnityState:
    mode: UnityMode
    coherence_level: float
    integration_strength: float
    behavioral_consistency: float
    timestamp: float

class IUnitySimulator(ABC):
    """Interface for simulating unified consciousness behavior."""

    @abstractmethod
    def simulate_unity(self, left_output: Any, right_output: Any, context: ProcessingContext) -> Any:
        """Simulate unified consciousness output from hemispheric inputs."""
        pass

    @abstractmethod
    def assess_unity_requirement(self, context: ProcessingContext) -> bool:
        """Assess whether unity simulation is required for context."""
        pass

    @abstractmethod
    def set_unity_mode(self, mode: UnityMode) -> None:
        """Set unity simulation mode."""
        pass

    @abstractmethod
    def get_unity_state(self) -> UnityState:
        """Get current unity simulation state."""
        pass

    @abstractmethod
    def measure_behavioral_coherence(self, outputs: List[Any]) -> float:
        """Measure coherence of behavioral outputs."""
        pass
```

### Memory Management Interface

**IHemisphericMemory**
```python
class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"

@dataclass
class MemoryItem:
    content: Any
    memory_type: MemoryType
    hemisphere: HemisphereType
    confidence: float
    timestamp: float
    access_count: int
    memory_id: str

class IHemisphericMemory(ABC):
    """Interface for hemispheric memory management."""

    @abstractmethod
    def store_memory(self, item: MemoryItem) -> str:
        """Store memory item in hemispheric memory system."""
        pass

    @abstractmethod
    def retrieve_memory(self, memory_id: str, hemisphere: HemisphereType) -> Optional[MemoryItem]:
        """Retrieve memory item from hemispheric memory."""
        pass

    @abstractmethod
    def search_memories(self, query: Any, hemisphere: HemisphereType, memory_type: Optional[MemoryType] = None) -> List[MemoryItem]:
        """Search memories within hemisphere."""
        pass

    @abstractmethod
    def transfer_memory(self, memory_id: str, source: HemisphereType, target: HemisphereType) -> bool:
        """Transfer memory between hemispheres if communication allows."""
        pass

    @abstractmethod
    def consolidate_memories(self, hemisphere: HemisphereType) -> int:
        """Consolidate memories within hemisphere."""
        pass
```

### Attention Management Interface

**IHemisphericAttention**
```python
class AttentionType(Enum):
    FOCUSED = "focused"
    DIVIDED = "divided"
    SUSTAINED = "sustained"
    SELECTIVE = "selective"

@dataclass
class AttentionState:
    attention_type: AttentionType
    focus_targets: List[Any]
    intensity: float
    duration: float
    hemisphere: HemisphereType

class IHemisphericAttention(ABC):
    """Interface for hemispheric attention management."""

    @abstractmethod
    def allocate_attention(self, targets: List[Any], hemisphere: HemisphereType) -> AttentionState:
        """Allocate attention resources within hemisphere."""
        pass

    @abstractmethod
    def shift_attention(self, new_target: Any, hemisphere: HemisphereType) -> AttentionState:
        """Shift attention to new target within hemisphere."""
        pass

    @abstractmethod
    def coordinate_attention(self, left_state: AttentionState, right_state: AttentionState) -> Tuple[AttentionState, AttentionState]:
        """Coordinate attention between hemispheres when possible."""
        pass

    @abstractmethod
    def detect_attention_conflicts(self, left_state: AttentionState, right_state: AttentionState) -> List[ConflictEvent]:
        """Detect attention conflicts between hemispheres."""
        pass
```

### Integration Interfaces

**IConsciousnessFormIntegration**
```python
class ConsciousnessForm(Enum):
    BASIC_AWARENESS = "form_01"
    INTENTIONAL = "form_05"
    SOCIAL = "form_09"
    META_CONSCIOUSNESS = "form_11"

@dataclass
class IntegrationContext:
    source_form: ConsciousnessForm
    integration_type: str
    priority: int
    bidirectional: bool

class IConsciousnessFormIntegration(ABC):
    """Interface for integrating with other consciousness forms."""

    @abstractmethod
    def integrate_with_form(self, form: ConsciousnessForm, data: Any, context: IntegrationContext) -> Any:
        """Integrate with another consciousness form."""
        pass

    @abstractmethod
    def provide_split_brain_data(self, requesting_form: ConsciousnessForm) -> Any:
        """Provide split-brain consciousness data to other forms."""
        pass

    @abstractmethod
    def receive_form_data(self, source_form: ConsciousnessForm, data: Any) -> Any:
        """Receive and process data from other consciousness forms."""
        pass

    @abstractmethod
    def establish_integration_channel(self, target_form: ConsciousnessForm) -> bool:
        """Establish integration channel with another consciousness form."""
        pass
```

### Monitoring and Diagnostics Interface

**ISplitBrainMonitor**
```python
@dataclass
class MonitoringMetrics:
    hemispheric_activity: Dict[HemisphereType, float]
    communication_frequency: float
    conflict_rate: float
    resolution_success_rate: float
    compensation_effectiveness: float
    unity_coherence: float
    timestamp: float

class ISplitBrainMonitor(ABC):
    """Interface for monitoring split-brain consciousness system."""

    @abstractmethod
    def collect_metrics(self) -> MonitoringMetrics:
        """Collect current system metrics."""
        pass

    @abstractmethod
    def detect_anomalies(self, metrics: MonitoringMetrics) -> List[str]:
        """Detect anomalies in system behavior."""
        pass

    @abstractmethod
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        pass

    @abstractmethod
    def set_monitoring_parameters(self, parameters: Dict[str, Any]) -> None:
        """Configure monitoring parameters."""
        pass
```

### Configuration Interface

**ISplitBrainConfiguration**
```python
@dataclass
class SystemConfiguration:
    disconnection_level: float  # 0.0 (full connection) to 1.0 (complete disconnection)
    compensation_enabled: bool
    unity_simulation_mode: UnityMode
    conflict_resolution_strategy: ResolutionStrategy
    monitoring_level: str
    hemispheric_specialization: Dict[HemisphereType, List[str]]

class ISplitBrainConfiguration(ABC):
    """Interface for system configuration management."""

    @abstractmethod
    def set_configuration(self, config: SystemConfiguration) -> None:
        """Set system configuration."""
        pass

    @abstractmethod
    def get_configuration(self) -> SystemConfiguration:
        """Get current system configuration."""
        pass

    @abstractmethod
    def update_configuration(self, updates: Dict[str, Any]) -> None:
        """Update specific configuration parameters."""
        pass

    @abstractmethod
    def validate_configuration(self, config: SystemConfiguration) -> bool:
        """Validate configuration for consistency and feasibility."""
        pass

    @abstractmethod
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        pass
```

These interface definitions provide a comprehensive framework for implementing split-brain consciousness systems, ensuring proper separation of concerns, modularity, and extensibility while maintaining the specialized functionality required for modeling hemispheric independence and integration.