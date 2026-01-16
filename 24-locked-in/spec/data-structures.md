# Form 24: Locked-in Syndrome Consciousness - Data Structures

## Core Data Models

### Patient and User Management

#### Patient Profile Data Structure
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum

class LockedInSeverity(Enum):
    COMPLETE = "complete"  # No voluntary movement except eyes
    INCOMPLETE = "incomplete"  # Minimal residual movement
    CLASSIC = "classic"  # Vertical eye movements and blinking only
    TOTAL = "total"  # Complete paralysis including eyes

class CommunicationCapability(Enum):
    EYE_MOVEMENT = "eye_movement"
    BLINKING = "blinking"
    MINIMAL_FACIAL = "minimal_facial"
    RESIDUAL_LIMB = "residual_limb"
    NONE_DETECTED = "none_detected"

@dataclass
class MedicalHistory:
    diagnosis: str
    onset_date: datetime
    etiology: str  # stroke, ALS, trauma, etc.
    affected_regions: List[str]
    comorbidities: List[str]
    medications: List[Dict[str, Any]]
    progression_notes: List[str]

@dataclass
class FunctionalAssessment:
    consciousness_level: str
    cognitive_function_preserved: bool
    communication_capabilities: List[CommunicationCapability]
    residual_motor_function: Dict[str, float]  # body part -> function level
    sensory_function: Dict[str, bool]
    last_assessment_date: datetime
    assessment_confidence: float

@dataclass
class PatientProfile:
    patient_id: str
    demographic_info: Dict[str, Any]
    medical_history: MedicalHistory
    functional_assessment: FunctionalAssessment
    locked_in_severity: LockedInSeverity
    communication_preferences: Dict[str, Any]
    technology_experience: Dict[str, Any]
    caregiver_contacts: List[Dict[str, str]]
    created_date: datetime
    last_updated: datetime

    def update_functional_assessment(self, new_assessment: FunctionalAssessment) -> None:
        """Update functional assessment with new data."""
        self.functional_assessment = new_assessment
        self.last_updated = datetime.now()

    def get_optimal_communication_modalities(self) -> List[str]:
        """Determine optimal communication modalities based on capabilities."""
        modalities = []

        if CommunicationCapability.EYE_MOVEMENT in self.functional_assessment.communication_capabilities:
            modalities.append("eye_tracking")

        if self.functional_assessment.cognitive_function_preserved:
            modalities.extend(["p300_bci", "ssvep_bci", "motor_imagery_bci"])

        if CommunicationCapability.RESIDUAL_LIMB in self.functional_assessment.communication_capabilities:
            modalities.append("switch_interface")

        return modalities
```

### Neural Signal Data Structures

#### EEG Signal Data
```python
import numpy as np
from typing import Tuple

@dataclass
class EEGChannelInfo:
    channel_name: str
    electrode_position: str
    impedance: float
    reference_type: str
    sampling_rate: int
    gain: float
    filter_settings: Dict[str, float]

@dataclass
class EEGSignalData:
    session_id: str
    patient_id: str
    start_timestamp: float
    sampling_rate: int
    channels: List[EEGChannelInfo]
    raw_data: np.ndarray  # shape: (n_channels, n_samples)
    processed_data: Optional[np.ndarray] = None
    artifacts: Optional[List[Tuple[float, float]]] = None  # time intervals
    quality_metrics: Optional[Dict[str, float]] = None
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add_event_marker(self, event_type: str, timestamp: float, description: str = "") -> None:
        """Add event marker to EEG data."""
        self.events.append({
            'type': event_type,
            'timestamp': timestamp,
            'description': description,
            'sample_index': int((timestamp - self.start_timestamp) * self.sampling_rate)
        })

    def get_signal_quality_score(self) -> float:
        """Calculate overall signal quality score."""
        if not self.quality_metrics:
            return 0.0

        weights = {
            'signal_to_noise_ratio': 0.3,
            'artifact_percentage': -0.25,  # negative weight
            'electrode_impedance_avg': -0.2,  # negative weight
            'spectral_power_distribution': 0.25
        }

        score = 0.0
        for metric, weight in weights.items():
            if metric in self.quality_metrics:
                score += weight * self.quality_metrics[metric]

        return max(0.0, min(1.0, score))
```

#### Brain-Computer Interface Data
```python
@dataclass
class BCITrialData:
    trial_id: str
    paradigm_type: str
    target_stimulus: str
    user_response: Optional[str]
    confidence_score: float
    reaction_time: float
    signal_features: Dict[str, float]
    classification_result: Dict[str, float]
    timestamp: float

@dataclass
class BCISessionData:
    session_id: str
    patient_id: str
    paradigm_type: str
    start_time: datetime
    end_time: Optional[datetime]
    trials: List[BCITrialData]
    calibration_data: Optional[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    user_feedback: Optional[str]

    def calculate_accuracy(self) -> float:
        """Calculate overall session accuracy."""
        if not self.trials:
            return 0.0

        correct_trials = sum(1 for trial in self.trials
                           if trial.target_stimulus == trial.user_response)
        return correct_trials / len(self.trials)

    def calculate_information_transfer_rate(self) -> float:
        """Calculate information transfer rate in bits per minute."""
        if not self.trials or not self.end_time:
            return 0.0

        accuracy = self.calculate_accuracy()
        n_targets = len(set(trial.target_stimulus for trial in self.trials))

        if accuracy <= 1/n_targets:
            return 0.0

        bit_rate = np.log2(n_targets) + accuracy * np.log2(accuracy) + (1-accuracy) * np.log2((1-accuracy)/(n_targets-1))
        session_duration_minutes = (self.end_time - self.start_time).total_seconds() / 60

        return bit_rate * len(self.trials) / session_duration_minutes
```

### Eye-Tracking Data Structures

#### Gaze Data
```python
@dataclass
class GazePoint:
    x: float  # screen coordinates (0-1 normalized)
    y: float
    timestamp: float
    confidence: float
    pupil_diameter_left: Optional[float]
    pupil_diameter_right: Optional[float]
    eye_openness_left: float
    eye_openness_right: float

@dataclass
class GazeCalibrationPoint:
    target_x: float
    target_y: float
    measured_x: float
    measured_y: float
    accuracy: float  # degrees visual angle
    validation_trials: int

@dataclass
class EyeTrackingSession:
    session_id: str
    patient_id: str
    start_time: datetime
    calibration_data: List[GazeCalibrationPoint]
    gaze_points: List[GazePoint]
    selections: List[Dict[str, Any]]
    system_parameters: Dict[str, Any]
    environment_conditions: Dict[str, Any]

    def get_average_accuracy(self) -> float:
        """Calculate average calibration accuracy."""
        if not self.calibration_data:
            return 0.0
        return np.mean([point.accuracy for point in self.calibration_data])

    def detect_fixations(self, velocity_threshold: float = 30.0,
                        duration_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect fixation periods in gaze data."""
        fixations = []
        current_fixation = None

        for i, point in enumerate(self.gaze_points[1:], 1):
            prev_point = self.gaze_points[i-1]

            # Calculate gaze velocity
            distance = np.sqrt((point.x - prev_point.x)**2 + (point.y - prev_point.y)**2)
            time_diff = point.timestamp - prev_point.timestamp
            velocity = distance / time_diff if time_diff > 0 else 0

            if velocity < velocity_threshold:
                if current_fixation is None:
                    current_fixation = {
                        'start_time': prev_point.timestamp,
                        'start_index': i-1,
                        'x_sum': prev_point.x,
                        'y_sum': prev_point.y,
                        'count': 1
                    }
                else:
                    current_fixation['x_sum'] += point.x
                    current_fixation['y_sum'] += point.y
                    current_fixation['count'] += 1
            else:
                if current_fixation and (point.timestamp - current_fixation['start_time']) >= duration_threshold:
                    fixations.append({
                        'start_time': current_fixation['start_time'],
                        'end_time': point.timestamp,
                        'duration': point.timestamp - current_fixation['start_time'],
                        'center_x': current_fixation['x_sum'] / current_fixation['count'],
                        'center_y': current_fixation['y_sum'] / current_fixation['count'],
                        'point_count': current_fixation['count']
                    })
                current_fixation = None

        return fixations
```

### Communication Data Structures

#### Communication Events
```python
class CommunicationModality(Enum):
    EYE_TRACKING = "eye_tracking"
    BCI_P300 = "bci_p300"
    BCI_SSVEP = "bci_ssvep"
    BCI_MOTOR_IMAGERY = "bci_motor_imagery"
    SWITCH_INTERFACE = "switch_interface"
    HYBRID = "hybrid"

@dataclass
class CommunicationEvent:
    event_id: str
    patient_id: str
    session_id: str
    modality: CommunicationModality
    content: str
    confidence: float
    timestamp: datetime
    processing_time: float
    user_effort_level: Optional[float]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'patient_id': self.patient_id,
            'session_id': self.session_id,
            'modality': self.modality.value,
            'content': self.content,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'user_effort_level': self.user_effort_level,
            'context': self.context
        }

@dataclass
class CommunicationSession:
    session_id: str
    patient_id: str
    start_time: datetime
    end_time: Optional[datetime]
    primary_modality: CommunicationModality
    backup_modalities: List[CommunicationModality]
    events: List[CommunicationEvent]
    performance_metrics: Dict[str, float]
    user_satisfaction: Optional[float]
    technical_issues: List[str]

    def calculate_communication_rate(self) -> float:
        """Calculate characters per minute communication rate."""
        if not self.events or not self.end_time:
            return 0.0

        total_characters = sum(len(event.content) for event in self.events)
        session_duration_minutes = (self.end_time - self.start_time).total_seconds() / 60

        return total_characters / session_duration_minutes if session_duration_minutes > 0 else 0.0

    def get_modality_usage_distribution(self) -> Dict[str, float]:
        """Get distribution of modality usage during session."""
        if not self.events:
            return {}

        modality_counts = {}
        for event in self.events:
            modality = event.modality.value
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        total_events = len(self.events)
        return {modality: count/total_events for modality, count in modality_counts.items()}
```

### Performance and Analytics Data Structures

#### Performance Metrics
```python
@dataclass
class PerformanceMetric:
    metric_id: str
    patient_id: str
    metric_name: str
    metric_value: float
    measurement_unit: str
    timestamp: datetime
    session_context: Dict[str, Any]
    confidence_interval: Optional[Tuple[float, float]]

    def is_within_normal_range(self, normal_range: Tuple[float, float]) -> bool:
        """Check if metric value is within normal range."""
        return normal_range[0] <= self.metric_value <= normal_range[1]

@dataclass
class PerformanceTrend:
    patient_id: str
    metric_name: str
    time_range: Tuple[datetime, datetime]
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0.0 to 1.0
    statistical_significance: float
    data_points: List[PerformanceMetric]
    prediction_confidence: float

    def calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement or decline."""
        if len(self.data_points) < 2:
            return 0.0

        values = [point.metric_value for point in sorted(self.data_points, key=lambda x: x.timestamp)]
        time_diffs = [(self.data_points[i].timestamp - self.data_points[i-1].timestamp).total_seconds()
                     for i in range(1, len(self.data_points))]

        if not time_diffs or sum(time_diffs) == 0:
            return 0.0

        value_change = values[-1] - values[0]
        total_time_hours = sum(time_diffs) / 3600

        return value_change / total_time_hours
```

#### System State and Configuration
```python
@dataclass
class SystemConfiguration:
    config_id: str
    patient_id: str
    active_modalities: List[CommunicationModality]
    bci_parameters: Dict[str, Any]
    eyetracking_parameters: Dict[str, Any]
    interface_settings: Dict[str, Any]
    safety_thresholds: Dict[str, float]
    adaptation_rules: Dict[str, Any]
    last_updated: datetime

    def update_parameters(self, modality: CommunicationModality,
                         new_parameters: Dict[str, Any]) -> None:
        """Update parameters for specific modality."""
        if modality == CommunicationModality.BCI_P300 or modality == CommunicationModality.BCI_SSVEP:
            self.bci_parameters.update(new_parameters)
        elif modality == CommunicationModality.EYE_TRACKING:
            self.eyetracking_parameters.update(new_parameters)

        self.last_updated = datetime.now()

@dataclass
class SystemState:
    state_id: str
    timestamp: datetime
    patient_id: str
    active_sessions: List[str]
    system_health: Dict[str, float]
    hardware_status: Dict[str, str]
    software_status: Dict[str, str]
    alert_conditions: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]

    def is_system_healthy(self) -> bool:
        """Check if system is in healthy operating state."""
        health_threshold = 0.8
        critical_components = ['signal_quality', 'processing_latency', 'hardware_connectivity']

        for component in critical_components:
            if component in self.system_health:
                if self.system_health[component] < health_threshold:
                    return False

        return len(self.alert_conditions) == 0
```

### Emergency and Safety Data Structures

#### Emergency Event Data
```python
class EmergencyType(Enum):
    MEDICAL_DISTRESS = "medical_distress"
    SYSTEM_FAILURE = "system_failure"
    COMMUNICATION_LOSS = "communication_loss"
    HARDWARE_MALFUNCTION = "hardware_malfunction"
    USER_UNRESPONSIVE = "user_unresponsive"

@dataclass
class EmergencyEvent:
    event_id: str
    patient_id: str
    emergency_type: EmergencyType
    severity_level: int  # 1-5, 5 being most severe
    detected_timestamp: datetime
    description: str
    automated_response_taken: List[str]
    manual_response_required: bool
    resolved_timestamp: Optional[datetime]
    resolution_notes: Optional[str]

    def get_response_time(self) -> Optional[float]:
        """Calculate emergency response time in seconds."""
        if not self.resolved_timestamp:
            return None
        return (self.resolved_timestamp - self.detected_timestamp).total_seconds()

    def is_resolved(self) -> bool:
        """Check if emergency has been resolved."""
        return self.resolved_timestamp is not None

@dataclass
class SafetyMonitoringData:
    monitoring_id: str
    patient_id: str
    start_time: datetime
    monitored_parameters: Dict[str, float]
    safety_thresholds: Dict[str, Tuple[float, float]]
    violations: List[Dict[str, Any]]
    system_interventions: List[str]
    caregiver_notifications: List[Dict[str, Any]]

    def check_safety_violations(self) -> List[str]:
        """Check for safety threshold violations."""
        violations = []

        for parameter, value in self.monitored_parameters.items():
            if parameter in self.safety_thresholds:
                min_threshold, max_threshold = self.safety_thresholds[parameter]
                if value < min_threshold:
                    violations.append(f"{parameter} below minimum threshold: {value} < {min_threshold}")
                elif value > max_threshold:
                    violations.append(f"{parameter} above maximum threshold: {value} > {max_threshold}")

        return violations
```

### Data Persistence and Serialization

#### Database Schema Models
```python
@dataclass
class DatabaseRecord:
    record_id: str
    table_name: str
    created_timestamp: datetime
    updated_timestamp: datetime
    data_checksum: str

    def calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity verification."""
        import hashlib
        import json
        data_string = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_string.encode()).hexdigest()

@dataclass
class DataExportBundle:
    export_id: str
    patient_id: str
    export_timestamp: datetime
    data_types: List[str]
    date_range: Tuple[datetime, datetime]
    anonymized: bool
    file_format: str
    compression_used: bool
    export_size_bytes: int

    def validate_export_integrity(self, file_path: str) -> bool:
        """Validate integrity of exported data file."""
        import os
        actual_size = os.path.getsize(file_path)
        return actual_size == self.export_size_bytes
```

These data structures provide a comprehensive foundation for managing all aspects of locked-in syndrome consciousness systems, from patient profiles and neural signals to communication events and safety monitoring. They support both real-time operation and long-term data analysis while maintaining data integrity and privacy.