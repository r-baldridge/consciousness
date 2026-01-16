# Form 24: Locked-in Syndrome Consciousness - API Specification

## API Overview and Design Principles

### RESTful API Architecture
The Locked-in Syndrome Consciousness API follows RESTful design principles with standardized HTTP methods, status codes, and JSON data exchange formats. All endpoints support versioning through URL path prefixes (e.g., `/api/v1/`) and include comprehensive error handling and authentication mechanisms.

### Base URL Structure
```
https://api.lis-consciousness.system/api/v1/
```

### Authentication and Security
- **Authentication Method**: OAuth 2.0 with PKCE for patient access, API keys for healthcare providers
- **Encryption**: TLS 1.3 for all communications
- **Data Protection**: HIPAA-compliant data handling with end-to-end encryption for sensitive neural data
- **Rate Limiting**: Configurable per user type (patients: 1000 req/hour, caregivers: 5000 req/hour, clinicians: 10000 req/hour)

## Core API Endpoints

### Patient Management API

#### Patient Profile Management
```http
GET /patients/{patient_id}
```
**Description**: Retrieve comprehensive patient profile including medical history and functional assessment.

**Parameters**:
- `patient_id` (path, required): Unique patient identifier
- `include_medical_history` (query, optional): Include detailed medical history (default: false)
- `include_assessments` (query, optional): Include latest functional assessments (default: true)

**Response Example**:
```json
{
  "patient_id": "lis_patient_001",
  "demographic_info": {
    "age": 45,
    "gender": "female",
    "language_preferences": ["english", "spanish"]
  },
  "functional_assessment": {
    "consciousness_level": "full",
    "cognitive_function_preserved": true,
    "communication_capabilities": ["eye_movement", "blinking"],
    "locked_in_severity": "classic",
    "last_assessment_date": "2024-03-15T10:30:00Z",
    "assessment_confidence": 0.92
  },
  "optimal_communication_modalities": ["eye_tracking", "p300_bci"],
  "last_updated": "2024-03-15T10:30:00Z"
}
```

```http
PUT /patients/{patient_id}/functional-assessment
```
**Description**: Update patient's functional assessment based on new evaluation.

**Request Body**:
```json
{
  "consciousness_level": "full",
  "cognitive_function_preserved": true,
  "communication_capabilities": ["eye_movement", "blinking", "minimal_facial"],
  "residual_motor_function": {
    "left_eyelid": 0.8,
    "right_eyelid": 0.9,
    "facial_muscles": 0.1
  },
  "assessment_notes": "Improved control over eyelid movements",
  "assessor_id": "clinician_123"
}
```

### Consciousness Detection API

#### Real-time Consciousness Assessment
```http
POST /consciousness/assess
```
**Description**: Perform real-time consciousness assessment using available modalities.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "assessment_config": {
    "methods": ["eeg_markers", "command_following", "fmri_networks"],
    "duration_minutes": 10,
    "priority": "high"
  },
  "context": {
    "time_of_day": "morning",
    "alertness_level": "normal",
    "recent_medications": []
  }
}
```

**Response Example**:
```json
{
  "assessment_id": "assess_20240315_103000",
  "consciousness_level": "full",
  "confidence": 0.94,
  "supporting_evidence": {
    "default_mode_network_integrity": 0.89,
    "command_following_accuracy": 0.95,
    "perturbational_complexity_index": 0.87,
    "eeg_consciousness_markers": 0.92
  },
  "recommendations": [
    "Consciousness clearly preserved - proceed with communication training",
    "Consider advanced BCI modalities for optimal communication"
  ],
  "assessment_duration": 8.5,
  "timestamp": "2024-03-15T10:30:00Z"
}
```

#### Consciousness Monitoring
```http
GET /consciousness/monitor/{patient_id}
```
**Description**: Retrieve consciousness monitoring data over specified time period.

**Parameters**:
- `start_time` (query, required): ISO 8601 timestamp
- `end_time` (query, required): ISO 8601 timestamp
- `granularity` (query, optional): Data granularity ("minute", "hour", "day") (default: "hour")

### Brain-Computer Interface API

#### BCI Session Management
```http
POST /bci/sessions
```
**Description**: Initialize new BCI communication session.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "paradigm": "p300_speller",
  "session_config": {
    "target_duration_minutes": 30,
    "difficulty_level": "adaptive",
    "feedback_enabled": true,
    "break_intervals": 5
  },
  "hardware_config": {
    "electrode_count": 32,
    "sampling_rate": 256,
    "impedance_threshold": 5000
  }
}
```

**Response Example**:
```json
{
  "session_id": "bci_session_20240315_103000",
  "status": "initialized",
  "calibration_required": true,
  "estimated_calibration_time": 300,
  "session_parameters": {
    "paradigm": "p300_speller",
    "stimulus_timing": 125,
    "inter_stimulus_interval": 175,
    "number_of_targets": 36
  },
  "hardware_status": "ready",
  "next_step": "start_calibration"
}
```

#### BCI Calibration
```http
POST /bci/sessions/{session_id}/calibrate
```
**Description**: Perform BCI system calibration for specific user and paradigm.

**Request Body**:
```json
{
  "calibration_type": "full",
  "target_accuracy": 0.85,
  "max_calibration_time": 600,
  "adaptive_parameters": true
}
```

#### Real-time BCI Communication
```http
POST /bci/sessions/{session_id}/communicate
```
**Description**: Process real-time neural signals for communication intent.

**Request Body**:
```json
{
  "neural_data": {
    "eeg_signals": "base64_encoded_signal_data",
    "timestamp": "2024-03-15T10:30:00.123Z",
    "channels": 32,
    "samples": 256
  },
  "context": {
    "current_interface": "speller",
    "user_focus_target": "letter_A",
    "session_time_elapsed": 180
  }
}
```

**Response Example**:
```json
{
  "communication_intent": {
    "detected_selection": "A",
    "confidence": 0.87,
    "processing_time_ms": 45
  },
  "signal_quality": {
    "overall_quality": 0.91,
    "noise_level": 0.08,
    "electrode_issues": []
  },
  "next_action": "continue_spelling",
  "performance_feedback": {
    "accuracy_trend": "stable",
    "fatigue_indicator": 0.2
  }
}
```

### Eye-Tracking Communication API

#### Eye-Tracking Session Management
```http
POST /eyetracking/sessions
```
**Description**: Initialize eye-tracking communication session.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "interface_type": "communication_board",
  "calibration_config": {
    "calibration_points": 9,
    "validation_points": 4,
    "max_calibration_time": 180
  },
  "tracking_parameters": {
    "sampling_rate": 120,
    "gaze_filter": "adaptive",
    "blink_detection": true
  }
}
```

#### Gaze-Based Selection
```http
POST /eyetracking/sessions/{session_id}/detect-selection
```
**Description**: Detect user selection through gaze patterns.

**Request Body**:
```json
{
  "gaze_data": [
    {
      "x": 0.52,
      "y": 0.38,
      "timestamp": "2024-03-15T10:30:00.123Z",
      "confidence": 0.94,
      "pupil_diameter": 3.2
    }
  ],
  "interface_targets": [
    {
      "target_id": "letter_H",
      "bounds": {"x": 0.5, "y": 0.35, "width": 0.08, "height": 0.08}
    }
  ],
  "dwell_threshold_ms": 800
}
```

### Hybrid Communication API

#### Modality Switching
```http
POST /communication/switch-modality
```
**Description**: Switch between communication modalities based on performance or user preference.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "current_modality": "eye_tracking",
  "requested_modality": "p300_bci",
  "reason": "eye_fatigue",
  "performance_context": {
    "current_accuracy": 0.72,
    "fatigue_level": 0.8,
    "session_duration": 1200
  }
}
```

#### Communication Synthesis
```http
POST /communication/synthesize
```
**Description**: Synthesize communication output from multiple input modalities.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "inputs": [
    {
      "modality": "eye_tracking",
      "content": "HELP",
      "confidence": 0.85,
      "timestamp": "2024-03-15T10:30:00Z"
    },
    {
      "modality": "bci_p300",
      "content": "HEL",
      "confidence": 0.78,
      "timestamp": "2024-03-15T10:30:01Z"
    }
  ],
  "context": {
    "conversation_history": ["Good morning", "How are you feeling?"],
    "user_preferences": {"word_prediction": true}
  }
}
```

### Performance Analytics API

#### Performance Metrics Collection
```http
POST /analytics/metrics
```
**Description**: Record performance metrics for analysis.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "metrics": [
    {
      "metric_name": "communication_accuracy",
      "value": 0.87,
      "unit": "percentage",
      "timestamp": "2024-03-15T10:30:00Z",
      "session_context": {
        "modality": "eye_tracking",
        "session_id": "et_session_123",
        "fatigue_level": 0.3
      }
    }
  ]
}
```

#### Trend Analysis
```http
GET /analytics/trends/{patient_id}
```
**Description**: Retrieve performance trend analysis over time.

**Parameters**:
- `metric_types` (query, required): Comma-separated list of metric types
- `time_range` (query, required): Time range in format "start_date,end_date"
- `granularity` (query, optional): Analysis granularity ("day", "week", "month")

**Response Example**:
```json
{
  "patient_id": "lis_patient_001",
  "analysis_period": {
    "start_date": "2024-02-15",
    "end_date": "2024-03-15",
    "granularity": "week"
  },
  "trends": [
    {
      "metric_name": "communication_accuracy",
      "trend_direction": "improving",
      "trend_strength": 0.73,
      "improvement_rate": 0.02,
      "statistical_significance": 0.95,
      "data_points": 28,
      "predictions": {
        "next_week_estimate": 0.89,
        "confidence_interval": [0.85, 0.93]
      }
    }
  ],
  "recommendations": [
    "Consider advancing to more complex BCI paradigms",
    "Maintain current training schedule for optimal progress"
  ]
}
```

### Emergency and Safety API

#### Emergency Alert System
```http
POST /emergency/alert
```
**Description**: Trigger emergency alert with automatic response protocols.

**Request Body**:
```json
{
  "patient_id": "lis_patient_001",
  "emergency_type": "medical_distress",
  "severity": 4,
  "description": "Patient indicating severe pain through emergency communication",
  "detected_by": "automated_monitoring",
  "location": "room_205_bed_A",
  "automated_actions_taken": [
    "caregiver_notification_sent",
    "backup_communication_activated"
  ]
}
```

#### Safety Monitoring
```http
GET /safety/monitoring/{patient_id}/status
```
**Description**: Get current safety monitoring status and recent alerts.

### System Configuration API

#### Configuration Management
```http
GET /config/system/{patient_id}
```
**Description**: Retrieve current system configuration for patient.

```http
PUT /config/system/{patient_id}
```
**Description**: Update system configuration parameters.

**Request Body**:
```json
{
  "active_modalities": ["eye_tracking", "p300_bci"],
  "bci_parameters": {
    "p300_paradigm": {
      "stimulus_duration": 125,
      "inter_stimulus_interval": 175,
      "number_of_sequences": 6
    }
  },
  "eyetracking_parameters": {
    "dwell_time_threshold": 800,
    "gaze_smoothing": "medium",
    "blink_compensation": true
  },
  "safety_thresholds": {
    "max_session_duration": 1800,
    "fatigue_threshold": 0.7,
    "accuracy_minimum": 0.6
  }
}
```

## WebSocket API for Real-time Communication

### Real-time Data Streaming
```javascript
// WebSocket connection for real-time neural signal streaming
const ws = new WebSocket('wss://api.lis-consciousness.system/ws/v1/realtime');

// Message format for real-time neural data
{
  "type": "neural_data",
  "patient_id": "lis_patient_001",
  "session_id": "bci_session_123",
  "timestamp": "2024-03-15T10:30:00.123Z",
  "data": {
    "eeg_channels": [/* channel data */],
    "signal_quality": 0.91,
    "processing_latency": 23
  }
}

// Real-time communication event
{
  "type": "communication_event",
  "patient_id": "lis_patient_001",
  "content": "H",
  "confidence": 0.87,
  "modality": "eye_tracking",
  "timestamp": "2024-03-15T10:30:00.123Z"
}
```

## Error Handling and Status Codes

### Standard HTTP Status Codes
- `200 OK`: Successful request
- `201 Created`: Resource successfully created
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (e.g., session already active)
- `422 Unprocessable Entity`: Valid request but unable to process
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_PATIENT_ID",
    "message": "The specified patient ID does not exist or is not accessible",
    "details": {
      "patient_id": "invalid_id_123",
      "suggestion": "Verify patient ID and access permissions"
    },
    "timestamp": "2024-03-15T10:30:00Z",
    "request_id": "req_abc123def456"
  }
}
```

## Rate Limiting and Quotas

### Rate Limiting Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1647345000
X-RateLimit-Retry-After: 3600
```

### Usage Quotas by User Type
- **Patients**: 1,000 requests/hour, 10 concurrent sessions
- **Caregivers**: 5,000 requests/hour, 25 concurrent sessions
- **Clinicians**: 10,000 requests/hour, 50 concurrent sessions
- **Researchers**: 25,000 requests/hour, 100 concurrent sessions

This API specification provides comprehensive endpoints for all aspects of locked-in syndrome consciousness systems, enabling secure, real-time communication and monitoring while maintaining HIPAA compliance and optimal user experience.