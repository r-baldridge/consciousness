# Somatosensory Consciousness System - API Specification

**Document**: Application Programming Interface Specification
**Form**: 03 - Somatosensory Consciousness
**Category**: Technical Specification & Design
**Version**: 1.0
**Date**: 2025-09-26

## Executive Summary

This document defines the comprehensive API specification for the Somatosensory Consciousness System, providing programmatic interfaces for tactile, thermal, pain, and proprioceptive consciousness generation, configuration, and integration with external systems.

## API Architecture Overview

### Base API Structure
```
/api/v1/somatosensory/
├── /tactile/           # Tactile consciousness endpoints
├── /thermal/           # Thermal consciousness endpoints
├── /pain/              # Pain consciousness endpoints (with safety protocols)
├── /proprioceptive/    # Proprioceptive consciousness endpoints
├── /integration/       # Cross-modal integration endpoints
├── /safety/            # Safety monitoring and control endpoints
├── /configuration/     # System configuration endpoints
└── /monitoring/        # Real-time monitoring endpoints
```

### Authentication and Security
```python
# API Authentication Requirements
class SomatosensoryAPIAuth:
    def __init__(self):
        self.auth_methods = ['API_KEY', 'JWT_TOKEN', 'OAUTH2']
        self.safety_clearance_levels = ['USER', 'RESEARCHER', 'CLINICAL', 'ADMIN']
        self.pain_access_restrictions = True
        self.thermal_safety_validation = True

    def validate_safety_clearance(self, endpoint: str, user_level: str) -> bool:
        pain_endpoints = ['/pain/', '/thermal/extreme/']
        if any(pe in endpoint for pe in pain_endpoints):
            return user_level in ['CLINICAL', 'ADMIN']
        return True
```

## Core API Endpoints

### 1. Tactile Consciousness API

#### POST /api/v1/somatosensory/tactile/stimulate
**Description**: Generate tactile consciousness experiences
**Authorization**: USER level required

```python
# Request Schema
class TactileStimulationRequest:
    body_region: str              # "fingertip", "palm", "forearm", etc.
    stimulus_type: str            # "pressure", "texture", "vibration", "touch"
    intensity: float              # 0.0 to 1.0
    duration_ms: int              # Milliseconds
    spatial_pattern: List[dict]   # Spatial distribution of stimuli
    temporal_pattern: dict        # Timing parameters

# Example Request
{
    "body_region": "fingertip",
    "stimulus_type": "texture",
    "intensity": 0.7,
    "duration_ms": 2000,
    "spatial_pattern": [
        {"x": 0, "y": 0, "radius": 5, "intensity": 0.8},
        {"x": 10, "y": 5, "radius": 3, "intensity": 0.6}
    ],
    "temporal_pattern": {
        "onset_ms": 0,
        "peak_ms": 500,
        "offset_ms": 2000,
        "adaptation_curve": "exponential"
    }
}

# Response Schema
class TactileConsciousnessResponse:
    session_id: str
    consciousness_id: str
    tactile_experience: dict
    quality_metrics: dict
    safety_status: str

# Example Response
{
    "session_id": "sess_12345",
    "consciousness_id": "tactile_67890",
    "tactile_experience": {
        "touch_quality": "smooth_texture",
        "pressure_awareness": 0.7,
        "spatial_localization": {"x": 5, "y": 2.5, "confidence": 0.95},
        "temporal_dynamics": {"onset": 15, "peak": 485, "offset": 1998}
    },
    "quality_metrics": {
        "realism_score": 0.92,
        "user_acceptance": 0.88,
        "latency_ms": 12
    },
    "safety_status": "SAFE"
}
```

#### GET /api/v1/somatosensory/tactile/status/{session_id}
**Description**: Get current tactile consciousness state
**Authorization**: USER level required

```python
# Response Schema
{
    "session_id": "sess_12345",
    "active_stimuli": [
        {
            "consciousness_id": "tactile_67890",
            "body_region": "fingertip",
            "intensity": 0.7,
            "remaining_duration_ms": 1500,
            "current_quality": "smooth_texture"
        }
    ],
    "tactile_sensitivity": {
        "fingertip": 0.9,
        "palm": 0.8,
        "forearm": 0.6
    },
    "processing_load": 0.3
}
```

### 2. Thermal Consciousness API

#### POST /api/v1/somatosensory/thermal/stimulate
**Description**: Generate thermal consciousness experiences with safety protocols
**Authorization**: USER level required, CLINICAL for extreme temperatures

```python
# Request Schema
class ThermalStimulationRequest:
    body_region: str         # Target body region
    temperature_celsius: float  # 5.0 to 45.0 (safety limited)
    duration_ms: int         # Duration in milliseconds
    thermal_gradient: dict   # Spatial temperature distribution
    comfort_override: bool   # Allow uncomfortable temperatures (requires CLINICAL auth)

# Example Request
{
    "body_region": "palm",
    "temperature_celsius": 35.0,
    "duration_ms": 5000,
    "thermal_gradient": {
        "center_temp": 35.0,
        "edge_temp": 32.0,
        "gradient_type": "radial"
    },
    "comfort_override": false
}

# Response Schema
{
    "session_id": "sess_12345",
    "consciousness_id": "thermal_11111",
    "thermal_experience": {
        "temperature_consciousness": "warm",
        "thermal_comfort": 0.6,
        "spatial_distribution": {"center": 35.0, "periphery": 32.5},
        "adaptation_progress": 0.2
    },
    "safety_validation": {
        "temperature_safe": true,
        "duration_approved": true,
        "user_comfort_acceptable": true
    }
}
```

#### POST /api/v1/somatosensory/thermal/emergency_cool
**Description**: Emergency thermal consciousness termination
**Authorization**: USER level required

```python
# Request Schema
{
    "session_id": "sess_12345",
    "consciousness_id": "thermal_11111",
    "emergency_reason": "user_discomfort"
}

# Response Schema
{
    "termination_status": "SUCCESS",
    "termination_time_ms": 95,
    "post_stimulus_monitoring": true,
    "safety_log_entry": "log_entry_id_789"
}
```

### 3. Pain Consciousness API (High Security)

#### POST /api/v1/somatosensory/pain/controlled_stimulate
**Description**: Generate controlled pain consciousness for research/clinical use
**Authorization**: CLINICAL or ADMIN level required

```python
# Request Schema
class PainStimulationRequest:
    body_region: str           # Target region
    pain_type: str             # "acute", "chronic", "therapeutic"
    intensity: float           # 0.0 to MAX_SAFE_INTENSITY (typically 7.0/10.0)
    duration_ms: int           # Maximum duration with safety limits
    clinical_justification: str # Required justification
    informed_consent_id: str   # Consent record ID
    monitoring_frequency_ms: int # Safety monitoring interval

# Example Request
{
    "body_region": "forearm",
    "pain_type": "therapeutic",
    "intensity": 4.0,
    "duration_ms": 30000,
    "clinical_justification": "Pain threshold research study IRB#2025-001",
    "informed_consent_id": "consent_abc123",
    "monitoring_frequency_ms": 1000
}

# Response Schema
{
    "session_id": "sess_12345",
    "consciousness_id": "pain_22222",
    "pain_experience": {
        "sensory_component": 4.0,
        "affective_component": 3.2,
        "pain_quality": "dull_aching",
        "localization_accuracy": 0.9
    },
    "safety_monitoring": {
        "max_duration_remaining_ms": 29500,
        "intensity_monitoring": "active",
        "emergency_stop_available": true,
        "consent_verified": true
    },
    "clinical_data": {
        "session_logged": true,
        "irb_compliance": "verified",
        "data_anonymization": "active"
    }
}
```

#### POST /api/v1/somatosensory/pain/emergency_stop
**Description**: Immediate pain consciousness termination
**Authorization**: USER level required (emergency endpoint)

```python
# Request Schema
{
    "session_id": "sess_12345",
    "consciousness_id": "pain_22222",
    "emergency_type": "USER_REQUEST"  # or "SAFETY_THRESHOLD", "SYSTEM_ERROR"
}

# Response Schema
{
    "termination_status": "IMMEDIATE_SUCCESS",
    "termination_time_ms": 45,
    "pain_cessation_confirmed": true,
    "post_care_monitoring": true,
    "incident_report_id": "incident_789"
}
```

### 4. Proprioceptive Consciousness API

#### POST /api/v1/somatosensory/proprioceptive/update_body_state
**Description**: Update body position consciousness
**Authorization**: USER level required

```python
# Request Schema
class ProprioceptiveUpdateRequest:
    joint_positions: dict        # Joint angles in degrees
    movement_velocities: dict    # Angular velocities
    body_orientation: dict       # Overall body orientation
    update_mode: str             # "continuous", "discrete", "prediction"

# Example Request
{
    "joint_positions": {
        "shoulder_flexion": 45.0,
        "elbow_flexion": 90.0,
        "wrist_flexion": 15.0,
        "finger_positions": [10, 15, 20, 25, 30]
    },
    "movement_velocities": {
        "shoulder_velocity": 2.5,
        "elbow_velocity": 0.0,
        "wrist_velocity": 1.2
    },
    "body_orientation": {
        "head_pitch": 0,
        "head_yaw": 15,
        "trunk_tilt": 5
    },
    "update_mode": "continuous"
}

# Response Schema
{
    "session_id": "sess_12345",
    "consciousness_id": "proprio_33333",
    "proprioceptive_experience": {
        "joint_position_consciousness": {
            "shoulder": {"angle": 45.0, "confidence": 0.95},
            "elbow": {"angle": 90.0, "confidence": 0.98},
            "wrist": {"angle": 15.0, "confidence": 0.92}
        },
        "movement_awareness": {
            "active_joints": ["shoulder", "wrist"],
            "movement_quality": "smooth",
            "coordination_assessment": 0.88
        },
        "body_schema_update": {
            "schema_confidence": 0.94,
            "spatial_accuracy": 0.91,
            "ownership_assessment": 0.97
        }
    },
    "processing_metrics": {
        "update_latency_ms": 8,
        "integration_success": true,
        "prediction_accuracy": 0.93
    }
}
```

### 5. Integration API

#### POST /api/v1/somatosensory/integration/cross_modal
**Description**: Integrate somatosensory consciousness with other modalities
**Authorization**: USER level required

```python
# Request Schema
class CrossModalIntegrationRequest:
    primary_modality: str        # "tactile", "thermal", "pain", "proprioceptive"
    secondary_modalities: List[str]  # Other modalities to integrate
    integration_weights: dict    # Relative importance of each modality
    temporal_alignment: dict     # Synchronization parameters

# Example Request
{
    "primary_modality": "tactile",
    "secondary_modalities": ["visual", "auditory"],
    "integration_weights": {
        "tactile": 0.6,
        "visual": 0.3,
        "auditory": 0.1
    },
    "temporal_alignment": {
        "synchronization_window_ms": 50,
        "prediction_compensation": true
    }
}

# Response Schema
{
    "integration_id": "integration_44444",
    "integrated_experience": {
        "unified_object_representation": {
            "tactile_properties": {"texture": "rough", "hardness": 0.8},
            "visual_properties": {"color": "red", "shape": "cylindrical"},
            "spatial_alignment": 0.94
        },
        "cross_modal_enhancement": {
            "tactile_enhancement": 1.15,
            "recognition_improvement": 1.32,
            "confidence_boost": 0.23
        }
    },
    "integration_quality": {
        "temporal_sync_accuracy": 0.96,
        "spatial_alignment_score": 0.91,
        "phenomenological_coherence": 0.89
    }
}
```

### 6. Safety and Monitoring API

#### GET /api/v1/somatosensory/safety/status
**Description**: Get comprehensive safety status
**Authorization**: USER level required

```python
# Response Schema
{
    "overall_safety_status": "SAFE",
    "active_safety_monitors": [
        {
            "monitor_type": "pain_intensity",
            "status": "active",
            "threshold": 7.0,
            "current_max": 4.2
        },
        {
            "monitor_type": "thermal_safety",
            "status": "active",
            "safe_range": [5.0, 45.0],
            "current_range": [22.0, 35.0]
        }
    ],
    "emergency_protocols": {
        "pain_emergency_stop": "armed",
        "thermal_emergency_cool": "armed",
        "session_termination": "available"
    },
    "user_controls": {
        "intensity_reduction": "available",
        "modality_disabling": "available",
        "session_pause": "available"
    }
}
```

#### POST /api/v1/somatosensory/safety/configure_limits
**Description**: Configure personal safety limits
**Authorization**: USER level required

```python
# Request Schema
{
    "pain_limits": {
        "max_intensity": 6.0,
        "max_duration_ms": 10000,
        "require_confirmation_above": 4.0
    },
    "thermal_limits": {
        "min_temperature": 10.0,
        "max_temperature": 40.0,
        "comfort_range": [18.0, 32.0]
    },
    "tactile_sensitivity": {
        "pressure_sensitivity": 0.8,
        "vibration_sensitivity": 0.9,
        "texture_sensitivity": 0.7
    }
}

# Response Schema
{
    "configuration_updated": true,
    "active_limits": {
        "pain_protection": "enhanced",
        "thermal_protection": "standard",
        "tactile_calibration": "personalized"
    },
    "safety_validation": "passed"
}
```

### 7. Configuration and Calibration API

#### POST /api/v1/somatosensory/configuration/calibrate
**Description**: Perform individual calibration
**Authorization**: USER level required

```python
# Request Schema
{
    "calibration_type": "full",  # "full", "tactile_only", "thermal_only", etc.
    "user_characteristics": {
        "age": 30,
        "dominant_hand": "right",
        "sensitivity_profile": "normal",
        "medical_conditions": []
    },
    "calibration_preferences": {
        "comfort_priority": 0.8,
        "accuracy_priority": 0.6,
        "safety_conservatism": 0.9
    }
}

# Response Schema
{
    "calibration_id": "cal_55555",
    "calibration_results": {
        "tactile_thresholds": {
            "pressure_threshold": 0.15,
            "vibration_threshold": 0.08,
            "texture_discrimination": 0.92
        },
        "thermal_calibration": {
            "comfort_zone": [20.0, 35.0],
            "detection_threshold": 0.8,
            "adaptation_rate": 1.2
        },
        "proprioceptive_accuracy": {
            "joint_position_error": 2.1,
            "movement_detection": 0.95,
            "body_schema_confidence": 0.91
        }
    },
    "personalization_active": true
}
```

## Streaming and Real-Time APIs

### WebSocket Connection for Real-Time Consciousness

```python
# WebSocket Endpoint: /ws/somatosensory/realtime
class SomatosensoryWebSocket:
    def on_connect(self, websocket, path):
        # Authentication and session initialization
        pass

    def on_message(self, websocket, message):
        # Real-time consciousness commands
        {
            "command": "update_tactile",
            "data": {
                "body_region": "fingertip",
                "pressure": 0.6,
                "timestamp": 1640995200000
            }
        }

    def on_consciousness_update(self, consciousness_data):
        # Real-time consciousness state broadcast
        {
            "type": "consciousness_update",
            "modality": "tactile",
            "experience": {
                "touch_quality": "smooth",
                "intensity": 0.6,
                "confidence": 0.95
            },
            "timestamp": 1640995200010
        }
```

## Error Handling and Status Codes

### HTTP Status Codes
- `200 OK`: Successful consciousness generation
- `400 Bad Request`: Invalid stimulation parameters
- `401 Unauthorized`: Insufficient safety clearance
- `403 Forbidden`: Safety protocol violation
- `429 Too Many Requests`: Rate limiting for safety
- `500 Internal Server Error`: System processing error
- `503 Service Unavailable`: Safety system override

### Error Response Schema
```python
{
    "error": {
        "code": "SAFETY_VIOLATION",
        "message": "Pain intensity exceeds user safety limits",
        "details": {
            "requested_intensity": 8.5,
            "user_max_intensity": 6.0,
            "safety_recommendation": "Reduce intensity or obtain clinical authorization"
        }
    },
    "timestamp": "2025-09-26T10:00:00Z",
    "request_id": "req_12345"
}
```

## Rate Limiting and Usage Quotas

### Rate Limits by Endpoint Type
- **Tactile stimulation**: 100 requests/minute
- **Thermal stimulation**: 20 requests/minute
- **Pain stimulation**: 5 requests/minute (CLINICAL auth)
- **Status queries**: 1000 requests/minute
- **Safety controls**: Unlimited (emergency use)

This API specification enables comprehensive, safe, and research-grade access to somatosensory consciousness capabilities while maintaining strict safety protocols and user control mechanisms.