# Interoceptive Consciousness System - API Specification

**Document**: API Specification
**Form**: 06 - Interoceptive Consciousness
**Category**: Technical Specification & Design
**Version**: 1.0
**Date**: 2025-09-27

## Executive Summary

This document defines the comprehensive API specification for the Interoceptive Consciousness System, including RESTful web APIs, WebSocket interfaces, and integration protocols for physiological monitoring, consciousness generation, and external system integration.

## API Architecture Overview

### Base URL Structure
```
Production: https://api.interoceptive-consciousness.com/v1
Staging: https://staging-api.interoceptive-consciousness.com/v1
Development: https://dev-api.interoceptive-consciousness.com/v1
```

### Authentication and Authorization
```
Authentication: Bearer JWT tokens
Authorization: Role-based access control (RBAC)
Rate Limiting: 1000 requests/minute per user
API Versioning: URL path versioning
```

## Core API Endpoints

### 1. Cardiovascular Consciousness API

#### Get Cardiovascular State
```http
GET /cardiovascular/state/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "timestamp": "2025-09-27T10:30:00Z",
  "heartbeat_awareness": {
    "intensity": 0.7,
    "clarity": 0.8,
    "confidence": 0.85
  },
  "heart_rate_bpm": 72,
  "hrv_metrics": {
    "rmssd_ms": 45.2,
    "sdnn_ms": 52.1,
    "lf_hf_ratio": 1.2
  },
  "cardiac_comfort": 8.5,
  "consciousness_quality": 0.82
}
```

#### Stream Cardiovascular Consciousness
```http
WebSocket: /cardiovascular/stream/{user_id}
Authorization: Bearer {token}

Message Format:
{
  "type": "cardiovascular_update",
  "data": {
    "timestamp": "2025-09-27T10:30:00Z",
    "heartbeat_detected": true,
    "rr_interval_ms": 833,
    "consciousness_intensity": 0.75
  }
}
```

#### Configure Cardiovascular Monitoring
```http
POST /cardiovascular/configure
Authorization: Bearer {token}
Content-Type: application/json

{
  "user_id": "user_123",
  "sensitivity_level": 0.8,
  "attention_modulation": true,
  "training_mode": {
    "enabled": true,
    "protocol": "heartbeat_counting",
    "duration_minutes": 10
  }
}

Response 201:
{
  "configuration_id": "config_456",
  "status": "active",
  "message": "Cardiovascular monitoring configured successfully"
}
```

### 2. Respiratory Consciousness API

#### Get Respiratory State
```http
GET /respiratory/state/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "timestamp": "2025-09-27T10:30:00Z",
  "breathing_awareness": {
    "pattern_consciousness": "normal",
    "effort_awareness": 0.3,
    "rhythm_awareness": 0.85
  },
  "respiratory_rate_bpm": 16,
  "tidal_volume_ml": 500,
  "respiratory_comfort": 9.0,
  "breath_control_consciousness": 0.7
}
```

#### Breathing Pattern Analysis
```http
GET /respiratory/pattern-analysis/{user_id}
Authorization: Bearer {token}
Query Parameters:
  - window_minutes: 5 (default)
  - include_variability: true

Response 200:
{
  "analysis_window": {
    "start": "2025-09-27T10:25:00Z",
    "end": "2025-09-27T10:30:00Z"
  },
  "pattern_classification": "eupnea",
  "pattern_confidence": 0.92,
  "breathing_variability": {
    "rate_variability": 0.15,
    "depth_variability": 0.12,
    "rhythm_stability": 0.88
  },
  "consciousness_indicators": {
    "voluntary_control_episodes": 3,
    "attention_to_breathing": 0.4
  }
}
```

### 3. Gastrointestinal Consciousness API

#### Get Digestive State
```http
GET /gastrointestinal/state/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "timestamp": "2025-09-27T10:30:00Z",
  "hunger_satiety": {
    "hunger_level": 6.5,
    "satiety_level": 2.0,
    "appetite_quality": "normal"
  },
  "gastric_awareness": {
    "fullness_sensation": 0.2,
    "comfort_level": 8.0,
    "motility_consciousness": 0.3
  },
  "hormone_levels": {
    "ghrelin_pg_ml": 650,
    "leptin_ng_ml": 8.2,
    "cck_pg_ml": 15
  }
}
```

#### Food Intake Tracking
```http
POST /gastrointestinal/food-intake
Authorization: Bearer {token}
Content-Type: application/json

{
  "user_id": "user_123",
  "meal_details": {
    "meal_type": "lunch",
    "food_items": [
      {"name": "grilled_chicken", "quantity_g": 150},
      {"name": "brown_rice", "quantity_g": 100}
    ],
    "eating_rate": "normal",
    "satisfaction_score": 8.5
  }
}

Response 201:
{
  "intake_id": "intake_789",
  "predicted_satiety_curve": [
    {"time_minutes": 0, "satiety": 2.0},
    {"time_minutes": 15, "satiety": 7.5},
    {"time_minutes": 30, "satiety": 8.5}
  ]
}
```

### 4. Thermoregulatory API

#### Get Thermal State
```http
GET /thermoregulatory/state/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "timestamp": "2025-09-27T10:30:00Z",
  "core_temperature_celsius": 37.1,
  "thermal_comfort": {
    "sensation_score": 0.5,
    "comfort_score": 8.0,
    "preference": "slightly_cooler"
  },
  "thermal_responses": {
    "sweating_rate": 0.2,
    "vasoconstriction": 0.1,
    "shivering_intensity": 0.0
  },
  "environmental_factors": {
    "ambient_temp_celsius": 22.5,
    "humidity_percent": 45,
    "air_velocity_ms": 0.1
  }
}
```

### 5. Homeostatic Consciousness API

#### Get Homeostatic Balance
```http
GET /homeostatic/balance/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "timestamp": "2025-09-27T10:30:00Z",
  "hydration_status": {
    "thirst_level": 3.5,
    "hydration_balance": 0.1,
    "urine_concentration": "normal"
  },
  "energy_status": {
    "energy_level": 7.5,
    "fatigue_level": 2.0,
    "sleep_pressure": 3.0
  },
  "stress_indicators": {
    "cortisol_level": "normal",
    "stress_score": 2.5,
    "recovery_status": 0.8
  }
}
```

## Integrated Consciousness APIs

### 1. Unified Interoceptive State
```http
GET /consciousness/integrated-state/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "timestamp": "2025-09-27T10:30:00Z",
  "unified_state": {
    "overall_interoceptive_awareness": 0.75,
    "cross_modal_coherence": 0.82,
    "homeostatic_balance": 0.88
  },
  "modality_contributions": {
    "cardiovascular": 0.8,
    "respiratory": 0.7,
    "gastrointestinal": 0.6,
    "thermoregulatory": 0.9
  },
  "somatic_markers": {
    "gut_feeling_strength": 0.6,
    "decision_confidence": 0.75,
    "emotional_body_connection": 0.7
  }
}
```

### 2. Consciousness Training API
```http
POST /consciousness/training/session
Authorization: Bearer {token}
Content-Type: application/json

{
  "user_id": "user_123",
  "training_type": "heartbeat_awareness",
  "session_config": {
    "duration_minutes": 15,
    "feedback_mode": "real_time",
    "difficulty_level": "intermediate"
  }
}

Response 201:
{
  "session_id": "training_session_101",
  "status": "started",
  "real_time_endpoint": "/consciousness/training/stream/training_session_101"
}
```

## Real-Time Streaming APIs

### WebSocket Connection for Real-Time Monitoring
```javascript
// WebSocket connection example
const ws = new WebSocket('wss://api.interoceptive-consciousness.com/v1/stream');

ws.onopen = function(event) {
  // Subscribe to user's interoceptive stream
  ws.send(JSON.stringify({
    type: 'subscribe',
    user_id: 'user_123',
    modalities: ['cardiovascular', 'respiratory'],
    update_frequency_hz: 10
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  // Handle real-time interoceptive consciousness updates
  console.log('Consciousness update:', data);
};
```

## Safety and Emergency APIs

### 1. Safety Monitoring
```http
GET /safety/status/{user_id}
Authorization: Bearer {token}

Response 200:
{
  "user_id": "user_123",
  "safety_level": "normal",
  "active_alerts": [],
  "physiological_bounds": {
    "heart_rate": {"current": 72, "safe_range": [40, 200]},
    "core_temperature": {"current": 37.1, "safe_range": [35.0, 39.0]}
  },
  "last_safety_check": "2025-09-27T10:29:55Z"
}
```

### 2. Emergency Response
```http
POST /safety/emergency-stop
Authorization: Bearer {token}
Content-Type: application/json

{
  "user_id": "user_123",
  "emergency_type": "user_request",
  "reason": "feeling_unwell"
}

Response 200:
{
  "emergency_id": "emergency_456",
  "status": "all_systems_stopped",
  "timestamp": "2025-09-27T10:30:00Z",
  "response_time_ms": 50
}
```

## Configuration and Personalization APIs

### 1. User Profile Configuration
```http
PUT /users/{user_id}/interoceptive-profile
Authorization: Bearer {token}
Content-Type: application/json

{
  "sensitivity_preferences": {
    "cardiovascular": 0.8,
    "respiratory": 0.6,
    "gastrointestinal": 0.7,
    "thermoregulatory": 0.9
  },
  "training_goals": [
    "improve_heartbeat_awareness",
    "reduce_anxiety_through_breathing"
  ],
  "medical_conditions": [
    "mild_hypertension"
  ],
  "comfort_thresholds": {
    "maximum_discomfort_level": 3.0,
    "preferred_training_intensity": 0.7
  }
}

Response 200:
{
  "profile_updated": true,
  "calibration_recommended": true,
  "message": "Profile updated successfully"
}
```

## Analytics and Reporting APIs

### 1. Historical Analysis
```http
GET /analytics/trends/{user_id}
Authorization: Bearer {token}
Query Parameters:
  - start_date: 2025-09-20
  - end_date: 2025-09-27
  - metrics: heartbeat_awareness,stress_levels

Response 200:
{
  "user_id": "user_123",
  "analysis_period": {
    "start": "2025-09-20T00:00:00Z",
    "end": "2025-09-27T23:59:59Z"
  },
  "trends": {
    "heartbeat_awareness": {
      "average": 0.72,
      "trend": "improving",
      "weekly_change": 0.08
    },
    "stress_levels": {
      "average": 3.2,
      "trend": "stable",
      "correlation_with_awareness": -0.45
    }
  }
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "INVALID_USER_ID",
    "message": "The specified user ID does not exist",
    "details": {
      "user_id": "invalid_user",
      "timestamp": "2025-09-27T10:30:00Z"
    },
    "help_url": "https://docs.interoceptive-consciousness.com/errors/invalid_user_id"
  }
}
```

### Common Error Codes
- `AUTHENTICATION_REQUIRED` (401)
- `AUTHORIZATION_DENIED` (403)
- `USER_NOT_FOUND` (404)
- `INVALID_SENSOR_DATA` (422)
- `SAFETY_THRESHOLD_EXCEEDED` (423)
- `RATE_LIMIT_EXCEEDED` (429)
- `INTERNAL_SERVER_ERROR` (500)

This comprehensive API specification provides robust interfaces for all aspects of interoceptive consciousness monitoring, training, and integration while maintaining high standards of safety, security, and usability.