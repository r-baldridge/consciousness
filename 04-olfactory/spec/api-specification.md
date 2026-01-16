# Olfactory Consciousness System - API Specification

**Document**: Application Programming Interface Specification
**Form**: 04 - Olfactory Consciousness
**Category**: Technical Specification & Design
**Version**: 1.0
**Date**: 2025-09-26

## Executive Summary

This document defines the comprehensive API specification for the Olfactory Consciousness System, providing programmatic interfaces for chemical detection, scent recognition, memory integration, emotional response generation, and consciousness experience creation with full support for cultural adaptation and personalization.

## API Architecture Overview

### Base API Structure
```
/api/v1/olfactory/
├── /chemical/          # Chemical detection and analysis endpoints
├── /scent/             # Scent recognition and classification endpoints
├── /memory/            # Memory integration endpoints
├── /emotion/           # Emotional response endpoints
├── /consciousness/     # Consciousness generation endpoints
├── /integration/       # Cross-modal integration endpoints
├── /culture/           # Cultural adaptation endpoints
├── /personalization/   # User personalization endpoints
├── /safety/            # Safety monitoring endpoints
└── /monitoring/        # Real-time system monitoring endpoints
```

### Authentication and Security
```python
# API Authentication Requirements
class OlfactoryAPIAuth:
    def __init__(self):
        self.auth_methods = ['API_KEY', 'JWT_TOKEN', 'OAUTH2']
        self.access_levels = ['PUBLIC', 'PERSONAL', 'RESEARCH', 'CLINICAL', 'ADMIN']
        self.privacy_protection = True
        self.cultural_sensitivity = True

    def validate_access_level(self, endpoint: str, user_level: str) -> bool:
        sensitive_endpoints = ['/memory/personal/', '/emotion/therapeutic/', '/personalization/']
        if any(se in endpoint for se in sensitive_endpoints):
            return user_level in ['PERSONAL', 'CLINICAL', 'ADMIN']
        return True
```

## Core API Endpoints

### 1. Chemical Detection API

#### POST /api/v1/olfactory/chemical/detect
**Description**: Detect and analyze chemical composition of air samples
**Authorization**: PUBLIC level required

```python
# Request Schema
class ChemicalDetectionRequest:
    sample_id: str                    # Unique sample identifier
    detection_mode: str               # "real_time", "batch", "high_sensitivity"
    concentration_range: tuple        # (min, max) concentration range
    target_molecules: List[str]       # Specific molecules to detect (optional)
    detection_threshold: float        # Sensitivity threshold
    temporal_window_ms: int           # Time window for detection

# Example Request
{
    "sample_id": "air_sample_001",
    "detection_mode": "real_time",
    "concentration_range": [1e-12, 1e-3],
    "target_molecules": ["limonene", "vanillin", "geraniol"],
    "detection_threshold": 1e-9,
    "temporal_window_ms": 1000
}

# Response Schema
class ChemicalDetectionResponse:
    sample_id: str
    detection_timestamp: str
    detected_molecules: List[dict]
    total_concentration: float
    detection_confidence: float
    processing_time_ms: int

# Example Response
{
    "sample_id": "air_sample_001",
    "detection_timestamp": "2025-09-26T10:00:00Z",
    "detected_molecules": [
        {
            "molecule": "limonene",
            "concentration": 5.2e-9,
            "confidence": 0.95,
            "chemical_formula": "C10H16",
            "molecular_weight": 136.23
        },
        {
            "molecule": "vanillin",
            "concentration": 2.1e-10,
            "confidence": 0.88,
            "chemical_formula": "C8H8O3",
            "molecular_weight": 152.15
        }
    ],
    "total_concentration": 5.41e-9,
    "detection_confidence": 0.92,
    "processing_time_ms": 45
}
```

#### GET /api/v1/olfactory/chemical/analyze/{sample_id}
**Description**: Get detailed chemical analysis of detected sample
**Authorization**: PUBLIC level required

```python
# Response Schema
{
    "sample_id": "air_sample_001",
    "molecular_composition": {
        "primary_components": [
            {"molecule": "limonene", "percentage": 89.2},
            {"molecule": "vanillin", "percentage": 10.8}
        ],
        "trace_components": [
            {"molecule": "linalool", "percentage": 0.02}
        ]
    },
    "chemical_properties": {
        "volatility": "high",
        "solubility": "lipophilic",
        "stability": "stable",
        "reactivity": "low"
    },
    "safety_assessment": {
        "toxicity_level": "safe",
        "allergen_potential": "low",
        "exposure_limits": {
            "safe_concentration": 1e-6,
            "maximum_exposure_time": 3600
        }
    }
}
```

### 2. Scent Recognition API

#### POST /api/v1/olfactory/scent/recognize
**Description**: Recognize and classify scent patterns from chemical data
**Authorization**: PUBLIC level required

```python
# Request Schema
class ScentRecognitionRequest:
    chemical_data: dict               # Chemical detection data
    recognition_mode: str             # "fast", "accurate", "comprehensive"
    cultural_context: str             # Cultural classification context
    personal_profile_id: str          # Personal scent profile (optional)

# Example Request
{
    "chemical_data": {
        "detected_molecules": [
            {"molecule": "limonene", "concentration": 5.2e-9},
            {"molecule": "citral", "concentration": 1.8e-9}
        ]
    },
    "recognition_mode": "accurate",
    "cultural_context": "western",
    "personal_profile_id": "user_12345"
}

# Response Schema
class ScentRecognitionResponse:
    scent_id: str
    primary_scent: dict
    secondary_scents: List[dict]
    odor_categories: List[str]
    cultural_associations: dict
    confidence_score: float

# Example Response
{
    "scent_id": "scent_rec_001",
    "primary_scent": {
        "name": "lemon",
        "category": "citrus",
        "confidence": 0.92,
        "hedonic_value": 0.8,
        "intensity": 0.7
    },
    "secondary_scents": [
        {
            "name": "lime",
            "category": "citrus",
            "confidence": 0.65,
            "contribution": 0.3
        }
    ],
    "odor_categories": ["citrus", "fresh", "clean"],
    "cultural_associations": {
        "western": ["freshness", "cleanliness", "energy"],
        "mediterranean": ["cooking", "natural", "outdoor"]
    },
    "confidence_score": 0.89
}
```

#### GET /api/v1/olfactory/scent/profile/{scent_id}
**Description**: Get detailed scent profile information
**Authorization**: PUBLIC level required

```python
# Response Schema
{
    "scent_id": "scent_rec_001",
    "scent_profile": {
        "molecular_signature": {
            "key_molecules": ["limonene", "citral", "pinene"],
            "molecular_weights": [136.23, 152.24, 136.23],
            "functional_groups": ["terpene", "aldehyde"]
        },
        "perceptual_qualities": {
            "brightness": 0.9,
            "freshness": 0.85,
            "sweetness": 0.4,
            "intensity": 0.7,
            "complexity": 0.3
        },
        "emotional_associations": {
            "positive_emotions": ["joy", "energy", "cleanliness"],
            "negative_emotions": [],
            "arousal_level": 0.6,
            "valence": 0.8
        }
    }
}
```

### 3. Memory Integration API

#### POST /api/v1/olfactory/memory/associate
**Description**: Create or retrieve memory associations with scents
**Authorization**: PERSONAL level required

```python
# Request Schema
class MemoryAssociationRequest:
    scent_data: dict                  # Scent recognition data
    user_id: str                      # User identifier
    context_information: dict        # Current context
    association_type: str             # "retrieve", "create", "update"

# Example Request
{
    "scent_data": {
        "primary_scent": "vanilla",
        "confidence": 0.92
    },
    "user_id": "user_12345",
    "context_information": {
        "location": "kitchen",
        "time_of_day": "evening",
        "activity": "cooking"
    },
    "association_type": "retrieve"
}

# Response Schema
class MemoryAssociationResponse:
    association_id: str
    triggered_memories: List[dict]
    emotional_responses: dict
    autobiographical_content: dict
    memory_vividness: float

# Example Response
{
    "association_id": "mem_assoc_001",
    "triggered_memories": [
        {
            "memory_id": "mem_001",
            "description": "Grandmother's kitchen during holidays",
            "time_period": "childhood",
            "emotional_intensity": 0.9,
            "vividness": 0.85,
            "confidence": 0.88
        },
        {
            "memory_id": "mem_002",
            "description": "First time baking cookies",
            "time_period": "adolescence",
            "emotional_intensity": 0.7,
            "vividness": 0.75,
            "confidence": 0.82
        }
    ],
    "emotional_responses": {
        "nostalgia": 0.9,
        "comfort": 0.8,
        "warmth": 0.85
    },
    "autobiographical_content": {
        "family_memories": 0.7,
        "cooking_experiences": 0.6,
        "holiday_traditions": 0.8
    },
    "memory_vividness": 0.82
}
```

#### POST /api/v1/olfactory/memory/encode
**Description**: Encode new scent-memory associations
**Authorization**: PERSONAL level required

```python
# Request Schema
{
    "user_id": "user_12345",
    "scent_data": {
        "primary_scent": "rose",
        "scent_context": "garden"
    },
    "memory_content": {
        "event_description": "Proposal in rose garden",
        "emotional_content": {
            "joy": 0.95,
            "love": 0.9,
            "surprise": 0.7
        },
        "contextual_details": {
            "location": "Botanical Garden",
            "date": "2025-06-15",
            "people_present": ["partner"]
        }
    },
    "encoding_strength": 0.9
}

# Response Schema
{
    "encoding_successful": true,
    "memory_id": "mem_new_001",
    "association_strength": 0.88,
    "predicted_retrieval_accuracy": 0.92,
    "consolidation_timeline": "24-48 hours"
}
```

### 4. Emotional Response API

#### POST /api/v1/olfactory/emotion/generate
**Description**: Generate emotional responses to olfactory stimuli
**Authorization**: PUBLIC level required

```python
# Request Schema
class EmotionalResponseRequest:
    scent_data: dict                  # Scent recognition data
    user_profile: dict                # User emotional profile
    context: dict                     # Environmental/social context
    cultural_setting: str             # Cultural interpretation context

# Example Request
{
    "scent_data": {
        "primary_scent": "lavender",
        "intensity": 0.6,
        "quality_descriptors": ["floral", "calming", "sweet"]
    },
    "user_profile": {
        "emotional_sensitivity": 0.7,
        "stress_level": 0.8,
        "current_mood": "anxious"
    },
    "context": {
        "environment": "bedroom",
        "time": "bedtime",
        "purpose": "relaxation"
    },
    "cultural_setting": "western"
}

# Response Schema
{
    "emotion_response_id": "emot_resp_001",
    "primary_emotions": {
        "calmness": 0.85,
        "relaxation": 0.9,
        "comfort": 0.75
    },
    "hedonic_evaluation": {
        "pleasantness": 0.8,
        "familiarity": 0.7,
        "appropriateness": 0.95
    },
    "physiological_responses": {
        "stress_reduction": 0.6,
        "heart_rate_change": -0.1,
        "breathing_rate_change": -0.15
    },
    "behavioral_tendencies": {
        "approach_behavior": 0.8,
        "inhalation_depth": 0.9,
        "duration_preference": "extended"
    }
}
```

### 5. Consciousness Generation API

#### POST /api/v1/olfactory/consciousness/generate
**Description**: Generate unified olfactory consciousness experience
**Authorization**: PERSONAL level required

```python
# Request Schema
class ConsciousnessGenerationRequest:
    chemical_detection: dict          # Chemical detection results
    scent_recognition: dict           # Scent recognition results
    memory_associations: dict         # Memory integration results
    emotional_responses: dict         # Emotional response results
    attention_state: dict            # Current attention configuration
    consciousness_preferences: dict   # User consciousness preferences

# Example Request
{
    "chemical_detection": {
        "molecules": ["linalool", "limonene"],
        "concentrations": [2.1e-9, 1.8e-9]
    },
    "scent_recognition": {
        "primary_scent": "bergamot",
        "confidence": 0.88
    },
    "memory_associations": {
        "triggered_memories": ["earl_grey_tea", "afternoon_reading"]
    },
    "emotional_responses": {
        "comfort": 0.8,
        "sophistication": 0.6
    },
    "attention_state": {
        "focus_intensity": 0.7,
        "selective_attention": ["bergamot"]
    },
    "consciousness_preferences": {
        "experience_depth": "rich",
        "cultural_context": "british"
    }
}

# Response Schema
class OlfactoryConsciousnessResponse:
    consciousness_id: str
    unified_experience: dict
    phenomenological_qualities: dict
    consciousness_intensity: float
    experience_coherence: float

# Example Response
{
    "consciousness_id": "olf_cons_001",
    "unified_experience": {
        "scent_identity": "bergamot",
        "qualitative_experience": {
            "citrus_brightness": 0.8,
            "floral_elegance": 0.6,
            "tea_association": 0.9
        },
        "emotional_coloring": {
            "sophistication": 0.7,
            "comfort": 0.8,
            "nostalgia": 0.5
        },
        "memory_resonance": {
            "afternoon_tea": 0.9,
            "library_reading": 0.6,
            "british_culture": 0.7
        }
    },
    "phenomenological_qualities": {
        "experience_richness": 0.85,
        "clarity": 0.9,
        "temporal_flow": 0.8,
        "spatial_presence": 0.7
    },
    "consciousness_intensity": 0.82,
    "experience_coherence": 0.88
}
```

### 6. Cross-Modal Integration API

#### POST /api/v1/olfactory/integration/cross_modal
**Description**: Integrate olfactory consciousness with other sensory modalities
**Authorization**: PUBLIC level required

```python
# Request Schema
{
    "olfactory_consciousness": {
        "primary_scent": "coffee",
        "intensity": 0.8
    },
    "other_modalities": {
        "gustatory": {
            "taste": "bitter",
            "intensity": 0.7
        },
        "auditory": {
            "sounds": ["percolating", "grinding"],
            "volume": 0.4
        },
        "visual": {
            "objects": ["coffee_cup", "steam"],
            "lighting": "warm"
        }
    },
    "integration_mode": "flavor_enhancement"
}

# Response Schema
{
    "integration_id": "cross_modal_001",
    "enhanced_experience": {
        "unified_coffee_experience": {
            "flavor_consciousness": 0.95,
            "aromatic_enhancement": 1.3,
            "multisensory_coherence": 0.9
        },
        "cross_modal_effects": {
            "taste_enhancement": 0.4,
            "visual_expectation_match": 0.85,
            "auditory_association_strength": 0.7
        }
    },
    "integration_quality": {
        "temporal_synchronization": 0.92,
        "spatial_alignment": 0.88,
        "semantic_coherence": 0.94
    }
}
```

### 7. Cultural Adaptation API

#### POST /api/v1/olfactory/culture/adapt
**Description**: Adapt olfactory consciousness to cultural contexts
**Authorization**: PUBLIC level required

```python
# Request Schema
{
    "scent_data": {
        "primary_scent": "durian",
        "intensity": 0.9
    },
    "cultural_contexts": ["southeast_asian", "western"],
    "adaptation_mode": "cultural_sensitivity"
}

# Response Schema
{
    "cultural_adaptation_id": "cult_adapt_001",
    "cultural_interpretations": {
        "southeast_asian": {
            "hedonic_value": 0.8,
            "cultural_significance": "delicacy",
            "emotional_associations": ["enjoyment", "tradition"],
            "social_meaning": "special_occasion"
        },
        "western": {
            "hedonic_value": -0.6,
            "cultural_significance": "unfamiliar",
            "emotional_associations": ["surprise", "caution"],
            "social_meaning": "exotic_experience"
        }
    },
    "adaptation_recommendations": {
        "context_sensitive_presentation": true,
        "cultural_education_content": "durian_cultural_significance",
        "gradual_exposure_protocol": "available"
    }
}
```

### 8. Safety Monitoring API

#### GET /api/v1/olfactory/safety/status
**Description**: Get current safety status of olfactory system
**Authorization**: PUBLIC level required

```python
# Response Schema
{
    "safety_status": "SAFE",
    "chemical_safety": {
        "detected_toxins": [],
        "allergen_alerts": [],
        "concentration_levels": "within_safe_limits",
        "exposure_duration": "normal"
    },
    "psychological_safety": {
        "emotional_intensity": "comfortable",
        "memory_sensitivity": "normal",
        "cultural_appropriateness": "verified"
    },
    "system_health": {
        "sensor_status": "operational",
        "processing_performance": "optimal",
        "data_integrity": "verified"
    }
}
```

## Streaming and Real-Time APIs

### WebSocket Connection for Real-Time Olfactory Processing

```python
# WebSocket Endpoint: /ws/olfactory/realtime
class OlfactoryWebSocket:
    def on_connect(self, websocket, path):
        # Authentication and session initialization
        pass

    def on_message(self, websocket, message):
        # Real-time chemical detection commands
        {
            "command": "start_detection",
            "parameters": {
                "sensitivity": 0.8,
                "target_molecules": ["vanilla", "cinnamon"]
            }
        }

    def on_consciousness_update(self, consciousness_data):
        # Real-time consciousness state broadcast
        {
            "type": "consciousness_update",
            "scent": "vanilla",
            "intensity": 0.7,
            "emotional_response": {
                "comfort": 0.8,
                "nostalgia": 0.6
            },
            "memory_activations": ["childhood_kitchen"],
            "timestamp": 1640995200000
        }
```

## Error Handling and Status Codes

### HTTP Status Codes
- `200 OK`: Successful olfactory processing
- `400 Bad Request`: Invalid chemical or scent parameters
- `401 Unauthorized`: Insufficient access level
- `403 Forbidden`: Cultural sensitivity violation
- `404 Not Found`: Scent or memory not found
- `429 Too Many Requests`: Rate limiting
- `500 Internal Server Error`: Processing system error

### Error Response Schema
```python
{
    "error": {
        "code": "CULTURAL_SENSITIVITY_VIOLATION",
        "message": "Requested scent presentation violates cultural sensitivity guidelines",
        "details": {
            "scent": "pork_aroma",
            "cultural_context": "islamic",
            "violation_type": "religious_dietary_restriction",
            "suggested_alternatives": ["vegetarian_umami", "herb_blend"]
        }
    },
    "timestamp": "2025-09-26T10:00:00Z",
    "request_id": "req_12345"
}
```

## Rate Limiting and Usage Quotas

### Rate Limits by Endpoint Type
- **Chemical detection**: 1000 requests/minute
- **Scent recognition**: 500 requests/minute
- **Memory integration**: 100 requests/minute (privacy protection)
- **Consciousness generation**: 200 requests/minute
- **Cultural adaptation**: 300 requests/minute
- **Safety monitoring**: Unlimited (safety priority)

This comprehensive API specification enables sophisticated, culturally-sensitive, and personalized olfactory consciousness experiences while maintaining strict safety, privacy, and cultural appropriateness standards.