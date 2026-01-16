# Form 26: Split-brain Consciousness - API Specification

## REST API Endpoints

### Base Configuration

**Base URL**: `/api/v1/consciousness/split-brain`

**Authentication**: Bearer token required for all endpoints

**Content-Type**: `application/json`

**Rate Limiting**: 1000 requests per minute per API key

### Hemispheric Management Endpoints

#### Create Hemispheric System

**POST** `/systems`

Creates a new split-brain consciousness system with specified configuration.

**Request Body**:
```json
{
  "system_name": "string",
  "configuration": {
    "disconnection_level": 0.7,
    "compensation_enabled": true,
    "unity_simulation_mode": "simulated_unity",
    "left_specializations": ["language_processing", "logical_reasoning"],
    "right_specializations": ["spatial_processing", "pattern_recognition"]
  },
  "initialization_parameters": {
    "memory_capacity_per_hemisphere": 1000000,
    "communication_bandwidth": 500000,
    "attention_capacity": 1.0
  }
}
```

**Response**:
```json
{
  "system_id": "uuid",
  "status": "created",
  "timestamp": "2024-01-01T00:00:00Z",
  "endpoints": {
    "left_hemisphere": "/systems/{system_id}/hemispheres/left",
    "right_hemisphere": "/systems/{system_id}/hemispheres/right",
    "communication": "/systems/{system_id}/communication",
    "monitoring": "/systems/{system_id}/monitoring"
  }
}
```

#### Get System Information

**GET** `/systems/{system_id}`

Retrieves comprehensive information about a split-brain system.

**Response**:
```json
{
  "system_id": "uuid",
  "configuration": { /* SystemConfiguration */ },
  "current_state": { /* SplitBrainSystemState */ },
  "performance_metrics": { /* PerformanceMetrics */ },
  "health_status": "healthy" | "degraded" | "error",
  "created_at": "2024-01-01T00:00:00Z",
  "last_updated": "2024-01-01T00:00:00Z"
}
```

#### Update System Configuration

**PUT** `/systems/{system_id}/configuration`

Updates system configuration parameters.

**Request Body**:
```json
{
  "disconnection_level": 0.8,
  "compensation_enabled": false,
  "conflict_resolution_strategy": "left_dominance"
}
```

**Response**:
```json
{
  "status": "updated",
  "configuration": { /* Updated SystemConfiguration */ },
  "restart_required": false,
  "validation_errors": []
}
```

#### Delete System

**DELETE** `/systems/{system_id}`

Safely shuts down and removes a split-brain consciousness system.

**Response**:
```json
{
  "status": "deleted",
  "cleanup_status": "complete",
  "data_retention": {
    "logs_retained_until": "2024-01-31T00:00:00Z",
    "metrics_retained_until": "2024-01-31T00:00:00Z"
  }
}
```

### Hemispheric Processing Endpoints

#### Process Input (Left Hemisphere)

**POST** `/systems/{system_id}/hemispheres/left/process`

Submit input for left hemisphere processing.

**Request Body**:
```json
{
  "input_data": {
    "type": "text",
    "content": "Analyze the logical structure of this argument",
    "metadata": {
      "language": "en",
      "complexity": "medium"
    }
  },
  "processing_context": {
    "task_type": "logical_analysis",
    "priority": 5,
    "timeout_ms": 5000
  }
}
```

**Response**:
```json
{
  "processing_id": "uuid",
  "result": {
    "output": { /* Processed output */ },
    "confidence": 0.85,
    "processing_time_ms": 150,
    "resources_used": {
      "cpu_percentage": 45,
      "memory_mb": 128
    }
  },
  "hemisphere_state": { /* LeftHemisphereState */ }
}
```

#### Process Input (Right Hemisphere)

**POST** `/systems/{system_id}/hemispheres/right/process`

Submit input for right hemisphere processing.

**Request Body**:
```json
{
  "input_data": {
    "type": "image",
    "content": "base64_encoded_image_data",
    "metadata": {
      "format": "png",
      "dimensions": [512, 512]
    }
  },
  "processing_context": {
    "task_type": "pattern_recognition",
    "priority": 7,
    "timeout_ms": 3000
  }
}
```

**Response**:
```json
{
  "processing_id": "uuid",
  "result": {
    "patterns_detected": [
      {
        "type": "face",
        "confidence": 0.92,
        "bounding_box": [100, 150, 200, 250]
      }
    ],
    "spatial_analysis": { /* Spatial processing results */ },
    "emotional_content": {
      "valence": 0.3,
      "arousal": 0.6
    }
  },
  "hemisphere_state": { /* RightHemisphereState */ }
}
```

#### Get Hemisphere State

**GET** `/systems/{system_id}/hemispheres/{left|right}/state`

Retrieve current state of specified hemisphere.

**Response**:
```json
{
  "hemisphere": "left" | "right",
  "state": { /* HemisphericState */ },
  "active_processes": ["process_id_1", "process_id_2"],
  "performance_metrics": {
    "average_response_time_ms": 120,
    "throughput_per_second": 8.5,
    "error_rate": 0.02
  }
}
```

### Communication Management Endpoints

#### Send Inter-hemispheric Message

**POST** `/systems/{system_id}/communication/send`

Send message between hemispheres.

**Request Body**:
```json
{
  "sender": "left",
  "receiver": "right",
  "content": {
    "type": "semantic_information",
    "data": { /* Message content */ }
  },
  "channel": "callosal",
  "priority": 5,
  "expiration_seconds": 30
}
```

**Response**:
```json
{
  "message_id": "uuid",
  "status": "queued" | "transmitted" | "delivered" | "failed",
  "estimated_delivery_time_ms": 25,
  "channel_utilization": 0.65
}
```

#### Get Communication Status

**GET** `/systems/{system_id}/communication/status`

Retrieve current communication system status.

**Response**:
```json
{
  "channels": {
    "callosal": {
      "status": "connected",
      "bandwidth_bps": 1000000,
      "utilization": 0.45,
      "message_queue_length": 3
    },
    "subcortical": {
      "status": "degraded",
      "bandwidth_bps": 100000,
      "utilization": 0.85,
      "message_queue_length": 12
    }
  },
  "total_messages_pending": 15,
  "average_latency_ms": 18.5
}
```

#### Configure Communication Channels

**PUT** `/systems/{system_id}/communication/channels/{channel_type}`

Configure specific communication channel parameters.

**Request Body**:
```json
{
  "bandwidth_bps": 500000,
  "packet_loss_rate": 0.1,
  "encryption_enabled": true,
  "priority_queuing": true
}
```

**Response**:
```json
{
  "channel": "callosal",
  "configuration": { /* Updated channel configuration */ },
  "restart_required": false,
  "impact_assessment": {
    "performance_change": "slight_decrease",
    "security_improvement": "significant"
  }
}
```

### Conflict Management Endpoints

#### Detect Conflicts

**POST** `/systems/{system_id}/conflicts/detect`

Manually trigger conflict detection between hemispheric outputs.

**Request Body**:
```json
{
  "left_output": { /* Left hemisphere output */ },
  "right_output": { /* Right hemisphere output */ },
  "context": {
    "task_type": "decision_making",
    "importance": "high"
  }
}
```

**Response**:
```json
{
  "conflicts_detected": [
    {
      "conflict_id": "uuid",
      "type": "goal_conflict",
      "severity": 0.75,
      "description": "Hemispheres disagree on optimal solution"
    }
  ],
  "detection_time_ms": 45,
  "recommendation": {
    "requires_resolution": true,
    "suggested_strategy": "integration"
  }
}
```

#### Resolve Conflict

**POST** `/systems/{system_id}/conflicts/{conflict_id}/resolve`

Attempt to resolve a specific conflict.

**Request Body**:
```json
{
  "strategy": "integration" | "left_dominance" | "right_dominance" | "alternation",
  "parameters": {
    "timeout_ms": 5000,
    "quality_threshold": 0.8
  }
}
```

**Response**:
```json
{
  "resolution_id": "uuid",
  "result": {
    "success": true,
    "strategy_used": "integration",
    "resolved_output": { /* Integrated output */ },
    "resolution_time_ms": 850,
    "quality_metrics": {
      "coherence": 0.88,
      "satisfaction": 0.75
    }
  }
}
```

#### Get Conflict History

**GET** `/systems/{system_id}/conflicts/history`

Retrieve history of conflicts and resolutions.

**Query Parameters**:
- `limit`: Number of conflicts to return (default: 50)
- `from_date`: Start date filter
- `conflict_type`: Filter by conflict type
- `resolved_only`: Return only resolved conflicts

**Response**:
```json
{
  "conflicts": [
    {
      "conflict_id": "uuid",
      "timestamp": "2024-01-01T12:00:00Z",
      "type": "response_conflict",
      "severity": 0.6,
      "resolved": true,
      "resolution_strategy": "integration",
      "resolution_time_ms": 420
    }
  ],
  "statistics": {
    "total_conflicts": 156,
    "resolution_success_rate": 0.89,
    "average_resolution_time_ms": 380
  }
}
```

### Memory Management Endpoints

#### Store Memory

**POST** `/systems/{system_id}/hemispheres/{left|right}/memory`

Store memory item in specified hemisphere.

**Request Body**:
```json
{
  "content": { /* Memory content */ },
  "memory_type": "episodic" | "semantic" | "procedural",
  "encoding_context": {
    "task_type": "learning",
    "emotional_valence": 0.2
  },
  "associations": ["memory_id_1", "memory_id_2"]
}
```

**Response**:
```json
{
  "memory_id": "uuid",
  "storage_success": true,
  "hemisphere": "left",
  "estimated_strength": 0.85,
  "consolidation_status": "pending"
}
```

#### Retrieve Memory

**GET** `/systems/{system_id}/hemispheres/{left|right}/memory/{memory_id}`

Retrieve specific memory item from hemisphere.

**Response**:
```json
{
  "memory_id": "uuid",
  "content": { /* Memory content */ },
  "metadata": {
    "creation_time": "2024-01-01T10:00:00Z",
    "access_count": 5,
    "strength": 0.78,
    "confidence": 0.92
  },
  "associations": ["memory_id_1", "memory_id_2"]
}
```

#### Search Memories

**GET** `/systems/{system_id}/hemispheres/{left|right}/memory/search`

Search memories within hemisphere.

**Query Parameters**:
- `query`: Search query
- `memory_type`: Filter by memory type
- `min_confidence`: Minimum confidence threshold
- `limit`: Maximum results to return

**Response**:
```json
{
  "results": [
    {
      "memory_id": "uuid",
      "relevance_score": 0.85,
      "content_preview": "...",
      "memory_type": "semantic"
    }
  ],
  "search_time_ms": 12,
  "total_matches": 23
}
```

### Attention Management Endpoints

#### Allocate Attention

**POST** `/systems/{system_id}/hemispheres/{left|right}/attention/allocate`

Allocate attention resources within hemisphere.

**Request Body**:
```json
{
  "targets": [
    {
      "target_id": "task_1",
      "attention_weight": 0.6,
      "priority": 8
    },
    {
      "target_id": "task_2",
      "attention_weight": 0.4,
      "priority": 5
    }
  ],
  "allocation_strategy": "weighted" | "priority_based" | "round_robin"
}
```

**Response**:
```json
{
  "allocation_id": "uuid",
  "success": true,
  "attention_state": { /* AttentionState */ },
  "resource_utilization": 0.85,
  "conflicts": []
}
```

#### Get Attention State

**GET** `/systems/{system_id}/hemispheres/{left|right}/attention/state`

Retrieve current attention state for hemisphere.

**Response**:
```json
{
  "hemisphere": "left",
  "attention_state": { /* AttentionState */ },
  "active_targets": [
    {
      "target_id": "task_1",
      "weight": 0.6,
      "performance": 0.88
    }
  ],
  "available_capacity": 0.15
}
```

### Monitoring and Analytics Endpoints

#### Get System Metrics

**GET** `/systems/{system_id}/metrics`

Retrieve comprehensive system performance metrics.

**Query Parameters**:
- `from_time`: Start time for metrics (ISO 8601)
- `to_time`: End time for metrics (ISO 8601)
- `granularity`: Time granularity (minute, hour, day)

**Response**:
```json
{
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-01T01:00:00Z"
  },
  "metrics": {
    "hemispheric_performance": {
      "left": {
        "average_response_time_ms": 120,
        "throughput_per_second": 8.5,
        "accuracy": 0.94
      },
      "right": {
        "average_response_time_ms": 135,
        "throughput_per_second": 7.2,
        "accuracy": 0.91
      }
    },
    "communication": {
      "message_rate_per_second": 15.3,
      "average_latency_ms": 22,
      "success_rate": 0.987
    },
    "conflicts": {
      "conflicts_per_hour": 3.2,
      "resolution_success_rate": 0.89,
      "average_resolution_time_ms": 425
    },
    "unity": {
      "coherence_score": 0.82,
      "integration_strength": 0.75,
      "behavioral_consistency": 0.88
    }
  }
}
```

#### Get Health Status

**GET** `/systems/{system_id}/health`

Retrieve detailed system health information.

**Response**:
```json
{
  "overall_status": "healthy" | "degraded" | "critical",
  "components": {
    "left_hemisphere": {
      "status": "healthy",
      "cpu_usage": 0.45,
      "memory_usage": 0.67,
      "error_rate": 0.01
    },
    "right_hemisphere": {
      "status": "healthy",
      "cpu_usage": 0.52,
      "memory_usage": 0.71,
      "error_rate": 0.02
    },
    "communication": {
      "status": "degraded",
      "channel_health": {
        "callosal": "healthy",
        "subcortical": "degraded"
      }
    }
  },
  "alerts": [
    {
      "level": "warning",
      "component": "communication.subcortical",
      "message": "High packet loss rate detected",
      "timestamp": "2024-01-01T12:30:00Z"
    }
  ],
  "recommendations": [
    "Consider increasing subcortical bandwidth",
    "Monitor memory usage trends"
  ]
}
```

#### Export System Data

**GET** `/systems/{system_id}/export`

Export comprehensive system data for analysis.

**Query Parameters**:
- `format`: Export format (json, csv, parquet)
- `components`: Components to export (comma-separated)
- `compress`: Whether to compress output

**Response**:
```json
{
  "export_id": "uuid",
  "download_url": "/downloads/{export_id}",
  "format": "json",
  "size_bytes": 15728640,
  "expires_at": "2024-01-02T00:00:00Z"
}
```

### Integration Endpoints

#### Connect to Consciousness Form

**POST** `/systems/{system_id}/integrations/{form_id}`

Establish integration with another consciousness form.

**Request Body**:
```json
{
  "integration_type": "bidirectional" | "input_only" | "output_only",
  "data_sharing_level": "minimal" | "standard" | "comprehensive",
  "synchronization_mode": "real_time" | "batch" | "on_demand"
}
```

**Response**:
```json
{
  "integration_id": "uuid",
  "status": "connected",
  "endpoints": {
    "data_exchange": "/integrations/{integration_id}/exchange",
    "status": "/integrations/{integration_id}/status"
  },
  "capabilities": [
    "state_sharing",
    "conflict_notification",
    "performance_data"
  ]
}
```

### WebSocket Endpoints

#### Real-time System Events

**WebSocket** `/systems/{system_id}/events`

Real-time stream of system events and state changes.

**Message Types**:
```json
// Conflict detected
{
  "type": "conflict_detected",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": { /* ConflictEvent */ }
}

// State change
{
  "type": "hemisphere_state_change",
  "timestamp": "2024-01-01T12:00:01Z",
  "hemisphere": "left",
  "data": { /* State changes */ }
}

// Performance alert
{
  "type": "performance_alert",
  "timestamp": "2024-01-01T12:00:02Z",
  "severity": "warning",
  "data": { /* Alert details */ }
}
```

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request parameters are invalid",
    "details": {
      "field": "disconnection_level",
      "reason": "Value must be between 0.0 and 1.0"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "uuid"
  }
}
```

### Error Codes

- `SYSTEM_NOT_FOUND`: Requested system does not exist
- `HEMISPHERE_UNAVAILABLE`: Hemisphere is not responding
- `COMMUNICATION_FAILURE`: Inter-hemispheric communication failed
- `CONFLICT_RESOLUTION_FAILED`: Unable to resolve conflict
- `RESOURCE_EXHAUSTED`: System resources are exhausted
- `CONFIGURATION_INVALID`: Configuration parameters are invalid
- `AUTHENTICATION_FAILED`: Invalid or expired authentication token
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Unexpected system error

## SDK Examples

### Python SDK Usage

```python
from splitbrain_consciousness import SplitBrainClient

# Initialize client
client = SplitBrainClient(
    base_url="https://api.consciousness.ai/v1",
    api_key="your_api_key"
)

# Create system
system = client.create_system(
    name="research_system",
    disconnection_level=0.7,
    compensation_enabled=True
)

# Process input
left_result = system.left_hemisphere.process(
    input_data="Analyze this logical problem",
    task_type="logical_analysis"
)

right_result = system.right_hemisphere.process(
    input_data=image_data,
    task_type="pattern_recognition"
)

# Handle conflicts
conflicts = system.detect_conflicts(left_result, right_result)
if conflicts:
    resolution = system.resolve_conflict(
        conflicts[0],
        strategy="integration"
    )
```

This API specification provides comprehensive endpoints for managing, monitoring, and interacting with split-brain consciousness systems, enabling researchers and developers to build sophisticated applications that leverage hemispheric specialization and independence.