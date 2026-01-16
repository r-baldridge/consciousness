# Collective Consciousness - API Specification
**Module 20: Collective Consciousness**
**Task B4: API Specification**
**Date:** September 27, 2025

## Overview

This document defines the comprehensive REST API specification for the Collective Consciousness module. The API provides endpoints for agent management, collective communication, shared state synchronization, decision-making processes, emergent behavior monitoring, and group awareness coordination.

## API Architecture

### Base Configuration

- **Base URL**: `https://api.collective-consciousness.ai/v1`
- **Protocol**: HTTPS only
- **Authentication**: Bearer token (JWT)
- **Content Type**: `application/json`
- **Rate Limiting**: 10,000 requests per hour per agent
- **API Version**: v1.0

### Common Response Format

```json
{
  "success": true,
  "data": {},
  "metadata": {
    "timestamp": "2025-09-27T10:00:00Z",
    "request_id": "req_123456789",
    "agent_id": "agent_abc123",
    "processing_time_ms": 150
  },
  "errors": [],
  "warnings": []
}
```

## 1. Agent Management API

### 1.1 Agent Registration and Identity

#### Register New Agent
```http
POST /agents/register
Content-Type: application/json
Authorization: Bearer {token}

{
  "agent_type": "ai",
  "capabilities": {
    "natural_language": "expert",
    "data_analysis": "advanced",
    "problem_solving": "intermediate"
  },
  "specializations": ["machine_learning", "data_science"],
  "resource_capacity": {
    "cpu_capacity": 8.0,
    "memory_capacity": 16777216,
    "network_bandwidth": 1000000,
    "concurrent_tasks": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "agent_id": "agent_abc123",
    "registration_timestamp": "2025-09-27T10:00:00Z",
    "public_key": "-----BEGIN PUBLIC KEY-----...",
    "initial_groups": ["default_collective"],
    "authentication_token": "jwt_token_here"
  }
}
```

#### Get Agent Profile
```http
GET /agents/{agent_id}
Authorization: Bearer {token}
```

#### Update Agent Capabilities
```http
PUT /agents/{agent_id}/capabilities
Content-Type: application/json
Authorization: Bearer {token}

{
  "capabilities": {
    "natural_language": "expert",
    "data_analysis": "expert"
  },
  "specializations": ["machine_learning", "nlp"]
}
```

#### Deregister Agent
```http
DELETE /agents/{agent_id}
Authorization: Bearer {token}
```

### 1.2 Agent Status and Health

#### Update Agent Status
```http
PUT /agents/{agent_id}/status
Content-Type: application/json
Authorization: Bearer {token}

{
  "status": "active",
  "location": {
    "type": "virtual",
    "coordinates": [0.0, 0.0, 0.0]
  },
  "health_metrics": {
    "cpu_usage": 0.75,
    "memory_usage": 0.60,
    "response_time_ms": 120
  }
}
```

#### Get Agent Health Metrics
```http
GET /agents/{agent_id}/health
Authorization: Bearer {token}
```

## 2. Communication API

### 2.1 Messaging

#### Send Direct Message
```http
POST /communication/messages
Content-Type: application/json
Authorization: Bearer {token}

{
  "recipients": ["agent_def456"],
  "content_type": "structured_data",
  "content": {
    "type": "coordination_request",
    "task_id": "task_789",
    "priority": "high"
  },
  "priority": "high",
  "delivery_guarantee": "exactly_once",
  "context": {
    "situation_id": "sit_123",
    "urgency_level": "high"
  }
}
```

#### Broadcast Message
```http
POST /communication/broadcast
Content-Type: application/json
Authorization: Bearer {token}

{
  "target_groups": ["emergency_response"],
  "content": {
    "type": "alert",
    "message": "Emergency situation detected",
    "coordinates": [40.7128, -74.0060]
  },
  "priority": "emergency"
}
```

#### Subscribe to Topic
```http
POST /communication/subscriptions
Content-Type: application/json
Authorization: Bearer {token}

{
  "topic": "emergency_alerts",
  "callback_url": "https://agent.example.com/webhook/messages",
  "filter_criteria": {
    "priority": ["high", "emergency"],
    "location_radius": 50.0
  }
}
```

#### Get Message History
```http
GET /communication/messages?agent_id={agent_id}&limit=50&offset=0
Authorization: Bearer {token}
```

### 2.2 Real-time Communication

#### WebSocket Connection
```
WSS /communication/realtime
Authorization: Bearer {token}
```

**Message Format:**
```json
{
  "type": "message",
  "data": {
    "message_id": "msg_123",
    "sender_id": "agent_abc123",
    "content": {},
    "timestamp": "2025-09-27T10:00:00Z"
  }
}
```

## 3. Collective State Management API

### 3.1 Shared State Operations

#### Read Shared State
```http
GET /state/{state_key}?consistency_level=strong
Authorization: Bearer {token}
```

#### Update Shared State
```http
PUT /state/{state_key}
Content-Type: application/json
Authorization: Bearer {token}

{
  "value": {
    "temperature": 23.5,
    "humidity": 65,
    "timestamp": "2025-09-27T10:00:00Z"
  },
  "condition": {
    "type": "version_match",
    "expected_version": 42
  }
}
```

#### Atomic State Transaction
```http
POST /state/transactions
Content-Type: application/json
Authorization: Bearer {token}

{
  "operations": [
    {
      "type": "update",
      "state_key": "resource_pool",
      "value": {"cpu": 0.8, "memory": 0.6}
    },
    {
      "type": "update",
      "state_key": "task_queue",
      "value": {"pending": 5, "active": 3}
    }
  ]
}
```

#### Subscribe to State Changes
```http
POST /state/subscriptions
Content-Type: application/json
Authorization: Bearer {token}

{
  "state_pattern": "sensor_data.*",
  "callback_url": "https://agent.example.com/webhook/state_change"
}
```

### 3.2 Collective Memory

#### Store Memory
```http
POST /memory/items
Content-Type: application/json
Authorization: Bearer {token}

{
  "memory_type": "episodic",
  "content": {
    "event": "successful_coordination",
    "participants": ["agent_abc123", "agent_def456"],
    "outcome": "task_completed",
    "efficiency_score": 0.85
  },
  "context": {
    "situation_description": "Multi-agent task coordination",
    "environmental_conditions": {"load": "high"},
    "goals_at_time": ["optimize_efficiency"]
  },
  "significance_score": 0.8
}
```

#### Retrieve Memories
```http
GET /memory/search?query=coordination&memory_type=episodic&limit=10
Authorization: Bearer {token}
```

#### Advanced Memory Query
```http
POST /memory/query
Content-Type: application/json
Authorization: Bearer {token}

{
  "query_type": "semantic",
  "content_similarity": {
    "text": "successful team coordination",
    "threshold": 0.7
  },
  "temporal_range": {
    "start": "2025-09-01T00:00:00Z",
    "end": "2025-09-27T23:59:59Z"
  },
  "filters": {
    "memory_type": ["episodic", "procedural"],
    "significance_threshold": 0.5
  }
}
```

## 4. Decision-Making API

### 4.1 Consensus Building

#### Initiate Consensus
```http
POST /decisions/consensus
Content-Type: application/json
Authorization: Bearer {token}

{
  "proposal": {
    "title": "Resource Allocation Strategy",
    "description": "Proposal for new resource allocation algorithm",
    "alternatives": [
      {
        "id": "alt_1",
        "name": "Equal Distribution",
        "description": "Distribute resources equally among agents"
      },
      {
        "id": "alt_2",
        "name": "Capability-Based",
        "description": "Allocate based on agent capabilities"
      }
    ]
  },
  "participants": ["agent_abc123", "agent_def456", "agent_ghi789"],
  "consensus_threshold": 0.8,
  "deadline": "2025-09-28T18:00:00Z"
}
```

#### Submit Vote
```http
POST /decisions/consensus/{session_id}/votes
Content-Type: application/json
Authorization: Bearer {token}

{
  "alternative_id": "alt_2",
  "confidence": 0.9,
  "reasoning": "Capability-based allocation is more efficient",
  "supporting_evidence": ["historical_performance_data", "simulation_results"]
}
```

#### Get Consensus Status
```http
GET /decisions/consensus/{session_id}/status
Authorization: Bearer {token}
```

### 4.2 Collective Planning

#### Create Planning Session
```http
POST /planning/sessions
Content-Type: application/json
Authorization: Bearer {token}

{
  "goal": {
    "objective": "Optimize collective efficiency",
    "success_criteria": ["efficiency > 0.85", "error_rate < 0.05"],
    "deadline": "2025-10-01T00:00:00Z"
  },
  "constraints": [
    {
      "type": "resource",
      "description": "Total CPU usage < 80%"
    },
    {
      "type": "temporal",
      "description": "Complete within 72 hours"
    }
  ],
  "participants": ["agent_abc123", "agent_def456"]
}
```

#### Contribute to Plan
```http
POST /planning/sessions/{session_id}/contributions
Content-Type: application/json
Authorization: Bearer {token}

{
  "contribution_type": "task_suggestion",
  "content": {
    "task_id": "optimize_algorithm",
    "description": "Optimize the coordination algorithm",
    "estimated_duration": "4 hours",
    "required_capabilities": ["algorithm_design", "optimization"],
    "dependencies": ["data_analysis_complete"]
  }
}
```

## 5. Emergent Behavior API

### 5.1 Swarm Intelligence

#### Join Swarm
```http
POST /swarm/{swarm_id}/join
Content-Type: application/json
Authorization: Bearer {token}

{
  "capabilities": {
    "processing_power": 8.0,
    "specializations": ["optimization", "search"]
  },
  "initial_position": [0.5, 0.3, 0.8],
  "objective_contribution": "global_optimization"
}
```

#### Update Swarm Position
```http
PUT /swarm/{swarm_id}/position
Content-Type: application/json
Authorization: Bearer {token}

{
  "position": [0.6, 0.4, 0.7],
  "velocity": [0.1, 0.1, -0.1],
  "fitness": 0.85,
  "local_best": [0.6, 0.4, 0.7]
}
```

#### Get Swarm Neighbors
```http
GET /swarm/{swarm_id}/neighbors?radius=0.2
Authorization: Bearer {token}
```

### 5.2 Emergence Detection

#### Register Emergence Pattern
```http
POST /emergence/patterns
Content-Type: application/json
Authorization: Bearer {token}

{
  "pattern_name": "Coordination_Spiral",
  "detection_criteria": [
    {
      "metric": "coordination_efficiency",
      "threshold": 0.9,
      "duration": "5 minutes"
    }
  ],
  "minimum_participants": 5,
  "complexity_indicators": ["entropy", "mutual_information"]
}
```

#### Detect Emergent Behaviors
```http
GET /emergence/detect?window_start=2025-09-27T09:00:00Z&window_end=2025-09-27T10:00:00Z
Authorization: Bearer {token}
```

## 6. Group Awareness API

### 6.1 Situational Awareness

#### Contribute Situation Data
```http
POST /awareness/situation/contribute
Content-Type: application/json
Authorization: Bearer {token}

{
  "observation": {
    "type": "environmental_change",
    "description": "Increased network latency detected",
    "location": {"region": "us-east-1"},
    "severity": "medium",
    "confidence": 0.8,
    "timestamp": "2025-09-27T10:00:00Z"
  }
}
```

#### Get Collective Situation
```http
GET /awareness/situation/current?location=us-east-1&scope=network
Authorization: Bearer {token}
```

#### Request Situation Focus
```http
POST /awareness/situation/focus
Content-Type: application/json
Authorization: Bearer {token}

{
  "focus_area": {
    "type": "geographic",
    "coordinates": [40.7128, -74.0060],
    "radius": 10.0
  },
  "priority": "high",
  "duration": "30 minutes",
  "reason": "Anomalous activity detected"
}
```

### 6.2 Group Identity

#### Define Group Identity
```http
POST /groups/{group_id}/identity
Content-Type: application/json
Authorization: Bearer {token}

{
  "group_name": "Emergency Response Team",
  "mission_statement": "Rapid response to emergency situations",
  "core_values": ["efficiency", "collaboration", "safety"],
  "decision_making_model": {
    "type": "consensus",
    "threshold": 0.75
  }
}
```

#### Update Group Values
```http
PUT /groups/{group_id}/values
Content-Type: application/json
Authorization: Bearer {token}

{
  "values_update": {
    "core_values": ["efficiency", "collaboration", "safety", "innovation"],
    "change_reason": "Adaptation to new challenges"
  }
}
```

## 7. Monitoring and Analytics API

### 7.1 Performance Metrics

#### Get Collective Performance
```http
GET /metrics/performance?period=24h&granularity=1h
Authorization: Bearer {token}
```

#### Get Real-time Metrics
```http
GET /metrics/realtime
Authorization: Bearer {token}
```

### 7.2 Health and Diagnostics

#### System Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-27T10:00:00Z",
  "components": {
    "communication_system": "healthy",
    "state_management": "healthy",
    "decision_engine": "healthy",
    "emergence_detector": "warning"
  },
  "active_agents": 1250,
  "active_groups": 47,
  "system_load": 0.72
}
```

## 8. WebSocket Events

### Event Types

#### Agent Events
- `agent.joined` - Agent joined collective
- `agent.left` - Agent left collective
- `agent.status_changed` - Agent status updated

#### Communication Events
- `message.received` - New message received
- `broadcast.sent` - Broadcast message sent

#### State Events
- `state.changed` - Shared state updated
- `state.conflict` - State conflict detected

#### Decision Events
- `consensus.started` - Consensus process initiated
- `consensus.completed` - Consensus reached
- `vote.submitted` - Vote submitted

#### Emergence Events
- `emergence.detected` - New emergent behavior detected
- `pattern.evolved` - Emergent pattern evolved

## Error Codes

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Custom Error Codes
```json
{
  "error_code": "CC_001",
  "error_type": "CONSENSUS_TIMEOUT",
  "message": "Consensus building timed out",
  "details": {
    "session_id": "consensus_123",
    "timeout_duration": "300s",
    "current_consensus": 0.65
  },
  "remediation": "Consider extending timeout or reducing consensus threshold"
}
```

## Rate Limiting

### Limits by Endpoint Category
- **Authentication**: 100 requests/hour
- **Agent Management**: 1,000 requests/hour
- **Communication**: 10,000 requests/hour
- **State Operations**: 5,000 requests/hour
- **Decision Making**: 500 requests/hour
- **Monitoring**: 2,000 requests/hour

### Rate Limit Headers
```http
X-RateLimit-Limit: 10000
X-RateLimit-Remaining: 9500
X-RateLimit-Reset: 1696118400
```

## Security Considerations

### Authentication
- JWT tokens with 24-hour expiration
- Refresh tokens for extended sessions
- Multi-factor authentication for sensitive operations

### Authorization
- Role-based access control (RBAC)
- Agent-level permissions
- Group-level access controls

### Data Protection
- TLS 1.3 for all communications
- End-to-end encryption for sensitive data
- Data anonymization for analytics

This API specification provides comprehensive access to all collective consciousness functionality while maintaining security, performance, and reliability standards.