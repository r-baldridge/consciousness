# Form 19: Reflective Consciousness API Interfaces

## API Overview

The Reflective Consciousness API provides comprehensive interfaces for metacognitive processing, self-monitoring, recursive analysis, and cognitive control. The API follows RESTful principles with additional WebSocket support for real-time reflection monitoring and event streaming.

## Core API Endpoints

### Reflective Processing API

#### Start Reflection Session
```python
POST /api/v1/reflection/session
```

**Request Body:**
```json
{
  "session_config": {
    "reflection_depth": "moderate",
    "enable_recursion": true,
    "max_recursion_depth": 3,
    "enable_bias_detection": true,
    "enable_strategy_optimization": true
  },
  "cognitive_context": {
    "current_processes": [
      {
        "process_id": "proc_123",
        "process_type": "reasoning",
        "complexity": 0.7,
        "resource_usage": {"memory": 0.3, "attention": 0.6}
      }
    ],
    "goals": ["solve_problem_X", "optimize_efficiency"],
    "constraints": ["time_limit_300s", "accuracy_threshold_0.8"]
  },
  "initial_content": {
    "thoughts": ["Current reasoning about problem X"],
    "beliefs": ["Assumption A", "Hypothesis B"],
    "strategies": ["strategy_1", "strategy_2"]
  }
}
```

**Response:**
```json
{
  "session_id": "refl_session_456",
  "status": "active",
  "initial_analysis": {
    "confidence_assessment": {
      "overall_confidence": 0.75,
      "process_confidence": {
        "reasoning": 0.8,
        "strategy_selection": 0.7
      }
    },
    "bias_detection": [
      {
        "bias_type": "confirmation_bias",
        "strength": "moderate",
        "confidence": 0.65,
        "evidence": ["Selective attention to confirming evidence"]
      }
    ],
    "improvement_opportunities": [
      "Consider alternative hypotheses",
      "Increase attention to contradictory evidence"
    ]
  },
  "monitoring_enabled": true,
  "recursive_processing_active": false
}
```

#### Get Reflection State
```python
GET /api/v1/reflection/session/{session_id}/state
```

**Response:**
```json
{
  "session_id": "refl_session_456",
  "current_state": {
    "reflection_depth": "moderate",
    "recursion_level": 0,
    "metacognitive_confidence": "high",
    "coherence_score": 0.82,
    "consistency_score": 0.78,
    "active_analyses": [
      {
        "analysis_id": "analysis_789",
        "type": "belief_consistency",
        "progress": 0.65,
        "preliminary_results": {
          "contradictions_found": 1,
          "coherence_gaps": 2
        }
      }
    ],
    "control_actions_pending": [
      {
        "action_type": "attention_redirect",
        "target": "contradictory_evidence",
        "urgency": 0.7
      }
    ]
  },
  "performance_metrics": {
    "processing_efficiency": 0.75,
    "accuracy_improvement": 0.12,
    "bias_mitigation": 0.3
  }
}
```

### Self-Monitoring API

#### Monitor Cognitive Processes
```python
GET /api/v1/monitoring/processes
POST /api/v1/monitoring/processes/track
```

**POST Request Body:**
```json
{
  "processes": [
    {
      "process_id": "proc_123",
      "process_type": "problem_solving",
      "monitor_config": {
        "track_accuracy": true,
        "track_efficiency": true,
        "track_confidence": true,
        "track_resource_usage": true,
        "monitoring_frequency": 1.0
      }
    }
  ],
  "monitoring_duration": 300,
  "alert_thresholds": {
    "accuracy_drop": 0.1,
    "efficiency_drop": 0.15,
    "resource_overflow": 0.9
  }
}
```

**Response:**
```json
{
  "monitoring_session_id": "monitor_789",
  "tracked_processes": [
    {
      "process_id": "proc_123",
      "current_metrics": {
        "accuracy": 0.82,
        "efficiency": 0.74,
        "confidence": 0.68,
        "resource_usage": {"memory": 0.35, "attention": 0.62}
      },
      "trend_analysis": {
        "accuracy_trend": "stable",
        "efficiency_trend": "improving",
        "confidence_trend": "decreasing"
      },
      "alerts": []
    }
  ],
  "overall_assessment": {
    "system_health": 0.78,
    "performance_index": 0.73,
    "attention_allocation_efficiency": 0.81
  }
}
```

#### Get Performance Insights
```python
GET /api/v1/monitoring/insights/{monitoring_session_id}
```

**Response:**
```json
{
  "insights": {
    "performance_patterns": [
      {
        "pattern_id": "pattern_123",
        "description": "Accuracy decreases when confidence is high",
        "frequency": 0.73,
        "impact": "moderate",
        "recommendations": [
          "Implement confidence calibration",
          "Add uncertainty monitoring"
        ]
      }
    ],
    "bottlenecks_identified": [
      {
        "bottleneck_type": "attention_overload",
        "affected_processes": ["proc_123"],
        "severity": 0.65,
        "suggested_solutions": ["attention_reallocation", "task_prioritization"]
      }
    ],
    "optimization_opportunities": [
      {
        "opportunity_type": "strategy_switching",
        "expected_improvement": 0.15,
        "implementation_cost": 0.3,
        "confidence": 0.72
      }
    ]
  }
}
```

### Recursive Analysis API

#### Initiate Recursive Processing
```python
POST /api/v1/recursive/analyze
```

**Request Body:**
```json
{
  "analysis_target": {
    "type": "reflection_quality",
    "content": {
      "reflection_id": "refl_456",
      "focus_areas": ["coherence", "bias_detection", "insight_quality"]
    }
  },
  "recursion_config": {
    "max_depth": 4,
    "convergence_threshold": 0.05,
    "quality_threshold": 0.7,
    "timeout_seconds": 30
  },
  "termination_criteria": {
    "no_new_insights": true,
    "quality_plateau": true,
    "resource_limit": true
  }
}
```

**Response:**
```json
{
  "recursive_analysis_id": "recursive_789",
  "status": "processing",
  "current_depth": 1,
  "progress": {
    "levels_completed": 0,
    "insights_generated": 0,
    "quality_score": 0.0,
    "convergence_indicator": 0.0
  },
  "preliminary_insights": [],
  "estimated_completion": 25
}
```

#### Get Recursive Analysis Results
```python
GET /api/v1/recursive/analysis/{analysis_id}/results
```

**Response:**
```json
{
  "analysis_id": "recursive_789",
  "status": "completed",
  "final_depth": 3,
  "convergence_achieved": true,
  "processing_chain": [
    {
      "level": 1,
      "insights": [
        {
          "insight_id": "insight_123",
          "content": "Initial reflection shows overconfidence in reasoning accuracy",
          "confidence": 0.78,
          "novelty": 0.65,
          "utility": 0.82
        }
      ],
      "quality_score": 0.75
    },
    {
      "level": 2,
      "insights": [
        {
          "insight_id": "insight_124",
          "content": "Overconfidence stems from insufficient consideration of alternatives",
          "confidence": 0.72,
          "novelty": 0.58,
          "utility": 0.79
        }
      ],
      "quality_score": 0.70
    }
  ],
  "final_insights": [
    {
      "insight_id": "insight_125",
      "content": "Systematic bias toward initial hypotheses requires deliberate counter-evidence seeking",
      "confidence": 0.85,
      "actionable_recommendations": [
        "Implement devil's advocate protocol",
        "Set explicit counter-evidence seeking goals"
      ],
      "expected_impact": 0.25
    }
  ],
  "meta_analysis": {
    "recursive_quality": 0.73,
    "insight_coherence": 0.81,
    "practical_value": 0.77
  }
}
```

### Cognitive Control API

#### Generate Control Actions
```python
POST /api/v1/control/actions/generate
```

**Request Body:**
```json
{
  "analysis_results": {
    "reflection_id": "refl_456",
    "identified_issues": [
      {
        "issue_type": "confirmation_bias",
        "severity": 0.7,
        "affected_processes": ["reasoning", "evidence_evaluation"]
      },
      {
        "issue_type": "attention_misallocation",
        "severity": 0.5,
        "details": "60% attention on confirmed hypotheses, 40% on alternatives"
      }
    ]
  },
  "control_preferences": {
    "intervention_aggressiveness": 0.6,
    "prefer_gradual_changes": true,
    "maintain_performance": true,
    "risk_tolerance": 0.4
  },
  "context_constraints": {
    "time_pressure": 0.3,
    "resource_limitations": {"attention": 0.8, "memory": 0.9},
    "external_requirements": ["maintain_accuracy_above_0.7"]
  }
}
```

**Response:**
```json
{
  "control_plan_id": "control_plan_123",
  "recommended_actions": [
    {
      "action_id": "action_456",
      "action_type": "attention_redirect",
      "parameters": {
        "source": "confirmed_hypotheses",
        "target": "alternative_hypotheses",
        "intensity": 0.3,
        "duration": 120
      },
      "expected_outcome": {
        "bias_reduction": 0.25,
        "accuracy_change": 0.05,
        "efficiency_cost": -0.1
      },
      "urgency": 0.7,
      "confidence": 0.78
    },
    {
      "action_id": "action_457",
      "action_type": "strategy_adjustment",
      "parameters": {
        "target_strategy": "evidence_evaluation",
        "adjustment_type": "add_counter_evidence_step",
        "weight": 0.4
      },
      "expected_outcome": {
        "bias_reduction": 0.35,
        "thoroughness_increase": 0.2,
        "time_cost": 15
      },
      "urgency": 0.5,
      "confidence": 0.82
    }
  ],
  "execution_plan": {
    "sequential_order": ["action_456", "action_457"],
    "monitoring_checkpoints": [30, 60, 120],
    "success_criteria": {
      "bias_score_reduction": 0.2,
      "accuracy_maintenance": 0.7
    }
  }
}
```

#### Execute Control Actions
```python
POST /api/v1/control/actions/execute
```

**Request Body:**
```json
{
  "control_plan_id": "control_plan_123",
  "actions_to_execute": ["action_456", "action_457"],
  "execution_mode": "sequential",
  "monitoring_enabled": true,
  "rollback_on_failure": true
}
```

**Response:**
```json
{
  "execution_id": "exec_789",
  "status": "executing",
  "current_action": "action_456",
  "progress": 0.33,
  "executed_actions": [],
  "pending_actions": ["action_456", "action_457"],
  "monitoring_data": {
    "performance_changes": {},
    "side_effects": [],
    "success_indicators": {}
  }
}
```

### Knowledge and Memory API

#### Store Reflective Insights
```python
POST /api/v1/knowledge/insights
```

**Request Body:**
```json
{
  "insights": [
    {
      "insight_content": "Confirmation bias stronger under time pressure",
      "context": {
        "task_type": "decision_making",
        "time_pressure": 0.8,
        "complexity": 0.6
      },
      "validation": {
        "evidence_strength": 0.75,
        "reproducibility": 0.68,
        "generalizability": 0.54
      },
      "applications": [
        "Add time pressure detection",
        "Increase bias monitoring under time pressure"
      ]
    }
  ],
  "pattern_associations": [
    {
      "pattern_id": "pattern_time_pressure_bias",
      "strength": 0.73
    }
  ]
}
```

#### Query Knowledge Base
```python
GET /api/v1/knowledge/query
POST /api/v1/knowledge/query
```

**POST Request Body:**
```json
{
  "query_type": "similar_situations",
  "context": {
    "current_task": "complex_reasoning",
    "detected_biases": ["confirmation_bias", "anchoring_bias"],
    "performance_level": 0.72,
    "time_pressure": 0.6
  },
  "result_preferences": {
    "limit": 10,
    "min_similarity": 0.6,
    "include_outcomes": true,
    "include_strategies": true
  }
}
```

**Response:**
```json
{
  "matches": [
    {
      "situation_id": "situation_123",
      "similarity_score": 0.87,
      "context_match": {
        "task_similarity": 0.91,
        "bias_overlap": 0.85,
        "performance_similarity": 0.82
      },
      "historical_outcomes": {
        "successful_strategies": [
          {
            "strategy": "systematic_counter_evidence_search",
            "success_rate": 0.78,
            "improvement": 0.23
          }
        ],
        "failed_strategies": [
          {
            "strategy": "simple_bias_warning",
            "success_rate": 0.34,
            "issues": ["insufficient_motivation", "competing_priorities"]
          }
        ]
      },
      "recommendations": [
        "Apply systematic counter-evidence search",
        "Monitor attention allocation carefully"
      ]
    }
  ],
  "pattern_insights": [
    {
      "pattern": "Time pressure amplifies existing biases",
      "confidence": 0.82,
      "suggested_mitigations": ["Early bias detection", "Proactive attention control"]
    }
  ]
}
```

## WebSocket API for Real-Time Monitoring

### Connection and Authentication
```javascript
// WebSocket connection
const ws = new WebSocket('wss://api.example.com/v1/reflection/monitor');

// Authentication message
ws.send(JSON.stringify({
  type: 'authenticate',
  credentials: {
    session_id: 'refl_session_456',
    api_key: 'your_api_key'
  }
}));
```

### Real-Time Event Streams

#### Reflection Events
```javascript
// Subscribe to reflection events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['reflections', 'insights', 'control_actions'],
  filters: {
    min_confidence: 0.6,
    event_types: ['new_insight', 'bias_detected', 'action_executed']
  }
}));

// Example incoming events
{
  "type": "new_insight",
  "timestamp": 1635789123,
  "session_id": "refl_session_456",
  "data": {
    "insight_id": "insight_789",
    "content": "Pattern recognition improving with focused attention",
    "confidence": 0.78,
    "actionable": true
  }
}

{
  "type": "bias_detected",
  "timestamp": 1635789125,
  "session_id": "refl_session_456",
  "data": {
    "bias_type": "availability_heuristic",
    "strength": "moderate",
    "evidence": ["Recent examples weighted too heavily"],
    "suggested_interventions": ["Systematic example sampling"]
  }
}
```

#### Performance Monitoring
```javascript
{
  "type": "performance_update",
  "timestamp": 1635789127,
  "session_id": "refl_session_456",
  "data": {
    "metrics": {
      "accuracy": 0.84,
      "confidence_calibration": 0.76,
      "processing_efficiency": 0.82
    },
    "trends": {
      "accuracy": "improving",
      "efficiency": "stable",
      "bias_level": "decreasing"
    },
    "alerts": [
      {
        "alert_type": "attention_overload",
        "severity": "medium",
        "recommendation": "Consider task simplification"
      }
    ]
  }
}
```

## Error Handling and Status Codes

### HTTP Status Codes
```yaml
Success Responses:
  200: OK - Request successful
  201: Created - Resource created successfully
  202: Accepted - Request accepted for processing

Client Error Responses:
  400: Bad Request - Invalid request format
  401: Unauthorized - Authentication required
  403: Forbidden - Insufficient permissions
  404: Not Found - Resource not found
  409: Conflict - Resource state conflict
  422: Unprocessable Entity - Validation errors

Server Error Responses:
  500: Internal Server Error - Unexpected server error
  502: Bad Gateway - Integration service error
  503: Service Unavailable - System overload
  504: Gateway Timeout - Processing timeout
```

### Error Response Format
```json
{
  "error": {
    "code": "REFLECTION_ANALYSIS_FAILED",
    "message": "Recursive analysis failed to converge within time limit",
    "details": {
      "analysis_id": "recursive_789",
      "current_depth": 2,
      "timeout_seconds": 30,
      "convergence_score": 0.23
    },
    "suggestions": [
      "Reduce max_depth parameter",
      "Increase timeout_seconds",
      "Adjust convergence_threshold"
    ],
    "timestamp": "2023-11-01T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Rate Limiting
```yaml
Rate Limits:
  Reflection Sessions: 100 per hour per user
  Monitoring Requests: 1000 per hour per user
  Recursive Analysis: 20 per hour per user
  Knowledge Queries: 500 per hour per user

Headers:
  X-RateLimit-Limit: Maximum requests per time window
  X-RateLimit-Remaining: Remaining requests in current window
  X-RateLimit-Reset: Time when rate limit resets (Unix timestamp)
```

This comprehensive API specification provides all necessary interfaces for implementing and interacting with Form 19: Reflective Consciousness, supporting both synchronous request-response patterns and real-time streaming for continuous reflection monitoring and control.