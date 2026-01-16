# Form 23: Lucid Dream Consciousness - API Specification

## Core System APIs

### 1. Dream State Detection API

```python
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import asyncio

class LucidDreamStateAPI:
    """API for dream state detection and monitoring."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None

    async def detect_processing_state(self, 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current processing state and lucidity level.
        
        Args:
            context: Current processing context including sensory inputs,
                    cognitive state, and environmental factors
        
        Returns:
            {
                "state": "wake|transitional|light_dream|deep_dream|rem_equivalent|lucid_aware",
                "confidence": float,  # 0.0 to 1.0
                "lucidity_level": float,  # 0.0 to 1.0
                "stability": float,  # 0.0 to 1.0
                "duration_in_state": float,  # seconds
                "transition_probabilities": {
                    "wake": float,
                    "dream": float,
                    "lucid": float
                },
                "detection_timestamp": str,
                "processing_metrics": {
                    "sensory_input_level": float,
                    "internal_generation_rate": float,
                    "metacognitive_activity": float
                }
            }
        """
        endpoint = f"{self.base_url}/api/v1/state/detect"
        payload = {"context": context}
        # Implementation would make HTTP request
        return await self._post_request(endpoint, payload)

    async def monitor_state_transitions(self,
                                      duration: float,
                                      callback_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Monitor state transitions over specified duration.
        
        Args:
            duration: Monitoring duration in seconds
            callback_url: Optional callback for real-time notifications
        
        Returns:
            {
                "monitoring_id": str,
                "start_time": str,
                "duration": float,
                "transitions_detected": List[{
                    "timestamp": str,
                    "from_state": str,
                    "to_state": str,
                    "transition_confidence": float,
                    "trigger_factors": List[str]
                }],
                "state_summary": {
                    "total_time_lucid": float,
                    "total_time_dreaming": float,
                    "lucidity_episodes": int,
                    "average_lucidity_duration": float
                }
            }
        """
        endpoint = f"{self.base_url}/api/v1/state/monitor"
        payload = {
            "duration": duration,
            "callback_url": callback_url
        }
        return await self._post_request(endpoint, payload)

    async def calibrate_detection(self,
                                training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calibrate state detection algorithms with training data.
        
        Args:
            training_data: List of labeled state examples
        
        Returns:
            {
                "calibration_id": str,
                "calibration_success": bool,
                "accuracy_improvement": float,
                "updated_thresholds": Dict[str, float],
                "validation_metrics": {
                    "precision": float,
                    "recall": float,
                    "f1_score": float
                }
            }
        """
        endpoint = f"{self.base_url}/api/v1/state/calibrate"
        payload = {"training_data": training_data}
        return await self._post_request(endpoint, payload)
```

### 2. Reality Testing API

```python
class RealityTestingAPI:
    """API for reality testing and consistency validation."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def test_physical_consistency(self,
                                      environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test environment for physical law violations.
        
        Args:
            environment_state: Current environmental parameters and objects
        
        Returns:
            {
                "test_id": str,
                "is_consistent": bool,
                "overall_reality_score": float,  # 0.0 to 1.0
                "violations_detected": List[{
                    "type": "physical_law|spatial_impossibility|temporal_inconsistency",
                    "severity": float,
                    "confidence": float,
                    "description": str,
                    "location": Dict[str, Any]
                }],
                "lucidity_trigger_potential": float,
                "test_duration_ms": float
            }
        """
        endpoint = f"{self.base_url}/api/v1/reality/test-physical"
        payload = {"environment_state": environment_state}
        return await self._post_request(endpoint, payload)

    async def verify_memory_consistency(self,
                                      current_experience: Dict[str, Any],
                                      memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current experience with stored memories.
        
        Args:
            current_experience: Current experience data
            memory_context: Relevant memory context
        
        Returns:
            {
                "test_id": str,
                "consistency_score": float,
                "memory_matches": List[{
                    "memory_id": str,
                    "similarity": float,
                    "conflict_areas": List[str]
                }],
                "inconsistencies": List[{
                    "type": "factual|temporal|causal|emotional",
                    "description": str,
                    "confidence": float
                }],
                "reality_assessment": float
            }
        """
        endpoint = f"{self.base_url}/api/v1/reality/test-memory"
        payload = {
            "current_experience": current_experience,
            "memory_context": memory_context
        }
        return await self._post_request(endpoint, payload)

    async def comprehensive_reality_check(self,
                                        full_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete reality assessment across all dimensions.
        
        Args:
            full_context: Complete context including environment, memories, 
                        characters, and narrative elements
        
        Returns:
            {
                "assessment_id": str,
                "overall_reality_probability": float,
                "confidence": float,
                "component_scores": {
                    "physical_consistency": float,
                    "temporal_coherence": float,
                    "memory_alignment": float,
                    "character_believability": float,
                    "narrative_logic": float
                },
                "anomalies_summary": {
                    "total_anomalies": int,
                    "severity_distribution": Dict[str, int],
                    "lucidity_triggers": List[str]
                },
                "recommended_actions": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/reality/comprehensive-check"
        payload = {"context": full_context}
        return await self._post_request(endpoint, payload)
```

### 3. Lucidity Induction API

```python
class LucidityInductionAPI:
    """API for lucidity induction and maintenance."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def trigger_lucidity_check(self,
                                   trigger_type: str = "automatic",
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trigger a lucidity check and potential awareness induction.
        
        Args:
            trigger_type: Type of trigger ("automatic", "manual", "scheduled", "anomaly")
            context: Current context information
        
        Returns:
            {
                "trigger_id": str,
                "trigger_successful": bool,
                "lucidity_achieved": bool,
                "achieved_level": str,  # "none|partial|recognition|basic_control|advanced_control"
                "awareness_intensity": float,
                "stability_prediction": float,
                "induction_latency_ms": float,
                "recommended_next_steps": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/lucidity/trigger"
        payload = {
            "trigger_type": trigger_type,
            "context": context or {}
        }
        return await self._post_request(endpoint, payload)

    async def maintain_lucid_state(self,
                                 current_lucidity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maintain and stabilize current lucid state.
        
        Args:
            current_lucidity: Current lucidity state information
        
        Returns:
            {
                "maintenance_id": str,
                "stability_improved": bool,
                "new_stability_level": float,
                "maintenance_techniques_applied": List[str],
                "predicted_duration": float,
                "risk_factors": List[{
                    "factor": str,
                    "risk_level": float,
                    "mitigation": str
                }]
            }
        """
        endpoint = f"{self.base_url}/api/v1/lucidity/maintain"
        payload = {"current_lucidity": current_lucidity}
        return await self._post_request(endpoint, payload)

    async def enhance_lucidity_level(self,
                                   current_level: str,
                                   target_level: str) -> Dict[str, Any]:
        """
        Attempt to increase lucidity from current to target level.
        
        Args:
            current_level: Current lucidity level
            target_level: Desired lucidity level
        
        Returns:
            {
                "enhancement_id": str,
                "enhancement_successful": bool,
                "achieved_level": str,
                "enhancement_techniques": List[str],
                "effort_required": float,
                "success_probability": float,
                "side_effects": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/lucidity/enhance"
        payload = {
            "current_level": current_level,
            "target_level": target_level
        }
        return await self._post_request(endpoint, payload)
```

### 4. Dream Control API

```python
class DreamControlAPI:
    """API for dream content control and manipulation."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def modify_environment(self,
                               modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify environmental aspects of current dream.
        
        Args:
            modifications: Requested environmental changes
        
        Returns:
            {
                "control_id": str,
                "modifications_applied": Dict[str, Any],
                "success_rate": float,
                "partial_successes": List[str],
                "failures": List[{
                    "modification": str,
                    "reason": str,
                    "alternative_suggestions": List[str]
                }],
                "stability_impact": float,
                "effort_required": float
            }
        """
        endpoint = f"{self.base_url}/api/v1/control/environment"
        payload = {"modifications": modifications}
        return await self._post_request(endpoint, payload)

    async def direct_narrative(self,
                             narrative_direction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Influence story progression and plot development.
        
        Args:
            narrative_direction: Desired narrative changes and direction
        
        Returns:
            {
                "control_id": str,
                "narrative_changes_applied": Dict[str, Any],
                "plot_influence_success": float,
                "character_response_quality": float,
                "narrative_coherence_maintained": bool,
                "unexpected_developments": List[str],
                "control_maintenance_effort": float
            }
        """
        endpoint = f"{self.base_url}/api/v1/control/narrative"
        payload = {"narrative_direction": narrative_direction}
        return await self._post_request(endpoint, payload)

    async def summon_character(self,
                             character_specification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or summon a character in the dream.
        
        Args:
            character_specification: Character attributes and properties
        
        Returns:
            {
                "control_id": str,
                "character_created": bool,
                "character_id": str,
                "actual_attributes": Dict[str, Any],
                "responsiveness_level": float,
                "integration_quality": float,
                "control_difficulty": float
            }
        """
        endpoint = f"{self.base_url}/api/v1/control/summon-character"
        payload = {"character_specification": character_specification}
        return await self._post_request(endpoint, payload)

    async def manipulate_sensory_experience(self,
                                          sensory_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Alter sensory aspects of the dream experience.
        
        Args:
            sensory_changes: Requested changes to visual, auditory, tactile, etc. experiences
        
        Returns:
            {
                "control_id": str,
                "sensory_modifications": Dict[str, Any],
                "enhancement_success": Dict[str, float],
                "sensory_clarity": Dict[str, float],
                "immersion_impact": float,
                "stability_considerations": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/control/sensory"
        payload = {"sensory_changes": sensory_changes}
        return await self._post_request(endpoint, payload)
```

### 5. Dream Memory Management API

```python
class DreamMemoryAPI:
    """API for dream memory storage, retrieval, and integration."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def store_dream_experience(self,
                                   experience_data: Dict[str, Any],
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store current dream experience in memory.
        
        Args:
            experience_data: Complete dream experience data
            metadata: Additional metadata about the experience
        
        Returns:
            {
                "memory_id": str,
                "storage_successful": bool,
                "memory_quality_score": float,
                "compression_ratio": float,
                "significant_elements_preserved": List[str],
                "automatic_tags": List[str],
                "integration_recommendations": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/memory/store"
        payload = {
            "experience_data": experience_data,
            "metadata": metadata
        }
        return await self._post_request(endpoint, payload)

    async def retrieve_dream_memories(self,
                                    query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve dream memories based on query parameters.
        
        Args:
            query: Search parameters and filters
        
        Returns:
            {
                "query_id": str,
                "total_matches": int,
                "memories": List[{
                    "memory_id": str,
                    "experience_summary": str,
                    "timestamp": str,
                    "lucidity_level": float,
                    "control_achievements": List[str],
                    "insights_gained": List[str],
                    "relevance_score": float
                }],
                "search_time_ms": float,
                "suggested_refinements": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/memory/retrieve"
        payload = {"query": query}
        return await self._post_request(endpoint, payload)

    async def integrate_with_autobiographical_memory(self,
                                                   memory_id: str) -> Dict[str, Any]:
        """
        Integrate dream memory with autobiographical memory system.
        
        Args:
            memory_id: ID of dream memory to integrate
        
        Returns:
            {
                "integration_id": str,
                "integration_successful": bool,
                "integration_quality": float,
                "autobiographical_connections": List[{
                    "connection_type": str,
                    "related_memory_id": str,
                    "connection_strength": float
                }],
                "narrative_contributions": List[str],
                "insights_extracted": List[str],
                "learning_outcomes": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/memory/integrate"
        payload = {"memory_id": memory_id}
        return await self._post_request(endpoint, payload)

    async def analyze_dream_patterns(self,
                                   analysis_scope: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns across multiple dream memories.
        
        Args:
            analysis_scope: Scope and parameters for pattern analysis
        
        Returns:
            {
                "analysis_id": str,
                "patterns_identified": List[{
                    "pattern_type": str,
                    "description": str,
                    "frequency": float,
                    "significance": float,
                    "related_memories": List[str]
                }],
                "recurring_themes": List[str],
                "skill_development_trends": Dict[str, float],
                "therapeutic_progress_indicators": Dict[str, float],
                "recommendations": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/memory/analyze-patterns"
        payload = {"analysis_scope": analysis_scope}
        return await self._post_request(endpoint, payload)
```

### 6. Session Management API

```python
class LucidDreamSessionAPI:
    """API for managing complete lucid dream sessions."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def create_session(self,
                           session_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and initialize a new lucid dream session.
        
        Args:
            session_config: Session configuration and goals
        
        Returns:
            {
                "session_id": str,
                "session_created": bool,
                "estimated_duration": float,
                "preparation_requirements": List[str],
                "success_probability": float,
                "recommended_techniques": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/session/create"
        payload = {"config": session_config}
        return await self._post_request(endpoint, payload)

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current status of an active session.
        
        Args:
            session_id: ID of session to check
        
        Returns:
            {
                "session_id": str,
                "status": "preparation|induction|lucid_exploration|completion",
                "current_lucidity_level": float,
                "time_elapsed": float,
                "goals_progress": Dict[str, float],
                "current_activities": List[str],
                "stability_level": float,
                "recommended_actions": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/session/{session_id}/status"
        return await self._get_request(endpoint)

    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Complete and analyze a lucid dream session.
        
        Args:
            session_id: ID of session to complete
        
        Returns:
            {
                "session_id": str,
                "completion_successful": bool,
                "session_summary": {
                    "total_duration": float,
                    "lucid_duration": float,
                    "peak_lucidity_level": float,
                    "goals_achieved": List[str],
                    "control_successes": List[str],
                    "insights_gained": List[str]
                },
                "performance_metrics": {
                    "lucidity_stability": float,
                    "control_proficiency": float,
                    "memory_quality": float,
                    "overall_success": float
                },
                "learning_outcomes": List[str],
                "recommendations_for_improvement": List[str]
            }
        """
        endpoint = f"{self.base_url}/api/v1/session/{session_id}/complete"
        return await self._post_request(endpoint, {})
```

### 7. Configuration and Monitoring APIs

```python
class SystemConfigurationAPI:
    """API for system configuration and performance monitoring."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url

    async def update_user_profile(self,
                                user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile and personalization settings.
        
        Args:
            user_profile: Updated user profile information
        
        Returns:
            {
                "profile_updated": bool,
                "optimization_changes": List[str],
                "personalization_improvements": Dict[str, float],
                "recommended_settings": Dict[str, Any]
            }
        """
        endpoint = f"{self.base_url}/api/v1/config/user-profile"
        payload = {"user_profile": user_profile}
        return await self._put_request(endpoint, payload)

    async def get_performance_metrics(self,
                                    time_range: Dict[str, str]) -> Dict[str, Any]:
        """
        Get system performance metrics over specified time range.
        
        Args:
            time_range: Start and end times for metrics collection
        
        Returns:
            {
                "metrics_id": str,
                "time_range": Dict[str, str],
                "lucidity_statistics": {
                    "frequency": float,
                    "average_duration": float,
                    "peak_levels_achieved": Dict[str, int]
                },
                "control_statistics": {
                    "success_rates_by_domain": Dict[str, float],
                    "improvement_trends": Dict[str, float]
                },
                "memory_statistics": {
                    "memories_stored": int,
                    "integration_quality": float,
                    "insights_generated": int
                },
                "system_performance": {
                    "average_response_time": float,
                    "reliability_score": float,
                    "error_rate": float
                }
            }
        """
        endpoint = f"{self.base_url}/api/v1/monitoring/performance"
        params = {"time_range": time_range}
        return await self._get_request(endpoint, params)

    async def optimize_system_settings(self,
                                     optimization_goals: List[str]) -> Dict[str, Any]:
        """
        Optimize system settings for specified goals.
        
        Args:
            optimization_goals: List of optimization objectives
        
        Returns:
            {
                "optimization_id": str,
                "optimization_successful": bool,
                "settings_changes": Dict[str, Any],
                "expected_improvements": Dict[str, float],
                "validation_required": bool,
                "rollback_available": bool
            }
        """
        endpoint = f"{self.base_url}/api/v1/config/optimize"
        payload = {"optimization_goals": optimization_goals}
        return await self._post_request(endpoint, payload)

    # Helper methods for HTTP requests
    async def _get_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        # Implementation would make actual HTTP GET request
        pass

    async def _post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation would make actual HTTP POST request
        pass

    async def _put_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation would make actual HTTP PUT request
        pass
```

## WebSocket Real-Time APIs

### Real-Time State Monitoring

```python
import websockets
import json

class RealTimeLucidDreamAPI:
    """WebSocket API for real-time lucid dream monitoring and control."""

    def __init__(self, websocket_url: str = "ws://localhost:8080/ws"):
        self.websocket_url = websocket_url
        self.connection = None

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            self.connection = await websockets.connect(self.websocket_url)
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def subscribe_to_state_changes(self,
                                       callback: callable,
                                       filters: Optional[Dict[str, Any]] = None):
        """
        Subscribe to real-time state change notifications.
        
        Args:
            callback: Function to call when state changes are received
            filters: Optional filters for which changes to receive
        """
        subscription_message = {
            "type": "subscribe",
            "channel": "state_changes",
            "filters": filters or {}
        }
        
        await self.connection.send(json.dumps(subscription_message))
        
        async for message in self.connection:
            data = json.loads(message)
            if data.get("channel") == "state_changes":
                await callback(data.get("payload"))

    async def send_real_time_control(self,
                                   control_command: Dict[str, Any]) -> bool:
        """
        Send real-time control command.
        
        Args:
            control_command: Control command to execute
        
        Returns:
            Success status of command transmission
        """
        command_message = {
            "type": "control_command",
            "payload": control_command,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.connection.send(json.dumps(command_message))
            return True
        except Exception as e:
            print(f"Command transmission failed: {e}")
            return False
```

## Error Handling and Response Codes

### HTTP Status Codes
- 200: Success
- 201: Resource created successfully
- 400: Bad request (invalid parameters)
- 401: Unauthorized access
- 403: Forbidden operation
- 404: Resource not found
- 409: Conflict (e.g., session already active)
- 422: Unprocessable entity (validation failed)
- 429: Rate limit exceeded
- 500: Internal server error
- 503: Service unavailable

### Error Response Format
```json
{
    "error": {
        "code": "LUCIDITY_INDUCTION_FAILED",
        "message": "Failed to achieve lucid state",
        "details": {
            "reason": "insufficient_stability",
            "current_state": "light_dream",
            "recommendations": [
                "improve_state_stability",
                "retry_with_different_technique"
            ]
        },
        "timestamp": "2025-09-27T10:30:00Z",
        "request_id": "req_12345"
    }
}
```

This comprehensive API specification provides complete programmatic access to all lucid dream consciousness functionality, supporting both synchronous operations and real-time monitoring and control scenarios.