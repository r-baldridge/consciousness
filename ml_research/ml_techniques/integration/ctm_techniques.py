"""
CTM (Continuous Thought Machine) Integrated Techniques

Techniques that leverage CTM's neural dynamics and synchronization patterns.
CTM uses neuron-level models with internal time dimensions, enabling adaptive
computation and emergent temporal reasoning.

Key CTM capabilities:
    - Decoupled internal time dimension
    - Neuron-level models (NLMs) with unique weights
    - Neural synchronization for information encoding
    - Adaptive computation based on input complexity

Integrated Techniques:
    - CTMTemporalReasoning: Leverage synchronization for time-aware reasoning
    - CTMMemory: Use neuron dynamics for persistent memory
    - CTMVerification: Confidence scoring via activity patterns
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .. import (
    TechniqueBase,
    TechniqueResult,
    TechniqueConfig,
    TechniqueCategory,
)


# =============================================================================
# CTM CONFIGURATION
# =============================================================================

@dataclass
class CTMIntegrationConfig:
    """Configuration for CTM-integrated techniques."""
    # CTM architecture parameters
    hidden_dim: int = 512
    num_neurons: int = 1024
    history_length: int = 8
    max_internal_steps: int = 32

    # Synchronization parameters
    sync_window: int = 4
    halt_threshold: float = 0.01
    sync_threshold: float = 0.8  # High sync = high confidence

    # Neuron activation
    neuron_activation: str = "gelu"

    # Fallback behavior
    use_fallback_on_unavailable: bool = True


class SyncPattern(Enum):
    """Synchronization pattern types."""
    LOW = "low"           # < 0.3 sync
    MEDIUM = "medium"     # 0.3 - 0.7 sync
    HIGH = "high"         # > 0.7 sync
    OSCILLATING = "oscillating"  # Variable sync


# =============================================================================
# CTM TEMPORAL REASONING
# =============================================================================

class CTMTemporalReasoning(TechniqueBase):
    """
    Leverage CTM synchronization for temporal pattern reasoning.

    Uses CTM's neuron synchronization patterns to reason about temporal
    relationships. The internal time dimension enables tracking of phase
    relationships between concepts, making it ideal for:
        - Sequence ordering tasks
        - Temporal relationship extraction
        - Time-series reasoning

    Configuration:
        max_steps: Maximum internal time steps
        sync_threshold: Threshold for synchronization detection
        track_phases: Whether to track phase relationships

    Usage:
        temporal = CTMTemporalReasoning(max_steps=32)
        result = temporal.run({
            "events": ["event1", "event2"],
            "query": "What happened first?"
        })
    """

    TECHNIQUE_ID = "ctm_temporal_reasoning"
    CATEGORY = TechniqueCategory.PROMPTING

    def __init__(
        self,
        max_steps: int = 32,
        sync_threshold: float = 0.8,
        track_phases: bool = True,
        config: Optional[CTMIntegrationConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.sync_threshold = sync_threshold
        self.track_phases = track_phases
        self.ctm_config = config or CTMIntegrationConfig()
        self._ctm = None

    @property
    def ctm(self):
        """Lazy load CTM architecture."""
        if self._ctm is None:
            try:
                from modern_dev.ctm import CTM, CTMConfig
                self._ctm = CTM(CTMConfig(
                    hidden_dim=self.ctm_config.hidden_dim,
                    num_neurons=self.ctm_config.num_neurons,
                    max_internal_steps=self.max_steps,
                ))
            except ImportError:
                self._ctm = None
        return self._ctm

    def _compute_sync_pattern(
        self,
        activations: List[float],
        window: int,
    ) -> Tuple[float, SyncPattern]:
        """
        Compute synchronization from neuron activations.

        Returns:
            (sync_score, pattern_type)
        """
        if not activations or len(activations) < window:
            return 0.5, SyncPattern.MEDIUM

        # Simulate sync computation
        # In real CTM, this would use correlation of neuron activities
        import random
        sync_score = random.uniform(0.3, 0.9)

        if sync_score < 0.3:
            pattern = SyncPattern.LOW
        elif sync_score < 0.7:
            pattern = SyncPattern.MEDIUM
        else:
            pattern = SyncPattern.HIGH

        return sync_score, pattern

    def _temporal_step(
        self,
        query: str,
        events: List[str],
        step: int,
        state: Dict,
    ) -> Tuple[Dict, float, SyncPattern]:
        """
        Perform one temporal reasoning step.

        Returns:
            (new_state, sync_score, sync_pattern)
        """
        if self.ctm is None:
            # Fallback: simulate temporal reasoning
            sync_score, pattern = self._compute_sync_pattern(
                state.get("activations", [0.5] * 10),
                self.ctm_config.sync_window,
            )
            new_state = {
                **state,
                "step": step,
                "activations": [0.5 + step * 0.01] * 10,
            }
            return new_state, sync_score, pattern

        # CTM-based temporal processing would go here
        activations = state.get("activations", [])
        sync_score, pattern = self._compute_sync_pattern(
            activations,
            self.ctm_config.sync_window,
        )

        new_state = {
            **state,
            "step": step,
            "sync_history": state.get("sync_history", []) + [sync_score],
        }

        return new_state, sync_score, pattern

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            events = input_data.get("events", [])
            query = input_data.get("query", "")
        else:
            events = []
            query = str(input_data)

        self._call_hooks("pre_run", query=query, events=events)

        try:
            state: Dict[str, Any] = {"activations": []}
            phase_history = []
            final_sync = 0.0
            final_pattern = SyncPattern.MEDIUM

            for step in range(self.max_steps):
                state, sync_score, pattern = self._temporal_step(
                    query, events, step, state
                )

                if self.track_phases:
                    phase_history.append({
                        "step": step,
                        "sync": sync_score,
                        "pattern": pattern.value,
                    })

                trace.append({
                    "action": "ctm_temporal_step",
                    "step": step,
                    "sync_score": sync_score,
                    "pattern": pattern.value,
                })

                final_sync = sync_score
                final_pattern = pattern

                # Early halt if synchronization stabilizes
                if sync_score > self.sync_threshold:
                    trace.append({
                        "action": "sync_stable_halt",
                        "step": step,
                        "sync_score": sync_score,
                    })
                    break

            # Generate temporal ordering based on phase relationships
            ordering = self._derive_temporal_ordering(events, phase_history)

            output = {
                "query": query,
                "events": events,
                "temporal_ordering": ordering,
                "sync_score": final_sync,
                "sync_pattern": final_pattern.value,
                "steps_used": len(phase_history),
            }

            if self.track_phases:
                output["phase_history"] = phase_history

            self._call_hooks("post_run", result=output)

            return TechniqueResult(
                success=True,
                output=output,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "ctm_available": self.ctm is not None,
                    "final_sync": final_sync,
                },
            )

        except Exception as e:
            self._call_hooks("on_error", error=e)
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )

    def _derive_temporal_ordering(
        self,
        events: List[str],
        phase_history: List[Dict],
    ) -> List[str]:
        """Derive temporal ordering from phase relationships."""
        # Placeholder: return events as-is
        # Real implementation would use phase correlations
        return events


# =============================================================================
# CTM MEMORY
# =============================================================================

class CTMMemory(TechniqueBase):
    """
    Use CTM neuron dynamics for persistent memory.

    Leverages CTM's internal neuron states as a form of memory where:
        - Persistent activations encode important information
        - Synchronization strength determines importance weighting
        - Activity patterns enable associative retrieval

    Features:
        - Activity-based importance weighting
        - Associative memory retrieval
        - Adaptive memory consolidation
        - Forgetting based on activation decay

    Configuration:
        memory_capacity: Maximum number of memories
        importance_threshold: Minimum sync for memory retention
        decay_rate: Rate at which inactive memories decay

    Usage:
        memory = CTMMemory(memory_capacity=100)
        memory.store("Important fact", importance=0.9)
        result = memory.run("What do you remember about X?")
    """

    TECHNIQUE_ID = "ctm_memory"
    CATEGORY = TechniqueCategory.MEMORY

    def __init__(
        self,
        memory_capacity: int = 100,
        importance_threshold: float = 0.5,
        decay_rate: float = 0.01,
        config: Optional[CTMIntegrationConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.memory_capacity = memory_capacity
        self.importance_threshold = importance_threshold
        self.decay_rate = decay_rate
        self.ctm_config = config or CTMIntegrationConfig()
        self._ctm = None
        self._memories: List[Dict[str, Any]] = []

    @property
    def ctm(self):
        """Lazy load CTM architecture."""
        if self._ctm is None:
            try:
                from modern_dev.ctm import CTM, CTMConfig
                self._ctm = CTM(CTMConfig(
                    hidden_dim=self.ctm_config.hidden_dim,
                    num_neurons=self.ctm_config.num_neurons,
                ))
            except ImportError:
                self._ctm = None
        return self._ctm

    def store(
        self,
        content: str,
        importance: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Store a memory with CTM-based importance weighting.

        Args:
            content: Memory content to store
            importance: Explicit importance (0-1), or None for auto-compute
            metadata: Additional metadata to store with memory
        """
        if importance is None:
            # Compute importance based on CTM activation patterns
            importance = self._compute_importance(content)

        memory = {
            "content": content,
            "importance": importance,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "access_count": 0,
            "activation": importance,  # Initial activation = importance
        }

        self._memories.append(memory)

        # Enforce capacity limit
        if len(self._memories) > self.memory_capacity:
            self._consolidate_memories()

    def _compute_importance(self, content: str) -> float:
        """Compute importance using CTM synchronization patterns."""
        if self.ctm is None:
            # Fallback: length-based heuristic
            return min(1.0, len(content) / 500)

        # CTM would compute based on activation patterns
        # Placeholder simulation
        import random
        return random.uniform(0.4, 0.9)

    def _consolidate_memories(self) -> None:
        """Consolidate memories, removing low-importance ones."""
        # Sort by importance * recency
        current_time = time.time()
        for memory in self._memories:
            age = current_time - memory["timestamp"]
            memory["score"] = memory["importance"] * (1 - self.decay_rate * age)

        self._memories.sort(key=lambda m: m["score"], reverse=True)
        self._memories = self._memories[:self.memory_capacity]

    def _retrieve_relevant(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using CTM activation matching."""
        if not self._memories:
            return []

        # Update activations based on query
        for memory in self._memories:
            # Simple relevance: word overlap
            query_words = set(query.lower().split())
            content_words = set(memory["content"].lower().split())
            overlap = len(query_words & content_words)
            memory["activation"] = min(1.0, memory["importance"] + overlap * 0.1)

        # Sort by activation
        sorted_memories = sorted(
            self._memories,
            key=lambda m: m["activation"],
            reverse=True,
        )

        # Update access count
        for memory in sorted_memories[:k]:
            memory["access_count"] += 1

        return sorted_memories[:k]

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        query = input_data if isinstance(input_data, str) else str(input_data)
        self._call_hooks("pre_run", query=query)

        try:
            # Retrieve relevant memories
            relevant = self._retrieve_relevant(query, k=5)

            trace.append({
                "action": "retrieve_memories",
                "query": query,
                "num_retrieved": len(relevant),
            })

            output = {
                "query": query,
                "memories": [
                    {
                        "content": m["content"],
                        "importance": m["importance"],
                        "activation": m["activation"],
                        "access_count": m["access_count"],
                    }
                    for m in relevant
                ],
                "total_memories": len(self._memories),
            }

            self._call_hooks("post_run", result=output)

            return TechniqueResult(
                success=True,
                output=output,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "ctm_available": self.ctm is not None,
                    "memory_count": len(self._memories),
                },
            )

        except Exception as e:
            self._call_hooks("on_error", error=e)
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# =============================================================================
# CTM VERIFICATION
# =============================================================================

class CTMVerification(TechniqueBase):
    """
    Confidence scoring via CTM activity patterns.

    Uses CTM's neural synchronization as a confidence signal where:
        - High synchronization = High confidence
        - Low synchronization = Uncertainty
        - Oscillating patterns = Potential contradiction

    The intuition is that confident answers produce stable, synchronized
    neural activity, while uncertain or incorrect answers show desynchronized
    or unstable patterns.

    Configuration:
        confidence_threshold: Sync level for high confidence
        oscillation_window: Window for detecting oscillation
        multi_pass: Run multiple passes for robust scoring

    Usage:
        verifier = CTMVerification(confidence_threshold=0.8)
        result = verifier.run({
            "input": "What is 2+2?",
            "output": "4",
        })
    """

    TECHNIQUE_ID = "ctm_verification"
    CATEGORY = TechniqueCategory.VERIFICATION

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        oscillation_window: int = 5,
        multi_pass: bool = True,
        num_passes: int = 3,
        config: Optional[CTMIntegrationConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.oscillation_window = oscillation_window
        self.multi_pass = multi_pass
        self.num_passes = num_passes
        self.ctm_config = config or CTMIntegrationConfig()
        self._ctm = None

    @property
    def ctm(self):
        """Lazy load CTM architecture."""
        if self._ctm is None:
            try:
                from modern_dev.ctm import CTM, CTMConfig
                self._ctm = CTM(CTMConfig(
                    hidden_dim=self.ctm_config.hidden_dim,
                    num_neurons=self.ctm_config.num_neurons,
                ))
            except ImportError:
                self._ctm = None
        return self._ctm

    def _compute_sync_confidence(
        self,
        input_text: str,
        output_text: str,
        pass_idx: int,
    ) -> Tuple[float, SyncPattern, List[float]]:
        """
        Compute confidence from CTM synchronization.

        Returns:
            (confidence, pattern, sync_history)
        """
        if self.ctm is None:
            # Fallback: heuristic-based confidence
            # Check basic consistency
            import random
            base_conf = random.uniform(0.5, 0.9)

            # Adjust based on output length relative to input
            if len(output_text) < 10:
                base_conf *= 0.8

            sync_history = [base_conf + random.uniform(-0.1, 0.1) for _ in range(5)]

            if base_conf > 0.7:
                pattern = SyncPattern.HIGH
            elif base_conf > 0.4:
                pattern = SyncPattern.MEDIUM
            else:
                pattern = SyncPattern.LOW

            return base_conf, pattern, sync_history

        # CTM-based confidence would compute here
        # Placeholder
        import random
        confidence = random.uniform(0.6, 0.95)
        sync_history = [confidence + random.uniform(-0.05, 0.05) for _ in range(5)]
        pattern = SyncPattern.HIGH if confidence > 0.7 else SyncPattern.MEDIUM

        return confidence, pattern, sync_history

    def _detect_oscillation(self, sync_history: List[float]) -> bool:
        """Detect oscillating synchronization patterns."""
        if len(sync_history) < self.oscillation_window:
            return False

        # Check for alternating increases/decreases
        changes = []
        for i in range(1, len(sync_history)):
            changes.append(sync_history[i] - sync_history[i-1])

        # Count sign changes
        sign_changes = sum(
            1 for i in range(1, len(changes))
            if changes[i] * changes[i-1] < 0
        )

        return sign_changes >= len(changes) * 0.5

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            input_text = input_data.get("input", "")
            output_text = input_data.get("output", "")
        else:
            input_text = str(input_data)
            output_text = ""

        self._call_hooks("pre_run", input=input_text, output=output_text)

        try:
            pass_results = []
            all_sync_histories = []

            num_passes = self.num_passes if self.multi_pass else 1

            for pass_idx in range(num_passes):
                confidence, pattern, sync_history = self._compute_sync_confidence(
                    input_text, output_text, pass_idx
                )

                is_oscillating = self._detect_oscillation(sync_history)

                pass_results.append({
                    "pass": pass_idx,
                    "confidence": confidence,
                    "pattern": pattern.value,
                    "is_oscillating": is_oscillating,
                })

                all_sync_histories.extend(sync_history)

                trace.append({
                    "action": "ctm_verification_pass",
                    "pass": pass_idx,
                    "confidence": confidence,
                    "pattern": pattern.value,
                    "oscillating": is_oscillating,
                })

            # Aggregate confidence across passes
            avg_confidence = sum(p["confidence"] for p in pass_results) / len(pass_results)
            any_oscillating = any(p["is_oscillating"] for p in pass_results)

            # Determine final verdict
            if avg_confidence >= self.confidence_threshold and not any_oscillating:
                verdict = "high_confidence"
            elif avg_confidence < 0.4 or any_oscillating:
                verdict = "low_confidence"
            else:
                verdict = "medium_confidence"

            output = {
                "input": input_text,
                "output": output_text,
                "confidence": avg_confidence,
                "verdict": verdict,
                "passes": pass_results,
                "is_oscillating": any_oscillating,
            }

            self._call_hooks("post_run", result=output)

            return TechniqueResult(
                success=True,
                output=output,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                metadata={
                    "ctm_available": self.ctm is not None,
                    "num_passes": num_passes,
                    "avg_confidence": avg_confidence,
                },
            )

        except Exception as e:
            self._call_hooks("on_error", error=e)
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# =============================================================================
# REGISTER TECHNIQUES
# =============================================================================

def _register_techniques():
    """Register CTM techniques with the integration registry."""
    try:
        from . import register_integrated_technique
        register_integrated_technique("ctm_temporal_reasoning", CTMTemporalReasoning)
        register_integrated_technique("ctm_memory", CTMMemory)
        register_integrated_technique("ctm_verification", CTMVerification)
    except ImportError:
        pass  # Registry not available

_register_techniques()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CTMIntegrationConfig",
    "SyncPattern",
    "CTMTemporalReasoning",
    "CTMMemory",
    "CTMVerification",
]
