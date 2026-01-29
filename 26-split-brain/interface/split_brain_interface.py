#!/usr/bin/env python3
"""
Split-Brain Consciousness Interface

Form 26: Models the dual consciousness phenomena observed after surgical
severing of the corpus callosum (callosotomy). Split-brain patients
exhibit a fascinating dissociation where the left and right hemispheres
operate as semi-independent conscious agents, each with specialized
processing capabilities.

This form explores hemispheric specialization (left: verbal, analytical;
right: spatial, holistic), inter-hemispheric conflict, confabulation
(the left hemisphere's tendency to generate explanations for right
hemisphere-driven behaviors), and the implications for theories of
unified consciousness.

Key phenomena modeled:
- Lateralized stimulus processing
- Hemispheric specialization and response
- Inter-hemispheric conflict detection
- Left-hemisphere confabulation
- Cross-cueing (indirect inter-hemispheric communication)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class Hemisphere(Enum):
    """
    Brain hemispheres.

    In split-brain patients, each hemisphere processes information from
    the contralateral visual field and controls the contralateral hand.
    """
    LEFT = "left"      # Typically: language, analytical, sequential
    RIGHT = "right"    # Typically: spatial, holistic, emotional


class ProcessingDomain(Enum):
    """
    Cognitive processing domains associated with hemispheric specialization.

    While the lateralization is not absolute, these represent the
    dominant hemisphere for each function in most right-handed individuals.
    """
    VERBAL = "verbal"                # Language production and comprehension (left)
    SPATIAL = "spatial"              # Visuospatial processing (right)
    EMOTIONAL = "emotional"          # Emotional processing and recognition (right)
    ANALYTICAL = "analytical"        # Logical/sequential analysis (left)
    HOLISTIC = "holistic"            # Pattern recognition, gestalt processing (right)
    MOTOR_SPEECH = "motor_speech"    # Speech motor control (left)


class InterhemisphericState(Enum):
    """
    State of inter-hemispheric interaction.

    In split-brain, the corpus callosum is severed, preventing direct
    communication. However, indirect cross-cueing can occur through
    subcortical pathways or external cues.
    """
    DISCONNECTED = "disconnected"    # No communication (complete callosotomy)
    PARTIAL = "partial"              # Some fibers intact (partial callosotomy)
    CROSS_CUEING = "cross_cueing"    # Indirect communication via subcortical or external
    INTEGRATED = "integrated"        # Normal (intact corpus callosum)


class ConflictType(Enum):
    """Types of inter-hemispheric conflict in split-brain patients."""
    MOTOR_CONFLICT = "motor_conflict"        # Alien hand: hands acting at cross purposes
    PERCEPTUAL_CONFLICT = "perceptual_conflict"  # Different percepts in each hemisphere
    DECISIONAL_CONFLICT = "decisional_conflict"  # Different decisions by each hemisphere
    EMOTIONAL_CONFLICT = "emotional_conflict"    # Emotional reactions conflict


class ConfabulationType(Enum):
    """
    Types of confabulation produced by the left hemisphere.

    When the left hemisphere observes behavior initiated by the right
    hemisphere (of which it has no direct knowledge), it generates
    plausible but incorrect explanations.
    """
    CAUSAL_ATTRIBUTION = "causal_attribution"    # Inventing a cause for behavior
    POST_HOC_RATIONALIZATION = "post_hoc_rationalization"  # Rationalizing after the fact
    GAP_FILLING = "gap_filling"                  # Filling memory gaps
    NARRATIVE_CONSTRUCTION = "narrative_construction"  # Building a coherent story


class LateralizedField(Enum):
    """Visual field for lateralized stimulus presentation."""
    LEFT_VISUAL_FIELD = "left_visual_field"      # Projects to right hemisphere
    RIGHT_VISUAL_FIELD = "right_visual_field"    # Projects to left hemisphere
    BILATERAL = "bilateral"                       # Both fields simultaneously
    CENTRAL = "central"                           # Central fixation


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SplitBrainInput:
    """
    Input representing a lateralized stimulus presented to a split-brain subject.

    In split-brain experiments, stimuli are briefly flashed to one visual
    field while the patient fixates centrally, ensuring only one hemisphere
    receives the information.
    """
    visual_field: LateralizedField
    stimulus_content: str                 # Description of the stimulus
    stimulus_type: str                    # image, word, object, face, emotion
    processing_domain: ProcessingDomain   # Which domain is engaged
    stimulus_duration_ms: float = 150.0   # Brief to prevent eye movement
    task_type: str = "identify"           # identify, match, respond, draw
    response_modality: str = "verbal"     # verbal, pointing, drawing, left_hand, right_hand
    interhemispheric_state: InterhemisphericState = InterhemisphericState.DISCONNECTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visual_field": self.visual_field.value,
            "stimulus_content": self.stimulus_content,
            "stimulus_type": self.stimulus_type,
            "processing_domain": self.processing_domain.value,
            "stimulus_duration_ms": self.stimulus_duration_ms,
            "task_type": self.task_type,
            "response_modality": self.response_modality,
            "interhemispheric_state": self.interhemispheric_state.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BilateralInput:
    """Input for bilateral stimulus presentation (different stimuli to each field)."""
    left_field_stimulus: str              # Stimulus to left visual field (right hemisphere)
    right_field_stimulus: str             # Stimulus to right visual field (left hemisphere)
    stimulus_type: str = "image"
    task_type: str = "identify"
    response_modality: str = "verbal"
    interhemispheric_state: InterhemisphericState = InterhemisphericState.DISCONNECTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_field_stimulus": self.left_field_stimulus,
            "right_field_stimulus": self.right_field_stimulus,
            "stimulus_type": self.stimulus_type,
            "task_type": self.task_type,
            "response_modality": self.response_modality,
            "interhemispheric_state": self.interhemispheric_state.value,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class HemisphereResponse:
    """
    Response generated by a single hemisphere.

    Each hemisphere may generate its own response to a stimulus,
    and these responses may differ or conflict.
    """
    hemisphere: Hemisphere
    response_content: str                 # What the hemisphere "says" or does
    processing_domain: ProcessingDomain
    confidence: float                     # 0.0-1.0: processing confidence
    can_verbalize: bool                   # Can this hemisphere verbally report?
    response_latency_ms: float = 0.0
    accuracy: float = 0.0               # 0.0-1.0: accuracy of response
    awareness_of_stimulus: bool = True    # Whether hemisphere is aware of input
    notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hemisphere": self.hemisphere.value,
            "response_content": self.response_content,
            "processing_domain": self.processing_domain.value,
            "confidence": self.confidence,
            "can_verbalize": self.can_verbalize,
            "response_latency_ms": self.response_latency_ms,
            "accuracy": self.accuracy,
            "awareness_of_stimulus": self.awareness_of_stimulus,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SplitBrainOutput:
    """
    Combined output from split-brain stimulus processing.

    Contains responses from both hemispheres, conflict detection,
    and any confabulation generated by the left hemisphere.
    """
    left_hemisphere: HemisphereResponse
    right_hemisphere: HemisphereResponse
    conflict_detected: bool
    conflict_type: Optional[ConflictType] = None
    confabulation: Optional[str] = None
    confabulation_type: Optional[ConfabulationType] = None
    integration_attempt: bool = False      # Whether cross-cueing was attempted
    integration_success: bool = False      # Whether integration succeeded
    dominant_response: str = ""           # The externally observable response
    notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_hemisphere": self.left_hemisphere.to_dict(),
            "right_hemisphere": self.right_hemisphere.to_dict(),
            "conflict_detected": self.conflict_detected,
            "conflict_type": self.conflict_type.value if self.conflict_type else None,
            "confabulation": self.confabulation,
            "confabulation_type": (
                self.confabulation_type.value if self.confabulation_type else None
            ),
            "integration_attempt": self.integration_attempt,
            "integration_success": self.integration_success,
            "dominant_response": self.dominant_response,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConflictAnalysis:
    """Detailed analysis of inter-hemispheric conflict."""
    conflict_present: bool
    conflict_type: Optional[ConflictType]
    severity: float                       # 0.0-1.0: conflict severity
    left_position: str                    # Left hemisphere's stance
    right_position: str                   # Right hemisphere's stance
    resolution: str = "unresolved"        # How/if conflict was resolved
    resolution_method: str = ""           # Method of resolution
    behavioral_manifestation: str = ""    # Observable behavior
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_present": self.conflict_present,
            "conflict_type": self.conflict_type.value if self.conflict_type else None,
            "severity": self.severity,
            "left_position": self.left_position,
            "right_position": self.right_position,
            "resolution": self.resolution,
            "resolution_method": self.resolution_method,
            "behavioral_manifestation": self.behavioral_manifestation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConfabulationOutput:
    """Output from confabulation modeling."""
    confabulation_generated: bool
    confabulation_content: str            # The confabulated explanation
    confabulation_type: ConfabulationType
    plausibility: float                   # 0.0-1.0: how plausible the confabulation is
    actual_cause: str                     # The real cause (right hemisphere action)
    awareness_of_confabulation: bool = False  # Whether subject knows they confabulated
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confabulation_generated": self.confabulation_generated,
            "confabulation_content": self.confabulation_content,
            "confabulation_type": self.confabulation_type.value,
            "plausibility": self.plausibility,
            "actual_cause": self.actual_cause,
            "awareness_of_confabulation": self.awareness_of_confabulation,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class SplitBrainInterface:
    """
    Main interface for Form 26: Split-Brain Consciousness.

    Models split-brain phenomena including lateralized stimulus processing,
    hemispheric specialization, inter-hemispheric conflict, confabulation,
    and the implications for understanding unified consciousness.
    """

    FORM_ID = "26-split-brain"
    FORM_NAME = "Split-Brain Consciousness"

    # Hemispheric specialization mapping
    DOMAIN_LATERALIZATION = {
        ProcessingDomain.VERBAL: Hemisphere.LEFT,
        ProcessingDomain.ANALYTICAL: Hemisphere.LEFT,
        ProcessingDomain.MOTOR_SPEECH: Hemisphere.LEFT,
        ProcessingDomain.SPATIAL: Hemisphere.RIGHT,
        ProcessingDomain.HOLISTIC: Hemisphere.RIGHT,
        ProcessingDomain.EMOTIONAL: Hemisphere.RIGHT,
    }

    def __init__(self):
        """Initialize the Split-Brain Consciousness Interface."""
        self._initialized = False
        self._trial_history: List[SplitBrainOutput] = []
        self._conflict_history: List[ConflictAnalysis] = []
        self._confabulation_history: List[ConfabulationOutput] = []
        self._interhemispheric_state: InterhemisphericState = (
            InterhemisphericState.DISCONNECTED
        )

    async def initialize(self) -> None:
        """Initialize the interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        self._interhemispheric_state = InterhemisphericState.DISCONNECTED

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def present_lateralized(
        self, stimulus_input: SplitBrainInput
    ) -> SplitBrainOutput:
        """
        Present a lateralized stimulus and model the split-brain response.

        When a stimulus is presented to one visual field, only the
        contralateral hemisphere directly processes it. The response
        depends on which hemisphere receives the information and which
        response modality is used.

        Args:
            stimulus_input: Lateralized stimulus parameters.

        Returns:
            SplitBrainOutput with responses from both hemispheres.
        """
        if not self._initialized:
            await self.initialize()

        # Determine which hemisphere receives the stimulus
        receiving_hemisphere = self._field_to_hemisphere(stimulus_input.visual_field)

        # Generate hemisphere responses
        left_response = self._generate_hemisphere_response(
            Hemisphere.LEFT, stimulus_input, receiving_hemisphere
        )
        right_response = self._generate_hemisphere_response(
            Hemisphere.RIGHT, stimulus_input, receiving_hemisphere
        )

        # Detect conflict
        conflict_detected, conflict_type = self._detect_hemispheric_conflict(
            left_response, right_response
        )

        # Check for confabulation
        confabulation = None
        confab_type = None
        if (receiving_hemisphere == Hemisphere.RIGHT and
                stimulus_input.response_modality == "verbal"):
            # Left hemisphere must verbally respond to something only
            # the right hemisphere saw - classic confabulation scenario
            confabulation = self._generate_confabulation(
                stimulus_input, left_response
            )
            confab_type = ConfabulationType.POST_HOC_RATIONALIZATION

        # Check for cross-cueing
        integration_attempt = (
            stimulus_input.interhemispheric_state == InterhemisphericState.CROSS_CUEING
        )
        integration_success = integration_attempt and conflict_type is None

        # Determine dominant observable response
        dominant = self._determine_dominant_response(
            left_response, right_response, stimulus_input.response_modality
        )

        output = SplitBrainOutput(
            left_hemisphere=left_response,
            right_hemisphere=right_response,
            conflict_detected=conflict_detected,
            conflict_type=conflict_type,
            confabulation=confabulation,
            confabulation_type=confab_type,
            integration_attempt=integration_attempt,
            integration_success=integration_success,
            dominant_response=dominant,
        )

        self._trial_history.append(output)
        return output

    async def assess_hemisphere_response(
        self,
        hemisphere: Hemisphere,
        domain: ProcessingDomain,
        stimulus_content: str,
    ) -> HemisphereResponse:
        """
        Assess a specific hemisphere's response capability for a given domain.

        Args:
            hemisphere: Which hemisphere to assess.
            domain: Processing domain to test.
            stimulus_content: The stimulus content.

        Returns:
            HemisphereResponse with the assessment.
        """
        if not self._initialized:
            await self.initialize()

        # Determine if this hemisphere is specialized for the domain
        specialized = self.DOMAIN_LATERALIZATION.get(domain) == hemisphere

        # Compute response quality
        if specialized:
            accuracy = 0.85
            confidence = 0.8
            latency = 250.0
        else:
            accuracy = 0.35
            confidence = 0.3
            latency = 500.0

        # Left hemisphere can verbalize
        can_verbalize = hemisphere == Hemisphere.LEFT

        notes = []
        if specialized:
            notes.append(f"Specialized for {domain.value} processing")
        else:
            notes.append(f"Non-dominant for {domain.value} processing")

        if not can_verbalize:
            notes.append("Cannot produce verbal report")

        return HemisphereResponse(
            hemisphere=hemisphere,
            response_content=f"{hemisphere.value}_response_to_{stimulus_content}",
            processing_domain=domain,
            confidence=confidence,
            can_verbalize=can_verbalize,
            response_latency_ms=latency,
            accuracy=accuracy,
            awareness_of_stimulus=True,
            notes=notes,
        )

    async def detect_conflict(
        self, bilateral_input: BilateralInput
    ) -> ConflictAnalysis:
        """
        Detect inter-hemispheric conflict from bilateral stimulation.

        When different stimuli are presented to each visual field,
        the hemispheres may generate conflicting responses.

        Args:
            bilateral_input: Different stimuli for each visual field.

        Returns:
            ConflictAnalysis with conflict assessment.
        """
        if not self._initialized:
            await self.initialize()

        # Each hemisphere processes its stimulus
        left_position = (
            f"Reports seeing '{bilateral_input.right_field_stimulus}'"
        )
        right_position = (
            f"Recognizes '{bilateral_input.left_field_stimulus}'"
        )

        # Conflict depends on response modality
        if bilateral_input.response_modality == "verbal":
            # Left hemisphere dominates verbal - reports only what it saw
            conflict_present = True
            conflict_type = ConflictType.PERCEPTUAL_CONFLICT
            severity = 0.7
            resolution = "left_hemisphere_verbal_dominance"
            behavioral = (
                f"Verbally reports '{bilateral_input.right_field_stimulus}' "
                f"but left hand may reach for '{bilateral_input.left_field_stimulus}'"
            )
        elif bilateral_input.response_modality in ("left_hand", "pointing"):
            # Left hand controlled by right hemisphere
            conflict_present = True
            conflict_type = ConflictType.MOTOR_CONFLICT
            severity = 0.8
            resolution = "right_hemisphere_motor_dominance"
            behavioral = (
                f"Left hand selects '{bilateral_input.left_field_stimulus}' "
                f"while patient verbally denies seeing it"
            )
        else:
            conflict_present = bilateral_input.left_field_stimulus != bilateral_input.right_field_stimulus
            conflict_type = ConflictType.DECISIONAL_CONFLICT if conflict_present else None
            severity = 0.5 if conflict_present else 0.0
            resolution = "ambiguous"
            behavioral = "Mixed behavioral response"

        analysis = ConflictAnalysis(
            conflict_present=conflict_present,
            conflict_type=conflict_type,
            severity=severity,
            left_position=left_position,
            right_position=right_position,
            resolution=resolution,
            resolution_method="hemispheric_dominance" if conflict_present else "agreement",
            behavioral_manifestation=behavioral,
        )

        self._conflict_history.append(analysis)
        return analysis

    async def model_confabulation(
        self,
        right_hemisphere_action: str,
        actual_cause: str,
    ) -> ConfabulationOutput:
        """
        Model the left hemisphere's confabulation in response to a
        right hemisphere-driven behavior.

        The left hemisphere (the "interpreter") generates plausible
        explanations for behaviors it did not initiate and cannot
        access the true cause of.

        Args:
            right_hemisphere_action: What the right hemisphere caused.
            actual_cause: The real reason for the behavior.

        Returns:
            ConfabulationOutput with the confabulated explanation.
        """
        if not self._initialized:
            await self.initialize()

        # Generate confabulated explanation
        confab_content = self._generate_confabulation_content(
            right_hemisphere_action
        )

        # Assess plausibility
        plausibility = self._assess_confabulation_plausibility(
            confab_content, right_hemisphere_action
        )

        output = ConfabulationOutput(
            confabulation_generated=True,
            confabulation_content=confab_content,
            confabulation_type=ConfabulationType.POST_HOC_RATIONALIZATION,
            plausibility=plausibility,
            actual_cause=actual_cause,
            awareness_of_confabulation=False,
        )

        self._confabulation_history.append(output)
        return output

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the interface state to a dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "interhemispheric_state": self._interhemispheric_state.value,
            "trial_count": len(self._trial_history),
            "conflict_count": len(self._conflict_history),
            "confabulation_count": len(self._confabulation_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the split-brain interface."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "interhemispheric_state": self._interhemispheric_state.value,
            "total_trials": len(self._trial_history),
            "conflicts_detected": sum(
                1 for c in self._conflict_history if c.conflict_present
            ),
            "confabulations_generated": len(self._confabulation_history),
        }

    # ========================================================================
    # PRIVATE COMPUTATION METHODS
    # ========================================================================

    def _field_to_hemisphere(self, field: LateralizedField) -> Optional[Hemisphere]:
        """Map visual field to receiving hemisphere."""
        if field == LateralizedField.LEFT_VISUAL_FIELD:
            return Hemisphere.RIGHT  # Contralateral
        elif field == LateralizedField.RIGHT_VISUAL_FIELD:
            return Hemisphere.LEFT   # Contralateral
        elif field == LateralizedField.BILATERAL:
            return None  # Both hemispheres
        else:
            return None  # Central - both

    def _generate_hemisphere_response(
        self,
        hemisphere: Hemisphere,
        stimulus: SplitBrainInput,
        receiving_hemisphere: Optional[Hemisphere],
    ) -> HemisphereResponse:
        """Generate a response for a specific hemisphere."""
        # Does this hemisphere directly receive the stimulus?
        receives_direct = (
            receiving_hemisphere is None or  # Bilateral/central
            receiving_hemisphere == hemisphere
        )

        # Is this hemisphere specialized for the processing domain?
        specialized = (
            self.DOMAIN_LATERALIZATION.get(stimulus.processing_domain) == hemisphere
        )

        # Compute processing quality
        if receives_direct and specialized:
            accuracy = 0.90
            confidence = 0.85
        elif receives_direct:
            accuracy = 0.65
            confidence = 0.6
        elif (not receives_direct and
              self._interhemispheric_state == InterhemisphericState.DISCONNECTED):
            accuracy = 0.0
            confidence = 0.0
        else:
            # Some cross-communication
            accuracy = 0.3
            confidence = 0.25

        can_verbalize = hemisphere == Hemisphere.LEFT
        awareness = receives_direct or self._interhemispheric_state != InterhemisphericState.DISCONNECTED

        notes = []
        if receives_direct:
            notes.append("Direct stimulus reception")
        else:
            notes.append("No direct stimulus access")

        if specialized:
            notes.append(f"Dominant for {stimulus.processing_domain.value}")

        if receives_direct:
            response_content = f"processed_{stimulus.stimulus_content}"
        else:
            response_content = "no_information" if not awareness else "indirect_information"

        return HemisphereResponse(
            hemisphere=hemisphere,
            response_content=response_content,
            processing_domain=stimulus.processing_domain,
            confidence=confidence,
            can_verbalize=can_verbalize,
            response_latency_ms=250.0 if receives_direct else 600.0,
            accuracy=accuracy,
            awareness_of_stimulus=awareness,
            notes=notes,
        )

    def _detect_hemispheric_conflict(
        self,
        left: HemisphereResponse,
        right: HemisphereResponse,
    ) -> Tuple[bool, Optional[ConflictType]]:
        """Detect conflict between hemisphere responses."""
        # No conflict if one hemisphere has no information
        if not left.awareness_of_stimulus or not right.awareness_of_stimulus:
            return False, None

        # Conflict if different responses to same stimulus
        if (left.response_content != right.response_content and
                left.confidence > 0.3 and right.confidence > 0.3):
            return True, ConflictType.PERCEPTUAL_CONFLICT

        return False, None

    def _generate_confabulation(
        self,
        stimulus: SplitBrainInput,
        left_response: HemisphereResponse,
    ) -> str:
        """Generate a left-hemisphere confabulation."""
        if left_response.awareness_of_stimulus:
            return ""  # No need to confabulate if aware

        # Classic confabulation: left hemisphere invents an explanation
        confabulations = {
            "image": f"I chose that because it seemed related to what I was thinking",
            "word": f"I think I saw something like that written somewhere",
            "object": f"I just had a feeling that was the right choice",
            "face": f"That person looked familiar to me",
            "emotion": f"I just felt like responding that way",
        }
        return confabulations.get(
            stimulus.stimulus_type,
            "I just had an intuition about the right answer"
        )

    def _generate_confabulation_content(self, action: str) -> str:
        """Generate confabulation content for a given action."""
        return (
            f"I performed '{action}' because it seemed like the right "
            f"thing to do at the time. I had a clear reason for it."
        )

    def _assess_confabulation_plausibility(
        self, confab: str, action: str
    ) -> float:
        """Assess how plausible a confabulation is."""
        # Confabulations are typically quite plausible
        base_plausibility = 0.7

        # Longer confabulations tend to be more elaborate and plausible
        if len(confab) > 50:
            base_plausibility += 0.1

        return min(1.0, base_plausibility)

    def _determine_dominant_response(
        self,
        left: HemisphereResponse,
        right: HemisphereResponse,
        modality: str,
    ) -> str:
        """Determine the externally dominant response."""
        if modality == "verbal":
            return left.response_content  # Left hemisphere controls speech
        elif modality in ("left_hand", "drawing"):
            return right.response_content  # Right hemisphere controls left hand
        elif modality == "right_hand":
            return left.response_content  # Left hemisphere controls right hand
        elif modality == "pointing":
            # Could be either hand
            return (
                right.response_content
                if right.confidence > left.confidence
                else left.response_content
            )
        else:
            # Default: verbal (left hemisphere)
            return left.response_content


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "Hemisphere",
    "ProcessingDomain",
    "InterhemisphericState",
    "ConflictType",
    "ConfabulationType",
    "LateralizedField",
    # Input dataclasses
    "SplitBrainInput",
    "BilateralInput",
    # Output dataclasses
    "HemisphereResponse",
    "SplitBrainOutput",
    "ConflictAnalysis",
    "ConfabulationOutput",
    # Interface
    "SplitBrainInterface",
    # Convenience
    "create_split_brain_interface",
]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_split_brain_interface() -> SplitBrainInterface:
    """
    Create and return a new SplitBrainInterface instance.

    Note: Call await interface.initialize() before use.

    Returns:
        A new SplitBrainInterface instance.
    """
    return SplitBrainInterface()
