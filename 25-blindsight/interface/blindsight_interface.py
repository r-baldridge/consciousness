#!/usr/bin/env python3
"""
Blindsight Consciousness Interface

Form 25: Models the dissociation between conscious visual experience and
unconscious visual processing known as blindsight. In blindsight, patients
with damage to primary visual cortex (V1) demonstrate above-chance
performance on forced-choice tasks for stimuli presented in their blind
field, despite reporting no conscious visual experience.

This form explores the distinction between ventral (conscious) and dorsal
(unconscious) visual processing pathways, the phenomenon of Type 1
(guessing without feeling) and Type 2 (vague feeling of something)
blindsight, and the implications for understanding the neural correlates
of conscious visual experience.
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

class BlindsightType(Enum):
    """
    Classification of blindsight subtypes.

    Type 1 involves above-chance performance with no subjective
    experience whatsoever. Type 2 involves a vague non-visual feeling
    or intuition that something is present.
    """
    TYPE_1_GUESSING = "type_1_guessing"    # Above-chance with no experience
    TYPE_2_FEELING = "type_2_feeling"      # Above-chance with vague feeling


class VisualFieldRegion(Enum):
    """
    Regions of the visual field for stimulus presentation.

    Blindsight occurs in the scotoma (blind region) resulting from
    V1 damage, while intact regions process normally.
    """
    INTACT_FOVEAL = "intact_foveal"          # Central intact vision
    INTACT_PERIPHERAL = "intact_peripheral"  # Peripheral intact vision
    BLIND_FIELD = "blind_field"              # Scotoma region (V1 damage)
    TRANSITION_ZONE = "transition_zone"      # Border of blind and intact


class ProcessingPathway(Enum):
    """
    Visual processing pathways in the brain.

    The ventral stream (V1 -> temporal cortex) supports conscious
    recognition. The dorsal stream can operate without V1, supporting
    unconscious spatial processing and action guidance.
    """
    VENTRAL_CONSCIOUS = "ventral_conscious"      # "What" pathway - conscious recognition
    DORSAL_UNCONSCIOUS = "dorsal_unconscious"    # "Where/How" pathway - action guidance
    SUBCORTICAL = "subcortical"                  # Superior colliculus pathway
    RESIDUAL_V1 = "residual_v1"                  # Spared V1 islands


class StimulusProperty(Enum):
    """Properties of visual stimuli that can be processed in blindsight."""
    MOTION = "motion"                    # Movement direction detection
    ORIENTATION = "orientation"          # Line/grating orientation
    COLOR = "color"                      # Chromatic properties (limited)
    SPATIAL_FREQUENCY = "spatial_frequency"  # Fine vs coarse detail
    EMOTION = "emotion"                  # Facial emotion (subcortical)
    LUMINANCE = "luminance"              # Brightness changes
    SHAPE = "shape"                      # Basic shape detection


class DetectionConfidence(Enum):
    """Patient's subjective confidence in their forced-choice response."""
    PURE_GUESS = "pure_guess"            # No feeling of knowing
    SLIGHT_HUNCH = "slight_hunch"        # Minimal intuition
    MODERATE_FEELING = "moderate_feeling"  # Some non-visual awareness
    STRONG_FEELING = "strong_feeling"    # Clear non-visual intuition


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class BlindsightInput:
    """
    Input representing a visual stimulus presented in the blind field.

    Encapsulates the stimulus properties, presentation location, and
    task parameters for blindsight assessment.
    """
    stimulus_region: VisualFieldRegion
    stimulus_property: StimulusProperty
    stimulus_intensity: float             # 0.0-1.0: stimulus strength
    stimulus_duration_ms: float           # Duration in milliseconds
    stimulus_size_degrees: float = 2.0    # Size in visual degrees
    contrast: float = 0.8               # 0.0-1.0: Michelson contrast
    eccentricity_degrees: float = 10.0   # Distance from fixation
    background_luminance: float = 0.5    # Background brightness
    task_type: str = "forced_choice"     # forced_choice, detection, discrimination
    response_options: int = 2            # Number of choices (2AFC, 4AFC, etc.)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stimulus_region": self.stimulus_region.value,
            "stimulus_property": self.stimulus_property.value,
            "stimulus_intensity": self.stimulus_intensity,
            "stimulus_duration_ms": self.stimulus_duration_ms,
            "stimulus_size_degrees": self.stimulus_size_degrees,
            "contrast": self.contrast,
            "eccentricity_degrees": self.eccentricity_degrees,
            "background_luminance": self.background_luminance,
            "task_type": self.task_type,
            "response_options": self.response_options,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ForcedChoiceTrialInput:
    """Input for a single forced-choice trial in blindsight testing."""
    stimulus: BlindsightInput
    correct_response: str                 # The correct answer
    trial_number: int = 0
    block_number: int = 0
    inter_trial_interval_ms: float = 1000.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stimulus": self.stimulus.to_dict(),
            "correct_response": self.correct_response,
            "trial_number": self.trial_number,
            "block_number": self.block_number,
            "inter_trial_interval_ms": self.inter_trial_interval_ms,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class BlindsightOutput:
    """
    Output from blindsight stimulus processing.

    Contains both the implicit detection result (above/below chance)
    and the pathway analysis indicating which neural routes processed
    the stimulus.
    """
    implicit_detection: bool              # Whether stimulus was implicitly detected
    forced_choice_accuracy: float         # 0.0-1.0: accuracy over trials
    above_chance: bool                    # Statistically above chance
    pathway_analysis: Dict[str, float] = field(default_factory=dict)
    blindsight_type: Optional[BlindsightType] = None
    subjective_confidence: DetectionConfidence = DetectionConfidence.PURE_GUESS
    response_time_ms: float = 0.0        # Response latency
    d_prime: float = 0.0                 # Signal detection sensitivity
    criterion: float = 0.0              # Response bias
    conscious_report: str = "nothing"    # What patient consciously reports
    confidence: float = 0.5              # Assessment confidence
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "implicit_detection": self.implicit_detection,
            "forced_choice_accuracy": self.forced_choice_accuracy,
            "above_chance": self.above_chance,
            "pathway_analysis": self.pathway_analysis,
            "blindsight_type": self.blindsight_type.value if self.blindsight_type else None,
            "subjective_confidence": self.subjective_confidence.value,
            "response_time_ms": self.response_time_ms,
            "d_prime": self.d_prime,
            "criterion": self.criterion,
            "conscious_report": self.conscious_report,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PathwayAnalysis:
    """Detailed analysis of visual processing pathway activation."""
    pathway: ProcessingPathway
    activation_level: float               # 0.0-1.0: pathway activation
    information_transmitted: List[str] = field(default_factory=list)
    latency_ms: float = 0.0             # Processing latency
    reaches_consciousness: bool = False   # Whether this pathway supports awareness
    notes: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pathway": self.pathway.value,
            "activation_level": self.activation_level,
            "information_transmitted": self.information_transmitted,
            "latency_ms": self.latency_ms,
            "reaches_consciousness": self.reaches_consciousness,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ImplicitDetectionResult:
    """Result from implicit detection assessment."""
    detected: bool
    detection_method: str                 # How detection was determined
    stimulus_property_detected: StimulusProperty
    accuracy: float                      # 0.0-1.0: forced-choice accuracy
    n_trials: int = 0
    chance_level: float = 0.5            # Baseline chance performance
    p_value: float = 1.0                 # Statistical significance
    effect_size: float = 0.0             # Cohen's d or similar
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "detection_method": self.detection_method,
            "stimulus_property_detected": self.stimulus_property_detected.value,
            "accuracy": self.accuracy,
            "n_trials": self.n_trials,
            "chance_level": self.chance_level,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BlindsightProfile:
    """Complete blindsight profile for a patient/subject."""
    blindsight_type: BlindsightType
    affected_field: VisualFieldRegion
    preserved_properties: List[StimulusProperty] = field(default_factory=list)
    active_pathways: List[ProcessingPathway] = field(default_factory=list)
    overall_accuracy: float = 0.0
    property_accuracies: Dict[str, float] = field(default_factory=dict)
    total_trials: int = 0
    assessment_sessions: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blindsight_type": self.blindsight_type.value,
            "affected_field": self.affected_field.value,
            "preserved_properties": [p.value for p in self.preserved_properties],
            "active_pathways": [p.value for p in self.active_pathways],
            "overall_accuracy": self.overall_accuracy,
            "property_accuracies": self.property_accuracies,
            "total_trials": self.total_trials,
            "assessment_sessions": self.assessment_sessions,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class BlindsightInterface:
    """
    Main interface for Form 25: Blindsight Consciousness.

    Models the dissociation between conscious and unconscious visual
    processing, providing methods for testing blind-field processing,
    forced-choice paradigms, implicit detection assessment, and
    visual pathway analysis.
    """

    FORM_ID = "25-blindsight"
    FORM_NAME = "Blindsight Consciousness"

    def __init__(self):
        """Initialize the Blindsight Consciousness Interface."""
        self._initialized = False
        self._trial_history: List[Dict[str, Any]] = []
        self._pathway_analyses: List[PathwayAnalysis] = []
        self._profile: Optional[BlindsightProfile] = None
        self._property_results: Dict[str, List[float]] = {}
        self._total_correct: int = 0
        self._total_trials: int = 0

    async def initialize(self) -> None:
        """Initialize the interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        self._property_results = {prop.value: [] for prop in StimulusProperty}
        self._total_correct = 0
        self._total_trials = 0

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def process_blind_field(
        self, stimulus_input: BlindsightInput
    ) -> BlindsightOutput:
        """
        Process a stimulus presented in the blind field.

        Simulates the blindsight phenomenon by modeling both conscious
        and unconscious processing pathways and determining whether
        implicit detection occurs.

        Args:
            stimulus_input: The visual stimulus parameters.

        Returns:
            BlindsightOutput with detection results and pathway analysis.
        """
        if not self._initialized:
            await self.initialize()

        # Determine if stimulus is in blind field
        in_blind_field = stimulus_input.stimulus_region in (
            VisualFieldRegion.BLIND_FIELD,
            VisualFieldRegion.TRANSITION_ZONE,
        )

        # Compute pathway activation
        pathway_activation = self._compute_pathway_activation(
            stimulus_input, in_blind_field
        )

        # Determine implicit detection
        implicit = self._compute_implicit_detection(
            stimulus_input, pathway_activation
        )

        # Compute forced-choice accuracy
        accuracy = self._compute_accuracy(stimulus_input, implicit)

        # Determine blindsight type
        bs_type = self._classify_blindsight_type(
            implicit, stimulus_input
        )

        # Subjective confidence
        confidence_level = self._assess_subjective_confidence(
            bs_type, implicit
        )

        # Conscious report
        if in_blind_field:
            conscious_report = "nothing" if bs_type == BlindsightType.TYPE_1_GUESSING else "vague_feeling"
        else:
            conscious_report = "clear_perception"

        # d-prime (signal detection measure)
        d_prime = self._compute_d_prime(accuracy)

        output = BlindsightOutput(
            implicit_detection=implicit,
            forced_choice_accuracy=accuracy,
            above_chance=accuracy > (1.0 / stimulus_input.response_options) + 0.1,
            pathway_analysis=pathway_activation,
            blindsight_type=bs_type if in_blind_field else None,
            subjective_confidence=confidence_level,
            response_time_ms=self._estimate_response_time(stimulus_input, in_blind_field),
            d_prime=d_prime,
            conscious_report=conscious_report,
            confidence=min(1.0, accuracy + 0.1),
        )

        # Update history
        self._trial_history.append({
            "input": stimulus_input.to_dict(),
            "output": output.to_dict(),
        })

        # Update property results
        prop_key = stimulus_input.stimulus_property.value
        if prop_key in self._property_results:
            self._property_results[prop_key].append(accuracy)

        return output

    async def forced_choice_test(
        self, trial_input: ForcedChoiceTrialInput
    ) -> Dict[str, Any]:
        """
        Run a single forced-choice trial.

        The patient is forced to choose between alternatives despite
        reporting no conscious experience of the stimulus.

        Args:
            trial_input: Trial parameters including correct response.

        Returns:
            Dictionary with trial results.
        """
        if not self._initialized:
            await self.initialize()

        # Process the stimulus
        output = await self.process_blind_field(trial_input.stimulus)

        # Determine if response would be correct
        correct = output.forced_choice_accuracy > 0.5

        self._total_trials += 1
        if correct:
            self._total_correct += 1

        running_accuracy = (
            self._total_correct / self._total_trials
            if self._total_trials > 0 else 0.0
        )

        return {
            "trial_number": trial_input.trial_number,
            "correct": correct,
            "response_time_ms": output.response_time_ms,
            "subjective_confidence": output.subjective_confidence.value,
            "conscious_report": output.conscious_report,
            "running_accuracy": running_accuracy,
            "total_trials": self._total_trials,
            "blindsight_type": output.blindsight_type.value if output.blindsight_type else None,
        }

    async def assess_implicit_detection(
        self, stimulus_property: StimulusProperty
    ) -> ImplicitDetectionResult:
        """
        Assess whether implicit detection exists for a given stimulus
        property based on accumulated trial data.

        Args:
            stimulus_property: The property to assess.

        Returns:
            ImplicitDetectionResult with statistical analysis.
        """
        if not self._initialized:
            await self.initialize()

        prop_key = stimulus_property.value
        results = self._property_results.get(prop_key, [])

        if len(results) == 0:
            return ImplicitDetectionResult(
                detected=False,
                detection_method="forced_choice",
                stimulus_property_detected=stimulus_property,
                accuracy=0.0,
                n_trials=0,
                p_value=1.0,
            )

        accuracy = sum(results) / len(results)
        chance = 0.5  # 2AFC default
        above_chance = accuracy > chance + 0.1
        n_trials = len(results)

        # Simplified p-value estimation
        p_value = max(0.001, 1.0 - (accuracy - chance) * n_trials * 0.1)

        # Effect size (simplified Cohen's d)
        effect_size = (accuracy - chance) / 0.25 if accuracy > chance else 0.0

        return ImplicitDetectionResult(
            detected=above_chance and p_value < 0.05,
            detection_method="forced_choice",
            stimulus_property_detected=stimulus_property,
            accuracy=accuracy,
            n_trials=n_trials,
            chance_level=chance,
            p_value=min(1.0, p_value),
            effect_size=effect_size,
        )

    async def analyze_pathway(
        self, stimulus_input: BlindsightInput
    ) -> List[PathwayAnalysis]:
        """
        Analyze which visual processing pathways are activated by a stimulus.

        Args:
            stimulus_input: The visual stimulus parameters.

        Returns:
            List of PathwayAnalysis for each relevant pathway.
        """
        if not self._initialized:
            await self.initialize()

        in_blind_field = stimulus_input.stimulus_region in (
            VisualFieldRegion.BLIND_FIELD,
            VisualFieldRegion.TRANSITION_ZONE,
        )

        analyses = []

        # Ventral pathway (conscious)
        ventral_activation = 0.0 if in_blind_field else stimulus_input.stimulus_intensity
        analyses.append(PathwayAnalysis(
            pathway=ProcessingPathway.VENTRAL_CONSCIOUS,
            activation_level=ventral_activation,
            information_transmitted=["identity", "color", "form"] if ventral_activation > 0.3 else [],
            latency_ms=150.0 if ventral_activation > 0 else 0.0,
            reaches_consciousness=ventral_activation > 0.3,
            notes=["Primary conscious pathway"] if not in_blind_field else ["Disrupted by V1 lesion"],
        ))

        # Dorsal pathway (unconscious action guidance)
        dorsal_activation = stimulus_input.stimulus_intensity * 0.7 if in_blind_field else stimulus_input.stimulus_intensity * 0.9
        dorsal_info = []
        if stimulus_input.stimulus_property == StimulusProperty.MOTION:
            dorsal_info.append("motion_direction")
            dorsal_activation *= 1.2
        if stimulus_input.stimulus_property == StimulusProperty.ORIENTATION:
            dorsal_info.append("orientation")
        dorsal_info.append("spatial_location")
        analyses.append(PathwayAnalysis(
            pathway=ProcessingPathway.DORSAL_UNCONSCIOUS,
            activation_level=min(1.0, dorsal_activation),
            information_transmitted=dorsal_info,
            latency_ms=80.0,
            reaches_consciousness=False,
            notes=["Processes spatial and motion information without V1"],
        ))

        # Subcortical pathway (superior colliculus)
        subcortical_activation = stimulus_input.stimulus_intensity * 0.5
        subcortical_info = ["basic_detection", "spatial_location"]
        if stimulus_input.stimulus_property == StimulusProperty.EMOTION:
            subcortical_info.append("facial_emotion")
            subcortical_activation *= 1.3
        analyses.append(PathwayAnalysis(
            pathway=ProcessingPathway.SUBCORTICAL,
            activation_level=min(1.0, subcortical_activation),
            information_transmitted=subcortical_info,
            latency_ms=60.0,
            reaches_consciousness=False,
            notes=["Fast subcortical route via superior colliculus"],
        ))

        self._pathway_analyses.extend(analyses)
        return analyses

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the interface state to a dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "total_trials": self._total_trials,
            "total_correct": self._total_correct,
            "overall_accuracy": (
                self._total_correct / self._total_trials
                if self._total_trials > 0 else 0.0
            ),
            "trial_history_length": len(self._trial_history),
            "pathway_analyses_count": len(self._pathway_analyses),
            "profile": self._profile.to_dict() if self._profile else None,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the blindsight interface."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "total_trials": self._total_trials,
            "accuracy": (
                self._total_correct / self._total_trials
                if self._total_trials > 0 else 0.0
            ),
            "properties_tested": sum(
                1 for v in self._property_results.values() if len(v) > 0
            ),
        }

    # ========================================================================
    # PRIVATE COMPUTATION METHODS
    # ========================================================================

    def _compute_pathway_activation(
        self,
        stimulus: BlindsightInput,
        in_blind_field: bool,
    ) -> Dict[str, float]:
        """Compute activation levels for each processing pathway."""
        activation = {}

        if in_blind_field:
            # V1 damaged: ventral is suppressed, dorsal and subcortical active
            activation[ProcessingPathway.VENTRAL_CONSCIOUS.value] = 0.0
            activation[ProcessingPathway.DORSAL_UNCONSCIOUS.value] = (
                stimulus.stimulus_intensity * 0.7
            )
            activation[ProcessingPathway.SUBCORTICAL.value] = (
                stimulus.stimulus_intensity * 0.5
            )
            activation[ProcessingPathway.RESIDUAL_V1.value] = (
                stimulus.stimulus_intensity * 0.1
            )
        else:
            # Intact field: all pathways active
            activation[ProcessingPathway.VENTRAL_CONSCIOUS.value] = (
                stimulus.stimulus_intensity * 0.9
            )
            activation[ProcessingPathway.DORSAL_UNCONSCIOUS.value] = (
                stimulus.stimulus_intensity * 0.8
            )
            activation[ProcessingPathway.SUBCORTICAL.value] = (
                stimulus.stimulus_intensity * 0.4
            )
            activation[ProcessingPathway.RESIDUAL_V1.value] = 0.0

        # Property-specific modulation
        if stimulus.stimulus_property == StimulusProperty.MOTION:
            activation[ProcessingPathway.DORSAL_UNCONSCIOUS.value] *= 1.3
        elif stimulus.stimulus_property == StimulusProperty.EMOTION:
            activation[ProcessingPathway.SUBCORTICAL.value] *= 1.4

        # Clamp values
        for key in activation:
            activation[key] = max(0.0, min(1.0, activation[key]))

        return activation

    def _compute_implicit_detection(
        self,
        stimulus: BlindsightInput,
        pathway_activation: Dict[str, float],
    ) -> bool:
        """Determine whether implicit detection occurs."""
        # Implicit detection depends on dorsal and subcortical pathways
        dorsal = pathway_activation.get(
            ProcessingPathway.DORSAL_UNCONSCIOUS.value, 0.0
        )
        subcortical = pathway_activation.get(
            ProcessingPathway.SUBCORTICAL.value, 0.0
        )

        combined = dorsal * 0.6 + subcortical * 0.4

        # Higher intensity and contrast improve detection
        combined *= (1.0 + stimulus.contrast * 0.3)

        # Duration affects detection
        if stimulus.stimulus_duration_ms > 100:
            combined *= 1.1

        return combined > 0.25

    def _compute_accuracy(
        self, stimulus: BlindsightInput, implicit_detection: bool
    ) -> float:
        """Compute forced-choice accuracy."""
        chance = 1.0 / stimulus.response_options

        if not implicit_detection:
            return chance

        # Above-chance performance
        boost = stimulus.stimulus_intensity * 0.3 + stimulus.contrast * 0.2
        accuracy = chance + boost

        # Property-specific accuracy
        property_bonuses = {
            StimulusProperty.MOTION: 0.15,
            StimulusProperty.LUMINANCE: 0.10,
            StimulusProperty.ORIENTATION: 0.08,
            StimulusProperty.EMOTION: 0.12,
            StimulusProperty.SPATIAL_FREQUENCY: 0.05,
            StimulusProperty.COLOR: 0.03,
            StimulusProperty.SHAPE: 0.06,
        }
        accuracy += property_bonuses.get(stimulus.stimulus_property, 0.05)

        return min(1.0, max(chance, accuracy))

    def _classify_blindsight_type(
        self, implicit_detection: bool, stimulus: BlindsightInput
    ) -> BlindsightType:
        """Classify the type of blindsight."""
        if not implicit_detection:
            return BlindsightType.TYPE_1_GUESSING

        # Type 2 is more likely with certain stimulus properties
        type_2_properties = {
            StimulusProperty.EMOTION,
            StimulusProperty.MOTION,
            StimulusProperty.LUMINANCE,
        }

        if (stimulus.stimulus_property in type_2_properties and
                stimulus.stimulus_intensity > 0.6):
            return BlindsightType.TYPE_2_FEELING

        return BlindsightType.TYPE_1_GUESSING

    def _assess_subjective_confidence(
        self,
        bs_type: Optional[BlindsightType],
        implicit: bool,
    ) -> DetectionConfidence:
        """Assess the patient's subjective confidence level."""
        if not implicit:
            return DetectionConfidence.PURE_GUESS

        if bs_type == BlindsightType.TYPE_2_FEELING:
            return DetectionConfidence.MODERATE_FEELING

        return DetectionConfidence.PURE_GUESS

    def _estimate_response_time(
        self, stimulus: BlindsightInput, in_blind_field: bool
    ) -> float:
        """Estimate response time in milliseconds."""
        if in_blind_field:
            # Longer response times in blind field
            base_rt = 600.0
            intensity_reduction = stimulus.stimulus_intensity * 100
            return base_rt - intensity_reduction
        else:
            return 350.0 - stimulus.stimulus_intensity * 50

    def _compute_d_prime(self, accuracy: float) -> float:
        """Compute d-prime (sensitivity) from accuracy."""
        if accuracy <= 0.0 or accuracy >= 1.0:
            return 0.0 if accuracy <= 0.5 else 3.0

        # Simplified d-prime: transform accuracy to z-score difference
        # Using linear approximation for simplicity
        d_prime = (accuracy - 0.5) * 6.0
        return max(0.0, min(4.0, d_prime))


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "BlindsightType",
    "VisualFieldRegion",
    "ProcessingPathway",
    "StimulusProperty",
    "DetectionConfidence",
    # Input dataclasses
    "BlindsightInput",
    "ForcedChoiceTrialInput",
    # Output dataclasses
    "BlindsightOutput",
    "PathwayAnalysis",
    "ImplicitDetectionResult",
    "BlindsightProfile",
    # Interface
    "BlindsightInterface",
    # Convenience
    "create_blindsight_interface",
]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_blindsight_interface() -> BlindsightInterface:
    """
    Create and return a new BlindsightInterface instance.

    Note: Call await interface.initialize() before use.

    Returns:
        A new BlindsightInterface instance.
    """
    return BlindsightInterface()
