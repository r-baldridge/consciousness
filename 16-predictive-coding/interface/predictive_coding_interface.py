#!/usr/bin/env python3
"""
Predictive Coding Consciousness Interface

Form 16: Implements Predictive Processing / Free Energy Principle as
proposed by Karl Friston and Andy Clark. The brain is fundamentally a
prediction machine that constructs top-down generative models of the world
and compares them with bottom-up sensory evidence. Prediction errors -
discrepancies between predictions and reality - drive learning and
perception. Consciousness emerges from the hierarchical minimization of
prediction error (free energy).

This module manages hierarchical predictions, computes prediction errors,
updates generative models, and minimizes free energy across processing
levels.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class PredictionLevel(Enum):
    """
    Hierarchical levels of predictive processing.

    Lower levels handle concrete sensory details; higher levels
    handle abstract concepts and models.
    """
    SENSORY = "sensory"                  # Lowest: raw sensory predictions
    PERCEPTUAL = "perceptual"            # Object/feature-level predictions
    CONCEPTUAL = "conceptual"            # Abstract concept-level predictions
    CONTEXTUAL = "contextual"            # Context and scene-level predictions
    NARRATIVE = "narrative"              # High-level narrative predictions


class ErrorType(Enum):
    """Types of prediction errors."""
    SENSORY_MISMATCH = "sensory_mismatch"        # Raw sensory mismatch
    FEATURE_ERROR = "feature_error"              # Feature detection error
    OBJECT_ERROR = "object_error"                # Object recognition error
    CONTEXT_ERROR = "context_error"              # Contextual expectation violation
    MODEL_VIOLATION = "model_violation"          # Deep model violation
    SURPRISE = "surprise"                        # Bayesian surprise


class PrecisionLevel(Enum):
    """Precision (inverse variance) of predictions and errors."""
    VERY_LOW = "very_low"                # Very uncertain
    LOW = "low"                          # Uncertain
    MODERATE = "moderate"                # Moderately certain
    HIGH = "high"                        # Quite certain
    VERY_HIGH = "very_high"              # Very certain


class UpdateStrategy(Enum):
    """Strategies for updating the generative model."""
    PERCEPTUAL_INFERENCE = "perceptual_inference"  # Update percept to match model
    MODEL_UPDATE = "model_update"                  # Update model to match evidence
    ACTIVE_INFERENCE = "active_inference"           # Act to make world match model
    ATTENTION_SHIFT = "attention_shift"            # Change precision weighting


class FreeEnergyComponent(Enum):
    """Components of the free energy functional."""
    PREDICTION_ERROR = "prediction_error"     # Accuracy term
    COMPLEXITY = "complexity"                  # KL divergence / complexity
    ENTROPY = "entropy"                        # Uncertainty / entropy
    EXPECTED_FREE_ENERGY = "expected_free_energy"  # Future-oriented


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class Prediction:
    """A top-down prediction from the generative model."""
    prediction_id: str
    level: PredictionLevel
    content: Dict[str, Any]             # What is predicted
    confidence: float                   # 0.0-1.0: Model confidence
    precision: float                    # 0.0-1.0: Inverse variance
    source_model: str = "default"       # Which generative model
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "level": self.level.value,
            "content": self.content,
            "confidence": round(self.confidence, 4),
            "precision": round(self.precision, 4),
            "source_model": self.source_model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SensoryEvidence:
    """Bottom-up sensory evidence (observation)."""
    evidence_id: str
    level: PredictionLevel
    content: Dict[str, Any]             # What is observed
    reliability: float                  # 0.0-1.0: Signal reliability
    precision: float                    # 0.0-1.0: Sensory precision
    modality: str = "general"           # Sensory modality
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "level": self.level.value,
            "content": self.content,
            "reliability": round(self.reliability, 4),
            "precision": round(self.precision, 4),
            "modality": self.modality,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictiveCodingInput:
    """Complete input for predictive coding processing."""
    predictions: List[Prediction]
    evidence: List[SensoryEvidence]
    current_context: Dict[str, Any] = field(default_factory=dict)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class PredictionError:
    """A prediction error (discrepancy between prediction and evidence)."""
    error_id: str
    prediction_id: str
    evidence_id: str
    level: PredictionLevel
    error_type: ErrorType
    magnitude: float                    # 0.0-1.0: Size of the error
    precision_weighted_error: float     # Error weighted by precision
    direction: Dict[str, float]         # Direction of the mismatch
    surprise: float                     # Bayesian surprise (-log p)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "prediction_id": self.prediction_id,
            "evidence_id": self.evidence_id,
            "level": self.level.value,
            "error_type": self.error_type.value,
            "magnitude": round(self.magnitude, 6),
            "precision_weighted_error": round(self.precision_weighted_error, 6),
            "surprise": round(self.surprise, 6),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ModelUpdate:
    """An update to the generative model."""
    update_id: str
    strategy: UpdateStrategy
    level: PredictionLevel
    learning_rate: float                # How much the model changed
    error_reduction: float              # How much error was reduced
    new_model_confidence: float         # Updated model confidence
    description: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "strategy": self.strategy.value,
            "level": self.level.value,
            "learning_rate": round(self.learning_rate, 6),
            "error_reduction": round(self.error_reduction, 6),
            "new_model_confidence": round(self.new_model_confidence, 4),
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FreeEnergyState:
    """Current state of the free energy functional."""
    total_free_energy: float            # Total free energy
    prediction_error_component: float   # Accuracy term
    complexity_component: float         # Complexity penalty
    entropy_component: float            # Uncertainty
    level_breakdown: Dict[str, float]   # Free energy by level
    is_minimized: bool                  # Whether at local minimum
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_free_energy": round(self.total_free_energy, 6),
            "prediction_error_component": round(self.prediction_error_component, 6),
            "complexity_component": round(self.complexity_component, 6),
            "entropy_component": round(self.entropy_component, 6),
            "level_breakdown": {k: round(v, 6) for k, v in self.level_breakdown.items()},
            "is_minimized": self.is_minimized,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictiveCodingOutput:
    """Complete output from predictive coding processing."""
    prediction_errors: List[PredictionError]
    model_updates: List[ModelUpdate]
    free_energy_state: FreeEnergyState
    total_surprise: float
    dominant_error_level: PredictionLevel
    update_strategy_used: UpdateStrategy
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_errors": len(self.prediction_errors),
            "num_updates": len(self.model_updates),
            "free_energy": self.free_energy_state.to_dict(),
            "total_surprise": round(self.total_surprise, 6),
            "dominant_error_level": self.dominant_error_level.value,
            "update_strategy": self.update_strategy_used.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PredictiveCodingSystemStatus:
    """Status of the predictive coding system."""
    is_initialized: bool
    total_predictions: int
    total_errors_computed: int
    total_model_updates: int
    current_free_energy: float
    average_surprise: float
    system_health: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# PREDICTION ERROR ENGINE
# ============================================================================

class PredictionErrorEngine:
    """
    Engine for computing prediction errors across the hierarchy.

    Compares top-down predictions with bottom-up evidence at each
    processing level to compute precision-weighted prediction errors.
    """

    ERROR_COUNTER = 0

    def __init__(self):
        self._error_history: List[PredictionError] = []
        self._max_history = 200

    def compute_prediction_error(
        self,
        prediction: Prediction,
        evidence: SensoryEvidence,
        attention_weight: float = 1.0
    ) -> PredictionError:
        """
        Compute the prediction error between a prediction and evidence.

        The error is precision-weighted, meaning more precise (certain)
        predictions and evidence produce stronger error signals.
        """
        PredictionErrorEngine.ERROR_COUNTER += 1
        error_id = f"pe_{PredictionErrorEngine.ERROR_COUNTER}"

        # Compute raw magnitude
        magnitude = self._compute_magnitude(prediction.content, evidence.content)

        # Compute precision weighting
        combined_precision = (prediction.precision + evidence.precision) / 2.0
        precision_weighted = magnitude * combined_precision * attention_weight

        # Compute Bayesian surprise: -log(p(evidence|model))
        # Higher magnitude = higher surprise
        surprise = -math.log(max(0.001, 1.0 - magnitude * 0.9))

        # Determine error type
        error_type = self._classify_error(prediction.level, magnitude)

        # Compute direction (simplified as signed difference)
        direction = self._compute_direction(prediction.content, evidence.content)

        error = PredictionError(
            error_id=error_id,
            prediction_id=prediction.prediction_id,
            evidence_id=evidence.evidence_id,
            level=prediction.level,
            error_type=error_type,
            magnitude=magnitude,
            precision_weighted_error=precision_weighted,
            direction=direction,
            surprise=surprise,
        )

        self._error_history.append(error)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        return error

    def _compute_magnitude(
        self, predicted: Dict[str, Any], observed: Dict[str, Any]
    ) -> float:
        """Compute magnitude of prediction error."""
        if not predicted or not observed:
            return 0.5

        # Compare common keys
        common_keys = set(predicted.keys()) & set(observed.keys())
        if not common_keys:
            return 0.5

        total_error = 0.0
        for key in common_keys:
            pred_val = predicted[key]
            obs_val = observed[key]

            if isinstance(pred_val, (int, float)) and isinstance(obs_val, (int, float)):
                total_error += abs(pred_val - obs_val)
            elif pred_val != obs_val:
                total_error += 1.0

        avg_error = total_error / len(common_keys)
        return max(0.0, min(1.0, avg_error))

    def _classify_error(self, level: PredictionLevel, magnitude: float) -> ErrorType:
        """Classify the type of prediction error."""
        if magnitude > 0.8:
            return ErrorType.MODEL_VIOLATION
        if magnitude > 0.6:
            return ErrorType.SURPRISE

        level_to_error = {
            PredictionLevel.SENSORY: ErrorType.SENSORY_MISMATCH,
            PredictionLevel.PERCEPTUAL: ErrorType.FEATURE_ERROR,
            PredictionLevel.CONCEPTUAL: ErrorType.OBJECT_ERROR,
            PredictionLevel.CONTEXTUAL: ErrorType.CONTEXT_ERROR,
            PredictionLevel.NARRATIVE: ErrorType.MODEL_VIOLATION,
        }
        return level_to_error.get(level, ErrorType.SENSORY_MISMATCH)

    def _compute_direction(
        self, predicted: Dict[str, Any], observed: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute direction of prediction error."""
        direction = {}
        common_keys = set(predicted.keys()) & set(observed.keys())
        for key in common_keys:
            pred_val = predicted[key]
            obs_val = observed[key]
            if isinstance(pred_val, (int, float)) and isinstance(obs_val, (int, float)):
                direction[key] = obs_val - pred_val
        return direction

    def classify_precision(self, precision: float) -> PrecisionLevel:
        """Classify a precision value into a level."""
        if precision < 0.2:
            return PrecisionLevel.VERY_LOW
        elif precision < 0.4:
            return PrecisionLevel.LOW
        elif precision < 0.6:
            return PrecisionLevel.MODERATE
        elif precision < 0.8:
            return PrecisionLevel.HIGH
        else:
            return PrecisionLevel.VERY_HIGH


# ============================================================================
# GENERATIVE MODEL ENGINE
# ============================================================================

class GenerativeModelEngine:
    """
    Engine for managing and updating the generative model.

    The generative model produces top-down predictions and is updated
    based on prediction errors through various strategies.
    """

    UPDATE_COUNTER = 0

    def __init__(self):
        self._model_parameters: Dict[str, Dict[str, Any]] = {}
        self._model_confidence: Dict[str, float] = {}
        self._update_history: List[ModelUpdate] = []
        self._max_history = 100

        # Initialize default model parameters per level
        for level in PredictionLevel:
            self._model_parameters[level.value] = {
                "learning_rate": 0.1,
                "prior_strength": 0.5,
            }
            self._model_confidence[level.value] = 0.5

    def generate_prediction(
        self,
        level: PredictionLevel,
        context: Dict[str, Any],
        prediction_id: str = ""
    ) -> Prediction:
        """Generate a top-down prediction based on the current model."""
        if not prediction_id:
            prediction_id = f"pred_{level.value}_{len(self._update_history)}"

        model_conf = self._model_confidence.get(level.value, 0.5)
        params = self._model_parameters.get(level.value, {})

        # Generate predicted content based on context and model
        predicted_content = {}
        for key, value in context.items():
            if isinstance(value, (int, float)):
                # Model predicts with some noise/bias
                predicted_content[key] = value * (0.8 + model_conf * 0.2)
            else:
                predicted_content[key] = value

        precision = model_conf * 0.8 + 0.2  # Higher confidence = higher precision

        return Prediction(
            prediction_id=prediction_id,
            level=level,
            content=predicted_content,
            confidence=model_conf,
            precision=precision,
            source_model=f"gen_model_{level.value}",
            context=context,
        )

    def update_model(
        self,
        errors: List[PredictionError],
        level: PredictionLevel,
    ) -> ModelUpdate:
        """Update the generative model based on prediction errors."""
        GenerativeModelEngine.UPDATE_COUNTER += 1
        update_id = f"update_{GenerativeModelEngine.UPDATE_COUNTER}"

        params = self._model_parameters.get(level.value, {})
        learning_rate = params.get("learning_rate", 0.1)

        # Compute average error for this level
        level_errors = [e for e in errors if e.level == level]
        avg_magnitude = sum(e.magnitude for e in level_errors) / max(1, len(level_errors))
        avg_weighted = sum(e.precision_weighted_error for e in level_errors) / max(1, len(level_errors))

        # Determine update strategy
        strategy = self._choose_strategy(avg_magnitude, avg_weighted)

        # Apply update
        old_confidence = self._model_confidence.get(level.value, 0.5)

        if strategy == UpdateStrategy.MODEL_UPDATE:
            # Model updates to accommodate the evidence
            confidence_delta = -avg_weighted * learning_rate
            new_confidence = max(0.1, min(0.99, old_confidence + confidence_delta))
            error_reduction = avg_magnitude * learning_rate * 0.5
        elif strategy == UpdateStrategy.PERCEPTUAL_INFERENCE:
            # Perception changes to match the model (small updates)
            new_confidence = old_confidence
            error_reduction = avg_magnitude * 0.3
        elif strategy == UpdateStrategy.ATTENTION_SHIFT:
            # Change precision weighting
            new_confidence = old_confidence
            error_reduction = avg_magnitude * 0.2
        else:
            new_confidence = old_confidence
            error_reduction = avg_magnitude * 0.1

        self._model_confidence[level.value] = new_confidence

        update = ModelUpdate(
            update_id=update_id,
            strategy=strategy,
            level=level,
            learning_rate=learning_rate,
            error_reduction=error_reduction,
            new_model_confidence=new_confidence,
            description=f"Updated {level.value} model via {strategy.value}",
        )

        self._update_history.append(update)
        if len(self._update_history) > self._max_history:
            self._update_history.pop(0)

        return update

    def _choose_strategy(
        self, avg_magnitude: float, avg_weighted: float
    ) -> UpdateStrategy:
        """Choose the best update strategy based on error characteristics."""
        if avg_magnitude > 0.7:
            return UpdateStrategy.MODEL_UPDATE
        elif avg_magnitude > 0.4:
            return UpdateStrategy.ATTENTION_SHIFT
        elif avg_weighted > 0.3:
            return UpdateStrategy.PERCEPTUAL_INFERENCE
        else:
            return UpdateStrategy.PERCEPTUAL_INFERENCE

    def get_model_confidence(self, level: PredictionLevel) -> float:
        """Get current model confidence for a level."""
        return self._model_confidence.get(level.value, 0.5)


# ============================================================================
# FREE ENERGY ENGINE
# ============================================================================

class FreeEnergyEngine:
    """
    Engine for computing and minimizing variational free energy.

    Free energy = prediction error (accuracy) + complexity (KL divergence).
    The system aims to minimize free energy over time.
    """

    def __init__(self):
        self._free_energy_history: List[float] = []
        self._max_history = 100

    def compute_free_energy(
        self,
        errors: List[PredictionError],
        model_confidences: Dict[str, float]
    ) -> FreeEnergyState:
        """Compute the current free energy state."""
        # Prediction error component (accuracy)
        pe_component = sum(e.precision_weighted_error for e in errors) / max(1, len(errors))

        # Complexity component (KL divergence approximation)
        complexity = 0.0
        for level, conf in model_confidences.items():
            # Higher confidence = more constrained model = higher complexity
            complexity += conf * 0.3

        complexity /= max(1, len(model_confidences))

        # Entropy component
        if errors:
            error_values = [e.magnitude for e in errors]
            mean_error = sum(error_values) / len(error_values)
            variance = sum((e - mean_error) ** 2 for e in error_values) / len(error_values)
            entropy = math.log(max(0.001, variance + 0.1))
        else:
            entropy = 0.0

        # Total free energy
        total = pe_component * 0.5 + complexity * 0.3 + max(0, entropy) * 0.2

        # Level breakdown
        level_breakdown = {}
        for level in PredictionLevel:
            level_errors = [e for e in errors if e.level == level]
            if level_errors:
                level_fe = sum(e.precision_weighted_error for e in level_errors) / len(level_errors)
                level_breakdown[level.value] = level_fe
            else:
                level_breakdown[level.value] = 0.0

        # Check if minimized (free energy is low and stable)
        self._free_energy_history.append(total)
        if len(self._free_energy_history) > self._max_history:
            self._free_energy_history.pop(0)

        is_minimized = total < 0.2 or self._is_stable()

        return FreeEnergyState(
            total_free_energy=total,
            prediction_error_component=pe_component,
            complexity_component=complexity,
            entropy_component=entropy,
            level_breakdown=level_breakdown,
            is_minimized=is_minimized,
        )

    def _is_stable(self) -> bool:
        """Check if free energy is stable (converged)."""
        if len(self._free_energy_history) < 5:
            return False

        recent = self._free_energy_history[-5:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return variance < 0.01


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class PredictiveCodingInterface:
    """
    Main interface for Form 16: Predictive Coding / Free Energy.

    Implements the predictive processing framework where the brain
    generates top-down predictions, computes prediction errors against
    bottom-up evidence, and updates its generative model to minimize
    free energy (surprise).
    """

    FORM_ID = "16-predictive-coding"
    FORM_NAME = "Predictive Coding / Free Energy Principle"

    def __init__(self):
        """Initialize the Predictive Coding interface."""
        self._error_engine = PredictionErrorEngine()
        self._model_engine = GenerativeModelEngine()
        self._free_energy_engine = FreeEnergyEngine()

        # Tracking
        self._is_initialized = False
        self._total_predictions = 0
        self._total_errors = 0
        self._total_updates = 0
        self._surprise_history: List[float] = []
        self._max_history = 100

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the predictive coding system."""
        self._is_initialized = True
        logger.info(f"{self.FORM_NAME} initialized and ready")

    async def generate_prediction(
        self,
        level: PredictionLevel,
        context: Dict[str, Any],
        prediction_id: str = ""
    ) -> Prediction:
        """
        Generate a top-down prediction at the specified level.

        Uses the current generative model to predict what should
        be observed given the current context.
        """
        prediction = self._model_engine.generate_prediction(level, context, prediction_id)
        self._total_predictions += 1
        return prediction

    async def compute_prediction_error(
        self,
        prediction: Prediction,
        evidence: SensoryEvidence,
        attention_weight: float = 1.0
    ) -> PredictionError:
        """
        Compute the prediction error between a prediction and evidence.

        Returns a precision-weighted error that drives model updating.
        """
        error = self._error_engine.compute_prediction_error(
            prediction, evidence, attention_weight
        )
        self._total_errors += 1
        self._surprise_history.append(error.surprise)
        if len(self._surprise_history) > self._max_history:
            self._surprise_history.pop(0)
        return error

    async def update_generative_model(
        self,
        errors: List[PredictionError],
        level: PredictionLevel
    ) -> ModelUpdate:
        """
        Update the generative model based on accumulated prediction errors.

        Selects the appropriate update strategy and applies changes
        to reduce future prediction errors.
        """
        update = self._model_engine.update_model(errors, level)
        self._total_updates += 1
        return update

    async def minimize_free_energy(
        self,
        pc_input: PredictiveCodingInput
    ) -> PredictiveCodingOutput:
        """
        Run the full predictive coding cycle to minimize free energy.

        This is the main entry point: generate predictions, compute
        errors, update the model, and assess free energy state.
        """
        all_errors = []
        all_updates = []

        # Process each prediction-evidence pair
        for prediction in pc_input.predictions:
            # Find matching evidence at the same level
            matching_evidence = [
                e for e in pc_input.evidence if e.level == prediction.level
            ]

            for evidence in matching_evidence:
                attention = pc_input.attention_weights.get(
                    prediction.level.value, 1.0
                )
                error = await self.compute_prediction_error(
                    prediction, evidence, attention
                )
                all_errors.append(error)

        # Update model at each level with errors
        levels_with_errors = set(e.level for e in all_errors)
        for level in levels_with_errors:
            update = await self.update_generative_model(all_errors, level)
            all_updates.append(update)

        # Compute free energy
        model_confidences = {
            level.value: self._model_engine.get_model_confidence(level)
            for level in PredictionLevel
        }
        free_energy = self._free_energy_engine.compute_free_energy(
            all_errors, model_confidences
        )

        # Compute total surprise
        total_surprise = sum(e.surprise for e in all_errors)

        # Find dominant error level
        if all_errors:
            level_errors = {}
            for e in all_errors:
                if e.level not in level_errors:
                    level_errors[e.level] = 0.0
                level_errors[e.level] += e.precision_weighted_error
            dominant = max(level_errors, key=level_errors.get)
        else:
            dominant = PredictionLevel.SENSORY

        # Determine update strategy
        strategy = all_updates[0].strategy if all_updates else UpdateStrategy.PERCEPTUAL_INFERENCE

        return PredictiveCodingOutput(
            prediction_errors=all_errors,
            model_updates=all_updates,
            free_energy_state=free_energy,
            total_surprise=total_surprise,
            dominant_error_level=dominant,
            update_strategy_used=strategy,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        avg_surprise = (
            sum(self._surprise_history) / len(self._surprise_history)
            if self._surprise_history else 0.0
        )
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "is_initialized": self._is_initialized,
            "total_predictions": self._total_predictions,
            "total_errors": self._total_errors,
            "total_updates": self._total_updates,
            "average_surprise": round(avg_surprise, 6),
            "model_confidences": {
                level.value: round(self._model_engine.get_model_confidence(level), 4)
                for level in PredictionLevel
            },
        }

    def get_status(self) -> PredictiveCodingSystemStatus:
        """Get current system status."""
        avg_surprise = (
            sum(self._surprise_history) / len(self._surprise_history)
            if self._surprise_history else 0.0
        )
        fe_history = self._free_energy_engine._free_energy_history
        current_fe = fe_history[-1] if fe_history else 0.0

        return PredictiveCodingSystemStatus(
            is_initialized=self._is_initialized,
            total_predictions=self._total_predictions,
            total_errors_computed=self._total_errors,
            total_model_updates=self._total_updates,
            current_free_energy=current_fe,
            average_surprise=avg_surprise,
            system_health=1.0 if self._is_initialized else 0.5,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_predictive_coding_interface() -> PredictiveCodingInterface:
    """Create and return a Predictive Coding interface."""
    return PredictiveCodingInterface()


def create_prediction(
    prediction_id: str,
    level: PredictionLevel = PredictionLevel.PERCEPTUAL,
    content: Optional[Dict[str, Any]] = None,
    confidence: float = 0.7,
) -> Prediction:
    """Create a prediction for testing."""
    return Prediction(
        prediction_id=prediction_id,
        level=level,
        content=content or {"value": 0.5},
        confidence=confidence,
        precision=confidence * 0.8 + 0.2,
    )


def create_evidence(
    evidence_id: str,
    level: PredictionLevel = PredictionLevel.PERCEPTUAL,
    content: Optional[Dict[str, Any]] = None,
    reliability: float = 0.8,
) -> SensoryEvidence:
    """Create sensory evidence for testing."""
    return SensoryEvidence(
        evidence_id=evidence_id,
        level=level,
        content=content or {"value": 0.6},
        reliability=reliability,
        precision=reliability * 0.8 + 0.2,
    )


__all__ = [
    # Enums
    "PredictionLevel",
    "ErrorType",
    "PrecisionLevel",
    "UpdateStrategy",
    "FreeEnergyComponent",
    # Input dataclasses
    "Prediction",
    "SensoryEvidence",
    "PredictiveCodingInput",
    # Output dataclasses
    "PredictionError",
    "ModelUpdate",
    "FreeEnergyState",
    "PredictiveCodingOutput",
    "PredictiveCodingSystemStatus",
    # Engines
    "PredictionErrorEngine",
    "GenerativeModelEngine",
    "FreeEnergyEngine",
    # Main interface
    "PredictiveCodingInterface",
    # Convenience functions
    "create_predictive_coding_interface",
    "create_prediction",
    "create_evidence",
]
