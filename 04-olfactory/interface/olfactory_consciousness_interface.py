#!/usr/bin/env python3
"""
Olfactory Consciousness Interface

Form 04: The olfactory processing system for consciousness.
Olfactory consciousness processes chemical signals (odorants) through
receptor activation patterns, odor identification, hedonic evaluation,
and memory association to construct conscious smell experience.

This form is uniquely connected to emotional and memory systems,
as the olfactory bulb projects directly to the limbic system,
bypassing thalamic relay.
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

class OdorCategory(Enum):
    """Primary categories of odor classification."""
    FLORAL = "floral"
    FRUITY = "fruity"
    WOODY = "woody"
    SPICY = "spicy"
    HERBAL = "herbal"
    EARTHY = "earthy"
    CHEMICAL = "chemical"
    FOOD = "food"
    SMOKE = "smoke"
    DECAY = "decay"
    BODY = "body"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class OlfactoryQuality(Enum):
    """Perceptual qualities of odor experience."""
    SWEET = "sweet"
    PUNGENT = "pungent"
    MUSKY = "musky"
    PUTRID = "putrid"
    CAMPHORACEOUS = "camphoraceous"
    ETHEREAL = "ethereal"
    MINTY = "minty"
    WARM = "warm"
    COOL = "cool"
    SHARP = "sharp"


class OdorIntensityLevel(Enum):
    """Intensity levels for odor perception."""
    THRESHOLD = "threshold"        # Barely detectable
    FAINT = "faint"
    MODERATE = "moderate"
    STRONG = "strong"
    OVERWHELMING = "overwhelming"


class HedonicValence(Enum):
    """Hedonic evaluation categories."""
    VERY_PLEASANT = "very_pleasant"
    PLEASANT = "pleasant"
    NEUTRAL = "neutral"
    UNPLEASANT = "unpleasant"
    VERY_UNPLEASANT = "very_unpleasant"


class OlfactoryAdaptationState(Enum):
    """State of olfactory adaptation (habituation)."""
    FRESH = "fresh"             # No adaptation, full sensitivity
    ADAPTING = "adapting"       # Partially adapted
    ADAPTED = "adapted"         # Fully adapted, minimal perception
    CROSS_ADAPTED = "cross_adapted"  # Adapted to similar odors


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class ChemicalFeatures:
    """Chemical feature representation of odorant molecules."""
    molecular_weight: float     # Normalized 0.0-1.0
    volatility: float           # 0.0-1.0
    hydrophobicity: float       # 0.0-1.0
    functional_groups: List[str]  # e.g., ["alcohol", "aldehyde"]
    receptor_activation: Dict[str, float]  # receptor_id -> activation level
    concentration: float = 0.5  # 0.0-1.0 normalized
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecular_weight": round(self.molecular_weight, 4),
            "volatility": round(self.volatility, 4),
            "hydrophobicity": round(self.hydrophobicity, 4),
            "functional_groups": self.functional_groups,
            "num_receptors_activated": len(self.receptor_activation),
            "concentration": round(self.concentration, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OlfactoryInput:
    """Input to the olfactory consciousness system."""
    chemical_features: Optional[ChemicalFeatures] = None
    intensity: float = 0.0       # 0.0-1.0 perceived intensity
    onset_detected: bool = False  # New odor onset
    concentration_change: float = 0.0  # Rate of change (-1.0 to 1.0)
    sniff_active: bool = False    # Active sniffing behavior
    ambient_odor_level: float = 0.0  # Background odor level
    num_distinct_odors: int = 1   # Number of separable odors
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_chemical_features": self.chemical_features is not None,
            "intensity": round(self.intensity, 4),
            "onset_detected": self.onset_detected,
            "concentration_change": round(self.concentration_change, 4),
            "sniff_active": self.sniff_active,
            "ambient_odor_level": round(self.ambient_odor_level, 4),
            "num_distinct_odors": self.num_distinct_odors,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class OdorIdentification:
    """Identification result for a detected odor."""
    category: OdorCategory
    label: str
    qualities: List[OlfactoryQuality]
    confidence: float
    intensity_level: OdorIntensityLevel
    familiarity: float         # 0.0-1.0
    distinctiveness: float     # 0.0-1.0 how distinct from background
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "label": self.label,
            "qualities": [q.value for q in self.qualities],
            "confidence": round(self.confidence, 4),
            "intensity_level": self.intensity_level.value,
            "familiarity": round(self.familiarity, 4),
            "distinctiveness": round(self.distinctiveness, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HedonicEvaluation:
    """Hedonic (pleasure/displeasure) evaluation of an odor."""
    valence: HedonicValence
    valence_score: float        # -1.0 to 1.0
    approach_tendency: float    # 0.0-1.0 (desire to approach)
    avoidance_tendency: float   # 0.0-1.0 (desire to avoid)
    emotional_associations: List[str]  # e.g., ["comfort", "nostalgia"]
    appetitive: bool = False    # Related to food/appetite
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence.value,
            "valence_score": round(self.valence_score, 4),
            "approach_tendency": round(self.approach_tendency, 4),
            "avoidance_tendency": round(self.avoidance_tendency, 4),
            "emotional_associations": self.emotional_associations,
            "appetitive": self.appetitive,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MemoryAssociation:
    """Memory associations triggered by an odor."""
    has_memory_trigger: bool
    memory_strength: float      # 0.0-1.0
    memory_type: str            # "episodic", "semantic", "emotional"
    memory_valence: float       # -1.0 to 1.0
    context_cues: List[str]     # Associated context elements
    proustian_effect: float     # 0.0-1.0 strength of involuntary memory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_memory_trigger": self.has_memory_trigger,
            "memory_strength": round(self.memory_strength, 4),
            "memory_type": self.memory_type,
            "memory_valence": round(self.memory_valence, 4),
            "context_cues": self.context_cues,
            "proustian_effect": round(self.proustian_effect, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OlfactoryOutput:
    """Complete output of olfactory consciousness processing."""
    odor_identification: OdorIdentification
    hedonic_evaluation: HedonicEvaluation
    memory_association: MemoryAssociation
    adaptation_state: OlfactoryAdaptationState
    overall_salience: float      # 0.0-1.0
    requires_attention: bool
    safety_alert: bool = False   # Dangerous odor detected
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "odor_identification": self.odor_identification.to_dict(),
            "hedonic_evaluation": self.hedonic_evaluation.to_dict(),
            "memory_association": self.memory_association.to_dict(),
            "adaptation_state": self.adaptation_state.value,
            "overall_salience": round(self.overall_salience, 4),
            "requires_attention": self.requires_attention,
            "safety_alert": self.safety_alert,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class OlfactoryConsciousnessInterface:
    """
    Main interface for Form 04: Olfactory Consciousness.

    Processes odorant stimuli through chemical analysis, pattern
    recognition, hedonic evaluation, and memory association to
    produce conscious smell experience. Uniquely linked to
    emotional and autobiographical memory systems.
    """

    FORM_ID = "04-olfactory"
    FORM_NAME = "Olfactory Consciousness"

    def __init__(self):
        """Initialize the olfactory consciousness interface."""
        self._initialized = False
        self._processing_count = 0
        self._current_output: Optional[OlfactoryOutput] = None
        self._adaptation_level: float = 0.0  # 0.0 (fresh) to 1.0 (fully adapted)
        self._odor_history: List[OdorIdentification] = []
        self._known_odors: Dict[str, float] = {}  # label -> familiarity
        self._memory_associations: Dict[str, List[str]] = {}
        self._max_history = 50
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the olfactory processing pipeline."""
        self._initialized = True
        self._adaptation_level = 0.0
        logger.info(f"{self.FORM_NAME} pipeline initialized")

    async def process_olfactory_input(
        self, olfactory_input: OlfactoryInput
    ) -> OlfactoryOutput:
        """
        Process olfactory input through the consciousness pipeline.

        Pipeline stages:
        1. Odor identification
        2. Hedonic evaluation
        3. Memory association
        4. Adaptation update
        5. Integration and salience
        """
        self._processing_count += 1

        # Stage 1: Identify odor
        odor_id = await self._identify_odor(olfactory_input)

        # Stage 2: Hedonic evaluation
        hedonic = await self._evaluate_hedonic(olfactory_input, odor_id)

        # Stage 3: Memory association
        memory = await self._associate_memories(odor_id, hedonic)

        # Stage 4: Update adaptation
        adaptation = self._update_adaptation(olfactory_input, odor_id)

        # Stage 5: Integration
        salience = self._compute_salience(olfactory_input, odor_id, hedonic, memory)
        requires_attention = salience > 0.6 or olfactory_input.onset_detected
        safety = self._check_safety(odor_id, olfactory_input)

        output = OlfactoryOutput(
            odor_identification=odor_id,
            hedonic_evaluation=hedonic,
            memory_association=memory,
            adaptation_state=adaptation,
            overall_salience=salience,
            requires_attention=requires_attention,
            safety_alert=safety,
        )

        self._current_output = output
        self._update_history(odor_id)

        return output

    async def _identify_odor(self, olfactory_input: OlfactoryInput) -> OdorIdentification:
        """Identify the odor from chemical and perceptual features."""
        if olfactory_input.intensity < 0.05:
            return OdorIdentification(
                category=OdorCategory.NEUTRAL,
                label="no_odor",
                qualities=[],
                confidence=0.9,
                intensity_level=OdorIntensityLevel.THRESHOLD,
                familiarity=1.0,
                distinctiveness=0.0,
            )

        # Determine category from chemical features
        category = OdorCategory.UNKNOWN
        qualities = []
        label = "unidentified"

        if olfactory_input.chemical_features:
            cf = olfactory_input.chemical_features
            if cf.volatility > 0.7:
                category = OdorCategory.CHEMICAL
                qualities.append(OlfactoryQuality.SHARP)
            elif cf.hydrophobicity > 0.6:
                category = OdorCategory.WOODY
                qualities.append(OlfactoryQuality.WARM)
            elif "alcohol" in cf.functional_groups:
                category = OdorCategory.FRUITY
                qualities.append(OlfactoryQuality.SWEET)
            elif "sulfur" in cf.functional_groups:
                category = OdorCategory.DECAY
                qualities.append(OlfactoryQuality.PUTRID)
            else:
                category = OdorCategory.HERBAL
                qualities.append(OlfactoryQuality.COOL)

            label = f"{category.value}_odor"
        else:
            # Heuristic without chemical features
            if olfactory_input.intensity > 0.7:
                category = OdorCategory.CHEMICAL
                qualities.append(OlfactoryQuality.PUNGENT)
            else:
                category = OdorCategory.FLORAL
                qualities.append(OlfactoryQuality.SWEET)
            label = f"{category.value}_odor"

        # Determine intensity level
        intensity = olfactory_input.intensity
        if intensity < 0.2:
            intensity_level = OdorIntensityLevel.FAINT
        elif intensity < 0.5:
            intensity_level = OdorIntensityLevel.MODERATE
        elif intensity < 0.8:
            intensity_level = OdorIntensityLevel.STRONG
        else:
            intensity_level = OdorIntensityLevel.OVERWHELMING

        # Adaptation reduces perceived intensity
        adapted_intensity = intensity * (1.0 - self._adaptation_level * 0.7)
        familiarity = self._known_odors.get(label, 0.0)
        distinctiveness = min(1.0, adapted_intensity * (1.0 - olfactory_input.ambient_odor_level))

        return OdorIdentification(
            category=category,
            label=label,
            qualities=qualities,
            confidence=0.6 + adapted_intensity * 0.3,
            intensity_level=intensity_level,
            familiarity=familiarity,
            distinctiveness=distinctiveness,
        )

    async def _evaluate_hedonic(
        self, olfactory_input: OlfactoryInput, odor_id: OdorIdentification
    ) -> HedonicEvaluation:
        """Evaluate the hedonic quality (pleasantness) of the odor."""
        # Category-based hedonic defaults
        hedonic_defaults = {
            OdorCategory.FLORAL: 0.6,
            OdorCategory.FRUITY: 0.5,
            OdorCategory.WOODY: 0.3,
            OdorCategory.SPICY: 0.1,
            OdorCategory.HERBAL: 0.2,
            OdorCategory.EARTHY: 0.0,
            OdorCategory.CHEMICAL: -0.3,
            OdorCategory.FOOD: 0.4,
            OdorCategory.SMOKE: -0.2,
            OdorCategory.DECAY: -0.8,
            OdorCategory.BODY: -0.4,
            OdorCategory.NEUTRAL: 0.0,
            OdorCategory.UNKNOWN: 0.0,
        }

        base_valence = hedonic_defaults.get(odor_id.category, 0.0)

        # Intensity affects hedonic: very strong odors become unpleasant
        if olfactory_input.intensity > 0.8:
            base_valence -= 0.3

        # Familiarity tends to increase pleasantness slightly
        base_valence += odor_id.familiarity * 0.1

        valence_score = max(-1.0, min(1.0, base_valence))

        # Determine hedonic category
        if valence_score > 0.4:
            valence = HedonicValence.VERY_PLEASANT
        elif valence_score > 0.1:
            valence = HedonicValence.PLEASANT
        elif valence_score > -0.1:
            valence = HedonicValence.NEUTRAL
        elif valence_score > -0.4:
            valence = HedonicValence.UNPLEASANT
        else:
            valence = HedonicValence.VERY_UNPLEASANT

        # Approach/avoidance
        approach = max(0.0, valence_score)
        avoidance = max(0.0, -valence_score)

        # Emotional associations
        emotions = []
        if valence_score > 0.3:
            emotions.append("comfort")
        if odor_id.familiarity > 0.5:
            emotions.append("nostalgia")
        if valence_score < -0.5:
            emotions.append("disgust")
        if odor_id.category == OdorCategory.FOOD:
            emotions.append("appetite")

        appetitive = odor_id.category in [OdorCategory.FOOD, OdorCategory.FRUITY]

        return HedonicEvaluation(
            valence=valence,
            valence_score=valence_score,
            approach_tendency=approach,
            avoidance_tendency=avoidance,
            emotional_associations=emotions,
            appetitive=appetitive,
        )

    async def _associate_memories(
        self, odor_id: OdorIdentification, hedonic: HedonicEvaluation
    ) -> MemoryAssociation:
        """Associate odor with memories (Proustian memory effect)."""
        has_trigger = odor_id.familiarity > 0.3
        memory_strength = odor_id.familiarity * 0.8
        proustian = 0.0

        if has_trigger:
            proustian = odor_id.familiarity * 0.6
            if abs(hedonic.valence_score) > 0.5:
                proustian = min(1.0, proustian + 0.2)

        memory_type = "emotional"
        if odor_id.familiarity > 0.7:
            memory_type = "episodic"
        elif odor_id.familiarity > 0.4:
            memory_type = "semantic"

        context_cues = self._memory_associations.get(odor_id.label, [])
        if not context_cues and has_trigger:
            context_cues = [f"context_{odor_id.category.value}"]

        return MemoryAssociation(
            has_memory_trigger=has_trigger,
            memory_strength=memory_strength,
            memory_type=memory_type,
            memory_valence=hedonic.valence_score,
            context_cues=context_cues,
            proustian_effect=proustian,
        )

    def _update_adaptation(
        self, olfactory_input: OlfactoryInput, odor_id: OdorIdentification
    ) -> OlfactoryAdaptationState:
        """Update olfactory adaptation state."""
        if olfactory_input.onset_detected:
            # New odor resets adaptation
            self._adaptation_level = 0.0
            return OlfactoryAdaptationState.FRESH

        if olfactory_input.intensity > 0.1:
            # Gradual adaptation
            self._adaptation_level = min(1.0, self._adaptation_level + 0.05)

        if self._adaptation_level < 0.2:
            return OlfactoryAdaptationState.FRESH
        elif self._adaptation_level < 0.7:
            return OlfactoryAdaptationState.ADAPTING
        else:
            return OlfactoryAdaptationState.ADAPTED

    def _compute_salience(
        self,
        olfactory_input: OlfactoryInput,
        odor_id: OdorIdentification,
        hedonic: HedonicEvaluation,
        memory: MemoryAssociation,
    ) -> float:
        """Compute overall salience of the olfactory experience."""
        salience = 0.0

        # Intensity contributes
        salience += olfactory_input.intensity * 0.3

        # Novel onset is salient
        if olfactory_input.onset_detected:
            salience += 0.3

        # Strong hedonic value increases salience
        salience += abs(hedonic.valence_score) * 0.2

        # Memory triggers increase salience
        salience += memory.proustian_effect * 0.2

        # Adaptation reduces salience
        salience *= (1.0 - self._adaptation_level * 0.5)

        return max(0.0, min(1.0, salience))

    def _check_safety(
        self, odor_id: OdorIdentification, olfactory_input: OlfactoryInput
    ) -> bool:
        """Check if odor indicates safety concern."""
        dangerous_categories = [OdorCategory.CHEMICAL, OdorCategory.SMOKE, OdorCategory.DECAY]
        if odor_id.category in dangerous_categories and olfactory_input.intensity > 0.6:
            return True
        return False

    def _update_history(self, odor_id: OdorIdentification) -> None:
        """Update odor history and familiarity."""
        self._odor_history.append(odor_id)
        if len(self._odor_history) > self._max_history:
            self._odor_history.pop(0)

        current = self._known_odors.get(odor_id.label, 0.0)
        self._known_odors[odor_id.label] = min(1.0, current + 0.1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for serialization."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "processing_count": self._processing_count,
            "adaptation_level": round(self._adaptation_level, 4),
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "known_odors": len(self._known_odors),
            "history_length": len(self._odor_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current form status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "operational": True,
            "processing_count": self._processing_count,
            "adaptation_level": round(self._adaptation_level, 4),
            "known_odors": len(self._known_odors),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_olfactory_interface() -> OlfactoryConsciousnessInterface:
    """Create and return an olfactory consciousness interface."""
    return OlfactoryConsciousnessInterface()


def create_simple_olfactory_input(
    intensity: float = 0.5,
    onset: bool = False,
    category_hint: str = "floral",
) -> OlfactoryInput:
    """Create a simple olfactory input for testing."""
    functional_groups = []
    if category_hint == "fruity":
        functional_groups = ["alcohol"]
    elif category_hint == "decay":
        functional_groups = ["sulfur"]

    return OlfactoryInput(
        chemical_features=ChemicalFeatures(
            molecular_weight=0.5,
            volatility=0.5,
            hydrophobicity=0.3,
            functional_groups=functional_groups,
            receptor_activation={"OR1": 0.7, "OR2": 0.4},
            concentration=intensity,
        ),
        intensity=intensity,
        onset_detected=onset,
    )


__all__ = [
    # Enums
    "OdorCategory",
    "OlfactoryQuality",
    "OdorIntensityLevel",
    "HedonicValence",
    "OlfactoryAdaptationState",
    # Input dataclasses
    "ChemicalFeatures",
    "OlfactoryInput",
    # Output dataclasses
    "OdorIdentification",
    "HedonicEvaluation",
    "MemoryAssociation",
    "OlfactoryOutput",
    # Main interface
    "OlfactoryConsciousnessInterface",
    # Convenience functions
    "create_olfactory_interface",
    "create_simple_olfactory_input",
]
