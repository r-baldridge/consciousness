#!/usr/bin/env python3
"""
Gustatory Consciousness Interface

Form 05: The gustatory processing system for consciousness.
Gustatory consciousness processes taste signals from the five primary
taste modalities (sweet, sour, salty, bitter, umami) through receptor
activation, flavor integration with olfactory and textural input,
and palatability assessment.

This form handles the transformation of chemical taste stimuli into
conscious taste experience, integrating with olfactory and somatosensory
systems for full flavor perception.
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

class TasteModality(Enum):
    """The five primary taste modalities."""
    SWEET = "sweet"
    SOUR = "sour"
    SALTY = "salty"
    BITTER = "bitter"
    UMAMI = "umami"


class FlavorProfile(Enum):
    """High-level flavor profiles combining taste and aroma."""
    SAVORY = "savory"
    SWEET_AROMATIC = "sweet_aromatic"
    SOUR_TANGY = "sour_tangy"
    BITTER_COMPLEX = "bitter_complex"
    SPICY_HOT = "spicy_hot"
    MILD = "mild"
    RICH = "rich"
    FRESH = "fresh"
    FERMENTED = "fermented"
    NEUTRAL = "neutral"


class TextureQuality(Enum):
    """Texture qualities relevant to taste experience."""
    SMOOTH = "smooth"
    CRUNCHY = "crunchy"
    CREAMY = "creamy"
    GRAINY = "grainy"
    CHEWY = "chewy"
    LIQUID = "liquid"
    FIZZY = "fizzy"
    DRY = "dry"


class PalatabilityLevel(Enum):
    """Overall palatability assessment levels."""
    DELICIOUS = "delicious"
    PLEASANT = "pleasant"
    ACCEPTABLE = "acceptable"
    BLAND = "bland"
    UNPLEASANT = "unpleasant"
    AVERSIVE = "aversive"


class AppetiteState(Enum):
    """Current appetite state affecting taste perception."""
    HUNGRY = "hungry"
    SATIATED = "satiated"
    CRAVING = "craving"
    NAUSEOUS = "nauseous"
    NEUTRAL = "neutral"


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class TasteReceptorData:
    """Taste receptor activation data."""
    modality_activations: Dict[str, float]  # TasteModality.value -> activation 0.0-1.0
    receptor_density: float = 0.5    # 0.0-1.0 sensitivity
    adaptation_level: float = 0.0    # 0.0-1.0 adaptation
    temperature: float = 0.5         # 0.0 (cold) - 1.0 (hot)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality_activations": {
                k: round(v, 4) for k, v in self.modality_activations.items()
            },
            "receptor_density": round(self.receptor_density, 4),
            "adaptation_level": round(self.adaptation_level, 4),
            "temperature": round(self.temperature, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GustatoryInput:
    """Input to the gustatory consciousness system."""
    taste_receptor_data: Optional[TasteReceptorData] = None
    overall_intensity: float = 0.0   # 0.0-1.0
    texture_quality: TextureQuality = TextureQuality.SMOOTH
    olfactory_contribution: float = 0.0  # 0.0-1.0 retronasal olfaction
    oral_temperature: float = 0.5    # 0.0 (cold) - 1.0 (hot)
    oral_irritation: float = 0.0     # 0.0-1.0 (capsaicin, menthol, etc.)
    appetite_state: AppetiteState = AppetiteState.NEUTRAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_receptor_data": self.taste_receptor_data is not None,
            "overall_intensity": round(self.overall_intensity, 4),
            "texture_quality": self.texture_quality.value,
            "olfactory_contribution": round(self.olfactory_contribution, 4),
            "oral_temperature": round(self.oral_temperature, 4),
            "oral_irritation": round(self.oral_irritation, 4),
            "appetite_state": self.appetite_state.value,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class TasteIdentification:
    """Identification of taste components."""
    dominant_modality: TasteModality
    modality_strengths: Dict[str, float]  # modality -> perceived strength
    taste_complexity: float      # 0.0-1.0 (simple to complex)
    taste_harmony: float         # 0.0-1.0 how well tastes blend
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_modality": self.dominant_modality.value,
            "modality_strengths": {
                k: round(v, 4) for k, v in self.modality_strengths.items()
            },
            "taste_complexity": round(self.taste_complexity, 4),
            "taste_harmony": round(self.taste_harmony, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FlavorIntegration:
    """Integrated flavor experience combining taste, aroma, and texture."""
    flavor_profile: FlavorProfile
    flavor_richness: float       # 0.0-1.0
    aroma_contribution: float    # 0.0-1.0 how much aroma adds
    texture_contribution: float  # 0.0-1.0 how much texture adds
    temperature_influence: float # -1.0 to 1.0
    overall_intensity: float     # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flavor_profile": self.flavor_profile.value,
            "flavor_richness": round(self.flavor_richness, 4),
            "aroma_contribution": round(self.aroma_contribution, 4),
            "texture_contribution": round(self.texture_contribution, 4),
            "temperature_influence": round(self.temperature_influence, 4),
            "overall_intensity": round(self.overall_intensity, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PalatabilityAssessment:
    """Assessment of overall palatability and food acceptability."""
    palatability: PalatabilityLevel
    palatability_score: float    # -1.0 to 1.0
    desire_to_consume: float     # 0.0-1.0
    satiety_signal: float        # 0.0-1.0 how filling
    safety_assessment: float     # 0.0-1.0 (safe to consume)
    novelty: float               # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "palatability": self.palatability.value,
            "palatability_score": round(self.palatability_score, 4),
            "desire_to_consume": round(self.desire_to_consume, 4),
            "satiety_signal": round(self.satiety_signal, 4),
            "safety_assessment": round(self.safety_assessment, 4),
            "novelty": round(self.novelty, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GustatoryOutput:
    """Complete output of gustatory consciousness processing."""
    taste_identification: TasteIdentification
    flavor_integration: FlavorIntegration
    palatability_assessment: PalatabilityAssessment
    overall_pleasure: float      # -1.0 to 1.0
    food_safety_alert: bool
    appetite_modulation: float   # -1.0 (suppressed) to 1.0 (enhanced)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taste_identification": self.taste_identification.to_dict(),
            "flavor_integration": self.flavor_integration.to_dict(),
            "palatability_assessment": self.palatability_assessment.to_dict(),
            "overall_pleasure": round(self.overall_pleasure, 4),
            "food_safety_alert": self.food_safety_alert,
            "appetite_modulation": round(self.appetite_modulation, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class GustatoryConsciousnessInterface:
    """
    Main interface for Form 05: Gustatory Consciousness.

    Processes taste stimuli through receptor activation analysis,
    flavor integration (taste + aroma + texture), and palatability
    assessment to produce conscious taste experience.
    """

    FORM_ID = "05-gustatory"
    FORM_NAME = "Gustatory Consciousness"

    def __init__(self):
        """Initialize the gustatory consciousness interface."""
        self._initialized = False
        self._processing_count = 0
        self._current_output: Optional[GustatoryOutput] = None
        self._taste_history: List[TasteIdentification] = []
        self._known_flavors: Dict[str, float] = {}  # profile -> familiarity
        self._appetite_state = AppetiteState.NEUTRAL
        self._satiety_level = 0.0
        self._max_history = 50
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the gustatory processing pipeline."""
        self._initialized = True
        self._appetite_state = AppetiteState.NEUTRAL
        logger.info(f"{self.FORM_NAME} pipeline initialized")

    async def process_gustatory_input(
        self, gustatory_input: GustatoryInput
    ) -> GustatoryOutput:
        """
        Process gustatory input through the consciousness pipeline.

        Pipeline stages:
        1. Taste identification (receptor analysis)
        2. Flavor integration (taste + aroma + texture)
        3. Palatability assessment
        4. Appetite modulation
        """
        self._processing_count += 1
        self._appetite_state = gustatory_input.appetite_state

        # Stage 1: Identify tastes
        taste_id = await self._identify_tastes(gustatory_input)

        # Stage 2: Integrate flavor
        flavor = await self._integrate_flavor(gustatory_input, taste_id)

        # Stage 3: Assess palatability
        palatability = await self._assess_palatability(gustatory_input, taste_id, flavor)

        # Stage 4: Compute outputs
        pleasure = self._compute_pleasure(palatability, gustatory_input)
        safety_alert = self._check_food_safety(taste_id, gustatory_input)
        appetite_mod = self._compute_appetite_modulation(palatability, gustatory_input)

        output = GustatoryOutput(
            taste_identification=taste_id,
            flavor_integration=flavor,
            palatability_assessment=palatability,
            overall_pleasure=pleasure,
            food_safety_alert=safety_alert,
            appetite_modulation=appetite_mod,
        )

        self._current_output = output
        self._update_history(taste_id, flavor)

        return output

    async def _identify_tastes(self, gustatory_input: GustatoryInput) -> TasteIdentification:
        """Identify taste components from receptor data."""
        modality_strengths = {}

        if gustatory_input.taste_receptor_data:
            rd = gustatory_input.taste_receptor_data
            for modality in TasteModality:
                activation = rd.modality_activations.get(modality.value, 0.0)
                # Adjust for adaptation and sensitivity
                perceived = activation * rd.receptor_density * (1.0 - rd.adaptation_level * 0.5)
                modality_strengths[modality.value] = max(0.0, min(1.0, perceived))
        else:
            # Default low activation across modalities
            for modality in TasteModality:
                modality_strengths[modality.value] = gustatory_input.overall_intensity * 0.3

        # Determine dominant modality
        if modality_strengths:
            dominant_name = max(modality_strengths, key=modality_strengths.get)
            dominant = TasteModality(dominant_name)
        else:
            dominant = TasteModality.SWEET

        # Compute complexity and harmony
        values = list(modality_strengths.values())
        active_count = sum(1 for v in values if v > 0.1)
        complexity = min(1.0, active_count / 5.0)

        # Harmony: similar strengths = harmonious
        if values and max(values) > 0:
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            harmony = max(0.0, 1.0 - variance * 10)
        else:
            harmony = 0.5

        confidence = 0.5 + gustatory_input.overall_intensity * 0.4

        return TasteIdentification(
            dominant_modality=dominant,
            modality_strengths=modality_strengths,
            taste_complexity=complexity,
            taste_harmony=harmony,
            confidence=min(1.0, confidence),
        )

    async def _integrate_flavor(
        self, gustatory_input: GustatoryInput, taste_id: TasteIdentification
    ) -> FlavorIntegration:
        """Integrate taste with aroma and texture for full flavor."""
        # Determine flavor profile from dominant taste and aroma
        dominant = taste_id.dominant_modality
        aroma = gustatory_input.olfactory_contribution

        profile = self._determine_flavor_profile(dominant, aroma, gustatory_input)

        # Compute richness from multiple inputs
        richness = (
            taste_id.taste_complexity * 0.4 +
            gustatory_input.olfactory_contribution * 0.3 +
            gustatory_input.overall_intensity * 0.3
        )

        # Temperature influence (-1 = cooling, +1 = warming)
        temp = gustatory_input.oral_temperature
        temp_influence = (temp - 0.5) * 2.0

        # Texture contribution
        texture_contrib = 0.3  # Base
        if gustatory_input.texture_quality in [TextureQuality.CREAMY, TextureQuality.CRUNCHY]:
            texture_contrib = 0.5
        elif gustatory_input.texture_quality == TextureQuality.FIZZY:
            texture_contrib = 0.6

        return FlavorIntegration(
            flavor_profile=profile,
            flavor_richness=min(1.0, richness),
            aroma_contribution=aroma,
            texture_contribution=texture_contrib,
            temperature_influence=max(-1.0, min(1.0, temp_influence)),
            overall_intensity=gustatory_input.overall_intensity,
        )

    async def _assess_palatability(
        self,
        gustatory_input: GustatoryInput,
        taste_id: TasteIdentification,
        flavor: FlavorIntegration,
    ) -> PalatabilityAssessment:
        """Assess overall palatability."""
        # Base palatability from flavor richness and harmony
        score = taste_id.taste_harmony * 0.4 + flavor.flavor_richness * 0.3

        # Appetite state modulates palatability
        if gustatory_input.appetite_state == AppetiteState.HUNGRY:
            score += 0.2
        elif gustatory_input.appetite_state == AppetiteState.NAUSEOUS:
            score -= 0.5
        elif gustatory_input.appetite_state == AppetiteState.SATIATED:
            score -= 0.1

        # Bitter at high levels reduces palatability
        bitter = taste_id.modality_strengths.get("bitter", 0.0)
        if bitter > 0.6:
            score -= 0.3

        # Sweet tends to increase palatability
        sweet = taste_id.modality_strengths.get("sweet", 0.0)
        score += sweet * 0.2

        # Oral irritation modulates
        if gustatory_input.oral_irritation > 0.5:
            score -= gustatory_input.oral_irritation * 0.3

        score = max(-1.0, min(1.0, score))

        # Determine level
        if score > 0.6:
            level = PalatabilityLevel.DELICIOUS
        elif score > 0.3:
            level = PalatabilityLevel.PLEASANT
        elif score > 0.0:
            level = PalatabilityLevel.ACCEPTABLE
        elif score > -0.2:
            level = PalatabilityLevel.BLAND
        elif score > -0.5:
            level = PalatabilityLevel.UNPLEASANT
        else:
            level = PalatabilityLevel.AVERSIVE

        desire = max(0.0, (score + 1.0) / 2.0)
        satiety = min(1.0, self._satiety_level + gustatory_input.overall_intensity * 0.1)
        self._satiety_level = satiety

        # Safety: bitter can signal toxins
        safety = 1.0
        if bitter > 0.8:
            safety = 0.5

        novelty = 1.0 - self._known_flavors.get(flavor.flavor_profile.value, 0.0)

        return PalatabilityAssessment(
            palatability=level,
            palatability_score=score,
            desire_to_consume=desire,
            satiety_signal=satiety,
            safety_assessment=safety,
            novelty=novelty,
        )

    def _determine_flavor_profile(
        self,
        dominant: TasteModality,
        aroma: float,
        gustatory_input: GustatoryInput,
    ) -> FlavorProfile:
        """Determine the overall flavor profile."""
        if gustatory_input.oral_irritation > 0.5:
            return FlavorProfile.SPICY_HOT
        if dominant == TasteModality.SWEET and aroma > 0.3:
            return FlavorProfile.SWEET_AROMATIC
        if dominant == TasteModality.UMAMI:
            return FlavorProfile.SAVORY
        if dominant == TasteModality.SOUR:
            return FlavorProfile.SOUR_TANGY
        if dominant == TasteModality.BITTER:
            return FlavorProfile.BITTER_COMPLEX
        if dominant == TasteModality.SWEET:
            return FlavorProfile.MILD
        if aroma > 0.5:
            return FlavorProfile.RICH
        return FlavorProfile.NEUTRAL

    def _compute_pleasure(
        self, palatability: PalatabilityAssessment, gustatory_input: GustatoryInput
    ) -> float:
        """Compute overall pleasure from taste experience."""
        pleasure = palatability.palatability_score
        # Hunger amplifies pleasure from food
        if gustatory_input.appetite_state == AppetiteState.HUNGRY:
            pleasure = min(1.0, pleasure * 1.3)
        return max(-1.0, min(1.0, pleasure))

    def _check_food_safety(
        self, taste_id: TasteIdentification, gustatory_input: GustatoryInput
    ) -> bool:
        """Check if taste indicates safety concern."""
        bitter = taste_id.modality_strengths.get("bitter", 0.0)
        sour = taste_id.modality_strengths.get("sour", 0.0)
        if bitter > 0.8:
            return True  # Potential toxin
        if sour > 0.9 and gustatory_input.overall_intensity > 0.8:
            return True  # Potential spoilage
        return False

    def _compute_appetite_modulation(
        self, palatability: PalatabilityAssessment, gustatory_input: GustatoryInput
    ) -> float:
        """Compute how the taste modulates appetite."""
        if gustatory_input.appetite_state == AppetiteState.NAUSEOUS:
            return -0.8
        mod = palatability.palatability_score * 0.5
        mod -= palatability.satiety_signal * 0.3
        return max(-1.0, min(1.0, mod))

    def _update_history(
        self, taste_id: TasteIdentification, flavor: FlavorIntegration
    ) -> None:
        """Update taste history."""
        self._taste_history.append(taste_id)
        if len(self._taste_history) > self._max_history:
            self._taste_history.pop(0)

        profile_key = flavor.flavor_profile.value
        current = self._known_flavors.get(profile_key, 0.0)
        self._known_flavors[profile_key] = min(1.0, current + 0.1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for serialization."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "processing_count": self._processing_count,
            "appetite_state": self._appetite_state.value,
            "satiety_level": round(self._satiety_level, 4),
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "known_flavors": len(self._known_flavors),
            "history_length": len(self._taste_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current form status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "operational": True,
            "processing_count": self._processing_count,
            "appetite_state": self._appetite_state.value,
            "satiety_level": round(self._satiety_level, 4),
            "known_flavors": len(self._known_flavors),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gustatory_interface() -> GustatoryConsciousnessInterface:
    """Create and return a gustatory consciousness interface."""
    return GustatoryConsciousnessInterface()


def create_simple_gustatory_input(
    sweet: float = 0.0,
    sour: float = 0.0,
    salty: float = 0.0,
    bitter: float = 0.0,
    umami: float = 0.0,
    intensity: float = 0.5,
    appetite: AppetiteState = AppetiteState.NEUTRAL,
) -> GustatoryInput:
    """Create a simple gustatory input for testing."""
    return GustatoryInput(
        taste_receptor_data=TasteReceptorData(
            modality_activations={
                "sweet": sweet,
                "sour": sour,
                "salty": salty,
                "bitter": bitter,
                "umami": umami,
            },
        ),
        overall_intensity=intensity,
        appetite_state=appetite,
    )


__all__ = [
    # Enums
    "TasteModality",
    "FlavorProfile",
    "TextureQuality",
    "PalatabilityLevel",
    "AppetiteState",
    # Input dataclasses
    "TasteReceptorData",
    "GustatoryInput",
    # Output dataclasses
    "TasteIdentification",
    "FlavorIntegration",
    "PalatabilityAssessment",
    "GustatoryOutput",
    # Main interface
    "GustatoryConsciousnessInterface",
    # Convenience functions
    "create_gustatory_interface",
    "create_simple_gustatory_input",
]
