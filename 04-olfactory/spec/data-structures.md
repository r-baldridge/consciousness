# Olfactory Consciousness System - Data Structures

**Document**: Data Structures Specification
**Form**: 04 - Olfactory Consciousness
**Category**: Technical Specification & Design
**Version**: 1.0
**Date**: 2025-09-26

## Executive Summary

This document defines comprehensive data structures for the Olfactory Consciousness System, including molecular detection representations, scent pattern models, memory association objects, emotional response structures, and cultural adaptation frameworks for rich, culturally-sensitive olfactory consciousness experiences.

## Core Data Structure Hierarchy

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum, IntEnum
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

# Base Classes and Enums
class OlfactoryModality(Enum):
    AIRBORNE = "airborne"           # Airborne chemical detection
    RETRONASAL = "retronasal"       # Through eating/drinking
    DIRECT = "direct"               # Direct contact with source
    TRACE = "trace"                 # Trace amounts/residual

class ScentCategory(Enum):
    FLORAL = "floral"
    FRUITY = "fruity"
    CITRUS = "citrus"
    WOODY = "woody"
    SPICY = "spicy"
    HERBACEOUS = "herbaceous"
    MARINE = "marine"
    GOURMAND = "gourmand"
    ANIMALIC = "animalic"
    AROMATIC = "aromatic"

class HedonicCategory(Enum):
    VERY_PLEASANT = "very_pleasant"     # +2
    PLEASANT = "pleasant"               # +1
    NEUTRAL = "neutral"                 # 0
    UNPLEASANT = "unpleasant"          # -1
    VERY_UNPLEASANT = "very_unpleasant" # -2

class CulturalRegion(Enum):
    WESTERN = "western"
    EAST_ASIAN = "east_asian"
    SOUTH_ASIAN = "south_asian"
    MIDDLE_EASTERN = "middle_eastern"
    AFRICAN = "african"
    LATIN_AMERICAN = "latin_american"
    MEDITERRANEAN = "mediterranean"
    NORDIC = "nordic"

class ConsciousnessClarity(IntEnum):
    SUBLIMINAL = 0          # Below conscious threshold
    BARELY_DETECTABLE = 1   # Just above threshold
    WEAK = 2               # Weak but conscious
    MODERATE = 3           # Clearly conscious
    STRONG = 4             # Strong consciousness
    OVERWHELMING = 5       # Overwhelming consciousness
```

## Molecular Detection Data Structures

### 1. Base Molecular Data Structure

```python
@dataclass
class BaseMolecularData:
    """Base class for all molecular detection data"""
    molecule_id: str
    detection_timestamp: datetime
    chemical_formula: str
    molecular_weight: float
    structural_representation: str  # SMILES notation
    detection_confidence: float     # 0.0-1.0
    data_quality_score: float      # 0.0-1.0
    sensor_source: str             # Source sensor/detector
    environmental_context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.detection_confidence <= 1.0):
            raise ValueError("Detection confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.data_quality_score <= 1.0):
            raise ValueError("Data quality score must be between 0.0 and 1.0")

@dataclass
class FunctionalGroup:
    """Chemical functional group representation"""
    group_name: str                 # "aldehyde", "ester", "terpene", etc.
    group_formula: str             # Chemical formula of group
    position_in_molecule: List[int] # Atom positions
    contribution_to_odor: float    # 0.0-1.0
    hedonic_influence: float       # -1.0 to 1.0

@dataclass
class MolecularProperties:
    """Comprehensive molecular properties"""
    volatility: float              # Vapor pressure related
    solubility_water: float       # Water solubility
    solubility_lipid: float       # Lipid solubility
    boiling_point: float          # °C
    melting_point: float          # °C
    stability: float              # Chemical stability 0.0-1.0
    reactivity: float             # Chemical reactivity 0.0-1.0
    polarity: float               # Molecular polarity 0.0-1.0
    hydrogen_bonding: bool        # Hydrogen bonding capability
    van_der_waals_volume: float   # Molecular volume
```

### 2. Chemical Detection Results

```python
@dataclass
class ConcentrationMeasurement:
    """Detailed concentration measurement"""
    value: float                   # Concentration value
    unit: str                     # "ppm", "ppb", "ppt", "ng/m³"
    measurement_method: str       # Detection method used
    accuracy_estimate: float      # Estimated accuracy ±%
    detection_limit: float        # Minimum detectable concentration
    saturation_limit: float       # Maximum measurable concentration
    temporal_stability: float     # Stability over time 0.0-1.0
    spatial_uniformity: float     # Spatial distribution uniformity 0.0-1.0
    interference_level: float     # Level of measurement interference 0.0-1.0

@dataclass
class ChemicalDetectionResult(BaseMolecularData):
    """Comprehensive chemical detection result"""
    # Basic molecular information
    common_name: str
    iupac_name: str
    cas_number: str               # Chemical Abstracts Service number
    functional_groups: List[FunctionalGroup]
    molecular_properties: MolecularProperties

    # Detection specifics
    concentration: ConcentrationMeasurement
    detection_method: str         # "GC-MS", "electronic_nose", etc.
    isotope_pattern: Optional[Dict[int, float]]  # Mass spectrum isotope pattern
    retention_time: Optional[float]  # Chromatographic retention time

    # Temporal characteristics
    detection_onset_time: Optional[datetime]
    peak_concentration_time: Optional[datetime]
    detection_offset_time: Optional[datetime]
    concentration_stability: float

    # Spatial characteristics
    detection_coordinates: Optional[Tuple[float, float, float]]  # 3D location
    spatial_distribution: Optional[np.ndarray]  # Spatial concentration map
    diffusion_characteristics: Dict[str, float]

    # Quality and reliability
    signal_to_noise_ratio: float
    baseline_interference: float
    cross_sensitivity: Dict[str, float]  # Interference from other molecules
    calibration_status: str       # "calibrated", "estimated", "uncalibrated"

    def calculate_odor_potential(self) -> float:
        """Calculate potential for odor perception"""
        # Consider volatility, concentration, and molecular properties
        volatility_factor = self.molecular_properties.volatility
        concentration_factor = min(self.concentration.value / 1e-9, 1.0)  # Normalize to ppb
        molecular_factor = len(self.functional_groups) / 10.0  # Functional group diversity

        return min(volatility_factor * concentration_factor * molecular_factor, 1.0)

    def get_dominant_functional_groups(self, threshold: float = 0.5) -> List[FunctionalGroup]:
        """Get functional groups with high odor contribution"""
        return [fg for fg in self.functional_groups if fg.contribution_to_odor >= threshold]
```

### 3. Chemical Mixture Analysis

```python
@dataclass
class MixtureComponent:
    """Individual component in a chemical mixture"""
    molecule: ChemicalDetectionResult
    relative_concentration: float  # Relative to total mixture
    contribution_to_overall_odor: float  # 0.0-1.0
    masking_effects: Dict[str, float]    # How this component masks others
    synergistic_effects: Dict[str, float]  # Synergistic interactions

@dataclass
class ChemicalMixtureAnalysis:
    """Analysis of complex chemical mixtures"""
    mixture_id: str
    detection_timestamp: datetime
    components: List[MixtureComponent]
    total_concentration: float
    mixture_complexity: float      # Number of significant components

    # Interaction effects
    masking_interactions: Dict[Tuple[str, str], float]  # (mask_id, masked_id) -> effect
    synergistic_interactions: Dict[Tuple[str, str], float]  # (mol1_id, mol2_id) -> effect
    emergent_properties: Dict[str, Any]  # Properties emerging from mixture

    # Stability and dynamics
    mixture_stability: float       # Temporal stability of mixture
    evaporation_profile: Dict[str, List[Tuple[int, float]]]  # Component evaporation over time
    diffusion_dynamics: Dict[str, float]  # Diffusion characteristics

    # Odor prediction
    predicted_odor_profile: Dict[str, float]  # Predicted scent characteristics
    hedonic_prediction: HedonicCategory
    intensity_prediction: float    # Predicted intensity 0.0-1.0

    def get_dominant_components(self, threshold: float = 0.1) -> List[MixtureComponent]:
        """Get components with significant contribution to overall odor"""
        return [comp for comp in self.components
                if comp.contribution_to_overall_odor >= threshold]

    def calculate_mixture_hedonic_value(self) -> float:
        """Calculate overall hedonic value of mixture"""
        weighted_hedonic = 0.0
        total_weight = 0.0

        for component in self.components:
            weight = component.contribution_to_overall_odor
            hedonic_contribution = sum(fg.hedonic_influence for fg in component.molecule.functional_groups)
            weighted_hedonic += weight * hedonic_contribution
            total_weight += weight

        return weighted_hedonic / total_weight if total_weight > 0 else 0.0
```

## Scent Recognition Data Structures

### 1. Scent Pattern Representations

```python
@dataclass
class ScentFingerprint:
    """Unique molecular fingerprint for scent identification"""
    fingerprint_id: str
    molecular_signature: Dict[str, float]  # molecule_id -> relative importance
    functional_group_pattern: Dict[str, float]  # functional_group -> weight
    concentration_ratios: Dict[str, float]  # Relative concentration patterns
    temporal_signature: List[Tuple[int, Dict[str, float]]]  # Time-based patterns

    # Pattern characteristics
    pattern_complexity: float      # 0.0-1.0
    pattern_uniqueness: float      # 0.0-1.0
    pattern_stability: float       # Temporal stability

    # Recognition metrics
    recognition_confidence: float  # Confidence in this fingerprint
    false_positive_rate: float    # Historical false positive rate
    false_negative_rate: float    # Historical false negative rate

@dataclass
class ScentClassification:
    """Multi-level scent classification"""
    primary_category: ScentCategory
    secondary_categories: List[ScentCategory]
    tertiary_descriptors: List[str]  # Fine-grained descriptors

    # Classification confidence
    category_confidence: Dict[ScentCategory, float]
    classification_ambiguity: float  # Uncertainty in classification

    # Cultural variations
    cultural_classifications: Dict[CulturalRegion, Dict[str, Any]]
    universal_descriptors: List[str]  # Culture-independent descriptors
    culture_specific_descriptors: Dict[CulturalRegion, List[str]]

@dataclass
class ScentRecognitionResult:
    """Comprehensive scent recognition result"""
    recognition_id: str
    timestamp: datetime
    input_mixture: ChemicalMixtureAnalysis

    # Primary identification
    identified_scent: str          # Main scent name
    scent_fingerprint: ScentFingerprint
    classification: ScentClassification
    recognition_confidence: float  # Overall confidence 0.0-1.0

    # Alternative identifications
    alternative_scents: List[Tuple[str, float]]  # (scent_name, confidence)
    ambiguous_components: List[str]  # Components with uncertain identification

    # Scent characteristics
    intensity_estimate: float      # Perceived intensity 0.0-1.0
    complexity_assessment: float   # Scent complexity 0.0-1.0
    familiarity_score: float      # How familiar this scent is 0.0-1.0
    novelty_score: float          # How novel/unusual this scent is 0.0-1.0

    # Quality and reliability
    signal_quality: float         # Quality of input signal
    recognition_method: str       # Method used for recognition
    database_match_score: float  # Match score against database
    learning_contribution: bool  # Whether this helps improve recognition

    def get_best_match(self) -> Tuple[str, float]:
        """Get the best matching scent and confidence"""
        if self.recognition_confidence > 0.5:
            return self.identified_scent, self.recognition_confidence
        elif self.alternative_scents:
            return max(self.alternative_scents, key=lambda x: x[1])
        else:
            return "unknown", 0.0

    def is_novel_scent(self, novelty_threshold: float = 0.7) -> bool:
        """Determine if this represents a novel scent"""
        return self.novelty_score >= novelty_threshold
```

## Memory Integration Data Structures

### 1. Olfactory Memory Representations

```python
@dataclass
class AutobiographicalMemory:
    """Autobiographical memory triggered by scent"""
    memory_id: str
    creation_timestamp: datetime
    trigger_scent: str

    # Memory content
    memory_description: str
    life_period: str              # "childhood", "adolescence", "early_adulthood", etc.
    approximate_age: Optional[int]  # Age when memory was formed
    location: Optional[str]       # Where the memory took place
    people_involved: List[str]    # People present in the memory
    activities: List[str]         # Activities in the memory

    # Emotional content
    primary_emotions: Dict[str, float]  # emotion -> intensity
    emotional_valence: float      # -1.0 (negative) to 1.0 (positive)
    emotional_arousal: float      # 0.0 (calm) to 1.0 (intense)

    # Memory characteristics
    vividness: float              # How vivid/clear the memory is 0.0-1.0
    confidence: float             # Confidence in memory accuracy 0.0-1.0
    importance: float             # Personal importance of memory 0.0-1.0
    frequency_recalled: int       # How often this memory is recalled

    # Sensory details
    visual_details: Dict[str, str]  # Visual aspects of memory
    auditory_details: Dict[str, str]  # Sounds in memory
    tactile_details: Dict[str, str]   # Touch sensations in memory
    other_scents: List[str]           # Other scents present in memory

    # Contextual information
    season: Optional[str]         # Season when memory occurred
    weather: Optional[str]        # Weather conditions
    time_of_day: Optional[str]    # Time of day
    social_context: str           # "family", "friends", "alone", etc.

@dataclass
class SemanticScentKnowledge:
    """Semantic knowledge about scents"""
    knowledge_id: str
    scent_name: str

    # Basic information
    description: str
    origin: str                   # Natural/synthetic source
    typical_contexts: List[str]   # Where this scent is typically encountered
    associated_objects: List[str] # Objects associated with this scent
    associated_activities: List[str]  # Activities associated with scent

    # Chemical knowledge
    key_molecules: List[str]      # Main chemical components
    molecular_families: List[str] # Chemical families represented

    # Cultural knowledge
    cultural_meanings: Dict[CulturalRegion, Dict[str, Any]]
    religious_significance: Dict[str, str]  # Religious associations
    historical_uses: List[str]   # Historical applications
    symbolic_meanings: List[str] # Symbolic associations

    # Practical knowledge
    commercial_uses: List[str]    # Commercial applications
    therapeutic_properties: Dict[str, float]  # Therapeutic effects
    safety_information: Dict[str, str]  # Safety considerations
    preparation_methods: List[str]  # How it's prepared/extracted

    # Hedonic knowledge
    typical_hedonic_response: HedonicCategory
    individual_variation: float  # How much individual variation exists
    age_preferences: Dict[str, float]  # Preferences by age group
    gender_preferences: Dict[str, float]  # Preferences by gender

@dataclass
class ScentMemoryAssociation:
    """Association between scent and memory"""
    association_id: str
    scent_identifier: str
    memory: Union[AutobiographicalMemory, SemanticScentKnowledge]

    # Association strength
    association_strength: float   # 0.0-1.0
    activation_threshold: float   # Minimum scent intensity to trigger
    activation_latency: float     # Time to activate memory (seconds)

    # Temporal characteristics
    first_formed: datetime        # When association was first formed
    last_activated: datetime      # When last activated
    activation_count: int         # Number of times activated

    # Context dependency
    context_dependence: float     # How context-dependent activation is 0.0-1.0
    facilitating_contexts: List[str]  # Contexts that enhance activation
    inhibiting_contexts: List[str]    # Contexts that suppress activation

    # Learning characteristics
    consolidation_level: float    # How well-consolidated 0.0-1.0
    interference_susceptibility: float  # Susceptibility to interference 0.0-1.0
    forgetting_rate: float        # Rate of memory decay 0.0-1.0

    def calculate_activation_probability(self, scent_intensity: float,
                                       current_context: Dict[str, Any]) -> float:
        """Calculate probability of memory activation"""
        # Base probability from scent intensity
        intensity_factor = max(0, scent_intensity - self.activation_threshold)

        # Context modulation
        context_factor = 1.0
        if current_context:
            context_match = self._assess_context_match(current_context)
            context_factor = 0.5 + 0.5 * context_match  # Range 0.5-1.0

        # Association strength factor
        strength_factor = self.association_strength

        return min(intensity_factor * context_factor * strength_factor, 1.0)

    def _assess_context_match(self, current_context: Dict[str, Any]) -> float:
        """Assess how well current context matches facilitating contexts"""
        if not self.facilitating_contexts:
            return 1.0

        matches = 0
        for context_key, context_value in current_context.items():
            if any(context_value in fc for fc in self.facilitating_contexts):
                matches += 1

        return matches / len(self.facilitating_contexts)
```

### 2. Memory Network Structures

```python
@dataclass
class MemoryNetwork:
    """Network of interconnected olfactory memories"""
    network_id: str
    central_scent: str
    connected_memories: List[ScentMemoryAssociation]

    # Network topology
    connection_strengths: Dict[Tuple[str, str], float]  # (memory1, memory2) -> strength
    clustering_coefficient: float  # How clustered the network is
    path_lengths: Dict[Tuple[str, str], int]  # Shortest paths between memories

    # Network dynamics
    activation_cascade: Dict[str, List[str]]  # memory -> activated_memories
    inhibition_patterns: Dict[str, List[str]]  # memory -> inhibited_memories
    network_stability: float      # Stability of network structure 0.0-1.0

    # Temporal evolution
    formation_history: List[Tuple[datetime, str]]  # Network formation timeline
    modification_history: List[Tuple[datetime, Dict[str, Any]]]  # Changes over time

    def calculate_network_activation(self, initial_activation: Dict[str, float]) -> Dict[str, float]:
        """Calculate network-wide activation pattern"""
        activation = initial_activation.copy()

        # Iterate activation spreading
        for _ in range(5):  # Max 5 iterations
            new_activation = activation.copy()

            for memory_id, current_activation in activation.items():
                if current_activation > 0.1:  # Only spread from significantly activated
                    for connected_memory in self.activation_cascade.get(memory_id, []):
                        connection_strength = self.connection_strengths.get(
                            (memory_id, connected_memory), 0.0
                        )
                        spread_activation = current_activation * connection_strength * 0.5
                        new_activation[connected_memory] = min(
                            new_activation.get(connected_memory, 0.0) + spread_activation, 1.0
                        )

            activation = new_activation

        return activation

    def find_memory_paths(self, source_memory: str, target_memory: str) -> List[List[str]]:
        """Find paths between memories in the network"""
        # Simplified path finding - in practice, use proper graph algorithms
        paths = []
        if source_memory in self.activation_cascade:
            for intermediate in self.activation_cascade[source_memory]:
                if intermediate == target_memory:
                    paths.append([source_memory, target_memory])
                elif intermediate in self.activation_cascade:
                    if target_memory in self.activation_cascade[intermediate]:
                        paths.append([source_memory, intermediate, target_memory])
        return paths
```

## Emotional Response Data Structures

### 1. Hedonic Evaluation

```python
@dataclass
class HedonicEvaluation:
    """Comprehensive hedonic evaluation of olfactory stimulus"""
    evaluation_id: str
    scent_identifier: str
    timestamp: datetime

    # Basic hedonic response
    pleasantness: float           # -1.0 (very unpleasant) to 1.0 (very pleasant)
    intensity: float              # 0.0 (very weak) to 1.0 (very strong)
    hedonic_category: HedonicCategory

    # Dimensional analysis
    arousal: float                # 0.0 (calming) to 1.0 (stimulating)
    dominance: float              # 0.0 (submissive) to 1.0 (dominant)
    approach_avoidance: float     # -1.0 (avoid) to 1.0 (approach)

    # Temporal dynamics
    initial_response: float       # Immediate hedonic response
    sustained_response: float     # Response after adaptation
    offset_response: float        # Response when scent disappears
    hedonic_adaptation_rate: float  # How quickly hedonic response adapts

    # Context modulation
    baseline_mood: Dict[str, float]  # Mood before scent exposure
    mood_change: Dict[str, float]    # Change in mood due to scent
    context_enhancement: float    # How context enhances/reduces response
    expectation_effect: float    # Effect of prior expectations

    # Individual factors
    personal_significance: float  # Personal meaning of this scent 0.0-1.0
    familiarity_effect: float    # Effect of familiarity on hedonic response
    cultural_influence: float    # Influence of cultural background

    # Physiological correlates
    autonomic_responses: Dict[str, float]  # HR, BP, skin conductance changes
    facial_expression_pattern: Dict[str, float]  # Facial muscle activations
    approach_behavior_tendency: float  # Tendency to approach/avoid

@dataclass
class EmotionalResponse:
    """Comprehensive emotional response to olfactory stimulus"""
    response_id: str
    trigger_scent: str
    timestamp: datetime

    # Basic emotions (using Ekman's basic emotions + additional)
    joy: float                    # 0.0-1.0
    sadness: float               # 0.0-1.0
    anger: float                 # 0.0-1.0
    fear: float                  # 0.0-1.0
    surprise: float              # 0.0-1.0
    disgust: float               # 0.0-1.0

    # Complex emotions
    nostalgia: float             # 0.0-1.0
    comfort: float               # 0.0-1.0
    excitement: float            # 0.0-1.0
    anxiety: float               # 0.0-1.0
    love: float                  # 0.0-1.0
    contempt: float              # 0.0-1.0

    # Emotional characteristics
    emotional_intensity: float   # Overall emotional intensity 0.0-1.0
    emotional_clarity: float     # How clear/distinct the emotions are 0.0-1.0
    emotional_complexity: int    # Number of distinct emotions > 0.3
    dominant_emotion: str        # Most prominent emotion

    # Cognitive components
    emotional_attribution: str   # What the person attributes emotion to
    emotional_control: float     # Sense of control over emotion 0.0-1.0
    emotional_appropriateness: float  # How appropriate emotion feels 0.0-1.0

    # Social emotions (if relevant)
    empathy_activation: float    # If scent activates empathetic responses
    social_connection: float     # Feeling of social connection
    social_memories: List[str]   # Activated social memories

    # Temporal pattern
    emotion_onset_latency: float # Time to emotion onset (seconds)
    emotion_peak_time: float     # Time to peak emotion (seconds)
    emotion_duration: float      # Duration of emotional response (seconds)
    emotion_offset_pattern: str  # "gradual", "abrupt", "cyclical"

    def get_dominant_emotions(self, threshold: float = 0.3) -> Dict[str, float]:
        """Get emotions above threshold intensity"""
        emotions = {
            'joy': self.joy,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'surprise': self.surprise,
            'disgust': self.disgust,
            'nostalgia': self.nostalgia,
            'comfort': self.comfort,
            'excitement': self.excitement,
            'anxiety': self.anxiety,
            'love': self.love,
            'contempt': self.contempt
        }
        return {emotion: intensity for emotion, intensity in emotions.items()
                if intensity >= threshold}

    def calculate_valence(self) -> float:
        """Calculate overall emotional valence"""
        positive_emotions = self.joy + self.love + self.comfort + self.excitement
        negative_emotions = self.sadness + self.anger + self.fear + self.disgust + self.anxiety + self.contempt

        # Nostalgia and surprise are neutral to slightly positive
        mixed_emotions = self.nostalgia * 0.3 + self.surprise * 0.1

        total_positive = positive_emotions + mixed_emotions
        total_negative = negative_emotions

        if total_positive + total_negative == 0:
            return 0.0

        return (total_positive - total_negative) / (total_positive + total_negative)
```

## Cultural Adaptation Data Structures

### 1. Cultural Context Representation

```python
@dataclass
class CulturalProfile:
    """Comprehensive cultural profile for scent adaptation"""
    profile_id: str
    cultural_region: CulturalRegion
    subculture: Optional[str]     # More specific cultural group

    # Demographic information
    age_group: str               # "child", "adolescent", "young_adult", etc.
    gender_identity: str
    socioeconomic_background: str
    education_level: str
    urbanity: str                # "urban", "suburban", "rural"

    # Cultural factors
    religious_background: List[str]
    dietary_restrictions: List[str]  # "vegetarian", "halal", "kosher", etc.
    linguistic_background: List[str]
    migration_history: Optional[str]  # If applicable

    # Scent-related cultural factors
    traditional_scents: List[str]     # Culturally significant scents
    taboo_scents: List[str]          # Culturally inappropriate scents
    ritual_scents: List[str]         # Scents used in cultural/religious rituals
    medicinal_scents: List[str]      # Traditional medicinal scents
    culinary_scents: List[str]       # Food-related cultural scents

    # Adaptation preferences
    cultural_sensitivity_level: float  # How sensitive to cultural adaptation 0.0-1.0
    openness_to_foreign_scents: float  # Willingness to experience foreign scents 0.0-1.0
    preference_for_traditional: float  # Preference for culturally familiar scents 0.0-1.0

@dataclass
class CulturalScentMapping:
    """Mapping of scents to cultural meanings and associations"""
    mapping_id: str
    scent_identifier: str
    cultural_region: CulturalRegion

    # Cultural meanings
    primary_cultural_meaning: str     # Main cultural association
    secondary_meanings: List[str]     # Additional cultural meanings
    symbolic_significance: Dict[str, float]  # symbol -> significance level

    # Usage contexts
    traditional_uses: List[str]       # How traditionally used
    modern_uses: List[str]           # How used in modern context
    ceremonial_uses: List[str]       # Ceremonial/ritual uses
    domestic_uses: List[str]         # Household uses
    medicinal_uses: List[str]        # Traditional medicine uses

    # Social associations
    social_class_associations: Dict[str, float]  # "working", "middle", "upper"
    gender_associations: Dict[str, float]        # "masculine", "feminine", "neutral"
    age_associations: Dict[str, float]           # Age group associations
    occasion_associations: List[str]             # Appropriate occasions

    # Emotional/psychological associations
    cultural_emotions: Dict[str, float]          # Culture-specific emotional responses
    psychological_effects: Dict[str, float]      # Cultural psychological associations
    memory_triggers: List[str]                   # Common cultural memories triggered

    # Appropriateness and taboos
    appropriateness_level: float      # How culturally appropriate 0.0-1.0
    taboo_level: float               # How taboo in this culture 0.0-1.0
    sensitivity_warnings: List[str]   # Specific sensitivity considerations
    alternative_scents: List[str]     # Culturally appropriate alternatives

    # Regional variations
    regional_variations: Dict[str, Dict[str, Any]]  # Variations within culture
    historical_changes: List[Tuple[str, Dict[str, Any]]]  # Changes over time

@dataclass
class CulturalAdaptationStrategy:
    """Strategy for adapting scent presentation to cultural context"""
    strategy_id: str
    source_scent: str
    target_cultural_profile: CulturalProfile

    # Adaptation approach
    adaptation_method: str           # "direct", "gradual", "alternative", "educational"
    adaptation_intensity: float      # How much adaptation needed 0.0-1.0

    # Presentation modifications
    intensity_adjustment: float      # Adjustment to scent intensity ±1.0
    duration_adjustment: float       # Adjustment to exposure duration ±1.0
    context_modifications: Dict[str, str]  # Changes to presentation context

    # Educational components
    cultural_education_content: Optional[str]  # Educational content to provide
    cross_cultural_comparison: Optional[Dict[str, Any]]  # Compare across cultures
    historical_context: Optional[str]  # Historical background

    # Safety and sensitivity
    sensitivity_precautions: List[str]  # Precautions to take
    opt_out_mechanisms: List[str]       # Ways user can opt out
    feedback_collection: Dict[str, str]  # How to collect cultural feedback

    # Effectiveness tracking
    success_metrics: Dict[str, str]     # How to measure adaptation success
    fallback_strategies: List[str]      # Backup strategies if adaptation fails

    def apply_adaptation(self, base_scent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cultural adaptation to base scent data"""
        adapted_data = base_scent_data.copy()

        # Apply intensity adjustment
        if 'intensity' in adapted_data:
            adapted_data['intensity'] *= (1.0 + self.intensity_adjustment)
            adapted_data['intensity'] = max(0.0, min(1.0, adapted_data['intensity']))

        # Apply duration adjustment
        if 'duration' in adapted_data:
            adapted_data['duration'] *= (1.0 + self.duration_adjustment)

        # Add cultural context
        adapted_data['cultural_context'] = {
            'target_culture': self.target_cultural_profile.cultural_region,
            'adaptation_method': self.adaptation_method,
            'cultural_appropriateness': self._assess_appropriateness(),
            'educational_content': self.cultural_education_content
        }

        # Add sensitivity information
        adapted_data['sensitivity_info'] = {
            'precautions': self.sensitivity_precautions,
            'opt_out_available': bool(self.opt_out_mechanisms),
            'cultural_sensitivity_level': self.target_cultural_profile.cultural_sensitivity_level
        }

        return adapted_data

    def _assess_appropriateness(self) -> float:
        """Assess cultural appropriateness of adapted scent"""
        # This would involve complex cultural knowledge processing
        # Simplified version here
        base_appropriateness = 0.8  # Assume most adaptations start reasonably appropriate

        # Adjust based on adaptation intensity
        appropriateness = base_appropriateness * (1.0 - self.adaptation_intensity * 0.2)

        return max(0.0, min(1.0, appropriateness))
```

This comprehensive data structure specification provides the foundation for implementing sophisticated, culturally-sensitive olfactory consciousness with rich molecular detection, pattern recognition, memory integration, emotional processing, and cultural adaptation capabilities.