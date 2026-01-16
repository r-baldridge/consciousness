# Gustatory Consciousness System - Data Structures

**Document**: Data Structures Specification
**Form**: 05 - Gustatory Consciousness
**Category**: Technical Specification & Design
**Version**: 1.0
**Date**: 2025-09-27

## Executive Summary

This document defines comprehensive data structures for the Gustatory Consciousness System, establishing standardized formats for taste detection, flavor integration, memory association, cultural adaptation, and conscious experience representation. These structures ensure consistent data handling while supporting biological authenticity, cultural sensitivity, and phenomenological richness across all gustatory consciousness operations.

## Core Data Structure Framework

### Base Data Types and Enumerations

#### Fundamental Enumerations
```python
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import uuid

class BasicTasteType(Enum):
    """Five basic taste modalities"""
    SWEET = "sweet"
    SOUR = "sour"
    SALTY = "salty"
    BITTER = "bitter"
    UMAMI = "umami"

class FlavorComplexity(IntEnum):
    """Flavor complexity levels"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4
    EXTREMELY_COMPLEX = 5

class CulturalSensitivityLevel(Enum):
    """Cultural sensitivity handling levels"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    EXPERT = "expert"
    SACRED = "sacred"

class MemoryType(Enum):
    """Types of gustatory memories"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CULTURAL = "cultural"
    AUTOBIOGRAPHICAL = "autobiographical"
    PROCEDURAL = "procedural"

class DietaryRestriction(Enum):
    """Dietary restriction categories"""
    HALAL = "halal"
    KOSHER = "kosher"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DIABETIC = "diabetic"
    ALLERGENIC = "allergenic"
    CUSTOM = "custom"

class ConsciousnessIntensity(Enum):
    """Consciousness experience intensity levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    PEAK = "peak"
```

### Base Data Structures

#### Universal Base Classes
```python
@dataclass
class BaseGustatoryData:
    """Base class for all gustatory consciousness data structures"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    cultural_context: Optional[str] = None
    confidence_score: float = 1.0  # 0.0-1.0
    validation_status: bool = True

@dataclass
class CulturalContext:
    """Cultural context information"""
    primary_culture: str
    cultural_background: List[str] = field(default_factory=list)
    religious_context: List[str] = field(default_factory=list)
    regional_preferences: Optional[str] = None
    dietary_restrictions: List[DietaryRestriction] = field(default_factory=list)
    cultural_sensitivity_level: CulturalSensitivityLevel = CulturalSensitivityLevel.ENHANCED
    traditional_knowledge_access: bool = True
    cross_cultural_adaptation: bool = True
```

## Taste Detection Data Structures

### Chemical and Molecular Data

#### Taste Compound Representation
```python
@dataclass
class TasteCompound(BaseGustatoryData):
    """Individual taste compound data structure"""
    compound_name: str
    molecular_formula: str
    molecular_weight: float
    smiles_notation: str  # Simplified molecular input line entry system

    # Taste characteristics
    primary_taste: BasicTasteType
    taste_intensity: float  # 0.0-1.0
    taste_threshold: float  # detection threshold concentration
    concentration: float  # actual concentration in sample

    # Biological properties
    receptor_binding_affinity: Dict[str, float]  # receptor_type: affinity
    solubility: float  # water solubility
    volatility: float  # vapor pressure indicator

    # Cultural and contextual information
    cultural_significance: Optional[str] = None
    traditional_use: Optional[str] = None
    safety_classification: str = "safe"
    allergen_potential: bool = False

@dataclass
class BasicTasteProfile(BaseGustatoryData):
    """Profile of five basic tastes"""
    sweetness: float  # 0.0-1.0 intensity
    sourness: float   # 0.0-1.0 intensity
    saltiness: float  # 0.0-1.0 intensity
    bitterness: float # 0.0-1.0 intensity
    umami: float      # 0.0-1.0 intensity

    # Quality metrics
    detection_accuracy: float = 0.0  # 0.0-1.0
    identification_confidence: float = 0.0  # 0.0-1.0

    # Individual variation factors
    individual_sensitivity_factors: Dict[BasicTasteType, float] = field(default_factory=dict)
    supertaster_adjustments: Dict[BasicTasteType, float] = field(default_factory=dict)

    # Interaction effects
    taste_interactions: Dict[str, float] = field(default_factory=dict)  # interaction_type: strength
```

#### Taste Receptor Response Data
```python
@dataclass
class TasteReceptorResponse(BaseGustatoryData):
    """Taste receptor activation response"""
    receptor_type: str  # T1R1, T1R2, T1R3, T2R variants
    receptor_subunit_composition: List[str] = field(default_factory=list)

    # Response characteristics
    activation_strength: float  # 0.0-1.0
    response_onset_time: float  # milliseconds
    response_peak_time: float   # milliseconds
    response_decay_time: float  # milliseconds

    # Adaptation and plasticity
    adaptation_state: float = 0.0  # 0.0-1.0, 0=fully adapted
    habituation_strength: float = 0.0  # 0.0-1.0
    sensitization_level: float = 0.0   # 0.0-1.0

    # Individual variation
    genetic_variation_factor: float = 1.0  # multiplier for genetic effects
    age_related_factor: float = 1.0        # age-related sensitivity changes
    sex_related_factor: float = 1.0        # sex-related sensitivity differences

    # Cross-receptor interactions
    cross_sensitivity: Dict[str, float] = field(default_factory=dict)
    inhibitory_effects: Dict[str, float] = field(default_factory=dict)
    facilitatory_effects: Dict[str, float] = field(default_factory=dict)

@dataclass
class TasteInteractionProfile(BaseGustatoryData):
    """Profile of taste interactions and modulations"""

    # Enhancement interactions
    taste_enhancements: List[Tuple[BasicTasteType, BasicTasteType, float]] = field(default_factory=list)

    # Suppression interactions
    taste_suppressions: List[Tuple[BasicTasteType, BasicTasteType, float]] = field(default_factory=list)

    # Masking effects
    taste_masking: List[Tuple[BasicTasteType, BasicTasteType, float]] = field(default_factory=list)

    # Temporal interactions
    sequential_effects: Dict[str, float] = field(default_factory=dict)
    simultaneous_effects: Dict[str, float] = field(default_factory=dict)

    # Concentration-dependent effects
    concentration_thresholds: Dict[str, float] = field(default_factory=dict)
    saturation_points: Dict[str, float] = field(default_factory=dict)

    # Cultural interaction patterns
    cultural_interaction_preferences: Dict[str, List[str]] = field(default_factory=dict)
```

## Flavor Integration Data Structures

### Cross-Modal Integration Data

#### Flavor Synthesis Data
```python
@dataclass
class FlavorProfile(BaseGustatoryData):
    """Comprehensive flavor profile data structure"""

    # Core flavor components
    taste_component: BasicTasteProfile
    olfactory_component: Dict[str, Any]  # From olfactory consciousness system
    trigeminal_component: 'TrigeminalSensationProfile'

    # Integrated flavor characteristics
    flavor_complexity: FlavorComplexity
    flavor_harmony: float  # 0.0-1.0, balance and coherence
    flavor_intensity: float  # 0.0-1.0, overall intensity
    flavor_persistence: float  # 0.0-1.0, lingering quality

    # Temporal flavor development
    temporal_profile: 'TemporalFlavorProfile'

    # Cultural and contextual classification
    flavor_category: str  # cultural flavor classification
    cuisine_classification: List[str] = field(default_factory=list)
    preparation_method_influence: Dict[str, float] = field(default_factory=dict)

    # Individual and cultural adaptation
    cultural_authenticity_score: float = 0.0  # 0.0-1.0
    personal_preference_alignment: float = 0.0  # 0.0-1.0
    novelty_score: float = 0.0  # 0.0-1.0, how novel the flavor is

@dataclass
class TrigeminalSensationProfile(BaseGustatoryData):
    """Trigeminal sensation data (temperature, texture, chemical irritation)"""

    # Temperature sensations
    temperature_celsius: float
    thermal_sensation_intensity: float  # 0.0-1.0
    thermal_preference_alignment: float  # 0.0-1.0

    # Texture sensations
    texture_descriptors: List[str] = field(default_factory=list)
    texture_intensity: float = 0.0  # 0.0-1.0
    mouthfeel_characteristics: Dict[str, float] = field(default_factory=dict)

    # Chemical irritation
    capsaicin_level: float = 0.0  # spiciness level
    carbonation_level: float = 0.0  # carbonation intensity
    astringency_level: float = 0.0  # astringent sensation
    cooling_sensation: float = 0.0  # menthol-like cooling
    warming_sensation: float = 0.0  # warming spices

    # Pain-pleasure balance
    nociceptive_intensity: float = 0.0  # pain/irritation level
    hedonic_response: float = 0.0  # pleasure response to sensation
    tolerance_level: float = 0.0  # individual tolerance

    # Cultural associations
    cultural_spice_tolerance: float = 0.0  # cultural norm alignment
    traditional_preparation_markers: List[str] = field(default_factory=list)

@dataclass
class TemporalFlavorProfile(BaseGustatoryData):
    """Temporal development of flavor experience"""

    # Phase timing
    initial_perception_duration: float  # milliseconds
    development_phase_duration: float   # milliseconds
    peak_intensity_duration: float      # milliseconds
    decline_phase_duration: float       # milliseconds
    aftertaste_duration: float         # milliseconds

    # Flavor component evolution
    taste_evolution: Dict[BasicTasteType, List[Tuple[float, float]]] = field(default_factory=dict)  # (time, intensity)
    aroma_evolution: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    trigeminal_evolution: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

    # Interaction development
    interaction_timeline: List[Tuple[float, str, float]] = field(default_factory=list)  # (time, interaction_type, strength)

    # Quality characteristics
    temporal_coherence: float = 0.0  # 0.0-1.0, smooth progression
    development_naturalness: float = 0.0  # 0.0-1.0, realistic development
    aftertaste_quality: float = 0.0  # 0.0-1.0, aftertaste pleasantness
```

### Retronasal Integration Data

#### Taste-Smell Binding Data
```python
@dataclass
class RetronasalIntegrationResult(BaseGustatoryData):
    """Result of retronasal olfaction integration with taste"""

    # Binding characteristics
    binding_strength: float  # 0.0-1.0
    spatial_coherence: float  # 0.0-1.0
    temporal_synchronization: float  # 0.0-1.0

    # Enhancement effects
    flavor_enhancement_factor: float  # multiplier for overall flavor intensity
    taste_smell_synergy: Dict[str, float] = field(default_factory=dict)  # specific synergistic combinations

    # Integration quality
    integration_naturalness: float = 0.0  # 0.0-1.0
    perceptual_unity: float = 0.0  # 0.0-1.0, unified perception strength
    cross_modal_consistency: float = 0.0  # 0.0-1.0

    # Individual and cultural factors
    individual_integration_ability: float = 1.0  # individual variation factor
    cultural_flavor_expectation_match: float = 0.0  # 0.0-1.0

    # Plasticity and learning
    integration_learning_state: float = 0.0  # 0.0-1.0, learned integration strength
    adaptation_history_influence: float = 0.0  # influence of past experiences
```

## Memory Integration Data Structures

### Gustatory Memory Data

#### Flavor Memory Objects
```python
@dataclass
class FlavorMemory(BaseGustatoryData):
    """Flavor-associated memory object"""

    # Memory identification
    memory_type: MemoryType
    memory_strength: float  # 0.0-1.0
    memory_vividness: float  # 0.0-1.0
    emotional_valence: float  # -1.0 to 1.0

    # Memory content
    memory_description: str
    associated_emotions: List[str] = field(default_factory=list)
    sensory_details: Dict[str, Any] = field(default_factory=dict)

    # Contextual information
    temporal_context: Optional[datetime] = None
    spatial_context: Optional[str] = None
    social_context: List[str] = field(default_factory=list)
    cultural_context_memory: str = ""

    # Personal significance
    personal_relevance: float = 0.0  # 0.0-1.0
    identity_significance: float = 0.0  # 0.0-1.0
    family_tradition_connection: float = 0.0  # 0.0-1.0

    # Memory formation factors
    formation_age: Optional[int] = None  # age when memory was formed
    repetition_frequency: int = 0  # how often experience was repeated
    emotional_intensity_at_formation: float = 0.0  # emotional state during formation

    # Privacy and access control
    privacy_level: str = "personal"  # public, personal, intimate
    sharing_permissions: Dict[str, bool] = field(default_factory=dict)

    # Cultural memory attributes (for cultural memory type)
    cultural_tradition_type: Optional[str] = None
    ritual_significance: Optional[str] = None
    symbolic_meaning: Optional[str] = None
    intergenerational_transmission: bool = False

@dataclass
class AutobiographicalFlavorMemory(FlavorMemory):
    """Specialized autobiographical flavor memory"""

    # Life period classification
    life_period: str  # childhood, adolescence, young_adult, etc.
    developmental_significance: float = 0.0  # 0.0-1.0

    # Family and social connections
    family_members_involved: List[str] = field(default_factory=list)
    family_traditions_learned: List[str] = field(default_factory=list)
    cultural_identity_formation: float = 0.0  # 0.0-1.0

    # Formative experience characteristics
    first_experience: bool = False
    learning_experience: bool = False
    milestone_association: Optional[str] = None

    # Memory enhancement through flavor
    vividness_enhancement_factor: float = 1.0  # how much flavor enhances memory vividness
    detail_richness_multiplier: float = 1.0    # detail enhancement through flavor cues
    emotional_amplification_factor: float = 1.0  # emotional intensity enhancement

@dataclass
class CulturalFlavorMemory(FlavorMemory):
    """Cultural flavor memory and knowledge"""

    # Cultural knowledge components
    traditional_preparation_methods: List[str] = field(default_factory=list)
    seasonal_associations: List[str] = field(default_factory=list)
    ceremonial_uses: List[str] = field(default_factory=list)

    # Cultural transmission
    knowledge_source: str = ""  # how knowledge was acquired
    authority_level: float = 0.0  # 0.0-1.0, authority of knowledge source
    consensus_level: float = 0.0  # 0.0-1.0, cultural consensus on information

    # Regional and historical context
    geographic_origin: Optional[str] = None
    historical_period: Optional[str] = None
    migration_adaptations: List[str] = field(default_factory=list)

    # Contemporary relevance
    modern_practice_frequency: float = 0.0  # 0.0-1.0
    adaptation_to_modern_context: float = 0.0  # 0.0-1.0
    preservation_priority: float = 0.0  # 0.0-1.0
```

### Memory Association Data

#### Memory-Flavor Binding
```python
@dataclass
class MemoryFlavorAssociation(BaseGustatoryData):
    """Association between flavors and memories"""

    # Association characteristics
    association_strength: float  # 0.0-1.0
    association_type: str  # direct, indirect, learned, cultural
    formation_mechanism: str  # how association was formed

    # Flavor trigger specificity
    flavor_specificity: float = 0.0  # 0.0-1.0, how specific flavor must be
    component_specificity: Dict[str, float] = field(default_factory=dict)  # specificity per flavor component
    context_dependency: float = 0.0  # 0.0-1.0, context requirement for activation

    # Memory retrieval characteristics
    retrieval_latency: float = 0.0  # milliseconds for memory activation
    retrieval_accuracy: float = 0.0  # 0.0-1.0
    retrieval_completeness: float = 0.0  # 0.0-1.0

    # Emotional and phenomenological aspects
    emotional_enhancement: float = 0.0  # emotional intensity enhancement
    vividness_enhancement: float = 0.0  # memory vividness enhancement
    phenomenological_richness: float = 0.0  # 0.0-1.0

    # Cultural and individual factors
    cultural_universality: float = 0.0  # 0.0-1.0, how universal the association is
    individual_uniqueness: float = 0.0  # 0.0-1.0, how unique to individual

    # Learning and plasticity
    association_plasticity: float = 0.0  # 0.0-1.0, how modifiable
    reinforcement_history: List[datetime] = field(default_factory=list)
    extinction_resistance: float = 0.0  # 0.0-1.0
```

## Cultural Adaptation Data Structures

### Cultural Context and Knowledge

#### Cultural Food Knowledge Base
```python
@dataclass
class CulturalFoodKnowledge(BaseGustatoryData):
    """Comprehensive cultural food knowledge structure"""

    # Cultural identification
    culture_identifier: str
    cultural_hierarchy: List[str] = field(default_factory=list)  # continent, country, region, etc.

    # Food tradition components
    traditional_foods: Dict[str, Any] = field(default_factory=dict)
    preparation_methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cooking_techniques: List[str] = field(default_factory=list)
    ingredient_traditions: Dict[str, str] = field(default_factory=dict)

    # Cultural food practices
    meal_patterns: Dict[str, Any] = field(default_factory=dict)
    eating_etiquette: List[str] = field(default_factory=list)
    food_sharing_customs: List[str] = field(default_factory=list)
    hospitality_traditions: List[str] = field(default_factory=list)

    # Religious and spiritual aspects
    religious_dietary_laws: List[DietaryRestriction] = field(default_factory=list)
    sacred_foods: List[str] = field(default_factory=list)
    ceremonial_foods: Dict[str, str] = field(default_factory=dict)
    fasting_traditions: List[str] = field(default_factory=list)

    # Seasonal and temporal aspects
    seasonal_foods: Dict[str, List[str]] = field(default_factory=dict)
    festival_foods: Dict[str, List[str]] = field(default_factory=dict)
    lifecycle_foods: Dict[str, List[str]] = field(default_factory=dict)  # birth, coming of age, marriage, etc.

    # Social and economic aspects
    status_foods: Dict[str, List[str]] = field(default_factory=dict)
    everyday_foods: List[str] = field(default_factory=list)
    comfort_foods: List[str] = field(default_factory=list)

    # Historical and evolutionary aspects
    historical_development: Dict[str, str] = field(default_factory=dict)
    foreign_influences: List[str] = field(default_factory=list)
    adaptation_patterns: List[str] = field(default_factory=list)

    # Contemporary context
    modern_adaptations: List[str] = field(default_factory=list)
    globalization_effects: List[str] = field(default_factory=list)
    preservation_efforts: List[str] = field(default_factory=list)

    # Knowledge validation
    knowledge_authority: str = ""  # source authority level
    community_consensus: float = 0.0  # 0.0-1.0
    expert_validation: bool = False
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DietaryRestrictionProfile(BaseGustatoryData):
    """Comprehensive dietary restriction profile"""

    # Restriction categories
    restriction_type: DietaryRestriction
    restriction_strictness: float  # 0.0-1.0
    enforcement_level: str  # voluntary, cultural_norm, religious_law, medical_necessity

    # Specific restrictions
    forbidden_ingredients: List[str] = field(default_factory=list)
    forbidden_combinations: List[Tuple[str, str]] = field(default_factory=list)
    forbidden_preparation_methods: List[str] = field(default_factory=list)

    # Permitted alternatives
    acceptable_substitutes: Dict[str, List[str]] = field(default_factory=dict)
    certified_products: List[str] = field(default_factory=list)

    # Contextual considerations
    situational_flexibility: float = 0.0  # 0.0-1.0
    emergency_provisions: List[str] = field(default_factory=list)
    travel_adaptations: List[str] = field(default_factory=list)

    # Authority and validation
    religious_authority: Optional[str] = None
    medical_authority: Optional[str] = None
    cultural_authority: Optional[str] = None
    certification_requirements: List[str] = field(default_factory=list)
```

### Personal Preference Data

#### Individual Preference Profile
```python
@dataclass
class PersonalTasteProfile(BaseGustatoryData):
    """Individual taste preference and characteristic profile"""

    # Basic taste preferences
    taste_preferences: Dict[BasicTasteType, float] = field(default_factory=dict)  # -1.0 to 1.0
    taste_sensitivity: Dict[BasicTasteType, float] = field(default_factory=dict)  # sensitivity multipliers

    # Flavor preferences
    preferred_flavor_categories: List[str] = field(default_factory=list)
    disliked_flavor_categories: List[str] = field(default_factory=list)
    flavor_adventure_level: float = 0.0  # 0.0-1.0, openness to new flavors

    # Texture and temperature preferences
    preferred_textures: List[str] = field(default_factory=list)
    disliked_textures: List[str] = field(default_factory=list)
    temperature_preferences: Dict[str, float] = field(default_factory=dict)

    # Spice and intensity preferences
    spice_tolerance: float = 0.0  # 0.0-1.0
    preferred_spice_types: List[str] = field(default_factory=list)
    intensity_preference: float = 0.0  # 0.0-1.0, mild to intense

    # Cultural food preferences
    cultural_food_comfort_level: Dict[str, float] = field(default_factory=dict)
    cross_cultural_openness: float = 0.0  # 0.0-1.0
    traditional_food_attachment: float = 0.0  # 0.0-1.0

    # Dietary considerations
    health_considerations: List[str] = field(default_factory=list)
    allergy_profile: List[str] = field(default_factory=list)
    nutritional_priorities: List[str] = field(default_factory=list)

    # Learning and adaptation characteristics
    preference_plasticity: float = 0.0  # 0.0-1.0, how easily preferences change
    learning_rate: float = 0.0  # rate of preference adaptation
    memory_influence_strength: float = 0.0  # how much memories influence preferences

    # Individual characteristics
    supertaster_classification: Optional[str] = None  # supertaster, medium, non-taster
    age_group: str = ""
    sex: Optional[str] = None
    cultural_background: List[str] = field(default_factory=list)

    # Preference evolution tracking
    preference_history: List[Tuple[datetime, str, float]] = field(default_factory=list)  # (time, preference_item, value)
    preference_trends: Dict[str, str] = field(default_factory=dict)  # increasing, decreasing, stable
```

## Consciousness Generation Data Structures

### Conscious Experience Data

#### Gustatory Consciousness Experience
```python
@dataclass
class GustatoryConsciousnessExperience(BaseGustatoryData):
    """Complete gustatory consciousness experience"""

    # Experience identification
    experience_type: str  # flavor_tasting, meal_experience, cultural_exploration, etc.
    consciousness_intensity: ConsciousnessIntensity

    # Phenomenological components
    phenomenological_qualities: 'PhenomenologicalQualities'
    temporal_experience_flow: 'TemporalExperienceFlow'
    emotional_response_profile: 'EmotionalResponseProfile'
    memory_experience_integration: 'MemoryExperienceIntegration'

    # Attention and awareness
    attention_distribution: Dict[str, float] = field(default_factory=dict)  # attention allocation
    mindful_awareness_level: float = 0.0  # 0.0-1.0
    distraction_level: float = 0.0  # 0.0-1.0

    # Individual and cultural resonance
    personal_significance: float = 0.0  # 0.0-1.0
    cultural_resonance: float = 0.0  # 0.0-1.0
    novelty_impact: float = 0.0  # 0.0-1.0

    # Quality assessment
    authenticity_score: float = 0.0  # 0.0-1.0
    richness_rating: float = 0.0  # 0.0-1.0
    coherence_score: float = 0.0  # 0.0-1.0
    satisfaction_level: float = 0.0  # 0.0-1.0

    # Integration quality
    cross_modal_integration_quality: float = 0.0  # 0.0-1.0
    memory_integration_quality: float = 0.0  # 0.0-1.0
    cultural_integration_quality: float = 0.0  # 0.0-1.0

@dataclass
class PhenomenologicalQualities(BaseGustatoryData):
    """Phenomenological qualities of gustatory consciousness"""

    # Sensory qualities
    flavor_richness: float = 0.0  # 0.0-1.0
    taste_clarity: float = 0.0  # 0.0-1.0
    aroma_prominence: float = 0.0  # 0.0-1.0
    texture_awareness: float = 0.0  # 0.0-1.0
    temperature_awareness: float = 0.0  # 0.0-1.0

    # Temporal qualities
    temporal_coherence: float = 0.0  # 0.0-1.0
    development_smoothness: float = 0.0  # 0.0-1.0
    transition_naturalness: float = 0.0  # 0.0-1.0

    # Emotional and hedonic qualities
    pleasantness: float = 0.0  # -1.0 to 1.0
    emotional_depth: float = 0.0  # 0.0-1.0
    nostalgic_quality: float = 0.0  # 0.0-1.0
    comfort_level: float = 0.0  # 0.0-1.0

    # Cognitive and memory qualities
    memory_vividness: float = 0.0  # 0.0-1.0
    associative_richness: float = 0.0  # 0.0-1.0
    cultural_recognition: float = 0.0  # 0.0-1.0
    personal_relevance: float = 0.0  # 0.0-1.0

    # Attentional qualities
    attention_capture: float = 0.0  # 0.0-1.0
    focus_intensity: float = 0.0  # 0.0-1.0
    mindful_presence: float = 0.0  # 0.0-1.0

    # Aesthetic qualities
    complexity_appreciation: float = 0.0  # 0.0-1.0
    harmony_perception: float = 0.0  # 0.0-1.0
    beauty_assessment: float = 0.0  # 0.0-1.0

@dataclass
class TemporalExperienceFlow(BaseGustatoryData):
    """Temporal flow of gustatory consciousness experience"""

    # Phase structure
    anticipation_phase: Dict[str, Any] = field(default_factory=dict)
    initial_contact_phase: Dict[str, Any] = field(default_factory=dict)
    development_phase: Dict[str, Any] = field(default_factory=dict)
    peak_experience_phase: Dict[str, Any] = field(default_factory=dict)
    decline_phase: Dict[str, Any] = field(default_factory=dict)
    aftertaste_phase: Dict[str, Any] = field(default_factory=dict)
    memory_integration_phase: Dict[str, Any] = field(default_factory=dict)

    # Temporal characteristics
    total_experience_duration: float = 0.0  # seconds
    peak_intensity_time: float = 0.0  # seconds from start
    attention_peak_time: float = 0.0  # seconds from start

    # Flow quality
    temporal_coherence: float = 0.0  # 0.0-1.0
    transition_smoothness: float = 0.0  # 0.0-1.0
    flow_naturalness: float = 0.0  # 0.0-1.0

    # Dynamic changes
    intensity_trajectory: List[Tuple[float, float]] = field(default_factory=list)  # (time, intensity)
    attention_trajectory: List[Tuple[float, float]] = field(default_factory=list)  # (time, attention)
    emotional_trajectory: List[Tuple[float, float]] = field(default_factory=list)  # (time, emotion)
```

### Emotional and Hedonic Response Data

#### Emotional Response Profile
```python
@dataclass
class EmotionalResponseProfile(BaseGustatoryData):
    """Emotional response to gustatory experience"""

    # Basic emotional responses
    primary_emotion: str = ""
    secondary_emotions: List[str] = field(default_factory=list)
    emotional_intensity: float = 0.0  # 0.0-1.0
    emotional_valence: float = 0.0  # -1.0 to 1.0

    # Hedonic evaluation
    pleasantness_rating: float = 0.0  # -5.0 to 5.0
    preference_strength: float = 0.0  # 0.0-1.0
    approach_avoidance_tendency: float = 0.0  # -1.0 to 1.0

    # Complex emotional states
    nostalgia_level: float = 0.0  # 0.0-1.0
    comfort_level: float = 0.0  # 0.0-1.0
    excitement_level: float = 0.0  # 0.0-1.0
    anxiety_level: float = 0.0  # 0.0-1.0

    # Cultural emotional responses
    cultural_pride: float = 0.0  # 0.0-1.0
    cultural_comfort: float = 0.0  # 0.0-1.0
    cultural_novelty_excitement: float = 0.0  # 0.0-1.0

    # Memory-related emotions
    memory_triggered_emotions: List[str] = field(default_factory=list)
    autobiographical_emotional_intensity: float = 0.0  # 0.0-1.0

    # Physiological emotional indicators
    autonomic_response_indicators: Dict[str, float] = field(default_factory=dict)
    facial_expression_indicators: List[str] = field(default_factory=list)

    # Social emotional aspects
    sharing_desire: float = 0.0  # 0.0-1.0, desire to share experience
    social_bonding_potential: float = 0.0  # 0.0-1.0

    # Temporal emotional dynamics
    emotional_development_timeline: List[Tuple[float, str, float]] = field(default_factory=list)  # (time, emotion, intensity)
    emotional_persistence: float = 0.0  # how long emotions persist after experience
```

## System Performance and Quality Data

### Quality Assessment Data

#### Experience Quality Metrics
```python
@dataclass
class GustatoryQualityMetrics(BaseGustatoryData):
    """Quality assessment metrics for gustatory consciousness"""

    # Authenticity metrics
    biological_plausibility: float = 0.0  # 0.0-1.0
    phenomenological_authenticity: float = 0.0  # 0.0-1.0
    cultural_authenticity: float = 0.0  # 0.0-1.0

    # Performance metrics
    processing_latency: float = 0.0  # milliseconds
    detection_accuracy: float = 0.0  # 0.0-1.0
    integration_quality: float = 0.0  # 0.0-1.0

    # User experience metrics
    user_satisfaction: float = 0.0  # 0.0-1.0
    engagement_level: float = 0.0  # 0.0-1.0
    learning_effectiveness: float = 0.0  # 0.0-1.0

    # Cultural sensitivity metrics
    cultural_appropriateness: float = 0.0  # 0.0-1.0
    religious_compliance: float = 0.0  # 0.0-1.0
    dietary_restriction_compliance: float = 0.0  # 0.0-1.0

    # System health metrics
    error_rates: Dict[str, float] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    scalability_indicators: Dict[str, float] = field(default_factory=dict)

    # Quality trends
    quality_trend_direction: str = "stable"  # improving, stable, declining
    quality_improvement_rate: float = 0.0  # rate of quality change
    consistency_score: float = 0.0  # 0.0-1.0, consistency across experiences
```

These comprehensive data structures provide the foundation for representing all aspects of gustatory consciousness, from basic taste detection through complex cultural adaptation and conscious experience generation, while maintaining biological authenticity, cultural sensitivity, and phenomenological richness.