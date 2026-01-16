# Interoceptive Consciousness System - Data Structures

**Document**: Data Structures Specification
**Form**: 06 - Interoceptive Consciousness
**Category**: Technical Specification & Design
**Version**: 1.0
**Date**: 2025-09-27

## Executive Summary

This document defines the comprehensive data structures for the Interoceptive Consciousness System, encompassing all physiological sensor inputs, consciousness representations, integration models, and communication protocols required for implementing rich interoceptive awareness.

## Core Data Structure Categories

### Physiological Sensor Data Structures
### Consciousness Experience Data Structures
### Integration and Processing Data Structures
### Safety and Control Data Structures
### Configuration and State Management Data Structures

## Physiological Sensor Data Structures

### 1. Cardiovascular Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum, IntEnum
from datetime import datetime, timezone
import numpy as np

class HeartRhythmClassification(Enum):
    NORMAL_SINUS_RHYTHM = "normal_sinus_rhythm"
    SINUS_BRADYCARDIA = "sinus_bradycardia"
    SINUS_TACHYCARDIA = "sinus_tachycardia"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"
    ATRIAL_FLUTTER = "atrial_flutter"
    VENTRICULAR_TACHYCARDIA = "ventricular_tachycardia"
    PREMATURE_VENTRICULAR_CONTRACTION = "pvc"
    PREMATURE_ATRIAL_CONTRACTION = "pac"
    UNKNOWN_RHYTHM = "unknown"

class CardiacSafetyLevel(IntEnum):
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4

@dataclass
class HeartRateVariabilityMetrics:
    """Comprehensive HRV analysis metrics"""
    rmssd_ms: float                    # Root mean square of successive differences
    sdnn_ms: float                     # Standard deviation of NN intervals
    pnn50_percent: float               # Percentage of NN50 divided by total NN intervals
    triangular_index: float            # HRV triangular index
    tinn_ms: float                     # Triangular interpolation of NN intervals

    # Frequency domain metrics
    lf_power_ms2: float                # Low frequency power (0.04-0.15 Hz)
    hf_power_ms2: float                # High frequency power (0.15-0.4 Hz)
    lf_hf_ratio: float                 # LF/HF ratio
    total_power_ms2: float             # Total spectral power
    vlf_power_ms2: float               # Very low frequency power (≤0.04 Hz)

    # Non-linear metrics
    sd1_ms: float                      # Poincaré plot SD1
    sd2_ms: float                      # Poincaré plot SD2
    sd1_sd2_ratio: float               # SD1/SD2 ratio
    approximate_entropy: float         # ApEn measure
    sample_entropy: float              # SampEn measure

    # Time window information
    analysis_window_minutes: int       # Duration of analysis window
    data_quality_score: float          # 0.0-1.0 quality assessment
    measurement_confidence: float      # 0.0-1.0 confidence level

@dataclass
class CardiovascularSensorReading:
    """Single cardiovascular sensor reading with comprehensive metrics"""
    sensor_id: str
    user_id: str
    timestamp_utc: datetime
    timestamp_ms: int                  # Milliseconds since epoch

    # Primary cardiac metrics
    instantaneous_heart_rate_bpm: float
    rr_interval_ms: int               # Most recent R-R interval
    rr_intervals_buffer: List[int] = field(default_factory=list)  # Last 300 intervals

    # Blood pressure metrics (if available)
    systolic_pressure_mmhg: Optional[float] = None
    diastolic_pressure_mmhg: Optional[float] = None
    mean_arterial_pressure_mmhg: Optional[float] = None
    pulse_pressure_mmhg: Optional[float] = None

    # Advanced cardiac metrics
    pulse_wave_velocity_ms: Optional[float] = None
    cardiac_output_l_min: Optional[float] = None
    stroke_volume_ml: Optional[float] = None
    ejection_fraction_percent: Optional[float] = None

    # Rhythm analysis
    rhythm_classification: HeartRhythmClassification = HeartRhythmClassification.UNKNOWN_RHYTHM
    rhythm_confidence: float = 0.0     # 0.0-1.0 classification confidence
    arrhythmia_burden_percent: float = 0.0  # Recent arrhythmia percentage

    # Heart rate variability
    hrv_metrics: Optional[HeartRateVariabilityMetrics] = None

    # Signal quality and reliability
    signal_quality_score: float = 0.0  # 0.0-1.0 signal quality
    motion_artifact_level: float = 0.0 # 0.0-1.0 motion artifact assessment
    electrode_contact_quality: float = 0.0  # 0.0-1.0 contact quality

    # Safety assessment
    safety_level: CardiacSafetyLevel = CardiacSafetyLevel.NORMAL
    safety_alerts: List[str] = field(default_factory=list)

    # Metadata
    sensor_type: str = "ecg"           # ecg, ppg, accelerometer, etc.
    sampling_rate_hz: int = 250        # Data sampling frequency
    calibration_timestamp: Optional[datetime] = None

@dataclass
class CardiovascularTimeSeries:
    """Time series data for cardiovascular monitoring"""
    user_id: str
    start_timestamp_utc: datetime
    end_timestamp_utc: datetime
    readings: List[CardiovascularSensorReading]

    # Aggregated metrics over time series
    average_heart_rate_bpm: float
    heart_rate_range_bpm: Tuple[float, float]  # (min, max)
    heart_rate_trend: str              # "increasing", "decreasing", "stable"
    rhythm_stability_score: float      # 0.0-1.0 rhythm stability

    # HRV trend analysis
    hrv_trend_analysis: Dict[str, Any]
    cardiac_load_assessment: Dict[str, float]
    recovery_indicators: Dict[str, float]

    # Data quality summary
    overall_data_quality: float        # 0.0-1.0 overall quality
    data_completeness_percent: float   # Percentage of expected data points
    interpolated_points_count: int     # Number of interpolated/estimated points
```

### 2. Respiratory Data Structures

```python
class RespiratoryPatternType(Enum):
    EUPNEA = "eupnea"                  # Normal breathing
    TACHYPNEA = "tachypnea"           # Rapid breathing
    BRADYPNEA = "bradypnea"           # Slow breathing
    HYPERPNEA = "hyperpnea"           # Deep breathing
    HYPOPNEA = "hypopnea"             # Shallow breathing
    DYSPNEA = "dyspnea"               # Difficult breathing
    APNEA = "apnea"                   # Absence of breathing
    CHEYNE_STOKES = "cheyne_stokes"   # Periodic breathing pattern
    KUSSMAUL = "kussmaul"             # Deep, labored breathing
    BIOT = "biot"                     # Irregular breathing with pauses

@dataclass
class RespiratoryMechanics:
    """Detailed respiratory mechanics measurements"""
    tidal_volume_ml: float             # Normal breath volume
    inspiratory_reserve_volume_ml: float  # Additional inhalation capacity
    expiratory_reserve_volume_ml: float   # Additional exhalation capacity
    residual_volume_ml: float          # Volume remaining after max exhalation
    functional_residual_capacity_ml: float  # Volume at rest
    total_lung_capacity_ml: float      # Maximum lung volume
    vital_capacity_ml: float           # Maximum breath volume

    # Flow rates
    peak_inspiratory_flow_l_min: float  # Maximum inhalation flow
    peak_expiratory_flow_l_min: float   # Maximum exhalation flow
    forced_expiratory_volume_1s_ml: float  # FEV1
    forced_vital_capacity_ml: float     # FVC
    fev1_fvc_ratio: float              # FEV1/FVC ratio

    # Timing measurements
    inspiration_time_ms: int           # Duration of inhalation
    expiration_time_ms: int            # Duration of exhalation
    inspiratory_pause_ms: int          # Pause after inhalation
    expiratory_pause_ms: int           # Pause after exhalation
    ie_ratio: float                    # Inspiration/Expiration ratio

    # Work of breathing
    respiratory_effort_score: float    # 0.0-10.0 effort assessment
    work_of_breathing_j_l: float       # Energy expenditure per liter
    respiratory_muscle_fatigue: float  # 0.0-1.0 fatigue level

@dataclass
class RespiratoryGasExchange:
    """Gas exchange and oxygenation metrics"""
    oxygen_saturation_percent: float   # SpO2
    partial_pressure_oxygen_mmhg: float  # PaO2
    partial_pressure_co2_mmhg: float    # PaCO2
    end_tidal_co2_mmhg: float          # EtCO2
    ph_level: float                    # Blood pH
    bicarbonate_mmol_l: float          # HCO3- concentration
    base_excess_mmol_l: float          # Base excess/deficit

    # Calculated indices
    alveolar_arterial_gradient: float  # A-a gradient
    respiratory_quotient: float        # RQ (CO2 production/O2 consumption)
    dead_space_ventilation_ratio: float  # VD/VT ratio
    shunt_fraction_percent: float      # Right-to-left shunt

    # Diffusion capacity
    dlco_ml_min_mmhg: Optional[float] = None  # Diffusion capacity for CO
    dlco_predicted_percent: Optional[float] = None  # % of predicted DLCO

@dataclass
class RespiratorySensorReading:
    """Comprehensive respiratory sensor reading"""
    sensor_id: str
    user_id: str
    timestamp_utc: datetime
    timestamp_ms: int

    # Basic respiratory metrics
    respiratory_rate_bpm: float
    breath_depth_ml: float             # Current breath volume
    minute_ventilation_l: float        # Total ventilation per minute

    # Pattern classification
    breathing_pattern: RespiratoryPatternType
    pattern_confidence: float          # 0.0-1.0 classification confidence
    pattern_regularity_score: float    # 0.0-1.0 regularity assessment

    # Detailed mechanics
    mechanics: RespiratoryMechanics

    # Gas exchange data
    gas_exchange: Optional[RespiratoryGasExchange] = None

    # Respiratory distress indicators
    dyspnea_score: float              # 0.0-10.0 breathing difficulty
    accessory_muscle_use: bool        # Use of accessory breathing muscles
    paradoxical_breathing: bool       # Paradoxical chest/abdomen movement
    stridor_present: bool             # Audible breathing sounds

    # Environmental factors
    ambient_air_quality_index: Optional[int] = None
    ambient_humidity_percent: Optional[float] = None
    ambient_pressure_kpa: Optional[float] = None

    # Signal quality
    signal_quality_score: float       # 0.0-1.0 overall signal quality
    motion_artifact_level: float      # 0.0-1.0 motion interference
    sensor_calibration_status: str     # "calibrated", "needs_calibration", "error"

    # Safety assessment
    respiratory_safety_level: CardiacSafetyLevel
    respiratory_alerts: List[str] = field(default_factory=list)
```

### 3. Gastrointestinal Data Structures

```python
class GastricMotilityPhase(Enum):
    PHASE_1_QUIESCENCE = "phase_1_quiescence"  # Minimal contractions
    PHASE_2_IRREGULAR = "phase_2_irregular"     # Irregular contractions
    PHASE_3_REGULAR = "phase_3_regular"         # Regular strong contractions
    PHASE_4_DECLINING = "phase_4_declining"     # Declining activity
    DIGESTIVE_ACTIVE = "digestive_active"       # Post-meal activity
    PATHOLOGICAL = "pathological"               # Abnormal patterns

@dataclass
class GastrointestinalHormones:
    """Gastrointestinal hormone levels and appetite regulation"""
    ghrelin_pg_ml: float              # Hunger hormone
    leptin_ng_ml: float               # Satiety hormone
    cholecystokinin_pg_ml: float      # CCK satiety signal
    glucagon_like_peptide_1_pmol_l: float  # GLP-1
    peptide_yy_pg_ml: float           # PYY satiety hormone
    insulin_mu_u_ml: float            # Insulin levels
    cortisol_ug_dl: float             # Stress hormone affecting appetite

    # Metabolic indicators
    blood_glucose_mg_dl: float        # Blood sugar level
    free_fatty_acids_mmol_l: float    # FFA levels
    triglycerides_mg_dl: float        # Triglyceride levels
    ketones_mmol_l: float             # Ketone bodies

    # Timing relative to meals
    time_since_last_meal_hours: float
    time_to_next_expected_meal_hours: float
    fasting_duration_hours: float

@dataclass
class GastricMechanics:
    """Gastric mechanical function and motility"""
    gastric_volume_ml: float          # Current stomach volume
    gastric_pressure_mmhg: float      # Intragastric pressure
    gastric_compliance_ml_mmhg: float # Stomach wall elasticity
    gastric_emptying_rate_percent_hour: float  # Emptying rate
    gastric_accommodation_ml: float   # Volume accommodation to food

    # Motility patterns
    antral_contractions_per_minute: float  # Antral wave frequency
    contraction_amplitude_mmhg: float      # Strength of contractions
    motility_index: float                  # Composite motility score
    migrating_motor_complex_phase: GastricMotilityPhase

    # Gastric secretion
    gastric_acid_production_meq_hour: float  # Acid secretion rate
    gastric_ph: float                        # Stomach pH
    pepsin_activity_units: float             # Pepsin enzyme activity
    intrinsic_factor_level: float            # B12 absorption factor

@dataclass
class DigestiveComfort:
    """Subjective digestive comfort and symptoms"""
    overall_comfort_score: float      # -5 (very uncomfortable) to +5 (very comfortable)
    nausea_intensity: float           # 0.0-10.0 nausea scale
    bloating_severity: float          # 0.0-10.0 bloating scale
    abdominal_pain_intensity: float   # 0.0-10.0 pain scale
    heartburn_severity: float         # 0.0-10.0 heartburn scale

    # Appetite and hunger sensations
    hunger_intensity: float           # 0.0-10.0 hunger scale
    satiety_level: float             # 0.0-10.0 fullness scale
    food_cravings: Dict[str, float]   # Specific food cravings by category
    appetite_quality: str             # "normal", "decreased", "increased", "distorted"

    # Digestive symptoms
    early_satiety: bool              # Feeling full quickly
    delayed_gastric_emptying: bool   # Gastroparesis symptoms
    dumping_syndrome: bool           # Rapid gastric emptying symptoms
    food_intolerances: List[str]     # Known food intolerances

@dataclass
class GastrointestinalSensorReading:
    """Comprehensive gastrointestinal sensor reading"""
    sensor_id: str
    user_id: str
    timestamp_utc: datetime
    timestamp_ms: int

    # Hormonal regulation
    hormones: GastrointestinalHormones

    # Gastric mechanics
    mechanics: GastricMechanics

    # Subjective comfort
    comfort: DigestiveComfort

    # Nutritional status
    nutritional_indicators: Dict[str, float]  # Vitamins, minerals, macronutrients
    hydration_status: float           # -2 to +2 standard deviations
    metabolic_rate_kcal_day: float    # Estimated daily energy expenditure

    # Meal and eating behavior
    recent_food_intake: List[Dict[str, Any]]  # Recent meals and snacks
    eating_rate_g_min: float          # Speed of food consumption
    meal_satisfaction_score: float    # 0.0-10.0 meal satisfaction

    # Microbiome indicators
    gut_microbiome_diversity_index: Optional[float] = None
    beneficial_bacteria_ratio: Optional[float] = None
    inflammatory_markers: Dict[str, float] = field(default_factory=dict)

    # Signal quality
    sensor_accuracy_level: float      # 0.0-1.0 measurement accuracy
    biological_variability: float     # Expected individual variation
    measurement_confidence: float     # 0.0-1.0 confidence level
```

### 4. Thermoregulatory Data Structures

```python
class ThermalState(Enum):
    SEVERE_HYPOTHERMIA = "severe_hypothermia"      # <32°C
    MODERATE_HYPOTHERMIA = "moderate_hypothermia"   # 32-35°C
    MILD_HYPOTHERMIA = "mild_hypothermia"          # 35-36°C
    NORMAL_CORE_TEMP = "normal_core_temp"          # 36-37.5°C
    MILD_HYPERTHERMIA = "mild_hyperthermia"        # 37.5-38.5°C
    MODERATE_HYPERTHERMIA = "moderate_hyperthermia" # 38.5-40°C
    SEVERE_HYPERTHERMIA = "severe_hyperthermia"    # >40°C

@dataclass
class ThermoregulationResponse:
    """Physiological thermoregulatory responses"""
    # Autonomic responses
    sweating_rate_g_m2_h: float       # Sweat production rate
    skin_blood_flow_ml_100g_min: float # Skin perfusion
    shivering_intensity: float        # 0.0-1.0 thermogenesis intensity
    goosebumps_response: bool         # Piloerection present

    # Vascular responses
    peripheral_vasoconstriction: float # 0.0-1.0 vasoconstriction level
    peripheral_vasodilation: float    # 0.0-1.0 vasodilation level
    core_to_shell_heat_transfer: float # W/m² heat transfer rate

    # Behavioral responses
    thermal_seeking_behavior: str     # "warming", "cooling", "neutral"
    clothing_adjustment_urge: float   # 0.0-1.0 urge to adjust clothing
    environment_seeking_urge: float   # 0.0-1.0 urge to change environment
    activity_modification_urge: float # 0.0-1.0 urge to change activity

    # Metabolic responses
    metabolic_heat_production_w: float # Heat production rate
    brown_adipose_tissue_activity: float # BAT thermogenesis
    thyroid_hormone_response: float   # T3/T4 mediated response
    stress_hormone_response: float    # Cortisol, epinephrine response

@dataclass
class ThermalComfort:
    """Thermal comfort and sensation assessment"""
    thermal_sensation_score: float    # -3 (very cold) to +3 (very hot)
    thermal_comfort_score: float      # -3 (very uncomfortable) to +3 (very comfortable)
    thermal_preference: str           # "warmer", "cooler", "no_change"
    thermal_acceptability: bool       # Environment thermally acceptable

    # Detailed thermal sensations
    whole_body_sensation: float       # Overall thermal sensation
    local_sensations: Dict[str, float] # Regional thermal sensations
    thermal_adaptation_level: float   # 0.0-1.0 adaptation to current conditions
    thermal_expectation_met: bool     # Thermal expectations satisfied

    # Individual factors
    individual_thermal_preference: float  # Personal thermal preference
    thermal_history_influence: float      # Influence of recent thermal history
    circadian_thermal_rhythm_phase: float # Current circadian thermal phase
    menstrual_cycle_thermal_influence: Optional[float] = None  # Hormonal influence

@dataclass
class EnvironmentalThermalData:
    """Environmental thermal conditions"""
    air_temperature_celsius: float
    radiant_temperature_celsius: float
    air_velocity_m_s: float
    relative_humidity_percent: float
    atmospheric_pressure_kpa: float

    # Derived thermal indices
    heat_index: float                 # Apparent temperature with humidity
    wind_chill_index: float          # Apparent temperature with wind
    wet_bulb_globe_temperature: float # WBGT thermal stress index
    thermal_comfort_index: float     # PMV predicted mean vote

    # Clothing and activity factors
    clothing_insulation_clo: float   # Clothing thermal resistance
    activity_metabolic_rate_met: float # Activity level in METs
    surface_contact_temperature: Dict[str, float] # Contact surface temps

@dataclass
class ThermoregulatoryReading:
    """Comprehensive thermoregulatory sensor reading"""
    sensor_id: str
    user_id: str
    timestamp_utc: datetime
    timestamp_ms: int

    # Core temperature measurements
    core_temperature_celsius: float   # Deep body temperature
    core_temperature_trend: str      # "rising", "falling", "stable"
    core_temperature_rate_change: float # °C/hour change rate

    # Skin temperature measurements
    mean_skin_temperature_celsius: float # Average skin temperature
    skin_temperature_distribution: Dict[str, float] # Regional skin temps
    skin_temperature_gradient: float  # Core-skin temperature difference

    # Thermal state assessment
    thermal_state: ThermalState
    thermal_state_confidence: float  # 0.0-1.0 classification confidence
    time_in_current_state_minutes: int # Duration in current thermal state

    # Thermoregulatory responses
    thermoregulation: ThermoregulationResponse

    # Thermal comfort
    comfort: ThermalComfort

    # Environmental conditions
    environment: EnvironmentalThermalData

    # Thermal load and stress
    thermal_load_w_m2: float         # Total thermal load
    thermal_stress_index: float      # Composite thermal stress
    heat_strain_score: float         # 0.0-1.0 heat strain level
    cold_strain_score: float         # 0.0-1.0 cold strain level

    # Thermoregulatory efficiency
    thermal_efficiency_score: float  # 0.0-1.0 efficiency assessment
    thermal_adaptation_capacity: float # Remaining adaptation capacity
    thermal_recovery_rate: float     # Rate of thermal recovery

    # Safety assessment
    thermal_safety_level: CardiacSafetyLevel
    thermal_safety_alerts: List[str] = field(default_factory=list)

    # Measurement quality
    sensor_calibration_offset: float  # Calibration correction
    measurement_uncertainty: float    # ±°C measurement uncertainty
    sensor_response_time_s: float     # Sensor thermal response time
```

## Consciousness Experience Data Structures

### 1. Interoceptive Consciousness Experience

```python
@dataclass
class InteroceptiveAttentionState:
    """Attention and awareness state for interoceptive consciousness"""
    overall_interoceptive_attention: float  # 0.0-1.0 general bodily attention
    modality_specific_attention: Dict[str, float]  # Attention to specific systems
    attention_stability: float             # 0.0-1.0 stability of attention
    attention_switching_frequency: float   # Attention switching rate per minute

    # Selective attention patterns
    focused_body_region: Optional[str] = None     # Currently focused region
    attention_breadth: float                      # 0.0-1.0 narrow to broad focus
    background_monitoring_level: float           # 0.0-1.0 background awareness

    # Metacognitive awareness
    interoceptive_metacognition: float           # Awareness of bodily awareness
    confidence_in_interoceptive_accuracy: float  # Confidence in bodily signals
    interoceptive_insight: float                 # Understanding of bodily patterns

@dataclass
class SomaticMarkers:
    """Somatic markers for decision-making and emotion"""
    gut_feeling_intensity: float          # 0.0-1.0 intuitive body sense
    decision_relevant_signals: Dict[str, float]  # Body signals affecting decisions
    emotional_body_mapping: Dict[str, Dict[str, float]]  # Emotion-body connections
    approach_avoidance_signals: Dict[str, float]  # Motivational body signals

    # Anticipatory responses
    anticipatory_anxiety_signals: Dict[str, float]    # Pre-event body changes
    reward_anticipation_signals: Dict[str, float]     # Positive anticipation markers
    threat_detection_signals: Dict[str, float]        # Danger detection markers

    # Social and interpersonal markers
    social_comfort_signals: Dict[str, float]          # Social situation body response
    empathic_resonance_signals: Dict[str, float]      # Empathic body responses
    attachment_security_signals: Dict[str, float]     # Attachment-related signals

@dataclass
class InteroceptiveMemory:
    """Memory components of interoceptive experience"""
    episodic_body_memories: List[Dict[str, Any]]      # Specific bodily experiences
    semantic_body_knowledge: Dict[str, Any]           # General bodily knowledge
    procedural_body_skills: Dict[str, float]          # Embodied procedural memory
    emotional_body_associations: Dict[str, Dict[str, float]]  # Emotion-body links

    # Memory consolidation
    memory_encoding_strength: float                   # 0.0-1.0 encoding strength
    memory_retrieval_accuracy: float                  # 0.0-1.0 retrieval accuracy
    memory_interference_level: float                  # 0.0-1.0 interference from other memories

    # Autobiographical body memory
    significant_body_events: List[Dict[str, Any]]     # Milestone bodily experiences
    body_narrative_coherence: float                   # 0.0-1.0 coherent body story
    body_identity_consistency: float                  # 0.0-1.0 consistent body identity

@dataclass
class InteroceptiveConsciousnessState:
    """Comprehensive interoceptive consciousness state representation"""
    state_id: str
    user_id: str
    timestamp_utc: datetime
    timestamp_ms: int

    # Primary interoceptive modalities
    cardiovascular_consciousness: Optional['CardiovascularConsciousnessState'] = None
    respiratory_consciousness: Optional['RespiratoryConsciousnessState'] = None
    gastrointestinal_consciousness: Optional['GastrointestinalConsciousnessState'] = None
    thermoregulatory_consciousness: Optional['ThermoregulatoryConsciousnessState'] = None

    # Integrated consciousness components
    unified_bodily_state: Dict[str, float]            # Integrated body state representation
    homeostatic_balance_awareness: float              # 0.0-1.0 overall balance awareness
    interoceptive_coherence: float                    # 0.0-1.0 cross-modal coherence

    # Attention and awareness
    attention_state: InteroceptiveAttentionState

    # Decision-making integration
    somatic_markers: SomaticMarkers

    # Memory integration
    interoceptive_memory: InteroceptiveMemory

    # Emotional integration
    embodied_emotions: Dict[str, float]               # Current emotional body states
    mood_body_correlation: Dict[str, float]           # Mood-body relationships
    stress_embodiment: Dict[str, float]               # Stress-related body changes

    # Consciousness quality metrics
    consciousness_clarity: float                      # 0.0-1.0 clarity of consciousness
    phenomenological_richness: float                 # 0.0-1.0 richness of experience
    temporal_continuity: float                        # 0.0-1.0 temporal coherence
    spatial_coherence: float                          # 0.0-1.0 spatial body mapping

    # Individual differences
    interoceptive_sensitivity: float                  # Individual sensitivity level
    interoceptive_accuracy: float                     # Objective accuracy score
    interoceptive_sensibility: float                  # Subjective confidence
    trait_body_awareness: float                       # Stable individual awareness trait

    # Context and environment
    environmental_context: Dict[str, Any]             # Current environmental factors
    social_context: Dict[str, Any]                    # Current social context
    activity_context: Dict[str, Any]                  # Current activity context

    # Safety and wellbeing
    overall_safety_status: CardiacSafetyLevel
    wellbeing_indicators: Dict[str, float]            # Various wellbeing metrics
    stress_indicators: Dict[str, float]               # Stress and strain indicators

    # System metadata
    data_sources: List[str]                           # Active sensor sources
    processing_confidence: float                      # 0.0-1.0 processing confidence
    system_performance_metrics: Dict[str, float]      # System performance indicators
```

This comprehensive data structure specification provides the foundation for representing all aspects of interoceptive consciousness, from raw physiological sensor data to complex consciousness states, enabling sophisticated modeling and analysis of internal bodily awareness.