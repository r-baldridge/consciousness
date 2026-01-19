#!/usr/bin/env python3
"""
Psychedelic/Entheogenic Consciousness Interface

Form 37: The comprehensive interface for psychedelic and entheogenic consciousness
research, integrating scientific neuroscience, phenomenological experience types,
traditional ceremonial contexts, and therapeutic applications.

Ethical Principles:
- Harm reduction and safety prioritization
- Cultural respect for traditional practices
- Evidence-based therapeutic claims
- Integration support for experiences
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class PsychedelicSubstance(Enum):
    """
    Major psychedelic and entheogenic substances.

    Organized by chemical class and traditional/clinical use.
    """

    # === CLASSICAL TRYPTAMINES ===
    PSILOCYBIN = "psilocybin"  # Psilocybe mushrooms
    DMT = "dmt"  # N,N-Dimethyltryptamine
    FIVE_MEO_DMT = "five_meo_dmt"  # 5-MeO-DMT (Bufo, plants)

    # === LYSERGAMIDES ===
    LSD = "lsd"  # Lysergic acid diethylamide

    # === PHENETHYLAMINES ===
    MESCALINE = "mescaline"  # Peyote, San Pedro

    # === EMPATHOGENS ===
    MDMA = "mdma"  # 3,4-Methylenedioxymethamphetamine

    # === DISSOCIATIVES ===
    KETAMINE = "ketamine"  # Arylcyclohexylamine dissociative

    # === TRADITIONAL ENTHEOGENS ===
    AYAHUASCA = "ayahuasca"  # DMT + MAO inhibitor brew
    IBOGAINE = "ibogaine"  # Tabernanthe iboga
    SALVIA_DIVINORUM = "salvia_divinorum"  # Salvinorin A
    CANNABIS_HIGH_DOSE = "cannabis_high_dose"  # Psychedelic-level THC
    AMANITA_MUSCARIA = "amanita_muscaria"  # Fly agaric mushroom
    SAN_PEDRO = "san_pedro"  # Echinopsis pachanoi (mescaline)


class ExperienceType(Enum):
    """
    Categories of psychedelic experience phenomena.

    Based on phenomenological research and clinical observation.
    """

    VISUAL_GEOMETRY = "visual_geometry"  # Form constants, patterns
    ENTITY_ENCOUNTER = "entity_encounter"  # Contact with beings
    EGO_DISSOLUTION = "ego_dissolution"  # Loss of self-boundaries
    MYSTICAL_UNITY = "mystical_unity"  # Unity consciousness, oneness
    TIME_DISTORTION = "time_distortion"  # Altered temporal perception
    SYNESTHESIA = "synesthesia"  # Cross-sensory perception
    EMOTIONAL_CATHARSIS = "emotional_catharsis"  # Emotional release
    INSIGHT_REVELATION = "insight_revelation"  # Noetic quality, understanding
    DEATH_REBIRTH = "death_rebirth"  # Ego death and renewal
    COSMIC_CONSCIOUSNESS = "cosmic_consciousness"  # Transcendent awareness
    ANCESTRAL_CONTACT = "ancestral_contact"  # Connection to ancestors
    HEALING_VISION = "healing_vision"  # Diagnostic/therapeutic imagery


class EntheogenicTradition(Enum):
    """
    Traditional and syncretic entheogenic practices.

    Represents living and historical traditions.
    """

    # === SOUTH AMERICAN ===
    AMAZONIAN_AYAHUASCA = "amazonian_ayahuasca"  # Indigenous Amazonian
    SANTO_DAIME = "santo_daime"  # Brazilian syncretic
    UDV = "udv"  # Uniao do Vegetal

    # === MESOAMERICAN ===
    MESOAMERICAN_MUSHROOM = "mesoamerican_mushroom"  # Mazatec, Aztec

    # === NORTH AMERICAN ===
    NATIVE_AMERICAN_PEYOTE = "native_american_peyote"  # NAC

    # === AFRICAN ===
    AFRICAN_IBOGA = "african_iboga"  # Bwiti tradition

    # === EURASIAN ===
    SIBERIAN_AMANITA = "siberian_amanita"  # Siberian shamanic

    # === HISTORICAL ===
    ELEUSINIAN_MYSTERIES = "eleusinian_mysteries"  # Ancient Greek
    VEDIC_SOMA = "vedic_soma"  # Ancient Vedic


class TherapeuticApplication(Enum):
    """
    Clinical indications for psychedelic-assisted therapy.

    Based on current research and clinical trials.
    """

    DEPRESSION = "depression"  # Major depressive disorder
    PTSD = "ptsd"  # Post-traumatic stress disorder
    ADDICTION = "addiction"  # Substance use disorders
    END_OF_LIFE_ANXIETY = "end_of_life_anxiety"  # Existential distress
    OCD = "ocd"  # Obsessive-compulsive disorder
    EATING_DISORDERS = "eating_disorders"  # Anorexia, bulimia


class NeuralMechanism(Enum):
    """
    Neurobiological mechanisms of psychedelic action.

    Key neural and receptor targets.
    """

    DMN_DISRUPTION = "dmn_disruption"  # Default mode network effects
    SEROTONIN_5HT2A = "serotonin_5ht2a"  # Primary receptor target
    NEURAL_ENTROPY = "neural_entropy"  # Increased signal complexity
    CONNECTIVITY_INCREASE = "connectivity_increase"  # Enhanced brain connectivity
    NEUROPLASTICITY = "neuroplasticity"  # Structural/functional plasticity


class SetSettingFactor(Enum):
    """
    Environmental and psychological factors influencing experience.

    Critical variables for outcome prediction.
    """

    INTENTION = "intention"  # Purpose and goals
    ENVIRONMENT = "environment"  # Physical setting
    GUIDE_PRESENCE = "guide_presence"  # Facilitator/therapist
    MUSIC = "music"  # Auditory environment
    PREPARATION = "preparation"  # Pre-experience work
    INTEGRATION = "integration"  # Post-experience processing


class MaturityLevel(Enum):
    """Depth of knowledge coverage."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SubstanceProfile:
    """
    Comprehensive profile of a psychedelic substance.

    Includes pharmacology, dosing, safety, and therapeutic potential.
    """
    substance_id: str
    substance: PsychedelicSubstance
    chemical_class: str
    receptor_targets: List[str]
    typical_dose: str
    duration: str
    subjective_effects: List[str] = field(default_factory=list)
    therapeutic_potential: List[TherapeuticApplication] = field(default_factory=list)
    safety_profile: str = ""
    legal_status: str = ""
    natural_sources: List[str] = field(default_factory=list)
    mechanisms: List[NeuralMechanism] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Substance: {self.substance.value}",
            f"Class: {self.chemical_class}",
            f"Effects: {', '.join(self.subjective_effects)}",
            f"Targets: {', '.join(self.receptor_targets)}"
        ]
        return " | ".join(parts)


@dataclass
class PsychedelicExperience:
    """
    Record of a psychedelic experience with phenomenological details.

    Captures substance, context, and experience qualities.
    """
    experience_id: str
    substance: PsychedelicSubstance
    dose: str
    set_setting: Dict[SetSettingFactor, str] = field(default_factory=dict)
    experience_types: List[ExperienceType] = field(default_factory=list)
    duration: str = ""
    insights: List[str] = field(default_factory=list)
    integration_notes: str = ""
    intensity_level: int = 3  # 1-5 scale (McKenna levels)
    mystical_score: Optional[float] = None  # MEQ30 equivalent
    ego_dissolution_score: Optional[float] = None  # EDI equivalent
    challenging_aspects: List[str] = field(default_factory=list)
    visual_phenomena: List[str] = field(default_factory=list)
    emotional_phenomena: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class EntheogenicCeremony:
    """
    Traditional or syncretic ceremonial context.

    Links to Form 29 for indigenous wisdom integration.
    """
    ceremony_id: str
    tradition: EntheogenicTradition
    substance: PsychedelicSubstance
    ritual_structure: List[str] = field(default_factory=list)
    spiritual_framework: str = ""
    preparation_requirements: List[str] = field(default_factory=list)
    integration_practices: List[str] = field(default_factory=list)
    facilitator_role: str = ""
    music_elements: List[str] = field(default_factory=list)
    dietary_protocols: List[str] = field(default_factory=list)
    form_29_link: Optional[str] = None  # Link to folk wisdom teaching
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class TherapeuticProtocol:
    """
    Clinical protocol for psychedelic-assisted therapy.

    Based on research protocols from major institutions.
    """
    protocol_id: str
    indication: TherapeuticApplication
    substance: PsychedelicSubstance
    dose_schedule: List[str] = field(default_factory=list)
    session_structure: Dict[str, str] = field(default_factory=dict)
    therapist_requirements: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    efficacy_data: Dict[str, Any] = field(default_factory=dict)
    preparation_sessions: int = 0
    dosing_sessions: int = 0
    integration_sessions: int = 0
    research_institution: str = ""
    trial_phase: str = ""
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExperiencePhenomenology:
    """
    Detailed phenomenological mapping of experience type.

    Describes subjective features and neural correlates.
    """
    phenomenology_id: str
    experience_type: ExperienceType
    description: str
    subjective_features: List[str] = field(default_factory=list)
    neural_correlates: List[NeuralMechanism] = field(default_factory=list)
    common_triggers: List[str] = field(default_factory=list)
    substances_associated: List[PsychedelicSubstance] = field(default_factory=list)
    therapeutic_relevance: str = ""
    measurement_scales: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PsychedelicConsciousnessMaturityState:
    """Tracks the maturity of psychedelic consciousness knowledge."""
    overall_maturity: float = 0.0
    substance_coverage: Dict[str, float] = field(default_factory=dict)
    substance_count: int = 0
    experience_type_count: int = 0
    ceremony_count: int = 0
    protocol_count: int = 0
    phenomenology_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class PsychedelicConsciousnessInterface:
    """
    Main interface for Form 37: Psychedelic/Entheogenic Consciousness.

    Provides methods for storing, retrieving, and querying psychedelic
    substance profiles, experience phenomenology, traditional ceremonies,
    and therapeutic protocols.
    """

    FORM_ID = "37-psychedelic-consciousness"
    FORM_NAME = "Psychedelic/Entheogenic Consciousness"

    def __init__(self):
        """Initialize the Psychedelic Consciousness Interface."""
        # Knowledge indexes
        self.substance_index: Dict[str, SubstanceProfile] = {}
        self.experience_index: Dict[str, PsychedelicExperience] = {}
        self.ceremony_index: Dict[str, EntheogenicCeremony] = {}
        self.protocol_index: Dict[str, TherapeuticProtocol] = {}
        self.phenomenology_index: Dict[str, ExperiencePhenomenology] = {}

        # Cross-reference indexes
        self.substance_type_index: Dict[PsychedelicSubstance, List[str]] = {}
        self.tradition_index: Dict[EntheogenicTradition, List[str]] = {}
        self.indication_index: Dict[TherapeuticApplication, List[str]] = {}
        self.experience_type_index: Dict[ExperienceType, List[str]] = {}

        # Maturity tracking
        self.maturity_state = PsychedelicConsciousnessMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize substance type index
        for substance in PsychedelicSubstance:
            self.substance_type_index[substance] = []

        # Initialize tradition index
        for tradition in EntheogenicTradition:
            self.tradition_index[tradition] = []

        # Initialize indication index
        for indication in TherapeuticApplication:
            self.indication_index[indication] = []

        # Initialize experience type index
        for exp_type in ExperienceType:
            self.experience_type_index[exp_type] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # SUBSTANCE METHODS
    # ========================================================================

    async def add_substance(self, substance: SubstanceProfile) -> None:
        """Add a substance profile to the index."""
        self.substance_index[substance.substance_id] = substance

        # Update substance type index
        if substance.substance in self.substance_type_index:
            self.substance_type_index[substance.substance].append(substance.substance_id)

        # Update indication index
        for indication in substance.therapeutic_potential:
            if indication in self.indication_index:
                self.indication_index[indication].append(substance.substance_id)

        # Update maturity
        self.maturity_state.substance_count = len(self.substance_index)
        await self._update_maturity()

    async def get_substance(self, substance_id: str) -> Optional[SubstanceProfile]:
        """Retrieve a substance profile by ID."""
        return self.substance_index.get(substance_id)

    async def query_substances_by_type(
        self,
        substance: PsychedelicSubstance,
        limit: int = 10
    ) -> List[SubstanceProfile]:
        """Query substance profiles by substance type."""
        substance_ids = self.substance_type_index.get(substance, [])[:limit]
        return [
            self.substance_index[sid]
            for sid in substance_ids
            if sid in self.substance_index
        ]

    async def query_substances_by_indication(
        self,
        indication: TherapeuticApplication,
        limit: int = 10
    ) -> List[SubstanceProfile]:
        """Query substances by therapeutic indication."""
        substance_ids = self.indication_index.get(indication, [])[:limit]
        return [
            self.substance_index[sid]
            for sid in substance_ids
            if sid in self.substance_index
        ]

    # ========================================================================
    # EXPERIENCE METHODS
    # ========================================================================

    async def add_experience(self, experience: PsychedelicExperience) -> None:
        """Add an experience record to the index."""
        self.experience_index[experience.experience_id] = experience

        # Update experience type index
        for exp_type in experience.experience_types:
            if exp_type in self.experience_type_index:
                self.experience_type_index[exp_type].append(experience.experience_id)

        await self._update_maturity()

    async def get_experience(self, experience_id: str) -> Optional[PsychedelicExperience]:
        """Retrieve an experience by ID."""
        return self.experience_index.get(experience_id)

    async def query_experiences_by_type(
        self,
        experience_type: ExperienceType,
        limit: int = 10
    ) -> List[PsychedelicExperience]:
        """Query experiences by experience type."""
        experience_ids = self.experience_type_index.get(experience_type, [])[:limit]
        return [
            self.experience_index[eid]
            for eid in experience_ids
            if eid in self.experience_index
        ]

    # ========================================================================
    # CEREMONY METHODS
    # ========================================================================

    async def add_ceremony(self, ceremony: EntheogenicCeremony) -> None:
        """Add a ceremony record to the index."""
        self.ceremony_index[ceremony.ceremony_id] = ceremony

        # Update tradition index
        if ceremony.tradition in self.tradition_index:
            self.tradition_index[ceremony.tradition].append(ceremony.ceremony_id)

        # Update maturity
        self.maturity_state.ceremony_count = len(self.ceremony_index)
        await self._update_maturity()

    async def get_ceremony(self, ceremony_id: str) -> Optional[EntheogenicCeremony]:
        """Retrieve a ceremony by ID."""
        return self.ceremony_index.get(ceremony_id)

    async def query_ceremonies_by_tradition(
        self,
        tradition: EntheogenicTradition,
        limit: int = 10
    ) -> List[EntheogenicCeremony]:
        """Query ceremonies by tradition."""
        ceremony_ids = self.tradition_index.get(tradition, [])[:limit]
        return [
            self.ceremony_index[cid]
            for cid in ceremony_ids
            if cid in self.ceremony_index
        ]

    # ========================================================================
    # PROTOCOL METHODS
    # ========================================================================

    async def add_protocol(self, protocol: TherapeuticProtocol) -> None:
        """Add a therapeutic protocol to the index."""
        self.protocol_index[protocol.protocol_id] = protocol

        # Update indication index
        if protocol.indication in self.indication_index:
            self.indication_index[protocol.indication].append(protocol.protocol_id)

        # Update maturity
        self.maturity_state.protocol_count = len(self.protocol_index)
        await self._update_maturity()

    async def get_protocol(self, protocol_id: str) -> Optional[TherapeuticProtocol]:
        """Retrieve a protocol by ID."""
        return self.protocol_index.get(protocol_id)

    async def query_protocols_by_indication(
        self,
        indication: TherapeuticApplication,
        limit: int = 10
    ) -> List[TherapeuticProtocol]:
        """Query protocols by therapeutic indication."""
        protocol_ids = self.indication_index.get(indication, [])[:limit]
        return [
            self.protocol_index[pid]
            for pid in protocol_ids
            if pid in self.protocol_index
        ]

    # ========================================================================
    # PHENOMENOLOGY METHODS
    # ========================================================================

    async def add_phenomenology(self, phenomenology: ExperiencePhenomenology) -> None:
        """Add a phenomenology record to the index."""
        self.phenomenology_index[phenomenology.phenomenology_id] = phenomenology

        # Update experience type index
        if phenomenology.experience_type in self.experience_type_index:
            self.experience_type_index[phenomenology.experience_type].append(
                phenomenology.phenomenology_id
            )

        # Update maturity
        self.maturity_state.phenomenology_count = len(self.phenomenology_index)
        await self._update_maturity()

    async def get_phenomenology(
        self,
        phenomenology_id: str
    ) -> Optional[ExperiencePhenomenology]:
        """Retrieve a phenomenology record by ID."""
        return self.phenomenology_index.get(phenomenology_id)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.substance_count +
            self.maturity_state.ceremony_count +
            self.maturity_state.protocol_count +
            self.maturity_state.phenomenology_count
        )

        # Simple maturity calculation
        target_items = 100  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update substance coverage
        for substance in PsychedelicSubstance:
            count = len(self.substance_type_index.get(substance, []))
            target_per_substance = 3
            self.maturity_state.substance_coverage[substance.value] = min(
                1.0, count / target_per_substance
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> PsychedelicConsciousnessMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_substances(self) -> List[Dict[str, Any]]:
        """Return seed substance profiles for initialization."""
        return [
            # Psilocybin
            {
                "substance_id": "psilocybin_profile",
                "substance": PsychedelicSubstance.PSILOCYBIN,
                "chemical_class": "Tryptamine",
                "receptor_targets": ["5-HT2A", "5-HT2C", "5-HT1A"],
                "typical_dose": "2-3.5g dried mushrooms (15-25mg pure)",
                "duration": "4-6 hours",
                "subjective_effects": [
                    "Visual geometry and color enhancement",
                    "Emotional intensity",
                    "Introspective insights",
                    "Nature connection",
                    "Time distortion",
                    "Potential ego dissolution"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION,
                    TherapeuticApplication.END_OF_LIFE_ANXIETY,
                    TherapeuticApplication.OCD
                ],
                "safety_profile": "Very low physiological toxicity. LD50 essentially non-existent at practical doses. Psychological risks include challenging experiences.",
                "legal_status": "Schedule I (US), decriminalized in some jurisdictions, legal in Jamaica and Netherlands (truffles)",
                "natural_sources": ["Psilocybe cubensis", "Psilocybe semilanceata", "Psilocybe azurescens"],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.NEUROPLASTICITY],
                "contraindications": ["Personal/family history of psychosis", "Lithium use", "Unstable cardiovascular conditions"],
            },
            # DMT
            {
                "substance_id": "dmt_profile",
                "substance": PsychedelicSubstance.DMT,
                "chemical_class": "Tryptamine",
                "receptor_targets": ["5-HT2A", "Sigma-1", "Trace amine receptors"],
                "typical_dose": "20-60mg smoked, 0.2-0.4mg/kg IV",
                "duration": "5-20 minutes smoked, 4-6 hours with MAOi (ayahuasca)",
                "subjective_effects": [
                    "Intense visual phenomena",
                    "Entity encounters",
                    "Breakthrough experiences",
                    "Time dissolution",
                    "Ineffability",
                    "Cosmic consciousness"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION
                ],
                "safety_profile": "Physiologically safe but intensely psychoactive. Requires careful set/setting. Not recommended for those with psychosis risk.",
                "legal_status": "Schedule I (US), religious exemptions for ayahuasca churches",
                "natural_sources": ["Psychotria viridis", "Mimosa hostilis", "Acacia confusa"],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION],
                "contraindications": ["MAOi interactions (when oral)", "Psychosis history", "Severe cardiac conditions"],
            },
            # 5-MeO-DMT
            {
                "substance_id": "five_meo_dmt_profile",
                "substance": PsychedelicSubstance.FIVE_MEO_DMT,
                "chemical_class": "Tryptamine",
                "receptor_targets": ["5-HT2A", "5-HT1A"],
                "typical_dose": "5-15mg smoked",
                "duration": "15-45 minutes",
                "subjective_effects": [
                    "Non-visual ego dissolution",
                    "White light experiences",
                    "Unity consciousness",
                    "Intense somatic effects",
                    "Complete self-transcendence"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION
                ],
                "safety_profile": "Higher risk than other tryptamines. Cardiac concerns. Intense ego death requires experienced facilitation.",
                "legal_status": "Schedule I (US), unscheduled in many countries",
                "natural_sources": ["Bufo alvarius toad venom", "Virola species", "Anadenanthera species"],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION],
                "contraindications": ["Cardiac conditions", "MAOi use", "Psychosis history", "Inexperienced setting"],
            },
            # LSD
            {
                "substance_id": "lsd_profile",
                "substance": PsychedelicSubstance.LSD,
                "chemical_class": "Lysergamide",
                "receptor_targets": ["5-HT2A", "D2", "5-HT1A", "5-HT2C", "Adrenergic"],
                "typical_dose": "75-150ug",
                "duration": "8-12 hours",
                "subjective_effects": [
                    "Visual distortions and geometry",
                    "Emotional variability",
                    "Cognitive enhancement",
                    "Stimulation",
                    "Synesthesia",
                    "Profound insights"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION,
                    TherapeuticApplication.END_OF_LIFE_ANXIETY
                ],
                "safety_profile": "Extremely potent (microgram doses). Long duration requires time commitment. No known lethal dose.",
                "legal_status": "Schedule I (US), universally controlled",
                "natural_sources": [],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.CONNECTIVITY_INCREASE],
                "contraindications": ["Psychosis history", "Lithium use", "Cardiac arrhythmias"],
            },
            # Mescaline
            {
                "substance_id": "mescaline_profile",
                "substance": PsychedelicSubstance.MESCALINE,
                "chemical_class": "Phenethylamine",
                "receptor_targets": ["5-HT2A", "5-HT2B", "5-HT2C"],
                "typical_dose": "200-400mg (mescaline), 6-12 peyote buttons",
                "duration": "8-12 hours",
                "subjective_effects": [
                    "Warm, colorful visuals",
                    "Heart-opening experiences",
                    "Slower onset than LSD",
                    "Nature appreciation",
                    "Spiritual connection"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION
                ],
                "safety_profile": "Initial nausea common. Long duration. Generally well-tolerated physically.",
                "legal_status": "Schedule I (US), religious exemptions for NAC members, San Pedro legal in many places",
                "natural_sources": ["Lophophora williamsii (peyote)", "Echinopsis pachanoi (San Pedro)"],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION],
                "contraindications": ["Psychosis history", "Cardiac conditions", "GI disorders"],
            },
            # MDMA
            {
                "substance_id": "mdma_profile",
                "substance": PsychedelicSubstance.MDMA,
                "chemical_class": "Empathogen/Entactogen",
                "receptor_targets": ["Serotonin release", "Dopamine release", "Norepinephrine release", "Oxytocin"],
                "typical_dose": "75-125mg with optional 60-75mg redose",
                "duration": "3-5 hours main effects",
                "subjective_effects": [
                    "Emotional openness",
                    "Empathy enhancement",
                    "Reduced fear response",
                    "Enhanced connection",
                    "Mild visual enhancement",
                    "Euphoria"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.PTSD,
                    TherapeuticApplication.END_OF_LIFE_ANXIETY
                ],
                "safety_profile": "Neurotoxicity concerns at high/frequent doses. Hyperthermia risk. Hyponatremia risk.",
                "legal_status": "Schedule I (US), potential rescheduling pending FDA approval",
                "natural_sources": [],
                "mechanisms": [NeuralMechanism.CONNECTIVITY_INCREASE, NeuralMechanism.NEUROPLASTICITY],
                "contraindications": ["Cardiac conditions", "Hyperthermia risk", "SSRIs", "MAOIs"],
            },
            # Ketamine
            {
                "substance_id": "ketamine_profile",
                "substance": PsychedelicSubstance.KETAMINE,
                "chemical_class": "Dissociative",
                "receptor_targets": ["NMDA antagonist", "Opioid receptors", "AMPA", "Sigma receptors"],
                "typical_dose": "50-100mg IM, 100-200mg for K-hole",
                "duration": "45-90 minutes IM",
                "subjective_effects": [
                    "Dissociation",
                    "Out-of-body experiences",
                    "Dream-like states",
                    "Anesthetic properties",
                    "K-hole (complete dissociation)"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.PTSD
                ],
                "safety_profile": "FDA-approved for depression (esketamine). Addiction potential with frequent use. Bladder damage with chronic abuse.",
                "legal_status": "Schedule III (US), widely available medically",
                "natural_sources": [],
                "mechanisms": [NeuralMechanism.NEUROPLASTICITY, NeuralMechanism.CONNECTIVITY_INCREASE],
                "contraindications": ["Uncontrolled hypertension", "History of substance abuse", "Psychosis"],
            },
            # Ibogaine
            {
                "substance_id": "ibogaine_profile",
                "substance": PsychedelicSubstance.IBOGAINE,
                "chemical_class": "Indole alkaloid",
                "receptor_targets": ["NMDA antagonist", "Kappa opioid", "Serotonin", "Dopamine", "Sigma"],
                "typical_dose": "15-25mg/kg (flood dose)",
                "duration": "24-36 hours acute, after-effects for days",
                "subjective_effects": [
                    "Visionary life review",
                    "Introspection",
                    "Physical intensity",
                    "Ataxia",
                    "Waking dream states"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.ADDICTION
                ],
                "safety_profile": "Significant cardiac risks (QT prolongation). Several reported deaths. Requires medical supervision.",
                "legal_status": "Schedule I (US), available in clinics in Mexico, New Zealand, Brazil",
                "natural_sources": ["Tabernanthe iboga root bark"],
                "mechanisms": [NeuralMechanism.NEUROPLASTICITY, NeuralMechanism.DMN_DISRUPTION],
                "contraindications": ["Cardiac conditions", "QT prolongation", "Hepatic impairment", "Opioid dependence (withdrawal risk)"],
            },
            # Salvia divinorum
            {
                "substance_id": "salvia_profile",
                "substance": PsychedelicSubstance.SALVIA_DIVINORUM,
                "chemical_class": "Neoclerodane diterpenoid",
                "receptor_targets": ["Kappa opioid receptor agonist"],
                "typical_dose": "0.25-0.5g dried leaf, 10-60mg extract",
                "duration": "5-30 minutes smoked",
                "subjective_effects": [
                    "Reality shattering",
                    "Identity dissolution",
                    "Bizarre physical sensations",
                    "Dysphoric potential",
                    "Uncontrollable laughter or terror"
                ],
                "therapeutic_potential": [],
                "safety_profile": "Unique non-serotonergic mechanism. Short duration but intense. Low addiction potential.",
                "legal_status": "Unscheduled federally (US), state laws vary",
                "natural_sources": ["Salvia divinorum plant (Oaxaca, Mexico)"],
                "mechanisms": [],
                "contraindications": ["Inexperienced setting", "Unsupervised use", "Mental health instability"],
            },
            # Ayahuasca
            {
                "substance_id": "ayahuasca_profile",
                "substance": PsychedelicSubstance.AYAHUASCA,
                "chemical_class": "DMT + MAO inhibitor combination",
                "receptor_targets": ["5-HT2A", "Sigma-1", "MAO-A inhibition"],
                "typical_dose": "Variable (traditional preparation)",
                "duration": "4-6 hours",
                "subjective_effects": [
                    "Intense visions",
                    "Purging (vomiting, diarrhea)",
                    "Emotional processing",
                    "Entity encounters",
                    "Healing visions",
                    "Spiritual insights"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION,
                    TherapeuticApplication.PTSD
                ],
                "safety_profile": "MAOi interactions critical. Tyramine diet restrictions. Requires experienced facilitation.",
                "legal_status": "Religious exemptions (UDV, Santo Daime in US), legal for traditional use in South America",
                "natural_sources": ["Banisteriopsis caapi", "Psychotria viridis", "Mimosa hostilis"],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION],
                "contraindications": ["MAOi medications", "SSRIs", "Tyramine-rich foods", "Cardiac conditions", "Psychosis"],
            },
            # Cannabis (high dose)
            {
                "substance_id": "cannabis_high_dose_profile",
                "substance": PsychedelicSubstance.CANNABIS_HIGH_DOSE,
                "chemical_class": "Cannabinoid",
                "receptor_targets": ["CB1", "CB2"],
                "typical_dose": "100-300mg+ THC (edibles, naive users)",
                "duration": "4-8 hours (edibles)",
                "subjective_effects": [
                    "Full psychedelic experiences possible",
                    "Visual phenomena",
                    "Temporal distortion",
                    "Anxiety/paranoia potential",
                    "Memory effects"
                ],
                "therapeutic_potential": [],
                "safety_profile": "Highly variable response. Set/setting dependent. Memory impairment.",
                "legal_status": "Variable by jurisdiction",
                "natural_sources": ["Cannabis sativa/indica"],
                "mechanisms": [],
                "contraindications": ["Psychosis history", "Anxiety disorders", "Adolescent brain development"],
            },
            # Amanita muscaria
            {
                "substance_id": "amanita_profile",
                "substance": PsychedelicSubstance.AMANITA_MUSCARIA,
                "chemical_class": "Isoxazole (muscimol, ibotenic acid)",
                "receptor_targets": ["GABA-A agonist (muscimol)", "Glutamate agonist (ibotenic acid)"],
                "typical_dose": "5-15g dried caps (after proper preparation)",
                "duration": "6-8 hours",
                "subjective_effects": [
                    "Deliriant effects",
                    "Dream-like states",
                    "Sedation",
                    "Size distortion (macropsia/micropsia)",
                    "Unpredictable experience"
                ],
                "therapeutic_potential": [],
                "safety_profile": "Ibotenic acid toxicity if improperly prepared. Unpredictable dosing. Requires proper decarboxylation.",
                "legal_status": "Legal in most jurisdictions",
                "natural_sources": ["Amanita muscaria (fly agaric)"],
                "mechanisms": [],
                "contraindications": ["Liver conditions", "GI disorders", "Inexperienced preparation"],
            },
            # San Pedro
            {
                "substance_id": "san_pedro_profile",
                "substance": PsychedelicSubstance.SAN_PEDRO,
                "chemical_class": "Phenethylamine (mescaline-containing)",
                "receptor_targets": ["5-HT2A", "5-HT2B", "5-HT2C"],
                "typical_dose": "30-50cm of cactus, prepared as tea",
                "duration": "8-12 hours",
                "subjective_effects": [
                    "Heart-opening experiences",
                    "Nature connection",
                    "Gentle visual enhancement",
                    "Emotional processing",
                    "Spiritual insights"
                ],
                "therapeutic_potential": [
                    TherapeuticApplication.DEPRESSION,
                    TherapeuticApplication.ADDICTION
                ],
                "safety_profile": "Nausea during onset common. Long duration. Generally safe with proper preparation.",
                "legal_status": "Legal to possess in most countries, illegal to consume for psychoactive purposes in some",
                "natural_sources": ["Echinopsis pachanoi", "Echinopsis peruviana (Peruvian Torch)"],
                "mechanisms": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION],
                "contraindications": ["Cardiac conditions", "Psychosis history", "GI disorders"],
            },
        ]

    def _get_seed_ceremonies(self) -> List[Dict[str, Any]]:
        """Return seed ceremony data for initialization."""
        return [
            # Amazonian Ayahuasca
            {
                "ceremony_id": "ayahuasca_traditional",
                "tradition": EntheogenicTradition.AMAZONIAN_AYAHUASCA,
                "substance": PsychedelicSubstance.AYAHUASCA,
                "ritual_structure": [
                    "Sunset preparation and intentions",
                    "Nighttime ceremony opening",
                    "Curandero/a serves ayahuasca",
                    "Icaros (medicine songs) sung throughout",
                    "Individual and collective healing work",
                    "Dawn closing and sharing"
                ],
                "spiritual_framework": "Plant spirit communication and healing. The vine as teacher. Direct access to spiritual realms through mareacion (intoxication).",
                "preparation_requirements": [
                    "Dieta (dietary restrictions) for 3-7+ days",
                    "No sexual activity",
                    "No pork, alcohol, or fermented foods",
                    "Intention setting with facilitator",
                    "Medical screening"
                ],
                "integration_practices": [
                    "Morning sharing circle",
                    "Continued dieta post-ceremony",
                    "Dream journaling",
                    "Follow-up sessions with curandero/a"
                ],
                "facilitator_role": "Curandero/a guides ceremony through icaros, holds space, performs healings",
                "music_elements": ["Icaros (medicine songs)", "Chakapa (leaf rattle)", "Sometimes instruments"],
                "dietary_protocols": ["No tyramine foods", "No salt", "No sugar", "No spices", "Plain foods only"],
                "form_29_link": "amazonian_plant_medicine",
            },
            # Native American Peyote
            {
                "ceremony_id": "nac_peyote_ceremony",
                "tradition": EntheogenicTradition.NATIVE_AMERICAN_PEYOTE,
                "substance": PsychedelicSubstance.MESCALINE,
                "ritual_structure": [
                    "Tipi setup and altar preparation",
                    "Evening opening prayers",
                    "Peyote distribution throughout night",
                    "Drum and singing rounds",
                    "Midnight water ceremony",
                    "Morning water ceremony",
                    "Communal breakfast and sharing"
                ],
                "spiritual_framework": "Direct communion with Creator through peyote medicine. Christian and indigenous elements synthesized. Prayer as central practice.",
                "preparation_requirements": [
                    "Clear intentions",
                    "Proper invitation from Road Chief",
                    "Appropriate dress and demeanor",
                    "Fasting before ceremony"
                ],
                "integration_practices": [
                    "Morning meal and sharing",
                    "Return to daily life with insights",
                    "Ongoing NAC community participation"
                ],
                "facilitator_role": "Road Chief leads ceremony. Drum Chief, Cedar Chief, Fire Chief assist. Male-female balance.",
                "music_elements": ["Water drum", "Gourd rattle", "Peyote songs", "Eagle bone whistle"],
                "dietary_protocols": ["Fasting before ceremony", "No alcohol"],
                "form_29_link": "plains_indigenous_ceremonies",
            },
            # Mazatec Mushroom Velada
            {
                "ceremony_id": "mazatec_velada",
                "tradition": EntheogenicTradition.MESOAMERICAN_MUSHROOM,
                "substance": PsychedelicSubstance.PSILOCYBIN,
                "ritual_structure": [
                    "Nighttime gathering in healer's home",
                    "Copal incense and prayers",
                    "Distribution of mushroom pairs",
                    "Healer and patient both consume",
                    "Chanting and diagnostic visions",
                    "Healing interventions",
                    "Dawn closing"
                ],
                "spiritual_framework": "Mushrooms as 'little saints' (nti si tho). Direct communication with spirits for healing and divination. Catholic and indigenous synthesis.",
                "preparation_requirements": [
                    "Sexual abstinence",
                    "Clear purpose (healing need)",
                    "Trust in curandera/o"
                ],
                "integration_practices": [
                    "Follow healer's recommendations",
                    "Dietary/behavioral prescriptions",
                    "Follow-up ceremonies if needed"
                ],
                "facilitator_role": "Curandera/o (like Maria Sabina) diagnoses and heals through mushroom visions",
                "music_elements": ["Chanting", "Clapping", "Repetitive prayers"],
                "dietary_protocols": ["Fasting day of ceremony"],
                "form_29_link": "mesoamerican_folk_medicine",
            },
        ]

    def _get_seed_protocols(self) -> List[Dict[str, Any]]:
        """Return seed therapeutic protocol data for initialization."""
        return [
            # Psilocybin for Depression
            {
                "protocol_id": "psilocybin_depression_jhu",
                "indication": TherapeuticApplication.DEPRESSION,
                "substance": PsychedelicSubstance.PSILOCYBIN,
                "dose_schedule": ["25mg psilocybin", "Optional second session 25mg"],
                "session_structure": {
                    "preparation": "2-3 sessions (90 min each) building rapport and intentions",
                    "dosing": "1-2 sessions (6-8 hours) with eyeshades and music",
                    "integration": "2-4 sessions (90 min each) processing experience"
                },
                "therapist_requirements": [
                    "Licensed mental health professional",
                    "Specialized training in psychedelic-assisted therapy",
                    "Experience with non-ordinary states",
                    "Typically male-female co-therapist dyad"
                ],
                "contraindications": [
                    "Personal/family history of psychotic disorders",
                    "Current use of lithium",
                    "Unstable medical conditions",
                    "Active suicidality"
                ],
                "efficacy_data": {
                    "response_rate": "50-70%",
                    "remission_rate": "30-50%",
                    "effect_size": "Large (d > 0.8)",
                    "durability": "Months to years in responders",
                    "onset": "Within days to weeks"
                },
                "preparation_sessions": 3,
                "dosing_sessions": 2,
                "integration_sessions": 3,
                "research_institution": "Johns Hopkins University",
                "trial_phase": "Phase 2, moving to Phase 3",
            },
            # MDMA for PTSD
            {
                "protocol_id": "mdma_ptsd_maps",
                "indication": TherapeuticApplication.PTSD,
                "substance": PsychedelicSubstance.MDMA,
                "dose_schedule": ["80-120mg initial", "40-60mg optional supplement at 1.5-2 hours"],
                "session_structure": {
                    "preparation": "3 sessions (90 min each) building therapeutic alliance",
                    "dosing": "3 sessions (8 hours each) spaced 3-5 weeks apart",
                    "integration": "9 sessions (90 min each) throughout treatment"
                },
                "therapist_requirements": [
                    "Licensed mental health professional",
                    "MAPS MDMA therapy training",
                    "Male-female co-therapist dyad",
                    "Inner Directed approach training"
                ],
                "contraindications": [
                    "Uncontrolled hypertension",
                    "Cardiac arrhythmias",
                    "Hepatic impairment",
                    "Current use of MAOIs or SSRIs"
                ],
                "efficacy_data": {
                    "response_rate": "67% no longer met PTSD criteria",
                    "improvement_rate": "88% clinically meaningful reduction",
                    "effect_size": "Very large (d = 0.91)",
                    "durability": "Sustained at 12-month follow-up",
                    "effective_for": "Treatment-resistant PTSD"
                },
                "preparation_sessions": 3,
                "dosing_sessions": 3,
                "integration_sessions": 9,
                "research_institution": "MAPS",
                "trial_phase": "Phase 3 completed",
            },
            # Psilocybin for Tobacco Addiction
            {
                "protocol_id": "psilocybin_smoking_jhu",
                "indication": TherapeuticApplication.ADDICTION,
                "substance": PsychedelicSubstance.PSILOCYBIN,
                "dose_schedule": ["20mg/70kg first session", "30mg/70kg second session", "Optional 30mg/70kg third"],
                "session_structure": {
                    "preparation": "4 sessions of cognitive behavioral therapy for smoking cessation",
                    "dosing": "2-3 psilocybin sessions with preparatory and follow-up therapy",
                    "integration": "Weekly meetings for 10 weeks, then follow-up"
                },
                "therapist_requirements": [
                    "Training in both CBT for smoking and psychedelic therapy",
                    "Experience with addiction treatment"
                ],
                "contraindications": [
                    "Psychotic disorders",
                    "Unstable psychiatric conditions",
                    "Cardiovascular disease"
                ],
                "efficacy_data": {
                    "abstinence_6_months": "80%",
                    "abstinence_12_months": "60%",
                    "comparison": "Best existing treatment ~35%",
                    "predictor": "Mystical experience intensity"
                },
                "preparation_sessions": 4,
                "dosing_sessions": 3,
                "integration_sessions": 10,
                "research_institution": "Johns Hopkins University",
                "trial_phase": "Pilot completed, larger trials ongoing",
            },
            # Ketamine for Depression
            {
                "protocol_id": "ketamine_depression_clinical",
                "indication": TherapeuticApplication.DEPRESSION,
                "substance": PsychedelicSubstance.KETAMINE,
                "dose_schedule": ["0.5mg/kg IV over 40 min", "Repeated 2-3x/week for 2-3 weeks"],
                "session_structure": {
                    "assessment": "Psychiatric evaluation and medical clearance",
                    "dosing": "IV infusion in clinical setting with monitoring",
                    "observation": "2-hour post-infusion monitoring",
                    "maintenance": "Booster infusions as needed (weekly to monthly)"
                },
                "therapist_requirements": [
                    "Anesthesiologist or trained physician for IV",
                    "Psychiatric oversight",
                    "REMS certification for esketamine nasal spray"
                ],
                "contraindications": [
                    "Uncontrolled hypertension",
                    "History of psychosis",
                    "Substance use disorder",
                    "Increased intracranial pressure"
                ],
                "efficacy_data": {
                    "response_rate": "~60% for TRD",
                    "onset": "Hours to days (rapid)",
                    "durability": "Requires maintenance dosing",
                    "effect_size": "Moderate to large"
                },
                "preparation_sessions": 1,
                "dosing_sessions": 6,
                "integration_sessions": 0,
                "research_institution": "Multiple (FDA approved)",
                "trial_phase": "FDA approved (esketamine)",
            },
        ]

    def _get_seed_phenomenology(self) -> List[Dict[str, Any]]:
        """Return seed phenomenology data for initialization."""
        return [
            # Visual Geometry
            {
                "phenomenology_id": "visual_geometry_phenom",
                "experience_type": ExperienceType.VISUAL_GEOMETRY,
                "description": "Perception of geometric patterns, fractals, and form constants. Includes Kluver's form constants: tunnels, spirals, cobwebs, and gratings.",
                "subjective_features": [
                    "Color enhancement and shifting",
                    "Surface breathing and morphing",
                    "Kaleidoscopic closed-eye imagery",
                    "Open-eye geometric overlays",
                    "Fractal and recursive patterns",
                    "Sacred geometry (mandalas, yantras)",
                    "Self-transforming structures"
                ],
                "neural_correlates": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.NEURAL_ENTROPY],
                "common_triggers": ["Moderate to high doses", "Closed eyes", "Darkness", "Rhythmic sounds"],
                "substances_associated": [
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.DMT,
                    PsychedelicSubstance.MESCALINE
                ],
                "therapeutic_relevance": "May indicate 5-HT2A activation. Not directly therapeutic but often accompanies deeper experiences.",
                "measurement_scales": ["5D-ASC Visual Restructuralization", "Hallucinogen Rating Scale"],
            },
            # Entity Encounter
            {
                "phenomenology_id": "entity_encounter_phenom",
                "experience_type": ExperienceType.ENTITY_ENCOUNTER,
                "description": "Contact with perceived autonomous beings during psychedelic states. Includes machine elves, insectoid beings, divine figures, and ancestral spirits.",
                "subjective_features": [
                    "Perception of autonomous beings",
                    "Telepathic or emotional communication",
                    "Sense of being expected or known",
                    "Experiences of teaching or showing",
                    "Testing or challenging interactions",
                    "Healing or energy work",
                    "Ineffable quality of encounter"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.SEROTONIN_5HT2A],
                "common_triggers": ["High doses", "DMT specifically", "Breakthrough experiences", "Ceremonial contexts"],
                "substances_associated": [
                    PsychedelicSubstance.DMT,
                    PsychedelicSubstance.AYAHUASCA,
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.IBOGAINE
                ],
                "therapeutic_relevance": "Can provide profound meaning, guidance, and healing. Integration of encounter content may be therapeutically significant.",
                "measurement_scales": ["Strassman's Entity Encounter Questionnaire", "Phenomenological interview"],
            },
            # Ego Dissolution
            {
                "phenomenology_id": "ego_dissolution_phenom",
                "experience_type": ExperienceType.EGO_DISSOLUTION,
                "description": "Loss of the sense of self and self-other boundaries. Ranges from partial weakening of ego to complete dissolution.",
                "subjective_features": [
                    "Loss of self-other boundary",
                    "Dissolution of personal narrative",
                    "Unity with environment",
                    "Experience of ego death",
                    "Pure awareness without subject",
                    "Oceanic boundlessness",
                    "Terror or bliss depending on surrender"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.NEURAL_ENTROPY, NeuralMechanism.CONNECTIVITY_INCREASE],
                "common_triggers": ["High doses", "Surrender and letting go", "5-MeO-DMT specifically", "Supportive setting"],
                "substances_associated": [
                    PsychedelicSubstance.FIVE_MEO_DMT,
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.KETAMINE
                ],
                "therapeutic_relevance": "Strongly correlated with therapeutic outcomes. May reset rigid self-referential patterns. Requires integration.",
                "measurement_scales": ["Ego Dissolution Inventory (EDI)", "5D-ASC Oceanic Boundlessness"],
            },
            # Mystical Unity
            {
                "phenomenology_id": "mystical_unity_phenom",
                "experience_type": ExperienceType.MYSTICAL_UNITY,
                "description": "Complete mystical experience characterized by unity, transcendence, sacredness, and noetic quality. Matches criteria of classical mystical experiences.",
                "subjective_features": [
                    "Internal unity (all aspects of self unified)",
                    "External unity (merger with world)",
                    "Transcendence of time and space",
                    "Sacredness and sense of the holy",
                    "Noetic quality (direct knowing)",
                    "Deeply positive mood",
                    "Ineffability and paradoxicality"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.CONNECTIVITY_INCREASE],
                "common_triggers": ["Moderate to high doses", "Optimal set and setting", "Spiritual intention", "Music"],
                "substances_associated": [
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.FIVE_MEO_DMT,
                    PsychedelicSubstance.MESCALINE
                ],
                "therapeutic_relevance": "Primary mediator of therapeutic outcomes. Predicts sustained positive changes. Central to end-of-life work.",
                "measurement_scales": ["Mystical Experience Questionnaire (MEQ30)", "Hood Mysticism Scale"],
            },
            # Time Distortion
            {
                "phenomenology_id": "time_distortion_phenom",
                "experience_type": ExperienceType.TIME_DISTORTION,
                "description": "Altered perception of temporal flow including dilation, compression, loops, and timelessness.",
                "subjective_features": [
                    "Time dilation (minutes feel like hours)",
                    "Time compression (hours feel like minutes)",
                    "Time loops (repetitive temporal experience)",
                    "Timelessness (absence of temporal flow)",
                    "Eternal present (infinite now)",
                    "Temporal fragmentation (non-linear)"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.NEURAL_ENTROPY],
                "common_triggers": ["Most psychedelic doses", "Internal focus", "Ego dissolution"],
                "substances_associated": [
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.DMT,
                    PsychedelicSubstance.KETAMINE
                ],
                "therapeutic_relevance": "May allow processing of life experiences from new temporal perspective.",
                "measurement_scales": ["5D-ASC", "Phenomenological interview"],
            },
            # Synesthesia
            {
                "phenomenology_id": "synesthesia_phenom",
                "experience_type": ExperienceType.SYNESTHESIA,
                "description": "Cross-sensory perception where stimulation of one sense produces experiences in another.",
                "subjective_features": [
                    "Audio-visual synesthesia (seeing music)",
                    "Tactile-visual cross-perception",
                    "Conceptual synesthesia (ideas have colors)",
                    "Emotional synesthesia (feelings as shapes)",
                    "Enhanced sensory integration"
                ],
                "neural_correlates": [NeuralMechanism.CONNECTIVITY_INCREASE, NeuralMechanism.NEURAL_ENTROPY],
                "common_triggers": ["Music", "Moderate doses", "Sensory stimulation"],
                "substances_associated": [
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.MESCALINE
                ],
                "therapeutic_relevance": "Enhances music therapy. May indicate increased sensory integration.",
                "measurement_scales": ["5D-ASC", "Synesthesia Battery"],
            },
            # Emotional Catharsis
            {
                "phenomenology_id": "emotional_catharsis_phenom",
                "experience_type": ExperienceType.EMOTIONAL_CATHARSIS,
                "description": "Intense release of repressed or unprocessed emotions. Central to trauma processing.",
                "subjective_features": [
                    "Release of repressed emotions",
                    "Traumatic memory processing",
                    "Grief release",
                    "Rage expression in safe container",
                    "Overwhelming joy and gratitude",
                    "Somatic emotional release"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.NEUROPLASTICITY],
                "common_triggers": ["Therapeutic setting", "MDMA specifically", "Supportive presence", "Music"],
                "substances_associated": [
                    PsychedelicSubstance.MDMA,
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.AYAHUASCA,
                    PsychedelicSubstance.IBOGAINE
                ],
                "therapeutic_relevance": "Core mechanism in PTSD treatment. Allows processing of defended material.",
                "measurement_scales": ["Challenging Experience Questionnaire", "Emotional Breakthrough Inventory"],
            },
            # Death-Rebirth
            {
                "phenomenology_id": "death_rebirth_phenom",
                "experience_type": ExperienceType.DEATH_REBIRTH,
                "description": "Experience of psychological death and subsequent rebirth. Central to Grof's perinatal matrices.",
                "subjective_features": [
                    "Complete loss of identity",
                    "Confrontation with mortality",
                    "Surrender and acceptance",
                    "Experience of dying",
                    "White light or void",
                    "Rebirth with new perspective",
                    "Profound relief and gratitude"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.NEURAL_ENTROPY],
                "common_triggers": ["High doses", "5-MeO-DMT", "Ibogaine", "Surrender"],
                "substances_associated": [
                    PsychedelicSubstance.FIVE_MEO_DMT,
                    PsychedelicSubstance.IBOGAINE,
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.KETAMINE
                ],
                "therapeutic_relevance": "Central to addiction treatment. May resolve death anxiety. Requires integration.",
                "measurement_scales": ["Ego Dissolution Inventory", "Near-Death Experience Scale"],
            },
            # Cosmic Consciousness
            {
                "phenomenology_id": "cosmic_consciousness_phenom",
                "experience_type": ExperienceType.COSMIC_CONSCIOUSNESS,
                "description": "Transcendent awareness of cosmic scope. Characterized by intellectual illumination, moral elevation, and sense of immortality.",
                "subjective_features": [
                    "Subjective inner light",
                    "Intellectual illumination",
                    "Moral elevation",
                    "Sense of immortality",
                    "Loss of fear of death",
                    "Lasting character change",
                    "Understanding of universe"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION, NeuralMechanism.CONNECTIVITY_INCREASE],
                "common_triggers": ["High doses", "Mystical experience", "Spiritual preparation"],
                "substances_associated": [
                    PsychedelicSubstance.FIVE_MEO_DMT,
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.PSILOCYBIN
                ],
                "therapeutic_relevance": "May provide lasting perspective shift. Central to end-of-life work.",
                "measurement_scales": ["MEQ30", "Hood Mysticism Scale", "Bucke's criteria"],
            },
            # Insight/Revelation
            {
                "phenomenology_id": "insight_revelation_phenom",
                "experience_type": ExperienceType.INSIGHT_REVELATION,
                "description": "Noetic experiences of direct knowing and insight. Information or understanding that feels deeply true and significant.",
                "subjective_features": [
                    "Direct knowing (noesis)",
                    "Life insights",
                    "Relationship understanding",
                    "Self-understanding",
                    "Existential clarity",
                    "Sometimes specific information"
                ],
                "neural_correlates": [NeuralMechanism.NEUROPLASTICITY, NeuralMechanism.DMN_DISRUPTION],
                "common_triggers": ["Intention setting", "Moderate doses", "Integration focus"],
                "substances_associated": [
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.LSD,
                    PsychedelicSubstance.AYAHUASCA
                ],
                "therapeutic_relevance": "Core therapeutic mechanism. Insights require integration to be lasting.",
                "measurement_scales": ["MEQ30 Noetic Quality subscale", "Phenomenological interview"],
            },
            # Ancestral Contact
            {
                "phenomenology_id": "ancestral_contact_phenom",
                "experience_type": ExperienceType.ANCESTRAL_CONTACT,
                "description": "Experiences of connection with ancestors, genetic memory, or collective human heritage.",
                "subjective_features": [
                    "Visions of ancestors",
                    "Access to genetic/ancestral memory",
                    "Collective human experience",
                    "Evolutionary visions",
                    "Past life experiences",
                    "Cultural/lineage information"
                ],
                "neural_correlates": [NeuralMechanism.DMN_DISRUPTION],
                "common_triggers": ["Ceremonial contexts", "Ibogaine specifically", "Ayahuasca", "Cultural preparation"],
                "substances_associated": [
                    PsychedelicSubstance.IBOGAINE,
                    PsychedelicSubstance.AYAHUASCA,
                    PsychedelicSubstance.PSILOCYBIN
                ],
                "therapeutic_relevance": "May provide healing through ancestral reconciliation. Important in Bwiti tradition.",
                "measurement_scales": ["Phenomenological interview"],
            },
            # Healing Vision
            {
                "phenomenology_id": "healing_vision_phenom",
                "experience_type": ExperienceType.HEALING_VISION,
                "description": "Visionary experiences related to healing, diagnosis, or therapeutic imagery.",
                "subjective_features": [
                    "Diagnosis visions (seeing illness)",
                    "Extraction imagery (removing pathology)",
                    "Energy healing experiences",
                    "Plant spirit communication",
                    "Guided surgical metaphors",
                    "Body scan experiences"
                ],
                "neural_correlates": [NeuralMechanism.SEROTONIN_5HT2A, NeuralMechanism.DMN_DISRUPTION],
                "common_triggers": ["Ceremonial contexts", "Healer facilitation", "Ayahuasca specifically"],
                "substances_associated": [
                    PsychedelicSubstance.AYAHUASCA,
                    PsychedelicSubstance.PSILOCYBIN,
                    PsychedelicSubstance.SAN_PEDRO
                ],
                "therapeutic_relevance": "Central to traditional healing. May provide somatic and psychological healing.",
                "measurement_scales": ["Phenomenological interview", "Curandero assessment"],
            },
        ]

    async def initialize_seed_substances(self) -> int:
        """Initialize with seed substance profiles."""
        seed_substances = self._get_seed_substances()
        count = 0

        for substance_data in seed_substances:
            substance = SubstanceProfile(
                substance_id=substance_data["substance_id"],
                substance=substance_data["substance"],
                chemical_class=substance_data["chemical_class"],
                receptor_targets=substance_data["receptor_targets"],
                typical_dose=substance_data["typical_dose"],
                duration=substance_data["duration"],
                subjective_effects=substance_data.get("subjective_effects", []),
                therapeutic_potential=substance_data.get("therapeutic_potential", []),
                safety_profile=substance_data.get("safety_profile", ""),
                legal_status=substance_data.get("legal_status", ""),
                natural_sources=substance_data.get("natural_sources", []),
                mechanisms=substance_data.get("mechanisms", []),
                contraindications=substance_data.get("contraindications", []),
            )
            await self.add_substance(substance)
            count += 1

        logger.info(f"Initialized {count} seed substances")
        return count

    async def initialize_seed_ceremonies(self) -> int:
        """Initialize with seed ceremony data."""
        seed_ceremonies = self._get_seed_ceremonies()
        count = 0

        for ceremony_data in seed_ceremonies:
            ceremony = EntheogenicCeremony(
                ceremony_id=ceremony_data["ceremony_id"],
                tradition=ceremony_data["tradition"],
                substance=ceremony_data["substance"],
                ritual_structure=ceremony_data.get("ritual_structure", []),
                spiritual_framework=ceremony_data.get("spiritual_framework", ""),
                preparation_requirements=ceremony_data.get("preparation_requirements", []),
                integration_practices=ceremony_data.get("integration_practices", []),
                facilitator_role=ceremony_data.get("facilitator_role", ""),
                music_elements=ceremony_data.get("music_elements", []),
                dietary_protocols=ceremony_data.get("dietary_protocols", []),
                form_29_link=ceremony_data.get("form_29_link"),
            )
            await self.add_ceremony(ceremony)
            count += 1

        logger.info(f"Initialized {count} seed ceremonies")
        return count

    async def initialize_seed_protocols(self) -> int:
        """Initialize with seed therapeutic protocols."""
        seed_protocols = self._get_seed_protocols()
        count = 0

        for protocol_data in seed_protocols:
            protocol = TherapeuticProtocol(
                protocol_id=protocol_data["protocol_id"],
                indication=protocol_data["indication"],
                substance=protocol_data["substance"],
                dose_schedule=protocol_data.get("dose_schedule", []),
                session_structure=protocol_data.get("session_structure", {}),
                therapist_requirements=protocol_data.get("therapist_requirements", []),
                contraindications=protocol_data.get("contraindications", []),
                efficacy_data=protocol_data.get("efficacy_data", {}),
                preparation_sessions=protocol_data.get("preparation_sessions", 0),
                dosing_sessions=protocol_data.get("dosing_sessions", 0),
                integration_sessions=protocol_data.get("integration_sessions", 0),
                research_institution=protocol_data.get("research_institution", ""),
                trial_phase=protocol_data.get("trial_phase", ""),
            )
            await self.add_protocol(protocol)
            count += 1

        logger.info(f"Initialized {count} seed protocols")
        return count

    async def initialize_seed_phenomenology(self) -> int:
        """Initialize with seed phenomenology data."""
        seed_phenomenology = self._get_seed_phenomenology()
        count = 0

        for phenom_data in seed_phenomenology:
            phenom = ExperiencePhenomenology(
                phenomenology_id=phenom_data["phenomenology_id"],
                experience_type=phenom_data["experience_type"],
                description=phenom_data["description"],
                subjective_features=phenom_data.get("subjective_features", []),
                neural_correlates=phenom_data.get("neural_correlates", []),
                common_triggers=phenom_data.get("common_triggers", []),
                substances_associated=phenom_data.get("substances_associated", []),
                therapeutic_relevance=phenom_data.get("therapeutic_relevance", ""),
                measurement_scales=phenom_data.get("measurement_scales", []),
            )
            await self.add_phenomenology(phenom)
            count += 1

        logger.info(f"Initialized {count} seed phenomenology records")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        substances_count = await self.initialize_seed_substances()
        ceremonies_count = await self.initialize_seed_ceremonies()
        protocols_count = await self.initialize_seed_protocols()
        phenomenology_count = await self.initialize_seed_phenomenology()

        total = substances_count + ceremonies_count + protocols_count + phenomenology_count

        return {
            "substances": substances_count,
            "ceremonies": ceremonies_count,
            "protocols": protocols_count,
            "phenomenology": phenomenology_count,
            "total": total
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "PsychedelicSubstance",
    "ExperienceType",
    "EntheogenicTradition",
    "TherapeuticApplication",
    "NeuralMechanism",
    "SetSettingFactor",
    "MaturityLevel",
    # Dataclasses
    "SubstanceProfile",
    "PsychedelicExperience",
    "EntheogenicCeremony",
    "TherapeuticProtocol",
    "ExperiencePhenomenology",
    "PsychedelicConsciousnessMaturityState",
    # Interface
    "PsychedelicConsciousnessInterface",
]
