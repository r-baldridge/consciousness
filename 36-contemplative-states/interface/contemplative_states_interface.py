#!/usr/bin/env python3
"""
Contemplative & Meditative States Interface

Form 36: The comprehensive interface for contemplative states, meditative practices,
and the scientific study of altered states of consciousness across diverse wisdom
traditions. This form bridges ancient contemplative wisdom with modern neuroscientific
research.

Scientific Foundation:
- Integration of first-person phenomenology with third-person neuroscience
- Neurophenomenological methodology (Varela, Lutz)
- Cross-tradition comparative analysis

Ethical Principles:
- Respect for source traditions
- Scientific integrity in representing findings
- Recognition of practice requirements and safety considerations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ContemplativeState(Enum):
    """
    Primary contemplative and meditative states across traditions.

    States are organized by type without implying hierarchy across traditions.
    Each state has specific phenomenological characteristics and entry conditions.
    """

    # === CONCENTRATION STATES ===
    ACCESS_CONCENTRATION = "access_concentration"  # Upacara samadhi
    FIRST_JHANA = "first_jhana"  # With vitakka, vicara, piti, sukha
    SECOND_JHANA = "second_jhana"  # Without vitakka/vicara, with piti, sukha
    THIRD_JHANA = "third_jhana"  # Without piti, with sukha
    FOURTH_JHANA = "fourth_jhana"  # Upekkha and one-pointedness

    # === FORMLESS JHANAS ===
    FIFTH_JHANA_INFINITE_SPACE = "fifth_jhana_infinite_space"  # Akasanancayatana
    SIXTH_JHANA_INFINITE_CONSCIOUSNESS = "sixth_jhana_infinite_consciousness"  # Vinnanancayatana
    SEVENTH_JHANA_NOTHINGNESS = "seventh_jhana_nothingness"  # Akincannayatana
    EIGHTH_JHANA_NEITHER_PERCEPTION = "eighth_jhana_neither_perception"  # Nevasannanasannayatana

    # === CESSATION ===
    CESSATION = "cessation"  # Nirodha samapatti

    # === SAMADHI STATES (HINDU YOGIC) ===
    SAMADHI = "samadhi"  # General absorption state

    # === ZEN AWAKENING ===
    KENSHO = "kensho"  # Initial seeing of true nature
    SATORI = "satori"  # Full awakening

    # === ADVAITA VEDANTA ===
    TURIYA = "turiya"  # Fourth state / witness consciousness

    # === SUFI ===
    FANA = "fana"  # Annihilation of self

    # === CROSS-TRADITION STATES ===
    UNITIVE_STATE = "unitive_state"  # Mystical unity experience
    WITNESS_CONSCIOUSNESS = "witness_consciousness"  # Pure observing awareness
    FLOW_STATE = "flow_state"  # Csikszentmihalyi flow
    PURE_CONSCIOUSNESS = "pure_consciousness"  # Contentless awareness


class ContemplativeTradition(Enum):
    """
    Major contemplative and meditative traditions.

    Each tradition has distinctive practices, maps of states,
    and philosophical frameworks for understanding experience.
    """

    THERAVADA_BUDDHIST = "theravada_buddhist"  # Pali Canon tradition
    ZEN_BUDDHIST = "zen_buddhist"  # Chan/Zen tradition
    TIBETAN_BUDDHIST = "tibetan_buddhist"  # Vajrayana, Dzogchen, Mahamudra
    HINDU_YOGIC = "hindu_yogic"  # Classical Yoga, Tantra
    ADVAITA_VEDANTA = "advaita_vedanta"  # Non-dual Vedanta
    CHRISTIAN_CARMELITE = "christian_carmelite"  # Teresa of Avila, John of the Cross
    CHRISTIAN_HESYCHAST = "christian_hesychast"  # Orthodox Christian mysticism
    SUFI_ISLAMIC = "sufi_islamic"  # Islamic mysticism
    JEWISH_KABBALISTIC = "jewish_kabbalistic"  # Jewish mysticism
    TAOIST = "taoist"  # Internal alchemy, Neidan
    SECULAR_MINDFULNESS = "secular_mindfulness"  # MBSR, MBCT


class PhenomenologicalQuality(Enum):
    """
    Phenomenological dimensions of contemplative experience.

    These qualities can be present in varying degrees across
    different contemplative states and traditions.
    """

    SPACIOUSNESS = "spaciousness"  # Sense of expanded awareness
    LUMINOSITY = "luminosity"  # Inner light or brightness
    STILLNESS = "stillness"  # Absence of mental movement
    BLISS = "bliss"  # Sukha, ananda - profound well-being
    EQUANIMITY = "equanimity"  # Upekkha - balanced awareness
    UNITY = "unity"  # Non-dual, oneness experience
    TIMELESSNESS = "timelessness"  # Transcendence of temporal sense
    SELFLESSNESS = "selflessness"  # Absence of self-reference
    CLARITY = "clarity"  # Mental brightness and precision
    PEACE = "peace"  # Profound tranquility


class NeuralCorrelate(Enum):
    """
    Neural correlates of contemplative states from research.

    These represent findings from neuroscience studies of
    meditation and contemplative practice.
    """

    GAMMA_WAVES = "gamma_waves"  # High-frequency oscillations (25-100 Hz)
    DMN_DEACTIVATION = "dmn_deactivation"  # Default mode network suppression
    ANTERIOR_CINGULATE = "anterior_cingulate"  # ACC activation/changes
    INSULA_ACTIVATION = "insula_activation"  # Interoceptive awareness
    THETA_INCREASE = "theta_increase"  # 4-8 Hz increase
    ALPHA_COHERENCE = "alpha_coherence"  # Alpha band synchrony


class PracticeType(Enum):
    """Types of meditation and contemplative practice."""

    CONCENTRATION = "concentration"  # Samatha, focused attention
    INSIGHT = "insight"  # Vipassana, investigative awareness
    OPEN_AWARENESS = "open_awareness"  # Shikantaza, choiceless awareness
    DEVOTIONAL = "devotional"  # Bhakti, prayer-based
    MOVEMENT = "movement"  # Tai chi, walking meditation
    VISUALIZATION = "visualization"  # Deity yoga, guided imagery
    MANTRA = "mantra"  # Repetition-based practice


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
class ContemplativeStateProfile:
    """
    Comprehensive profile of a contemplative state.

    Includes phenomenological description, prerequisites,
    neural correlates, and progression path information.
    """
    state_id: str
    state: ContemplativeState
    tradition_origin: ContemplativeTradition
    phenomenology: List[PhenomenologicalQuality]
    prerequisites: List[str] = field(default_factory=list)
    typical_duration: str = "variable"
    neural_correlates: List[NeuralCorrelate] = field(default_factory=list)
    progression_path: List[str] = field(default_factory=list)
    description: str = ""
    entry_methods: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    classical_references: List[str] = field(default_factory=list)
    research_references: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"State: {self.state.value}",
            f"Tradition: {self.tradition_origin.value}",
            f"Phenomenology: {', '.join(p.value for p in self.phenomenology)}",
            f"Description: {self.description}"
        ]
        return " | ".join(parts)


@dataclass
class MeditationSession:
    """
    Record of a single meditation session with states accessed
    and phenomenological observations.
    """
    session_id: str
    practitioner_id: str
    practice_type: PracticeType
    duration: int  # minutes
    states_accessed: List[ContemplativeState] = field(default_factory=list)
    phenomenological_report: str = ""
    qualities_experienced: List[PhenomenologicalQuality] = field(default_factory=list)
    tradition: Optional[ContemplativeTradition] = None
    technique: str = ""
    depth_rating: float = 0.5  # 0.0 to 1.0
    stability_rating: float = 0.5  # 0.0 to 1.0
    clarity_rating: float = 0.5  # 0.0 to 1.0
    notes: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class TraditionProfile:
    """
    Comprehensive profile of a contemplative tradition.

    Includes history, key practices, state maps, and
    notable teachers/texts.
    """
    tradition_id: str
    tradition: ContemplativeTradition
    origin: str
    key_practices: List[str] = field(default_factory=list)
    state_maps: List[str] = field(default_factory=list)  # Progression systems
    notable_teachers: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    description: str = ""
    philosophy: str = ""
    entry_requirements: List[str] = field(default_factory=list)
    contemporary_centers: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT


@dataclass
class NeuralFinding:
    """
    A neuroscience research finding related to contemplative states.

    Links scientific studies to specific states and correlates.
    """
    finding_id: str
    state: ContemplativeState
    neural_correlate: NeuralCorrelate
    study_reference: str
    methodology: str  # fMRI, EEG, SPECT, etc.
    sample_size: int
    key_result: str
    effect_size: Optional[float] = None
    replication_status: str = "unreplicated"  # unreplicated, replicated, contested
    year: int = 0
    researchers: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    doi: Optional[str] = None


@dataclass
class ContemplativeStatesMaturityState:
    """Tracks the maturity of contemplative states knowledge."""
    overall_maturity: float = 0.0
    tradition_coverage: Dict[str, float] = field(default_factory=dict)
    state_profile_count: int = 0
    session_count: int = 0
    tradition_count: int = 0
    finding_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class ContemplativeStatesInterface:
    """
    Main interface for Form 36: Contemplative & Meditative States.

    Provides methods for storing, retrieving, and querying contemplative
    states, meditation sessions, tradition profiles, and neuroscience
    findings across diverse wisdom traditions.
    """

    FORM_ID = "36-contemplative-states"
    FORM_NAME = "Contemplative & Meditative States"

    def __init__(self):
        """Initialize the Contemplative States Interface."""
        # Knowledge indexes
        self.state_profile_index: Dict[str, ContemplativeStateProfile] = {}
        self.session_index: Dict[str, MeditationSession] = {}
        self.tradition_index: Dict[str, TraditionProfile] = {}
        self.finding_index: Dict[str, NeuralFinding] = {}

        # Cross-reference indexes
        self.state_index: Dict[ContemplativeState, List[str]] = {}
        self.tradition_state_index: Dict[ContemplativeTradition, List[str]] = {}
        self.correlate_index: Dict[NeuralCorrelate, List[str]] = {}

        # Maturity tracking
        self.maturity_state = ContemplativeStatesMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize state index
        for state in ContemplativeState:
            self.state_index[state] = []

        # Initialize tradition index
        for tradition in ContemplativeTradition:
            self.tradition_state_index[tradition] = []

        # Initialize correlate index
        for correlate in NeuralCorrelate:
            self.correlate_index[correlate] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # STATE PROFILE METHODS
    # ========================================================================

    async def add_state_profile(self, profile: ContemplativeStateProfile) -> None:
        """Add a contemplative state profile to the index."""
        self.state_profile_index[profile.state_id] = profile

        # Update state index
        if profile.state in self.state_index:
            self.state_index[profile.state].append(profile.state_id)

        # Update tradition index
        if profile.tradition_origin in self.tradition_state_index:
            self.tradition_state_index[profile.tradition_origin].append(profile.state_id)

        # Update correlate index
        for correlate in profile.neural_correlates:
            if correlate in self.correlate_index:
                self.correlate_index[correlate].append(profile.state_id)

        # Update maturity
        self.maturity_state.state_profile_count = len(self.state_profile_index)
        await self._update_maturity()

    async def get_state_profile(self, state_id: str) -> Optional[ContemplativeStateProfile]:
        """Retrieve a state profile by ID."""
        return self.state_profile_index.get(state_id)

    async def query_profiles_by_state(
        self,
        state: ContemplativeState,
        limit: int = 10
    ) -> List[ContemplativeStateProfile]:
        """Query profiles by contemplative state."""
        profile_ids = self.state_index.get(state, [])[:limit]
        return [
            self.state_profile_index[pid]
            for pid in profile_ids
            if pid in self.state_profile_index
        ]

    async def query_profiles_by_tradition(
        self,
        tradition: ContemplativeTradition,
        limit: int = 10
    ) -> List[ContemplativeStateProfile]:
        """Query profiles by tradition."""
        profile_ids = self.tradition_state_index.get(tradition, [])[:limit]
        return [
            self.state_profile_index[pid]
            for pid in profile_ids
            if pid in self.state_profile_index
        ]

    async def query_profiles_by_correlate(
        self,
        correlate: NeuralCorrelate,
        limit: int = 10
    ) -> List[ContemplativeStateProfile]:
        """Query profiles by neural correlate."""
        profile_ids = self.correlate_index.get(correlate, [])[:limit]
        return [
            self.state_profile_index[pid]
            for pid in profile_ids
            if pid in self.state_profile_index
        ]

    async def get_state_progression(
        self,
        tradition: ContemplativeTradition
    ) -> List[ContemplativeStateProfile]:
        """Get ordered progression of states in a tradition."""
        profiles = await self.query_profiles_by_tradition(tradition, limit=100)
        # Sort by progression path length (simpler states first)
        return sorted(profiles, key=lambda p: len(p.prerequisites))

    # ========================================================================
    # SESSION METHODS
    # ========================================================================

    async def add_session(self, session: MeditationSession) -> None:
        """Add a meditation session to the index."""
        self.session_index[session.session_id] = session

        # Update maturity
        self.maturity_state.session_count = len(self.session_index)
        await self._update_maturity()

    async def get_session(self, session_id: str) -> Optional[MeditationSession]:
        """Retrieve a session by ID."""
        return self.session_index.get(session_id)

    async def query_sessions_by_practitioner(
        self,
        practitioner_id: str,
        limit: int = 50
    ) -> List[MeditationSession]:
        """Query sessions by practitioner."""
        sessions = [
            s for s in self.session_index.values()
            if s.practitioner_id == practitioner_id
        ]
        sessions.sort(key=lambda s: s.timestamp or datetime.min, reverse=True)
        return sessions[:limit]

    # ========================================================================
    # TRADITION METHODS
    # ========================================================================

    async def add_tradition(self, tradition: TraditionProfile) -> None:
        """Add a tradition profile to the index."""
        self.tradition_index[tradition.tradition_id] = tradition

        # Update maturity
        self.maturity_state.tradition_count = len(self.tradition_index)
        await self._update_maturity()

    async def get_tradition(self, tradition_id: str) -> Optional[TraditionProfile]:
        """Retrieve a tradition by ID."""
        return self.tradition_index.get(tradition_id)

    async def get_all_traditions(self) -> List[TraditionProfile]:
        """Get all tradition profiles."""
        return list(self.tradition_index.values())

    # ========================================================================
    # NEURAL FINDING METHODS
    # ========================================================================

    async def add_finding(self, finding: NeuralFinding) -> None:
        """Add a neural finding to the index."""
        self.finding_index[finding.finding_id] = finding

        # Update maturity
        self.maturity_state.finding_count = len(self.finding_index)
        await self._update_maturity()

    async def get_finding(self, finding_id: str) -> Optional[NeuralFinding]:
        """Retrieve a finding by ID."""
        return self.finding_index.get(finding_id)

    async def query_findings_by_state(
        self,
        state: ContemplativeState,
        limit: int = 20
    ) -> List[NeuralFinding]:
        """Query findings by contemplative state."""
        findings = [
            f for f in self.finding_index.values()
            if f.state == state
        ]
        return findings[:limit]

    async def query_findings_by_correlate(
        self,
        correlate: NeuralCorrelate,
        limit: int = 20
    ) -> List[NeuralFinding]:
        """Query findings by neural correlate."""
        findings = [
            f for f in self.finding_index.values()
            if f.neural_correlate == correlate
        ]
        return findings[:limit]

    # ========================================================================
    # CROSS-TRADITION ANALYSIS
    # ========================================================================

    async def find_equivalent_states(
        self,
        state_id: str,
        min_overlap: float = 0.5
    ) -> List[Tuple[ContemplativeStateProfile, float]]:
        """Find equivalent states across traditions with similarity score."""
        profile = await self.get_state_profile(state_id)
        if not profile:
            return []

        results = []
        source_qualities = set(profile.phenomenology)

        for other_id, other_profile in self.state_profile_index.items():
            if other_id == state_id:
                continue

            other_qualities = set(other_profile.phenomenology)
            if not source_qualities or not other_qualities:
                continue

            # Calculate Jaccard similarity
            intersection = len(source_qualities & other_qualities)
            union = len(source_qualities | other_qualities)
            similarity = intersection / union if union > 0 else 0

            if similarity >= min_overlap:
                results.append((other_profile, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def get_common_phenomenology(
        self,
        state_ids: List[str]
    ) -> List[PhenomenologicalQuality]:
        """Get phenomenological qualities common to multiple states."""
        if not state_ids:
            return []

        profiles = [
            await self.get_state_profile(sid)
            for sid in state_ids
        ]
        profiles = [p for p in profiles if p is not None]

        if not profiles:
            return []

        # Find intersection of all phenomenology sets
        common = set(profiles[0].phenomenology)
        for profile in profiles[1:]:
            common &= set(profile.phenomenology)

        return list(common)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.state_profile_count +
            self.maturity_state.tradition_count +
            self.maturity_state.finding_count
        )

        # Simple maturity calculation
        target_items = 200  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update tradition coverage
        for tradition in ContemplativeTradition:
            count = len(self.tradition_state_index.get(tradition, []))
            target_per_tradition = 10
            self.maturity_state.tradition_coverage[tradition.value] = min(
                1.0, count / target_per_tradition
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> ContemplativeStatesMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_state_profiles(self) -> List[Dict[str, Any]]:
        """Return seed state profiles for initialization."""
        return [
            # === JHANA STATES ===
            {
                "state_id": "access_concentration_theravada",
                "state": ContemplativeState.ACCESS_CONCENTRATION,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.STILLNESS,
                    PhenomenologicalQuality.CLARITY,
                ],
                "description": "Access concentration (upacara samadhi) is the preparatory stage preceding full meditative absorption. The five hindrances are suppressed but absorption factors are not yet firmly established. The counterpart sign (patibhaga nimitta) appears.",
                "prerequisites": [],
                "typical_duration": "minutes to hours",
                "entry_methods": ["focused attention on meditation object", "anapanasati (breath)"],
                "neural_correlates": [NeuralCorrelate.ALPHA_COHERENCE],
                "progression_path": ["first_jhana_theravada"],
            },
            {
                "state_id": "first_jhana_theravada",
                "state": ContemplativeState.FIRST_JHANA,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.BLISS,
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.STILLNESS,
                ],
                "description": "First jhana with five factors: applied thought (vitakka), sustained thought (vicara), rapture (piti), happiness (sukha), and one-pointedness (ekaggata). Seclusion from sensory desires and unwholesome states while thinking and examining still present.",
                "prerequisites": ["access_concentration_theravada"],
                "typical_duration": "minutes to hours",
                "entry_methods": ["deepening concentration from access", "absorption into nimitta"],
                "neural_correlates": [NeuralCorrelate.GAMMA_WAVES, NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["second_jhana_theravada"],
            },
            {
                "state_id": "second_jhana_theravada",
                "state": ContemplativeState.SECOND_JHANA,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.BLISS,
                    PhenomenologicalQuality.PEACE,
                    PhenomenologicalQuality.STILLNESS,
                ],
                "description": "Second jhana with three factors: rapture (piti), happiness (sukha), one-pointedness (ekaggata). Stilling of applied and sustained thought brings internal tranquility and noble silence. Joy without conceptual overlay.",
                "prerequisites": ["first_jhana_theravada"],
                "typical_duration": "minutes to hours",
                "entry_methods": ["letting go of thinking within first jhana"],
                "neural_correlates": [NeuralCorrelate.GAMMA_WAVES, NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["third_jhana_theravada"],
            },
            {
                "state_id": "third_jhana_theravada",
                "state": ContemplativeState.THIRD_JHANA,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.PEACE,
                    PhenomenologicalQuality.EQUANIMITY,
                    PhenomenologicalQuality.CLARITY,
                ],
                "description": "Third jhana with two factors: happiness (sukha), one-pointedness (ekaggata). Rapture fades, replaced by contentment, mindfulness, and clear comprehension. Equanimity toward pleasure develops.",
                "prerequisites": ["second_jhana_theravada"],
                "typical_duration": "minutes to hours",
                "entry_methods": ["releasing attachment to rapture"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION, NeuralCorrelate.ANTERIOR_CINGULATE],
                "progression_path": ["fourth_jhana_theravada"],
            },
            {
                "state_id": "fourth_jhana_theravada",
                "state": ContemplativeState.FOURTH_JHANA,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.EQUANIMITY,
                    PhenomenologicalQuality.STILLNESS,
                    PhenomenologicalQuality.CLARITY,
                ],
                "description": "Fourth jhana with two factors: equanimity (upekkha), one-pointedness (ekaggata). Neither pleasure nor pain; purified equanimity and mindfulness. Profound stillness where breath may become imperceptible.",
                "prerequisites": ["third_jhana_theravada"],
                "typical_duration": "minutes to hours",
                "entry_methods": ["releasing attachment to sukha"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION, NeuralCorrelate.THETA_INCREASE],
                "progression_path": ["fifth_jhana_theravada"],
            },
            {
                "state_id": "fifth_jhana_theravada",
                "state": ContemplativeState.FIFTH_JHANA_INFINITE_SPACE,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.SPACIOUSNESS,
                    PhenomenologicalQuality.EQUANIMITY,
                    PhenomenologicalQuality.STILLNESS,
                ],
                "description": "Fifth jhana - Sphere of Infinite Space (Akasanancayatana). Transcendence of perception of form. Attention to boundless space. Complete release from material perception.",
                "prerequisites": ["fourth_jhana_theravada"],
                "typical_duration": "variable",
                "entry_methods": ["expanding attention beyond form"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["sixth_jhana_theravada"],
            },
            {
                "state_id": "sixth_jhana_theravada",
                "state": ContemplativeState.SIXTH_JHANA_INFINITE_CONSCIOUSNESS,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.SPACIOUSNESS,
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.SELFLESSNESS,
                ],
                "description": "Sixth jhana - Sphere of Infinite Consciousness (Vinnanancayatana). Attention shifts from space to the consciousness that knows space. Awareness of boundless consciousness itself.",
                "prerequisites": ["fifth_jhana_theravada"],
                "typical_duration": "variable",
                "entry_methods": ["attending to consciousness knowing space"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["seventh_jhana_theravada"],
            },
            {
                "state_id": "seventh_jhana_theravada",
                "state": ContemplativeState.SEVENTH_JHANA_NOTHINGNESS,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.STILLNESS,
                    PhenomenologicalQuality.SELFLESSNESS,
                    PhenomenologicalQuality.PEACE,
                ],
                "description": "Seventh jhana - Sphere of Nothingness (Akincannayatana). Contemplation of 'there is nothing'. Subtle perception of non-existence of objects. State reached by Buddha's teacher Alara Kalama.",
                "prerequisites": ["sixth_jhana_theravada"],
                "typical_duration": "variable",
                "entry_methods": ["contemplating absence of objects"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["eighth_jhana_theravada"],
            },
            {
                "state_id": "eighth_jhana_theravada",
                "state": ContemplativeState.EIGHTH_JHANA_NEITHER_PERCEPTION,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.STILLNESS,
                    PhenomenologicalQuality.SELFLESSNESS,
                ],
                "description": "Eighth jhana - Sphere of Neither Perception nor Non-Perception (Nevasannanasannayatana). Beyond negation of perception. Liminal state: not engaged in perception, not wholly unconscious. State reached by Buddha's teacher Uddaka Ramaputta. Most refined state of ordinary consciousness.",
                "prerequisites": ["seventh_jhana_theravada"],
                "typical_duration": "variable",
                "entry_methods": ["transcending nothingness"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["cessation_theravada"],
            },
            {
                "state_id": "cessation_theravada",
                "state": ContemplativeState.CESSATION,
                "tradition_origin": ContemplativeTradition.THERAVADA_BUDDHIST,
                "phenomenology": [],  # No phenomenology - consciousness ceases
                "description": "Nirodha Samapatti - the highest meditative attainment. Complete cessation of all mental activity, sensations, and awareness. Total absence of conscious experience. Can last up to 7 days according to tradition. Practitioners report profound reset upon emerging.",
                "prerequisites": ["eighth_jhana_theravada"],
                "typical_duration": "minutes to days",
                "entry_methods": ["sequential jhana mastery", "insight into cessation"],
                "neural_correlates": [],  # Research ongoing
                "progression_path": [],
            },

            # === ZEN STATES ===
            {
                "state_id": "kensho_zen",
                "state": ContemplativeState.KENSHO,
                "tradition_origin": ContemplativeTradition.ZEN_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.UNITY,
                    PhenomenologicalQuality.SELFLESSNESS,
                ],
                "description": "Kensho - 'seeing nature' or seeing one's true nature. Direct realization that self has no fixed or independent essence. Experiential, not conceptual understanding. Often initial or partial awakening, usually spontaneous and sudden. Ego ceases to exist temporarily.",
                "prerequisites": [],
                "typical_duration": "momentary",
                "entry_methods": ["koan practice", "shikantaza", "spontaneous"],
                "neural_correlates": [NeuralCorrelate.GAMMA_WAVES, NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["satori_zen"],
            },
            {
                "state_id": "satori_zen",
                "state": ContemplativeState.SATORI,
                "tradition_origin": ContemplativeTradition.ZEN_BUDDHIST,
                "phenomenology": [
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.UNITY,
                    PhenomenologicalQuality.SELFLESSNESS,
                    PhenomenologicalQuality.PEACE,
                ],
                "description": "Satori - 'awakening' or 'comprehension'. Full, deep enlightenment like Buddha's. Transcends distinction between knower and knowledge. Blissful realization of 'originally pure mind'. Illuminating emptiness, dynamic thusness.",
                "prerequisites": ["kensho_zen"],
                "typical_duration": "variable",
                "entry_methods": ["deepening of kensho", "ongoing practice"],
                "neural_correlates": [NeuralCorrelate.GAMMA_WAVES, NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": [],
            },

            # === ADVAITA VEDANTA ===
            {
                "state_id": "turiya_advaita",
                "state": ContemplativeState.TURIYA,
                "tradition_origin": ContemplativeTradition.ADVAITA_VEDANTA,
                "phenomenology": [
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.PEACE,
                    PhenomenologicalQuality.SELFLESSNESS,
                    PhenomenologicalQuality.LUMINOSITY,
                ],
                "description": "Turiya - 'the fourth'. The true self (atman) beyond the three common states (waking, dreaming, deep sleep). Not truly a fourth state but the substrate underlying all three. The ever-present witness behind all experience. Silent ground of being, pure consciousness undefiled by contents.",
                "prerequisites": [],
                "typical_duration": "variable to permanent",
                "entry_methods": ["self-inquiry (atma vichara)", "neti neti", "meditation"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION, NeuralCorrelate.ALPHA_COHERENCE],
                "progression_path": [],
            },
            {
                "state_id": "witness_consciousness_advaita",
                "state": ContemplativeState.WITNESS_CONSCIOUSNESS,
                "tradition_origin": ContemplativeTradition.ADVAITA_VEDANTA,
                "phenomenology": [
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.EQUANIMITY,
                    PhenomenologicalQuality.STILLNESS,
                ],
                "description": "Sakshi - witness consciousness that observes three states without identification. Detached observation with equanimity toward contents of awareness. Foundation for self-inquiry practice. Unchanging awareness behind changing experiences.",
                "prerequisites": [],
                "typical_duration": "variable",
                "entry_methods": ["self-inquiry", "observing without reaction"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": ["turiya_advaita"],
            },

            # === SUFI ===
            {
                "state_id": "fana_sufi",
                "state": ContemplativeState.FANA,
                "tradition_origin": ContemplativeTradition.SUFI_ISLAMIC,
                "phenomenology": [
                    PhenomenologicalQuality.UNITY,
                    PhenomenologicalQuality.SELFLESSNESS,
                    PhenomenologicalQuality.BLISS,
                ],
                "description": "Fana - 'passing away' or 'annihilation' of the self. Complete denial of self and realization of God. 'To die before one dies.' Loss of awareness of earthly existence. Step toward achieving union with God.",
                "prerequisites": [],
                "typical_duration": "variable",
                "entry_methods": ["dhikr (remembrance)", "devotion", "surrender"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": [],
            },

            # === SAMADHI (HINDU YOGIC) ===
            {
                "state_id": "samadhi_yogic",
                "state": ContemplativeState.SAMADHI,
                "tradition_origin": ContemplativeTradition.HINDU_YOGIC,
                "phenomenology": [
                    PhenomenologicalQuality.UNITY,
                    PhenomenologicalQuality.BLISS,
                    PhenomenologicalQuality.STILLNESS,
                    PhenomenologicalQuality.CLARITY,
                ],
                "description": "Samadhi - the eighth limb of Patanjali's yoga. Complete absorption where body, mind, and object become one. Loss of subject-object duality. Includes varieties: savikalpa (with distinctions), nirvikalpa (without distinctions), dharmamegha (cloud of dharma).",
                "prerequisites": [],
                "typical_duration": "variable",
                "entry_methods": ["dharana-dhyana progression", "pranayama", "pratyahara"],
                "neural_correlates": [NeuralCorrelate.GAMMA_WAVES, NeuralCorrelate.THETA_INCREASE],
                "progression_path": [],
            },

            # === CROSS-TRADITION ===
            {
                "state_id": "unitive_state_general",
                "state": ContemplativeState.UNITIVE_STATE,
                "tradition_origin": ContemplativeTradition.SECULAR_MINDFULNESS,
                "phenomenology": [
                    PhenomenologicalQuality.UNITY,
                    PhenomenologicalQuality.PEACE,
                    PhenomenologicalQuality.BLISS,
                    PhenomenologicalQuality.TIMELESSNESS,
                ],
                "description": "Unitive states characterized by sense of unity or 'oneness' transcending ordinary subject-object duality. Simple unity or pure consciousness. Non-duality of observer and observed. Feelings of sacredness, peace, and bliss. Cross-tradition phenomenon.",
                "prerequisites": [],
                "typical_duration": "variable",
                "entry_methods": ["various meditation practices", "spontaneous"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION, NeuralCorrelate.GAMMA_WAVES],
                "progression_path": [],
            },
            {
                "state_id": "flow_state_general",
                "state": ContemplativeState.FLOW_STATE,
                "tradition_origin": ContemplativeTradition.SECULAR_MINDFULNESS,
                "phenomenology": [
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.TIMELESSNESS,
                    PhenomenologicalQuality.SELFLESSNESS,
                ],
                "description": "Flow state (Csikszentmihalyi) - complete absorption in activity. Intense concentration on present moment, merging of action and awareness, loss of self-consciousness, sense of control, distorted time sense, intrinsic reward. 'Active, moving meditation'.",
                "prerequisites": [],
                "typical_duration": "minutes to hours",
                "entry_methods": ["engaging activity with challenge-skill balance"],
                "neural_correlates": [NeuralCorrelate.ANTERIOR_CINGULATE, NeuralCorrelate.DMN_DEACTIVATION],
                "progression_path": [],
            },
            {
                "state_id": "pure_consciousness_general",
                "state": ContemplativeState.PURE_CONSCIOUSNESS,
                "tradition_origin": ContemplativeTradition.SECULAR_MINDFULNESS,
                "phenomenology": [
                    PhenomenologicalQuality.CLARITY,
                    PhenomenologicalQuality.STILLNESS,
                    PhenomenologicalQuality.LUMINOSITY,
                ],
                "description": "Pure consciousness - awareness without intentional content. Consciousness without objects of consciousness. Self-luminous quality, timeless or eternal quality. Often described as 'empty' yet 'full'. Distinct from unconsciousness.",
                "prerequisites": [],
                "typical_duration": "variable",
                "entry_methods": ["deep meditation", "spontaneous"],
                "neural_correlates": [NeuralCorrelate.DMN_DEACTIVATION, NeuralCorrelate.ALPHA_COHERENCE],
                "progression_path": [],
            },
        ]

    def _get_seed_traditions(self) -> List[Dict[str, Any]]:
        """Return seed tradition profiles for initialization."""
        return [
            {
                "tradition_id": "theravada_buddhist",
                "tradition": ContemplativeTradition.THERAVADA_BUDDHIST,
                "origin": "Sri Lanka, Myanmar, Thailand, Cambodia, Laos - based on Pali Canon",
                "key_practices": [
                    "Samatha (tranquility meditation)",
                    "Vipassana (insight meditation)",
                    "Anapanasati (breath mindfulness)",
                    "Kasina practices",
                    "Four foundations of mindfulness",
                    "Body scanning (Goenka tradition)",
                    "Noting practice (Mahasi tradition)",
                ],
                "state_maps": ["Jhana progression (1-8 + cessation)", "Progress of Insight (16 nanas)"],
                "notable_teachers": [
                    "Mahasi Sayadaw",
                    "S.N. Goenka",
                    "U Ba Khin",
                    "Ajahn Chah",
                    "Ajahn Brahm",
                    "Pa Auk Sayadaw",
                ],
                "texts": ["Visuddhimagga", "Satipatthana Sutta", "Anapanasati Sutta"],
                "description": "Teaching of the Elders - oldest surviving Buddhist school emphasizing meditation leading to nibbana through development of concentration and insight.",
            },
            {
                "tradition_id": "zen_buddhist",
                "tradition": ContemplativeTradition.ZEN_BUDDHIST,
                "origin": "China (Chan, 6th century) transmitted to Japan (Zen, 12th century)",
                "key_practices": [
                    "Zazen (seated meditation)",
                    "Shikantaza (just sitting)",
                    "Koan practice",
                    "Kinhin (walking meditation)",
                    "Samu (work practice)",
                    "Sesshin (intensive retreats)",
                ],
                "state_maps": ["Ten Ox-Herding Pictures", "Koan progression"],
                "notable_teachers": [
                    "Bodhidharma",
                    "Dogen Zenji",
                    "Hakuin Ekaku",
                    "Shunryu Suzuki",
                    "Thich Nhat Hanh",
                ],
                "texts": ["Platform Sutra", "Shobogenzo", "Mumonkan", "Blue Cliff Record"],
                "description": "Direct pointing to the nature of mind through meditation and koan study. Emphasis on sudden awakening (kensho/satori) and integration of practice with daily life.",
            },
            {
                "tradition_id": "tibetan_buddhist",
                "tradition": ContemplativeTradition.TIBETAN_BUDDHIST,
                "origin": "Tibet (7th century onwards) - synthesis of Indian Buddhism with native Bon",
                "key_practices": [
                    "Ngondro (preliminary practices)",
                    "Shamatha and vipashyana",
                    "Deity yoga",
                    "Tonglen (giving and taking)",
                    "Guru yoga",
                    "Dzogchen (Trekcho, Togal)",
                    "Mahamudra",
                    "Mantra recitation",
                ],
                "state_maps": ["Dzogchen rigpa progression", "Four yogas of Mahamudra", "Bardo states"],
                "notable_teachers": [
                    "Padmasambhava",
                    "Milarepa",
                    "Longchenpa",
                    "Tsongkhapa",
                    "Dalai Lama",
                    "Dilgo Khyentse",
                ],
                "texts": ["Tibetan Book of the Dead", "Jewel Ornament of Liberation", "Treasury of Precious Qualities"],
                "description": "Vajrayana Buddhism incorporating tantric methods, deity yoga, and direct pointing traditions (Dzogchen, Mahamudra) for rapid awakening.",
            },
            {
                "tradition_id": "hindu_yogic",
                "tradition": ContemplativeTradition.HINDU_YOGIC,
                "origin": "India - systematized in Yoga Sutras of Patanjali (c. 200 BCE-200 CE)",
                "key_practices": [
                    "Raja Yoga (eight limbs)",
                    "Pranayama (breath control)",
                    "Dharana (concentration)",
                    "Dhyana (meditation)",
                    "Pratyahara (sense withdrawal)",
                    "Kundalini Yoga",
                    "Hatha Yoga",
                ],
                "state_maps": ["Eight limbs of yoga", "Samadhi varieties"],
                "notable_teachers": [
                    "Patanjali",
                    "Swami Vivekananda",
                    "Paramahansa Yogananda",
                    "B.K.S. Iyengar",
                    "Krishnamacharya",
                ],
                "texts": ["Yoga Sutras", "Hatha Yoga Pradipika", "Bhagavad Gita"],
                "description": "Classical yoga system based on Patanjali's eight limbs leading to kaivalya (liberation) through progressive refinement of body, breath, and mind.",
            },
            {
                "tradition_id": "advaita_vedanta",
                "tradition": ContemplativeTradition.ADVAITA_VEDANTA,
                "origin": "India - systematized by Adi Shankaracharya (8th century CE)",
                "key_practices": [
                    "Self-inquiry (Atma Vichara)",
                    "Neti Neti (not this, not this)",
                    "Sravana, Manana, Nididhyasana",
                    "Meditation on 'I am'",
                ],
                "state_maps": ["Recognition of turiya", "Jivanmukti (living liberation)"],
                "notable_teachers": [
                    "Adi Shankaracharya",
                    "Ramana Maharshi",
                    "Nisargadatta Maharaj",
                    "Papaji",
                    "Ramesh Balsekar",
                ],
                "texts": ["Upanishads", "Vivekachudamani", "I Am That", "Who Am I?"],
                "description": "Non-dual philosophy recognizing atman (individual self) as identical with Brahman (universal Self). Practice centers on self-inquiry and discrimination between real and unreal.",
            },
            {
                "tradition_id": "christian_carmelite",
                "tradition": ContemplativeTradition.CHRISTIAN_CARMELITE,
                "origin": "Spain (16th century) - Carmelite reform movement",
                "key_practices": [
                    "Mental prayer",
                    "Contemplative prayer",
                    "Lectio divina",
                    "Examination of conscience",
                    "Purification through dark nights",
                ],
                "state_maps": ["Seven Mansions (Teresa)", "Dark Night of Soul (John)"],
                "notable_teachers": [
                    "Teresa of Avila",
                    "John of the Cross",
                    "Therese of Lisieux",
                    "Edith Stein",
                ],
                "texts": ["Interior Castle", "Dark Night of the Soul", "Ascent of Mount Carmel", "Living Flame of Love"],
                "description": "Christian contemplative tradition emphasizing progressive prayer leading through purification to transforming union with God.",
            },
            {
                "tradition_id": "christian_hesychast",
                "tradition": ContemplativeTradition.CHRISTIAN_HESYCHAST,
                "origin": "Eastern Christianity - Desert Fathers (3rd-7th century), codified Byzantine period",
                "key_practices": [
                    "Jesus Prayer",
                    "Hesychia (stillness)",
                    "Psychophysical method",
                    "Prayer of the heart",
                    "Continuous prayer",
                ],
                "state_maps": ["Theosis (deification)", "Vision of uncreated light"],
                "notable_teachers": [
                    "Anthony the Great",
                    "Evagrius Ponticus",
                    "Gregory Palamas",
                    "Seraphim of Sarov",
                ],
                "texts": ["Philokalia", "Way of a Pilgrim", "Ladder of Divine Ascent"],
                "description": "Orthodox Christian tradition seeking stillness (hesychia) through continuous Jesus Prayer, leading to theosis and vision of divine light.",
            },
            {
                "tradition_id": "sufi_islamic",
                "tradition": ContemplativeTradition.SUFI_ISLAMIC,
                "origin": "Middle East (8th century CE) - mystical dimension of Islam",
                "key_practices": [
                    "Dhikr (remembrance)",
                    "Sama (spiritual concert)",
                    "Muraqaba (meditation)",
                    "Whirling (Mevlevi)",
                    "Following a tariqa (path)",
                ],
                "state_maps": ["Maqamat (stations)", "Ahwal (states)", "Fana-Baqa progression"],
                "notable_teachers": [
                    "Rabia al-Adawiyya",
                    "Al-Hallaj",
                    "Al-Ghazali",
                    "Rumi",
                    "Ibn Arabi",
                ],
                "texts": ["Masnavi", "Ihya Ulum al-Din", "Fusus al-Hikam"],
                "description": "Islamic mystical tradition seeking direct experience of divine presence through purification, devotion, and remembrance (dhikr).",
            },
            {
                "tradition_id": "jewish_kabbalistic",
                "tradition": ContemplativeTradition.JEWISH_KABBALISTIC,
                "origin": "Jewish mysticism - Sefer Yetzirah (early CE), Zohar (13th century)",
                "key_practices": [
                    "Hitbodedut (self-seclusion)",
                    "Kavvanah (focused intention)",
                    "Letter meditation",
                    "Divine name meditation",
                    "Sefirot contemplation",
                ],
                "state_maps": ["Tree of Life (Sefirot)", "Devekut (cleaving to God)"],
                "notable_teachers": [
                    "Isaac Luria",
                    "Moses Cordovero",
                    "Baal Shem Tov",
                    "Nachman of Breslov",
                    "Abraham Abulafia",
                ],
                "texts": ["Zohar", "Sefer Yetzirah", "Tanya", "Etz Chaim"],
                "description": "Jewish mystical tradition working with divine names, letters, and sefirot to achieve devekut (cleaving) with the divine.",
            },
            {
                "tradition_id": "taoist",
                "tradition": ContemplativeTradition.TAOIST,
                "origin": "China - Lao Tzu, Chuang Tzu, internal alchemy traditions",
                "key_practices": [
                    "Zuowang (sitting and forgetting)",
                    "Shouyi (guarding the one)",
                    "Neidan (internal alchemy)",
                    "Xingqi (circulating qi)",
                    "Taixi (embryonic breathing)",
                    "Dantian cultivation",
                ],
                "state_maps": ["Three treasures refinement", "Return to Tao"],
                "notable_teachers": [
                    "Lao Tzu",
                    "Chuang Tzu",
                    "Zhang Boduan",
                    "Wang Chongyang",
                ],
                "texts": ["Tao Te Ching", "Chuang Tzu", "Secret of the Golden Flower", "Awakening to Reality"],
                "description": "Chinese tradition using body as alchemical vessel to refine essence (jing) to breath (qi) to spirit (shen) and return to primordial Tao.",
            },
            {
                "tradition_id": "secular_mindfulness",
                "tradition": ContemplativeTradition.SECULAR_MINDFULNESS,
                "origin": "USA (1979) - Jon Kabat-Zinn at University of Massachusetts",
                "key_practices": [
                    "MBSR (Mindfulness-Based Stress Reduction)",
                    "MBCT (Mindfulness-Based Cognitive Therapy)",
                    "Body scan",
                    "Sitting meditation",
                    "Mindful movement",
                    "ACT (Acceptance and Commitment Therapy)",
                ],
                "state_maps": ["Stress reduction progression", "Attention training"],
                "notable_teachers": [
                    "Jon Kabat-Zinn",
                    "Mark Williams",
                    "Zindel Segal",
                    "Tara Brach",
                    "Jack Kornfield",
                ],
                "texts": ["Full Catastrophe Living", "Wherever You Go There You Are", "The Mindful Way Through Depression"],
                "description": "Secular adaptation of Buddhist mindfulness for clinical settings, focusing on stress reduction, emotional regulation, and well-being.",
            },
        ]

    def _get_seed_findings(self) -> List[Dict[str, Any]]:
        """Return seed neural findings for initialization."""
        return [
            {
                "finding_id": "lutz_2004_gamma",
                "state": ContemplativeState.SAMADHI,
                "neural_correlate": NeuralCorrelate.GAMMA_WAVES,
                "study_reference": "Lutz et al. (2004). Long-term meditators self-induce high-amplitude gamma synchrony during mental practice. PNAS.",
                "methodology": "EEG",
                "sample_size": 8,
                "key_result": "Long-term Buddhist practitioners showed sustained high-amplitude gamma-band oscillations (25-42 Hz) and phase-synchrony during compassion meditation. Gamma activity levels were highest reported in literature in non-pathological context.",
                "replication_status": "replicated",
                "year": 2004,
                "researchers": ["Antoine Lutz", "Lawrence Greischar", "Nancy Rawlings", "Matthieu Ricard", "Richard Davidson"],
            },
            {
                "finding_id": "brewer_2011_dmn",
                "state": ContemplativeState.ACCESS_CONCENTRATION,
                "neural_correlate": NeuralCorrelate.DMN_DEACTIVATION,
                "study_reference": "Brewer et al. (2011). Meditation experience is associated with differences in default mode network activity and connectivity. PNAS.",
                "methodology": "fMRI",
                "sample_size": 12,
                "key_result": "Experienced meditators (mean 9,676 hours practice) showed decreased DMN activity across meditation types. Main nodes (mPFC and PCC) relatively deactivated. Stronger coupling between PCC, dACC, and dlPFC.",
                "replication_status": "replicated",
                "year": 2011,
                "researchers": ["Judson Brewer", "Patrick Worhunsky", "Jeremy Gray", "Yi-Yuan Tang", "Jochen Weber", "Hedy Kober"],
            },
            {
                "finding_id": "lazar_2005_cortical",
                "state": ContemplativeState.ACCESS_CONCENTRATION,
                "neural_correlate": NeuralCorrelate.INSULA_ACTIVATION,
                "study_reference": "Lazar et al. (2005). Meditation experience is associated with increased cortical thickness. NeuroReport.",
                "methodology": "MRI",
                "sample_size": 20,
                "key_result": "Regular meditation associated with increased cortical thickness in brain regions associated with attention, interoception, and sensory processing. Right anterior insula and right prefrontal cortex significantly thicker.",
                "replication_status": "replicated",
                "year": 2005,
                "researchers": ["Sara Lazar", "Catherine Kerr", "Rachel Wasserman", "Jeremy Gray", "Douglas Greve", "Michael Treadway"],
            },
            {
                "finding_id": "lutz_2008_compassion",
                "state": ContemplativeState.SAMADHI,
                "neural_correlate": NeuralCorrelate.INSULA_ACTIVATION,
                "study_reference": "Lutz et al. (2008). Regulation of the neural circuitry of emotion by compassion meditation. PLoS ONE.",
                "methodology": "fMRI",
                "sample_size": 16,
                "key_result": "During compassion meditation, expert meditators showed greater activation in limbic regions (insula, cingulate cortices) than novices when exposed to emotional sounds. Suggests enhanced empathic response.",
                "replication_status": "replicated",
                "year": 2008,
                "researchers": ["Antoine Lutz", "Julie Brefczynski-Lewis", "Tom Johnstone", "Richard Davidson"],
            },
            {
                "finding_id": "tang_2015_acc",
                "state": ContemplativeState.ACCESS_CONCENTRATION,
                "neural_correlate": NeuralCorrelate.ANTERIOR_CINGULATE,
                "study_reference": "Tang et al. (2015). Short-term meditation increases blood flow in anterior cingulate cortex and insula. Frontiers in Psychology.",
                "methodology": "fMRI",
                "sample_size": 40,
                "key_result": "Five days (30 min/day) of Integrative Body-Mind Training enhanced cerebral blood flow in subgenual/ventral ACC and bilateral insula. Associated with improved self-regulation.",
                "replication_status": "replicated",
                "year": 2015,
                "researchers": ["Yi-Yuan Tang", "Yinghua Ma", "Yaxin Fan", "Junhong Wang"],
            },
            {
                "finding_id": "mgh_2023_jhana",
                "state": ContemplativeState.FIRST_JHANA,
                "neural_correlate": NeuralCorrelate.DMN_DEACTIVATION,
                "study_reference": "Massachusetts General Hospital Jhana Study (2023-2025). 7T MRI investigation of jhana states.",
                "methodology": "7T fMRI",
                "sample_size": 1,
                "key_result": "First use of ultra-high field 7T MRI in meditation research. Adept with 25+ years experience showed distinctive patterns in cortical, subcortical, brainstem, and cerebellar regions. Reward system activation observed.",
                "replication_status": "unreplicated",
                "year": 2023,
                "researchers": ["Matthew Sacchet", "other MGH researchers"],
                "limitations": ["Single subject case study", "Requires replication with larger samples"],
            },
            {
                "finding_id": "cross_tradition_gamma",
                "state": ContemplativeState.SAMADHI,
                "neural_correlate": NeuralCorrelate.GAMMA_WAVES,
                "study_reference": "Cross-tradition gamma study - meta-analysis of meditation EEG studies.",
                "methodology": "EEG",
                "sample_size": 100,
                "key_result": "Higher parieto-occipital 60-110 Hz gamma amplitude observed across three different meditation traditions compared to controls. Gamma power positively correlated with meditation experience. Hours of practice predict baseline gamma.",
                "replication_status": "replicated",
                "year": 2015,
                "researchers": ["Multiple research groups"],
            },
            {
                "finding_id": "laukkonen_2023_cessation",
                "state": ContemplativeState.CESSATION,
                "neural_correlate": NeuralCorrelate.DMN_DEACTIVATION,
                "study_reference": "Laukkonen et al. (2023). Cessations of consciousness in meditation: Advancing a scientific understanding of nirodha samapatti. Progress in Brain Research.",
                "methodology": "Theoretical/Review",
                "sample_size": 0,
                "key_result": "First comprehensive scientific framework for understanding nirodha samapatti (cessation). Proposes integration with cognitive-neurocomputational and active inference frameworks. Practitioners impervious to external stimulation during state.",
                "replication_status": "unreplicated",
                "year": 2023,
                "researchers": ["Ruben Laukkonen", "other researchers"],
            },
        ]

    async def initialize_seed_state_profiles(self) -> int:
        """Initialize with seed state profiles."""
        seed_profiles = self._get_seed_state_profiles()
        count = 0

        for profile_data in seed_profiles:
            profile = ContemplativeStateProfile(
                state_id=profile_data["state_id"],
                state=profile_data["state"],
                tradition_origin=profile_data["tradition_origin"],
                phenomenology=profile_data["phenomenology"],
                description=profile_data.get("description", ""),
                prerequisites=profile_data.get("prerequisites", []),
                typical_duration=profile_data.get("typical_duration", "variable"),
                entry_methods=profile_data.get("entry_methods", []),
                neural_correlates=profile_data.get("neural_correlates", []),
                progression_path=profile_data.get("progression_path", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_state_profile(profile)
            count += 1

        logger.info(f"Initialized {count} seed state profiles")
        return count

    async def initialize_seed_traditions(self) -> int:
        """Initialize with seed traditions."""
        seed_traditions = self._get_seed_traditions()
        count = 0

        for tradition_data in seed_traditions:
            tradition = TraditionProfile(
                tradition_id=tradition_data["tradition_id"],
                tradition=tradition_data["tradition"],
                origin=tradition_data["origin"],
                key_practices=tradition_data.get("key_practices", []),
                state_maps=tradition_data.get("state_maps", []),
                notable_teachers=tradition_data.get("notable_teachers", []),
                texts=tradition_data.get("texts", []),
                description=tradition_data.get("description", ""),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_tradition(tradition)
            count += 1

        logger.info(f"Initialized {count} seed traditions")
        return count

    async def initialize_seed_findings(self) -> int:
        """Initialize with seed neural findings."""
        seed_findings = self._get_seed_findings()
        count = 0

        for finding_data in seed_findings:
            finding = NeuralFinding(
                finding_id=finding_data["finding_id"],
                state=finding_data["state"],
                neural_correlate=finding_data["neural_correlate"],
                study_reference=finding_data["study_reference"],
                methodology=finding_data["methodology"],
                sample_size=finding_data["sample_size"],
                key_result=finding_data["key_result"],
                replication_status=finding_data.get("replication_status", "unreplicated"),
                year=finding_data.get("year", 0),
                researchers=finding_data.get("researchers", []),
                limitations=finding_data.get("limitations", []),
            )
            await self.add_finding(finding)
            count += 1

        logger.info(f"Initialized {count} seed findings")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        profiles_count = await self.initialize_seed_state_profiles()
        traditions_count = await self.initialize_seed_traditions()
        findings_count = await self.initialize_seed_findings()

        return {
            "state_profiles": profiles_count,
            "traditions": traditions_count,
            "findings": findings_count,
            "total": profiles_count + traditions_count + findings_count
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ContemplativeState",
    "ContemplativeTradition",
    "PhenomenologicalQuality",
    "NeuralCorrelate",
    "PracticeType",
    "MaturityLevel",
    # Dataclasses
    "ContemplativeStateProfile",
    "MeditationSession",
    "TraditionProfile",
    "NeuralFinding",
    "ContemplativeStatesMaturityState",
    # Interface
    "ContemplativeStatesInterface",
]
