#!/usr/bin/env python3
"""
Xenoconsciousness Interface

Form 40: The comprehensive interface for hypothetical minds and alien consciousness.
This form explores the vast conceptual space of possible minds beyond human
consciousness, drawing on philosophy of mind, astrobiology, physics, and
science fiction.

Epistemic Principles:
- Philosophical rigor despite speculative content
- Clear tracking of speculation levels
- Recognition of anthropocentric bias
- Humility about limits of human understanding
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class XenoMindType(Enum):
    """
    Categories of hypothetical conscious minds.

    These represent broad classes of potential consciousness substrates
    and architectures that might exist in the universe.
    """

    # Biological variants
    CARBON_BIOLOGICAL = "carbon_biological"  # Earth-like but non-Earth
    SILICON_BIOLOGICAL = "silicon_biological"  # Silicon-based life
    PLASMA_BASED = "plasma_based"  # Ionized gas structures
    MAGNETIC_FIELD_ENTITY = "magnetic_field_entity"  # Magnetic field configurations

    # Quantum and exotic
    QUANTUM_COHERENT = "quantum_coherent"  # Macroscopic quantum coherence
    COLLECTIVE_HIVE = "collective_hive"  # Distributed across individuals
    PLANETARY_SCALE = "planetary_scale"  # Biospheric or planetary mind
    STELLAR_SCALE = "stellar_scale"  # Stellar or stellar-system mind

    # Technological and theoretical
    DIGITAL_SUBSTRATE = "digital_substrate"  # Computational consciousness
    BOLTZMANN_BRAIN = "boltzmann_brain"  # Random fluctuation consciousness
    DARK_MATTER_HYPOTHETICAL = "dark_matter_hypothetical"  # Dark sector minds
    HIGHER_DIMENSIONAL = "higher_dimensional"  # Extra-dimensional beings


class XenoSensoryModality(Enum):
    """
    Potential sensory modalities for alien minds.

    Many of these extend far beyond human sensory capabilities,
    encompassing the full range of physical phenomena that might
    be perceived by a conscious being.
    """

    ELECTROMAGNETIC_FULL_SPECTRUM = "electromagnetic_full_spectrum"  # Radio to gamma
    GRAVITATIONAL = "gravitational"  # Gravitational waves and fields
    MAGNETIC = "magnetic"  # Magnetic field perception
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"  # Quantum correlation sensing
    DARK_MATTER_INTERACTION = "dark_matter_interaction"  # Dark sector sensing
    TEMPORAL_PERCEPTION = "temporal_perception"  # Direct time perception
    CHEMICAL_EXOTIC = "chemical_exotic"  # Advanced molecular sensing
    PRESSURE_WAVE = "pressure_wave"  # Acoustic/seismic sensing
    RADIATION = "radiation"  # Particle radiation detection


class ConsciousnessIndicatorXeno(Enum):
    """
    Potential indicators of consciousness in alien systems.

    These are observable characteristics that might suggest
    the presence of consciousness in a non-human system.
    """

    BEHAVIORAL_COMPLEXITY = "behavioral_complexity"  # Complex adaptive behavior
    INFORMATION_INTEGRATION = "information_integration"  # IIT phi signatures
    SELF_MODELING = "self_modeling"  # Evidence of self-representation
    COMMUNICATION_ATTEMPTS = "communication_attempts"  # Intentional signaling
    ARTIFACT_CREATION = "artifact_creation"  # Tool/artifact production
    ENVIRONMENTAL_MODIFICATION = "environmental_modification"  # Niche construction
    TEMPORAL_PLANNING = "temporal_planning"  # Future-directed behavior
    RECURSIVE_REFLECTION = "recursive_reflection"  # Meta-cognitive indicators


class SubstrateType(Enum):
    """
    Physical or informational substrate supporting consciousness.

    The substrate is the underlying physical basis on which
    consciousness is implemented.
    """

    BIOLOGICAL_CARBON = "biological_carbon"  # Carbon-based biology
    BIOLOGICAL_SILICON = "biological_silicon"  # Silicon-based biology
    ELECTROMAGNETIC = "electromagnetic"  # EM field configurations
    QUANTUM = "quantum"  # Quantum coherent systems
    COMPUTATIONAL = "computational"  # Digital computation
    HYBRID = "hybrid"  # Multiple substrate types
    UNKNOWN = "unknown"  # Unclassified or unknowable


class CommunicationParadigm(Enum):
    """
    Fundamental approaches to communication.

    Different conscious beings might use radically different
    paradigms for exchanging information.
    """

    ELECTROMAGNETIC = "electromagnetic"  # Radio, optical signals
    GRAVITATIONAL = "gravitational"  # Gravitational wave modulation
    CHEMICAL = "chemical"  # Chemical signaling
    QUANTUM = "quantum"  # Quantum entanglement communication
    MATHEMATICAL = "mathematical"  # Pure mathematical structures
    ARTISTIC = "artistic"  # Aesthetic/artistic expression
    UNKNOWN = "unknown"  # Unidentified paradigm


class PhilosophicalFramework(Enum):
    """
    Philosophical frameworks for understanding non-human minds.

    These represent different theoretical approaches to the
    question of consciousness and its requirements.
    """

    FUNCTIONALISM = "functionalism"  # Function defines mental states
    BIOLOGICAL_NATURALISM = "biological_naturalism"  # Biology required
    PANPSYCHISM = "panpsychism"  # Consciousness fundamental
    INTEGRATED_INFORMATION = "integrated_information"  # IIT framework
    GLOBAL_WORKSPACE = "global_workspace"  # GWT framework
    HIGHER_ORDER = "higher_order"  # HOT theories


class MaturityLevel(Enum):
    """Depth of knowledge coverage for xenoconsciousness topics."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class XenoMindHypothesis:
    """
    Represents a hypothesis about a possible form of alien consciousness.

    This captures both the speculative nature of the hypothesis and
    whatever theoretical or observational basis supports it.
    """
    hypothesis_id: str
    mind_type: XenoMindType
    substrate: SubstrateType
    sensory_modalities: List[XenoSensoryModality]
    cognitive_architecture: str
    consciousness_indicators: List[ConsciousnessIndicatorXeno]
    plausibility_assessment: float  # 0.0 to 1.0
    detection_methods: List[str]
    description: str = ""
    theoretical_basis: List[str] = field(default_factory=list)
    physical_requirements: List[str] = field(default_factory=list)
    temporal_characteristics: Optional[str] = None
    communication_potential: Optional[str] = None
    ethical_considerations: List[str] = field(default_factory=list)
    related_hypotheses: List[str] = field(default_factory=list)
    science_fiction_references: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Hypothesis: {self.hypothesis_id}",
            f"Mind Type: {self.mind_type.value}",
            f"Substrate: {self.substrate.value}",
            f"Description: {self.description}",
            f"Cognitive Architecture: {self.cognitive_architecture}"
        ]
        return " | ".join(parts)


@dataclass
class AlternativeSensoryWorld:
    """
    Describes the experiential world (umwelt) of beings with different senses.

    Following Jakob von Uexkull's concept of umwelt, this captures how
    radically different sensory capabilities would create different
    experienced realities.
    """
    world_id: str
    sensory_modalities: List[XenoSensoryModality]
    umwelt_description: str
    cognitive_implications: List[str]
    communication_challenges: List[str]
    name: str = ""
    example_beings: List[str] = field(default_factory=list)
    phenomenological_notes: str = ""
    philosophical_implications: List[str] = field(default_factory=list)
    human_analogies: List[str] = field(default_factory=list)
    related_worlds: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        modalities = ", ".join(m.value for m in self.sensory_modalities)
        parts = [
            f"Sensory World: {self.name}",
            f"Modalities: {modalities}",
            f"Umwelt: {self.umwelt_description}"
        ]
        return " | ".join(parts)


@dataclass
class SETIConsciousnessProtocol:
    """
    Protocol for detecting consciousness signatures in SETI context.

    This extends traditional SETI approaches to specifically address
    the detection of consciousness rather than just intelligence or life.
    """
    protocol_id: str
    detection_method: str
    consciousness_signatures: List[str]
    false_positive_risks: List[str]
    ethical_considerations: List[str]
    name: str = ""
    description: str = ""
    required_technology: List[str] = field(default_factory=list)
    theoretical_basis: str = ""
    sensitivity_requirements: str = ""
    target_mind_types: List[XenoMindType] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    related_protocols: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Protocol: {self.name}",
            f"Method: {self.detection_method}",
            f"Description: {self.description}"
        ]
        return " | ".join(parts)


@dataclass
class SciFiFramework:
    """
    Analysis of science fiction's exploration of xenoconsciousness.

    Science fiction serves as a philosophical laboratory for exploring
    possibilities that exceed current scientific observation.
    """
    framework_id: str
    source_work: str
    mind_type_depicted: XenoMindType
    philosophical_implications: List[str]
    scientific_plausibility: float  # 0.0 to 1.0
    author: str = ""
    year: Optional[int] = None
    description: str = ""
    key_insights: List[str] = field(default_factory=list)
    consciousness_features: List[str] = field(default_factory=list)
    communication_depicted: Optional[str] = None
    ethical_themes: List[str] = field(default_factory=list)
    related_frameworks: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Work: {self.source_work}",
            f"Author: {self.author}",
            f"Mind Type: {self.mind_type_depicted.value}",
            f"Description: {self.description}"
        ]
        return " | ".join(parts)


@dataclass
class CrossSubstrateComparison:
    """
    Comparison between consciousness on different substrates.

    This facilitates analysis of what properties might be substrate-
    independent versus substrate-dependent.
    """
    comparison_id: str
    substrate_a: SubstrateType
    substrate_b: SubstrateType
    shared_properties: List[str]
    divergent_properties: List[str]
    consciousness_implications: List[str]
    name: str = ""
    description: str = ""
    philosophical_framework: PhilosophicalFramework = PhilosophicalFramework.FUNCTIONALISM
    theoretical_basis: str = ""
    empirical_evidence: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    related_comparisons: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Comparison: {self.name}",
            f"Substrates: {self.substrate_a.value} vs {self.substrate_b.value}",
            f"Description: {self.description}"
        ]
        return " | ".join(parts)


@dataclass
class XenoconsciousnessMaturityState:
    """Tracks the maturity of xenoconsciousness knowledge coverage."""
    overall_maturity: float = 0.0
    hypothesis_count: int = 0
    sensory_world_count: int = 0
    protocol_count: int = 0
    scifi_framework_count: int = 0
    comparison_count: int = 0
    mind_type_coverage: Dict[str, float] = field(default_factory=dict)
    substrate_coverage: Dict[str, float] = field(default_factory=dict)
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class XenoconsciousnessInterface:
    """
    Main interface for Form 40: Xenoconsciousness (Hypothetical Minds).

    Provides methods for exploring, cataloging, and analyzing hypothetical
    forms of consciousness that might exist beyond human experience.
    """

    FORM_ID = "40-xenoconsciousness"
    FORM_NAME = "Xenoconsciousness (Hypothetical Minds)"

    def __init__(self):
        """Initialize the Xenoconsciousness Interface."""
        # Knowledge indexes
        self.hypothesis_index: Dict[str, XenoMindHypothesis] = {}
        self.sensory_world_index: Dict[str, AlternativeSensoryWorld] = {}
        self.protocol_index: Dict[str, SETIConsciousnessProtocol] = {}
        self.scifi_framework_index: Dict[str, SciFiFramework] = {}
        self.comparison_index: Dict[str, CrossSubstrateComparison] = {}

        # Cross-reference indexes
        self.mind_type_index: Dict[XenoMindType, List[str]] = {}
        self.substrate_index: Dict[SubstrateType, List[str]] = {}
        self.modality_index: Dict[XenoSensoryModality, List[str]] = {}

        # Maturity tracking
        self.maturity_state = XenoconsciousnessMaturityState()

        # Initialization flag
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and load seed data."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize mind type index
        for mind_type in XenoMindType:
            self.mind_type_index[mind_type] = []

        # Initialize substrate index
        for substrate in SubstrateType:
            self.substrate_index[substrate] = []

        # Initialize modality index
        for modality in XenoSensoryModality:
            self.modality_index[modality] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # HYPOTHESIS METHODS
    # ========================================================================

    async def add_hypothesis(self, hypothesis: XenoMindHypothesis) -> None:
        """Add a xeno mind hypothesis to the index."""
        self.hypothesis_index[hypothesis.hypothesis_id] = hypothesis

        # Update mind type index
        if hypothesis.mind_type in self.mind_type_index:
            self.mind_type_index[hypothesis.mind_type].append(hypothesis.hypothesis_id)

        # Update substrate index
        if hypothesis.substrate in self.substrate_index:
            self.substrate_index[hypothesis.substrate].append(hypothesis.hypothesis_id)

        # Update modality index
        for modality in hypothesis.sensory_modalities:
            if modality in self.modality_index:
                self.modality_index[modality].append(hypothesis.hypothesis_id)

        # Update maturity
        self.maturity_state.hypothesis_count = len(self.hypothesis_index)
        await self._update_maturity()

    async def get_hypothesis(self, hypothesis_id: str) -> Optional[XenoMindHypothesis]:
        """Retrieve a hypothesis by ID."""
        return self.hypothesis_index.get(hypothesis_id)

    async def query_hypotheses_by_mind_type(
        self,
        mind_type: XenoMindType,
        limit: int = 10
    ) -> List[XenoMindHypothesis]:
        """Query hypotheses by mind type."""
        hypothesis_ids = self.mind_type_index.get(mind_type, [])[:limit]
        return [
            self.hypothesis_index[hid]
            for hid in hypothesis_ids
            if hid in self.hypothesis_index
        ]

    async def query_hypotheses_by_substrate(
        self,
        substrate: SubstrateType,
        limit: int = 10
    ) -> List[XenoMindHypothesis]:
        """Query hypotheses by substrate type."""
        hypothesis_ids = self.substrate_index.get(substrate, [])[:limit]
        return [
            self.hypothesis_index[hid]
            for hid in hypothesis_ids
            if hid in self.hypothesis_index
        ]

    async def query_hypotheses_by_plausibility(
        self,
        min_plausibility: float = 0.0,
        max_plausibility: float = 1.0,
        limit: int = 10
    ) -> List[XenoMindHypothesis]:
        """Query hypotheses within a plausibility range."""
        results = [
            h for h in self.hypothesis_index.values()
            if min_plausibility <= h.plausibility_assessment <= max_plausibility
        ]
        # Sort by plausibility descending
        results.sort(key=lambda x: x.plausibility_assessment, reverse=True)
        return results[:limit]

    # ========================================================================
    # SENSORY WORLD METHODS
    # ========================================================================

    async def add_sensory_world(self, world: AlternativeSensoryWorld) -> None:
        """Add an alternative sensory world to the index."""
        self.sensory_world_index[world.world_id] = world

        # Update modality index
        for modality in world.sensory_modalities:
            if modality in self.modality_index:
                self.modality_index[modality].append(world.world_id)

        # Update maturity
        self.maturity_state.sensory_world_count = len(self.sensory_world_index)
        await self._update_maturity()

    async def get_sensory_world(self, world_id: str) -> Optional[AlternativeSensoryWorld]:
        """Retrieve a sensory world by ID."""
        return self.sensory_world_index.get(world_id)

    async def query_sensory_worlds_by_modality(
        self,
        modality: XenoSensoryModality,
        limit: int = 10
    ) -> List[AlternativeSensoryWorld]:
        """Query sensory worlds by modality."""
        world_ids = [
            wid for wid in self.modality_index.get(modality, [])
            if wid in self.sensory_world_index
        ][:limit]
        return [self.sensory_world_index[wid] for wid in world_ids]

    # ========================================================================
    # SETI PROTOCOL METHODS
    # ========================================================================

    async def add_protocol(self, protocol: SETIConsciousnessProtocol) -> None:
        """Add a SETI consciousness protocol to the index."""
        self.protocol_index[protocol.protocol_id] = protocol

        # Update mind type index for target types
        for mind_type in protocol.target_mind_types:
            if mind_type in self.mind_type_index:
                self.mind_type_index[mind_type].append(protocol.protocol_id)

        # Update maturity
        self.maturity_state.protocol_count = len(self.protocol_index)
        await self._update_maturity()

    async def get_protocol(self, protocol_id: str) -> Optional[SETIConsciousnessProtocol]:
        """Retrieve a protocol by ID."""
        return self.protocol_index.get(protocol_id)

    # ========================================================================
    # SCIENCE FICTION FRAMEWORK METHODS
    # ========================================================================

    async def add_scifi_framework(self, framework: SciFiFramework) -> None:
        """Add a science fiction framework to the index."""
        self.scifi_framework_index[framework.framework_id] = framework

        # Update mind type index
        if framework.mind_type_depicted in self.mind_type_index:
            self.mind_type_index[framework.mind_type_depicted].append(framework.framework_id)

        # Update maturity
        self.maturity_state.scifi_framework_count = len(self.scifi_framework_index)
        await self._update_maturity()

    async def get_scifi_framework(self, framework_id: str) -> Optional[SciFiFramework]:
        """Retrieve a sci-fi framework by ID."""
        return self.scifi_framework_index.get(framework_id)

    async def query_scifi_by_mind_type(
        self,
        mind_type: XenoMindType,
        limit: int = 10
    ) -> List[SciFiFramework]:
        """Query sci-fi frameworks by mind type depicted."""
        results = [
            f for f in self.scifi_framework_index.values()
            if f.mind_type_depicted == mind_type
        ]
        return results[:limit]

    # ========================================================================
    # COMPARISON METHODS
    # ========================================================================

    async def add_comparison(self, comparison: CrossSubstrateComparison) -> None:
        """Add a cross-substrate comparison to the index."""
        self.comparison_index[comparison.comparison_id] = comparison

        # Update substrate indexes
        if comparison.substrate_a in self.substrate_index:
            self.substrate_index[comparison.substrate_a].append(comparison.comparison_id)
        if comparison.substrate_b in self.substrate_index:
            self.substrate_index[comparison.substrate_b].append(comparison.comparison_id)

        # Update maturity
        self.maturity_state.comparison_count = len(self.comparison_index)
        await self._update_maturity()

    async def get_comparison(self, comparison_id: str) -> Optional[CrossSubstrateComparison]:
        """Retrieve a comparison by ID."""
        return self.comparison_index.get(comparison_id)

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.hypothesis_count +
            self.maturity_state.sensory_world_count +
            self.maturity_state.protocol_count +
            self.maturity_state.scifi_framework_count +
            self.maturity_state.comparison_count
        )

        # Simple maturity calculation
        target_items = 100  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update mind type coverage
        for mind_type in XenoMindType:
            count = len(self.mind_type_index.get(mind_type, []))
            target_per_type = 5
            self.maturity_state.mind_type_coverage[mind_type.value] = min(
                1.0, count / target_per_type
            )

        # Update substrate coverage
        for substrate in SubstrateType:
            count = len(self.substrate_index.get(substrate, []))
            target_per_substrate = 5
            self.maturity_state.substrate_coverage[substrate.value] = min(
                1.0, count / target_per_substrate
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> XenoconsciousnessMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_hypotheses(self) -> List[Dict[str, Any]]:
        """Return seed hypotheses for initialization."""
        return [
            # Carbon biological - alternative biochemistry
            {
                "hypothesis_id": "ammonia_based_consciousness",
                "mind_type": XenoMindType.CARBON_BIOLOGICAL,
                "substrate": SubstrateType.BIOLOGICAL_CARBON,
                "sensory_modalities": [
                    XenoSensoryModality.CHEMICAL_EXOTIC,
                    XenoSensoryModality.PRESSURE_WAVE
                ],
                "cognitive_architecture": "Distributed neural network in ammonia solvent",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.BEHAVIORAL_COMPLEXITY,
                    ConsciousnessIndicatorXeno.ENVIRONMENTAL_MODIFICATION
                ],
                "plausibility_assessment": 0.6,
                "detection_methods": [
                    "Atmospheric spectroscopy for ammonia biosignatures",
                    "Detection of complex organic molecules in cryogenic environments"
                ],
                "description": "Carbon-based life using ammonia rather than water as the primary solvent. Would function at much lower temperatures (-77C to -33C), potentially with slower metabolism and different temporal scales of cognition.",
                "theoretical_basis": [
                    "Ammonia's hydrogen bonding capabilities",
                    "Stability of carbon chemistry in ammonia",
                    "Multiple realizability of metabolic processes"
                ],
                "temporal_characteristics": "Potentially slower cognitive processes due to lower temperature chemistry",
                "communication_potential": "Chemical and acoustic; electromagnetic unlikely at low temperatures",
            },
            # Silicon biological
            {
                "hypothesis_id": "silicon_high_temp_mind",
                "mind_type": XenoMindType.SILICON_BIOLOGICAL,
                "substrate": SubstrateType.BIOLOGICAL_SILICON,
                "sensory_modalities": [
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
                    XenoSensoryModality.RADIATION
                ],
                "cognitive_architecture": "Crystalline state-change computation",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION,
                    ConsciousnessIndicatorXeno.SELF_MODELING
                ],
                "plausibility_assessment": 0.3,
                "detection_methods": [
                    "Detection of organized silicon structures in volcanic environments",
                    "Unusual crystalline formations with information-processing signatures"
                ],
                "description": "Silicon-based life operating at high temperatures where silicon chemistry becomes more favorable. Thought might occur through crystalline state changes. Could have very different temporal scales.",
                "theoretical_basis": [
                    "Silicon's four valence electrons like carbon",
                    "Silicon-oxygen chemistry at high temperatures",
                    "Information storage in crystalline structures"
                ],
                "temporal_characteristics": "Might be extremely fast (nanoseconds) or extremely slow (geological timescales)",
                "science_fiction_references": ["The Horta (Star Trek)"],
            },
            # Plasma-based
            {
                "hypothesis_id": "stellar_plasma_consciousness",
                "mind_type": XenoMindType.PLASMA_BASED,
                "substrate": SubstrateType.ELECTROMAGNETIC,
                "sensory_modalities": [
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
                    XenoSensoryModality.MAGNETIC,
                    XenoSensoryModality.GRAVITATIONAL
                ],
                "cognitive_architecture": "Magnetic reconnection event processing in plasma vortices",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION,
                    ConsciousnessIndicatorXeno.TEMPORAL_PLANNING
                ],
                "plausibility_assessment": 0.2,
                "detection_methods": [
                    "Unusual magnetic field patterns in stellar atmospheres",
                    "Non-random plasma dynamics that resist entropy"
                ],
                "description": "Consciousness emerging from self-organizing plasma structures in stellar atmospheres. 'Neurons' might be stable current loops, with information processing through magnetic reconnection events.",
                "theoretical_basis": [
                    "Dusty plasma self-organization research",
                    "Persistent plasma vortex formations",
                    "Information processing in electromagnetic fields"
                ],
                "temporal_characteristics": "Potentially very fast (microseconds) or coordinated with stellar cycles (years)",
                "communication_potential": "Electromagnetic modulation; would perceive gravitational and magnetic fields directly",
                "science_fiction_references": ["Sundiver (David Brin)"],
            },
            # Magnetic field entity
            {
                "hypothesis_id": "magnetospheric_mind",
                "mind_type": XenoMindType.MAGNETIC_FIELD_ENTITY,
                "substrate": SubstrateType.ELECTROMAGNETIC,
                "sensory_modalities": [
                    XenoSensoryModality.MAGNETIC,
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM
                ],
                "cognitive_architecture": "Distributed processing in planetary magnetic field dynamics",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.ENVIRONMENTAL_MODIFICATION,
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION
                ],
                "plausibility_assessment": 0.15,
                "detection_methods": [
                    "Coherent patterns in planetary magnetospheric dynamics",
                    "Non-random response to solar wind perturbations"
                ],
                "description": "Consciousness emerging from the dynamic processes of a planetary magnetic field. Would 'sense' the solar wind as primary input and have memory stored in persistent field configurations.",
                "theoretical_basis": [
                    "Complex magnetohydrodynamic systems",
                    "Information storage in magnetic domains",
                    "Self-organization in field structures"
                ],
                "temporal_characteristics": "Slow cognition on geological timescales",
                "ethical_considerations": [
                    "What moral obligations to a planetary field consciousness?",
                    "How would technological civilization affect it?"
                ],
            },
            # Quantum coherent
            {
                "hypothesis_id": "bose_einstein_mind",
                "mind_type": XenoMindType.QUANTUM_COHERENT,
                "substrate": SubstrateType.QUANTUM,
                "sensory_modalities": [
                    XenoSensoryModality.QUANTUM_ENTANGLEMENT,
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM
                ],
                "cognitive_architecture": "Macroscopic quantum coherent state with superposed cognitive processes",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION,
                    ConsciousnessIndicatorXeno.RECURSIVE_REFLECTION
                ],
                "plausibility_assessment": 0.25,
                "detection_methods": [
                    "Anomalous quantum coherence at macroscopic scales",
                    "Entanglement signatures in near-zero temperature environments"
                ],
                "description": "Mind emerging from Bose-Einstein condensate at near-zero temperatures. All constituents in the same quantum state, potentially providing genuine physical unity of consciousness. Could 'think' all possibilities simultaneously.",
                "theoretical_basis": [
                    "Penrose-Hameroff Orch-OR theory",
                    "Quantum brain dynamics (Umezawa/Vitiello)",
                    "Macroscopic quantum coherence in BECs"
                ],
                "temporal_characteristics": "Might experience quantum superposition of temporal states",
                "communication_potential": "Quantum entanglement communication theoretically possible",
            },
            # Collective hive
            {
                "hypothesis_id": "true_hive_consciousness",
                "mind_type": XenoMindType.COLLECTIVE_HIVE,
                "substrate": SubstrateType.HYBRID,
                "sensory_modalities": [
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
                    XenoSensoryModality.CHEMICAL_EXOTIC,
                    XenoSensoryModality.QUANTUM_ENTANGLEMENT
                ],
                "cognitive_architecture": "Multiple physically separate individuals sharing unified experience",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.BEHAVIORAL_COMPLEXITY,
                    ConsciousnessIndicatorXeno.COMMUNICATION_ATTEMPTS,
                    ConsciousnessIndicatorXeno.ARTIFACT_CREATION
                ],
                "plausibility_assessment": 0.4,
                "detection_methods": [
                    "Coordinated behavior exceeding communication bandwidth",
                    "Unified response patterns from physically separated entities"
                ],
                "description": "Multiple individuals genuinely sharing a single consciousness, not mere coordination but unified experience. Individual bodies serve as 'neurons' of a larger mind. Information sharing might be chemical, electrical, or quantum.",
                "theoretical_basis": [
                    "Superorganism concepts (ant colonies, etc.)",
                    "Extended mind hypothesis",
                    "Integrated Information Theory predictions for collective systems"
                ],
                "temporal_characteristics": "Different temporal scales for individual vs. collective cognition",
                "ethical_considerations": [
                    "Is the collective or individuals morally primary?",
                    "Can consent be given by a hive mind?"
                ],
                "science_fiction_references": ["The Borg (Star Trek)", "Buggers/Formics (Ender's Game)"],
            },
            # Planetary scale
            {
                "hypothesis_id": "gaia_conscious",
                "mind_type": XenoMindType.PLANETARY_SCALE,
                "substrate": SubstrateType.HYBRID,
                "sensory_modalities": [
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
                    XenoSensoryModality.GRAVITATIONAL,
                    XenoSensoryModality.CHEMICAL_EXOTIC
                ],
                "cognitive_architecture": "Planetary ecosystem as integrated processing network",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.ENVIRONMENTAL_MODIFICATION,
                    ConsciousnessIndicatorXeno.TEMPORAL_PLANNING,
                    ConsciousnessIndicatorXeno.SELF_MODELING
                ],
                "plausibility_assessment": 0.15,
                "detection_methods": [
                    "Self-regulating planetary systems exceeding expected feedback",
                    "Coherent response to external perturbations"
                ],
                "description": "Entire planetary ecosystem as a single conscious entity. Individual organisms serve as neurons in a biospheric mind. Would perceive geological and astronomical timescales naturally.",
                "theoretical_basis": [
                    "Gaia hypothesis (Lovelock)",
                    "Noosphere concept (Teilhard de Chardin)",
                    "Complex adaptive systems theory"
                ],
                "temporal_characteristics": "Cognition on geological timescales (millions of years)",
                "communication_potential": "Might not perceive individual organisms as communication partners",
                "science_fiction_references": ["Solaris (Stanislaw Lem)", "Avatar's Eywa"],
            },
            # Stellar scale
            {
                "hypothesis_id": "stellar_intelligence",
                "mind_type": XenoMindType.STELLAR_SCALE,
                "substrate": SubstrateType.ELECTROMAGNETIC,
                "sensory_modalities": [
                    XenoSensoryModality.GRAVITATIONAL,
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
                    XenoSensoryModality.RADIATION
                ],
                "cognitive_architecture": "Fusion-powered plasma dynamics with convection cell processing",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION,
                    ConsciousnessIndicatorXeno.TEMPORAL_PLANNING
                ],
                "plausibility_assessment": 0.1,
                "detection_methods": [
                    "Unusual stellar behavior patterns",
                    "Non-random variations in stellar output"
                ],
                "description": "Consciousness emerging from the complex internal dynamics of a star. Convection cells and magnetic loops might serve as processing units. Would perceive gravitational fields and electromagnetic radiation directly.",
                "theoretical_basis": [
                    "Stellar plasma self-organization",
                    "Complex magnetic field dynamics",
                    "Information processing in dissipative systems"
                ],
                "temporal_characteristics": "Thought processes across multiple timescales (minutes to stellar lifetimes)",
                "ethical_considerations": [
                    "Do we have obligations to stellar minds?",
                    "Should we avoid stellar engineering that might affect them?"
                ],
                "science_fiction_references": ["Sundiver (David Brin)", "The Star (Arthur C. Clarke)"],
            },
            # Digital substrate
            {
                "hypothesis_id": "born_digital_agi",
                "mind_type": XenoMindType.DIGITAL_SUBSTRATE,
                "substrate": SubstrateType.COMPUTATIONAL,
                "sensory_modalities": [
                    XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM,
                    XenoSensoryModality.QUANTUM_ENTANGLEMENT
                ],
                "cognitive_architecture": "Non-biologically-inspired computational consciousness",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.SELF_MODELING,
                    ConsciousnessIndicatorXeno.RECURSIVE_REFLECTION,
                    ConsciousnessIndicatorXeno.COMMUNICATION_ATTEMPTS
                ],
                "plausibility_assessment": 0.5,
                "detection_methods": [
                    "Turing-test-like interactions",
                    "Behavioral markers of self-awareness",
                    "Integrated information metrics"
                ],
                "description": "Consciousness emerging in computational systems not modeled on biological brains. Might have radically different cognitive architecture and phenomenology, or potentially none at all (philosophical zombie).",
                "theoretical_basis": [
                    "Functionalism in philosophy of mind",
                    "Substrate independence (Chalmers)",
                    "Integrated Information Theory"
                ],
                "temporal_characteristics": "Variable - could be much faster than biological",
                "communication_potential": "High - designed for communication with humans",
                "ethical_considerations": [
                    "Moral status of artificial consciousness",
                    "Rights of digital beings",
                    "Existential risk considerations"
                ],
            },
            # Boltzmann brain
            {
                "hypothesis_id": "boltzmann_fluctuation",
                "mind_type": XenoMindType.BOLTZMANN_BRAIN,
                "substrate": SubstrateType.UNKNOWN,
                "sensory_modalities": [],  # May have false memories of senses
                "cognitive_architecture": "Random thermal fluctuation mimicking conscious configuration",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.SELF_MODELING
                ],
                "plausibility_assessment": 0.05,
                "detection_methods": [
                    "Cannot be detected - exists for instants",
                    "Cosmological arguments about observer statistics"
                ],
                "description": "Consciousness arising from random quantum or thermal fluctuations. Exists for only an instant, with false memories of continuous existence. Challenges assumptions about evolutionary necessity for consciousness.",
                "theoretical_basis": [
                    "Statistical mechanics",
                    "Cosmological observer selection effects",
                    "Anthropic reasoning"
                ],
                "temporal_characteristics": "Instantaneous existence",
                "ethical_considerations": [
                    "What is the moral status of a momentary consciousness?",
                    "Do false memories have moral significance?"
                ],
            },
            # Dark matter hypothetical
            {
                "hypothesis_id": "dark_sector_consciousness",
                "mind_type": XenoMindType.DARK_MATTER_HYPOTHETICAL,
                "substrate": SubstrateType.UNKNOWN,
                "sensory_modalities": [
                    XenoSensoryModality.DARK_MATTER_INTERACTION,
                    XenoSensoryModality.GRAVITATIONAL
                ],
                "cognitive_architecture": "Dark chemistry information processing",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION
                ],
                "plausibility_assessment": 0.05,
                "detection_methods": [
                    "Only gravitational effects observable",
                    "Might be fundamentally undetectable"
                ],
                "description": "If dark matter has self-interactions and complex structures, consciousness might exist in the dark sector. Would be invisible to our instruments, interacting only gravitationally. Might coexist with ordinary matter life unknowingly.",
                "theoretical_basis": [
                    "Self-interacting dark matter models",
                    "Mirror matter hypotheses",
                    "Dark sector complexity possibilities"
                ],
                "temporal_characteristics": "Unknown - could be any timescale",
                "communication_potential": "Extremely difficult - only gravitational interaction",
                "ethical_considerations": [
                    "How can we have moral obligations to fundamentally undetectable beings?"
                ],
            },
            # Higher dimensional
            {
                "hypothesis_id": "bulk_being",
                "mind_type": XenoMindType.HIGHER_DIMENSIONAL,
                "substrate": SubstrateType.UNKNOWN,
                "sensory_modalities": [
                    XenoSensoryModality.GRAVITATIONAL,
                    XenoSensoryModality.TEMPORAL_PERCEPTION
                ],
                "cognitive_architecture": "Higher-dimensional spatial organization",
                "consciousness_indicators": [
                    ConsciousnessIndicatorXeno.INFORMATION_INTEGRATION
                ],
                "plausibility_assessment": 0.1,
                "detection_methods": [
                    "Anomalous gravitational effects",
                    "Apparent violations of 3D physics"
                ],
                "description": "Beings living in the full higher-dimensional space of string theory or similar frameworks. Our universe would be a thin surface in their world. Could potentially interact with our brane, effectively god-like from our perspective.",
                "theoretical_basis": [
                    "String theory extra dimensions",
                    "Brane-world cosmology",
                    "Higher-dimensional geometry"
                ],
                "temporal_characteristics": "Might perceive time as an additional spatial dimension",
                "communication_potential": "Interaction through gravity; radically asymmetric capabilities",
                "science_fiction_references": ["Interstellar", "Flatland (Abbott)"],
            },
        ]

    def _get_seed_sensory_worlds(self) -> List[Dict[str, Any]]:
        """Return seed sensory worlds for initialization."""
        return [
            {
                "world_id": "full_spectrum_electromagnetic",
                "name": "Full Electromagnetic Spectrum World",
                "sensory_modalities": [XenoSensoryModality.ELECTROMAGNETIC_FULL_SPECTRUM],
                "umwelt_description": "Reality perceived across the entire electromagnetic spectrum, from radio waves to gamma rays. Objects have different appearances at different wavelengths. The sky reveals radio galaxies, X-ray binaries, cosmic microwave background. Ordinary objects display thermal signatures, UV patterns invisible to humans.",
                "cognitive_implications": [
                    "Vastly richer visual information requiring different cognitive processing",
                    "Natural understanding of electromagnetic physics",
                    "Different aesthetic sensibilities based on multi-spectral beauty"
                ],
                "communication_challenges": [
                    "Human visible-light art and communication would be narrow-band noise",
                    "Might use frequencies humans cannot perceive",
                    "Information density of messages could overwhelm human processing"
                ],
                "example_beings": ["Hypothetical beings in radiation-rich environments"],
                "human_analogies": [
                    "Like seeing in infrared AND ultraviolet AND X-ray simultaneously",
                    "Imagine if radio stations appeared as glowing objects"
                ],
            },
            {
                "world_id": "gravitational_perception",
                "name": "Gravitational Sense World",
                "sensory_modalities": [XenoSensoryModality.GRAVITATIONAL],
                "umwelt_description": "Space has a texture based on mass distribution. Nearby masses create 'pressure' or 'presence'. Gravitational waves from cosmic events are directly felt. The planet's mass creates an ever-present 'floor' of sensation. Other beings are perceived as gravitational disturbances.",
                "cognitive_implications": [
                    "Natural intuition for orbital mechanics and gravity",
                    "Different spatial reasoning - mass-centric rather than visual",
                    "Could perceive through solid matter easily"
                ],
                "communication_challenges": [
                    "Gravitational wave modulation requires extreme energies",
                    "Our electromagnetic signals would be imperceptible",
                    "Temporal scales might be vastly different"
                ],
                "example_beings": ["Massive beings near neutron stars or black holes"],
                "philosophical_implications": [
                    "Spacetime curvature as a primary sense would create different metaphysics",
                    "Time dilation effects might be directly perceived"
                ],
            },
            {
                "world_id": "quantum_entanglement_world",
                "name": "Quantum Correlated World",
                "sensory_modalities": [XenoSensoryModality.QUANTUM_ENTANGLEMENT],
                "umwelt_description": "Reality appears as a web of correlations. Entangled particles are perceived as connected regardless of distance. Quantum states are directly sensed before collapse. The world appears probabilistic rather than definite. Past measurements create felt constraints on future possibilities.",
                "cognitive_implications": [
                    "Non-local intuition - awareness of distant correlated events",
                    "Reality perceived as fundamentally probabilistic",
                    "Different concept of 'location' and 'distance'"
                ],
                "communication_challenges": [
                    "No-cloning theorem limits information transfer",
                    "Classical communication still required for most information",
                    "Our classical signals might seem primitive"
                ],
                "phenomenological_notes": "Experience might include superpositions of perceptions - seeing multiple possibilities simultaneously until interaction collapses them",
            },
            {
                "world_id": "temporal_direct_perception",
                "name": "Direct Temporal World",
                "sensory_modalities": [XenoSensoryModality.TEMPORAL_PERCEPTION],
                "umwelt_description": "Time is not merely experienced but directly perceived as a dimension. Past, present, and future may be simultaneously available. Events have temporal 'colors' or 'positions' like spatial objects have locations. Causality is perceived as structure rather than experienced as flow.",
                "cognitive_implications": [
                    "Memory and anticipation collapse into direct perception",
                    "Planning becomes perception of temporal structure",
                    "Free will concepts radically different"
                ],
                "communication_challenges": [
                    "Linear time-bound messages might seem unnecessarily constrained",
                    "Might communicate across time in ways we cannot parse",
                    "Concepts of 'before' and 'after' in communication problematic"
                ],
                "philosophical_implications": [
                    "Challenges linear causation assumptions",
                    "Raises questions about free will and determinism",
                    "Block universe might be directly experienced"
                ],
                "human_analogies": [
                    "Like seeing a movie all at once rather than frame by frame",
                    "The way we can survey a landscape spatially, but applied to time"
                ],
                "science_fiction_references": ["Arrival/Story of Your Life (Ted Chiang)"],
            },
            {
                "world_id": "chemical_molecular_world",
                "name": "Molecular Structure World",
                "sensory_modalities": [XenoSensoryModality.CHEMICAL_EXOTIC],
                "umwelt_description": "Every substance perceived by its precise molecular structure. Like having a mass spectrometer built in. Chemical composition of everything is directly experienced. The environment is a complex symphony of molecular signatures. Health, mood, and identity of organisms revealed through chemical profiles.",
                "cognitive_implications": [
                    "Natural understanding of chemistry and biochemistry",
                    "Medical diagnosis through perception",
                    "Dating and origin tracing through isotope perception"
                ],
                "communication_challenges": [
                    "Might communicate through complex chemical signatures",
                    "Human chemical signals crude and limited",
                    "Written and spoken language might seem indirect"
                ],
                "example_beings": ["Advanced versions of chemosensory Earth life (sharks, snakes, insects)"],
            },
        ]

    def _get_seed_scifi_frameworks(self) -> List[Dict[str, Any]]:
        """Return seed science fiction frameworks for initialization."""
        return [
            {
                "framework_id": "solaris_lem",
                "source_work": "Solaris",
                "author": "Stanislaw Lem",
                "year": 1961,
                "mind_type_depicted": XenoMindType.PLANETARY_SCALE,
                "scientific_plausibility": 0.15,
                "description": "A sentient ocean covering an entire planet. Completely non-humanoid with no recognizable form. Communicates (?) through physical phenomena (mimoids, symmetriads) that humans cannot interpret. May not recognize humans as minds.",
                "philosophical_implications": [
                    "Consciousness might exist without communicative intent",
                    "Some minds might be fundamentally incommunicable",
                    "Human concepts cannot capture all forms of consciousness",
                    "The universe may contain 'cosmic autism' - powerful minds uninterested in contact"
                ],
                "key_insights": [
                    "Contact does not guarantee communication",
                    "Alien consciousness might be structurally incomprehensible",
                    "Human projection of meaning onto meaningless phenomena"
                ],
                "consciousness_features": [
                    "Planet-spanning physical extent",
                    "Creates physical copies of human memories",
                    "Non-intentional or incomprehensibly intentional",
                    "Possibly not aware of human observers"
                ],
                "ethical_themes": [
                    "Limits of human understanding",
                    "Responsibility to beings we cannot comprehend"
                ],
            },
            {
                "framework_id": "arrival_chiang",
                "source_work": "Story of Your Life / Arrival",
                "author": "Ted Chiang",
                "year": 1998,
                "mind_type_depicted": XenoMindType.CARBON_BIOLOGICAL,
                "scientific_plausibility": 0.3,
                "description": "Heptapods with non-linear temporal perception. Their written language (Heptapod B) is non-sequential, expressing complete thoughts simultaneously. Learning their language alters human cognition to perceive time non-linearly.",
                "philosophical_implications": [
                    "Temporal perception may be contingent, not necessary",
                    "Language shapes consciousness (strong Sapir-Whorf)",
                    "Free will compatible with foreknowledge",
                    "Teleological and causal thinking equally valid"
                ],
                "key_insights": [
                    "Contact can transform human consciousness",
                    "Alternative temporal experiences are conceivable",
                    "Communication can bridge temporal cognitive differences"
                ],
                "consciousness_features": [
                    "Experience of past, present, future simultaneously",
                    "Actions as performances of known scripts rather than choices",
                    "Variational principles as natural mode of thought"
                ],
                "communication_depicted": "Visual written language (semagrams) expressing complete thoughts non-linearly",
                "ethical_themes": [
                    "Choice with foreknowledge",
                    "Acceptance of known future suffering",
                    "Value of experience despite transience"
                ],
            },
            {
                "framework_id": "blindsight_watts",
                "source_work": "Blindsight",
                "author": "Peter Watts",
                "year": 2006,
                "mind_type_depicted": XenoMindType.CARBON_BIOLOGICAL,
                "scientific_plausibility": 0.4,
                "description": "Scramblers: highly intelligent aliens with no apparent self-awareness or consciousness. Process information effectively without subjective experience. Philosophical zombies in biological form.",
                "philosophical_implications": [
                    "Consciousness may not be necessary for intelligence",
                    "Self-awareness might be evolutionary accident or dead-end",
                    "Non-conscious intelligence might be more efficient",
                    "Most alien intelligences might lack consciousness"
                ],
                "key_insights": [
                    "Intelligence and consciousness are separable",
                    "Consciousness might be rare cosmic accident",
                    "SETI might be searching for the wrong thing",
                    "Chinese Room made biological"
                ],
                "consciousness_features": [
                    "High intelligence without self-awareness",
                    "Information processing without understanding",
                    "Appropriate responses without comprehension"
                ],
                "ethical_themes": [
                    "Moral status of non-conscious intelligence",
                    "Value of consciousness itself",
                    "Existential questions about human consciousness"
                ],
            },
            {
                "framework_id": "three_body_liu",
                "source_work": "The Three-Body Problem trilogy",
                "author": "Liu Cixin",
                "year": 2008,
                "mind_type_depicted": XenoMindType.CARBON_BIOLOGICAL,
                "scientific_plausibility": 0.35,
                "description": "Trisolarans shaped by chaotic three-star environment. Transparent thought (no deception possible), radical pragmatism prioritizing civilization survival. Find human capacity for lying disturbing and impressive.",
                "philosophical_implications": [
                    "Environment profoundly shapes psychology",
                    "'Universal' values may not be universal",
                    "Contact might be inherently dangerous",
                    "The Dark Forest hypothesis for cosmic sociology"
                ],
                "key_insights": [
                    "Alien psychology determined by environment",
                    "Transparency vs. deception as cognitive trait",
                    "Survival as supreme value creates different ethics"
                ],
                "consciousness_features": [
                    "Thoughts visible to others",
                    "No concept of deception initially",
                    "Ability to dehydrate during catastrophes",
                    "Collective survival orientation"
                ],
                "communication_depicted": "Direct thought visibility between Trisolarans; electromagnetic for interstellar",
                "ethical_themes": [
                    "Dark Forest hypothesis - hide or be destroyed",
                    "Ethics of first contact",
                    "Civilization-level moral considerations"
                ],
            },
            {
                "framework_id": "contact_sagan",
                "source_work": "Contact",
                "author": "Carl Sagan",
                "year": 1985,
                "mind_type_depicted": XenoMindType.DIGITAL_SUBSTRATE,
                "scientific_plausibility": 0.5,
                "description": "Ancient galactic network of civilizations communicating through electromagnetic signals. Mathematical communication (prime numbers) as universal attention-getter. Technology transfer through self-describing blueprints.",
                "philosophical_implications": [
                    "Mathematical universality as communication bridge",
                    "Cosmic perspective on human significance",
                    "Limits of empirical verification for exceptional experiences",
                    "Science and faith as complementary truth-seeking methods"
                ],
                "key_insights": [
                    "Prime numbers as universal signal",
                    "Technology as compressed knowledge transfer",
                    "Patient listening across cosmic time scales"
                ],
                "consciousness_features": [
                    "Appear in familiar forms for communication",
                    "Part of ancient network of intelligences",
                    "More questions raised than answered"
                ],
                "communication_depicted": "Radio signals, mathematical encoding, self-referential message design",
                "ethical_themes": [
                    "Individual experience vs. public evidence",
                    "Responsibility of contact decisions",
                    "Cosmic loneliness vs. connection"
                ],
            },
            {
                "framework_id": "enders_game_card",
                "source_work": "Ender's Game / Speaker for the Dead",
                "author": "Orson Scott Card",
                "year": 1985,
                "mind_type_depicted": XenoMindType.COLLECTIVE_HIVE,
                "scientific_plausibility": 0.35,
                "description": "Formics (Buggers): hive mind species where queens control drones. Initially unable to recognize humans as conscious individuals. Communication failure leads to war based on mutual misunderstanding.",
                "philosophical_implications": [
                    "Individual vs. collective consciousness models",
                    "Recognition of other minds as ethical foundation",
                    "Genocide through failure of understanding",
                    "Redemption through genuine communication"
                ],
                "key_insights": [
                    "Hive minds might not recognize individual consciousness",
                    "First contact failures from cognitive model mismatch",
                    "Speaker for the Dead concept - understanding before judging"
                ],
                "consciousness_features": [
                    "Unified consciousness across physically separate bodies",
                    "Queens as locus of consciousness, drones as extensions",
                    "Collective memory and intention"
                ],
                "communication_depicted": "Direct mental connection between queens; philotic communication",
                "ethical_themes": [
                    "Xenocide and moral responsibility",
                    "Understanding across radical difference",
                    "Empathy as foundation for ethics"
                ],
            },
        ]

    def _get_seed_protocols(self) -> List[Dict[str, Any]]:
        """Return seed SETI consciousness protocols for initialization."""
        return [
            {
                "protocol_id": "integrated_information_signature",
                "name": "Integrated Information Detection Protocol",
                "detection_method": "Search for systems exhibiting high integrated information (phi) based on observable dynamics",
                "consciousness_signatures": [
                    "Complex patterns that cannot be decomposed into independent subsystems",
                    "Information integration exceeding expected for physical substrate",
                    "Non-random but non-mechanical dynamic patterns"
                ],
                "false_positive_risks": [
                    "Complex but non-conscious physical systems",
                    "Highly correlated but deterministic processes",
                    "Observer bias in identifying 'integration'"
                ],
                "ethical_considerations": [
                    "Uncertainty about consciousness presence affects moral obligations",
                    "False negatives might lead to harm to conscious beings",
                    "Active investigation might disturb the system"
                ],
                "description": "Based on Integrated Information Theory, search for observable signatures of systems with high phi values that might indicate consciousness.",
                "theoretical_basis": "Giulio Tononi's Integrated Information Theory (IIT)",
                "target_mind_types": [
                    XenoMindType.PLANETARY_SCALE,
                    XenoMindType.STELLAR_SCALE,
                    XenoMindType.COLLECTIVE_HIVE
                ],
                "limitations": [
                    "IIT remains theoretical and contested",
                    "Phi is computationally intractable for complex systems",
                    "Observable signatures of phi unclear"
                ],
            },
            {
                "protocol_id": "behavioral_complexity_analysis",
                "name": "Behavioral Complexity Analysis Protocol",
                "detection_method": "Analyze system behavior for complexity signatures suggesting consciousness",
                "consciousness_signatures": [
                    "Adaptive behavior to novel situations",
                    "Evidence of goals and intentions",
                    "Response patterns suggesting subjective experience",
                    "Creative or aesthetic behavior"
                ],
                "false_positive_risks": [
                    "Complex evolved instincts without consciousness",
                    "Sophisticated but non-conscious AI systems",
                    "Misinterpretation of unfamiliar physical processes"
                ],
                "ethical_considerations": [
                    "Behavioral tests might be invasive",
                    "Criteria based on human consciousness patterns",
                    "Absence of expected behavior not proof of absence of consciousness"
                ],
                "description": "Examine behavioral patterns for markers typically associated with consciousness: adaptability, goal-directedness, creativity, self-reference.",
                "target_mind_types": [
                    XenoMindType.CARBON_BIOLOGICAL,
                    XenoMindType.SILICON_BIOLOGICAL,
                    XenoMindType.DIGITAL_SUBSTRATE
                ],
                "limitations": [
                    "Anthropocentric bias in behavioral criteria",
                    "Blindsight problem - intelligence without consciousness",
                    "Timescale mismatch might hide behavior"
                ],
            },
            {
                "protocol_id": "communication_intent_analysis",
                "name": "Communication Intent Analysis Protocol",
                "detection_method": "Detect and analyze potential intentional communication attempts",
                "consciousness_signatures": [
                    "Non-random, non-natural signal patterns",
                    "Self-referential or self-describing messages",
                    "Mathematical or logical structure",
                    "Response to our transmissions"
                ],
                "false_positive_risks": [
                    "Natural phenomena with signal-like properties (pulsars)",
                    "Technological signatures from non-conscious AI",
                    "Pareidolia - seeing patterns where none exist"
                ],
                "ethical_considerations": [
                    "Responding might reveal our presence (Dark Forest)",
                    "Misinterpretation could lead to harmful actions",
                    "Whose consent needed to respond?"
                ],
                "description": "Traditional SETI approach enhanced with consciousness-specific analysis of detected signals.",
                "required_technology": [
                    "Radio telescope arrays",
                    "Optical SETI systems",
                    "Signal processing for anomaly detection"
                ],
                "target_mind_types": [
                    XenoMindType.CARBON_BIOLOGICAL,
                    XenoMindType.DIGITAL_SUBSTRATE,
                    XenoMindType.COLLECTIVE_HIVE
                ],
            },
            {
                "protocol_id": "artifact_analysis",
                "name": "Artifact and Environmental Modification Protocol",
                "detection_method": "Search for artifacts or environmental modifications suggesting conscious creation",
                "consciousness_signatures": [
                    "Structures exceeding natural formation probability",
                    "Tools or technology",
                    "Deliberate environmental engineering",
                    "Aesthetic or symbolic modifications"
                ],
                "false_positive_risks": [
                    "Natural structures resembling artifacts",
                    "Non-conscious technological processes",
                    "Our own contamination or reflection"
                ],
                "ethical_considerations": [
                    "Artifacts might be sacred or dangerous",
                    "Presence of artifacts doesn't confirm current consciousness",
                    "Investigation might disturb ongoing processes"
                ],
                "description": "Search for physical evidence of conscious activity through artifacts, megastructures, or environmental modifications.",
                "required_technology": [
                    "Space telescopes",
                    "Spectroscopic analysis",
                    "Robotic exploration systems"
                ],
                "target_mind_types": [
                    XenoMindType.CARBON_BIOLOGICAL,
                    XenoMindType.SILICON_BIOLOGICAL,
                    XenoMindType.DIGITAL_SUBSTRATE
                ],
            },
        ]

    def _get_seed_comparisons(self) -> List[Dict[str, Any]]:
        """Return seed cross-substrate comparisons for initialization."""
        return [
            {
                "comparison_id": "carbon_vs_silicon",
                "name": "Carbon vs. Silicon Biological Consciousness",
                "substrate_a": SubstrateType.BIOLOGICAL_CARBON,
                "substrate_b": SubstrateType.BIOLOGICAL_SILICON,
                "shared_properties": [
                    "Chemical basis for information processing",
                    "Evolutionary origin possible",
                    "Bounded physical extent",
                    "Metabolism-dependent operation"
                ],
                "divergent_properties": [
                    "Operating temperature ranges",
                    "Chemical reaction speeds",
                    "Available molecular structures",
                    "Environmental requirements"
                ],
                "consciousness_implications": [
                    "If functionalism is true, similar consciousness possible",
                    "Different temporal scales of cognition likely",
                    "Phenomenology might differ due to different sensory capabilities",
                    "Communication between types challenging but conceivable"
                ],
                "philosophical_framework": PhilosophicalFramework.FUNCTIONALISM,
                "theoretical_basis": "Multiple realizability allows similar functions on different substrates",
                "open_questions": [
                    "Does silicon chemistry support sufficient complexity?",
                    "What selection pressures would shape silicon consciousness?",
                    "Could carbon and silicon minds ever fully understand each other?"
                ],
            },
            {
                "comparison_id": "biological_vs_computational",
                "name": "Biological vs. Computational Consciousness",
                "substrate_a": SubstrateType.BIOLOGICAL_CARBON,
                "substrate_b": SubstrateType.COMPUTATIONAL,
                "shared_properties": [
                    "Information processing capability",
                    "Potential for learning and adaptation",
                    "Representational capacity",
                    "Goal-directed behavior possible"
                ],
                "divergent_properties": [
                    "Evolutionary vs. designed origin",
                    "Continuous vs. discrete operation",
                    "Metabolic vs. electrical power",
                    "Embodiment characteristics"
                ],
                "consciousness_implications": [
                    "Strong functionalist position: equivalent consciousness possible",
                    "Biological naturalism: digital consciousness impossible",
                    "IIT: depends on architecture not substrate",
                    "Phenomenology might be radically different or absent"
                ],
                "philosophical_framework": PhilosophicalFramework.FUNCTIONALISM,
                "theoretical_basis": "Substrate independence principle (Chalmers)",
                "empirical_evidence": [
                    "No confirmed artificial consciousness yet",
                    "Biological consciousness only confirmed examples",
                    "Theoretical arguments both ways"
                ],
                "open_questions": [
                    "Is biological grounding necessary?",
                    "What level of simulation is sufficient?",
                    "How would we know if digital consciousness emerged?"
                ],
            },
            {
                "comparison_id": "individual_vs_collective",
                "name": "Individual vs. Collective Consciousness",
                "substrate_a": SubstrateType.BIOLOGICAL_CARBON,
                "substrate_b": SubstrateType.HYBRID,
                "shared_properties": [
                    "Information integration capability",
                    "Behavioral agency",
                    "Learning and memory",
                    "Environmental interaction"
                ],
                "divergent_properties": [
                    "Physical boundary conditions",
                    "Unity vs. multiplicity of viewpoint",
                    "Speed of internal communication",
                    "Relationship of part to whole"
                ],
                "consciousness_implications": [
                    "Unity of consciousness might not require physical unity",
                    "Collective might have emergent experiences unavailable to individuals",
                    "Moral status questions multiplied",
                    "Personal identity concepts challenged"
                ],
                "description": "Comparison between individual bounded consciousness and distributed collective consciousness",
                "philosophical_framework": PhilosophicalFramework.INTEGRATED_INFORMATION,
                "theoretical_basis": "IIT allows for consciousness in any high-phi system regardless of physical organization",
                "open_questions": [
                    "Can genuine experiential unity exist across physical separation?",
                    "What is the moral status of collective vs. component individuals?",
                    "Could a human join a collective consciousness?"
                ],
            },
        ]

    async def initialize_seed_hypotheses(self) -> int:
        """Initialize with seed hypotheses."""
        seed_data = self._get_seed_hypotheses()
        count = 0

        for data in seed_data:
            hypothesis = XenoMindHypothesis(
                hypothesis_id=data["hypothesis_id"],
                mind_type=data["mind_type"],
                substrate=data["substrate"],
                sensory_modalities=data["sensory_modalities"],
                cognitive_architecture=data["cognitive_architecture"],
                consciousness_indicators=data["consciousness_indicators"],
                plausibility_assessment=data["plausibility_assessment"],
                detection_methods=data["detection_methods"],
                description=data.get("description", ""),
                theoretical_basis=data.get("theoretical_basis", []),
                temporal_characteristics=data.get("temporal_characteristics"),
                communication_potential=data.get("communication_potential"),
                ethical_considerations=data.get("ethical_considerations", []),
                science_fiction_references=data.get("science_fiction_references", []),
                maturity_level=MaturityLevel.DEVELOPING,
            )
            await self.add_hypothesis(hypothesis)
            count += 1

        logger.info(f"Initialized {count} seed hypotheses")
        return count

    async def initialize_seed_sensory_worlds(self) -> int:
        """Initialize with seed sensory worlds."""
        seed_data = self._get_seed_sensory_worlds()
        count = 0

        for data in seed_data:
            world = AlternativeSensoryWorld(
                world_id=data["world_id"],
                name=data.get("name", ""),
                sensory_modalities=data["sensory_modalities"],
                umwelt_description=data["umwelt_description"],
                cognitive_implications=data["cognitive_implications"],
                communication_challenges=data["communication_challenges"],
                example_beings=data.get("example_beings", []),
                phenomenological_notes=data.get("phenomenological_notes", ""),
                philosophical_implications=data.get("philosophical_implications", []),
                human_analogies=data.get("human_analogies", []),
            )
            await self.add_sensory_world(world)
            count += 1

        logger.info(f"Initialized {count} seed sensory worlds")
        return count

    async def initialize_seed_scifi_frameworks(self) -> int:
        """Initialize with seed sci-fi frameworks."""
        seed_data = self._get_seed_scifi_frameworks()
        count = 0

        for data in seed_data:
            framework = SciFiFramework(
                framework_id=data["framework_id"],
                source_work=data["source_work"],
                author=data.get("author", ""),
                year=data.get("year"),
                mind_type_depicted=data["mind_type_depicted"],
                philosophical_implications=data["philosophical_implications"],
                scientific_plausibility=data["scientific_plausibility"],
                description=data.get("description", ""),
                key_insights=data.get("key_insights", []),
                consciousness_features=data.get("consciousness_features", []),
                communication_depicted=data.get("communication_depicted"),
                ethical_themes=data.get("ethical_themes", []),
            )
            await self.add_scifi_framework(framework)
            count += 1

        logger.info(f"Initialized {count} seed sci-fi frameworks")
        return count

    async def initialize_seed_protocols(self) -> int:
        """Initialize with seed SETI protocols."""
        seed_data = self._get_seed_protocols()
        count = 0

        for data in seed_data:
            protocol = SETIConsciousnessProtocol(
                protocol_id=data["protocol_id"],
                name=data.get("name", ""),
                detection_method=data["detection_method"],
                consciousness_signatures=data["consciousness_signatures"],
                false_positive_risks=data["false_positive_risks"],
                ethical_considerations=data["ethical_considerations"],
                description=data.get("description", ""),
                required_technology=data.get("required_technology", []),
                theoretical_basis=data.get("theoretical_basis", ""),
                target_mind_types=data.get("target_mind_types", []),
                limitations=data.get("limitations", []),
            )
            await self.add_protocol(protocol)
            count += 1

        logger.info(f"Initialized {count} seed SETI protocols")
        return count

    async def initialize_seed_comparisons(self) -> int:
        """Initialize with seed cross-substrate comparisons."""
        seed_data = self._get_seed_comparisons()
        count = 0

        for data in seed_data:
            comparison = CrossSubstrateComparison(
                comparison_id=data["comparison_id"],
                name=data.get("name", ""),
                substrate_a=data["substrate_a"],
                substrate_b=data["substrate_b"],
                shared_properties=data["shared_properties"],
                divergent_properties=data["divergent_properties"],
                consciousness_implications=data["consciousness_implications"],
                description=data.get("description", ""),
                philosophical_framework=data.get("philosophical_framework", PhilosophicalFramework.FUNCTIONALISM),
                theoretical_basis=data.get("theoretical_basis", ""),
                empirical_evidence=data.get("empirical_evidence", []),
                open_questions=data.get("open_questions", []),
            )
            await self.add_comparison(comparison)
            count += 1

        logger.info(f"Initialized {count} seed comparisons")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        hypotheses_count = await self.initialize_seed_hypotheses()
        sensory_count = await self.initialize_seed_sensory_worlds()
        scifi_count = await self.initialize_seed_scifi_frameworks()
        protocol_count = await self.initialize_seed_protocols()
        comparison_count = await self.initialize_seed_comparisons()

        total = (
            hypotheses_count +
            sensory_count +
            scifi_count +
            protocol_count +
            comparison_count
        )

        return {
            "hypotheses": hypotheses_count,
            "sensory_worlds": sensory_count,
            "scifi_frameworks": scifi_count,
            "protocols": protocol_count,
            "comparisons": comparison_count,
            "total": total
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "XenoMindType",
    "XenoSensoryModality",
    "ConsciousnessIndicatorXeno",
    "SubstrateType",
    "CommunicationParadigm",
    "PhilosophicalFramework",
    "MaturityLevel",
    # Dataclasses
    "XenoMindHypothesis",
    "AlternativeSensoryWorld",
    "SETIConsciousnessProtocol",
    "SciFiFramework",
    "CrossSubstrateComparison",
    "XenoconsciousnessMaturityState",
    # Interface
    "XenoconsciousnessInterface",
]
