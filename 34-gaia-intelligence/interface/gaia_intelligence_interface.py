#!/usr/bin/env python3
"""
Ecological/Planetary Intelligence (Gaia) Interface

Form 34: The comprehensive interface for Gaia theory, Earth system science,
planetary boundaries, indigenous Earth traditions, and ecological intelligence.
This form explores Earth as a self-regulating, potentially conscious system.

Scientific Foundation:
- James Lovelock's Gaia hypothesis
- Lynn Margulis's symbiogenesis contributions
- Earth System Science integration
- Planetary Boundaries framework (Rockstrom)

Ethical Framework:
- Respect for indigenous Earth traditions
- Recognition of Earth's intrinsic value
- Intergenerational responsibility
- Integration with Form 29 (Folk Wisdom) perspectives
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

class GaiaSystem(Enum):
    """
    Major Earth system components that contribute to planetary regulation.

    These systems interact through complex feedbacks to maintain
    conditions suitable for life - the core of Gaia theory.
    """
    ATMOSPHERE = "atmosphere"  # Gas composition, ozone, greenhouse effect
    OCEAN_CIRCULATION = "ocean_circulation"  # Thermohaline, currents, heat transport
    CARBON_CYCLE = "carbon_cycle"  # Biological and geological carbon cycling
    WATER_CYCLE = "water_cycle"  # Evaporation, precipitation, transpiration
    NITROGEN_CYCLE = "nitrogen_cycle"  # Microbial fixation and transformation
    CLIMATE_FEEDBACK = "climate_feedback"  # Amplifying and dampening mechanisms
    BIODIVERSITY_NETWORKS = "biodiversity_networks"  # Food webs, mutualistic networks
    SOIL_ECOSYSTEMS = "soil_ecosystems"  # Decomposition, nutrient cycling, carbon storage
    MAGNETIC_FIELD = "magnetic_field"  # Geodynamo, radiation protection
    ALBEDO_REGULATION = "albedo_regulation"  # Surface reflectivity, ice, clouds


class PlanetaryBoundary(Enum):
    """
    The nine planetary boundaries identified by Rockstrom et al. (2009).

    These define a safe operating space for humanity within which
    Earth system processes can continue to function.
    """
    CLIMATE_CHANGE = "climate_change"  # CO2, radiative forcing
    BIOSPHERE_INTEGRITY = "biosphere_integrity"  # Genetic and functional diversity
    LAND_SYSTEM_CHANGE = "land_system_change"  # Forest cover, land use
    FRESHWATER_USE = "freshwater_use"  # Water consumption, green/blue water
    BIOGEOCHEMICAL_FLOWS = "biogeochemical_flows"  # Nitrogen and phosphorus cycles
    OCEAN_ACIDIFICATION = "ocean_acidification"  # Carbonate saturation state
    ATMOSPHERIC_AEROSOLS = "atmospheric_aerosols"  # Aerosol optical depth
    STRATOSPHERIC_OZONE = "stratospheric_ozone"  # O3 concentration
    NOVEL_ENTITIES = "novel_entities"  # Chemical pollution, plastics, GMOs


class EcologicalIntelligenceType(Enum):
    """
    Types of intelligence exhibited by ecological and planetary systems.

    These represent different aspects of how Earth systems demonstrate
    behavior analogous to cognition or intelligence.
    """
    HOMEOSTATIC = "homeostatic"  # Maintaining stable conditions
    ADAPTIVE = "adaptive"  # Evolving responses to change
    REGENERATIVE = "regenerative"  # Recovering from disturbance
    COMMUNICATIVE = "communicative"  # Information transfer in networks
    ANTICIPATORY = "anticipatory"  # Seasonal preparation, predictive behavior
    RESILIENT = "resilient"  # Absorbing disturbance while maintaining function


class FeedbackType(Enum):
    """
    Types of feedback loops in Earth system dynamics.
    """
    POSITIVE = "positive"  # Amplifying, reinforcing
    NEGATIVE = "negative"  # Dampening, stabilizing
    COMPLEX = "complex"  # Mixed or conditional effects


class IndigenousEarthTradition(Enum):
    """
    Major indigenous traditions with Earth consciousness concepts.

    Each tradition offers unique perspective on Earth as living entity.
    Links to Form 29 (Folk Wisdom) for detailed exploration.
    """
    PACHAMAMA_ANDEAN = "pachamama_andean"  # Quechua, Aymara Earth Mother
    ABORIGINAL_COUNTRY = "aboriginal_country"  # Australian Aboriginal
    LAKOTA_EARTH = "lakota_earth"  # Mitakuye Oyasin, all relations
    MAORI_PAPATUANUKU = "maori_papatuanuku"  # New Zealand Maori
    AFRICAN_EARTH_SPIRITS = "african_earth_spirits"  # Asase Yaa, Ala, Ubuntu


class GaiaPhilosophy(Enum):
    """
    Major philosophical frameworks for understanding Earth intelligence.
    """
    LOVELOCK_GAIA = "lovelock_gaia"  # Original Gaia hypothesis
    DEEP_ECOLOGY = "deep_ecology"  # Arne Naess, biospherical egalitarianism
    ECOFEMINISM = "ecofeminism"  # Earth as feminine, critique of domination
    SYSTEMS_ECOLOGY = "systems_ecology"  # Capra, networks, emergence


class MaturityLevel(Enum):
    """Depth of knowledge coverage."""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


class BoundaryStatus(Enum):
    """Status of a planetary boundary."""
    SAFE = "safe"  # Within safe operating space
    INCREASING_RISK = "increasing_risk"  # Approaching threshold
    HIGH_RISK = "high_risk"  # Beyond safe threshold
    UNCERTAIN = "uncertain"  # Not quantified or data insufficient


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EarthSystemComponent:
    """
    Represents a major Earth system component and its current state.

    These components form the regulatory mechanisms of planetary homeostasis,
    demonstrating aspects of Gaian self-regulation.
    """
    system_id: str
    name: str
    gaia_system: GaiaSystem
    description: str = ""
    current_state: str = ""  # Narrative description of current status
    feedback_loops: List[str] = field(default_factory=list)  # Related feedbacks
    tipping_points: List[str] = field(default_factory=list)  # Associated tipping points
    resilience_indicators: List[str] = field(default_factory=list)
    key_processes: List[str] = field(default_factory=list)
    biological_drivers: List[str] = field(default_factory=list)
    human_impacts: List[str] = field(default_factory=list)
    related_systems: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"System: {self.name}",
            f"Type: {self.gaia_system.value}",
            f"Description: {self.description}",
            f"State: {self.current_state}"
        ]
        return " | ".join(parts)


@dataclass
class PlanetaryBoundaryState:
    """
    Represents the current state of a planetary boundary.

    Based on Rockstrom et al. (2009) and subsequent updates,
    these define safe operating space for humanity.
    """
    boundary_id: str
    name: str
    boundary_type: PlanetaryBoundary
    description: str = ""
    control_variable: str = ""  # What is measured
    current_value: str = ""  # Current measurement
    safe_threshold: str = ""  # Safe operating boundary
    danger_zone: str = ""  # High-risk threshold
    status: BoundaryStatus = BoundaryStatus.UNCERTAIN
    trend: str = ""  # Improving, stable, worsening
    key_drivers: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    mitigation_options: List[str] = field(default_factory=list)
    last_assessment: Optional[str] = None  # Date of last scientific assessment
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Boundary: {self.name}",
            f"Status: {self.status.value}",
            f"Current: {self.current_value}",
            f"Threshold: {self.safe_threshold}"
        ]
        return " | ".join(parts)


@dataclass
class ClimateFeedback:
    """
    Represents a climate feedback mechanism.

    Feedbacks are central to Gaia theory - they demonstrate how
    Earth systems self-regulate (negative) or amplify changes (positive).
    """
    feedback_id: str
    name: str
    feedback_type: FeedbackType
    description: str = ""
    systems_involved: List[GaiaSystem] = field(default_factory=list)
    mechanism: str = ""  # How the feedback operates
    magnitude: str = ""  # Strength of effect
    timescale: str = ""  # How quickly it operates
    certainty: str = ""  # Scientific confidence level
    current_activation: str = ""  # Is it currently active/observable
    tipping_potential: bool = False  # Can trigger tipping point
    related_feedbacks: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Feedback: {self.name}",
            f"Type: {self.feedback_type.value}",
            f"Mechanism: {self.mechanism}",
            f"Timescale: {self.timescale}"
        ]
        return " | ".join(parts)


@dataclass
class IndigenousEarthPerspective:
    """
    Represents an indigenous tradition's conception of Earth as living entity.

    These perspectives often predate and parallel scientific Gaia theory,
    offering ethical frameworks that science lacks.
    """
    perspective_id: str
    name: str
    tradition: IndigenousEarthTradition
    description: str = ""
    earth_conception: str = ""  # How Earth is understood
    human_role: str = ""  # Human place in relation to Earth
    ethical_principles: List[str] = field(default_factory=list)
    practices: List[str] = field(default_factory=list)  # Ceremonies, rituals
    key_concepts: List[str] = field(default_factory=list)  # Important terms
    source_communities: List[str] = field(default_factory=list)
    legal_recognition: List[str] = field(default_factory=list)  # Legal/constitutional
    form_29_link: Optional[str] = None  # Link to Folk Wisdom form
    related_perspectives: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Tradition: {self.name}",
            f"Earth Conception: {self.earth_conception}",
            f"Human Role: {self.human_role}",
            f"Principles: {', '.join(self.ethical_principles[:3])}"
        ]
        return " | ".join(parts)


@dataclass
class TippingPoint:
    """
    Represents an Earth system tipping point.

    Tipping points are thresholds beyond which systems shift to
    new states, often irreversibly on human timescales.
    """
    tipping_id: str
    name: str
    system: GaiaSystem
    description: str = ""
    threshold: str = ""  # The critical threshold
    current_distance: str = ""  # How far from threshold
    trigger_mechanisms: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    reversibility: str = ""  # Whether it can be reversed
    timescale: str = ""  # How quickly effects manifest
    early_warnings: List[str] = field(default_factory=list)
    cascading_effects: List[str] = field(default_factory=list)  # Other systems affected
    related_tipping_points: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Tipping Point: {self.name}",
            f"System: {self.system.value}",
            f"Threshold: {self.threshold}",
            f"Reversibility: {self.reversibility}"
        ]
        return " | ".join(parts)


@dataclass
class GaiaIntelligenceMaturityState:
    """Tracks the maturity of Gaia intelligence knowledge."""
    overall_maturity: float = 0.0
    system_coverage: Dict[str, float] = field(default_factory=dict)
    earth_system_count: int = 0
    boundary_count: int = 0
    feedback_count: int = 0
    perspective_count: int = 0
    tipping_point_count: int = 0
    cross_references: int = 0
    last_updated: Optional[datetime] = None


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class GaiaIntelligenceInterface:
    """
    Main interface for Form 34: Ecological/Planetary Intelligence (Gaia).

    Provides methods for storing, retrieving, and querying information
    about Earth systems, planetary boundaries, climate feedbacks,
    indigenous perspectives, and tipping points.
    """

    FORM_ID = "34-gaia-intelligence"
    FORM_NAME = "Ecological/Planetary Intelligence (Gaia)"

    def __init__(self):
        """Initialize the Gaia Intelligence Interface."""
        # Knowledge indexes
        self.earth_system_index: Dict[str, EarthSystemComponent] = {}
        self.boundary_index: Dict[str, PlanetaryBoundaryState] = {}
        self.feedback_index: Dict[str, ClimateFeedback] = {}
        self.perspective_index: Dict[str, IndigenousEarthPerspective] = {}
        self.tipping_point_index: Dict[str, TippingPoint] = {}

        # Cross-reference indexes
        self.gaia_system_index: Dict[GaiaSystem, List[str]] = {}
        self.boundary_type_index: Dict[PlanetaryBoundary, List[str]] = {}
        self.tradition_index: Dict[IndigenousEarthTradition, List[str]] = {}

        # Maturity tracking
        self.maturity_state = GaiaIntelligenceMaturityState()

        # Initialize
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the interface and prepare indexes."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")

        # Initialize system index
        for system in GaiaSystem:
            self.gaia_system_index[system] = []

        # Initialize boundary type index
        for boundary in PlanetaryBoundary:
            self.boundary_type_index[boundary] = []

        # Initialize tradition index
        for tradition in IndigenousEarthTradition:
            self.tradition_index[tradition] = []

        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # EARTH SYSTEM METHODS
    # ========================================================================

    async def add_earth_system(self, system: EarthSystemComponent) -> None:
        """Add an Earth system component to the index."""
        self.earth_system_index[system.system_id] = system

        # Update system index
        if system.gaia_system in self.gaia_system_index:
            self.gaia_system_index[system.gaia_system].append(system.system_id)

        # Update maturity
        self.maturity_state.earth_system_count = len(self.earth_system_index)
        await self._update_maturity()

    async def get_earth_system(self, system_id: str) -> Optional[EarthSystemComponent]:
        """Retrieve an Earth system by ID."""
        return self.earth_system_index.get(system_id)

    async def query_systems_by_type(
        self,
        gaia_system: GaiaSystem,
        limit: int = 10
    ) -> List[EarthSystemComponent]:
        """Query Earth systems by type."""
        system_ids = self.gaia_system_index.get(gaia_system, [])[:limit]
        return [
            self.earth_system_index[sid]
            for sid in system_ids
            if sid in self.earth_system_index
        ]

    # ========================================================================
    # PLANETARY BOUNDARY METHODS
    # ========================================================================

    async def add_boundary(self, boundary: PlanetaryBoundaryState) -> None:
        """Add a planetary boundary to the index."""
        self.boundary_index[boundary.boundary_id] = boundary

        # Update boundary type index
        if boundary.boundary_type in self.boundary_type_index:
            self.boundary_type_index[boundary.boundary_type].append(boundary.boundary_id)

        # Update maturity
        self.maturity_state.boundary_count = len(self.boundary_index)
        await self._update_maturity()

    async def get_boundary(self, boundary_id: str) -> Optional[PlanetaryBoundaryState]:
        """Retrieve a planetary boundary by ID."""
        return self.boundary_index.get(boundary_id)

    async def query_boundaries_by_status(
        self,
        status: BoundaryStatus,
        limit: int = 10
    ) -> List[PlanetaryBoundaryState]:
        """Query boundaries by their current status."""
        results = []
        for boundary in self.boundary_index.values():
            if boundary.status == status:
                results.append(boundary)
                if len(results) >= limit:
                    break
        return results

    async def get_transgressed_boundaries(self) -> List[PlanetaryBoundaryState]:
        """Get all boundaries that have been transgressed (high risk)."""
        return await self.query_boundaries_by_status(BoundaryStatus.HIGH_RISK, limit=20)

    # ========================================================================
    # CLIMATE FEEDBACK METHODS
    # ========================================================================

    async def add_feedback(self, feedback: ClimateFeedback) -> None:
        """Add a climate feedback to the index."""
        self.feedback_index[feedback.feedback_id] = feedback

        # Update system index for involved systems
        for system in feedback.systems_involved:
            if system in self.gaia_system_index:
                self.gaia_system_index[system].append(feedback.feedback_id)

        # Update maturity
        self.maturity_state.feedback_count = len(self.feedback_index)
        await self._update_maturity()

    async def get_feedback(self, feedback_id: str) -> Optional[ClimateFeedback]:
        """Retrieve a climate feedback by ID."""
        return self.feedback_index.get(feedback_id)

    async def query_feedbacks_by_type(
        self,
        feedback_type: FeedbackType,
        limit: int = 10
    ) -> List[ClimateFeedback]:
        """Query feedbacks by type (positive/negative/complex)."""
        results = []
        for feedback in self.feedback_index.values():
            if feedback.feedback_type == feedback_type:
                results.append(feedback)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # INDIGENOUS PERSPECTIVE METHODS
    # ========================================================================

    async def add_perspective(self, perspective: IndigenousEarthPerspective) -> None:
        """Add an indigenous perspective to the index."""
        self.perspective_index[perspective.perspective_id] = perspective

        # Update tradition index
        if perspective.tradition in self.tradition_index:
            self.tradition_index[perspective.tradition].append(perspective.perspective_id)

        # Update maturity
        self.maturity_state.perspective_count = len(self.perspective_index)
        await self._update_maturity()

    async def get_perspective(self, perspective_id: str) -> Optional[IndigenousEarthPerspective]:
        """Retrieve an indigenous perspective by ID."""
        return self.perspective_index.get(perspective_id)

    async def query_perspectives_by_tradition(
        self,
        tradition: IndigenousEarthTradition,
        limit: int = 10
    ) -> List[IndigenousEarthPerspective]:
        """Query perspectives by tradition."""
        perspective_ids = self.tradition_index.get(tradition, [])[:limit]
        return [
            self.perspective_index[pid]
            for pid in perspective_ids
            if pid in self.perspective_index
        ]

    # ========================================================================
    # TIPPING POINT METHODS
    # ========================================================================

    async def add_tipping_point(self, tipping_point: TippingPoint) -> None:
        """Add a tipping point to the index."""
        self.tipping_point_index[tipping_point.tipping_id] = tipping_point

        # Update system index
        if tipping_point.system in self.gaia_system_index:
            self.gaia_system_index[tipping_point.system].append(tipping_point.tipping_id)

        # Update maturity
        self.maturity_state.tipping_point_count = len(self.tipping_point_index)
        await self._update_maturity()

    async def get_tipping_point(self, tipping_id: str) -> Optional[TippingPoint]:
        """Retrieve a tipping point by ID."""
        return self.tipping_point_index.get(tipping_id)

    async def query_tipping_points_by_system(
        self,
        system: GaiaSystem,
        limit: int = 10
    ) -> List[TippingPoint]:
        """Query tipping points by Earth system."""
        results = []
        for tp in self.tipping_point_index.values():
            if tp.system == system:
                results.append(tp)
                if len(results) >= limit:
                    break
        return results

    # ========================================================================
    # MATURITY METHODS
    # ========================================================================

    async def _update_maturity(self) -> None:
        """Update maturity state based on current knowledge."""
        total_items = (
            self.maturity_state.earth_system_count +
            self.maturity_state.boundary_count +
            self.maturity_state.feedback_count +
            self.maturity_state.perspective_count +
            self.maturity_state.tipping_point_count
        )

        # Simple maturity calculation
        target_items = 100  # Target for full maturity
        self.maturity_state.overall_maturity = min(1.0, total_items / target_items)

        # Update system coverage
        for system in GaiaSystem:
            count = len(self.gaia_system_index.get(system, []))
            target_per_system = 5
            self.maturity_state.system_coverage[system.value] = min(
                1.0, count / target_per_system
            )

        self.maturity_state.last_updated = datetime.now(timezone.utc)

    async def get_maturity_state(self) -> GaiaIntelligenceMaturityState:
        """Get current maturity state."""
        return self.maturity_state

    # ========================================================================
    # SEED DATA
    # ========================================================================

    def _get_seed_boundaries(self) -> List[Dict[str, Any]]:
        """Return seed data for planetary boundaries."""
        return [
            {
                "boundary_id": "climate_change",
                "name": "Climate Change",
                "boundary_type": PlanetaryBoundary.CLIMATE_CHANGE,
                "description": "Atmospheric CO2 concentration and radiative forcing as indicators of climate system stability.",
                "control_variable": "Atmospheric CO2 concentration (ppm) and radiative forcing (W/m2)",
                "current_value": "~420 ppm (2024), 2.91 W/m2",
                "safe_threshold": "350 ppm CO2, 1.0 W/m2",
                "danger_zone": ">450 ppm CO2",
                "status": BoundaryStatus.HIGH_RISK,
                "trend": "Worsening - approximately 2.5 ppm increase per year",
                "key_drivers": ["Fossil fuel combustion", "Deforestation", "Industrial processes", "Agriculture"],
                "consequences": ["Global temperature rise", "Sea level rise", "Extreme weather events", "Ecosystem disruption"],
            },
            {
                "boundary_id": "biosphere_integrity",
                "name": "Biosphere Integrity",
                "boundary_type": PlanetaryBoundary.BIOSPHERE_INTEGRITY,
                "description": "Genetic diversity and functional diversity of Earth's biosphere.",
                "control_variable": "Extinction rate (E/MSY) and Biodiversity Intactness Index (BII)",
                "current_value": "100-1000 E/MSY, BII below safe level in many regions",
                "safe_threshold": "<10 E/MSY (extinctions per million species-years)",
                "danger_zone": ">100 E/MSY",
                "status": BoundaryStatus.HIGH_RISK,
                "trend": "Worsening - sixth mass extinction underway",
                "key_drivers": ["Habitat destruction", "Overexploitation", "Invasive species", "Climate change", "Pollution"],
                "consequences": ["Ecosystem collapse", "Loss of ecosystem services", "Reduced resilience", "Food security threats"],
            },
            {
                "boundary_id": "land_system_change",
                "name": "Land-System Change",
                "boundary_type": PlanetaryBoundary.LAND_SYSTEM_CHANGE,
                "description": "Conversion of forests and other ecosystems to agricultural and other land uses.",
                "control_variable": "Global forest cover as percentage of original forest",
                "current_value": "~62% of original forest cover remains globally",
                "safe_threshold": "75% forest cover maintained",
                "danger_zone": "<54% forest cover",
                "status": BoundaryStatus.INCREASING_RISK,
                "trend": "Worsening - continued deforestation, especially in tropics",
                "key_drivers": ["Agricultural expansion", "Urbanization", "Infrastructure development", "Logging"],
                "consequences": ["Carbon release", "Biodiversity loss", "Disrupted water cycles", "Soil degradation"],
            },
            {
                "boundary_id": "freshwater_use",
                "name": "Freshwater Use",
                "boundary_type": PlanetaryBoundary.FRESHWATER_USE,
                "description": "Human appropriation of freshwater resources.",
                "control_variable": "Consumptive use of blue water (km3/yr)",
                "current_value": "~2600 km3/yr",
                "safe_threshold": "4000 km3/yr",
                "danger_zone": ">6000 km3/yr",
                "status": BoundaryStatus.SAFE,
                "trend": "Stable globally but regional crises emerging",
                "key_drivers": ["Agriculture (irrigation)", "Industry", "Domestic use", "Climate change"],
                "consequences": ["Water scarcity", "Ecosystem degradation", "Food insecurity", "Conflict"],
            },
            {
                "boundary_id": "biogeochemical_flows",
                "name": "Biogeochemical Flows (N and P)",
                "boundary_type": PlanetaryBoundary.BIOGEOCHEMICAL_FLOWS,
                "description": "Human interference with nitrogen and phosphorus cycles.",
                "control_variable": "N: Industrial fixation (Tg N/yr); P: Flow to oceans (Tg P/yr)",
                "current_value": "N: ~150 Tg N/yr; P: ~22 Tg P/yr",
                "safe_threshold": "N: 62 Tg N/yr; P: 11 Tg P/yr",
                "danger_zone": "N: >150 Tg N/yr; P: >100 Tg P/yr",
                "status": BoundaryStatus.HIGH_RISK,
                "trend": "Worsening - agricultural intensification continues",
                "key_drivers": ["Fertilizer use", "Fossil fuel combustion", "Animal agriculture", "Sewage"],
                "consequences": ["Eutrophication", "Dead zones", "Groundwater contamination", "Algal blooms"],
            },
            {
                "boundary_id": "ocean_acidification",
                "name": "Ocean Acidification",
                "boundary_type": PlanetaryBoundary.OCEAN_ACIDIFICATION,
                "description": "Decrease in ocean pH due to CO2 absorption.",
                "control_variable": "Carbonate ion concentration (aragonite saturation state)",
                "current_value": "~84% of pre-industrial aragonite saturation in surface ocean",
                "safe_threshold": ">80% of pre-industrial aragonite saturation",
                "danger_zone": "<70% saturation",
                "status": BoundaryStatus.SAFE,
                "trend": "Worsening - pH declining as CO2 increases",
                "key_drivers": ["Atmospheric CO2 absorption", "Linked to climate change boundary"],
                "consequences": ["Coral bleaching", "Shell dissolution", "Marine food web disruption", "Fisheries decline"],
            },
            {
                "boundary_id": "atmospheric_aerosols",
                "name": "Atmospheric Aerosol Loading",
                "boundary_type": PlanetaryBoundary.ATMOSPHERIC_AEROSOLS,
                "description": "Particulate matter affecting climate and health.",
                "control_variable": "Aerosol optical depth (regional)",
                "current_value": "Variable by region - some areas severely affected",
                "safe_threshold": "Not globally quantified",
                "danger_zone": "Regional thresholds exceeded in South Asia, China",
                "status": BoundaryStatus.UNCERTAIN,
                "trend": "Mixed - improving in some regions, worsening in others",
                "key_drivers": ["Fossil fuel combustion", "Biomass burning", "Industrial emissions", "Dust"],
                "consequences": ["Health impacts", "Monsoon disruption", "Agricultural effects", "Climate forcing"],
            },
            {
                "boundary_id": "stratospheric_ozone",
                "name": "Stratospheric Ozone Depletion",
                "boundary_type": PlanetaryBoundary.STRATOSPHERIC_OZONE,
                "description": "Depletion of the protective ozone layer.",
                "control_variable": "Stratospheric O3 concentration (DU)",
                "current_value": "Recovery ongoing - Antarctic ozone hole still present seasonally",
                "safe_threshold": "<5% reduction from pre-industrial (275 DU)",
                "danger_zone": ">5% sustained reduction",
                "status": BoundaryStatus.SAFE,
                "trend": "Improving - Montreal Protocol success",
                "key_drivers": ["CFCs (largely phased out)", "HCFCs", "N2O", "Climate change interactions"],
                "consequences": ["UV radiation increase", "Skin cancer", "Crop damage", "Marine ecosystem effects"],
            },
            {
                "boundary_id": "novel_entities",
                "name": "Novel Entities",
                "boundary_type": PlanetaryBoundary.NOVEL_ENTITIES,
                "description": "Chemical pollution, plastics, nuclear waste, and other novel substances.",
                "control_variable": "Not yet quantified - multiple metrics proposed",
                "current_value": "350,000+ registered chemicals; 80,000+ in commerce",
                "safe_threshold": "Not quantified",
                "danger_zone": "Not quantified",
                "status": BoundaryStatus.HIGH_RISK,
                "trend": "Worsening - chemical production increasing exponentially",
                "key_drivers": ["Chemical industry", "Plastics production", "Pharmaceuticals", "Pesticides"],
                "consequences": ["Ecosystem disruption", "Endocrine disruption", "Bioaccumulation", "Unknown long-term effects"],
            },
        ]

    def _get_seed_feedbacks(self) -> List[Dict[str, Any]]:
        """Return seed data for climate feedbacks."""
        return [
            {
                "feedback_id": "ice_albedo",
                "name": "Ice-Albedo Feedback",
                "feedback_type": FeedbackType.POSITIVE,
                "description": "Warming melts ice, reducing reflectivity and causing more warming.",
                "systems_involved": [GaiaSystem.ALBEDO_REGULATION, GaiaSystem.CLIMATE_FEEDBACK],
                "mechanism": "Ice and snow reflect 80-90% of incoming solar radiation. As warming melts ice, darker ocean and land surfaces absorb more heat, causing further warming and more ice melt.",
                "magnitude": "Strong - estimated to amplify warming by 20-30%",
                "timescale": "Decades to centuries",
                "certainty": "High - well observed in satellite record",
                "current_activation": "Actively operating - Arctic sea ice declining ~13% per decade",
                "tipping_potential": True,
            },
            {
                "feedback_id": "water_vapor",
                "name": "Water Vapor Feedback",
                "feedback_type": FeedbackType.POSITIVE,
                "description": "Warming increases evaporation; water vapor is a greenhouse gas.",
                "systems_involved": [GaiaSystem.WATER_CYCLE, GaiaSystem.CLIMATE_FEEDBACK, GaiaSystem.ATMOSPHERE],
                "mechanism": "As temperature rises, evaporation increases, putting more water vapor into the atmosphere. Water vapor is a powerful greenhouse gas, trapping more heat and causing further warming.",
                "magnitude": "Strong - roughly doubles the warming from CO2 alone",
                "timescale": "Days to weeks (fast feedback)",
                "certainty": "Very high - fundamental physics",
                "current_activation": "Continuously active",
                "tipping_potential": False,
            },
            {
                "feedback_id": "permafrost_carbon",
                "name": "Permafrost Carbon Feedback",
                "feedback_type": FeedbackType.POSITIVE,
                "description": "Warming thaws permafrost, releasing stored carbon as CO2 and methane.",
                "systems_involved": [GaiaSystem.CARBON_CYCLE, GaiaSystem.SOIL_ECOSYSTEMS, GaiaSystem.CLIMATE_FEEDBACK],
                "mechanism": "Permafrost contains ~1,500 Gt of carbon, twice atmospheric carbon. As warming thaws permafrost, decomposition releases CO2 and CH4, causing more warming and more thawing.",
                "magnitude": "Potentially large - could add 0.3-1.0C by 2100",
                "timescale": "Decades to centuries",
                "certainty": "Medium-high - process understood, magnitude uncertain",
                "current_activation": "Beginning - permafrost warming and thawing observed",
                "tipping_potential": True,
            },
            {
                "feedback_id": "vegetation_albedo",
                "name": "Vegetation-Albedo Feedback",
                "feedback_type": FeedbackType.POSITIVE,
                "description": "Warming allows forests to expand into tundra, reducing albedo.",
                "systems_involved": [GaiaSystem.BIODIVERSITY_NETWORKS, GaiaSystem.ALBEDO_REGULATION, GaiaSystem.CLIMATE_FEEDBACK],
                "mechanism": "Trees are darker than snow-covered tundra. As warming allows trees to expand northward, surface albedo decreases, absorbing more heat and amplifying warming.",
                "magnitude": "Moderate - regional effects significant",
                "timescale": "Decades to centuries",
                "certainty": "Medium - vegetation dynamics complex",
                "current_activation": "Beginning - Arctic greening observed",
                "tipping_potential": False,
            },
            {
                "feedback_id": "silicate_weathering",
                "name": "Silicate Weathering Feedback",
                "feedback_type": FeedbackType.NEGATIVE,
                "description": "Warming increases weathering rates, which consume CO2.",
                "systems_involved": [GaiaSystem.CARBON_CYCLE, GaiaSystem.CLIMATE_FEEDBACK],
                "mechanism": "Higher temperatures and rainfall increase chemical weathering of silicate rocks. This process consumes CO2, reducing atmospheric concentrations and causing cooling.",
                "magnitude": "Strong over geological time - planetary thermostat",
                "timescale": "100,000 to millions of years",
                "certainty": "High - has maintained habitability for billions of years",
                "current_activation": "Always active but too slow for current crisis",
                "tipping_potential": False,
            },
            {
                "feedback_id": "cloud_feedback",
                "name": "Cloud Feedback",
                "feedback_type": FeedbackType.COMPLEX,
                "description": "Changes in cloud cover and type affect both cooling and warming.",
                "systems_involved": [GaiaSystem.WATER_CYCLE, GaiaSystem.ATMOSPHERE, GaiaSystem.CLIMATE_FEEDBACK],
                "mechanism": "Low clouds cool by reflecting sunlight; high clouds warm by trapping heat. Net effect depends on how cloud distributions change with warming - a major source of climate uncertainty.",
                "magnitude": "Potentially large - may amplify or dampen warming",
                "timescale": "Days to decades",
                "certainty": "Low - largest uncertainty in climate sensitivity",
                "current_activation": "Complex changes occurring",
                "tipping_potential": False,
            },
            {
                "feedback_id": "ocean_circulation",
                "name": "Ocean Circulation Feedback",
                "feedback_type": FeedbackType.COMPLEX,
                "description": "Changes in thermohaline circulation affect global heat distribution.",
                "systems_involved": [GaiaSystem.OCEAN_CIRCULATION, GaiaSystem.CLIMATE_FEEDBACK],
                "mechanism": "Freshwater from melting ice can slow the Atlantic meridional overturning circulation (AMOC), potentially causing rapid regional cooling even as global temperatures rise.",
                "magnitude": "Potentially large - regional effects could be dramatic",
                "timescale": "Decades to centuries",
                "certainty": "Medium - AMOC already weakening but tipping threshold uncertain",
                "current_activation": "AMOC has weakened ~15% since mid-20th century",
                "tipping_potential": True,
            },
        ]

    def _get_seed_perspectives(self) -> List[Dict[str, Any]]:
        """Return seed data for indigenous Earth perspectives."""
        return [
            {
                "perspective_id": "pachamama",
                "name": "Pachamama - Andean Earth Mother",
                "tradition": IndigenousEarthTradition.PACHAMAMA_ANDEAN,
                "description": "Pachamama is the Earth Mother goddess in Andean cosmology, embodying the living planet as a conscious, feeling being who provides for her children and requires reciprocity.",
                "earth_conception": "Earth as conscious, maternal being who feels, responds, and provides. Not simply 'earth' but the entire space-time cosmos in maternal form. Local manifestations in mountains (Apus) and springs.",
                "human_role": "Humans are children of Pachamama with obligation to maintain ayni (sacred reciprocity). We must give back through offerings (despachos) and respectful practice. Neglect causes illness and disaster.",
                "ethical_principles": ["Ayni (reciprocity)", "Ama sua (don't steal)", "Ama llulla (don't lie)", "Ama quella (don't be lazy)", "Respect for all life"],
                "practices": ["Ch'alla (offering libations)", "Despacho ceremonies", "Inti Raymi (sun festival)", "Coca leaf offerings", "Agricultural ceremonies"],
                "key_concepts": ["Ayni", "Pacha (world-time)", "Apus (mountain spirits)", "Kawsay (living energy)"],
                "source_communities": ["Quechua", "Aymara", "Andean cultures of Peru, Bolivia, Ecuador"],
                "legal_recognition": ["Ecuador Constitution 2008 - Rights of Pachamama", "Bolivia Law of Mother Earth 2010"],
                "form_29_link": "andean_ayni",
            },
            {
                "perspective_id": "aboriginal_country",
                "name": "Aboriginal Country - Living Land",
                "tradition": IndigenousEarthTradition.ABORIGINAL_COUNTRY,
                "description": "In Aboriginal Australian worldviews, 'Country' is a specific living area to which people belong. Country is conscious, has feelings, communicates with those who know how to listen.",
                "earth_conception": "Country is alive and conscious with needs, feelings, and intentions. Includes land, water, sky, all beings. People belong to Country, not the reverse. Ancestral beings from Dreamtime remain present in the land.",
                "human_role": "Humans are custodians responsible for caring for Country through ceremony, proper behavior, and sustainable practice. When people care for Country, Country cares for them. Neglecting Country causes it to 'get sick.'",
                "ethical_principles": ["Care for Country", "Kinship obligations", "Law (Tjukurpa)", "Respect for ancestors", "Sustainable harvest"],
                "practices": ["Fire management (cool burns)", "Songlines maintenance", "Increase ceremonies", "Sorry business", "Men's and women's ceremonies"],
                "key_concepts": ["Tjukurpa/Dreaming/Law", "Country", "Songlines", "Totem", "Sorry business"],
                "source_communities": ["Western Desert peoples", "Arnhem Land peoples", "Torres Strait Islanders", "Hundreds of language groups"],
                "legal_recognition": ["Native Title Act 1993", "Caring for Country programs", "Co-management arrangements"],
                "form_29_link": "aboriginal_dreaming_law",
            },
            {
                "perspective_id": "lakota_earth",
                "name": "Mitakuye Oyasin - All My Relations",
                "tradition": IndigenousEarthTradition.LAKOTA_EARTH,
                "description": "Lakota and related Plains nations understand Earth as mother and all beings as relatives. Mitakuye Oyasin ('All Are Related') expresses fundamental interconnection.",
                "earth_conception": "Earth is our mother who feeds, shelters, and sustains all her children. She is not a resource but a relative. The sacred hoop of life includes all beings: four-leggeds, winged ones, crawling beings, standing people (trees), and humans.",
                "human_role": "Humans are part of the sacred hoop, not above it. We have responsibility to all our relatives - past, present, and future (seven generations). Living well means maintaining harmony with all creation.",
                "ethical_principles": ["Mitakuye Oyasin (all related)", "Seven Generations thinking", "Respect for all life", "Generosity", "Bravery", "Wisdom", "Fortitude"],
                "practices": ["Inipi (sweat lodge)", "Sun Dance", "Vision Quest", "Pipe ceremonies", "Seasonal ceremonies"],
                "key_concepts": ["Wakan Tanka (Great Mystery)", "Mitakuye Oyasin", "Sacred Hoop", "Four Directions", "Medicine Wheel"],
                "source_communities": ["Lakota", "Dakota", "Nakota", "Cheyenne", "Blackfoot"],
                "legal_recognition": ["American Indian Religious Freedom Act 1978", "Standing Rock water protection movement"],
                "form_29_link": "lakota_mitakuye_oyasin",
            },
            {
                "perspective_id": "maori_papatuanuku",
                "name": "Papatuanuku - Maori Earth Mother",
                "tradition": IndigenousEarthTradition.MAORI_PAPATUANUKU,
                "description": "In Maori cosmology, Papatuanuku (Earth Mother) was separated from Ranginui (Sky Father), and from their union all life descends. Whakapapa (genealogy) connects all beings.",
                "earth_conception": "Papatuanuku is the Earth Mother, ancestress of all life. She was embraced with Ranginui in darkness until their children separated them to create the world. Her body is the land; she nourishes all her descendants.",
                "human_role": "Humans are descendants of Papatuanuku and Ranginui, kin to all beings through whakapapa. As tangata whenua (people of the land), Maori have kaitiakitanga (guardianship) responsibilities.",
                "ethical_principles": ["Kaitiakitanga (guardianship)", "Whakapapa (genealogical connection)", "Mana (spiritual authority)", "Tapu (sacred restrictions)", "Utu (reciprocity)"],
                "practices": ["Karakia (prayers/incantations)", "Powhiri (welcome ceremony)", "Rahui (resource restrictions)", "Tangi (funeral rites)", "Marae ceremonies"],
                "key_concepts": ["Whakapapa", "Mana", "Tapu", "Mauri (life force)", "Kaitiakitanga", "Tangata whenua"],
                "source_communities": ["Maori iwi (tribes) of Aotearoa/New Zealand"],
                "legal_recognition": ["Treaty of Waitangi 1840", "Whanganui River legal personhood 2017", "Te Urewera legal personhood 2014"],
                "form_29_link": None,
            },
            {
                "perspective_id": "african_earth_spirits",
                "name": "African Earth Spirits and Ubuntu",
                "tradition": IndigenousEarthTradition.AFRICAN_EARTH_SPIRITS,
                "description": "Across Africa, diverse traditions recognize Earth as sacred, often personified as goddess (Asase Yaa, Ala) or understood through Ubuntu - humanity embedded in web of relationships including land.",
                "earth_conception": "Earth as sacred mother or goddess who gives life and receives the dead. Not property to be owned but relative to be honored. Asase Yaa (Akan), Ala (Igbo) are Earth goddesses. Land holds ancestors and spiritual power.",
                "human_role": "Humans are part of community that includes living, dead, unborn, and land. Ubuntu ('I am because we are') extends to relationship with Earth. We are custodians, not owners. Crimes against community are crimes against Earth.",
                "ethical_principles": ["Ubuntu (interconnectedness)", "Respect for ancestors", "Community harmony", "Sacred prohibition (taboo)", "Generational responsibility"],
                "practices": ["Libation pouring", "Harvest ceremonies", "First fruits offerings", "Taboo days for farming (Asase Yaa Thursday)", "Ancestral veneration"],
                "key_concepts": ["Ubuntu", "Asase Yaa", "Ala/Ani", "Mami Wata", "Taboo", "Ancestor veneration"],
                "source_communities": ["Akan (Ghana)", "Igbo (Nigeria)", "Yoruba", "Zulu", "San", "Numerous African cultures"],
                "legal_recognition": ["Ubuntu principles in South African Constitution", "Traditional authority in land management"],
                "form_29_link": "yoruba_ashe",
            },
        ]

    def _get_seed_tipping_points(self) -> List[Dict[str, Any]]:
        """Return seed data for tipping points."""
        return [
            {
                "tipping_id": "greenland_ice_sheet",
                "name": "Greenland Ice Sheet Collapse",
                "system": GaiaSystem.ALBEDO_REGULATION,
                "description": "Complete or near-complete loss of Greenland ice sheet.",
                "threshold": "~1.5-2.5C global warming above pre-industrial",
                "current_distance": "Already in uncertainty zone - 1.1C warming reached",
                "trigger_mechanisms": ["Surface melting", "Marine ice sheet instability", "Meltwater lubrication"],
                "consequences": ["~7m sea level rise", "AMOC disruption", "Global weather pattern changes", "Coastal flooding worldwide"],
                "reversibility": "Essentially irreversible on human timescales (thousands of years)",
                "timescale": "Centuries to millennia for full collapse",
                "early_warnings": ["Accelerating mass loss", "Increasing melt extent", "Ice stream acceleration"],
                "cascading_effects": ["AMOC weakening", "Arctic ecosystem disruption", "Global sea level rise"],
            },
            {
                "tipping_id": "amazon_dieback",
                "name": "Amazon Rainforest Dieback",
                "system": GaiaSystem.BIODIVERSITY_NETWORKS,
                "description": "Large-scale transition of Amazon from rainforest to savanna.",
                "threshold": "~20-25% deforestation (currently ~17%) plus 3-4C regional warming",
                "current_distance": "Approaching - deforestation and drought increasing",
                "trigger_mechanisms": ["Deforestation", "Drought", "Fire", "Reduced transpiration"],
                "consequences": ["Massive carbon release (~90 Gt C)", "Biodiversity catastrophe", "Regional climate change", "Indigenous peoples displacement"],
                "reversibility": "May be irreversible once self-sustaining - forest creates own rainfall",
                "timescale": "Decades to century",
                "early_warnings": ["Increasing dry season length", "More intense fires", "Reduced productivity", "Eastern Amazon already carbon source"],
                "cascading_effects": ["Global carbon cycle disruption", "South American rainfall changes", "Accelerated warming"],
            },
            {
                "tipping_id": "amoc_collapse",
                "name": "Atlantic Meridional Overturning Circulation Collapse",
                "system": GaiaSystem.OCEAN_CIRCULATION,
                "description": "Shutdown or major weakening of the Atlantic conveyor belt.",
                "threshold": "Freshwater input threshold - 1.5-2.0C warming may be sufficient",
                "current_distance": "AMOC already weakened ~15% - approaching tipping zone",
                "trigger_mechanisms": ["Freshwater input from Greenland", "Arctic sea ice melt", "Changed precipitation"],
                "consequences": ["Rapid European cooling", "Shifted tropical rainfall", "Disrupted monsoons", "Sea level changes"],
                "reversibility": "May recover over centuries if freshwater input stops",
                "timescale": "Decades to centuries",
                "early_warnings": ["Continued AMOC weakening", "North Atlantic cooling blob", "Changed deep water formation"],
                "cascading_effects": ["European climate disruption", "African and Asian monsoon changes", "Marine ecosystem shifts"],
            },
            {
                "tipping_id": "permafrost_collapse",
                "name": "Permafrost Carbon Release",
                "system": GaiaSystem.SOIL_ECOSYSTEMS,
                "description": "Large-scale thawing of Arctic permafrost releasing stored carbon.",
                "threshold": "~1.5-2C global warming (already occurring)",
                "current_distance": "Already beginning - permafrost is thawing",
                "trigger_mechanisms": ["Arctic amplification", "Thermokarst formation", "Microbial decomposition activation"],
                "consequences": ["Large CO2 and CH4 release", "Accelerated warming", "Infrastructure damage", "Landscape transformation"],
                "reversibility": "Carbon release is irreversible; permafrost would take millennia to reform",
                "timescale": "Ongoing over decades to centuries",
                "early_warnings": ["Ground subsidence", "Thermokarst lakes", "Increased winter CO2", "Methane seeps"],
                "cascading_effects": ["Amplified global warming", "Arctic ecosystem transformation", "Infrastructure damage"],
            },
            {
                "tipping_id": "coral_reef_collapse",
                "name": "Coral Reef Die-off",
                "system": GaiaSystem.BIODIVERSITY_NETWORKS,
                "description": "Mass mortality of coral reefs from warming and acidification.",
                "threshold": "~1.5C warming (reefs begin dying at current levels)",
                "current_distance": "Already in crisis - mass bleaching events increasing",
                "trigger_mechanisms": ["Ocean warming", "Ocean acidification", "Bleaching", "Disease"],
                "consequences": ["Loss of 25% of marine species habitat", "Fisheries collapse", "Coastal protection loss", "Economic devastation"],
                "reversibility": "Recovery possible if conditions improve, but unlikely at >1.5C",
                "timescale": "Already underway - decades to see full effects",
                "early_warnings": ["Mass bleaching events", "Reduced calcification", "Algae dominance", "Structural collapse"],
                "cascading_effects": ["Marine food web disruption", "Coastal community impacts", "Tourism collapse"],
            },
            {
                "tipping_id": "west_antarctic_ice",
                "name": "West Antarctic Ice Sheet Collapse",
                "system": GaiaSystem.ALBEDO_REGULATION,
                "description": "Irreversible marine ice sheet instability in West Antarctica.",
                "threshold": "Possibly already triggered at current warming levels",
                "current_distance": "May have already crossed - Thwaites glacier retreating",
                "trigger_mechanisms": ["Warm water intrusion", "Marine ice sheet instability", "Ice cliff collapse"],
                "consequences": ["3-5m sea level rise potential", "Global coastal flooding", "Changed ocean circulation"],
                "reversibility": "Irreversible on human timescales once triggered",
                "timescale": "Centuries for full collapse, but committed",
                "early_warnings": ["Grounding line retreat", "Accelerating ice loss", "Warm water observations"],
                "cascading_effects": ["Global sea level rise", "Southern Ocean changes", "Antarctic ecosystem disruption"],
            },
        ]

    async def initialize_seed_boundaries(self) -> int:
        """Initialize with seed boundary data."""
        seed_boundaries = self._get_seed_boundaries()
        count = 0

        for boundary_data in seed_boundaries:
            boundary = PlanetaryBoundaryState(
                boundary_id=boundary_data["boundary_id"],
                name=boundary_data["name"],
                boundary_type=boundary_data["boundary_type"],
                description=boundary_data.get("description", ""),
                control_variable=boundary_data.get("control_variable", ""),
                current_value=boundary_data.get("current_value", ""),
                safe_threshold=boundary_data.get("safe_threshold", ""),
                danger_zone=boundary_data.get("danger_zone", ""),
                status=boundary_data.get("status", BoundaryStatus.UNCERTAIN),
                trend=boundary_data.get("trend", ""),
                key_drivers=boundary_data.get("key_drivers", []),
                consequences=boundary_data.get("consequences", []),
            )
            await self.add_boundary(boundary)
            count += 1

        logger.info(f"Initialized {count} seed planetary boundaries")
        return count

    async def initialize_seed_feedbacks(self) -> int:
        """Initialize with seed feedback data."""
        seed_feedbacks = self._get_seed_feedbacks()
        count = 0

        for feedback_data in seed_feedbacks:
            feedback = ClimateFeedback(
                feedback_id=feedback_data["feedback_id"],
                name=feedback_data["name"],
                feedback_type=feedback_data["feedback_type"],
                description=feedback_data.get("description", ""),
                systems_involved=feedback_data.get("systems_involved", []),
                mechanism=feedback_data.get("mechanism", ""),
                magnitude=feedback_data.get("magnitude", ""),
                timescale=feedback_data.get("timescale", ""),
                certainty=feedback_data.get("certainty", ""),
                current_activation=feedback_data.get("current_activation", ""),
                tipping_potential=feedback_data.get("tipping_potential", False),
            )
            await self.add_feedback(feedback)
            count += 1

        logger.info(f"Initialized {count} seed climate feedbacks")
        return count

    async def initialize_seed_perspectives(self) -> int:
        """Initialize with seed indigenous perspectives."""
        seed_perspectives = self._get_seed_perspectives()
        count = 0

        for perspective_data in seed_perspectives:
            perspective = IndigenousEarthPerspective(
                perspective_id=perspective_data["perspective_id"],
                name=perspective_data["name"],
                tradition=perspective_data["tradition"],
                description=perspective_data.get("description", ""),
                earth_conception=perspective_data.get("earth_conception", ""),
                human_role=perspective_data.get("human_role", ""),
                ethical_principles=perspective_data.get("ethical_principles", []),
                practices=perspective_data.get("practices", []),
                key_concepts=perspective_data.get("key_concepts", []),
                source_communities=perspective_data.get("source_communities", []),
                legal_recognition=perspective_data.get("legal_recognition", []),
                form_29_link=perspective_data.get("form_29_link"),
            )
            await self.add_perspective(perspective)
            count += 1

        logger.info(f"Initialized {count} seed indigenous perspectives")
        return count

    async def initialize_seed_tipping_points(self) -> int:
        """Initialize with seed tipping point data."""
        seed_tipping_points = self._get_seed_tipping_points()
        count = 0

        for tp_data in seed_tipping_points:
            tipping_point = TippingPoint(
                tipping_id=tp_data["tipping_id"],
                name=tp_data["name"],
                system=tp_data["system"],
                description=tp_data.get("description", ""),
                threshold=tp_data.get("threshold", ""),
                current_distance=tp_data.get("current_distance", ""),
                trigger_mechanisms=tp_data.get("trigger_mechanisms", []),
                consequences=tp_data.get("consequences", []),
                reversibility=tp_data.get("reversibility", ""),
                timescale=tp_data.get("timescale", ""),
                early_warnings=tp_data.get("early_warnings", []),
                cascading_effects=tp_data.get("cascading_effects", []),
            )
            await self.add_tipping_point(tipping_point)
            count += 1

        logger.info(f"Initialized {count} seed tipping points")
        return count

    async def initialize_all_seed_data(self) -> Dict[str, int]:
        """Initialize all seed data."""
        await self.initialize()

        boundaries_count = await self.initialize_seed_boundaries()
        feedbacks_count = await self.initialize_seed_feedbacks()
        perspectives_count = await self.initialize_seed_perspectives()
        tipping_points_count = await self.initialize_seed_tipping_points()

        total = boundaries_count + feedbacks_count + perspectives_count + tipping_points_count

        return {
            "boundaries": boundaries_count,
            "feedbacks": feedbacks_count,
            "perspectives": perspectives_count,
            "tipping_points": tipping_points_count,
            "total": total
        }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "GaiaSystem",
    "PlanetaryBoundary",
    "EcologicalIntelligenceType",
    "FeedbackType",
    "IndigenousEarthTradition",
    "GaiaPhilosophy",
    "MaturityLevel",
    "BoundaryStatus",
    # Dataclasses
    "EarthSystemComponent",
    "PlanetaryBoundaryState",
    "ClimateFeedback",
    "IndigenousEarthPerspective",
    "TippingPoint",
    "GaiaIntelligenceMaturityState",
    # Interface
    "GaiaIntelligenceInterface",
]
