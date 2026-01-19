"""
Ecosystem Adapters - Forms 31-34
Plant, fungal, swarm, and planetary/Gaia intelligence adapters.
Part of the Neural Network module for the Consciousness system.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base_adapter import SpecializedAdapter

logger = logging.getLogger(__name__)


class PlantIntelligenceAdapter(SpecializedAdapter):
    """
    Adapter for Form 31: Plant Intelligence & Vegetal Consciousness.

    Engages with knowledge about plant cognition, signaling, and behavior,
    integrating scientific research with indigenous perspectives on
    plant intelligence and human-plant relationships.

    Features:
    - Chemical signaling and root communication
    - Memory and habituation studies
    - Decision-making and resource allocation
    - Defense coordination networks
    - Integration with Form 29 (indigenous plant wisdom)
    """

    FORM_ID = "31-plant-intelligence"
    NAME = "Plant Intelligence & Vegetal Consciousness"
    SPECIALIZATION = "plant_intelligence"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._species_context: Dict[str, Any] = {}
        self._cognition_domains: List[str] = []
        self._indigenous_integration: bool = True

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess input for plant intelligence analysis."""
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            species = input_data.get('species', [])
            domains = input_data.get('domains', [])
        else:
            query = str(input_data)
            species = []
            domains = []

        detected_species = species or self._detect_plant_groups(query)
        detected_domains = domains or self._detect_cognition_domains(query)

        return {
            'query': query,
            'species': detected_species,
            'domains': detected_domains,
            'include_indigenous': self._indigenous_integration,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess plant intelligence output."""
        self._record_inference()

        if isinstance(model_output, dict):
            profiles = model_output.get('profiles', [])
            behaviors = model_output.get('behaviors', [])
            indigenous_wisdom = model_output.get('indigenous_wisdom', [])
        else:
            profiles = []
            behaviors = []
            indigenous_wisdom = []

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'species_profiles': profiles,
            'behavior_insights': behaviors,
            'indigenous_wisdom': indigenous_wisdom,
            'signaling_mechanisms': self._extract_signaling_mechanisms(profiles),
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run plant intelligence analysis."""
        processed = await self.preprocess(input_data)
        self._cognition_domains = processed['domains']

        result = {
            'profiles': [],
            'behaviors': [],
            'indigenous_wisdom': [] if processed['include_indigenous'] else None,
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data or 'species' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'plant_intelligence_query',
            'fields': {
                'query': 'string',
                'species': 'optional_list',
                'domains': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'plant_intelligence_response',
            'fields': {
                'species_profiles': 'list',
                'behavior_insights': 'list',
                'indigenous_wisdom': 'list',
                'signaling_mechanisms': 'dict',
            }
        }

    def _detect_plant_groups(self, query: str) -> List[str]:
        """Detect relevant plant groups from query."""
        groups = []
        query_lower = query.lower()

        group_keywords = {
            'trees': ['tree', 'forest', 'oak', 'pine', 'redwood', 'birch'],
            'flowering': ['flower', 'rose', 'orchid', 'sunflower'],
            'vines': ['vine', 'climbing', 'ivy'],
            'grasses': ['grass', 'bamboo', 'wheat', 'rice'],
            'carnivorous': ['venus', 'sundew', 'pitcher', 'carnivorous'],
            'sensitive': ['mimosa', 'sensitive plant', 'touch-me-not'],
        }

        for group, keywords in group_keywords.items():
            if any(kw in query_lower for kw in keywords):
                groups.append(group)

        return groups or ['general']

    def _detect_cognition_domains(self, query: str) -> List[str]:
        """Detect relevant cognition domains from query."""
        domains = []
        query_lower = query.lower()

        domain_keywords = {
            'chemical_signaling': ['signal', 'chemical', 'volatile', 'hormone'],
            'root_communication': ['root', 'mycorrhiza', 'underground', 'network'],
            'memory_habituation': ['memory', 'learn', 'habituate', 'remember'],
            'decision_making': ['decision', 'choice', 'allocate', 'resource'],
            'defense_coordination': ['defense', 'predator', 'herbivore', 'protect'],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)

        return domains or ['general']

    def _extract_signaling_mechanisms(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract signaling mechanisms from profiles."""
        return {
            'volatile_compounds': [],
            'root_exudates': [],
            'electrical_signals': [],
            'hydraulic_signals': [],
        }


class FungalIntelligenceAdapter(SpecializedAdapter):
    """
    Adapter for Form 32: Fungal Networks & Mycorrhizal Intelligence.

    Engages with knowledge about fungal cognition, network optimization,
    and the "wood wide web" - mycorrhizal networks connecting forest ecosystems.

    Features:
    - Network optimization algorithms (slime mold studies)
    - Resource distribution and allocation
    - Chemical communication systems
    - Memory traces and learning
    - Integration with Forms 31 (plant) and 34 (Gaia)
    """

    FORM_ID = "32-fungal-intelligence"
    NAME = "Fungal Networks & Mycorrhizal Intelligence"
    SPECIALIZATION = "fungal_intelligence"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._network_context: Dict[str, Any] = {}
        self._intelligence_domains: List[str] = []

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess input for fungal network analysis."""
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            fungal_types = input_data.get('fungal_types', [])
            domains = input_data.get('domains', [])
        else:
            query = str(input_data)
            fungal_types = []
            domains = []

        detected_types = fungal_types or self._detect_fungal_types(query)
        detected_domains = domains or self._detect_intelligence_domains(query)

        return {
            'query': query,
            'fungal_types': detected_types,
            'domains': detected_domains,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess fungal network output."""
        self._record_inference()

        if isinstance(model_output, dict):
            profiles = model_output.get('profiles', [])
            experiments = model_output.get('experiments', [])
            indigenous = model_output.get('indigenous_wisdom', [])
        else:
            profiles = []
            experiments = []
            indigenous = []

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'network_profiles': profiles,
            'slime_mold_experiments': experiments,
            'indigenous_wisdom': indigenous,
            'network_properties': self._extract_network_properties(profiles),
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run fungal network analysis."""
        processed = await self.preprocess(input_data)
        self._intelligence_domains = processed['domains']

        result = {
            'profiles': [],
            'experiments': [],
            'indigenous_wisdom': [],
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'fungal_network_query',
            'fields': {
                'query': 'string',
                'fungal_types': 'optional_list',
                'domains': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'fungal_network_response',
            'fields': {
                'network_profiles': 'list',
                'slime_mold_experiments': 'list',
                'indigenous_wisdom': 'list',
                'network_properties': 'dict',
            }
        }

    def _detect_fungal_types(self, query: str) -> List[str]:
        """Detect relevant fungal types from query."""
        types = []
        query_lower = query.lower()

        type_keywords = {
            'mycorrhizal': ['mycorrhiza', 'wood wide web', 'tree network'],
            'slime_mold': ['slime mold', 'physarum', 'dictyostelium'],
            'decomposers': ['decompos', 'saprotroph', 'decay'],
            'parasitic': ['parasit', 'cordyceps', 'ophiocordyceps'],
            'symbiotic': ['symbio', 'lichen', 'endophyte'],
        }

        for ftype, keywords in type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                types.append(ftype)

        return types or ['general']

    def _detect_intelligence_domains(self, query: str) -> List[str]:
        """Detect relevant intelligence domains from query."""
        domains = []
        query_lower = query.lower()

        domain_keywords = {
            'network_optimization': ['network', 'optim', 'path', 'efficient'],
            'resource_distribution': ['resource', 'nutri', 'distribut', 'share'],
            'chemical_communication': ['chemical', 'signal', 'communicat'],
            'memory_traces': ['memory', 'learn', 'remember', 'adapt'],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)

        return domains or ['general']

    def _extract_network_properties(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract network properties from profiles."""
        return {
            'topology': [],
            'efficiency_metrics': [],
            'resilience_factors': [],
            'information_flow': [],
        }


class SwarmIntelligenceAdapter(SpecializedAdapter):
    """
    Adapter for Form 33: Swarm & Collective Intelligence.

    Engages with knowledge about emergent collective behavior in
    biological and social systems - from ant colonies to human crowds.

    Features:
    - Ant colony optimization algorithms
    - Bee hive decision-making
    - Bird flocking and fish schooling dynamics
    - Human crowd behavior
    - Emergence detection and analysis
    """

    FORM_ID = "33-swarm-intelligence"
    NAME = "Swarm & Collective Intelligence"
    SPECIALIZATION = "swarm_intelligence"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._system_context: Dict[str, Any] = {}
        self._collective_systems: List[str] = []

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess input for swarm intelligence analysis."""
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            systems = input_data.get('systems', [])
            phenomena = input_data.get('phenomena', [])
        else:
            query = str(input_data)
            systems = []
            phenomena = []

        detected_systems = systems or self._detect_collective_systems(query)
        detected_phenomena = phenomena or self._detect_phenomena(query)

        return {
            'query': query,
            'systems': detected_systems,
            'phenomena': detected_phenomena,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess swarm intelligence output."""
        self._record_inference()

        if isinstance(model_output, dict):
            profiles = model_output.get('profiles', [])
            observations = model_output.get('observations', [])
            emergence = model_output.get('emergence_events', [])
        else:
            profiles = []
            observations = []
            emergence = []

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_profiles': profiles,
            'behavior_observations': observations,
            'emergence_events': emergence,
            'collective_properties': self._extract_collective_properties(profiles),
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run swarm intelligence analysis."""
        processed = await self.preprocess(input_data)
        self._collective_systems = processed['systems']

        result = {
            'profiles': [],
            'observations': [],
            'emergence_events': [],
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'swarm_intelligence_query',
            'fields': {
                'query': 'string',
                'systems': 'optional_list',
                'phenomena': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'swarm_intelligence_response',
            'fields': {
                'system_profiles': 'list',
                'behavior_observations': 'list',
                'emergence_events': 'list',
                'collective_properties': 'dict',
            }
        }

    def _detect_collective_systems(self, query: str) -> List[str]:
        """Detect relevant collective systems from query."""
        systems = []
        query_lower = query.lower()

        system_keywords = {
            'ant_colonies': ['ant', 'colony', 'foraging', 'pheromone'],
            'bee_hives': ['bee', 'hive', 'waggle', 'honey'],
            'bird_flocks': ['bird', 'flock', 'murmuration', 'starling'],
            'fish_schools': ['fish', 'school', 'shoal', 'swim'],
            'human_crowds': ['crowd', 'human', 'social', 'market'],
        }

        for system, keywords in system_keywords.items():
            if any(kw in query_lower for kw in keywords):
                systems.append(system)

        return systems or ['general']

    def _detect_phenomena(self, query: str) -> List[str]:
        """Detect relevant swarm phenomena from query."""
        phenomena = []
        query_lower = query.lower()

        phenomena_keywords = {
            'emergence': ['emerg', 'self-organiz', 'spontaneous'],
            'coordination': ['coordinat', 'synchron', 'align'],
            'decision_making': ['decision', 'consensus', 'vote'],
            'optimization': ['optim', 'efficient', 'path'],
            'adaptation': ['adapt', 'learn', 'evolve'],
        }

        for phenomenon, keywords in phenomena_keywords.items():
            if any(kw in query_lower for kw in keywords):
                phenomena.append(phenomenon)

        return phenomena or ['general']

    def _extract_collective_properties(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract collective properties from profiles."""
        return {
            'emergence_indicators': [],
            'coordination_mechanisms': [],
            'information_transfer': [],
            'collective_memory': [],
        }


class GaiaIntelligenceAdapter(SpecializedAdapter):
    """
    Adapter for Form 34: Ecological/Planetary Intelligence (Gaia).

    Engages with Earth system science, the Gaia hypothesis, and
    planetary-scale feedback loops that maintain conditions for life.

    Features:
    - Atmosphere-biosphere interactions
    - Ocean circulation and carbon cycle
    - Planetary boundary analysis
    - Biodiversity network effects
    - Indigenous perspectives on Earth as living system
    """

    FORM_ID = "34-gaia-intelligence"
    NAME = "Ecological/Planetary Intelligence"
    SPECIALIZATION = "gaia_intelligence"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._system_context: Dict[str, Any] = {}
        self._gaia_systems: List[str] = []
        self._indigenous_integration: bool = True

    async def preprocess(self, input_data: Any) -> Any:
        """Preprocess input for Gaia system analysis."""
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            systems = input_data.get('systems', [])
            boundaries = input_data.get('boundaries', [])
        else:
            query = str(input_data)
            systems = []
            boundaries = []

        detected_systems = systems or self._detect_gaia_systems(query)
        detected_boundaries = boundaries or self._detect_planetary_boundaries(query)

        return {
            'query': query,
            'systems': detected_systems,
            'boundaries': detected_boundaries,
            'include_indigenous': self._indigenous_integration,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess Gaia system output."""
        self._record_inference()

        if isinstance(model_output, dict):
            components = model_output.get('components', [])
            boundaries = model_output.get('boundary_states', [])
            feedback = model_output.get('feedback_analysis', [])
            indigenous = model_output.get('indigenous_perspectives', [])
        else:
            components = []
            boundaries = []
            feedback = []
            indigenous = []

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_components': components,
            'boundary_states': boundaries,
            'feedback_analysis': feedback,
            'indigenous_perspectives': indigenous,
            'planetary_health': self._assess_planetary_health(boundaries),
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run Gaia system analysis."""
        processed = await self.preprocess(input_data)
        self._gaia_systems = processed['systems']

        result = {
            'components': [],
            'boundary_states': [],
            'feedback_analysis': [],
            'indigenous_perspectives': [] if processed['include_indigenous'] else None,
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'gaia_system_query',
            'fields': {
                'query': 'string',
                'systems': 'optional_list',
                'boundaries': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'gaia_system_response',
            'fields': {
                'system_components': 'list',
                'boundary_states': 'list',
                'feedback_analysis': 'list',
                'indigenous_perspectives': 'list',
                'planetary_health': 'dict',
            }
        }

    def _detect_gaia_systems(self, query: str) -> List[str]:
        """Detect relevant Gaia systems from query."""
        systems = []
        query_lower = query.lower()

        system_keywords = {
            'atmosphere': ['atmospher', 'air', 'climate', 'weather', 'ozone'],
            'ocean_circulation': ['ocean', 'current', 'thermohaline', 'sea'],
            'carbon_cycle': ['carbon', 'co2', 'greenhouse', 'sequester'],
            'biodiversity_networks': ['biodivers', 'species', 'ecosystem', 'food web'],
            'water_cycle': ['water', 'hydro', 'rain', 'evapor'],
        }

        for system, keywords in system_keywords.items():
            if any(kw in query_lower for kw in keywords):
                systems.append(system)

        return systems or ['general']

    def _detect_planetary_boundaries(self, query: str) -> List[str]:
        """Detect relevant planetary boundaries from query."""
        boundaries = []
        query_lower = query.lower()

        boundary_keywords = {
            'climate_change': ['climate', 'warming', 'temperature'],
            'biosphere_integrity': ['biodiversity', 'extinction', 'species loss'],
            'land_system': ['deforest', 'land use', 'agriculture'],
            'freshwater': ['freshwater', 'aquifer', 'river'],
            'ocean_acidification': ['acidif', 'ph', 'coral'],
            'nitrogen_cycle': ['nitrogen', 'phosphorus', 'nutrient'],
        }

        for boundary, keywords in boundary_keywords.items():
            if any(kw in query_lower for kw in keywords):
                boundaries.append(boundary)

        return boundaries or ['general']

    def _assess_planetary_health(self, boundaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess planetary health from boundary states."""
        return {
            'boundaries_crossed': [],
            'safe_operating_space': [],
            'tipping_point_risks': [],
            'regeneration_potential': [],
        }
