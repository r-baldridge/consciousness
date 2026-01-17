"""
Extended Adapters - Forms 28-30
Philosophical consciousness, folk wisdom, and animal cognition adapters.
Part of the Neural Network module for the Consciousness system.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base_adapter import SpecializedAdapter

logger = logging.getLogger(__name__)


class PhilosophyAdapter(SpecializedAdapter):
    """
    Adapter for Form 28: Philosophical Consciousness.

    Integrates Western and Eastern philosophical traditions with agentic
    research capabilities for systematic inquiry into fundamental questions
    about existence, knowledge, values, reason, mind, and language.

    Features:
    - Cross-tradition synthesis (Western, Eastern, Indigenous)
    - Dialectical reasoning and argument analysis
    - Ontological, epistemological, and axiological processing
    - Integration with Forms 10 (Memory), 11 (Meta), 14 (Global Workspace)
    - Agentic research for knowledge expansion
    """

    FORM_ID = "28-philosophy"
    NAME = "Philosophical Consciousness"
    SPECIALIZATION = "philosophical_reasoning"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._philosophical_context: Dict[str, Any] = {}
        self._active_traditions: List[str] = []
        self._reasoning_mode: str = "dialectical"
        self._maturity_level: float = 0.0

    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input for philosophical analysis.

        Detects relevant philosophical traditions and domains,
        prepares context for reasoning engine.
        """
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            context = input_data.get('context', {})
            traditions = input_data.get('traditions', [])
        else:
            query = str(input_data)
            context = {}
            traditions = []

        # Detect philosophical domains
        domains = self._detect_domains(query)

        return {
            'query': query,
            'context': context,
            'traditions': traditions or self._detect_traditions(query),
            'domains': domains,
            'reasoning_mode': self._reasoning_mode,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess philosophical reasoning output.

        Formats synthesis, identifies cross-tradition insights,
        and prepares for integration with other forms.
        """
        self._record_inference()

        if isinstance(model_output, dict):
            synthesis = model_output.get('synthesis', {})
            arguments = model_output.get('arguments', [])
            traditions_consulted = model_output.get('traditions', [])
        else:
            synthesis = {'raw_output': model_output}
            arguments = []
            traditions_consulted = []

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'synthesis': synthesis,
            'arguments': arguments,
            'traditions_consulted': traditions_consulted,
            'cross_tradition_insights': self._extract_cross_tradition_insights(synthesis),
            'maturity_level': self._maturity_level,
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run philosophical reasoning inference."""
        processed = await self.preprocess(input_data)

        # Simulate philosophical reasoning (actual model would be used in production)
        result = {
            'synthesis': {
                'query': processed['query'],
                'domains': processed['domains'],
                'perspective': 'integrated_multi_tradition',
            },
            'arguments': [],
            'traditions': processed['traditions'],
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        """Validate philosophical query input."""
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'philosophical_query',
            'fields': {
                'query': 'string',
                'context': 'optional_dict',
                'traditions': 'optional_list',
                'domains': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'philosophical_synthesis',
            'fields': {
                'synthesis': 'dict',
                'arguments': 'list',
                'traditions_consulted': 'list',
                'cross_tradition_insights': 'list',
            }
        }

    def _detect_domains(self, query: str) -> List[str]:
        """Detect relevant philosophical domains from query."""
        domains = []
        query_lower = query.lower()

        domain_keywords = {
            'ontology': ['existence', 'being', 'reality', 'substance', 'essence'],
            'epistemology': ['knowledge', 'truth', 'belief', 'justification', 'certainty'],
            'ethics': ['moral', 'ethical', 'virtue', 'duty', 'good', 'right', 'wrong'],
            'aesthetics': ['beauty', 'art', 'sublime', 'taste', 'aesthetic'],
            'logic': ['argument', 'validity', 'inference', 'reason', 'logic'],
            'metaphysics': ['mind', 'consciousness', 'free will', 'causation', 'time'],
            'phenomenology': ['experience', 'perception', 'intentionality', 'lived'],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)

        return domains or ['general']

    def _detect_traditions(self, query: str) -> List[str]:
        """Detect relevant philosophical traditions from query."""
        traditions = []
        query_lower = query.lower()

        tradition_keywords = {
            'western_analytic': ['logic', 'language', 'science', 'analysis'],
            'western_continental': ['phenomenology', 'existential', 'hermeneutic'],
            'eastern_buddhist': ['buddhist', 'dharma', 'suffering', 'emptiness', 'mindfulness'],
            'eastern_hindu': ['hindu', 'vedanta', 'yoga', 'atman', 'brahman'],
            'eastern_taoist': ['tao', 'taoist', 'wu wei', 'yin yang'],
            'eastern_confucian': ['confucian', 'virtue', 'harmony', 'filial'],
        }

        for tradition, keywords in tradition_keywords.items():
            if any(kw in query_lower for kw in keywords):
                traditions.append(tradition)

        return traditions or ['cross_tradition']

    def _extract_cross_tradition_insights(self, synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights that bridge multiple traditions."""
        return []  # Populated by actual reasoning engine

    async def get_philosophical_state(self) -> Dict[str, Any]:
        """Get current philosophical processing state."""
        return {
            'form_id': self.FORM_ID,
            'active_traditions': self._active_traditions,
            'reasoning_mode': self._reasoning_mode,
            'maturity_level': self._maturity_level,
            'context_size': len(self._philosophical_context),
        }

    def set_reasoning_mode(self, mode: str) -> None:
        """Set the reasoning mode (dialectical, analytic, phenomenological, etc.)."""
        valid_modes = ['dialectical', 'analytic', 'phenomenological', 'hermeneutic', 'pragmatic']
        if mode in valid_modes:
            self._reasoning_mode = mode


class FolkWisdomAdapter(SpecializedAdapter):
    """
    Adapter for Form 29: Folk & Indigenous Wisdom.

    Engages with folk traditions, indigenous wisdom, and animistic practices
    from cultures worldwide, bridging formal philosophy with lived wisdom
    embedded in oral traditions, ceremonial practices, and traditional
    ecological knowledge.

    Features:
    - Global coverage (Africa, Europe, Asia, Oceania, Americas)
    - Oral tradition processing (stories, songs, proverbs)
    - Animistic practice understanding
    - Indigenous cosmology integration
    - Ethical handling with source attribution
    """

    FORM_ID = "29-folk-wisdom"
    NAME = "Folk & Indigenous Wisdom"
    SPECIALIZATION = "folk_indigenous_wisdom"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._active_regions: List[str] = []
        self._wisdom_context: Dict[str, Any] = {}
        self._ethical_flags: Dict[str, bool] = {
            'source_attribution': True,
            'cultural_context': True,
            'sacred_boundaries': True,
        }

    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input for folk wisdom retrieval and analysis.

        Identifies relevant regions, traditions, and content categories.
        """
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            regions = input_data.get('regions', [])
            categories = input_data.get('categories', [])
        else:
            query = str(input_data)
            regions = []
            categories = []

        # Detect regions and categories
        detected_regions = regions or self._detect_regions(query)
        detected_categories = categories or self._detect_categories(query)

        return {
            'query': query,
            'regions': detected_regions,
            'categories': detected_categories,
            'ethical_context': self._ethical_flags,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess folk wisdom output.

        Ensures proper attribution and cultural context,
        formats for integration with Forms 28 and 30.
        """
        self._record_inference()

        if isinstance(model_output, dict):
            teachings = model_output.get('teachings', [])
            sources = model_output.get('sources', [])
            cultural_context = model_output.get('cultural_context', {})
        else:
            teachings = []
            sources = []
            cultural_context = {}

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'teachings': teachings,
            'sources': sources,
            'cultural_context': cultural_context,
            'regions_consulted': self._active_regions,
            'attribution_complete': True,
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run folk wisdom retrieval and synthesis."""
        processed = await self.preprocess(input_data)
        self._active_regions = processed['regions']

        # Simulate folk wisdom retrieval
        result = {
            'teachings': [],
            'sources': [],
            'cultural_context': {
                'regions': processed['regions'],
                'categories': processed['categories'],
            },
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        """Validate folk wisdom query input."""
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'folk_wisdom_query',
            'fields': {
                'query': 'string',
                'regions': 'optional_list',
                'categories': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'folk_wisdom_synthesis',
            'fields': {
                'teachings': 'list',
                'sources': 'list',
                'cultural_context': 'dict',
            }
        }

    def _detect_regions(self, query: str) -> List[str]:
        """Detect relevant world regions from query."""
        regions = []
        query_lower = query.lower()

        region_keywords = {
            'africa': ['african', 'yoruba', 'akan', 'zulu', 'maasai', 'dogon'],
            'europe': ['celtic', 'norse', 'slavic', 'baltic', 'finnish', 'germanic'],
            'asia': ['siberian', 'mongol', 'tibetan', 'balinese', 'japanese folk'],
            'oceania': ['polynesian', 'maori', 'aboriginal', 'hawaiian', 'fijian'],
            'americas': ['inuit', 'lakota', 'maya', 'amazonian', 'andean', 'cherokee'],
        }

        for region, keywords in region_keywords.items():
            if any(kw in query_lower for kw in keywords):
                regions.append(region)

        return regions or ['global']

    def _detect_categories(self, query: str) -> List[str]:
        """Detect relevant content categories from query."""
        categories = []
        query_lower = query.lower()

        category_keywords = {
            'wisdom_teachings': ['wisdom', 'teaching', 'elder', 'proverb'],
            'animistic_practices': ['spirit', 'ancestor', 'ritual', 'ceremony'],
            'cosmologies': ['creation', 'world', 'cosmos', 'origin'],
            'oral_traditions': ['story', 'song', 'legend', 'myth', 'tale'],
            'ecological_knowledge': ['nature', 'animal', 'plant', 'land', 'season'],
        }

        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                categories.append(category)

        return categories or ['general']

    async def get_folk_wisdom_state(self) -> Dict[str, Any]:
        """Get current folk wisdom processing state."""
        return {
            'form_id': self.FORM_ID,
            'active_regions': self._active_regions,
            'ethical_flags': self._ethical_flags,
            'context_size': len(self._wisdom_context),
        }


class AnimalCognitionAdapter(SpecializedAdapter):
    """
    Adapter for Form 30: Animal Cognition & Ethology.

    Engages with knowledge about animal minds, behavior, and consciousness,
    integrating Western scientific research with indigenous perspectives
    on animal intelligence and human-animal relationships.

    Features:
    - Cognitive profiles for mammals, birds, reptiles, fish, invertebrates
    - Multiple cognition domains (memory, learning, social, self-awareness)
    - Consciousness indicators tracking
    - Integration with Form 29 for indigenous animal wisdom
    - Cross-species synthesis capabilities
    """

    FORM_ID = "30-animal-cognition"
    NAME = "Animal Cognition & Ethology"
    SPECIALIZATION = "animal_cognition"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.SPECIALIZATION)
        self._species_context: Dict[str, Any] = {}
        self._cognition_domains: List[str] = []
        self._indigenous_integration: bool = True

    async def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input for animal cognition analysis.

        Identifies relevant species, cognition domains, and evidence types.
        """
        if isinstance(input_data, dict):
            query = input_data.get('query', input_data.get('input', ''))
            species = input_data.get('species', [])
            domains = input_data.get('domains', [])
        else:
            query = str(input_data)
            species = []
            domains = []

        # Detect species and domains
        detected_species = species or self._detect_species(query)
        detected_domains = domains or self._detect_cognition_domains(query)

        return {
            'query': query,
            'species': detected_species,
            'domains': detected_domains,
            'include_indigenous': self._indigenous_integration,
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess animal cognition output.

        Formats cognitive profiles, integrates indigenous perspectives,
        and prepares for cross-form integration.
        """
        self._record_inference()

        if isinstance(model_output, dict):
            profiles = model_output.get('profiles', [])
            evidence = model_output.get('evidence', [])
            indigenous_wisdom = model_output.get('indigenous_wisdom', [])
        else:
            profiles = []
            evidence = []
            indigenous_wisdom = []

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cognitive_profiles': profiles,
            'evidence': evidence,
            'indigenous_wisdom': indigenous_wisdom,
            'consciousness_indicators': self._extract_consciousness_indicators(profiles),
            'integration_ready': True,
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """Run animal cognition analysis."""
        processed = await self.preprocess(input_data)
        self._cognition_domains = processed['domains']

        # Simulate animal cognition analysis
        result = {
            'profiles': [],
            'evidence': [],
            'indigenous_wisdom': [] if processed['include_indigenous'] else None,
        }

        return await self.postprocess(result)

    def validate_input(self, input_data: Any) -> bool:
        """Validate animal cognition query input."""
        if isinstance(input_data, dict):
            return 'query' in input_data or 'input' in input_data or 'species' in input_data
        return isinstance(input_data, str) and len(input_data) > 0

    def get_input_spec(self) -> Dict[str, Any]:
        return {
            'type': 'animal_cognition_query',
            'fields': {
                'query': 'string',
                'species': 'optional_list',
                'domains': 'optional_list',
            }
        }

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'type': 'animal_cognition_synthesis',
            'fields': {
                'cognitive_profiles': 'list',
                'evidence': 'list',
                'indigenous_wisdom': 'list',
                'consciousness_indicators': 'dict',
            }
        }

    def _detect_species(self, query: str) -> List[str]:
        """Detect relevant species from query."""
        species = []
        query_lower = query.lower()

        species_keywords = {
            'great_apes': ['chimpanzee', 'bonobo', 'gorilla', 'orangutan', 'ape'],
            'cetaceans': ['dolphin', 'whale', 'orca', 'porpoise'],
            'elephants': ['elephant'],
            'corvids': ['crow', 'raven', 'jay', 'magpie', 'corvid'],
            'parrots': ['parrot', 'african grey', 'kea', 'macaw'],
            'cephalopods': ['octopus', 'cuttlefish', 'squid'],
            'canids': ['dog', 'wolf', 'fox'],
            'primates': ['monkey', 'capuchin', 'macaque', 'lemur'],
        }

        for species_group, keywords in species_keywords.items():
            if any(kw in query_lower for kw in keywords):
                species.append(species_group)

        return species or ['general']

    def _detect_cognition_domains(self, query: str) -> List[str]:
        """Detect relevant cognition domains from query."""
        domains = []
        query_lower = query.lower()

        domain_keywords = {
            'memory': ['memory', 'remember', 'recall', 'episodic'],
            'learning': ['learn', 'tool', 'problem', 'causal'],
            'social_cognition': ['social', 'cooperation', 'empathy', 'theory of mind'],
            'self_awareness': ['mirror', 'self', 'metacognition', 'agency'],
            'communication': ['communication', 'language', 'signal', 'syntax'],
            'emotion': ['emotion', 'grief', 'play', 'joy'],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)

        return domains or ['general']

    def _extract_consciousness_indicators(self, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract consciousness indicators from cognitive profiles."""
        return {
            'behavioral': [],
            'neuroanatomical': [],
            'neurophysiological': [],
            'indigenous_observation': [],
        }

    async def get_animal_cognition_state(self) -> Dict[str, Any]:
        """Get current animal cognition processing state."""
        return {
            'form_id': self.FORM_ID,
            'active_domains': self._cognition_domains,
            'indigenous_integration': self._indigenous_integration,
            'species_context_size': len(self._species_context),
        }

    def set_indigenous_integration(self, enabled: bool) -> None:
        """Enable or disable integration with Form 29 indigenous animal wisdom."""
        self._indigenous_integration = enabled
