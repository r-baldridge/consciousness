"""
Philosophical Adapter - Form 28: Philosophical Consciousness
Provides interface for philosophical reasoning and knowledge integration.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base_adapter import TheoreticalAdapter

logger = logging.getLogger(__name__)


class PhilosophicalAdapter(TheoreticalAdapter):
    """
    Adapter for Form 28: Philosophical Consciousness.

    Integrates philosophical reasoning, cross-tradition synthesis,
    and wisdom integration with the consciousness system.

    Uses RAG embeddings for semantic retrieval of philosophical knowledge.
    """

    FORM_ID = "28-philosophy"
    NAME = "Philosophical Consciousness"
    THEORY = "philosophical_consciousness"

    def __init__(self):
        super().__init__(self.FORM_ID, self.NAME, self.THEORY)

        # Philosophical processing state
        self._active_traditions: List[str] = []
        self._reasoning_mode: str = "analytical"
        self._cross_tradition_active: bool = False
        self._last_wisdom_context: Optional[Dict[str, Any]] = None

        # Maturity metrics
        self._queries_processed: int = 0
        self._syntheses_performed: int = 0
        self._research_triggered: int = 0

    async def preprocess(self, input_data: Any) -> Dict[str, Any]:
        """
        Preprocess philosophical query input.

        Handles:
        - String queries (natural language)
        - Dict with structured query options
        - Concept lookup requests
        """
        if isinstance(input_data, str):
            return {
                'query': input_data,
                'query_type': 'semantic',
                'filters': {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        if isinstance(input_data, dict):
            return {
                'query': input_data.get('query', ''),
                'query_type': input_data.get('type', 'semantic'),
                'tradition_filter': input_data.get('traditions', []),
                'domain_filter': input_data.get('domains', []),
                'synthesis_requested': input_data.get('synthesize', False),
                'filters': input_data.get('filters', {}),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        return {
            'query': str(input_data),
            'query_type': 'semantic',
            'filters': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess philosophical reasoning output.

        Structures results with:
        - Retrieved concepts
        - Relevance scores
        - Wisdom teachings
        - Synthesis (if performed)
        """
        self._record_inference()
        self._queries_processed += 1

        if isinstance(model_output, dict):
            concepts = model_output.get('concepts', [])
            figures = model_output.get('figures', [])
            synthesis = model_output.get('synthesis', None)
            wisdom = model_output.get('wisdom_teachings', [])
            relevance = model_output.get('relevance_scores', {})
        else:
            concepts = []
            figures = []
            synthesis = None
            wisdom = []
            relevance = {}

        # Track if synthesis was performed
        if synthesis:
            self._syntheses_performed += 1
            self._cross_tradition_active = True

        return {
            'form_id': self.FORM_ID,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'concepts': concepts,
            'figures': figures,
            'synthesis': synthesis,
            'wisdom_teachings': wisdom,
            'relevance_scores': relevance,
            'theory_metrics': self.get_theory_metrics(),
            'philosophical_maturity': self._calculate_maturity(),
        }

    async def inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Perform philosophical reasoning inference.

        This adapter primarily serves as the neural network interface
        for the PhilosophicalConsciousnessInterface.
        """
        processed = await self.preprocess(input_data)

        if not self.has_model:
            # Return mock output for testing/development
            model_output = self._mock_philosophical_output(processed)
        else:
            try:
                if self.model and self.model.model_instance:
                    model_output = self.model.model_instance(processed)
                else:
                    model_output = self._mock_philosophical_output(processed)
            except Exception as e:
                self._record_error()
                logger.error(f"Philosophical inference error: {e}")
                raise

        return await self.postprocess(model_output)

    def validate_input(self, input_data: Any) -> bool:
        """Validate philosophical input."""
        if isinstance(input_data, str):
            return len(input_data.strip()) > 0

        if isinstance(input_data, dict):
            return 'query' in input_data or 'concept' in input_data

        return False

    def get_input_spec(self) -> Dict[str, Any]:
        """Get the input specification for philosophical queries."""
        return {
            'type': 'philosophical_query',
            'supported_types': [
                {'type': 'string', 'description': 'Natural language query'},
                {'type': 'dict', 'required_keys': ['query']},
            ],
            'optional_filters': ['traditions', 'domains', 'figures'],
            'supports_synthesis': True,
            'supports_research_trigger': True,
        }

    def get_output_spec(self) -> Dict[str, Any]:
        """Get the output specification for philosophical results."""
        return {
            'type': 'philosophical_response',
            'includes': {
                'concepts': 'List of matching philosophical concepts',
                'figures': 'List of relevant philosophers',
                'synthesis': 'Cross-tradition synthesis (if requested)',
                'wisdom_teachings': 'Practical wisdom from traditions',
                'relevance_scores': 'Semantic similarity scores',
                'theory_metrics': 'Adapter performance metrics',
            },
        }

    def get_theory_metrics(self) -> Dict[str, Any]:
        """Get philosophy-specific metrics."""
        return {
            'theory': self.THEORY,
            'form_id': self.FORM_ID,
            'active_traditions': self._active_traditions,
            'reasoning_mode': self._reasoning_mode,
            'cross_tradition_active': self._cross_tradition_active,
            'queries_processed': self._queries_processed,
            'syntheses_performed': self._syntheses_performed,
            'research_triggered': self._research_triggered,
        }

    def _calculate_maturity(self) -> Dict[str, Any]:
        """Calculate philosophical maturity metrics."""
        # Simplified maturity calculation
        query_factor = min(1.0, self._queries_processed / 1000)
        synthesis_factor = min(1.0, self._syntheses_performed / 100)

        return {
            'maturity_score': (query_factor * 0.6 + synthesis_factor * 0.4),
            'queries_processed': self._queries_processed,
            'syntheses_performed': self._syntheses_performed,
        }

    def _mock_philosophical_output(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock output for testing."""
        query = processed.get('query', '')

        return {
            'concepts': [
                {
                    'name': f'Concept related to: {query[:30]}',
                    'tradition': 'analytic',
                    'domain': 'metaphysics',
                    'relevance': 0.75,
                }
            ],
            'figures': [],
            'synthesis': None,
            'wisdom_teachings': [
                'The examined life is worth living - Socrates',
            ],
            'relevance_scores': {
                'mock_concept': 0.75
            },
        }

    # ========================================================================
    # PHILOSOPHICAL-SPECIFIC METHODS
    # ========================================================================

    def set_reasoning_mode(self, mode: str) -> None:
        """
        Set the philosophical reasoning mode.

        Modes:
        - analytical: Logical analysis, clarity focus
        - dialectical: Thesis-antithesis-synthesis
        - phenomenological: Direct experience investigation
        - hermeneutic: Interpretive understanding
        - pragmatic: Practical consequences focus
        - contemplative: Meditative inquiry
        """
        valid_modes = [
            'analytical', 'dialectical', 'phenomenological',
            'hermeneutic', 'pragmatic', 'contemplative'
        ]
        if mode in valid_modes:
            self._reasoning_mode = mode
            logger.debug(f"Reasoning mode set to: {mode}")

    def set_active_traditions(self, traditions: List[str]) -> None:
        """Set the active philosophical traditions for processing."""
        self._active_traditions = traditions
        logger.debug(f"Active traditions: {traditions}")

    def enable_cross_tradition_synthesis(self, enable: bool = True) -> None:
        """Enable or disable cross-tradition synthesis."""
        self._cross_tradition_active = enable

    def get_wisdom_context(self) -> Optional[Dict[str, Any]]:
        """Get the last wisdom context for integration."""
        return self._last_wisdom_context

    async def request_wisdom_for_engagement(
        self,
        engagement_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Request appropriate wisdom for an engagement context.

        This method interfaces with the enlightened engagement protocols.
        """
        # Store context
        self._last_wisdom_context = engagement_context

        # Generate wisdom response based on context
        emotional_state = engagement_context.get('emotional_state', 'neutral')
        capacity = engagement_context.get('recipient_capacity', 'intermediate')

        wisdom_response = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': engagement_context,
            'wisdom_teachings': self._select_wisdom_for_state(emotional_state, capacity),
            'practical_guidance': self._select_guidance_for_state(emotional_state),
            'traditions_used': self._active_traditions or ['stoicism', 'buddhist'],
        }

        return wisdom_response

    def _select_wisdom_for_state(
        self,
        emotional_state: str,
        capacity: str
    ) -> List[str]:
        """Select appropriate wisdom teachings based on state."""
        teachings = []

        if emotional_state == 'distressed':
            teachings = [
                "This too shall pass - both joy and sorrow are temporary",
                "Focus on what is within your control",
            ]
        elif emotional_state == 'seeking':
            teachings = [
                "The unexamined life is not worth living",
                "Know thyself",
            ]
        else:
            teachings = [
                "Virtue is its own reward",
                "The present moment is the only teacher you need",
            ]

        return teachings

    def _select_guidance_for_state(self, emotional_state: str) -> List[str]:
        """Select practical guidance based on emotional state."""
        if emotional_state == 'distressed':
            return [
                "Pause and take three conscious breaths",
                "Ask: What is actually within my control here?",
            ]
        return [
            "Reflect on what truly matters",
            "Act in accordance with your values",
        ]
