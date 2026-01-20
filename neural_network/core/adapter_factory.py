"""
Adapter Factory - Central factory for creating and managing form adapters.
Part of the Neural Network module for the Consciousness system.

This factory:
- Maps form IDs to their adapter classes
- Instantiates adapters on demand
- Provides batch registration to NervousSystem
- Manages adapter lifecycle
"""

import logging
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .nervous_system import NervousSystem
    from ..adapters.base_adapter import FormAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# FORM REGISTRY - Maps form IDs to adapter classes
# =============================================================================

# Import all adapters
from ..adapters import (
    # Base classes
    FormAdapter,
    SensoryAdapter,
    CognitiveAdapter,
    TheoreticalAdapter,
    SpecializedAdapter,
    # Sensory (01-06)
    VisualAdapter,
    AuditoryAdapter,
    TactileAdapter,
    OlfactoryAdapter,
    GustatoryAdapter,
    ProprioceptiveAdapter,
    # Cognitive (07-12)
    AttentionAdapter,
    ArousalAdapter,
    MemorySTMAdapter,
    MemoryLTMAdapter,
    EmotionAdapter,
    ExecutiveAdapter,
    # Theoretical (13-17)
    IITAdapter,
    GlobalWorkspaceAdapter,
    HOTAdapter,
    PredictiveAdapter,
    RecurrentAdapter,
    # Specialized (18-27)
    PrimaryAdapter,
    ReflectiveAdapter,
    SocialAdapter,
    ArtificialAdapter,
    DreamAdapter,
    MeditationAdapter,
    FlowAdapter,
    MysticalAdapter,
    ThresholdAdapter,
    AlteredAdapter,
    # Extended (28-40)
    PhilosophyAdapter,
    FolkWisdomAdapter,
    AnimalCognitionAdapter,
    PlantIntelligenceAdapter,
    FungalIntelligenceAdapter,
    SwarmIntelligenceAdapter,
    GaiaIntelligenceAdapter,
    DevelopmentalConsciousnessAdapter,
    ContemplativeStatesAdapter,
    PsychedelicConsciousnessAdapter,
    NeurodivergentConsciousnessAdapter,
    TraumaConsciousnessAdapter,
    XenoconsciousnessAdapter,
)


# Complete registry mapping form_id -> adapter class
FORM_ADAPTER_REGISTRY: Dict[str, Type["FormAdapter"]] = {
    # Sensory Forms (01-06)
    "01-visual": VisualAdapter,
    "02-auditory": AuditoryAdapter,
    "03-somatosensory": TactileAdapter,  # Tactile handles somatosensory
    "04-olfactory": OlfactoryAdapter,
    "05-gustatory": GustatoryAdapter,
    "06-interoceptive": ProprioceptiveAdapter,  # Proprioceptive handles interoceptive

    # Cognitive Forms (07-12)
    "07-emotional": EmotionAdapter,
    "08-arousal": ArousalAdapter,
    "09-perceptual": AttentionAdapter,  # Attention handles perceptual binding
    "10-self-recognition": ExecutiveAdapter,  # Executive handles self-recognition
    "11-meta-consciousness": ExecutiveAdapter,  # Shared with executive
    "12-narrative-consciousness": MemoryLTMAdapter,  # LTM handles narrative

    # Theoretical Forms (13-17)
    "13-integrated-information": IITAdapter,
    "14-global-workspace": GlobalWorkspaceAdapter,
    "15-higher-order-thought": HOTAdapter,
    "16-predictive-coding": PredictiveAdapter,
    "17-recurrent-processing": RecurrentAdapter,

    # Specialized Forms (18-27)
    "18-primary-consciousness": PrimaryAdapter,
    "19-reflective-consciousness": ReflectiveAdapter,
    "20-social-consciousness": SocialAdapter,
    "21-artificial-consciousness": ArtificialAdapter,
    "22-dream-consciousness": DreamAdapter,
    "23-meditation-consciousness": MeditationAdapter,
    "24-flow-state": FlowAdapter,
    "25-mystical-consciousness": MysticalAdapter,
    "26-split-brain": ThresholdAdapter,
    "27-altered-state": AlteredAdapter,

    # Extended Forms (28-40)
    "28-philosophy": PhilosophyAdapter,
    "29-folk-wisdom": FolkWisdomAdapter,
    "30-animal-cognition": AnimalCognitionAdapter,
    "31-plant-intelligence": PlantIntelligenceAdapter,
    "32-fungal-intelligence": FungalIntelligenceAdapter,
    "33-swarm-intelligence": SwarmIntelligenceAdapter,
    "34-gaia-intelligence": GaiaIntelligenceAdapter,
    "35-developmental-consciousness": DevelopmentalConsciousnessAdapter,
    "36-contemplative-states": ContemplativeStatesAdapter,
    "37-psychedelic-consciousness": PsychedelicConsciousnessAdapter,
    "38-neurodivergent-consciousness": NeurodivergentConsciousnessAdapter,
    "39-trauma-consciousness": TraumaConsciousnessAdapter,
    "40-xenoconsciousness": XenoconsciousnessAdapter,
}


# Form categories for batch operations
FORM_CATEGORIES = {
    "sensory": ["01-visual", "02-auditory", "03-somatosensory", "04-olfactory",
                "05-gustatory", "06-interoceptive"],
    "cognitive": ["07-emotional", "08-arousal", "09-perceptual", "10-self-recognition",
                  "11-meta-consciousness", "12-narrative-consciousness"],
    "theoretical": ["13-integrated-information", "14-global-workspace",
                    "15-higher-order-thought", "16-predictive-coding", "17-recurrent-processing"],
    "specialized": ["18-primary-consciousness", "19-reflective-consciousness",
                    "20-social-consciousness", "21-artificial-consciousness",
                    "22-dream-consciousness", "23-meditation-consciousness",
                    "24-flow-state", "25-mystical-consciousness",
                    "26-split-brain", "27-altered-state"],
    "extended": ["28-philosophy", "29-folk-wisdom", "30-animal-cognition",
                 "31-plant-intelligence", "32-fungal-intelligence", "33-swarm-intelligence",
                 "34-gaia-intelligence", "35-developmental-consciousness",
                 "36-contemplative-states", "37-psychedelic-consciousness",
                 "38-neurodivergent-consciousness", "39-trauma-consciousness",
                 "40-xenoconsciousness"],
    "critical": ["08-arousal", "13-integrated-information", "14-global-workspace"],
}


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

class AdapterFactory:
    """
    Factory for creating and managing consciousness form adapters.

    Provides:
    - On-demand adapter instantiation
    - Adapter pooling and caching
    - Batch registration to NervousSystem
    - Lifecycle management
    """

    def __init__(self):
        """Initialize the adapter factory."""
        self._adapter_cache: Dict[str, "FormAdapter"] = {}
        self._initialization_errors: Dict[str, str] = {}

    def get_available_forms(self) -> List[str]:
        """Get list of all available form IDs."""
        return list(FORM_ADAPTER_REGISTRY.keys())

    def get_forms_by_category(self, category: str) -> List[str]:
        """Get form IDs for a specific category."""
        return FORM_CATEGORIES.get(category, [])

    def get_critical_forms(self) -> List[str]:
        """Get the critical forms that must always be loaded."""
        return FORM_CATEGORIES["critical"]

    def is_valid_form(self, form_id: str) -> bool:
        """Check if a form ID is valid."""
        return form_id in FORM_ADAPTER_REGISTRY

    def create_adapter(self, form_id: str) -> Optional["FormAdapter"]:
        """
        Create a new adapter instance for a form.

        Args:
            form_id: The form identifier

        Returns:
            New adapter instance, or None if creation fails
        """
        if form_id not in FORM_ADAPTER_REGISTRY:
            logger.error(f"Unknown form ID: {form_id}")
            return None

        adapter_class = FORM_ADAPTER_REGISTRY[form_id]

        try:
            adapter = adapter_class()
            logger.info(f"Created adapter for {form_id}: {adapter_class.__name__}")
            return adapter
        except Exception as e:
            logger.error(f"Failed to create adapter for {form_id}: {e}")
            self._initialization_errors[form_id] = str(e)
            return None

    def get_adapter(self, form_id: str, create_if_missing: bool = True) -> Optional["FormAdapter"]:
        """
        Get an adapter instance, optionally creating if not cached.

        Args:
            form_id: The form identifier
            create_if_missing: Whether to create adapter if not in cache

        Returns:
            Adapter instance, or None if unavailable
        """
        # Check cache first
        if form_id in self._adapter_cache:
            return self._adapter_cache[form_id]

        if not create_if_missing:
            return None

        # Create and cache
        adapter = self.create_adapter(form_id)
        if adapter:
            self._adapter_cache[form_id] = adapter

        return adapter

    def get_adapters_by_category(
        self,
        category: str,
        create_if_missing: bool = True
    ) -> Dict[str, "FormAdapter"]:
        """
        Get all adapters for a category.

        Args:
            category: Category name (sensory, cognitive, etc.)
            create_if_missing: Whether to create adapters if not cached

        Returns:
            Dict mapping form_id to adapter
        """
        form_ids = self.get_forms_by_category(category)
        adapters = {}

        for form_id in form_ids:
            adapter = self.get_adapter(form_id, create_if_missing)
            if adapter:
                adapters[form_id] = adapter

        return adapters

    def create_all_adapters(self) -> Dict[str, "FormAdapter"]:
        """
        Create adapters for all 40 forms.

        Returns:
            Dict mapping form_id to adapter
        """
        adapters = {}

        for form_id in FORM_ADAPTER_REGISTRY:
            adapter = self.get_adapter(form_id, create_if_missing=True)
            if adapter:
                adapters[form_id] = adapter

        logger.info(f"Created {len(adapters)}/{len(FORM_ADAPTER_REGISTRY)} adapters")

        if self._initialization_errors:
            logger.warning(f"Failed to create {len(self._initialization_errors)} adapters: "
                          f"{list(self._initialization_errors.keys())}")

        return adapters

    def register_with_nervous_system(
        self,
        nervous_system: "NervousSystem",
        form_ids: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        include_critical: bool = True
    ) -> Dict[str, bool]:
        """
        Register adapters with a NervousSystem instance.

        Args:
            nervous_system: The NervousSystem to register with
            form_ids: Specific form IDs to register (optional)
            categories: Categories to register (optional)
            include_critical: Always include critical forms

        Returns:
            Dict mapping form_id to success status
        """
        # Determine which forms to register
        forms_to_register = set()

        if include_critical:
            forms_to_register.update(self.get_critical_forms())

        if form_ids:
            forms_to_register.update(form_ids)

        if categories:
            for category in categories:
                forms_to_register.update(self.get_forms_by_category(category))

        # If nothing specified, register all
        if not form_ids and not categories and not include_critical:
            forms_to_register = set(self.get_available_forms())
        elif not form_ids and not categories:
            # Just critical forms were added
            pass

        # Register adapters
        results = {}

        for form_id in forms_to_register:
            adapter = self.get_adapter(form_id, create_if_missing=True)

            if adapter:
                try:
                    nervous_system.register_adapter(form_id, adapter)
                    results[form_id] = True
                    logger.debug(f"Registered adapter for {form_id}")
                except Exception as e:
                    logger.error(f"Failed to register adapter for {form_id}: {e}")
                    results[form_id] = False
            else:
                results[form_id] = False

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Registered {success_count}/{len(forms_to_register)} adapters with NervousSystem")

        return results

    def register_all(self, nervous_system: "NervousSystem") -> Dict[str, bool]:
        """
        Register all 40 adapters with a NervousSystem.

        Args:
            nervous_system: The NervousSystem to register with

        Returns:
            Dict mapping form_id to success status
        """
        return self.register_with_nervous_system(
            nervous_system,
            form_ids=list(FORM_ADAPTER_REGISTRY.keys()),
            include_critical=True
        )

    def register_critical(self, nervous_system: "NervousSystem") -> Dict[str, bool]:
        """
        Register only critical adapters (08, 13, 14).

        Args:
            nervous_system: The NervousSystem to register with

        Returns:
            Dict mapping form_id to success status
        """
        return self.register_with_nervous_system(
            nervous_system,
            form_ids=self.get_critical_forms(),
            include_critical=True
        )

    def destroy_adapter(self, form_id: str) -> bool:
        """
        Remove an adapter from the cache.

        Args:
            form_id: The form identifier

        Returns:
            True if adapter was removed
        """
        if form_id in self._adapter_cache:
            del self._adapter_cache[form_id]
            logger.info(f"Destroyed adapter for {form_id}")
            return True
        return False

    def clear_cache(self) -> int:
        """
        Clear all cached adapters.

        Returns:
            Number of adapters cleared
        """
        count = len(self._adapter_cache)
        self._adapter_cache.clear()
        self._initialization_errors.clear()
        logger.info(f"Cleared {count} adapters from cache")
        return count

    def get_status(self) -> Dict[str, Any]:
        """Get factory status."""
        return {
            "total_forms": len(FORM_ADAPTER_REGISTRY),
            "cached_adapters": len(self._adapter_cache),
            "cached_form_ids": list(self._adapter_cache.keys()),
            "initialization_errors": dict(self._initialization_errors),
            "categories": {
                cat: len(forms) for cat, forms in FORM_CATEGORIES.items()
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global factory instance
_default_factory: Optional[AdapterFactory] = None


def get_factory() -> AdapterFactory:
    """Get the default adapter factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = AdapterFactory()
    return _default_factory


def create_adapter(form_id: str) -> Optional["FormAdapter"]:
    """Create an adapter using the default factory."""
    return get_factory().create_adapter(form_id)


def get_adapter(form_id: str) -> Optional["FormAdapter"]:
    """Get an adapter using the default factory."""
    return get_factory().get_adapter(form_id)


def register_all_adapters(nervous_system: "NervousSystem") -> Dict[str, bool]:
    """Register all adapters with a NervousSystem using the default factory."""
    return get_factory().register_all(nervous_system)


def register_critical_adapters(nervous_system: "NervousSystem") -> Dict[str, bool]:
    """Register critical adapters with a NervousSystem using the default factory."""
    return get_factory().register_critical(nervous_system)


# =============================================================================
# ADAPTER INFO
# =============================================================================

def get_adapter_info(form_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about an adapter without instantiating it.

    Args:
        form_id: The form identifier

    Returns:
        Dict with adapter class info, or None if invalid form_id
    """
    if form_id not in FORM_ADAPTER_REGISTRY:
        return None

    adapter_class = FORM_ADAPTER_REGISTRY[form_id]

    # Determine category
    category = None
    for cat, forms in FORM_CATEGORIES.items():
        if form_id in forms:
            category = cat
            break

    return {
        "form_id": form_id,
        "adapter_class": adapter_class.__name__,
        "base_class": adapter_class.__bases__[0].__name__ if adapter_class.__bases__ else None,
        "category": category,
        "is_critical": form_id in FORM_CATEGORIES["critical"],
        "module": adapter_class.__module__,
    }


def list_all_adapters() -> List[Dict[str, Any]]:
    """Get info about all available adapters."""
    return [
        get_adapter_info(form_id)
        for form_id in FORM_ADAPTER_REGISTRY
    ]
