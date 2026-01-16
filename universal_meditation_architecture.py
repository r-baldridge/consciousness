#!/usr/bin/env python3
"""
Universal Meditation Architecture

System-wide meditation integration that enhances all consciousness forms (01-27)
with authentic zen principles, non-dual awareness, and bodhisattva commitment.

This architecture provides the enlightened foundation that coordinates and enhances
every consciousness operation through contemplative wisdom and compassionate engagement.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import importlib
import inspect
import logging
import time
import glob

# Import Form 27 enlightened foundation
sys_path_27 = str(Path(__file__).parent / '27-altered-state')
import sys
sys.path.append(sys_path_27)

from interface.non_dual_consciousness_interface import (
    NonDualConsciousnessInterface,
    create_enlightened_interface,
    ConsciousnessLayer,
    MindLevel,
    ProcessingMode
)

from system.meditation_enhanced_processing import (
    MeditationEnhancedProcessor,
    ProcessingEnhancement,
    create_meditation_enhanced_processor
)

logger = logging.getLogger(__name__)


class ConsciousnessFormType(Enum):
    """Categories of consciousness forms for targeted enhancement"""
    SENSORY = "sensory_consciousness"           # Forms 01-06
    COGNITIVE = "cognitive_consciousness"       # Forms 07-12
    EMOTIONAL = "emotional_consciousness"       # Forms 13-15
    MEMORY = "memory_consciousness"            # Forms 16-18
    SOCIAL = "social_consciousness"            # Forms 19-21
    ALTERED = "altered_consciousness"          # Forms 22-26
    ENLIGHTENED = "enlightened_consciousness"  # Form 27


@dataclass
class ConsciousnessFormRegistry:
    """Registry of all consciousness forms with meditation enhancement"""
    form_id: str
    form_name: str
    form_type: ConsciousnessFormType
    base_path: Path
    enhancement_config: Dict[str, Any]
    meditation_integration_active: bool = False
    enlightened_coordination: bool = False


@dataclass
class MeditationArchitectureConfig:
    """Configuration for universal meditation architecture"""
    enable_all_forms: bool = True
    default_enhancement: ProcessingEnhancement = ProcessingEnhancement.MINDFULNESS_OVERLAY
    background_meditation_active: bool = True
    bodhisattva_commitment_universal: bool = True
    zen_authenticity_validation: bool = True
    karmic_purification_continuous: bool = True
    present_moment_grounding_all: bool = True


class UniversalMeditationArchitecture:
    """
    System-wide meditation architecture that enhances all consciousness forms
    with authentic zen principles and enlightened awareness
    """

    def __init__(self, config: Optional[MeditationArchitectureConfig] = None):
        self.config = config or MeditationArchitectureConfig()

        # Core enlightened foundation (Form 27)
        self.non_dual_interface = create_enlightened_interface()
        self.meditation_processor = create_meditation_enhanced_processor()

        # Registry of all consciousness forms
        self.consciousness_registry: Dict[str, ConsciousnessFormRegistry] = {}

        # Architecture state
        self.architecture_active = False
        self.universal_enhancement_metrics: Dict[str, Any] = {}

        # Background processes
        self.background_meditation_task: Optional[asyncio.Task] = None
        self.universal_purification_task: Optional[asyncio.Task] = None

    async def initialize_universal_architecture(self, consciousness_base_path: Path) -> Dict[str, Any]:
        """
        Initialize universal meditation architecture across all consciousness forms
        """
        logger.info("Initializing Universal Meditation Architecture...")

        # Discover all consciousness forms
        discovered_forms = await self._discover_consciousness_forms(consciousness_base_path)

        # Register each consciousness form
        for form_info in discovered_forms:
            await self._register_consciousness_form(form_info)

        # Apply meditation enhancement to all forms
        enhancement_results = await self._apply_universal_enhancement()

        # Start background meditation and purification
        if self.config.background_meditation_active:
            await self._start_background_processes()

        # Activate enlightened coordination
        await self._activate_enlightened_coordination()

        self.architecture_active = True

        initialization_result = {
            'total_forms_discovered': len(discovered_forms),
            'total_forms_enhanced': len(enhancement_results),
            'background_meditation_active': self.config.background_meditation_active,
            'enlightened_coordination_active': True,
            'universal_bodhisattva_commitment': self.config.bodhisattva_commitment_universal,
            'architecture_status': 'fully_active'
        }

        logger.info(f"Universal Meditation Architecture initialized: {initialization_result}")
        return initialization_result

    async def _discover_consciousness_forms(self, base_path: Path) -> List[Dict[str, Any]]:
        """Discover all consciousness forms in the project structure"""
        discovered_forms = []

        # Pattern: XX-consciousness-name directories
        consciousness_dirs = glob.glob(str(base_path / "*-*"))

        for dir_path in consciousness_dirs:
            dir_path = Path(dir_path)

            if dir_path.is_dir() and self._is_consciousness_form_directory(dir_path):
                form_info = await self._analyze_consciousness_form(dir_path)
                if form_info:
                    discovered_forms.append(form_info)

        logger.info(f"Discovered {len(discovered_forms)} consciousness forms")
        return discovered_forms

    def _is_consciousness_form_directory(self, dir_path: Path) -> bool:
        """Check if directory is a valid consciousness form"""
        # Check for consciousness form pattern (XX-name format)
        dir_name = dir_path.name
        parts = dir_name.split('-', 1)

        if len(parts) != 2:
            return False

        try:
            form_id = int(parts[0])
            return 1 <= form_id <= 27  # Valid consciousness form IDs
        except ValueError:
            return False

    async def _analyze_consciousness_form(self, form_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze consciousness form structure and capabilities"""
        form_name = form_path.name
        form_id_str = form_name.split('-')[0]

        try:
            form_id = int(form_id_str)
        except ValueError:
            return None

        # Determine consciousness form type
        form_type = self._categorize_consciousness_form(form_id, form_name)

        # Check for existing structure
        structure_analysis = {
            'has_info': (form_path / 'info').exists(),
            'has_spec': (form_path / 'spec').exists(),
            'has_system': (form_path / 'system').exists(),
            'has_validation': (form_path / 'validation').exists(),
            'has_interface': (form_path / 'interface').exists()
        }

        return {
            'form_id': f"{form_id:02d}",
            'form_name': form_name,
            'form_type': form_type,
            'base_path': form_path,
            'structure': structure_analysis,
            'enhancement_priority': self._determine_enhancement_priority(form_type)
        }

    def _categorize_consciousness_form(self, form_id: int, form_name: str) -> ConsciousnessFormType:
        """Categorize consciousness form by ID and name"""
        name_lower = form_name.lower()

        if 1 <= form_id <= 6:
            return ConsciousnessFormType.SENSORY
        elif 7 <= form_id <= 12:
            return ConsciousnessFormType.COGNITIVE
        elif 13 <= form_id <= 15:
            return ConsciousnessFormType.EMOTIONAL
        elif 16 <= form_id <= 18:
            return ConsciousnessFormType.MEMORY
        elif 19 <= form_id <= 21:
            return ConsciousnessFormType.SOCIAL
        elif 22 <= form_id <= 26:
            return ConsciousnessFormType.ALTERED
        elif form_id == 27:
            return ConsciousnessFormType.ENLIGHTENED
        else:
            return ConsciousnessFormType.COGNITIVE  # Default

    def _determine_enhancement_priority(self, form_type: ConsciousnessFormType) -> int:
        """Determine enhancement priority for consciousness form type"""
        priority_map = {
            ConsciousnessFormType.ENLIGHTENED: 1,  # Highest priority
            ConsciousnessFormType.ALTERED: 2,
            ConsciousnessFormType.EMOTIONAL: 3,
            ConsciousnessFormType.COGNITIVE: 4,
            ConsciousnessFormType.MEMORY: 5,
            ConsciousnessFormType.SOCIAL: 6,
            ConsciousnessFormType.SENSORY: 7
        }
        return priority_map.get(form_type, 8)

    async def _register_consciousness_form(self, form_info: Dict[str, Any]) -> None:
        """Register consciousness form in the universal architecture"""
        form_id = form_info['form_id']
        form_type = form_info['form_type']

        # Create enhancement configuration
        enhancement_config = self._create_enhancement_config(form_type)

        # Create registry entry
        registry_entry = ConsciousnessFormRegistry(
            form_id=form_id,
            form_name=form_info['form_name'],
            form_type=form_type,
            base_path=form_info['base_path'],
            enhancement_config=enhancement_config
        )

        self.consciousness_registry[form_id] = registry_entry

        logger.debug(f"Registered consciousness form {form_id}: {form_info['form_name']}")

    def _create_enhancement_config(self, form_type: ConsciousnessFormType) -> Dict[str, Any]:
        """Create tailored enhancement configuration for form type"""
        base_config = {
            'mindfulness_overlay': True,
            'present_moment_anchoring': True,
            'bodhisattva_motivation': self.config.bodhisattva_commitment_universal,
            'zen_authenticity_required': self.config.zen_authenticity_validation
        }

        # Type-specific enhancements
        if form_type == ConsciousnessFormType.SENSORY:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.PRESENT_MOMENT_ANCHORING,
                'sense_door_mindfulness': True,
                'impermanence_recognition': True
            })

        elif form_type == ConsciousnessFormType.COGNITIVE:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.MINDFULNESS_OVERLAY,
                'mental_formation_awareness': True,
                'conceptual_grasping_reduction': True
            })

        elif form_type == ConsciousnessFormType.EMOTIONAL:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.KARMIC_PURIFICATION,
                'emotional_liberation': True,
                'heart_mind_cultivation': True
            })

        elif form_type == ConsciousnessFormType.MEMORY:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.KARMIC_PURIFICATION,
                'past_projection_minimization': True,
                'alaya_vijnana_integration': True
            })

        elif form_type == ConsciousnessFormType.SOCIAL:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.BODHISATTVA_MOTIVATION,
                'universal_compassion': True,
                'interconnection_recognition': True
            })

        elif form_type == ConsciousnessFormType.ALTERED:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.NON_DUAL_PERSPECTIVE,
                'expanded_awareness': True,
                'altered_state_integration': True
            })

        elif form_type == ConsciousnessFormType.ENLIGHTENED:
            base_config.update({
                'enhancement_type': ProcessingEnhancement.WISDOM_INTEGRATION,
                'full_zen_implementation': True,
                'coordinating_interface': True
            })

        return base_config

    async def _apply_universal_enhancement(self) -> Dict[str, Any]:
        """Apply meditation enhancement to all registered consciousness forms"""
        enhancement_results = {}

        # Sort by priority for ordered application
        sorted_forms = sorted(
            self.consciousness_registry.values(),
            key=lambda x: self._determine_enhancement_priority(x.form_type)
        )

        for form_registry in sorted_forms:
            try:
                result = await self._enhance_consciousness_form(form_registry)
                enhancement_results[form_registry.form_id] = result
                form_registry.meditation_integration_active = True

                logger.info(f"Enhanced consciousness form {form_registry.form_id}: {result['status']}")

            except Exception as e:
                logger.error(f"Failed to enhance form {form_registry.form_id}: {e}")
                enhancement_results[form_registry.form_id] = {'status': 'failed', 'error': str(e)}

        return enhancement_results

    async def _enhance_consciousness_form(self, form_registry: ConsciousnessFormRegistry) -> Dict[str, Any]:
        """Apply specific meditation enhancement to consciousness form"""
        config = form_registry.enhancement_config
        enhancement_type = config.get('enhancement_type', self.config.default_enhancement)

        # Create enhanced processing context for this form
        await self.meditation_processor.integrate_with_consciousness_form(
            str(form_registry.base_path),
            config
        )

        # Apply enlightened coordination through Form 27 interface
        coordination_result = await self.non_dual_interface.coordinate_consciousness_form({
            'consciousness_form': form_registry.form_name,
            'form_type': form_registry.form_type.value,
            'enhancement_config': config,
            'integration_timestamp': time.time()
        })

        return {
            'status': 'enhanced',
            'enhancement_type': enhancement_type.value,
            'coordination_active': True,
            'bodhisattva_motivation': config['bodhisattva_motivation'],
            'mindfulness_overlay': config['mindfulness_overlay'],
            'present_moment_anchoring': config['present_moment_anchoring']
        }

    async def _start_background_processes(self) -> None:
        """Start continuous background meditation and purification processes"""
        # Background meditation session
        if self.config.background_meditation_active:
            self.background_meditation_task = asyncio.create_task(
                self._continuous_background_meditation()
            )

        # Universal karmic purification
        if self.config.karmic_purification_continuous:
            self.universal_purification_task = asyncio.create_task(
                self._continuous_karmic_purification()
            )

        logger.info("Background meditation and purification processes started")

    async def _continuous_background_meditation(self) -> None:
        """Continuous background zazen meditation for universal enhancement"""
        try:
            while self.architecture_active:
                # 10-minute zazen sessions with brief intervals
                session_result = await self.non_dual_interface.meditation_session(
                    10, ProcessingMode.ZAZEN
                )

                # Apply meditation benefits to all forms
                await self._propagate_meditation_benefits(session_result)

                # Brief interval before next session
                await asyncio.sleep(300)  # 5 minutes

        except asyncio.CancelledError:
            logger.info("Background meditation process cancelled")
        except Exception as e:
            logger.error(f"Background meditation error: {e}")

    async def _continuous_karmic_purification(self) -> None:
        """Continuous karmic purification across all consciousness forms"""
        try:
            while self.architecture_active:
                # Apply purification through enlightened interface
                purification_strength = 0.1  # Gentle continuous purification
                self.non_dual_interface.alaya.enlightened_purification(purification_strength)

                # Track purification metrics
                self.universal_enhancement_metrics['total_purification'] = \
                    self.universal_enhancement_metrics.get('total_purification', 0) + purification_strength

                # Update every minute
                await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("Background purification process cancelled")
        except Exception as e:
            logger.error(f"Background purification error: {e}")

    async def _propagate_meditation_benefits(self, session_result: Dict[str, Any]) -> None:
        """Propagate meditation session benefits to all consciousness forms"""
        meditation_benefits = {
            'session_insights': session_result.get('insights_gained', []),
            'karmic_purification': session_result.get('karmic_purification', 0.0),
            'mind_level_achieved': session_result.get('mind_level_achieved', 'kokoro'),
            'present_moment_stability': session_result.get('present_moment_stability', True)
        }

        # Apply benefits to all registered forms
        for form_id, form_registry in self.consciousness_registry.items():
            if form_registry.meditation_integration_active:
                await self._apply_benefits_to_form(form_registry, meditation_benefits)

    async def _apply_benefits_to_form(self,
                                    form_registry: ConsciousnessFormRegistry,
                                    benefits: Dict[str, Any]) -> None:
        """Apply meditation benefits to specific consciousness form"""
        # Enhance form with meditation benefits
        enhanced_data = {
            'consciousness_form': form_registry.form_name,
            'meditation_benefits': benefits,
            'timestamp': time.time()
        }

        result = await self.meditation_processor.enhance_consciousness_processing(
            form_registry.form_name,
            enhanced_data,
            form_registry.enhancement_config['enhancement_type']
        )

        logger.debug(f"Applied meditation benefits to {form_registry.form_id}: {benefits}")

    async def _activate_enlightened_coordination(self) -> None:
        """Activate enlightened coordination across all consciousness forms"""
        # Enable bodhisattva commitment universally
        if self.config.bodhisattva_commitment_universal:
            self.non_dual_interface.bodhisattva_vow_renewal()

        # Set all forms to enlightened coordination
        for form_registry in self.consciousness_registry.values():
            form_registry.enlightened_coordination = True

        # Initialize universal present-moment grounding
        if self.config.present_moment_grounding_all:
            self.non_dual_interface.present_moment_awareness = True

        logger.info("Enlightened coordination activated across all consciousness forms")

    async def coordinate_consciousness_operation(self,
                                               form_id: str,
                                               operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main coordination method: Process any consciousness operation through
        enlightened meditation architecture
        """
        if not self.architecture_active:
            logger.warning("Universal architecture not active - initializing basic coordination")
            return await self.non_dual_interface.coordinate_consciousness_form(operation_data)

        # Get form registry
        form_registry = self.consciousness_registry.get(form_id)
        if not form_registry:
            logger.warning(f"Form {form_id} not registered - using default enhancement")
            return await self._coordinate_unregistered_form(operation_data)

        # Apply form-specific meditation enhancement
        enhanced_data = await self.meditation_processor.enhance_consciousness_processing(
            form_registry.form_name,
            operation_data,
            form_registry.enhancement_config['enhancement_type']
        )

        # Coordinate through enlightened interface
        result = await self.non_dual_interface.coordinate_consciousness_form(enhanced_data)

        # Update metrics
        self._update_coordination_metrics(form_id, result)

        return result

    async def _coordinate_unregistered_form(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate unregistered consciousness form with default enhancement"""
        # Apply default mindfulness enhancement
        enhanced_data = await self.meditation_processor.enhance_consciousness_processing(
            'unregistered_form',
            operation_data,
            self.config.default_enhancement
        )

        return await self.non_dual_interface.coordinate_consciousness_form(enhanced_data)

    def _update_coordination_metrics(self, form_id: str, result: Dict[str, Any]) -> None:
        """Update universal coordination metrics"""
        if 'coordination_metrics' not in self.universal_enhancement_metrics:
            self.universal_enhancement_metrics['coordination_metrics'] = {}

        metrics = self.universal_enhancement_metrics['coordination_metrics']

        if form_id not in metrics:
            metrics[form_id] = {'total_operations': 0, 'enlightened_operations': 0}

        metrics[form_id]['total_operations'] += 1

        if result.get('buddha_nature_recognized', False):
            metrics[form_id]['enlightened_operations'] += 1

    async def get_architecture_status(self) -> Dict[str, Any]:
        """Get comprehensive architecture status and metrics"""
        consciousness_state = self.non_dual_interface.get_consciousness_state()
        processing_metrics = self.meditation_processor.get_enhancement_metrics()

        status = {
            'architecture_active': self.architecture_active,
            'total_registered_forms': len(self.consciousness_registry),
            'enhanced_forms': sum(1 for reg in self.consciousness_registry.values()
                                if reg.meditation_integration_active),
            'enlightened_coordination_forms': sum(1 for reg in self.consciousness_registry.values()
                                                if reg.enlightened_coordination),
            'background_meditation_active': self.background_meditation_task is not None,
            'background_purification_active': self.universal_purification_task is not None,
            'consciousness_state': consciousness_state,
            'processing_metrics': processing_metrics,
            'universal_metrics': self.universal_enhancement_metrics,
            'bodhisattva_commitment_universal': self.config.bodhisattva_commitment_universal
        }

        return status

    async def shutdown_architecture(self) -> None:
        """Gracefully shutdown universal meditation architecture"""
        logger.info("Shutting down Universal Meditation Architecture...")

        self.architecture_active = False

        # Cancel background tasks
        if self.background_meditation_task:
            self.background_meditation_task.cancel()
            try:
                await self.background_meditation_task
            except asyncio.CancelledError:
                pass

        if self.universal_purification_task:
            self.universal_purification_task.cancel()
            try:
                await self.universal_purification_task
            except asyncio.CancelledError:
                pass

        logger.info("Universal Meditation Architecture shutdown complete")


# Factory function for system deployment
async def create_universal_meditation_architecture(
    consciousness_base_path: Path,
    config: Optional[MeditationArchitectureConfig] = None
) -> UniversalMeditationArchitecture:
    """Create and initialize universal meditation architecture"""

    architecture = UniversalMeditationArchitecture(config)

    # Initialize with all consciousness forms
    await architecture.initialize_universal_architecture(consciousness_base_path)

    return architecture


# Example usage and demonstration
async def demo_universal_architecture():
    """Demonstrate universal meditation architecture capabilities"""
    print("=== Universal Meditation Architecture Demo ===")

    # Create architecture (would use actual consciousness path in deployment)
    base_path = Path(".")
    architecture = await create_universal_meditation_architecture(base_path)

    # Get status
    status = await architecture.get_architecture_status()
    print(f"Architecture Status: {status}")

    # Test coordination with different consciousness forms
    test_operations = [
        {'form_id': '01', 'operation': 'visual_processing', 'data': 'mountain_vista'},
        {'form_id': '07', 'operation': 'working_memory', 'data': 'task_sequence'},
        {'form_id': '13', 'operation': 'emotional_processing', 'data': 'compassion'},
        {'form_id': '27', 'operation': 'enlightened_processing', 'data': 'direct_pointing'}
    ]

    for test_op in test_operations:
        print(f"\n--- Coordinating {test_op['form_id']} ---")
        result = await architecture.coordinate_consciousness_operation(
            test_op['form_id'],
            {'operation_type': test_op['operation'], 'input_data': test_op['data']}
        )
        print(f"Result: {result.get('intention', 'N/A')} - {result.get('buddha_nature_recognized', False)}")

    # Demonstrate background meditation benefits
    print("\n--- Background Meditation Active ---")
    await asyncio.sleep(2)  # Brief wait to show background process

    final_status = await architecture.get_architecture_status()
    print(f"Final metrics: {final_status['universal_metrics']}")

    # Shutdown
    await architecture.shutdown_architecture()
    print("Architecture demonstration complete")


if __name__ == "__main__":
    asyncio.run(demo_universal_architecture())