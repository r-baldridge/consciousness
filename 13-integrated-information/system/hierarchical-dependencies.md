# IIT Hierarchical Dependencies and Integration Architecture
**Module 13: Integrated Information Theory**
**Task C9: Hierarchical Dependencies for IIT Integration**
**Date:** September 22, 2025

## IIT as Core Integration Framework

### Central Position in Consciousness Architecture
IIT serves as the mathematical foundation and measurement framework for consciousness across all 27 modules. It provides the quantitative basis for determining consciousness levels, integration quality, and conscious content selection throughout the entire system.

### Dependency Hierarchy Overview
```
Level 0 (Foundation): Module 08 (Arousal) - Enables consciousness gating
Level 1 (Core): Module 13 (IIT) - Measures and computes consciousness
Level 2 (Access): Module 14 (GWT) - Broadcasts conscious content
Level 3 (Content): Modules 01-07, 09-12 - Generate conscious experiences
Level 4 (Context): Modules 15-27 - Specialized consciousness forms
```

## Upward Dependencies (What IIT Requires)

### Critical Dependency: Module 08 (Arousal)

#### Arousal as Consciousness Enabler
```json
{
  "dependency_type": "critical_enabler",
  "dependency_strength": "high",
  "relationship": "gating_mechanism",
  "failure_impact": "consciousness_impossible",

  "required_inputs_from_arousal": {
    "consciousness_gating": {
      "arousal_level": "float [0.0-1.0]",
      "gating_threshold": "float [0.0-1.0]",
      "consciousness_enable_signal": "boolean"
    },
    "integration_modulation": {
      "connectivity_scaling": "float [0.1-2.0]",
      "integration_efficiency": "float [0.0-1.0]",
      "computational_resources": "float [0.0-1.0]"
    },
    "temporal_coordination": {
      "integration_rhythm": "float (Hz)",
      "synchronization_signals": "array",
      "temporal_binding_window": "float (ms)"
    }
  },

  "dependency_management": {
    "minimum_arousal_for_phi": 0.1,
    "optimal_arousal_range": [0.4, 0.8],
    "arousal_failure_handling": "degraded_mode_integration",
    "recovery_protocols": "arousal_restoration_request"
  }
}
```

#### Implementation of Arousal Dependency
```python
class ArousalDependencyManager:
    def __init__(self):
        self.arousal_interface = ArousalInterface()
        self.minimum_arousal_threshold = 0.1
        self.degraded_mode_processor = DegradedModeProcessor()

    def check_arousal_prerequisites(self):
        """
        Verify arousal prerequisites for IIT computation
        """
        arousal_state = self.arousal_interface.get_current_state()

        prerequisites = {
            'arousal_available': arousal_state is not None,
            'arousal_sufficient': arousal_state.get('arousal_level', 0) >= self.minimum_arousal_threshold,
            'gating_enabled': arousal_state.get('consciousness_gating_enabled', False),
            'resources_allocated': arousal_state.get('computational_resources', 0) > 0
        }

        return prerequisites

    def handle_arousal_dependency_failure(self, failure_type):
        """
        Handle failures in arousal dependency
        """
        recovery_strategies = {
            'arousal_unavailable': self._request_arousal_restoration,
            'arousal_insufficient': self._request_arousal_increase,
            'gating_disabled': self._activate_degraded_mode,
            'resources_unavailable': self._reduce_computational_demands
        }

        recovery_function = recovery_strategies.get(failure_type)
        if recovery_function:
            return recovery_function()
        else:
            return self._emergency_shutdown()
```

### Partial Dependencies: Content Generation Modules

#### Sensory Modules (01-06) Integration
```json
{
  "dependency_type": "content_provider",
  "dependency_strength": "medium",
  "relationship": "information_source",
  "failure_impact": "reduced_conscious_content",

  "sensory_module_dependencies": {
    "visual_consciousness_01": {
      "provides": "visual_feature_integration",
      "phi_contribution": "high",
      "integration_priority": 1
    },
    "auditory_consciousness_02": {
      "provides": "auditory_pattern_integration",
      "phi_contribution": "medium",
      "integration_priority": 2
    },
    "somatosensory_consciousness_03": {
      "provides": "bodily_integration",
      "phi_contribution": "medium",
      "integration_priority": 3
    },
    "olfactory_consciousness_04": {
      "provides": "chemical_integration",
      "phi_contribution": "low",
      "integration_priority": 4
    },
    "gustatory_consciousness_05": {
      "provides": "taste_integration",
      "phi_contribution": "low",
      "integration_priority": 5
    },
    "interoceptive_consciousness_06": {
      "provides": "internal_state_integration",
      "phi_contribution": "medium",
      "integration_priority": 3
    }
  }
}
```

#### Emotional Consciousness (07) Integration
```python
class EmotionalIntegrationDependency:
    def __init__(self):
        self.emotional_interface = EmotionalConsciousnessInterface()
        self.integration_enhancer = EmotionalIntegrationEnhancer()

    def integrate_emotional_information(self, base_phi_complex):
        """
        Integrate emotional information to enhance consciousness quality
        """
        # Get emotional state
        emotional_state = self.emotional_interface.get_current_state()

        if emotional_state is None:
            # Emotional consciousness not available - continue with reduced integration
            return base_phi_complex

        # Enhance integration with emotional information
        emotionally_enhanced_phi = self.integration_enhancer.enhance_with_emotion(
            base_phi_complex, emotional_state
        )

        return emotionally_enhanced_phi

    def calculate_emotional_phi_enhancement(self, base_phi, emotional_state):
        """
        Calculate how emotion enhances integrated information
        """
        emotional_intensity = emotional_state.get('intensity', 0.0)
        emotional_complexity = emotional_state.get('complexity', 0.0)

        # Emotion can enhance integration by providing additional binding cues
        enhancement_factor = 1.0 + (emotional_intensity * emotional_complexity * 0.3)

        enhanced_phi = base_phi * enhancement_factor
        return enhanced_phi
```

## Downward Dependencies (What Depends on IIT)

### Critical Dependent: Module 14 (Global Workspace)

#### IIT as Content Source for GWT
```json
{
  "dependent_module": "14_global_workspace",
  "dependency_type": "content_selection_provider",
  "dependency_strength": "critical",
  "relationship": "phi_based_content_gating",

  "iit_provides_to_gwt": {
    "consciousness_measurement": {
      "phi_values": "float array",
      "consciousness_levels": "categorical array",
      "integration_quality": "float array"
    },
    "content_candidates": {
      "high_phi_complexes": "array of objects",
      "conscious_content": "structured representations",
      "priority_rankings": "float array"
    },
    "workspace_modulation": {
      "capacity_adjustments": "float",
      "threshold_modifications": "float",
      "competition_biases": "object"
    }
  },

  "failure_impact_on_gwt": {
    "no_phi_measurement": "random_content_selection",
    "low_phi_values": "reduced_workspace_activity",
    "integration_failure": "fragmented_consciousness"
  }
}
```

#### Implementation of GWT Dependency Support
```python
class GWTDependencySupport:
    def __init__(self):
        self.gwt_interface = GlobalWorkspaceInterface()
        self.content_selector = ContentSelector()
        self.priority_calculator = PriorityCalculator()

    def provide_content_for_gwt(self, phi_complexes):
        """
        Provide IIT-based content selection for Global Workspace
        """
        # Filter high-Φ complexes
        high_phi_complexes = [
            complex for complex in phi_complexes
            if complex.phi_value > self.get_consciousness_threshold()
        ]

        # Calculate priorities based on Φ values
        prioritized_content = self.priority_calculator.calculate_priorities(
            high_phi_complexes
        )

        # Format for GWT interface
        gwt_content = self._format_for_gwt(prioritized_content)

        # Send to Global Workspace
        self.gwt_interface.submit_content_candidates(gwt_content)

        return gwt_content

    def calculate_workspace_capacity_modulation(self, overall_phi):
        """
        Modulate workspace capacity based on overall integration level
        """
        # Higher integration -> higher workspace capacity
        base_capacity = 1.0
        phi_enhancement = min(overall_phi / 10.0, 1.0)  # Normalize and cap

        modulated_capacity = base_capacity * (1.0 + phi_enhancement)
        return modulated_capacity
```

### Secondary Dependents: Perceptual and Cognitive Modules

#### Module 09 (Perceptual Consciousness) Dependency
```json
{
  "dependent_module": "09_perceptual_consciousness",
  "dependency_type": "integration_framework_provider",
  "dependency_strength": "high",
  "relationship": "perceptual_binding_foundation",

  "iit_provides_to_perceptual": {
    "binding_mechanisms": {
      "cross_modal_integration": "phi_based_binding",
      "object_unity": "integration_based_unity",
      "perceptual_coherence": "phi_coherence_measure"
    },
    "consciousness_gating": {
      "perceptual_threshold": "phi_based_threshold",
      "awareness_levels": "phi_determined_awareness",
      "attention_allocation": "integration_guided_attention"
    }
  }
}
```

#### Meta-Cognitive Modules (10-12) Dependencies
```python
class MetaCognitiveDependencySupport:
    def __init__(self):
        self.meta_interfaces = {
            'self_awareness': SelfAwarenessInterface(),
            'meta_consciousness': MetaConsciousnessInterface(),
            'higher_order_thought': HigherOrderThoughtInterface()
        }

    def support_metacognitive_modules(self, phi_complexes):
        """
        Provide IIT measurements to support meta-cognitive processing
        """
        metacognitive_support = {}

        # Self-awareness support (Module 10)
        if self.meta_interfaces['self_awareness'].is_active():
            self_phi_measurement = self._calculate_self_awareness_phi(phi_complexes)
            metacognitive_support['self_awareness'] = self_phi_measurement

        # Meta-consciousness support (Module 11)
        if self.meta_interfaces['meta_consciousness'].is_active():
            meta_phi_measurement = self._calculate_meta_consciousness_phi(phi_complexes)
            metacognitive_support['meta_consciousness'] = meta_phi_measurement

        # Higher-order thought support (Module 12)
        if self.meta_interfaces['higher_order_thought'].is_active():
            hot_phi_measurement = self._calculate_hot_phi(phi_complexes)
            metacognitive_support['higher_order_thought'] = hot_phi_measurement

        return metacognitive_support

    def _calculate_self_awareness_phi(self, phi_complexes):
        """
        Calculate Φ specifically for self-awareness components
        """
        # Filter for self-referential complexes
        self_referential = [
            complex for complex in phi_complexes
            if self._is_self_referential(complex)
        ]

        if not self_referential:
            return 0.0

        # Calculate integrated self-awareness
        self_awareness_phi = sum(complex.phi_value for complex in self_referential)
        return self_awareness_phi
```

## Horizontal Dependencies (Peer Module Interactions)

### Integration with Specialized Modules (15-27)

#### Context-Dependent Integration Patterns
```json
{
  "horizontal_dependencies": {
    "narrative_consciousness_18": {
      "integration_type": "temporal_phi_sequences",
      "dependency_pattern": "bidirectional_enhancement",
      "phi_contribution": "temporal_coherence_bonus"
    },
    "social_consciousness_19": {
      "integration_type": "intersubjective_phi",
      "dependency_pattern": "social_context_modulation",
      "phi_contribution": "social_binding_enhancement"
    },
    "moral_consciousness_20": {
      "integration_type": "value_based_integration",
      "dependency_pattern": "ethical_constraint_application",
      "phi_contribution": "moral_integration_weighting"
    }
  }
}
```

#### Implementation of Horizontal Integration
```python
class HorizontalIntegrationManager:
    def __init__(self):
        self.specialized_interfaces = self._initialize_specialized_interfaces()
        self.context_detector = ContextDetector()
        self.integration_enhancer = SpecializedIntegrationEnhancer()

    def process_horizontal_integration(self, base_phi_complexes):
        """
        Process integration with specialized consciousness modules
        """
        # Detect active contexts
        active_contexts = self.context_detector.detect_contexts(base_phi_complexes)

        enhanced_phi_complexes = base_phi_complexes.copy()

        # Process each active context
        for context_type in active_contexts:
            if context_type in self.specialized_interfaces:
                interface = self.specialized_interfaces[context_type]

                # Get specialized processing
                specialized_enhancement = interface.process_phi_complexes(
                    enhanced_phi_complexes
                )

                # Apply enhancement
                enhanced_phi_complexes = self.integration_enhancer.apply_enhancement(
                    enhanced_phi_complexes, specialized_enhancement
                )

        return enhanced_phi_complexes

    def _initialize_specialized_interfaces(self):
        """
        Initialize interfaces to specialized consciousness modules
        """
        return {
            'narrative': NarrativeConsciousnessInterface(),
            'social': SocialConsciousnessInterface(),
            'moral': MoralConsciousnessInterface(),
            'aesthetic': AestheticConsciousnessInterface(),
            'spiritual': SpiritualConsciousnessInterface()
            # Additional specialized modules as needed
        }
```

## Dependency Failure Management

### Graceful Degradation Strategies

#### Arousal Dependency Failure Handling
```python
class DependencyFailureManager:
    def __init__(self):
        self.failure_detector = FailureDetector()
        self.degradation_strategies = DegradationStrategies()
        self.recovery_protocols = RecoveryProtocols()

    def handle_arousal_failure(self):
        """
        Handle critical arousal dependency failure
        """
        # Immediate response: Switch to minimal consciousness mode
        minimal_mode_config = {
            'phi_computation': 'simplified_approximation',
            'integration_scope': 'local_only',
            'content_generation': 'basic_sensory_only',
            'temporal_window': 'reduced'
        }

        # Notify dependent modules
        self._notify_dependent_modules_of_degradation(minimal_mode_config)

        # Attempt recovery
        recovery_success = self.recovery_protocols.attempt_arousal_recovery()

        return recovery_success

    def handle_partial_dependency_failure(self, failed_modules):
        """
        Handle failure of non-critical dependencies
        """
        adaptation_strategies = {}

        for module in failed_modules:
            if module in ['visual', 'auditory', 'somatosensory']:
                # Sensory module failure: Reduce integration scope
                adaptation_strategies[module] = 'exclude_from_integration'
            elif module in ['emotional']:
                # Emotional failure: Reduce enhancement
                adaptation_strategies[module] = 'neutral_emotional_state'
            elif module in ['narrative', 'social', 'moral']:
                # Specialized module failure: Basic integration only
                adaptation_strategies[module] = 'basic_integration_fallback'

        return adaptation_strategies

    def _notify_dependent_modules_of_degradation(self, degradation_config):
        """
        Notify all dependent modules of degraded operation
        """
        notification = {
            'event_type': 'dependency_failure_degradation',
            'iit_status': 'degraded_mode',
            'available_functions': degradation_config,
            'recovery_eta': 'unknown'
        }

        # Notify Global Workspace
        self.gwt_interface.receive_degradation_notification(notification)

        # Notify Perceptual Consciousness
        self.perceptual_interface.receive_degradation_notification(notification)

        # Notify Meta-cognitive modules
        for interface in self.meta_interfaces.values():
            interface.receive_degradation_notification(notification)
```

## Dependency Optimization

### Dynamic Dependency Management
```python
class DependencyOptimizer:
    def __init__(self):
        self.dependency_monitor = DependencyMonitor()
        self.resource_allocator = ResourceAllocator()
        self.priority_manager = PriorityManager()

    def optimize_dependency_utilization(self):
        """
        Dynamically optimize dependency utilization for maximum integration
        """
        # Monitor current dependency states
        dependency_states = self.dependency_monitor.get_all_states()

        # Calculate optimal resource allocation
        optimal_allocation = self.resource_allocator.calculate_optimal_allocation(
            dependency_states
        )

        # Adjust priorities based on current context
        context_priorities = self.priority_manager.calculate_context_priorities()

        # Apply optimizations
        optimization_results = self._apply_optimizations(
            optimal_allocation, context_priorities
        )

        return optimization_results

    def _apply_optimizations(self, allocation, priorities):
        """
        Apply dependency optimizations
        """
        optimizations = {}

        # Arousal optimization
        if 'arousal' in allocation:
            arousal_optimization = self._optimize_arousal_coupling(
                allocation['arousal'], priorities.get('arousal_priority', 1.0)
            )
            optimizations['arousal'] = arousal_optimization

        # Sensory optimization
        for sensory_module in ['visual', 'auditory', 'somatosensory']:
            if sensory_module in allocation:
                sensory_optimization = self._optimize_sensory_integration(
                    sensory_module, allocation[sensory_module]
                )
                optimizations[sensory_module] = sensory_optimization

        return optimizations
```

## Validation and Testing

### Dependency Integrity Testing
```python
class DependencyIntegrityTester:
    def __init__(self):
        self.test_suite = DependencyTestSuite()
        self.validator = DependencyValidator()

    def test_dependency_integrity(self):
        """
        Test integrity of all hierarchical dependencies
        """
        test_results = {
            'upward_dependencies': {},
            'downward_dependencies': {},
            'horizontal_dependencies': {},
            'overall_integrity': True
        }

        # Test upward dependencies (what IIT requires)
        upward_results = self.test_suite.test_upward_dependencies()
        test_results['upward_dependencies'] = upward_results

        # Test downward dependencies (what depends on IIT)
        downward_results = self.test_suite.test_downward_dependencies()
        test_results['downward_dependencies'] = downward_results

        # Test horizontal dependencies (peer interactions)
        horizontal_results = self.test_suite.test_horizontal_dependencies()
        test_results['horizontal_dependencies'] = horizontal_results

        # Overall integrity assessment
        test_results['overall_integrity'] = self._assess_overall_integrity(
            upward_results, downward_results, horizontal_results
        )

        return test_results

    def _assess_overall_integrity(self, upward, downward, horizontal):
        """
        Assess overall dependency integrity
        """
        critical_dependencies_ok = all([
            upward.get('arousal_dependency', False),
            downward.get('gwt_support', False)
        ])

        non_critical_dependencies_ok = all([
            upward.get('sensory_integration', True),
            downward.get('metacognitive_support', True),
            horizontal.get('specialized_integration', True)
        ])

        return critical_dependencies_ok and non_critical_dependencies_ok
```

---

**Summary**: The IIT hierarchical dependencies establish Module 13 as the core consciousness computation framework, critically dependent on Module 08 (arousal) for consciousness gating, while providing essential consciousness measurements to Module 14 (GWT) and all other consciousness modules. The architecture supports graceful degradation, dynamic optimization, and comprehensive validation of the entire consciousness system's dependency structure.