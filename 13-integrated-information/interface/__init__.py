"""
Form 13: Integrated Information Theory (IIT) Consciousness Interface

Implements Tononi's IIT for computing integrated information (phi).
"""

from .iit_consciousness_interface import (
    # Enums
    IntegrationLevel,
    ComplexType,
    CauseEffectStructure,
    PartitionType,
    ExperienceQuality,
    # Input dataclasses
    SystemElement,
    IITInput,
    Partition,
    # Output dataclasses
    Concept,
    ConceptualStructure,
    Complex,
    IITOutput,
    IITSystemStatus,
    # Engine
    IITComputationEngine,
    # Main interface
    IITConsciousnessInterface,
    # Convenience functions
    create_iit_interface,
    create_simple_iit_input,
)

__all__ = [
    # Enums
    "IntegrationLevel",
    "ComplexType",
    "CauseEffectStructure",
    "PartitionType",
    "ExperienceQuality",
    # Input dataclasses
    "SystemElement",
    "IITInput",
    "Partition",
    # Output dataclasses
    "Concept",
    "ConceptualStructure",
    "Complex",
    "IITOutput",
    "IITSystemStatus",
    # Engine
    "IITComputationEngine",
    # Main interface
    "IITConsciousnessInterface",
    # Convenience functions
    "create_iit_interface",
    "create_simple_iit_input",
]
