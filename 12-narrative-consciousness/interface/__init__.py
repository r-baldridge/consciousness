"""
Form 12: Narrative Consciousness Interface

Self-narrative, autobiographical memory, and temporal consciousness.
"""

from .narrative_consciousness_interface import (
    # Enums
    NarrativeMode,
    TemporalOrientation,
    MemoryType,
    NarrativeCoherence,
    IdentityTheme,
    # Input dataclasses
    EventInput,
    MemoryInput,
    NarrativeInput,
    # Output dataclasses
    NarrativeSegment,
    NarrativeStructure,
    TemporalFrame,
    IdentityCoherenceReport,
    NarrativeOutput,
    NarrativeSystemStatus,
    # Engines
    NarrativeConstructionEngine,
    TemporalConsciousnessEngine,
    AutobiographicalMemoryEngine,
    # Main interface
    NarrativeConsciousnessInterface,
    # Convenience
    create_narrative_consciousness_interface,
)

__all__ = [
    "NarrativeMode",
    "TemporalOrientation",
    "MemoryType",
    "NarrativeCoherence",
    "IdentityTheme",
    "EventInput",
    "MemoryInput",
    "NarrativeInput",
    "NarrativeSegment",
    "NarrativeStructure",
    "TemporalFrame",
    "IdentityCoherenceReport",
    "NarrativeOutput",
    "NarrativeSystemStatus",
    "NarrativeConstructionEngine",
    "TemporalConsciousnessEngine",
    "AutobiographicalMemoryEngine",
    "NarrativeConsciousnessInterface",
    "create_narrative_consciousness_interface",
]
