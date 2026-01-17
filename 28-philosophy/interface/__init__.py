"""
Form 28: Philosophical Consciousness Interface

This module provides the core interface for philosophical consciousness,
integrating comprehensive knowledge of Western and Eastern philosophical
traditions with agentic research capabilities and cross-form integration.
"""

from .philosophical_consciousness_interface import (
    PhilosophicalConsciousnessInterface,
    PhilosophicalTradition,
    PhilosophicalDomain,
    PhilosophicalConcept,
    PhilosophicalFigure,
    PhilosophicalText,
    PhilosophicalArgument,
    PhilosophicalMaturityState,
    CrossTraditionSynthesis,
    ArgumentType,
    MaturityLevel,
)

from .philosophical_reasoning_engine import (
    PhilosophicalReasoningEngine,
    ReasoningMode,
    ArgumentAnalysis,
    DialecticalResult,
)

from .research_agent_coordinator import (
    PhilosophicalResearchAgentCoordinator,
    ResearchSource,
    ResearchTask,
    ResearchResult,
)

__all__ = [
    # Main interface
    "PhilosophicalConsciousnessInterface",
    # Enums
    "PhilosophicalTradition",
    "PhilosophicalDomain",
    "ArgumentType",
    "MaturityLevel",
    "ReasoningMode",
    "ResearchSource",
    # Data classes
    "PhilosophicalConcept",
    "PhilosophicalFigure",
    "PhilosophicalText",
    "PhilosophicalArgument",
    "PhilosophicalMaturityState",
    "CrossTraditionSynthesis",
    "ArgumentAnalysis",
    "DialecticalResult",
    "ResearchTask",
    "ResearchResult",
    # Components
    "PhilosophicalReasoningEngine",
    "PhilosophicalResearchAgentCoordinator",
]
