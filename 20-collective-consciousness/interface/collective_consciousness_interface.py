#!/usr/bin/env python3
"""
Collective Consciousness Interface

Form 20: Collective consciousness as described by Emile Durkheim and
subsequent social theorists. Collective consciousness refers to the shared
beliefs, ideas, attitudes, and knowledge that are common to a social group
or society and function as a unifying force.

Core concepts modeled:
- Shared representations: beliefs and norms held in common
- Social cohesion: the binding force of collective sentiment
- Group mind states: emergent mental properties of collectives
- Meme propagation: how ideas spread through social networks
- Emergence: collective properties not present in individuals

This form connects to Form 19 (individual reflection) by modeling how
individual minds aggregate into shared consciousness, and to Form 21
(artificial consciousness) by addressing distributed awareness.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class CollectiveType(Enum):
    """
    Types of collectives that can manifest shared consciousness.

    Each type has different dynamics of belief formation and propagation.
    """
    CROWD = "crowd"                # Temporary gathering, emotional contagion
    COMMUNITY = "community"        # Sustained local group, shared norms
    INSTITUTION = "institution"    # Formal organization, codified beliefs
    CULTURE = "culture"            # Broad shared worldview across population
    DIGITAL = "digital"            # Online collective, viral propagation
    MOVEMENT = "movement"          # Purpose-driven collective, ideology
    SPECIES = "species"            # Species-wide shared cognitive patterns


class SocialCohesion(Enum):
    """
    Level of social cohesion within a collective.

    Following Durkheim's distinction between mechanical and organic
    solidarity, cohesion can range from fragmented to tightly integrated.
    """
    FRAGMENTED = "fragmented"          # Minimal shared beliefs, anomie
    LOOSELY_CONNECTED = "loosely_connected"  # Some shared values, weak ties
    MODERATE = "moderate"              # Shared core values, mixed ties
    COHESIVE = "cohesive"              # Strong shared identity, dense ties
    TIGHTLY_INTEGRATED = "tightly_integrated"  # Near-unanimous beliefs, rigid norms


class GroupMindState(Enum):
    """
    Emergent mental states of a collective.
    """
    DORMANT = "dormant"            # No active collective processing
    DIFFUSE = "diffuse"            # Weakly shared attention
    FOCUSED = "focused"            # Shared attention on common object
    POLARIZED = "polarized"        # Split into opposing factions
    MOBILIZED = "mobilized"        # Coordinated collective action
    EUPHORIC = "euphoric"          # Collective effervescence (Durkheim)
    PANIC = "panic"                # Collective fear response


class BeliefStrength(Enum):
    """Strength with which a belief is held."""
    WEAK = "weak"                  # Peripheral, easily changed
    MODERATE = "moderate"          # Held but open to revision
    STRONG = "strong"              # Core belief, resistant to change
    SACRED = "sacred"              # Inviolable, identity-defining


class PropagationMode(Enum):
    """How ideas propagate through the collective."""
    CONTAGION = "contagion"        # Emotional spread, mimicry
    PERSUASION = "persuasion"      # Reasoned argument
    AUTHORITY = "authority"        # Top-down imposition
    IMITATION = "imitation"        # Copying observed behavior
    RITUAL = "ritual"              # Ceremonial reinforcement
    MEDIA = "media"                # Mass communication channels
    VIRAL = "viral"                # Exponential digital sharing


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class IndividualBelief:
    """A belief held by an individual member of the collective."""
    agent_id: str
    belief_id: str
    content: str
    strength: BeliefStrength
    confidence: float = 0.5        # 0.0-1.0
    emotional_valence: float = 0.0  # -1.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "belief_id": self.belief_id,
            "content": self.content,
            "strength": self.strength.value,
            "confidence": round(self.confidence, 4),
            "emotional_valence": round(self.emotional_valence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SocialSignal:
    """A social signal observed in the collective."""
    signal_id: str
    signal_type: str               # "agreement", "dissent", "enthusiasm", "apathy"
    source_agent: str
    intensity: float               # 0.0-1.0
    reach: float                   # 0.0-1.0, fraction of group reached
    propagation_mode: PropagationMode = PropagationMode.CONTAGION
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source_agent": self.source_agent,
            "intensity": round(self.intensity, 4),
            "reach": round(self.reach, 4),
            "propagation_mode": self.propagation_mode.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CollectiveInput:
    """
    Input for collective consciousness processing.
    """
    collective_type: CollectiveType
    individual_beliefs: List[IndividualBelief] = field(default_factory=list)
    social_signals: List[SocialSignal] = field(default_factory=list)
    group_size: int = 0
    context: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collective_type": self.collective_type.value,
            "belief_count": len(self.individual_beliefs),
            "signal_count": len(self.social_signals),
            "group_size": self.group_size,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class SharedRepresentation:
    """A belief or idea shared across the collective."""
    representation_id: str
    content: str
    adoption_rate: float           # 0.0-1.0, fraction holding this belief
    strength: BeliefStrength
    emotional_charge: float        # -1.0 to 1.0
    stability: float               # 0.0-1.0, resistance to change
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "content": self.content,
            "adoption_rate": round(self.adoption_rate, 4),
            "strength": self.strength.value,
            "emotional_charge": round(self.emotional_charge, 4),
            "stability": round(self.stability, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmergentProperty:
    """An emergent property of the collective not present in individuals."""
    property_id: str
    name: str
    description: str
    intensity: float               # 0.0-1.0
    contributing_agents: int       # Number of agents contributing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_id": self.property_id,
            "name": self.name,
            "description": self.description,
            "intensity": round(self.intensity, 4),
            "contributing_agents": self.contributing_agents,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CollectiveOutput:
    """
    Complete output of a collective consciousness processing cycle.
    """
    collective_type: CollectiveType
    group_mind_state: GroupMindState
    cohesion: SocialCohesion
    shared_representations: List[SharedRepresentation]
    emergent_properties: List[EmergentProperty]
    belief_diversity: float        # 0.0-1.0, Shannon-like diversity
    consensus_level: float         # 0.0-1.0
    polarization_index: float      # 0.0-1.0
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collective_type": self.collective_type.value,
            "group_mind_state": self.group_mind_state.value,
            "cohesion": self.cohesion.value,
            "shared_representation_count": len(self.shared_representations),
            "emergent_property_count": len(self.emergent_properties),
            "belief_diversity": round(self.belief_diversity, 4),
            "consensus_level": round(self.consensus_level, 4),
            "polarization_index": round(self.polarization_index, 4),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class CollectiveConsciousnessInterface:
    """
    Main interface for Form 20: Collective Consciousness (Durkheim).

    Models how individual beliefs aggregate into shared representations,
    how social cohesion emerges from collective sentiment, and how
    ideas propagate through social networks.
    """

    FORM_ID = "20-collective-consciousness"
    FORM_NAME = "Collective Consciousness (Durkheim)"

    def __init__(self):
        """Initialize the collective consciousness interface."""
        # Belief aggregation state
        self._belief_pool: Dict[str, List[IndividualBelief]] = {}
        self._shared_representations: Dict[str, SharedRepresentation] = {}
        self._emergent_properties: Dict[str, EmergentProperty] = {}

        # Counters
        self._representation_counter: int = 0
        self._property_counter: int = 0
        self._cycle_counter: int = 0

        # Meme propagation state
        self._active_memes: Dict[str, Dict[str, Any]] = {}
        self._meme_counter: int = 0

        # Configuration
        self._consensus_threshold: float = 0.6
        self._emergence_threshold: float = 0.4
        self._max_history: int = 100

        # History
        self._output_history: List[CollectiveOutput] = []

        self._initialized = False
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the collective consciousness interface."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.FORM_NAME}...")
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized successfully")

    # ========================================================================
    # CORE METHODS
    # ========================================================================

    async def aggregate_beliefs(
        self, collective_input: CollectiveInput
    ) -> List[SharedRepresentation]:
        """
        Aggregate individual beliefs into shared representations.

        Examines the beliefs held by individual agents and identifies
        common beliefs that rise above the consensus threshold. These
        shared representations form the substance of collective consciousness.

        Args:
            collective_input: Input with individual beliefs and context.

        Returns:
            List of SharedRepresentation objects.
        """
        # Group beliefs by content
        content_groups: Dict[str, List[IndividualBelief]] = {}
        for belief in collective_input.individual_beliefs:
            key = belief.belief_id
            if key not in content_groups:
                content_groups[key] = []
            content_groups[key].append(belief)

        # Update belief pool
        for key, beliefs in content_groups.items():
            self._belief_pool[key] = beliefs

        representations: List[SharedRepresentation] = []
        group_size = max(collective_input.group_size, len(set(
            b.agent_id for b in collective_input.individual_beliefs
        )), 1)

        for belief_id, beliefs in content_groups.items():
            adoption_rate = len(beliefs) / group_size

            if adoption_rate >= self._consensus_threshold:
                self._representation_counter += 1

                # Aggregate strength
                strengths = [b.strength for b in beliefs]
                agg_strength = self._aggregate_strength(strengths)

                # Aggregate emotional charge
                avg_emotion = sum(b.emotional_valence for b in beliefs) / len(beliefs)

                # Compute stability
                stability = self._compute_belief_stability(beliefs)

                rep = SharedRepresentation(
                    representation_id=f"sr_{self._representation_counter:06d}",
                    content=beliefs[0].content,
                    adoption_rate=adoption_rate,
                    strength=agg_strength,
                    emotional_charge=avg_emotion,
                    stability=stability,
                )
                representations.append(rep)
                self._shared_representations[rep.representation_id] = rep

        return representations

    async def detect_emergence(
        self, collective_input: CollectiveInput
    ) -> List[EmergentProperty]:
        """
        Detect emergent properties of the collective.

        Emergent properties are collective-level phenomena that cannot be
        reduced to individual properties -- collective effervescence,
        group polarization, herd behavior, etc.

        Args:
            collective_input: Input with signals and beliefs.

        Returns:
            List of EmergentProperty objects detected.
        """
        properties: List[EmergentProperty] = []
        signals = collective_input.social_signals
        beliefs = collective_input.individual_beliefs

        # Detect collective effervescence (high shared enthusiasm)
        enthusiasm_signals = [
            s for s in signals if s.signal_type == "enthusiasm"
        ]
        if enthusiasm_signals:
            avg_intensity = sum(s.intensity for s in enthusiasm_signals) / len(enthusiasm_signals)
            avg_reach = sum(s.reach for s in enthusiasm_signals) / len(enthusiasm_signals)
            if avg_intensity * avg_reach > self._emergence_threshold:
                self._property_counter += 1
                properties.append(EmergentProperty(
                    property_id=f"ep_{self._property_counter:06d}",
                    name="collective_effervescence",
                    description="Heightened shared emotional energy (Durkheim)",
                    intensity=avg_intensity * avg_reach,
                    contributing_agents=len(enthusiasm_signals),
                ))

        # Detect polarization (opposing beliefs with strong conviction)
        if beliefs:
            positive = [b for b in beliefs if b.emotional_valence > 0.3]
            negative = [b for b in beliefs if b.emotional_valence < -0.3]
            if positive and negative:
                balance = min(len(positive), len(negative)) / max(len(positive), len(negative))
                if balance > 0.3:  # Roughly balanced opposition
                    self._property_counter += 1
                    properties.append(EmergentProperty(
                        property_id=f"ep_{self._property_counter:06d}",
                        name="group_polarization",
                        description="Division into opposing factions",
                        intensity=balance,
                        contributing_agents=len(positive) + len(negative),
                    ))

        # Detect herd behavior (uniform signals with high reach)
        agreement_signals = [
            s for s in signals if s.signal_type == "agreement"
        ]
        if agreement_signals:
            avg_reach = sum(s.reach for s in agreement_signals) / len(agreement_signals)
            if avg_reach > 0.7:
                self._property_counter += 1
                properties.append(EmergentProperty(
                    property_id=f"ep_{self._property_counter:06d}",
                    name="herd_behavior",
                    description="Uniform collective response without individual deliberation",
                    intensity=avg_reach,
                    contributing_agents=len(agreement_signals),
                ))

        for prop in properties:
            self._emergent_properties[prop.property_id] = prop

        return properties

    async def measure_cohesion(
        self, collective_input: CollectiveInput
    ) -> SocialCohesion:
        """
        Measure the social cohesion of the collective.

        Cohesion reflects how strongly members share beliefs, values,
        and identities. Higher cohesion corresponds to Durkheim's
        mechanical solidarity; lower cohesion may indicate anomie.

        Args:
            collective_input: Input with beliefs and signals.

        Returns:
            SocialCohesion level.
        """
        beliefs = collective_input.individual_beliefs
        if not beliefs:
            return SocialCohesion.FRAGMENTED

        # Compute belief agreement rate
        unique_beliefs = set(b.belief_id for b in beliefs)
        agents = set(b.agent_id for b in beliefs)

        if len(agents) <= 1:
            return SocialCohesion.TIGHTLY_INTEGRATED

        # Average overlap: for each pair of agents, what fraction of beliefs match?
        agent_beliefs: Dict[str, Set[str]] = {}
        for b in beliefs:
            if b.agent_id not in agent_beliefs:
                agent_beliefs[b.agent_id] = set()
            agent_beliefs[b.agent_id].add(b.belief_id)

        agent_list = list(agent_beliefs.keys())
        overlap_sum = 0.0
        pair_count = 0
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                s1 = agent_beliefs[agent_list[i]]
                s2 = agent_beliefs[agent_list[j]]
                union = len(s1 | s2)
                intersection = len(s1 & s2)
                if union > 0:
                    overlap_sum += intersection / union
                pair_count += 1

        avg_overlap = overlap_sum / pair_count if pair_count > 0 else 0.0

        # Map to cohesion level
        if avg_overlap > 0.8:
            return SocialCohesion.TIGHTLY_INTEGRATED
        elif avg_overlap > 0.6:
            return SocialCohesion.COHESIVE
        elif avg_overlap > 0.4:
            return SocialCohesion.MODERATE
        elif avg_overlap > 0.2:
            return SocialCohesion.LOOSELY_CONNECTED
        else:
            return SocialCohesion.FRAGMENTED

    async def propagate_meme(
        self,
        meme_content: str,
        initial_adopters: int,
        group_size: int,
        propagation_mode: PropagationMode = PropagationMode.VIRAL,
        generations: int = 5,
    ) -> Dict[str, Any]:
        """
        Simulate the propagation of an idea (meme) through the collective.

        Models how an idea spreads from initial adopters through a group
        using the specified propagation mode. Returns adoption statistics
        over time.

        Args:
            meme_content: Description of the idea.
            initial_adopters: Number of agents initially holding the idea.
            group_size: Total number of agents.
            propagation_mode: How the idea spreads.
            generations: Number of propagation cycles to simulate.

        Returns:
            Dictionary with propagation statistics.
        """
        self._meme_counter += 1
        meme_id = f"meme_{self._meme_counter:06d}"

        # Propagation rates by mode
        mode_rates = {
            PropagationMode.CONTAGION: 0.3,
            PropagationMode.PERSUASION: 0.15,
            PropagationMode.AUTHORITY: 0.4,
            PropagationMode.IMITATION: 0.25,
            PropagationMode.RITUAL: 0.1,
            PropagationMode.MEDIA: 0.35,
            PropagationMode.VIRAL: 0.5,
        }
        rate = mode_rates.get(propagation_mode, 0.2)

        # Simulate propagation
        adopters = initial_adopters
        history = [adopters]
        for gen in range(generations):
            non_adopters = group_size - adopters
            new_adopters = int(non_adopters * rate * (adopters / group_size))
            adopters = min(group_size, adopters + max(1, new_adopters))
            history.append(adopters)

        final_adoption = adopters / group_size if group_size > 0 else 0.0

        result = {
            "meme_id": meme_id,
            "content": meme_content,
            "propagation_mode": propagation_mode.value,
            "initial_adopters": initial_adopters,
            "group_size": group_size,
            "generations": generations,
            "final_adopters": adopters,
            "adoption_rate": round(final_adoption, 4),
            "adoption_history": history,
        }

        self._active_memes[meme_id] = result
        return result

    # ========================================================================
    # STATUS AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "shared_representations": len(self._shared_representations),
            "emergent_properties": len(self._emergent_properties),
            "active_memes": len(self._active_memes),
            "belief_pool_size": sum(len(v) for v in self._belief_pool.values()),
            "cycle_count": self._cycle_counter,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current operational status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "shared_representation_count": len(self._shared_representations),
            "emergent_property_count": len(self._emergent_properties),
            "active_meme_count": len(self._active_memes),
            "total_beliefs_tracked": sum(len(v) for v in self._belief_pool.values()),
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _aggregate_strength(self, strengths: List[BeliefStrength]) -> BeliefStrength:
        """Aggregate belief strengths."""
        strength_order = [
            BeliefStrength.WEAK,
            BeliefStrength.MODERATE,
            BeliefStrength.STRONG,
            BeliefStrength.SACRED,
        ]
        indices = [strength_order.index(s) for s in strengths]
        avg_idx = sum(indices) / len(indices)
        return strength_order[min(int(round(avg_idx)), len(strength_order) - 1)]

    def _compute_belief_stability(self, beliefs: List[IndividualBelief]) -> float:
        """Compute stability of a shared belief."""
        if not beliefs:
            return 0.0

        # Higher confidence and stronger beliefs = more stable
        avg_confidence = sum(b.confidence for b in beliefs) / len(beliefs)
        strength_scores = {
            BeliefStrength.WEAK: 0.2,
            BeliefStrength.MODERATE: 0.5,
            BeliefStrength.STRONG: 0.8,
            BeliefStrength.SACRED: 1.0,
        }
        avg_strength = sum(
            strength_scores.get(b.strength, 0.5) for b in beliefs
        ) / len(beliefs)

        return (avg_confidence * 0.4 + avg_strength * 0.6)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_collective_consciousness_interface() -> CollectiveConsciousnessInterface:
    """Create and return a collective consciousness interface instance."""
    return CollectiveConsciousnessInterface()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "CollectiveType",
    "SocialCohesion",
    "GroupMindState",
    "BeliefStrength",
    "PropagationMode",
    # Input dataclasses
    "IndividualBelief",
    "SocialSignal",
    "CollectiveInput",
    # Output dataclasses
    "SharedRepresentation",
    "EmergentProperty",
    "CollectiveOutput",
    # Interface
    "CollectiveConsciousnessInterface",
    # Convenience
    "create_collective_consciousness_interface",
]
