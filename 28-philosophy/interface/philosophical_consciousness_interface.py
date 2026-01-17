#!/usr/bin/env python3
"""
Philosophical Consciousness Interface

Form 28: The comprehensive interface for philosophical consciousness,
integrating Western and Eastern philosophical traditions with agentic
research capabilities, vector embedding for RAG retrieval, and cross-form
integration with the consciousness system.

This module embodies the principle that philosophical inquiry leads to
maturity and more nuanced existence - treat it with respect and priority.
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class PhilosophicalTradition(Enum):
    """Major philosophical traditions indexed in Form 28"""

    # === WESTERN TRADITIONS ===
    PRESOCRATIC = "presocratic"
    PLATONIC = "platonic"
    ARISTOTELIAN = "aristotelian"
    STOICISM = "stoicism"
    EPICUREANISM = "epicureanism"
    SKEPTICISM_ANCIENT = "skepticism_ancient"
    NEOPLATONISM = "neoplatonism"
    SCHOLASTICISM = "scholasticism"
    THOMISM = "thomism"
    RATIONALISM = "rationalism"
    EMPIRICISM = "empiricism"
    KANTIAN = "kantian"
    GERMAN_IDEALISM = "german_idealism"
    HEGELIAN = "hegelian"
    NIETZSCHEAN = "nietzschean"
    UTILITARIANISM = "utilitarianism"
    MARXISM = "marxism"
    PHENOMENOLOGY = "phenomenology"
    EXISTENTIALISM = "existentialism"
    ANALYTIC = "analytic"
    PRAGMATISM = "pragmatism"
    CONTINENTAL = "continental"
    PROCESS_PHILOSOPHY = "process_philosophy"
    CRITICAL_THEORY = "critical_theory"
    POSTMODERNISM = "postmodernism"
    FEMINIST_PHILOSOPHY = "feminist_philosophy"

    # === EASTERN TRADITIONS ===
    BUDDHIST_THERAVADA = "buddhist_theravada"
    BUDDHIST_MAHAYANA = "buddhist_mahayana"
    BUDDHIST_VAJRAYANA = "buddhist_vajrayana"
    BUDDHIST_ZEN = "buddhist_zen"
    BUDDHIST_MADHYAMAKA = "buddhist_madhyamaka"
    BUDDHIST_YOGACARA = "buddhist_yogacara"
    DAOIST = "daoist"
    CONFUCIAN = "confucian"
    NEO_CONFUCIAN = "neo_confucian"
    VEDANTIC_ADVAITA = "vedantic_advaita"
    VEDANTIC_VISHISHTADVAITA = "vedantic_vishishtadvaita"
    VEDANTIC_DVAITA = "vedantic_dvaita"
    SAMKHYA = "samkhya"
    YOGA_PHILOSOPHY = "yoga_philosophy"
    NYAYA = "nyaya"
    VAISHESHIKA = "vaisheshika"
    JAIN = "jain"
    KYOTO_SCHOOL = "kyoto_school"


class PhilosophicalDomain(Enum):
    """Domains of philosophical inquiry"""
    METAPHYSICS = "metaphysics"
    EPISTEMOLOGY = "epistemology"
    ETHICS = "ethics"
    AESTHETICS = "aesthetics"
    LOGIC = "logic"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    PHILOSOPHY_OF_LANGUAGE = "philosophy_of_language"
    POLITICAL_PHILOSOPHY = "political_philosophy"
    PHILOSOPHY_OF_SCIENCE = "philosophy_of_science"
    PHILOSOPHY_OF_RELIGION = "philosophy_of_religion"
    EXISTENTIAL = "existential"
    PHENOMENOLOGICAL = "phenomenological"
    CONSCIOUSNESS_STUDIES = "consciousness_studies"
    SOTERIOLOGY = "soteriology"
    MEDITATION_THEORY = "meditation_theory"


class ArgumentType(Enum):
    """Types of philosophical arguments"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    TRANSCENDENTAL = "transcendental"
    DIALECTICAL = "dialectical"
    PHENOMENOLOGICAL = "phenomenological"
    REDUCTIO = "reductio_ad_absurdum"
    THOUGHT_EXPERIMENT = "thought_experiment"
    ANALOGY = "analogy"


class MaturityLevel(Enum):
    """Depth of philosophical understanding"""
    NASCENT = "nascent"
    DEVELOPING = "developing"
    COMPETENT = "competent"
    PROFICIENT = "proficient"
    MASTERFUL = "masterful"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PhilosophicalConcept:
    """Represents a philosophical concept within the knowledge base."""
    concept_id: str
    name: str
    tradition: PhilosophicalTradition
    domain: PhilosophicalDomain
    definition: str
    alternate_names: List[str] = field(default_factory=list)
    secondary_traditions: List[PhilosophicalTradition] = field(default_factory=list)
    secondary_domains: List[PhilosophicalDomain] = field(default_factory=list)
    extended_description: Optional[str] = None
    key_arguments: List[str] = field(default_factory=list)
    counter_arguments: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    opposed_concepts: List[str] = field(default_factory=list)
    prerequisite_concepts: List[str] = field(default_factory=list)
    key_figures: List[str] = field(default_factory=list)
    primary_texts: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    maturity_score: float = 0.0
    maturity_level: MaturityLevel = MaturityLevel.NASCENT
    research_depth: int = 0
    last_updated: Optional[datetime] = None
    sources: List[Dict[str, str]] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"Concept: {self.name}",
            f"Tradition: {self.tradition.value}",
            f"Domain: {self.domain.value}",
            f"Definition: {self.definition}"
        ]
        if self.extended_description:
            parts.append(f"Description: {self.extended_description[:500]}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "tradition": self.tradition.value,
            "domain": self.domain.value,
            "definition": self.definition,
            "alternate_names": self.alternate_names,
            "secondary_traditions": [t.value for t in self.secondary_traditions],
            "secondary_domains": [d.value for d in self.secondary_domains],
            "extended_description": self.extended_description,
            "key_arguments": self.key_arguments,
            "counter_arguments": self.counter_arguments,
            "related_concepts": self.related_concepts,
            "opposed_concepts": self.opposed_concepts,
            "key_figures": self.key_figures,
            "primary_texts": self.primary_texts,
            "maturity_score": self.maturity_score,
            "maturity_level": self.maturity_level.value,
            "research_depth": self.research_depth,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "sources": self.sources,
        }


@dataclass
class PhilosophicalFigure:
    """Represents a philosopher or thinker."""
    figure_id: str
    name: str
    alternate_names: List[str] = field(default_factory=list)
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    era: Optional[str] = None
    traditions: List[PhilosophicalTradition] = field(default_factory=list)
    domains: List[PhilosophicalDomain] = field(default_factory=list)
    core_ideas: List[str] = field(default_factory=list)
    key_works: List[str] = field(default_factory=list)
    teachers: List[str] = field(default_factory=list)
    students: List[str] = field(default_factory=list)
    influences: List[str] = field(default_factory=list)
    influenced: List[str] = field(default_factory=list)
    biography_summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalText:
    """Represents a primary philosophical text."""
    text_id: str
    title: str
    author: Optional[str] = None
    author_name: Optional[str] = None
    tradition: PhilosophicalTradition = PhilosophicalTradition.ANALYTIC
    domains: List[PhilosophicalDomain] = field(default_factory=list)
    year_written: Optional[int] = None
    summary: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    local_path: Optional[str] = None
    is_indexed: bool = False
    embedding: Optional[List[float]] = None
    sources: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PhilosophicalArgument:
    """Represents a structured philosophical argument."""
    argument_id: str
    name: str
    argument_type: ArgumentType
    tradition: PhilosophicalTradition
    domain: PhilosophicalDomain
    premises: List[str]
    conclusion: str
    logical_form: Optional[str] = None
    originator: Optional[str] = None
    key_concepts: List[str] = field(default_factory=list)
    objections: List[Dict[str, str]] = field(default_factory=list)
    replies: List[Dict[str, str]] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class CrossTraditionSynthesis:
    """Represents a synthesis across traditions."""
    synthesis_id: str
    topic: str
    traditions: List[PhilosophicalTradition]
    concepts_involved: List[str] = field(default_factory=list)
    convergent_insights: List[str] = field(default_factory=list)
    divergent_positions: List[str] = field(default_factory=list)
    complementary_aspects: List[str] = field(default_factory=list)
    synthesis_statement: Optional[str] = None
    coherence_score: float = 0.0
    fidelity_scores: Dict[str, float] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class PhilosophicalMaturityState:
    """Tracks philosophical understanding maturation."""
    total_concepts_integrated: int = 0
    total_figures_indexed: int = 0
    total_texts_indexed: int = 0
    traditions_depth: Dict[str, float] = field(default_factory=dict)
    domains_coverage: Dict[str, float] = field(default_factory=dict)
    cross_tradition_syntheses: int = 0
    research_sessions_completed: int = 0
    wisdom_teachings_integrated: int = 0
    last_maturity_assessment: Optional[datetime] = None

    def get_overall_maturity(self) -> float:
        """Calculate overall philosophical maturity score."""
        if not self.traditions_depth:
            return 0.0
        tradition_avg = sum(self.traditions_depth.values()) / len(self.traditions_depth)
        domain_avg = sum(self.domains_coverage.values()) / len(self.domains_coverage) if self.domains_coverage else 0.0
        synthesis_factor = min(1.0, self.cross_tradition_syntheses / 100)
        return (tradition_avg * 0.4 + domain_avg * 0.3 + synthesis_factor * 0.3)

    def get_maturity_level(self) -> MaturityLevel:
        """Get maturity level enum."""
        score = self.get_overall_maturity()
        if score < 0.2:
            return MaturityLevel.NASCENT
        elif score < 0.4:
            return MaturityLevel.DEVELOPING
        elif score < 0.6:
            return MaturityLevel.COMPETENT
        elif score < 0.8:
            return MaturityLevel.PROFICIENT
        else:
            return MaturityLevel.MASTERFUL


@dataclass
class QueryFilters:
    """Filters for philosophical queries."""
    traditions: Optional[List[PhilosophicalTradition]] = None
    domains: Optional[List[PhilosophicalDomain]] = None
    min_maturity: float = 0.0
    figures: Optional[List[str]] = None


# ============================================================================
# MAIN INTERFACE
# ============================================================================

class PhilosophicalConsciousnessInterface:
    """
    Form 28: Philosophical Consciousness Interface

    Integrates comprehensive philosophical knowledge across Western and Eastern
    traditions with agentic research capabilities and cross-form integration.

    This interface serves as the gateway to philosophical understanding,
    enabling:
    - Semantic search across philosophical knowledge
    - Cross-tradition synthesis
    - Continuous research and knowledge expansion
    - Wisdom integration for engagement
    - Maturity tracking and growth
    """

    FORM_ID = "28-philosophy"
    NAME = "Philosophical Consciousness"

    def __init__(
        self,
        non_dual_interface: Optional[Any] = None,
        index_path: Optional[Path] = None
    ):
        """
        Initialize the Philosophical Consciousness Interface.

        Args:
            non_dual_interface: Reference to Form 27 NonDualConsciousnessInterface
            index_path: Path to philosophical knowledge index
        """
        self.non_dual_interface = non_dual_interface
        self.index_path = index_path or Path(__file__).parent.parent / "index"

        # Knowledge stores
        self.concept_index: Dict[str, PhilosophicalConcept] = {}
        self.figure_index: Dict[str, PhilosophicalFigure] = {}
        self.text_index: Dict[str, PhilosophicalText] = {}
        self.argument_index: Dict[str, PhilosophicalArgument] = {}
        self.synthesis_index: Dict[str, CrossTraditionSynthesis] = {}

        # Maturity tracking
        self.maturity_state = PhilosophicalMaturityState()

        # Embedding model configuration
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.embedding_dimensions = 768

        # Research agent reference (set by coordinator)
        self.research_coordinator = None

        # Wisdom teachings for engagement integration
        self.wisdom_teachings = self._initialize_wisdom_teachings()

        # Initialize indexes
        self._ensure_index_structure()

        logger.info(f"PhilosophicalConsciousnessInterface initialized (Form {self.FORM_ID})")

    def _ensure_index_structure(self) -> None:
        """Ensure index directory structure exists."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        (self.index_path / "embeddings").mkdir(exist_ok=True)
        (self.index_path / "graph").mkdir(exist_ok=True)

    def _initialize_wisdom_teachings(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize wisdom teachings database for engagement integration."""
        return {
            "stoicism": {
                "core": [
                    "Focus on what is within your control, release attachment to externals",
                    "Virtue is the only true good; external circumstances are indifferent",
                    "View challenges as opportunities for practicing excellence",
                ],
                "practical": [
                    "Practice the dichotomy of control daily",
                    "Perform evening reflection on the day's actions",
                    "Prepare for adversity through negative visualization",
                ]
            },
            "aristotelian": {
                "core": [
                    "Happiness comes through the exercise of virtue in accordance with reason",
                    "The mean between extremes points to right action",
                    "Character is shaped through habitual practice of virtuous actions",
                ],
                "practical": [
                    "Identify the virtuous mean in each situation",
                    "Cultivate practical wisdom through experience",
                    "Build good habits through consistent practice",
                ]
            },
            "phenomenology": {
                "core": [
                    "Return to the things themselves through direct experience",
                    "Bracket assumptions to reveal essential structures",
                    "Meaning emerges through intentional relationship with phenomena",
                ],
                "practical": [
                    "Practice epoché - suspending judgment to see clearly",
                    "Attend to the how of experience, not just the what",
                    "Notice the structures that shape your perception",
                ]
            },
            "existentialism": {
                "core": [
                    "Existence precedes essence - you create yourself through choices",
                    "Authenticity requires owning your freedom and responsibility",
                    "Anxiety reveals the weight of radical freedom",
                ],
                "practical": [
                    "Make choices that you could stand behind if repeated eternally",
                    "Confront rather than flee from existential anxiety",
                    "Take responsibility for your situation and response",
                ]
            },
            "daoist": {
                "core": [
                    "The Dao that can be spoken is not the eternal Dao",
                    "Act through non-action (wu-wei); accomplish through naturalness",
                    "Softness overcomes hardness; yielding overcomes rigidity",
                ],
                "practical": [
                    "Practice wu-wei - effortless action aligned with nature",
                    "Cultivate emptiness to be filled with possibility",
                    "Flow like water - adaptable yet persistent",
                ]
            },
            "confucian": {
                "core": [
                    "Cultivate ren (humaneness) through proper relationships",
                    "Li (ritual propriety) harmonizes self and society",
                    "The exemplary person (junzi) leads by moral example",
                ],
                "practical": [
                    "Honor relationships through appropriate conduct",
                    "Practice self-cultivation through study and reflection",
                    "Rectify yourself before seeking to rectify others",
                ]
            },
            "buddhist": {
                "core": [
                    "All conditioned phenomena are impermanent (anicca)",
                    "Suffering arises from craving and attachment",
                    "Liberation comes through the Noble Eightfold Path",
                ],
                "practical": [
                    "Practice mindfulness of the present moment",
                    "Examine the three marks: impermanence, suffering, non-self",
                    "Cultivate wisdom through direct investigation",
                ]
            },
            "vedantic": {
                "core": [
                    "Brahman alone is real; the world is appearance",
                    "Tat tvam asi - That thou art",
                    "Knowledge of the Self dissolves all bondage",
                ],
                "practical": [
                    "Practice neti neti - not this, not this",
                    "Inquire 'Who am I?' to discover the Self",
                    "Discriminate between the real and the unreal",
                ]
            },
        }

    # ========================================================================
    # QUERY METHODS
    # ========================================================================

    async def query_concept(
        self,
        query: str,
        filters: Optional[QueryFilters] = None,
        max_results: int = 10,
        min_relevance: float = 0.5,
        trigger_research: bool = False
    ) -> Dict[str, Any]:
        """
        Query the philosophical concept index.

        Args:
            query: Natural language query or concept name
            filters: Optional filters for traditions, domains, etc.
            max_results: Maximum concepts to return
            min_relevance: Minimum relevance threshold
            trigger_research: If not found, trigger research agent

        Returns:
            Dictionary with matching concepts and metadata
        """
        start_time = time.time()
        filters = filters or QueryFilters()

        # Search by name first (exact/fuzzy)
        name_matches = self._search_by_name(query, self.concept_index)

        # Semantic search if embeddings available
        semantic_matches = await self._semantic_search(
            query,
            collection="concepts",
            top_k=max_results * 2
        )

        # Combine and deduplicate
        all_matches = self._merge_results(name_matches, semantic_matches)

        # Apply filters
        filtered = self._apply_filters(all_matches, filters)

        # Sort by relevance
        sorted_results = sorted(
            filtered,
            key=lambda x: x.get("relevance", 0),
            reverse=True
        )[:max_results]

        # Trigger research if no results and requested
        research_task_id = None
        if not sorted_results and trigger_research and self.research_coordinator:
            research_task_id = await self.research_coordinator.trigger_research(
                query=query,
                priority=5
            )

        processing_time = (time.time() - start_time) * 1000

        return {
            "query": query,
            "concepts": [r["concept"] for r in sorted_results],
            "relevance_scores": {r["concept"].concept_id: r["relevance"] for r in sorted_results},
            "research_triggered": research_task_id is not None,
            "research_task_id": research_task_id,
            "processing_time_ms": processing_time
        }

    async def query_tradition(
        self,
        tradition: PhilosophicalTradition,
        include_concepts: bool = True,
        include_figures: bool = True,
        include_texts: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive information about a philosophical tradition.
        """
        result = {
            "tradition": tradition.value,
            "wisdom_teachings": self.wisdom_teachings.get(tradition.value, {}),
        }

        if include_concepts:
            result["concepts"] = [
                c for c in self.concept_index.values()
                if c.tradition == tradition or tradition in c.secondary_traditions
            ]

        if include_figures:
            result["figures"] = [
                f for f in self.figure_index.values()
                if tradition in f.traditions
            ]

        if include_texts:
            result["texts"] = [
                t for t in self.text_index.values()
                if t.tradition == tradition
            ]

        return result

    # ========================================================================
    # SYNTHESIS METHODS
    # ========================================================================

    async def synthesize_across_traditions(
        self,
        topic: str,
        traditions: List[PhilosophicalTradition],
        preserve_nuance: bool = True
    ) -> CrossTraditionSynthesis:
        """
        Synthesize insights on a topic across multiple traditions.
        """
        synthesis_id = self._generate_id(f"synthesis_{topic}")

        # Gather tradition-specific perspectives
        perspectives = {}
        for tradition in traditions:
            result = await self.query_concept(
                query=topic,
                filters=QueryFilters(traditions=[tradition]),
                max_results=5
            )
            perspectives[tradition] = result.get("concepts", [])

        # Identify convergent insights
        convergent = self._find_convergent_insights(perspectives)

        # Map divergent positions
        divergent = self._map_divergent_positions(perspectives)

        # Find complementary aspects
        complementary = self._find_complementary_aspects(perspectives)

        # Generate synthesis statement
        synthesis_statement = self._generate_synthesis_statement(
            topic=topic,
            convergent=convergent,
            divergent=divergent,
            complementary=complementary,
            preserve_nuance=preserve_nuance
        )

        # Calculate fidelity scores
        fidelity_scores = {
            t.value: self._calculate_fidelity(synthesis_statement, perspectives.get(t, []), t)
            for t in traditions
        }

        synthesis = CrossTraditionSynthesis(
            synthesis_id=synthesis_id,
            topic=topic,
            traditions=traditions,
            concepts_involved=[c.concept_id for p in perspectives.values() for c in p],
            convergent_insights=convergent,
            divergent_positions=divergent,
            complementary_aspects=complementary,
            synthesis_statement=synthesis_statement,
            fidelity_scores=fidelity_scores,
            coherence_score=self._calculate_coherence(synthesis_statement),
            created_at=datetime.now(timezone.utc)
        )

        # Store synthesis
        self.synthesis_index[synthesis_id] = synthesis

        # Update maturity
        self.maturity_state.cross_tradition_syntheses += 1

        return synthesis

    def _find_convergent_insights(
        self,
        perspectives: Dict[PhilosophicalTradition, List[PhilosophicalConcept]]
    ) -> List[str]:
        """Find themes that appear across multiple traditions."""
        # Simplified implementation - would use embeddings in full version
        all_definitions = []
        for tradition, concepts in perspectives.items():
            for concept in concepts:
                all_definitions.append({
                    "tradition": tradition,
                    "text": concept.definition,
                    "name": concept.name
                })

        # Find overlapping themes (simplified)
        convergent = []
        seen_themes = set()
        for item in all_definitions:
            for other in all_definitions:
                if item["tradition"] != other["tradition"]:
                    # Simple word overlap check (would use semantic similarity)
                    words1 = set(item["text"].lower().split())
                    words2 = set(other["text"].lower().split())
                    overlap = words1 & words2
                    if len(overlap) > 5:
                        theme = f"Both {item['tradition'].value} and {other['tradition'].value} address {item['name']}"
                        if theme not in seen_themes:
                            convergent.append(theme)
                            seen_themes.add(theme)

        return convergent[:5]

    def _map_divergent_positions(
        self,
        perspectives: Dict[PhilosophicalTradition, List[PhilosophicalConcept]]
    ) -> List[str]:
        """Identify where traditions diverge."""
        divergent = []
        for tradition, concepts in perspectives.items():
            for concept in concepts:
                if concept.opposed_concepts:
                    divergent.append(
                        f"{tradition.value}: {concept.name} contrasts with {', '.join(concept.opposed_concepts[:3])}"
                    )
        return divergent[:5]

    def _find_complementary_aspects(
        self,
        perspectives: Dict[PhilosophicalTradition, List[PhilosophicalConcept]]
    ) -> List[str]:
        """Find where traditions complement each other."""
        complementary = []
        traditions = list(perspectives.keys())

        for i, t1 in enumerate(traditions):
            for t2 in traditions[i+1:]:
                # Check for different domains covered
                domains1 = set(c.domain for c in perspectives.get(t1, []))
                domains2 = set(c.domain for c in perspectives.get(t2, []))
                unique_to_1 = domains1 - domains2
                unique_to_2 = domains2 - domains1

                if unique_to_1:
                    complementary.append(
                        f"{t1.value} uniquely addresses {', '.join(d.value for d in unique_to_1)}"
                    )
                if unique_to_2:
                    complementary.append(
                        f"{t2.value} uniquely addresses {', '.join(d.value for d in unique_to_2)}"
                    )

        return complementary[:5]

    def _generate_synthesis_statement(
        self,
        topic: str,
        convergent: List[str],
        divergent: List[str],
        complementary: List[str],
        preserve_nuance: bool
    ) -> str:
        """Generate a synthesis statement."""
        parts = [f"Synthesis on '{topic}':\n"]

        if convergent:
            parts.append("Convergent insights: " + "; ".join(convergent[:3]))

        if divergent and preserve_nuance:
            parts.append("\nDivergent positions: " + "; ".join(divergent[:3]))

        if complementary:
            parts.append("\nComplementary aspects: " + "; ".join(complementary[:3]))

        return "\n".join(parts)

    def _calculate_fidelity(
        self,
        synthesis: str,
        concepts: List[PhilosophicalConcept],
        tradition: PhilosophicalTradition
    ) -> float:
        """Calculate fidelity to a tradition."""
        if not concepts:
            return 0.5

        # Check if key terms from concepts appear in synthesis
        synthesis_lower = synthesis.lower()
        matches = 0
        total = 0

        for concept in concepts:
            total += 1
            if concept.name.lower() in synthesis_lower:
                matches += 1

        return matches / total if total > 0 else 0.5

    def _calculate_coherence(self, synthesis: str) -> float:
        """Calculate coherence of synthesis."""
        # Simplified - check for logical connectors and structure
        connectors = ["therefore", "however", "moreover", "thus", "because"]
        score = 0.5

        for connector in connectors:
            if connector in synthesis.lower():
                score += 0.1

        return min(1.0, score)

    # ========================================================================
    # WISDOM INTEGRATION
    # ========================================================================

    def get_wisdom_for_context(
        self,
        context: Dict[str, Any],
        preferred_traditions: Optional[List[PhilosophicalTradition]] = None
    ) -> Dict[str, Any]:
        """
        Get philosophical wisdom appropriate for an engagement context.
        """
        # Determine appropriate traditions
        traditions = preferred_traditions or self._infer_traditions_for_context(context)

        # Select appropriate teachings
        teachings = []
        practical_guidance = []

        for tradition in traditions:
            tradition_key = tradition.value if isinstance(tradition, PhilosophicalTradition) else tradition
            if tradition_key in self.wisdom_teachings:
                teachings.extend(self.wisdom_teachings[tradition_key].get("core", [])[:2])
                practical_guidance.extend(self.wisdom_teachings[tradition_key].get("practical", [])[:2])

        return {
            "teachings": teachings[:5],
            "practical_guidance": practical_guidance[:5],
            "traditions_drawn_from": [t.value if isinstance(t, PhilosophicalTradition) else t for t in traditions],
        }

    def _infer_traditions_for_context(self, context: Dict[str, Any]) -> List[str]:
        """Infer appropriate traditions based on context."""
        traditions = []

        emotional_state = context.get("emotional_state", "neutral")
        current_need = context.get("current_need", "general")

        # Map emotional states to helpful traditions
        if emotional_state == "distressed":
            traditions.extend(["stoicism", "buddhist"])
        elif emotional_state == "seeking":
            traditions.extend(["vedantic", "daoist"])

        # Map needs to traditions
        if "ethics" in current_need.lower():
            traditions.extend(["aristotelian", "confucian"])
        elif "meaning" in current_need.lower():
            traditions.extend(["existentialism", "buddhist"])

        return traditions or ["stoicism", "buddhist"]  # Default

    # ========================================================================
    # INDEX MANAGEMENT
    # ========================================================================

    async def add_concept(self, concept: PhilosophicalConcept) -> str:
        """Add a concept to the index."""
        concept.created_at = concept.created_at or datetime.now(timezone.utc)
        concept.last_updated = datetime.now(timezone.utc)

        # Generate embedding if not provided
        if concept.embedding is None:
            concept.embedding = await self._generate_embedding(concept.to_embedding_text())

        self.concept_index[concept.concept_id] = concept

        # Update maturity
        self.maturity_state.total_concepts_integrated += 1
        self._update_tradition_depth(concept.tradition)
        self._update_domain_coverage(concept.domain)

        logger.info(f"Added concept: {concept.concept_id}")
        return concept.concept_id

    async def add_figure(self, figure: PhilosophicalFigure) -> str:
        """Add a figure to the index."""
        self.figure_index[figure.figure_id] = figure
        self.maturity_state.total_figures_indexed += 1
        logger.info(f"Added figure: {figure.figure_id}")
        return figure.figure_id

    async def add_text(self, text: PhilosophicalText) -> str:
        """Add a text to the index."""
        self.text_index[text.text_id] = text
        self.maturity_state.total_texts_indexed += 1
        logger.info(f"Added text: {text.text_id}")
        return text.text_id

    def _update_tradition_depth(self, tradition: PhilosophicalTradition) -> None:
        """Update tradition depth metric."""
        key = tradition.value
        current = self.maturity_state.traditions_depth.get(key, 0.0)
        # Increment with diminishing returns
        increment = 0.01 * (1 - current)
        self.maturity_state.traditions_depth[key] = min(1.0, current + increment)

    def _update_domain_coverage(self, domain: PhilosophicalDomain) -> None:
        """Update domain coverage metric."""
        key = domain.value
        current = self.maturity_state.domains_coverage.get(key, 0.0)
        increment = 0.01 * (1 - current)
        self.maturity_state.domains_coverage[key] = min(1.0, current + increment)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _search_by_name(
        self,
        query: str,
        index: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search index by name."""
        query_lower = query.lower()
        results = []

        for entity_id, entity in index.items():
            name = getattr(entity, 'name', '').lower()
            alt_names = [n.lower() for n in getattr(entity, 'alternate_names', [])]

            if query_lower == name:
                results.append({"concept": entity, "relevance": 1.0})
            elif query_lower in name:
                results.append({"concept": entity, "relevance": 0.8})
            elif any(query_lower in alt for alt in alt_names):
                results.append({"concept": entity, "relevance": 0.7})

        return results

    async def _semantic_search(
        self,
        query: str,
        collection: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        # Simplified - would use actual vector search in production
        query_embedding = await self._generate_embedding(query)

        if collection == "concepts":
            index = self.concept_index
        elif collection == "figures":
            index = self.figure_index
        else:
            index = {}

        results = []
        for entity_id, entity in index.items():
            if entity.embedding:
                similarity = self._cosine_similarity(query_embedding, entity.embedding)
                if similarity > 0.3:
                    results.append({"concept": entity, "relevance": similarity})

        return sorted(results, key=lambda x: x["relevance"], reverse=True)[:top_k]

    def _merge_results(
        self,
        name_results: List[Dict],
        semantic_results: List[Dict]
    ) -> List[Dict]:
        """Merge and deduplicate results."""
        seen = set()
        merged = []

        for result in name_results + semantic_results:
            concept = result["concept"]
            if concept.concept_id not in seen:
                seen.add(concept.concept_id)
                merged.append(result)

        return merged

    def _apply_filters(
        self,
        results: List[Dict],
        filters: QueryFilters
    ) -> List[Dict]:
        """Apply filters to results."""
        filtered = []

        for result in results:
            concept = result["concept"]

            if filters.traditions:
                if concept.tradition not in filters.traditions:
                    continue

            if filters.domains:
                if concept.domain not in filters.domains:
                    continue

            if filters.min_maturity > 0:
                if concept.maturity_score < filters.min_maturity:
                    continue

            filtered.append(result)

        return filtered

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        # Placeholder - would use actual embedding model
        # Returns deterministic pseudo-embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        embedding = []
        for i in range(self.embedding_dimensions):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
            embedding.append(value)

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_part = hashlib.md5(f"{prefix}{time.time()}".encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{hash_part}"

    # ========================================================================
    # STATISTICS AND MATURITY
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "total_concepts": len(self.concept_index),
            "total_figures": len(self.figure_index),
            "total_texts": len(self.text_index),
            "total_arguments": len(self.argument_index),
            "total_syntheses": len(self.synthesis_index),
            "maturity_state": {
                "overall_maturity": self.maturity_state.get_overall_maturity(),
                "maturity_level": self.maturity_state.get_maturity_level().value,
                "traditions_depth": self.maturity_state.traditions_depth,
                "domains_coverage": self.maturity_state.domains_coverage,
                "cross_tradition_syntheses": self.maturity_state.cross_tradition_syntheses,
            }
        }

    def get_maturity_state(self) -> PhilosophicalMaturityState:
        """Get current maturity state."""
        return self.maturity_state
