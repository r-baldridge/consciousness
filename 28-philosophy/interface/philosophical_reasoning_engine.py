#!/usr/bin/env python3
"""
Philosophical Reasoning Engine

Core reasoning algorithms for Form 28, including argument analysis,
dialectical synthesis, and multi-hop philosophical reasoning.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .philosophical_consciousness_interface import (
    PhilosophicalConcept,
    PhilosophicalArgument,
    PhilosophicalTradition,
    PhilosophicalDomain,
    ArgumentType,
)

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Modes of philosophical reasoning"""
    ANALYTICAL = "analytical"           # Logical analysis, clarity
    DIALECTICAL = "dialectical"         # Thesis-antithesis-synthesis
    PHENOMENOLOGICAL = "phenomenological"  # Direct experience investigation
    HERMENEUTIC = "hermeneutic"         # Interpretive understanding
    PRAGMATIC = "pragmatic"             # Practical consequences
    CONTEMPLATIVE = "contemplative"      # Meditative inquiry


@dataclass
class ArgumentAnalysis:
    """Result of analyzing a philosophical argument."""
    argument_id: str
    validity_assessment: str
    validity_score: float  # 0.0-1.0
    premise_assessments: List[Dict[str, Any]]
    hidden_assumptions: List[str]
    soundness_assessment: str
    soundness_score: float
    related_arguments: List[str]
    objections_summary: List[str]
    overall_evaluation: str


@dataclass
class DialecticalResult:
    """Result of dialectical synthesis."""
    thesis: str
    antithesis: str
    synthesis: str
    contradiction_identified: str
    thesis_truth_preserved: str
    antithesis_truth_preserved: str
    aufhebung_analysis: Dict[str, Any]


@dataclass
class ReasoningStep:
    """A step in multi-hop reasoning."""
    step_number: int
    question: str
    answer: str
    concepts_used: List[str]
    confidence: float


@dataclass
class MultiHopResult:
    """Result of multi-hop philosophical reasoning."""
    original_question: str
    reasoning_chain: List[ReasoningStep]
    final_answer: str
    confidence: float
    traditions_consulted: List[PhilosophicalTradition]


class PhilosophicalReasoningEngine:
    """
    Engine for philosophical reasoning operations.

    Provides:
    - Argument analysis and evaluation
    - Dialectical synthesis
    - Multi-hop reasoning
    - Cross-tradition comparison
    """

    def __init__(self, consciousness_interface):
        """
        Initialize the reasoning engine.

        Args:
            consciousness_interface: Reference to PhilosophicalConsciousnessInterface
        """
        self.interface = consciousness_interface
        self.reasoning_mode = ReasoningMode.ANALYTICAL
        logger.info("PhilosophicalReasoningEngine initialized")

    def set_reasoning_mode(self, mode: ReasoningMode) -> None:
        """Set the current reasoning mode."""
        self.reasoning_mode = mode
        logger.debug(f"Reasoning mode set to: {mode.value}")

    # ========================================================================
    # ARGUMENT ANALYSIS
    # ========================================================================

    async def analyze_argument(
        self,
        argument: PhilosophicalArgument
    ) -> ArgumentAnalysis:
        """
        Analyze a philosophical argument for validity and soundness.

        Evaluates:
        - Logical validity (does conclusion follow from premises?)
        - Premise plausibility
        - Hidden assumptions
        - Relation to objections
        """
        # Assess validity
        validity_result = self._assess_validity(argument)

        # Assess each premise
        premise_assessments = []
        for i, premise in enumerate(argument.premises):
            assessment = await self._assess_premise(premise, argument.tradition)
            premise_assessments.append({
                "premise_number": i + 1,
                "premise": premise,
                "plausibility": assessment["plausibility"],
                "support": assessment["support"],
                "challenges": assessment["challenges"]
            })

        # Identify hidden assumptions
        assumptions = self._identify_assumptions(argument)

        # Calculate soundness
        avg_plausibility = sum(p["plausibility"] for p in premise_assessments) / len(premise_assessments) if premise_assessments else 0
        soundness_score = validity_result["score"] * avg_plausibility

        # Find related arguments
        related = await self._find_related_arguments(argument)

        # Summarize objections
        objection_summary = [obj.get("objection", "") for obj in argument.objections[:3]]

        # Overall evaluation
        overall = self._generate_overall_evaluation(
            validity_result,
            premise_assessments,
            soundness_score
        )

        return ArgumentAnalysis(
            argument_id=argument.argument_id,
            validity_assessment=validity_result["assessment"],
            validity_score=validity_result["score"],
            premise_assessments=premise_assessments,
            hidden_assumptions=assumptions,
            soundness_assessment=f"Soundness score: {soundness_score:.2f}",
            soundness_score=soundness_score,
            related_arguments=related,
            objections_summary=objection_summary,
            overall_evaluation=overall
        )

    def _assess_validity(self, argument: PhilosophicalArgument) -> Dict[str, Any]:
        """Assess logical validity of an argument."""
        # Check argument type-specific validity criteria
        if argument.argument_type == ArgumentType.DEDUCTIVE:
            # Deductive: conclusion must necessarily follow
            # Simplified check based on logical form
            if argument.logical_form:
                return {
                    "assessment": "Deductive argument with explicit logical form",
                    "score": 0.9
                }
            return {
                "assessment": "Deductive argument - validity depends on logical structure",
                "score": 0.7
            }

        elif argument.argument_type == ArgumentType.INDUCTIVE:
            return {
                "assessment": "Inductive argument - conclusion probably follows",
                "score": 0.6
            }

        elif argument.argument_type == ArgumentType.TRANSCENDENTAL:
            return {
                "assessment": "Transcendental argument - establishes necessary conditions",
                "score": 0.8
            }

        elif argument.argument_type == ArgumentType.DIALECTICAL:
            return {
                "assessment": "Dialectical argument - synthesis emerges from opposition",
                "score": 0.75
            }

        elif argument.argument_type == ArgumentType.REDUCTIO:
            return {
                "assessment": "Reductio ad absurdum - valid if contradiction genuinely follows",
                "score": 0.85
            }

        return {
            "assessment": f"{argument.argument_type.value} argument",
            "score": 0.5
        }

    async def _assess_premise(
        self,
        premise: str,
        tradition: PhilosophicalTradition
    ) -> Dict[str, Any]:
        """Assess plausibility of a premise."""
        # Search for supporting concepts
        result = await self.interface.query_concept(
            query=premise,
            max_results=3
        )

        supporting = []
        challenging = []

        for concept in result.get("concepts", []):
            # Simple check - would use semantic similarity in production
            if concept.tradition == tradition:
                supporting.append(concept.name)
            elif concept.opposed_concepts:
                challenging.append(concept.name)

        # Calculate plausibility
        plausibility = 0.5  # Base
        plausibility += len(supporting) * 0.1
        plausibility -= len(challenging) * 0.05
        plausibility = max(0.1, min(1.0, plausibility))

        return {
            "plausibility": plausibility,
            "support": supporting,
            "challenges": challenging
        }

    def _identify_assumptions(self, argument: PhilosophicalArgument) -> List[str]:
        """Identify hidden assumptions in an argument."""
        assumptions = []

        # Check for common implicit assumptions based on tradition
        tradition_assumptions = {
            PhilosophicalTradition.KANTIAN: [
                "Reason can operate independently of experience",
                "Moral principles must be universalizable"
            ],
            PhilosophicalTradition.EMPIRICISM: [
                "All knowledge derives from sensory experience",
                "There are no innate ideas"
            ],
            PhilosophicalTradition.RATIONALISM: [
                "Reason can grasp truths independently of experience",
                "Clear and distinct ideas are reliable"
            ],
            PhilosophicalTradition.UTILITARIANISM: [
                "Consequences determine moral worth",
                "Pleasure/happiness is intrinsically valuable"
            ],
        }

        if argument.tradition in tradition_assumptions:
            assumptions.extend(tradition_assumptions[argument.tradition])

        # Add argument-specific assumptions
        if argument.assumptions:
            assumptions.extend(argument.assumptions)

        return assumptions[:5]

    async def _find_related_arguments(
        self,
        argument: PhilosophicalArgument
    ) -> List[str]:
        """Find arguments related to the given one."""
        related = list(argument.related_arguments)

        # Search for arguments with similar concepts
        for concept_id in argument.key_concepts[:3]:
            # Would search argument index in production
            pass

        return related[:5]

    def _generate_overall_evaluation(
        self,
        validity: Dict[str, Any],
        premises: List[Dict[str, Any]],
        soundness: float
    ) -> str:
        """Generate overall argument evaluation."""
        if soundness > 0.8:
            strength = "strong"
        elif soundness > 0.5:
            strength = "moderate"
        else:
            strength = "weak"

        weak_premises = [p for p in premises if p["plausibility"] < 0.5]

        evaluation = f"This is a {strength} argument. "
        evaluation += validity["assessment"] + ". "

        if weak_premises:
            evaluation += f"However, {len(weak_premises)} premise(s) have questionable plausibility. "

        return evaluation

    # ========================================================================
    # DIALECTICAL SYNTHESIS
    # ========================================================================

    async def dialectical_synthesis(
        self,
        thesis: str,
        antithesis: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DialecticalResult:
        """
        Perform Hegelian-style dialectical synthesis.

        Takes a thesis and antithesis, identifies the contradiction,
        and generates a synthesis that preserves the truth in both.
        """
        # Analyze both positions
        thesis_analysis = await self._analyze_position(thesis, context)
        antithesis_analysis = await self._analyze_position(antithesis, context)

        # Identify the core contradiction
        contradiction = self._identify_contradiction(thesis_analysis, antithesis_analysis)

        # Extract kernels of truth
        thesis_truth = self._extract_truth_kernel(thesis_analysis)
        antithesis_truth = self._extract_truth_kernel(antithesis_analysis)

        # Generate synthesis
        synthesis = self._generate_dialectical_synthesis(
            thesis_truth, antithesis_truth, contradiction
        )

        # Analyze aufhebung (what is preserved/negated/transcended)
        aufhebung = self._analyze_aufhebung(thesis, antithesis, synthesis)

        return DialecticalResult(
            thesis=thesis,
            antithesis=antithesis,
            synthesis=synthesis,
            contradiction_identified=contradiction,
            thesis_truth_preserved=thesis_truth,
            antithesis_truth_preserved=antithesis_truth,
            aufhebung_analysis=aufhebung
        )

    async def _analyze_position(
        self,
        position: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze a philosophical position."""
        # Query for related concepts
        result = await self.interface.query_concept(position, max_results=5)
        concepts = result.get("concepts", [])

        return {
            "position": position,
            "related_concepts": [c.name for c in concepts],
            "traditions": list(set(c.tradition.value for c in concepts)),
            "domains": list(set(c.domain.value for c in concepts)),
            "key_claims": [position],  # Would extract multiple claims in production
        }

    def _identify_contradiction(
        self,
        thesis_analysis: Dict[str, Any],
        antithesis_analysis: Dict[str, Any]
    ) -> str:
        """Identify the core contradiction between thesis and antithesis."""
        thesis = thesis_analysis["position"]
        antithesis = antithesis_analysis["position"]

        return f"The thesis '{thesis[:50]}...' and antithesis '{antithesis[:50]}...' represent opposing positions"

    def _extract_truth_kernel(self, analysis: Dict[str, Any]) -> str:
        """Extract the kernel of truth from a position."""
        position = analysis["position"]
        # Simplified - would use more sophisticated analysis
        return f"The valid insight in '{position[:50]}...' is its emphasis on its core point"

    def _generate_dialectical_synthesis(
        self,
        thesis_truth: str,
        antithesis_truth: str,
        contradiction: str
    ) -> str:
        """Generate synthesis that preserves both truths."""
        synthesis = (
            f"The synthesis preserves the truth of the thesis ({thesis_truth[:50]}...) "
            f"while also incorporating the antithesis ({antithesis_truth[:50]}...). "
            f"This moves beyond the original contradiction to a higher understanding."
        )
        return synthesis

    def _analyze_aufhebung(
        self,
        thesis: str,
        antithesis: str,
        synthesis: str
    ) -> Dict[str, Any]:
        """Analyze what is preserved, negated, and transcended."""
        return {
            "preserved_from_thesis": f"Core insight of: {thesis[:30]}...",
            "preserved_from_antithesis": f"Core insight of: {antithesis[:30]}...",
            "negated": "The one-sidedness of both original positions",
            "transcended": "The original contradiction itself",
            "new_understanding": synthesis[:100]
        }

    # ========================================================================
    # MULTI-HOP REASONING
    # ========================================================================

    async def multi_hop_reasoning(
        self,
        question: str,
        max_hops: int = 3
    ) -> MultiHopResult:
        """
        Answer complex questions requiring multiple reasoning steps.

        Decomposes the question, answers sub-questions iteratively,
        and synthesizes into a final answer.
        """
        # Decompose question
        sub_questions = self._decompose_question(question)

        # Answer iteratively
        reasoning_chain = []
        accumulated_context = []
        traditions_consulted = set()

        for i, sub_q in enumerate(sub_questions[:max_hops]):
            # Query with accumulated context
            context_query = sub_q + " " + " ".join(accumulated_context[-2:])
            result = await self.interface.query_concept(context_query, max_results=3)

            concepts = result.get("concepts", [])
            for c in concepts:
                traditions_consulted.add(c.tradition)

            # Generate answer for this step
            answer = self._generate_step_answer(sub_q, concepts)
            accumulated_context.append(answer)

            reasoning_chain.append(ReasoningStep(
                step_number=i + 1,
                question=sub_q,
                answer=answer,
                concepts_used=[c.concept_id for c in concepts],
                confidence=0.7 if concepts else 0.3
            ))

        # Synthesize final answer
        final_answer = self._synthesize_final_answer(question, reasoning_chain)

        # Calculate overall confidence
        avg_confidence = sum(s.confidence for s in reasoning_chain) / len(reasoning_chain) if reasoning_chain else 0.5

        return MultiHopResult(
            original_question=question,
            reasoning_chain=reasoning_chain,
            final_answer=final_answer,
            confidence=avg_confidence,
            traditions_consulted=list(traditions_consulted)
        )

    def _decompose_question(self, question: str) -> List[str]:
        """Decompose a complex question into sub-questions."""
        # Simplified decomposition - would use more sophisticated NLP
        sub_questions = [question]

        # Check for compound questions
        if " and " in question.lower():
            parts = question.lower().split(" and ")
            sub_questions = [p.strip() + "?" for p in parts]

        # Add clarifying sub-questions
        if "why" in question.lower():
            sub_questions.append(f"What are the causes of {question[4:].strip('?')}?")
        if "how" in question.lower():
            sub_questions.append(f"What are the mechanisms of {question[4:].strip('?')}?")

        return sub_questions[:3]

    def _generate_step_answer(
        self,
        question: str,
        concepts: List[PhilosophicalConcept]
    ) -> str:
        """Generate answer for a reasoning step."""
        if not concepts:
            return f"No directly relevant concepts found for: {question}"

        concept_summaries = [f"{c.name}: {c.definition[:100]}" for c in concepts[:2]]
        return f"Based on {', '.join(c.name for c in concepts[:2])}: " + "; ".join(concept_summaries)

    def _synthesize_final_answer(
        self,
        question: str,
        chain: List[ReasoningStep]
    ) -> str:
        """Synthesize final answer from reasoning chain."""
        if not chain:
            return "Unable to answer - no reasoning steps completed."

        answer_parts = [f"To answer '{question[:50]}...':\n"]

        for step in chain:
            answer_parts.append(f"Step {step.step_number}: {step.answer[:100]}...")

        answer_parts.append(
            f"\nConclusion: Based on this analysis, the answer integrates insights from {len(chain)} reasoning steps."
        )

        return "\n".join(answer_parts)

    # ========================================================================
    # TRADITION COMPARISON
    # ========================================================================

    async def compare_traditions(
        self,
        topic: str,
        traditions: List[PhilosophicalTradition]
    ) -> Dict[str, Any]:
        """
        Compare how different traditions approach a topic.
        """
        comparisons = {}

        for tradition in traditions:
            result = await self.interface.query_concept(
                topic,
                filters={"traditions": [tradition]},
                max_results=3
            )
            concepts = result.get("concepts", [])

            comparisons[tradition.value] = {
                "concepts": [c.name for c in concepts],
                "definitions": [c.definition for c in concepts],
                "domain_focus": list(set(c.domain.value for c in concepts)),
            }

        # Find commonalities and differences
        all_domains = set()
        for data in comparisons.values():
            all_domains.update(data["domain_focus"])

        shared_domains = all_domains.copy()
        for data in comparisons.values():
            shared_domains &= set(data["domain_focus"])

        return {
            "topic": topic,
            "traditions_compared": [t.value for t in traditions],
            "tradition_perspectives": comparisons,
            "shared_domain_focus": list(shared_domains),
            "diverse_domains": list(all_domains - shared_domains),
        }
