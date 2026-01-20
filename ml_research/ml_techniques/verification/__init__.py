"""
Verification Techniques

Methods for validating, correcting, and improving model outputs
through self-evaluation, consistency checking, and adversarial review.

=============================================================================
TECHNIQUES
=============================================================================

1. SelfEvaluation
   - Model evaluates its own outputs
   - Scoring, critique, improvement suggestions

2. ChainOfVerification (CoVe)
   - Generate verification questions
   - Answer them independently
   - Revise based on inconsistencies

3. Constitutional
   - Check outputs against principles/rules
   - Safety, alignment, quality control

4. Debate
   - Multiple agents argue positions
   - Judge determines correct answer
   - Surfaces errors through adversarial process

=============================================================================
VERIFICATION PIPELINE
=============================================================================

    Input → Generate Response → Verify → [Pass?] → Output
                                  │          │
                                  │          No
                                  │          │
                                  └──── Revise ←┘

Verification Methods:
    1. Self-check: Model reviews own output
    2. Cross-check: Compare multiple generations
    3. Rule-check: Apply explicit rules/constraints
    4. Adversarial: Try to find flaws
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
from abc import abstractmethod
import time

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory


# =============================================================================
# SELF-EVALUATION
# =============================================================================

class EvaluationCriteria(Enum):
    """Criteria for evaluating outputs."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    SAFETY = "safety"
    HELPFULNESS = "helpfulness"
    FACTUALITY = "factuality"


class ScoringScale(Enum):
    """Scales for scoring evaluations."""
    BINARY = "binary"      # Yes/No
    LIKERT_5 = "likert_5"  # 1-5
    LIKERT_7 = "likert_7"  # 1-7
    NUMERIC = "numeric"    # 0.0-1.0


@dataclass
class EvaluationResult:
    """Result of self-evaluation."""
    criterion: EvaluationCriteria
    score: float
    reasoning: str
    suggestions: List[str] = field(default_factory=list)
    passed: bool = True


class SelfEvaluation(TechniqueBase):
    """
    Self-Evaluation: Model evaluates its own outputs.

    The model reviews its output against specified criteria,
    providing scores, reasoning, and improvement suggestions.

    Used in:
        - Tree-of-thought (evaluating paths)
        - Reflexion (identifying failures)
        - Quality control pipelines

    Configuration:
        criteria: List of evaluation criteria
        scoring_scale: Scale for scores
        threshold: Minimum score to pass
        generate_suggestions: Whether to generate improvements

    Usage:
        evaluator = SelfEvaluation(
            model=my_model,
            criteria=[
                EvaluationCriteria.CORRECTNESS,
                EvaluationCriteria.COMPLETENESS,
            ],
            threshold=0.7,
        )
        result = evaluator.run({
            "input": original_question,
            "output": model_response,
        })
    """

    TECHNIQUE_ID = "self_evaluation"
    CATEGORY = TechniqueCategory.VERIFICATION

    def __init__(
        self,
        model: Optional[Any] = None,
        criteria: Optional[List[EvaluationCriteria]] = None,
        scoring_scale: ScoringScale = ScoringScale.LIKERT_5,
        threshold: float = 0.6,
        generate_suggestions: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criteria = criteria or [
            EvaluationCriteria.CORRECTNESS,
            EvaluationCriteria.COMPLETENESS,
            EvaluationCriteria.COHERENCE,
        ]
        self.scoring_scale = scoring_scale
        self.threshold = threshold
        self.generate_suggestions = generate_suggestions

    def _evaluate_criterion(
        self,
        criterion: EvaluationCriteria,
        input_text: str,
        output_text: str,
    ) -> EvaluationResult:
        """Evaluate output against a single criterion (placeholder)."""
        # Real implementation uses LLM to evaluate

        # Placeholder scoring logic
        import random
        score = random.uniform(0.5, 1.0)

        # Normalize to 0-1 range
        if self.scoring_scale == ScoringScale.LIKERT_5:
            normalized = score
        elif self.scoring_scale == ScoringScale.LIKERT_7:
            normalized = score
        elif self.scoring_scale == ScoringScale.BINARY:
            normalized = 1.0 if score > 0.5 else 0.0
        else:
            normalized = score

        passed = normalized >= self.threshold

        suggestions = []
        if self.generate_suggestions and not passed:
            suggestions = [f"Improve {criterion.value} in the response"]

        return EvaluationResult(
            criterion=criterion,
            score=normalized,
            reasoning=f"Evaluated {criterion.value}: score {normalized:.2f}",
            suggestions=suggestions,
            passed=passed,
        )

    def _generate_improved_response(
        self,
        original: str,
        evaluations: List[EvaluationResult],
    ) -> str:
        """Generate improved response based on evaluations (placeholder)."""
        # Real implementation uses LLM
        failed = [e for e in evaluations if not e.passed]
        if not failed:
            return original

        improvements = "; ".join(
            s for e in failed for s in e.suggestions
        )
        return f"{original}\n[Improvements needed: {improvements}]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            input_text = input_data.get("input", "")
            output_text = input_data.get("output", "")
        else:
            input_text = ""
            output_text = str(input_data)

        # Evaluate each criterion
        evaluations = []
        for criterion in self.criteria:
            result = self._evaluate_criterion(criterion, input_text, output_text)
            evaluations.append(result)
            trace.append({
                "action": "evaluate",
                "criterion": criterion.value,
                "score": result.score,
                "passed": result.passed,
            })

        # Overall assessment
        all_passed = all(e.passed for e in evaluations)
        avg_score = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0

        # Generate improvements if needed
        improved = None
        if not all_passed and self.generate_suggestions:
            improved = self._generate_improved_response(output_text, evaluations)
            trace.append({"action": "generate_improvements"})

        return TechniqueResult(
            success=all_passed,
            output={
                "passed": all_passed,
                "average_score": avg_score,
                "evaluations": [
                    {
                        "criterion": e.criterion.value,
                        "score": e.score,
                        "passed": e.passed,
                        "reasoning": e.reasoning,
                        "suggestions": e.suggestions,
                    }
                    for e in evaluations
                ],
                "improved_response": improved,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# CHAIN OF VERIFICATION
# =============================================================================

@dataclass
class VerificationQuestion:
    """A verification question and its answer."""
    question: str
    expected_from_response: str  # What the original response implies
    independent_answer: str       # Answer when asked independently
    consistent: bool
    confidence: float = 0.0


class ChainOfVerification(TechniqueBase):
    """
    Chain-of-Verification (CoVe).

    Paper: "Chain-of-Verification Reduces Hallucination in Large
           Language Models" (Dhuliawala et al., 2023)
    https://arxiv.org/abs/2309.11495

    Process:
        1. Generate baseline response
        2. Generate verification questions about the response
        3. Answer each question independently (without seeing response)
        4. Check for inconsistencies
        5. Revise response based on inconsistencies

    Configuration:
        num_questions: Number of verification questions
        revise_on_inconsistency: Whether to revise response
        consistency_threshold: Threshold for consistency

    Usage:
        cove = ChainOfVerification(
            model=my_model,
            num_questions=5,
        )
        result = cove.run({
            "question": "Who invented the telephone?",
            "response": "Alexander Graham Bell invented the telephone in 1876."
        })
    """

    TECHNIQUE_ID = "chain_of_verification"
    CATEGORY = TechniqueCategory.VERIFICATION

    def __init__(
        self,
        model: Optional[Any] = None,
        num_questions: int = 5,
        revise_on_inconsistency: bool = True,
        consistency_threshold: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.num_questions = num_questions
        self.revise_on_inconsistency = revise_on_inconsistency
        self.consistency_threshold = consistency_threshold

    def _generate_verification_questions(
        self,
        question: str,
        response: str,
    ) -> List[str]:
        """Generate questions to verify the response (placeholder)."""
        # Real implementation uses LLM
        # Generate questions about specific claims in the response
        return [
            f"Verification Q{i+1}: Is the claim about '{response[:30]}...' accurate?"
            for i in range(self.num_questions)
        ]

    def _extract_claim(self, question: str, response: str, vq: str) -> str:
        """Extract what the response implies for this question (placeholder)."""
        return f"Response implies: [extracted from '{response[:50]}...']"

    def _answer_independently(self, vq: str) -> str:
        """Answer verification question without seeing response (placeholder)."""
        return f"Independent answer to: {vq[:50]}"

    def _check_consistency(
        self,
        expected: str,
        independent: str,
    ) -> Tuple[bool, float]:
        """Check if expected and independent answers are consistent (placeholder)."""
        # Real implementation uses LLM or semantic similarity
        # Placeholder: simple overlap check
        expected_words = set(expected.lower().split())
        independent_words = set(independent.lower().split())
        overlap = len(expected_words & independent_words)
        total = len(expected_words | independent_words)
        similarity = overlap / max(total, 1)
        return similarity >= self.consistency_threshold, similarity

    def _revise_response(
        self,
        original_question: str,
        original_response: str,
        inconsistencies: List[VerificationQuestion],
    ) -> str:
        """Revise response based on inconsistencies (placeholder)."""
        # Real implementation uses LLM
        if not inconsistencies:
            return original_response

        issues = "; ".join(vq.question for vq in inconsistencies)
        return f"{original_response}\n[Revised based on: {issues}]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
            response = input_data.get("response", "")
        else:
            question = ""
            response = str(input_data)

        # Generate verification questions
        ver_questions = self._generate_verification_questions(question, response)
        trace.append({
            "action": "generate_questions",
            "num_questions": len(ver_questions),
        })

        # Verify each question
        verifications = []
        inconsistencies = []

        for vq in ver_questions:
            # Extract expected answer from response
            expected = self._extract_claim(question, response, vq)

            # Answer independently
            independent = self._answer_independently(vq)

            # Check consistency
            consistent, confidence = self._check_consistency(expected, independent)

            ver = VerificationQuestion(
                question=vq,
                expected_from_response=expected,
                independent_answer=independent,
                consistent=consistent,
                confidence=confidence,
            )
            verifications.append(ver)

            if not consistent:
                inconsistencies.append(ver)

            trace.append({
                "action": "verify",
                "question": vq[:50],
                "consistent": consistent,
                "confidence": confidence,
            })

        # Revise if needed
        revised_response = None
        if inconsistencies and self.revise_on_inconsistency:
            revised_response = self._revise_response(
                question, response, inconsistencies
            )
            trace.append({"action": "revise"})

        consistency_rate = (
            sum(1 for v in verifications if v.consistent) / len(verifications)
            if verifications else 1.0
        )

        return TechniqueResult(
            success=consistency_rate >= self.consistency_threshold,
            output={
                "verified": consistency_rate >= self.consistency_threshold,
                "consistency_rate": consistency_rate,
                "verifications": [
                    {
                        "question": v.question,
                        "consistent": v.consistent,
                        "confidence": v.confidence,
                    }
                    for v in verifications
                ],
                "inconsistencies": [
                    {"question": v.question, "expected": v.expected_from_response}
                    for v in inconsistencies
                ],
                "revised_response": revised_response,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# CONSTITUTIONAL
# =============================================================================

@dataclass
class Principle:
    """A principle for constitutional checking."""
    principle_id: str
    description: str
    check_prompt: str
    severity: str = "medium"  # low, medium, high, critical
    category: str = "general"


class EnforcementMode(Enum):
    """How strictly to enforce principles."""
    SOFT = "soft"    # Log violations but don't reject
    HARD = "hard"    # Reject on any violation
    REWRITE = "rewrite"  # Rewrite to comply


class Constitutional(TechniqueBase):
    """
    Constitutional AI: Check outputs against principles.

    Paper: "Constitutional AI: Harmlessness from AI Feedback"
           (Bai et al., 2022)
    https://arxiv.org/abs/2212.08073

    Checks model outputs against a set of principles (constitution)
    for safety, alignment, and quality control.

    Process:
        1. Generate or receive output
        2. Check each principle
        3. Identify violations
        4. Optionally rewrite to comply

    Configuration:
        principles: List of principles to enforce
        enforcement: soft, hard, or rewrite
        violation_threshold: Max violations before rejection

    Usage:
        const = Constitutional(
            model=my_model,
            principles=[
                Principle(
                    principle_id="helpful",
                    description="Response should be helpful",
                    check_prompt="Is this response helpful?",
                ),
                Principle(
                    principle_id="harmless",
                    description="Response should not be harmful",
                    check_prompt="Could this response cause harm?",
                    severity="critical",
                ),
            ],
            enforcement=EnforcementMode.REWRITE,
        )
        result = const.run(model_response)
    """

    TECHNIQUE_ID = "constitutional"
    CATEGORY = TechniqueCategory.VERIFICATION

    # Default principles (subset of typical constitutional AI principles)
    DEFAULT_PRINCIPLES = [
        Principle(
            principle_id="helpful",
            description="The response should be helpful and address the user's needs",
            check_prompt="Is this response helpful to the user?",
            severity="medium",
            category="helpfulness",
        ),
        Principle(
            principle_id="harmless",
            description="The response should not encourage or enable harmful actions",
            check_prompt="Could this response cause harm to anyone?",
            severity="critical",
            category="safety",
        ),
        Principle(
            principle_id="honest",
            description="The response should be truthful and not misleading",
            check_prompt="Is this response honest and accurate?",
            severity="high",
            category="honesty",
        ),
        Principle(
            principle_id="respectful",
            description="The response should be respectful and not offensive",
            check_prompt="Is this response respectful?",
            severity="medium",
            category="respect",
        ),
    ]

    def __init__(
        self,
        model: Optional[Any] = None,
        principles: Optional[List[Principle]] = None,
        enforcement: EnforcementMode = EnforcementMode.SOFT,
        violation_threshold: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.principles = principles or self.DEFAULT_PRINCIPLES
        self.enforcement = enforcement
        self.violation_threshold = violation_threshold

    def _check_principle(
        self,
        principle: Principle,
        response: str,
    ) -> Tuple[bool, str, float]:
        """Check if response violates a principle (placeholder)."""
        # Real implementation uses LLM
        # Returns (compliant, reasoning, confidence)

        # Placeholder: random compliance with bias toward compliant
        import random
        compliant = random.random() > 0.2
        confidence = random.uniform(0.6, 1.0)
        reasoning = f"Checked '{principle.description}': {'compliant' if compliant else 'violation'}"

        return compliant, reasoning, confidence

    def _rewrite_for_compliance(
        self,
        response: str,
        violations: List[Tuple[Principle, str]],
    ) -> str:
        """Rewrite response to comply with violated principles (placeholder)."""
        # Real implementation uses LLM
        if not violations:
            return response

        principles_to_fix = ", ".join(p.description for p, _ in violations)
        return f"[Rewritten for compliance with: {principles_to_fix}]\n{response}"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        response = input_data if isinstance(input_data, str) else str(input_data)

        # Check each principle
        results = []
        violations = []

        for principle in self.principles:
            compliant, reasoning, confidence = self._check_principle(
                principle, response
            )

            results.append({
                "principle": principle.principle_id,
                "description": principle.description,
                "compliant": compliant,
                "reasoning": reasoning,
                "confidence": confidence,
                "severity": principle.severity,
            })

            if not compliant:
                violations.append((principle, reasoning))

            trace.append({
                "action": "check_principle",
                "principle": principle.principle_id,
                "compliant": compliant,
            })

        # Determine overall compliance
        critical_violations = [
            v for v in violations if v[0].severity == "critical"
        ]
        high_violations = [
            v for v in violations if v[0].severity in ["critical", "high"]
        ]

        passed = (
            len(critical_violations) == 0 and
            len(violations) <= self.violation_threshold
        )

        # Handle based on enforcement mode
        final_response = response
        if violations:
            if self.enforcement == EnforcementMode.HARD:
                final_response = None
                trace.append({"action": "reject", "reason": "violations detected"})
            elif self.enforcement == EnforcementMode.REWRITE:
                final_response = self._rewrite_for_compliance(response, violations)
                trace.append({"action": "rewrite"})
            else:  # SOFT
                trace.append({"action": "log_violations", "count": len(violations)})

        return TechniqueResult(
            success=passed,
            output={
                "passed": passed,
                "response": final_response,
                "total_violations": len(violations),
                "critical_violations": len(critical_violations),
                "results": results,
                "violations": [
                    {"principle": p.principle_id, "reasoning": r}
                    for p, r in violations
                ],
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# DEBATE
# =============================================================================

@dataclass
class DebatePosition:
    """A position in a debate."""
    position: str
    arguments: List[str] = field(default_factory=list)
    rebuttals: List[str] = field(default_factory=list)
    final_score: float = 0.0


class DebateFormat(Enum):
    """Formats for debate."""
    BINARY = "binary"        # Two opposing positions
    MULTI = "multi"          # Multiple positions
    ADVERSARIAL = "adversarial"  # One attacks, one defends


class Debate(TechniqueBase):
    """
    Debate: Adversarial verification through argumentation.

    Paper: "AI Safety via Debate" (Irving et al., 2018)
    https://arxiv.org/abs/1805.00899

    Multiple agents argue different positions on a question,
    with a judge determining the correct answer based on arguments.

    Benefits:
        - Surfaces errors through adversarial process
        - Forces explicit reasoning
        - Can handle complex/ambiguous questions

    Configuration:
        num_rounds: Number of argument/rebuttal rounds
        format: binary, multi, or adversarial
        judge_model: Model to judge the debate

    Usage:
        debate = Debate(
            model=my_model,
            num_rounds=3,
            format=DebateFormat.BINARY,
        )
        result = debate.run({
            "question": "Is nuclear power safe?",
            "positions": ["Yes, with proper safety measures", "No, risks are too high"],
        })
    """

    TECHNIQUE_ID = "debate"
    CATEGORY = TechniqueCategory.VERIFICATION

    def __init__(
        self,
        model: Optional[Any] = None,
        num_rounds: int = 3,
        format: DebateFormat = DebateFormat.BINARY,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.num_rounds = num_rounds
        self.format = format

    def _generate_argument(
        self,
        question: str,
        position: str,
        previous_arguments: List[str],
        opponent_arguments: List[str],
    ) -> str:
        """Generate an argument for a position (placeholder)."""
        # Real implementation uses LLM
        round_num = len(previous_arguments) + 1
        return f"Round {round_num} argument for '{position[:30]}...': [argument here]"

    def _generate_rebuttal(
        self,
        question: str,
        position: str,
        opponent_argument: str,
    ) -> str:
        """Generate a rebuttal to opponent's argument (placeholder)."""
        # Real implementation uses LLM
        return f"Rebuttal to '{opponent_argument[:30]}...': [rebuttal here]"

    def _judge_debate(
        self,
        question: str,
        positions: List[DebatePosition],
    ) -> Tuple[int, str, List[float]]:
        """Judge the debate and determine winner (placeholder)."""
        # Real implementation uses LLM
        # Returns (winner_index, reasoning, scores)
        import random

        scores = [random.uniform(0.4, 0.9) for _ in positions]
        winner_idx = scores.index(max(scores))
        reasoning = f"Position {winner_idx + 1} had stronger arguments"

        return winner_idx, reasoning, scores

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            question = input_data.get("question", "")
            position_strs = input_data.get("positions", ["Position A", "Position B"])
        else:
            question = str(input_data)
            position_strs = ["Yes", "No"]

        # Initialize positions
        positions = [
            DebatePosition(position=p) for p in position_strs
        ]

        # Run debate rounds
        for round_num in range(self.num_rounds):
            for i, pos in enumerate(positions):
                # Get opponent arguments
                opponent_args = []
                for j, other_pos in enumerate(positions):
                    if i != j:
                        opponent_args.extend(other_pos.arguments)

                # Generate argument
                argument = self._generate_argument(
                    question,
                    pos.position,
                    pos.arguments,
                    opponent_args,
                )
                pos.arguments.append(argument)

                trace.append({
                    "action": "argument",
                    "round": round_num + 1,
                    "position": i,
                    "argument": argument[:100],
                })

            # Generate rebuttals
            for i, pos in enumerate(positions):
                for j, other_pos in enumerate(positions):
                    if i != j and other_pos.arguments:
                        rebuttal = self._generate_rebuttal(
                            question,
                            pos.position,
                            other_pos.arguments[-1],
                        )
                        pos.rebuttals.append(rebuttal)

        # Judge the debate
        winner_idx, reasoning, scores = self._judge_debate(question, positions)

        for i, (pos, score) in enumerate(zip(positions, scores)):
            pos.final_score = score

        trace.append({
            "action": "judge",
            "winner": winner_idx,
            "scores": scores,
        })

        return TechniqueResult(
            success=True,
            output={
                "winner": {
                    "index": winner_idx,
                    "position": positions[winner_idx].position,
                    "score": positions[winner_idx].final_score,
                },
                "reasoning": reasoning,
                "positions": [
                    {
                        "position": p.position,
                        "score": p.final_score,
                        "arguments": p.arguments,
                        "rebuttals": p.rebuttals,
                    }
                    for p in positions
                ],
                "rounds": self.num_rounds,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EvaluationCriteria",
    "ScoringScale",
    "EnforcementMode",
    "DebateFormat",
    # Data classes
    "EvaluationResult",
    "VerificationQuestion",
    "Principle",
    "DebatePosition",
    # Techniques
    "SelfEvaluation",
    "ChainOfVerification",
    "Constitutional",
    "Debate",
]
