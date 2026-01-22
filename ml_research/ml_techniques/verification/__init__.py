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
# SELF-REFINE
# =============================================================================

class StopCondition(Enum):
    """Conditions for stopping the self-refine loop."""
    MAX_ITERATIONS = "max_iterations"      # Stop after N iterations
    NO_CHANGES = "no_changes"              # Stop when refinement produces same output
    QUALITY_THRESHOLD = "quality_threshold"  # Stop when quality score exceeds threshold
    FEEDBACK_EMPTY = "feedback_empty"      # Stop when no more feedback


class SelfRefine(TechniqueBase):
    """
    Self-Refine: Iterative self-refinement without external feedback.

    Paper: "Self-Refine: Iterative Refinement with Self-Feedback"
           (Madaan et al., 2023)
    https://arxiv.org/abs/2303.17651

    The model iteratively improves its own output through a feedback loop:
        1. Generate initial output
        2. Generate feedback on the output
        3. Refine the output based on feedback
        4. Repeat until stopping condition is met

    Unlike methods requiring external feedback or reward models, Self-Refine
    uses the same model for generation, feedback, and refinement.

    Configuration:
        max_iterations: Maximum refinement iterations (default: 3)
        feedback_prompt: Template for generating feedback
        refine_prompt: Template for refinement based on feedback
        stop_condition: When to stop iterating

    Usage:
        refiner = SelfRefine(
            model=my_model,
            max_iterations=5,
            stop_condition=StopCondition.QUALITY_THRESHOLD,
        )
        result = refiner.run({
            "task": "Write a poem about the ocean",
            "initial_output": "The ocean is blue and vast...",
        })
    """

    TECHNIQUE_ID = "self_refine"
    CATEGORY = TechniqueCategory.VERIFICATION

    DEFAULT_FEEDBACK_PROMPT = """
Review the following output and provide specific, actionable feedback for improvement.
Focus on clarity, accuracy, completeness, and quality.

Task: {task}
Output: {output}

Provide feedback on what can be improved:
"""

    DEFAULT_REFINE_PROMPT = """
Refine the following output based on the feedback provided.
Make specific improvements while maintaining the original intent.

Task: {task}
Current Output: {output}
Feedback: {feedback}

Provide an improved version:
"""

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[Any] = None,
        max_iterations: int = 3,
        feedback_prompt: Optional[str] = None,
        refine_prompt: Optional[str] = None,
        stop_condition: StopCondition = StopCondition.MAX_ITERATIONS,
        quality_threshold: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend or model
        self.max_iterations = max_iterations
        self.feedback_prompt = feedback_prompt or self.DEFAULT_FEEDBACK_PROMPT
        self.refine_prompt = refine_prompt or self.DEFAULT_REFINE_PROMPT
        self.stop_condition = stop_condition
        self.quality_threshold = quality_threshold

    def _generate_feedback(self, task: str, output: str) -> str:
        """Generate feedback on the current output (placeholder)."""
        # Real implementation uses LLM with feedback_prompt
        return f"Feedback: Consider improving clarity and adding more detail to '{output[:50]}...'"

    def _refine_output(self, task: str, output: str, feedback: str) -> str:
        """Refine output based on feedback (placeholder)."""
        # Real implementation uses LLM with refine_prompt
        return f"{output}\n[Refined based on: {feedback[:100]}]"

    def _evaluate_quality(self, task: str, output: str) -> float:
        """Evaluate quality of output (placeholder)."""
        # Real implementation uses LLM or quality metrics
        import random
        return random.uniform(0.6, 1.0)

    def _check_stop_condition(
        self,
        iteration: int,
        current_output: str,
        previous_output: str,
        feedback: str,
        quality: float,
    ) -> bool:
        """Check if refinement should stop."""
        if self.stop_condition == StopCondition.MAX_ITERATIONS:
            return iteration >= self.max_iterations
        elif self.stop_condition == StopCondition.NO_CHANGES:
            return current_output == previous_output
        elif self.stop_condition == StopCondition.QUALITY_THRESHOLD:
            return quality >= self.quality_threshold
        elif self.stop_condition == StopCondition.FEEDBACK_EMPTY:
            return not feedback or feedback.strip() == ""
        return iteration >= self.max_iterations

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            task = input_data.get("task", "")
            current_output = input_data.get("initial_output", "")
            if not current_output:
                current_output = input_data.get("output", str(input_data))
        else:
            task = ""
            current_output = str(input_data)

        previous_output = ""
        all_feedback = []
        all_versions = [current_output]

        for iteration in range(self.max_iterations):
            # Generate feedback
            feedback = self._generate_feedback(task, current_output)
            all_feedback.append(feedback)
            trace.append({
                "action": "generate_feedback",
                "iteration": iteration + 1,
                "feedback": feedback[:200],
            })

            # Evaluate quality
            quality = self._evaluate_quality(task, current_output)
            trace.append({
                "action": "evaluate_quality",
                "iteration": iteration + 1,
                "quality": quality,
            })

            # Check stop condition
            if self._check_stop_condition(
                iteration + 1, current_output, previous_output, feedback, quality
            ):
                trace.append({
                    "action": "stop",
                    "iteration": iteration + 1,
                    "reason": self.stop_condition.value,
                })
                break

            # Refine output
            previous_output = current_output
            current_output = self._refine_output(task, current_output, feedback)
            all_versions.append(current_output)
            trace.append({
                "action": "refine",
                "iteration": iteration + 1,
            })

        final_quality = self._evaluate_quality(task, current_output)

        return TechniqueResult(
            success=True,
            output={
                "final_output": current_output,
                "iterations": len(all_versions) - 1,
                "final_quality": final_quality,
                "all_versions": all_versions,
                "all_feedback": all_feedback,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# RECURSIVE CRITICISM AND IMPROVEMENT (RCI)
# =============================================================================

class RCI(TechniqueBase):
    """
    Recursive Criticism and Improvement (RCI).

    Related to Self-Refine but with recursive critique at multiple levels.
    The model critiques its output at progressively deeper levels, then
    applies improvements from the deepest level back up.

    Process:
        1. Generate initial output
        2. Critique at level 1 (surface issues)
        3. Critique at level 2 (deeper structural issues)
        4. ... continue to max depth
        5. Apply improvements from deepest to shallowest
        6. Optionally repeat the entire process

    Configuration:
        critique_depth: Number of critique levels (default: 3)
        improvement_prompt: Template for generating improvements

    Usage:
        rci = RCI(
            model=my_model,
            critique_depth=3,
        )
        result = rci.run({
            "task": "Write a business proposal",
            "output": "Our proposal is to...",
        })
    """

    TECHNIQUE_ID = "rci"
    CATEGORY = TechniqueCategory.VERIFICATION

    CRITIQUE_LEVELS = [
        ("surface", "Check for basic errors: grammar, spelling, formatting, clarity"),
        ("structural", "Check for logical structure: organization, flow, completeness"),
        ("semantic", "Check for deeper issues: accuracy, coherence, argumentation"),
        ("meta", "Check for high-level concerns: alignment with goals, audience fit, overall quality"),
    ]

    DEFAULT_IMPROVEMENT_PROMPT = """
Based on the following critiques at multiple levels, improve the output.
Address issues from the deepest level first, then work toward surface-level fixes.

Original Output: {output}

Critiques:
{critiques}

Provide an improved version that addresses all identified issues:
"""

    def __init__(
        self,
        model: Optional[Any] = None,
        backend: Optional[Any] = None,
        critique_depth: int = 3,
        improvement_prompt: Optional[str] = None,
        max_iterations: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend or model
        self.critique_depth = min(critique_depth, len(self.CRITIQUE_LEVELS))
        self.improvement_prompt = improvement_prompt or self.DEFAULT_IMPROVEMENT_PROMPT
        self.max_iterations = max_iterations

    def _critique_at_level(
        self,
        level: int,
        output: str,
        previous_critiques: List[str],
    ) -> str:
        """Generate critique at a specific level (placeholder)."""
        # Real implementation uses LLM
        level_name, level_desc = self.CRITIQUE_LEVELS[level]
        return f"[Level {level + 1} - {level_name}]: Critique of '{output[:50]}...' - {level_desc}"

    def _apply_improvements(
        self,
        output: str,
        critiques: List[Tuple[str, str]],
    ) -> str:
        """Apply improvements based on all critiques (placeholder)."""
        # Real implementation uses LLM
        critique_summary = "; ".join(f"{name}: {critique[:50]}" for name, critique in critiques)
        return f"{output}\n[Improved based on: {critique_summary}]"

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            task = input_data.get("task", "")
            current_output = input_data.get("output", str(input_data))
        else:
            task = ""
            current_output = str(input_data)

        all_iterations = []

        for iteration in range(self.max_iterations):
            critiques: List[Tuple[str, str]] = []
            previous_critique_texts: List[str] = []

            # Generate critiques at each level
            for level in range(self.critique_depth):
                level_name = self.CRITIQUE_LEVELS[level][0]
                critique = self._critique_at_level(level, current_output, previous_critique_texts)
                critiques.append((level_name, critique))
                previous_critique_texts.append(critique)

                trace.append({
                    "action": "critique",
                    "iteration": iteration + 1,
                    "level": level + 1,
                    "level_name": level_name,
                    "critique": critique[:200],
                })

            # Apply improvements (from deepest to shallowest)
            improved_output = self._apply_improvements(current_output, list(reversed(critiques)))
            trace.append({
                "action": "improve",
                "iteration": iteration + 1,
            })

            all_iterations.append({
                "iteration": iteration + 1,
                "input": current_output,
                "critiques": critiques,
                "output": improved_output,
            })

            current_output = improved_output

        return TechniqueResult(
            success=True,
            output={
                "final_output": current_output,
                "total_iterations": len(all_iterations),
                "critique_depth": self.critique_depth,
                "iterations": all_iterations,
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
    "StopCondition",
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
    "SelfRefine",
    "RCI",
]
