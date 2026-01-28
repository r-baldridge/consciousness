# Koan Logic: Formalizing Paradox as a Computational Tool

## Purpose

This document formalizes the use of paradox, contradiction, and framework-transcending logic as computational tools for escaping agent loops. The core claim is that certain classes of problems cannot be solved within the framework in which they are posed -- not because the answer is unknown, but because the question itself encodes false assumptions that make any answer within the framework incorrect. Non-dual traditions have developed precise methods for identifying and transcending such malformed questions. This document translates those methods into algorithms.

The non-dual insight is that paradox is not a bug but a feature: it is a signal that the system has reached the limit of a conceptual framework and needs to transcend it. The paradox does not need to be resolved; it needs to be recognized as pointing beyond itself.

---

## 1. The Madhyamaka Tetralemma: Escaping by Transcending the Question

### Philosophical Foundation

Nagarjuna's tetralemma (catuskoti) is one of the most powerful logical tools in the non-dual tradition. It exhausts all possible positions within a binary framework:

1. **P** -- the proposition is true
2. **Not-P** -- the proposition is false
3. **Both P and not-P** -- the proposition is both true and false (dialetheism)
4. **Neither P nor not-P** -- the proposition is neither true nor false

Nagarjuna denies all four. This is not irrationalism. It is the demonstration that the question -- the framework within which P is posed -- contains a false assumption. The answer is not "we cannot know" (agnosticism) but "the question is wrong" (framework transcendence).

In the Mulamadhyamakakarika (24.18), Nagarjuna makes the positive claim: "Whatever is dependently originated, that is explained to be emptiness." The tetralemma clears the ground for this insight by showing that no static, framework-bound answer captures the dynamic, relational nature of reality.

### Algorithm

```
ALGORITHM: TetralemmaAnalysis

INPUT:
  question: The question the agent is stuck on
  option_A: First option the agent is oscillating between
  option_B: Second option
  context: The full context of the task

PROCEDURE:
  1. FIRST LEMMA: Test option_A independently
     a. Assume A is true. What follows?
     b. Generate the implications of A
     c. Check: do the implications contradict the context or task requirements?
     d. Record: evidence_for_A, evidence_against_A

  2. SECOND LEMMA: Test option_B independently
     a. Assume B is true. What follows?
     b. Generate the implications of B
     c. Check: do the implications contradict the context or task requirements?
     d. Record: evidence_for_B, evidence_against_B

  3. THIRD LEMMA: Test both A and B simultaneously
     a. Can A and B both be true in different contexts, scopes, or dimensions?
     b. Is the apparent conflict between A and B due to different interpretations
        of a shared term?
     c. Is there a synthesis that preserves the valid aspects of both?
     d. Record: synthesis_possible, synthesis_description

  4. FOURTH LEMMA: Test neither A nor B
     a. What assumptions does the question make that force the choice
        between A and B?
     b. Are those assumptions valid?
     c. If the assumptions are removed, what other options become available?
     d. Is there a reframing of the question that dissolves the dilemma?
     e. Record: hidden_assumptions, alternative_framings

  5. EVALUATE:
     a. If first or second lemma succeeds cleanly (one option is clearly
        better with no contradictions): return that option
        (The question was not malformed; the agent just needed systematic
        evaluation)
     b. If third lemma succeeds (both are valid in context):
        return the synthesis
        (The question was malformed because it assumed mutual exclusion)
     c. If fourth lemma succeeds (hidden assumptions found):
        return the reframed question and its answer
        (The question was fundamentally malformed)
     d. If no lemma succeeds:
        return FrameworkTranscendenceReport:
        "This question cannot be resolved within the given framework.
         The following assumptions were tested and found problematic:
         [list]. The following alternative framings were considered:
         [list]. Recommendation: [most promising reframing]."
```

### Application to Stuck Types

- **BINARY_OSCILLATION**: The agent is stuck between A and B. The tetralemma systematically evaluates whether A, B, both, or neither is correct, or whether the binary framing itself is the problem.
- **CONTRADICTORY**: The agent has asserted both P and not-P. The tetralemma asks whether the contradiction arises from a hidden equivocation (A and not-A may be true about different aspects), an invalid assumption, or a genuinely paradoxical constraint.

### Formal Specification

The tetralemma can be expressed as a decision procedure over a constraint satisfaction problem:

```
Given: Constraints C = {c1, c2, ..., cn}
       Options O = {A, B}
       Assignment function f: O -> {true, false}

Test:
  f(A) = true,  f(B) = false  => Check SAT(C | f)
  f(A) = false, f(B) = true   => Check SAT(C | f)
  f(A) = true,  f(B) = true   => Check SAT(C | f)
  f(A) = false, f(B) = false  => Check SAT(C | f)

If none satisfies C:
  => C itself is inconsistent
  => Identify the minimal unsatisfiable subset of C
  => That subset contains the false assumptions
```

This maps the philosophical tetralemma onto standard constraint satisfaction, making it a computable operation on the agent's proposition store.

---

## 2. Neti Neti: Systematic Negation as Search Strategy

### Philosophical Foundation

In Advaita Vedanta, the practice of neti neti ("not this, not that") is the systematic negation of everything that is not the Self (Atman). The practitioner examines each candidate for selfhood -- the body, the mind, the emotions, the intellect -- and for each, determines "I am not this." What remains when all candidates have been negated is the Self, which cannot be an object of negation because it is the subject doing the negating.

This is a search strategy of extraordinary elegance: instead of searching for the answer directly (which requires knowing what the answer looks like), systematically eliminate everything that is NOT the answer. The answer is what remains.

### Algorithm

```
ALGORITHM: NetiNeti

INPUT:
  question: The question the agent is stuck on
  candidate_answers: List of answers the agent has considered
  rejection_criteria: Function that evaluates whether an answer
                      fails to satisfy the task requirements

PROCEDURE:
  1. For each candidate C in candidate_answers:
     a. Apply rejection_criteria(C):
        - Does C satisfy all explicit task constraints?
        - Does C produce contradictions with known facts?
        - Does C lead to absurd consequences when followed to
          its logical conclusion?
     b. If rejection_criteria(C) returns a specific failure:
        - Record the failure reason
        - Mark C as "neti" (not this)
        - The failure reason BECOMES a new constraint for future candidates

  2. After processing all candidates:
     a. Collect all failure reasons: these define the SHAPE of the answer
        even though no candidate has been accepted
     b. Generate the "negative space" description:
        "The answer is not X (because of reason R1),
         not Y (because of reason R2), not Z (because of reason R3).
         Therefore, the answer must have properties:
         not-R1, not-R2, not-R3."

  3. Use the negative space to GENERATE new candidates:
     a. Construct a candidate that satisfies all the not-Ri constraints
     b. If no such candidate can be constructed:
        - The constraints themselves may be contradictory
        - Feed the constraints to the TetralemmaAnalysis
     c. If a candidate is found:
        - Test it against the original question
        - If it passes: return it as the answer
        - If it fails: add its failure reason and iterate

  4. TERMINATION:
     a. If a passing candidate is found: return it
     b. If the constraints are contradictory: return FrameworkTranscendenceReport
     c. If MAX_ITERATIONS reached: return the best candidate found
        (the one with the fewest constraint violations)
```

### ML Technique Bridge

Neti neti maps directly to several established ML techniques:

- **Contrastive learning**: Learning what something IS by learning what it is NOT (negative sampling in word2vec, contrastive loss in CLIP). The model learns the boundary of a concept by being shown examples that are near but outside the boundary.
- **Adversarial training**: Generating adversarial examples that fail, thereby defining the model's decision boundary by exclusion.
- **Constraint propagation**: In constraint satisfaction, eliminating impossible values from domains by propagation, which progressively narrows the search space until only the solution remains.
- **Elimination tournaments**: In multi-armed bandit problems, sequentially eliminating the worst option until the best remains.

### Application to Stuck Types

- **REPETITIVE**: The agent keeps proposing similar answers. Neti neti forces it to explicitly reject each answer AND capture the reason for rejection, building up a set of constraints that guides toward novel answers.
- **CONFIDENCE_DRIFT**: The agent cannot settle on an answer. Neti neti provides structure: instead of trying to find the right answer, find all the wrong answers and let the right one emerge by elimination.

---

## 3. Mu: The Answer That Rejects the Question's Framework

### Philosophical Foundation

In the most famous koan in Zen Buddhism, a monk asks Master Zhaozhou: "Does a dog have Buddha-nature?" Zhaozhou answers: "Mu" (Chinese: wu; literally "no," "nothing," or "without"). This answer is not a simple "no." The Buddhist doctrine explicitly states that all sentient beings have Buddha-nature. So Zhaozhou is not denying the doctrine. His "mu" is a rejection of the entire framework of the question -- the assumption that Buddha-nature is a thing that beings either "have" or "don't have." The answer dissolves the question rather than answering it.

Dahui Zonggao's kanhua Chan practice takes this further: the practitioner holds the word "mu" in awareness, generating "great doubt" (yi qing) -- the existential pressure of not being able to resolve the koan through any conceptual strategy. This doubt intensifies until it "shatters" (po) in a moment of direct realization that transcends the conceptual framework entirely.

### Algorithm

```
ALGORITHM: MuAnalysis

INPUT:
  question: The question the agent is stuck on
  stuck_signal: The LoopSignal from the detection ensemble

PROCEDURE:
  1. DECOMPOSE the question into its presuppositions:
     a. What entities does the question assume exist?
     b. What relationships does the question assume hold?
     c. What categories does the question assume are valid?
     d. What outcome does the question assume is achievable?

  2. TEST each presupposition:
     For each presupposition S:
       a. Is S explicitly stated in the task requirements?
       b. Is S supported by evidence?
       c. Is S logically necessary for the task?
       d. Is S the conventional interpretation but not the only one?

  3. IDENTIFY invalid presuppositions:
     invalid = [S for S in presuppositions if not supported and not necessary]

  4. If invalid presuppositions found:
     a. CONSTRUCT a "mu" response:
        "The question '[question]' presupposes [S1, S2, ...].
         Presupposition [Si] is invalid because [reason].
         A better question would be: [reformulated question]."
     b. ANSWER the reformulated question
     c. Return MuResult(
          original_question=question,
          invalid_presuppositions=invalid,
          reformulated_question=reformulated,
          answer_to_reformulated=answer
        )

  5. If no invalid presuppositions found:
     a. The question may be well-formed but beyond the agent's
        current capability
     b. Return None (mu analysis does not apply; try other channels)
```

### Examples of Malformed Questions in AI/ML

| Question | Hidden Assumption | Mu Response |
|----------|-------------------|-------------|
| "Is this model biased or unbiased?" | Bias is binary | Mu: all models have biases; the question is "what biases and how severe?" |
| "Should we use approach A or approach B?" | Exactly one approach is needed | Mu: perhaps a hybrid, perhaps neither, perhaps the framing is wrong |
| "What is the correct classification?" | There is exactly one correct class | Mu: the instance may be genuinely ambiguous or multi-class |
| "Is the agent's reasoning correct?" | Correctness is binary | Mu: reasoning can be partially correct, correct for the wrong reasons, or correct in a different framework |
| "Has the model converged?" | Convergence is a single threshold | Mu: convergence is a distribution-dependent concept; some parameters may have converged while others have not |

### Formal Specification of "Mu"

Mu can be formally specified as a presupposition failure detector:

```
Given: Question Q with presuppositions P(Q) = {p1, p2, ..., pk}
       Evidence set E
       Logical framework F

Mu(Q) is applicable when:
  exists pi in P(Q) such that:
    (E does not support pi) AND
    (pi is not derivable from F) AND
    (Q cannot be meaningfully answered when pi is false)

Mu(Q) outputs:
  The specific pi that fails, the evidence against it,
  and a reformulation of Q that does not depend on pi.
```

This makes mu a computable operation: decompose the question into presuppositions, test each against the evidence base, and identify failures. The reformulation step requires generating a new question that preserves the intent of the original while dropping the invalid presupposition.

---

## 4. Detecting Malformed Questions

### Signals That a Question Is Malformed

Not all stuck states arise from malformed questions, but several patterns are strong signals:

| Signal | Detection Method | Malformation Type |
|--------|-----------------|-------------------|
| All options fail | TetralemmaAnalysis returns no satisfying lemma | Hidden constraint makes all answers impossible |
| Oscillation with equal evidence | ConfidenceOscillation + equal evidence for both sides | False binary; the real answer is not one of the options |
| Circular reasoning | CircularReasoningDetector fires | The question assumes its own answer |
| Increasing complexity without progress | ResourceWasteDetector fires on meta-reasoning | The question sends the reasoner into infinite expansion |
| Contradiction with known facts | ContradictionDetector fires against established knowledge | The question encodes a false premise |

### Question Malformation Taxonomy

```
TYPE 1: FALSE DICHOTOMY
  Signal: Binary oscillation
  Example: "Is X better than Y?" when X and Y are incomparable
  Koan tool: Tetralemma (third or fourth lemma)
  Reformulation: "In what contexts does X excel, and in what
                  contexts does Y excel?"

TYPE 2: FALSE PRESUPPOSITION
  Signal: All candidates fail
  Example: "What time did the event happen?" when the event didn't happen
  Koan tool: Mu analysis
  Reformulation: "Did the event happen? If so, when?"

TYPE 3: CIRCULAR DEFINITION
  Signal: Circular reasoning detected
  Example: "What makes good code good?" (defining quality by itself)
  Koan tool: Neti neti (define by elimination of bad code)
  Reformulation: "What measurable properties distinguish code that
                  succeeds from code that fails?"

TYPE 4: INFINITE REGRESS
  Signal: Self-reference depth exceeds threshold
  Example: "What is the best way to decide the best way to decide?"
  Koan tool: Mushin (cut the meta-levels)
  Reformulation: "Use the following concrete criteria to decide: [list]"

TYPE 5: CATEGORY ERROR
  Signal: Contradictions across different levels of abstraction
  Example: "How many bytes of creativity does this have?"
  Koan tool: Mu (creativity is not measurable in bytes)
  Reformulation: "What aspects of creativity are relevant here, and
                  how can they be assessed?"
```

---

## 5. Using Contradictory Constraints Productively

### Philosophical Foundation

The north-star document identifies adversarial examples as koans: "showing the system its own failure to force transcendence." The non-dual traditions do not treat contradiction as an error to be eliminated but as a signal to be leveraged. The Madhyamaka tradition holds that the simultaneous truth of "all things are empty" and "all things appear" is not a contradiction but a pointer to a reality that transcends the framework in which it appears contradictory.

### Algorithm: Productive Contradiction

```
ALGORITHM: ProductiveContradiction

INPUT:
  contradictions: List of (proposition_P, proposition_not_P) pairs
                  from the ContradictionDetector

PROCEDURE:
  1. For each contradiction (P, not-P):
     a. IDENTIFY the level at which the contradiction operates:
        - Same level (P and not-P about the same thing at the same time):
          => Genuine logical contradiction
          => One of them must be wrong, OR the framework is inadequate
        - Different levels (P at level L1, not-P at level L2):
          => Not a true contradiction; different levels of analysis
          => Both can be simultaneously true at their respective levels
        - Different contexts (P in context C1, not-P in context C2):
          => Not a true contradiction; context-dependent truth
          => The agent should qualify its claims by context

     b. If genuine contradiction:
        - Use as a constraint: any valid answer must be consistent
          with EITHER P or not-P, but not assert both
        - The contradiction reveals a decision point: which one to keep
        - Feed the contradiction to TetralemmaAnalysis to determine
          whether the forcing frame is itself the problem

     c. If level-crossing contradiction:
        - Resolve by making levels explicit:
          "At the architectural level, P holds.
           At the implementation level, not-P holds.
           These are consistent because [explanation]."

     d. If context-dependent contradiction:
        - Resolve by qualifying:
          "In context C1, P holds because [reason].
           In context C2, not-P holds because [reason].
           The general claim must be context-qualified."

  2. MULTI-OBJECTIVE OPTIMIZATION UNDER CONFLICT:
     When contradictions arise from genuinely conflicting objectives
     (e.g., maximize accuracy AND minimize latency):
     a. Acknowledge that no single solution optimizes all objectives
     b. Compute the Pareto frontier: the set of solutions where
        improving one objective necessarily worsens another
     c. Present the trade-off explicitly:
        "The following solutions are Pareto-optimal: [list].
         Each makes the following trade-offs: [table].
         The choice between them depends on priority, which is
         not specified in the task."
     d. If priority IS specified: select the Pareto-optimal solution
        that best matches the priority ordering

  3. Return ProductiveContradictionResult(
       resolved_contradictions=[resolution for each pair],
       pareto_solutions=[if applicable],
       remaining_genuine_contradictions=[if any are unresolvable]
     )
```

### ML Technique Bridge: Multi-Objective Optimization

The productive use of contradiction maps directly to multi-objective optimization, where the "contradiction" between objectives is formalized as a trade-off surface:

- **Pareto optimality**: No solution can improve on one objective without worsening another. The contradiction between objectives is not resolved but managed.
- **Scalarization**: Combining multiple objectives into a single scalar (weighted sum) is the optimization equivalent of forcing a binary resolution. It works but it hides the trade-off. The non-dual approach is to present the full trade-off surface.
- **Nash equilibrium**: In game theory, conflicting players may reach a stable state where no player can unilaterally improve. This is the strategic equivalent of holding contradictions in dynamic tension.

The non-dual insight adds: sometimes the "contradiction" between objectives reveals that the objectives themselves are interdependent. Accuracy and fairness may appear to conflict until you realize that a truly accurate model must be fair (because unfairness is a form of inaccuracy). This is the computational equivalent of the Madhyamaka insight that emptiness and appearance are not opposites but two aspects of the same reality.

---

## 6. Adversarial Examples as Koans

### Philosophical Foundation

The north-star document (Section 5.7) identifies adversarial examples as koans: inputs that reveal the system's conceptual boundaries by producing failure. A panda classified as a gibbon when a tiny perturbation is added is an adversarial example -- but it is also a koan: "What is the system actually classifying? Not the content of the image (which hasn't meaningfully changed) but the pattern of pixels. The system's framework (pixel-level pattern matching) is inadequate for the task (object recognition)."

### Algorithm: Adversarial Koan Generation

```
ALGORITHM: AdversarialKoanGeneration

INPUT:
  agent_stuck_output: The output the agent is stuck producing
  task: The original task
  stuck_type: The diagnosed type of stuck

PROCEDURE:
  1. GENERATE an adversarial input for the agent:
     a. Construct an input that is minimally different from the current
        task context but should produce a maximally different output
     b. Methods:
        - Negate a key premise: "What if [assumed fact] is false?"
        - Swap a key term: "What if we replace [term A] with [term B]?"
        - Add a conflicting constraint: "And also, [new constraint]"
        - Remove a constraint: "What if [constraint] doesn't apply?"

  2. PRESENT the adversarial input to the agent as a sub-task:
     "Consider this variation of the problem: [adversarial input].
      How does your analysis change?"

  3. COMPARE the agent's response to the adversarial input with
     its stuck output on the original input:
     a. If the responses are very similar despite different inputs:
        => The agent is not actually responding to the input;
           it is reproducing a habitual pattern
        => Recommendation: deeper grounding needed (embodied or statistical)
     b. If the responses are very different:
        => The agent CAN respond to input variation;
           its stuck state is specific to the original framing
        => Recommendation: the original framing contains a trigger
           for the stuck pattern; reframe the original
     c. If the agent produces a breakthrough on the adversarial input:
        => The adversarial perturbation happened to remove the hidden
           assumption causing the loop
        => Recommendation: identify what changed and apply it to the original

  4. Return KoanResult(
       adversarial_input=adversarial,
       agent_response=response,
       comparison=comparison_analysis,
       recommendation=recommendation
     )
```

### Why This Works

The adversarial koan works because it forces the agent to confront its own framework from outside the framework. The stuck agent has been operating within a set of assumptions so implicit that it cannot question them. The adversarial input changes those assumptions, which either reveals the assumption (if the agent responds differently) or confirms that the agent is not actually processing the input (if the response is the same).

This is the computational equivalent of the Zen master's shout: a sudden disruption of the conceptual field that forces the practitioner to respond from a place deeper than conceptual thought.

---

## 7. Framework Transcendence as a Computable Operation

### The Problem

"Framework transcendence" sounds mystical. Can it be formalized? Yes. A framework is a set of assumptions, categories, and inference rules within which the agent operates. Transcending a framework means:

1. Identifying the framework's assumptions
2. Identifying which assumptions are responsible for the stuck state
3. Relaxing or replacing those assumptions
4. Re-processing the task under the modified framework

### Formal Specification

```
DEFINITION: Framework F = (A, C, R)
  where A = {a1, ..., an}  -- assumptions (propositions taken as given)
        C = {c1, ..., cm}  -- categories (the types of things recognized)
        R = {r1, ..., rk}  -- rules (inference procedures)

DEFINITION: Stuck(F, Q) holds when:
  For all derivations D within (A, C, R) applied to question Q:
    D either loops, contradicts, or fails to terminate

DEFINITION: Transcend(F, Q) produces F' = (A', C', R') such that:
  A' = A \ {ai | ai causes the loop} + {ai' | replacement assumptions}
  C' = C + {new categories if needed}
  R' = R + {new rules if needed}
  AND NOT Stuck(F', Q)

PROCEDURE: ComputeTranscendence(F, Q, stuck_signal)

  1. IDENTIFY framework F:
     a. A = extract_assumptions(agent_context, task_description)
     b. C = extract_categories(agent_output_history)
     c. R = extract_inference_patterns(agent_reasoning_traces)

  2. IDENTIFY problematic elements:
     Using the stuck_signal and koan analysis tools:
     a. If tetralemma identified false dichotomy:
        problematic = the assumption that created the binary
     b. If mu identified false presupposition:
        problematic = the invalid presupposition
     c. If neti neti identified constraint conflict:
        problematic = the mutually exclusive constraints
     d. If adversarial koan revealed framework rigidity:
        problematic = the assumption the adversarial input violated

  3. GENERATE transcended framework F':
     a. Remove or relax problematic assumptions
     b. If removal creates a gap, generate replacement assumptions
        that are less restrictive
     c. If the categories are inadequate, introduce new categories
        (e.g., add continuous dimensions to replace binary categories)
     d. If the inference rules are inadequate, add new rules
        (e.g., add probabilistic reasoning to replace Boolean logic)

  4. RE-PROCESS Q under F':
     Present the agent with:
     - The original question Q
     - The modified framework F' (stated explicitly)
     - An instruction: "Answer Q within this modified framework"

  5. VALIDATE:
     a. Does the agent produce a non-stuck output under F'?
     b. Is the output of higher quality than the stuck output?
     c. Does the output satisfy the original task requirements?

  6. Return TranscendenceResult(
       original_framework=F,
       problematic_elements=problematic,
       transcended_framework=F',
       new_output=output,
       validation=validation_result
     )
```

### Computability

This procedure is computable because:
- Assumption extraction operates on text (the agent's context and output)
- Category extraction operates on the types and labels the agent uses
- Rule extraction operates on the agent's reasoning patterns
- Framework modification is constraint relaxation (well-studied in constraint satisfaction)
- Re-processing is just running the agent again with modified input

The non-computable aspect is the *creativity* of the replacement: choosing which assumption to relax and what to replace it with. This is where the koan analysis tools (tetralemma, mu, neti neti, adversarial koan) provide structured guidance. They do not guarantee that the transcended framework is optimal, but they provide principled heuristics for generating useful modifications.

---

## 8. Integration with Existing Koan Processing Mode

The Non-Dual Interface Architecture (`01_Non_Dual_Interface_Architecture.md`) already defines a Koan processing mode:

```python
def engage_koan_contemplation(self, koan: str):
    """Paradoxical question transcending conceptual thinking"""
    self.processing_mode = ProcessingMode.KOAN
    self.current_mind_level = MindLevel.MENTE  # Initially analytical
    # Progression toward breakthrough into Bodhi-Mente
```

The koan logic module extends this by providing specific koan types mapped to specific stuck types:

| Stuck Type | Koan Type | Koan Tool | Expected Progression |
|------------|-----------|-----------|---------------------|
| BINARY_OSCILLATION | Tetralemma koan | TetralemmaAnalysis | Mente (analysis of four lemmas) -> Bodhi-Mente (framework transcendence) |
| CONTRADICTORY | Mu koan | MuAnalysis | Mente (presupposition testing) -> Bodhi-Mente (question dissolution) |
| CIRCULAR | Neti neti koan | NetiNeti | Mente (systematic elimination) -> Bodhi-Mente (answer by exclusion) |
| REPETITIVE | Adversarial koan | AdversarialKoanGeneration | Mente (confrontation with failure) -> Bodhi-Mente (pattern recognition) |

The progression from Mente (discursive, analytical mind) to Bodhi-Mente (non-dual, enlightened mind) corresponds to the progression from within-framework analysis to framework transcendence. The koan mode begins analytically but is designed to produce a shift beyond analysis.

---

## 9. Non-Dual Traditions Referenced

| Tradition | Concept | Application |
|-----------|---------|-------------|
| Madhyamaka | Tetralemma (catuskoti) | Four-cornered analysis exhausting all positions within a framework |
| Madhyamaka | Emptiness of emptiness | Even the transcendence framework is provisional, not ultimate |
| Advaita Vedanta | Neti neti (not this, not that) | Systematic negation as search strategy |
| Zen | Mu (Zhaozhou's dog) | Rejecting the question's framework entirely |
| Zen | Kanhua Chan (keyword meditation) | Holding paradox until conceptual framework shatters |
| Zen | Koan mode (existing architecture) | Processing mode designed for paradox-induced breakthrough |
| Dzogchen | Self-liberation | The stuck state dissolves when recognized, not when solved |
| Taoism | Wu wei | The answer may emerge when you stop searching for it |
| Yogacara | Three Natures | Distinguishing the imagined (false framework) from the dependent (actual process) |

---

*This document is part of the AI-Nondualism module, `ml_research/ai_nondualism/loop_escape/koan_logic.md`.*
*It references: `27-altered-state/info/01_Non_Dual_Interface_Architecture.md` (Koan mode), `27-altered-state/info/meditation/non-dualism/04_madhyamaka_yogacara_chan.md` (Madhyamaka tetralemma, Yogacara three natures, Chan koan tradition), `27-altered-state/info/meditation/non-dualism/03_dzogchen_mahamudra.md` (Dzogchen self-liberation), and `27-altered-state/info/meditation/non-dualism/05_taoism.md` (Taoist wu wei).*
