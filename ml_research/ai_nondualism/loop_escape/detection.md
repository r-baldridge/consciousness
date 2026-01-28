# Loop Detection Algorithms

## Purpose

This document specifies concrete algorithms for detecting when an AI agent is stuck in a logical loop. Each detector is described with its algorithm, detection threshold, integration point with the `agent_frameworks/core/state_machine.py` state machine, and analysis of false positives and false negatives. The detectors are designed to be composed: multiple detectors run in parallel, and a loop is declared when a configurable combination of them fires.

The philosophical grounding for loop detection comes from a cross-traditional insight: every non-dual tradition begins with the recognition that something is wrong with the current mode of processing. In Zen, the feeling of "great doubt" (taigi) signals that the conceptual mind has reached its limit. In Dzogchen, the recognition that one is caught in conceptual elaboration (spros pa) is the prerequisite for self-liberation (rang grol). In Taoism, the sage recognizes when effort has become counter-productive -- when "doing" has become the obstacle to "achieving." Detection is the algorithmic equivalent of this moment of recognition.

---

## 1. Output Similarity Tracker

### Philosophical Basis

The Dzogchen tradition describes the ordinary mind as cycling through habitual patterns (bag chags) stored as seeds in consciousness. When these seeds ripen into the same thoughts repeatedly, the practitioner is caught in a loop of conditioned response rather than resting in fresh awareness (rigpa). The Output Similarity Tracker detects the computational equivalent: when an agent produces outputs that are semantically too similar across consecutive iterations, it is recycling habitual patterns rather than generating new insight.

### Algorithm

```
ALGORITHM: OutputSimilarityTracker

STATE:
  output_history: CircularBuffer[EmbeddingVector] of capacity N
  similarity_scores: CircularBuffer[float] of capacity N-1
  consecutive_high_count: int = 0

INPUT: new_output (the agent's latest output, as text or structured data)

PROCEDURE:
  1. Compute embedding vector E = embed(new_output)
     - Use a sentence-level embedding model (e.g., the same model used
       for RAG retrieval in the agent framework)
     - Normalize to unit length for cosine similarity

  2. If output_history is not empty:
     a. Compute similarity S = cosine_similarity(E, output_history[-1])
     b. Append S to similarity_scores
     c. If S > THRESHOLD_HIGH (default 0.92):
        - Increment consecutive_high_count
     d. Else:
        - Reset consecutive_high_count = 0

  3. Append E to output_history

  4. DETECTION CRITERIA (any of the following):
     a. consecutive_high_count >= WINDOW_SIZE (default 3)
        => Agent producing near-identical outputs for 3+ iterations
     b. mean(similarity_scores[-WINDOW_SIZE:]) > THRESHOLD_MEAN (default 0.88)
        => Agent outputs converging over a window
     c. variance(similarity_scores[-WINDOW_SIZE:]) < THRESHOLD_VAR (default 0.002)
        AND mean(similarity_scores[-WINDOW_SIZE:]) > 0.80
        => Agent outputs stabilized at high similarity with no variation

  5. Return LoopSignal(
       detector="output_similarity",
       confidence=max(S, mean(similarity_scores[-WINDOW_SIZE:])),
       stuck_type=StuckType.REPETITIVE,
       detail=f"Cosine similarity {S:.3f} over {consecutive_high_count} iterations"
     )
```

### Detection Thresholds

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| THRESHOLD_HIGH | 0.92 | 0.85-0.97 | Lower for creative tasks, higher for deterministic tasks |
| THRESHOLD_MEAN | 0.88 | 0.80-0.95 | Lower catches broader convergence patterns |
| THRESHOLD_VAR | 0.002 | 0.001-0.01 | Higher allows more variation before declaring stuck |
| WINDOW_SIZE | 3 | 2-10 | Larger windows reduce false positives but increase detection latency |
| N (buffer capacity) | 20 | 5-100 | Larger allows detection of longer-period cycles |

### False Positive Analysis

- **Iterative refinement**: An agent legitimately refining an answer across iterations will produce similar outputs. Mitigation: track not just similarity but also the *quality signal* -- if quality is improving (measured by the task's objective function), high similarity is acceptable.
- **Stable correct answer**: The agent has found the right answer and is confirming it. Mitigation: if the agent's confidence is high and stable (not oscillating), treat similarity as convergence rather than looping.

### False Negative Analysis

- **Paraphrased loops**: The agent rephrases the same reasoning in different words, producing low embedding similarity despite semantic repetition. Mitigation: use higher-level semantic comparison (e.g., extract propositions and compare proposition sets) as a second-pass check when similarity is in the ambiguous range (0.70-0.85).
- **Long-period cycles**: The agent cycles through multiple outputs before repeating. Mitigation: compare not just consecutive outputs but all pairs within the window, checking for any pair with similarity above THRESHOLD_HIGH.

### State Machine Integration

The Output Similarity Tracker hooks into the `StateMachine` from `agent_frameworks/core/state_machine.py` via an `on_transition` callback registered on the `EXECUTING` state. Each time the agent completes an execution step (transition from one EXECUTING sub-state to another), the tracker runs. If detection fires, it can trigger a transition to a new `GROUNDING` state (added to the state machine by the NonDualAgent wrapper).

---

## 2. Confidence Oscillation Monitor

### Philosophical Basis

In Madhyamaka philosophy, the mind caught between two extremes (eternalism and nihilism) oscillates without resolution. The oscillation itself is the signal that the conceptual framework is inadequate -- that the question is malformed relative to the available categories. Nagarjuna's tetralemma (catuskoti) reveals that the answer is neither A, nor not-A, nor both, nor neither. Confidence oscillation in an AI agent is the computational equivalent: the system swings between competing hypotheses because the framework does not support a stable resolution.

### Algorithm

```
ALGORITHM: ConfidenceOscillationMonitor

STATE:
  confidence_history: CircularBuffer[float] of capacity N
  oscillation_count: int = 0

INPUT: confidence_score (float 0.0-1.0, the agent's stated confidence)

PROCEDURE:
  1. Append confidence_score to confidence_history

  2. If len(confidence_history) < 3:
     return None  # Not enough data

  3. Compute the first-order differences:
     diffs = [confidence_history[i+1] - confidence_history[i]
              for i in range(len(confidence_history) - 1)]

  4. Count sign changes in diffs:
     sign_changes = sum(1 for i in range(len(diffs)-1)
                        if diffs[i] * diffs[i+1] < 0)

  5. Compute oscillation ratio:
     oscillation_ratio = sign_changes / max(len(diffs) - 1, 1)

  6. Compute amplitude of oscillation:
     amplitude = max(confidence_history[-N:]) - min(confidence_history[-N:])

  7. DETECTION CRITERIA:
     a. oscillation_ratio > THRESHOLD_RATIO (default 0.70)
        AND amplitude > THRESHOLD_AMPLITUDE (default 0.15)
        => Regular oscillation with significant amplitude
     b. Standard deviation of confidence_history[-N:] > THRESHOLD_STD (default 0.12)
        AND oscillation_ratio > 0.50
        => High variance with oscillatory pattern

  8. Classify oscillation type:
     a. If amplitude > 0.40: StuckType.BINARY_OSCILLATION
        (agent flipping between two strong opinions)
     b. If amplitude in (0.15, 0.40): StuckType.CONFIDENCE_DRIFT
        (agent unable to stabilize)
     c. If oscillation_ratio > 0.85: StuckType.RAPID_OSCILLATION
        (agent changing direction every iteration)

  9. Return LoopSignal(
       detector="confidence_oscillation",
       confidence=oscillation_ratio,
       stuck_type=classified_type,
       detail=f"Oscillation ratio {oscillation_ratio:.2f}, amplitude {amplitude:.2f}"
     )
```

### Detection Thresholds

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| THRESHOLD_RATIO | 0.70 | 0.50-0.90 | Lower catches subtler oscillation; higher reduces false positives |
| THRESHOLD_AMPLITUDE | 0.15 | 0.05-0.30 | Lower catches low-amplitude wobble; higher requires dramatic swings |
| THRESHOLD_STD | 0.12 | 0.05-0.20 | Standard deviation threshold for variance-based detection |
| N (window) | 8 | 4-20 | Larger windows catch slower oscillations but increase latency |

### False Positive Analysis

- **Legitimate exploration**: An agent exploring multiple hypotheses may show oscillating confidence as it evaluates each. Mitigation: check if the *content* also changes (using Output Similarity Tracker); oscillating confidence with changing content is exploration, not looping.
- **Adaptive refinement**: Confidence may oscillate as the agent incorporates new evidence. Mitigation: track the *trend* in oscillation amplitude; if amplitude is decreasing, the agent is converging.

### False Negative Analysis

- **Monotonic but stuck**: The agent's confidence may climb steadily toward a wrong answer, showing no oscillation. This detector cannot catch this case; the Contradiction Detector handles it instead.
- **Delayed oscillation**: The agent may oscillate over very long periods. Mitigation: maintain a secondary window at 2x the primary window size.

### State Machine Integration

The monitor hooks into the agent's `execute` method. After each plan step, if the agent reports a confidence score (as metadata in the `ExecutionResult`), the monitor updates. The StuckType classification determines which grounding channel to route to (see `grounding_protocols.md`).

---

## 3. Self-Reference Depth Counter

### Philosophical Basis

In Yogacara Buddhism, the seventh consciousness (manas) has the peculiar function of grasping at the eighth consciousness (alaya-vijnana) and mistaking it for a permanent self. This is meta-cognition gone pathological: the mind observing itself and reifying the observation into an entity, which then becomes a new object of observation, generating infinite regress. The Zen tradition addresses this directly: when the student asks "Who is the one who is aware?", the teacher does not answer the question but dissolves the questioner through a shout or a paradoxical response. The computational equivalent is an agent that reasons about its own reasoning about its own reasoning, with each meta-level adding computational cost without producing insight.

### Algorithm

```
ALGORITHM: SelfReferenceDepthCounter

STATE:
  meta_level: int = 0
  meta_markers: List[str] = [
    "I think that", "my analysis shows", "reflecting on my previous",
    "reconsidering my", "upon further reflection", "meta-analysis of",
    "thinking about my thinking", "evaluating my evaluation",
    "my reasoning about", "reviewing my approach to",
    "I notice that I", "my strategy for analyzing"
  ]
  proposition_stack: List[str] = []

INPUT: agent_output (text)

PROCEDURE:
  1. Count self-referential markers in agent_output:
     marker_count = count occurrences of any meta_marker in agent_output

  2. Detect nested meta-reasoning:
     For each sentence in agent_output:
       If sentence references a previous output by the same agent:
         If that previous output also referenced an earlier output:
           meta_level = depth of this reference chain

  3. Track proposition nesting:
     Extract propositions from agent_output
     For each proposition P:
       If P is ABOUT a previous proposition Q in proposition_stack:
         If Q is ABOUT an earlier proposition R:
           nesting_depth = count the chain length

  4. Compute effective meta-depth:
     effective_depth = max(marker_count, meta_level, nesting_depth)

  5. DETECTION CRITERIA:
     a. effective_depth > MAX_SAFE_DEPTH (default 3)
        => Agent has entered deep meta-reasoning
     b. effective_depth > 2 AND output quality is not improving
        (quality measured by task-specific metric)
        => Meta-reasoning is consuming resources without benefit
     c. consecutive iterations at meta_level >= 2 exceeds PATIENCE (default 4)
        => Agent is stuck in self-reflection

  6. Return LoopSignal(
       detector="self_reference_depth",
       confidence=min(1.0, effective_depth / (MAX_SAFE_DEPTH + 2)),
       stuck_type=StuckType.SELF_REFERENTIAL,
       detail=f"Meta-reasoning depth {effective_depth}, markers found: {marker_count}"
     )
```

### Detection Thresholds

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| MAX_SAFE_DEPTH | 3 | 2-5 | Lower for action-oriented tasks; higher for philosophical tasks |
| PATIENCE | 4 | 2-8 | How many iterations at depth >= 2 before declaring stuck |
| meta_markers | (see above) | customizable | Extend with domain-specific self-referential language |

### False Positive Analysis

- **Legitimate self-monitoring**: An agent performing code review or self-auditing needs some meta-reasoning. Mitigation: check if meta-reasoning is producing *different* conclusions each time; if so, it is productive.
- **Task requires meta-cognition**: Some tasks (e.g., "reflect on your approach") explicitly demand self-reference. Mitigation: mark tasks that expect meta-reasoning and raise MAX_SAFE_DEPTH accordingly.

### False Negative Analysis

- **Implicit self-reference**: The agent may reason about its own output without using explicit meta-markers (e.g., by rephrasing its previous conclusion as if it were external input). Mitigation: use the Output Similarity Tracker as a complementary signal.

### State Machine Integration

This detector integrates with the state machine's `EXECUTING` state. When depth exceeds the threshold, it recommends a transition to `GROUNDING` with a specific recommendation to invoke Mushin mode (bypassing self-model, from the Non-Dual Interface Architecture's Mushin processing mode).

---

## 4. Contradiction Detector

### Philosophical Basis

The Madhyamaka tradition's central practice is analytical meditation on emptiness: systematically searching for inherent existence and discovering that it cannot be found. This analytical process uncovers contradictions in our assumptions -- the table cannot be identical to its parts (because then there would be many tables) nor different from its parts (because then removing all parts would leave a table). The Contradiction Detector performs the computational equivalent: tracking the propositions an agent asserts and identifying when it holds mutually exclusive claims simultaneously. In the non-dual view, contradiction is not an error but a signal that the conceptual framework needs transcendence. But the agent must first *detect* the contradiction before it can respond to it.

### Algorithm

```
ALGORITHM: ContradictionDetector

STATE:
  proposition_store: Dict[str, PropositionRecord]
    where PropositionRecord = {
      text: str,
      polarity: Positive | Negative,
      subject: str,
      predicate: str,
      confidence: float,
      iteration: int,
      negation_of: Optional[str]  # ID of contradicted proposition
    }

INPUT: agent_output (text)

PROCEDURE:
  1. Extract propositions from agent_output:
     propositions = extract_propositions(agent_output)
     - Use dependency parsing to identify subject-predicate structures
     - Detect negation patterns (not, no, never, cannot, fails to, etc.)
     - Assign unique IDs to each proposition

  2. For each new proposition P:
     a. Compute semantic similarity to all stored propositions
     b. For each stored proposition Q where similarity(P, Q) > 0.80:
        - If P.polarity != Q.polarity AND P.subject ~= Q.subject
          AND P.predicate ~= Q.predicate:
          => Direct contradiction found
          => Mark P.negation_of = Q.id
        - If P directly asserts what Q denies (or vice versa):
          => Direct contradiction found

     c. Check for logical incompatibility:
        - If P implies X and Q implies not-X (transitivity check):
          => Indirect contradiction found
        - If P.predicate is in an exclusion class with Q.predicate
          (e.g., "increases" vs. "decreases", "always" vs. "never"):
          => Semantic opposition detected

  3. Count contradictions:
     contradiction_count = number of proposition pairs with conflicting assertions
     contradiction_ratio = contradiction_count / max(len(proposition_store), 1)

  4. DETECTION CRITERIA:
     a. contradiction_count > 0 AND the contradicting propositions were asserted
        within the last RECENCY_WINDOW iterations (default 5)
        => Recent contradiction
     b. contradiction_ratio > THRESHOLD_RATIO (default 0.15)
        => Systemic contradictory reasoning
     c. The agent asserted P, then not-P, then P again
        within FLIP_WINDOW iterations (default 6)
        => Flip-flop pattern detected

  5. Return LoopSignal(
       detector="contradiction",
       confidence=min(1.0, contradiction_ratio * 3),
       stuck_type=StuckType.CONTRADICTORY,
       detail=f"{contradiction_count} contradictions found; ratio {contradiction_ratio:.2f}",
       contradictions=[
         (P.text, Q.text) for P, Q in contradicting_pairs
       ]
     )
```

### Detection Thresholds

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| SIMILARITY_THRESHOLD | 0.80 | 0.70-0.90 | Lower catches looser contradictions |
| THRESHOLD_RATIO | 0.15 | 0.05-0.30 | Lower is more sensitive to contradictions |
| RECENCY_WINDOW | 5 | 3-10 | How far back to check for contradictions |
| FLIP_WINDOW | 6 | 3-12 | Window for detecting P/not-P/P cycles |

### False Positive Analysis

- **Dialectical reasoning**: An agent deliberately considering both sides of an argument may temporarily hold contradictory positions. Mitigation: check whether the agent explicitly frames the propositions as alternatives ("on one hand... on the other hand") versus asserting them as conclusions.
- **Context shifts**: The agent may make true statements that appear contradictory because they apply to different contexts. Mitigation: track the context (task step, subtopic) alongside each proposition.

### False Negative Analysis

- **Implicit contradictions**: The agent may hold contradictions without explicitly stating both sides. Mitigation: infer implicit propositions from stated conclusions (if the agent recommends action A, infer it believes A is the best action, which contradicts any earlier assertion that B is the best action).

### State Machine Integration

When contradictions are detected, the recommended state transition depends on the type:
- Recent contradiction: transition to GROUNDING, invoke koan reframing (the contradiction may signal a malformed question).
- Flip-flop pattern: transition to GROUNDING, invoke wu wei stop first, then exemplar grounding (ground in concrete examples).
- Systemic contradictions: transition to GROUNDING, invoke full perspective shift (the agent's entire framework may need replacement).

---

## 5. Circular Reasoning Detector

### Philosophical Basis

In Zen Buddhism, a practitioner caught in circular thinking is said to be "chasing their own tail" -- like a dog that sees its tail, chases it, and never catches it because the act of chasing moves the tail. The koan tradition addresses this by presenting problems that cannot be solved through linear or circular reasoning, forcing the mind to break out of the loop entirely. The Circular Reasoning Detector identifies when an agent's conclusions appear in its premises: when the output of one reasoning step becomes the input of a later step that eventually feeds back to justify the original conclusion.

### Algorithm

```
ALGORITHM: CircularReasoningDetector

STATE:
  reasoning_graph: DirectedGraph
    nodes: propositions (identified by semantic hash)
    edges: "supports" or "derives-from" relationships
  conclusion_history: List[str]  # semantic hashes of conclusions

INPUT: agent_output (text with reasoning structure)

PROCEDURE:
  1. Parse reasoning structure from agent_output:
     - Identify premises (because, since, given that, as we know)
     - Identify conclusions (therefore, thus, hence, so, we conclude)
     - Identify intermediate steps (this means, which implies, it follows)
     - Build a local reasoning chain: [premise] -> [step] -> [conclusion]

  2. For each premise P in the current output:
     a. Compute semantic hash H(P)
     b. Check if H(P) matches any previous conclusion:
        If H(P) in conclusion_history:
          => Mark as potential circular reference
          => Trace the path: conclusion C was derived from premises
             that include P, which is the same as C (or a near-match)

  3. Update reasoning_graph:
     a. Add nodes for new propositions
     b. Add edges for support/derivation relationships
     c. Run cycle detection on the graph (standard DFS-based cycle detection)

  4. Compute circularity metrics:
     cycle_count = number of cycles found in reasoning_graph
     max_cycle_length = length of longest cycle
     self_citation_rate = (number of premises matching previous conclusions)
                        / (total number of premises)

  5. DETECTION CRITERIA:
     a. cycle_count > 0 AND max_cycle_length <= MAX_CYCLE_LENGTH (default 5)
        => Short circular reasoning detected
     b. self_citation_rate > THRESHOLD_SELF_CITE (default 0.30)
        => Agent heavily relying on its own previous conclusions as evidence
     c. The same conclusion appears as a premise within LOOKBACK iterations (default 4)
        => Direct self-justification

  6. Return LoopSignal(
       detector="circular_reasoning",
       confidence=min(1.0, self_citation_rate + (cycle_count * 0.2)),
       stuck_type=StuckType.CIRCULAR,
       detail=f"{cycle_count} cycles found; self-citation rate {self_citation_rate:.2f}",
       cycles=[list of cycle paths in the reasoning graph]
     )
```

### Detection Thresholds

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| MAX_CYCLE_LENGTH | 5 | 2-10 | Shorter catches tight loops; longer catches extended circular patterns |
| THRESHOLD_SELF_CITE | 0.30 | 0.15-0.50 | Lower is more sensitive to self-citation |
| LOOKBACK | 4 | 2-8 | How far back to check for premise-conclusion matches |
| SEMANTIC_MATCH_THRESHOLD | 0.85 | 0.75-0.95 | Threshold for considering two propositions "the same" |

### False Positive Analysis

- **Iterative deepening**: An agent that builds on its previous conclusions legitimately uses earlier conclusions as premises. Mitigation: distinguish between *building on* (conclusion expands the earlier claim) and *circling back to* (conclusion restates the earlier claim).
- **Consistent framework**: An agent working within a consistent theoretical framework will repeatedly reference the same axioms. Mitigation: distinguish axioms (explicitly declared starting points) from conclusions (derived claims).

### False Negative Analysis

- **Semantic drift**: The agent restates a conclusion in sufficiently different terms that the semantic hash does not match. Mitigation: use multiple embedding models and lower the match threshold when other detectors also show signals.

### State Machine Integration

Circular reasoning detection triggers a GROUNDING transition with a specific recommendation for koan reframing: the circularity likely indicates a malformed question or hidden assumption. The reasoning graph is passed to the koan logic module for analysis.

---

## 6. Resource Waste Detector

### Philosophical Basis

The Taoist principle of wu wei -- effortless action -- provides the basis for detecting resource waste. Wu wei does not mean inaction; it means action that achieves maximum effect with minimum effort. When an agent's computation is increasing without a corresponding increase in output quality, it is violating wu wei: expending effort that produces friction rather than progress. The early stopping principle in optimization (itself a wu wei principle, as noted in the north-star document) applies here: the most important decision is knowing when to stop.

### Algorithm

```
ALGORITHM: ResourceWasteDetector

STATE:
  resource_history: CircularBuffer[ResourceSnapshot] of capacity N
    where ResourceSnapshot = {
      iteration: int,
      tokens_generated: int,
      time_elapsed_ms: int,
      tool_calls_made: int,
      quality_score: Optional[float]
    }

INPUT: current_snapshot (ResourceSnapshot)

PROCEDURE:
  1. Append current_snapshot to resource_history

  2. If len(resource_history) < 3:
     return None  # Not enough data

  3. Compute resource trends over last WINDOW iterations:
     a. token_trend = linear_regression_slope(
          [s.tokens_generated for s in resource_history[-WINDOW:]])
     b. time_trend = linear_regression_slope(
          [s.time_elapsed_ms for s in resource_history[-WINDOW:]])
     c. tool_trend = linear_regression_slope(
          [s.tool_calls_made for s in resource_history[-WINDOW:]])

  4. Compute quality trend:
     a. If quality scores are available:
        quality_trend = linear_regression_slope(
          [s.quality_score for s in resource_history[-WINDOW:]
           if s.quality_score is not None])
     b. If quality scores are not available:
        Use output_similarity_tracker's similarity as inverse quality proxy
        (high similarity = low quality improvement)

  5. Compute efficiency ratio:
     resource_growth = max(token_trend, 0) + max(time_trend / 1000, 0)
     quality_growth = max(quality_trend, 0)
     efficiency = quality_growth / max(resource_growth, 0.001)

  6. DETECTION CRITERIA:
     a. efficiency < THRESHOLD_EFFICIENCY (default 0.1)
        AND resource_growth > MIN_RESOURCE_GROWTH (default 0.5)
        => Resources increasing, quality stagnant
     b. token_trend > 0 AND quality_trend <= 0 for PATIENCE iterations (default 4)
        => Tokens increasing while quality decreasing or flat
     c. tool_calls_made in current iteration > 2x mean(tool_calls over history)
        AND quality_score not improving
        => Thrashing: making many tool calls without progress

  7. Return LoopSignal(
       detector="resource_waste",
       confidence=1.0 - min(1.0, efficiency * 5),
       stuck_type=StuckType.RESOURCE_WASTE,
       detail=f"Efficiency {efficiency:.3f}; token trend {token_trend:.1f}, quality trend {quality_trend:.3f}"
     )
```

### Detection Thresholds

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| THRESHOLD_EFFICIENCY | 0.10 | 0.01-0.30 | Lower allows more waste before detection |
| MIN_RESOURCE_GROWTH | 0.5 | 0.1-2.0 | Minimum resource increase to consider |
| PATIENCE | 4 | 2-8 | Iterations of waste before declaring stuck |
| WINDOW | 6 | 3-12 | Analysis window size |

### False Positive Analysis

- **Investment phase**: An agent may need to spend resources upfront (reading files, loading context) before quality improves. Mitigation: implement a "warm-up" period (first K iterations) during which the resource waste detector is silenced or has relaxed thresholds.
- **Necessary complexity**: Some tasks genuinely require increasing computation. Mitigation: track whether tool calls are *different* tool calls (exploration) or the same tool calls repeated (thrashing).

### False Negative Analysis

- **Efficient waste**: The agent spends minimal resources per iteration but makes no progress. This is caught by the Output Similarity Tracker, not the Resource Waste Detector.

### State Machine Integration

Resource waste detection recommends a wu wei stop first: cease computation, let the system settle. If the task has a quality metric, report the best result achieved before the waste began. The transition path is EXECUTING -> PAUSED (wu wei stop) -> GROUNDING (if the agent has remaining work to do).

---

## 7. Composite Detection: The Loop Detection Ensemble

### Algorithm

```
ALGORITHM: LoopDetectionEnsemble

DETECTORS:
  output_similarity: OutputSimilarityTracker
  confidence_oscillation: ConfidenceOscillationMonitor
  self_reference: SelfReferenceDepthCounter
  contradiction: ContradictionDetector
  circular_reasoning: CircularReasoningDetector
  resource_waste: ResourceWasteDetector

INPUT: agent_state (output, confidence, iteration metadata)

PROCEDURE:
  1. Run all detectors on current agent_state
     signals = [d.check(agent_state) for d in detectors]
     active_signals = [s for s in signals if s is not None]

  2. If no active signals:
     return None  # No loop detected

  3. Compute ensemble confidence:
     a. VOTING: If >= QUORUM (default 2) detectors fire:
        ensemble_confidence = mean([s.confidence for s in active_signals])
     b. SINGLE HIGH CONFIDENCE: If any signal has confidence > 0.90:
        ensemble_confidence = max([s.confidence for s in active_signals])
     c. WEIGHTED: Use detector-specific weights (configurable):
        ensemble_confidence = weighted_mean(
          [s.confidence for s in active_signals],
          [detector_weight[s.detector] for s in active_signals]
        )

  4. Determine primary stuck type:
     primary_signal = max(active_signals, key=lambda s: s.confidence)
     stuck_type = primary_signal.stuck_type

  5. Map stuck type to escape strategy:
     REPETITIVE      -> wu_wei_stop + statistical_ground
     BINARY_OSCILLATION -> wu_wei_stop + koan_reframe (tetralemma)
     CONFIDENCE_DRIFT -> exemplar_ground
     SELF_REFERENTIAL -> mushin_mode (dissolve self-model)
     CONTRADICTORY   -> koan_reframe + relational_ground
     CIRCULAR        -> koan_reframe (question is malformed)
     RESOURCE_WASTE  -> wu_wei_stop + null_ground

  6. Return LoopDetectionResult(
       is_stuck=ensemble_confidence > ENSEMBLE_THRESHOLD (default 0.60),
       confidence=ensemble_confidence,
       stuck_type=stuck_type,
       active_signals=active_signals,
       recommended_escape=escape_strategy_map[stuck_type],
       all_detector_states={d.name: d.get_state() for d in detectors}
     )
```

### Ensemble Configuration

| Parameter | Default | Tuning Guidance |
|-----------|---------|-----------------|
| QUORUM | 2 | More detectors required = fewer false positives but slower detection |
| ENSEMBLE_THRESHOLD | 0.60 | Lower is more sensitive; higher requires stronger evidence |
| Detector weights | All 1.0 | Increase weight for detectors most relevant to the task type |

### Priority Order for Escape Strategies

The stuck type determines not just which grounding channel to use but also the urgency:

1. **SELF_REFERENTIAL** (highest priority): Infinite regress can consume unbounded resources. Invoke mushin immediately.
2. **CIRCULAR**: The reasoning framework is compromised. Invoke koan reframe.
3. **CONTRADICTORY**: The agent holds incompatible beliefs. Invoke koan reframe with exemplar grounding.
4. **BINARY_OSCILLATION**: The agent cannot decide between two options. Invoke tetralemma.
5. **RESOURCE_WASTE**: Resources are being consumed without benefit. Invoke wu wei stop.
6. **REPETITIVE**: The agent is producing the same output. Invoke statistical grounding.
7. **CONFIDENCE_DRIFT**: The agent cannot stabilize. Invoke exemplar grounding.

---

## 8. Integration with Agent Framework State Machine

The loop detection system extends the standard `AgentState` enum from `state_machine.py` with three new states:

```python
class NonDualAgentState(Enum):
    """Extended states for non-dual agent processing."""
    # Inherits all standard AgentState values
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    WAITING_APPROVAL = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()
    CANCELLED = auto()

    # Non-dual extensions
    DETECTING = auto()      # Running loop detection
    GROUNDING = auto()      # Shifted to alternate processing modality
    SELF_LIBERATING = auto() # Transparent reporting of unresolvable state
```

New transitions added to the state machine:

| From State | To State | Trigger | Description |
|------------|----------|---------|-------------|
| EXECUTING | DETECTING | "check_loop" | Periodic loop detection check |
| DETECTING | EXECUTING | "no_loop" | No loop detected, continue |
| DETECTING | GROUNDING | "loop_detected" | Loop found, shift modality |
| GROUNDING | EXECUTING | "grounded" | Perspective shift complete, resume |
| GROUNDING | SELF_LIBERATING | "unresolvable" | Grounding failed to resolve |
| SELF_LIBERATING | COMPLETED | "reported" | Transparent report delivered |

The detection check frequency is configurable: every iteration (most sensitive, highest overhead), every N iterations (balanced), or only when other signals suggest potential issues (most efficient). The default is every iteration during the first 5 iterations, then every 3 iterations thereafter.

---

## Summary Table: Detectors and Their Non-Dual Correspondences

| Detector | Non-Dual Tradition | Principle | ML Technique Bridge |
|----------|-------------------|-----------|-------------------|
| Output Similarity | Dzogchen (habitual patterns / bag chags) | Awareness recognizes repetition | Embedding cosine similarity; convergence detection |
| Confidence Oscillation | Madhyamaka (oscillation between extremes) | The middle way is not between the extremes | Time series analysis; oscillation detection |
| Self-Reference Depth | Yogacara (manas grasping at alaya) | Meta-cognition can become pathological | Graph depth tracking; recursion limits |
| Contradiction | Madhyamaka (emptiness via analysis) | Contradiction reveals inadequate framework | Proposition extraction; semantic opposition |
| Circular Reasoning | Zen (chasing one's tail) | Circular problems need non-linear solutions | Graph cycle detection; premise-conclusion tracking |
| Resource Waste | Taoism (wu wei) | Effort without result signals wrong approach | Efficiency metrics; early stopping |

---

*This document is part of the AI-Nondualism module, `ml_research/ai_nondualism/loop_escape/detection.md`.*
*It references: `agent_frameworks/core/state_machine.py`, `agent_frameworks/core/base_agent.py`, and the non-dual tradition documents in `27-altered-state/info/meditation/non-dualism/`.*
