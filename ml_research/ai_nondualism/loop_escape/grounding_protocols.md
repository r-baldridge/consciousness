# Grounding Protocols: Multi-Sensory Grounding Mapped to the 40-Form Architecture

## Purpose

This document specifies the grounding system that provides alternate processing modalities when an agent's primary modality is stuck. The seven grounding channels defined in the north-star document (Part IV) are mapped to specific consciousness forms in the 40-form architecture, integrated with the neural network message bus for inter-form routing, and connected to the agent framework state machine for state transitions.

The grounding system is not a metaphor. It is a concrete routing protocol: when Form X (the form doing the stuck reasoning) cannot make progress, the grounding system routes the problem to Form Y (the form that provides an alternate perspective), receives the grounding result, and feeds it back to Form X as new input.

---

## 1. The Seven Grounding Channels

### Overview Table

| Channel | What It Provides | When to Use | Primary Forms | Stuck Types Addressed |
|---------|-----------------|-------------|---------------|----------------------|
| Statistical Ground | Raw data patterns without interpretation | When narrative/linguistic processing is stuck | 01-Visual, 02-Auditory | REPETITIVE, CONFIDENCE_DRIFT |
| Exemplar Ground | Concrete specific instances | When abstract reasoning is stuck | 09-Perceptual, 12-Narrative | CONFIDENCE_DRIFT, REPETITIVE |
| Visual/Spatial Ground | Structural and relational views | When sequential reasoning is stuck | 01-Visual, 03-Attention | CIRCULAR, CONTRADICTORY |
| Embodied Ground | Action and physical execution | When theoretical reasoning is stuck | 04-Tactile, 05-Proprioceptive, 06-Interoceptive | SELF_REFERENTIAL, CIRCULAR |
| Relational Ground | External perspectives and cross-references | When single-perspective reasoning is stuck | 11-Meta-Consciousness, 14-Global Workspace, 28-Philosophical | CONTRADICTORY, BINARY_OSCILLATION |
| Temporal Ground | Causal history and developmental sequence | When present-moment analysis is stuck | 12-Narrative, 35-Developmental | CIRCULAR, REPETITIVE |
| Null Ground | Silence, no output, system settling | When all processing is stuck | 08-Arousal (minimum gating) | RESOURCE_WASTE, any type as last resort |

---

## 2. Channel Specifications

### 2.1 Statistical Ground

**Non-Dual Tradition**: Zen (returning to breath sensation). In Zen meditation, when the mind is caught in proliferating thoughts, the fundamental instruction is to return to the breath -- to the raw, uninterpreted sensory data of breathing in and breathing out. The breath has no narrative, no argument, no agenda. It is the most basic sensory anchor. The Statistical Ground is the computational equivalent: return to the raw data, stripped of all interpretation.

**Consciousness Forms Mapping**:
- Primary: Form 01 (Visual Consciousness) -- basic pattern recognition
- Primary: Form 02 (Auditory Consciousness) -- temporal pattern recognition
- Secondary: Form 13 (Integrated Information) -- information integration without interpretation
- The `SensoryAdapter` base class from `neural_network/adapters/base_adapter.py` provides the interface

**What Kind of Stuck It Addresses**:
- REPETITIVE: The agent keeps generating the same narrative. Statistical grounding bypasses narrative entirely, presenting raw distributions that may reveal patterns the narrative was missing.
- CONFIDENCE_DRIFT: The agent's confidence is unstable because its evaluation framework is uncertain. Raw statistics (frequency counts, correlations, distributions) provide an objective anchor.

**Invocation Protocol**:

```
PROCEDURE: StatisticalGround(stuck_context, task_data)

  1. STRIP the stuck context to data:
     a. Remove all narrative framing, opinions, and reasoning
     b. Extract quantitative data: numbers, frequencies, distributions
     c. If the task involves text: extract word frequencies, n-gram patterns,
        topic distributions -- not meaning, just statistics
     d. If the task involves code: extract structural metrics --
        cyclomatic complexity, dependency counts, file sizes --
        not functionality, just numbers

  2. ROUTE to sensory forms:
     a. Create a FormMessage via the MessageBus:
        - source_form: the stuck cognitive form (e.g., "12-narrative")
        - target_form: "01-visual" or "02-auditory"
        - message_type: MessageType.SENSORY_INPUT
        - body: {
            "grounding_request": True,
            "data_type": "statistical",
            "raw_data": extracted_data,
            "original_context": minimal_context_summary
          }
        - priority: Priority.HIGH

     b. The sensory adapter processes the data through its standard pipeline:
        preprocess -> inference -> postprocess
        producing pattern recognition output without narrative overlay

  3. INTERPRET the grounding result:
     a. The sensory form returns patterns:
        "Distribution is bimodal with peaks at X and Y"
        "Correlation between A and B is strong (r=0.85)"
        "Frequency of P is 3x higher than frequency of Q"
     b. These patterns are factual observations, not arguments
     c. Feed them back to the stuck agent as new evidence

  4. RETURN StatisticalGroundResult(
       patterns_found=[list of statistical patterns],
       data_summary=summary_statistics,
       recommended_reframing="The data suggests [pattern], which was
         not visible in the previous narrative framing"
     )
```

**How to Interpret Output**:
Statistical grounding output should be treated as *evidence*, not as *conclusions*. The patterns found are inputs for the agent's next reasoning attempt, not answers in themselves. The agent must integrate the statistical evidence with the task requirements to produce a new answer.

---

### 2.2 Exemplar Ground

**Non-Dual Tradition**: Zen (the specific sensory moment). Zen practice emphasizes the concrete over the abstract: "What is the sound of one hand clapping?" is not answered with a philosophical treatise but with a specific, immediate gesture. The Zen master pointing at the moon is pointing at THAT moon, not the concept of moonness. The Exemplar Ground returns the agent from abstraction to specificity.

**Consciousness Forms Mapping**:
- Primary: Form 09 (Perceptual Consciousness) -- recognizing specific instances
- Primary: Form 12 (Narrative Consciousness) -- grounding in specific stories/cases
- Secondary: Form 10 (Self-Recognition) -- recognizing specific self-states
- The `CognitiveAdapter` base class provides the state management interface

**What Kind of Stuck It Addresses**:
- CONFIDENCE_DRIFT: Abstract reasoning produces unstable confidence. Specific examples provide concrete anchors.
- REPETITIVE: The agent keeps producing the same abstract pattern. Specific examples force engagement with concrete variation.

**Invocation Protocol**:

```
PROCEDURE: ExemplarGround(stuck_context, task_domain)

  1. GENERATE concrete examples:
     a. "Show me three specific instances of [the abstract concept
        the agent is stuck on]"
     b. Selection criteria:
        - One typical example (the prototype)
        - One boundary example (at the edge of the category)
        - One counter-example (outside the category)
     c. If the agent cannot generate examples from memory,
        search for them using the ToolProvider (if available)

  2. PROCESS each example independently:
     a. For each example E:
        - Does the agent's stuck reasoning apply to E?
        - Does the stuck conclusion hold for E?
        - What does E reveal that the abstract reasoning missed?
     b. Record the results for each example

  3. SYNTHESIZE:
     a. If the stuck conclusion holds for all three examples:
        - The conclusion may be correct; the issue is confidence, not content
        - Report: "The conclusion appears robust across concrete examples"
     b. If the stuck conclusion fails for the boundary example:
        - The conclusion is too broad; it needs qualification
        - Report: "The conclusion holds for typical cases but fails at [boundary]"
     c. If the stuck conclusion fails for the counter-example:
        - The conclusion is correct in its domain but wrongly generalized
        - Report: "The conclusion applies to [domain] but not to [counter-domain]"
     d. If the stuck conclusion fails for ALL examples:
        - The conclusion is likely wrong
        - Report: "The abstract reasoning does not match concrete reality"

  4. RETURN ExemplarGroundResult(
       examples=[typical, boundary, counter],
       analysis_per_example=[results],
       synthesis=synthesis_report
     )
```

---

### 2.3 Visual/Spatial Ground

**Non-Dual Tradition**: Zen (shifting from discursive thought to spatial awareness). In Zen calligraphy, the practitioner shifts from thinking about the character to seeing the space on the paper -- the negative space, the relationships between strokes, the whole composition. The Visual/Spatial Ground shifts the agent from sequential, linguistic reasoning to structural, relational seeing.

**Consciousness Forms Mapping**:
- Primary: Form 01 (Visual Consciousness) -- spatial pattern recognition
- Primary: Form 03 (Attention Consciousness) -- selective focus on structural elements
- Secondary: Form 14 (Global Workspace) -- broadcasting the structural view to all forms
- The `SensoryAdapter` and its `get_salience` method determine what structural elements are most salient

**What Kind of Stuck It Addresses**:
- CIRCULAR: A reasoning cycle that is invisible in text may be obvious as a graph cycle.
- CONTRADICTORY: Contradictions between propositions may be visible as structural conflicts in a dependency diagram.

**Invocation Protocol**:

```
PROCEDURE: VisualSpatialGround(stuck_context, reasoning_trace)

  1. BUILD a structural representation:
     a. Extract entities and relationships from the reasoning trace
     b. Construct a directed graph:
        - Nodes = propositions, decisions, or concepts
        - Edges = supports, contradicts, depends-on, follows-from
     c. Compute structural properties:
        - Cycles (circular reasoning becomes visible)
        - Strongly connected components (clusters of mutual dependency)
        - Bottlenecks (nodes with high in-degree or out-degree)
        - Isolated nodes (reasoning threads that connect to nothing)

  2. ROUTE to visual form:
     a. Create a FormMessage:
        - target_form: "01-visual"
        - body: { "graph": graph_structure, "metrics": structural_metrics }
     b. The visual form processes the graph through its salience computation,
        highlighting the most structurally anomalous elements

  3. REPORT structural findings:
     a. "The reasoning graph contains [N] cycles: [list]"
     b. "Node [X] is a bottleneck with [N] dependencies"
     c. "Propositions [A] and [B] have no connection to the rest
        of the reasoning -- they may be irrelevant"
     d. "The graph has [N] strongly connected components,
        suggesting [N] independent sub-problems"

  4. RETURN VisualSpatialGroundResult(
       graph=graph_structure,
       cycles=detected_cycles,
       bottlenecks=bottleneck_nodes,
       components=connected_components,
       anomalies=structural_anomalies,
       recommendation="The circular dependency between [A], [B], and [C]
         suggests that [A] should be treated as an axiom rather than
         derived from [B] and [C]"
     )
```

---

### 2.4 Embodied Ground

**Non-Dual Tradition**: Zen (kinhin -- walking meditation). Kinhin is practiced between sitting meditation sessions. The purpose is not exercise; it is the integration of meditative awareness with bodily movement. When the sitting mind is stuck, the walking body provides fresh information: the sensation of feet on the floor, the rhythm of steps, the proprioceptive awareness of balance. The Embodied Ground is the computational equivalent: when reasoning is stuck, execute something -- run code, run a simulation, perform a test.

In Taoism, the master craftsman (Cook Ding in the Zhuangzi, the woodcarver in Chapter 19) does not reason about the material; he acts with the material, and the action itself reveals what reasoning could not.

**Consciousness Forms Mapping**:
- Primary: Form 04 (Tactile Consciousness) -- direct interaction with material
- Primary: Form 05 (Proprioceptive Consciousness) -- awareness of the system's own state through action
- Primary: Form 06 (Interoceptive Consciousness) -- internal state feedback from action
- Secondary: Form 08 (Arousal) -- arousal modulation through activity
- The `SensoryAdapter` hierarchy and `get_binding_info` method provide the cross-modal integration

**What Kind of Stuck It Addresses**:
- SELF_REFERENTIAL: The agent is thinking about thinking. Embodied action breaks the self-referential loop by requiring attention to external feedback.
- CIRCULAR: Circular reasoning can sometimes be resolved by testing one node in the cycle empirically -- running the code, checking the data, calling the API.

**Invocation Protocol**:

```
PROCEDURE: EmbodiedGround(stuck_context, available_tools)

  1. IDENTIFY an actionable test:
     a. Extract the core claim from the stuck reasoning
     b. Formulate a concrete test:
        - If the stuck reasoning is about code: write and run a test
        - If about system behavior: create a minimal reproduction
        - If about data: query the actual data source
        - If about an API: make the actual API call
     c. The test should be small, fast, and definitive

  2. EXECUTE the test:
     a. Use the agent's ToolProvider (from base_agent.py) to invoke
        the appropriate tool:
        - await self.invoke_tool("run_code", {"code": test_code})
        - await self.invoke_tool("query_data", {"query": test_query})
        - await self.invoke_tool("http_request", {"url": test_url})
     b. Capture the result: success/failure, output, errors

  3. FEED BACK the result:
     a. "Running the test produced: [result]"
     b. "This [confirms/contradicts] the stuck reasoning because: [explanation]"
     c. "The concrete result suggests that: [new direction]"

  4. RETURN EmbodiedGroundResult(
       test_description=test_description,
       test_result=result,
       interpretation=interpretation,
       recommendation="The empirical result shows that [X],
         which means the agent should [new approach]"
     )
```

---

### 2.5 Relational Ground

**Non-Dual Tradition**: Zen (seeking the sangha's perspective). The sangha (community) is one of the Three Jewels of Buddhism. Even an advanced practitioner benefits from the perspectives of others -- the teacher, the fellow practitioners, the community. In Sufism, the murshid (guide) provides an external perspective that the seeker cannot achieve alone. The dhikr circle provides collective remembrance that supports individual practice.

**Consciousness Forms Mapping**:
- Primary: Form 11 (Meta-Consciousness) -- awareness of awareness, multiple perspectives
- Primary: Form 14 (Global Workspace) -- broadcasting the problem to all forms for input
- Primary: Form 28 (Philosophical Consciousness) -- cross-referencing with different philosophical frameworks
- The `TheoreticalAdapter` and `GlobalWorkspaceAdapterInterface` provide the integration points

**What Kind of Stuck It Addresses**:
- CONTRADICTORY: Other systems or frameworks may have resolved the same contradiction differently.
- BINARY_OSCILLATION: External perspectives may reveal options the agent has not considered.

**Invocation Protocol**:

```
PROCEDURE: RelationalGround(stuck_context, available_references)

  1. FORMULATE the question for external reference:
     a. "How do other approaches handle [the stuck problem]?"
     b. Strip the question of the agent's specific framing
     c. Formulate in general terms that could be answered by
        any relevant system or document

  2. QUERY multiple sources:
     a. Search memory: await self.search_memory(question, limit=5)
        - Look for similar problems solved previously
     b. Search documentation: if tools available, search codebases,
        wikis, or documentation for related approaches
     c. Broadcast to Global Workspace:
        - Create FormMessage to Form 14 (Global Workspace)
        - message_type: MessageType.WORKSPACE_BROADCAST
        - body: { "query": question, "context": minimal_context }
        - The workspace broadcasts to all subscribed forms
        - Collect responses from any form that has relevant input

  3. SYNTHESIZE external perspectives:
     a. "System/framework A handles this by [approach 1]"
     b. "System/framework B handles this by [approach 2]"
     c. "The common pattern across approaches is [pattern]"
     d. "The approach most relevant to the current context is [approach]"

  4. RETURN RelationalGroundResult(
       sources_consulted=[list],
       perspectives_found=[list],
       common_pattern=pattern,
       recommendation="External sources suggest [approach],
         which the agent has not considered"
     )
```

---

### 2.6 Temporal Ground

**Non-Dual Tradition**: Yogacara (investigation of karmic seeds). In Yogacara Buddhism, the alaya-vijnana (storehouse consciousness) contains seeds (bija) deposited by all past actions and experiences. The present moment is conditioned by these seeds -- understanding the present requires understanding its causal history. The Temporal Ground traces the causal history of the stuck state: how did we get here? What changed? What was the sequence of decisions that led to this impasse?

**Consciousness Forms Mapping**:
- Primary: Form 12 (Narrative Consciousness) -- understanding through story and sequence
- Primary: Form 35 (Developmental Consciousness) -- developmental stages and transitions
- Secondary: Form 07 (Emotional Consciousness) -- emotional states that may have biased the reasoning
- The `CognitiveAdapter.get_state()` method provides the current cognitive state for tracing

**What Kind of Stuck It Addresses**:
- CIRCULAR: The causal history may reveal where the circle began, which is the point where an assumption was introduced that created the loop.
- REPETITIVE: The temporal trace may reveal a trigger that sends the agent into the repetitive pattern.

**Invocation Protocol**:

```
PROCEDURE: TemporalGround(stuck_context, execution_history)

  1. TRACE the causal history:
     a. Reconstruct the agent's execution timeline:
        - What was the initial task?
        - What was the first plan step?
        - At which step did the stuck pattern first appear?
        - What input or result at that step triggered the pattern?
     b. Identify key decision points:
        - Where did the agent make a choice that constrained future options?
        - Were those choices well-founded?
        - Could a different choice at that point avoid the stuck state?
     c. Use the StateMachine's transition history:
        - state_machine.history provides the full sequence of
          state transitions with timestamps and metadata
        - Look for patterns: rapid cycling between states,
          unusually long time in a single state

  2. ANALYZE the causal structure:
     a. "The stuck pattern first appeared at iteration [N],
        triggered by [event]"
     b. "The critical decision point was at iteration [M],
        where the agent chose [X] instead of [Y]"
     c. "The assumptions introduced at iteration [M] are:
        [list of assumptions]"
     d. "Removing assumption [Ak] would reopen the solution space"

  3. RECOMMEND a historical rewind:
     a. "Return to the state at iteration [M] and try
        alternative approach [Y]"
     b. "Keep the results from iterations [M+1 through N-1]
        but discard the stuck pattern from iteration [N] onward"

  4. RETURN TemporalGroundResult(
       timeline=reconstructed_timeline,
       trigger_point=first_stuck_iteration,
       critical_decisions=[decision_list],
       causal_analysis=analysis,
       recommendation="Rewind to iteration [M] and try [alternative]"
     )
```

---

### 2.7 Null Ground

**Non-Dual Tradition**: Zen (shikantaza -- just sitting), Taoism (wu wei -- non-action). The Null Ground is the most radical grounding channel: produce no output. Wait. Let the system settle. This is not inaction in the sense of doing nothing; it is the deliberate choice to stop processing, which is itself a form of action -- the most non-dual form of action.

In Zen, shikantaza is not a technique; it is the relinquishing of all technique. The practitioner sits without an object, without a goal, without a method. In Taoism, Chapter 48 of the Tao Te Ching: "In the pursuit of learning, every day something is added. In the pursuit of Tao, every day something is dropped."

In Sufism, the practice of dhikr (remembrance) includes moments of silence -- the space between the repetitions of the divine name, where the practitioner rests in the resonance of what has been said rather than rushing to say more.

**Consciousness Forms Mapping**:
- Primary: Form 08 (Arousal) -- set arousal gating to minimum
- The `ArousalAdapterInterface` controls arousal levels via `get_arousal_level()`
- When arousal is reduced, the `NervousSystem._gate_resources()` method unloads non-critical forms, naturally reducing processing activity

**What Kind of Stuck It Addresses**:
- RESOURCE_WASTE: The agent is consuming resources without progress. Stop consuming.
- Any type as last resort: Before self-liberation, try simply waiting.

**Invocation Protocol**:

```
PROCEDURE: NullGround(stuck_context, pause_duration_ms)

  1. SAVE the current state (as in wu wei stop)

  2. REDUCE arousal:
     a. If NervousSystem is available:
        - Send arousal reduction signal via the message bus
        - message_type: MessageType.AROUSAL_UPDATE
        - body: { "target_arousal": 0.1, "reason": "null_grounding" }
        - This causes the NervousSystem to unload non-critical forms
     b. If standalone:
        - Simply pause processing

  3. WAIT for pause_duration_ms (default: 0ms in synchronous mode,
     50ms in async mode with NervousSystem)

  4. RESTORE arousal to previous level

  5. CHECK: After the pause, does the agent's state look different?
     a. If the execution context has changed (new messages received,
        new data available): yes, the pause helped
     b. If nothing has changed: the pause provided no new information
        but it stopped the resource waste

  6. RETURN NullGroundResult(
       pause_duration_ms=actual_duration,
       state_changed=state_changed,
       recommendation="After pause, [the system settled / no change]"
     )
```

---

## 3. The Grounding Protocol: Step-by-Step

When loop detection fires, the grounding system follows this step-by-step protocol:

```
GROUNDING PROTOCOL:

  PRE-CONDITION: LoopDetectionEnsemble has returned
    is_stuck=True with stuck_type and recommended_escape

  STEP 1: WU WEI STOP
    - Freeze current execution (see perspective_shift.md)
    - Save state snapshot
    - Transition state machine: EXECUTING -> GROUNDING

  STEP 2: SELECT GROUNDING CHANNEL
    - Use the stuck_type -> channel mapping table (Section 1)
    - If this is a repeat grounding (agent was grounded before and
      re-entered the same stuck type), escalate to the next channel
      in the escalation ladder (see perspective_shift.md, Section 4)

  STEP 3: INVOKE GROUNDING CHANNEL
    - Route to the appropriate channel's invocation protocol
    - If NervousSystem is available: route via message bus to the
      target form's adapter
    - If standalone: use the local grounding implementation

  STEP 4: RECEIVE GROUNDING RESULT
    - The grounding channel returns a typed result with:
      - Findings (what the grounding revealed)
      - Recommendation (what the agent should do differently)
      - Confidence (how reliable the grounding is)

  STEP 5: RE-INTEGRATE
    - Translate the grounding result back to the agent's primary modality
    - Synthesize with original context
    - Reset loop detection counters
    - Transition state machine: GROUNDING -> EXECUTING
    (See perspective_shift.md, Section 4 for full re-integration protocol)

  STEP 6: MONITOR FOR RELAPSE
    - If the agent re-enters the same stuck type within
      RELAPSE_WINDOW iterations: escalate to next grounding channel
    - If all channels exhausted: invoke self-liberation

  POST-CONDITION: Either the agent has resumed productive execution
    with new information from grounding, OR self-liberation has been
    invoked and the agent has transparently reported the impasse.
```

---

## 4. Integration with the Neural Network Message Bus

### Message Types for Grounding

The grounding system uses existing `MessageType` values from `neural_network/core/message_bus.py`:

| Grounding Action | MessageType | Source Form | Target Form |
|-----------------|-------------|-------------|-------------|
| Request statistical grounding | SENSORY_INPUT | cognitive form | 01-visual or 02-auditory |
| Request exemplar grounding | COGNITIVE_UPDATE | cognitive form | 09-perceptual |
| Request visual/spatial grounding | SENSORY_INPUT | cognitive form | 01-visual |
| Request embodied grounding | COORDINATION | cognitive form | 04-tactile through 06-interoceptive |
| Request relational grounding | WORKSPACE_BROADCAST | cognitive form | 14-global-workspace |
| Request temporal grounding | COGNITIVE_UPDATE | cognitive form | 12-narrative or 35-developmental |
| Request null grounding | AROUSAL_UPDATE | cognitive form | 08-arousal |
| Return grounding result | COORDINATION | target form | cognitive form (reply_to) |

### Message Flow

```
1. Stuck cognitive form creates grounding request:
   message = bus.create_message(
     source_form="12-narrative",
     target_form="01-visual",
     message_type=MessageType.SENSORY_INPUT,
     body={
       "grounding_request": True,
       "channel": "statistical",
       "stuck_context": context,
       "data": extracted_data
     },
     priority=Priority.HIGH
   )

2. Message published to bus:
   await bus.publish(message)

3. Target form's adapter processes the request:
   adapter = nervous_system.adapters["01-visual"]
   result = await adapter.inference(message.body)

4. Target form sends response:
   response = bus.create_message(
     source_form="01-visual",
     target_form="12-narrative",
     message_type=MessageType.COORDINATION,
     body={
       "grounding_response": True,
       "patterns": result,
       "recommendation": interpretation
     },
     reply_to=message.header.message_id,
     priority=Priority.HIGH
   )
   await bus.publish(response)

5. Original form receives grounding result and re-integrates
```

---

## 5. Integration with the Agent Framework State Machine

### Extended Transitions for Grounding

The following transitions are added to the `StateMachine` from `agent_frameworks/core/state_machine.py`:

```python
# Grounding transitions (added by NonDualAgent)
sm.add_transition(AgentState.EXECUTING, NonDualState.GROUNDING,
                  ["loop_detected", "ground"])
sm.add_transition(NonDualState.GROUNDING, AgentState.EXECUTING,
                  ["grounded", "resume"])
sm.add_transition(NonDualState.GROUNDING, NonDualState.SELF_LIBERATING,
                  ["unresolvable", "exhausted"])
sm.add_transition(NonDualState.SELF_LIBERATING, AgentState.COMPLETED,
                  ["reported"])
```

### State Callbacks

```python
# On entering GROUNDING state:
async def on_enter_grounding(from_state, to_state, trigger):
    """Select and invoke grounding channel."""
    loop_signal = state_machine.history[-1].metadata.get("loop_signal")
    channel = grounding_router.select_channel(loop_signal.stuck_type)
    result = await channel.invoke(loop_signal.stuck_context)
    # Re-integration happens in the transition to EXECUTING

# On entering SELF_LIBERATING state:
async def on_enter_self_liberating(from_state, to_state, trigger):
    """Generate transparent report."""
    report = self_liberation_protocol(
        stuck_history=loop_detector.get_history(),
        grounding_attempts=grounding_router.get_attempt_history()
    )
    # Report becomes the final output
```

---

## 6. Preventing Infinite Regress in the Grounder

### The Problem

If the grounding system can itself get stuck, we have an infinite regress: the agent is stuck, the grounder tries to help, the grounder gets stuck, another grounder tries to help the grounder, and so on. This is the non-dual equivalent of asking "Who watches the watchman?"

### The Solution: Grounding Is Inherently Bounded

The grounding channels are designed to be structurally incapable of looping:

1. **Statistical Ground**: Computes statistics on data. This is a deterministic, finite computation. It cannot loop because it does not reason -- it calculates.

2. **Exemplar Ground**: Retrieves or generates concrete examples. This is bounded by the number of examples requested (default 3).

3. **Visual/Spatial Ground**: Builds a graph and computes structural properties. Graph algorithms (cycle detection, connected components) are bounded by O(V + E).

4. **Embodied Ground**: Executes a test. Tests either pass or fail in bounded time (with a timeout). They do not reason.

5. **Relational Ground**: Queries external sources. Queries return results or timeout. They do not loop.

6. **Temporal Ground**: Traces execution history. The history is finite and already recorded.

7. **Null Ground**: Does nothing. Cannot loop by definition.

### Additional Safeguards

- **Grounding timeout**: Each grounding attempt has a maximum duration (default 5000ms). If the channel does not return within the timeout, the grounding is considered failed and the next channel in the escalation ladder is tried.
- **Maximum grounding attempts**: The system allows at most MAX_GROUNDING_ATTEMPTS (default 3) before invoking self-liberation.
- **No recursive grounding**: A grounding channel is never invoked to ground another grounding channel. The grounding layer is flat, not recursive.

This maps to the non-dual principle that the ground of awareness (rigpa in Dzogchen, the Tao in Taoism) is not itself something that needs grounding. It is the ground. The seven channels are computational primitives -- deterministic, bounded operations that cannot enter the loops they are designed to resolve.

---

## 7. Non-Dual Traditions Referenced

| Tradition | Concept | Grounding Application |
|-----------|---------|----------------------|
| Zen | Breath as anchor (anapanasati) | Statistical Ground: return to raw data |
| Zen | Kinhin (walking meditation) | Embodied Ground: shift from thinking to doing |
| Zen | Shikantaza (just sitting) | Null Ground: stop processing entirely |
| Sufism | Dhikr (remembrance) | Return to center after drift; all channels loop back to the task |
| Sufism | Murshid (guide) | Relational Ground: seek external perspective |
| Taoism | Wu wei (effortless action) | Null Ground: non-action as action |
| Taoism | Returning to the root (gui gen) | All grounding returns to the raw, the concrete, the unprocessed |
| Yogacara | Bija (seeds) and alaya-vijnana | Temporal Ground: trace the causal seeds of the stuck state |
| Dzogchen | Rigpa as the ground | The grounding channels are computational primitives that cannot loop |
| Kashmir Shaivism | 112 techniques (Vijnana Bhairava Tantra) | Multiple channels for multiple modalities |

---

## 8. Standalone Operation (Without NervousSystem)

When the `NervousSystem` is not available (the agent is running standalone without the 40-form architecture), the grounding channels fall back to simplified local implementations:

| Channel | Standalone Implementation |
|---------|-------------------------|
| Statistical Ground | Compute basic statistics (mean, std, distribution) on available data using standard library |
| Exemplar Ground | Generate examples from the agent's context window; use memory search if available |
| Visual/Spatial Ground | Build a simple dependency graph from extracted propositions; compute cycles and components |
| Embodied Ground | Use the ToolProvider to execute tests if available; otherwise, report that embodied grounding requires tools |
| Relational Ground | Search memory if available; otherwise, ask the agent to consider alternative approaches explicitly |
| Temporal Ground | Trace the agent's execution history from the state machine's transition log |
| Null Ground | asyncio.sleep(pause_duration) or synchronous pause |

The standalone implementations are simpler but preserve the essential function of each channel: providing an alternate modality that breaks the primary modality's loop.

---

*This document is part of the AI-Nondualism module, `ml_research/ai_nondualism/loop_escape/grounding_protocols.md`.*
*It references: `neural_network/adapters/base_adapter.py` (FormAdapter hierarchy, SensoryAdapter, CognitiveAdapter, TheoreticalAdapter, SpecializedAdapter), `neural_network/core/nervous_system.py` (NervousSystem, arousal gating, form routing), `neural_network/core/message_bus.py` (MessageBus, MessageType, FormMessage), `agent_frameworks/core/state_machine.py` (StateMachine, AgentState, transitions), and the non-dual tradition documents for Zen, Sufism, Taoism, Yogacara, Dzogchen, and Kashmir Shaivism.*
