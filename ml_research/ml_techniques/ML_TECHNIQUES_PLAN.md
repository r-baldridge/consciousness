# ML Techniques Development Plan

## Overview

This plan outlines the development of missing/stub implementations in ml_techniques and integration with modern_dev architectures.

**Current State**: 50+ techniques indexed, ~90% implemented, key gaps identified

---

## Gap Analysis

### 1. Stub Implementations Needing Full Development

| Technique | Location | Status | Priority |
|-----------|----------|--------|----------|
| HierarchicalTaskDecomposition | decomposition/__init__.py | Stub | HIGH |
| MultiAgent | agentic/__init__.py | Stub | HIGH |

### 2. Missing Techniques to Add

| Technique | Category | Description | Priority |
|-----------|----------|-------------|----------|
| ProgramSynthesis | code_synthesis | Generate programs from specifications | HIGH |
| CodeAsPolicy | code_synthesis | Use code for action/policy execution | MEDIUM |
| NeuroSymbolicReasoning | agentic | Combine neural + symbolic reasoning | MEDIUM |
| CausalReasoning | prompting | Explicit cause-effect analysis | LOW |

### 3. Architecture Integration Needed

| Architecture | Techniques to Connect | Priority |
|--------------|----------------------|----------|
| TRM | recursive_decomposition, chain_of_thought | HIGH |
| CTM | temporal reasoning, memory patterns | HIGH |
| Mamba | long_context_memory, streaming_inference | HIGH |

---

## Development Tasks

### Phase 1: Stub Completion (Parallel Agents)

#### Agent-T1: HierarchicalTaskDecomposition
```yaml
task_id: htn_implementation
category: decomposition
status: needs_implementation
dependencies: none

goal: |
  Implement full HTN-style planning with:
  - Task templates with preconditions/effects
  - Method ordering (multiple ways to achieve goals)
  - Partial-order planning
  - Dynamic replanning on failure

files_to_modify:
  - ml_techniques/decomposition/__init__.py

implementation:
  classes:
    - TaskTemplate: Define reusable task patterns
    - Method: Alternative ways to decompose a task
    - HTNPlanner: Core planning algorithm
    - PlanValidator: Check plan feasibility

  key_methods:
    - decompose_task(goal) -> TaskTree
    - find_methods(task) -> List[Method]
    - apply_method(task, method) -> subtasks
    - validate_plan(plan, state) -> bool
    - replan_on_failure(plan, failed_task) -> Plan

acceptance_criteria:
  - Correctly decomposes goals into subtask hierarchies
  - Supports multiple methods per task
  - Handles task dependencies
  - Replans when subtasks fail
  - Integrates with existing TechniqueBase
```

#### Agent-T2: MultiAgent Coordination
```yaml
task_id: multiagent_implementation
category: agentic
status: needs_implementation
dependencies: none

goal: |
  Implement full multi-agent patterns:
  - Debate: Agents argue positions, judge decides
  - Division: Specialized agents handle subtasks
  - Hierarchical: Manager coordinates workers
  - Peer review: Agents check each other's work

files_to_modify:
  - ml_techniques/agentic/__init__.py

implementation:
  patterns:
    debate:
      - Proposer generates initial answer
      - Critic finds flaws
      - Proposer refines
      - Judge evaluates final

    division:
      - Coordinator decomposes task
      - Workers claim subtasks
      - Results aggregated

    hierarchical:
      - Manager plans strategy
      - Workers execute
      - Manager reviews and adjusts

    peer_review:
      - Worker A produces output
      - Worker B reviews
      - Revisions based on feedback

acceptance_criteria:
  - All 4 coordination patterns working
  - Configurable agent roles
  - Message passing between agents
  - Consensus mechanisms
  - Traces full agent interactions
```

### Phase 2: New Techniques

#### Agent-T3: ProgramSynthesis
```yaml
task_id: program_synthesis
category: code_synthesis
status: new_implementation
dependencies: none

goal: |
  Implement program synthesis from specifications:
  - Input-output examples
  - Natural language specs
  - Formal constraints
  - Iterative refinement

files_to_create:
  - ml_techniques/code_synthesis/program_synthesis.py

implementation:
  classes:
    - SpecificationParser: Parse specs from various formats
    - ProgramGenerator: Generate candidate programs
    - TestExecutor: Run against examples
    - Synthesizer: Orchestrate synthesis loop

  synthesis_strategies:
    - example_based: Learn from I/O pairs
    - sketch_based: Fill holes in template
    - decomposition: Break spec into subproblems
    - enumeration: Search program space

acceptance_criteria:
  - Generates working programs from I/O examples
  - Handles natural language specs
  - Iteratively refines based on failures
  - Integrates with RLM infrastructure
```

#### Agent-T4: Architecture Integration
```yaml
task_id: architecture_integration
category: integration
status: new_implementation
dependencies: [htn_implementation, multiagent_implementation]

goal: |
  Connect techniques with modern_dev architectures:
  - TRM + RecursiveDecomposition + ChainOfThought
  - CTM + TemporalMemory + Verification
  - Mamba + LongContextRAG + StreamingInference

files_to_create:
  - ml_techniques/integration/__init__.py
  - ml_techniques/integration/trm_techniques.py
  - ml_techniques/integration/ctm_techniques.py
  - ml_techniques/integration/mamba_techniques.py

implementation:
  trm_integration:
    - TRMDecomposer: Use TRM iterations for decomposition
    - TRMChainOfThought: TRM-backed reasoning

  ctm_integration:
    - CTMTemporalReasoning: Leverage synchronization
    - CTMVerification: Use neuron activity for confidence

  mamba_integration:
    - MambaRAG: Long-context retrieval
    - MambaStreaming: Efficient streaming inference

acceptance_criteria:
  - Each architecture has technique wrappers
  - Techniques can specify preferred architecture
  - Automatic architecture selection via Orchestrator
  - Performance benchmarks available
```

---

## Implementation Order

```
Phase 1: Week 1 (Parallel)
├── Agent-T1: HTN Decomposition
├── Agent-T2: MultiAgent Coordination
└── Tests for Phase 1

Phase 2: Week 2 (Parallel, after Phase 1)
├── Agent-T3: Program Synthesis
├── Agent-T4: Architecture Integration
└── Tests for Phase 2

Phase 3: Week 3 (Sequential)
├── Integration tests
├── Documentation
└── Example notebooks
```

---

## Agent Prompts

### Agent-T1 Prompt: HTN Implementation
```
You are implementing HierarchicalTaskDecomposition for ml_research/ml_techniques.

CONTEXT:
- Base class: TechniqueBase from ml_techniques/__init__.py
- Location: ml_techniques/decomposition/__init__.py
- Existing stubs: RecursiveDecomposition, LeastToMost (working)

TASK:
Replace the stub HierarchicalTaskDecomposition with a full implementation.

REQUIREMENTS:
1. Create TaskTemplate class for reusable task patterns
2. Create Method class for alternative decomposition strategies
3. Implement HTN planning algorithm with:
   - Task hierarchy construction
   - Method selection (prioritized ordering)
   - Precondition/effect tracking
   - Partial-order planning
4. Add PlanValidator for feasibility checking
5. Support dynamic replanning on failure

PATTERNS TO FOLLOW:
- Match RecursiveDecomposition's style
- Use TechniqueResult for return values
- Add hooks for step tracking
- Include trace in intermediate_steps

TESTS:
Create tests in ml_techniques/tests/test_decomposition.py
- test_basic_decomposition
- test_method_selection
- test_precondition_checking
- test_replanning

DO NOT modify RecursiveDecomposition or LeastToMost.
```

### Agent-T2 Prompt: MultiAgent Implementation
```
You are implementing full MultiAgent coordination for ml_research/ml_techniques.

CONTEXT:
- Base class: TechniqueBase from ml_techniques/__init__.py
- Location: ml_techniques/agentic/__init__.py
- Existing: AgentSpec, MultiAgent (stub with placeholder)

TASK:
Replace the MultiAgent stub with full multi-agent coordination patterns.

REQUIREMENTS:
1. Implement 4 coordination patterns:
   - debate: Multiple agents argue, judge decides
   - division: Coordinator splits work among specialists
   - hierarchical: Manager directs workers
   - peer_review: Agents review each other's work

2. Create supporting classes:
   - AgentMessage: Message passing between agents
   - ConversationHistory: Track agent interactions
   - ConsensusProtocol: Reach agreement

3. Configuration options:
   - num_agents
   - coordination_pattern
   - max_rounds
   - consensus_threshold

PATTERNS TO FOLLOW:
- Match ReAct/Planning technique style
- Use backend for agent responses
- Full trace of agent interactions
- Configurable via constructor

TESTS:
Create tests in ml_techniques/tests/test_agentic.py
- test_debate_pattern
- test_division_pattern
- test_hierarchical_pattern
- test_peer_review_pattern

DO NOT modify other agentic techniques.
```

### Agent-T3 Prompt: Program Synthesis
```
You are implementing ProgramSynthesis for ml_research/ml_techniques.

CONTEXT:
- Base class: TechniqueBase from ml_techniques/__init__.py
- Related: RLM in code_synthesis/ (for reference)
- New file: ml_techniques/code_synthesis/program_synthesis.py

TASK:
Create a new ProgramSynthesis technique for generating code from specifications.

REQUIREMENTS:
1. SpecificationParser class:
   - Parse I/O examples
   - Parse natural language specs
   - Parse formal constraints

2. ProgramGenerator class:
   - Generate candidate programs
   - Support multiple languages (Python primary)
   - Template/sketch-based generation

3. Synthesizer class (main technique):
   - Orchestrate synthesis loop
   - Test against examples
   - Refine on failure
   - Return best program

4. Synthesis strategies:
   - EXAMPLE_BASED: Learn from I/O pairs
   - SKETCH_BASED: Fill template holes
   - DECOMPOSITION: Break into subproblems

PATTERNS TO FOLLOW:
- Match RLM's style in code_synthesis/
- Return TechniqueResult with program
- Include execution trace
- Handle failures gracefully

TESTS:
Create tests in ml_techniques/tests/test_code_synthesis.py
- test_simple_synthesis (add two numbers)
- test_string_manipulation
- test_list_operations
- test_refinement_loop
```

### Agent-T4 Prompt: Architecture Integration
```
You are creating architecture integrations for ml_research/ml_techniques.

CONTEXT:
- Architectures: TRM, CTM, Mamba in modern_dev/
- Techniques: All in ml_techniques/
- Unified Index: core/unified_index.py

TASK:
Create integration layer connecting techniques to architectures.

FILES TO CREATE:
1. ml_techniques/integration/__init__.py
2. ml_techniques/integration/trm_techniques.py
3. ml_techniques/integration/ctm_techniques.py
4. ml_techniques/integration/mamba_techniques.py

REQUIREMENTS:

1. TRM Integration (trm_techniques.py):
   - TRMDecomposer: Use TRM iterations for recursive decomposition
   - TRMChainOfThought: TRM-backed reasoning chains
   - TRMCodeRepair: Connect to code repair pipeline

2. CTM Integration (ctm_techniques.py):
   - CTMTemporalReasoning: Leverage synchronization patterns
   - CTMMemory: Use neuron dynamics for memory
   - CTMVerification: Confidence from activity patterns

3. Mamba Integration (mamba_techniques.py):
   - MambaRAG: Long-context retrieval (10K+ tokens)
   - MambaStreaming: Efficient streaming inference
   - MambaCompression: State-space context compression

4. Registry (__init__.py):
   - ARCHITECTURE_TECHNIQUE_MAP
   - get_technique_for_architecture()
   - get_architecture_for_technique()

PATTERNS:
- Each wrapper inherits from TechniqueBase
- Lazy-load architecture components
- Fall back to base technique if arch unavailable

TESTS:
Create ml_techniques/tests/test_integration.py
```

---

## Verification Checklist

### Phase 1 Completion
- [ ] HierarchicalTaskDecomposition passes all tests
- [ ] MultiAgent all 4 patterns working
- [ ] No regressions in existing techniques
- [ ] Documentation updated

### Phase 2 Completion
- [ ] ProgramSynthesis generates correct code
- [ ] Architecture integrations functional
- [ ] Unified Index updated
- [ ] Full test coverage

### Final Verification
```bash
# Run all technique tests
python -m pytest ml_techniques/tests/ -v

# Verify imports
python -c "from ml_research.ml_techniques import *; print('All imports OK')"

# Check integration
python -c "from ml_research import get_compatible_techniques; print(get_compatible_techniques('trm'))"
```

---

## Notes

1. **Backend Abstraction**: All techniques use `backend` parameter for LLM calls
2. **Trace Everything**: Every step should be in `intermediate_steps`
3. **Composable**: Techniques should work with Pipeline and ParallelComposition
4. **Type Hints**: Full typing for all new code
5. **Docstrings**: Match existing style with paper references
