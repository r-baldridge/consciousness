# CTM Code Repair - Project Specification

## Project Overview

Adapt the Continuous Thought Machine (CTM) architecture for Python code repair, leveraging its unique temporal processing and neural synchronization mechanisms to iteratively reason about and fix buggy code.

**Hypothesis**: CTM's internal temporal axis and neuron-level models may enable superior reasoning about code structure compared to single-pass models, particularly for complex logic bugs requiring multi-step reasoning.

---

## Architecture Configuration

### CTM-CodeRepair Model

```yaml
model:
  name: CTM-CodeRepair
  base: Continuous Thought Machine (Sakana AI)

  # Core dimensions
  d_model: 384              # Latent dimension (balanced for code)
  d_input: 256              # Attention output dimension
  heads: 8                  # Attention heads
  n_synch_out: 48           # Output synchronization neurons
  n_synch_action: 48        # Action synchronization neurons

  # Temporal processing
  iterations: 64            # Internal thought ticks
  memory_length: 32         # NLM history window
  synapse_depth: 6          # U-NET depth

  # Neuron-level models
  deep_nlms: true
  memory_hidden_dims: 64

  # Code-specific
  vocab_size: 32768         # BPE vocabulary
  grid_height: 64           # Max lines
  grid_width: 48            # Max tokens per line

  # Backbone
  backbone_type: code-encoder  # Custom for code
  positional_embedding: learnable-fourier-2d

  # Estimated parameters
  parameters: ~15M          # Larger than TRM-7M due to NLMs
```

### Parameter Breakdown

| Component | Parameters | Notes |
|-----------|------------|-------|
| Code Backbone | 3.2M | Conv + projection |
| Token Embedding | 12.6M | 32K × 384 |
| Positional Embedding | 0.1M | Fourier features |
| Synapse U-NET | 2.4M | 6-layer U-NET |
| Neuron-Level Models | 4.8M | D × unique MLPs |
| Attention | 1.2M | Q, K, V projections |
| Output Projection | 0.4M | Sync → vocab |
| **Total** | **~15M** | |

---

## Data Pipeline

### Shared Infrastructure with TRM

The CTM code repair project shares the data collection infrastructure with TRM:

```
shared/data/
├── collectors/          # GitHub, SO, Synthetic, Linters
├── processors/          # Tokenizer, Validator, Augmenter
├── taxonomy/            # Bug types
└── utils/               # Dataset builder
```

### CTM-Specific Data Format

```python
@dataclass
class CTMCodeSample:
    """Training sample for CTM code repair."""

    # Input
    buggy_grid: np.ndarray      # [64, 48] token IDs
    buggy_mask: np.ndarray      # [64, 48] valid positions

    # Target
    fixed_grid: np.ndarray      # [64, 48] token IDs
    diff_mask: np.ndarray       # [64, 48] positions that change

    # Metadata
    bug_type: str
    difficulty: float
    source: str

    # Optional: intermediate supervision
    bug_location: Tuple[int, int]  # For early iterations
```

### Grid Encoding for CTM

```python
class CTMGridEncoder:
    """Encode code for CTM with 2D positional information."""

    def encode(self, code: str) -> Dict[str, np.ndarray]:
        # Tokenize to grid
        grid = self.tokenizer.encode_to_grid(code)

        # Create attention mask
        mask = (grid != self.pad_token).astype(np.float32)

        # 2D positions for Fourier embedding
        positions = self._create_position_grid(grid.shape)

        return {
            'input_ids': grid,
            'attention_mask': mask,
            'positions': positions,
        }
```

---

## Training Configuration

### Optimizer & Schedule

```yaml
training:
  optimizer: AdamW
  learning_rate: 3e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]

  scheduler: cosine_with_warmup
  warmup_steps: 2000
  min_lr: 1e-6

  gradient_clipping: 1.0
  mixed_precision: bf16

  batch_size: 32
  effective_batch_size: 128  # With gradient accumulation
  gradient_accumulation: 4
```

### Loss Function

```python
def ctm_code_repair_loss(
    predictions: List[Tensor],     # [T] list of [B, H, W, V]
    certainties: List[Tensor],     # [T] list of [B]
    targets: Tensor,               # [B, H, W]
    diff_mask: Tensor,             # [B, H, W]
    iteration_weights: str = 'exponential',
) -> Tensor:
    """
    Multi-iteration loss for CTM code repair.

    Supervises all iterations with increasing weight toward later iterations.
    Focuses loss on positions that differ between buggy and fixed.
    """
    T = len(predictions)
    total_loss = 0

    for t, (pred, cert) in enumerate(zip(predictions, certainties)):
        # Per-token cross entropy
        ce_loss = F.cross_entropy(
            pred.view(-1, pred.size(-1)),
            targets.view(-1),
            reduction='none'
        ).view_as(targets)

        # Weight by diff mask (focus on changed positions)
        masked_loss = (ce_loss * diff_mask).sum() / diff_mask.sum().clamp(min=1)

        # Iteration weight (later iterations matter more)
        if iteration_weights == 'exponential':
            weight = 2 ** (t - T + 1)  # [0.5^(T-1), ..., 0.5, 1]
        elif iteration_weights == 'linear':
            weight = (t + 1) / T
        else:
            weight = 1.0

        total_loss += weight * masked_loss

    # Certainty regularization (encourage decisive outputs)
    cert_loss = -torch.mean(torch.stack(certainties))

    return total_loss + 0.1 * cert_loss
```

### Curriculum Learning

```yaml
curriculum:
  # Stage 1: Simple bugs, few iterations
  stage_1:
    iterations: 16
    samples: 200,000
    difficulty: 0.0-0.3
    bug_types: [syntax]
    epochs: 2
    lr: 3e-4

  # Stage 2: Medium bugs, more iterations
  stage_2:
    iterations: 32
    samples: 300,000
    difficulty: 0.0-0.5
    bug_types: [syntax, simple_logic]
    epochs: 2
    lr: 2e-4

  # Stage 3: Complex bugs, full iterations
  stage_3:
    iterations: 48
    samples: 500,000
    difficulty: 0.0-0.7
    bug_types: [all_logic]
    epochs: 2
    lr: 1e-4

  # Stage 4: Full training
  stage_4:
    iterations: 64
    samples: 1,000,000
    difficulty: 0.0-1.0
    bug_types: [all]
    epochs: 3
    lr: 5e-5
```

---

## Implementation Plan

### Phase 1: Core CTM Implementation (Week 1-2)

```
ctm/src/
├── __init__.py
├── model.py              # Main CTM class
├── modules/
│   ├── __init__.py
│   ├── synapse.py        # U-NET synapse network
│   ├── nlm.py            # Neuron-level models
│   ├── synchronization.py # Sync computation
│   ├── attention.py      # Multi-head attention
│   └── backbone.py       # Code encoder backbone
├── positional/
│   ├── __init__.py
│   ├── fourier.py        # Learnable Fourier
│   └── rotational.py     # Rotational embeddings
└── config.py             # Model configuration
```

**Key Classes**:

```python
class CTMConfig:
    d_model: int = 384
    iterations: int = 64
    memory_length: int = 32
    synapse_depth: int = 6
    heads: int = 8
    n_synch_out: int = 48
    n_synch_action: int = 48
    deep_nlms: bool = True
    memory_hidden_dims: int = 64
    vocab_size: int = 32768
    grid_height: int = 64
    grid_width: int = 48


class ContinuousThoughtMachine(nn.Module):
    def __init__(self, config: CTMConfig):
        self.config = config
        self.backbone = CodeBackbone(config)
        self.positional = LearnableFourier2D(config)
        self.synapse = SynapseUNET(config)
        self.nlm = NeuronLevelModels(config)
        self.attention = MultiHeadAttention(config)
        self.sync_computer = SynchronizationModule(config)
        self.output_proj = nn.Linear(config.n_synch_out, config.vocab_size)

    def forward(self, input_ids, attention_mask, positions):
        # Backbone + positional
        features = self.backbone(input_ids)
        features = features + self.positional(positions)

        # KV projection for attention
        kv = self.kv_proj(features)

        # Initialize state
        state_trace = self.init_trace(batch_size)
        activated_state = self.init_state(batch_size)
        sync_accum = self.init_sync_accumulators(batch_size)

        outputs = []
        certainties = []

        # Internal recurrence
        for t in range(self.config.iterations):
            # Compute action synchronization
            action_sync = self.sync_computer.action_sync(
                activated_state, sync_accum
            )

            # Attention with sync-derived query
            query = self.query_proj(action_sync)
            attn_out = self.attention(query, kv, attention_mask)

            # Synapse processing
            combined = torch.cat([attn_out, activated_state], dim=-1)
            pre_activations = self.synapse(combined)

            # Update trace
            state_trace = self.update_trace(state_trace, pre_activations)

            # Neuron-level models
            activated_state = self.nlm(state_trace)

            # Output synchronization
            output_sync = self.sync_computer.output_sync(
                activated_state, sync_accum
            )

            # Predictions and certainty
            logits = self.output_proj(output_sync)
            certainty = self.compute_certainty(logits)

            outputs.append(logits)
            certainties.append(certainty)

        return outputs, certainties
```

### Phase 2: Training Infrastructure (Week 2-3)

```
ctm/
├── training/
│   ├── __init__.py
│   ├── trainer.py        # Main trainer
│   ├── losses.py         # CTM-specific losses
│   ├── curriculum.py     # Curriculum scheduler
│   └── callbacks.py      # Training callbacks
├── cli/
│   ├── __init__.py
│   ├── train.py          # Training CLI
│   └── evaluate.py       # Evaluation CLI
└── configs/
    ├── base.yaml
    ├── code_repair.yaml
    └── curriculum/
        ├── stage_1.yaml
        ├── stage_2.yaml
        ├── stage_3.yaml
        └── stage_4.yaml
```

### Phase 3: Evaluation & Analysis (Week 3-4)

```
ctm/
├── evaluation/
│   ├── __init__.py
│   ├── benchmarks.py     # Benchmark runners
│   ├── metrics.py        # Evaluation metrics
│   └── visualizations.py # Sync/attention viz
└── analysis/
    ├── __init__.py
    ├── iteration_analysis.py   # Per-iteration accuracy
    ├── sync_patterns.py        # Synchronization analysis
    └── comparison.py           # CTM vs TRM comparison
```

---

## Evaluation Framework

### Metrics

| Metric | Description |
|--------|-------------|
| **Fix Accuracy** | % of bugs correctly fixed |
| **Token Accuracy** | % of tokens correctly predicted |
| **Iteration Efficiency** | Accuracy vs iteration count |
| **Certainty Calibration** | Certainty correlation with accuracy |
| **Bug Localization** | How well sync identifies bug location |

### Benchmark Suite

```yaml
benchmarks:
  syntax_benchmark:
    samples: 5000
    bug_types: [syntax_errors]
    target_accuracy: 95%

  logic_benchmark:
    samples: 5000
    bug_types: [logic_errors]
    target_accuracy: 75%

  security_benchmark:
    samples: 2000
    bug_types: [security_vulns]
    target_accuracy: 80%

  iteration_scaling:
    samples: 1000
    iterations: [8, 16, 32, 48, 64]
    measure: accuracy_vs_compute

  comparison_vs_trm:
    samples: 5000
    models: [ctm, trm]
    measure: accuracy, calibration, speed
```

### Analysis Outputs

1. **Iteration Progression**: How predictions improve over iterations
2. **Synchronization Patterns**: What neural pairs activate for different bugs
3. **Attention Maps**: Where the model focuses at each iteration
4. **Certainty Curves**: When the model becomes confident

---

## Expected Outcomes

### Performance Targets

| Task | CTM Target | TRM Comparison |
|------|------------|----------------|
| Syntax Fix | >93% | TRM: >90% |
| Logic Fix | >72% | TRM: >70% |
| Security Fix | >78% | TRM: >80% |
| Overall | >76% | TRM: >75% |

### Hypothesized Advantages

1. **Complex Logic Bugs**: Multi-step reasoning via iterations
2. **Calibration**: Better certainty estimates
3. **Interpretability**: Sync patterns may be meaningful

### Hypothesized Disadvantages

1. **Speed**: Sequential iterations (cannot parallelize)
2. **Memory**: State traces grow with iterations
3. **Training**: May be harder to stabilize

---

## Resource Requirements

### Hardware

- **GPU Memory**: 80GB (larger than TRM due to traces)
- **Training Time**: ~40 GPU-hours (longer due to iterations)
- **Data Storage**: Same as TRM (150 GB)

### Compute Budget

| Phase | GPU-Hours | Notes |
|-------|-----------|-------|
| Implementation | 10 | Debugging, validation |
| Stage 1 Training | 5 | 16 iterations, fast |
| Stage 2 Training | 8 | 32 iterations |
| Stage 3 Training | 12 | 48 iterations |
| Stage 4 Training | 15 | 64 iterations, full data |
| Evaluation | 5 | Benchmarks, analysis |
| **Total** | **55** | |

---

## Success Criteria

### Minimum Viable
- [ ] CTM trains stably on code repair task
- [ ] Accuracy improves with more iterations
- [ ] Performance matches TRM baseline

### Target
- [ ] Exceeds TRM on complex logic bugs
- [ ] Good certainty calibration
- [ ] Meaningful synchronization patterns

### Stretch
- [ ] Novel insights from sync analysis
- [ ] State-of-the-art on code repair benchmarks
- [ ] Efficient early-stopping via certainty
