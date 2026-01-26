# Quick Start Guide

Get the ML Research Code Repair Pipeline running in under 30 minutes.

## Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- GPU with 4GB+ VRAM (optional but recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ml-research.git
cd ml-research/consciousness/ml_research/modern_dev
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
python -c "from modern_dev.orchestrator.router import MLOrchestrator; print('Installation successful!')"
```

---

## Basic Usage

### Repair Code with Default Settings

```python
from modern_dev.orchestrator.router import MLOrchestrator, Request

# Initialize orchestrator
orchestrator = MLOrchestrator()

# Repair buggy code
response = orchestrator.process(Request(
    task_type="code_repair",
    input_data={"buggy_code": "def add(a, b): return a + c"},
))

if response.success:
    print(f"Architecture used: {response.architecture_used}")
    print(f"Output: {response.output}")
else:
    print(f"Error: {response.metadata}")
```

### Direct TRM Usage

```python
from modern_dev.trm.src.model import CodeRepairTRM, CodeRepairConfig

# Create model with preset configuration
config = CodeRepairConfig.for_code_repair_small()
model = CodeRepairTRM(config)

# Check model size
print(f"Parameters: {model.num_parameters():,}")

# Generate repair (simplified example)
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 64, 48))
result = model.generate(input_ids)
print(f"Iterations: {result['iterations']}, Confidence: {result['confidence'].item():.2%}")
```

### Direct Mamba Usage

```python
from modern_dev.mamba_impl.src.mamba_model import MambaLM, MambaLMConfig

# Create model
config = MambaLMConfig(
    vocab_size=32000,
    d_model=512,
    n_layers=12,
)
model = MambaLM(config)

# Generate text
import torch
prompt = torch.randint(0, config.vocab_size, (1, 10))
generated = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
)
print(f"Generated {generated.shape[1]} tokens")
```

---

## Using the CLI

### Repair a File

```bash
python -m modern_dev.cli repair --input buggy.py --output fixed.py
```

### Process a Directory

```bash
python -m modern_dev.cli repair --input src/ --output fixed_src/ --recursive
```

### Check Status

```bash
python -m modern_dev.cli status
```

---

## Common Tasks

### Task 1: Repair Simple Syntax Errors

```python
from modern_dev.orchestrator.router import MLOrchestrator, Request

orch = MLOrchestrator()

# Variable typo
result = orch.process(Request(
    task_type="code_repair",
    input_data={
        "buggy_code": "def greet(name): return 'Hello, ' + nme",
        "error_message": "NameError: name 'nme' is not defined",
    },
))
```

### Task 2: Fix Logic Errors

```python
result = orch.process(Request(
    task_type="code_repair",
    input_data={
        "buggy_code": """
def divide(a, b):
    return a / b
""",
        "error_message": "ZeroDivisionError: division by zero",
        "test_cases": [
            {"input": {"a": 10, "b": 2}, "expected": 5},
            {"input": {"a": 10, "b": 0}, "expected": "error"},
        ],
    },
))
```

### Task 3: Analyze Long Code Files

```python
# Use Mamba for long context
result = orch.process(Request(
    task_type="analysis",
    input_data={"code": large_codebase},
    constraints={"context_length": 50000},
))
```

### Task 4: Generate Code from Specification

```python
result = orch.process(Request(
    task_type="code_generation",
    input_data={
        "specification": "Create a function that computes fibonacci numbers",
        "language": "python",
    },
))
```

---

## Model Presets

### TRM Configurations

| Preset | Parameters | Use Case |
|--------|-----------|----------|
| `for_code_repair_tiny` | ~1M | Unit testing |
| `for_code_repair_small` | ~9M | Quick experiments |
| `for_code_repair_base` | ~12M | Standard training |
| `for_code_repair_large` | ~23M | Complex repairs |

```python
from modern_dev.trm.src.model import CodeRepairConfig

config = CodeRepairConfig.for_code_repair_base()
```

### Mamba Configurations

```python
from modern_dev.mamba_impl.src.mamba_model import MambaLMConfig

# Small (fast inference)
config = MambaLMConfig(d_model=512, n_layers=12)

# Medium (balanced)
config = MambaLMConfig(d_model=768, n_layers=24)

# Large (high quality)
config = MambaLMConfig(d_model=1024, n_layers=32)
```

---

## Environment Configuration

Create a `.env` file in your project root:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Model Paths
MODEL_CACHE_DIR=~/.cache/modern_dev/models

# Logging
LOG_LEVEL=INFO

# Optional: Weights & Biases
WANDB_API_KEY=your_key_here
WANDB_PROJECT=ml-research
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'modern_dev'"

Ensure you're in the correct directory and have installed the package:

```bash
cd /path/to/ml-research/consciousness/ml_research
pip install -e ".[dev]"
```

### "CUDA out of memory"

Reduce batch size or use a smaller model preset:

```python
# Use smaller model
config = CodeRepairConfig.for_code_repair_small()

# Or enable gradient checkpointing
config.use_gradient_checkpointing = True
```

### "No GPU available"

Check PyTorch CUDA installation:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### "Model loading too slow"

Pre-load models on startup:

```python
orchestrator = MLOrchestrator()
# Force model loading
_ = orchestrator.registry.get_instance("trm")
_ = orchestrator.registry.get_instance("mamba")
```

---

## Performance Tips

1. **Use GPU**: 10-50x speedup for inference
2. **Enable Caching**: Mamba supports O(1) inference with cache
3. **Batch Processing**: Process multiple files together
4. **Early Stopping**: TRM stops when confident (saves iterations)

---

## Next Steps

- **API Reference**: See [API.md](./API.md) for complete documentation
- **Architecture**: See [ARCHITECTURE.md](./ARCHITECTURE.md) for system design
- **Examples**: See `/examples` directory for working code samples
- **TRM Details**: See `/trm/docs/` for TRM-specific documentation
