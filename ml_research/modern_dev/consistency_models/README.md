# Consistency Models

> A generative modeling approach enabling one-step generation by learning consistency mappings that map any point on a diffusion trajectory directly to its origin, bypassing iterative sampling.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Consistency Models](https://arxiv.org/abs/2303.01469)
**Year:** 2023
**Organization:** OpenAI

## Overview

Consistency Models address the fundamental limitation of diffusion models: their slow iterative sampling process. The key insight is that all points on the same trajectory of a diffusion ODE map to the same initial data point. Rather than iterating through the trajectory step by step, Consistency Models learn this mapping directly, enabling generation in as few as one step while maintaining high quality.

The architecture learns a consistency function f(x_t, t) such that f(x_t, t) = f(x_s, s) for any two points on the same trajectory. This self-consistency property, combined with a boundary condition f(x, epsilon) = x, allows the model to directly predict the clean data point from any noisy version. The model can be trained either by distillation from a pre-trained diffusion model (Consistency Distillation) or from scratch (Consistency Training).

Latent Consistency Models (LCM) extend this approach to the latent space of Stable Diffusion, enabling 1-4 step high-quality image generation with real-time performance. This has made Consistency Models practically important for applications requiring fast generation, such as interactive image editing and real-time synthesis.

## Key Innovations

- **Consistency Mapping**: Learns to map any point on a diffusion trajectory directly to its origin, bypassing iterative denoising entirely.

- **Self-Consistency Property**: Ensures f(x_t, t) = f(x_s, s) for all points on the same trajectory, enabling single-step generation.

- **Dual Training Methods**: Can be trained via distillation from diffusion models (CD) or from scratch (CT), providing flexibility in deployment.

## Architecture Diagram

```
Diffusion Model Sampling (iterative):
    x_T --> x_{T-1} --> x_{T-2} --> ... --> x_1 --> x_0
     |         |          |                  |       |
    [denoise] [denoise]  [denoise]        [denoise]
    (many steps required)

Consistency Model Sampling (direct):
    x_T -----------------------> x_0
                |
           f_theta(x_T, T)
    (single step possible!)

Consistency Function Properties:
    +-----------------------------------------------+
    |  For any trajectory point (x_t, t):           |
    |                                               |
    |  f(x_t, t) = f(x_s, s) = x_0                 |
    |  (all points on trajectory map to same x_0)  |
    |                                               |
    |  f(x, epsilon) = x                           |
    |  (boundary condition at t=epsilon)           |
    +-----------------------------------------------+

Architecture:
    f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)

    where c_skip and c_out ensure boundary condition:
        c_skip(epsilon) = 1
        c_out(epsilon) = 0
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| ConsistencyModel | Stub | Main model class |
| ConsistencyDistillation | Stub | Training with teacher |
| ConsistencyTraining | Stub | From-scratch training |
| SkipScaling | Stub | Boundary condition |
| TimeSchedule | Stub | Karras schedule |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | CM and LCM configs |

## Training Methods

| Method | Description | Requirements | Quality |
|--------|-------------|--------------|---------|
| Consistency Distillation (CD) | Distill from pre-trained diffusion | Teacher model | Higher |
| Consistency Training (CT) | Train from scratch | No teacher | Good |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- diffusers (for pre-trained models)
- einops
- torchdiffeq (for ODE solving)
- accelerate (for distributed training)

### Hardware
- GPU recommended for training
- Single GPU sufficient for inference
- Memory scales with model size

### External Resources
- [ ] Official implementation: [openai/consistency_models](https://github.com/openai/consistency_models)
- [ ] Pre-trained diffusion models for distillation
- [ ] LCM implementation: [luosiallen/latent-consistency-model](https://github.com/luosiallen/latent-consistency-model)

## Quick Start

```python
from consciousness.ml_research.modern_dev.consistency_models import CONSISTENCY_CONFIG, LCM_CONFIG

# Consistency Model Configuration
config = {
    "model_type": "unet",
    "image_size": 64,
    "num_channels": 128,
    "num_res_blocks": 2,
    "channel_mult": [1, 2, 2, 2],
    "attention_resolutions": [16, 8],
    "dropout": 0.0,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.5,
    "rho": 7,
    "num_timesteps": 18,
    "ema_rate": 0.999,
}

# Latent Consistency Model Configuration
lcm_config = {
    "base_model": "stabilityai/stable-diffusion-2-1",
    "lora_rank": 64,
    "num_inference_steps": 4,
    "guidance_scale": 7.5,
    "lcm_origin_steps": 50,
    "use_fp16": True,
}

# When implemented:
# from consciousness.ml_research.modern_dev.consistency_models.src.consistency import ConsistencyModel
# model = ConsistencyModel(**config)
```

## File Structure

```
consistency_models/
├── __init__.py       # Module documentation and metadata
├── README.md         # This file
├── src/
│   ├── consistency.py    # ConsistencyModel and ConsistencyConfig
│   ├── distillation.py   # ConsistencyDistillation
│   ├── training.py       # ConsistencyTraining
│   └── schedule.py       # TimeSchedule, KarrasSchedule
├── configs/
│   ├── consistency.yaml  # Standard CM config
│   └── lcm.yaml          # Latent Consistency Model config
├── cli/
│   ├── train.py      # Training script (cm-train)
│   ├── sample.py     # Fast sampling (cm-sample)
│   └── distill.py    # Distillation (cm-distill)
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**PF-ODE (defines trajectory):**
```
dx/dt = f(x, t) = -0.5 * beta(t) * [x + s(x, t)]

where s(x, t) = nabla_x log p_t(x) is the score
```

**Consistency Function:**
```
f: (R^d x [epsilon, T]) -> R^d

Properties:
1. f(x_t, t) = f(x_s, s)  for all (x_t, t), (x_s, s) on same trajectory
2. f(x, epsilon) = x      boundary condition
```

**Parameterization:**
```
f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)

c_skip(t) = sigma_data^2 / ((t - epsilon)^2 + sigma_data^2)
c_out(t) = sigma_data * (t - epsilon) / sqrt((t - epsilon)^2 + sigma_data^2)

This ensures f_theta(x, epsilon) = x automatically
```

**Consistency Distillation Loss:**
```
L_CD = E_{n, x}[d(f_theta(x_{t_{n+1}}, t_{n+1}), f_{theta-}(hat_x_{t_n}, t_n))]

where:
- hat_x_{t_n} = one ODE step from x_{t_{n+1}} using teacher
- theta- = EMA of theta
- d = distance metric (L2 or LPIPS)
```

**Consistency Training Loss:**
```
L_CT = E_{n, x}[d(f_theta(x + sigma_{t_{n+1}} * z, t_{n+1}),
                   f_{theta-}(x + sigma_{t_n} * z, t_n))]

Uses curriculum learning: gradually increase N (discretization steps)
```

## Training Algorithms

**Consistency Distillation:**
```python
for each training step:
    x ~ p_data
    n ~ Uniform{1, ..., N-1}
    x_{t_n+1} = sqrt(alpha_{t_n+1}) * x + sqrt(1-alpha_{t_n+1}) * eps

    # Teacher: ODE step using pre-trained model
    x_hat_{t_n} = ODE_step(x_{t_n+1}, s_phi, t_{n+1} -> t_n)

    # Consistency loss
    loss = ||f_theta(x_{t_n+1}, t_{n+1}) - f_{theta-}(x_hat_{t_n}, t_n)||

    theta = theta - lr * grad(loss)
    theta- = mu * theta- + (1-mu) * theta  # EMA update
```

**Consistency Training (from scratch):**
```python
# Similar but estimate ODE step using score matching
# No pre-trained model needed
# Uses curriculum: gradually increase N (discretization steps)
```

## Benchmarks

**Consistency Models (CIFAR-10):**

| Method | Steps | FID |
|--------|-------|-----|
| CD (Distillation) | 1 | 3.55 |
| CD (Distillation) | 2 | 2.93 |
| CT (Training) | 1 | 8.70 |

**ImageNet 64x64:**

| Method | Steps | FID |
|--------|-------|-----|
| CD (Distillation) | 1 | 6.20 |
| CD (Distillation) | 2 | 4.70 |

**Latent Consistency Models:**
- Stable Diffusion quality in 1-4 steps
- Real-time image synthesis possible
- Widely adopted in production

## Comparison with Other Methods

|                    | Consistency | Flow Matching | Diffusion |
|--------------------|-------------|---------------|-----------|
| One-Step Gen | Yes | No (few-step) | No |
| Training | CD or CT | Simulation-free | Score matching |
| Sampling Steps | 1-4 | 5-50 | 50-1000 |
| Quality (1-step) | High | N/A | N/A |
| Quality (multi) | Very High | High | Highest |
| Training Stability | Moderate | High | High |

## Related Work

| Method | Description | Paper |
|--------|-------------|-------|
| Latent Consistency Models | CM in SD latent space | [arXiv:2310.04378](https://arxiv.org/abs/2310.04378) |
| Progressive Distillation | Iteratively halve steps | [arXiv:2202.00512](https://arxiv.org/abs/2202.00512) |
| Rectified Flow | Flow straightening | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) |

## References

- Song, Y., et al. "Consistency Models" (2023). arXiv:2303.01469
- Luo, S., et al. "Latent Consistency Models" (2023). arXiv:2310.04378
- [Official OpenAI Implementation](https://github.com/openai/consistency_models)
- Related: DDPM, Score Matching, Progressive Distillation

## Contributing

To complete this implementation:

1. **Phase 1: Consistency Models Core**
   - Implement ConsistencyModel architecture
   - Create skip connection parameterization
   - Add time discretization schedule
   - Build basic training loop

2. **Phase 2: Consistency Training Methods**
   - Implement Consistency Distillation (with teacher)
   - Implement Consistency Training (from scratch)
   - Add curriculum learning for CT
   - Create EMA target network

3. **Phase 3: Latent Consistency Models**
   - Integrate with Stable Diffusion latent space
   - Add LoRA fine-tuning support
   - Create fast inference pipeline
   - Build real-time generation demo

4. **Phase 4: Evaluation**
   - Implement FID/IS computation
   - Create sampling quality comparison
   - Add step-wise quality analysis
