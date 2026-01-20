# Flow Matching for Generative Modeling

> A framework for training continuous normalizing flows by regressing onto target vector fields, enabling straighter generation trajectories than diffusion models for faster and higher-quality sampling.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
**Year:** 2022
**Organization:** Meta AI

## Overview

Flow Matching provides an elegant alternative to diffusion models for generative modeling. While diffusion models add noise gradually and learn to reverse this process through curved trajectories in data space, Flow Matching defines straight paths from noise to data using optimal transport principles and learns the velocity field along these paths. The result is significantly faster sampling-achieving high quality in 1-10 steps compared to diffusion's 20-1000 steps.

The key insight is that optimal transport paths between noise and data distributions are straight lines in expectation. By training a neural network to predict the velocity field along these paths (a simple regression problem), we can generate samples by integrating this velocity field from noise to data. Unlike diffusion models that require complex noise schedules, Flow Matching uses simple linear interpolation between source and target.

Rectified Flow (Reflow) takes this further by iteratively straightening flow trajectories. After training an initial flow, we generate coupled noise-data pairs and retrain on these, producing even straighter paths. This process can be repeated until one-step generation approaches multi-step quality. Flow Matching now powers state-of-the-art image generation systems including Stable Diffusion 3 and Flux.

## Key Innovations

- **Optimal Transport Formulation**: Learns velocity fields that transport noise to data along straight (OT) paths, requiring fewer sampling steps than curved diffusion trajectories.

- **Simulation-Free Training**: Direct regression onto conditional velocity fields without expensive ODE/SDE simulation, enabling scaling to large models.

- **Rectified Flow (Reflow)**: Iterative trajectory straightening that enables one-step generation with quality approaching multi-step sampling.

## Architecture Diagram

```
Diffusion Models:          Flow Matching:

    noise                     noise
      |                         |
      ~                         |
       ~                        |
        ~                       |
         ~                      |
    data                      data
    (curved path)            (straight path)

Flow Matching Training:
    +---------------------------------------------------+
    |                                                    |
    |  x_0 ~ N(0, I)    x_1 ~ p_data                   |
    |        |              |                           |
    |        +------+-------+                           |
    |               |                                   |
    |               v                                   |
    |         x_t = (1-t) * x_0 + t * x_1              |
    |               |                                   |
    |               v                                   |
    |     +-------------------+                         |
    |     |  Velocity Network | --> v_pred              |
    |     |   v_theta(x_t, t) |                        |
    |     +-------------------+                         |
    |               |                                   |
    |               v                                   |
    |         Target: u_t = x_1 - x_0                  |
    |               |                                   |
    |               v                                   |
    |         Loss = ||v_pred - u_t||^2                |
    +---------------------------------------------------+

Sampling (Euler):
    x = noise
    for t in [0, 1]:
        v = model(x, t)
        x = x + v * dt
    return x  # generated sample
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| VelocityNetwork | Stub | UNet/Transformer backbone |
| FlowMatchingTrainer | Stub | CFM loss implementation |
| ODESampler | Stub | Euler, Heun, RK4 solvers |
| ReflowTrainer | Stub | Trajectory straightening |
| OTCoupler | Stub | Optimal transport pairing |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | Default and solver configs |

## Flow Matching Variants

| Variant | Description | Key Feature |
|---------|-------------|-------------|
| Basic FM | Standard flow matching | Simple linear paths |
| CFM | Conditional Flow Matching | Tractable training |
| OT-CFM | Optimal Transport CFM | Mini-batch OT coupling |
| Rectified Flow | Iterative straightening | One-step generation |
| Stochastic Interpolants | Bridge diffusion and FM | Flexible noise schedule |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- torchdiffeq (optional, for advanced ODE solvers)
- einops
- torchvision (for image processing)

### Hardware
- GPU recommended for training
- Single GPU sufficient for inference
- Memory scales with model size (UNet/Transformer)

### External Resources
- [ ] Reference implementations in diffusers library
- [ ] Training data: ImageNet, CIFAR-10, etc.
- [ ] FID/IS evaluation tools

## Quick Start

```python
from consciousness.ml_research.modern_dev.flow_matching import DEFAULT_CONFIG, SOLVER_CONFIGS

# Default Configuration
config = {
    # Model
    "model_type": "unet",  # or "transformer"
    "hidden_channels": 256,
    "num_res_blocks": 2,
    "attention_resolutions": [16, 8],
    "channel_mult": [1, 2, 4, 8],

    # Training
    "learning_rate": 1e-4,
    "batch_size": 64,
    "num_iterations": 100000,
    "ema_decay": 0.9999,

    # Sampling
    "num_steps": 50,
    "solver": "euler",  # euler, heun, rk4

    # Data
    "image_size": 64,
    "num_channels": 3,
}

# Solver configurations
# SOLVER_CONFIGS["euler"] -> 1 function eval per step
# SOLVER_CONFIGS["heun"] -> 2 function evals per step
# SOLVER_CONFIGS["rk4"] -> 4 function evals per step

# When implemented:
# from consciousness.ml_research.modern_dev.flow_matching.src.trainer import FlowMatchingTrainer
# trainer = FlowMatchingTrainer(**config)
```

## File Structure

```
flow_matching/
├── __init__.py       # Module documentation and metadata
├── README.md         # This file
├── src/
│   ├── velocity_network.py  # VelocityUNet, VelocityTransformer
│   ├── trainer.py           # FlowMatchingTrainer
│   ├── sampler.py           # ODESampler, euler_step, heun_step
│   └── reflow.py            # ReflowTrainer
├── configs/
│   ├── flow_matching.yaml   # Default configuration
│   └── reflow.yaml          # Rectified flow config
├── cli/
│   ├── train.py      # Training script (flow-train)
│   ├── sample.py     # Sampling (flow-sample)
│   ├── reflow.py     # Straightening (flow-reflow)
│   └── evaluate.py   # FID/IS computation
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**Probability Path:**
```
p_t = (1-t) * p_noise + t * p_data  (marginal distribution)
```

**Conditional Flow (given data point x_1):**
```
x_t = (1-t) * x_0 + t * x_1

where x_0 ~ N(0, I), x_1 ~ p_data
```

**Conditional Vector Field:**
```
u_t(x | x_1) = (x_1 - x_0) = (x_1 - x) / (1 - t)  [for t < 1]

Equivalently: u_t = x_1 - x_0 (constant velocity)
```

**Flow Matching Loss:**
```
L_FM(theta) = E_{t~U[0,1], x_0~N(0,I), x_1~p_data}[
    || v_theta(x_t, t) - (x_1 - x_0) ||^2
]
```

**Sampling via ODE:**
```
dx/dt = v_theta(x, t)
x(0) ~ N(0, I)
x(1) = generated sample
```

**Rectified Flow (Reflow):**
```
Given trained v_theta, generate coupled pairs:
    x_0 ~ N(0, I)
    x_1 = ODE_solve(v_theta, x_0, t: 0->1)

Retrain on coupled (x_0, x_1) pairs
New paths are straighter (no crossing)
```

## ODE Solvers

| Solver | Order | Evals/Step | Recommended Steps |
|--------|-------|------------|-------------------|
| Euler | 1 | 1 | 50 |
| Heun | 2 | 2 | 25 |
| RK4 | 4 | 4 | 10 |
| DOPRI5 | 5 | Adaptive | Auto |

## Benchmarks

**ImageNet 256x256 (FID):**

| Method | Steps | FID |
|--------|-------|-----|
| DDPM | 1000 | 3.17 |
| Flow Matching | 250 | 2.95 |
| Rectified Flow | 1 | 8.48 |
| Rectified Flow | 10 | 2.58 |

**Key Results:**
- Matches diffusion quality with fewer steps
- One-step generation competitive with multi-step diffusion
- Training time similar to diffusion
- Inference 10-100x faster

## Comparison with Diffusion

|                    | Flow Matching | Diffusion |
|--------------------|---------------|-----------|
| Training Objective | Predict velocity | Predict noise/score |
| Path Shape | Straight (OT) | Curved (noise schedule) |
| Sampling Steps | 1-10 typical | 20-50 typical |
| Training | Simulation-free | Simulation-free |
| Theory | Optimal transport | Score matching |

## Applications

**Production Systems:**
- **Stable Diffusion 3**: Uses flow matching (rectified flow)
- **Flux**: Black Forest Labs flow-based model
- State-of-the-art image generation quality

**Modalities:**
- Image generation (primary use case)
- Audio/video synthesis
- 3D generation (point clouds, meshes)
- Molecular design

## References

- Lipman, Y., et al. "Flow Matching for Generative Modeling" (2022). arXiv:2210.02747
- Liu, X., et al. "Rectified Flow" (2022). arXiv:2209.03003
- Albergo, M., et al. "Stochastic Interpolants" (2023). arXiv:2303.08797
- Related: Continuous Normalizing Flows, Score Matching

## Contributing

To complete this implementation:

1. **Phase 1: Core Implementation**
   - Implement velocity network (UNet backbone)
   - Create basic flow matching trainer
   - Build Euler sampler
   - Test on MNIST/CIFAR

2. **Phase 2: Advanced Features**
   - Implement rectified flow / reflow training
   - Add OT-CFM mini-batch coupling
   - Create higher-order ODE solvers (Heun, RK4)
   - Add classifier-free guidance support

3. **Phase 3: Scaling**
   - Implement Transformer backbone (DiT-style)
   - Add ImageNet training support
   - Enable distributed training
   - Add mixed precision training

4. **Phase 4: Evaluation**
   - Implement FID/IS computation
   - Create sample visualization tools
   - Add trajectory straightness metrics
