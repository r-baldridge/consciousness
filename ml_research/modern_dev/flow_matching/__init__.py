"""
Flow Matching for Generative Modeling - Meta AI / Various, 2022-2024

A framework for training continuous normalizing flows by regressing onto
target vector fields. Enables straighter generation trajectories than
diffusion models, resulting in faster and higher-quality sampling.

Paper: "Flow Matching for Generative Modeling"
arXiv: https://arxiv.org/abs/2210.02747 (Original)
Related: https://arxiv.org/abs/2302.00482 (Rectified Flow)
Authors: Yaron Lipman, Ricky T. Q. Chen, et al. (Meta AI)

=============================================================================
KEY INNOVATIONS
=============================================================================

1. OPTIMAL TRANSPORT FORMULATION
   - Learns vector field that transports noise to data
   - OT paths are STRAIGHT lines in expectation
   - Straighter paths = fewer sampling steps needed
   - No complex noise schedules required

2. SIMULATION-FREE TRAINING
   - No ODE/SDE simulation during training
   - Direct regression onto conditional vector field
   - Orders of magnitude faster than CNF training
   - Enables scaling to large models

3. RECTIFIED FLOW / REFLOW
   - Iterative straightening of flow paths
   - Each iteration produces straighter trajectories
   - One-step generation becomes possible
   - Quality improves with each reflow iteration

=============================================================================
CORE CONCEPT
=============================================================================

Diffusion Models:
    - Add noise gradually (forward process)
    - Learn to denoise (reverse process)
    - Curved trajectories in data space
    - Many steps needed (20-1000)

Flow Matching:
    - Define straight paths from noise to data
    - Learn vector field along these paths
    - Straighter trajectories = fewer steps
    - Can achieve high quality in 1-4 steps

    Trajectory Comparison:

    Diffusion:          Flow Matching:

    noise               noise
      |                   |
      ~                   |
       ~                  |
        ~                 |
         ~                |
    data                data
    (curved path)       (straight path)

=============================================================================
MATHEMATICAL FOUNDATION
=============================================================================

Probability Path:
    p_t(x) interpolates between p_0 (noise) and p_1 (data)

    For optimal transport:
        p_t(x) = (1-t) * p_0 + t * p_1  (in distribution sense)

Conditional Flow:
    x_t = (1-t) * x_0 + t * x_1

    where:
        x_0 ~ p_0 (noise, e.g., N(0,I))
        x_1 ~ p_1 (data)
        t in [0, 1]

Target Vector Field:
    u_t(x_t | x_1) = x_1 - x_0

    This is the CONSTANT velocity that moves x_0 to x_1

Flow Matching Objective:
    L_FM = E_{t, x_0, x_1}[ ||v_theta(x_t, t) - u_t(x_t | x_1)||^2 ]

    Train neural network v_theta to predict the velocity field

=============================================================================
COMPARISON WITH DIFFUSION
=============================================================================

                    Diffusion           Flow Matching
Forward Process     Noising SDE/ODE     Linear interpolation
Training Target     Score/noise         Velocity field
Path Shape          Curved              Straight (OT)
Typical Steps       20-50 (DDPM)        1-10 (FM)
                    1000 (original)
Training            Simulation-free     Simulation-free
Theory              Score matching      Optimal transport

Key Insight:
    - Diffusion: Forward = noise schedule, Reverse = learned
    - Flow: Both directions defined by same vector field
    - Straight paths enable larger step sizes

=============================================================================
RECTIFIED FLOW (REFLOW)
=============================================================================

Problem:
    - Initial flow paths can still be slightly curved
    - Individual paths cross each other
    - Limits one-step generation quality

Reflow Algorithm:
    1. Train initial flow matching model v_theta
    2. Generate pairs: (noise x_0, data x_1) by sampling
    3. Re-train on these COUPLED pairs
    4. New paths are straighter (don't cross)
    5. Repeat for even straighter paths

    After K reflows:
        - Paths are nearly straight
        - One-step generation approaches multi-step quality
        - Distillation-like effect without teacher

=============================================================================
ARCHITECTURE VARIANTS
=============================================================================

1. Basic Flow Matching:
   - UNet or Transformer backbone
   - Predict velocity v(x, t)
   - Euler or higher-order ODE solver

2. Conditional Flow Matching (CFM):
   - Conditions on data point x_1
   - Simpler optimization landscape
   - Equivalent marginal vector field

3. Optimal Transport CFM (OT-CFM):
   - Mini-batch optimal transport coupling
   - Even straighter paths within batch
   - Better for multi-modal distributions

4. Stochastic Interpolants:
   - Adds controlled noise to path
   - Bridges diffusion and flow matching
   - Flexible noise schedule

=============================================================================
APPLICATIONS IN MODERN SYSTEMS
=============================================================================

Stable Diffusion 3:
    - Uses flow matching (rectified flow)
    - Improved image quality
    - Faster sampling than SD 1.x/2.x

Flux:
    - Black Forest Labs model
    - Flow matching backbone
    - State-of-the-art image generation

Audio/Video:
    - Faster than diffusion for temporal data
    - Consistent trajectories across frames

3D Generation:
    - Point cloud flow matching
    - Mesh deformation flows

=============================================================================
IMPLEMENTATION DETAILS
=============================================================================

Training Loop:
    ```
    for batch in dataloader:
        x_1 = batch  # Real data
        x_0 = torch.randn_like(x_1)  # Noise
        t = torch.rand(batch_size)  # Time

        # Linear interpolation
        x_t = (1 - t) * x_0 + t * x_1

        # Target velocity (constant for OT)
        u_t = x_1 - x_0

        # Predict velocity
        v_pred = model(x_t, t)

        # Loss
        loss = F.mse_loss(v_pred, u_t)
        loss.backward()
        optimizer.step()
    ```

Sampling (Euler):
    ```
    x = torch.randn(shape)  # Start from noise

    for t in linspace(0, 1, num_steps):
        v = model(x, t)
        x = x + v * dt  # Euler step

    return x  # Generated sample
    ```

=============================================================================
PERFORMANCE BENCHMARKS
=============================================================================

ImageNet 256x256 (FID):
    Method              Steps   FID
    DDPM                1000    3.17
    Flow Matching       250     2.95
    Rectified Flow      1       8.48
    Rectified Flow      10      2.58

Key Results:
    - Matches diffusion quality with fewer steps
    - One-step generation competitive with multi-step diffusion
    - Training time similar to diffusion
    - Inference 10-100x faster

=============================================================================
IMPLEMENTATION REQUIREMENTS
=============================================================================

Dependencies:
    - PyTorch >= 2.0
    - torchdiffeq (optional, for advanced ODE solvers)
    - einops
    - Standard image processing (PIL, torchvision)

Key Components to Implement:
    1. VelocityNetwork (UNet/Transformer)
    2. FlowMatchingTrainer
    3. ODESampler (Euler, Heun, RK4)
    4. ReflowTrainer (for straightening)
    5. OTCoupler (optimal transport pairing)

=============================================================================
ADVANCED TOPICS
=============================================================================

Guidance:
    - Classifier-free guidance works with flow matching
    - v_guided = v_uncond + w * (v_cond - v_uncond)
    - Same principle as diffusion guidance

Multi-Modal Coupling:
    - Standard: Random pairing of noise and data
    - OT: Mini-batch optimal transport pairing
    - Improves path straightness significantly

Continuous Normalizing Flows Connection:
    - Flow matching is efficient CNF training
    - Avoids costly trace computation
    - Same generative model, better training

=============================================================================
DEVELOPMENT ROADMAP
=============================================================================

Phase 1: Core Implementation
    - [ ] Velocity network (UNet backbone)
    - [ ] Basic flow matching trainer
    - [ ] Euler sampler
    - [ ] MNIST/CIFAR proof of concept

Phase 2: Advanced Features
    - [ ] Rectified flow / reflow
    - [ ] OT-CFM coupling
    - [ ] Higher-order ODE solvers
    - [ ] Classifier-free guidance

Phase 3: Scaling
    - [ ] Transformer backbone (DiT-style)
    - [ ] ImageNet training
    - [ ] Distributed training support
    - [ ] Mixed precision

Phase 4: CLI Tools
    - [ ] flow-train: Training script
    - [ ] flow-sample: Sampling with various solvers
    - [ ] flow-reflow: Straightening pipeline
    - [ ] flow-evaluate: FID/IS computation
"""

__version__ = "0.1.0"
__status__ = "indexed"

# Architecture metadata
ARCHITECTURE_INFO = {
    "name": "Flow Matching",
    "abbreviation": "FM",
    "year": 2022,
    "organization": "Meta AI",
    "paper_url": "https://arxiv.org/abs/2210.02747",
    "related_papers": [
        {"title": "Rectified Flow", "url": "https://arxiv.org/abs/2209.03003"},
        {"title": "Stochastic Interpolants", "url": "https://arxiv.org/abs/2303.08797"},
    ],
    "github_url": None,  # Various implementations exist
    "authors": ["Yaron Lipman", "Ricky T. Q. Chen", "Heli Ben-Hamu", "Maximilian Nickel"],
    "key_contribution": "Simulation-free training of flows with optimal transport paths",
}

# Mathematical formulation
FORMULATION = """
Flow Matching Objective:

Probability Path:
    p_t = (1-t) * p_noise + t * p_data  (marginal distribution)

Conditional Flow (given data point x_1):
    x_t = (1-t) * x_0 + t * x_1

    where x_0 ~ N(0, I), x_1 ~ p_data

Conditional Vector Field:
    u_t(x | x_1) = (x_1 - x_0) = (x_1 - x) / (1 - t)  [for t < 1]

    Equivalently: u_t = x_1 - x_0 (constant velocity)

Flow Matching Loss:
    L_FM(theta) = E_{t~U[0,1], x_0~N(0,I), x_1~p_data}[
        || v_theta(x_t, t) - (x_1 - x_0) ||^2
    ]

Sampling via ODE:
    dx/dt = v_theta(x, t)
    x(0) ~ N(0, I)
    x(1) = generated sample

Rectified Flow (Reflow):
    Given trained v_theta, generate coupled pairs:
        x_0 ~ N(0, I)
        x_1 = ODE_solve(v_theta, x_0, t: 0->1)

    Retrain on coupled (x_0, x_1) pairs
    New paths are straighter (no crossing)
"""

# Default configuration
DEFAULT_CONFIG = {
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
SOLVER_CONFIGS = {
    "euler": {
        "order": 1,
        "function_evals_per_step": 1,
        "recommended_steps": 50,
    },
    "heun": {
        "order": 2,
        "function_evals_per_step": 2,
        "recommended_steps": 25,
    },
    "rk4": {
        "order": 4,
        "function_evals_per_step": 4,
        "recommended_steps": 10,
    },
    "dopri5": {
        "order": 5,
        "function_evals_per_step": "adaptive",
        "recommended_steps": "auto",
    },
}

# Comparison with diffusion
COMPARISON_WITH_DIFFUSION = {
    "training_objective": {
        "diffusion": "Predict noise or score",
        "flow_matching": "Predict velocity field",
    },
    "path_shape": {
        "diffusion": "Curved (follows noise schedule)",
        "flow_matching": "Straight (optimal transport)",
    },
    "sampling_steps": {
        "diffusion": "20-50 typical (DDIM), 1000 original",
        "flow_matching": "1-10 typical after reflow",
    },
    "theory": {
        "diffusion": "Score matching, denoising",
        "flow_matching": "Optimal transport, CNF",
    },
}

# Placeholder imports
# from .src.velocity_network import VelocityUNet, VelocityTransformer
# from .src.trainer import FlowMatchingTrainer
# from .src.sampler import ODESampler, euler_step, heun_step
# from .src.reflow import ReflowTrainer
# from .cli.train import main as train
# from .cli.sample import main as sample
