"""
Learning Rate Schedulers

Methods for adapting the learning rate during training to improve
convergence and final model performance.

Key Methods:
    - Step Decay: Reduce LR at fixed epochs
    - Cosine Annealing: Smooth decay following cosine curve
    - Warmup: Gradually increase LR at start of training
    - OneCycleLR: Warmup + annealing with momentum scheduling
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Step Decay
# =============================================================================

STEP_DECAY = MLMethod(
    method_id="step_decay",
    name="Step Decay (StepLR)",
    year=1998,
    era=MethodEra.CLASSICAL,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Various"],
    paper_title="Standard practice in neural network training",
    paper_url=None,
    key_innovation="""
    Reduces the learning rate by a fixed factor at predetermined epochs.
    Simple and effective, widely used in image classification where
    learning rate is typically divided by 10 at specific milestones.
    """,
    mathematical_formulation="""
    Step Decay:
    -----------

    Basic Step Decay:
        lr_t = lr_0 * gamma^(floor(epoch / step_size))

    Where:
        lr_0        = initial learning rate
        gamma       = decay factor (typically 0.1)
        step_size   = epochs between decays
        epoch       = current epoch number

    Multi-Step Decay:
        lr_t = lr_0 * gamma^n

        where n = number of milestones <= current epoch

    Example (ImageNet training):
        milestones = [30, 60, 90]
        gamma = 0.1

        epochs 0-29:   lr = 0.1
        epochs 30-59:  lr = 0.01
        epochs 60-89:  lr = 0.001
        epochs 90+:    lr = 0.0001

    Variants:
        1. Fixed schedule: decay at epochs [30, 60, 90]
        2. Regular intervals: decay every N epochs
        3. Exponential: lr = lr_0 * gamma^epoch (continuous)

    Exponential Decay (continuous version):
        lr_t = lr_0 * exp(-k * t)
        lr_t = lr_0 * gamma^t
    """,
    predecessors=[],
    successors=["cosine_annealing", "warmup"],
    tags=["learning-rate", "scheduler", "decay", "step"],
    notes="""
    Typical Usage:
    - ImageNet: Decay by 10x at epochs 30, 60, 90
    - CIFAR: Decay by 5x at epochs 60, 120, 160

    Advantages:
    - Simple to implement and understand
    - Proven effective for many tasks
    - Interpretable schedule

    Disadvantages:
    - Requires manual tuning of milestones
    - Abrupt changes may cause training instability
    - Not adaptive to training progress

    PyTorch:
        torch.optim.lr_scheduler.StepLR
        torch.optim.lr_scheduler.MultiStepLR
    """
)


def step_decay_schedule(
    epoch: int,
    initial_lr: float,
    step_size: int = 30,
    gamma: float = 0.1
) -> float:
    """
    Compute learning rate with step decay.

    Pseudocode:
        lr = initial_lr * gamma^(epoch // step_size)
    """
    # return initial_lr * (gamma ** (epoch // step_size))
    pass


def multi_step_schedule(
    epoch: int,
    initial_lr: float,
    milestones: list,
    gamma: float = 0.1
) -> float:
    """
    Compute learning rate with multi-step decay.

    Pseudocode:
        num_decays = sum(1 for m in milestones if epoch >= m)
        lr = initial_lr * gamma^num_decays
    """
    pass


# =============================================================================
# Cosine Annealing
# =============================================================================

COSINE_ANNEALING = MLMethod(
    method_id="cosine_annealing",
    name="Cosine Annealing",
    year=2016,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Ilya Loshchilov", "Frank Hutter"],
    paper_title="SGDR: Stochastic Gradient Descent with Warm Restarts",
    paper_url="https://arxiv.org/abs/1608.03983",
    key_innovation="""
    Smoothly decreases learning rate following a cosine curve from initial
    value to near zero. Provides gradual annealing without abrupt changes,
    and can be combined with warm restarts for periodic learning rate cycles.
    """,
    mathematical_formulation="""
    Cosine Annealing:
    -----------------

    Basic Cosine Annealing:
        lr_t = lr_min + (lr_max - lr_min) * (1 + cos(pi * t / T)) / 2

    Where:
        lr_max      = initial/maximum learning rate
        lr_min      = minimum learning rate (often 0 or small value)
        t           = current step/epoch
        T           = total number of steps/epochs

    Simplified (lr_min = 0):
        lr_t = lr_max * (1 + cos(pi * t / T)) / 2

    At key points:
        t = 0:      lr = lr_max (starts at maximum)
        t = T/2:    lr = (lr_max + lr_min) / 2 (halfway)
        t = T:      lr = lr_min (ends at minimum)

    Cosine Annealing with Warm Restarts (SGDR):
        After each cycle of T_i epochs, restart from lr_max.

        T_{i+1} = T_i * T_mult    (cycles can grow)

        Within cycle i at epoch t:
            lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * T_cur / T_i)) / 2

        where T_cur = epochs since last restart

    Example (with restarts):
        T_0 = 10, T_mult = 2
        Cycle 1: epochs 0-9 (length 10)
        Cycle 2: epochs 10-29 (length 20)
        Cycle 3: epochs 30-69 (length 40)
    """,
    predecessors=["step_decay"],
    successors=["one_cycle_lr"],
    tags=["learning-rate", "scheduler", "cosine", "smooth", "warm-restarts"],
    notes="""
    Advantages:
    - Smooth decay (no abrupt changes)
    - Warm restarts can help escape local minima
    - Easy to tune (just T and lr_max)

    Popular Configurations:
    1. Single cycle: Cosine from lr_max to 0 over training
    2. SGDR: Multiple cycles with restarts
    3. Linear warmup + cosine decay (very common)

    PyTorch:
        torch.optim.lr_scheduler.CosineAnnealingLR
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    Typical Usage:
    - Vision: Often with linear warmup
    - Transformers: Warmup + cosine is standard
    - Fine-tuning: Shorter cosine schedule

    Empirical Results:
    - Generally better than step decay
    - Warm restarts help for longer training
    - Works well with SGD and Adam
    """
)


def cosine_annealing_schedule(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0
) -> float:
    """
    Compute learning rate with cosine annealing.

    Pseudocode:
        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * step / total_steps)) / 2
    """
    # import math
    # return lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps)) / 2
    pass


def cosine_with_restarts_schedule(
    step: int,
    lr_max: float,
    lr_min: float,
    T_0: int,
    T_mult: int = 1
) -> float:
    """
    Compute learning rate with cosine annealing and warm restarts.

    Pseudocode:
        # Find current cycle and position within cycle
        if T_mult == 1:
            cycle = step // T_0
            t_cur = step % T_0
            T_i = T_0
        else:
            # Compute which cycle we're in
            # T_0 + T_0*T_mult + T_0*T_mult^2 + ... >= step
            ...

        lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * t_cur / T_i)) / 2
    """
    pass


# =============================================================================
# Warmup
# =============================================================================

WARMUP = MLMethod(
    method_id="warmup",
    name="Learning Rate Warmup",
    year=2017,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Priya Goyal", "Piotr Dollar", "Ross Girshick", "Pieter Noordhuis",
             "Lukasz Wesolowski", "Aapo Kyrola", "Andrew Tulloch",
             "Yangqing Jia", "Kaiming He"],
    paper_title="Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour",
    paper_url="https://arxiv.org/abs/1706.02677",
    key_innovation="""
    Gradually increases learning rate from a small value to the target value
    during the first few epochs/steps. Stabilizes training in the critical
    early phase when parameters are far from optimal and gradients are noisy.
    """,
    mathematical_formulation="""
    Learning Rate Warmup:
    ---------------------

    Linear Warmup:
        lr_t = lr_target * (t / warmup_steps)    for t < warmup_steps
        lr_t = lr_target                         for t >= warmup_steps

    Or more precisely:
        lr_t = lr_target * min(1, t / warmup_steps)

    Warmup + Linear Decay:
        if t < warmup_steps:
            lr = lr_max * t / warmup_steps
        else:
            lr = lr_max * (1 - (t - warmup_steps) / (total_steps - warmup_steps))

    Warmup + Cosine Decay (common for transformers):
        if t < warmup_steps:
            lr = lr_max * t / warmup_steps
        else:
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * progress)) / 2

    Gradual Warmup (for very large batches):
        epoch 1: lr = lr_target / k
        epoch 2: lr = 2 * lr_target / k
        ...
        epoch k: lr = lr_target

    Why Warmup Helps:
        - Early gradients are noisy (random initialization)
        - Large LR + noisy gradients = unstable updates
        - Gradual increase allows network to find stable region
        - Especially important for large batch training
    """,
    predecessors=["step_decay"],
    successors=["one_cycle_lr"],
    tags=["learning-rate", "warmup", "large-batch", "transformer"],
    notes="""
    When Warmup is Essential:
    - Large batch training
    - Transformers (almost always use warmup)
    - Large learning rates
    - Complex architectures

    Typical Warmup Length:
    - Vision: 5 epochs or ~5% of training
    - Transformers: 1-10% of total steps
    - Large batch: Longer warmup needed

    Warmup + Scheduler Combinations:
    - Warmup + step decay
    - Warmup + cosine annealing (most common)
    - Warmup + linear decay

    Implementation:
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        for step in range(total_steps):
            if step < warmup_steps:
                lr = max_lr * step / warmup_steps
            else:
                lr = scheduler.get_lr(step - warmup_steps)
    """
)


def warmup_schedule(
    step: int,
    warmup_steps: int,
    target_lr: float,
    initial_lr: float = 0.0
) -> float:
    """
    Compute learning rate during warmup phase.

    Pseudocode:
        if step < warmup_steps:
            lr = initial_lr + (target_lr - initial_lr) * step / warmup_steps
        else:
            lr = target_lr
    """
    # if step < warmup_steps:
    #     return initial_lr + (target_lr - initial_lr) * step / warmup_steps
    # else:
    #     return target_lr
    pass


def warmup_cosine_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float = 0.0
) -> float:
    """
    Compute learning rate with warmup followed by cosine decay.

    Pseudocode:
        if step < warmup_steps:
            lr = lr_max * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * progress)) / 2
    """
    pass


# =============================================================================
# OneCycleLR
# =============================================================================

ONE_CYCLE_LR = MLMethod(
    method_id="one_cycle_lr",
    name="OneCycleLR (1cycle Policy)",
    year=2018,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Leslie N. Smith", "Nicholay Topin"],
    paper_title="Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates",
    paper_url="https://arxiv.org/abs/1708.07120",
    key_innovation="""
    Combines learning rate warmup and annealing with cyclical momentum
    scheduling. Uses a high maximum learning rate discovered via LR range
    test, enabling "super-convergence" - training to high accuracy in
    significantly fewer epochs.
    """,
    mathematical_formulation="""
    OneCycleLR (1cycle Policy):
    ---------------------------

    The schedule has three phases:

    Phase 1 - Warmup (typically 30-45% of training):
        lr: lr_min -> lr_max (increase)
        momentum: mom_max -> mom_min (decrease)

    Phase 2 - High LR (symmetric with Phase 1):
        lr: lr_max -> lr_min (decrease)
        momentum: mom_min -> mom_max (increase)

    Phase 3 - Annihilation (remaining steps):
        lr: lr_min -> lr_min/100 or lower (decrease further)
        momentum: mom_max (stay constant)

    Learning Rate:
        Phase 1 & 2 (first 2*pct_start of steps):
            lr = lr_min + (lr_max - lr_min) * (1 - |1 - 2*progress|)

            where progress = step / (2 * pct_start * total_steps)

        Phase 3 (remaining steps):
            Cosine annealing from lr_min to final_lr

    Momentum (inverse relationship):
        When lr is high -> momentum is low (0.85)
        When lr is low -> momentum is high (0.95)

    Typical Parameters:
        lr_max: Found via LR range test (e.g., 0.1)
        lr_min: lr_max / 25 (e.g., 0.004)
        final_lr: lr_min / 1e4 (e.g., 4e-7)
        pct_start: 0.3 (30% warmup)
        mom_max: 0.95
        mom_min: 0.85

    Super-Convergence:
        Using high lr_max (discovered via LR range test) enables
        training in far fewer epochs with similar or better accuracy.
        Example: ResNet on CIFAR-10 in 25 epochs vs 200+ typical.
    """,
    predecessors=["cosine_annealing", "warmup", "lr_range_test"],
    successors=[],
    tags=["learning-rate", "scheduler", "super-convergence", "momentum", "fast-training"],
    notes="""
    Key Components:
    1. LR warmup to high value
    2. LR annealing back down
    3. Momentum anti-cycling (opposite to LR)
    4. Final annealing phase

    Why It Works:
    - High LR acts as regularization
    - Escapes sharp minima (finds flatter optima)
    - Momentum cycling prevents oscillations
    - Final phase finds precise optimum

    Typical Usage:
        lr_max = find_lr()  # LR range test
        scheduler = OneCycleLR(
            max_lr=lr_max,
            total_steps=total_epochs * steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,      # initial_lr = max_lr/25
            final_div_factor=1e4  # final_lr = initial_lr/1e4
        )

    Results:
    - CIFAR-10: 94% in 25 epochs (vs 200+ standard)
    - ImageNet: Competitive results, much faster
    - Works with SGD (best) and Adam (good)

    PyTorch:
        torch.optim.lr_scheduler.OneCycleLR
    """
)


def one_cycle_schedule(
    step: int,
    total_steps: int,
    lr_max: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
) -> tuple:
    """
    Compute learning rate and momentum for OneCycleLR.

    Returns:
        Tuple of (learning_rate, momentum)

    Pseudocode:
        lr_min = lr_max / div_factor
        final_lr = lr_min / final_div_factor

        warmup_steps = int(pct_start * total_steps)
        anneal_steps = total_steps - 2 * warmup_steps

        if step < warmup_steps:
            # Phase 1: warmup
            progress = step / warmup_steps
            lr = lr_min + (lr_max - lr_min) * progress
            momentum = 0.95 - (0.95 - 0.85) * progress
        elif step < 2 * warmup_steps:
            # Phase 2: mirror of warmup
            progress = (step - warmup_steps) / warmup_steps
            lr = lr_max - (lr_max - lr_min) * progress
            momentum = 0.85 + (0.95 - 0.85) * progress
        else:
            # Phase 3: final annealing
            progress = (step - 2*warmup_steps) / anneal_steps
            lr = lr_min - (lr_min - final_lr) * (1 - cos(pi * progress)) / 2
            momentum = 0.95
    """
    pass


# =============================================================================
# Comparison
# =============================================================================

LR_SCHEDULER_COMPARISON = """
Learning Rate Scheduler Comparison:
===================================

| Scheduler       | Pattern          | Best For                    |
|-----------------|------------------|-----------------------------|
| Step Decay      | Discrete drops   | Simple training, baselines  |
| Cosine Annealing| Smooth cosine    | General purpose, fine-tuning|
| Warmup + Cosine | Ramp up + cosine | Transformers, large batch   |
| OneCycleLR      | Up + down + low  | Fast training, SGD          |

Typical Combinations:
    CNNs (long training):     Step decay or Cosine
    CNNs (fast training):     OneCycleLR
    Transformers:             Warmup + Cosine decay
    Fine-tuning:              Short cosine or constant

Modern Best Practices:
    Vision:
        - SGD + momentum + weight decay + OneCycleLR
        - SGD + cosine annealing (long training)

    Transformers:
        - AdamW + warmup (10% of steps) + cosine decay
        - Linear warmup + linear decay (BERT-style)

    General:
        - Always consider warmup for large models
        - LR range test to find max_lr
        - Cosine generally better than step
"""


def get_all_scheduler_methods():
    """Return all LR scheduler MLMethod entries."""
    return [STEP_DECAY, COSINE_ANNEALING, WARMUP, ONE_CYCLE_LR]
