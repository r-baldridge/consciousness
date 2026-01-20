"""
Learning Rate Range Test (LR Finder)

Smith's method for finding the optimal learning rate by gradually
increasing LR and monitoring the loss.

Key Method:
    - LR Range Test: Systematic method to find optimal learning rate
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Learning Rate Range Test
# =============================================================================

LR_RANGE_TEST = MLMethod(
    method_id="lr_range_test",
    name="Learning Rate Range Test (LR Finder)",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Leslie N. Smith"],
    paper_title="Cyclical Learning Rates for Training Neural Networks",
    paper_url="https://arxiv.org/abs/1506.01186",
    key_innovation="""
    Provides a systematic method to find the optimal learning rate by
    training for a short period while exponentially increasing the learning
    rate. The optimal LR is typically where loss decreases fastest (steepest
    negative slope), before it starts increasing.
    """,
    mathematical_formulation="""
    Learning Rate Range Test:
    -------------------------

    Setup:
        lr_min = starting learning rate (e.g., 1e-7)
        lr_max = ending learning rate (e.g., 10)
        num_iter = number of iterations (e.g., 100-300)

    LR Schedule (exponential increase):
        lr_t = lr_min * (lr_max / lr_min)^(t / num_iter)

        Or equivalently:
        lr_t = lr_min * exp(t * log(lr_max / lr_min) / num_iter)

    Linear Schedule (alternative):
        lr_t = lr_min + (lr_max - lr_min) * (t / num_iter)

    Algorithm:
        1. Initialize model weights (or use pretrained)
        2. For each batch:
            a. Compute loss
            b. Update model with current lr
            c. Record (lr, loss) pair
            d. Increase lr exponentially
            e. Stop if loss explodes (e.g., > 4 * minimum loss)
        3. Plot loss vs lr (log scale for lr)
        4. Find optimal lr from plot

    Finding Optimal LR:
        Method 1 - Steepest Descent:
            optimal_lr = lr where d(loss)/d(log(lr)) is most negative
            i.e., where loss is decreasing fastest

        Method 2 - Minimum Loss:
            optimal_lr = lr at minimum loss / 10
            (slightly before minimum to have safety margin)

        Method 3 - Valley:
            If using for cyclical LR, find the range where loss decreases:
            lr_min = start of descent
            lr_max = point before loss increases

    Smoothing (recommended):
        Apply exponential moving average to loss:
        smoothed_loss_t = beta * smoothed_loss_{t-1} + (1 - beta) * loss_t
        where beta ~ 0.98

    For OneCycleLR:
        max_lr = optimal_lr * 10  (or point where loss starts increasing)
        min_lr = max_lr / 25

    Pseudocode:
        losses = []
        lrs = []
        best_loss = infinity
        avg_loss = 0
        beta = 0.98

        for batch_idx, (x, y) in enumerate(dataloader):
            # Compute current lr (exponential increase)
            lr = lr_min * (lr_max / lr_min) ** (batch_idx / num_batches)
            set_learning_rate(optimizer, lr)

            # Forward + backward
            loss = compute_loss(model(x), y)
            loss.backward()
            optimizer.step()

            # Smooth the loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (batch_idx + 1))

            # Record
            losses.append(smoothed_loss)
            lrs.append(lr)

            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss:
                break
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

        # Plot losses vs lrs (log scale for x-axis)
        plot(lrs, losses, xscale='log')
    """,
    predecessors=[],
    successors=["one_cycle_lr"],
    tags=["learning-rate", "hyperparameter-tuning", "practical", "diagnostic"],
    notes="""
    When to Use:
    - Starting a new project
    - New architecture
    - New dataset
    - Before using cyclical LR or OneCycleLR

    Interpretation of Plot:
        Loss vs LR (log scale):

                |
           loss |  ----___
                |         \___
                |             \____  <- optimal region
                |                  \______
                |                         \__
                |                            |
                +-----------------------------+---> lr (log)
               low                          high

        - Start: Loss relatively flat (LR too small)
        - Descent: Loss decreasing (good LR range)
        - Valley: Optimal learning rate region
        - Explosion: Loss increases rapidly (LR too large)

    Practical Tips:
    1. Run for ~100-300 iterations (1 epoch often sufficient)
    2. Use same batch size as training
    3. Reset model weights before actual training
    4. Use smoothed loss for cleaner curve
    5. Stop early when loss explodes

    For Different Optimizers:
        SGD: LR range typically 0.01 - 1.0
        Adam: LR range typically 1e-4 - 1e-2
        AdamW: Similar to Adam

    Libraries:
        - fastai: learn.lr_find()
        - PyTorch: torch-lr-finder package
        - Manual implementation (straightforward)

    Gotchas:
    - Different from optimal LR for full training
    - Provides upper bound for max_lr
    - May need adjustment for learning rate warmup
    - Results vary with batch size
    """
)


def lr_range_test(
    model,
    dataloader,
    lr_min: float = 1e-7,
    lr_max: float = 10.0,
    num_iter: int = 100,
    smooth_factor: float = 0.98,
    diverge_threshold: float = 4.0
) -> tuple:
    """
    Pseudocode for Learning Rate Range Test.

    Args:
        model: Neural network model
        dataloader: Training data loader
        lr_min: Starting learning rate
        lr_max: Ending learning rate
        num_iter: Number of iterations
        smooth_factor: EMA smoothing factor
        diverge_threshold: Stop when loss > threshold * best_loss

    Returns:
        Tuple of (lrs, losses) for plotting

    Pseudocode:
        # Initialize
        lrs = []
        losses = []
        best_loss = float('inf')
        avg_loss = 0

        # Save initial state
        initial_state = model.state_dict()

        for i, (x, y) in enumerate(dataloader):
            if i >= num_iter:
                break

            # Exponentially increase learning rate
            lr = lr_min * (lr_max / lr_min) ** (i / num_iter)
            set_lr(optimizer, lr)

            # Forward pass
            output = model(x)
            loss = criterion(output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Smooth the loss
            avg_loss = smooth_factor * avg_loss + (1 - smooth_factor) * loss.item()
            smoothed_loss = avg_loss / (1 - smooth_factor ** (i + 1))

            # Record
            lrs.append(lr)
            losses.append(smoothed_loss)

            # Update best and check for divergence
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            if smoothed_loss > diverge_threshold * best_loss:
                break

        # Restore initial state
        model.load_state_dict(initial_state)

        return lrs, losses
    """
    pass


def suggest_lr(lrs: list, losses: list) -> dict:
    """
    Analyze LR range test results and suggest optimal learning rate.

    Returns:
        Dictionary with suggested learning rates

    Pseudocode:
        # Method 1: Steepest gradient
        gradients = diff(losses) / diff(log(lrs))
        steepest_idx = argmin(gradients)
        lr_steepest = lrs[steepest_idx]

        # Method 2: Minimum loss with margin
        min_idx = argmin(losses)
        lr_min_loss = lrs[min_idx] / 10

        # Method 3: 10x before explosion
        # Find where loss starts increasing significantly
        for i in range(len(losses) - 1):
            if losses[i+1] > 1.5 * losses[i]:
                lr_before_explosion = lrs[i] / 10
                break

        return {
            'lr_steepest': lr_steepest,
            'lr_min_loss': lr_min_loss,
            'suggested': lr_steepest,  # Most reliable
            'max_lr': lr_before_explosion,  # For OneCycleLR
        }
    """
    pass


# =============================================================================
# Extended LR Finding Methods
# =============================================================================

LR_FINDING_VARIANTS = """
LR Finding Variants:
====================

1. Original LR Range Test (Smith, 2015):
   - Exponential increase
   - Find steepest descent point
   - Used for cyclical LR bounds

2. Fast.ai Method:
   - Similar to original
   - Automatic suggestion
   - Integrated with OneCycleLR

3. Gradient-based Selection:
   - Use d(loss)/d(log(lr))
   - More principled selection
   - Robust to noise

4. Multi-run Average:
   - Run test multiple times
   - Average the curves
   - More reliable estimate

Practical Recommendations:
-------------------------
For SGD + OneCycleLR:
    1. Run LR range test
    2. max_lr = lr where loss starts to increase
    3. min_lr = max_lr / 25
    4. Use OneCycleLR with these bounds

For Adam/AdamW:
    1. Run LR range test
    2. Use lr = steepest_point / 10
    3. Or simply use 1e-4 to 1e-3 (usually works)

For Fine-tuning:
    1. Use lower LR than pretraining
    2. Typically max_lr / 10 to max_lr / 100
    3. LR range test still useful
"""

LR_FINDER_INTERPRETATION = """
How to Read LR Finder Plot:
===========================

          Loss
            |
    high    |____
            |    \____
            |         \___
            |             \___
   optimal  |                 \___  <- Steepest descent
            |                     \___
            |                         \_
    low     |                          |
            |                           \_____/ <- Explosion
            +--------------------------------------> LR (log scale)
         1e-7   1e-5    1e-3    1e-1    10

Regions:
    A. Flat region (lr < 1e-5): LR too small, minimal learning
    B. Descent region: Good LR range, loss decreasing
    C. Steepest point: Optimal LR for OneCycleLR max
    D. Valley/minimum: Good stable LR
    E. Explosion: LR too high, training diverges

Selection Rules:
    - For max_lr (OneCycleLR): Point C or slightly before
    - For constant lr: Point D / 10
    - For SGD: Usually higher than Adam
    - For fine-tuning: Much lower than training from scratch

Common Pitfalls:
    - Using lr at minimum (too aggressive)
    - Ignoring explosion (need safety margin)
    - Not smoothing (noisy curve)
    - Running too few iterations (incomplete picture)
"""


def get_lr_finder_info():
    """Return the LR Range Test MLMethod entry."""
    return LR_RANGE_TEST
