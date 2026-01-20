"""
Data Augmentation Methods

Data augmentation techniques that increase effective training data
through transformations, improving model generalization.

Key Methods:
    - Mixup: Linear interpolation of samples
    - CutMix: Cut and paste patches between images
    - RandAugment: Simplified automated augmentation
    - AutoAugment: Learned augmentation policies
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# Mixup
# =============================================================================

MIXUP = MLMethod(
    method_id="mixup",
    name="Mixup",
    year=2017,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Hongyi Zhang", "Moustapha Cisse", "Yann N. Dauphin", "David Lopez-Paz"],
    paper_title="mixup: Beyond Empirical Risk Minimization",
    paper_url="https://arxiv.org/abs/1710.09412",
    key_innovation="""
    Creates virtual training examples by linearly interpolating pairs of
    examples and their labels. This encourages the model to behave linearly
    between training examples, leading to smoother decision boundaries and
    better generalization.
    """,
    mathematical_formulation="""
    Mixup:
    ------

    For a pair of training examples (x_i, y_i) and (x_j, y_j):

    1. Sample mixing coefficient:
        lambda ~ Beta(alpha, alpha)

    2. Create virtual example:
        x_tilde = lambda * x_i + (1 - lambda) * x_j
        y_tilde = lambda * y_i + (1 - lambda) * y_j

    Where:
        alpha       = hyperparameter controlling interpolation strength
        lambda      = mixing coefficient in [0, 1]
        x_tilde     = virtual input (blend of two images)
        y_tilde     = virtual label (blend of two one-hot vectors)

    Beta Distribution:
        alpha = 0.1: Most lambda near 0 or 1 (light mixing)
        alpha = 0.2: Common setting (moderate mixing)
        alpha = 0.4: Stronger mixing
        alpha = 1.0: Uniform distribution (heavy mixing)

    Training:
        x_batch, y_batch = get_batch()
        x_shuffled, y_shuffled = shuffle(x_batch, y_batch)
        lambda = sample_beta(alpha, alpha)
        x_mixed = lambda * x_batch + (1 - lambda) * x_shuffled
        y_mixed = lambda * y_batch + (1 - lambda) * y_shuffled
        loss = criterion(model(x_mixed), y_mixed)

    Label Smoothing Effect:
        Mixup naturally provides soft labels when mixing classes.
        Example: If mixing cat (label 0) and dog (label 1) with lambda=0.7:
            y_mixed = [0.7, 0.3] (soft label)

    Theoretical Motivation:
        - Vicinal Risk Minimization (VRM)
        - Encourages linear behavior in input space
        - Smooths decision boundaries
    """,
    predecessors=[],
    successors=["cutmix", "manifold_mixup"],
    tags=["augmentation", "regularization", "interpolation", "soft-labels"],
    notes="""
    Benefits:
    - Improves generalization
    - Better calibration (soft predictions)
    - Reduces overconfident predictions
    - Increases robustness

    Typical alpha values:
    - alpha = 0.2: Common for image classification
    - alpha = 0.4: For stronger regularization
    - alpha = 1.0: Maximum mixing (uniform)

    Implementation Tips:
    - Mix within mini-batch (shuffle and combine)
    - Use for training only, not inference
    - Works with any differentiable model
    - Combine with other augmentations

    Variants:
    - Manifold Mixup: Mix hidden representations
    - CutMix: Spatial mixing instead of global
    - Input Mixup: Standard (this method)
    """
)


def mixup_data(
    x: "ndarray",
    y: "ndarray",
    alpha: float = 0.2
) -> tuple:
    """
    Pseudocode for Mixup data augmentation.

    Args:
        x: Input batch of shape (N, ...)
        y: Labels of shape (N, num_classes) - one-hot encoded
        alpha: Beta distribution parameter

    Returns:
        Tuple of (mixed_x, mixed_y, lambda)

    Pseudocode:
        # Sample mixing coefficient
        if alpha > 0:
            lambda = sample_beta(alpha, alpha)
        else:
            lambda = 1.0

        # Get random permutation
        indices = random_permutation(N)

        # Mix inputs and labels
        x_mixed = lambda * x + (1 - lambda) * x[indices]
        y_mixed = lambda * y + (1 - lambda) * y[indices]

        return x_mixed, y_mixed, lambda
    """
    pass


# =============================================================================
# CutMix
# =============================================================================

CUTMIX = MLMethod(
    method_id="cutmix",
    name="CutMix",
    year=2019,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Sangdoo Yun", "Dongyoon Han", "Seong Joon Oh",
             "Sanghyuk Chun", "Junsuk Choe", "Youngjoon Yoo"],
    paper_title="CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features",
    paper_url="https://arxiv.org/abs/1905.04899",
    key_innovation="""
    Instead of globally mixing images like Mixup, CutMix cuts a patch from
    one image and pastes it onto another. This preserves local statistics
    while still mixing labels proportionally to the area of the patch.
    Leads to better localization ability than Mixup.
    """,
    mathematical_formulation="""
    CutMix:
    -------

    For a pair of images (x_A, y_A) and (x_B, y_B):

    1. Sample mixing ratio:
        lambda ~ Beta(alpha, alpha)

    2. Sample bounding box coordinates:
        r_x ~ Uniform(0, W)     # center x
        r_y ~ Uniform(0, H)     # center y
        r_w = W * sqrt(1 - lambda)   # width
        r_h = H * sqrt(1 - lambda)   # height

        # Bounding box
        x1 = clip(r_x - r_w/2, 0, W)
        x2 = clip(r_x + r_w/2, 0, W)
        y1 = clip(r_y - r_h/2, 0, H)
        y2 = clip(r_y + r_h/2, 0, H)

    3. Create binary mask M:
        M[y1:y2, x1:x2] = 1
        M elsewhere = 0

    4. Create mixed image:
        x_tilde = (1 - M) * x_A + M * x_B

    5. Compute actual mixing ratio (after clipping):
        lambda_adjusted = 1 - (area of box) / (W * H)

    6. Mix labels:
        y_tilde = lambda_adjusted * y_A + (1 - lambda_adjusted) * y_B

    Where:
        M           = binary mask
        lambda      = target mixing ratio
        lambda_adj  = actual ratio (after boundary effects)
        (W, H)      = image dimensions

    Area Calculation:
        box_area = (x2 - x1) * (y2 - y1)
        lambda_adjusted = 1 - box_area / (W * H)

    Key Difference from Mixup:
        Mixup:   x_mixed = lambda * x_A + (1-lambda) * x_B  (global blend)
        CutMix:  x_mixed = x_A with patch from x_B         (spatial)
    """,
    predecessors=["mixup", "cutout"],
    successors=["saliency_mix", "resizemix"],
    tags=["augmentation", "regularization", "spatial", "localization"],
    notes="""
    Benefits vs Mixup:
    - Better localization (uses local features)
    - More efficient training (sees full regions)
    - Improves object detection transfer
    - No loss of spatial information

    Implementation:
    - Random box location and size
    - Label proportional to visible area
    - Apply within mini-batch

    Typical Settings:
    - alpha = 1.0 (uniform lambda)
    - Apply with probability 0.5-1.0

    Combined with other augmentation:
    - Often combined with standard augmentation
    - Can be combined with Mixup (choose randomly)
    - Popular in vision training recipes
    """
)


def cutmix_data(
    x: "ndarray",
    y: "ndarray",
    alpha: float = 1.0
) -> tuple:
    """
    Pseudocode for CutMix data augmentation.

    Args:
        x: Input batch of shape (N, C, H, W)
        y: Labels of shape (N, num_classes)
        alpha: Beta distribution parameter

    Returns:
        Tuple of (mixed_x, mixed_y, lambda)

    Pseudocode:
        # Sample mixing ratio
        lambda = sample_beta(alpha, alpha)

        # Get random permutation
        indices = random_permutation(N)

        # Compute box dimensions
        N, C, H, W = x.shape
        cut_w = int(W * sqrt(1 - lambda))
        cut_h = int(H * sqrt(1 - lambda))

        # Sample center point
        cx = random_int(0, W)
        cy = random_int(0, H)

        # Compute box boundaries (with clipping)
        x1 = clip(cx - cut_w // 2, 0, W)
        x2 = clip(cx + cut_w // 2, 0, W)
        y1 = clip(cy - cut_h // 2, 0, H)
        y2 = clip(cy + cut_h // 2, 0, H)

        # Create mixed image
        x_mixed = x.copy()
        x_mixed[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]

        # Adjust lambda based on actual box area
        lambda_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)

        # Mix labels
        y_mixed = lambda_adjusted * y + (1 - lambda_adjusted) * y[indices]

        return x_mixed, y_mixed, lambda_adjusted
    """
    pass


# =============================================================================
# RandAugment
# =============================================================================

RANDAUGMENT = MLMethod(
    method_id="randaugment",
    name="RandAugment",
    year=2019,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Ekin D. Cubuk", "Barret Zoph", "Jonathon Shlens", "Quoc V. Le"],
    paper_title="RandAugment: Practical automated data augmentation with a reduced search space",
    paper_url="https://arxiv.org/abs/1909.13719",
    key_innovation="""
    Simplifies automated augmentation to just TWO hyperparameters:
    N (number of augmentations) and M (magnitude). Randomly samples
    N operations from a fixed set and applies each at magnitude M.
    Much simpler than AutoAugment while achieving comparable results.
    """,
    mathematical_formulation="""
    RandAugment:
    ------------

    Parameters:
        N = number of augmentation operations to apply
        M = magnitude of each operation (shared across all)

    Operations (typical set of 14):
        1. Identity (no change)
        2. AutoContrast
        3. Equalize (histogram equalization)
        4. Rotate (by M degrees)
        5. Solarize (invert above threshold M)
        6. Color (adjust saturation by M)
        7. Posterize (reduce bits by M)
        8. Contrast (adjust by M)
        9. Brightness (adjust by M)
        10. Sharpness (adjust by M)
        11. ShearX (shear horizontal by M)
        12. ShearY (shear vertical by M)
        13. TranslateX (translate horizontal by M)
        14. TranslateY (translate vertical by M)

    Algorithm:
        For each image:
            ops = randomly_select_N(operations, N)
            for op in ops:
                image = apply_op(image, magnitude=M)

    Magnitude M:
        - Range: typically 0-30 or 0-10
        - Higher = stronger augmentation
        - Scales the strength of each operation

    Pseudocode:
        def randaugment(image, N, M):
            operations = [rotate, shear, translate, ...]
            for _ in range(N):
                op = random.choice(operations)
                image = op(image, magnitude=M)
            return image

    Search Space:
        AutoAugment: 10^32 possible policies
        RandAugment: ~10^2 (just tune N and M)
    """,
    predecessors=["autoaugment"],
    successors=["trivialaugment"],
    tags=["augmentation", "automated", "simple", "practical"],
    notes="""
    Why RandAugment?
    - AutoAugment requires expensive search
    - RandAugment achieves similar results
    - Only 2 hyperparameters to tune
    - Easy to implement and reproduce

    Typical Settings:
    - N = 2-3 (number of operations)
    - M = 9-15 (magnitude, scale varies by implementation)

    Implementation:
    - Apply N random operations sequentially
    - Same magnitude for all operations
    - Simple grid search to tune N and M

    Comparison to AutoAugment:
    - Much simpler (2 params vs learned policy)
    - Similar or better performance
    - No search required
    - More robust across datasets

    Used In:
    - EfficientNet training
    - Vision Transformer training
    - Many SOTA models
    """
)


# =============================================================================
# AutoAugment
# =============================================================================

AUTOAUGMENT = MLMethod(
    method_id="autoaugment",
    name="AutoAugment",
    year=2018,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.REGULARIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE],
    authors=["Ekin D. Cubuk", "Barret Zoph", "Dandelion Mane",
             "Vijay Vasudevan", "Quoc V. Le"],
    paper_title="AutoAugment: Learning Augmentation Policies from Data",
    paper_url="https://arxiv.org/abs/1805.09501",
    key_innovation="""
    Uses reinforcement learning to search for the best augmentation policy
    for a given dataset. Discovers combinations and orderings of augmentation
    operations that outperform hand-designed augmentation strategies.
    """,
    mathematical_formulation="""
    AutoAugment:
    ------------

    Policy Structure:
        - A policy consists of 5 sub-policies
        - Each sub-policy has 2 operations
        - Each operation has: (operation_type, probability, magnitude)

    Sub-policy format:
        (op1, prob1, mag1, op2, prob2, mag2)

    Example sub-policy:
        (ShearX, 0.9, 7, Rotate, 0.8, 3)
        = Apply ShearX with prob 0.9 at magnitude 7, then
          Apply Rotate with prob 0.8 at magnitude 3

    Training:
        For each batch:
            sub_policy = random.choice(5 sub-policies)
            for (op, prob, mag) in sub_policy:
                if random() < prob:
                    image = op(image, mag)

    Search Method:
        Use RL (reinforcement learning) to optimize policy:
        - Controller RNN proposes policies
        - Child network trained with policy
        - Validation accuracy as reward
        - Controller updated via policy gradient

    Search Space Size:
        - 16 operations, 10 magnitudes, 11 probabilities
        - (16 * 10 * 11)^10 = ~10^32 possible policies

    Discovered Policies (ImageNet):
        - Sub-policy 1: (Posterize, 0.4, 8), (Rotate, 0.6, 9)
        - Sub-policy 2: (Solarize, 0.6, 5), (AutoContrast, 0.6, -)
        - ... (5 sub-policies total)
    """,
    predecessors=[],
    successors=["randaugment", "fast_autoaugment"],
    tags=["augmentation", "automated", "reinforcement-learning", "neural-architecture-search"],
    notes="""
    Key Contributions:
    - First learned augmentation policy
    - Significant improvements on CIFAR-10, ImageNet
    - Policies transfer across datasets
    - Inspired many follow-up works

    Limitations:
    - Very expensive search (~15,000 GPU hours)
    - Requires training many proxy models
    - Dataset-specific policies

    Transferred Policies:
    - ImageNet policy works well on other datasets
    - CIFAR policy published and widely used

    Follow-up Works:
    - Fast AutoAugment (more efficient search)
    - RandAugment (no search needed)
    - TrivialAugment (even simpler)
    - Population Based Augmentation
    """
)


# =============================================================================
# Comparison
# =============================================================================

DATA_AUGMENTATION_COMPARISON = """
Data Augmentation Methods Comparison:
=====================================

| Method       | Key Idea                    | Hyperparameters      | Overhead |
|--------------|-----------------------------|-----------------------|----------|
| Mixup        | Linear interpolation        | alpha                 | Minimal  |
| CutMix       | Cut and paste patches       | alpha                 | Minimal  |
| RandAugment  | N random ops, magnitude M   | N, M                  | Low      |
| AutoAugment  | Learned augmentation policy | Searched policy       | High*    |

*AutoAugment overhead is during search; deployment is low

When to Use:
- Mixup: General regularization, smoother predictions
- CutMix: Better localization, works well with Mixup
- RandAugment: Simple, effective, easy to tune
- AutoAugment: Maximum performance, have compute for search

Modern Training Recipes:
    Many SOTA models use combinations:
    - RandAugment + Mixup + CutMix
    - Apply with probability (e.g., 0.5 each)
    - Increases effective dataset diversity
"""


def get_all_augmentation_methods():
    """Return all data augmentation MLMethod entries."""
    return [MIXUP, CUTMIX, RANDAUGMENT, AUTOAUGMENT]
