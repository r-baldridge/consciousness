"""
DeiT: Data-efficient Image Transformers - 2021

Authors: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa,
         Alexandre Sablayrolles, Herve Jegou (Facebook AI Research)

Paper: "Training data-efficient image transformers & distillation through attention"
       ICML 2021
       https://arxiv.org/abs/2012.12877

Key Innovation:
    - Knowledge distillation specifically designed for transformers
    - Introduced distillation token alongside [CLS] token
    - Strong data augmentation and regularization for training on ImageNet-1k only
    - Achieves competitive results without large-scale pretraining

Architecture:
    Same as ViT with addition of distillation token:
    1. Patch Embedding: Split image into patches
    2. [CLS] + [DIST] Tokens: Two special tokens
    3. Transformer Encoder: Standard ViT encoder
    4. Dual Heads: One for classification, one for distillation

Knowledge Distillation:
    - Soft distillation: KL divergence on softmax outputs
    - Hard distillation: Cross-entropy with teacher's hard labels
    - Hard-label distillation outperforms soft distillation

Mathematical Formulation:
    Token sequence:
        z_0 = [x_class; x_dist; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos

    Training objectives:
        Soft distillation:
            L = (1-lambda) * L_CE(psi(Z_s), y) + lambda * tau^2 * KL(psi(Z_s/tau), psi(Z_t/tau))

        Hard distillation:
            L = (1/2) * L_CE(psi(Z_s), y) + (1/2) * L_CE(psi(Z_s), y_t)
            where y_t = argmax_c Z_t(c) is the hard decision of the teacher

    Inference (hard distillation):
        y = (softmax(z_class) + softmax(z_dist)) / 2

Pseudocode:
    ```
    def deit_forward(image, teacher_model=None, training=False):
        # 1. Create patches
        patches = image_to_patches(image, patch_size=16)
        N = patches.shape[0]

        # 2. Linear projection
        patch_embeddings = linear(patches, hidden_dim)

        # 3. Prepend [CLS] and [DIST] tokens
        cls_token = learnable_parameter(1, hidden_dim)
        dist_token = learnable_parameter(1, hidden_dim)
        tokens = concat([cls_token, dist_token, patch_embeddings], dim=0)

        # 4. Add position embeddings (N+2 positions)
        pos_embed = learnable_parameter(N+2, hidden_dim)
        tokens = tokens + pos_embed

        # 5. Transformer encoder (same as ViT)
        for layer in transformer_layers:
            tokens = layer(tokens)

        # 6. Dual classification heads
        cls_output = layer_norm(tokens[0])   # [CLS] token
        dist_output = layer_norm(tokens[1])  # [DIST] token

        cls_logits = classification_head(cls_output)
        dist_logits = distillation_head(dist_output)

        if training and teacher_model is not None:
            # Knowledge distillation loss
            with no_grad():
                teacher_logits = teacher_model(image)
            return cls_logits, dist_logits, teacher_logits
        else:
            # Inference: average both heads
            return (cls_logits + dist_logits) / 2
    ```

Data Augmentation Strategy:
    - RandAugment: Random augmentation policy
    - Mixup: alpha=0.8, interpolate images and labels
    - CutMix: Random patch replacement
    - Random Erasing: Randomly erase image regions
    - Color Jitter: Brightness, contrast, saturation variation

Regularization:
    - Stochastic Depth: Drop layers during training (rate ~0.1)
    - Dropout: Standard dropout in attention and MLP
    - Label Smoothing: Soft labels (epsilon=0.1)
    - Weight Decay: AdamW with decay ~0.05

Model Variants:
    - DeiT-Ti (Tiny):  5M params, 72.2% top-1
    - DeiT-S (Small):  22M params, 79.9% top-1
    - DeiT-B (Base):   86M params, 81.8% top-1
    - DeiT-B distilled: 86M params, 83.4% top-1 (with RegNet teacher)

Teacher Models:
    - RegNetY-16GF: Best CNN teacher
    - Larger DeiT: Transformer-to-transformer distillation
    - Convnets work better as teachers than transformers

Historical Significance:
    - Made ViT practical without massive pretraining
    - Introduced attention-based distillation mechanism
    - Established training recipe for vision transformers
    - Democratized vision transformer research

Limitations:
    - Still benefits from distillation (needs pretrained teacher)
    - Requires extensive augmentation/regularization
    - Distillation adds training complexity
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


FORMULATION = r"""
DeiT: Data-efficient Image Transformer

Token Embedding with [CLS] and [DIST]:
    \mathbf{z}_0 = [x_{class}; x_{dist}; \mathbf{x}_p^1 \mathbf{E}; ...;
                   \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}

    where:
        x_{class} \in \mathbb{R}^D - learnable classification token
        x_{dist} \in \mathbb{R}^D - learnable distillation token
        \mathbf{E}_{pos} \in \mathbb{R}^{(N+2) \times D} - position embeddings

Soft Distillation Loss:
    \mathcal{L}_{soft} = (1-\lambda) \cdot \mathcal{L}_{CE}(\psi(Z_s), y) +
                         \lambda \cdot \tau^2 \cdot KL(\psi(Z_s/\tau) \| \psi(Z_t/\tau))

    where:
        \psi = \text{softmax}
        Z_s, Z_t = student, teacher logits
        \tau = temperature for softening distributions
        \lambda = balancing coefficient

Hard-Label Distillation Loss (better performance):
    \mathcal{L}_{hard} = \frac{1}{2} \mathcal{L}_{CE}(\psi(Z_s^{cls}), y) +
                         \frac{1}{2} \mathcal{L}_{CE}(\psi(Z_s^{dist}), y_t)

    where:
        y_t = \arg\max_c Z_t(c) (teacher's hard prediction)
        Z_s^{cls} = logits from [CLS] token
        Z_s^{dist} = logits from [DIST] token

Inference (joint prediction):
    y = \frac{1}{2}(\text{softmax}(z_{class}) + \text{softmax}(z_{dist}))

Stochastic Depth:
    \mathbf{z}_\ell = \begin{cases}
        f_\ell(\mathbf{z}_{\ell-1}) + \mathbf{z}_{\ell-1} & \text{with prob } 1-p_\ell \\
        \mathbf{z}_{\ell-1} & \text{with prob } p_\ell
    \end{cases}

    where p_\ell = \frac{\ell}{L} \cdot p (linear increase)
"""


# Research index entry
DEIT = MLMethod(
    method_id="deit_2021",
    name="DeiT: Data-efficient Image Transformer",
    year=2021,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Hugo Touvron",
        "Matthieu Cord",
        "Matthijs Douze",
        "Francisco Massa",
        "Alexandre Sablayrolles",
        "Herve Jegou",
    ],
    paper_title="Training data-efficient image transformers & distillation through attention",
    paper_url="https://arxiv.org/abs/2012.12877",
    key_innovation=(
        "Introduced distillation token for attention-based knowledge transfer and "
        "established strong training recipe (augmentation + regularization) enabling "
        "ViT training on ImageNet-1k without large-scale pretraining."
    ),
    mathematical_formulation=FORMULATION,
    predecessors=[
        "vision_transformer_2020",
        "knowledge_distillation_2015",
        "randaugment_2019",
        "mixup_2017",
        "cutmix_2019",
    ],
    successors=[
        "deit_iii_2022",
        "cait_2021",
        "levit_2021",
    ],
    tags=[
        "vision-transformer",
        "knowledge-distillation",
        "data-augmentation",
        "regularization",
        "imagenet-1k",
        "distillation-token",
        "facebook-ai",
    ],
    notes=(
        "DeiT made vision transformers accessible to researchers without massive compute. "
        "The distillation token learns to produce predictions similar to a CNN teacher, "
        "inheriting CNN's inductive biases through distillation. Hard-label distillation "
        "surprisingly outperforms soft distillation."
    ),
)


def get_augmentation_config() -> Dict[str, Dict]:
    """
    Data augmentation configuration used in DeiT training.

    Returns:
        Dictionary of augmentation methods and their parameters
    """
    return {
        "RandAugment": {
            "num_ops": 2,
            "magnitude": 9,
            "description": "Random selection of augmentation operations",
        },
        "Mixup": {
            "alpha": 0.8,
            "description": "Convex combination of image pairs and their labels",
            "formula": "x_mix = lambda * x_i + (1-lambda) * x_j",
        },
        "CutMix": {
            "alpha": 1.0,
            "description": "Replace image region with patch from another image",
            "formula": "x_mix = M * x_i + (1-M) * x_j, where M is binary mask",
        },
        "Random_Erasing": {
            "probability": 0.25,
            "description": "Randomly erase rectangular region",
        },
        "Color_Jitter": {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.4,
            "description": "Random color transformations",
        },
        "AutoAugment_Policy": {
            "policy": "rand-m9-mstd0.5-inc1",
            "description": "Automated augmentation policy selection",
        },
    }


def get_regularization_config() -> Dict[str, Dict]:
    """
    Regularization techniques used in DeiT training.

    Returns:
        Dictionary of regularization methods and their parameters
    """
    return {
        "Stochastic_Depth": {
            "drop_rate": 0.1,
            "mode": "linear",
            "description": "Randomly drop transformer layers during training",
            "formula": "p_l = l/L * p_max",
        },
        "Dropout": {
            "attention_dropout": 0.0,
            "projection_dropout": 0.0,
            "description": "Standard dropout (often 0 with other regularization)",
        },
        "Label_Smoothing": {
            "epsilon": 0.1,
            "description": "Soft labels for cross-entropy",
            "formula": "y_soft = (1-eps)*y_hard + eps/K",
        },
        "Weight_Decay": {
            "value": 0.05,
            "description": "L2 regularization with AdamW",
        },
        "Gradient_Clipping": {
            "max_norm": 1.0,
            "description": "Clip gradients to prevent explosion",
        },
    }


def get_model_variants() -> Dict[str, Dict]:
    """
    DeiT model variants and their specifications.

    Returns:
        Dictionary of model variants with architecture and performance
    """
    return {
        "DeiT-Ti": {
            "hidden_dim": 192,
            "num_layers": 12,
            "num_heads": 3,
            "mlp_ratio": 4,
            "params": "5M",
            "top1_no_distill": "72.2%",
            "top1_distilled": "74.5%",
        },
        "DeiT-S": {
            "hidden_dim": 384,
            "num_layers": 12,
            "num_heads": 6,
            "mlp_ratio": 4,
            "params": "22M",
            "top1_no_distill": "79.9%",
            "top1_distilled": "81.2%",
        },
        "DeiT-B": {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "params": "86M",
            "top1_no_distill": "81.8%",
            "top1_distilled": "83.4%",
        },
        "DeiT-B-384": {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
            "resolution": 384,
            "params": "86M",
            "top1_distilled": "85.2%",
            "notes": "Fine-tuned at higher resolution",
        },
    }


def get_teacher_comparison() -> Dict[str, Dict]:
    """
    Comparison of different teacher models for DeiT distillation.

    Returns:
        Dictionary comparing teacher effectiveness
    """
    return {
        "RegNetY-16GF": {
            "teacher_top1": "82.9%",
            "student_top1": "83.4%",
            "type": "CNN",
            "notes": "Best performing teacher - CNN inductive bias helps",
        },
        "DeiT-B": {
            "teacher_top1": "81.8%",
            "student_top1": "82.0%",
            "type": "Transformer",
            "notes": "Transformer-to-transformer distillation less effective",
        },
        "EfficientNet-B7": {
            "teacher_top1": "84.4%",
            "student_top1": "84.0%",
            "type": "CNN",
            "notes": "Very strong teacher",
        },
    }


def soft_distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> float:
    """
    Compute soft distillation loss.

    L = (1-alpha) * CE(student, labels) + alpha * T^2 * KL(student/T || teacher/T)

    Args:
        student_logits: Student model outputs (B, C)
        teacher_logits: Teacher model outputs (B, C)
        labels: Ground truth labels (B,)
        temperature: Softening temperature
        alpha: Balance between CE and KL

    Returns:
        Combined distillation loss
    """
    def softmax(x, T=1.0):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted / T)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def cross_entropy(logits, labels):
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-10)
        return -np.mean(log_probs[np.arange(len(labels)), labels])

    def kl_divergence(p, q):
        return np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)), axis=-1).mean()

    # Cross-entropy with ground truth
    ce_loss = cross_entropy(student_logits, labels)

    # KL divergence with soft teacher labels
    student_soft = softmax(student_logits, temperature)
    teacher_soft = softmax(teacher_logits, temperature)
    kl_loss = kl_divergence(teacher_soft, student_soft)

    # Combined loss
    loss = (1 - alpha) * ce_loss + alpha * (temperature ** 2) * kl_loss

    return float(loss)


def hard_distillation_loss(
    cls_logits: np.ndarray,
    dist_logits: np.ndarray,
    labels: np.ndarray,
    teacher_logits: np.ndarray,
) -> float:
    """
    Compute hard-label distillation loss (used by DeiT).

    L = 0.5 * CE(cls_logits, labels) + 0.5 * CE(dist_logits, teacher_hard_labels)

    Args:
        cls_logits: Logits from [CLS] token (B, C)
        dist_logits: Logits from [DIST] token (B, C)
        labels: Ground truth labels (B,)
        teacher_logits: Teacher model outputs (B, C)

    Returns:
        Combined hard distillation loss
    """
    def softmax(x):
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def cross_entropy(logits, labels):
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-10)
        return -np.mean(log_probs[np.arange(len(labels)), labels])

    # Hard labels from teacher
    teacher_hard_labels = np.argmax(teacher_logits, axis=-1)

    # Classification loss (ground truth)
    cls_loss = cross_entropy(cls_logits, labels)

    # Distillation loss (teacher hard labels)
    dist_loss = cross_entropy(dist_logits, teacher_hard_labels)

    # Average both losses
    loss = 0.5 * cls_loss + 0.5 * dist_loss

    return float(loss)


def key_equations() -> Dict[str, str]:
    """Return key equations for DeiT."""
    return {
        "token_embedding": r"z_0 = [x_{class}; x_{dist}; x_p^1 E; ...; x_p^N E] + E_{pos}",
        "soft_distillation": r"\mathcal{L} = (1-\lambda)CE(Z_s, y) + \lambda\tau^2 KL(\sigma(Z_s/\tau) \| \sigma(Z_t/\tau))",
        "hard_distillation": r"\mathcal{L} = \frac{1}{2}CE(Z_s^{cls}, y) + \frac{1}{2}CE(Z_s^{dist}, y_t)",
        "inference": r"y = \frac{1}{2}(\text{softmax}(z_{class}) + \text{softmax}(z_{dist}))",
        "stochastic_depth": r"p_\ell = \frac{\ell}{L} \cdot p_{max}",
    }
