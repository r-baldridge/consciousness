"""
Vision Transformers - Attention-based architectures for computer vision.

This module contains research documentation for major vision transformer architectures
that revolutionized computer vision by applying transformer attention mechanisms to images.

Timeline:
    - 2020: ViT (Google) - First pure transformer for image classification
    - 2021: DeiT (Facebook AI) - Data-efficient training on ImageNet-1k
    - 2021: Swin (Microsoft) - Hierarchical vision transformer with shifted windows
    - 2023: SAM (Meta AI) - Foundation model for promptable segmentation

Key Innovations:
    - Patch tokenization: Treating image patches as sequence tokens
    - Position embeddings: Encoding spatial information
    - Hierarchical features: Multi-scale representations like CNNs
    - Efficient attention: Linear complexity through windowing
    - Foundation models: Zero-shot generalization across tasks

All methods follow the MLMethod documentation template from core.taxonomy.
"""

# Vision Transformer (ViT) - 2020
from .vit import (
    VISION_TRANSFORMER,
    FORMULATION as VIT_FORMULATION,
    image_to_patches,
    get_position_embeddings,
    get_model_config as get_vit_config,
    key_equations as vit_equations,
    get_pretraining_datasets,
)

# DeiT: Data-efficient Image Transformer - 2021
from .deit import (
    DEIT,
    FORMULATION as DEIT_FORMULATION,
    get_augmentation_config,
    get_regularization_config,
    get_model_variants as get_deit_variants,
    get_teacher_comparison,
    soft_distillation_loss,
    hard_distillation_loss,
    key_equations as deit_equations,
)

# Swin Transformer - 2021
from .swin import (
    SWIN_TRANSFORMER,
    FORMULATION as SWIN_FORMULATION,
    get_model_variants as get_swin_variants,
    get_stage_info,
    window_partition,
    window_reverse,
    cyclic_shift,
    get_relative_position_index,
    compute_attention_mask,
    patch_merging,
    key_equations as swin_equations,
    get_downstream_tasks as get_swin_tasks,
)

# Segment Anything Model (SAM) - 2023
from .sam import (
    SEGMENT_ANYTHING,
    FORMULATION as SAM_FORMULATION,
    get_model_variants as get_sam_variants,
    get_prompt_types,
    get_sa1b_dataset_info,
    get_architecture_details as get_sam_architecture,
    positional_encoding_2d,
    encode_point_prompt,
    encode_box_prompt,
    key_equations as sam_equations,
    get_downstream_applications,
)

# All MLMethod entries for indexing
ALL_METHODS = [
    VISION_TRANSFORMER,
    DEIT,
    SWIN_TRANSFORMER,
    SEGMENT_ANYTHING,
]

# Method lookup by ID
METHODS_BY_ID = {method.method_id: method for method in ALL_METHODS}

# Export all public symbols
__all__ = [
    # MLMethod entries
    "VISION_TRANSFORMER",
    "DEIT",
    "SWIN_TRANSFORMER",
    "SEGMENT_ANYTHING",
    "ALL_METHODS",
    "METHODS_BY_ID",

    # ViT exports
    "VIT_FORMULATION",
    "image_to_patches",
    "get_position_embeddings",
    "get_vit_config",
    "vit_equations",
    "get_pretraining_datasets",

    # DeiT exports
    "DEIT_FORMULATION",
    "get_augmentation_config",
    "get_regularization_config",
    "get_deit_variants",
    "get_teacher_comparison",
    "soft_distillation_loss",
    "hard_distillation_loss",
    "deit_equations",

    # Swin exports
    "SWIN_FORMULATION",
    "get_swin_variants",
    "get_stage_info",
    "window_partition",
    "window_reverse",
    "cyclic_shift",
    "get_relative_position_index",
    "compute_attention_mask",
    "patch_merging",
    "swin_equations",
    "get_swin_tasks",

    # SAM exports
    "SAM_FORMULATION",
    "get_sam_variants",
    "get_prompt_types",
    "get_sa1b_dataset_info",
    "get_sam_architecture",
    "positional_encoding_2d",
    "encode_point_prompt",
    "encode_box_prompt",
    "sam_equations",
    "get_downstream_applications",
]
