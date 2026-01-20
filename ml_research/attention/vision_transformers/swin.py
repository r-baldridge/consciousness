"""
Swin Transformer - 2021

Authors: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang,
         Stephen Lin, Baining Guo (Microsoft Research Asia)

Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
       ICCV 2021 (Best Paper)
       https://arxiv.org/abs/2103.14030

Key Innovation:
    - Hierarchical feature maps like CNNs (enables dense prediction tasks)
    - Shifted window attention for cross-window connections
    - Linear computational complexity O(n) vs O(n^2) for global attention
    - General-purpose backbone for vision (classification, detection, segmentation)

Architecture:
    1. Patch Partition: Split image into 4x4 patches
    2. Stage 1: Linear embedding + Swin Transformer blocks
    3. Stage 2-4: Patch merging (2x downsampling) + Swin Transformer blocks
    4. Each stage has alternating W-MSA and SW-MSA blocks

Window Attention Mechanism:
    - W-MSA (Window Multi-head Self-Attention):
      Partition features into non-overlapping MxM windows
      Apply self-attention within each window independently

    - SW-MSA (Shifted Window MSA):
      Shift windows by (M/2, M/2) pixels before partitioning
      Enables cross-window information flow
      Uses efficient batch computation with cyclic shift

Mathematical Formulation:
    Standard Window Attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d) + B)V
        where B is relative position bias

    Computational Complexity:
        Global MSA: O(4hwC^2 + 2(hw)^2 C)  -- quadratic in hw
        W-MSA:      O(4hwC^2 + 2M^2 hwC)   -- linear in hw

    Swin Transformer Block (consecutive layers):
        z^l = W-MSA(LN(z^(l-1))) + z^(l-1)
        z^l = MLP(LN(z^l)) + z^l
        z^(l+1) = SW-MSA(LN(z^l)) + z^l
        z^(l+1) = MLP(LN(z^(l+1))) + z^(l+1)

Pseudocode:
    ```
    def swin_transformer(image, num_classes=1000):
        # Stage 1: Patch embedding (4x4 patches -> C features)
        x = patch_embed(image)  # (H/4, W/4, C)

        # Stage 1: Swin Transformer blocks (no downsampling)
        for i in range(num_blocks[0]):
            if i % 2 == 0:
                x = window_attention(x, shift=0)   # W-MSA
            else:
                x = window_attention(x, shift=M//2)  # SW-MSA
            x = mlp(layer_norm(x)) + x

        # Stages 2-4: Patch merging + Swin blocks
        for stage in range(1, 4):
            # Patch merging: 2x2 -> 1, double channels
            x = patch_merge(x)  # (H/2, W/2, 2C)

            for i in range(num_blocks[stage]):
                if i % 2 == 0:
                    x = window_attention(x, shift=0)
                else:
                    x = window_attention(x, shift=M//2)
                x = mlp(layer_norm(x)) + x

        # Classification head
        x = global_avg_pool(x)
        logits = linear(x, num_classes)
        return logits

    def window_attention(x, window_size=7, shift=0):
        H, W, C = x.shape

        # Cyclic shift (for SW-MSA)
        if shift > 0:
            x = cyclic_shift(x, shift)

        # Partition into windows
        windows = partition_windows(x, window_size)  # (num_windows, M*M, C)

        # Self-attention within each window
        attn = multi_head_attention(windows)

        # Merge windows back
        x = merge_windows(attn, H, W)

        # Reverse cyclic shift
        if shift > 0:
            x = cyclic_shift(x, -shift)

        return x
    ```

Relative Position Bias:
    - Learnable bias matrix B for each head
    - B indexed by relative position (range [-M+1, M-1] for both axes)
    - Parameterized as B-hat with size (2M-1) x (2M-1)
    - Significantly improves performance over absolute positions

Patch Merging:
    - Concatenate 2x2 neighboring patches (4C channels)
    - Linear layer to reduce to 2C channels
    - Similar to strided convolution / pooling

Model Variants:
    - Swin-T (Tiny):  29M params, 81.3% ImageNet top-1
    - Swin-S (Small): 50M params, 83.0% ImageNet top-1
    - Swin-B (Base):  88M params, 83.5% ImageNet top-1
    - Swin-L (Large): 197M params, 86.4% ImageNet top-1 (ImageNet-22k pretrain)

Historical Significance:
    - Made transformers practical for dense prediction tasks
    - Became dominant backbone for detection and segmentation
    - ICCV 2021 Best Paper - unified vision backbone
    - Inspired SwinV2, Swin-UNETR, and many variants

Applications:
    - Image Classification (ImageNet)
    - Object Detection (COCO) - with Cascade Mask R-CNN
    - Semantic Segmentation (ADE20K) - with UperNet
    - Instance Segmentation
    - Video Classification (Kinetics)

Limitations:
    - Window size is fixed (typically 7x7)
    - Cyclic shift implementation complexity
    - Still requires careful training recipe
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
Swin Transformer:

Window-based Multi-head Self-Attention (W-MSA):
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V

    where B \in \mathbb{R}^{M^2 \times M^2} is relative position bias

Computational Complexity Comparison:
    Global MSA:  \Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C
    Window MSA:  \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC

    where h,w = feature map size, M = window size, C = channels

Swin Transformer Block (two consecutive layers):
    \hat{\mathbf{z}}^\ell = \text{W-MSA}(\text{LN}(\mathbf{z}^{\ell-1})) + \mathbf{z}^{\ell-1}
    \mathbf{z}^\ell = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^\ell)) + \hat{\mathbf{z}}^\ell
    \hat{\mathbf{z}}^{\ell+1} = \text{SW-MSA}(\text{LN}(\mathbf{z}^\ell)) + \mathbf{z}^\ell
    \mathbf{z}^{\ell+1} = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{\ell+1})) + \hat{\mathbf{z}}^{\ell+1}

Relative Position Bias:
    B = \hat{B}[\text{relative\_position\_index}]
    \hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}

    where relative\_position\_index encodes spatial relationships

Patch Merging (downsampling):
    \mathbf{x}_{merged} = \text{Linear}([\mathbf{x}_{0,0}; \mathbf{x}_{0,1}; \mathbf{x}_{1,0}; \mathbf{x}_{1,1}])

    Input: H \times W \times C \rightarrow Output: \frac{H}{2} \times \frac{W}{2} \times 2C
"""


# Research index entry
SWIN_TRANSFORMER = MLMethod(
    method_id="swin_transformer_2021",
    name="Swin Transformer",
    year=2021,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Ze Liu",
        "Yutong Lin",
        "Yue Cao",
        "Han Hu",
        "Yixuan Wei",
        "Zheng Zhang",
        "Stephen Lin",
        "Baining Guo",
    ],
    paper_title="Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
    paper_url="https://arxiv.org/abs/2103.14030",
    key_innovation=(
        "Hierarchical vision transformer with shifted window attention achieving "
        "linear computational complexity. First transformer to serve as general-purpose "
        "backbone for dense prediction tasks (detection, segmentation)."
    ),
    mathematical_formulation=FORMULATION,
    predecessors=[
        "vision_transformer_2020",
        "deit_2021",
        "resnet_2015",
        "fpn_2017",
    ],
    successors=[
        "swin_v2_2022",
        "swin_unetr_2022",
        "cswin_transformer_2022",
        "focal_transformer_2021",
    ],
    tags=[
        "vision-transformer",
        "hierarchical",
        "shifted-windows",
        "linear-complexity",
        "dense-prediction",
        "object-detection",
        "segmentation",
        "microsoft-research",
        "iccv-best-paper",
    ],
    notes=(
        "Swin Transformer won ICCV 2021 Best Paper for unifying vision backbones. "
        "The shifted window mechanism enables cross-window information flow while "
        "maintaining linear complexity. Relative position bias is crucial for performance. "
        "Became the dominant backbone for detection and segmentation tasks."
    ),
)


def get_model_variants() -> Dict[str, Dict]:
    """
    Swin Transformer model variants and their configurations.

    Returns:
        Dictionary of model variants with architecture and performance
    """
    return {
        "Swin-T": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "params": "29M",
            "flops": "4.5G",
            "imagenet_top1": "81.3%",
        },
        "Swin-S": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": 7,
            "params": "50M",
            "flops": "8.7G",
            "imagenet_top1": "83.0%",
        },
        "Swin-B": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
            "window_size": 7,
            "params": "88M",
            "flops": "15.4G",
            "imagenet_top1": "83.5%",
        },
        "Swin-L": {
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "window_size": 7,
            "params": "197M",
            "flops": "34.5G",
            "imagenet_top1": "86.4%",
            "notes": "Pretrained on ImageNet-22k",
        },
    }


def get_stage_info() -> Dict[str, Dict]:
    """
    Information about each stage of Swin Transformer.

    Returns:
        Dictionary describing each hierarchical stage
    """
    return {
        "Stage_1": {
            "patch_size": "4x4",
            "resolution": "H/4 x W/4",
            "channels": "C (96 for Swin-T)",
            "operation": "Patch embed + Swin blocks",
            "downsampling": "None (initial embedding)",
        },
        "Stage_2": {
            "resolution": "H/8 x W/8",
            "channels": "2C (192 for Swin-T)",
            "operation": "Patch merge + Swin blocks",
            "downsampling": "2x via patch merging",
        },
        "Stage_3": {
            "resolution": "H/16 x W/16",
            "channels": "4C (384 for Swin-T)",
            "operation": "Patch merge + Swin blocks",
            "downsampling": "2x via patch merging",
        },
        "Stage_4": {
            "resolution": "H/32 x W/32",
            "channels": "8C (768 for Swin-T)",
            "operation": "Patch merge + Swin blocks",
            "downsampling": "2x via patch merging",
        },
    }


def window_partition(
    x: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: Input feature map (H, W, C)
        window_size: Size of each window (M)

    Returns:
        Windows of shape (num_windows, M*M, C)
    """
    H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        f"Feature map size must be divisible by window size"

    # Reshape to (H/M, M, W/M, M, C)
    x = x.reshape(H // window_size, window_size, W // window_size, window_size, C)

    # Transpose to (H/M, W/M, M, M, C)
    x = x.transpose(0, 2, 1, 3, 4)

    # Flatten windows: (num_windows, M*M, C)
    num_windows = (H // window_size) * (W // window_size)
    windows = x.reshape(num_windows, window_size * window_size, C)

    return windows


def window_reverse(
    windows: np.ndarray,
    window_size: int,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Reverse window partition back to feature map.

    Args:
        windows: Windowed features (num_windows, M*M, C)
        window_size: Size of each window (M)
        H, W: Original feature map height and width

    Returns:
        Feature map of shape (H, W, C)
    """
    num_windows, _, C = windows.shape
    n_h = H // window_size
    n_w = W // window_size

    # Reshape to (H/M, W/M, M, M, C)
    x = windows.reshape(n_h, n_w, window_size, window_size, C)

    # Transpose to (H/M, M, W/M, M, C)
    x = x.transpose(0, 2, 1, 3, 4)

    # Reshape to (H, W, C)
    x = x.reshape(H, W, C)

    return x


def cyclic_shift(
    x: np.ndarray,
    shift: int,
) -> np.ndarray:
    """
    Apply cyclic shift to feature map for shifted window attention.

    Args:
        x: Input feature map (H, W, C)
        shift: Shift amount (typically M//2 or -M//2)

    Returns:
        Cyclically shifted feature map (H, W, C)
    """
    return np.roll(np.roll(x, shift, axis=0), shift, axis=1)


def get_relative_position_index(
    window_size: int,
) -> np.ndarray:
    """
    Generate relative position index for a window.

    Args:
        window_size: Size of attention window (M)

    Returns:
        Relative position index of shape (M*M, M*M)
    """
    # Create coordinate grid
    coords_h = np.arange(window_size)
    coords_w = np.arange(window_size)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, M, M)

    # Flatten coordinates
    coords_flatten = coords.reshape(2, -1)  # (2, M*M)

    # Compute relative positions
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, M*M, M*M)

    # Shift to start from 0
    relative_coords[0] += window_size - 1
    relative_coords[1] += window_size - 1

    # Convert to single index
    relative_coords[0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(axis=0)  # (M*M, M*M)

    return relative_position_index.astype(np.int64)


def compute_attention_mask(
    H: int,
    W: int,
    window_size: int,
    shift_size: int,
) -> np.ndarray:
    """
    Compute attention mask for shifted window attention.

    Different regions should not attend to each other after cyclic shift.

    Args:
        H, W: Feature map dimensions
        window_size: Window size (M)
        shift_size: Shift amount for SW-MSA

    Returns:
        Attention mask of shape (num_windows, M*M, M*M)
    """
    if shift_size == 0:
        # No mask needed for W-MSA
        return None

    # Create region labels
    img_mask = np.zeros((H, W))
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[h, w] = cnt
            cnt += 1

    # Partition into windows
    mask_windows = window_partition(img_mask[:, :, np.newaxis], window_size)
    mask_windows = mask_windows.squeeze(-1)  # (num_windows, M*M)

    # Create attention mask
    attn_mask = mask_windows[:, :, np.newaxis] - mask_windows[:, np.newaxis, :]
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0)

    return attn_mask


def patch_merging(
    x: np.ndarray,
    output_channels: int,
) -> np.ndarray:
    """
    Patch merging layer for downsampling (conceptual).

    Concatenates 2x2 neighboring patches and projects to output_channels.

    Args:
        x: Input features (H, W, C)
        output_channels: Output channel dimension (typically 2C)

    Returns:
        Merged features (H/2, W/2, output_channels)
    """
    H, W, C = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Dimensions must be even"

    # Gather 2x2 patches
    x0 = x[0::2, 0::2, :]  # Top-left
    x1 = x[1::2, 0::2, :]  # Bottom-left
    x2 = x[0::2, 1::2, :]  # Top-right
    x3 = x[1::2, 1::2, :]  # Bottom-right

    # Concatenate along channel dimension: (H/2, W/2, 4C)
    x_concat = np.concatenate([x0, x1, x2, x3], axis=-1)

    # Note: In practice, a linear layer would project 4C -> 2C
    # This is a placeholder showing the concatenation structure
    return x_concat


def key_equations() -> Dict[str, str]:
    """Return key equations for Swin Transformer."""
    return {
        "window_attention": r"\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d} + B)V",
        "global_complexity": r"\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C",
        "window_complexity": r"\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC",
        "w_msa_block": r"\hat{z}^\ell = \text{W-MSA}(\text{LN}(z^{\ell-1})) + z^{\ell-1}",
        "sw_msa_block": r"\hat{z}^{\ell+1} = \text{SW-MSA}(\text{LN}(z^\ell)) + z^\ell",
        "relative_pos_bias": r"B = \hat{B}[\text{relative\_position\_index}]",
    }


def get_downstream_tasks() -> Dict[str, Dict]:
    """
    Downstream task performance with Swin Transformer backbone.

    Returns:
        Dictionary of task performance metrics
    """
    return {
        "ImageNet_Classification": {
            "model": "Swin-L (ImageNet-22k)",
            "metric": "Top-1 Accuracy",
            "score": "86.4%",
        },
        "COCO_Object_Detection": {
            "model": "Swin-L + Cascade Mask R-CNN",
            "metric": "box AP",
            "score": "58.7",
        },
        "COCO_Instance_Segmentation": {
            "model": "Swin-L + Cascade Mask R-CNN",
            "metric": "mask AP",
            "score": "51.1",
        },
        "ADE20K_Semantic_Segmentation": {
            "model": "Swin-L + UperNet",
            "metric": "mIoU",
            "score": "53.5",
        },
    }
