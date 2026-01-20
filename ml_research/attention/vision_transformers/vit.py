"""
Vision Transformer (ViT) - 2020

Authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
         Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
         Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Research)

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
       ICLR 2021
       https://arxiv.org/abs/2010.11929

Key Innovation:
    - Treats images as sequences of patches (tokens), applying standard Transformer
    - Demonstrates pure attention-based models can match/exceed CNNs at scale
    - Simple, scalable architecture without convolutions

Architecture:
    1. Patch Embedding: Split image into fixed-size patches (16x16 or 14x14)
    2. Linear Projection: Flatten and project patches to embedding dimension
    3. Position Embeddings: Add learnable 1D position embeddings
    4. [CLS] Token: Prepend learnable classification token
    5. Transformer Encoder: Standard multi-head self-attention + MLP blocks
    6. MLP Head: Classification head on [CLS] token output

Mathematical Formulation:
    Image to patches:
        x in R^(H x W x C) -> x_p in R^(N x (P^2 * C))
        where N = HW/P^2 is number of patches, P is patch size

    Patch embedding:
        z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos
        where E in R^((P^2*C) x D) is patch projection, E_pos in R^((N+1) x D)

    Transformer encoder:
        z'_l = MSA(LN(z_{l-1})) + z_{l-1}           (Multi-head Self-Attention)
        z_l = MLP(LN(z'_l)) + z'_l                   (Feed-forward)

    Classification:
        y = LN(z_L^0)                                (Use [CLS] token output)

Pseudocode:
    ```
    def vit_forward(image, patch_size=16, hidden_dim=768, num_layers=12, num_heads=12):
        # 1. Create patches
        patches = image_to_patches(image, patch_size)  # (N, P*P*C)
        N = patches.shape[0]

        # 2. Linear projection
        patch_embeddings = linear(patches, hidden_dim)  # (N, D)

        # 3. Prepend [CLS] token
        cls_token = learnable_parameter(1, hidden_dim)
        tokens = concat([cls_token, patch_embeddings], dim=0)  # (N+1, D)

        # 4. Add position embeddings
        pos_embed = learnable_parameter(N+1, hidden_dim)
        tokens = tokens + pos_embed

        # 5. Transformer encoder blocks
        for layer in range(num_layers):
            # Multi-head self-attention with residual
            tokens_norm = layer_norm(tokens)
            attn_out = multi_head_attention(tokens_norm, tokens_norm, tokens_norm, num_heads)
            tokens = tokens + attn_out

            # MLP with residual
            tokens_norm = layer_norm(tokens)
            mlp_out = mlp(tokens_norm, hidden_dim * 4)
            tokens = tokens + mlp_out

        # 6. Classification head
        cls_output = layer_norm(tokens[0])  # Take [CLS] token
        logits = linear(cls_output, num_classes)

        return logits
    ```

Model Variants:
    - ViT-Base:  12 layers, 768 hidden, 12 heads, 86M params
    - ViT-Large: 24 layers, 1024 hidden, 16 heads, 307M params
    - ViT-Huge:  32 layers, 1280 hidden, 16 heads, 632M params

Training Requirements:
    - Requires large-scale pretraining (ImageNet-21k or JFT-300M)
    - Without large data, underperforms CNNs due to lack of inductive bias
    - Benefits from strong data augmentation

Historical Significance:
    - Demonstrated transformers can replace CNNs for vision tasks
    - Sparked explosion of vision transformer research
    - Foundation for multimodal models (CLIP, DALL-E, etc.)
    - Showed importance of scale in vision models

Limitations:
    - Requires large-scale pretraining data
    - Lacks inductive biases of CNNs (locality, translation equivariance)
    - Quadratic complexity O(N^2) with sequence length
    - Fixed resolution at inference time
"""

from typing import List, Dict, Optional
import numpy as np

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


FORMULATION = r"""
Vision Transformer (ViT):

Image to Patch Sequence:
    x \in \mathbb{R}^{H \times W \times C} \rightarrow
    \mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}

    where N = \frac{H \cdot W}{P^2} (number of patches), P = patch size

Patch Embedding with [CLS] Token:
    \mathbf{z}_0 = [x_{class}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; ...;
                   \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}

    where:
        \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D} - patch projection matrix
        \mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D} - position embeddings
        x_{class} - learnable [CLS] token

Transformer Encoder (L layers):
    \mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
    \mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell

Multi-Head Self-Attention:
    \text{MSA}(\mathbf{z}) = \text{Concat}(head_1, ..., head_h)\mathbf{W}^O
    head_i = \text{Attention}(\mathbf{z}\mathbf{W}_i^Q, \mathbf{z}\mathbf{W}_i^K, \mathbf{z}\mathbf{W}_i^V)
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

MLP Block:
    \text{MLP}(\mathbf{z}) = \text{GELU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2

Classification Output:
    \mathbf{y} = \text{LN}(\mathbf{z}_L^0)  \text{  (output from [CLS] token position)}
"""


# Research index entry
VISION_TRANSFORMER = MLMethod(
    method_id="vision_transformer_2020",
    name="Vision Transformer (ViT)",
    year=2020,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Alexey Dosovitskiy",
        "Lucas Beyer",
        "Alexander Kolesnikov",
        "Dirk Weissenborn",
        "Xiaohua Zhai",
        "Thomas Unterthiner",
        "Mostafa Dehghani",
        "Matthias Minderer",
        "Georg Heigold",
        "Sylvain Gelly",
        "Jakob Uszkoreit",
        "Neil Houlsby",
    ],
    paper_title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    paper_url="https://arxiv.org/abs/2010.11929",
    key_innovation=(
        "First pure transformer architecture for image classification, treating images "
        "as sequences of patches. Demonstrated that with sufficient pretraining data, "
        "transformers can match or exceed CNN performance without convolutions."
    ),
    mathematical_formulation=FORMULATION,
    predecessors=[
        "transformer_2017",
        "bert_2018",
        "resnet_2015",
    ],
    successors=[
        "deit_2021",
        "swin_transformer_2021",
        "clip_2021",
        "mae_2021",
        "beit_2021",
    ],
    tags=[
        "vision-transformer",
        "image-classification",
        "patch-embedding",
        "self-attention",
        "foundation-model",
        "transfer-learning",
        "google-research",
    ],
    notes=(
        "ViT showed that the lack of inductive biases in transformers (compared to CNNs) "
        "can be overcome with sufficient data and compute. This work catalyzed the shift "
        "from CNNs to transformers in computer vision. The patch-based tokenization became "
        "standard for vision transformers."
    ),
)


def image_to_patches(
    image: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Convert image to sequence of flattened patches.

    Args:
        image: Input image of shape (H, W, C) or (B, H, W, C)
        patch_size: Size of each square patch (default 16)

    Returns:
        Patches of shape (N, P*P*C) or (B, N, P*P*C)
        where N = (H/P) * (W/P)
    """
    if image.ndim == 3:
        # Single image (H, W, C)
        H, W, C = image.shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image dimensions must be divisible by patch_size"

        n_h = H // patch_size
        n_w = W // patch_size
        N = n_h * n_w

        # Reshape to (n_h, P, n_w, P, C) then to (N, P*P*C)
        patches = image.reshape(n_h, patch_size, n_w, patch_size, C)
        patches = patches.transpose(0, 2, 1, 3, 4)  # (n_h, n_w, P, P, C)
        patches = patches.reshape(N, -1)  # (N, P*P*C)

        return patches

    elif image.ndim == 4:
        # Batch of images (B, H, W, C)
        B, H, W, C = image.shape
        assert H % patch_size == 0 and W % patch_size == 0

        n_h = H // patch_size
        n_w = W // patch_size
        N = n_h * n_w

        patches = image.reshape(B, n_h, patch_size, n_w, patch_size, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)  # (B, n_h, n_w, P, P, C)
        patches = patches.reshape(B, N, -1)  # (B, N, P*P*C)

        return patches

    else:
        raise ValueError(f"Expected 3D or 4D image, got {image.ndim}D")


def get_position_embeddings(
    num_patches: int,
    hidden_dim: int,
    include_cls: bool = True,
) -> np.ndarray:
    """
    Initialize learnable position embeddings (conceptual).

    In practice, these are learnable parameters. This function shows
    the sinusoidal initialization option from the original transformer.

    Args:
        num_patches: Number of patches N
        hidden_dim: Embedding dimension D
        include_cls: Whether to include position for [CLS] token

    Returns:
        Position embeddings of shape (N+1, D) or (N, D)
    """
    seq_len = num_patches + 1 if include_cls else num_patches

    # Sinusoidal position encoding (alternative to learnable)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, hidden_dim, 2) * -(np.log(10000.0) / hidden_dim))

    pos_embed = np.zeros((seq_len, hidden_dim))
    pos_embed[:, 0::2] = np.sin(position * div_term)
    pos_embed[:, 1::2] = np.cos(position * div_term)

    return pos_embed


def get_model_config(variant: str = "base") -> Dict:
    """
    Get ViT model configuration for standard variants.

    Args:
        variant: One of 'tiny', 'small', 'base', 'large', 'huge'

    Returns:
        Dictionary with model hyperparameters
    """
    configs = {
        "tiny": {
            "hidden_dim": 192,
            "num_layers": 12,
            "num_heads": 3,
            "mlp_dim": 768,
            "patch_size": 16,
            "params": "5.7M",
        },
        "small": {
            "hidden_dim": 384,
            "num_layers": 12,
            "num_heads": 6,
            "mlp_dim": 1536,
            "patch_size": 16,
            "params": "22M",
        },
        "base": {
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "patch_size": 16,
            "params": "86M",
        },
        "large": {
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "mlp_dim": 4096,
            "patch_size": 16,
            "params": "307M",
        },
        "huge": {
            "hidden_dim": 1280,
            "num_layers": 32,
            "num_heads": 16,
            "mlp_dim": 5120,
            "patch_size": 14,
            "params": "632M",
        },
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    return configs[variant]


def key_equations() -> Dict[str, str]:
    """Return key equations for ViT."""
    return {
        "patch_embedding": r"z_0 = [x_{class}; x_p^1 E; ...; x_p^N E] + E_{pos}",
        "msa_residual": r"z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}",
        "mlp_residual": r"z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell",
        "attention": r"\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V",
        "output": r"y = \text{LN}(z_L^0)",
    }


def get_pretraining_datasets() -> Dict[str, Dict]:
    """Information about datasets used for ViT pretraining."""
    return {
        "ImageNet-1k": {
            "size": "1.28M images",
            "classes": 1000,
            "notes": "Standard benchmark, insufficient for ViT without augmentation",
        },
        "ImageNet-21k": {
            "size": "14M images",
            "classes": 21843,
            "notes": "Larger dataset, enables good ViT performance",
        },
        "JFT-300M": {
            "size": "300M images",
            "classes": 18291,
            "notes": "Google internal dataset, used for best ViT results",
        },
    }
