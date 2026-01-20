"""
Segment Anything Model (SAM) - 2023

Authors: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland,
         Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg,
         Wan-Yen Lo, Piotr Dollar, Ross Girshick (Meta AI Research / FAIR)

Paper: "Segment Anything"
       ICCV 2023
       https://arxiv.org/abs/2304.02643

Key Innovation:
    - Foundation model for promptable image segmentation
    - Can segment any object with zero-shot generalization
    - Multiple prompt types: points, boxes, masks, text
    - SA-1B dataset: 1 billion masks on 11 million images

Architecture:
    1. Image Encoder: MAE-pretrained ViT (heavy, runs once per image)
    2. Prompt Encoder: Encodes sparse (points, boxes) and dense (masks) prompts
    3. Mask Decoder: Lightweight transformer decoder predicting masks

    Image Encoder:
        - ViT-H (632M params) pretrained with MAE
        - Processes image to create embeddings
        - Run once, cached for multiple prompts

    Prompt Encoder:
        - Points: Positional encoding + learned embeddings for foreground/background
        - Boxes: Positional encoding of corner points
        - Masks: Convolutional embedding
        - Text: CLIP text encoder (optional)

    Mask Decoder:
        - Two-way transformer: tokens attend to image, image attends to tokens
        - Outputs multiple masks with confidence scores
        - Handles ambiguity (multiple valid segmentations)

Mathematical Formulation:
    Image Encoding:
        F = ImageEncoder(I)
        F in R^(H/16 x W/16 x C), C=256

    Prompt Encoding:
        For point (x, y) with label l in {0, 1} (background/foreground):
            p = PE(x, y) + E_l, where E_l is learned label embedding

        For box (x1, y1, x2, y2):
            p = [PE(x1, y1) + E_tl; PE(x2, y2) + E_br]

        For mask M:
            p_dense = Conv(M), downsampled to match F

    Mask Decoder (two-way attention):
        tokens = [IoU_token; mask_tokens; prompt_tokens]
        For each layer:
            tokens = self_attention(tokens)
            tokens = cross_attention(tokens -> image_embedding)
            image_embedding = cross_attention(image_embedding -> tokens)
            tokens = MLP(tokens)

    Output:
        masks = MLP(mask_tokens) @ upsampled_image_embedding
        IoU_scores = MLP(IoU_token)

Pseudocode:
    ```
    def sam_forward(image, prompts):
        # 1. Image encoding (run once per image)
        image_embedding = image_encoder(image)  # ViT-H: (64, 64, 256)

        # 2. Encode each prompt
        sparse_embeddings, dense_embeddings = prompt_encoder(prompts)
        # sparse: points, boxes -> (num_prompts, embed_dim)
        # dense: masks -> (H/16, W/16, embed_dim)

        # 3. Mask decoder
        masks, iou_scores = mask_decoder(
            image_embedding=image_embedding,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        # masks: (num_masks, H, W) - multiple masks per prompt
        # iou_scores: (num_masks,) - confidence for each mask

        return masks, iou_scores

    def mask_decoder(image_embedding, sparse_prompts, dense_prompts):
        # Prepare tokens
        output_tokens = concat([iou_token, mask_tokens], sparse_prompts)

        # Add dense prompt to image embedding
        image_embedding = image_embedding + dense_prompts

        # Two-way transformer layers
        for layer in transformer_layers:
            # Self-attention among tokens
            output_tokens = self_attention(output_tokens)

            # Cross-attention: tokens attend to image
            output_tokens = cross_attention(
                q=output_tokens, k=image_embedding, v=image_embedding
            )

            # Cross-attention: image attends to tokens
            image_embedding = cross_attention(
                q=image_embedding, k=output_tokens, v=output_tokens
            )

            output_tokens = mlp(output_tokens)

        # Generate masks
        iou_pred = mlp(output_tokens[0])  # IoU prediction
        mask_tokens_out = output_tokens[1:num_masks+1]
        upscaled_embedding = upsample(image_embedding)  # 4x
        masks = mask_tokens_out @ upscaled_embedding

        return masks, iou_pred
    ```

SA-1B Dataset:
    - 11 million diverse, high-resolution images
    - 1.1 billion high-quality segmentation masks
    - Automatic mask generation using SAM in the loop
    - 400x more masks than any prior dataset

Model Variants:
    - SAM (ViT-H): 632M params, highest quality
    - SAM (ViT-L): 308M params, good quality/speed tradeoff
    - SAM (ViT-B): 91M params, faster inference

Training:
    - Pre-train image encoder with MAE
    - Joint training of prompt encoder and mask decoder
    - Focal loss + Dice loss for mask prediction
    - Trained on SA-1B with automatic + interactive masks

Historical Significance:
    - First foundation model for image segmentation
    - Enabled zero-shot segmentation of any object
    - Sparked "segment anything" research direction
    - Foundation for SAM 2 (video), Grounded SAM, etc.

Limitations:
    - Large image encoder (slow without caching)
    - Struggles with fine details (thin structures)
    - No semantic understanding (segments anything, not specific classes)
    - Requires prompts (not fully automatic)
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


FORMULATION = r"""
Segment Anything Model (SAM):

Image Encoding:
    \mathbf{F} = \text{ImageEncoder}(I)
    \mathbf{F} \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times C}

    where ImageEncoder is MAE-pretrained ViT-H, C = 256

Point Prompt Encoding:
    \mathbf{p}_{point} = \text{PE}(x, y) + \mathbf{E}_l

    where l \in \{0, 1\} indicates background/foreground

Box Prompt Encoding:
    \mathbf{p}_{box} = [\text{PE}(x_1, y_1) + \mathbf{E}_{tl}; \text{PE}(x_2, y_2) + \mathbf{E}_{br}]

Mask Prompt Encoding:
    \mathbf{p}_{mask} = \text{Conv}(M_{input})

Two-Way Transformer Attention:
    Layer updates for tokens T and image embedding F:
    \mathbf{T}' = \text{SelfAttn}(\mathbf{T}) + \mathbf{T}
    \mathbf{T}'' = \text{CrossAttn}(\mathbf{T}', \mathbf{F}) + \mathbf{T}'
    \mathbf{F}' = \text{CrossAttn}(\mathbf{F}, \mathbf{T}'') + \mathbf{F}
    \mathbf{T}_{out} = \text{MLP}(\mathbf{T}'') + \mathbf{T}''

Mask Prediction:
    \mathbf{M} = \mathbf{t}_{mask} \cdot \text{Upsample}(\mathbf{F}')^T

    where t_{mask} is the output mask token

IoU Prediction:
    \hat{IoU} = \text{MLP}(\mathbf{t}_{iou})

Training Loss:
    \mathcal{L} = \lambda_{focal} \mathcal{L}_{focal} + \lambda_{dice} \mathcal{L}_{dice} + \lambda_{iou} \mathcal{L}_{iou}
"""


# Research index entry
SEGMENT_ANYTHING = MLMethod(
    method_id="segment_anything_2023",
    name="Segment Anything Model (SAM)",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
    authors=[
        "Alexander Kirillov",
        "Eric Mintun",
        "Nikhila Ravi",
        "Hanzi Mao",
        "Chloe Rolland",
        "Laura Gustafson",
        "Tete Xiao",
        "Spencer Whitehead",
        "Alexander C. Berg",
        "Wan-Yen Lo",
        "Piotr Dollar",
        "Ross Girshick",
    ],
    paper_title="Segment Anything",
    paper_url="https://arxiv.org/abs/2304.02643",
    key_innovation=(
        "Foundation model for promptable image segmentation with zero-shot generalization. "
        "Introduced the SA-1B dataset (1B masks) and demonstrated that a single model "
        "can segment any object given point, box, or mask prompts."
    ),
    mathematical_formulation=FORMULATION,
    predecessors=[
        "vision_transformer_2020",
        "mae_2021",
        "clip_2021",
        "mask_rcnn_2017",
    ],
    successors=[
        "sam_2_2024",
        "grounded_sam_2023",
        "fastsam_2023",
        "mobile_sam_2023",
        "efficient_sam_2024",
    ],
    tags=[
        "foundation-model",
        "segmentation",
        "promptable",
        "zero-shot",
        "vision-transformer",
        "meta-ai",
        "sa-1b-dataset",
        "interactive-segmentation",
    ],
    notes=(
        "SAM established the paradigm of foundation models for image segmentation. "
        "The SA-1B dataset is the largest segmentation dataset ever created. "
        "The two-way transformer decoder enables efficient multi-mask prediction. "
        "SAM 2 extended this to video with streaming memory."
    ),
)


def get_model_variants() -> Dict[str, Dict]:
    """
    SAM model variants with different ViT backbones.

    Returns:
        Dictionary of model variants and their specifications
    """
    return {
        "SAM-ViT-B": {
            "image_encoder": "ViT-Base",
            "encoder_params": "86M",
            "total_params": "91M",
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "notes": "Fastest, suitable for real-time applications",
        },
        "SAM-ViT-L": {
            "image_encoder": "ViT-Large",
            "encoder_params": "304M",
            "total_params": "308M",
            "encoder_embed_dim": 1024,
            "encoder_depth": 24,
            "encoder_num_heads": 16,
            "notes": "Good quality/speed tradeoff",
        },
        "SAM-ViT-H": {
            "image_encoder": "ViT-Huge",
            "encoder_params": "628M",
            "total_params": "632M",
            "encoder_embed_dim": 1280,
            "encoder_depth": 32,
            "encoder_num_heads": 16,
            "notes": "Highest quality, used for dataset generation",
        },
    }


def get_prompt_types() -> Dict[str, Dict]:
    """
    Different prompt types supported by SAM.

    Returns:
        Dictionary describing each prompt type
    """
    return {
        "point": {
            "input": "Single (x, y) coordinate",
            "label": "Foreground (1) or Background (0)",
            "encoding": "Positional encoding + learned label embedding",
            "use_case": "Quick single-click segmentation",
            "ambiguity": "High - can produce multiple valid masks",
        },
        "points_multiple": {
            "input": "Multiple (x, y) coordinates with labels",
            "encoding": "Sum of individual point encodings",
            "use_case": "Refining segmentation with positive/negative points",
            "ambiguity": "Lower - more constraints reduce ambiguity",
        },
        "box": {
            "input": "Bounding box (x1, y1, x2, y2)",
            "encoding": "Positional encodings of corners + learned embeddings",
            "use_case": "Object detection to segmentation",
            "ambiguity": "Low - box typically specifies single object",
        },
        "mask": {
            "input": "Coarse binary mask",
            "encoding": "Convolutional embedding, added to image features",
            "use_case": "Mask refinement, iterative segmentation",
            "ambiguity": "Very low - explicit spatial prior",
        },
        "text": {
            "input": "Text description (optional, via CLIP)",
            "encoding": "CLIP text encoder embedding",
            "use_case": "Open-vocabulary segmentation",
            "notes": "Not included in base SAM, added in variants like Grounded SAM",
        },
    }


def get_sa1b_dataset_info() -> Dict[str, any]:
    """
    Information about the SA-1B dataset.

    Returns:
        Dictionary with dataset statistics and properties
    """
    return {
        "name": "SA-1B (Segment Anything 1 Billion)",
        "images": "11 million",
        "masks": "1.1 billion",
        "masks_per_image": "~100 average",
        "image_resolution": "~1500x2250 average",
        "image_source": "Licensed from photo providers",
        "annotation_method": "Model-assisted with SAM in the loop",
        "annotation_stages": [
            "1. Assisted-manual: Annotators click, SAM predicts",
            "2. Semi-automatic: SAM auto-generates, annotators refine",
            "3. Fully automatic: SAM generates all masks",
        ],
        "comparison": {
            "COCO": "164k images, 1.5M masks",
            "OpenImages": "9M images, 2.7M masks",
            "SA-1B": "11M images, 1.1B masks (400x more masks)",
        },
        "diversity": "Geographic and income diversity ensured",
        "license": "Apache 2.0 (masks), varies (images)",
    }


def get_architecture_details() -> Dict[str, Dict]:
    """
    Detailed architecture specifications for SAM components.

    Returns:
        Dictionary with component specifications
    """
    return {
        "image_encoder": {
            "type": "ViT with MAE pretraining",
            "input_resolution": "1024 x 1024",
            "patch_size": 16,
            "output_resolution": "64 x 64",
            "output_channels": 256,
            "window_attention": True,
            "global_attention_blocks": [5, 11, 17, 23],
            "notes": "Run once per image, cached for multiple prompts",
        },
        "prompt_encoder": {
            "sparse_embed_dim": 256,
            "dense_embed_dim": 256,
            "positional_encoding": "Fourier features",
            "num_point_embeddings": 4,  # pos, neg, top-left, bottom-right
            "mask_embed_channels": [16, 32, 256],
        },
        "mask_decoder": {
            "transformer_dim": 256,
            "num_heads": 8,
            "num_layers": 2,
            "mlp_dim": 2048,
            "num_mask_tokens": 4,  # 3 masks + 1 for multi-mask
            "iou_head_layers": 3,
            "upscaling": "2x transposed conv",
        },
    }


def positional_encoding_2d(
    embed_dim: int,
    height: int,
    width: int,
    temperature: float = 10000.0,
) -> np.ndarray:
    """
    Generate 2D positional encoding for spatial coordinates.

    Args:
        embed_dim: Embedding dimension
        height: Height of the grid
        width: Width of the grid
        temperature: Temperature for frequency scaling

    Returns:
        Positional encoding of shape (height, width, embed_dim)
    """
    y_embed = np.arange(height)[:, np.newaxis]
    x_embed = np.arange(width)[np.newaxis, :]

    # Normalize to [0, 1]
    y_embed = y_embed / height
    x_embed = x_embed / width

    dim_t = np.arange(embed_dim // 4)
    dim_t = temperature ** (2 * (dim_t // 2) / (embed_dim // 4))

    pos_y = y_embed[:, :, np.newaxis] / dim_t
    pos_x = x_embed[:, :, np.newaxis] / dim_t

    pos_y = np.stack([np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])], axis=-1)
    pos_x = np.stack([np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])], axis=-1)

    pos_y = pos_y.reshape(height, width, -1)
    pos_x = pos_x.reshape(height, width, -1)

    pos = np.concatenate([pos_y, pos_x], axis=-1)

    return pos


def encode_point_prompt(
    point: Tuple[int, int],
    label: int,
    image_size: Tuple[int, int],
    embed_dim: int = 256,
) -> np.ndarray:
    """
    Encode a point prompt (conceptual implementation).

    Args:
        point: (x, y) coordinates
        label: 0 for background, 1 for foreground
        image_size: (height, width) of the image
        embed_dim: Embedding dimension

    Returns:
        Point embedding of shape (embed_dim,)
    """
    x, y = point
    h, w = image_size

    # Normalize coordinates
    x_norm = x / w
    y_norm = y / h

    # Generate positional encoding (simplified Fourier features)
    num_pos_feats = embed_dim // 4
    temperature = 10000.0

    dim_t = np.arange(num_pos_feats)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_norm / dim_t
    pos_y = y_norm / dim_t

    pe_x = np.concatenate([np.sin(pos_x[::2]), np.cos(pos_x[1::2])])
    pe_y = np.concatenate([np.sin(pos_y[::2]), np.cos(pos_y[1::2])])

    pos_encoding = np.concatenate([pe_x, pe_y])

    # Add learned label embedding (simulated)
    label_embedding = np.random.randn(embed_dim) * 0.1  # Would be learned
    if label == 0:  # background
        label_embedding = -label_embedding

    point_embedding = pos_encoding[:embed_dim] + label_embedding

    return point_embedding


def encode_box_prompt(
    box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    embed_dim: int = 256,
) -> np.ndarray:
    """
    Encode a box prompt (conceptual implementation).

    Args:
        box: (x1, y1, x2, y2) bounding box coordinates
        image_size: (height, width) of the image
        embed_dim: Embedding dimension

    Returns:
        Box embedding of shape (2, embed_dim) for corners
    """
    x1, y1, x2, y2 = box
    h, w = image_size

    # Encode top-left corner
    tl_embed = encode_point_prompt((x1, y1), 1, image_size, embed_dim)

    # Encode bottom-right corner
    br_embed = encode_point_prompt((x2, y2), 1, image_size, embed_dim)

    # Add learned corner embeddings (simulated)
    corner_embed_tl = np.random.randn(embed_dim) * 0.1
    corner_embed_br = np.random.randn(embed_dim) * 0.1

    box_embedding = np.stack([
        tl_embed + corner_embed_tl,
        br_embed + corner_embed_br
    ])

    return box_embedding


def key_equations() -> Dict[str, str]:
    """Return key equations for SAM."""
    return {
        "image_encoding": r"F = \text{ImageEncoder}(I) \in \mathbb{R}^{64 \times 64 \times 256}",
        "point_encoding": r"p_{point} = \text{PE}(x, y) + E_l, \quad l \in \{0, 1\}",
        "box_encoding": r"p_{box} = [\text{PE}(x_1, y_1) + E_{tl}; \text{PE}(x_2, y_2) + E_{br}]",
        "two_way_attn": r"T' = \text{CrossAttn}(T, F), \quad F' = \text{CrossAttn}(F, T')",
        "mask_prediction": r"M = t_{mask} \cdot \text{Upsample}(F')^T",
        "iou_prediction": r"\hat{IoU} = \text{MLP}(t_{iou})",
        "training_loss": r"\mathcal{L} = \mathcal{L}_{focal} + \mathcal{L}_{dice} + \mathcal{L}_{iou}",
    }


def get_downstream_applications() -> Dict[str, Dict]:
    """
    Applications and extensions of SAM.

    Returns:
        Dictionary of downstream applications
    """
    return {
        "Grounded_SAM": {
            "description": "Combines Grounding DINO with SAM for text-prompted segmentation",
            "input": "Image + text description",
            "pipeline": "Text -> Grounding DINO -> Boxes -> SAM -> Masks",
        },
        "SAM_2": {
            "description": "Extension to video with memory mechanism",
            "input": "Video + prompts on any frame",
            "key_feature": "Streaming memory for temporal consistency",
        },
        "Mobile_SAM": {
            "description": "Distilled lightweight version",
            "encoder": "TinyViT",
            "speedup": "~50x faster on mobile",
        },
        "Fast_SAM": {
            "description": "Real-time SAM using YOLO backbone",
            "architecture": "YOLOv8-seg",
            "tradeoff": "Speed vs accuracy",
        },
        "Medical_SAM": {
            "description": "Fine-tuned for medical image segmentation",
            "domains": ["CT", "MRI", "X-ray", "Ultrasound"],
        },
        "Track_Anything": {
            "description": "Combines SAM with video object tracking",
            "capability": "Track and segment objects through video",
        },
    }
