"""
R-CNN Family - Region-based Convolutional Neural Networks

This module documents the evolution of region-based object detection methods,
from the original R-CNN to Mask R-CNN for instance segmentation.

Timeline:
    - R-CNN (2014): Region proposals + CNN features
    - Fast R-CNN (2015): RoI pooling, single-stage training
    - Faster R-CNN (2015): Region Proposal Network (RPN)
    - Mask R-CNN (2017): Instance segmentation branch

Key Innovation: Two-stage detection pipeline
    1. Region proposal: Generate candidate object locations
    2. Classification: Classify and refine each proposal
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# R-CNN (2014) - Girshick et al.
# =============================================================================

R_CNN = MLMethod(
    method_id="r_cnn_2014",
    name="R-CNN (Regions with CNN features)",
    year=2014,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Ross Girshick", "Jeff Donahue", "Trevor Darrell", "Jitendra Malik"],
    paper_title="Rich feature hierarchies for accurate object detection and semantic segmentation",
    paper_url="https://arxiv.org/abs/1311.2524",
    key_innovation="""
    Combined selective search region proposals with CNN feature extraction.
    First successful application of deep CNNs to object detection, achieving
    significant improvements over traditional methods on PASCAL VOC.

    Pipeline:
    1. Generate ~2000 region proposals using Selective Search
    2. Warp each region to fixed size (227x227)
    3. Extract 4096-dim features using AlexNet
    4. Classify with class-specific linear SVMs
    5. Refine boxes with bounding box regression

    Limitations:
    - Slow inference (~47s per image): Each region processed independently
    - Multi-stage training: CNN, SVMs, and regressors trained separately
    - Large storage: Features cached to disk for SVM training

    Impact: Established CNN-based object detection as state-of-the-art,
    inspiring subsequent work on faster and more unified architectures.
    """,
    mathematical_formulation="""
    Region Proposal: Selective Search algorithm generates candidate regions R = {r_1, ..., r_n}

    Feature Extraction:
        phi(r_i) = CNN(warp(r_i)) in R^4096

    Classification (per class c):
        score_c(r_i) = w_c^T * phi(r_i)

    Bounding Box Regression:
        t_x = (G_x - P_x) / P_w
        t_y = (G_y - P_y) / P_h
        t_w = log(G_w / P_w)
        t_h = log(G_h / P_h)

    Where:
        - P = (P_x, P_y, P_w, P_h): Proposal box (center, width, height)
        - G = (G_x, G_y, G_w, G_h): Ground truth box
        - t = (t_x, t_y, t_w, t_h): Regression targets

    Training Loss (SVM):
        L = sum_i max(0, 1 - y_i * f(x_i))  (hinge loss)
    """,
    predecessors=["alexnet_2012", "selective_search_2013"],
    successors=["fast_r_cnn_2015", "spp_net_2014"],
    tags=["object_detection", "region_based", "two_stage", "selective_search"],
)


# =============================================================================
# Fast R-CNN (2015) - Girshick
# =============================================================================

FAST_R_CNN = MLMethod(
    method_id="fast_r_cnn_2015",
    name="Fast R-CNN",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Ross Girshick"],
    paper_title="Fast R-CNN",
    paper_url="https://arxiv.org/abs/1504.08083",
    key_innovation="""
    Single-stage training with shared CNN computation and RoI pooling.

    Key Improvements over R-CNN:
    1. Single CNN forward pass for entire image (not per-region)
    2. RoI Pooling: Extract fixed-size features from arbitrary regions
    3. Multi-task loss: Joint classification and bbox regression
    4. End-to-end training (except region proposals)

    Result: 9x faster training, 213x faster inference than R-CNN.

    RoI Pooling enables sharing CNN computation across all proposals.
    Still depends on external region proposal method (Selective Search).
    """,
    mathematical_formulation="""
    RoI Pooling:
        Given feature map F of size H x W x C and region (r_x, r_y, r_w, r_h):
        1. Project region onto feature map
        2. Divide into H' x W' grid (e.g., 7x7)
        3. Max pool each grid cell
        Output: H' x W' x C fixed-size feature

    Multi-task Loss:
        L(p, u, t^u, v) = L_cls(p, u) + lambda * [u >= 1] * L_loc(t^u, v)

    Where:
        - p = (p_0, ..., p_K): Softmax class probabilities
        - u: Ground truth class label
        - t^u = (t_x^u, t_y^u, t_w^u, t_h^u): Predicted bbox for class u
        - v = (v_x, v_y, v_w, v_h): Ground truth bbox regression targets

    Classification Loss:
        L_cls(p, u) = -log(p_u)  (cross-entropy)

    Localization Loss (Smooth L1):
        L_loc(t^u, v) = sum_i smooth_L1(t_i^u - v_i)

        smooth_L1(x) = 0.5 * x^2     if |x| < 1
                       |x| - 0.5     otherwise
    """,
    predecessors=["r_cnn_2014", "spp_net_2014"],
    successors=["faster_r_cnn_2015"],
    tags=["object_detection", "region_based", "two_stage", "roi_pooling", "multi_task"],
)


# =============================================================================
# Faster R-CNN (2015) - Ren et al.
# =============================================================================

FASTER_R_CNN = MLMethod(
    method_id="faster_r_cnn_2015",
    name="Faster R-CNN",
    year=2015,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Shaoqing Ren", "Kaiming He", "Ross Girshick", "Jian Sun"],
    paper_title="Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
    paper_url="https://arxiv.org/abs/1506.01497",
    key_innovation="""
    Region Proposal Network (RPN): Learnable region proposals sharing CNN features.

    Key Components:
    1. Backbone CNN: Shared feature extraction (VGG16, ResNet)
    2. RPN: Predicts objectness + bbox at each spatial location
    3. RoI Pooling + Detection Head: Classification and refinement

    Anchor Boxes: Multi-scale, multi-aspect-ratio reference boxes at each location.
    Default: 3 scales x 3 ratios = 9 anchors per location

    Result: ~10x faster than Fast R-CNN, enabling real-time detection.

    Anchor boxes became a fundamental concept in object detection, later
    adopted by single-stage detectors (YOLO v2+, SSD).
    The alternating training procedure was later simplified to joint training.
    Feature Pyramid Networks (FPN) further improved multi-scale detection.
    """,
    mathematical_formulation="""
    Region Proposal Network (RPN):
        Input: Feature map F of size H x W x C

        For each spatial location (i, j):
            - k anchor boxes {a_1, ..., a_k} of different scales/ratios
            - Predict: objectness scores + bbox deltas for each anchor

        Objectness Score:
            p_obj = sigmoid(W_cls * f_ij)  in [0, 1]

        Bounding Box Regression:
            t = (t_x, t_y, t_w, t_h) = W_reg * f_ij

    Anchor Definition:
        anchor(i, j, s, r) = (x_i, y_j, s * sqrt(r), s / sqrt(r))
        - Scales s in {128, 256, 512} pixels
        - Aspect ratios r in {0.5, 1, 2}

    RPN Loss:
        L_RPN = (1/N_cls) * sum_i L_cls(p_i, p_i*)
              + lambda * (1/N_reg) * sum_i p_i* * L_reg(t_i, t_i*)

        Where:
            - p_i*: Ground truth objectness (1 if IoU > 0.7, 0 if IoU < 0.3)
            - t_i*: Ground truth bbox regression targets

    IoU (Intersection over Union):
        IoU(A, B) = |A intersection B| / |A union B|

    Two-Stage Training:
        1. Train RPN
        2. Train Fast R-CNN using RPN proposals
        3. Fine-tune RPN with shared conv layers
        4. Fine-tune detection head
    """,
    predecessors=["fast_r_cnn_2015"],
    successors=["mask_r_cnn_2017", "fpn_2017"],
    tags=["object_detection", "region_based", "two_stage", "rpn", "anchor_boxes"],
)


# =============================================================================
# Mask R-CNN (2017) - He et al.
# =============================================================================

MASK_R_CNN = MLMethod(
    method_id="mask_r_cnn_2017",
    name="Mask R-CNN",
    year=2017,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Kaiming He", "Georgia Gkioxari", "Piotr Dollar", "Ross Girshick"],
    paper_title="Mask R-CNN",
    paper_url="https://arxiv.org/abs/1703.06870",
    key_innovation="""
    Extended Faster R-CNN with parallel instance segmentation branch.

    Key Contributions:
    1. Mask branch: Per-pixel binary mask for each detected object
    2. RoI Align: Bilinear interpolation for precise spatial alignment
    3. Decoupled mask and class prediction: Binary mask per class

    RoI Align fixes the quantization issues of RoI Pooling, enabling
    pixel-accurate mask predictions.

    Mask R-CNN became the foundation for many instance segmentation methods.
    The decoupled mask prediction (binary mask per class vs. multi-class mask)
    was a key design choice that improved results.

    Applications: Instance segmentation, human pose estimation (Keypoint R-CNN),
    panoptic segmentation.
    """,
    mathematical_formulation="""
    Architecture:
        Faster R-CNN backbone + FPN + Mask Head

    RoI Align (vs RoI Pooling):
        Instead of quantizing region boundaries:
        1. Divide RoI into H' x W' bins
        2. Sample 4 points per bin using bilinear interpolation
        3. Aggregate (max or average pool)

        Bilinear Interpolation at (x, y):
            f(x, y) = sum_{i,j in neighbors} w_ij * f(x_i, y_j)
            w_ij = (1 - |x - x_i|) * (1 - |y - y_j|)

    Mask Branch:
        Input: 14x14xC RoI features (from RoI Align)
        Architecture: 4 conv layers + deconv -> 28x28xK mask
        Output: K binary masks (one per class), each 28x28

        For class k:
            m_k(i, j) = sigmoid(F_k(i, j))  in [0, 1]

    Multi-task Loss:
        L = L_cls + L_box + L_mask

        Mask Loss (per-pixel binary cross-entropy):
        L_mask = -(1/m^2) * sum_{i,j} [y_ij * log(m_k(i,j))
                                      + (1-y_ij) * log(1-m_k(i,j))]

        Where:
            - k: Ground truth class
            - y_ij: Ground truth mask pixel
            - m: Mask resolution (28)

    Key Insight: Predicting K independent masks (one per class) avoids
    competition between classes, improving both mask and classification quality.
    """,
    predecessors=["faster_r_cnn_2015", "fpn_2017", "fcn_2014"],
    successors=["cascade_mask_rcnn_2018", "pointrend_2020"],
    tags=["object_detection", "instance_segmentation", "region_based", "roi_align"],
)


# =============================================================================
# Supporting Functions and Concepts
# =============================================================================

def selective_search() -> str:
    """
    Selective Search algorithm for generating region proposals.

    Used in R-CNN and Fast R-CNN before the introduction of RPN.
    """
    return """
    Selective Search Algorithm (Uijlings et al., 2013):

    Hierarchical grouping strategy:
    1. Start with pixel-level segmentation (graph-based)
    2. Greedily merge similar regions based on:
       - Color similarity (histogram intersection)
       - Texture similarity (SIFT-like gradients)
       - Size similarity (prefer merging small regions)
       - Fill similarity (prefer regions that fill gaps)
    3. Continue until single region remains
    4. Output all intermediate regions as proposals

    Similarity:
        S(r_i, r_j) = a_1 * S_color + a_2 * S_texture + a_3 * S_size + a_4 * S_fill

    Typically generates ~2000 proposals per image covering ~98% of objects.
    """


def roi_pooling() -> str:
    """
    RoI Pooling operation for extracting fixed-size features from arbitrary regions.

    Introduced in Fast R-CNN.
    """
    return """
    RoI Pooling:

    Input:
        - Feature map: H x W x C
        - Region of Interest: (x, y, w, h)
        - Output size: H' x W' (e.g., 7x7)

    Algorithm:
        1. Project RoI onto feature map (divide by stride)
        2. Quantize coordinates to integers
        3. Divide into H' x W' grid
        4. Max pool each grid cell

    Output: H' x W' x C

    Issue: Quantization causes spatial misalignment (~0.5 pixel error)
    Solution: RoI Align (Mask R-CNN)
    """


def region_proposal_network() -> str:
    """
    Region Proposal Network (RPN) architecture and training.

    Introduced in Faster R-CNN.
    """
    return """
    Region Proposal Network (RPN):

    Architecture:
        1. 3x3 conv with 512 channels (shared)
        2. Two sibling 1x1 conv layers:
           - cls: 2k scores (objectness for k anchors)
           - reg: 4k coordinates (bbox deltas for k anchors)

    Anchor Design:
        - k anchors per spatial location
        - Multiple scales: {128^2, 256^2, 512^2}
        - Multiple ratios: {1:2, 1:1, 2:1}
        - Total: k = 9 anchors

    Positive/Negative Assignment:
        - Positive: IoU > 0.7 with any GT box, or highest IoU with GT
        - Negative: IoU < 0.3 with all GT boxes
        - Ignore: 0.3 <= IoU <= 0.7

    Training:
        - Mini-batch: 256 anchors (128 pos, 128 neg)
        - Loss: Multi-task (classification + regression)
    """


def roi_align() -> str:
    """
    RoI Align operation for precise spatial feature extraction.

    Introduced in Mask R-CNN to fix quantization issues in RoI Pooling.
    """
    return """
    RoI Align:

    Key Difference from RoI Pooling:
        - No quantization of region boundaries
        - Use bilinear interpolation for sub-pixel sampling

    Algorithm:
        1. Project RoI onto feature map (floating point)
        2. Divide into H' x W' bins
        3. For each bin:
           a. Sample 4 regular points (2x2 grid)
           b. Compute each point's value via bilinear interpolation
           c. Aggregate (max or average)

    Bilinear Interpolation:
        f(x, y) = (1-dx)(1-dy) * f[x0,y0] + dx(1-dy) * f[x1,y0]
                + (1-dx)dy * f[x0,y1] + dx*dy * f[x1,y1]

        Where:
            - (x0, y0), (x1, y1): Neighboring integer coordinates
            - dx = x - x0, dy = y - y0

    Impact: Improved mask AP by ~3% on COCO by enabling pixel-accurate masks.
    """
