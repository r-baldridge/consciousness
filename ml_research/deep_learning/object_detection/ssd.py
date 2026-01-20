"""
SSD - Single Shot MultiBox Detector

This module documents the SSD object detection architecture,
a single-shot detector using multi-scale feature maps.

Paper: "SSD: Single Shot MultiBox Detector" (Liu et al., 2016)

Key Concept: Multi-scale feature maps for detecting objects at different sizes
    - Use feature maps from multiple layers of the backbone
    - Larger maps detect smaller objects (more spatial resolution)
    - Smaller maps detect larger objects (more semantic context)
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# SSD (2016) - Liu et al.
# =============================================================================

SSD = MLMethod(
    method_id="ssd_2016",
    name="SSD: Single Shot MultiBox Detector",
    year=2016,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Wei Liu", "Dragomir Anguelov", "Dumitru Erhan", "Christian Szegedy",
             "Scott Reed", "Cheng-Yang Fu", "Alexander C. Berg"],
    paper_title="SSD: Single Shot MultiBox Detector",
    paper_url="https://arxiv.org/abs/1512.02325",
    key_innovation="""
    Multi-scale feature map predictions in a single forward pass.

    Key Concepts:
    1. Base network (VGG16) truncated before classification layers
    2. Auxiliary feature layers added for additional scales
    3. Predictions at multiple resolutions (38x38, 19x19, 10x10, 5x5, 3x3, 1x1)
    4. Default boxes (anchors) with varying aspect ratios at each location

    Advantages over YOLO:
    - Better accuracy due to multi-scale predictions
    - Better detection of small objects (high-resolution feature maps)

    Advantages over Faster R-CNN:
    - Single-shot (no region proposal stage)
    - 3x faster while maintaining accuracy

    SSD300: 300x300 input, 59 FPS, 74.3% mAP on VOC2007
    SSD512: 512x512 input, 22 FPS, 76.8% mAP on VOC2007
    """,
    mathematical_formulation="""
    Multi-scale Feature Maps:
        VGG16 base network extracts features at multiple scales:

        Layer          | Size    | Default Boxes/Location | Total Boxes
        --------------|---------|------------------------|------------
        Conv4_3       | 38x38   | 4                      | 5776
        Conv7 (FC7)   | 19x19   | 6                      | 2166
        Conv8_2       | 10x10   | 6                      | 600
        Conv9_2       | 5x5     | 6                      | 150
        Conv10_2      | 3x3     | 4                      | 36
        Conv11_2      | 1x1     | 4                      | 4
        --------------|---------|------------------------|------------
        Total         |         |                        | 8732

    Default Box (Anchor) Definition:
        Scale at layer k:
            s_k = s_min + (s_max - s_min) * (k - 1) / (m - 1)

            Where s_min = 0.2, s_max = 0.9, m = number of feature maps

        Aspect ratios: a_r in {1, 2, 3, 1/2, 1/3}

        Box dimensions:
            w_k^a = s_k * sqrt(a_r)
            h_k^a = s_k / sqrt(a_r)

        Additional box for a_r = 1:
            s'_k = sqrt(s_k * s_{k+1})

    Prediction per Feature Map Location:
        For each default box:
            - 4 offsets: (delta_cx, delta_cy, delta_w, delta_h)
            - C+1 class scores (including background)

        Total predictions per location: k * (4 + C+1)

    Offset Parameterization:
        cx = d_cx + 0.1 * delta_cx * d_w
        cy = d_cy + 0.1 * delta_cy * d_h
        w = d_w * exp(0.2 * delta_w)
        h = d_h * exp(0.2 * delta_h)

        Where (d_cx, d_cy, d_w, d_h) is the default box

    Matching Strategy:
        1. Match each ground truth to default box with highest IoU
        2. Match default boxes to any ground truth with IoU > 0.5

        Ensures: Every ground truth has at least one matching default box

    Training Loss:
        L(x, c, l, g) = (1/N) * (L_conf(x, c) + alpha * L_loc(x, l, g))

        Where:
            - N: Number of matched default boxes
            - x_{ij}^p = {1, 0}: Match indicator for i-th default to j-th ground truth of class p
            - c: Class confidences
            - l: Predicted box offsets
            - g: Ground truth box offsets
            - alpha = 1 (weight balance)

        Localization Loss (Smooth L1):
        L_loc(x, l, g) = sum_{i in Pos} sum_{m in {cx,cy,w,h}} x_{ij}^k * smooth_L1(l_i^m - g_j^m)

        Confidence Loss (Softmax):
        L_conf(x, c) = -sum_{i in Pos} x_{ij}^p * log(c_i^p) - sum_{i in Neg} log(c_i^0)

        Where c_i^p = exp(c_i^p) / sum_p exp(c_i^p)
    """,
    predecessors=["faster_r_cnn_2015", "yolo_v1_2016", "multibox_2014"],
    successors=["dssd_2017", "fssd_2017", "retinanet_2017"],
    tags=["object_detection", "single_stage", "multi_scale", "anchor_boxes"],
)


# =============================================================================
# Supporting Functions and Concepts
# =============================================================================

def multi_scale_feature_maps() -> str:
    """
    Multi-scale feature map strategy for detecting objects at different sizes.
    """
    return """
    Multi-Scale Feature Map Detection:

    Intuition:
        - Early layers: High spatial resolution, low-level features
        - Later layers: Low spatial resolution, high-level semantic features

        Use both for detection:
        - Large feature maps (38x38): Small objects
        - Small feature maps (1x1): Large objects

    SSD Feature Map Sizes (300x300 input):
        38x38 -> 19x19 -> 10x10 -> 5x5 -> 3x3 -> 1x1

    Feature Pyramid Alternative:
        SSD: Top-down only (no feature fusion)
        FPN: Top-down + lateral connections for feature fusion

        FPN improves small object detection by adding high-level
        semantics to high-resolution feature maps.

    Receptive Field Coverage:
        Layer       | Receptive Field | Best for
        ------------|-----------------|------------------
        38x38       | Small           | Small objects
        19x19       | Medium          | Medium objects
        10x10       | Medium-Large    | Medium-Large objects
        5x5 - 1x1   | Large           | Large objects

    Computational Efficiency:
        Making predictions at smaller feature maps is cheap:
        - 38x38 with 4 boxes: 5776 predictions (most computation)
        - 1x1 with 4 boxes: 4 predictions (negligible)

        Total 8732 predictions computed in single forward pass.
    """


def default_boxes() -> str:
    """
    Default boxes (anchors) in SSD with multi-scale aspect ratios.
    """
    return """
    Default Boxes (Priors/Anchors):

    Design Philosophy:
        Cover the space of possible object shapes systematically
        with boxes of different scales and aspect ratios.

    Scale Calculation:
        For m feature maps with k in {1, ..., m}:

        s_k = s_min + (s_max - s_min) * (k - 1) / (m - 1)

        Default: s_min = 0.2, s_max = 0.9

        For 6 feature maps:
        s = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9] (relative to image)

    Aspect Ratios:
        Standard: {1, 2, 1/2, 3, 1/3}

        For aspect ratio a_r:
            width = s_k * sqrt(a_r)
            height = s_k / sqrt(a_r)

        Additional box for a_r = 1:
            s'_k = sqrt(s_k * s_{k+1})

    Number of Boxes per Location:
        - 4 boxes: {1, 1', 2, 1/2} - used at extreme scales
        - 6 boxes: {1, 1', 2, 1/2, 3, 1/3} - used at intermediate scales

    Box Center Locations:
        For feature map of size f x f:
        center_x = (i + 0.5) / f  for i in {0, ..., f-1}
        center_y = (j + 0.5) / f  for j in {0, ..., f-1}

    Total Boxes Example (SSD300):
        38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732
    """


def hard_negative_mining() -> str:
    """
    Hard negative mining strategy to handle class imbalance in SSD.
    """
    return """
    Hard Negative Mining:

    Problem:
        Most default boxes are negatives (background)
        Ratio of neg:pos can be 100:1 or worse
        Training dominated by easy negatives -> poor performance

    Solution: Hard Negative Mining
        Select hardest negative examples for training

        Algorithm:
        1. For each training image:
           a. Compute loss for all negative boxes
           b. Sort negatives by confidence loss (descending)
           c. Select top negatives such that neg:pos ratio <= 3:1

        This ensures:
        - Balanced training batches
        - Network focuses on hard examples
        - Easy negatives (sky, walls) ignored

    Loss Sorting:
        For negative box i, sort by:
        L_i = -log(c_i^0)  where c_i^0 = P(background)

        High loss = network incorrectly confident it's an object
        These are the "hard" negatives we want to train on.

    Implementation:
        def hard_negative_mining(conf_loss, pos_mask, neg_mask, ratio=3):
            # Count positives per image
            num_pos = pos_mask.sum(dim=1, keepdim=True)
            num_neg = num_pos * ratio

            # Sort negatives by loss
            neg_loss = conf_loss * neg_mask
            _, neg_idx = neg_loss.sort(dim=1, descending=True)

            # Select top-k negatives
            neg_rank = neg_idx.argsort(dim=1)
            hard_neg = neg_rank < num_neg

            return pos_mask | hard_neg

    Alternative: Focal Loss (RetinaNet)
        Instead of sampling, reweight loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Easy examples (high p_t) get down-weighted automatically.
    """
