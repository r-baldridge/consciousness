"""
YOLO Family - You Only Look Once

This module documents the evolution of YOLO object detection models,
from YOLOv1 (2016) through YOLOv8 (2023).

Key Concept: Single-shot detection using S x S grid
    - Divide image into S x S grid cells
    - Each cell predicts B bounding boxes and class probabilities
    - Direct regression from image pixels to detection outputs

Timeline:
    - YOLOv1 (2016): Original single-shot detector
    - YOLOv2 (2017): Batch norm, anchor boxes, multi-scale
    - YOLOv3 (2018): Feature pyramid, multi-scale predictions
    - YOLOv4 (2020): CSPNet, PANet, Mish activation
    - YOLOv5 (2020): PyTorch implementation, auto-learning anchors
    - YOLOv8 (2023): Anchor-free, decoupled head
"""

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# YOLOv1 (2016) - Redmon et al.
# =============================================================================

YOLO_V1 = MLMethod(
    method_id="yolo_v1_2016",
    name="YOLO v1",
    year=2016,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Joseph Redmon", "Santosh Divvala", "Ross Girshick", "Ali Farhadi"],
    paper_title="You Only Look Once: Unified, Real-Time Object Detection",
    paper_url="https://arxiv.org/abs/1506.02640",
    key_innovation="""
    First unified, real-time object detection system.

    Key Concepts:
    1. Single neural network predicts bboxes and class probabilities directly
    2. Frame detection as regression problem (not classification)
    3. Global reasoning: sees entire image during prediction

    Grid-based Prediction:
    - Divide image into S x S grid (S=7)
    - Each cell predicts B bounding boxes (B=2) and C class probabilities
    - Each bbox: (x, y, w, h, confidence)

    Speed: 45 FPS (Fast YOLO: 155 FPS) vs R-CNN family (~0.5 FPS)
    """,
    mathematical_formulation="""
    Grid Prediction (S x S x (B*5 + C)):
        For each grid cell (i, j):
            - B bounding boxes, each with 5 values:
              (x, y): Center relative to cell, in [0, 1]
              (w, h): Relative to image size, in [0, 1]
              confidence: P(Object) * IoU(pred, truth)

            - C class probabilities: P(Class_c | Object)

    Output Tensor: S x S x (B*5 + C)
        For PASCAL VOC: 7 x 7 x (2*5 + 20) = 7 x 7 x 30

    Detection Score at Test Time:
        P(Class_c) * IoU = P(Class_c | Object) * P(Object) * IoU(pred, truth)

    Loss Function (Multi-part):
        L = lambda_coord * L_xy + lambda_coord * L_wh + L_conf_obj + lambda_noobj * L_conf_noobj + L_class

        Localization Loss:
        L_xy = sum_{i=0}^{S^2} sum_{j=0}^{B} 1_{ij}^obj [(x_i - x_i_hat)^2 + (y_i - y_i_hat)^2]

        L_wh = sum_{i=0}^{S^2} sum_{j=0}^{B} 1_{ij}^obj [(sqrt(w_i) - sqrt(w_i_hat))^2 + (sqrt(h_i) - sqrt(h_i_hat))^2]

        Confidence Loss:
        L_conf_obj = sum_{i=0}^{S^2} sum_{j=0}^{B} 1_{ij}^obj (C_i - C_i_hat)^2
        L_conf_noobj = sum_{i=0}^{S^2} sum_{j=0}^{B} 1_{ij}^noobj (C_i - C_i_hat)^2

        Classification Loss:
        L_class = sum_{i=0}^{S^2} 1_i^obj sum_{c in classes} (p_i(c) - p_i_hat(c))^2

        Where:
            - 1_{ij}^obj = 1 if object in cell i, bbox j responsible
            - lambda_coord = 5 (weight for localization)
            - lambda_noobj = 0.5 (weight for no-object confidence)

    Why sqrt(w), sqrt(h)?
        Small deviations matter more for small boxes than large boxes.
        sqrt penalizes small box errors more heavily.
    """,
    predecessors=["overfeat_2013", "alexnet_2012"],
    successors=["yolo_v2_2017"],
    tags=["object_detection", "single_stage", "real_time", "grid_based"],
)


# =============================================================================
# YOLOv2/YOLO9000 (2017) - Redmon & Farhadi
# =============================================================================

YOLO_V2 = MLMethod(
    method_id="yolo_v2_2017",
    name="YOLO v2 (YOLO9000)",
    year=2017,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Joseph Redmon", "Ali Farhadi"],
    paper_title="YOLO9000: Better, Faster, Stronger",
    paper_url="https://arxiv.org/abs/1612.08242",
    key_innovation="""
    Better: Improved accuracy through architectural changes
    Faster: New backbone (Darknet-19)
    Stronger: Joint training on detection and classification data

    Key Improvements:
    1. Batch Normalization: Added to all conv layers
    2. High-resolution classifier: 448x448 pretraining
    3. Anchor boxes: Predefined box shapes from data clustering
    4. Dimension clusters: K-means on ground truth boxes
    5. Direct location prediction: Constrained predictions
    6. Passthrough layer: Fine-grained features for small objects
    7. Multi-scale training: Random input sizes during training

    YOLO9000: Jointly trained on ImageNet + COCO using WordTree
    """,
    mathematical_formulation="""
    Anchor Boxes via K-means Clustering:
        Cluster ground truth boxes by shape (w, h)
        Distance metric: d(box, centroid) = 1 - IoU(box, centroid)

        Result: k=5 anchors covering common aspect ratios
        Priors: (1.3, 1.7), (3.3, 4.0), (5.5, 8.6), (10.1, 10.5), (15.2, 16.1)

    Direct Location Prediction:
        Given anchor (p_w, p_h) at grid cell (c_x, c_y):

        b_x = sigma(t_x) + c_x
        b_y = sigma(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)

        Where:
            - (t_x, t_y, t_w, t_h): Network predictions
            - sigma(): Sigmoid to constrain center to cell
            - (b_x, b_y, b_w, b_h): Final bounding box

    Objectness:
        P(object) * IoU(b, object) = sigma(t_o)

    Passthrough Layer:
        Concatenate high-resolution (26x26) features with low-resolution (13x13)
        - Reshape 26x26x512 -> 13x13x2048
        - Concat with 13x13x1024 -> 13x13x3072

    Multi-scale Training:
        Every 10 batches, randomly choose input size from {320, 352, ..., 608}
        All sizes multiples of 32 (network stride)

    Output: 13 x 13 x (5 * (5 + C))
        5 anchors per cell, each with (t_x, t_y, t_w, t_h, t_o) + C classes
    """,
    predecessors=["yolo_v1_2016", "batch_norm_2015"],
    successors=["yolo_v3_2018"],
    tags=["object_detection", "single_stage", "anchor_boxes", "multi_scale"],
)


# =============================================================================
# YOLOv3 (2018) - Redmon & Farhadi
# =============================================================================

YOLO_V3 = MLMethod(
    method_id="yolo_v3_2018",
    name="YOLO v3",
    year=2018,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Joseph Redmon", "Ali Farhadi"],
    paper_title="YOLOv3: An Incremental Improvement",
    paper_url="https://arxiv.org/abs/1804.02767",
    key_innovation="""
    Multi-scale predictions using Feature Pyramid Network-like structure.

    Key Improvements:
    1. Darknet-53 backbone: Residual connections, 53 conv layers
    2. Multi-scale detection: Predictions at 3 scales
    3. Better backbone: Comparable to ResNet-152 but 2x faster
    4. Independent logistic classifiers: Multi-label classification

    Detection at 3 Scales:
        - 13x13: Large objects (stride 32)
        - 26x26: Medium objects (stride 16)
        - 52x52: Small objects (stride 8)
    """,
    mathematical_formulation="""
    Darknet-53 Architecture:
        Convolutional: 3x3, 1x1 alternating with residual connections
        No pooling layers: Downsampling via stride-2 convolutions

        Residual Block:
            y = x + Conv(Conv(x))

    Multi-scale Predictions:
        Scale 1 (13x13): Detect at final layer
        Scale 2 (26x26): Upsample + concat with earlier features
        Scale 3 (52x52): Upsample + concat with earlier features

        Each scale predicts 3 anchors using k-means on COCO:
        Scale 1: (116,90), (156,198), (373,326)  - large objects
        Scale 2: (30,61), (62,45), (59,119)      - medium objects
        Scale 3: (10,13), (16,30), (33,23)       - small objects

    Output per Scale:
        N x N x [3 * (4 + 1 + 80)]
        = N x N x 255 (for COCO with 80 classes)

        Per anchor: (t_x, t_y, t_w, t_h, objectness, class_1, ..., class_80)

    Classification: Independent Logistic Classifiers
        P(class_i) = sigma(output_i)  (not softmax)

        Allows multi-label classification (e.g., "woman" and "person")

    Loss: Sum of losses across all 3 scales
        L = L_scale1 + L_scale2 + L_scale3

        Each scale loss similar to YOLOv2:
        L = L_coord + L_obj + L_noobj + L_class
    """,
    predecessors=["yolo_v2_2017", "resnet_2015", "fpn_2017"],
    successors=["yolo_v4_2020"],
    tags=["object_detection", "single_stage", "feature_pyramid", "multi_scale"],
)


# =============================================================================
# YOLOv4 (2020) - Bochkovskiy et al.
# =============================================================================

YOLO_V4 = MLMethod(
    method_id="yolo_v4_2020",
    name="YOLO v4",
    year=2020,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Alexey Bochkovskiy", "Chien-Yao Wang", "Hong-Yuan Mark Liao"],
    paper_title="YOLOv4: Optimal Speed and Accuracy of Object Detection",
    paper_url="https://arxiv.org/abs/2004.10934",
    key_innovation="""
    Comprehensive study of 'bag of freebies' and 'bag of specials' for detection.

    Architecture:
    - Backbone: CSPDarknet53 (Cross Stage Partial connections)
    - Neck: SPP (Spatial Pyramid Pooling) + PANet (Path Aggregation Network)
    - Head: YOLOv3-style multi-scale predictions

    Bag of Freebies (training tricks, no inference cost):
    - Mosaic augmentation: 4 images combined
    - Self-adversarial training (SAT)
    - CutMix, MixUp data augmentation
    - DropBlock regularization
    - Label smoothing

    Bag of Specials (architecture changes):
    - Mish activation: x * tanh(softplus(x))
    - Cross-stage partial connections (CSP)
    - Spatial attention module (SAM)
    - DIoU-NMS
    """,
    mathematical_formulation="""
    CSPDarknet53:
        Cross Stage Partial (CSP) splits feature map into two parts:
        - Part 1: Passes through dense block
        - Part 2: Direct connection
        - Concatenate outputs

        Benefits: Reduced computation, better gradient flow

    Mish Activation:
        f(x) = x * tanh(softplus(x))
             = x * tanh(ln(1 + e^x))

        Properties: Smooth, non-monotonic, unbounded above, bounded below

    Spatial Pyramid Pooling (SPP):
        Parallel max pooling at multiple scales: {1x1, 5x5, 9x9, 13x13}
        Concatenate outputs for multi-scale features

    Path Aggregation Network (PANet):
        Bottom-up path augmentation after FPN:
        - FPN: Top-down for semantics
        - PANet: Bottom-up for localization

    CIoU Loss (Complete IoU):
        L_CIoU = 1 - IoU + rho^2(b, b_gt) / c^2 + alpha * v

        Where:
            - rho(b, b_gt): Euclidean distance between box centers
            - c: Diagonal of smallest enclosing box
            - v = (4/pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
            - alpha = v / ((1 - IoU) + v)

        Considers: Overlap, center distance, and aspect ratio

    DIoU-NMS:
        Score = s_i * (1 - R_DIoU(M, B_i))^beta  if R_DIoU >= threshold

        Uses distance-based IoU for better handling of occluded objects
    """,
    predecessors=["yolo_v3_2018", "cspnet_2020", "panet_2018"],
    successors=["yolo_v5_2020", "scaled_yolov4_2020"],
    tags=["object_detection", "single_stage", "cspnet", "panet", "mish"],
)


# =============================================================================
# YOLOv5 (2020) - Ultralytics
# =============================================================================

YOLO_V5 = MLMethod(
    method_id="yolo_v5_2020",
    name="YOLO v5",
    year=2020,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Glenn Jocher", "Ultralytics Team"],
    paper_title="YOLOv5",
    paper_url="https://github.com/ultralytics/yolov5",
    key_innovation="""
    PyTorch-native implementation with engineering optimizations.

    Key Features:
    1. PyTorch implementation: Easy to use, modify, deploy
    2. Auto-learning anchors: Compute optimal anchors for custom datasets
    3. Multiple model sizes: n, s, m, l, x (nano to extra-large)
    4. Efficient training: Mixed precision, optimized data loading
    5. Export support: ONNX, TensorRT, CoreML, TFLite

    Note: Not an official YOLO paper; controversial naming but widely adopted.
    """,
    mathematical_formulation="""
    Model Scaling (width, depth multipliers):
        YOLOv5n: width=0.25, depth=0.33
        YOLOv5s: width=0.50, depth=0.33
        YOLOv5m: width=0.75, depth=0.67
        YOLOv5l: width=1.00, depth=1.00
        YOLOv5x: width=1.25, depth=1.33

    Focus Layer (replaced in later versions):
        Slice input 640x640x3 into 320x320x12:
        - Extract alternate pixels to create 4 sub-images
        - Concatenate along channel dimension
        - Apply convolution

    Auto-learning Anchors:
        1. Compute k-means on dataset bounding boxes
        2. Optimize with genetic algorithm for best recall
        3. Use metric: 1 - IoU for clustering distance

    Loss Function:
        L = L_box + L_obj + L_cls

        Box Loss: CIoU loss
        Objectness Loss: BCE with positive/negative weighting
        Classification Loss: BCE (multi-label)

    Architecture Components:
        - C3 module: CSP Bottleneck with 3 convolutions
        - SPPF: Fast SPP (sequential instead of parallel)
        - Conv: Conv2d + BatchNorm + SiLU (Swish)

    SiLU Activation (Swish):
        f(x) = x * sigmoid(x)
    """,
    predecessors=["yolo_v4_2020"],
    successors=["yolo_v8_2023"],
    tags=["object_detection", "single_stage", "pytorch", "production"],
)


# =============================================================================
# YOLOv8 (2023) - Ultralytics
# =============================================================================

YOLO_V8 = MLMethod(
    method_id="yolo_v8_2023",
    name="YOLO v8",
    year=2023,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.CNN_LINE],
    authors=["Glenn Jocher", "Ultralytics Team"],
    paper_title="YOLOv8",
    paper_url="https://github.com/ultralytics/ultralytics",
    key_innovation="""
    Anchor-free detection with decoupled head architecture.

    Key Changes from YOLOv5:
    1. Anchor-free: No predefined anchor boxes
    2. Decoupled head: Separate branches for classification and regression
    3. New backbone: C2f module (CSP with 2 convolutions)
    4. Distribution Focal Loss: For bounding box regression
    5. Task Heads: Unified architecture for detect, segment, pose, classify

    Supported Tasks:
    - Detection: Object detection
    - Segmentation: Instance segmentation
    - Pose: Keypoint estimation
    - Classification: Image classification
    - OBB: Oriented bounding boxes
    """,
    mathematical_formulation="""
    Anchor-Free Prediction:
        Instead of predicting offsets from anchor boxes:
        - Directly predict: (x, y, w, h) for each grid cell
        - No anchor priors needed

        Prediction: (x, y) as distance from grid cell center
                   (w, h) as direct size predictions

    Decoupled Head:
        Split head into parallel branches:

        Input Features -> C2f -> Split
                              |-> Classification Branch -> cls logits
                              |-> Regression Branch -> bbox coords

        Benefits: Task-specific optimization, better gradients

    C2f Module (Cross-stage 2 convolutions):
        x -> Split -> [Conv, Conv, ..., Conv] -> Concat -> Conv -> output
             |__________________|
             (gradient flow)

        Improvement over C3: Better gradient flow with more bottlenecks

    Distribution Focal Loss (DFL):
        Regress bounding box as discrete distribution over bins:

        y = sum_{i=0}^{n-1} P(y_i) * y_i

        L_DFL = -((y_{i+1} - y) * log(P(y_i)) + (y - y_i) * log(P(y_{i+1})))

        Where y is between y_i and y_{i+1}

        Benefits: Better handling of uncertainty in bbox regression

    Loss Function:
        L = L_cls + L_box + L_dfl

        - Classification: BCE loss with focal loss weighting
        - Box: CIoU loss
        - DFL: Distribution focal loss for bbox regression

    Model Variants:
        YOLOv8n: 3.2M params, 225 FLOPs
        YOLOv8s: 11.2M params, 28.6 FLOPs
        YOLOv8m: 25.9M params, 78.9 FLOPs
        YOLOv8l: 43.7M params, 165.2 FLOPs
        YOLOv8x: 68.2M params, 257.8 FLOPs
    """,
    predecessors=["yolo_v5_2020", "fcos_2019", "ppyoloe_2022"],
    successors=[],
    tags=["object_detection", "anchor_free", "decoupled_head", "multi_task"],
)


# =============================================================================
# Supporting Functions and Concepts
# =============================================================================

def yolo_grid_prediction() -> str:
    """
    Core YOLO prediction mechanism: S x S grid with B boxes per cell.
    """
    return """
    YOLO Grid Prediction:

    Image divided into S x S grid (e.g., 7x7 for YOLOv1, 13x13 for later versions)

    For each grid cell:
        - Cell is "responsible" for object if object center falls in cell
        - Predicts B bounding boxes (B=2 for v1, B=3-5 for later versions)
        - Each box: (x, y, w, h, confidence)
        - Also predicts C class probabilities (shared across all boxes in cell)

    Bounding Box Prediction:
        (x, y): Offset from cell corner, normalized to [0, 1]
        (w, h): Normalized by image dimensions
        confidence: P(Object) * IoU(pred, truth)

    Output Tensor Shape:
        S x S x (B*5 + C)  for YOLOv1
        S x S x (B*(5+C))  for YOLOv2+ (per-box class predictions)

    At inference:
        class_prob * confidence = P(Class_i | Object) * P(Object) * IoU
        Apply threshold and NMS to get final detections
    """


def anchor_boxes() -> str:
    """
    Anchor boxes (priors) for YOLO v2+.
    """
    return """
    Anchor Boxes (Dimension Priors):

    Purpose: Provide shape priors for common object aspect ratios

    Generation via K-means:
        1. Collect all ground truth boxes (w, h) from training data
        2. Run k-means with distance = 1 - IoU(box, centroid)
        3. Select k anchors (typically k=5 or k=9)

    Prediction with Anchors:
        Given anchor (p_w, p_h) at grid cell (c_x, c_y):

        Network predicts: (t_x, t_y, t_w, t_h)

        Final box:
            b_x = sigmoid(t_x) + c_x
            b_y = sigmoid(t_y) + c_y
            b_w = p_w * exp(t_w)
            b_h = p_h * exp(t_h)

    Benefits:
        - Network learns offsets (easier) rather than absolute sizes
        - Different anchors specialize for different object shapes
        - Improves recall especially for extreme aspect ratios

    COCO Anchors (YOLOv3 example):
        Small:  (10,13), (16,30), (33,23)
        Medium: (30,61), (62,45), (59,119)
        Large:  (116,90), (156,198), (373,326)

    Anchor-Free Alternative (YOLOv8):
        Directly predict (x, y, w, h) without anchor priors
        Benefits: Simpler, no hyperparameter tuning for anchors
    """


def non_max_suppression() -> str:
    """
    Non-maximum suppression (NMS) for removing duplicate detections.
    """
    return """
    Non-Maximum Suppression (NMS):

    Problem: Object detector outputs many overlapping boxes for same object
    Solution: Keep only the best box, suppress highly overlapping ones

    Standard NMS Algorithm:
        1. Sort boxes by confidence score (descending)
        2. Select box with highest score
        3. Remove all boxes with IoU > threshold (e.g., 0.5) with selected box
        4. Repeat steps 2-3 with remaining boxes
        5. Output: List of non-overlapping boxes

    Pseudocode:
        def nms(boxes, scores, iou_threshold):
            keep = []
            order = argsort(scores, descending=True)

            while order:
                i = order[0]
                keep.append(i)

                ious = compute_iou(boxes[i], boxes[order[1:]])
                remaining = order[1:][ious <= iou_threshold]
                order = remaining

            return keep

    Variants:
        - Soft-NMS: Decay scores instead of hard removal
            s_i = s_i * exp(-IoU^2 / sigma)

        - DIoU-NMS: Consider center distance in addition to IoU

        - Batched NMS: Per-class NMS (prevent cross-class suppression)

    Soft-NMS Benefits:
        - Better for occluded objects
        - Smoother score decay
        - Improved recall at similar precision
    """
