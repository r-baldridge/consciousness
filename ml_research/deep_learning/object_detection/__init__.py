"""
Deep Learning Object Detection Module

This module contains research indices for deep learning object detection methods,
including two-stage detectors (R-CNN family) and single-stage detectors (YOLO, SSD).

Key Method Families:
    - R-CNN Family (2014-2017): Region-based CNNs with progressively faster designs
    - YOLO Family (2016-2023): Real-time single-shot detection with grid-based predictions
    - SSD (2016): Multi-scale feature map detection

The object detection task: Given an image, localize and classify all objects by
predicting bounding boxes (x, y, w, h) and class probabilities for each detection.
"""

from .rcnn_family import (
    # Method entries
    R_CNN,
    FAST_R_CNN,
    FASTER_R_CNN,
    MASK_R_CNN,
    # Key functions/concepts
    selective_search,
    roi_pooling,
    region_proposal_network,
    roi_align,
)

from .yolo import (
    # Method entries
    YOLO_V1,
    YOLO_V2,
    YOLO_V3,
    YOLO_V4,
    YOLO_V5,
    YOLO_V8,
    # Key functions/concepts
    yolo_grid_prediction,
    anchor_boxes,
    non_max_suppression,
)

from .ssd import (
    # Method entries
    SSD,
    # Key functions/concepts
    multi_scale_feature_maps,
    default_boxes,
    hard_negative_mining,
)

__all__ = [
    # R-CNN Family
    "R_CNN",
    "FAST_R_CNN",
    "FASTER_R_CNN",
    "MASK_R_CNN",
    "selective_search",
    "roi_pooling",
    "region_proposal_network",
    "roi_align",
    # YOLO Family
    "YOLO_V1",
    "YOLO_V2",
    "YOLO_V3",
    "YOLO_V4",
    "YOLO_V5",
    "YOLO_V8",
    "yolo_grid_prediction",
    "anchor_boxes",
    "non_max_suppression",
    # SSD
    "SSD",
    "multi_scale_feature_maps",
    "default_boxes",
    "hard_negative_mining",
]
