# Joint Embedding Predictive Architecture (JEPA)

> A family of self-supervised learning architectures that predict representations in latent space rather than pixel/token space, forming the foundation for building world models and general visual understanding.

**Status:** Scaffolding Complete - Implementation Pending
**Paper:** [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
**Year:** 2023-2025
**Organization:** Meta AI

## Overview

JEPA represents Yann LeCun's vision for building AI systems that learn world models through prediction in representation space. Rather than trying to predict every pixel (an intractable task given the world's inherent unpredictability), JEPA learns to predict abstract representations of content, focusing on what's possible versus impossible to build common sense reasoning.

The architecture uses an energy-based model foundation where compatible observations receive low energy and incompatible ones receive high energy. Unlike contrastive methods that explicitly push apart negative examples, JEPA prevents representation collapse through careful regularization of the representation space, making it more computationally efficient (2-10x fewer GPU-hours than alternatives like MAE or DINO).

The JEPA family has evolved through several variants: I-JEPA (2023) for images, V-JEPA (2024) for video, V-JEPA 2 (2024/2025) for advanced world modeling with 1.2B parameters, and VL-JEPA (2025) for vision-language understanding. Each variant maintains the core principle of non-generative, semantic-focused learning with minimal data augmentation requirements.

## Key Innovations

- **Prediction in Latent Space**: Predicts abstract representations rather than raw pixels, enabling the model to focus on semantic content and high-level features.

- **Non-Contrastive Learning**: Avoids the need for negative samples by regularizing the representation space, reducing computational requirements.

- **Multi-Block Masking Strategy**: Uses semantically coherent block masking (not random patches) to encourage learning of spatial relationships and object-level representations.

## Architecture Diagram

```
                    +-------------------+
    Image/Video --> | Context Encoder   | --> z_context
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |    Predictor      | --> z_predicted
                    +-------------------+
                              |
                              v
                    Compare with...
                              |
                    +-------------------+
    Masked Region -->| Target Encoder   | --> z_target (stop-gradient)
                    +-------------------+

Loss = ||z_predicted - z_target||^2  (in latent space)

Key Properties:
    - Non-generative: No pixel reconstruction
    - Single view: No heavy data augmentation
    - Semantic focus: Learns high-level features
```

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| VisionTransformer Encoder | Stub | ViT backbone defined |
| Predictor Network | Stub | MLP architecture ready |
| Multi-Block Masking | Stub | Strategy defined |
| EMA Update | Stub | Target encoder update |
| Training Loop | Stub | CLI ready |
| Tests | Ready | Import tests |
| Configs | Ready | I-JEPA and V-JEPA configs |

## JEPA Family Variants

| Variant | Year | Modality | Paper | GitHub |
|---------|------|----------|-------|--------|
| I-JEPA | 2023 | Image | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) |
| V-JEPA | 2024 | Video | [arXiv:2402.03192](https://arxiv.org/abs/2402.03192) | [facebookresearch/jepa](https://github.com/facebookresearch/jepa) |
| V-JEPA 2 | 2025 | Video | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) |
| VL-JEPA | 2025 | Vision-Language | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942) | - |

## Requirements for Full Implementation

### Dependencies
- PyTorch >= 2.0
- torchvision
- timm (for ViT backbones)
- einops

### Hardware
- 16 A100 GPUs for full ImageNet training (72 hours)
- Single GPU sufficient for fine-tuning and inference
- Memory scales with encoder size (ViT-L/ViT-H)

### External Resources
- [ ] Pretrained weights from [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)
- [ ] ImageNet-1K for I-JEPA training
- [ ] Video datasets (Kinetics, SSv2) for V-JEPA

## Quick Start

```python
from consciousness.ml_research.modern_dev.jepa import I_JEPA_CONFIG, V_JEPA_CONFIG

# I-JEPA Configuration
config = {
    "encoder": "vit_large_patch16",
    "predictor_depth": 12,
    "predictor_embed_dim": 384,
    "num_targets": 4,
    "target_aspect_ratio": (0.75, 1.5),
    "target_scale": (0.15, 0.2),
    "ema_momentum": 0.996,
    "learning_rate": 1e-4,
    "weight_decay": 0.04,
}

# When implemented:
# from consciousness.ml_research.modern_dev.jepa.src.encoder import JEPAEncoder
# from consciousness.ml_research.modern_dev.jepa.src.predictor import JEPAPredictor
# encoder = JEPAEncoder(config)
# predictor = JEPAPredictor(config)
```

## File Structure

```
jepa/
├── __init__.py       # Module documentation and variant metadata
├── README.md         # This file
├── src/
│   ├── encoder.py    # VisionTransformer encoder (context/target)
│   ├── predictor.py  # Predictor network
│   └── masking.py    # Multi-block masking strategy
├── configs/
│   ├── ijepa.yaml    # I-JEPA configuration
│   └── vjepa.yaml    # V-JEPA configuration
├── cli/
│   ├── train.py      # Training script (jepa-train)
│   ├── embed.py      # Extract embeddings (jepa-embed)
│   └── visualize.py  # Masking visualization
├── models/           # Pretrained checkpoints
├── docs/             # Additional documentation
└── tests/
    └── test_model.py # Unit tests
```

## Mathematical Formulation

**JEPA Objective:**
```
Loss = E[||predictor(z_context, mask) - z_target||^2]

where:
    z_context = context_encoder(visible_patches)
    z_target = target_encoder(masked_patches)  # stop-gradient
    predictor learns spatial/temporal relationships
```

**Target Encoder Update (EMA):**
```
theta_target = tau * theta_target + (1 - tau) * theta_context

typical tau = 0.996 to 0.9999
```

**Multi-Block Masking:**
```
Sample K target blocks (typically K=4)
Each block: aspect ratio in [0.75, 1.5], scale in [0.15, 0.2]
Context = all patches not in any target block
```

## Benchmarks

| Model | Task | Performance |
|-------|------|-------------|
| I-JEPA | ImageNet low-shot (12 examples/class) | SOTA |
| V-JEPA 2 | Something-Something v2 Top-1 | 77.3% |
| V-JEPA 2 | Epic-Kitchens-100 Recall@5 | 39.7 |
| V-JEPA 2 | Zero-shot robot planning | 65-80% success |

## Comparison with Other Methods

|               | JEPA | CLIP | MAE |
|---------------|------|------|-----|
| Approach | Predictive | Contrastive | Generative |
| Prediction Space | Latent | Latent | Pixel |
| Augmentation | Minimal | Heavy | None |
| Modality | Single | Cross-modal | Single |
| Negatives | No | Yes | No |
| Focus | Semantic | Alignment | Reconstruction |

## References

- Assran, M., et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023)
- Bardes, A., et al. "V-JEPA: Video Joint Embedding Predictive Architecture" (2024)
- Bardes, A., et al. "V-JEPA 2: Self-Supervised Video World Models" (2024)
- [Official I-JEPA Implementation](https://github.com/facebookresearch/ijepa)

## Contributing

To complete this implementation:

1. **Phase 1: I-JEPA Implementation**
   - Implement ViT encoder with proper initialization
   - Create multi-block masking with semantic block sampling
   - Build predictor network with cross-attention
   - Add training loop with EMA target updates

2. **Phase 2: V-JEPA Extension**
   - Extend to spatiotemporal patch embedding
   - Implement video data loading with tubelet extraction
   - Add temporal masking strategies

3. **Phase 3: VL-JEPA Integration**
   - Add text encoder integration
   - Implement selective decoding mechanism
   - Create VQA evaluation pipeline
