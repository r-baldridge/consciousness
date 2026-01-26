"""
Mamba Implementation - State Space Models

Core SSM components for the Mamba architecture, including:
- S4DKernel: Diagonal State Space Model kernel
- S4DLayer: Full S4D layer with normalization
- SelectiveSSM: Input-dependent selective state space
- SelectiveScan: Comprehensive selective mechanism (MAMBA-003)
- HiPPO: Principled initialization for long-range dependencies
- MambaBlock: Complete Mamba block architecture
- MambaLayer: Mamba layer with normalization and residual
- MambaStack: Stack of Mamba layers for building models
- MambaLM: Full Mamba Language Model (MAMBA-006)
- MambaCache: Inference cache for O(1) per-token generation
- MambaTrainer: Training wrapper with utilities
"""

from .model import MambaConfig, Mamba, MambaBlock
from .layers import SelectiveSSM, S6Layer, CausalConv1d, Discretization, RMSNorm
from .ssm import S4DKernel, S4DLayer
from .parameterization import HiPPO, SSMInit
from .block import MambaBlock as MambaBlockV2, MambaLayer, MambaStack
from .selective import SelectiveScan, SelectiveScanConfig, selective_scan_ref
from .scan import (
    AssociativeScan,
    ChunkedScan,
    parallel_scan,
    sequential_scan,
    selective_ssm_scan,
    select_scan_impl,
    ScanConfig,
    benchmark_scan,
)
from .mamba_model import (
    MambaLMConfig,
    MambaLM,
    MambaCache,
    MambaTrainer,
    MambaTrainerConfig,
    load_pretrained,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Model (original)
    "MambaConfig",
    "Mamba",
    "MambaBlock",
    # Full Language Model (MAMBA-006)
    "MambaLMConfig",
    "MambaLM",
    "MambaCache",
    "MambaTrainer",
    "MambaTrainerConfig",
    "load_pretrained",
    "save_checkpoint",
    "load_checkpoint",
    # Block (standalone module)
    "MambaBlockV2",
    "MambaLayer",
    "MambaStack",
    # Layers
    "SelectiveSSM",
    "S6Layer",
    "CausalConv1d",
    "Discretization",
    "RMSNorm",
    # Selective Mechanism (MAMBA-003)
    "SelectiveScan",
    "SelectiveScanConfig",
    "selective_scan_ref",
    # S4D SSM
    "S4DKernel",
    "S4DLayer",
    # Parameterization
    "HiPPO",
    "SSMInit",
    # Parallel Scan (MAMBA-005)
    "AssociativeScan",
    "ChunkedScan",
    "parallel_scan",
    "sequential_scan",
    "selective_ssm_scan",
    "select_scan_impl",
    "ScanConfig",
    "benchmark_scan",
]
