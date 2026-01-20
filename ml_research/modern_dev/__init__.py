"""
Modern ML Architectures Development Module

This module indexes state-of-the-art and emerging ML architectures (2024-2025+)
with subdirectories prepared for developing each into working CLI models.

Unlike the research index modules, these are intended for ACTIVE DEVELOPMENT
into usable implementations with command-line interfaces.

=============================================================================
INDEXED ARCHITECTURES
=============================================================================

Tier 1 - Production Ready (open source, well-documented):
---------------------------------------------------------
- CTM (Continuous Thought Machine) - Sakana AI, NeurIPS 2025
- JEPA Family (I-JEPA, V-JEPA, V-JEPA 2, VL-JEPA) - Meta AI
- xLSTM (Extended LSTM) - NXAI Lab
- RWKV (RNN-Transformer Hybrid) - BlinkDL
- Griffin (Gated Linear Recurrence) - Google DeepMind
- Mamba/SSM implementations - Gu & Dao

Tier 2 - Research Ready (papers available, some code):
------------------------------------------------------
- TTT (Test-Time Training) - Stanford/NVIDIA
- Hyena (Long Convolutions) - Hazy Research
- Consistency Models / Flow Matching - OpenAI/DeepMind
- Ring Attention (Infinite Context) - Berkeley
- Titans (Meta In-Context Memory) - Google

Tier 3 - Emerging (promising but early):
----------------------------------------
- KAN (Kolmogorov-Arnold Networks)
- Liquid Neural Networks
- Graph Neural Networks (modern variants)

=============================================================================
DIRECTORY STRUCTURE (per architecture)
=============================================================================

Each architecture has:
    src/        - Core implementation code
    models/     - Pre-trained model configs and weights references
    configs/    - YAML/JSON configuration files
    tests/      - Unit and integration tests
    docs/       - Architecture documentation and usage guides
    cli/        - Command-line interface tools

=============================================================================
USAGE
=============================================================================

# Import specific architecture
from consciousness.ml_research.modern_dev import ctm
from consciousness.ml_research.modern_dev import jepa

# Get architecture info
from consciousness.ml_research.modern_dev import ARCHITECTURES, get_architecture_info
info = get_architecture_info('ctm')

# List all architectures
for arch in ARCHITECTURES:
    print(arch['name'], arch['status'])
"""

__version__ = "0.1.0"
__status__ = "development"

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class DevelopmentStatus(Enum):
    """Development status of an architecture implementation."""
    INDEXED = "indexed"              # Research indexed, not yet implemented
    IN_PROGRESS = "in_progress"      # Active development
    ALPHA = "alpha"                  # Basic functionality working
    BETA = "beta"                    # Feature complete, testing
    PRODUCTION = "production"        # Ready for use


class ImplementationTier(Enum):
    """Implementation readiness tier."""
    TIER_1 = "production_ready"      # Open source, well-documented
    TIER_2 = "research_ready"        # Papers available, some code
    TIER_3 = "emerging"              # Promising but early


@dataclass
class ArchitectureIndex:
    """Index entry for a modern ML architecture."""
    id: str
    name: str
    year: int
    authors: List[str]
    organization: str
    paper_title: str
    paper_url: str
    github_url: Optional[str]
    tier: ImplementationTier
    status: DevelopmentStatus
    key_innovation: str
    use_cases: List[str]
    dependencies: List[str] = field(default_factory=list)
    related_architectures: List[str] = field(default_factory=list)
    notes: Optional[str] = None


# =============================================================================
# ARCHITECTURE REGISTRY
# =============================================================================

ARCHITECTURES: List[ArchitectureIndex] = [
    # Tier 1 - Production Ready
    ArchitectureIndex(
        id="ctm",
        name="Continuous Thought Machine",
        year=2025,
        authors=["Llion Jones", "David Ha", "et al."],
        organization="Sakana AI",
        paper_title="Continuous Thought Machines",
        paper_url="https://arxiv.org/abs/2505.05522",
        github_url="https://github.com/SakanaAI/continuous-thought-machines",
        tier=ImplementationTier.TIER_1,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Neural dynamics as core representation with neuron-level temporal "
            "processing and neural synchronization. Decoupled internal time "
            "dimension enables adaptive computation that emerges naturally."
        ),
        use_cases=[
            "Reasoning tasks requiring variable computation",
            "Maze solving with emergent algorithms",
            "Image classification with interpretable attention",
            "Tasks requiring step-by-step processing"
        ],
        related_architectures=["transformer", "neural_ode"],
    ),

    ArchitectureIndex(
        id="jepa",
        name="Joint Embedding Predictive Architecture",
        year=2023,
        authors=["Yann LeCun", "Mahmoud Assran", "et al."],
        organization="Meta AI",
        paper_title="Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
        paper_url="https://arxiv.org/abs/2301.08243",
        github_url="https://github.com/facebookresearch/ijepa",
        tier=ImplementationTier.TIER_1,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Predicts representations in latent space rather than pixel space. "
            "Non-generative approach that learns semantic features without "
            "intensive data augmentation. Foundation for I-JEPA, V-JEPA, VL-JEPA."
        ),
        use_cases=[
            "Self-supervised visual representation learning",
            "Video understanding and prediction",
            "Vision-language alignment",
            "World modeling for robotics"
        ],
        related_architectures=["vit", "clip", "mae"],
    ),

    ArchitectureIndex(
        id="xlstm",
        name="Extended LSTM",
        year=2024,
        authors=["Maximilian Beck", "Sepp Hochreiter", "et al."],
        organization="NXAI Lab / JKU Linz",
        paper_title="xLSTM: Extended Long Short-Term Memory",
        paper_url="https://arxiv.org/abs/2405.04517",
        github_url="https://github.com/NX-AI/xlstm",
        tier=ImplementationTier.TIER_1,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Modernized LSTM with exponential gating, matrix memory (mLSTM), "
            "and parallelizable training. Achieves transformer-level performance "
            "with linear complexity and better interpretability."
        ),
        use_cases=[
            "Long sequence modeling",
            "Time series forecasting",
            "Speech processing",
            "Biomedical signal analysis"
        ],
        related_architectures=["lstm", "mamba", "rwkv"],
    ),

    ArchitectureIndex(
        id="rwkv",
        name="RWKV (Receptance Weighted Key Value)",
        year=2023,
        authors=["Bo Peng (BlinkDL)"],
        organization="RWKV Foundation",
        paper_title="RWKV: Reinventing RNNs for the Transformer Era",
        paper_url="https://arxiv.org/abs/2305.13048",
        github_url="https://github.com/BlinkDL/RWKV-LM",
        tier=ImplementationTier.TIER_1,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Combines RNN efficiency with transformer parallelizable training. "
            "O(n) complexity, constant memory, infinite context length. "
            "RWKV-7 'Goose' introduces dynamic state evolution."
        ),
        use_cases=[
            "Long-context generation",
            "Efficient edge deployment",
            "Real-time streaming",
            "Unlimited context scenarios"
        ],
        related_architectures=["transformer", "lstm", "mamba"],
    ),

    ArchitectureIndex(
        id="griffin",
        name="Griffin",
        year=2024,
        authors=["Soham De", "et al."],
        organization="Google DeepMind",
        paper_title="Griffin: Mixing Gated Linear Recurrences with Local Attention",
        paper_url="https://arxiv.org/abs/2402.19427",
        github_url="https://github.com/google-deepmind/recurrentgemma",
        tier=ImplementationTier.TIER_1,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Replaces global attention with gated linear recurrences + local "
            "sliding window attention. O(n) complexity with global context "
            "awareness. Scales to 14B parameters."
        ),
        use_cases=[
            "Long document processing",
            "Efficient long-context inference",
            "Memory-constrained deployment",
            "Streaming applications"
        ],
        related_architectures=["transformer", "mamba", "rwkv"],
    ),

    ArchitectureIndex(
        id="mamba_impl",
        name="Mamba (Implementation)",
        year=2023,
        authors=["Albert Gu", "Tri Dao"],
        organization="CMU / Princeton",
        paper_title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        paper_url="https://arxiv.org/abs/2312.00752",
        github_url="https://github.com/state-spaces/mamba",
        tier=ImplementationTier.TIER_1,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Selective state space model with input-dependent parameters. "
            "Hardware-aware parallel scan algorithm. O(N) training, O(1) "
            "inference per token with content awareness."
        ),
        use_cases=[
            "Long sequence modeling",
            "Language modeling",
            "DNA/genomics",
            "Audio processing"
        ],
        related_architectures=["s4", "transformer", "rwkv"],
    ),

    # Tier 2 - Research Ready
    ArchitectureIndex(
        id="ttt",
        name="Test-Time Training",
        year=2024,
        authors=["Yu Sun", "et al."],
        organization="Stanford / NVIDIA",
        paper_title="Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
        paper_url="https://arxiv.org/abs/2407.04620",
        github_url=None,
        tier=ImplementationTier.TIER_2,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Models with learnable hidden states that adapt during inference. "
            "Hidden state is a small trainable model (linear or MLP) that "
            "learns from context in real-time. 2.7x faster than attention on 128K."
        ),
        use_cases=[
            "Long-context reasoning",
            "Adaptive in-context learning",
            "Mathematical reasoning",
            "Memory-efficient sequence modeling"
        ],
        related_architectures=["transformer", "mamba", "titans"],
    ),

    ArchitectureIndex(
        id="hyena",
        name="Hyena",
        year=2023,
        authors=["Michael Poli", "et al."],
        organization="Hazy Research / Stanford",
        paper_title="Hyena Hierarchy: Towards Larger Convolutional Language Models",
        paper_url="https://arxiv.org/abs/2302.10866",
        github_url="https://github.com/HazyResearch/safari",
        tier=ImplementationTier.TIER_2,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Subquadratic attention replacement using implicit long convolutions "
            "and multiplicative gating. 100x speedup at 64K sequences. "
            "SE(3)-Hyena adds equivariant geometric support."
        ),
        use_cases=[
            "Very long sequences (64K+)",
            "Scientific computing with symmetry",
            "Efficient dense attention replacement",
            "Genomics and biomedical"
        ],
        related_architectures=["transformer", "s4", "mamba"],
    ),

    ArchitectureIndex(
        id="consistency_models",
        name="Consistency Models",
        year=2023,
        authors=["Yang Song", "et al."],
        organization="OpenAI",
        paper_title="Consistency Models",
        paper_url="https://arxiv.org/abs/2303.01469",
        github_url=None,
        tier=ImplementationTier.TIER_2,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Fast generative models through learned consistency mappings. "
            "Reduces sampling from 1000+ steps to single digits. "
            "One-step generation possible with quality comparable to diffusion."
        ),
        use_cases=[
            "Fast image generation",
            "Real-time video generation",
            "Robotic policy learning",
            "Interactive generative applications"
        ],
        related_architectures=["diffusion", "flow_matching", "vae"],
    ),

    ArchitectureIndex(
        id="ring_attention",
        name="Ring Attention",
        year=2023,
        authors=["Hao Liu", "Pieter Abbeel", "et al."],
        organization="UC Berkeley",
        paper_title="Ring Attention with Blockwise Transformers for Near-Infinite Context",
        paper_url="https://arxiv.org/abs/2310.01889",
        github_url=None,
        tier=ImplementationTier.TIER_2,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Distributes long sequences across GPUs with overlapped communication. "
            "Enables near-infinite context by scaling linearly with devices. "
            "Zero approximation, full quadratic attention properties preserved."
        ),
        use_cases=[
            "Book/document processing (100K+ tokens)",
            "Long-form reasoning",
            "Video frame sequences",
            "Multi-document analysis"
        ],
        related_architectures=["transformer", "flash_attention"],
    ),

    ArchitectureIndex(
        id="flow_matching",
        name="Flow Matching",
        year=2024,
        authors=["Yaron Lipman", "et al."],
        organization="Meta AI / DeepMind",
        paper_title="Flow Matching for Generative Modeling",
        paper_url="https://arxiv.org/abs/2210.02747",
        github_url=None,
        tier=ImplementationTier.TIER_2,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Learns optimal transport paths between distributions. "
            "Straighter trajectories than diffusion enable faster sampling. "
            "Foundation for Stable Diffusion 3, Flux, and modern image generators."
        ),
        use_cases=[
            "High-quality image generation",
            "Video synthesis",
            "3D generation",
            "Audio synthesis"
        ],
        related_architectures=["diffusion", "consistency_models", "vae"],
    ),

    ArchitectureIndex(
        id="titans",
        name="Titans",
        year=2025,
        authors=["Google Research"],
        organization="Google",
        paper_title="Titans: Learning to Memorize at Test Time",
        paper_url="https://arxiv.org/abs/2501.00663",
        github_url=None,
        tier=ImplementationTier.TIER_2,
        status=DevelopmentStatus.INDEXED,
        key_innovation=(
            "Meta in-context memory learning at test-time. Learns HOW to "
            "memorize as data arrives, combining attention with learnable "
            "memory mechanisms. Faster and handles massive contexts."
        ),
        use_cases=[
            "Continual learning",
            "Massive context processing",
            "Adaptive memory systems",
            "Long-term dependencies"
        ],
        related_architectures=["transformer", "ttt", "mamba"],
    ),
]

# Create lookup dictionary
ARCHITECTURE_BY_ID: Dict[str, ArchitectureIndex] = {
    arch.id: arch for arch in ARCHITECTURES
}


def get_architecture_info(arch_id: str) -> ArchitectureIndex:
    """Get detailed information about an architecture."""
    if arch_id not in ARCHITECTURE_BY_ID:
        raise KeyError(f"Unknown architecture: {arch_id}")
    return ARCHITECTURE_BY_ID[arch_id]


def list_architectures(tier: Optional[ImplementationTier] = None) -> List[ArchitectureIndex]:
    """List all architectures, optionally filtered by tier."""
    if tier is None:
        return ARCHITECTURES
    return [a for a in ARCHITECTURES if a.tier == tier]


def get_by_status(status: DevelopmentStatus) -> List[ArchitectureIndex]:
    """Get architectures by development status."""
    return [a for a in ARCHITECTURES if a.status == status]


# Import submodules when available
# from . import ctm
# from . import jepa
# from . import xlstm
# from . import rwkv
# from . import griffin
# from . import mamba_impl
# from . import ttt
# from . import hyena
# from . import consistency_models
# from . import ring_attention
# from . import flow_matching
# from . import titans

# Orchestrator for dynamic architecture selection and execution
from .orchestrator import (
    Orchestrator,
    TaskType,
    TaskSpec,
    TaskResult,
    run_task,
    ARCHITECTURE_CAPABILITIES,
)
from .orchestrator.base import ArchitectureBase, StubArchitecture

__all__ = [
    # Enums
    "DevelopmentStatus",
    "ImplementationTier",
    "TaskType",
    # Data classes
    "ArchitectureIndex",
    "TaskSpec",
    "TaskResult",
    # Base class for implementations
    "ArchitectureBase",
    "StubArchitecture",
    # Orchestrator
    "Orchestrator",
    "ARCHITECTURE_CAPABILITIES",
    # Registry
    "ARCHITECTURES",
    "ARCHITECTURE_BY_ID",
    # Functions
    "get_architecture_info",
    "list_architectures",
    "get_by_status",
    "run_task",
]
