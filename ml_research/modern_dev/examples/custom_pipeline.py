"""
Custom Pipeline Example

Shows how to customize the pipeline with different models and settings.

Features demonstrated:
- Custom TRM configuration
- Custom Mamba configuration
- Direct model usage (bypassing orchestrator)
- Custom architecture registration
- Technique configuration
- Performance tuning

Usage:
    python -m modern_dev.examples.custom_pipeline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

# TRM imports
from modern_dev.trm.src.model import (
    CodeRepairTRM,
    CodeRepairConfig,
    TRM,
    TRMConfig,
)

# Mamba imports
from modern_dev.mamba_impl.src.mamba_model import (
    MambaLM,
    MambaLMConfig,
    MambaCache,
)

# Orchestrator imports
from modern_dev.orchestrator.router import (
    MLOrchestrator,
    Request,
    Response,
    ArchitectureRegistry,
    ArchitectureCapability,
    UnifiedIndex,
    TaskRouter,
    RoutingStrategy,
)


def custom_trm_configuration() -> None:
    """Example 1: Custom TRM configuration."""
    print("=" * 60)
    print("Example 1: Custom TRM Configuration")
    print("=" * 60)

    # Create a custom configuration for code repair
    config = CodeRepairConfig(
        # Grid configuration (code representation)
        grid_height=64,           # 64 lines of code
        grid_width=48,            # 48 tokens per line
        vocab_size=32768,         # BPE vocabulary size

        # Model architecture
        embed_dim=256,            # Embedding dimension
        n_heads=8,                # Attention heads
        ffn_dim=1024,             # Feed-forward dimension
        n_blocks=6,               # Number of recursive blocks

        # Recursion parameters
        max_iterations=8,         # Maximum recursive iterations
        min_iterations=2,         # Minimum before early stopping

        # Training/inference
        dropout=0.0,              # No dropout for inference
        q_threshold=0.95,         # Confidence threshold for early stopping

        # Memory optimization
        use_gradient_checkpointing=False,  # Disable for faster inference
    )

    # Create model
    model = CodeRepairTRM(config)

    print(f"Configuration:")
    print(f"  Grid size: {config.grid_height}x{config.grid_width}")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Effective depth: {config.effective_depth} layers")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Parameters: {model.num_parameters():,}")

    # Run inference
    model.eval()
    with torch.no_grad():
        # Create dummy input
        x = torch.randint(0, config.vocab_size, (1, 64, 48))

        start = time.time()
        result = model.generate(x)
        elapsed = (time.time() - start) * 1000

        print(f"\nInference:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Confidence: {result['confidence'].item():.2%}")
        print(f"  Time: {elapsed:.2f}ms")

    print()


def custom_mamba_configuration() -> None:
    """Example 2: Custom Mamba configuration."""
    print("=" * 60)
    print("Example 2: Custom Mamba Configuration")
    print("=" * 60)

    # Create a custom configuration for language modeling
    config = MambaLMConfig(
        vocab_size=32000,         # Vocabulary size
        d_model=512,              # Model dimension
        n_layers=12,              # Number of Mamba blocks
        d_state=16,               # SSM state dimension
        d_conv=4,                 # Convolution kernel width
        expand=2,                 # Expansion factor (d_inner = expand * d_model)

        # Discretization parameters
        dt_rank="auto",           # Auto-compute delta rank
        dt_min=0.001,             # Minimum delta
        dt_max=0.1,               # Maximum delta
        dt_init="random",         # Delta initialization

        # Training options
        tie_embeddings=True,      # Tie input/output embeddings
        bias=False,               # No bias in linear layers
        dropout=0.0,              # No dropout
        residual_in_fp32=True,    # Keep residuals in fp32

        # Vocabulary padding
        pad_vocab_size_multiple=8,  # Pad to multiple of 8
    )

    # Create model
    model = MambaLM(config)

    print(f"Configuration:")
    print(f"  Vocab size: {config.vocab_size:,}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Inner dim: {config.d_inner}")
    print(f"  Layers: {config.n_layers}")
    print(f"  State dim: {config.d_state}")
    print(f"  Parameters: {model.num_parameters():,}")

    # Get memory footprint
    mem = model.get_memory_footprint()
    print(f"\nMemory footprint:")
    print(f"  Embedding: {mem['embedding'] / 1024 / 1024:.2f}MB")
    print(f"  Layers: {mem['layers'] / 1024 / 1024:.2f}MB")
    print(f"  Total: {mem['total'] / 1024 / 1024:.2f}MB")

    # Run inference with caching
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, config.vocab_size, (1, 10))

        start = time.time()
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            use_cache=True,  # Use O(1) caching
        )
        elapsed = (time.time() - start) * 1000

        print(f"\nGeneration:")
        print(f"  Prompt length: 10")
        print(f"  Generated: {generated.shape[1] - 10} tokens")
        print(f"  Time: {elapsed:.2f}ms")
        print(f"  Speed: {(generated.shape[1] - 10) / (elapsed / 1000):.1f} tokens/sec")

    print()


def custom_orchestrator_setup() -> None:
    """Example 3: Custom orchestrator with custom architectures."""
    print("=" * 60)
    print("Example 3: Custom Orchestrator Setup")
    print("=" * 60)

    # Create custom registry
    registry = ArchitectureRegistry()

    # Register custom TRM variant
    registry.register(
        "trm_fast",
        ArchitectureCapability(
            name="TRM-Fast",
            supported_tasks=[
                "code_repair",
                "iterative_refinement",
                "puzzle_solving",
            ],
            max_context_length=3072,
            inference_speed="fast",
            memory_requirement="low",
            strengths=[
                "fast iteration",
                "early stopping",
                "code repair",
            ],
            supports_early_stopping=True,
        ),
        lambda: create_fast_trm(),
    )

    # Register custom Mamba variant
    registry.register(
        "mamba_xl",
        ArchitectureCapability(
            name="Mamba-XL",
            supported_tasks=[
                "code_generation",
                "long_context",
                "text_generation",
                "streaming",
            ],
            max_context_length=100000,
            inference_speed="fast",
            memory_requirement="medium",
            strengths=[
                "very long context",
                "efficient generation",
                "streaming output",
            ],
            supports_streaming=True,
        ),
        lambda: create_xl_mamba(),
    )

    # Create custom router
    router = TaskRouter(
        registry,
        default_strategy=RoutingStrategy.QUALITY_OPTIMIZED,
    )

    # Create custom index
    index = UnifiedIndex()

    # Register custom technique
    index.register_technique(
        "fast_repair",
        architectures=["trm_fast"],
        configs={
            "trm_fast": {
                "max_iterations": 4,  # Fewer iterations
                "q_threshold": 0.9,   # Lower threshold
            },
        },
        metadata={"description": "Fast code repair with early stopping"},
    )

    index.register_technique(
        "deep_analysis",
        architectures=["mamba_xl"],
        configs={
            "mamba_xl": {
                "max_context_tokens": 50000,
                "temperature": 0.3,
            },
        },
        metadata={"description": "Deep code analysis with long context"},
    )

    print(f"Registered architectures: {registry.list_architectures()}")
    print(f"Registered techniques: {index.list_techniques()}")

    # Test routing
    from modern_dev.orchestrator.router import Task

    task = Task.create("code_repair", {"buggy_code": "def foo(): retrun 1"})
    decision = router.route(task)

    print(f"\nRouting decision for 'code_repair':")
    print(f"  Primary: {decision.primary}")
    print(f"  Fallback: {decision.fallback}")
    print(f"  Confidence: {decision.confidence:.2%}")

    # Get technique config
    config = index.get_config("fast_repair", "trm_fast")
    print(f"\nFast repair config: {config}")

    print()


def create_fast_trm() -> CodeRepairTRM:
    """Factory for fast TRM variant."""
    config = CodeRepairConfig.for_code_repair_small()
    config.max_iterations = 4
    config.q_threshold = 0.9
    return CodeRepairTRM(config)


def create_xl_mamba() -> MambaLM:
    """Factory for XL Mamba variant."""
    config = MambaLMConfig(
        d_model=1024,
        n_layers=32,
    )
    return MambaLM(config)


def direct_model_usage() -> None:
    """Example 4: Direct model usage without orchestrator."""
    print("=" * 60)
    print("Example 4: Direct Model Usage")
    print("=" * 60)

    # Use TRM presets
    print("TRM Presets:")
    for preset_name, preset_fn in [
        ("Sudoku", TRMConfig.for_sudoku),
        ("Maze (30x30)", lambda: TRMConfig.for_maze(30)),
        ("ARC-AGI", TRMConfig.for_arc_agi),
    ]:
        config = preset_fn()
        model = TRM(config)
        print(f"  {preset_name}:")
        print(f"    Grid: {config.grid_size}x{config.grid_size}")
        print(f"    Effective depth: {config.effective_depth}")
        print(f"    Parameters: {model.num_parameters():,}")

    print("\nCodeRepair Presets:")
    for preset_name, preset_fn in [
        ("Tiny", CodeRepairConfig.for_code_repair_tiny),
        ("Small", CodeRepairConfig.for_code_repair_small),
        ("Base", CodeRepairConfig.for_code_repair_base),
        ("Large", CodeRepairConfig.for_code_repair_large),
    ]:
        config = preset_fn()
        model = CodeRepairTRM(config)
        print(f"  {preset_name}:")
        print(f"    Embed dim: {config.embed_dim}")
        print(f"    Parameters: {model.num_parameters():,}")

    print()


def performance_tuning() -> None:
    """Example 5: Performance tuning tips."""
    print("=" * 60)
    print("Example 5: Performance Tuning")
    print("=" * 60)

    # TRM tuning
    print("TRM Performance Tips:")
    print("  1. Reduce max_iterations for faster inference")
    print("  2. Lower q_threshold for earlier stopping")
    print("  3. Use smaller embed_dim (128 vs 256)")
    print("  4. Use gradient checkpointing for training")

    config = CodeRepairConfig.for_code_repair_small()
    base_model = CodeRepairTRM(config)
    base_params = base_model.num_parameters()

    # Fast config
    config.max_iterations = 4
    config.q_threshold = 0.85
    fast_model = CodeRepairTRM(config)

    print(f"\n  Base config: {base_params:,} params, 8 iterations")
    print(f"  Fast config: {fast_model.num_parameters():,} params, 4 iterations")
    print(f"  Expected speedup: ~2x")

    # Mamba tuning
    print("\nMamba Performance Tips:")
    print("  1. Always use cache for generation (use_cache=True)")
    print("  2. Use fp16/bf16 for 2x memory reduction")
    print("  3. Increase batch size for throughput")
    print("  4. Use greedy decoding (do_sample=False) for fastest inference")

    config = MambaLMConfig(d_model=512, n_layers=12)
    model = MambaLM(config)

    # Benchmark with/without cache
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        # Without cache (simulated by regenerating each time)
        start = time.time()
        for _ in range(5):
            _ = model(prompt)
        no_cache_time = (time.time() - start) / 5 * 1000

        # With cache
        start = time.time()
        for _ in range(5):
            _ = model.generate(prompt, max_new_tokens=10, use_cache=True)
        with_cache_time = (time.time() - start) / 5 * 1000

    print(f"\n  Forward pass (no generation): {no_cache_time:.2f}ms")
    print(f"  Generation (10 tokens, cached): {with_cache_time:.2f}ms")

    print()


def hybrid_pipeline() -> None:
    """Example 6: Hybrid pipeline combining models."""
    print("=" * 60)
    print("Example 6: Hybrid Pipeline")
    print("=" * 60)

    print("Hybrid Architecture Concept:")
    print("""
    Long Context Input
           |
           v
    [Mamba Encoder]  <- O(N) encoding of full context
           |
           v
    [Task Decomposer] <- Break into subtasks
           |
           v
    [TRM Refiner]    <- Iterative refinement per subtask
           |
           v
    [Mamba Decoder]  <- Generate coherent output
           |
           v
    Final Output
    """)

    # Create components
    mamba_config = MambaLMConfig(d_model=256, n_layers=6)
    trm_config = CodeRepairConfig.for_code_repair_tiny()

    mamba_encoder = MambaLM(mamba_config)
    trm_refiner = CodeRepairTRM(trm_config)

    print(f"Components:")
    print(f"  Mamba encoder: {mamba_encoder.num_parameters():,} params")
    print(f"  TRM refiner: {trm_refiner.num_parameters():,} params")
    print(f"  Total: {mamba_encoder.num_parameters() + trm_refiner.num_parameters():,} params")

    print("\nBenefits of hybrid approach:")
    print("  - Mamba handles long context efficiently (O(N))")
    print("  - TRM provides iterative refinement capability")
    print("  - Each model used for its strength")
    print("  - Better than either model alone")

    print()


def custom_training_config() -> None:
    """Example 7: Custom training configuration."""
    print("=" * 60)
    print("Example 7: Training Configuration")
    print("=" * 60)

    from modern_dev.mamba_impl.src.mamba_model import MambaTrainerConfig

    # Custom training config
    trainer_config = MambaTrainerConfig(
        learning_rate=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_grad_norm=1.0,
        warmup_steps=1000,
        max_steps=100000,
        gradient_accumulation_steps=4,
        log_interval=10,
        eval_interval=500,
        save_interval=1000,
    )

    print("Training Configuration:")
    print(f"  Learning rate: {trainer_config.learning_rate}")
    print(f"  Weight decay: {trainer_config.weight_decay}")
    print(f"  Gradient accumulation: {trainer_config.gradient_accumulation_steps}")
    print(f"  Max steps: {trainer_config.max_steps}")
    print(f"  Warmup steps: {trainer_config.warmup_steps}")

    print("\nTraining Tips:")
    print("  1. Use gradient accumulation for larger effective batch")
    print("  2. Warmup helps with training stability")
    print("  3. Save checkpoints frequently")
    print("  4. Monitor with W&B or TensorBoard")

    print()


def main():
    """Run all custom pipeline examples."""
    print("\n" + "=" * 60)
    print("ML Research - Custom Pipeline Examples")
    print("=" * 60 + "\n")

    custom_trm_configuration()
    custom_mamba_configuration()
    custom_orchestrator_setup()
    direct_model_usage()
    performance_tuning()
    hybrid_pipeline()
    custom_training_config()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
