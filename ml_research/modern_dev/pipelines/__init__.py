"""
Pipelines Module - End-to-End ML Pipelines (INTEG-002)

This module provides end-to-end pipelines combining multiple ML architectures
for code understanding, repair, and synthesis tasks.

Main pipelines:
- CodeRepairPipeline: High-level interface for code repair
- MambaTRMRLMPipeline: Full Mamba-TRM-RLM pipeline

The pipelines integrate:
- Mamba: Long-context understanding with O(N) complexity
- TRM: Iterative refinement with recursive reasoning
- RLM: Task decomposition and code synthesis

Example:
    from modern_dev.pipelines import CodeRepairPipeline

    # Quick usage
    pipeline = CodeRepairPipeline()
    result = pipeline.repair(
        buggy_code="def foo(): retrun 1",
        error_message="SyntaxError: invalid syntax"
    )
    print(result.repaired_code)

    # With configuration
    pipeline = CodeRepairPipeline.from_yaml("config/code_repair.yaml")
    result = pipeline.repair_file("src/buggy.py", line_range=(10, 20))
"""

# Handle both package and standalone imports
try:
    from .code_repair import (
        # Configuration
        CodeRepairConfig,
        # Result types
        RepairResult,
        FileRepairResult,
        RepairRequest,
        # Main pipeline
        CodeRepairPipeline,
        # Convenience functions
        repair_code,
        repair_file,
    )

    from .mamba_trm_rlm import (
        # Configuration
        PipelineConfig,
        MambaConfig,
        TRMConfig,
        RLMConfig,
        # Request/Response types
        PipelineRequest,
        PipelineResponse,
        TaskDecomposition,
        # Main pipeline
        MambaTRMRLMPipeline,
        # Component loaders
        load_mamba,
        load_trm,
        load_rlm,
    )
except ImportError:
    from code_repair import (
        # Configuration
        CodeRepairConfig,
        # Result types
        RepairResult,
        FileRepairResult,
        RepairRequest,
        # Main pipeline
        CodeRepairPipeline,
        # Convenience functions
        repair_code,
        repair_file,
    )

    from mamba_trm_rlm import (
        # Configuration
        PipelineConfig,
        MambaConfig,
        TRMConfig,
        RLMConfig,
        # Request/Response types
        PipelineRequest,
        PipelineResponse,
        TaskDecomposition,
        # Main pipeline
        MambaTRMRLMPipeline,
        # Component loaders
        load_mamba,
        load_trm,
        load_rlm,
    )


__all__ = [
    # Code Repair Pipeline
    "CodeRepairConfig",
    "RepairResult",
    "FileRepairResult",
    "RepairRequest",
    "CodeRepairPipeline",
    "repair_code",
    "repair_file",
    # Mamba-TRM-RLM Pipeline
    "PipelineConfig",
    "MambaConfig",
    "TRMConfig",
    "RLMConfig",
    "PipelineRequest",
    "PipelineResponse",
    "TaskDecomposition",
    "MambaTRMRLMPipeline",
    "load_mamba",
    "load_trm",
    "load_rlm",
]

__version__ = "0.1.0"
