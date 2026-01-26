"""
ML Research Pipeline Examples

This package contains runnable examples demonstrating the ML Research
Code Repair Pipeline.

Examples:
    basic_repair.py      - Simple code repair examples
    batch_processing.py  - Batch file processing with parallelization
    custom_pipeline.py   - Advanced customization and configuration

Usage:
    # Run individual examples
    python -m modern_dev.examples.basic_repair
    python -m modern_dev.examples.batch_processing
    python -m modern_dev.examples.custom_pipeline

    # Or import and use
    from modern_dev.examples import basic_repair
    basic_repair.main()
"""

from . import basic_repair
from . import batch_processing
from . import custom_pipeline

__all__ = [
    "basic_repair",
    "batch_processing",
    "custom_pipeline",
]
