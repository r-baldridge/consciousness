"""
TRM Data Collection & Processing Module

Tools for building Python code repair datasets for TRM training.

Usage:
    from trm.data import (
        PythonTokenizer,
        SyntheticBugGenerator,
        GitHubCollector,
        DataValidator,
    )

    # Tokenize code
    tokenizer = PythonTokenizer()
    grid = tokenizer.encode(code)

    # Generate synthetic bugs
    generator = SyntheticBugGenerator()
    buggy, fixed, bug_type = generator.generate(clean_code)

    # Validate pairs
    validator = DataValidator()
    is_valid, reason = validator.validate(buggy, fixed, bug_type)
"""

from .processors.tokenizer import PythonTokenizer, build_vocabulary
from .processors.encoder import GridEncoder
from .processors.validator import DataValidator
from .processors.augmenter import AugmentationPipeline

from .collectors.synthetic import SyntheticBugGenerator

from .taxonomy.bug_types import BugType, BugCategory, BUG_TAXONOMY, get_bug_info

__all__ = [
    # Processors
    "PythonTokenizer",
    "GridEncoder",
    "DataValidator",
    "AugmentationPipeline",
    "build_vocabulary",
    # Collectors
    "SyntheticBugGenerator",
    # Taxonomy
    "BugType",
    "BugCategory",
    "BUG_TAXONOMY",
    "get_bug_info",
]
