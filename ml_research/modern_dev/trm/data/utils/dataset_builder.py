"""
Dataset Builder for TRM Code Repair

High-level utility for building training datasets from Python source files.
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np

from ..collectors.synthetic import SyntheticBugGenerator
from ..processors.tokenizer import PythonTokenizer, build_vocabulary
from ..processors.encoder import GridEncoder, GridConfig, create_default_encoder
from ..processors.validator import DataValidator, ValidationConfig
from ..processors.augmenter import AugmentationPipeline, AugmentationConfig
from ..taxonomy.bug_types import BugType, BugCategory


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    # Paths
    source_dir: Optional[Path] = None
    output_dir: Path = Path("./dataset")
    vocab_path: Optional[Path] = None

    # Generation settings
    bugs_per_file: int = 3
    augmentations_per_pair: int = 2
    seed: int = 42

    # Quality settings
    min_file_lines: int = 5
    max_file_lines: int = 200

    # Grid settings
    grid_height: int = 64
    grid_width: int = 48
    vocab_size: int = 512


@dataclass
class DatasetSample:
    """A single training sample."""
    sample_id: str
    buggy_grid: np.ndarray
    fixed_grid: np.ndarray
    bug_type: str
    bug_category: str
    source_file: Optional[str] = None
    augmentations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON serialization)."""
        return {
            "sample_id": self.sample_id,
            "buggy_grid": self.buggy_grid.tolist(),
            "fixed_grid": self.fixed_grid.tolist(),
            "bug_type": self.bug_type,
            "bug_category": self.bug_category,
            "source_file": self.source_file,
            "augmentations": self.augmentations,
        }


class DatasetBuilder:
    """
    Builds training datasets for TRM code repair.

    Pipeline:
    1. Load Python source files
    2. Generate synthetic bugs
    3. Validate pairs
    4. Augment data
    5. Encode to grids
    6. Save dataset

    Example:
        builder = DatasetBuilder(config)
        builder.build_from_directory("./python_code")
        builder.save("./dataset")
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        Initialize the dataset builder.

        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()

        # Initialize components
        self.generator = SyntheticBugGenerator(seed=self.config.seed)
        self.validator = DataValidator()
        self.augmenter = AugmentationPipeline(seed=self.config.seed)
        self.encoder = create_default_encoder()

        # Storage
        self.samples: List[DatasetSample] = []
        self.statistics: Dict = {}

    def build_from_files(
        self,
        files: List[Path],
        show_progress: bool = True,
    ) -> int:
        """
        Build dataset from a list of Python files.

        Args:
            files: List of Python file paths
            show_progress: Show progress information

        Returns:
            Number of samples generated
        """
        total_generated = 0

        for i, file_path in enumerate(files):
            if show_progress and i % 10 == 0:
                print(f"Processing file {i+1}/{len(files)}: {file_path.name}")

            try:
                code = file_path.read_text(encoding='utf-8')
            except Exception:
                continue

            # Skip files outside length bounds
            lines = code.count('\n') + 1
            if lines < self.config.min_file_lines or lines > self.config.max_file_lines:
                continue

            # Generate bugs
            samples = self._process_code(code, source_file=str(file_path))
            self.samples.extend(samples)
            total_generated += len(samples)

        self._compute_statistics()
        return total_generated

    def build_from_directory(
        self,
        directory: Path,
        pattern: str = "**/*.py",
        show_progress: bool = True,
    ) -> int:
        """
        Build dataset from a directory of Python files.

        Args:
            directory: Directory containing Python files
            pattern: Glob pattern for finding files
            show_progress: Show progress information

        Returns:
            Number of samples generated
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))

        if show_progress:
            print(f"Found {len(files)} Python files")

        return self.build_from_files(files, show_progress)

    def build_from_code(
        self,
        code_samples: List[str],
        show_progress: bool = True,
    ) -> int:
        """
        Build dataset from a list of code strings.

        Args:
            code_samples: List of Python code strings
            show_progress: Show progress information

        Returns:
            Number of samples generated
        """
        total_generated = 0

        for i, code in enumerate(code_samples):
            if show_progress and i % 100 == 0:
                print(f"Processing sample {i+1}/{len(code_samples)}")

            samples = self._process_code(code)
            self.samples.extend(samples)
            total_generated += len(samples)

        self._compute_statistics()
        return total_generated

    def _process_code(
        self,
        code: str,
        source_file: Optional[str] = None,
    ) -> List[DatasetSample]:
        """Process a single code sample."""
        samples = []

        # Generate multiple bugs per code
        for _ in range(self.config.bugs_per_file):
            result = self.generator.generate(code)
            if result is None:
                continue

            buggy, fixed, bug_type = result

            # Validate
            report = self.validator.validate(buggy, fixed, bug_type)
            if not report.is_valid:
                continue

            # Create original sample
            sample = self._create_sample(buggy, fixed, bug_type, source_file)
            if sample is not None:
                samples.append(sample)

            # Augment
            for _ in range(self.config.augmentations_per_pair - 1):
                augmented = self.augmenter.augment(buggy, fixed, bug_type)
                aug_sample = self._create_sample(
                    augmented.buggy_code,
                    augmented.fixed_code,
                    bug_type,
                    source_file,
                    augmentations=[a.value for a in augmented.augmentations_applied],
                )
                if aug_sample is not None:
                    samples.append(aug_sample)

        return samples

    def _create_sample(
        self,
        buggy: str,
        fixed: str,
        bug_type: BugType,
        source_file: Optional[str] = None,
        augmentations: Optional[List[str]] = None,
    ) -> Optional[DatasetSample]:
        """Create a dataset sample."""
        try:
            buggy_grid, fixed_grid = self.encoder.encode_pair(buggy, fixed)

            # Generate unique ID
            content_hash = hashlib.md5(
                (buggy + fixed).encode()
            ).hexdigest()[:12]
            sample_id = f"{bug_type.value}_{content_hash}"

            # Get category
            category = self._get_category(bug_type)

            return DatasetSample(
                sample_id=sample_id,
                buggy_grid=buggy_grid,
                fixed_grid=fixed_grid,
                bug_type=bug_type.value,
                bug_category=category.value,
                source_file=source_file,
                augmentations=augmentations or [],
            )
        except Exception:
            return None

    def _get_category(self, bug_type: BugType) -> BugCategory:
        """Get category for bug type."""
        syntax_bugs = {
            BugType.MISSING_COLON, BugType.MISSING_PARENTHESIS,
            BugType.MISSING_BRACKET, BugType.INDENTATION_ERROR,
            BugType.MISSING_COMMA,
        }
        if bug_type in syntax_bugs:
            return BugCategory.SYNTAX
        return BugCategory.LOGIC

    def _compute_statistics(self):
        """Compute dataset statistics."""
        self.statistics = {
            "total_samples": len(self.samples),
            "bug_type_counts": {},
            "category_counts": {},
            "augmented_count": 0,
        }

        for sample in self.samples:
            # Bug type counts
            bt = sample.bug_type
            self.statistics["bug_type_counts"][bt] = \
                self.statistics["bug_type_counts"].get(bt, 0) + 1

            # Category counts
            cat = sample.bug_category
            self.statistics["category_counts"][cat] = \
                self.statistics["category_counts"].get(cat, 0) + 1

            # Augmented count
            if sample.augmentations:
                self.statistics["augmented_count"] += 1

    def save(
        self,
        output_dir: Optional[Path] = None,
        format: str = "npz",
    ):
        """
        Save the dataset.

        Args:
            output_dir: Output directory
            format: Output format ('npz', 'json', or 'both')
        """
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format in ('npz', 'both'):
            self._save_npz(output_dir)

        if format in ('json', 'both'):
            self._save_json(output_dir)

        # Save metadata
        self._save_metadata(output_dir)

        print(f"Dataset saved to {output_dir}")
        print(f"Total samples: {len(self.samples)}")

    def _save_npz(self, output_dir: Path):
        """Save as numpy arrays."""
        buggy_grids = np.stack([s.buggy_grid for s in self.samples])
        fixed_grids = np.stack([s.fixed_grid for s in self.samples])
        bug_types = np.array([s.bug_type for s in self.samples])

        np.savez_compressed(
            output_dir / "dataset.npz",
            buggy_grids=buggy_grids,
            fixed_grids=fixed_grids,
            bug_types=bug_types,
        )

    def _save_json(self, output_dir: Path):
        """Save as JSON lines."""
        with open(output_dir / "dataset.jsonl", 'w') as f:
            for sample in self.samples:
                # Convert grids to lists for JSON
                record = {
                    "sample_id": sample.sample_id,
                    "bug_type": sample.bug_type,
                    "bug_category": sample.bug_category,
                    "source_file": sample.source_file,
                    "augmentations": sample.augmentations,
                }
                f.write(json.dumps(record) + '\n')

    def _save_metadata(self, output_dir: Path):
        """Save dataset metadata."""
        metadata = {
            "config": {
                "bugs_per_file": self.config.bugs_per_file,
                "augmentations_per_pair": self.config.augmentations_per_pair,
                "grid_height": self.config.grid_height,
                "grid_width": self.config.grid_width,
                "vocab_size": self.config.vocab_size,
            },
            "statistics": self.statistics,
        }

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return self.statistics

    def iterate_samples(self) -> Iterator[DatasetSample]:
        """Iterate over samples."""
        yield from self.samples

    def get_sample(self, idx: int) -> DatasetSample:
        """Get sample by index."""
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetSample:
        return self.samples[idx]
