"""
Base data loader for architecture-specific loaders.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
import json
import numpy as np

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    pq = None
    PYARROW_AVAILABLE = False


# Curriculum stage definitions (shared across architectures)
CURRICULUM_STAGES = {
    1: {
        'max_difficulty': 0.3,
        'bug_categories': ['syntax'],
        'description': 'Simple syntax errors',
        'trm_iterations': 4,
        'ctm_iterations': 16,
    },
    2: {
        'max_difficulty': 0.5,
        'bug_categories': ['syntax', 'logic'],
        'description': 'Syntax + simple logic',
        'trm_iterations': 6,
        'ctm_iterations': 32,
    },
    3: {
        'max_difficulty': 0.7,
        'bug_categories': ['syntax', 'logic', 'style'],
        'description': 'Most bug types',
        'trm_iterations': 8,
        'ctm_iterations': 48,
    },
    4: {
        'max_difficulty': 1.0,
        'bug_categories': ['syntax', 'logic', 'style', 'security', 'framework'],
        'description': 'Full dataset',
        'trm_iterations': 8,
        'ctm_iterations': 64,
    },
}


class BaseDataLoader(ABC):
    """Abstract base class for architecture-specific data loaders."""

    def __init__(
        self,
        data_path: Path,
        tokenizer_path: Path,
        config: Dict[str, Any],
        split: str = 'train',
        curriculum_stage: Optional[int] = None,
        quality_tiers: Optional[List[str]] = None,
    ):
        """
        Initialize the data loader.

        Args:
            data_path: Path to canonical data directory
            tokenizer_path: Path to tokenizer directory
            config: Architecture-specific configuration
            split: Data split ('train', 'val', 'test')
            curriculum_stage: Optional curriculum stage (1-4)
            quality_tiers: Optional list of quality tiers to include
        """
        self.data_path = Path(data_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.config = config
        self.split = split
        self.curriculum_stage = curriculum_stage
        self.quality_tiers = quality_tiers

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Load or generate indices
        self._indices: Optional[List[str]] = None

    @property
    def indices(self) -> List[str]:
        """Lazy load indices."""
        if self._indices is None:
            self._indices = self._load_indices()
        return self._indices

    def _load_tokenizer(self):
        """Load shared BPE tokenizer."""
        try:
            from tokenizers import Tokenizer
            tokenizer_file = self.tokenizer_path / 'tokenizer.json'
            if tokenizer_file.exists():
                return Tokenizer.from_file(str(tokenizer_file))
        except ImportError:
            pass

        # Fallback: return None, will use pre-tokenized data
        return None

    def _load_indices(self) -> List[str]:
        """Load sample IDs for this split, applying filters."""
        indices = self._load_split_indices()

        # Apply curriculum filtering
        if self.curriculum_stage is not None:
            indices = self._filter_by_curriculum(indices)

        # Apply quality tier filtering
        if self.quality_tiers is not None:
            indices = self._filter_by_quality(indices)

        return indices

    def _load_split_indices(self) -> List[str]:
        """Load sample IDs for this split."""
        splits_file = self.data_path / 'metadata' / 'splits.json'

        if splits_file.exists():
            with open(splits_file) as f:
                splits = json.load(f)
            return splits.get(self.split, [])

        # Fallback: scan parquet files for sample_ids
        return self._scan_sample_ids()

    def _scan_sample_ids(self) -> List[str]:
        """Scan parquet files for sample IDs."""
        if not PYARROW_AVAILABLE:
            return []

        canonical_path = self.data_path / 'canonical'
        if not canonical_path.exists():
            return []

        sample_ids = []
        for parquet_file in canonical_path.rglob('*.parquet'):
            table = pq.read_table(parquet_file, columns=['sample_id'])
            sample_ids.extend(table['sample_id'].to_pylist())

        return sample_ids

    def _filter_by_curriculum(self, indices: List[str]) -> List[str]:
        """Filter indices based on curriculum stage."""
        if self.curriculum_stage not in CURRICULUM_STAGES:
            return indices

        stage_config = CURRICULUM_STAGES[self.curriculum_stage]
        max_difficulty = stage_config['max_difficulty']
        allowed_categories = stage_config['bug_categories']

        # Build filter conditions
        filters = [
            ('difficulty', '<=', max_difficulty),
            ('bug_category', 'in', allowed_categories),
        ]

        return self._query_indices_with_filters(indices, filters)

    def _filter_by_quality(self, indices: List[str]) -> List[str]:
        """Filter indices by quality tier."""
        filters = [('quality_tier', 'in', self.quality_tiers)]
        return self._query_indices_with_filters(indices, filters)

    def _query_indices_with_filters(
        self,
        indices: List[str],
        filters: List[tuple]
    ) -> List[str]:
        """Query parquet with filters and return matching indices."""
        if not PYARROW_AVAILABLE:
            return indices

        canonical_path = self.data_path / 'canonical'
        if not canonical_path.exists():
            return indices

        # Build pyarrow filter expressions
        filter_exprs = []
        for field, op, value in filters:
            if op == '<=':
                filter_exprs.append((field, '<=', value))
            elif op == 'in':
                filter_exprs.append((field, 'in', value))

        # Read with filters
        try:
            table = pq.read_table(
                canonical_path,
                columns=['sample_id', 'difficulty', 'bug_category', 'quality_tier'],
                filters=filter_exprs if filter_exprs else None,
            )

            filtered_ids = set(table['sample_id'].to_pylist())
            return [idx for idx in indices if idx in filtered_ids]
        except Exception:
            # Fallback: return all indices
            return indices

    def _load_canonical(self, sample_id: str) -> Dict:
        """Load a canonical sample by ID."""
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow is required for loading canonical samples")

        canonical_path = self.data_path / 'canonical'

        # Search in parquet files
        for parquet_file in canonical_path.rglob('*.parquet'):
            table = pq.read_table(
                parquet_file,
                filters=[('sample_id', '=', sample_id)]
            )
            if len(table) > 0:
                return table.to_pydict()

        raise ValueError(f"Sample {sample_id} not found")

    @abstractmethod
    def _transform_sample(self, canonical: Dict) -> Any:
        """Transform canonical format to architecture-specific format."""
        pass

    def __iter__(self) -> Iterator[Any]:
        """Iterate over samples in architecture-specific format."""
        for sample_id in self.indices:
            try:
                canonical = self._load_canonical(sample_id)
                yield self._transform_sample(canonical)
            except Exception as e:
                # Log and skip failed samples
                print(f"Warning: Failed to load sample {sample_id}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.indices)

    def get_curriculum_config(self) -> Dict[str, Any]:
        """Get curriculum configuration for current stage."""
        if self.curriculum_stage is None:
            return {}
        return CURRICULUM_STAGES.get(self.curriculum_stage, {})
