"""
TRM (Tiny Recursive Model) specific data loader.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base import BaseDataLoader


# TRM default configuration
TRM_CONFIG = {
    'grid_height': 64,          # Max lines
    'grid_width': 48,           # Max tokens per line
    'vocab_size': 32768,        # BPE vocabulary
    'pad_token': 0,
    'unk_token': 1,
    'newline_token': 2,
}


@dataclass
class TRMSample:
    """TRM-specific training sample."""

    # Core grids
    buggy_grid: np.ndarray      # [64, 48] int32 token IDs
    fixed_grid: np.ndarray      # [64, 48] int32 token IDs

    # Masks
    buggy_mask: np.ndarray      # [64, 48] bool - valid positions
    diff_mask: np.ndarray       # [64, 48] bool - changed positions

    # Metadata (for analysis/curriculum)
    bug_type: str
    bug_category: str
    difficulty: float
    sample_id: str


class GridEncoder:
    """Encode tokenized code to 2D grid format."""

    def __init__(
        self,
        height: int = 64,
        width: int = 48,
        pad_token: int = 0,
        newline_token: int = 2,
        tokenizer=None,
    ):
        self.height = height
        self.width = width
        self.pad_token = pad_token
        self.newline_token = newline_token
        self.tokenizer = tokenizer

    def tokens_to_grid(self, tokens: List[int]) -> np.ndarray:
        """
        Convert flat token sequence to 2D grid.

        Splits on newline tokens and pads to grid dimensions.
        """
        grid = np.full((self.height, self.width), self.pad_token, dtype=np.int32)

        row = 0
        col = 0

        for token in tokens:
            if token == self.newline_token:
                row += 1
                col = 0
                if row >= self.height:
                    break
            else:
                if col < self.width:
                    grid[row, col] = token
                    col += 1

        return grid

    def code_to_grid(self, code: str) -> np.ndarray:
        """
        Tokenize code and convert to grid.

        Requires tokenizer to be set.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for code_to_grid")

        encoding = self.tokenizer.encode(code)
        return self.tokens_to_grid(encoding.ids)

    def grid_to_tokens(self, grid: np.ndarray) -> List[int]:
        """Convert grid back to flat token sequence."""
        tokens = []

        for row in range(grid.shape[0]):
            row_tokens = []
            for col in range(grid.shape[1]):
                token = grid[row, col]
                if token != self.pad_token:
                    row_tokens.append(int(token))

            if row_tokens:
                tokens.extend(row_tokens)
                tokens.append(self.newline_token)
            elif tokens:
                # Empty row after content - add newline
                tokens.append(self.newline_token)

        # Remove trailing newlines
        while tokens and tokens[-1] == self.newline_token:
            tokens.pop()

        return tokens


class TRMDataLoader(BaseDataLoader):
    """Data loader for TRM architecture."""

    def __init__(
        self,
        data_path: Path,
        tokenizer_path: Path,
        config: Optional[Dict[str, Any]] = None,
        split: str = 'train',
        curriculum_stage: Optional[int] = None,
        quality_tiers: Optional[List[str]] = None,
    ):
        # Merge with defaults
        full_config = {**TRM_CONFIG, **(config or {})}

        super().__init__(
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            config=full_config,
            split=split,
            curriculum_stage=curriculum_stage,
            quality_tiers=quality_tiers,
        )

        self.grid_encoder = GridEncoder(
            height=self.config['grid_height'],
            width=self.config['grid_width'],
            pad_token=self.config['pad_token'],
            newline_token=self.config.get('newline_token', 2),
            tokenizer=self.tokenizer,
        )

    def _transform_sample(self, canonical: Dict) -> TRMSample:
        """Transform canonical to TRM format."""

        # Get tokens (may be stored as list or need extraction)
        buggy_tokens = self._extract_tokens(canonical, 'buggy_tokens')
        fixed_tokens = self._extract_tokens(canonical, 'fixed_tokens')

        # Encode to grids
        buggy_grid = self.grid_encoder.tokens_to_grid(buggy_tokens)
        fixed_grid = self.grid_encoder.tokens_to_grid(fixed_tokens)

        # Create masks
        buggy_mask = (buggy_grid != self.config['pad_token'])
        diff_mask = self._compute_diff_mask(buggy_grid, fixed_grid)

        return TRMSample(
            buggy_grid=buggy_grid,
            fixed_grid=fixed_grid,
            buggy_mask=buggy_mask.astype(np.float32),
            diff_mask=diff_mask.astype(np.float32),
            bug_type=self._extract_scalar(canonical, 'bug_type', ''),
            bug_category=self._extract_scalar(canonical, 'bug_category', ''),
            difficulty=self._extract_scalar(canonical, 'difficulty', 0.5),
            sample_id=self._extract_scalar(canonical, 'sample_id', ''),
        )

    def _extract_tokens(self, canonical: Dict, field: str) -> List[int]:
        """Extract tokens from canonical dict, handling various formats."""
        value = canonical.get(field)

        if value is None:
            # Fall back to tokenizing code
            code_field = field.replace('_tokens', '_code')
            code = canonical.get(code_field, '')
            if self.tokenizer and code:
                return self.tokenizer.encode(code).ids
            return []

        if isinstance(value, list):
            # Direct list
            if len(value) > 0 and isinstance(value[0], list):
                # Nested list (batch format) - take first
                return value[0]
            return value

        if isinstance(value, np.ndarray):
            return value.tolist()

        return []

    def _extract_scalar(self, canonical: Dict, field: str, default: Any) -> Any:
        """Extract scalar value from canonical dict."""
        value = canonical.get(field, default)

        if isinstance(value, (list, np.ndarray)):
            if len(value) > 0:
                return value[0]
            return default

        return value

    def _compute_diff_mask(self, buggy: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Compute mask of positions that differ."""
        return (buggy != fixed).astype(np.float32)


if TORCH_AVAILABLE:
    class TRMDataset(Dataset):
        """PyTorch Dataset wrapper for TRM."""

        def __init__(
            self,
            loader: TRMDataLoader,
            preload: bool = True,
        ):
            """
            Initialize TRM dataset.

            Args:
                loader: TRMDataLoader instance
                preload: If True, preload all samples into memory
            """
            self.loader = loader

            if preload:
                self.samples = list(loader)
            else:
                self.samples = None
                self.indices = loader.indices

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if self.samples is not None:
                sample = self.samples[idx]
            else:
                sample_id = self.indices[idx]
                canonical = self.loader._load_canonical(sample_id)
                sample = self.loader._transform_sample(canonical)

            return {
                'buggy_grid': torch.from_numpy(sample.buggy_grid).long(),
                'fixed_grid': torch.from_numpy(sample.fixed_grid).long(),
                'buggy_mask': torch.from_numpy(sample.buggy_mask).float(),
                'diff_mask': torch.from_numpy(sample.diff_mask).float(),
                'difficulty': torch.tensor(sample.difficulty, dtype=torch.float32),
            }

        def __len__(self) -> int:
            if self.samples is not None:
                return len(self.samples)
            return len(self.indices)

        def get_collate_fn(self):
            """Return collate function for DataLoader."""
            def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
                return {
                    key: torch.stack([b[key] for b in batch])
                    for key in batch[0].keys()
                }
            return collate
else:
    # Stub for when torch is not available
    class TRMDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TRMDataset")
