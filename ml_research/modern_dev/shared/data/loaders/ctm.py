"""
CTM (Continuous Thought Machine) specific data loader.
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
from .trm import GridEncoder


# CTM default configuration
CTM_CONFIG = {
    'grid_height': 64,          # Max lines
    'grid_width': 48,           # Max tokens per line
    'vocab_size': 32768,        # BPE vocabulary
    'pad_token': 0,
    'unk_token': 1,
    'newline_token': 2,
    'position_normalize': True,  # Normalize positions to [0, 1]
}


@dataclass
class CTMSample:
    """CTM-specific training sample."""

    # Core grids (same as TRM)
    buggy_grid: np.ndarray      # [64, 48] int32 token IDs
    fixed_grid: np.ndarray      # [64, 48] int32 token IDs

    # Masks
    buggy_mask: np.ndarray      # [64, 48] float32 - valid positions
    diff_mask: np.ndarray       # [64, 48] float32 - changed positions

    # 2D positions for Fourier embeddings (CTM-specific)
    positions: np.ndarray       # [64, 48, 2] float32 - (row, col) normalized

    # Bug location for intermediate supervision (CTM-specific)
    bug_location: Tuple[int, int]  # (row, col) of bug start in grid
    bug_location_mask: np.ndarray  # [64, 48] float32 - bug region mask

    # Metadata
    bug_type: str
    bug_category: str
    difficulty: float
    sample_id: str


class CTMDataLoader(BaseDataLoader):
    """Data loader for CTM architecture."""

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
        full_config = {**CTM_CONFIG, **(config or {})}

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

        # Pre-compute position grid (shared across all samples)
        self.position_grid = self._create_position_grid()

    def _create_position_grid(self) -> np.ndarray:
        """Create normalized 2D position grid for Fourier embeddings."""
        h = self.config['grid_height']
        w = self.config['grid_width']

        if self.config.get('position_normalize', True):
            rows = np.arange(h, dtype=np.float32) / h  # Normalized to [0, 1)
            cols = np.arange(w, dtype=np.float32) / w
        else:
            rows = np.arange(h, dtype=np.float32)
            cols = np.arange(w, dtype=np.float32)

        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
        return np.stack([row_grid, col_grid], axis=-1)

    def _transform_sample(self, canonical: Dict) -> CTMSample:
        """Transform canonical to CTM format."""

        # Get tokens
        buggy_tokens = self._extract_tokens(canonical, 'buggy_tokens')
        fixed_tokens = self._extract_tokens(canonical, 'fixed_tokens')

        # Encode to grids (same as TRM)
        buggy_grid = self.grid_encoder.tokens_to_grid(buggy_tokens)
        fixed_grid = self.grid_encoder.tokens_to_grid(fixed_tokens)

        # Create masks
        buggy_mask = (buggy_grid != self.config['pad_token']).astype(np.float32)
        diff_mask = self._compute_diff_mask(buggy_grid, fixed_grid)

        # Bug location (CTM-specific)
        bug_location = self._compute_bug_location(canonical, buggy_grid, diff_mask)
        bug_location_mask = self._compute_bug_location_mask(
            canonical, buggy_grid.shape, diff_mask
        )

        return CTMSample(
            buggy_grid=buggy_grid,
            fixed_grid=fixed_grid,
            buggy_mask=buggy_mask,
            diff_mask=diff_mask,
            positions=self.position_grid.copy(),
            bug_location=bug_location,
            bug_location_mask=bug_location_mask,
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
            if len(value) > 0 and isinstance(value[0], list):
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

    def _compute_bug_location(
        self,
        canonical: Dict,
        buggy_grid: np.ndarray,
        diff_mask: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Compute bug location as grid coordinates.

        Priority:
        1. Use explicit bug_start_line/col if available
        2. Use changed_tokens to find first change
        3. Use diff_mask to find first difference
        4. Default to (0, 0)
        """
        h, w = self.config['grid_height'], self.config['grid_width']

        # Try explicit location
        start_line = self._extract_scalar(canonical, 'bug_start_line', None)
        if start_line is not None:
            row = min(int(start_line) - 1, h - 1)  # Convert 1-indexed to 0-indexed
            col = min(self._extract_scalar(canonical, 'bug_start_col', 0), w - 1)
            return (max(0, row), max(0, col))

        # Try changed_tokens
        changed_tokens = canonical.get('changed_tokens')
        if changed_tokens and len(changed_tokens) > 0:
            first_changed = changed_tokens[0] if isinstance(changed_tokens, list) else changed_tokens
            if isinstance(first_changed, (int, np.integer)):
                row = int(first_changed) // w
                col = int(first_changed) % w
                return (min(row, h - 1), min(col, w - 1))

        # Use diff_mask
        diff_indices = np.argwhere(diff_mask > 0)
        if len(diff_indices) > 0:
            return tuple(diff_indices[0])

        return (0, 0)

    def _compute_bug_location_mask(
        self,
        canonical: Dict,
        shape: Tuple[int, int],
        diff_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create soft mask highlighting bug region.

        For CTM intermediate supervision - helps early iterations
        focus on the bug location.
        """
        mask = np.zeros(shape, dtype=np.float32)
        h, w = shape

        # Try explicit region
        start_line = self._extract_scalar(canonical, 'bug_start_line', None)
        if start_line is not None:
            start_row = max(0, int(start_line) - 1)
            end_row = max(0, int(self._extract_scalar(canonical, 'bug_end_line', start_line)) - 1)
            start_col = max(0, int(self._extract_scalar(canonical, 'bug_start_col', 0)))
            end_col = int(self._extract_scalar(canonical, 'bug_end_col', w))

            # Clip to bounds
            start_row = min(start_row, h - 1)
            end_row = min(end_row, h - 1)
            start_col = min(start_col, w - 1)
            end_col = min(end_col, w - 1)

            mask[start_row:end_row+1, start_col:end_col+1] = 1.0
            return mask

        # Fall back to diff_mask (use positions that changed)
        mask = diff_mask.copy()

        # Optionally expand the mask slightly for context
        if mask.sum() > 0:
            # Dilate the mask by 1 position in each direction
            expanded = np.zeros_like(mask)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    rolled = np.roll(np.roll(mask, dr, axis=0), dc, axis=1)
                    expanded = np.maximum(expanded, rolled * 0.5)  # Soft expansion

            mask = np.maximum(mask, expanded)

        return mask


if TORCH_AVAILABLE:
    class CTMDataset(Dataset):
        """PyTorch Dataset wrapper for CTM."""

        def __init__(
            self,
            loader: CTMDataLoader,
            preload: bool = True,
        ):
            """
            Initialize CTM dataset.

            Args:
                loader: CTMDataLoader instance
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
                'positions': torch.from_numpy(sample.positions).float(),
                'bug_location': torch.tensor(sample.bug_location, dtype=torch.long),
                'bug_location_mask': torch.from_numpy(sample.bug_location_mask).float(),
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
    class CTMDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for CTMDataset")
