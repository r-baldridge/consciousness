# Unified Data Pipeline Specification

## Overview

This document specifies a unified data pipeline for training code repair models across multiple architectures (TRM, CTM, and future models). The pipeline collects, processes, and stores training data in a canonical format with architecture-specific adapters for consumption.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED DATA PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA SOURCES                    CANONICAL STORAGE                          │
│  ════════════                    ═════════════════                          │
│  ┌─────────────┐                                                            │
│  │   GitHub    │──┐                                                         │
│  │   Commits   │  │                                                         │
│  └─────────────┘  │             ┌─────────────────────┐                    │
│  ┌─────────────┐  │             │                     │                    │
│  │    Stack    │──┼────────────▶│  Canonical Format   │                    │
│  │   Overflow  │  │             │  (Apache Parquet)   │                    │
│  └─────────────┘  │             │                     │                    │
│  ┌─────────────┐  │             └──────────┬──────────┘                    │
│  │  Synthetic  │──┤                        │                               │
│  │  Generator  │  │                        │                               │
│  └─────────────┘  │                        ▼                               │
│  ┌─────────────┐  │             ┌─────────────────────┐                    │
│  │   Linter    │──┘             │  Architecture       │                    │
│  │   Outputs   │                │  Adapters           │                    │
│  └─────────────┘                └──────────┬──────────┘                    │
│                                            │                               │
│                              ┌─────────────┼─────────────┐                 │
│                              │             │             │                 │
│                              ▼             ▼             ▼                 │
│                         ┌────────┐   ┌────────┐   ┌────────┐              │
│                         │  TRM   │   │  CTM   │   │ Future │              │
│                         │ Loader │   │ Loader │   │ Loader │              │
│                         └────────┘   └────────┘   └────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Canonical Data Format

### Core Schema

All training samples are stored in a canonical format that captures the superset of information needed by any architecture.

```python
@dataclass
class CanonicalCodeSample:
    """
    Canonical training sample for code repair models.

    This format stores all information that any architecture might need.
    Architecture-specific loaders extract relevant fields.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # REQUIRED FIELDS (All architectures)
    # ═══════════════════════════════════════════════════════════════════════

    sample_id: str                    # Unique identifier (UUID)
    buggy_code: str                   # Original buggy source code
    fixed_code: str                   # Corrected source code
    bug_type: str                     # Primary bug classification
    source: str                       # Data source (github/stackoverflow/synthetic/linter)

    # ═══════════════════════════════════════════════════════════════════════
    # TOKENIZATION (Computed on ingestion)
    # ═══════════════════════════════════════════════════════════════════════

    buggy_tokens: List[int]           # BPE token IDs (flat sequence)
    fixed_tokens: List[int]           # BPE token IDs (flat sequence)
    buggy_token_count: int            # Number of tokens in buggy code
    fixed_token_count: int            # Number of tokens in fixed code

    # ═══════════════════════════════════════════════════════════════════════
    # BUG METADATA
    # ═══════════════════════════════════════════════════════════════════════

    bug_category: str                 # Category (syntax/logic/style/security/framework)
    bug_subcategory: Optional[str]    # Finer classification
    difficulty: float                 # 0.0 (trivial) to 1.0 (expert)

    # Bug location (character offsets in buggy_code)
    bug_start_char: Optional[int]     # Start of buggy region
    bug_end_char: Optional[int]       # End of buggy region

    # Bug location (line/column)
    bug_start_line: Optional[int]     # 1-indexed line number
    bug_start_col: Optional[int]      # 0-indexed column
    bug_end_line: Optional[int]
    bug_end_col: Optional[int]

    # Bug location (token offsets)
    bug_start_token: Optional[int]    # Token index where bug starts
    bug_end_token: Optional[int]      # Token index where bug ends

    # ═══════════════════════════════════════════════════════════════════════
    # DIFF INFORMATION
    # ═══════════════════════════════════════════════════════════════════════

    diff_unified: str                 # Unified diff format
    changed_lines: List[int]          # Line numbers that changed (1-indexed)
    changed_tokens: List[int]         # Token indices that changed
    edit_distance: int                # Levenshtein distance (characters)
    token_edit_distance: int          # Levenshtein distance (tokens)

    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL INFORMATION (Optional, computed on demand)
    # ═══════════════════════════════════════════════════════════════════════

    buggy_ast: Optional[str]          # AST as JSON string (if parseable)
    fixed_ast: Optional[str]          # AST as JSON string
    ast_diff: Optional[str]           # Structural diff of ASTs
    is_syntactically_valid_buggy: bool  # Does buggy code parse?
    is_syntactically_valid_fixed: bool  # Does fixed code parse?

    # ═══════════════════════════════════════════════════════════════════════
    # SEMANTIC INFORMATION (Optional, expensive to compute)
    # ═══════════════════════════════════════════════════════════════════════

    buggy_symbols: Optional[List[str]]     # Defined symbols (functions, classes, vars)
    fixed_symbols: Optional[List[str]]
    imports: Optional[List[str]]           # Import statements
    complexity_cyclomatic: Optional[int]   # Code complexity metric

    # ═══════════════════════════════════════════════════════════════════════
    # PROVENANCE
    # ═══════════════════════════════════════════════════════════════════════

    source_url: Optional[str]         # GitHub commit URL, SO question URL, etc.
    source_repo: Optional[str]        # Repository name (for GitHub)
    source_commit: Optional[str]      # Commit SHA (for GitHub)
    source_file_path: Optional[str]   # Original file path
    collection_timestamp: str         # ISO 8601 timestamp

    # ═══════════════════════════════════════════════════════════════════════
    # QUALITY METRICS
    # ═══════════════════════════════════════════════════════════════════════

    similarity_score: float           # Similarity between buggy and fixed (0-1)
    quality_score: float              # Overall sample quality (0-1)
    validation_passed: bool           # Passed all validation checks
    validation_notes: Optional[str]   # Any validation warnings
```

### Storage Format

```yaml
storage:
  format: Apache Parquet
  compression: zstd
  partitioning:
    - bug_category
    - difficulty_bucket  # [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    - source

  file_structure:
    data/
    ├── canonical/
    │   ├── bug_category=syntax/
    │   │   ├── difficulty_bucket=0.0-0.2/
    │   │   │   ├── source=github/
    │   │   │   │   └── part-00000.parquet
    │   │   │   ├── source=synthetic/
    │   │   │   └── source=stackoverflow/
    │   │   ├── difficulty_bucket=0.2-0.4/
    │   │   └── ...
    │   ├── bug_category=logic/
    │   ├── bug_category=style/
    │   ├── bug_category=security/
    │   └── bug_category=framework/
    ├── tokenizer/
    │   ├── tokenizer.json
    │   ├── vocab.json
    │   └── merges.txt
    ├── metadata/
    │   ├── schema.json
    │   ├── statistics.json
    │   └── splits.json
    └── cache/
        ├── trm/
        ├── ctm/
        └── {future_arch}/
```

### Parquet Schema

```python
PARQUET_SCHEMA = pa.schema([
    # Required fields
    ('sample_id', pa.string()),
    ('buggy_code', pa.large_string()),
    ('fixed_code', pa.large_string()),
    ('bug_type', pa.string()),
    ('source', pa.string()),

    # Tokenization
    ('buggy_tokens', pa.list_(pa.int32())),
    ('fixed_tokens', pa.list_(pa.int32())),
    ('buggy_token_count', pa.int32()),
    ('fixed_token_count', pa.int32()),

    # Bug metadata
    ('bug_category', pa.string()),
    ('bug_subcategory', pa.string()),
    ('difficulty', pa.float32()),
    ('bug_start_char', pa.int32()),
    ('bug_end_char', pa.int32()),
    ('bug_start_line', pa.int32()),
    ('bug_start_col', pa.int32()),
    ('bug_end_line', pa.int32()),
    ('bug_end_col', pa.int32()),
    ('bug_start_token', pa.int32()),
    ('bug_end_token', pa.int32()),

    # Diff information
    ('diff_unified', pa.large_string()),
    ('changed_lines', pa.list_(pa.int32())),
    ('changed_tokens', pa.list_(pa.int32())),
    ('edit_distance', pa.int32()),
    ('token_edit_distance', pa.int32()),

    # Structural (optional)
    ('buggy_ast', pa.large_string()),
    ('fixed_ast', pa.large_string()),
    ('ast_diff', pa.large_string()),
    ('is_syntactically_valid_buggy', pa.bool_()),
    ('is_syntactically_valid_fixed', pa.bool_()),

    # Semantic (optional)
    ('buggy_symbols', pa.list_(pa.string())),
    ('fixed_symbols', pa.list_(pa.string())),
    ('imports', pa.list_(pa.string())),
    ('complexity_cyclomatic', pa.int32()),

    # Provenance
    ('source_url', pa.string()),
    ('source_repo', pa.string()),
    ('source_commit', pa.string()),
    ('source_file_path', pa.string()),
    ('collection_timestamp', pa.string()),

    # Quality
    ('similarity_score', pa.float32()),
    ('quality_score', pa.float32()),
    ('validation_passed', pa.bool_()),
    ('validation_notes', pa.string()),
])
```

---

## Architecture Requirements Matrix

### Field Usage by Architecture

| Field | TRM | CTM | Notes |
|-------|-----|-----|-------|
| **Required** |
| sample_id | ✓ | ✓ | Tracking |
| buggy_code | ✓ | ✓ | Raw text backup |
| fixed_code | ✓ | ✓ | Raw text backup |
| bug_type | ✓ | ✓ | Curriculum/analysis |
| source | ✓ | ✓ | Analysis |
| buggy_tokens | ✓ | ✓ | → Grid encoding |
| fixed_tokens | ✓ | ✓ | → Grid encoding |
| **Bug Metadata** |
| bug_category | ✓ | ✓ | Curriculum learning |
| difficulty | ✓ | ✓ | Curriculum learning |
| bug_start_line/col | ○ | ✓ | CTM intermediate supervision |
| bug_start_token | ○ | ✓ | CTM bug localization |
| **Diff Information** |
| changed_tokens | ✓ | ✓ | → diff_mask |
| **Structural** |
| buggy_ast | ○ | ○ | Future: AST-aware models |
| is_syntactically_valid_* | ✓ | ✓ | Validation filtering |

Legend: ✓ = Required, ○ = Optional/Beneficial

### Architecture-Specific Requirements

#### TRM (Tiny Recursive Model)

```python
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

# Grid configuration
TRM_CONFIG = {
    'grid_height': 64,          # Max lines
    'grid_width': 48,           # Max tokens per line
    'vocab_size': 32768,        # BPE vocabulary
    'pad_token': 0,
    'unk_token': 1,
}
```

#### CTM (Continuous Thought Machine)

```python
@dataclass
class CTMSample:
    """CTM-specific training sample."""

    # Core grids (same as TRM)
    buggy_grid: np.ndarray      # [64, 48] int32 token IDs
    fixed_grid: np.ndarray      # [64, 48] int32 token IDs

    # Masks
    buggy_mask: np.ndarray      # [64, 48] bool - valid positions
    diff_mask: np.ndarray       # [64, 48] bool - changed positions

    # 2D positions for Fourier embeddings
    positions: np.ndarray       # [64, 48, 2] float32 - (row, col) normalized

    # Bug location for intermediate supervision
    bug_location: Tuple[int, int]  # (row, col) of bug start in grid
    bug_location_mask: np.ndarray  # [64, 48] bool - bug region

    # Metadata
    bug_type: str
    bug_category: str
    difficulty: float
    sample_id: str

# CTM configuration
CTM_CONFIG = {
    'grid_height': 64,
    'grid_width': 48,
    'vocab_size': 32768,
    'pad_token': 0,
    'unk_token': 1,
    'position_normalize': True,  # Normalize positions to [0, 1]
}
```

#### Future Architectures (Extensible)

```python
@dataclass
class SequenceSample:
    """For sequence-to-sequence models (future)."""

    buggy_tokens: List[int]     # Variable length
    fixed_tokens: List[int]     # Variable length
    buggy_mask: List[bool]

    # Attention mask for cross-attention to bug location
    bug_attention_mask: List[float]  # Soft attention over tokens

    max_length: int = 2048

@dataclass
class ASTSample:
    """For AST-aware models (future)."""

    buggy_ast_nodes: List[int]  # Linearized AST node types
    buggy_ast_edges: List[Tuple[int, int]]  # Parent-child edges
    fixed_ast_nodes: List[int]
    fixed_ast_edges: List[Tuple[int, int]]

    # Mapping back to tokens
    node_to_token: Dict[int, List[int]]

@dataclass
class HybridSample:
    """For hybrid token+AST models (future)."""

    # Token representation
    buggy_grid: np.ndarray
    fixed_grid: np.ndarray

    # AST representation
    buggy_ast_embedding: np.ndarray  # Pre-computed AST embedding
    fixed_ast_embedding: np.ndarray

    # Alignment
    token_to_ast_node: np.ndarray  # [64, 48] int - AST node ID per token
```

---

## Data Loaders

### Base Loader Interface

```python
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import pyarrow.parquet as pq

class BaseDataLoader(ABC):
    """Abstract base class for architecture-specific data loaders."""

    def __init__(
        self,
        data_path: Path,
        tokenizer_path: Path,
        config: Dict[str, Any],
        split: str = 'train',  # train/val/test
        curriculum_stage: Optional[int] = None,
    ):
        self.data_path = data_path
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.config = config
        self.split = split
        self.curriculum_stage = curriculum_stage

        # Load split indices
        self.indices = self._load_split_indices()

        # Curriculum filtering
        if curriculum_stage is not None:
            self.indices = self._filter_by_curriculum(curriculum_stage)

    @abstractmethod
    def _transform_sample(self, canonical: Dict) -> Any:
        """Transform canonical format to architecture-specific format."""
        pass

    def _load_tokenizer(self, path: Path):
        """Load shared BPE tokenizer."""
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(path / 'tokenizer.json'))

    def _load_split_indices(self) -> List[str]:
        """Load sample IDs for this split."""
        splits_file = self.data_path / 'metadata' / 'splits.json'
        with open(splits_file) as f:
            splits = json.load(f)
        return splits[self.split]

    def _filter_by_curriculum(self, stage: int) -> List[str]:
        """Filter indices based on curriculum stage."""
        # Stage configs define difficulty and bug_type filters
        stage_config = CURRICULUM_STAGES[stage]

        # Query parquet with filters
        filters = [
            ('difficulty', '<=', stage_config['max_difficulty']),
            ('bug_category', 'in', stage_config['bug_categories']),
        ]

        # Return filtered indices
        return self._query_indices(filters)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over samples in architecture-specific format."""
        for sample_id in self.indices:
            canonical = self._load_canonical(sample_id)
            yield self._transform_sample(canonical)

    def __len__(self) -> int:
        return len(self.indices)


# Curriculum stage definitions (shared across architectures)
CURRICULUM_STAGES = {
    1: {
        'max_difficulty': 0.3,
        'bug_categories': ['syntax'],
        'description': 'Simple syntax errors',
    },
    2: {
        'max_difficulty': 0.5,
        'bug_categories': ['syntax', 'logic'],
        'description': 'Syntax + simple logic',
    },
    3: {
        'max_difficulty': 0.7,
        'bug_categories': ['syntax', 'logic', 'style'],
        'description': 'Most bug types',
    },
    4: {
        'max_difficulty': 1.0,
        'bug_categories': ['syntax', 'logic', 'style', 'security', 'framework'],
        'description': 'Full dataset',
    },
}
```

### TRM Data Loader

```python
class TRMDataLoader(BaseDataLoader):
    """Data loader for TRM architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_encoder = GridEncoder(
            height=self.config.get('grid_height', 64),
            width=self.config.get('grid_width', 48),
            tokenizer=self.tokenizer,
        )

    def _transform_sample(self, canonical: Dict) -> TRMSample:
        """Transform canonical to TRM format."""

        # Encode to grids
        buggy_grid = self.grid_encoder.tokens_to_grid(canonical['buggy_tokens'])
        fixed_grid = self.grid_encoder.tokens_to_grid(canonical['fixed_tokens'])

        # Create masks
        buggy_mask = (buggy_grid != self.config['pad_token'])
        diff_mask = self._compute_diff_mask(buggy_grid, fixed_grid)

        return TRMSample(
            buggy_grid=buggy_grid,
            fixed_grid=fixed_grid,
            buggy_mask=buggy_mask,
            diff_mask=diff_mask,
            bug_type=canonical['bug_type'],
            bug_category=canonical['bug_category'],
            difficulty=canonical['difficulty'],
            sample_id=canonical['sample_id'],
        )

    def _compute_diff_mask(self, buggy: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Compute mask of positions that differ."""
        return buggy != fixed


class TRMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for TRM."""

    def __init__(self, loader: TRMDataLoader):
        self.loader = loader
        self.samples = list(loader)  # Pre-load for random access

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'buggy_grid': torch.from_numpy(sample.buggy_grid).long(),
            'fixed_grid': torch.from_numpy(sample.fixed_grid).long(),
            'buggy_mask': torch.from_numpy(sample.buggy_mask).bool(),
            'diff_mask': torch.from_numpy(sample.diff_mask).bool(),
            'difficulty': torch.tensor(sample.difficulty),
        }

    def __len__(self) -> int:
        return len(self.samples)
```

### CTM Data Loader

```python
class CTMDataLoader(BaseDataLoader):
    """Data loader for CTM architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_encoder = GridEncoder(
            height=self.config.get('grid_height', 64),
            width=self.config.get('grid_width', 48),
            tokenizer=self.tokenizer,
        )
        self.position_grid = self._create_position_grid()

    def _create_position_grid(self) -> np.ndarray:
        """Create normalized 2D position grid for Fourier embeddings."""
        h, w = self.config['grid_height'], self.config['grid_width']
        rows = np.arange(h) / h  # Normalized to [0, 1]
        cols = np.arange(w) / w
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
        return np.stack([row_grid, col_grid], axis=-1).astype(np.float32)

    def _transform_sample(self, canonical: Dict) -> CTMSample:
        """Transform canonical to CTM format."""

        # Encode to grids (same as TRM)
        buggy_grid = self.grid_encoder.tokens_to_grid(canonical['buggy_tokens'])
        fixed_grid = self.grid_encoder.tokens_to_grid(canonical['fixed_tokens'])

        # Create masks
        buggy_mask = (buggy_grid != self.config['pad_token'])
        diff_mask = self._compute_diff_mask(buggy_grid, fixed_grid)

        # Bug location (CTM-specific)
        bug_location = self._compute_bug_location(canonical)
        bug_location_mask = self._compute_bug_location_mask(canonical, buggy_grid.shape)

        return CTMSample(
            buggy_grid=buggy_grid,
            fixed_grid=fixed_grid,
            buggy_mask=buggy_mask,
            diff_mask=diff_mask,
            positions=self.position_grid.copy(),
            bug_location=bug_location,
            bug_location_mask=bug_location_mask,
            bug_type=canonical['bug_type'],
            bug_category=canonical['bug_category'],
            difficulty=canonical['difficulty'],
            sample_id=canonical['sample_id'],
        )

    def _compute_bug_location(self, canonical: Dict) -> Tuple[int, int]:
        """Compute bug location as grid coordinates."""
        if canonical.get('bug_start_line') is not None:
            row = min(canonical['bug_start_line'] - 1, self.config['grid_height'] - 1)
            col = min(canonical.get('bug_start_col', 0), self.config['grid_width'] - 1)
            return (row, col)

        # Fallback: find first changed token
        if canonical.get('changed_tokens'):
            first_changed = canonical['changed_tokens'][0]
            row = first_changed // self.config['grid_width']
            col = first_changed % self.config['grid_width']
            return (min(row, self.config['grid_height'] - 1), col)

        return (0, 0)  # Unknown

    def _compute_bug_location_mask(self, canonical: Dict, shape: Tuple) -> np.ndarray:
        """Create mask highlighting bug region."""
        mask = np.zeros(shape, dtype=np.float32)

        if canonical.get('bug_start_line') is not None:
            start_row = canonical['bug_start_line'] - 1
            end_row = canonical.get('bug_end_line', start_row + 1) - 1
            start_col = canonical.get('bug_start_col', 0)
            end_col = canonical.get('bug_end_col', shape[1])

            # Clip to grid bounds
            start_row = max(0, min(start_row, shape[0] - 1))
            end_row = max(0, min(end_row, shape[0] - 1))
            start_col = max(0, min(start_col, shape[1] - 1))
            end_col = max(0, min(end_col, shape[1] - 1))

            mask[start_row:end_row+1, start_col:end_col+1] = 1.0
        else:
            # Fallback: use diff_mask
            # Will be computed in _transform_sample
            pass

        return mask


class CTMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for CTM."""

    def __init__(self, loader: CTMDataLoader):
        self.loader = loader
        self.samples = list(loader)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'buggy_grid': torch.from_numpy(sample.buggy_grid).long(),
            'fixed_grid': torch.from_numpy(sample.fixed_grid).long(),
            'buggy_mask': torch.from_numpy(sample.buggy_mask).bool(),
            'diff_mask': torch.from_numpy(sample.diff_mask).bool(),
            'positions': torch.from_numpy(sample.positions).float(),
            'bug_location': torch.tensor(sample.bug_location),
            'bug_location_mask': torch.from_numpy(sample.bug_location_mask).float(),
            'difficulty': torch.tensor(sample.difficulty),
        }

    def __len__(self) -> int:
        return len(self.samples)
```

---

## Data Collection Pipeline

### Collector Interface

```python
from abc import ABC, abstractmethod

class BaseCollector(ABC):
    """Abstract base for data collectors."""

    @abstractmethod
    def collect(self, limit: Optional[int] = None) -> Iterator[CanonicalCodeSample]:
        """Yield canonical samples from this source."""
        pass

    @abstractmethod
    def estimate_total(self) -> int:
        """Estimate total available samples."""
        pass


class GitHubCollector(BaseCollector):
    """Collect bug fixes from GitHub commits."""

    def __init__(self, token: str, languages: List[str] = ['python']):
        self.github = Github(token)
        self.languages = languages

    def collect(self, limit: Optional[int] = None) -> Iterator[CanonicalCodeSample]:
        # Implementation details in trm/data/collectors/github.py
        pass


class SyntheticCollector(BaseCollector):
    """Generate synthetic bug/fix pairs."""

    def __init__(self, source_files: List[Path], bug_types: List[str]):
        self.source_files = source_files
        self.bug_types = bug_types
        self.generator = SyntheticBugGenerator()

    def collect(self, limit: Optional[int] = None) -> Iterator[CanonicalCodeSample]:
        # Implementation details in trm/data/collectors/synthetic.py
        pass


class StackOverflowCollector(BaseCollector):
    """Collect code corrections from Stack Overflow."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def collect(self, limit: Optional[int] = None) -> Iterator[CanonicalCodeSample]:
        # Implementation in trm/data/collectors/stackoverflow.py
        pass


class LinterCollector(BaseCollector):
    """Collect auto-fixes from linters (pylint, flake8, black)."""

    def __init__(self, source_files: List[Path]):
        self.source_files = source_files

    def collect(self, limit: Optional[int] = None) -> Iterator[CanonicalCodeSample]:
        # Implementation in trm/data/collectors/linter.py
        pass
```

### Pipeline Orchestrator

```python
class DataPipeline:
    """Orchestrates data collection, processing, and storage."""

    def __init__(
        self,
        output_path: Path,
        tokenizer_path: Path,
        collectors: List[BaseCollector],
    ):
        self.output_path = output_path
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path / 'tokenizer.json'))
        self.collectors = collectors
        self.validator = DataValidator()
        self.stats = CollectionStats()

    def run(
        self,
        target_samples: int = 1_000_000,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        """Run the full collection pipeline."""

        # Default weights
        if source_weights is None:
            source_weights = {
                'github': 0.5,
                'synthetic': 0.4,
                'stackoverflow': 0.08,
                'linter': 0.02,
            }

        # Collection phase
        print(f"Collecting {target_samples} samples...")
        samples = []

        for collector in self.collectors:
            source_name = collector.__class__.__name__.replace('Collector', '').lower()
            source_target = int(target_samples * source_weights.get(source_name, 0.1))

            print(f"  {source_name}: targeting {source_target} samples")

            for sample in tqdm(collector.collect(limit=source_target), total=source_target):
                # Tokenize
                sample = self._tokenize_sample(sample)

                # Validate
                if self.validator.validate(sample):
                    samples.append(sample)
                    self.stats.record_valid(source_name)
                else:
                    self.stats.record_invalid(source_name)

        # Compute derived fields
        print("Computing derived fields...")
        samples = [self._compute_derived(s) for s in tqdm(samples)]

        # Create splits
        print("Creating train/val/test splits...")
        splits = self._create_splits(samples)

        # Save
        print("Saving to Parquet...")
        self._save_parquet(samples)
        self._save_splits(splits)
        self._save_statistics()

        print(f"Complete: {len(samples)} samples saved")
        return self.stats

    def _tokenize_sample(self, sample: CanonicalCodeSample) -> CanonicalCodeSample:
        """Add tokenization fields."""
        buggy_enc = self.tokenizer.encode(sample.buggy_code)
        fixed_enc = self.tokenizer.encode(sample.fixed_code)

        sample.buggy_tokens = buggy_enc.ids
        sample.fixed_tokens = fixed_enc.ids
        sample.buggy_token_count = len(buggy_enc.ids)
        sample.fixed_token_count = len(fixed_enc.ids)

        return sample

    def _compute_derived(self, sample: CanonicalCodeSample) -> CanonicalCodeSample:
        """Compute derived fields (diff, AST, etc.)."""

        # Unified diff
        sample.diff_unified = self._compute_diff(sample.buggy_code, sample.fixed_code)

        # Changed tokens
        sample.changed_tokens = self._find_changed_tokens(
            sample.buggy_tokens, sample.fixed_tokens
        )

        # Edit distances
        sample.edit_distance = Levenshtein.distance(sample.buggy_code, sample.fixed_code)
        sample.token_edit_distance = Levenshtein.distance(
            sample.buggy_tokens, sample.fixed_tokens
        )

        # Similarity
        sample.similarity_score = 1 - (sample.edit_distance / max(
            len(sample.buggy_code), len(sample.fixed_code), 1
        ))

        # AST (optional, may fail for buggy code)
        try:
            sample.buggy_ast = json.dumps(ast.dump(ast.parse(sample.buggy_code)))
            sample.is_syntactically_valid_buggy = True
        except SyntaxError:
            sample.is_syntactically_valid_buggy = False

        try:
            sample.fixed_ast = json.dumps(ast.dump(ast.parse(sample.fixed_code)))
            sample.is_syntactically_valid_fixed = True
        except SyntaxError:
            sample.is_syntactically_valid_fixed = False

        return sample

    def _create_splits(
        self,
        samples: List[CanonicalCodeSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, List[str]]:
        """Create stratified train/val/test splits."""

        # Group by bug_category and difficulty bucket for stratification
        groups = defaultdict(list)
        for sample in samples:
            diff_bucket = int(sample.difficulty * 5) / 5  # 0.0, 0.2, 0.4, 0.6, 0.8
            key = (sample.bug_category, diff_bucket)
            groups[key].append(sample.sample_id)

        splits = {'train': [], 'val': [], 'test': []}

        for key, ids in groups.items():
            random.shuffle(ids)
            n = len(ids)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            splits['train'].extend(ids[:train_end])
            splits['val'].extend(ids[train_end:val_end])
            splits['test'].extend(ids[val_end:])

        return splits
```

---

## Supplying Architecture-Specific Data

### How Each Architecture Gets What It Needs

#### TRM: Standard Grid Loading

```python
# TRM uses the base grid encoding
# No additional data required beyond canonical format

from shared.data import TRMDataLoader, TRMDataset

loader = TRMDataLoader(
    data_path=Path('data/canonical'),
    tokenizer_path=Path('data/tokenizer'),
    config=TRM_CONFIG,
    split='train',
    curriculum_stage=1,
)

dataset = TRMDataset(loader)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

#### CTM: Grid + Positions + Bug Location

```python
# CTM needs additional fields computed by loader:
# - positions: 2D position grid for Fourier embeddings
# - bug_location: (row, col) tuple for intermediate supervision
# - bug_location_mask: soft mask over bug region

from shared.data import CTMDataLoader, CTMDataset

loader = CTMDataLoader(
    data_path=Path('data/canonical'),
    tokenizer_path=Path('data/tokenizer'),
    config=CTM_CONFIG,
    split='train',
    curriculum_stage=1,
)

dataset = CTMDataset(loader)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# CTM-specific fields in batch:
# batch['positions']         # [B, 64, 48, 2] normalized (row, col)
# batch['bug_location']      # [B, 2] (row, col) of bug start
# batch['bug_location_mask'] # [B, 64, 48] soft attention mask
```

#### Future: Sequence Models

```python
# Sequence models need variable-length token sequences
# Canonical format already stores buggy_tokens/fixed_tokens as lists

class SequenceDataLoader(BaseDataLoader):
    def _transform_sample(self, canonical: Dict) -> SequenceSample:
        return SequenceSample(
            buggy_tokens=canonical['buggy_tokens'][:self.max_length],
            fixed_tokens=canonical['fixed_tokens'][:self.max_length],
            buggy_mask=[True] * len(canonical['buggy_tokens']),
            bug_attention_mask=self._compute_attention_mask(canonical),
        )
```

#### Future: AST-Aware Models

```python
# AST models need parsed AST structures
# Canonical format stores buggy_ast/fixed_ast as JSON strings
# Loader can parse and convert to graph structures

class ASTDataLoader(BaseDataLoader):
    def _transform_sample(self, canonical: Dict) -> ASTSample:
        if canonical['buggy_ast'] is None:
            return None  # Skip unparseable samples

        buggy_ast = json.loads(canonical['buggy_ast'])
        fixed_ast = json.loads(canonical['fixed_ast'])

        return ASTSample(
            buggy_ast_nodes=self._linearize_ast(buggy_ast),
            buggy_ast_edges=self._extract_edges(buggy_ast),
            fixed_ast_nodes=self._linearize_ast(fixed_ast),
            fixed_ast_edges=self._extract_edges(fixed_ast),
            node_to_token=self._align_ast_to_tokens(buggy_ast, canonical['buggy_tokens']),
        )
```

---

## Data Quality Tiers

### Tier System

Not all samples are equal quality. The canonical format includes quality signals that loaders can use for filtering.

```python
class QualityTier(Enum):
    GOLD = "gold"       # Human-verified or high-confidence real bugs
    SILVER = "silver"   # Real bugs from commits, moderate confidence
    BRONZE = "bronze"   # Synthetic bugs, linter fixes

def assign_quality_tier(sample: CanonicalCodeSample) -> QualityTier:
    """Assign quality tier based on source and validation signals."""

    # Gold: Human-curated or verified real bugs
    if sample.source == 'curated':
        return QualityTier.GOLD

    # Silver: Real bugs from GitHub with good signals
    if sample.source == 'github':
        if (sample.is_syntactically_valid_fixed and
            'fix' in sample.source_commit.lower() and
            sample.similarity_score > 0.7):
            return QualityTier.SILVER

    # Bronze: Everything else
    return QualityTier.BRONZE


# Loaders can filter by quality tier
loader = TRMDataLoader(
    data_path=Path('data/canonical'),
    config=TRM_CONFIG,
    quality_tiers=[QualityTier.GOLD, QualityTier.SILVER],  # Exclude bronze
)
```

---

## Directory Structure

```
shared/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── schema.py              # CanonicalCodeSample definition
│   ├── pipeline.py            # DataPipeline orchestrator
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseCollector ABC
│   │   ├── github.py          # GitHub commit collector
│   │   ├── synthetic.py       # Synthetic bug generator
│   │   ├── stackoverflow.py   # Stack Overflow collector
│   │   └── linter.py          # Linter output collector
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── tokenizer.py       # BPE tokenizer wrapper
│   │   ├── validator.py       # Data validation
│   │   ├── augmenter.py       # Data augmentation
│   │   └── grid_encoder.py    # Token → grid conversion
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseDataLoader ABC
│   │   ├── trm.py             # TRM-specific loader
│   │   ├── ctm.py             # CTM-specific loader
│   │   └── sequence.py        # Sequence model loader (future)
│   └── utils/
│       ├── __init__.py
│       ├── diff.py            # Diff computation
│       ├── ast_utils.py       # AST parsing utilities
│       └── quality.py         # Quality tier assignment
├── taxonomy/
│   ├── __init__.py
│   └── bug_types.py           # Bug taxonomy (shared)
├── tokenizer/
│   ├── train_tokenizer.py     # BPE tokenizer training
│   └── config.yaml            # Tokenizer configuration
└── docs/
    └── UNIFIED_DATA_PIPELINE.md  # This document
```

---

## Usage Examples

### Full Pipeline Run

```bash
# Train tokenizer (once)
python -m shared.tokenizer.train_tokenizer \
    --input data/raw_python/*.py \
    --output data/tokenizer \
    --vocab-size 32768

# Run collection pipeline
python -m shared.data.pipeline \
    --output data/canonical \
    --tokenizer data/tokenizer \
    --github-token $GITHUB_TOKEN \
    --stackoverflow-key $SO_API_KEY \
    --target-samples 1000000 \
    --source-weights github=0.5,synthetic=0.4,stackoverflow=0.08,linter=0.02

# Verify collection
python -m shared.data.verify \
    --data data/canonical \
    --check schema,splits,statistics
```

### Architecture-Specific Training

```python
# TRM Training
from shared.data import TRMDataLoader, TRMDataset
from trm.model import TinyRecursiveModel

train_loader = DataLoader(
    TRMDataset(TRMDataLoader(
        data_path=Path('data/canonical'),
        tokenizer_path=Path('data/tokenizer'),
        config=TRM_CONFIG,
        split='train',
        curriculum_stage=1,
    )),
    batch_size=64,
    shuffle=True,
)

model = TinyRecursiveModel(TRM_CONFIG)
# ... training loop


# CTM Training
from shared.data import CTMDataLoader, CTMDataset
from ctm.model import ContinuousThoughtMachine

train_loader = DataLoader(
    CTMDataset(CTMDataLoader(
        data_path=Path('data/canonical'),
        tokenizer_path=Path('data/tokenizer'),
        config=CTM_CONFIG,
        split='train',
        curriculum_stage=1,
    )),
    batch_size=32,
    shuffle=True,
)

model = ContinuousThoughtMachine(CTM_CONFIG)
# ... training loop
```

---

## Summary

| Aspect | Specification |
|--------|---------------|
| **Storage Format** | Apache Parquet with zstd compression |
| **Partitioning** | bug_category / difficulty_bucket / source |
| **Tokenization** | 32K BPE vocabulary, shared across architectures |
| **Grid Encoding** | 64×48 tokens (computed by loaders) |
| **Required Fields** | 5 core + tokenization + bug metadata |
| **Optional Fields** | AST, semantic, provenance |
| **Architecture Adapters** | TRM, CTM, Sequence, AST (extensible) |
| **Quality Tiers** | Gold, Silver, Bronze |
| **Curriculum Support** | 4 stages, filtered by difficulty + bug_category |
