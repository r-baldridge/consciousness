"""
Canonical data schema for code repair training samples.

This schema defines the superset of information needed by any architecture.
Architecture-specific loaders extract relevant fields.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import uuid
from datetime import datetime


class QualityTier(Enum):
    """Quality tier for training samples."""
    GOLD = "gold"       # Human-verified or high-confidence real bugs
    SILVER = "silver"   # Real bugs from commits, moderate confidence
    BRONZE = "bronze"   # Synthetic bugs, linter fixes


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

    buggy_code: str                   # Original buggy source code
    fixed_code: str                   # Corrected source code
    bug_type: str                     # Primary bug classification
    source: str                       # Data source (github/stackoverflow/synthetic/linter)

    # Auto-generated
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ═══════════════════════════════════════════════════════════════════════
    # TOKENIZATION (Computed on ingestion)
    # ═══════════════════════════════════════════════════════════════════════

    buggy_tokens: List[int] = field(default_factory=list)
    fixed_tokens: List[int] = field(default_factory=list)
    buggy_token_count: int = 0
    fixed_token_count: int = 0

    # ═══════════════════════════════════════════════════════════════════════
    # BUG METADATA
    # ═══════════════════════════════════════════════════════════════════════

    bug_category: str = ""            # Category (syntax/logic/style/security/framework)
    bug_subcategory: Optional[str] = None
    difficulty: float = 0.5           # 0.0 (trivial) to 1.0 (expert)

    # Bug location (character offsets in buggy_code)
    bug_start_char: Optional[int] = None
    bug_end_char: Optional[int] = None

    # Bug location (line/column)
    bug_start_line: Optional[int] = None
    bug_start_col: Optional[int] = None
    bug_end_line: Optional[int] = None
    bug_end_col: Optional[int] = None

    # Bug location (token offsets)
    bug_start_token: Optional[int] = None
    bug_end_token: Optional[int] = None

    # ═══════════════════════════════════════════════════════════════════════
    # DIFF INFORMATION
    # ═══════════════════════════════════════════════════════════════════════

    diff_unified: str = ""            # Unified diff format
    changed_lines: List[int] = field(default_factory=list)
    changed_tokens: List[int] = field(default_factory=list)
    edit_distance: int = 0            # Levenshtein distance (characters)
    token_edit_distance: int = 0      # Levenshtein distance (tokens)

    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURAL INFORMATION (Optional, computed on demand)
    # ═══════════════════════════════════════════════════════════════════════

    buggy_ast: Optional[str] = None   # AST as JSON string (if parseable)
    fixed_ast: Optional[str] = None   # AST as JSON string
    ast_diff: Optional[str] = None    # Structural diff of ASTs
    is_syntactically_valid_buggy: bool = False
    is_syntactically_valid_fixed: bool = True

    # ═══════════════════════════════════════════════════════════════════════
    # SEMANTIC INFORMATION (Optional, expensive to compute)
    # ═══════════════════════════════════════════════════════════════════════

    buggy_symbols: Optional[List[str]] = None
    fixed_symbols: Optional[List[str]] = None
    imports: Optional[List[str]] = None
    complexity_cyclomatic: Optional[int] = None

    # ═══════════════════════════════════════════════════════════════════════
    # PROVENANCE
    # ═══════════════════════════════════════════════════════════════════════

    source_url: Optional[str] = None
    source_repo: Optional[str] = None
    source_commit: Optional[str] = None
    source_file_path: Optional[str] = None
    collection_timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    # ═══════════════════════════════════════════════════════════════════════
    # QUALITY METRICS
    # ═══════════════════════════════════════════════════════════════════════

    similarity_score: float = 0.0     # Similarity between buggy and fixed (0-1)
    quality_score: float = 0.0        # Overall sample quality (0-1)
    quality_tier: str = QualityTier.BRONZE.value
    validation_passed: bool = False
    validation_notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'sample_id': self.sample_id,
            'buggy_code': self.buggy_code,
            'fixed_code': self.fixed_code,
            'bug_type': self.bug_type,
            'source': self.source,
            'buggy_tokens': self.buggy_tokens,
            'fixed_tokens': self.fixed_tokens,
            'buggy_token_count': self.buggy_token_count,
            'fixed_token_count': self.fixed_token_count,
            'bug_category': self.bug_category,
            'bug_subcategory': self.bug_subcategory,
            'difficulty': self.difficulty,
            'bug_start_char': self.bug_start_char,
            'bug_end_char': self.bug_end_char,
            'bug_start_line': self.bug_start_line,
            'bug_start_col': self.bug_start_col,
            'bug_end_line': self.bug_end_line,
            'bug_end_col': self.bug_end_col,
            'bug_start_token': self.bug_start_token,
            'bug_end_token': self.bug_end_token,
            'diff_unified': self.diff_unified,
            'changed_lines': self.changed_lines,
            'changed_tokens': self.changed_tokens,
            'edit_distance': self.edit_distance,
            'token_edit_distance': self.token_edit_distance,
            'buggy_ast': self.buggy_ast,
            'fixed_ast': self.fixed_ast,
            'ast_diff': self.ast_diff,
            'is_syntactically_valid_buggy': self.is_syntactically_valid_buggy,
            'is_syntactically_valid_fixed': self.is_syntactically_valid_fixed,
            'buggy_symbols': self.buggy_symbols,
            'fixed_symbols': self.fixed_symbols,
            'imports': self.imports,
            'complexity_cyclomatic': self.complexity_cyclomatic,
            'source_url': self.source_url,
            'source_repo': self.source_repo,
            'source_commit': self.source_commit,
            'source_file_path': self.source_file_path,
            'collection_timestamp': self.collection_timestamp,
            'similarity_score': self.similarity_score,
            'quality_score': self.quality_score,
            'quality_tier': self.quality_tier,
            'validation_passed': self.validation_passed,
            'validation_notes': self.validation_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CanonicalCodeSample':
        """Create from dictionary."""
        return cls(**data)


def assign_quality_tier(sample: CanonicalCodeSample) -> QualityTier:
    """Assign quality tier based on source and validation signals."""

    # Gold: Human-curated or verified real bugs
    if sample.source == 'curated':
        return QualityTier.GOLD

    # Silver: Real bugs from GitHub with good signals
    if sample.source == 'github':
        commit_msg = (sample.source_commit or '').lower()
        if (sample.is_syntactically_valid_fixed and
            any(kw in commit_msg for kw in ['fix', 'bug', 'repair', 'patch']) and
            sample.similarity_score > 0.7):
            return QualityTier.SILVER

    # Silver: Stack Overflow with accepted answers
    if sample.source == 'stackoverflow':
        if sample.is_syntactically_valid_fixed and sample.quality_score > 0.7:
            return QualityTier.SILVER

    # Bronze: Everything else
    return QualityTier.BRONZE
