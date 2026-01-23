"""
Data Validator for TRM Code Repair Dataset

Validates buggy/fixed code pairs to ensure data quality.
"""

import ast
import re
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum

from ..taxonomy.bug_types import BugType, BugCategory


class ValidationResult(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID_FIXED = "invalid_fixed"       # Fixed code doesn't parse
    INVALID_BUGGY = "invalid_buggy"       # Buggy code parses (should have error)
    TOO_SIMILAR = "too_similar"           # Codes are too similar
    TOO_DIFFERENT = "too_different"       # Codes are too different
    TOO_SHORT = "too_short"               # Code is too short
    TOO_LONG = "too_long"                 # Code is too long
    TRIVIAL_CHANGE = "trivial_change"     # Change is too trivial
    MULTIPLE_BUGS = "multiple_bugs"       # Contains multiple unrelated bugs
    BAD_BUG_TYPE = "bad_bug_type"         # Bug type doesn't match actual bug


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    min_lines: int = 3
    max_lines: int = 100
    min_diff_tokens: int = 1
    max_diff_tokens: int = 50
    min_similarity: float = 0.5    # Buggy/fixed should be somewhat similar
    max_similarity: float = 1.0    # Allow very similar (1 char changes are valid)
    require_syntax_error: bool = True  # For syntax bugs
    allow_multiple_changes: bool = False
    skip_similarity_for_syntax: bool = True  # Skip similarity check for syntax bugs


@dataclass
class ValidationReport:
    """Detailed validation report."""
    result: ValidationResult
    is_valid: bool
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class DataValidator:
    """
    Validates code repair training pairs.

    Ensures that:
    1. Fixed code is valid Python
    2. Buggy code has the expected defect
    3. The difference is appropriate for the bug type
    4. Code length is within bounds

    Example:
        validator = DataValidator()
        result = validator.validate(buggy, fixed, BugType.MISSING_COLON)
        if result.is_valid:
            # Use the pair
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validator.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def validate(
        self,
        buggy_code: str,
        fixed_code: str,
        bug_type: BugType,
    ) -> ValidationReport:
        """
        Validate a buggy/fixed code pair.

        Args:
            buggy_code: Code containing the bug
            fixed_code: Corrected code
            bug_type: Type of bug

        Returns:
            ValidationReport with result and details
        """
        warnings = []
        metrics = {}

        # Check fixed code is valid Python
        if not self._is_valid_python(fixed_code):
            return ValidationReport(
                result=ValidationResult.INVALID_FIXED,
                is_valid=False,
                reason="Fixed code is not valid Python",
                metrics=metrics,
                warnings=warnings,
            )

        # Check buggy code has appropriate error for syntax bugs
        category = self._get_category(bug_type)
        if category == BugCategory.SYNTAX and self.config.require_syntax_error:
            if self._is_valid_python(buggy_code):
                return ValidationReport(
                    result=ValidationResult.INVALID_BUGGY,
                    is_valid=False,
                    reason="Buggy code should have syntax error but parses successfully",
                    metrics=metrics,
                    warnings=warnings,
                )

        # Check code length
        fixed_lines = fixed_code.count('\n') + 1
        metrics['fixed_lines'] = fixed_lines

        if fixed_lines < self.config.min_lines:
            return ValidationReport(
                result=ValidationResult.TOO_SHORT,
                is_valid=False,
                reason=f"Code too short: {fixed_lines} lines (min: {self.config.min_lines})",
                metrics=metrics,
                warnings=warnings,
            )

        if fixed_lines > self.config.max_lines:
            return ValidationReport(
                result=ValidationResult.TOO_LONG,
                is_valid=False,
                reason=f"Code too long: {fixed_lines} lines (max: {self.config.max_lines})",
                metrics=metrics,
                warnings=warnings,
            )

        # Compute similarity
        similarity = self._compute_similarity(buggy_code, fixed_code)
        metrics['similarity'] = similarity

        # Skip similarity check for syntax bugs if configured
        skip_similarity = (
            self.config.skip_similarity_for_syntax and
            category == BugCategory.SYNTAX
        )

        if not skip_similarity and similarity > self.config.max_similarity:
            return ValidationReport(
                result=ValidationResult.TOO_SIMILAR,
                is_valid=False,
                reason=f"Codes too similar: {similarity:.2%} (max: {self.config.max_similarity:.2%})",
                metrics=metrics,
                warnings=warnings,
            )

        if similarity < self.config.min_similarity:
            return ValidationReport(
                result=ValidationResult.TOO_DIFFERENT,
                is_valid=False,
                reason=f"Codes too different: {similarity:.2%} (min: {self.config.min_similarity:.2%})",
                metrics=metrics,
                warnings=warnings,
            )

        # Compute diff size
        diff_tokens = self._count_diff_tokens(buggy_code, fixed_code)
        metrics['diff_tokens'] = diff_tokens

        if diff_tokens < self.config.min_diff_tokens:
            return ValidationReport(
                result=ValidationResult.TRIVIAL_CHANGE,
                is_valid=False,
                reason=f"Change too trivial: {diff_tokens} tokens (min: {self.config.min_diff_tokens})",
                metrics=metrics,
                warnings=warnings,
            )

        if diff_tokens > self.config.max_diff_tokens:
            return ValidationReport(
                result=ValidationResult.TOO_DIFFERENT,
                is_valid=False,
                reason=f"Change too large: {diff_tokens} tokens (max: {self.config.max_diff_tokens})",
                metrics=metrics,
                warnings=warnings,
            )

        # Check if multiple independent changes
        if not self.config.allow_multiple_changes:
            change_regions = self._count_change_regions(buggy_code, fixed_code)
            metrics['change_regions'] = change_regions

            if change_regions > 2:
                warnings.append(f"Multiple change regions detected: {change_regions}")
                # Allow but warn

        # Verify bug type matches actual change
        if not self._verify_bug_type(buggy_code, fixed_code, bug_type):
            warnings.append(f"Bug type {bug_type.value} may not match actual change")

        return ValidationReport(
            result=ValidationResult.VALID,
            is_valid=True,
            reason="Validation passed",
            metrics=metrics,
            warnings=warnings,
        )

    def validate_batch(
        self,
        pairs: List[Tuple[str, str, BugType]],
    ) -> Tuple[List[Tuple[str, str, BugType]], List[ValidationReport]]:
        """
        Validate a batch of pairs, returning valid ones.

        Args:
            pairs: List of (buggy, fixed, bug_type) tuples

        Returns:
            Tuple of (valid_pairs, all_reports)
        """
        valid_pairs = []
        reports = []

        for buggy, fixed, bug_type in pairs:
            report = self.validate(buggy, fixed, bug_type)
            reports.append(report)

            if report.is_valid:
                valid_pairs.append((buggy, fixed, bug_type))

        return valid_pairs, reports

    def get_statistics(
        self,
        reports: List[ValidationReport],
    ) -> Dict[str, any]:
        """
        Get statistics from validation reports.

        Args:
            reports: List of ValidationReport objects

        Returns:
            Dictionary with statistics
        """
        total = len(reports)
        valid = sum(1 for r in reports if r.is_valid)

        result_counts = {}
        for r in reports:
            result_counts[r.result.value] = result_counts.get(r.result.value, 0) + 1

        metrics = {}
        for key in ['similarity', 'diff_tokens', 'fixed_lines']:
            values = [r.metrics.get(key) for r in reports if key in r.metrics]
            if values:
                metrics[f'avg_{key}'] = sum(values) / len(values)
                metrics[f'min_{key}'] = min(values)
                metrics[f'max_{key}'] = max(values)

        return {
            'total': total,
            'valid': valid,
            'valid_rate': valid / total if total > 0 else 0,
            'result_counts': result_counts,
            'metrics': metrics,
        }

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _get_category(self, bug_type: BugType) -> BugCategory:
        """Get the category for a bug type."""
        # Map bug types to categories
        syntax_bugs = {
            BugType.MISSING_COLON, BugType.MISSING_PARENTHESIS,
            BugType.MISSING_BRACKET, BugType.MISSING_QUOTE,
            BugType.INDENTATION_ERROR, BugType.INVALID_SYNTAX,
            BugType.MISSING_COMMA, BugType.FSTRING_ERROR,
        }

        if bug_type in syntax_bugs:
            return BugCategory.SYNTAX
        return BugCategory.LOGIC  # Default

    def _compute_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity between two code strings."""
        # Token-based Jaccard similarity
        tokens1 = set(self._tokenize(code1))
        tokens2 = set(self._tokenize(code2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _count_diff_tokens(self, code1: str, code2: str) -> int:
        """Count the number of tokens that differ."""
        tokens1 = self._tokenize(code1)
        tokens2 = self._tokenize(code2)

        # Simple diff count
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, tokens1, tokens2)

        diff_count = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                diff_count += max(i2 - i1, j2 - j1)

        return diff_count

    def _count_change_regions(self, code1: str, code2: str) -> int:
        """Count the number of separate change regions."""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')

        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, lines1, lines2)

        regions = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                regions += 1

        return regions

    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code into a list of tokens."""
        return re.findall(r'\b\w+\b|[^\s\w]', code)

    def _verify_bug_type(
        self,
        buggy_code: str,
        fixed_code: str,
        bug_type: BugType,
    ) -> bool:
        """Verify that the bug type matches the actual change."""
        # Simple heuristic checks
        checks = {
            BugType.MISSING_COLON: lambda b, f: ':' in f and b.count(':') < f.count(':'),
            BugType.MISSING_PARENTHESIS: lambda b, f: b.count('(') != b.count(')'),
            BugType.MISSING_BRACKET: lambda b, f: b.count('[') != b.count(']') or b.count('{') != b.count('}'),
            BugType.INDENTATION_ERROR: lambda b, f: self._has_indent_diff(b, f),
            BugType.OFF_BY_ONE: lambda b, f: self._has_numeric_diff(b, f),
            BugType.WRONG_OPERATOR: lambda b, f: self._has_operator_diff(b, f),
        }

        check = checks.get(bug_type)
        if check is None:
            return True  # Can't verify, assume valid

        return check(buggy_code, fixed_code)

    def _has_indent_diff(self, code1: str, code2: str) -> bool:
        """Check if there's an indentation difference."""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')

        for l1, l2 in zip(lines1, lines2):
            indent1 = len(l1) - len(l1.lstrip())
            indent2 = len(l2) - len(l2.lstrip())
            if indent1 != indent2:
                return True
        return False

    def _has_numeric_diff(self, code1: str, code2: str) -> bool:
        """Check if there's a numeric difference."""
        nums1 = set(re.findall(r'\b\d+\b', code1))
        nums2 = set(re.findall(r'\b\d+\b', code2))
        return nums1 != nums2

    def _has_operator_diff(self, code1: str, code2: str) -> bool:
        """Check if there's an operator difference."""
        ops = ['+', '-', '*', '/', '//', '**', '%', '==', '!=', '<', '>', '<=', '>=']
        for op in ops:
            if code1.count(op) != code2.count(op):
                return True
        return False
