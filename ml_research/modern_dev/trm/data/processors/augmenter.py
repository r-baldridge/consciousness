"""
Data Augmentation Pipeline for TRM Code Repair

Applies various augmentation strategies to increase dataset diversity
while preserving semantic equivalence.
"""

import random
import re
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..taxonomy.bug_types import BugType


class AugmentationType(Enum):
    """Types of augmentation."""
    VARIABLE_RENAME = "variable_rename"
    WHITESPACE_NORMALIZE = "whitespace_normalize"
    COMMENT_ADD = "comment_add"
    COMMENT_REMOVE = "comment_remove"
    DOCSTRING_MODIFY = "docstring_modify"
    STRING_QUOTE_FLIP = "string_quote_flip"
    IMPORT_REORDER = "import_reorder"
    BLANK_LINES = "blank_lines"
    EQUIVALENT_REWRITE = "equivalent_rewrite"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    enabled_augmentations: List[AugmentationType] = field(default_factory=lambda: [
        AugmentationType.VARIABLE_RENAME,
        AugmentationType.WHITESPACE_NORMALIZE,
        AugmentationType.COMMENT_REMOVE,
        AugmentationType.STRING_QUOTE_FLIP,
        AugmentationType.BLANK_LINES,
    ])
    max_augmentations_per_sample: int = 3
    augmentation_probability: float = 0.5


@dataclass
class AugmentedSample:
    """Result of augmentation."""
    buggy_code: str
    fixed_code: str
    bug_type: BugType
    augmentations_applied: List[AugmentationType]
    original_buggy: str
    original_fixed: str


class AugmentationPipeline:
    """
    Pipeline for augmenting code repair training data.

    Applies semantic-preserving transformations to increase
    dataset diversity and improve model generalization.

    Example:
        pipeline = AugmentationPipeline(seed=42)
        augmented = pipeline.augment(buggy, fixed, bug_type)
    """

    # Variable name pools for renaming
    VARIABLE_NAMES = [
        # Single letters
        'x', 'y', 'z', 'a', 'b', 'c', 'i', 'j', 'k', 'n', 'm',
        # Common names
        'data', 'result', 'value', 'item', 'elem', 'temp', 'var',
        'count', 'total', 'num', 'idx', 'pos', 'key', 'val',
        # Descriptive
        'current', 'previous', 'next_val', 'first', 'last',
        'input_data', 'output', 'buffer', 'cache', 'state',
    ]

    COMMENTS = [
        "# TODO: refactor this",
        "# FIXME: potential issue",
        "# NOTE: important",
        "# Process the data",
        "# Initialize variables",
        "# Main logic",
        "# Helper function",
        "# Edge case handling",
    ]

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the augmentation pipeline.

        Args:
            config: Pipeline configuration
            seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        self.rng = random.Random(seed)

        # Register augmentation functions
        self._augmentations: Dict[AugmentationType, Callable] = {
            AugmentationType.VARIABLE_RENAME: self._augment_variable_rename,
            AugmentationType.WHITESPACE_NORMALIZE: self._augment_whitespace,
            AugmentationType.COMMENT_ADD: self._augment_add_comment,
            AugmentationType.COMMENT_REMOVE: self._augment_remove_comment,
            AugmentationType.STRING_QUOTE_FLIP: self._augment_quote_flip,
            AugmentationType.IMPORT_REORDER: self._augment_import_reorder,
            AugmentationType.BLANK_LINES: self._augment_blank_lines,
            AugmentationType.EQUIVALENT_REWRITE: self._augment_equivalent_rewrite,
        }

    def augment(
        self,
        buggy_code: str,
        fixed_code: str,
        bug_type: BugType,
        num_augmentations: Optional[int] = None,
    ) -> AugmentedSample:
        """
        Augment a single code pair.

        Args:
            buggy_code: Code with bug
            fixed_code: Corrected code
            bug_type: Type of bug
            num_augmentations: Number of augmentations to apply

        Returns:
            AugmentedSample with transformed code
        """
        if num_augmentations is None:
            num_augmentations = self.rng.randint(1, self.config.max_augmentations_per_sample)

        # Select augmentations to apply
        available = [
            aug for aug in self.config.enabled_augmentations
            if self.rng.random() < self.config.augmentation_probability
        ]

        if not available:
            return AugmentedSample(
                buggy_code=buggy_code,
                fixed_code=fixed_code,
                bug_type=bug_type,
                augmentations_applied=[],
                original_buggy=buggy_code,
                original_fixed=fixed_code,
            )

        selected = self.rng.sample(
            available,
            min(num_augmentations, len(available))
        )

        # Apply augmentations to both buggy and fixed code
        augmented_buggy = buggy_code
        augmented_fixed = fixed_code
        applied = []

        for aug_type in selected:
            aug_fn = self._augmentations.get(aug_type)
            if aug_fn is None:
                continue

            # Apply same augmentation to both to preserve relationship
            result = aug_fn(augmented_buggy, augmented_fixed)
            if result is not None:
                augmented_buggy, augmented_fixed = result
                applied.append(aug_type)

        return AugmentedSample(
            buggy_code=augmented_buggy,
            fixed_code=augmented_fixed,
            bug_type=bug_type,
            augmentations_applied=applied,
            original_buggy=buggy_code,
            original_fixed=fixed_code,
        )

    def augment_batch(
        self,
        pairs: List[Tuple[str, str, BugType]],
        samples_per_pair: int = 2,
    ) -> List[AugmentedSample]:
        """
        Augment a batch of code pairs.

        Args:
            pairs: List of (buggy, fixed, bug_type) tuples
            samples_per_pair: Number of augmented versions per pair

        Returns:
            List of AugmentedSample objects
        """
        results = []

        for buggy, fixed, bug_type in pairs:
            # Always include original
            results.append(AugmentedSample(
                buggy_code=buggy,
                fixed_code=fixed,
                bug_type=bug_type,
                augmentations_applied=[],
                original_buggy=buggy,
                original_fixed=fixed,
            ))

            # Generate augmented versions
            for _ in range(samples_per_pair - 1):
                augmented = self.augment(buggy, fixed, bug_type)
                results.append(augmented)

        return results

    def _augment_variable_rename(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Rename a variable consistently in both codes."""
        # Find local variable names (simple pattern)
        pattern = r'\b([a-z_][a-z0-9_]*)\s*='
        matches = re.findall(pattern, fixed)

        if not matches:
            return None

        # Filter out common names we shouldn't rename
        exclude = {'self', 'cls', 'args', 'kwargs', 'return', 'result'}
        candidates = [m for m in matches if m not in exclude and len(m) > 1]

        if not candidates:
            return None

        old_name = self.rng.choice(candidates)
        new_name = self.rng.choice([
            n for n in self.VARIABLE_NAMES
            if n != old_name and n not in fixed
        ])

        if new_name is None:
            return None

        # Replace in both (word boundary)
        new_buggy = re.sub(rf'\b{old_name}\b', new_name, buggy)
        new_fixed = re.sub(rf'\b{old_name}\b', new_name, fixed)

        return new_buggy, new_fixed

    def _augment_whitespace(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Normalize whitespace around operators."""
        # Add/remove spaces around operators
        patterns = [
            (r'(\w)\s*([+\-*/])\s*(\w)', r'\1 \2 \3'),  # Add spaces
            (r'(\w)\s+([+\-*/])\s+(\w)', r'\1\2\3'),    # Remove spaces
        ]

        pattern, replacement = self.rng.choice(patterns)

        new_buggy = re.sub(pattern, replacement, buggy)
        new_fixed = re.sub(pattern, replacement, fixed)

        return new_buggy, new_fixed

    def _augment_add_comment(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Add a comment to the code."""
        comment = self.rng.choice(self.COMMENTS)

        lines = fixed.split('\n')
        if len(lines) < 2:
            return None

        # Find a line to add comment before
        valid_lines = [
            i for i, line in enumerate(lines)
            if line.strip() and not line.strip().startswith('#')
        ]

        if not valid_lines:
            return None

        insert_idx = self.rng.choice(valid_lines)

        # Get indentation from next line
        indent = len(lines[insert_idx]) - len(lines[insert_idx].lstrip())
        indent_str = ' ' * indent

        # Insert in both
        buggy_lines = buggy.split('\n')
        fixed_lines = fixed.split('\n')

        if insert_idx < len(buggy_lines):
            buggy_lines.insert(insert_idx, indent_str + comment)
        fixed_lines.insert(insert_idx, indent_str + comment)

        return '\n'.join(buggy_lines), '\n'.join(fixed_lines)

    def _augment_remove_comment(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Remove comments from the code."""
        # Remove single-line comments
        pattern = r'^(\s*)#.*$'

        new_buggy = re.sub(pattern, '', buggy, flags=re.MULTILINE)
        new_fixed = re.sub(pattern, '', fixed, flags=re.MULTILINE)

        # Clean up multiple blank lines
        new_buggy = re.sub(r'\n\n+', '\n\n', new_buggy)
        new_fixed = re.sub(r'\n\n+', '\n\n', new_fixed)

        return new_buggy.strip(), new_fixed.strip()

    def _augment_quote_flip(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Flip string quotes between single and double."""
        # Simple single-to-double quote flip
        # Only for simple strings without internal quotes

        def flip_quotes(code: str) -> str:
            # Find simple single-quoted strings
            result = []
            i = 0
            while i < len(code):
                if code[i] == "'" and (i == 0 or code[i-1] != '\\'):
                    # Find matching quote
                    j = i + 1
                    while j < len(code):
                        if code[j] == "'" and code[j-1] != '\\':
                            # Check if no double quotes inside
                            content = code[i+1:j]
                            if '"' not in content and self.rng.random() < 0.5:
                                result.append('"')
                                result.append(content)
                                result.append('"')
                                i = j + 1
                                break
                        j += 1
                    else:
                        result.append(code[i])
                        i += 1
                else:
                    result.append(code[i])
                    i += 1
            return ''.join(result)

        new_buggy = flip_quotes(buggy)
        new_fixed = flip_quotes(fixed)

        return new_buggy, new_fixed

    def _augment_import_reorder(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Reorder import statements."""
        def reorder_imports(code: str) -> str:
            lines = code.split('\n')
            import_lines = []
            other_lines = []
            import_section_ended = False

            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    if not import_section_ended:
                        import_lines.append(line)
                    else:
                        other_lines.append(line)
                else:
                    if import_lines and stripped:
                        import_section_ended = True
                    other_lines.append(line)

            # Shuffle imports
            self.rng.shuffle(import_lines)

            return '\n'.join(import_lines + other_lines)

        new_buggy = reorder_imports(buggy)
        new_fixed = reorder_imports(fixed)

        return new_buggy, new_fixed

    def _augment_blank_lines(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Add or remove blank lines."""
        action = self.rng.choice(['add', 'remove'])

        if action == 'add':
            # Add blank line before a random line
            lines = fixed.split('\n')
            if len(lines) < 3:
                return None

            idx = self.rng.randint(1, len(lines) - 1)

            buggy_lines = buggy.split('\n')
            fixed_lines = fixed.split('\n')

            if idx < len(buggy_lines):
                buggy_lines.insert(idx, '')
            fixed_lines.insert(idx, '')

            return '\n'.join(buggy_lines), '\n'.join(fixed_lines)
        else:
            # Remove a blank line
            new_buggy = re.sub(r'\n\n+', '\n\n', buggy)
            new_fixed = re.sub(r'\n\n+', '\n\n', fixed)
            return new_buggy, new_fixed

    def _augment_equivalent_rewrite(
        self,
        buggy: str,
        fixed: str,
    ) -> Optional[Tuple[str, str]]:
        """Apply equivalent code transformations."""
        transformations = [
            # if x == True -> if x
            (r'if\s+(\w+)\s*==\s*True:', r'if \1:'),
            # if x == False -> if not x
            (r'if\s+(\w+)\s*==\s*False:', r'if not \1:'),
            # x = x + 1 -> x += 1
            (r'(\w+)\s*=\s*\1\s*\+\s*1', r'\1 += 1'),
            # len(x) == 0 -> not x
            (r'len\((\w+)\)\s*==\s*0', r'not \1'),
            # != None -> is not None
            (r'!=\s*None', 'is not None'),
        ]

        pattern, replacement = self.rng.choice(transformations)

        new_buggy = re.sub(pattern, replacement, buggy)
        new_fixed = re.sub(pattern, replacement, fixed)

        # Only return if something changed
        if new_fixed == fixed:
            return None

        return new_buggy, new_fixed

    def get_statistics(
        self,
        samples: List[AugmentedSample],
    ) -> Dict[str, any]:
        """
        Get statistics about augmented samples.

        Args:
            samples: List of AugmentedSample objects

        Returns:
            Dictionary with statistics
        """
        aug_counts = {}
        for sample in samples:
            for aug in sample.augmentations_applied:
                aug_counts[aug.value] = aug_counts.get(aug.value, 0) + 1

        return {
            'total_samples': len(samples),
            'original_samples': sum(1 for s in samples if not s.augmentations_applied),
            'augmented_samples': sum(1 for s in samples if s.augmentations_applied),
            'augmentation_counts': aug_counts,
            'avg_augmentations': sum(len(s.augmentations_applied) for s in samples) / len(samples) if samples else 0,
        }
