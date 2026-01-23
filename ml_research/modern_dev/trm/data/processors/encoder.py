"""
Grid Encoder for TRM Code Repair

Converts tokenized Python code into the 2D grid representation
required by TRM's GridEmbedding layer.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for grid encoding."""
    max_lines: int = 64
    max_tokens_per_line: int = 48
    vocab_size: int = 512

    # Special token IDs
    pad_token: int = 0
    unk_token: int = 1
    newline_token: int = 2
    indent_token: int = 3
    dedent_token: int = 4
    bos_token: int = 5
    eos_token: int = 6

    # Bug marker tokens
    bug_start_token: int = 7
    bug_end_token: int = 8
    fix_start_token: int = 9
    fix_end_token: int = 10


class GridEncoder:
    """
    Encodes Python code into fixed-size grids for TRM.

    The grid format represents code spatially:
    - Rows represent lines of code
    - Columns represent tokens within each line
    - Values are vocabulary IDs

    This 2D representation preserves the visual structure of code,
    which is important for bug localization and repair.

    Example:
        encoder = GridEncoder(vocab)
        grid = encoder.encode(code)  # Shape: (64, 48)
        code = encoder.decode(grid)
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        config: Optional[GridConfig] = None,
    ):
        """
        Initialize the grid encoder.

        Args:
            vocab: Token to ID mapping
            config: Grid configuration
        """
        self.vocab = vocab
        self.config = config or GridConfig()
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(
        self,
        code: str,
        mark_bug_region: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        Encode code into a grid.

        Args:
            code: Python source code
            mark_bug_region: Optional (start_line, start_col, end_line, end_col)
                           to mark with bug tokens

        Returns:
            numpy array of shape (max_lines, max_tokens_per_line)
        """
        grid = np.full(
            (self.config.max_lines, self.config.max_tokens_per_line),
            self.config.pad_token,
            dtype=np.int32,
        )

        lines = code.split('\n')

        for line_idx, line in enumerate(lines[:self.config.max_lines]):
            if not line:
                # Empty line
                grid[line_idx, 0] = self.config.newline_token
                continue

            # Calculate indentation
            stripped = line.lstrip()
            indent_level = (len(line) - len(stripped)) // 4

            col_idx = 0

            # Add indent tokens
            for _ in range(min(indent_level, 8)):  # Max 8 indent levels
                if col_idx < self.config.max_tokens_per_line:
                    grid[line_idx, col_idx] = self.config.indent_token
                    col_idx += 1

            # Tokenize the line content
            tokens = self._tokenize_line(stripped)

            # Check if this line is in bug region
            in_bug = (
                mark_bug_region is not None and
                mark_bug_region[0] <= line_idx <= mark_bug_region[2]
            )

            # Add bug start marker if applicable
            if in_bug and line_idx == mark_bug_region[0]:
                if col_idx < self.config.max_tokens_per_line:
                    grid[line_idx, col_idx] = self.config.bug_start_token
                    col_idx += 1

            # Add tokens
            for token in tokens:
                if col_idx >= self.config.max_tokens_per_line:
                    break

                token_id = self.vocab.get(token, self.config.unk_token)
                grid[line_idx, col_idx] = token_id
                col_idx += 1

            # Add bug end marker if applicable
            if in_bug and line_idx == mark_bug_region[2]:
                if col_idx < self.config.max_tokens_per_line:
                    grid[line_idx, col_idx] = self.config.bug_end_token
                    col_idx += 1

        return grid

    def encode_pair(
        self,
        buggy_code: str,
        fixed_code: str,
        bug_location: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a buggy/fixed code pair.

        Args:
            buggy_code: Code with bug
            fixed_code: Corrected code
            bug_location: Optional (line, column) of bug

        Returns:
            Tuple of (buggy_grid, fixed_grid)
        """
        # Determine bug region if location provided
        bug_region = None
        if bug_location is not None:
            line, col = bug_location
            # Mark single line for simplicity
            bug_region = (line - 1, col, line - 1, col + 10)

        buggy_grid = self.encode(buggy_code, mark_bug_region=bug_region)
        fixed_grid = self.encode(fixed_code)

        return buggy_grid, fixed_grid

    def decode(self, grid: np.ndarray) -> str:
        """
        Decode a grid back to code.

        Args:
            grid: numpy array of shape (max_lines, max_tokens_per_line)

        Returns:
            Reconstructed Python code
        """
        lines = []

        for line_idx in range(grid.shape[0]):
            line_tokens = []
            indent_count = 0

            for col_idx in range(grid.shape[1]):
                token_id = int(grid[line_idx, col_idx])

                # Handle special tokens
                if token_id == self.config.pad_token:
                    break
                elif token_id == self.config.indent_token:
                    indent_count += 1
                elif token_id == self.config.newline_token:
                    break
                elif token_id in (
                    self.config.bug_start_token,
                    self.config.bug_end_token,
                    self.config.fix_start_token,
                    self.config.fix_end_token,
                ):
                    # Skip markers
                    continue
                else:
                    token = self.id_to_token.get(token_id, '<UNK>')
                    line_tokens.append(token)

            # Reconstruct line
            if line_tokens or indent_count > 0:
                indent = '    ' * indent_count
                line = indent + ' '.join(line_tokens)
                lines.append(line)
            else:
                lines.append('')

        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        return '\n'.join(lines)

    def compute_diff_mask(
        self,
        buggy_grid: np.ndarray,
        fixed_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a mask indicating which positions differ.

        Args:
            buggy_grid: Grid of buggy code
            fixed_grid: Grid of fixed code

        Returns:
            Boolean mask where True indicates a difference
        """
        return buggy_grid != fixed_grid

    def get_diff_locations(
        self,
        buggy_grid: np.ndarray,
        fixed_grid: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Get list of (line, col) positions that differ.

        Args:
            buggy_grid: Grid of buggy code
            fixed_grid: Grid of fixed code

        Returns:
            List of (line, col) tuples where grids differ
        """
        diff_mask = self.compute_diff_mask(buggy_grid, fixed_grid)
        positions = np.where(diff_mask)
        return list(zip(positions[0].tolist(), positions[1].tolist()))

    def _tokenize_line(self, line: str) -> List[str]:
        """Simple tokenization of a line."""
        # Split on whitespace and common separators
        import re
        tokens = re.findall(
            r'\b\w+\b|[+\-*/=<>!&|^%@~]+|[(){}\[\],.:;\'"]',
            line
        )
        return tokens

    def get_statistics(self, grid: np.ndarray) -> Dict[str, int]:
        """
        Get statistics about a grid.

        Args:
            grid: Encoded grid

        Returns:
            Dictionary with statistics
        """
        non_pad = grid != self.config.pad_token
        non_special = ~np.isin(grid, [
            self.config.pad_token,
            self.config.indent_token,
            self.config.newline_token,
        ])

        return {
            'total_tokens': int(non_pad.sum()),
            'content_tokens': int(non_special.sum()),
            'num_lines': int((grid[:, 0] != self.config.pad_token).sum()),
            'avg_tokens_per_line': float(non_pad.sum(axis=1).mean()),
            'max_tokens_in_line': int(non_pad.sum(axis=1).max()),
        }


def create_default_encoder() -> GridEncoder:
    """Create an encoder with the default vocabulary."""
    # Minimal default vocabulary for testing
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<NEWLINE>': 2,
        '<INDENT>': 3,
        '<DEDENT>': 4,
        '<BOS>': 5,
        '<EOS>': 6,
        '<BUG_START>': 7,
        '<BUG_END>': 8,
        '<FIX_START>': 9,
        '<FIX_END>': 10,
    }

    # Add Python keywords
    keywords = [
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
        'while', 'with', 'yield',
    ]

    for kw in keywords:
        vocab[kw] = len(vocab)

    # Add common operators and punctuation
    operators = [
        '+', '-', '*', '/', '//', '**', '%', '@',
        '==', '!=', '<', '>', '<=', '>=',
        '=', '+=', '-=', '*=', '/=',
        '(', ')', '[', ']', '{', '}',
        ',', '.', ':', ';', '->', '=>',
        '&', '|', '^', '~', '<<', '>>',
    ]

    for op in operators:
        vocab[op] = len(vocab)

    # Add common builtins
    builtins = [
        'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
        'set', 'tuple', 'type', 'isinstance', 'open', 'read', 'write',
        'append', 'extend', 'pop', 'get', 'items', 'keys', 'values',
        'self', 'cls', '__init__', '__str__', '__repr__',
    ]

    for b in builtins:
        vocab[b] = len(vocab)

    return GridEncoder(vocab)
