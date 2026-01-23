"""
Python Code Tokenizer for TRM

Converts Python source code to token sequences and grid representations.
"""

import tokenize
import keyword
import builtins
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np


@dataclass
class CodeToken:
    """Represents a single token in Python code."""
    type: str           # Token type (NAME, NUMBER, OP, etc.)
    value: str          # Original string value
    token_id: int       # Vocabulary ID
    line: int           # Source line number (1-indexed)
    col: int            # Source column number (0-indexed)


# Special token IDs (0-15 reserved)
SPECIAL_TOKENS = {
    "PAD": 0,
    "UNK": 1,
    "MASK": 2,
    "BOS": 3,
    "EOS": 4,
    "NEWLINE": 5,
    "INDENT": 6,
    "DEDENT": 7,
    "ERROR": 8,
    "COMMENT": 9,
    "STRING": 10,
    "FSTRING": 11,
    "NUMBER": 12,
    "NAME": 13,         # Generic name (not in vocab)
    "EMPTY": 14,
    "CONTINUATION": 15,
}


def build_vocabulary(
    corpus_path: Optional[str] = None,
    vocab_size: int = 512,
    save_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Build token vocabulary for Python code.

    Args:
        corpus_path: Path to Python files for counting common names
        vocab_size: Maximum vocabulary size
        save_path: Path to save vocabulary JSON

    Returns:
        Dictionary mapping token strings to IDs
    """
    vocab = dict(SPECIAL_TOKENS)
    next_id = len(SPECIAL_TOKENS)

    # Python keywords (35 tokens)
    for kw in keyword.kwlist:
        if kw not in vocab:
            vocab[kw] = next_id
            next_id += 1

    # Soft keywords (Python 3.10+)
    soft_keywords = ["match", "case", "_", "type"]
    for kw in soft_keywords:
        if kw not in vocab:
            vocab[kw] = next_id
            next_id += 1

    # Operators
    operators = [
        # Arithmetic
        "+", "-", "*", "/", "//", "%", "**", "@",
        # Comparison
        "==", "!=", "<", ">", "<=", ">=",
        # Assignment
        "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=",
        "&=", "|=", "^=", ">>=", "<<=", ":=",
        # Bitwise
        "&", "|", "^", "~", "<<", ">>",
        # Logical (already in keywords: and, or, not)
        # Membership (already in keywords: in, not in, is, is not)
        # Other
        "->", "...",
    ]
    for op in operators:
        if op not in vocab:
            vocab[op] = next_id
            next_id += 1

    # Delimiters
    delimiters = [
        "(", ")", "[", "]", "{", "}", ",", ":", ";", ".", "\\",
    ]
    for delim in delimiters:
        if delim not in vocab:
            vocab[delim] = next_id
            next_id += 1

    # Built-in functions
    builtin_names = [
        "abs", "aiter", "all", "any", "anext", "ascii", "bin", "bool",
        "breakpoint", "bytearray", "bytes", "callable", "chr", "classmethod",
        "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate",
        "eval", "exec", "filter", "float", "format", "frozenset", "getattr",
        "globals", "hasattr", "hash", "help", "hex", "id", "input", "int",
        "isinstance", "issubclass", "iter", "len", "list", "locals", "map",
        "max", "memoryview", "min", "next", "object", "oct", "open", "ord",
        "pow", "print", "property", "range", "repr", "reversed", "round",
        "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum",
        "super", "tuple", "type", "vars", "zip", "__import__",
    ]
    for name in builtin_names:
        if name not in vocab:
            vocab[name] = next_id
            next_id += 1

    # Built-in constants
    builtin_constants = ["True", "False", "None", "NotImplemented", "Ellipsis", "__debug__"]
    for const in builtin_constants:
        if const not in vocab:
            vocab[const] = next_id
            next_id += 1

    # Built-in exceptions
    exceptions = [
        "BaseException", "Exception", "ArithmeticError", "AssertionError",
        "AttributeError", "BlockingIOError", "BrokenPipeError", "BufferError",
        "BytesWarning", "ChildProcessError", "ConnectionError", "ConnectionAbortedError",
        "ConnectionRefusedError", "ConnectionResetError", "DeprecationWarning",
        "EOFError", "EnvironmentError", "FileExistsError", "FileNotFoundError",
        "FloatingPointError", "FutureWarning", "GeneratorExit", "IOError",
        "ImportError", "ImportWarning", "IndentationError", "IndexError",
        "InterruptedError", "IsADirectoryError", "KeyError", "KeyboardInterrupt",
        "LookupError", "MemoryError", "ModuleNotFoundError", "NameError",
        "NotADirectoryError", "NotImplementedError", "OSError", "OverflowError",
        "PendingDeprecationWarning", "PermissionError", "ProcessLookupError",
        "RecursionError", "ReferenceError", "ResourceWarning", "RuntimeError",
        "RuntimeWarning", "StopAsyncIteration", "StopIteration", "SyntaxError",
        "SyntaxWarning", "SystemError", "SystemExit", "TabError", "TimeoutError",
        "TypeError", "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError",
        "UnicodeError", "UnicodeTranslateError", "UnicodeWarning", "UserWarning",
        "ValueError", "Warning", "ZeroDivisionError",
    ]
    for exc in exceptions:
        if exc not in vocab and next_id < vocab_size:
            vocab[exc] = next_id
            next_id += 1

    # Type hints
    type_hints = [
        "int", "str", "float", "bool", "bytes", "List", "Dict", "Set",
        "Tuple", "Optional", "Union", "Any", "Callable", "Type", "Generic",
        "TypeVar", "Sequence", "Mapping", "Iterable", "Iterator", "Generator",
        "Coroutine", "AsyncIterator", "AsyncGenerator", "Awaitable", "Final",
        "Literal", "ClassVar", "Protocol", "TypedDict", "Self",
    ]
    for hint in type_hints:
        if hint not in vocab and next_id < vocab_size:
            vocab[hint] = next_id
            next_id += 1

    # Common identifiers
    common_names = [
        "self", "cls", "args", "kwargs", "result", "data", "value", "item",
        "items", "key", "keys", "values", "index", "name", "path", "file",
        "line", "text", "string", "number", "count", "size", "length",
        "start", "end", "first", "last", "new", "old", "current", "next",
        "prev", "temp", "tmp", "i", "j", "k", "n", "x", "y", "z", "a", "b", "c",
        "df", "ax", "fig", "model", "config", "options", "params", "settings",
        "response", "request", "url", "api", "client", "server", "db",
        "connection", "cursor", "query", "row", "column", "table",
        "user", "username", "password", "email", "token", "session",
        "error", "message", "status", "code", "output", "input",
        "func", "fn", "callback", "handler", "wrapper", "decorator",
        "logger", "log", "debug", "info", "warning", "exception",
        "test", "setUp", "tearDown", "mock", "patch", "assert",
        "__init__", "__str__", "__repr__", "__call__", "__enter__", "__exit__",
        "__getitem__", "__setitem__", "__len__", "__iter__", "__next__",
        "__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__",
        "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
        "__name__", "__main__", "__file__", "__doc__", "__class__",
    ]
    for name in common_names:
        if name not in vocab and next_id < vocab_size:
            vocab[name] = next_id
            next_id += 1

    # Common module names
    modules = [
        "os", "sys", "re", "json", "math", "random", "time", "datetime",
        "collections", "itertools", "functools", "typing", "pathlib",
        "logging", "unittest", "pytest", "argparse", "subprocess",
        "threading", "multiprocessing", "asyncio", "socket", "http",
        "urllib", "requests", "numpy", "pandas", "torch", "tensorflow",
        "flask", "django", "fastapi", "sqlalchemy", "pydantic",
    ]
    for mod in modules:
        if mod not in vocab and next_id < vocab_size:
            vocab[mod] = next_id
            next_id += 1

    # If corpus provided, add most common names from it
    if corpus_path and next_id < vocab_size:
        name_counts = _count_names_in_corpus(corpus_path)
        for name, _ in sorted(name_counts.items(), key=lambda x: -x[1]):
            if name not in vocab and next_id < vocab_size:
                if _is_valid_identifier(name):
                    vocab[name] = next_id
                    next_id += 1

    # Save vocabulary
    if save_path:
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)

    return vocab


def _count_names_in_corpus(corpus_path: str) -> Counter:
    """Count identifier frequencies in a corpus."""
    counts = Counter()
    corpus = Path(corpus_path)

    if corpus.is_file():
        files = [corpus]
    else:
        files = list(corpus.rglob("*.py"))

    for file_path in files[:1000]:  # Limit for speed
        try:
            code = file_path.read_text(encoding="utf-8", errors="ignore")
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type == tokenize.NAME:
                    counts[tok.string] += 1
        except Exception:
            continue

    return counts


def _is_valid_identifier(name: str) -> bool:
    """Check if name is a valid Python identifier."""
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))


class PythonTokenizer:
    """
    Tokenizer for Python code that produces TRM-compatible sequences.

    Features:
        - Preserves structure (NEWLINE, INDENT, DEDENT)
        - Handles syntax errors gracefully
        - Maps to fixed vocabulary
        - Encodes to 2D grid format
    """

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        max_lines: int = 64,
        max_tokens_per_line: int = 48,
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_path: Path to vocabulary JSON file
            vocab: Vocabulary dictionary (alternative to vocab_path)
            max_lines: Maximum lines in grid
            max_tokens_per_line: Maximum tokens per line
        """
        if vocab is not None:
            self.vocab = vocab
        elif vocab_path:
            with open(vocab_path) as f:
                self.vocab = json.load(f)
        else:
            self.vocab = build_vocabulary()

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_lines = max_lines
        self.max_tokens_per_line = max_tokens_per_line

    def tokenize(self, code: str) -> List[CodeToken]:
        """
        Tokenize Python code into CodeToken objects.

        Args:
            code: Python source code

        Returns:
            List of CodeToken objects
        """
        tokens = []

        try:
            # Try standard tokenization
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                token = self._process_token(tok)
                if token:
                    tokens.append(token)
        except tokenize.TokenError as e:
            # Handle incomplete code / syntax errors
            tokens = self._fallback_tokenize(code)

        return tokens

    def _process_token(self, tok) -> Optional[CodeToken]:
        """Process a single token from tokenize module."""
        tok_type = tokenize.tok_name[tok.type]
        tok_string = tok.string
        line, col = tok.start

        # Skip encoding and endmarker
        if tok_type in ("ENCODING", "ENDMARKER"):
            return None

        # Handle different token types
        if tok_type == "NEWLINE" or tok_type == "NL":
            return CodeToken("NEWLINE", "\\n", self.vocab["NEWLINE"], line, col)

        elif tok_type == "INDENT":
            return CodeToken("INDENT", "INDENT", self.vocab["INDENT"], line, col)

        elif tok_type == "DEDENT":
            return CodeToken("DEDENT", "DEDENT", self.vocab["DEDENT"], line, col)

        elif tok_type == "COMMENT":
            return CodeToken("COMMENT", tok_string, self.vocab["COMMENT"], line, col)

        elif tok_type == "STRING":
            # Check for f-string
            if tok_string.startswith(("f'", 'f"', "F'", 'F"')):
                token_id = self.vocab["FSTRING"]
            else:
                token_id = self.vocab["STRING"]
            return CodeToken("STRING", tok_string, token_id, line, col)

        elif tok_type == "NUMBER":
            return CodeToken("NUMBER", tok_string, self.vocab["NUMBER"], line, col)

        elif tok_type == "NAME":
            # Look up in vocabulary
            token_id = self.vocab.get(tok_string, self.vocab["NAME"])
            return CodeToken("NAME", tok_string, token_id, line, col)

        elif tok_type == "OP":
            # Look up operator
            token_id = self.vocab.get(tok_string, self.vocab["UNK"])
            return CodeToken("OP", tok_string, token_id, line, col)

        else:
            # Unknown token type
            return CodeToken(tok_type, tok_string, self.vocab["UNK"], line, col)

    def _fallback_tokenize(self, code: str) -> List[CodeToken]:
        """
        Fallback tokenization for code with syntax errors.

        Uses regex-based tokenization that's more permissive.
        """
        tokens = []
        lines = code.split("\n")

        # Token patterns
        patterns = [
            (r"#.*", "COMMENT"),
            (r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "STRING"),
            (r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'', "STRING"),
            (r"\b\d+\.?\d*(?:e[+-]?\d+)?\b", "NUMBER"),
            (r"[a-zA-Z_][a-zA-Z0-9_]*", "NAME"),
            (r"[+\-*/%@&|^~<>=!:;,.\[\](){}]+", "OP"),
        ]

        for line_num, line in enumerate(lines, 1):
            col = 0
            # Calculate indent
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                # Simplified: just track indent level
                if indent > 0 and col == 0:
                    tokens.append(CodeToken("INDENT", "INDENT", self.vocab["INDENT"], line_num, 0))

            remaining = stripped
            while remaining:
                matched = False
                for pattern, tok_type in patterns:
                    match = re.match(pattern, remaining)
                    if match:
                        value = match.group()
                        if tok_type == "NAME":
                            token_id = self.vocab.get(value, self.vocab["NAME"])
                        elif tok_type == "OP":
                            token_id = self.vocab.get(value, self.vocab["UNK"])
                        else:
                            token_id = self.vocab.get(tok_type, self.vocab["UNK"])

                        tokens.append(CodeToken(tok_type, value, token_id, line_num, col))
                        remaining = remaining[len(value):].lstrip()
                        col += len(value)
                        matched = True
                        break

                if not matched:
                    # Skip unrecognized character
                    remaining = remaining[1:]
                    col += 1

            # Add newline at end of line
            tokens.append(CodeToken("NEWLINE", "\\n", self.vocab["NEWLINE"], line_num, col))

        return tokens

    def encode(self, code: str) -> np.ndarray:
        """
        Encode Python code to a 2D grid of token IDs.

        Args:
            code: Python source code

        Returns:
            numpy array of shape (max_lines, max_tokens_per_line)
        """
        tokens = self.tokenize(code)
        grid = np.zeros((self.max_lines, self.max_tokens_per_line), dtype=np.int32)
        grid.fill(self.vocab["PAD"])

        line_idx = 0
        col_idx = 0

        for token in tokens:
            if token.type == "NEWLINE":
                line_idx += 1
                col_idx = 0
                if line_idx >= self.max_lines:
                    break
            else:
                if col_idx < self.max_tokens_per_line:
                    grid[line_idx, col_idx] = token.token_id
                    col_idx += 1

        return grid

    def decode(self, grid: np.ndarray) -> str:
        """
        Decode a token grid back to Python code.

        Args:
            grid: numpy array of token IDs

        Returns:
            Reconstructed Python code (approximate)
        """
        lines = []
        current_line = []
        indent_level = 0

        for row in grid:
            for token_id in row:
                if token_id == self.vocab["PAD"]:
                    continue
                elif token_id == self.vocab["NEWLINE"]:
                    # Finish current line
                    if current_line:
                        lines.append("    " * indent_level + " ".join(current_line))
                        current_line = []
                elif token_id == self.vocab["INDENT"]:
                    indent_level += 1
                elif token_id == self.vocab["DEDENT"]:
                    indent_level = max(0, indent_level - 1)
                else:
                    token_str = self.inv_vocab.get(token_id, "<UNK>")
                    # Skip special tokens in output
                    if not token_str.startswith("<"):
                        current_line.append(token_str)

            # End of row - if we have content, it's a line
            if current_line:
                lines.append("    " * indent_level + " ".join(current_line))
                current_line = []

        return "\n".join(lines)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def from_vocab_file(cls, path: str, **kwargs) -> "PythonTokenizer":
        """Create tokenizer from vocabulary file."""
        return cls(vocab_path=path, **kwargs)
