# TRM Code Repair Dataset Specification

## Executive Summary

This document specifies the data acquisition, processing, and structure for training a TRM-based Python code repair system. The goal is to create a model that can:

1. **Detect** bugs and code issues in Python code
2. **Localize** the specific location of problems
3. **Fix** issues through iterative refinement
4. **Explain** the reasoning (via the z-state trajectory)

**Target**: 1M+ training pairs across 50+ bug categories

---

## Table of Contents

1. [Code-to-Grid Representation](#1-code-to-grid-representation)
2. [Bug Taxonomy](#2-bug-taxonomy)
3. [Dataset Sources](#3-dataset-sources)
4. [Data Collection Pipeline](#4-data-collection-pipeline)
5. [Preprocessing & Tokenization](#5-preprocessing--tokenization)
6. [Data Augmentation](#6-data-augmentation)
7. [Quality Assurance](#7-quality-assurance)
8. [Dataset Structure](#8-dataset-structure)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Phased Implementation](#10-phased-implementation)
11. [Tools & Infrastructure](#11-tools--infrastructure)
12. [Appendices](#appendices)

---

## 1. Code-to-Grid Representation

### 1.1 Primary Encoding: Token Grid

Convert Python code to a 2D grid of tokens:

```
Grid Dimensions: 64 lines × 48 tokens per line = 3,072 cells
Vocabulary Size: 512 tokens (covers 99%+ of Python code)
```

**Encoding Process:**
```python
# Original code
def add(a, b):
    return a + b

# Tokenized (using custom Python tokenizer)
['def', 'add', '(', 'a', ',', 'b', ')', ':', 'NEWLINE',
 'INDENT', 'return', 'a', '+', 'b', 'NEWLINE', 'DEDENT']

# Grid representation (64×48, padded)
[[DEF, NAME, LPAREN, NAME, COMMA, NAME, RPAREN, COLON, NEWLINE, PAD, ...],
 [INDENT, RETURN, NAME, PLUS, NAME, NEWLINE, DEDENT, PAD, PAD, PAD, ...],
 [PAD, PAD, PAD, ...],
 ...]
```

### 1.2 Token Vocabulary (512 tokens)

| Category | Count | Examples |
|----------|-------|----------|
| Keywords | 35 | `def`, `class`, `if`, `for`, `try`, `import`, ... |
| Operators | 45 | `+`, `-`, `==`, `!=`, `and`, `or`, `in`, ... |
| Delimiters | 20 | `(`, `)`, `[`, `]`, `{`, `}`, `:`, `,`, ... |
| Literals | 50 | `NUM_INT`, `NUM_FLOAT`, `STR`, `FSTR`, `TRUE`, `FALSE`, `NONE` |
| Built-ins | 70 | `len`, `range`, `print`, `list`, `dict`, `str`, ... |
| Common Names | 200 | `self`, `cls`, `args`, `kwargs`, `result`, `data`, `item`, ... |
| Special | 30 | `NEWLINE`, `INDENT`, `DEDENT`, `PAD`, `UNK`, `MASK`, ... |
| Type Hints | 40 | `int`, `str`, `List`, `Dict`, `Optional`, `Union`, ... |
| Exceptions | 22 | `Exception`, `ValueError`, `TypeError`, `KeyError`, ... |

### 1.3 Special Tokens

```python
SPECIAL_TOKENS = {
    0: 'PAD',      # Padding
    1: 'UNK',      # Unknown token
    2: 'MASK',     # Masked position (for MLM-style training)
    3: 'BOS',      # Beginning of sequence
    4: 'EOS',      # End of sequence
    5: 'NEWLINE',  # Line break
    6: 'INDENT',   # Indentation increase
    7: 'DEDENT',   # Indentation decrease
    8: 'ERROR',    # Syntax error marker
    9: 'FIX_START', # Start of fix region
    10: 'FIX_END',  # End of fix region
}
```

### 1.4 Alternative Encodings (for experimentation)

**A. Character-Level Grid**
```
Grid: 64 lines × 120 characters
Vocab: 128 (ASCII printable + special)
Pros: No tokenization errors, captures formatting
Cons: Longer sequences, less semantic
```

**B. AST-Based Grid**
```
Grid: AST nodes arranged by depth × breadth
Vocab: ~100 AST node types
Pros: Semantic structure preserved
Cons: Loses formatting, complex mapping
```

**C. Hybrid: Token + Position**
```
Each cell: (token_id, line_num, col_num, indent_level)
Allows model to reason about structure
```

---

## 2. Bug Taxonomy

### 2.1 Priority Tiers

**Tier 1: High Frequency, Clear Signal (Train First)**
| Bug Type | Frequency | Example |
|----------|-----------|---------|
| `SYNTAX_ERROR` | 25% | Missing colon, unmatched brackets |
| `NAME_ERROR` | 15% | Undefined variable, typo in name |
| `TYPE_ERROR` | 12% | Wrong argument type |
| `INDENTATION_ERROR` | 10% | Incorrect indent/dedent |
| `ATTRIBUTE_ERROR` | 8% | Non-existent attribute access |
| `INDEX_ERROR` | 6% | List index out of range |
| `KEY_ERROR` | 5% | Missing dictionary key |
| `IMPORT_ERROR` | 4% | Module not found, wrong import |

**Tier 2: Logic & Semantic Bugs**
| Bug Type | Frequency | Example |
|----------|-----------|---------|
| `OFF_BY_ONE` | Common | `range(n)` vs `range(n+1)` |
| `WRONG_OPERATOR` | Common | `=` vs `==`, `and` vs `or` |
| `WRONG_RETURN` | Common | Missing return, wrong value |
| `NONE_CHECK` | Common | Not checking for None |
| `EXCEPTION_HANDLING` | Moderate | Bare except, wrong exception type |
| `RESOURCE_LEAK` | Moderate | File not closed, connection leak |
| `INFINITE_LOOP` | Rare | Loop condition never false |
| `RACE_CONDITION` | Rare | Threading issues |

**Tier 3: Style & Best Practices**
| Bug Type | Frequency | Example |
|----------|-----------|---------|
| `UNUSED_VARIABLE` | Common | Assigned but never used |
| `UNUSED_IMPORT` | Common | Import but never used |
| `SHADOWING` | Moderate | Variable shadows built-in |
| `MUTABLE_DEFAULT` | Moderate | `def f(x=[]):` |
| `GLOBAL_USAGE` | Moderate | Unnecessary global |
| `MAGIC_NUMBER` | Common | Unexplained numeric literals |
| `COMPLEXITY` | Moderate | Overly complex expressions |

**Tier 4: Security Vulnerabilities**
| Bug Type | CWE ID | Example |
|----------|--------|---------|
| `SQL_INJECTION` | CWE-89 | String formatting in SQL |
| `COMMAND_INJECTION` | CWE-78 | `os.system(user_input)` |
| `PATH_TRAVERSAL` | CWE-22 | `open(user_path)` |
| `HARDCODED_SECRET` | CWE-798 | Passwords in code |
| `INSECURE_DESERIALIZE` | CWE-502 | `pickle.loads(untrusted)` |
| `WEAK_CRYPTO` | CWE-327 | MD5, SHA1 for passwords |
| `SSRF` | CWE-918 | Unvalidated URL fetch |
| `XSS` | CWE-79 | Unsanitized HTML output |

**Tier 5: API Misuse & Framework-Specific**
| Bug Type | Framework | Example |
|----------|-----------|---------|
| `DJANGO_N_PLUS_1` | Django | Query in loop |
| `FLASK_DEBUG_PROD` | Flask | Debug mode in production |
| `PANDAS_CHAINED_ASSIGN` | Pandas | Chained assignment warning |
| `NUMPY_BROADCAST_ERROR` | NumPy | Incompatible shapes |
| `ASYNCIO_BLOCKING` | asyncio | Blocking call in async |
| `REQUESTS_TIMEOUT` | requests | No timeout specified |

### 2.2 Bug Category Details

For each bug category, we define:

```yaml
bug_category:
  id: "NAME_ERROR_TYPO"
  parent: "NAME_ERROR"
  tier: 1
  description: "Variable name typo causing NameError"

  patterns:
    - pattern: "Common character swaps (teh -> the)"
    - pattern: "Missing/extra character"
    - pattern: "Case sensitivity (Data vs data)"
    - pattern: "Similar variable names confused"

  detection_signals:
    - "NameError in traceback"
    - "Levenshtein distance to valid name < 3"
    - "Similar name exists in scope"

  fix_strategies:
    - "Replace with closest valid name"
    - "Add missing definition"
    - "Fix import statement"

  training_data_sources:
    - "GitHub commits with 'typo' or 'fix name'"
    - "Synthetic: mutate valid code"
    - "Stack Overflow NameError questions"

  examples:
    - buggy: "print(resutl)"
      fixed: "print(result)"
      context: "result = compute()"

    - buggy: "for i in rnage(10):"
      fixed: "for i in range(10):"
```

---

## 3. Dataset Sources

### 3.1 Primary Sources

#### A. GitHub Bug-Fix Commits (Target: 500K pairs)

**Collection Strategy:**
```python
SEARCH_QUERIES = [
    # Direct fix mentions
    "fix bug language:python",
    "bugfix language:python",
    "fix typo language:python",
    "fix error language:python",
    "patch language:python",

    # Error type specific
    "fix NameError language:python",
    "fix TypeError language:python",
    "fix IndexError language:python",
    "fix KeyError language:python",
    "fix AttributeError language:python",

    # Semantic fixes
    "fix off by one language:python",
    "fix null check language:python",
    "fix return value language:python",
    "fix exception handling language:python",

    # Security fixes
    "fix sql injection language:python",
    "fix security vulnerability language:python",
    "fix xss language:python",
]

QUALITY_FILTERS = {
    "min_stars": 10,           # Repo quality signal
    "max_files_changed": 3,    # Focused fixes
    "max_lines_changed": 50,   # Small, focused changes
    "min_lines_changed": 1,    # Not just whitespace
    "has_tests": True,         # Verifiable
    "not_fork": True,          # Original repos
}
```

**Extraction Pipeline:**
```
1. Search GitHub API for matching commits
2. Filter by quality criteria
3. Clone repo at commit and commit~1
4. Extract changed Python files
5. Parse before/after pairs
6. Validate syntax (both must parse OR before has syntax error)
7. Classify bug type using heuristics
8. Store with metadata
```

#### B. GitHub Issues + PRs (Target: 200K pairs)

Issues with linked PRs provide:
- Bug description (natural language)
- Buggy code (issue)
- Fixed code (PR)
- Discussion (reasoning)

```python
ISSUE_LABELS = [
    "bug", "bugfix", "fix", "defect", "error",
    "type: bug", "kind/bug", "category: bug"
]
```

#### C. Static Analysis Tools Output (Target: 300K pairs)

Run linters on large codebases, collect issues and auto-fixes:

| Tool | Bug Types | Auto-fix Available |
|------|-----------|-------------------|
| **pylint** | Style, errors, refactoring | Some |
| **flake8** | Style, complexity | Few |
| **mypy** | Type errors | No (but clear signals) |
| **bandit** | Security | No |
| **black** | Formatting | Yes (all) |
| **isort** | Import ordering | Yes (all) |
| **autoflake** | Unused imports/vars | Yes (all) |
| **pyupgrade** | Modernization | Yes (all) |
| **ruff** | All of above | Yes (many) |

**Pipeline:**
```bash
# For each Python file in corpus:
ruff check file.py --output-format=json > issues.json
ruff check file.py --fix --diff > fixes.diff
# Parse and pair issues with fixes
```

#### D. Stack Overflow (Target: 100K pairs)

**High-Value Tags:**
```python
SO_TAGS = [
    "python", "python-3.x",
    "pandas", "numpy", "django", "flask",
    "debugging", "error-handling",
    "nameerror", "typeerror", "syntaxerror"
]
```

**Extraction:**
1. Questions with code blocks showing errors
2. Accepted answers with corrected code
3. Parse before/after from Q&A
4. Use vote count as quality signal

#### E. Synthetic Generation (Target: 500K pairs)

**Mutation-Based Generation:**
```python
MUTATIONS = {
    # Syntax mutations
    "delete_colon": lambda code: remove_random_colon(code),
    "delete_bracket": lambda code: remove_random_bracket(code),
    "wrong_indent": lambda code: change_random_indent(code),

    # Name mutations
    "typo": lambda code: introduce_typo(code),
    "wrong_case": lambda code: change_case(code),
    "shadow_builtin": lambda code: shadow_random_builtin(code),

    # Operator mutations
    "swap_operator": lambda code: swap_comparison_op(code),
    "wrong_boolean": lambda code: swap_and_or(code),

    # Logic mutations
    "off_by_one": lambda code: mutate_range_bounds(code),
    "remove_return": lambda code: remove_random_return(code),
    "remove_none_check": lambda code: remove_if_none(code),

    # API mutations
    "wrong_method": lambda code: swap_similar_method(code),
    "wrong_arg_order": lambda code: swap_arguments(code),
}

def generate_synthetic_pair(clean_code: str) -> Tuple[str, str, str]:
    """Generate (buggy_code, clean_code, bug_type) triple."""
    mutation = random.choice(list(MUTATIONS.keys()))
    buggy_code = MUTATIONS[mutation](clean_code)
    return buggy_code, clean_code, mutation
```

**Source Code for Mutations:**
- Top 1000 PyPI packages
- Python standard library
- Popular GitHub repos (>1000 stars)
- Code from coding tutorials/courses

#### F. Curated High-Quality Sets (Target: 10K pairs)

Hand-curated examples for:
- Edge cases
- Complex multi-line fixes
- Security vulnerabilities
- Framework-specific patterns

Sources:
- Real-World Python Security Issues (CVE database)
- Django/Flask security advisories
- Published bug datasets (Defects4J Python equivalent)
- Academic datasets (ManySStuBs4J adapted for Python)

### 3.2 Source Priority Matrix

| Source | Volume | Quality | Diversity | Effort | Priority |
|--------|--------|---------|-----------|--------|----------|
| GitHub Commits | High | Medium | High | Medium | ⭐⭐⭐⭐⭐ |
| Synthetic | Very High | High | Medium | Low | ⭐⭐⭐⭐⭐ |
| Linter Outputs | High | High | Medium | Low | ⭐⭐⭐⭐ |
| GitHub Issues | Medium | High | High | High | ⭐⭐⭐⭐ |
| Stack Overflow | Medium | Medium | High | Medium | ⭐⭐⭐ |
| Curated | Low | Very High | High | Very High | ⭐⭐⭐ |

---

## 4. Data Collection Pipeline

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  GitHub  │  │  Stack   │  │  Linter  │  │Synthetic │        │
│  │   API    │  │ Overflow │  │  Output  │  │   Gen    │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       ▼             ▼             ▼             ▼               │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   RAW COLLECTOR                      │        │
│  │  • Rate limiting  • Deduplication  • Caching        │        │
│  └───────────────────────┬─────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   PARSER / VALIDATOR                 │        │
│  │  • Syntax check  • Extract before/after  • Validate │        │
│  └───────────────────────┬─────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   BUG CLASSIFIER                     │        │
│  │  • Rule-based  • Error message parsing  • ML tagger │        │
│  └───────────────────────┬─────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   TOKENIZER / ENCODER                │        │
│  │  • Python tokenizer  • Grid encoding  • Padding     │        │
│  └───────────────────────┬─────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   QUALITY FILTER                     │        │
│  │  • Dedup  • Length filter  • Complexity check       │        │
│  └───────────────────────┬─────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                   DATASET STORAGE                    │        │
│  │  • Parquet files  • Metadata DB  • Version control  │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Collection Scripts

**GitHub Commit Collector:**
```python
# tools/collectors/github_commits.py

import requests
from dataclasses import dataclass
from typing import Iterator, Optional
import time

@dataclass
class BugFixCommit:
    repo: str
    sha: str
    message: str
    files_changed: list
    before_code: str
    after_code: str
    timestamp: str
    author: str

class GitHubCommitCollector:
    """Collect bug-fix commits from GitHub."""

    def __init__(self, token: str, rate_limit: int = 30):
        self.token = token
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"token {token}"

    def search_commits(
        self,
        query: str,
        max_results: int = 1000,
    ) -> Iterator[BugFixCommit]:
        """Search for commits matching query."""
        # Implementation: paginated search, rate limiting
        ...

    def extract_diff(self, repo: str, sha: str) -> dict:
        """Extract before/after code from commit."""
        # Implementation: fetch commit, parse diff
        ...

    def filter_quality(self, commit: BugFixCommit) -> bool:
        """Apply quality filters."""
        # Implementation: check file count, line count, etc.
        ...
```

**Synthetic Generator:**
```python
# tools/generators/synthetic_bugs.py

import ast
import random
from typing import Tuple, Optional

class SyntheticBugGenerator:
    """Generate synthetic buggy code from clean code."""

    MUTATION_WEIGHTS = {
        "typo": 0.20,
        "missing_colon": 0.10,
        "wrong_indent": 0.10,
        "off_by_one": 0.08,
        "wrong_operator": 0.08,
        "missing_return": 0.07,
        "wrong_method": 0.07,
        "none_check": 0.06,
        "wrong_exception": 0.05,
        "unused_variable": 0.05,
        "mutable_default": 0.04,
        "string_format": 0.04,
        "import_error": 0.03,
        "security_issue": 0.03,
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.typo_chars = self._load_typo_map()

    def generate(
        self,
        clean_code: str,
        bug_type: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Generate (buggy, clean, bug_type) triple."""

        if bug_type is None:
            bug_type = self._sample_bug_type()

        mutation_fn = getattr(self, f"_mutate_{bug_type}")
        buggy_code = mutation_fn(clean_code)

        return buggy_code, clean_code, bug_type

    def _mutate_typo(self, code: str) -> str:
        """Introduce realistic typo."""
        # Find variable names, introduce character swap/deletion
        ...

    def _mutate_off_by_one(self, code: str) -> str:
        """Mutate range/index boundaries."""
        # Find range() calls, +/- 1 from bounds
        ...

    # ... more mutation methods
```

### 4.3 Running Collection

```bash
# Collection commands
python -m trm.data.collect github --query "fix bug" --max-results 10000
python -m trm.data.collect stackoverflow --tags python,debugging --max-results 5000
python -m trm.data.collect synthetic --source pypi-top1000 --mutations 100000
python -m trm.data.collect linters --corpus ./python_corpus --tools ruff,mypy

# Combine and deduplicate
python -m trm.data.merge --output ./dataset/combined.parquet

# Generate train/val/test splits
python -m trm.data.split --input ./dataset/combined.parquet --output ./dataset/
```

---

## 5. Preprocessing & Tokenization

### 5.1 Python Tokenizer

```python
# tools/tokenizer/python_tokenizer.py

import tokenize
import io
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class CodeToken:
    type: str      # Token type (NAME, NUMBER, OP, etc.)
    value: str     # Original string value
    token_id: int  # Vocabulary ID
    line: int      # Source line number
    col: int       # Source column number

class PythonCodeTokenizer:
    """Tokenize Python code for TRM grid encoding."""

    def __init__(self, vocab_path: str = "vocab.json"):
        self.vocab = self._load_vocab(vocab_path)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, code: str) -> List[CodeToken]:
        """Tokenize Python code."""
        tokens = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                token = self._process_token(tok)
                if token:
                    tokens.append(token)
        except tokenize.TokenError:
            # Handle syntax errors gracefully
            tokens = self._fallback_tokenize(code)

        return tokens

    def encode(self, code: str, max_lines: int = 64, max_cols: int = 48) -> np.ndarray:
        """Encode code to grid."""
        tokens = self.tokenize(code)
        grid = np.zeros((max_lines, max_cols), dtype=np.int32)
        grid.fill(self.vocab["PAD"])

        line_idx = 0
        col_idx = 0

        for token in tokens:
            if token.type == "NEWLINE":
                line_idx += 1
                col_idx = 0
                if line_idx >= max_lines:
                    break
            elif token.type == "INDENT":
                # Add indent token
                if col_idx < max_cols:
                    grid[line_idx, col_idx] = self.vocab["INDENT"]
                    col_idx += 1
            else:
                if col_idx < max_cols:
                    grid[line_idx, col_idx] = token.token_id
                    col_idx += 1

        return grid

    def decode(self, grid: np.ndarray) -> str:
        """Decode grid back to code."""
        lines = []
        current_line = []
        indent_level = 0

        for row in grid:
            for token_id in row:
                if token_id == self.vocab["PAD"]:
                    continue
                elif token_id == self.vocab["NEWLINE"]:
                    lines.append("    " * indent_level + " ".join(current_line))
                    current_line = []
                elif token_id == self.vocab["INDENT"]:
                    indent_level += 1
                elif token_id == self.vocab["DEDENT"]:
                    indent_level = max(0, indent_level - 1)
                else:
                    current_line.append(self.inv_vocab.get(token_id, "<UNK>"))

        if current_line:
            lines.append("    " * indent_level + " ".join(current_line))

        return "\n".join(lines)
```

### 5.2 Vocabulary Construction

```python
# Build vocabulary from corpus
def build_vocabulary(corpus_path: str, vocab_size: int = 512) -> dict:
    """Build token vocabulary from code corpus."""

    # Reserved tokens (IDs 0-31)
    vocab = {
        "PAD": 0, "UNK": 1, "MASK": 2, "BOS": 3, "EOS": 4,
        "NEWLINE": 5, "INDENT": 6, "DEDENT": 7, "ERROR": 8,
        # ... more special tokens
    }

    # Python keywords (IDs 32-66)
    import keyword
    for i, kw in enumerate(keyword.kwlist):
        vocab[kw] = 32 + i

    # Operators and delimiters (IDs 67-120)
    operators = ['+', '-', '*', '/', '//', '%', '**', '==', '!=',
                 '<', '>', '<=', '>=', 'and', 'or', 'not', 'in',
                 'is', '&', '|', '^', '~', '<<', '>>', '=', '+=',
                 '-=', '*=', '/=', '//=', '%=', '**=', '&=', '|=',
                 '^=', '>>=', '<<=', '@', '@=', '->', ':=']
    delimiters = ['(', ')', '[', ']', '{', '}', ',', ':', '.', ';',
                  '...', '\\']
    for i, op in enumerate(operators + delimiters):
        vocab[op] = 67 + i

    # Built-in functions (IDs 121-190)
    builtins_list = ['abs', 'all', 'any', 'ascii', 'bin', 'bool',
                     'breakpoint', 'bytearray', 'bytes', 'callable',
                     'chr', 'classmethod', 'compile', 'complex',
                     'delattr', 'dict', 'dir', 'divmod', 'enumerate',
                     'eval', 'exec', 'filter', 'float', 'format',
                     'frozenset', 'getattr', 'globals', 'hasattr',
                     'hash', 'help', 'hex', 'id', 'input', 'int',
                     'isinstance', 'issubclass', 'iter', 'len',
                     'list', 'locals', 'map', 'max', 'memoryview',
                     'min', 'next', 'object', 'oct', 'open', 'ord',
                     'pow', 'print', 'property', 'range', 'repr',
                     'reversed', 'round', 'set', 'setattr', 'slice',
                     'sorted', 'staticmethod', 'str', 'sum', 'super',
                     'tuple', 'type', 'vars', 'zip', '__import__']
    for i, b in enumerate(builtins_list):
        vocab[b] = 121 + i

    # Common names from corpus (IDs 191-450)
    name_counts = count_names_in_corpus(corpus_path)
    common_names = sorted(name_counts.items(), key=lambda x: -x[1])
    for i, (name, _) in enumerate(common_names[:260]):
        if name not in vocab:
            vocab[name] = 191 + i

    # Type hints (IDs 451-490)
    type_hints = ['int', 'str', 'float', 'bool', 'bytes', 'None',
                  'List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union',
                  'Any', 'Callable', 'Type', 'Generic', 'TypeVar',
                  'Sequence', 'Mapping', 'Iterable', 'Iterator',
                  'Generator', 'Coroutine', 'AsyncIterator',
                  'AsyncGenerator', 'Awaitable', 'Final', 'Literal',
                  'ClassVar', 'Protocol', 'TypedDict']
    for i, th in enumerate(type_hints):
        if th not in vocab:
            vocab[th] = 451 + i

    # Exceptions (IDs 491-511)
    exceptions = ['BaseException', 'Exception', 'ArithmeticError',
                  'AssertionError', 'AttributeError', 'BlockingIOError',
                  'BrokenPipeError', 'BufferError', 'BytesWarning',
                  'ChildProcessError', 'ConnectionError', 'EOFError',
                  'FileExistsError', 'FileNotFoundError', 'FloatingPointError',
                  'ImportError', 'IndentationError', 'IndexError',
                  'KeyError', 'KeyboardInterrupt', 'LookupError',
                  'MemoryError', 'ModuleNotFoundError', 'NameError',
                  'NotImplementedError', 'OSError', 'OverflowError',
                  'PermissionError', 'RecursionError', 'ReferenceError',
                  'RuntimeError', 'StopIteration', 'SyntaxError',
                  'SystemError', 'SystemExit', 'TabError', 'TimeoutError',
                  'TypeError', 'UnboundLocalError', 'UnicodeError',
                  'ValueError', 'ZeroDivisionError']
    for i, ex in enumerate(exceptions[:21]):
        if ex not in vocab:
            vocab[ex] = 491 + i

    return vocab
```

### 5.3 Grid Encoding Examples

**Example 1: Simple Function**
```python
# Original code
def add(a, b):
    return a + b

# Token sequence
[def, add, (, a, ,, b, ), :, NEWLINE, INDENT, return, a, +, b, NEWLINE, DEDENT]

# Grid (showing first 2 rows)
Row 0: [DEF, NAME_add, LPAREN, NAME_a, COMMA, NAME_b, RPAREN, COLON, NEWLINE, PAD, ...]
Row 1: [INDENT, RETURN, NAME_a, PLUS, NAME_b, NEWLINE, DEDENT, PAD, PAD, PAD, ...]
```

**Example 2: Buggy Code**
```python
# Buggy code (missing colon)
def add(a, b)
    return a + b

# Grid (ERROR token marks the issue)
Row 0: [DEF, NAME_add, LPAREN, NAME_a, COMMA, NAME_b, RPAREN, ERROR, NEWLINE, PAD, ...]
Row 1: [INDENT, RETURN, NAME_a, PLUS, NAME_b, NEWLINE, DEDENT, PAD, PAD, PAD, ...]
```

---

## 6. Data Augmentation

### 6.1 Augmentation Strategies

**A. Identifier Renaming**
```python
def augment_rename(code: str, prob: float = 0.5) -> str:
    """Randomly rename variables while preserving semantics."""
    tree = ast.parse(code)
    names = collect_user_defined_names(tree)

    rename_map = {}
    for name in names:
        if random.random() < prob:
            rename_map[name] = generate_plausible_name(name)

    return apply_renames(code, rename_map)

# Example:
# Before: def calculate(x, y): return x + y
# After:  def compute(a, b): return a + b
```

**B. Code Style Variation**
```python
def augment_style(code: str) -> str:
    """Apply random style variations."""
    variations = [
        lambda c: add_type_hints(c),
        lambda c: remove_type_hints(c),
        lambda c: expand_comprehensions(c),
        lambda c: contract_to_comprehensions(c),
        lambda c: change_string_quotes(c),
        lambda c: add_docstrings(c),
        lambda c: change_import_style(c),
    ]

    for variation in random.sample(variations, k=random.randint(1, 3)):
        code = variation(code)

    return code
```

**C. Context Padding**
```python
def augment_context(buggy: str, fixed: str) -> Tuple[str, str]:
    """Add surrounding code context."""
    prefix = generate_random_imports()
    suffix = generate_random_function()

    return (
        prefix + "\n" + buggy + "\n" + suffix,
        prefix + "\n" + fixed + "\n" + suffix,
    )
```

**D. Difficulty Scaling**
```python
def augment_difficulty(buggy: str, fixed: str, level: int) -> Tuple[str, str]:
    """Scale difficulty by adding more issues."""
    if level == 1:
        return buggy, fixed  # Single bug

    # Add more bugs for higher difficulty
    for _ in range(level - 1):
        mutation = random.choice(MUTATIONS)
        buggy = mutation(buggy)

    return buggy, fixed
```

### 6.2 Augmentation Pipeline

```python
class AugmentationPipeline:
    """Chain of augmentations for training data."""

    def __init__(self, config: dict):
        self.config = config
        self.augmenters = [
            ("rename", self.rename, config.get("rename_prob", 0.3)),
            ("style", self.style_vary, config.get("style_prob", 0.2)),
            ("context", self.add_context, config.get("context_prob", 0.4)),
            ("whitespace", self.vary_whitespace, config.get("ws_prob", 0.5)),
        ]

    def __call__(self, buggy: str, fixed: str) -> Tuple[str, str]:
        for name, augmenter, prob in self.augmenters:
            if random.random() < prob:
                buggy, fixed = augmenter(buggy, fixed)
        return buggy, fixed
```

---

## 7. Quality Assurance

### 7.1 Validation Checks

```python
class DataValidator:
    """Validate training pairs."""

    def validate(self, buggy: str, fixed: str, bug_type: str) -> Tuple[bool, str]:
        """
        Validate a training pair.
        Returns (is_valid, reason).
        """
        checks = [
            self._check_syntax,
            self._check_difference,
            self._check_length,
            self._check_not_duplicate,
            self._check_semantic_validity,
        ]

        for check in checks:
            valid, reason = check(buggy, fixed, bug_type)
            if not valid:
                return False, reason

        return True, "passed"

    def _check_syntax(self, buggy: str, fixed: str, bug_type: str) -> Tuple[bool, str]:
        """Fixed code must parse (buggy may or may not)."""
        try:
            ast.parse(fixed)
        except SyntaxError:
            return False, "fixed code has syntax error"

        # For syntax bug types, buggy should NOT parse
        if bug_type in ["SYNTAX_ERROR", "INDENTATION_ERROR"]:
            try:
                ast.parse(buggy)
                return False, "syntax bug but buggy code parses"
            except SyntaxError:
                pass  # Expected

        return True, ""

    def _check_difference(self, buggy: str, fixed: str, bug_type: str) -> Tuple[bool, str]:
        """Buggy and fixed must be different."""
        if buggy.strip() == fixed.strip():
            return False, "buggy and fixed are identical"

        # But not TOO different (suggests wrong pairing)
        similarity = difflib.SequenceMatcher(None, buggy, fixed).ratio()
        if similarity < 0.5:
            return False, f"too different (similarity={similarity:.2f})"

        return True, ""

    def _check_length(self, buggy: str, fixed: str, bug_type: str) -> Tuple[bool, str]:
        """Check length constraints."""
        max_lines = 64
        max_line_length = 200

        for code, name in [(buggy, "buggy"), (fixed, "fixed")]:
            lines = code.split("\n")
            if len(lines) > max_lines:
                return False, f"{name} has {len(lines)} lines (max {max_lines})"
            if any(len(line) > max_line_length for line in lines):
                return False, f"{name} has line > {max_line_length} chars"

        return True, ""
```

### 7.2 Deduplication

```python
class Deduplicator:
    """Remove duplicate and near-duplicate pairs."""

    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self.seen_hashes = set()
        self.minhash_index = MinHashLSH(threshold=0.9, num_perm=128)

    def is_duplicate(self, buggy: str, fixed: str) -> bool:
        """Check if this pair is a duplicate."""
        # Exact duplicate check
        pair_hash = hashlib.md5(f"{buggy}|||{fixed}".encode()).hexdigest()
        if pair_hash in self.seen_hashes:
            return True

        # Near-duplicate check using MinHash LSH
        combined = f"{buggy}\n{fixed}"
        minhash = self._compute_minhash(combined)

        if self.minhash_index.query(minhash):
            return True

        # Not a duplicate - add to index
        self.seen_hashes.add(pair_hash)
        self.minhash_index.insert(pair_hash, minhash)

        return False
```

### 7.3 Quality Metrics

Track quality metrics during collection:

```python
@dataclass
class DatasetQualityMetrics:
    total_collected: int = 0
    passed_validation: int = 0
    duplicates_removed: int = 0

    # By source
    by_source: Dict[str, int] = field(default_factory=dict)

    # By bug type
    by_bug_type: Dict[str, int] = field(default_factory=dict)

    # Validation failures
    failure_reasons: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return f"""
Dataset Quality Summary:
  Total collected: {self.total_collected:,}
  Passed validation: {self.passed_validation:,} ({100*self.passed_validation/self.total_collected:.1f}%)
  Duplicates removed: {self.duplicates_removed:,}

  By Source:
    {chr(10).join(f'    {k}: {v:,}' for k, v in sorted(self.by_source.items(), key=lambda x: -x[1]))}

  By Bug Type:
    {chr(10).join(f'    {k}: {v:,}' for k, v in sorted(self.by_bug_type.items(), key=lambda x: -x[1])[:20])}
        """
```

---

## 8. Dataset Structure

### 8.1 File Format

**Primary Format: Parquet**
```
dataset/
├── train/
│   ├── part-00000.parquet  (100K samples each)
│   ├── part-00001.parquet
│   └── ...
├── validation/
│   └── validation.parquet  (50K samples)
├── test/
│   ├── test_in_domain.parquet  (10K samples)
│   └── test_out_domain.parquet (10K samples, different bug types)
├── metadata/
│   ├── vocab.json
│   ├── bug_taxonomy.json
│   ├── quality_report.json
│   └── statistics.json
└── raw/  (optional, for reproducibility)
    ├── github_commits.jsonl
    ├── stackoverflow.jsonl
    └── synthetic.jsonl
```

### 8.2 Schema

```python
# Parquet schema
SCHEMA = pa.schema([
    ("id", pa.string()),                    # Unique identifier
    ("buggy_code", pa.string()),            # Original buggy code
    ("fixed_code", pa.string()),            # Corrected code
    ("buggy_grid", pa.list_(pa.list_(pa.int32()))),  # Encoded grid (64x48)
    ("fixed_grid", pa.list_(pa.list_(pa.int32()))),  # Encoded grid (64x48)
    ("bug_type", pa.string()),              # Primary bug category
    ("bug_subtypes", pa.list_(pa.string())), # Additional tags
    ("source", pa.string()),                # Data source
    ("difficulty", pa.int32()),             # 1-5 difficulty rating
    ("metadata", pa.string()),              # JSON blob with extras
])

# Example record
{
    "id": "gh_commit_abc123_file1",
    "buggy_code": "def add(a, b)\n    return a + b",
    "fixed_code": "def add(a, b):\n    return a + b",
    "buggy_grid": [[10, 45, 20, ...], ...],  # 64x48 int array
    "fixed_grid": [[10, 45, 20, ...], ...],
    "bug_type": "SYNTAX_ERROR",
    "bug_subtypes": ["MISSING_COLON"],
    "source": "github_commits",
    "difficulty": 1,
    "metadata": "{\"repo\": \"user/repo\", \"sha\": \"abc123\", ...}"
}
```

### 8.3 Dataset Splits

| Split | Size | Purpose | Characteristics |
|-------|------|---------|-----------------|
| **Train** | 900K | Model training | All bug types, all sources |
| **Validation** | 50K | Hyperparameter tuning | Stratified sample |
| **Test (In-Domain)** | 25K | Primary evaluation | Same distribution as train |
| **Test (Out-Domain)** | 25K | Generalization testing | Held-out bug types, sources |

**Stratification:**
- Proportional representation of bug types
- Minimum 1000 samples per bug category
- Balanced across difficulty levels

---

## 9. Evaluation Framework

### 9.1 Metrics

**Primary Metrics:**
```python
@dataclass
class EvaluationMetrics:
    # Exact match
    exact_match: float          # Fixed code exactly matches prediction

    # Token-level
    token_accuracy: float       # % tokens correctly predicted
    token_f1: float            # F1 score for changed tokens

    # Semantic
    syntax_valid: float        # % predictions that parse
    tests_pass: float          # % that pass unit tests (if available)

    # Localization
    bug_localized: float       # Correctly identified bug location
    fix_localized: float       # Fix applied to correct location

    # By category
    by_bug_type: Dict[str, float]  # Accuracy per bug type
    by_difficulty: Dict[int, float]  # Accuracy per difficulty
```

**Evaluation Script:**
```python
def evaluate(model, test_dataset, max_samples=None):
    """Evaluate model on test dataset."""
    results = []

    for sample in tqdm(test_dataset):
        buggy_grid = sample["buggy_grid"]
        fixed_grid = sample["fixed_grid"]

        # Get model prediction
        with torch.no_grad():
            result = model.solve(
                torch.tensor(buggy_grid).unsqueeze(0),
                max_steps=16,
                return_trajectory=True
            )

        pred_grid = result["solution"].squeeze(0)

        # Compute metrics
        exact_match = (pred_grid == torch.tensor(fixed_grid)).all().item()
        token_accuracy = (pred_grid == torch.tensor(fixed_grid)).float().mean().item()

        # Decode and check syntax
        pred_code = tokenizer.decode(pred_grid.numpy())
        try:
            ast.parse(pred_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False

        results.append({
            "id": sample["id"],
            "exact_match": exact_match,
            "token_accuracy": token_accuracy,
            "syntax_valid": syntax_valid,
            "bug_type": sample["bug_type"],
            "difficulty": sample["difficulty"],
            "steps": result["steps"],
            "confidence": result["confidence"],
        })

    return aggregate_results(results)
```

### 9.2 Benchmark Suites

**Tier 1: Core Python Bugs**
- 5000 samples across Tier 1 bug categories
- Focus: Syntax, name, type, index, key errors

**Tier 2: Logic Bugs**
- 2500 samples of semantic bugs
- Focus: Off-by-one, wrong operator, missing return

**Tier 3: Security**
- 1000 samples from security bug categories
- Focus: Injection, traversal, secrets

**Tier 4: Framework-Specific**
- 1000 samples per major framework
- Focus: Django, Flask, Pandas, NumPy

### 9.3 Human Evaluation (Sample)

For 500 random samples, collect:
- Is the fix correct? (Yes/No/Partial)
- Is the fix minimal? (Yes/No)
- Would you accept this PR? (Yes/No)

---

## 10. Phased Implementation

### Phase 1: Foundation (Week 1-2)
**Goal**: Basic pipeline working end-to-end

- [ ] Implement Python tokenizer
- [ ] Build vocabulary (512 tokens)
- [ ] Create grid encoder/decoder
- [ ] Set up data storage (Parquet)
- [ ] Implement validation checks
- [ ] Create synthetic generator (5 mutation types)

**Deliverable**: 10K synthetic pairs, working tokenizer

### Phase 2: Core Dataset (Week 3-4)
**Goal**: 100K high-quality pairs

- [ ] GitHub commit collector
- [ ] Linter output collector (ruff, mypy)
- [ ] Bug classifier (rule-based)
- [ ] Deduplication pipeline
- [ ] Quality metrics dashboard

**Deliverable**: 100K validated pairs, 20 bug types

### Phase 3: Scale Up (Week 5-6)
**Goal**: 500K pairs with full taxonomy

- [ ] Stack Overflow collector
- [ ] Enhanced synthetic generator (all mutations)
- [ ] Augmentation pipeline
- [ ] Distributed collection (parallel processing)
- [ ] Full bug taxonomy implementation

**Deliverable**: 500K pairs, 50 bug types

### Phase 4: Refinement (Week 7-8)
**Goal**: 1M+ pairs, production-ready

- [ ] ML-based bug classifier
- [ ] Hard negative mining
- [ ] Curriculum learning splits
- [ ] Comprehensive test suites
- [ ] Documentation and tooling

**Deliverable**: Final dataset, evaluation framework

---

## 11. Tools & Infrastructure

### 11.1 Required Tools

```bash
# Core dependencies
pip install torch numpy pandas pyarrow
pip install GitPython PyGithub requests
pip install ruff mypy pylint bandit
pip install tqdm rich typer

# Optional
pip install datasketch  # MinHash deduplication
pip install tree-sitter tree-sitter-python  # Better parsing
pip install transformers  # For comparison baselines
```

### 11.2 Directory Structure

```
trm/
├── data/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── github.py
│   │   ├── stackoverflow.py
│   │   ├── linters.py
│   │   └── synthetic.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── encoder.py
│   │   ├── validator.py
│   │   └── augmenter.py
│   ├── taxonomy/
│   │   ├── __init__.py
│   │   ├── bug_types.py
│   │   └── classifier.py
│   └── utils/
│       ├── __init__.py
│       ├── dedup.py
│       └── metrics.py
├── configs/
│   ├── collection.yaml
│   ├── taxonomy.yaml
│   └── vocabulary.json
└── scripts/
    ├── collect.py
    ├── process.py
    ├── validate.py
    └── evaluate.py
```

### 11.3 Configuration

```yaml
# configs/collection.yaml
github:
  token: ${GITHUB_TOKEN}
  rate_limit: 30  # requests per minute
  queries:
    - "fix bug language:python"
    - "bugfix language:python"
  filters:
    min_stars: 10
    max_files_changed: 3
    max_lines_changed: 50

synthetic:
  source_repos:
    - "python/cpython"
    - "django/django"
    - "pandas-dev/pandas"
  mutations_per_file: 10
  mutation_weights:
    typo: 0.20
    missing_colon: 0.10
    # ...

processing:
  max_lines: 64
  max_tokens_per_line: 48
  vocab_size: 512

augmentation:
  rename_prob: 0.3
  style_prob: 0.2
  context_prob: 0.4
```

---

## Appendices

### A. Full Bug Type Catalog

[See separate file: `taxonomy/bug_catalog.yaml`]

### B. Vocabulary File

[See separate file: `configs/vocabulary.json`]

### C. Example Training Pairs

**Example 1: Syntax Error (Missing Colon)**
```python
# Buggy
def calculate_sum(numbers)
    total = 0
    for num in numbers:
        total += num
    return total

# Fixed
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# Metadata
{
  "bug_type": "SYNTAX_ERROR",
  "bug_subtypes": ["MISSING_COLON"],
  "difficulty": 1,
  "source": "synthetic"
}
```

**Example 2: Logic Error (Off-by-One)**
```python
# Buggy
def get_last_n_items(items, n):
    return items[len(items)-n+1:]

# Fixed
def get_last_n_items(items, n):
    return items[len(items)-n:]

# Metadata
{
  "bug_type": "LOGIC_ERROR",
  "bug_subtypes": ["OFF_BY_ONE", "SLICE_BOUNDS"],
  "difficulty": 3,
  "source": "github_commits"
}
```

**Example 3: Security (SQL Injection)**
```python
# Buggy
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)

# Fixed
def get_user(username):
    query = "SELECT * FROM users WHERE name = ?"
    return db.execute(query, (username,))

# Metadata
{
  "bug_type": "SECURITY",
  "bug_subtypes": ["SQL_INJECTION", "CWE-89"],
  "difficulty": 4,
  "source": "curated"
}
```

### D. References

1. **Datasets**:
   - ManySStuBs4J: https://github.com/mast-group/manysstubs4j
   - Defects4J: https://github.com/rjust/defects4j
   - BugsInPy: https://github.com/soarsmu/BugsInPy
   - PyPIBugs: https://github.com/pypibugs/pypibugs

2. **Tools**:
   - Ruff: https://github.com/astral-sh/ruff
   - Mypy: https://github.com/python/mypy
   - Bandit: https://github.com/PyCQA/bandit

3. **Papers**:
   - "Learning to Fix Programs" (DeepFix)
   - "Neural Program Repair" (SequenceR)
   - "CURE: Code-Aware Neural Machine Translation for Automatic Program Repair"

---

*Document Version: 1.0*
*Last Updated: 2025-01-23*
*Author: ML Research Team*
