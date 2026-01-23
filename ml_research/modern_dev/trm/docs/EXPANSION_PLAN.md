# TRM Code Repair - Expansion Plan

## Current State Assessment

### What We Have (Proof of Concept)
- Basic bug taxonomy (~100 types defined, 12 implemented)
- Simple synthetic mutations (colon removal, operator swaps)
- 512-token vocabulary (too small)
- Single-file, function-level scope
- No real data sources (synthetic only)
- No semantic validation

### Critical Gaps for Real-World Use

| Gap | Current | Required |
|-----|---------|----------|
| Training samples | ~100s synthetic | 1M+ real + synthetic |
| Bug mutations | 12 basic | 100+ sophisticated |
| Vocabulary | 512 tokens | 32K-64K tokens |
| Code scope | Single function | Multi-file projects |
| Validation | Syntax only | Semantic + tests |
| Data sources | Synthetic only | GitHub, SO, linters |
| Framework support | None | Django, FastAPI, pandas, etc. |

---

## Phase 1: Real Data Sources (Critical)

### 1.1 GitHub Bug-Fix Mining
**Goal**: Extract 500K+ real bug-fix pairs from commit history

```
data/collectors/
├── github/
│   ├── __init__.py
│   ├── commit_miner.py      # Mine bug-fix commits
│   ├── diff_parser.py       # Parse unified diffs
│   ├── heuristics.py        # Identify bug-fix commits
│   ├── repository_filter.py # Quality repo selection
│   └── rate_limiter.py      # API rate limiting
```

**Implementation**:
- Use GitHub API + `pygit2` for commit analysis
- Heuristics for bug-fix identification:
  - Commit message patterns: "fix", "bug", "issue #", "closes #"
  - Small diffs (< 50 lines changed)
  - Single-file changes preferred
  - Test file additions alongside fixes
- Filter criteria:
  - Python 3.6+ syntax
  - Well-maintained repos (>100 stars, active)
  - Clear commit history
  - Permissive licenses (MIT, Apache, BSD)

**Target repos** (prioritized):
```python
PRIORITY_REPOS = [
    # Web frameworks
    "django/django",
    "pallets/flask",
    "tiangolo/fastapi",
    "encode/starlette",

    # Data science
    "pandas-dev/pandas",
    "numpy/numpy",
    "scikit-learn/scikit-learn",

    # Utilities
    "psf/requests",
    "aio-libs/aiohttp",
    "python-attrs/attrs",

    # CLI/DevTools
    "pytest-dev/pytest",
    "python/mypy",
    "PyCQA/pylint",
]
```

### 1.2 Stack Overflow Q&A Mining
**Goal**: Extract 100K+ question-answer code pairs

```
data/collectors/
├── stackoverflow/
│   ├── __init__.py
│   ├── api_client.py        # SO API wrapper
│   ├── code_extractor.py    # Extract code from posts
│   ├── pair_matcher.py      # Match questions to answers
│   └── quality_scorer.py    # Score answer quality
```

**Implementation**:
- Use Stack Exchange API
- Filter by:
  - Python tag
  - Accepted answer or high score (>10)
  - Code blocks in both Q and A
  - Error messages in question
- Extract pairs:
  - Question code (buggy) → Answer code (fixed)
  - Question + traceback → Working solution

### 1.3 Linter/Type Checker Integration
**Goal**: Generate 300K+ samples from static analysis

```
data/collectors/
├── static_analysis/
│   ├── __init__.py
│   ├── pylint_runner.py     # Run pylint, extract issues
│   ├── mypy_runner.py       # Run mypy, extract type errors
│   ├── ruff_runner.py       # Fast linting with ruff
│   ├── bandit_runner.py     # Security analysis
│   ├── autofix_mapper.py    # Map issues to fixes
│   └── rule_database.py     # Linter rule explanations
```

**Implementation**:
- Run linters on clean code
- Generate violations synthetically
- Map linter rules to bug types:
  ```python
  LINTER_BUG_MAPPING = {
      "E501": BugType.LINE_TOO_LONG,
      "W0612": BugType.UNUSED_VARIABLE,
      "E1101": BugType.ATTRIBUTE_ERROR,
      "C0111": BugType.MISSING_DOCSTRING,
      # ... 200+ mappings
  }
  ```

### 1.4 Issue Tracker Mining
**Goal**: Extract 50K+ bug reports with fixes

```
data/collectors/
├── issues/
│   ├── __init__.py
│   ├── github_issues.py     # Mine GitHub issues
│   ├── jira_connector.py    # Enterprise issue tracking
│   ├── bug_report_parser.py # Extract repro code
│   └── fix_linker.py        # Link issues to fix commits
```

---

## Phase 2: Sophisticated Bug Generation

### 2.1 Logic Bug Mutations (Priority)

Current logic mutations are trivial. Need:

```python
# data/collectors/synthetic/logic_mutations.py

class LogicBugMutator:
    """Sophisticated logic bug generation."""

    MUTATIONS = {
        # Control flow bugs
        "inverted_condition": InvertCondition(),
        "missing_else": RemoveElseBranch(),
        "wrong_loop_variable": SwapLoopVariables(),
        "early_return": InsertEarlyReturn(),
        "missing_break": RemoveBreakStatement(),
        "infinite_loop": CreateInfiniteLoop(),

        # Data handling bugs
        "off_by_one_advanced": OffByOneAdvanced(),
        "boundary_condition": BoundaryConditionError(),
        "null_dereference": RemoveNullCheck(),
        "uninitialized_variable": RemoveInitialization(),
        "wrong_default": ChangeDefaultValue(),
        "shallow_vs_deep_copy": ChangeCopyType(),

        # Collection bugs
        "modify_during_iteration": ModifyWhileIterating(),
        "wrong_collection_method": SwapCollectionMethods(),
        "index_out_of_bounds": CreateIndexError(),
        "key_error": CreateKeyError(),

        # Numeric bugs
        "integer_overflow": CreateOverflow(),
        "floating_point_comparison": FloatEqualityCheck(),
        "division_by_zero": CreateDivisionByZero(),
        "wrong_rounding": ChangeRoundingMethod(),

        # String bugs
        "encoding_error": CreateEncodingIssue(),
        "format_string_error": BreakFormatString(),
        "regex_error": BreakRegexPattern(),
    }
```

### 2.2 Async/Concurrency Bugs

```python
# data/collectors/synthetic/async_mutations.py

class AsyncBugMutator:
    """Async and concurrency bug generation."""

    MUTATIONS = {
        # Async basics
        "missing_await": RemoveAwait(),
        "await_in_sync": AddAwaitToSync(),
        "blocking_in_async": AddBlockingCall(),

        # Task management
        "uncancelled_task": RemoveTaskCancellation(),
        "missing_gather": RemoveAsyncioGather(),
        "wrong_event_loop": WrongEventLoopUsage(),

        # Resource management
        "unclosed_connection": RemoveAsyncClose(),
        "context_manager_missing": RemoveAsyncWith(),

        # Race conditions
        "race_condition": CreateRaceCondition(),
        "deadlock": CreateDeadlock(),
        "missing_lock": RemoveLockUsage(),
    }
```

### 2.3 Framework-Specific Bugs

```python
# data/collectors/synthetic/framework_mutations/

# Django
class DjangoBugMutator:
    MUTATIONS = {
        "n_plus_one": RemoveSelectRelated(),
        "missing_migration": BreakMigration(),
        "csrf_vulnerability": RemoveCSRFToken(),
        "raw_sql_injection": ConvertToRawSQL(),
        "wrong_queryset": SwapQuerySetMethods(),
        "missing_transaction": RemoveAtomicBlock(),
        "template_injection": CreateTemplateInjection(),
    }

# FastAPI
class FastAPIBugMutator:
    MUTATIONS = {
        "missing_validation": RemovePydanticModel(),
        "wrong_status_code": ChangeStatusCode(),
        "blocking_endpoint": AddBlockingToAsync(),
        "missing_dependency": RemoveDependsCall(),
        "cors_misconfiguration": BreakCORSConfig(),
    }

# Pandas
class PandasBugMutator:
    MUTATIONS = {
        "chained_assignment": CreateChainedAssignment(),
        "settingwithcopy": CreateSettingWithCopy(),
        "wrong_merge": SwapMergeType(),
        "memory_explosion": CreateMemoryLeak(),
        "dtype_mismatch": ChangeDtype(),
        "missing_na_handling": RemoveNAHandling(),
    }
```

### 2.4 Security Vulnerability Injection

```python
# data/collectors/synthetic/security_mutations.py

class SecurityBugMutator:
    """Generate security vulnerabilities for training."""

    # OWASP Top 10 coverage
    MUTATIONS = {
        # Injection
        "sql_injection": CreateSQLInjection(),
        "command_injection": CreateCommandInjection(),
        "ldap_injection": CreateLDAPInjection(),
        "xpath_injection": CreateXPathInjection(),

        # Authentication
        "weak_password_hash": WeakenPasswordHash(),
        "hardcoded_credentials": AddHardcodedCredentials(),
        "missing_auth_check": RemoveAuthCheck(),

        # Data exposure
        "sensitive_data_logging": AddSensitiveLogging(),
        "error_message_leak": AddVerboseErrors(),

        # Deserialization
        "unsafe_pickle": AddUnsafePickle(),
        "unsafe_yaml": AddUnsafeYAML(),
        "unsafe_eval": AddEvalUsage(),

        # Path traversal
        "path_traversal": CreatePathTraversal(),
        "symlink_attack": CreateSymlinkVulnerability(),
    }
```

---

## Phase 3: Vocabulary & Tokenization Expansion

### 3.1 Large Vocabulary (32K+ tokens)

Current 512 tokens is severely limiting. Need:

```python
# data/processors/vocabulary/

class VocabularyBuilder:
    """Build comprehensive Python vocabulary."""

    def __init__(self):
        self.target_size = 32768  # 32K tokens

    def build_from_corpus(self, corpus_path: Path) -> Vocabulary:
        """Build vocabulary from large Python corpus."""

        # Reserved tokens (0-1023)
        vocab = self._add_special_tokens()      # 0-31
        vocab.update(self._add_python_keywords()) # 32-127
        vocab.update(self._add_builtins())       # 128-511
        vocab.update(self._add_common_stdlib())  # 512-1023

        # Learned tokens from corpus (1024-32767)
        # Use BPE or WordPiece
        vocab.update(self._learn_subwords(corpus_path))

        return vocab

    def _add_common_stdlib(self) -> Dict[str, int]:
        """Add common stdlib names."""
        return {
            # os module
            "os.path", "os.environ", "os.listdir", "os.makedirs",
            # sys module
            "sys.argv", "sys.exit", "sys.path",
            # collections
            "defaultdict", "Counter", "deque", "OrderedDict",
            # itertools
            "chain", "combinations", "permutations", "groupby",
            # functools
            "partial", "lru_cache", "reduce", "wraps",
            # typing
            "Optional", "List", "Dict", "Tuple", "Union", "Any",
            "Callable", "TypeVar", "Generic", "Protocol",
            # pathlib
            "Path", "PurePath",
            # datetime
            "datetime", "timedelta", "timezone",
            # json
            "json.dumps", "json.loads",
            # re
            "re.match", "re.search", "re.sub", "re.compile",
            # ... 500+ more
        }
```

### 3.2 Context-Aware Tokenization

```python
# data/processors/tokenizer_v2.py

class ContextAwareTokenizer:
    """Tokenizer that understands Python semantics."""

    def tokenize(self, code: str) -> TokenizedCode:
        """Tokenize with full context."""

        # Parse AST for context
        tree = ast.parse(code)

        # Extract semantic information
        context = CodeContext(
            imports=self._extract_imports(tree),
            classes=self._extract_classes(tree),
            functions=self._extract_functions(tree),
            variables=self._extract_variables(tree),
        )

        # Tokenize with context awareness
        tokens = []
        for node in ast.walk(tree):
            token = self._tokenize_node(node, context)
            tokens.append(token)

        return TokenizedCode(tokens=tokens, context=context)

    def _tokenize_node(self, node: ast.AST, ctx: CodeContext) -> Token:
        """Context-aware node tokenization."""

        if isinstance(node, ast.Name):
            # Identify if it's a class, function, variable, import
            if node.id in ctx.classes:
                return Token(node.id, TokenType.CLASS_NAME)
            elif node.id in ctx.functions:
                return Token(node.id, TokenType.FUNCTION_NAME)
            elif node.id in ctx.imports:
                return Token(node.id, TokenType.IMPORT_NAME)
            else:
                return Token(node.id, TokenType.VARIABLE)
```

### 3.3 Subword Tokenization (BPE)

```python
# data/processors/bpe_tokenizer.py

class PythonBPETokenizer:
    """BPE tokenizer trained on Python code."""

    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def train(self, corpus: List[str]):
        """Train BPE on Python corpus."""
        # Pre-tokenize preserving Python structure
        # Learn merges
        # Build final vocabulary

    def encode(self, code: str) -> List[int]:
        """Encode code to token IDs."""

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to code."""
```

---

## Phase 4: Multi-File & Project Context

### 4.1 Project-Level Understanding

Current system only handles single functions. Real bugs often span:
- Multiple files
- Class hierarchies
- Import chains
- Configuration files

```python
# data/processors/project_context.py

@dataclass
class ProjectContext:
    """Full project context for bug understanding."""

    # File structure
    files: Dict[Path, ParsedFile]

    # Import graph
    import_graph: nx.DiGraph

    # Class hierarchy
    class_hierarchy: Dict[str, List[str]]

    # Function call graph
    call_graph: nx.DiGraph

    # Type information (from stubs or inference)
    type_info: Dict[str, TypeAnnotation]

    # Configuration
    config_files: Dict[str, Any]  # pyproject.toml, setup.py, etc.


class ProjectAnalyzer:
    """Analyze full Python projects."""

    def analyze(self, project_path: Path) -> ProjectContext:
        """Build full project context."""

        # Find all Python files
        files = self._find_python_files(project_path)

        # Parse each file
        parsed = {f: self._parse_file(f) for f in files}

        # Build import graph
        import_graph = self._build_import_graph(parsed)

        # Build class hierarchy
        class_hierarchy = self._build_class_hierarchy(parsed)

        # Build call graph
        call_graph = self._build_call_graph(parsed)

        # Extract type information
        type_info = self._run_type_inference(project_path)

        return ProjectContext(
            files=parsed,
            import_graph=import_graph,
            class_hierarchy=class_hierarchy,
            call_graph=call_graph,
            type_info=type_info,
        )
```

### 4.2 Multi-File Bug Representation

```python
# data/processors/multi_file_encoder.py

class MultiFileEncoder:
    """Encode multi-file bugs for TRM."""

    def encode_bug(
        self,
        buggy_files: Dict[Path, str],
        fixed_files: Dict[Path, str],
        context: ProjectContext,
    ) -> MultiFileGrid:
        """Encode a multi-file bug."""

        # Identify changed files
        changed = self._find_changed_files(buggy_files, fixed_files)

        # Encode each changed file
        file_grids = {}
        for path in changed:
            buggy_grid = self.single_encoder.encode(buggy_files[path])
            fixed_grid = self.single_encoder.encode(fixed_files[path])
            file_grids[path] = (buggy_grid, fixed_grid)

        # Encode relevant context files
        context_grids = self._encode_context(changed, context)

        return MultiFileGrid(
            changed_files=file_grids,
            context_files=context_grids,
            file_relationships=self._encode_relationships(changed, context),
        )
```

---

## Phase 5: Semantic Validation

### 5.1 Test Execution Validation

```python
# data/processors/test_validator.py

class TestValidator:
    """Validate fixes by running tests."""

    def __init__(self):
        self.sandbox = DockerSandbox()

    def validate(
        self,
        buggy_code: str,
        fixed_code: str,
        test_code: str,
    ) -> ValidationResult:
        """Validate that fix passes tests."""

        # Run tests with buggy code (should fail)
        buggy_result = self.sandbox.run_tests(buggy_code, test_code)
        if buggy_result.passed:
            return ValidationResult(
                valid=False,
                reason="Buggy code should fail tests but passed",
            )

        # Run tests with fixed code (should pass)
        fixed_result = self.sandbox.run_tests(fixed_code, test_code)
        if not fixed_result.passed:
            return ValidationResult(
                valid=False,
                reason=f"Fixed code fails tests: {fixed_result.error}",
            )

        return ValidationResult(valid=True)
```

### 5.2 Static Analysis Validation

```python
# data/processors/static_validator.py

class StaticValidator:
    """Validate using static analysis tools."""

    def validate(
        self,
        buggy_code: str,
        fixed_code: str,
        bug_type: BugType,
    ) -> ValidationResult:
        """Validate fix using static analysis."""

        # Run appropriate analyzer based on bug type
        if bug_type.category == BugCategory.TYPE:
            return self._validate_with_mypy(buggy_code, fixed_code)
        elif bug_type.category == BugCategory.STYLE:
            return self._validate_with_pylint(buggy_code, fixed_code)
        elif bug_type.category == BugCategory.SECURITY:
            return self._validate_with_bandit(buggy_code, fixed_code)
        else:
            return self._validate_with_all(buggy_code, fixed_code)

    def _validate_with_mypy(self, buggy: str, fixed: str) -> ValidationResult:
        """Validate type errors are fixed."""

        buggy_errors = self._run_mypy(buggy)
        fixed_errors = self._run_mypy(fixed)

        # Fixed should have fewer type errors
        if len(fixed_errors) >= len(buggy_errors):
            return ValidationResult(
                valid=False,
                reason="Fix does not reduce type errors",
            )

        return ValidationResult(valid=True)
```

### 5.3 Semantic Equivalence Checking

```python
# data/processors/semantic_validator.py

class SemanticValidator:
    """Check semantic equivalence of code transformations."""

    def validate_equivalence(
        self,
        original: str,
        transformed: str,
        test_inputs: List[Any],
    ) -> bool:
        """Check if transformation preserves semantics."""

        # Compile both versions
        original_fn = self._compile_function(original)
        transformed_fn = self._compile_function(transformed)

        # Test with various inputs
        for input_val in test_inputs:
            try:
                orig_result = original_fn(input_val)
                trans_result = transformed_fn(input_val)

                if orig_result != trans_result:
                    return False
            except Exception:
                # Different exception behavior is not equivalent
                return False

        return True
```

---

## Phase 6: Scale & Infrastructure

### 6.1 Distributed Processing

```python
# data/infrastructure/distributed.py

class DistributedPipeline:
    """Distributed data processing with Ray."""

    def __init__(self, num_workers: int = 32):
        ray.init()
        self.num_workers = num_workers

    def process_repositories(self, repos: List[str]) -> Dataset:
        """Process repositories in parallel."""

        # Distribute work
        futures = []
        for repo in repos:
            future = self._process_repo.remote(repo)
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        # Combine into dataset
        return self._combine_results(results)

    @ray.remote
    def _process_repo(self, repo: str) -> List[Sample]:
        """Process single repository (runs on worker)."""
        collector = GitHubCollector()
        return collector.collect(repo)
```

### 6.2 Data Storage

```python
# data/infrastructure/storage.py

class DatasetStorage:
    """Efficient storage for large datasets."""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save_shard(self, samples: List[Sample], shard_id: int):
        """Save a shard of data."""

        # Convert to Arrow format
        table = self._to_arrow(samples)

        # Save as Parquet with compression
        pq.write_table(
            table,
            self.base_path / f"shard_{shard_id:05d}.parquet",
            compression="zstd",
        )

    def load_dataset(self) -> Dataset:
        """Load full dataset."""
        return pq.read_table(self.base_path)

    def stream_dataset(self) -> Iterator[Sample]:
        """Stream dataset without loading all into memory."""
        for shard in sorted(self.base_path.glob("shard_*.parquet")):
            table = pq.read_table(shard)
            for row in table.to_pylist():
                yield Sample(**row)
```

### 6.3 Quality Pipeline

```python
# data/infrastructure/quality.py

class QualityPipeline:
    """Ensure dataset quality at scale."""

    def __init__(self):
        self.validators = [
            SyntaxValidator(),
            SemanticValidator(),
            DuplicateDetector(),
            DifficultyEstimator(),
            BalanceChecker(),
        ]

    def process(self, dataset: Dataset) -> QualityReport:
        """Run quality checks on dataset."""

        report = QualityReport()

        # Deduplicate
        dataset = self._deduplicate(dataset)
        report.duplicates_removed = ...

        # Validate each sample
        valid_samples = []
        for sample in dataset:
            if self._is_valid(sample):
                valid_samples.append(sample)
            else:
                report.rejected.append(sample)

        # Check balance
        report.bug_type_distribution = self._compute_distribution(valid_samples)

        # Estimate difficulty distribution
        report.difficulty_distribution = self._estimate_difficulties(valid_samples)

        return report

    def _deduplicate(self, dataset: Dataset) -> Dataset:
        """Remove near-duplicates using LSH."""
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=0.9, num_perm=128)

        unique_samples = []
        for sample in dataset:
            mh = self._compute_minhash(sample)

            # Check for duplicates
            if not lsh.query(mh):
                lsh.insert(sample.id, mh)
                unique_samples.append(sample)

        return unique_samples
```

---

## Phase 7: Training Data Curriculum

### 7.1 Difficulty Estimation

```python
# data/curriculum/difficulty.py

class DifficultyEstimator:
    """Estimate bug fix difficulty for curriculum learning."""

    FACTORS = {
        "lines_changed": 0.1,
        "tokens_changed": 0.15,
        "ast_depth": 0.1,
        "num_variables": 0.1,
        "cyclomatic_complexity": 0.15,
        "bug_type_difficulty": 0.2,
        "context_required": 0.2,
    }

    def estimate(self, sample: Sample) -> float:
        """Estimate difficulty score 0-1."""

        scores = {}

        # Lines changed
        scores["lines_changed"] = min(sample.lines_changed / 20, 1.0)

        # Tokens changed
        scores["tokens_changed"] = min(sample.tokens_changed / 50, 1.0)

        # AST depth of change
        scores["ast_depth"] = min(sample.ast_depth / 10, 1.0)

        # Number of variables involved
        scores["num_variables"] = min(sample.num_variables / 15, 1.0)

        # Cyclomatic complexity
        scores["cyclomatic_complexity"] = min(sample.complexity / 20, 1.0)

        # Bug type inherent difficulty
        scores["bug_type_difficulty"] = BUG_DIFFICULTY[sample.bug_type]

        # Context required (multi-file, imports, etc.)
        scores["context_required"] = sample.context_score

        # Weighted sum
        return sum(
            scores[k] * self.FACTORS[k]
            for k in self.FACTORS
        )
```

### 7.2 Curriculum Stages

```python
# data/curriculum/stages.py

CURRICULUM_STAGES = {
    "stage_1_syntax": {
        "difficulty_range": (0.0, 0.2),
        "bug_types": [
            BugType.MISSING_COLON,
            BugType.MISSING_PARENTHESIS,
            BugType.INDENTATION_ERROR,
            BugType.MISSING_COMMA,
        ],
        "max_lines": 20,
        "samples": 100_000,
    },

    "stage_2_simple_logic": {
        "difficulty_range": (0.2, 0.4),
        "bug_types": [
            BugType.OFF_BY_ONE,
            BugType.WRONG_OPERATOR,
            BugType.WRONG_COMPARISON,
            BugType.MISSING_RETURN,
        ],
        "max_lines": 50,
        "samples": 200_000,
    },

    "stage_3_complex_logic": {
        "difficulty_range": (0.4, 0.6),
        "bug_types": [
            BugType.MUTABLE_DEFAULT,
            BugType.SHALLOW_COPY,
            BugType.NONE_CHECK_MISSING,
            BugType.WRONG_EXCEPTION,
        ],
        "max_lines": 100,
        "samples": 200_000,
    },

    "stage_4_advanced": {
        "difficulty_range": (0.6, 0.8),
        "bug_types": [
            BugType.ASYNC_NOT_AWAITED,
            BugType.RACE_CONDITION,
            BugType.SQL_INJECTION,
            BugType.DJANGO_ORM_N_PLUS_1,
        ],
        "max_lines": 200,
        "samples": 300_000,
    },

    "stage_5_expert": {
        "difficulty_range": (0.8, 1.0),
        "bug_types": "all",
        "max_lines": None,
        "multi_file": True,
        "samples": 200_000,
    },
}
```

---

## Implementation Roadmap

### Milestone 1: Core Data Sources (4 weeks)
- [ ] GitHub commit mining infrastructure
- [ ] Stack Overflow API integration
- [ ] Linter integration (pylint, mypy, ruff)
- [ ] Initial data collection: 100K samples

### Milestone 2: Bug Generation (3 weeks)
- [ ] 50 additional logic bug mutations
- [ ] Async/concurrency bugs
- [ ] Security vulnerability injection
- [ ] Framework-specific bugs (Django, FastAPI, pandas)

### Milestone 3: Vocabulary & Tokenization (2 weeks)
- [ ] 32K BPE vocabulary trained on 10GB Python code
- [ ] Context-aware tokenizer
- [ ] Subword handling for identifiers

### Milestone 4: Validation (3 weeks)
- [ ] Docker sandbox for test execution
- [ ] Static analysis integration
- [ ] Semantic equivalence checking
- [ ] Deduplication pipeline

### Milestone 5: Scale (2 weeks)
- [ ] Distributed processing with Ray
- [ ] Parquet storage with streaming
- [ ] Quality pipeline

### Milestone 6: Curriculum (2 weeks)
- [ ] Difficulty estimation model
- [ ] Curriculum stage datasets
- [ ] Evaluation benchmarks

---

## Target Dataset Statistics

| Metric | Target |
|--------|--------|
| Total samples | 1.5M |
| Unique bug types | 100+ |
| Source: GitHub | 500K |
| Source: Synthetic | 500K |
| Source: Linters | 300K |
| Source: Stack Overflow | 100K |
| Source: Issues | 100K |
| Vocabulary size | 32K |
| Max code length | 64 lines / 2048 tokens |
| Multi-file samples | 50K |
| Validated by tests | 200K |
| Human-reviewed | 10K |

---

## Required Infrastructure

1. **Compute**
   - 32-core machine for processing
   - GPU cluster for training
   - Docker for sandboxed execution

2. **Storage**
   - 500GB for raw data
   - 100GB for processed dataset
   - S3/GCS for distributed access

3. **APIs**
   - GitHub API (with token for rate limits)
   - Stack Exchange API
   - Optional: BigQuery for GitHub Archive

4. **Tools**
   - Ray for distributed processing
   - Docker for sandboxing
   - Parquet/Arrow for storage
   - MinHash/LSH for deduplication
