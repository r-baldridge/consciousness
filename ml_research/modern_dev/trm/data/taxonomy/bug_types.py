"""
Bug Taxonomy for Python Code Repair

Comprehensive classification of Python bugs organized into 5 tiers:
- Tier 1: Syntax Errors (easiest to detect/fix)
- Tier 2: Logic Errors (require understanding)
- Tier 3: Style/Best Practices (code quality)
- Tier 4: Security Issues (critical)
- Tier 5: Framework-Specific (domain knowledge)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


class BugCategory(Enum):
    """Top-level bug categories (Tiers)."""
    SYNTAX = "syntax"           # Tier 1: Parse errors
    LOGIC = "logic"             # Tier 2: Runtime/semantic errors
    STYLE = "style"             # Tier 3: Code quality
    SECURITY = "security"       # Tier 4: Vulnerabilities
    FRAMEWORK = "framework"     # Tier 5: Library-specific


class BugType(Enum):
    """Specific bug types within each category."""

    # Tier 1: Syntax Errors (20 types)
    MISSING_COLON = "missing_colon"
    MISSING_PARENTHESIS = "missing_parenthesis"
    MISSING_BRACKET = "missing_bracket"
    MISSING_QUOTE = "missing_quote"
    INDENTATION_ERROR = "indentation_error"
    INVALID_SYNTAX = "invalid_syntax"
    UNEXPECTED_INDENT = "unexpected_indent"
    MISSING_COMMA = "missing_comma"
    INVALID_ASSIGNMENT = "invalid_assignment"
    KEYWORD_AS_VARIABLE = "keyword_as_variable"
    UNCLOSED_STRING = "unclosed_string"
    MIXED_TABS_SPACES = "mixed_tabs_spaces"
    INVALID_ESCAPE = "invalid_escape"
    MISSING_CONTINUATION = "missing_continuation"
    INVALID_CHARACTER = "invalid_character"
    EOF_IN_STRING = "eof_in_string"
    INVALID_TOKEN = "invalid_token"
    DEDENT_MISMATCH = "dedent_mismatch"
    FSTRING_ERROR = "fstring_error"
    WALRUS_SYNTAX = "walrus_syntax"

    # Tier 2: Logic Errors (25 types)
    OFF_BY_ONE = "off_by_one"
    WRONG_OPERATOR = "wrong_operator"
    WRONG_COMPARISON = "wrong_comparison"
    NONE_CHECK_MISSING = "none_check_missing"
    TYPE_MISMATCH = "type_mismatch"
    UNDEFINED_VARIABLE = "undefined_variable"
    UNUSED_VARIABLE = "unused_variable"
    WRONG_RETURN = "wrong_return"
    MISSING_RETURN = "missing_return"
    INFINITE_LOOP = "infinite_loop"
    WRONG_CONDITION = "wrong_condition"
    BOUNDARY_ERROR = "boundary_error"
    EMPTY_SEQUENCE = "empty_sequence"
    MUTABLE_DEFAULT = "mutable_default"
    SHALLOW_COPY = "shallow_copy"
    WRONG_EXCEPTION = "wrong_exception"
    BARE_EXCEPT = "bare_except"
    EXCEPTION_SWALLOW = "exception_swallow"
    WRONG_ITERATION = "wrong_iteration"
    DICT_KEY_ERROR = "dict_key_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    CIRCULAR_IMPORT = "circular_import"
    SCOPE_ERROR = "scope_error"
    CLOSURE_ISSUE = "closure_issue"

    # Tier 3: Style/Best Practices (15 types)
    NAMING_CONVENTION = "naming_convention"
    LINE_TOO_LONG = "line_too_long"
    MISSING_DOCSTRING = "missing_docstring"
    COMPLEX_EXPRESSION = "complex_expression"
    DUPLICATE_CODE = "duplicate_code"
    MAGIC_NUMBER = "magic_number"
    HARDCODED_VALUE = "hardcoded_value"
    DEAD_CODE = "dead_code"
    UNUSED_IMPORT = "unused_import"
    WILDCARD_IMPORT = "wildcard_import"
    INCONSISTENT_RETURN = "inconsistent_return"
    GOD_FUNCTION = "god_function"
    DEEP_NESTING = "deep_nesting"
    TYPE_HINT_MISSING = "type_hint_missing"
    ANTIPATTERN = "antipattern"

    # Tier 4: Security Issues (15 types)
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    XSS_VULNERABILITY = "xss_vulnerability"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_CRYPTO = "weak_crypto"
    INSECURE_RANDOM = "insecure_random"
    DEBUG_ENABLED = "debug_enabled"
    ASSERT_SECURITY = "assert_security"
    EVAL_USAGE = "eval_usage"
    EXEC_USAGE = "exec_usage"
    PICKLE_UNTRUSTED = "pickle_untrusted"
    YAML_UNSAFE = "yaml_unsafe"
    SSRF_VULNERABILITY = "ssrf_vulnerability"

    # Tier 5: Framework-Specific (25 types)
    # Django
    DJANGO_ORM_N_PLUS_1 = "django_orm_n_plus_1"
    DJANGO_CSRF_MISSING = "django_csrf_missing"
    DJANGO_RAW_SQL = "django_raw_sql"
    DJANGO_DEBUG_TRUE = "django_debug_true"
    DJANGO_SECRET_KEY = "django_secret_key"

    # Flask
    FLASK_DEBUG_MODE = "flask_debug_mode"
    FLASK_SECRET_KEY = "flask_secret_key"
    FLASK_UNSAFE_REDIRECT = "flask_unsafe_redirect"

    # FastAPI
    FASTAPI_VALIDATION = "fastapi_validation"
    FASTAPI_ASYNC_BLOCKING = "fastapi_async_blocking"

    # Pandas
    PANDAS_CHAIN_ASSIGNMENT = "pandas_chain_assignment"
    PANDAS_INPLACE_ISSUE = "pandas_inplace_issue"
    PANDAS_MEMORY_LEAK = "pandas_memory_leak"
    PANDAS_SETTINGWITHCOPY = "pandas_settingwithcopy"

    # NumPy
    NUMPY_BROADCASTING = "numpy_broadcasting"
    NUMPY_COPY_VIEW = "numpy_copy_view"
    NUMPY_DTYPE_MISMATCH = "numpy_dtype_mismatch"

    # Async
    ASYNC_NOT_AWAITED = "async_not_awaited"
    ASYNC_BLOCKING_CALL = "async_blocking_call"
    ASYNC_RACE_CONDITION = "async_race_condition"
    ASYNC_DEADLOCK = "async_deadlock"

    # Testing
    TEST_ISOLATION = "test_isolation"
    TEST_FLAKY = "test_flaky"
    MOCK_INCORRECT = "mock_incorrect"
    FIXTURE_SCOPE = "fixture_scope"


@dataclass
class BugInfo:
    """Detailed information about a bug type."""
    bug_type: BugType
    category: BugCategory
    name: str
    description: str
    severity: int  # 1-5, 5 being most severe
    frequency: float  # 0-1, estimated occurrence rate
    difficulty: int  # 1-5, difficulty to fix
    examples: List[str] = field(default_factory=list)
    fix_patterns: List[str] = field(default_factory=list)
    related_bugs: List[BugType] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)  # Tools that can detect


# Complete bug taxonomy with metadata
BUG_TAXONOMY: Dict[BugType, BugInfo] = {
    # ===== Tier 1: Syntax Errors =====
    BugType.MISSING_COLON: BugInfo(
        bug_type=BugType.MISSING_COLON,
        category=BugCategory.SYNTAX,
        name="Missing Colon",
        description="Missing colon after if/for/while/def/class/try/except/with statements",
        severity=1,
        frequency=0.15,
        difficulty=1,
        examples=[
            "if x > 0\n    print(x)",  # buggy
            "if x > 0:\n    print(x)",  # fixed
        ],
        fix_patterns=["Add colon at end of control flow statement"],
        tools=["python", "pylint", "pyflakes"],
    ),

    BugType.MISSING_PARENTHESIS: BugInfo(
        bug_type=BugType.MISSING_PARENTHESIS,
        category=BugCategory.SYNTAX,
        name="Missing Parenthesis",
        description="Unbalanced parentheses in function calls or expressions",
        severity=1,
        frequency=0.12,
        difficulty=1,
        examples=[
            "print(foo(bar)",
            "print(foo(bar))",
        ],
        fix_patterns=["Balance parentheses"],
        tools=["python", "pylint"],
    ),

    BugType.INDENTATION_ERROR: BugInfo(
        bug_type=BugType.INDENTATION_ERROR,
        category=BugCategory.SYNTAX,
        name="Indentation Error",
        description="Incorrect indentation level",
        severity=1,
        frequency=0.20,
        difficulty=1,
        examples=[
            "def foo():\nprint('hello')",
            "def foo():\n    print('hello')",
        ],
        fix_patterns=["Correct indentation to 4 spaces"],
        tools=["python", "pylint", "black"],
    ),

    BugType.MISSING_COMMA: BugInfo(
        bug_type=BugType.MISSING_COMMA,
        category=BugCategory.SYNTAX,
        name="Missing Comma",
        description="Missing comma in list, dict, tuple, or function arguments",
        severity=1,
        frequency=0.08,
        difficulty=1,
        examples=[
            "items = ['a' 'b', 'c']",
            "items = ['a', 'b', 'c']",
        ],
        fix_patterns=["Add comma between elements"],
        tools=["python", "pylint"],
    ),

    BugType.FSTRING_ERROR: BugInfo(
        bug_type=BugType.FSTRING_ERROR,
        category=BugCategory.SYNTAX,
        name="F-string Error",
        description="Invalid f-string syntax or unescaped braces",
        severity=1,
        frequency=0.05,
        difficulty=2,
        examples=[
            'f"Value: {x:}"',
            'f"Value: {x}"',
        ],
        fix_patterns=["Fix f-string format specifier"],
        tools=["python", "pylint"],
    ),

    # ===== Tier 2: Logic Errors =====
    BugType.OFF_BY_ONE: BugInfo(
        bug_type=BugType.OFF_BY_ONE,
        category=BugCategory.LOGIC,
        name="Off-by-One Error",
        description="Loop or index boundary off by one",
        severity=3,
        frequency=0.18,
        difficulty=3,
        examples=[
            "for i in range(len(items)):\n    print(items[i+1])",
            "for i in range(len(items)-1):\n    print(items[i+1])",
        ],
        fix_patterns=["Adjust loop bounds", "Fix index calculation"],
        related_bugs=[BugType.BOUNDARY_ERROR],
        tools=["pytest", "hypothesis"],
    ),

    BugType.WRONG_OPERATOR: BugInfo(
        bug_type=BugType.WRONG_OPERATOR,
        category=BugCategory.LOGIC,
        name="Wrong Operator",
        description="Using wrong arithmetic or logical operator",
        severity=3,
        frequency=0.10,
        difficulty=2,
        examples=[
            "total = price + quantity",
            "total = price * quantity",
        ],
        fix_patterns=["Replace operator with correct one"],
        tools=["pytest", "mypy"],
    ),

    BugType.WRONG_COMPARISON: BugInfo(
        bug_type=BugType.WRONG_COMPARISON,
        category=BugCategory.LOGIC,
        name="Wrong Comparison",
        description="Using == instead of is, or wrong comparison operator",
        severity=2,
        frequency=0.08,
        difficulty=2,
        examples=[
            "if x == None:",
            "if x is None:",
        ],
        fix_patterns=["Use 'is' for None/True/False comparisons"],
        tools=["pylint", "flake8"],
    ),

    BugType.MUTABLE_DEFAULT: BugInfo(
        bug_type=BugType.MUTABLE_DEFAULT,
        category=BugCategory.LOGIC,
        name="Mutable Default Argument",
        description="Using mutable object as default function argument",
        severity=4,
        frequency=0.06,
        difficulty=3,
        examples=[
            "def add_item(item, items=[]):\n    items.append(item)",
            "def add_item(item, items=None):\n    if items is None:\n        items = []",
        ],
        fix_patterns=["Use None as default, create mutable in function body"],
        tools=["pylint", "flake8", "bandit"],
    ),

    BugType.NONE_CHECK_MISSING: BugInfo(
        bug_type=BugType.NONE_CHECK_MISSING,
        category=BugCategory.LOGIC,
        name="Missing None Check",
        description="Accessing attribute or method on potentially None value",
        severity=3,
        frequency=0.12,
        difficulty=2,
        examples=[
            "result = obj.method()",
            "result = obj.method() if obj is not None else None",
        ],
        fix_patterns=["Add None check before access"],
        tools=["mypy", "pyright"],
    ),

    BugType.SHALLOW_COPY: BugInfo(
        bug_type=BugType.SHALLOW_COPY,
        category=BugCategory.LOGIC,
        name="Shallow Copy Issue",
        description="Using shallow copy when deep copy is needed",
        severity=3,
        frequency=0.04,
        difficulty=3,
        examples=[
            "new_list = old_list.copy()",
            "import copy\nnew_list = copy.deepcopy(old_list)",
        ],
        fix_patterns=["Use copy.deepcopy for nested structures"],
        tools=["pylint"],
    ),

    BugType.BARE_EXCEPT: BugInfo(
        bug_type=BugType.BARE_EXCEPT,
        category=BugCategory.LOGIC,
        name="Bare Except Clause",
        description="Using bare except without specifying exception type",
        severity=3,
        frequency=0.07,
        difficulty=2,
        examples=[
            "try:\n    risky()\nexcept:",
            "try:\n    risky()\nexcept Exception as e:",
        ],
        fix_patterns=["Specify exception type"],
        tools=["pylint", "flake8", "bandit"],
    ),

    # ===== Tier 3: Style/Best Practices =====
    BugType.NAMING_CONVENTION: BugInfo(
        bug_type=BugType.NAMING_CONVENTION,
        category=BugCategory.STYLE,
        name="Naming Convention Violation",
        description="Variable/function/class names don't follow PEP 8",
        severity=1,
        frequency=0.25,
        difficulty=1,
        examples=[
            "def MyFunction():",
            "def my_function():",
        ],
        fix_patterns=["snake_case for functions/variables", "PascalCase for classes"],
        tools=["pylint", "flake8", "black"],
    ),

    BugType.UNUSED_IMPORT: BugInfo(
        bug_type=BugType.UNUSED_IMPORT,
        category=BugCategory.STYLE,
        name="Unused Import",
        description="Importing module that is never used",
        severity=1,
        frequency=0.15,
        difficulty=1,
        examples=[
            "import os\nimport sys\nprint('hello')",
            "print('hello')",
        ],
        fix_patterns=["Remove unused import"],
        tools=["pylint", "flake8", "autoflake"],
    ),

    BugType.MAGIC_NUMBER: BugInfo(
        bug_type=BugType.MAGIC_NUMBER,
        category=BugCategory.STYLE,
        name="Magic Number",
        description="Using unexplained numeric literal in code",
        severity=2,
        frequency=0.10,
        difficulty=1,
        examples=[
            "if count > 100:",
            "MAX_ITEMS = 100\nif count > MAX_ITEMS:",
        ],
        fix_patterns=["Extract to named constant"],
        tools=["pylint"],
    ),

    # ===== Tier 4: Security Issues =====
    BugType.SQL_INJECTION: BugInfo(
        bug_type=BugType.SQL_INJECTION,
        category=BugCategory.SECURITY,
        name="SQL Injection",
        description="String formatting in SQL queries allows injection",
        severity=5,
        frequency=0.03,
        difficulty=2,
        examples=[
            "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
            "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        ],
        fix_patterns=["Use parameterized queries"],
        tools=["bandit", "semgrep"],
    ),

    BugType.COMMAND_INJECTION: BugInfo(
        bug_type=BugType.COMMAND_INJECTION,
        category=BugCategory.SECURITY,
        name="Command Injection",
        description="Unsanitized input in shell command execution",
        severity=5,
        frequency=0.02,
        difficulty=2,
        examples=[
            "os.system(f'rm {filename}')",
            "subprocess.run(['rm', filename], check=True)",
        ],
        fix_patterns=["Use subprocess with list arguments", "Validate/sanitize input"],
        tools=["bandit", "semgrep"],
    ),

    BugType.HARDCODED_SECRET: BugInfo(
        bug_type=BugType.HARDCODED_SECRET,
        category=BugCategory.SECURITY,
        name="Hardcoded Secret",
        description="API keys, passwords, or secrets in source code",
        severity=5,
        frequency=0.04,
        difficulty=1,
        examples=[
            "API_KEY = 'sk-1234567890abcdef'",
            "API_KEY = os.environ.get('API_KEY')",
        ],
        fix_patterns=["Use environment variables", "Use secrets manager"],
        tools=["bandit", "gitleaks", "trufflehog"],
    ),

    BugType.EVAL_USAGE: BugInfo(
        bug_type=BugType.EVAL_USAGE,
        category=BugCategory.SECURITY,
        name="Unsafe eval() Usage",
        description="Using eval() with untrusted input",
        severity=5,
        frequency=0.02,
        difficulty=2,
        examples=[
            "result = eval(user_input)",
            "import ast\nresult = ast.literal_eval(user_input)",
        ],
        fix_patterns=["Use ast.literal_eval for literals", "Avoid eval entirely"],
        tools=["bandit", "semgrep"],
    ),

    # ===== Tier 5: Framework-Specific =====
    BugType.DJANGO_ORM_N_PLUS_1: BugInfo(
        bug_type=BugType.DJANGO_ORM_N_PLUS_1,
        category=BugCategory.FRAMEWORK,
        name="Django N+1 Query",
        description="Missing select_related/prefetch_related causing N+1 queries",
        severity=3,
        frequency=0.08,
        difficulty=3,
        examples=[
            "for book in Book.objects.all():\n    print(book.author.name)",
            "for book in Book.objects.select_related('author'):\n    print(book.author.name)",
        ],
        fix_patterns=["Use select_related for ForeignKey", "Use prefetch_related for M2M"],
        tools=["django-debug-toolbar", "nplusone"],
    ),

    BugType.PANDAS_SETTINGWITHCOPY: BugInfo(
        bug_type=BugType.PANDAS_SETTINGWITHCOPY,
        category=BugCategory.FRAMEWORK,
        name="Pandas SettingWithCopyWarning",
        description="Chained assignment may not work as expected",
        severity=3,
        frequency=0.10,
        difficulty=3,
        examples=[
            "df[df['a'] > 0]['b'] = 1",
            "df.loc[df['a'] > 0, 'b'] = 1",
        ],
        fix_patterns=["Use .loc for assignment"],
        tools=["pandas"],
    ),

    BugType.ASYNC_NOT_AWAITED: BugInfo(
        bug_type=BugType.ASYNC_NOT_AWAITED,
        category=BugCategory.FRAMEWORK,
        name="Coroutine Not Awaited",
        description="Async function called without await",
        severity=4,
        frequency=0.06,
        difficulty=2,
        examples=[
            "async def main():\n    fetch_data()",
            "async def main():\n    await fetch_data()",
        ],
        fix_patterns=["Add await before coroutine call"],
        tools=["pylint", "mypy", "pyright"],
    ),

    BugType.ASYNC_BLOCKING_CALL: BugInfo(
        bug_type=BugType.ASYNC_BLOCKING_CALL,
        category=BugCategory.FRAMEWORK,
        name="Blocking Call in Async",
        description="Using blocking I/O in async function",
        severity=3,
        frequency=0.05,
        difficulty=3,
        examples=[
            "async def read_file():\n    return open('file.txt').read()",
            "async def read_file():\n    async with aiofiles.open('file.txt') as f:\n        return await f.read()",
        ],
        fix_patterns=["Use async I/O libraries", "Run in thread pool"],
        tools=["pylint-aiohttp", "flake8-async"],
    ),
}


def get_bug_info(bug_type: BugType) -> Optional[BugInfo]:
    """Get detailed information about a bug type."""
    return BUG_TAXONOMY.get(bug_type)


def get_bugs_by_category(category: BugCategory) -> List[BugInfo]:
    """Get all bugs in a category."""
    return [info for info in BUG_TAXONOMY.values() if info.category == category]


def get_bugs_by_severity(min_severity: int = 1) -> List[BugInfo]:
    """Get bugs at or above a severity level."""
    return [info for info in BUG_TAXONOMY.values() if info.severity >= min_severity]


def get_high_frequency_bugs(threshold: float = 0.05) -> List[BugInfo]:
    """Get bugs above a frequency threshold."""
    return [info for info in BUG_TAXONOMY.values() if info.frequency >= threshold]
