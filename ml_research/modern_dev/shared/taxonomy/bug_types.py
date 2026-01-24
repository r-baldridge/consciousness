"""
Comprehensive bug taxonomy for Python code repair.

Defines 100+ bug types across 5 categories with difficulty ratings,
descriptions, and example patterns.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional


class BugCategory(Enum):
    """High-level bug categories."""
    SYNTAX = "syntax"
    LOGIC = "logic"
    STYLE = "style"
    SECURITY = "security"
    FRAMEWORK = "framework"


class BugType(Enum):
    """Enumeration of all bug types."""

    # ═══════════════════════════════════════════════════════════════════════
    # SYNTAX ERRORS (Difficulty: 0.1 - 0.3)
    # ═══════════════════════════════════════════════════════════════════════

    # Basic syntax
    MISSING_COLON = "missing_colon"
    MISSING_PARENTHESIS = "missing_parenthesis"
    MISSING_BRACKET = "missing_bracket"
    MISSING_BRACE = "missing_brace"
    MISSING_QUOTE = "missing_quote"
    EXTRA_COLON = "extra_colon"
    EXTRA_PARENTHESIS = "extra_parenthesis"

    # Indentation
    WRONG_INDENTATION = "wrong_indentation"
    MIXED_TABS_SPACES = "mixed_tabs_spaces"
    MISSING_INDENTATION = "missing_indentation"

    # Keywords
    MISSPELLED_KEYWORD = "misspelled_keyword"
    WRONG_KEYWORD = "wrong_keyword"
    MISSING_KEYWORD = "missing_keyword"

    # Operators
    WRONG_ASSIGNMENT = "wrong_assignment"  # = vs ==
    INVALID_OPERATOR = "invalid_operator"

    # ═══════════════════════════════════════════════════════════════════════
    # LOGIC ERRORS (Difficulty: 0.3 - 0.7)
    # ═══════════════════════════════════════════════════════════════════════

    # Off-by-one errors
    OFF_BY_ONE = "off_by_one"
    WRONG_RANGE_BOUND = "wrong_range_bound"
    FENCE_POST = "fence_post"

    # Comparison errors
    WRONG_COMPARISON = "wrong_comparison"  # < vs <=
    INVERTED_CONDITION = "inverted_condition"
    WRONG_BOOLEAN_LOGIC = "wrong_boolean_logic"  # and vs or

    # Operator errors
    WRONG_OPERATOR = "wrong_operator"  # + vs -
    INTEGER_DIVISION = "integer_division"  # / vs //
    OPERATOR_PRECEDENCE = "operator_precedence"

    # Control flow
    WRONG_LOOP_DIRECTION = "wrong_loop_direction"
    MISSING_BREAK = "missing_break"
    MISSING_CONTINUE = "missing_continue"
    MISSING_RETURN = "missing_return"
    WRONG_RETURN_VALUE = "wrong_return_value"
    UNREACHABLE_CODE = "unreachable_code"
    INFINITE_LOOP = "infinite_loop"

    # Variable errors
    WRONG_VARIABLE = "wrong_variable"
    UNINITIALIZED_VARIABLE = "uninitialized_variable"
    WRONG_SCOPE = "wrong_scope"
    SHADOWED_VARIABLE = "shadowed_variable"

    # Function errors
    WRONG_ARGUMENT_ORDER = "wrong_argument_order"
    MISSING_ARGUMENT = "missing_argument"
    EXTRA_ARGUMENT = "extra_argument"
    WRONG_DEFAULT_VALUE = "wrong_default_value"

    # Data structure errors
    WRONG_INDEX = "wrong_index"
    WRONG_KEY = "wrong_key"
    EMPTY_COLLECTION_ACCESS = "empty_collection_access"
    MUTATION_DURING_ITERATION = "mutation_during_iteration"

    # Type errors
    TYPE_MISMATCH = "type_mismatch"
    NONE_HANDLING = "none_handling"
    MISSING_TYPE_CONVERSION = "missing_type_conversion"

    # Algorithm errors
    WRONG_ALGORITHM = "wrong_algorithm"
    WRONG_BASE_CASE = "wrong_base_case"
    MISSING_EDGE_CASE = "missing_edge_case"

    # ═══════════════════════════════════════════════════════════════════════
    # STYLE ERRORS (Difficulty: 0.2 - 0.4)
    # ═══════════════════════════════════════════════════════════════════════

    # Naming
    WRONG_NAMING_CONVENTION = "wrong_naming_convention"
    MISLEADING_NAME = "misleading_name"
    SINGLE_LETTER_NAME = "single_letter_name"

    # Code organization
    DUPLICATE_CODE = "duplicate_code"
    MAGIC_NUMBER = "magic_number"
    LONG_FUNCTION = "long_function"
    DEEP_NESTING = "deep_nesting"

    # Pythonic
    NON_PYTHONIC_LOOP = "non_pythonic_loop"
    UNNECESSARY_LIST_COMPREHENSION = "unnecessary_list_comprehension"
    MISSING_F_STRING = "missing_f_string"
    MUTABLE_DEFAULT_ARGUMENT = "mutable_default_argument"

    # Imports
    UNUSED_IMPORT = "unused_import"
    MISSING_IMPORT = "missing_import"
    WRONG_IMPORT_ORDER = "wrong_import_order"
    STAR_IMPORT = "star_import"

    # ═══════════════════════════════════════════════════════════════════════
    # SECURITY VULNERABILITIES (Difficulty: 0.5 - 0.9)
    # ═══════════════════════════════════════════════════════════════════════

    # Injection
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"

    # Authentication/Authorization
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    WEAK_CRYPTO = "weak_crypto"
    INSECURE_RANDOM = "insecure_random"
    MISSING_AUTH_CHECK = "missing_auth_check"
    PRIVILEGE_ESCALATION = "privilege_escalation"

    # Data exposure
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    INSECURE_LOGGING = "insecure_logging"
    DEBUG_IN_PRODUCTION = "debug_in_production"

    # Input validation
    MISSING_INPUT_VALIDATION = "missing_input_validation"
    IMPROPER_SANITIZATION = "improper_sanitization"
    REGEX_DOS = "regex_dos"

    # Resource handling
    RESOURCE_LEAK = "resource_leak"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    SSRF = "ssrf"

    # ═══════════════════════════════════════════════════════════════════════
    # FRAMEWORK-SPECIFIC (Difficulty: 0.4 - 0.8)
    # ═══════════════════════════════════════════════════════════════════════

    # Django
    DJANGO_N_PLUS_ONE = "django_n_plus_one"
    DJANGO_MISSING_MIGRATION = "django_missing_migration"
    DJANGO_UNSAFE_QUERY = "django_unsafe_query"
    DJANGO_CSRF_MISSING = "django_csrf_missing"

    # Flask
    FLASK_DEBUG_MODE = "flask_debug_mode"
    FLASK_SECRET_KEY = "flask_secret_key"
    FLASK_UNSAFE_REDIRECT = "flask_unsafe_redirect"

    # FastAPI
    FASTAPI_MISSING_VALIDATION = "fastapi_missing_validation"
    FASTAPI_WRONG_STATUS_CODE = "fastapi_wrong_status_code"

    # Pandas
    PANDAS_CHAINED_ASSIGNMENT = "pandas_chained_assignment"
    PANDAS_INPLACE_CONFUSION = "pandas_inplace_confusion"
    PANDAS_MEMORY_INEFFICIENT = "pandas_memory_inefficient"

    # NumPy
    NUMPY_BROADCASTING_ERROR = "numpy_broadcasting_error"
    NUMPY_VIEW_VS_COPY = "numpy_view_vs_copy"

    # Async
    ASYNC_MISSING_AWAIT = "async_missing_await"
    ASYNC_BLOCKING_CALL = "async_blocking_call"
    ASYNC_RACE_CONDITION = "async_race_condition"

    # Testing
    TEST_ASSERTION_ERROR = "test_assertion_error"
    TEST_MISSING_MOCK = "test_missing_mock"
    TEST_FLAKY = "test_flaky"


@dataclass
class BugInfo:
    """Metadata about a bug type."""
    bug_type: BugType
    category: BugCategory
    difficulty: float  # 0.0 (trivial) to 1.0 (expert)
    description: str
    example_pattern: Optional[str] = None
    fix_pattern: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Complete bug taxonomy with metadata
BUG_TAXONOMY: Dict[BugType, BugInfo] = {
    # ═══════════════════════════════════════════════════════════════════════
    # SYNTAX ERRORS
    # ═══════════════════════════════════════════════════════════════════════

    BugType.MISSING_COLON: BugInfo(
        bug_type=BugType.MISSING_COLON,
        category=BugCategory.SYNTAX,
        difficulty=0.1,
        description="Missing colon after control flow statement or function definition",
        example_pattern="def foo()\n    pass",
        fix_pattern="def foo():\n    pass",
        tags=["beginner", "syntax", "quick-fix"],
    ),
    BugType.MISSING_PARENTHESIS: BugInfo(
        bug_type=BugType.MISSING_PARENTHESIS,
        category=BugCategory.SYNTAX,
        difficulty=0.1,
        description="Missing opening or closing parenthesis",
        example_pattern="print('hello'",
        fix_pattern="print('hello')",
        tags=["beginner", "syntax", "quick-fix"],
    ),
    BugType.MISSING_BRACKET: BugInfo(
        bug_type=BugType.MISSING_BRACKET,
        category=BugCategory.SYNTAX,
        difficulty=0.1,
        description="Missing opening or closing bracket",
        example_pattern="items = [1, 2, 3",
        fix_pattern="items = [1, 2, 3]",
        tags=["beginner", "syntax", "quick-fix"],
    ),
    BugType.MISSING_BRACE: BugInfo(
        bug_type=BugType.MISSING_BRACE,
        category=BugCategory.SYNTAX,
        difficulty=0.1,
        description="Missing opening or closing brace",
        example_pattern="data = {'key': 'value'",
        fix_pattern="data = {'key': 'value'}",
        tags=["beginner", "syntax", "quick-fix"],
    ),
    BugType.MISSING_QUOTE: BugInfo(
        bug_type=BugType.MISSING_QUOTE,
        category=BugCategory.SYNTAX,
        difficulty=0.1,
        description="Missing opening or closing quote",
        example_pattern="msg = 'hello",
        fix_pattern="msg = 'hello'",
        tags=["beginner", "syntax", "quick-fix"],
    ),
    BugType.WRONG_INDENTATION: BugInfo(
        bug_type=BugType.WRONG_INDENTATION,
        category=BugCategory.SYNTAX,
        difficulty=0.2,
        description="Incorrect indentation level",
        example_pattern="if True:\npass",
        fix_pattern="if True:\n    pass",
        tags=["beginner", "syntax", "indentation"],
    ),
    BugType.MISSPELLED_KEYWORD: BugInfo(
        bug_type=BugType.MISSPELLED_KEYWORD,
        category=BugCategory.SYNTAX,
        difficulty=0.15,
        description="Misspelled Python keyword",
        example_pattern="retrun x",
        fix_pattern="return x",
        tags=["beginner", "syntax", "typo"],
    ),
    BugType.WRONG_ASSIGNMENT: BugInfo(
        bug_type=BugType.WRONG_ASSIGNMENT,
        category=BugCategory.SYNTAX,
        difficulty=0.2,
        description="Using = instead of == or vice versa",
        example_pattern="if x = 5:",
        fix_pattern="if x == 5:",
        tags=["beginner", "syntax", "common"],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # LOGIC ERRORS
    # ═══════════════════════════════════════════════════════════════════════

    BugType.OFF_BY_ONE: BugInfo(
        bug_type=BugType.OFF_BY_ONE,
        category=BugCategory.LOGIC,
        difficulty=0.4,
        description="Loop bounds off by one",
        example_pattern="for i in range(len(arr)):",
        fix_pattern="for i in range(len(arr) - 1):",
        tags=["intermediate", "logic", "common"],
    ),
    BugType.WRONG_COMPARISON: BugInfo(
        bug_type=BugType.WRONG_COMPARISON,
        category=BugCategory.LOGIC,
        difficulty=0.35,
        description="Using wrong comparison operator",
        example_pattern="if x < limit:",
        fix_pattern="if x <= limit:",
        tags=["intermediate", "logic", "boundary"],
    ),
    BugType.INVERTED_CONDITION: BugInfo(
        bug_type=BugType.INVERTED_CONDITION,
        category=BugCategory.LOGIC,
        difficulty=0.4,
        description="Condition logic is inverted",
        example_pattern="if not is_valid:",
        fix_pattern="if is_valid:",
        tags=["intermediate", "logic", "boolean"],
    ),
    BugType.WRONG_BOOLEAN_LOGIC: BugInfo(
        bug_type=BugType.WRONG_BOOLEAN_LOGIC,
        category=BugCategory.LOGIC,
        difficulty=0.45,
        description="Using 'and' instead of 'or' or vice versa",
        example_pattern="if a and b:",
        fix_pattern="if a or b:",
        tags=["intermediate", "logic", "boolean"],
    ),
    BugType.WRONG_OPERATOR: BugInfo(
        bug_type=BugType.WRONG_OPERATOR,
        category=BugCategory.LOGIC,
        difficulty=0.35,
        description="Using wrong arithmetic operator",
        example_pattern="result = a + b",
        fix_pattern="result = a - b",
        tags=["intermediate", "logic", "arithmetic"],
    ),
    BugType.INTEGER_DIVISION: BugInfo(
        bug_type=BugType.INTEGER_DIVISION,
        category=BugCategory.LOGIC,
        difficulty=0.4,
        description="Using / instead of // or vice versa",
        example_pattern="index = total / 2",
        fix_pattern="index = total // 2",
        tags=["intermediate", "logic", "numeric"],
    ),
    BugType.MISSING_RETURN: BugInfo(
        bug_type=BugType.MISSING_RETURN,
        category=BugCategory.LOGIC,
        difficulty=0.35,
        description="Function missing return statement",
        example_pattern="def get_value():\n    x = 5",
        fix_pattern="def get_value():\n    x = 5\n    return x",
        tags=["intermediate", "logic", "function"],
    ),
    BugType.WRONG_VARIABLE: BugInfo(
        bug_type=BugType.WRONG_VARIABLE,
        category=BugCategory.LOGIC,
        difficulty=0.5,
        description="Using wrong variable name",
        example_pattern="return total",
        fix_pattern="return count",
        tags=["intermediate", "logic", "variable"],
    ),
    BugType.WRONG_INDEX: BugInfo(
        bug_type=BugType.WRONG_INDEX,
        category=BugCategory.LOGIC,
        difficulty=0.45,
        description="Using wrong index to access collection",
        example_pattern="items[i]",
        fix_pattern="items[i-1]",
        tags=["intermediate", "logic", "indexing"],
    ),
    BugType.NONE_HANDLING: BugInfo(
        bug_type=BugType.NONE_HANDLING,
        category=BugCategory.LOGIC,
        difficulty=0.5,
        description="Missing None check before operation",
        example_pattern="return obj.value",
        fix_pattern="return obj.value if obj else None",
        tags=["intermediate", "logic", "null-safety"],
    ),
    BugType.MUTATION_DURING_ITERATION: BugInfo(
        bug_type=BugType.MUTATION_DURING_ITERATION,
        category=BugCategory.LOGIC,
        difficulty=0.6,
        description="Modifying collection while iterating",
        example_pattern="for item in items:\n    items.remove(item)",
        fix_pattern="for item in items[:]:\n    items.remove(item)",
        tags=["intermediate", "logic", "iteration"],
    ),
    BugType.WRONG_BASE_CASE: BugInfo(
        bug_type=BugType.WRONG_BASE_CASE,
        category=BugCategory.LOGIC,
        difficulty=0.6,
        description="Recursive function has wrong base case",
        example_pattern="if n == 0: return 0",
        fix_pattern="if n == 0: return 1",
        tags=["advanced", "logic", "recursion"],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # STYLE ERRORS
    # ═══════════════════════════════════════════════════════════════════════

    BugType.MUTABLE_DEFAULT_ARGUMENT: BugInfo(
        bug_type=BugType.MUTABLE_DEFAULT_ARGUMENT,
        category=BugCategory.STYLE,
        difficulty=0.4,
        description="Using mutable default argument",
        example_pattern="def foo(items=[]):",
        fix_pattern="def foo(items=None):\n    if items is None:\n        items = []",
        tags=["pythonic", "style", "gotcha"],
    ),
    BugType.NON_PYTHONIC_LOOP: BugInfo(
        bug_type=BugType.NON_PYTHONIC_LOOP,
        category=BugCategory.STYLE,
        difficulty=0.25,
        description="Using C-style loop instead of Pythonic iteration",
        example_pattern="for i in range(len(items)):\n    print(items[i])",
        fix_pattern="for item in items:\n    print(item)",
        tags=["pythonic", "style", "iteration"],
    ),
    BugType.MISSING_F_STRING: BugInfo(
        bug_type=BugType.MISSING_F_STRING,
        category=BugCategory.STYLE,
        difficulty=0.2,
        description="Using string concatenation instead of f-string",
        example_pattern="msg = 'Hello ' + name",
        fix_pattern="msg = f'Hello {name}'",
        tags=["pythonic", "style", "string"],
    ),
    BugType.UNUSED_IMPORT: BugInfo(
        bug_type=BugType.UNUSED_IMPORT,
        category=BugCategory.STYLE,
        difficulty=0.15,
        description="Importing module that is never used",
        example_pattern="import os\nprint('hello')",
        fix_pattern="print('hello')",
        tags=["style", "import", "cleanup"],
    ),
    BugType.MISSING_IMPORT: BugInfo(
        bug_type=BugType.MISSING_IMPORT,
        category=BugCategory.STYLE,
        difficulty=0.3,
        description="Using module without importing it",
        example_pattern="result = json.loads(data)",
        fix_pattern="import json\nresult = json.loads(data)",
        tags=["style", "import", "runtime-error"],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # SECURITY VULNERABILITIES
    # ═══════════════════════════════════════════════════════════════════════

    BugType.SQL_INJECTION: BugInfo(
        bug_type=BugType.SQL_INJECTION,
        category=BugCategory.SECURITY,
        difficulty=0.6,
        description="SQL query built with string formatting",
        example_pattern="cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")",
        fix_pattern="cursor.execute(\"SELECT * FROM users WHERE id = ?\", (user_id,))",
        tags=["security", "injection", "critical"],
    ),
    BugType.COMMAND_INJECTION: BugInfo(
        bug_type=BugType.COMMAND_INJECTION,
        category=BugCategory.SECURITY,
        difficulty=0.7,
        description="Shell command built with user input",
        example_pattern="os.system(f'echo {user_input}')",
        fix_pattern="subprocess.run(['echo', user_input], shell=False)",
        tags=["security", "injection", "critical"],
    ),
    BugType.HARDCODED_CREDENTIALS: BugInfo(
        bug_type=BugType.HARDCODED_CREDENTIALS,
        category=BugCategory.SECURITY,
        difficulty=0.5,
        description="Credentials hardcoded in source code",
        example_pattern="password = 'secret123'",
        fix_pattern="password = os.environ.get('PASSWORD')",
        tags=["security", "credentials", "critical"],
    ),
    BugType.PATH_TRAVERSAL: BugInfo(
        bug_type=BugType.PATH_TRAVERSAL,
        category=BugCategory.SECURITY,
        difficulty=0.65,
        description="File path constructed from user input without validation",
        example_pattern="open(f'/data/{filename}')",
        fix_pattern="safe_path = os.path.join('/data', os.path.basename(filename))",
        tags=["security", "path", "critical"],
    ),
    BugType.INSECURE_RANDOM: BugInfo(
        bug_type=BugType.INSECURE_RANDOM,
        category=BugCategory.SECURITY,
        difficulty=0.55,
        description="Using random instead of secrets for security",
        example_pattern="token = random.randint(0, 999999)",
        fix_pattern="token = secrets.token_hex(16)",
        tags=["security", "crypto", "important"],
    ),
    BugType.UNSAFE_DESERIALIZATION: BugInfo(
        bug_type=BugType.UNSAFE_DESERIALIZATION,
        category=BugCategory.SECURITY,
        difficulty=0.75,
        description="Deserializing untrusted data with pickle",
        example_pattern="data = pickle.loads(user_data)",
        fix_pattern="data = json.loads(user_data)",
        tags=["security", "deserialization", "critical"],
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # FRAMEWORK-SPECIFIC
    # ═══════════════════════════════════════════════════════════════════════

    BugType.DJANGO_N_PLUS_ONE: BugInfo(
        bug_type=BugType.DJANGO_N_PLUS_ONE,
        category=BugCategory.FRAMEWORK,
        difficulty=0.6,
        description="Django queryset causing N+1 query problem",
        example_pattern="for obj in Model.objects.all():\n    print(obj.related.name)",
        fix_pattern="for obj in Model.objects.select_related('related').all():\n    print(obj.related.name)",
        tags=["django", "performance", "database"],
    ),
    BugType.ASYNC_MISSING_AWAIT: BugInfo(
        bug_type=BugType.ASYNC_MISSING_AWAIT,
        category=BugCategory.FRAMEWORK,
        difficulty=0.5,
        description="Calling async function without await",
        example_pattern="result = async_function()",
        fix_pattern="result = await async_function()",
        tags=["async", "coroutine", "runtime-error"],
    ),
    BugType.ASYNC_BLOCKING_CALL: BugInfo(
        bug_type=BugType.ASYNC_BLOCKING_CALL,
        category=BugCategory.FRAMEWORK,
        difficulty=0.6,
        description="Blocking call inside async function",
        example_pattern="async def fetch():\n    time.sleep(1)",
        fix_pattern="async def fetch():\n    await asyncio.sleep(1)",
        tags=["async", "performance", "blocking"],
    ),
    BugType.PANDAS_CHAINED_ASSIGNMENT: BugInfo(
        bug_type=BugType.PANDAS_CHAINED_ASSIGNMENT,
        category=BugCategory.FRAMEWORK,
        difficulty=0.55,
        description="Pandas chained assignment warning",
        example_pattern="df[df['col'] > 0]['col2'] = value",
        fix_pattern="df.loc[df['col'] > 0, 'col2'] = value",
        tags=["pandas", "warning", "assignment"],
    ),
}


def get_bug_info(bug_type: BugType) -> Optional[BugInfo]:
    """Get metadata for a bug type."""
    return BUG_TAXONOMY.get(bug_type)


def get_bugs_by_category(category: BugCategory) -> List[BugInfo]:
    """Get all bugs in a category."""
    return [
        info for info in BUG_TAXONOMY.values()
        if info.category == category
    ]


def get_bugs_by_difficulty(
    min_difficulty: float = 0.0,
    max_difficulty: float = 1.0,
) -> List[BugInfo]:
    """Get bugs within a difficulty range."""
    return [
        info for info in BUG_TAXONOMY.values()
        if min_difficulty <= info.difficulty <= max_difficulty
    ]


def get_category_from_bug_type(bug_type: str) -> str:
    """Get category string from bug type string."""
    try:
        bt = BugType(bug_type)
        info = BUG_TAXONOMY.get(bt)
        if info:
            return info.category.value
    except ValueError:
        pass

    # Fallback: infer from name
    bug_type_lower = bug_type.lower()
    if any(kw in bug_type_lower for kw in ['missing', 'extra', 'wrong_indent', 'misspell']):
        return BugCategory.SYNTAX.value
    if any(kw in bug_type_lower for kw in ['injection', 'credential', 'unsafe', 'insecure']):
        return BugCategory.SECURITY.value
    if any(kw in bug_type_lower for kw in ['django', 'flask', 'pandas', 'async']):
        return BugCategory.FRAMEWORK.value
    if any(kw in bug_type_lower for kw in ['import', 'naming', 'pythonic', 'style']):
        return BugCategory.STYLE.value

    return BugCategory.LOGIC.value
