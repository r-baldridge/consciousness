"""
Self-Debugging System for RLM Code Synthesis

Implements the self-debugging technique from "Self-Debugging: Teaching LLMs to Debug Their Own Code".
This module provides:
- ExecutionSandbox: Safe, sandboxed code execution with timeout handling
- ExecutionResult: Structured result from code execution
- ErrorAnalyzer: Classifies errors and suggests fix strategies
- SelfDebugger: Main class that iteratively debugs generated code

The self-debugging loop:
    1. Execute code in sandbox
    2. If error, analyze and classify it
    3. Generate fix based on error analysis
    4. Apply fix and re-execute
    5. Repeat until success or max attempts
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import traceback
import sys
import io
import re
import signal
import time
from contextlib import contextmanager


# =============================================================================
# ENUMS
# =============================================================================

class ErrorCategory(Enum):
    """Categories of errors that can occur during code execution."""
    SYNTAX = "syntax"           # SyntaxError, IndentationError
    TYPE = "type"               # TypeError
    NAME = "name"               # NameError
    VALUE = "value"             # ValueError
    INDEX = "index"             # IndexError
    KEY = "key"                 # KeyError
    ATTRIBUTE = "attribute"     # AttributeError
    IMPORT = "import"           # ImportError, ModuleNotFoundError
    LOGIC = "logic"             # Wrong output, assertion failures
    TIMEOUT = "timeout"         # Execution timeout
    MEMORY = "memory"           # MemoryError
    RECURSION = "recursion"     # RecursionError
    ZERO_DIVISION = "zero_division"  # ZeroDivisionError
    UNKNOWN = "unknown"         # Unclassified errors


class FixStrategyType(Enum):
    """Types of fix strategies that can be applied."""
    REPLACE_LINE = "replace_line"         # Replace a specific line
    INSERT_LINE = "insert_line"           # Insert a new line
    DELETE_LINE = "delete_line"           # Delete a line
    MODIFY_EXPRESSION = "modify_expression"  # Modify an expression
    ADD_IMPORT = "add_import"             # Add missing import
    FIX_INDENTATION = "fix_indentation"   # Fix indentation issues
    ADD_TYPE_CONVERSION = "add_type_conversion"  # Add type conversion
    ADD_NULL_CHECK = "add_null_check"     # Add null/None check
    ADD_BOUNDS_CHECK = "add_bounds_check"  # Add index bounds check
    WRAP_TRY_EXCEPT = "wrap_try_except"   # Wrap in try-except
    REGENERATE = "regenerate"             # Regenerate entire function


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExecutionResult:
    """
    Result from executing code in the sandbox.

    Attributes:
        success: Whether execution completed without errors
        output: The return value or result of execution
        stdout: Captured standard output
        stderr: Captured standard error
        error_type: Type of error if failed (e.g., "SyntaxError")
        error_message: Error message if failed
        traceback: Full traceback string if failed
        execution_time_ms: Time taken to execute in milliseconds
        namespace: Variables defined during execution (for inspection)
    """
    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    execution_time_ms: float = 0.0
    namespace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": str(self.output) if self.output is not None else None,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ErrorAnalysis:
    """
    Analysis of an error from code execution.

    Attributes:
        category: Classified error category
        error_type: Original error type name
        message: Error message
        line_number: Line number where error occurred (if determinable)
        column: Column number (if available)
        code_context: Lines of code around the error
        root_cause: Inferred root cause of the error
        suggested_fixes: List of suggested fixes in order of preference
    """
    category: ErrorCategory
    error_type: str
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_context: Optional[str] = None
    root_cause: str = ""
    suggested_fixes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "error_type": self.error_type,
            "message": self.message,
            "line_number": self.line_number,
            "column": self.column,
            "code_context": self.code_context,
            "root_cause": self.root_cause,
            "suggested_fixes": self.suggested_fixes,
        }


@dataclass
class FixStrategy:
    """
    A strategy for fixing an error.

    Attributes:
        strategy_type: Type of fix to apply
        target_line: Line number to modify (if applicable)
        suggestion: Description of the fix
        new_code: New code to use (for replace/insert strategies)
        confidence: Confidence that this fix will work (0.0-1.0)
        reasoning: Explanation of why this fix should work
    """
    strategy_type: FixStrategyType
    target_line: Optional[int] = None
    suggestion: str = ""
    new_code: Optional[str] = None
    confidence: float = 0.5
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_type": self.strategy_type.value,
            "target_line": self.target_line,
            "suggestion": self.suggestion,
            "new_code": self.new_code,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class FixAttempt:
    """
    Record of a single fix attempt.

    Attributes:
        attempt_num: Attempt number (1-indexed)
        code_before: Code before the fix
        code_after: Code after the fix
        error_before: Error analysis before the fix
        fix_applied: The fix strategy that was applied
        result: Execution result after applying the fix
        success: Whether this fix attempt succeeded
    """
    attempt_num: int
    code_before: str
    code_after: str
    error_before: Optional[ErrorAnalysis]
    fix_applied: FixStrategy
    result: ExecutionResult
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attempt_num": self.attempt_num,
            "code_before": self.code_before,
            "code_after": self.code_after,
            "error_before": self.error_before.to_dict() if self.error_before else None,
            "fix_applied": self.fix_applied.to_dict(),
            "result": self.result.to_dict(),
            "success": self.success,
        }


@dataclass
class DebugResult:
    """
    Final result of the self-debugging process.

    Attributes:
        original_code: The original code before any fixes
        final_code: The final code after all fixes
        success: Whether debugging ultimately succeeded
        attempts: Number of fix attempts made
        fix_history: History of all fix attempts
        all_tests_passed: Whether all provided test cases pass
        final_result: The final execution result
        debug_trace: Step-by-step trace of the debugging process
    """
    original_code: str
    final_code: str
    success: bool
    attempts: int
    fix_history: List[FixAttempt]
    all_tests_passed: bool
    final_result: Optional[ExecutionResult] = None
    debug_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_code": self.original_code,
            "final_code": self.final_code,
            "success": self.success,
            "attempts": self.attempts,
            "fix_history": [f.to_dict() for f in self.fix_history],
            "all_tests_passed": self.all_tests_passed,
            "final_result": self.final_result.to_dict() if self.final_result else None,
            "debug_trace": self.debug_trace,
        }


# =============================================================================
# TIMEOUT HANDLING
# =============================================================================

class TimeoutError(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def timeout_context(seconds: float):
    """
    Context manager for timing out code execution.

    Uses SIGALRM on Unix systems. On Windows, this is a no-op
    (timeout must be handled differently).

    Args:
        seconds: Maximum execution time in seconds
    """
    if sys.platform == 'win32':
        # Windows doesn't support SIGALRM, yield without timeout
        yield
        return

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)

    # Set the alarm (convert to integer seconds, minimum 1)
    signal.alarm(max(1, int(seconds)))

    try:
        yield
    finally:
        # Cancel the alarm and restore the old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# EXECUTION SANDBOX
# =============================================================================

class ExecutionSandbox:
    """
    Safe, sandboxed environment for executing generated code.

    Provides:
    - Restricted builtins (no file I/O, network, system access)
    - Timeout handling
    - Stdout/stderr capture
    - Memory limits (optional)

    Example:
        >>> sandbox = ExecutionSandbox()
        >>> result = sandbox.execute("print(2 + 2)")
        >>> print(result.stdout)  # "4\\n"
        >>> print(result.success)  # True
    """

    # Safe builtins that don't allow dangerous operations
    SAFE_BUILTINS = {
        # Type constructors
        'bool': bool,
        'int': int,
        'float': float,
        'str': str,
        'bytes': bytes,
        'bytearray': bytearray,
        'list': list,
        'tuple': tuple,
        'set': set,
        'frozenset': frozenset,
        'dict': dict,
        'complex': complex,
        'type': type,
        'object': object,

        # Built-in functions (safe subset)
        'abs': abs,
        'all': all,
        'any': any,
        'ascii': ascii,
        'bin': bin,
        'callable': callable,
        'chr': chr,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'format': format,
        'getattr': getattr,
        'hasattr': hasattr,
        'hash': hash,
        'hex': hex,
        'id': id,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'iter': iter,
        'len': len,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print,  # Will be redirected to capture stdout
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'slice': slice,
        'sorted': sorted,
        'sum': sum,
        'zip': zip,

        # Constants
        'True': True,
        'False': False,
        'None': None,

        # Exception types (for try/except)
        'Exception': Exception,
        'BaseException': BaseException,
        'TypeError': TypeError,
        'ValueError': ValueError,
        'IndexError': IndexError,
        'KeyError': KeyError,
        'AttributeError': AttributeError,
        'RuntimeError': RuntimeError,
        'StopIteration': StopIteration,
        'ZeroDivisionError': ZeroDivisionError,
        'AssertionError': AssertionError,
        'NameError': NameError,
        'ImportError': ImportError,
        'LookupError': LookupError,
        'ArithmeticError': ArithmeticError,
        'OverflowError': OverflowError,
        'RecursionError': RecursionError,
    }

    # Modules that are safe to import
    SAFE_MODULES = {
        'math', 'cmath', 'decimal', 'fractions', 'statistics',
        'random', 'itertools', 'functools', 'operator',
        'collections', 'heapq', 'bisect',
        'string', 're', 'json',
        'datetime', 'time', 'calendar',
        'copy', 'types', 'typing',
    }

    def __init__(
        self,
        timeout: float = 5.0,
        max_output_length: int = 10000,
        allow_imports: bool = True,
        additional_builtins: Optional[Dict[str, Any]] = None,
        additional_modules: Optional[set] = None,
    ):
        """
        Initialize the execution sandbox.

        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum length of captured stdout/stderr
            allow_imports: Whether to allow (safe) imports
            additional_builtins: Additional builtins to allow
            additional_modules: Additional modules to allow importing
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.allow_imports = allow_imports

        # Build builtins dict
        self._builtins = dict(self.SAFE_BUILTINS)
        if additional_builtins:
            self._builtins.update(additional_builtins)

        # Build allowed modules set
        self._allowed_modules = set(self.SAFE_MODULES)
        if additional_modules:
            self._allowed_modules.update(additional_modules)

    def _create_namespace(self) -> Dict[str, Any]:
        """Create a fresh execution namespace with safe builtins."""
        namespace = {"__builtins__": dict(self._builtins)}

        if self.allow_imports:
            # Add safe import function
            def safe_import(name, *args, **kwargs):
                if name not in self._allowed_modules:
                    raise ImportError(f"Import of '{name}' is not allowed in sandbox")
                return __import__(name, *args, **kwargs)

            namespace["__builtins__"]["__import__"] = safe_import

        return namespace

    def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
        namespace: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox and return the result.

        Args:
            code: Python code to execute
            timeout: Override default timeout (seconds)
            namespace: Pre-populated namespace (merged with safe builtins)

        Returns:
            ExecutionResult with execution details
        """
        timeout = timeout or self.timeout

        # Create namespace
        exec_namespace = self._create_namespace()
        if namespace:
            exec_namespace.update(namespace)

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        start_time = time.time()

        try:
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr

            # Execute with timeout
            with timeout_context(timeout):
                exec(compile(code, "<sandbox>", "exec"), exec_namespace)

            execution_time = (time.time() - start_time) * 1000

            # Get stdout/stderr content
            stdout = captured_stdout.getvalue()[:self.max_output_length]
            stderr = captured_stderr.getvalue()[:self.max_output_length]

            # Get result (look for common result variable names)
            result = None
            for name in ['result', 'output', 'answer', '_']:
                if name in exec_namespace:
                    result = exec_namespace[name]
                    break

            return ExecutionResult(
                success=True,
                output=result,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time,
                namespace={k: v for k, v in exec_namespace.items()
                          if not k.startswith('_') and k != '__builtins__'},
            )

        except SyntaxError as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                stdout=captured_stdout.getvalue()[:self.max_output_length],
                stderr=captured_stderr.getvalue()[:self.max_output_length],
                error_type="SyntaxError",
                error_message=str(e),
                traceback=traceback.format_exc(),
                execution_time_ms=execution_time,
            )

        except TimeoutError as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                stdout=captured_stdout.getvalue()[:self.max_output_length],
                stderr=captured_stderr.getvalue()[:self.max_output_length],
                error_type="TimeoutError",
                error_message=str(e),
                traceback=traceback.format_exc(),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ExecutionResult(
                success=False,
                stdout=captured_stdout.getvalue()[:self.max_output_length],
                stderr=captured_stderr.getvalue()[:self.max_output_length],
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
                execution_time_ms=execution_time,
            )

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def execute_with_inputs(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        function_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Execute code with multiple test cases.

        Each test case should have:
        - "inputs": Dict of input values OR positional args list
        - "expected": Expected output (optional, for verification)

        Args:
            code: Python code containing a function to test
            test_cases: List of test case dictionaries
            function_name: Name of function to call (auto-detected if None)
            timeout: Timeout per test case

        Returns:
            List of ExecutionResult, one per test case
        """
        timeout = timeout or self.timeout
        results = []

        # First, compile and get the function
        exec_namespace = self._create_namespace()

        try:
            exec(compile(code, "<sandbox>", "exec"), exec_namespace)
        except Exception as e:
            # Compilation failed - return error for all test cases
            error_result = ExecutionResult(
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback=traceback.format_exc(),
            )
            return [error_result] * len(test_cases)

        # Find the function
        if function_name and function_name in exec_namespace:
            func = exec_namespace[function_name]
        else:
            # Auto-detect: look for callable that's not a builtin
            func = None
            for name, obj in exec_namespace.items():
                if callable(obj) and not name.startswith('_') and name != '__builtins__':
                    if not hasattr(obj, '__module__') or obj.__module__ is None:
                        func = obj
                        function_name = name
                        break

        if func is None:
            error_result = ExecutionResult(
                success=False,
                error_type="NameError",
                error_message="No callable function found in code",
            )
            return [error_result] * len(test_cases)

        # Run each test case
        for test_case in test_cases:
            inputs = test_case.get("inputs", test_case.get("input", {}))
            expected = test_case.get("expected", test_case.get("output"))

            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()

            start_time = time.time()

            try:
                sys.stdout = captured_stdout
                sys.stderr = captured_stderr

                with timeout_context(timeout):
                    if isinstance(inputs, dict):
                        result = func(**inputs)
                    elif isinstance(inputs, (list, tuple)):
                        result = func(*inputs)
                    else:
                        result = func(inputs)

                execution_time = (time.time() - start_time) * 1000
                stdout = captured_stdout.getvalue()[:self.max_output_length]
                stderr = captured_stderr.getvalue()[:self.max_output_length]

                # Check if result matches expected (if provided)
                success = True
                if expected is not None and result != expected:
                    success = False

                results.append(ExecutionResult(
                    success=success,
                    output=result,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time_ms=execution_time,
                ))

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                results.append(ExecutionResult(
                    success=False,
                    stdout=captured_stdout.getvalue()[:self.max_output_length],
                    stderr=captured_stderr.getvalue()[:self.max_output_length],
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                    execution_time_ms=execution_time,
                ))

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        return results


# =============================================================================
# ERROR ANALYZER
# =============================================================================

class ErrorAnalyzer:
    """
    Analyzes execution errors and suggests fix strategies.

    Provides:
    - Error classification into categories
    - Line number extraction from tracebacks
    - Root cause analysis
    - Fix strategy suggestions

    Example:
        >>> analyzer = ErrorAnalyzer()
        >>> result = ExecutionResult(success=False, error_type="NameError",
        ...                          error_message="name 'foo' is not defined")
        >>> analysis = analyzer.analyze(result)
        >>> print(analysis.category)  # ErrorCategory.NAME
    """

    # Mapping from error type names to categories
    ERROR_TYPE_MAPPING = {
        "SyntaxError": ErrorCategory.SYNTAX,
        "IndentationError": ErrorCategory.SYNTAX,
        "TabError": ErrorCategory.SYNTAX,
        "TypeError": ErrorCategory.TYPE,
        "NameError": ErrorCategory.NAME,
        "ValueError": ErrorCategory.VALUE,
        "IndexError": ErrorCategory.INDEX,
        "KeyError": ErrorCategory.KEY,
        "AttributeError": ErrorCategory.ATTRIBUTE,
        "ImportError": ErrorCategory.IMPORT,
        "ModuleNotFoundError": ErrorCategory.IMPORT,
        "AssertionError": ErrorCategory.LOGIC,
        "TimeoutError": ErrorCategory.TIMEOUT,
        "MemoryError": ErrorCategory.MEMORY,
        "RecursionError": ErrorCategory.RECURSION,
        "ZeroDivisionError": ErrorCategory.ZERO_DIVISION,
    }

    def __init__(self):
        """Initialize the error analyzer."""
        # Patterns for extracting information from tracebacks
        self._line_pattern = re.compile(r'File ".*?", line (\d+)')
        self._column_pattern = re.compile(r'\s*\^\s*$', re.MULTILINE)
        self._name_pattern = re.compile(r"name '(\w+)' is not defined")
        self._type_pattern = re.compile(r"'(\w+)' object")
        self._key_pattern = re.compile(r"KeyError: (.+)")
        self._index_pattern = re.compile(r"index (\d+)")

    def analyze(
        self,
        result: ExecutionResult,
        code: Optional[str] = None,
    ) -> ErrorAnalysis:
        """
        Analyze an execution result to understand the error.

        Args:
            result: ExecutionResult from sandbox execution
            code: Original code (for context extraction)

        Returns:
            ErrorAnalysis with classification and suggestions
        """
        if result.success:
            return ErrorAnalysis(
                category=ErrorCategory.UNKNOWN,
                error_type="",
                message="No error - execution succeeded",
            )

        # Classify error
        category = self.classify_error(result.error_type, result.error_message)

        # Extract line number
        line_num = self.extract_line_number(result.traceback)

        # Extract code context
        code_context = None
        if code and line_num:
            code_context = self._get_code_context(code, line_num)

        # Analyze root cause
        root_cause = self._analyze_root_cause(
            category, result.error_type, result.error_message, result.traceback
        )

        # Generate suggested fixes
        suggested_fixes = self._generate_suggestions(
            category, result.error_type, result.error_message, code, line_num
        )

        return ErrorAnalysis(
            category=category,
            error_type=result.error_type or "Unknown",
            message=result.error_message or "",
            line_number=line_num,
            code_context=code_context,
            root_cause=root_cause,
            suggested_fixes=suggested_fixes,
        )

    def classify_error(
        self,
        error_type: Optional[str],
        message: Optional[str],
    ) -> ErrorCategory:
        """
        Classify an error into an ErrorCategory.

        Args:
            error_type: Name of the exception type
            message: Error message

        Returns:
            Appropriate ErrorCategory
        """
        if not error_type:
            return ErrorCategory.UNKNOWN

        # Direct mapping
        if error_type in self.ERROR_TYPE_MAPPING:
            return self.ERROR_TYPE_MAPPING[error_type]

        # Check message for hints
        if message:
            message_lower = message.lower()
            if 'timeout' in message_lower:
                return ErrorCategory.TIMEOUT
            if 'memory' in message_lower:
                return ErrorCategory.MEMORY
            if 'recursion' in message_lower:
                return ErrorCategory.RECURSION

        return ErrorCategory.UNKNOWN

    def extract_line_number(self, tb: Optional[str]) -> Optional[int]:
        """
        Extract the line number where the error occurred from a traceback.

        Args:
            tb: Traceback string

        Returns:
            Line number or None if not found
        """
        if not tb:
            return None

        # Find all line numbers, take the last one (innermost frame)
        matches = self._line_pattern.findall(tb)
        if matches:
            return int(matches[-1])

        # For SyntaxErrors, look for direct line indication
        syntax_match = re.search(r'line (\d+)', tb)
        if syntax_match:
            return int(syntax_match.group(1))

        return None

    def suggest_fix_strategy(
        self,
        analysis: ErrorAnalysis,
        code: str,
    ) -> FixStrategy:
        """
        Suggest a fix strategy based on error analysis.

        Args:
            analysis: ErrorAnalysis from analyze()
            code: The code that produced the error

        Returns:
            FixStrategy with suggested approach
        """
        category = analysis.category

        if category == ErrorCategory.SYNTAX:
            return self._suggest_syntax_fix(analysis, code)
        elif category == ErrorCategory.NAME:
            return self._suggest_name_fix(analysis, code)
        elif category == ErrorCategory.TYPE:
            return self._suggest_type_fix(analysis, code)
        elif category == ErrorCategory.INDEX:
            return self._suggest_index_fix(analysis, code)
        elif category == ErrorCategory.KEY:
            return self._suggest_key_fix(analysis, code)
        elif category == ErrorCategory.ATTRIBUTE:
            return self._suggest_attribute_fix(analysis, code)
        elif category == ErrorCategory.VALUE:
            return self._suggest_value_fix(analysis, code)
        elif category == ErrorCategory.IMPORT:
            return self._suggest_import_fix(analysis, code)
        elif category == ErrorCategory.ZERO_DIVISION:
            return self._suggest_zero_division_fix(analysis, code)
        elif category == ErrorCategory.TIMEOUT:
            return FixStrategy(
                strategy_type=FixStrategyType.REGENERATE,
                suggestion="Code is too slow or has infinite loop. Regenerate with more efficient algorithm.",
                confidence=0.3,
                reasoning="Timeout errors typically require algorithmic changes.",
            )
        else:
            return FixStrategy(
                strategy_type=FixStrategyType.REGENERATE,
                suggestion="Unable to determine specific fix. Regenerate the code.",
                confidence=0.2,
                reasoning=f"Unknown error category: {category}",
            )

    def _get_code_context(self, code: str, line_num: int, context_lines: int = 2) -> str:
        """Get lines of code around the error location."""
        lines = code.split('\n')
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)

        context = []
        for i in range(start, end):
            marker = ">>> " if i == line_num - 1 else "    "
            context.append(f"{marker}{i + 1}: {lines[i]}")

        return '\n'.join(context)

    def _analyze_root_cause(
        self,
        category: ErrorCategory,
        error_type: Optional[str],
        message: Optional[str],
        tb: Optional[str],
    ) -> str:
        """Analyze the root cause of the error."""
        if not message:
            return "Unknown root cause"

        if category == ErrorCategory.NAME:
            match = self._name_pattern.search(message)
            if match:
                return f"Variable or function '{match.group(1)}' is used but not defined"

        elif category == ErrorCategory.TYPE:
            return f"Type mismatch: {message}"

        elif category == ErrorCategory.KEY:
            match = self._key_pattern.search(message)
            if match:
                return f"Key {match.group(1)} does not exist in dictionary"

        elif category == ErrorCategory.INDEX:
            return "List index out of range - trying to access element that doesn't exist"

        elif category == ErrorCategory.SYNTAX:
            return f"Invalid Python syntax: {message}"

        elif category == ErrorCategory.ATTRIBUTE:
            return f"Object doesn't have the expected attribute: {message}"

        return message

    def _generate_suggestions(
        self,
        category: ErrorCategory,
        error_type: Optional[str],
        message: Optional[str],
        code: Optional[str],
        line_num: Optional[int],
    ) -> List[str]:
        """Generate fix suggestions based on error type."""
        suggestions = []

        if category == ErrorCategory.NAME:
            match = self._name_pattern.search(message or "")
            if match:
                name = match.group(1)
                suggestions.append(f"Define '{name}' before using it")
                suggestions.append(f"Check if '{name}' is misspelled")
                suggestions.append(f"Import '{name}' if it's from a module")
            else:
                # Generic suggestions when pattern doesn't match
                suggestions.append("Define the variable before using it")
                suggestions.append("Check for misspelled variable names")
                suggestions.append("Import the module if it's from an external package")

        elif category == ErrorCategory.TYPE:
            suggestions.append("Add type conversion (int(), str(), float())")
            suggestions.append("Check that operands are compatible types")

        elif category == ErrorCategory.INDEX:
            suggestions.append("Add bounds checking before accessing index")
            suggestions.append("Use try/except to handle missing elements")
            suggestions.append("Check list length before accessing")

        elif category == ErrorCategory.KEY:
            suggestions.append("Use dict.get() with default value")
            suggestions.append("Check if key exists with 'in' operator")
            suggestions.append("Use try/except for KeyError")

        elif category == ErrorCategory.SYNTAX:
            suggestions.append("Check for missing colons, parentheses, or brackets")
            suggestions.append("Check indentation is consistent")
            suggestions.append("Ensure strings are properly quoted")

        elif category == ErrorCategory.ATTRIBUTE:
            suggestions.append("Check object type has expected method/attribute")
            suggestions.append("Add None check before accessing attribute")

        elif category == ErrorCategory.ZERO_DIVISION:
            suggestions.append("Add check for zero before division")
            suggestions.append("Use try/except for ZeroDivisionError")

        elif category == ErrorCategory.VALUE:
            suggestions.append("Validate input values before processing")
            suggestions.append("Add input validation at function start")

        return suggestions

    def _suggest_syntax_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for syntax errors."""
        message = analysis.message.lower()

        if 'indent' in message:
            return FixStrategy(
                strategy_type=FixStrategyType.FIX_INDENTATION,
                target_line=analysis.line_number,
                suggestion="Fix indentation - use consistent spaces (4 per level)",
                confidence=0.7,
                reasoning="Indentation errors are usually about inconsistent spacing",
            )

        if 'expected' in message and ':' in message:
            return FixStrategy(
                strategy_type=FixStrategyType.REPLACE_LINE,
                target_line=analysis.line_number,
                suggestion="Add missing colon at end of statement",
                confidence=0.6,
                reasoning="Missing colon after if/for/def/class statements",
            )

        return FixStrategy(
            strategy_type=FixStrategyType.REPLACE_LINE,
            target_line=analysis.line_number,
            suggestion="Fix syntax error on this line",
            confidence=0.5,
            reasoning=f"Syntax error: {analysis.message}",
        )

    def _suggest_name_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for name errors."""
        match = self._name_pattern.search(analysis.message)
        if match:
            undefined_name = match.group(1)

            # Check if it might be a typo
            lines = code.split('\n')
            defined_names = set()
            for line in lines:
                # Extract variable assignments
                assign_match = re.match(r'^\s*(\w+)\s*=', line)
                if assign_match:
                    defined_names.add(assign_match.group(1))

            # Simple typo detection
            for name in defined_names:
                if self._is_similar(undefined_name, name):
                    return FixStrategy(
                        strategy_type=FixStrategyType.MODIFY_EXPRESSION,
                        target_line=analysis.line_number,
                        suggestion=f"Replace '{undefined_name}' with '{name}' (possible typo)",
                        new_code=name,
                        confidence=0.8,
                        reasoning=f"'{name}' is defined and similar to '{undefined_name}'",
                    )

        return FixStrategy(
            strategy_type=FixStrategyType.INSERT_LINE,
            target_line=analysis.line_number,
            suggestion="Define the variable before using it",
            confidence=0.5,
            reasoning="Variable used before definition",
        )

    def _suggest_type_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for type errors."""
        message = analysis.message

        if 'str' in message and ('int' in message or 'float' in message):
            return FixStrategy(
                strategy_type=FixStrategyType.ADD_TYPE_CONVERSION,
                target_line=analysis.line_number,
                suggestion="Convert string to number with int() or float()",
                confidence=0.7,
                reasoning="String cannot be used in numeric operation",
            )

        if 'NoneType' in message:
            return FixStrategy(
                strategy_type=FixStrategyType.ADD_NULL_CHECK,
                target_line=analysis.line_number,
                suggestion="Add check for None before operation",
                confidence=0.7,
                reasoning="Operation attempted on None value",
            )

        return FixStrategy(
            strategy_type=FixStrategyType.MODIFY_EXPRESSION,
            target_line=analysis.line_number,
            suggestion="Check types and add appropriate conversion",
            confidence=0.5,
            reasoning=f"Type error: {message}",
        )

    def _suggest_index_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for index errors."""
        return FixStrategy(
            strategy_type=FixStrategyType.ADD_BOUNDS_CHECK,
            target_line=analysis.line_number,
            suggestion="Add bounds check: if idx < len(list)",
            confidence=0.7,
            reasoning="Index out of bounds - list is shorter than expected",
        )

    def _suggest_key_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for key errors."""
        return FixStrategy(
            strategy_type=FixStrategyType.MODIFY_EXPRESSION,
            target_line=analysis.line_number,
            suggestion="Use dict.get(key, default) instead of dict[key]",
            confidence=0.8,
            reasoning="Key not found in dictionary",
        )

    def _suggest_attribute_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for attribute errors."""
        if 'NoneType' in analysis.message:
            return FixStrategy(
                strategy_type=FixStrategyType.ADD_NULL_CHECK,
                target_line=analysis.line_number,
                suggestion="Add 'if obj is not None:' check before accessing attribute",
                confidence=0.7,
                reasoning="Attempting to access attribute of None",
            )

        return FixStrategy(
            strategy_type=FixStrategyType.MODIFY_EXPRESSION,
            target_line=analysis.line_number,
            suggestion="Check object type or use hasattr()",
            confidence=0.5,
            reasoning=f"Attribute error: {analysis.message}",
        )

    def _suggest_value_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for value errors."""
        return FixStrategy(
            strategy_type=FixStrategyType.WRAP_TRY_EXCEPT,
            target_line=analysis.line_number,
            suggestion="Add input validation or wrap in try/except",
            confidence=0.5,
            reasoning=f"Invalid value: {analysis.message}",
        )

    def _suggest_import_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for import errors."""
        return FixStrategy(
            strategy_type=FixStrategyType.ADD_IMPORT,
            suggestion="Add missing import statement or use allowed module",
            confidence=0.6,
            reasoning="Module not found or not allowed",
        )

    def _suggest_zero_division_fix(self, analysis: ErrorAnalysis, code: str) -> FixStrategy:
        """Suggest fix for zero division errors."""
        return FixStrategy(
            strategy_type=FixStrategyType.MODIFY_EXPRESSION,
            target_line=analysis.line_number,
            suggestion="Add check: 'if divisor != 0:' before division",
            confidence=0.8,
            reasoning="Division by zero - need to check divisor",
        )

    def _is_similar(self, s1: str, s2: str, threshold: int = 2) -> bool:
        """Check if two strings are similar (Levenshtein distance <= threshold)."""
        if abs(len(s1) - len(s2)) > threshold:
            return False

        # Simple edit distance
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1) <= threshold

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1] <= threshold


# =============================================================================
# SELF DEBUGGER
# =============================================================================

class SelfDebugger:
    """
    Main self-debugging class that iteratively fixes code errors.

    Implements the self-debugging loop:
    1. Execute code in sandbox
    2. If error, analyze and classify it
    3. Generate fix based on error analysis (optionally using LLM)
    4. Apply fix and re-execute
    5. Repeat until success or max attempts

    Example:
        >>> debugger = SelfDebugger(max_attempts=5)
        >>> code = "def add(a, b): return a + b"
        >>> test_cases = [{"inputs": {"a": 1, "b": 2}, "expected": 3}]
        >>> result = debugger.debug(code, "add two numbers", test_cases)
        >>> print(result.success)
    """

    def __init__(
        self,
        backend: Optional[Any] = None,
        max_attempts: int = 5,
        sandbox_timeout: float = 5.0,
        verbose: bool = False,
    ):
        """
        Initialize the self-debugger.

        Args:
            backend: Optional LLM backend for generating fixes
            max_attempts: Maximum number of fix attempts
            sandbox_timeout: Timeout for each code execution
            verbose: Whether to print debug information
        """
        self.backend = backend
        self.max_attempts = max_attempts
        self.verbose = verbose

        self.sandbox = ExecutionSandbox(timeout=sandbox_timeout)
        self.analyzer = ErrorAnalyzer()

    def debug(
        self,
        code: str,
        spec: str,
        test_cases: List[Dict[str, Any]],
        function_name: Optional[str] = None,
    ) -> DebugResult:
        """
        Debug code by iteratively fixing errors until tests pass.

        Args:
            code: Python code to debug
            spec: Natural language specification of what the code should do
            test_cases: List of test cases to verify correctness
            function_name: Name of function to test (auto-detected if None)

        Returns:
            DebugResult with debugging history and final code
        """
        original_code = code
        current_code = code
        fix_history: List[FixAttempt] = []
        debug_trace: List[Dict[str, Any]] = []

        debug_trace.append({
            "step": "start",
            "code_length": len(code),
            "test_cases": len(test_cases),
        })

        for attempt in range(1, self.max_attempts + 1):
            if self.verbose:
                print(f"\n--- Attempt {attempt}/{self.max_attempts} ---")

            # Execute with test cases
            results = self.sandbox.execute_with_inputs(
                current_code, test_cases, function_name
            )

            # Check if all tests pass
            all_passed = all(r.success for r in results)

            debug_trace.append({
                "step": f"attempt_{attempt}",
                "all_passed": all_passed,
                "results": [r.to_dict() for r in results],
            })

            if all_passed:
                if self.verbose:
                    print("All tests passed!")

                return DebugResult(
                    original_code=original_code,
                    final_code=current_code,
                    success=True,
                    attempts=attempt,
                    fix_history=fix_history,
                    all_tests_passed=True,
                    final_result=results[0] if results else None,
                    debug_trace=debug_trace,
                )

            # Find first failing test
            failed_result = next((r for r in results if not r.success), results[0])

            # Analyze error
            analysis = self.analyzer.analyze(failed_result, current_code)

            if self.verbose:
                print(f"Error: {analysis.error_type} - {analysis.message}")
                print(f"Category: {analysis.category.value}")
                if analysis.line_number:
                    print(f"Line: {analysis.line_number}")

            # Generate fix
            fix_strategy = self._generate_fix(current_code, analysis, spec)

            if self.verbose:
                print(f"Fix strategy: {fix_strategy.strategy_type.value}")
                print(f"Suggestion: {fix_strategy.suggestion}")

            # Apply fix
            new_code = self._apply_fix(current_code, fix_strategy, analysis, spec)

            # Record attempt
            fix_attempt = FixAttempt(
                attempt_num=attempt,
                code_before=current_code,
                code_after=new_code,
                error_before=analysis,
                fix_applied=fix_strategy,
                result=failed_result,
                success=False,  # Will be updated if next attempt succeeds
            )
            fix_history.append(fix_attempt)

            # Update current code
            current_code = new_code

        # Max attempts reached without success
        debug_trace.append({
            "step": "max_attempts_reached",
            "attempts": self.max_attempts,
        })

        # Run final check
        final_results = self.sandbox.execute_with_inputs(
            current_code, test_cases, function_name
        )
        all_passed = all(r.success for r in final_results)

        return DebugResult(
            original_code=original_code,
            final_code=current_code,
            success=all_passed,
            attempts=self.max_attempts,
            fix_history=fix_history,
            all_tests_passed=all_passed,
            final_result=final_results[0] if final_results else None,
            debug_trace=debug_trace,
        )

    def _generate_fix(
        self,
        code: str,
        error_analysis: ErrorAnalysis,
        spec: str,
    ) -> FixStrategy:
        """
        Generate a fix strategy for the error.

        If a backend LLM is available, uses it to generate fixes.
        Otherwise, uses rule-based heuristics.

        Args:
            code: Current code
            error_analysis: Analysis of the error
            spec: Original specification

        Returns:
            FixStrategy to apply
        """
        if self.backend:
            return self._generate_fix_with_llm(code, error_analysis, spec)
        else:
            return self.analyzer.suggest_fix_strategy(error_analysis, code)

    def _generate_fix_with_llm(
        self,
        code: str,
        error_analysis: ErrorAnalysis,
        spec: str,
    ) -> FixStrategy:
        """Generate fix using LLM backend."""
        prompt = f"""The following Python code has an error:

```python
{code}
```

Error: {error_analysis.error_type}
Message: {error_analysis.message}
{f"Line: {error_analysis.line_number}" if error_analysis.line_number else ""}

The code should: {spec}

Please provide the corrected code. Only output the Python code, no explanations.
```python"""

        try:
            if hasattr(self.backend, 'generate'):
                response = self.backend.generate(prompt)
            elif callable(self.backend):
                response = self.backend(prompt)
            else:
                # Fallback to rule-based
                return self.analyzer.suggest_fix_strategy(error_analysis, code)

            # Extract code from response
            fixed_code = self._extract_code(response)

            return FixStrategy(
                strategy_type=FixStrategyType.REGENERATE,
                new_code=fixed_code,
                suggestion="LLM-generated fix",
                confidence=0.7,
                reasoning="Generated by language model based on error analysis",
            )

        except Exception:
            # Fallback to rule-based on LLM error
            return self.analyzer.suggest_fix_strategy(error_analysis, code)

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Look for def keyword
        if "def " in response:
            lines = response.split('\n')
            code_lines = []
            in_function = False
            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)
            if code_lines:
                return '\n'.join(code_lines).strip()

        return response.strip()

    def _apply_fix(
        self,
        code: str,
        fix_strategy: FixStrategy,
        error_analysis: ErrorAnalysis,
        spec: str,
    ) -> str:
        """
        Apply a fix strategy to the code.

        Args:
            code: Current code
            fix_strategy: Strategy to apply
            error_analysis: Error analysis for context
            spec: Original specification

        Returns:
            Fixed code
        """
        strategy_type = fix_strategy.strategy_type

        if strategy_type == FixStrategyType.REGENERATE:
            if fix_strategy.new_code:
                return fix_strategy.new_code
            return self._regenerate_code(code, error_analysis, spec)

        if strategy_type == FixStrategyType.REPLACE_LINE:
            return self._replace_line(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.INSERT_LINE:
            return self._insert_line(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.DELETE_LINE:
            return self._delete_line(code, fix_strategy)

        if strategy_type == FixStrategyType.FIX_INDENTATION:
            return self._fix_indentation(code, fix_strategy)

        if strategy_type == FixStrategyType.ADD_TYPE_CONVERSION:
            return self._add_type_conversion(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.ADD_NULL_CHECK:
            return self._add_null_check(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.ADD_BOUNDS_CHECK:
            return self._add_bounds_check(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.ADD_IMPORT:
            return self._add_import(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.MODIFY_EXPRESSION:
            return self._modify_expression(code, fix_strategy, error_analysis)

        if strategy_type == FixStrategyType.WRAP_TRY_EXCEPT:
            return self._wrap_try_except(code, fix_strategy, error_analysis)

        # Default: return unchanged (shouldn't happen)
        return code

    def _regenerate_code(
        self,
        code: str,
        error_analysis: ErrorAnalysis,
        spec: str,
    ) -> str:
        """Regenerate code when fix strategy is REGENERATE but no new_code provided."""
        # Without LLM, we can't truly regenerate
        # Apply simple fixes based on error type
        if error_analysis.category == ErrorCategory.SYNTAX:
            return self._attempt_syntax_fix(code, error_analysis)
        return code

    def _attempt_syntax_fix(self, code: str, analysis: ErrorAnalysis) -> str:
        """Attempt to fix syntax errors heuristically."""
        lines = code.split('\n')

        if analysis.line_number and analysis.line_number <= len(lines):
            line_idx = analysis.line_number - 1
            line = lines[line_idx]

            # Common fixes
            # Missing colon
            if 'expected' in analysis.message.lower() and ':' in analysis.message:
                stripped = line.rstrip()
                if not stripped.endswith(':') and any(
                    stripped.lstrip().startswith(kw) for kw in ['if', 'else', 'elif', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with']
                ):
                    lines[line_idx] = stripped + ':'

            # Unclosed parenthesis
            if 'parenthes' in analysis.message.lower():
                open_count = line.count('(')
                close_count = line.count(')')
                if open_count > close_count:
                    lines[line_idx] = line.rstrip() + ')' * (open_count - close_count)
                elif close_count > open_count:
                    lines[line_idx] = '(' * (close_count - open_count) + line

        return '\n'.join(lines)

    def _replace_line(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Replace a specific line in the code."""
        if not strategy.target_line or not strategy.new_code:
            return code

        lines = code.split('\n')
        if 1 <= strategy.target_line <= len(lines):
            # Preserve indentation
            old_line = lines[strategy.target_line - 1]
            indent = len(old_line) - len(old_line.lstrip())
            lines[strategy.target_line - 1] = ' ' * indent + strategy.new_code.lstrip()

        return '\n'.join(lines)

    def _insert_line(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Insert a new line at the specified position."""
        if not strategy.target_line or not strategy.new_code:
            return code

        lines = code.split('\n')
        insert_idx = max(0, min(len(lines), strategy.target_line - 1))

        # Match indentation of surrounding code
        if insert_idx < len(lines):
            ref_line = lines[insert_idx]
            indent = len(ref_line) - len(ref_line.lstrip())
        else:
            indent = 0

        lines.insert(insert_idx, ' ' * indent + strategy.new_code.lstrip())
        return '\n'.join(lines)

    def _delete_line(self, code: str, strategy: FixStrategy) -> str:
        """Delete a specific line from the code."""
        if not strategy.target_line:
            return code

        lines = code.split('\n')
        if 1 <= strategy.target_line <= len(lines):
            del lines[strategy.target_line - 1]

        return '\n'.join(lines)

    def _fix_indentation(self, code: str, strategy: FixStrategy) -> str:
        """Fix indentation issues in the code."""
        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            if '\t' in line:
                # Replace tabs with 4 spaces
                line = line.replace('\t', '    ')
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _add_type_conversion(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Add type conversion to fix type errors."""
        # This is a simplified implementation
        # A full implementation would parse the AST
        if 'str' in analysis.message and 'int' in analysis.message:
            # Try to wrap string operations with int()
            if analysis.line_number:
                lines = code.split('\n')
                if analysis.line_number <= len(lines):
                    line = lines[analysis.line_number - 1]
                    # Simple heuristic: wrap variable in int()
                    # This is very basic and would need AST analysis for robust handling
                    lines[analysis.line_number - 1] = re.sub(
                        r'\b([a-zA-Z_]\w*)\b(?!\s*\()',
                        r'int(\1)' if 'int' in analysis.message else r'str(\1)',
                        line,
                        count=1
                    )
                    return '\n'.join(lines)
        return code

    def _add_null_check(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Add null/None check before operation."""
        if not analysis.line_number:
            return code

        lines = code.split('\n')
        if analysis.line_number <= len(lines):
            line = lines[analysis.line_number - 1]
            indent = len(line) - len(line.lstrip())

            # Insert check before the problematic line
            check = ' ' * indent + "if result is not None:"
            lines.insert(analysis.line_number - 1, check)
            # Indent the original line
            lines[analysis.line_number] = '    ' + lines[analysis.line_number]

        return '\n'.join(lines)

    def _add_bounds_check(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Add index bounds check."""
        if not analysis.line_number:
            return code

        lines = code.split('\n')
        if analysis.line_number <= len(lines):
            line = lines[analysis.line_number - 1]
            indent = len(line) - len(line.lstrip())

            # Look for list[index] pattern
            match = re.search(r'(\w+)\[(\w+)\]', line)
            if match:
                list_name, index_name = match.groups()
                check = f"{' ' * indent}if {index_name} < len({list_name}):"
                lines.insert(analysis.line_number - 1, check)
                lines[analysis.line_number] = '    ' + lines[analysis.line_number]

        return '\n'.join(lines)

    def _add_import(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Add missing import statement."""
        # Extract module name from error
        match = re.search(r"No module named '(\w+)'", analysis.message)
        if not match:
            match = re.search(r"name '(\w+)' is not defined", analysis.message)

        if match:
            module_name = match.group(1)
            # Only add if it's in allowed modules
            if module_name in self.sandbox._allowed_modules:
                return f"import {module_name}\n" + code

        return code

    def _modify_expression(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Modify an expression to fix an error."""
        if strategy.new_code and analysis.line_number:
            return self._replace_line(code, strategy, analysis)

        # For KeyError, try to add .get()
        if analysis.category == ErrorCategory.KEY and analysis.line_number:
            lines = code.split('\n')
            if analysis.line_number <= len(lines):
                line = lines[analysis.line_number - 1]
                # Replace dict[key] with dict.get(key)
                new_line = re.sub(r'(\w+)\[([^\]]+)\]', r'\1.get(\2)', line)
                lines[analysis.line_number - 1] = new_line
                return '\n'.join(lines)

        return code

    def _wrap_try_except(
        self,
        code: str,
        strategy: FixStrategy,
        analysis: ErrorAnalysis,
    ) -> str:
        """Wrap code in try/except block."""
        if not analysis.line_number:
            return code

        lines = code.split('\n')
        if analysis.line_number <= len(lines):
            line = lines[analysis.line_number - 1]
            indent = len(line) - len(line.lstrip())
            base_indent = ' ' * indent

            # Wrap the line in try/except
            try_line = base_indent + "try:"
            indented_line = base_indent + "    " + line.lstrip()
            except_line = base_indent + f"except {analysis.error_type or 'Exception'}:"
            pass_line = base_indent + "    pass  # Handle error"

            lines[analysis.line_number - 1] = '\n'.join([
                try_line, indented_line, except_line, pass_line
            ])

        return '\n'.join(lines)

    def _verify_fix(
        self,
        new_code: str,
        test_cases: List[Dict[str, Any]],
        function_name: Optional[str] = None,
    ) -> bool:
        """Verify that the fix passes all test cases."""
        results = self.sandbox.execute_with_inputs(
            new_code, test_cases, function_name
        )
        return all(r.success for r in results)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ErrorCategory",
    "FixStrategyType",
    # Data classes
    "ExecutionResult",
    "ErrorAnalysis",
    "FixStrategy",
    "FixAttempt",
    "DebugResult",
    # Classes
    "ExecutionSandbox",
    "ErrorAnalyzer",
    "SelfDebugger",
    # Exceptions
    "TimeoutError",
]
