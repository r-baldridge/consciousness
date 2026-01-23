"""
Synthetic Bug Generator for TRM Code Repair Dataset

Generates training pairs (buggy_code, fixed_code, bug_type) by
introducing realistic bugs into clean Python code.
"""

import random
import re
import ast
import tokenize
import io
from typing import Tuple, List, Optional, Dict, Callable, Set
from dataclasses import dataclass
from enum import Enum

from ..taxonomy.bug_types import BugType, BugCategory


@dataclass
class BugMutation:
    """Result of applying a bug mutation."""
    buggy_code: str
    fixed_code: str
    bug_type: BugType
    bug_location: Tuple[int, int]  # (line, column)
    description: str


class SyntheticBugGenerator:
    """
    Generates synthetic buggy code from clean Python code.

    Implements mutation-based bug injection for creating training pairs.
    Supports Tier 1-3 bugs (syntax, logic, style) which can be
    programmatically generated.

    Example:
        generator = SyntheticBugGenerator(seed=42)
        buggy, fixed, bug_type = generator.generate(clean_code)
    """

    # Mutation functions by bug type
    MUTATIONS: Dict[BugType, Callable] = {}

    # Bug type weights for random selection (based on frequency)
    DEFAULT_WEIGHTS = {
        # Tier 1: Syntax (high frequency, easy to generate)
        BugType.MISSING_COLON: 0.15,
        BugType.MISSING_PARENTHESIS: 0.10,
        BugType.INDENTATION_ERROR: 0.15,
        BugType.MISSING_COMMA: 0.08,
        BugType.MISSING_BRACKET: 0.06,

        # Tier 2: Logic (medium frequency, require care)
        BugType.OFF_BY_ONE: 0.12,
        BugType.WRONG_OPERATOR: 0.10,
        BugType.WRONG_COMPARISON: 0.08,
        BugType.MUTABLE_DEFAULT: 0.04,
        BugType.BARE_EXCEPT: 0.05,

        # Tier 3: Style (lower priority for synthetic)
        BugType.NAMING_CONVENTION: 0.04,
        BugType.UNUSED_IMPORT: 0.03,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        weights: Optional[Dict[BugType, float]] = None,
        enabled_tiers: Optional[Set[BugCategory]] = None,
    ):
        """
        Initialize the synthetic bug generator.

        Args:
            seed: Random seed for reproducibility
            weights: Custom weights for bug type selection
            enabled_tiers: Set of BugCategory to enable (default: all)
        """
        self.rng = random.Random(seed)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.enabled_tiers = enabled_tiers or {
            BugCategory.SYNTAX,
            BugCategory.LOGIC,
            BugCategory.STYLE,
        }

        # Register mutation functions
        self._register_mutations()

    def _register_mutations(self):
        """Register all mutation functions."""
        self.MUTATIONS = {
            # Tier 1: Syntax
            BugType.MISSING_COLON: self._mutate_missing_colon,
            BugType.MISSING_PARENTHESIS: self._mutate_missing_parenthesis,
            BugType.INDENTATION_ERROR: self._mutate_indentation,
            BugType.MISSING_COMMA: self._mutate_missing_comma,
            BugType.MISSING_BRACKET: self._mutate_missing_bracket,

            # Tier 2: Logic
            BugType.OFF_BY_ONE: self._mutate_off_by_one,
            BugType.WRONG_OPERATOR: self._mutate_wrong_operator,
            BugType.WRONG_COMPARISON: self._mutate_wrong_comparison,
            BugType.MUTABLE_DEFAULT: self._mutate_mutable_default,
            BugType.BARE_EXCEPT: self._mutate_bare_except,

            # Tier 3: Style
            BugType.NAMING_CONVENTION: self._mutate_naming,
            BugType.UNUSED_IMPORT: self._mutate_add_unused_import,
        }

    def generate(
        self,
        clean_code: str,
        bug_type: Optional[BugType] = None,
        max_attempts: int = 10,
    ) -> Optional[Tuple[str, str, BugType]]:
        """
        Generate a buggy version of clean code.

        Args:
            clean_code: Valid Python source code
            bug_type: Specific bug type to inject (random if None)
            max_attempts: Maximum attempts to find applicable mutation

        Returns:
            Tuple of (buggy_code, fixed_code, bug_type) or None if failed
        """
        # Validate input is valid Python
        if not self._is_valid_python(clean_code):
            return None

        # Select bug type
        if bug_type is None:
            bug_type = self._select_bug_type(clean_code)

        if bug_type is None:
            return None

        # Get mutation function
        mutation_fn = self.MUTATIONS.get(bug_type)
        if mutation_fn is None:
            return None

        # Apply mutation with retries
        for _ in range(max_attempts):
            result = mutation_fn(clean_code)
            if result is not None:
                return (result.buggy_code, result.fixed_code, result.bug_type)

        return None

    def generate_batch(
        self,
        clean_codes: List[str],
        bugs_per_code: int = 3,
    ) -> List[Tuple[str, str, BugType]]:
        """
        Generate multiple buggy versions from a list of clean codes.

        Args:
            clean_codes: List of valid Python source codes
            bugs_per_code: Number of different bugs to generate per code

        Returns:
            List of (buggy_code, fixed_code, bug_type) tuples
        """
        results = []

        for code in clean_codes:
            used_types = set()

            for _ in range(bugs_per_code):
                # Try to get a different bug type
                available_types = [
                    bt for bt in self.weights.keys()
                    if bt not in used_types
                ]

                if not available_types:
                    break

                bug_type = self.rng.choice(available_types)
                result = self.generate(code, bug_type=bug_type)

                if result is not None:
                    results.append(result)
                    used_types.add(bug_type)

        return results

    def _select_bug_type(self, code: str) -> Optional[BugType]:
        """Select an applicable bug type based on code structure."""
        # Filter to applicable mutations
        applicable = []
        weights = []

        for bug_type, weight in self.weights.items():
            if self._is_applicable(code, bug_type):
                applicable.append(bug_type)
                weights.append(weight)

        if not applicable:
            return None

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        return self.rng.choices(applicable, weights=weights, k=1)[0]

    def _is_applicable(self, code: str, bug_type: BugType) -> bool:
        """Check if a bug type can be applied to this code."""
        checks = {
            BugType.MISSING_COLON: lambda c: re.search(
                r'^[ \t]*(if|for|while|def|class|try|except|with|elif|else)\b',
                c, re.MULTILINE
            ),
            BugType.MISSING_PARENTHESIS: lambda c: '(' in c,
            BugType.INDENTATION_ERROR: lambda c: '\n    ' in c or '\n\t' in c,
            BugType.MISSING_COMMA: lambda c: re.search(r'\[.*,.*\]|\(.*,.*\)', c),
            BugType.MISSING_BRACKET: lambda c: '[' in c or '{' in c,
            BugType.OFF_BY_ONE: lambda c: re.search(r'range\s*\(|len\s*\(', c),
            BugType.WRONG_OPERATOR: lambda c: re.search(r'[+\-*/]', c),
            BugType.WRONG_COMPARISON: lambda c: re.search(r'==|!=|<|>|<=|>=', c),
            BugType.MUTABLE_DEFAULT: lambda c: re.search(r'def\s+\w+\s*\([^)]*=', c),
            BugType.BARE_EXCEPT: lambda c: re.search(
                r'except\s+\w+',
                c
            ),
            BugType.NAMING_CONVENTION: lambda c: re.search(r'def\s+\w+', c),
            BugType.UNUSED_IMPORT: lambda c: True,  # Can always add import
        }

        check = checks.get(bug_type)
        return check is not None and check(code)

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    # ===== Tier 1: Syntax Mutations =====

    def _mutate_missing_colon(self, code: str) -> Optional[BugMutation]:
        """Remove a colon from a control flow statement."""
        # Find lines ending with colon
        pattern = r'^([ \t]*(?:if|for|while|def|class|try|except|with|elif|else)[^:]*):[ \t]*$'
        matches = list(re.finditer(pattern, code, re.MULTILINE))

        if not matches:
            return None

        match = self.rng.choice(matches)
        line_start = match.start()
        colon_pos = match.end() - 1

        # Remove the colon
        buggy = code[:colon_pos] + code[colon_pos + 1:]

        # Find line number
        line_num = code[:line_start].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.MISSING_COLON,
            bug_location=(line_num, match.end() - match.start() - 1),
            description=f"Removed colon from line {line_num}",
        )

    def _mutate_missing_parenthesis(self, code: str) -> Optional[BugMutation]:
        """Remove a closing parenthesis."""
        # Find balanced parentheses
        parens = []
        depth = 0
        in_string = False
        string_char = None

        for i, char in enumerate(code):
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and (i == 0 or code[i-1] != '\\'):
                    in_string = False

            if not in_string:
                if char == '(':
                    depth += 1
                elif char == ')' and depth > 0:
                    depth -= 1
                    parens.append(i)

        if not parens:
            return None

        # Remove a random closing paren
        pos = self.rng.choice(parens)
        buggy = code[:pos] + code[pos + 1:]

        line_num = code[:pos].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.MISSING_PARENTHESIS,
            bug_location=(line_num, pos - code[:pos].rfind('\n') - 1),
            description=f"Removed closing parenthesis at position {pos}",
        )

    def _mutate_indentation(self, code: str) -> Optional[BugMutation]:
        """Introduce an indentation error."""
        lines = code.split('\n')
        indented_lines = [
            i for i, line in enumerate(lines)
            if line.startswith('    ') or line.startswith('\t')
        ]

        if not indented_lines:
            return None

        line_idx = self.rng.choice(indented_lines)
        line = lines[line_idx]

        # Choose mutation: remove indent, add extra, or wrong level
        mutation = self.rng.choice(['remove', 'extra', 'wrong'])

        if mutation == 'remove':
            # Remove one level of indentation
            if line.startswith('    '):
                lines[line_idx] = line[4:]
            elif line.startswith('\t'):
                lines[line_idx] = line[1:]
        elif mutation == 'extra':
            # Add extra indentation
            lines[line_idx] = '    ' + line
        else:
            # Wrong level (off by one space)
            if line.startswith('    '):
                lines[line_idx] = '   ' + line.lstrip()

        buggy = '\n'.join(lines)

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.INDENTATION_ERROR,
            bug_location=(line_idx + 1, 0),
            description=f"Modified indentation on line {line_idx + 1}",
        )

    def _mutate_missing_comma(self, code: str) -> Optional[BugMutation]:
        """Remove a comma from a list or tuple."""
        # Find commas in lists/tuples
        pattern = r'(\[[^\]]*),([^\]]*\])|(\([^\)]*),([^\)]*\))'
        matches = list(re.finditer(r',', code))

        # Filter to commas likely in collections
        valid_commas = []
        for m in matches:
            pos = m.start()
            # Check if inside brackets
            before = code[:pos]
            open_brackets = before.count('[') - before.count(']')
            open_parens = before.count('(') - before.count(')')
            if open_brackets > 0 or open_parens > 0:
                valid_commas.append(pos)

        if not valid_commas:
            return None

        pos = self.rng.choice(valid_commas)
        buggy = code[:pos] + code[pos + 1:]

        line_num = code[:pos].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.MISSING_COMMA,
            bug_location=(line_num, pos - code[:pos].rfind('\n') - 1),
            description=f"Removed comma at position {pos}",
        )

    def _mutate_missing_bracket(self, code: str) -> Optional[BugMutation]:
        """Remove a closing bracket."""
        brackets = {'[': ']', '{': '}'}
        positions = []

        for close_char in brackets.values():
            for i, char in enumerate(code):
                if char == close_char:
                    positions.append(i)

        if not positions:
            return None

        pos = self.rng.choice(positions)
        buggy = code[:pos] + code[pos + 1:]

        line_num = code[:pos].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.MISSING_BRACKET,
            bug_location=(line_num, pos - code[:pos].rfind('\n') - 1),
            description=f"Removed closing bracket at position {pos}",
        )

    # ===== Tier 2: Logic Mutations =====

    def _mutate_off_by_one(self, code: str) -> Optional[BugMutation]:
        """Introduce an off-by-one error in range or indexing."""
        # Find range() calls
        pattern = r'range\s*\(\s*(\d+)\s*\)'
        matches = list(re.finditer(pattern, code))

        if not matches:
            # Try len() based patterns
            pattern = r'range\s*\(\s*len\s*\([^)]+\)\s*\)'
            matches = list(re.finditer(pattern, code))

        if not matches:
            return None

        match = self.rng.choice(matches)

        # Choose mutation: +1 or -1
        if 'len(' in match.group():
            # range(len(x)) -> range(len(x) + 1) or range(len(x) - 1)
            offset = self.rng.choice([' + 1', ' - 1'])
            buggy = (
                code[:match.end() - 1] +
                offset +
                code[match.end() - 1:]
            )
        else:
            # range(n) -> range(n+1) or range(n-1)
            num = int(match.group(1))
            new_num = num + self.rng.choice([1, -1])
            buggy = code[:match.start(1)] + str(new_num) + code[match.end(1):]

        line_num = code[:match.start()].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.OFF_BY_ONE,
            bug_location=(line_num, match.start() - code[:match.start()].rfind('\n') - 1),
            description=f"Introduced off-by-one error in range()",
        )

    def _mutate_wrong_operator(self, code: str) -> Optional[BugMutation]:
        """Replace an arithmetic operator with a wrong one."""
        operators = {
            '+': ['-', '*'],
            '-': ['+', '*'],
            '*': ['+', '/'],
            '/': ['*', '//'],
            '//': ['/', '*'],
        }

        # Find operators
        matches = []
        for op in operators.keys():
            for m in re.finditer(re.escape(op), code):
                # Avoid operators in strings
                pos = m.start()
                if not self._in_string(code, pos):
                    matches.append((pos, op))

        if not matches:
            return None

        pos, op = self.rng.choice(matches)
        new_op = self.rng.choice(operators[op])

        buggy = code[:pos] + new_op + code[pos + len(op):]

        line_num = code[:pos].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.WRONG_OPERATOR,
            bug_location=(line_num, pos - code[:pos].rfind('\n') - 1),
            description=f"Changed '{op}' to '{new_op}'",
        )

    def _mutate_wrong_comparison(self, code: str) -> Optional[BugMutation]:
        """Replace a comparison operator with a wrong one."""
        comparisons = {
            '==': ['!=', 'is'],
            '!=': ['==', 'is not'],
            '<': ['<=', '>'],
            '>': ['>=', '<'],
            '<=': ['<', '>='],
            '>=': ['>', '<='],
        }

        # Find comparison operators
        matches = []
        for op in sorted(comparisons.keys(), key=len, reverse=True):
            for m in re.finditer(re.escape(op), code):
                pos = m.start()
                if not self._in_string(code, pos):
                    matches.append((pos, op, len(op)))

        if not matches:
            return None

        pos, op, length = self.rng.choice(matches)
        new_op = self.rng.choice(comparisons[op])

        buggy = code[:pos] + new_op + code[pos + length:]

        line_num = code[:pos].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.WRONG_COMPARISON,
            bug_location=(line_num, pos - code[:pos].rfind('\n') - 1),
            description=f"Changed '{op}' to '{new_op}'",
        )

    def _mutate_mutable_default(self, code: str) -> Optional[BugMutation]:
        """Change None default to mutable default."""
        # Find function definitions with None defaults
        pattern = r'(def\s+\w+\s*\([^)]*\w+)\s*=\s*None([^)]*\))'
        matches = list(re.finditer(pattern, code))

        if not matches:
            return None

        match = self.rng.choice(matches)

        # Replace None with []
        buggy = code[:match.start()] + match.group(1) + '=[]' + match.group(2) + code[match.end():]

        line_num = code[:match.start()].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.MUTABLE_DEFAULT,
            bug_location=(line_num, 0),
            description="Changed None default to mutable []",
        )

    def _mutate_bare_except(self, code: str) -> Optional[BugMutation]:
        """Change specific except to bare except."""
        pattern = r'except\s+(\w+)(\s+as\s+\w+)?:'
        matches = list(re.finditer(pattern, code))

        if not matches:
            return None

        match = self.rng.choice(matches)

        buggy = code[:match.start()] + 'except:' + code[match.end():]

        line_num = code[:match.start()].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.BARE_EXCEPT,
            bug_location=(line_num, 0),
            description=f"Changed 'except {match.group(1)}' to bare except",
        )

    # ===== Tier 3: Style Mutations =====

    def _mutate_naming(self, code: str) -> Optional[BugMutation]:
        """Change snake_case function name to camelCase."""
        pattern = r'def\s+([a-z]+_[a-z_]+)\s*\('
        matches = list(re.finditer(pattern, code))

        if not matches:
            return None

        match = self.rng.choice(matches)
        name = match.group(1)

        # Convert to camelCase
        parts = name.split('_')
        camel = parts[0] + ''.join(p.capitalize() for p in parts[1:])

        buggy = code[:match.start(1)] + camel + code[match.end(1):]

        # Also replace other occurrences
        buggy = buggy.replace(name, camel)

        line_num = code[:match.start()].count('\n') + 1

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.NAMING_CONVENTION,
            bug_location=(line_num, 0),
            description=f"Changed '{name}' to camelCase '{camel}'",
        )

    def _mutate_add_unused_import(self, code: str) -> Optional[BugMutation]:
        """Add an unused import."""
        unused_imports = [
            'import os',
            'import sys',
            'import json',
            'import re',
            'import math',
            'from collections import defaultdict',
            'from typing import List',
        ]

        # Find first non-docstring line
        lines = code.split('\n')
        insert_line = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                insert_line = i
                break

        # Choose an import not already present
        available = [imp for imp in unused_imports if imp not in code]
        if not available:
            return None

        import_stmt = self.rng.choice(available)

        lines.insert(insert_line, import_stmt)
        buggy = '\n'.join(lines)

        return BugMutation(
            buggy_code=buggy,
            fixed_code=code,
            bug_type=BugType.UNUSED_IMPORT,
            bug_location=(insert_line + 1, 0),
            description=f"Added unused import: {import_stmt}",
        )

    def _in_string(self, code: str, pos: int) -> bool:
        """Check if position is inside a string literal."""
        # Simple check - count quotes before position
        before = code[:pos]
        single_quotes = before.count("'") - before.count("\\'")
        double_quotes = before.count('"') - before.count('\\"')

        return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)
