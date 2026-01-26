"""
Variable Extractor for RLM (Recursive Language Model)

Extracts variables, types, and constraints from natural language specifications
for systematic code synthesis through problem decomposition.

This module provides:
- VariableType enum with comprehensive Python type mappings
- ExtractedVariable dataclass for structured variable representation
- VariableExtractor class for parsing specifications into variables
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Set
import re


class VariableType(Enum):
    """
    Comprehensive types for extracted variables.

    Maps to Python types for code generation.
    """
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    SET = "set"
    TUPLE = "tuple"
    OBJECT = "object"
    FUNCTION = "function"
    CLASS = "class"
    NONE = "none"
    UNKNOWN = "unknown"

    def to_python_type(self) -> str:
        """Convert to Python type annotation string."""
        mapping = {
            VariableType.INTEGER: "int",
            VariableType.FLOAT: "float",
            VariableType.STRING: "str",
            VariableType.BOOLEAN: "bool",
            VariableType.LIST: "List",
            VariableType.DICT: "Dict",
            VariableType.SET: "Set",
            VariableType.TUPLE: "Tuple",
            VariableType.OBJECT: "object",
            VariableType.FUNCTION: "Callable",
            VariableType.CLASS: "Type",
            VariableType.NONE: "None",
            VariableType.UNKNOWN: "Any",
        }
        return mapping.get(self, "Any")

    def default_value(self) -> str:
        """Return default value for this type as code string."""
        defaults = {
            VariableType.INTEGER: "0",
            VariableType.FLOAT: "0.0",
            VariableType.STRING: '""',
            VariableType.BOOLEAN: "False",
            VariableType.LIST: "[]",
            VariableType.DICT: "{}",
            VariableType.SET: "set()",
            VariableType.TUPLE: "()",
            VariableType.NONE: "None",
        }
        return defaults.get(self, "None")


@dataclass
class ExtractedVariable:
    """
    A variable extracted from natural language specification.

    Attributes:
        name: Python-valid variable name
        var_type: Inferred variable type
        description: Human-readable description
        constraints: List of constraint strings (e.g., "min_inclusive:0")
        dependencies: Names of other variables this depends on
        is_input: Whether this is an input parameter
        is_output: Whether this is a return value
        source_text: Original text this was extracted from
        element_type: For collections, the type of elements
        confidence: Confidence score for type inference (0.0-1.0)
    """
    name: str
    var_type: VariableType
    description: str
    constraints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    source_text: Optional[str] = None
    element_type: Optional["VariableType"] = None
    confidence: float = 0.5

    def to_python_annotation(self) -> str:
        """Generate Python type annotation."""
        base_type = self.var_type.to_python_type()

        if self.var_type in (VariableType.LIST, VariableType.SET):
            if self.element_type:
                return f"{base_type}[{self.element_type.to_python_type()}]"
            return f"{base_type}[Any]"

        if self.var_type == VariableType.DICT:
            return "Dict[str, Any]"

        if self.var_type == VariableType.TUPLE:
            return "Tuple[Any, ...]"

        return base_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.var_type.value,
            "description": self.description,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "is_input": self.is_input,
            "is_output": self.is_output,
            "source_text": self.source_text,
            "element_type": self.element_type.value if self.element_type else None,
            "confidence": self.confidence,
        }


class VariableExtractor:
    """
    Extracts variables, types, and constraints from natural language specifications.

    Uses pattern matching and heuristics to identify:
    - Input parameters and their types
    - Output/return values
    - Intermediate variables
    - Type constraints and relationships

    Example:
        >>> extractor = VariableExtractor()
        >>> spec = "Create a function that takes a list of integers and returns the sum of all even numbers."
        >>> variables = extractor.extract(spec)
        >>> for v in variables:
        ...     print(f"{v.name}: {v.var_type.value} (input={v.is_input}, output={v.is_output})")
        numbers: list (input=True, output=False)
        result: integer (input=False, output=True)
    """

    # Type inference patterns - ordered by specificity
    TYPE_PATTERNS: Dict[VariableType, List[str]] = {
        VariableType.INTEGER: [
            r'\binteger[s]?\b', r'\bint[s]?\b', r'\bwhole\s+number[s]?\b',
            r'\bcount[s]?\b', r'\bindex(?:es|ices)?\b', r'\bposition[s]?\b',
            r'\blength[s]?\b', r'\bsize[s]?\b', r'\bage[s]?\b',
        ],
        VariableType.FLOAT: [
            r'\bfloat(?:ing)?(?:\s+point)?\s*(?:number)?[s]?\b', r'\bdecimal[s]?\b', r'\breal\s+number[s]?\b',
            r'\bpercentage[s]?\b', r'\bratio[s]?\b', r'\bproportion[s]?\b',
            r'\baverage[s]?\b', r'\bmean[s]?\b', r'\bprice[s]?\b', r'\bcost[s]?\b',
            r'\bdouble[s]?\b',
        ],
        VariableType.STRING: [
            r'\bstring[s]?\b', r'\btext[s]?\b', r'\bword[s]?\b',
            r'\bcharacter[s]?\b', r'\bchar[s]?\b', r'\bname[s]?\b',
            r'\bmessage[s]?\b', r'\blabel[s]?\b', r'\btitle[s]?\b',
        ],
        VariableType.LIST: [
            r'\blist[s]?\b', r'\barray[s]?\b', r'\bsequence[s]?\b',
            r'\bcollection[s]?\b', r'\belements?\b', r'\bitems?\b',
            r'\bvector[s]?\b',
        ],
        VariableType.DICT: [
            r'\bdictionary\b', r'\bdict[s]?\b', r'\bmap[s]?\b',
            r'\bhashmap[s]?\b', r'\bhash\s+table[s]?\b', r'\bkey[s]?\s*[-/]?\s*value[s]?\b',
            r'\bmapping[s]?\b', r'\blookup[s]?\b',
        ],
        VariableType.SET: [
            r'\bset[s]?\b', r'\bunique\s+(?:elements?|items?|values?)\b',
            r'\bdistinct\s+(?:elements?|items?|values?)\b',
        ],
        VariableType.BOOLEAN: [
            r'\bbool(?:ean)?[s]?\b', r'\btrue\s+(?:or|if|when)\s+false\b',
            r'\btrue\s+if\b', r'\bfalse\s+otherwise\b',
            r'\bflag[s]?\b', r'\bis\s+(?:valid|empty|present|found)\b',
            r'\bcheck[s]?\s+(?:if|whether)\b', r'\bexist[s]?\b',
        ],
        VariableType.TUPLE: [
            r'\btuple[s]?\b', r'\bpair[s]?\b', r'\bcoordinate[s]?\b',
            r'\bpoint[s]?\b',
        ],
        VariableType.FUNCTION: [
            r'\bfunction[s]?\b', r'\bcallback[s]?\b', r'\blambda[s]?\b',
            r'\bpredicate[s]?\b', r'\bcomparator[s]?\b',
        ],
    }

    # Constraint patterns - (regex, constraint_type)
    CONSTRAINT_PATTERNS: List[Tuple[str, str]] = [
        (r'greater\s+than\s+(\d+(?:\.\d+)?)', 'min_exclusive'),
        (r'less\s+than\s+(\d+(?:\.\d+)?)', 'max_exclusive'),
        (r'at\s+least\s+(\d+(?:\.\d+)?)', 'min_inclusive'),
        (r'at\s+most\s+(\d+(?:\.\d+)?)', 'max_inclusive'),
        (r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)', 'range'),
        (r'minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)', 'min_inclusive'),
        (r'maximum\s+(?:of\s+)?(\d+(?:\.\d+)?)', 'max_inclusive'),
        (r'>=\s*(\d+(?:\.\d+)?)', 'min_inclusive'),
        (r'<=\s*(\d+(?:\.\d+)?)', 'max_inclusive'),
        (r'>\s*(\d+(?:\.\d+)?)', 'min_exclusive'),
        (r'<\s*(\d+(?:\.\d+)?)', 'max_exclusive'),
        (r'\bnon[-\s]?negative\b', 'non_negative'),
        (r'\bpositive\b', 'positive'),
        (r'\bnon[-\s]?empty\b', 'non_empty'),
        (r'\bunique\b', 'unique'),
        (r'\bsorted\b', 'sorted'),
        (r'\bdistinct\b', 'unique'),
        (r'\bascending\b', 'sorted_asc'),
        (r'\bdescending\b', 'sorted_desc'),
        (r'\bnot\s+null\b', 'not_null'),
        (r'\brequired\b', 'required'),
        (r'\boptional\b', 'optional'),
    ]

    # Input identification patterns
    INPUT_PATTERNS: List[str] = [
        r'takes?\s+(?:a\s+|an\s+)?(.+?)(?:\s+and\s+|\s*,\s*|\s+as\s+|\s*$)',
        r'given\s+(?:a\s+|an\s+)?(.+?)(?:\s+and\s+|\s*,\s*|\.|\s*$)',
        r'input[s]?\s*(?:is|are|:)\s*(.+?)(?:\.|,|$)',
        r'accepts?\s+(?:a\s+|an\s+)?(.+?)(?:\s+and\s+|\s*,\s*|\.|\s*$)',
        r'receives?\s+(?:a\s+|an\s+)?(.+?)(?:\s+and\s+|\s*,\s*|\.|\s*$)',
        r'with\s+(?:a\s+|an\s+)?(.+?)\s+as\s+(?:input|parameter)',
        r'parameter[s]?\s*(?:is|are|:)\s*(.+?)(?:\.|,|$)',
        r'argument[s]?\s*(?:is|are|:)\s*(.+?)(?:\.|,|$)',
        # Handle numbered list items like "1. Takes a list of strings"
        r'\d+\.\s*takes?\s+(?:a\s+|an\s+)?(.+?)(?:\s*$)',
    ]

    # Output identification patterns
    OUTPUT_PATTERNS: List[str] = [
        r'returns?\s+(?:the\s+)?(.+?)(?:\.|,|$)',
        r'output[s]?\s*(?:is|are|:)\s*(.+?)(?:\.|,|$)',
        r'produces?\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|,|$)',
        r'calculates?\s+(?:the\s+)?(.+?)(?:\.|,|$)',
        r'computes?\s+(?:the\s+)?(.+?)(?:\.|,|$)',
        r'generates?\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|,|$)',
        r'yields?\s+(?:the\s+)?(.+?)(?:\.|,|$)',
        r'results?\s+(?:is|in|:)\s*(.+?)(?:\.|,|$)',
        r'finds?\s+(?:the\s+)?(.+?)(?:\.|,|$)',
        r'determines?\s+(?:the\s+)?(.+?)(?:\.|,|$)',
    ]

    # Words to filter out of entity names
    STOP_WORDS: Set[str] = {
        'a', 'an', 'the', 'of', 'for', 'to', 'from', 'with', 'by', 'in', 'on',
        'and', 'or', 'that', 'which', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'these', 'those', 'their', 'them',
    }

    def __init__(self, use_llm: bool = False):
        """
        Initialize the variable extractor.

        Args:
            use_llm: Whether to use LLM for enhanced extraction (not implemented)
        """
        self.use_llm = use_llm
        self._compiled_type_patterns: Dict[VariableType, List[re.Pattern]] = {}
        self._compiled_constraint_patterns: List[Tuple[re.Pattern, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for var_type, patterns in self.TYPE_PATTERNS.items():
            self._compiled_type_patterns[var_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._compiled_constraint_patterns = [
            (re.compile(p, re.IGNORECASE), ct)
            for p, ct in self.CONSTRAINT_PATTERNS
        ]

    def extract(self, specification: str) -> List[ExtractedVariable]:
        """
        Extract variables from a natural language specification.

        Args:
            specification: Natural language description of a programming task

        Returns:
            List of ExtractedVariable objects representing identified variables

        Example:
            >>> extractor = VariableExtractor()
            >>> spec = "Create a function that takes a list of integers and returns the sum of all even numbers."
            >>> variables = extractor.extract(spec)
        """
        variables: List[ExtractedVariable] = []
        seen_names: Set[str] = set()

        # 1. Extract input variables
        inputs = self._extract_inputs(specification)
        for entity in inputs:
            var = self._create_variable(entity, specification, is_input=True)
            if var.name not in seen_names:
                variables.append(var)
                seen_names.add(var.name)

        # 2. Extract output variables
        outputs = self._extract_outputs(specification)
        for entity in outputs:
            var = self._create_variable(entity, specification, is_output=True)
            if var.name not in seen_names:
                variables.append(var)
                seen_names.add(var.name)

        # 3. Extract intermediate variables
        intermediates = self._extract_intermediates(specification, variables)
        for var in intermediates:
            if var.name not in seen_names:
                variables.append(var)
                seen_names.add(var.name)

        # 4. Build dependency graph
        self._resolve_dependencies(variables, specification)

        # 5. Refine types based on context
        self._refine_types(variables, specification)

        return variables

    def _extract_inputs(self, text: str) -> List[str]:
        """Extract input variable descriptions from specification."""
        inputs: List[str] = []
        text_lower = text.lower()

        for pattern in self.INPUT_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Handle tuple matches (from groups)
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                cleaned = self._clean_entity(match)
                if cleaned and len(cleaned) > 1:
                    inputs.append(cleaned)

        # Deduplicate while preserving order
        seen = set()
        unique_inputs = []
        for inp in inputs:
            if inp not in seen:
                seen.add(inp)
                unique_inputs.append(inp)

        return unique_inputs

    def _extract_outputs(self, text: str) -> List[str]:
        """Extract output variable descriptions from specification."""
        outputs: List[str] = []
        text_lower = text.lower()

        for pattern in self.OUTPUT_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                cleaned = self._clean_entity(match)
                if cleaned and len(cleaned) > 1:
                    outputs.append(cleaned)

        # Deduplicate
        seen = set()
        unique_outputs = []
        for out in outputs:
            if out not in seen:
                seen.add(out)
                unique_outputs.append(out)

        return unique_outputs

    def _clean_entity(self, entity: str) -> str:
        """Clean and normalize an extracted entity."""
        if not entity:
            return ""

        # Remove extra whitespace
        entity = re.sub(r'\s+', ' ', entity.strip())

        # Remove trailing punctuation
        entity = entity.rstrip('.,;:!?')

        # Limit length
        if len(entity) > 100:
            entity = entity[:100]

        return entity

    def _create_variable(
        self,
        entity: str,
        context: str,
        is_input: bool = False,
        is_output: bool = False
    ) -> ExtractedVariable:
        """Create an ExtractedVariable from an entity description."""
        var_type, element_type, confidence = self._infer_type(entity, context)
        name = self._generate_name(entity, var_type)
        constraints = self._extract_constraints(entity, context)

        return ExtractedVariable(
            name=name,
            var_type=var_type,
            description=entity,
            constraints=constraints,
            is_input=is_input,
            is_output=is_output,
            source_text=entity,
            element_type=element_type,
            confidence=confidence,
        )

    def _infer_type(
        self,
        entity: str,
        context: str
    ) -> Tuple[VariableType, Optional[VariableType], float]:
        """
        Infer variable type from entity and context.

        Returns:
            Tuple of (primary_type, element_type, confidence)
        """
        entity_lower = entity.lower()
        text = f"{entity} {context}".lower()

        type_scores: Dict[VariableType, float] = {}
        element_type: Optional[VariableType] = None

        # First, check for "container of elements" pattern in entity
        # This should override simple keyword matching
        container_of_pattern = re.compile(
            r'\b(list|array|collection|set|sequence|tuple)\s+of\s+(\w+)',
            re.IGNORECASE
        )
        container_match = container_of_pattern.search(entity_lower)

        if container_match:
            # Matched "list of X" pattern - prioritize container type
            container_word = container_match.group(1).lower()
            element_word = container_match.group(2).lower()

            # Map container word to type
            container_type_map = {
                'list': VariableType.LIST,
                'array': VariableType.LIST,
                'collection': VariableType.LIST,
                'sequence': VariableType.LIST,
                'set': VariableType.SET,
                'tuple': VariableType.TUPLE,
            }
            primary_type = container_type_map.get(container_word, VariableType.LIST)

            # Infer element type
            for etype, patterns in self._compiled_type_patterns.items():
                if etype in (VariableType.LIST, VariableType.DICT, VariableType.SET, VariableType.TUPLE):
                    continue  # Skip container types as element types
                for pattern in patterns:
                    if pattern.search(element_word):
                        element_type = etype
                        break
                if element_type:
                    break

            return primary_type, element_type, 0.9

        # Standard scoring for non-container patterns
        for var_type, patterns in self._compiled_type_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Higher weight for matches in entity vs context
                    entity_matches = pattern.findall(entity_lower)
                    # Give extra weight if match is at start of entity
                    position_bonus = 1.5 if entity_matches and pattern.search(entity_lower[:20]) else 1.0
                    score += len(entity_matches) * 2.0 * position_bonus + len(matches) * 0.5
            if score > 0:
                type_scores[var_type] = score

        if not type_scores:
            return VariableType.UNKNOWN, None, 0.3

        # Get highest scoring type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        primary_type = best_type[0]
        confidence = min(0.95, 0.5 + best_type[1] * 0.1)

        # For collections detected via scoring, try to infer element type
        if primary_type in (VariableType.LIST, VariableType.SET, VariableType.TUPLE):
            of_pattern = re.compile(
                r'\b(?:list|array|collection|set|sequence)\s+of\s+(\w+)',
                re.IGNORECASE
            )
            match = of_pattern.search(text)
            if match:
                element_text = match.group(1)
                for etype, patterns in self._compiled_type_patterns.items():
                    if etype in (VariableType.LIST, VariableType.DICT, VariableType.SET):
                        continue
                    for pattern in patterns:
                        if pattern.search(element_text):
                            element_type = etype
                            break
                    if element_type:
                        break

        return primary_type, element_type, confidence

    def _generate_name(self, entity: str, var_type: VariableType) -> str:
        """Generate a valid Python variable name from entity description."""
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', entity.lower())

        # Filter stop words
        words = [w for w in words if w not in self.STOP_WORDS]

        # Take first 3 meaningful words max
        words = words[:3]

        if words:
            name = '_'.join(words)
        else:
            # Fallback based on type
            type_defaults = {
                VariableType.INTEGER: "num",
                VariableType.FLOAT: "value",
                VariableType.STRING: "text",
                VariableType.LIST: "items",
                VariableType.DICT: "data",
                VariableType.SET: "unique_items",
                VariableType.BOOLEAN: "flag",
                VariableType.TUPLE: "pair",
                VariableType.FUNCTION: "func",
            }
            name = type_defaults.get(var_type, "var")

        # Ensure valid Python identifier
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')

        # Ensure doesn't start with number
        if name and name[0].isdigit():
            name = f"var_{name}"

        # Ensure not empty
        if not name:
            name = "var"

        return name

    def _extract_constraints(self, entity: str, context: str) -> List[str]:
        """Extract constraints for a variable from entity and context."""
        text = f"{entity} {context}".lower()
        constraints: List[str] = []

        for pattern, constraint_type in self._compiled_constraint_patterns:
            match = pattern.search(text)
            if match:
                if match.groups():
                    # Has captured groups - include values
                    values = ':'.join(match.groups())
                    constraints.append(f"{constraint_type}:{values}")
                else:
                    # No captured groups - just the constraint type
                    constraints.append(constraint_type)

        return constraints

    def _extract_intermediates(
        self,
        text: str,
        existing: List[ExtractedVariable]
    ) -> List[ExtractedVariable]:
        """Extract intermediate variables from processing steps."""
        intermediates: List[ExtractedVariable] = []
        existing_names = {v.name for v in existing}

        # Look for step indicators
        step_patterns = [
            (r'(\d+)\.\s*(.+?)(?=\d+\.|$)', 'numbered'),
            (r'first[,]?\s+(.+?)(?:then|next|finally|and\s+then|$)', 'first'),
            (r'then[,]?\s+(.+?)(?:then|next|finally|and\s+then|$)', 'then'),
            (r'next[,]?\s+(.+?)(?:then|next|finally|and\s+then|$)', 'next'),
            (r'finally[,]?\s+(.+?)(?:$)', 'finally'),
            (r'after\s+(?:that|this)[,]?\s+(.+?)(?:then|next|finally|$)', 'after'),
        ]

        # Intermediate result indicators
        intermediate_keywords = [
            'result', 'filtered', 'converted', 'transformed', 'processed',
            'calculated', 'computed', 'intermediate', 'temporary', 'temp',
            'sorted', 'grouped', 'mapped', 'reduced',
        ]

        for pattern, step_type in step_patterns:
            matches = re.findall(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            for match in matches:
                step_text = match[-1] if isinstance(match, tuple) else match
                step_text = step_text.strip()

                # Check if this step produces an intermediate result
                has_intermediate = any(kw in step_text for kw in intermediate_keywords)

                if has_intermediate and len(step_text) > 5:
                    var = self._create_variable(step_text[:80], text)

                    # Generate unique name
                    base_name = f"intermediate_{step_type}"
                    name = base_name
                    counter = 1
                    while name in existing_names:
                        name = f"{base_name}_{counter}"
                        counter += 1

                    var.name = name
                    var.is_input = False
                    var.is_output = False

                    intermediates.append(var)
                    existing_names.add(name)

        return intermediates

    def _resolve_dependencies(
        self,
        variables: List[ExtractedVariable],
        context: str
    ) -> None:
        """Resolve dependencies between variables based on specification."""
        input_vars = [v for v in variables if v.is_input]
        output_vars = [v for v in variables if v.is_output]
        intermediate_vars = [v for v in variables if not v.is_input and not v.is_output]

        input_names = [v.name for v in input_vars]

        # Outputs depend on inputs by default
        for out_var in output_vars:
            for inp_name in input_names:
                if inp_name not in out_var.dependencies:
                    out_var.dependencies.append(inp_name)

        # Intermediate variables
        # - Depend on inputs
        # - Outputs depend on them
        for inter_var in intermediate_vars:
            for inp_name in input_names:
                if inp_name not in inter_var.dependencies:
                    inter_var.dependencies.append(inp_name)

            # Add intermediate to output dependencies
            for out_var in output_vars:
                if inter_var.name not in out_var.dependencies:
                    out_var.dependencies.append(inter_var.name)

        # Look for explicit dependency indicators in context
        dep_patterns = [
            r'(\w+)\s+depends?\s+on\s+(\w+)',
            r'(\w+)\s+uses?\s+(\w+)',
            r'(\w+)\s+from\s+(\w+)',
            r'using\s+(\w+)\s+(?:to\s+)?(?:compute|calculate|get)\s+(\w+)',
        ]

        var_names = {v.name for v in variables}
        for pattern in dep_patterns:
            matches = re.findall(pattern, context.lower())
            for dependent, dependency in matches:
                # Try to match to actual variable names
                for var in variables:
                    if dependent in var.name.lower() or var.name.lower() in dependent:
                        for dep_var in variables:
                            if dependency in dep_var.name.lower() or dep_var.name.lower() in dependency:
                                if dep_var.name not in var.dependencies:
                                    var.dependencies.append(dep_var.name)

    def _refine_types(
        self,
        variables: List[ExtractedVariable],
        context: str
    ) -> None:
        """Refine variable types based on relationships and context."""
        # If an output is a sum/count/length of a list, it should be numeric
        numeric_operations = ['sum', 'count', 'length', 'average', 'mean', 'total']

        for var in variables:
            if var.is_output:
                # Check if description mentions numeric operations
                desc_lower = var.description.lower()
                if any(op in desc_lower for op in numeric_operations):
                    if var.var_type == VariableType.UNKNOWN:
                        var.var_type = VariableType.INTEGER
                        var.confidence = 0.7

        # If a variable has "even" or "odd" in context, likely integer
        for var in variables:
            if var.var_type == VariableType.UNKNOWN:
                if 'even' in var.description.lower() or 'odd' in var.description.lower():
                    var.var_type = VariableType.INTEGER
                    var.confidence = 0.6

        # List element types from common patterns
        for var in variables:
            if var.var_type == VariableType.LIST and not var.element_type:
                desc_lower = var.description.lower()
                if any(p in desc_lower for p in ['integer', 'int', 'number', 'digit']):
                    var.element_type = VariableType.INTEGER
                elif any(p in desc_lower for p in ['string', 'word', 'text', 'name']):
                    var.element_type = VariableType.STRING
                elif any(p in desc_lower for p in ['float', 'decimal', 'real']):
                    var.element_type = VariableType.FLOAT
