"""
Lineage Tracker module for ML Research.

Tracks the evolutionary relationships between ML methods,
including predecessors, successors, and influence graphs.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from .taxonomy import MLMethod, MethodLineage


@dataclass
class LineageNode:
    """
    A node in the method lineage graph.

    Attributes:
        method_id: Unique identifier of the method.
        name: Human-readable name.
        year: Year of introduction.
        lineages: List of lineages this method belongs to.
        predecessors: Set of method_ids this method builds upon.
        successors: Set of method_ids that build upon this method.
    """
    method_id: str
    name: str
    year: int
    lineages: List[MethodLineage]
    predecessors: Set[str]
    successors: Set[str]


@dataclass
class LineageInfo:
    """
    Information about a specific lineage.

    Attributes:
        lineage: The MethodLineage enum value.
        description: Human-readable description.
        founding_method: The method_id of the lineage's founding method.
        methods: List of method_ids in this lineage.
    """
    lineage: MethodLineage
    description: str
    founding_method: Optional[str]
    methods: List[str]


class LineageTracker:
    """
    Tracks evolutionary relationships between ML methods.

    Manages a graph of method relationships, tracking how methods
    build upon predecessors and influence successors.

    Example:
        tracker = LineageTracker()
        tracker.add_method(lstm_method)
        predecessors = tracker.get_predecessors("lstm_1997")
        graph = tracker.get_influence_graph()
    """

    LINEAGE_DESCRIPTIONS: Dict[MethodLineage, str] = {
        MethodLineage.PERCEPTRON_LINE: (
            "From the Perceptron (1958) through MLPs to modern deep networks. "
            "The foundational line of feedforward neural networks."
        ),
        MethodLineage.CNN_LINE: (
            "From Neocognitron (1980) through LeNet to modern architectures like "
            "ResNet and EfficientNet. The visual processing lineage."
        ),
        MethodLineage.RNN_LINE: (
            "From simple recurrent networks through LSTM and GRU to modern "
            "sequence models. The sequential processing lineage."
        ),
        MethodLineage.ATTENTION_LINE: (
            "From early attention mechanisms through self-attention to "
            "Transformers and beyond. The attention-based processing lineage."
        ),
        MethodLineage.GENERATIVE_LINE: (
            "From RBMs through VAEs and GANs to diffusion models. "
            "The generative modeling lineage."
        ),
        MethodLineage.RL_LINE: (
            "From Q-learning through DQN to PPO and RLHF. "
            "The reinforcement learning lineage."
        ),
    }

    def __init__(self) -> None:
        """Initialize an empty lineage tracker."""
        self._methods: Dict[str, MLMethod] = {}
        self._nodes: Dict[str, LineageNode] = {}

    def add_method(self, method: MLMethod) -> None:
        """
        Add a method to the lineage tracker.

        Automatically updates predecessor/successor relationships.

        Args:
            method: The MLMethod to add.
        """
        self._methods[method.method_id] = method

        # Create or update node
        node = LineageNode(
            method_id=method.method_id,
            name=method.name,
            year=method.year,
            lineages=method.lineages,
            predecessors=set(method.predecessors),
            successors=set(method.successors)
        )
        self._nodes[method.method_id] = node

        # Update predecessor nodes to include this as a successor
        for pred_id in method.predecessors:
            if pred_id in self._nodes:
                self._nodes[pred_id].successors.add(method.method_id)

        # Update successor nodes to include this as a predecessor
        for succ_id in method.successors:
            if succ_id in self._nodes:
                self._nodes[succ_id].predecessors.add(method.method_id)

    def remove_method(self, method_id: str) -> bool:
        """
        Remove a method from the lineage tracker.

        Also cleans up references in related methods.

        Args:
            method_id: The unique identifier of the method to remove.

        Returns:
            True if removed, False if not found.
        """
        if method_id not in self._nodes:
            return False

        node = self._nodes[method_id]

        # Remove from predecessors' successor lists
        for pred_id in node.predecessors:
            if pred_id in self._nodes:
                self._nodes[pred_id].successors.discard(method_id)

        # Remove from successors' predecessor lists
        for succ_id in node.successors:
            if succ_id in self._nodes:
                self._nodes[succ_id].predecessors.discard(method_id)

        del self._nodes[method_id]
        del self._methods[method_id]
        return True

    def show_lineage(self, lineage: MethodLineage) -> LineageInfo:
        """
        Get detailed information about a specific lineage.

        Args:
            lineage: The MethodLineage to display.

        Returns:
            LineageInfo with description and methods in this lineage.
        """
        methods_in_lineage = [
            m.method_id
            for m in self._methods.values()
            if lineage in m.lineages
        ]

        # Sort by year to find founding method
        methods_sorted = sorted(
            [(m.method_id, m.year) for m in self._methods.values() if lineage in m.lineages],
            key=lambda x: x[1]
        )

        founding = methods_sorted[0][0] if methods_sorted else None

        return LineageInfo(
            lineage=lineage,
            description=self.LINEAGE_DESCRIPTIONS.get(lineage, ""),
            founding_method=founding,
            methods=methods_in_lineage
        )

    def get_predecessors(
        self,
        method_id: str,
        *,
        depth: Optional[int] = None,
        include_indirect: bool = True
    ) -> List[str]:
        """
        Get predecessors of a method.

        Args:
            method_id: The method to find predecessors for.
            depth: Maximum depth to traverse (None for unlimited).
            include_indirect: If True, include transitive predecessors.

        Returns:
            List of method_ids that are predecessors.
        """
        if method_id not in self._nodes:
            return []

        if not include_indirect:
            return list(self._nodes[method_id].predecessors)

        visited: Set[str] = set()
        to_visit = [(p, 1) for p in self._nodes[method_id].predecessors]

        while to_visit:
            current_id, current_depth = to_visit.pop(0)

            if current_id in visited:
                continue
            if depth is not None and current_depth > depth:
                continue

            visited.add(current_id)

            if current_id in self._nodes:
                for pred_id in self._nodes[current_id].predecessors:
                    if pred_id not in visited:
                        to_visit.append((pred_id, current_depth + 1))

        return list(visited)

    def get_successors(
        self,
        method_id: str,
        *,
        depth: Optional[int] = None,
        include_indirect: bool = True
    ) -> List[str]:
        """
        Get successors of a method.

        Args:
            method_id: The method to find successors for.
            depth: Maximum depth to traverse (None for unlimited).
            include_indirect: If True, include transitive successors.

        Returns:
            List of method_ids that are successors.
        """
        if method_id not in self._nodes:
            return []

        if not include_indirect:
            return list(self._nodes[method_id].successors)

        visited: Set[str] = set()
        to_visit = [(s, 1) for s in self._nodes[method_id].successors]

        while to_visit:
            current_id, current_depth = to_visit.pop(0)

            if current_id in visited:
                continue
            if depth is not None and current_depth > depth:
                continue

            visited.add(current_id)

            if current_id in self._nodes:
                for succ_id in self._nodes[current_id].successors:
                    if succ_id not in visited:
                        to_visit.append((succ_id, current_depth + 1))

        return list(visited)

    def get_influence_graph(
        self,
        *,
        lineage: Optional[MethodLineage] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Get the full influence graph of methods.

        Args:
            lineage: Optional filter for a specific lineage.

        Returns:
            Dictionary mapping method_ids to their relationships:
            {
                "method_id": {
                    "name": str,
                    "year": int,
                    "predecessors": List[str],
                    "successors": List[str]
                }
            }
        """
        graph: Dict[str, Dict[str, any]] = {}

        for method_id, node in self._nodes.items():
            if lineage is not None and lineage not in node.lineages:
                continue

            graph[method_id] = {
                "name": node.name,
                "year": node.year,
                "predecessors": list(node.predecessors),
                "successors": list(node.successors)
            }

        return graph

    def get_lineage_chain(
        self,
        lineage: MethodLineage
    ) -> List[Tuple[str, str, int]]:
        """
        Get a chronological chain of methods in a lineage.

        Args:
            lineage: The lineage to get the chain for.

        Returns:
            List of (method_id, name, year) tuples sorted by year.
        """
        methods_in_lineage = [
            (m.method_id, m.name, m.year)
            for m in self._methods.values()
            if lineage in m.lineages
        ]
        methods_in_lineage.sort(key=lambda x: x[2])
        return methods_in_lineage

    def find_common_ancestors(
        self,
        method_id_1: str,
        method_id_2: str
    ) -> List[str]:
        """
        Find common ancestors between two methods.

        Args:
            method_id_1: First method's identifier.
            method_id_2: Second method's identifier.

        Returns:
            List of method_ids that are ancestors of both methods.
        """
        ancestors_1 = set(self.get_predecessors(method_id_1))
        ancestors_2 = set(self.get_predecessors(method_id_2))
        return list(ancestors_1.intersection(ancestors_2))

    def find_common_descendants(
        self,
        method_id_1: str,
        method_id_2: str
    ) -> List[str]:
        """
        Find common descendants of two methods.

        Args:
            method_id_1: First method's identifier.
            method_id_2: Second method's identifier.

        Returns:
            List of method_ids that are descendants of both methods.
        """
        descendants_1 = set(self.get_successors(method_id_1))
        descendants_2 = set(self.get_successors(method_id_2))
        return list(descendants_1.intersection(descendants_2))

    def get_influence_score(self, method_id: str) -> int:
        """
        Calculate an influence score based on successor count.

        Args:
            method_id: The method to score.

        Returns:
            Number of direct and indirect successors.
        """
        return len(self.get_successors(method_id, include_indirect=True))
