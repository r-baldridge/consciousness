"""
Timeline module for ML Research.

Provides historical timeline functionality for visualizing and
querying the chronological development of ML methods.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .taxonomy import MLMethod, MethodEra


@dataclass
class TimelineEntry:
    """
    A single entry in the ML timeline.

    Attributes:
        year: Year of the method's introduction.
        method_id: Unique identifier of the method.
        name: Human-readable name of the method.
        era: Historical era of the method.
        key_innovation: Brief description of the innovation.
    """
    year: int
    method_id: str
    name: str
    era: MethodEra
    key_innovation: str


class Timeline:
    """
    Historical timeline of ML methods.

    Manages a chronological view of ML method development,
    supporting queries by era, year range, and chronological ordering.

    Example:
        timeline = Timeline()
        timeline.add_method(transformer_method)
        entries = timeline.get_methods_by_year_range(2015, 2020)
    """

    # Era definitions with year ranges
    ERA_RANGES: Dict[MethodEra, Tuple[int, int]] = {
        MethodEra.FOUNDATIONAL: (1943, 1979),
        MethodEra.CLASSICAL: (1980, 2005),
        MethodEra.DEEP_LEARNING: (2006, 2016),
        MethodEra.ATTENTION: (2017, 2022),
        MethodEra.NOVEL: (2023, 2100),  # Open-ended for future
    }

    ERA_DESCRIPTIONS: Dict[MethodEra, str] = {
        MethodEra.FOUNDATIONAL: "Birth of neural networks and early learning algorithms",
        MethodEra.CLASSICAL: "SVMs, ensemble methods, and the AI winter recovery",
        MethodEra.DEEP_LEARNING: "Deep belief networks, CNNs, and the deep learning revolution",
        MethodEra.ATTENTION: "Transformer architecture and attention-based models",
        MethodEra.NOVEL: "Emerging paradigms, multimodal AI, and foundation models",
    }

    def __init__(self) -> None:
        """Initialize an empty timeline."""
        self._methods: Dict[str, MLMethod] = {}

    def add_method(self, method: MLMethod) -> None:
        """
        Add a method to the timeline.

        Args:
            method: The MLMethod to add.
        """
        self._methods[method.method_id] = method

    def remove_method(self, method_id: str) -> bool:
        """
        Remove a method from the timeline.

        Args:
            method_id: The unique identifier of the method to remove.

        Returns:
            True if removed, False if not found.
        """
        if method_id in self._methods:
            del self._methods[method_id]
            return True
        return False

    def show_era(self, era: MethodEra) -> Dict[str, any]:
        """
        Get detailed information about a specific era.

        Args:
            era: The MethodEra to display.

        Returns:
            Dictionary containing era information:
                - name: Era enum name
                - value: Era string value
                - year_range: Tuple of (start_year, end_year)
                - description: Brief description of the era
                - methods: List of TimelineEntry for methods in this era
        """
        year_range = self.ERA_RANGES.get(era, (0, 9999))
        methods_in_era = [
            TimelineEntry(
                year=m.year,
                method_id=m.method_id,
                name=m.name,
                era=m.era,
                key_innovation=m.key_innovation
            )
            for m in self._methods.values()
            if m.era == era
        ]
        # Sort by year
        methods_in_era.sort(key=lambda x: x.year)

        return {
            "name": era.name,
            "value": era.value,
            "year_range": year_range,
            "description": self.ERA_DESCRIPTIONS.get(era, ""),
            "methods": methods_in_era,
            "method_count": len(methods_in_era)
        }

    def get_methods_by_year_range(
        self,
        start_year: int,
        end_year: int
    ) -> List[TimelineEntry]:
        """
        Get all methods within a year range.

        Args:
            start_year: Start of the range (inclusive).
            end_year: End of the range (inclusive).

        Returns:
            List of TimelineEntry for methods in the range, sorted by year.
        """
        entries = [
            TimelineEntry(
                year=m.year,
                method_id=m.method_id,
                name=m.name,
                era=m.era,
                key_innovation=m.key_innovation
            )
            for m in self._methods.values()
            if start_year <= m.year <= end_year
        ]
        entries.sort(key=lambda x: x.year)
        return entries

    def get_chronological_list(
        self,
        *,
        era: Optional[MethodEra] = None,
        reverse: bool = False
    ) -> List[TimelineEntry]:
        """
        Get a chronologically sorted list of all methods.

        Args:
            era: Optional filter for a specific era.
            reverse: If True, sort from newest to oldest.

        Returns:
            List of TimelineEntry sorted by year.
        """
        methods = self._methods.values()
        if era is not None:
            methods = [m for m in methods if m.era == era]

        entries = [
            TimelineEntry(
                year=m.year,
                method_id=m.method_id,
                name=m.name,
                era=m.era,
                key_innovation=m.key_innovation
            )
            for m in methods
        ]
        entries.sort(key=lambda x: x.year, reverse=reverse)
        return entries

    def get_era_for_year(self, year: int) -> Optional[MethodEra]:
        """
        Determine which era a given year belongs to.

        Args:
            year: The year to classify.

        Returns:
            The MethodEra for the year, or None if outside all ranges.
        """
        for era, (start, end) in self.ERA_RANGES.items():
            if start <= year <= end:
                return era
        return None

    def get_all_eras(self) -> List[Dict[str, any]]:
        """
        Get information about all eras.

        Returns:
            List of era information dictionaries, sorted chronologically.
        """
        eras = []
        for era in MethodEra:
            era_info = self.show_era(era)
            eras.append(era_info)
        return eras

    def get_method_count_by_era(self) -> Dict[MethodEra, int]:
        """
        Get count of methods per era.

        Returns:
            Dictionary mapping MethodEra to count of methods.
        """
        counts: Dict[MethodEra, int] = {era: 0 for era in MethodEra}
        for method in self._methods.values():
            counts[method.era] += 1
        return counts

    def get_method_count_by_year(self) -> Dict[int, int]:
        """
        Get count of methods per year.

        Returns:
            Dictionary mapping year to count of methods introduced that year.
        """
        counts: Dict[int, int] = {}
        for method in self._methods.values():
            counts[method.year] = counts.get(method.year, 0) + 1
        return counts

    def format_timeline_text(
        self,
        *,
        era: Optional[MethodEra] = None,
        max_entries: Optional[int] = None
    ) -> str:
        """
        Generate a text representation of the timeline.

        Args:
            era: Optional filter for a specific era.
            max_entries: Optional limit on number of entries.

        Returns:
            Formatted string representation of the timeline.
        """
        entries = self.get_chronological_list(era=era)
        if max_entries:
            entries = entries[:max_entries]

        if not entries:
            return "No methods in timeline."

        lines = ["ML Methods Timeline", "=" * 50]

        current_era = None
        for entry in entries:
            if entry.era != current_era:
                current_era = entry.era
                lines.append(f"\n--- {current_era.value.upper()} ERA ---")

            lines.append(f"  {entry.year}: {entry.name}")
            lines.append(f"         {entry.key_innovation[:60]}...")

        return "\n".join(lines)
