"""
Method Registry for ML Research.

Central registry for all ML methods, providing storage, retrieval,
and search capabilities across the method database.
"""

from typing import Dict, List, Optional
from .taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


class MethodRegistry:
    """
    Central registry for managing ML methods.

    Provides a unified interface for registering, retrieving, and
    searching ML methods by various criteria including era, category,
    lineage, and free-text search.

    Example:
        registry = MethodRegistry()
        registry.register(transformer_method)
        methods = registry.get_by_era(MethodEra.ATTENTION)
    """

    def __init__(self) -> None:
        """Initialize an empty method registry."""
        self._methods: Dict[str, MLMethod] = {}

    def register(self, method: MLMethod) -> None:
        """
        Register a new ML method in the registry.

        Args:
            method: The MLMethod instance to register.

        Raises:
            ValueError: If a method with the same method_id already exists.
        """
        if method.method_id in self._methods:
            raise ValueError(f"Method '{method.method_id}' is already registered")
        self._methods[method.method_id] = method

    def get(self, method_id: str) -> Optional[MLMethod]:
        """
        Retrieve a method by its unique identifier.

        Args:
            method_id: The unique identifier of the method.

        Returns:
            The MLMethod if found, None otherwise.
        """
        return self._methods.get(method_id)

    def get_all(self) -> List[MLMethod]:
        """
        Retrieve all registered methods.

        Returns:
            List of all MLMethod instances in the registry.
        """
        return list(self._methods.values())

    def get_by_era(self, era: MethodEra) -> List[MLMethod]:
        """
        Retrieve all methods from a specific historical era.

        Args:
            era: The MethodEra to filter by.

        Returns:
            List of MLMethod instances from the specified era.
        """
        return [m for m in self._methods.values() if m.era == era]

    def get_by_category(self, category: MethodCategory) -> List[MLMethod]:
        """
        Retrieve all methods of a specific category.

        Args:
            category: The MethodCategory to filter by.

        Returns:
            List of MLMethod instances of the specified category.
        """
        return [m for m in self._methods.values() if m.category == category]

    def get_by_lineage(self, lineage: MethodLineage) -> List[MLMethod]:
        """
        Retrieve all methods belonging to a specific lineage.

        Args:
            lineage: The MethodLineage to filter by.

        Returns:
            List of MLMethod instances belonging to the specified lineage.
        """
        return [m for m in self._methods.values() if lineage in m.lineages]

    def search(
        self,
        query: str,
        *,
        era: Optional[MethodEra] = None,
        category: Optional[MethodCategory] = None,
        lineage: Optional[MethodLineage] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> List[MLMethod]:
        """
        Search for methods matching given criteria.

        Performs a flexible search across method names, key innovations,
        authors, and tags. Can be combined with filters for era, category,
        lineage, year range, and specific tags.

        Args:
            query: Free-text search string (searches name, key_innovation,
                   authors, paper_title, and tags).
            era: Optional era filter.
            category: Optional category filter.
            lineage: Optional lineage filter.
            year_start: Optional start year (inclusive).
            year_end: Optional end year (inclusive).
            tags: Optional list of tags to match (method must have all).

        Returns:
            List of MLMethod instances matching all specified criteria.
        """
        results = list(self._methods.values())

        # Apply era filter
        if era is not None:
            results = [m for m in results if m.era == era]

        # Apply category filter
        if category is not None:
            results = [m for m in results if m.category == category]

        # Apply lineage filter
        if lineage is not None:
            results = [m for m in results if lineage in m.lineages]

        # Apply year range filter
        if year_start is not None:
            results = [m for m in results if m.year >= year_start]
        if year_end is not None:
            results = [m for m in results if m.year <= year_end]

        # Apply tags filter (must have all specified tags)
        if tags:
            results = [m for m in results if all(tag in m.tags for tag in tags)]

        # Apply free-text search
        if query:
            query_lower = query.lower()
            filtered = []
            for m in results:
                searchable = " ".join([
                    m.name,
                    m.key_innovation,
                    m.paper_title,
                    " ".join(m.authors),
                    " ".join(m.tags)
                ]).lower()
                if query_lower in searchable:
                    filtered.append(m)
            results = filtered

        return results

    def unregister(self, method_id: str) -> bool:
        """
        Remove a method from the registry.

        Args:
            method_id: The unique identifier of the method to remove.

        Returns:
            True if the method was removed, False if it wasn't found.
        """
        if method_id in self._methods:
            del self._methods[method_id]
            return True
        return False

    def count(self) -> int:
        """
        Get the total number of registered methods.

        Returns:
            The count of methods in the registry.
        """
        return len(self._methods)

    def clear(self) -> None:
        """Remove all methods from the registry."""
        self._methods.clear()


# Global registry instance for convenience
_global_registry: Optional[MethodRegistry] = None


def get_global_registry() -> MethodRegistry:
    """
    Get the global method registry instance.

    Creates the registry on first access (singleton pattern).

    Returns:
        The global MethodRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = MethodRegistry()
    return _global_registry
