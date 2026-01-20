"""
Paper Index module for ML Research.

Database of key academic papers in machine learning, with
functionality for registration, retrieval, and search.
"""

from typing import Dict, List, Optional
from .taxonomy import Paper


class PaperIndex:
    """
    Database of ML papers.

    Provides storage and retrieval for academic papers that
    introduced or significantly advanced ML methods.

    Example:
        index = PaperIndex()
        index.register_paper(attention_paper)
        papers = index.get_papers_by_year(2017)
    """

    def __init__(self) -> None:
        """Initialize an empty paper index."""
        self._papers: Dict[str, Paper] = {}
        self._method_to_papers: Dict[str, List[str]] = {}
        self._year_to_papers: Dict[int, List[str]] = {}

    def register_paper(self, paper: Paper) -> None:
        """
        Register a new paper in the index.

        Args:
            paper: The Paper instance to register.

        Raises:
            ValueError: If a paper with the same paper_id already exists.
        """
        if paper.paper_id in self._papers:
            raise ValueError(f"Paper '{paper.paper_id}' is already registered")

        self._papers[paper.paper_id] = paper

        # Index by methods introduced
        for method_id in paper.methods_introduced:
            if method_id not in self._method_to_papers:
                self._method_to_papers[method_id] = []
            self._method_to_papers[method_id].append(paper.paper_id)

        # Index by year
        if paper.year not in self._year_to_papers:
            self._year_to_papers[paper.year] = []
        self._year_to_papers[paper.year].append(paper.paper_id)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Retrieve a paper by its unique identifier.

        Args:
            paper_id: The unique identifier of the paper.

        Returns:
            The Paper if found, None otherwise.
        """
        return self._papers.get(paper_id)

    def get_papers_by_method(self, method_id: str) -> List[Paper]:
        """
        Get all papers that introduced a specific method.

        Args:
            method_id: The method identifier to search for.

        Returns:
            List of Paper instances that introduced the method.
        """
        paper_ids = self._method_to_papers.get(method_id, [])
        return [self._papers[pid] for pid in paper_ids if pid in self._papers]

    def get_papers_by_year(self, year: int) -> List[Paper]:
        """
        Get all papers published in a specific year.

        Args:
            year: The publication year to filter by.

        Returns:
            List of Paper instances from that year.
        """
        paper_ids = self._year_to_papers.get(year, [])
        return [self._papers[pid] for pid in paper_ids if pid in self._papers]

    def get_papers_by_year_range(
        self,
        start_year: int,
        end_year: int
    ) -> List[Paper]:
        """
        Get all papers within a year range.

        Args:
            start_year: Start of the range (inclusive).
            end_year: End of the range (inclusive).

        Returns:
            List of Paper instances within the range, sorted by year.
        """
        papers = [
            p for p in self._papers.values()
            if start_year <= p.year <= end_year
        ]
        papers.sort(key=lambda p: p.year)
        return papers

    def get_papers_by_author(self, author: str) -> List[Paper]:
        """
        Get all papers by a specific author.

        Args:
            author: Author name to search for (partial match).

        Returns:
            List of Paper instances with matching author.
        """
        author_lower = author.lower()
        return [
            p for p in self._papers.values()
            if any(author_lower in a.lower() for a in p.authors)
        ]

    def get_papers_by_venue(self, venue: str) -> List[Paper]:
        """
        Get all papers from a specific venue.

        Args:
            venue: Venue name to search for (partial match).

        Returns:
            List of Paper instances from matching venues.
        """
        venue_lower = venue.lower()
        return [
            p for p in self._papers.values()
            if venue_lower in p.venue.lower()
        ]

    def search_papers(
        self,
        query: str,
        *,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        min_citations: Optional[int] = None
    ) -> List[Paper]:
        """
        Search papers by title and abstract content.

        Args:
            query: Free-text search string.
            year_start: Optional start year filter.
            year_end: Optional end year filter.
            min_citations: Optional minimum citations filter.

        Returns:
            List of matching Paper instances.
        """
        results = list(self._papers.values())

        # Apply year filters
        if year_start is not None:
            results = [p for p in results if p.year >= year_start]
        if year_end is not None:
            results = [p for p in results if p.year <= year_end]

        # Apply citation filter
        if min_citations is not None:
            results = [p for p in results if p.citations >= min_citations]

        # Apply text search
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.title.lower() or query_lower in p.abstract.lower()
            ]

        return results

    def get_most_cited(self, limit: int = 10) -> List[Paper]:
        """
        Get the most cited papers.

        Args:
            limit: Maximum number of papers to return.

        Returns:
            List of Paper instances sorted by citations (descending).
        """
        papers = sorted(
            self._papers.values(),
            key=lambda p: p.citations,
            reverse=True
        )
        return papers[:limit]

    def unregister_paper(self, paper_id: str) -> bool:
        """
        Remove a paper from the index.

        Args:
            paper_id: The unique identifier of the paper to remove.

        Returns:
            True if removed, False if not found.
        """
        if paper_id not in self._papers:
            return False

        paper = self._papers[paper_id]

        # Remove from method index
        for method_id in paper.methods_introduced:
            if method_id in self._method_to_papers:
                self._method_to_papers[method_id] = [
                    pid for pid in self._method_to_papers[method_id]
                    if pid != paper_id
                ]

        # Remove from year index
        if paper.year in self._year_to_papers:
            self._year_to_papers[paper.year] = [
                pid for pid in self._year_to_papers[paper.year]
                if pid != paper_id
            ]

        del self._papers[paper_id]
        return True

    def get_all_papers(self) -> List[Paper]:
        """
        Get all papers in the index.

        Returns:
            List of all Paper instances.
        """
        return list(self._papers.values())

    def count(self) -> int:
        """
        Get the total number of papers in the index.

        Returns:
            The count of papers.
        """
        return len(self._papers)

    def clear(self) -> None:
        """Remove all papers from the index."""
        self._papers.clear()
        self._method_to_papers.clear()
        self._year_to_papers.clear()

    def get_citation_stats(self) -> Dict[str, any]:
        """
        Get citation statistics across all papers.

        Returns:
            Dictionary with citation statistics:
                - total_citations: Sum of all citations
                - average_citations: Average per paper
                - max_citations: Highest citation count
                - min_citations: Lowest citation count
                - paper_count: Number of papers
        """
        if not self._papers:
            return {
                "total_citations": 0,
                "average_citations": 0.0,
                "max_citations": 0,
                "min_citations": 0,
                "paper_count": 0
            }

        citations = [p.citations for p in self._papers.values()]
        return {
            "total_citations": sum(citations),
            "average_citations": sum(citations) / len(citations),
            "max_citations": max(citations),
            "min_citations": min(citations),
            "paper_count": len(citations)
        }
