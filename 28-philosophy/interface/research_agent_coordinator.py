#!/usr/bin/env python3
"""
Philosophical Research Agent Coordinator

Coordinates autonomous research agents for continuous philosophical
knowledge expansion from external sources.
"""

import asyncio
import logging
import time
import re
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ResearchSource(Enum):
    """Available research sources"""
    STANFORD_ENCYCLOPEDIA = "sep"
    PHILPAPERS = "philpapers"
    INTERNET_ENCYCLOPEDIA = "iep"
    LOCAL_CORPUS = "local"
    WEB_SEARCH = "web"


class ResearchStatus(Enum):
    """Status of a research task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchTask:
    """Defines an autonomous research task."""
    task_id: str
    query: str
    sources: List[ResearchSource]
    priority: int = 5  # 1-10
    max_depth: int = 2
    status: ResearchStatus = ResearchStatus.PENDING
    triggered_by: str = "user"
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0


@dataclass
class ResearchResult:
    """Result of a completed research task."""
    task_id: str
    query: str
    status: ResearchStatus
    concepts_discovered: List[Dict[str, Any]] = field(default_factory=list)
    figures_discovered: List[Dict[str, Any]] = field(default_factory=list)
    texts_discovered: List[Dict[str, Any]] = field(default_factory=list)
    sources_consulted: List[str] = field(default_factory=list)
    raw_content: Dict[str, str] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class KnowledgeGap:
    """Represents an identified gap in knowledge."""
    gap_id: str
    gap_type: str  # "shallow_tradition", "missing_concept", "missing_figure"
    description: str
    tradition: Optional[str] = None
    domain: Optional[str] = None
    suggested_query: str = ""
    priority: int = 5


class PhilosophicalResearchAgentCoordinator:
    """
    Coordinates autonomous research agents for philosophical knowledge expansion.

    Features:
    - Multi-source research (SEP, PhilPapers, IEP, local corpus, web)
    - Task queue management with priorities
    - Knowledge gap detection and proactive research
    - Content extraction and concept mining
    """

    def __init__(
        self,
        consciousness_interface,
        max_concurrent_tasks: int = 3
    ):
        """
        Initialize the research coordinator.

        Args:
            consciousness_interface: Reference to PhilosophicalConsciousnessInterface
            max_concurrent_tasks: Maximum concurrent research tasks
        """
        self.interface = consciousness_interface
        self.max_concurrent = max_concurrent_tasks

        # Task management
        self.task_queue: List[ResearchTask] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, ResearchResult] = {}

        # Source configuration
        self.source_configs = {
            ResearchSource.STANFORD_ENCYCLOPEDIA: {
                "base_url": "https://plato.stanford.edu/entries/",
                "search_url": "https://plato.stanford.edu/search/searcher.py",
                "rate_limit_seconds": 2.0,
            },
            ResearchSource.PHILPAPERS: {
                "base_url": "https://philpapers.org/",
                "api_url": "https://philpapers.org/api/",
                "rate_limit_seconds": 1.0,
            },
            ResearchSource.INTERNET_ENCYCLOPEDIA: {
                "base_url": "https://iep.utm.edu/",
                "rate_limit_seconds": 2.0,
            },
        }

        # Rate limiting
        self.last_request_time: Dict[ResearchSource, float] = {}

        # Register with interface
        self.interface.research_coordinator = self

        logger.info("PhilosophicalResearchAgentCoordinator initialized")

    # ========================================================================
    # TASK MANAGEMENT
    # ========================================================================

    async def trigger_research(
        self,
        query: str,
        sources: Optional[List[ResearchSource]] = None,
        priority: int = 5,
        max_depth: int = 2,
        triggered_by: str = "user"
    ) -> str:
        """
        Trigger a new research task.

        Args:
            query: Topic or question to research
            sources: Which sources to consult
            priority: 1-10, higher = more urgent
            max_depth: How deep to follow related topics
            triggered_by: What initiated this research

        Returns:
            task_id for tracking
        """
        if sources is None:
            sources = [
                ResearchSource.STANFORD_ENCYCLOPEDIA,
                ResearchSource.PHILPAPERS,
                ResearchSource.WEB_SEARCH
            ]

        task_id = self._generate_task_id(query)

        task = ResearchTask(
            task_id=task_id,
            query=query,
            sources=sources,
            priority=priority,
            max_depth=max_depth,
            triggered_by=triggered_by,
            created_at=datetime.now(timezone.utc)
        )

        # Add to queue (sorted by priority)
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)

        logger.info(f"Research task queued: {task_id} for query '{query}'")

        # Start processing if we have capacity
        await self._process_queue()

        return task_id

    async def _process_queue(self) -> None:
        """Process queued research tasks."""
        while self.task_queue and len(self.active_tasks) < self.max_concurrent:
            task = self.task_queue.pop(0)
            await self._start_task(task)

    async def _start_task(self, task: ResearchTask) -> None:
        """Start executing a research task."""
        task.status = ResearchStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)

        # Create async task
        async_task = asyncio.create_task(self._execute_task(task))
        self.active_tasks[task.task_id] = async_task

        logger.info(f"Research task started: {task.task_id}")

    async def _execute_task(self, task: ResearchTask) -> ResearchResult:
        """Execute a research task."""
        start_time = time.time()
        result = ResearchResult(
            task_id=task.task_id,
            query=task.query,
            status=ResearchStatus.IN_PROGRESS
        )

        try:
            # Query each source
            for i, source in enumerate(task.sources):
                task.progress = (i / len(task.sources)) * 100

                source_result = await self._fetch_from_source(source, task.query)
                if source_result:
                    result.sources_consulted.append(source.value)
                    result.raw_content[source.value] = source_result.get("content", "")

                    # Extract concepts from content
                    concepts = self._extract_concepts(source_result, source)
                    result.concepts_discovered.extend(concepts)

                    # Extract figures
                    figures = self._extract_figures(source_result, source)
                    result.figures_discovered.extend(figures)

            # Deduplicate
            result.concepts_discovered = self._deduplicate_concepts(result.concepts_discovered)
            result.figures_discovered = self._deduplicate_figures(result.figures_discovered)

            # Integrate into knowledge base
            await self._integrate_results(result)

            task.status = ResearchStatus.COMPLETED
            result.status = ResearchStatus.COMPLETED
            task.progress = 100.0

        except Exception as e:
            logger.error(f"Research task failed: {task.task_id} - {str(e)}")
            task.status = ResearchStatus.FAILED
            task.error_message = str(e)
            result.status = ResearchStatus.FAILED
            result.error_message = str(e)

        finally:
            task.completed_at = datetime.now(timezone.utc)
            result.processing_time_seconds = time.time() - start_time

            # Store result
            self.completed_tasks[task.task_id] = result

            # Remove from active
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            # Update maturity
            self.interface.maturity_state.research_sessions_completed += 1

            logger.info(f"Research task completed: {task.task_id}")

        return result

    # ========================================================================
    # SOURCE FETCHING
    # ========================================================================

    async def _fetch_from_source(
        self,
        source: ResearchSource,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch content from a research source."""
        # Rate limiting
        await self._respect_rate_limit(source)

        if source == ResearchSource.STANFORD_ENCYCLOPEDIA:
            return await self._fetch_sep(query)
        elif source == ResearchSource.PHILPAPERS:
            return await self._fetch_philpapers(query)
        elif source == ResearchSource.INTERNET_ENCYCLOPEDIA:
            return await self._fetch_iep(query)
        elif source == ResearchSource.LOCAL_CORPUS:
            return await self._fetch_local(query)
        elif source == ResearchSource.WEB_SEARCH:
            return await self._fetch_web(query)

        return None

    async def _respect_rate_limit(self, source: ResearchSource) -> None:
        """Respect rate limits for a source."""
        config = self.source_configs.get(source, {})
        rate_limit = config.get("rate_limit_seconds", 1.0)

        last_time = self.last_request_time.get(source, 0)
        elapsed = time.time() - last_time

        if elapsed < rate_limit:
            await asyncio.sleep(rate_limit - elapsed)

        self.last_request_time[source] = time.time()

    async def _fetch_sep(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch from Stanford Encyclopedia of Philosophy."""
        # Simulated fetch - in production would use actual HTTP requests
        logger.debug(f"Fetching from SEP: {query}")

        # Generate mock response based on query
        slug = self._slugify(query)

        return {
            "source": "sep",
            "url": f"https://plato.stanford.edu/entries/{slug}/",
            "title": query.title(),
            "content": self._generate_mock_sep_content(query),
            "related_entries": [f"{query}_related_1", f"{query}_related_2"],
        }

    async def _fetch_philpapers(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch from PhilPapers."""
        logger.debug(f"Fetching from PhilPapers: {query}")

        return {
            "source": "philpapers",
            "query": query,
            "results": [
                {
                    "title": f"On {query.title()}",
                    "author": "Sample Author",
                    "year": 2020,
                    "abstract": f"This paper examines {query} from multiple perspectives...",
                }
            ],
        }

    async def _fetch_iep(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch from Internet Encyclopedia of Philosophy."""
        logger.debug(f"Fetching from IEP: {query}")

        return {
            "source": "iep",
            "title": query.title(),
            "content": f"The Internet Encyclopedia of Philosophy article on {query}...",
        }

    async def _fetch_local(self, query: str) -> Optional[Dict[str, Any]]:
        """Search local corpus."""
        logger.debug(f"Searching local corpus: {query}")
        # Would search local indexed texts
        return None

    async def _fetch_web(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform general web search."""
        logger.debug(f"Web search: {query}")

        return {
            "source": "web",
            "query": f"{query} philosophy",
            "results": [],  # Would contain actual search results
        }

    def _generate_mock_sep_content(self, query: str) -> str:
        """Generate mock SEP content for testing."""
        return f"""
        {query.title()}

        1. Introduction
        The concept of {query} has been central to philosophical inquiry since ancient times.
        It addresses fundamental questions about the nature of reality, knowledge, and value.

        2. Historical Development
        The term {query} derives from Greek and Latin philosophical vocabulary.
        Major philosophers who have addressed this concept include various thinkers
        from different traditions.

        3. Contemporary Debates
        Modern philosophers continue to debate the significance and implications of {query}.
        Key areas of disagreement include its metaphysical status and practical applications.

        4. Related Concepts
        {query} is closely connected to several other philosophical concepts including
        consciousness, existence, and understanding.
        """

    # ========================================================================
    # CONTENT EXTRACTION
    # ========================================================================

    def _extract_concepts(
        self,
        content: Dict[str, Any],
        source: ResearchSource
    ) -> List[Dict[str, Any]]:
        """Extract philosophical concepts from content."""
        concepts = []

        raw_content = content.get("content", "")
        title = content.get("title", "")

        # Simple extraction based on content structure
        if title:
            concepts.append({
                "name": title,
                "definition": self._extract_definition(raw_content),
                "source": source.value,
                "source_url": content.get("url", ""),
            })

        # Extract mentioned concepts (simplified)
        mentioned = self._extract_mentioned_concepts(raw_content)
        for name in mentioned[:5]:
            concepts.append({
                "name": name,
                "definition": f"Related concept mentioned in {source.value}",
                "source": source.value,
            })

        return concepts

    def _extract_definition(self, content: str) -> str:
        """Extract definition from content."""
        # Look for introduction section
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if "introduction" in line.lower():
                # Get next few lines as definition
                definition_lines = lines[i+1:i+4]
                return " ".join(definition_lines).strip()

        # Fallback to first substantial paragraph
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
        return paragraphs[0] if paragraphs else "Definition not available"

    def _extract_mentioned_concepts(self, content: str) -> List[str]:
        """Extract concepts mentioned in content."""
        # Simple pattern matching for philosophical terms
        philosophical_terms = [
            "consciousness", "existence", "knowledge", "truth", "reality",
            "ethics", "virtue", "justice", "freedom", "reason", "being",
            "metaphysics", "epistemology", "ontology", "phenomenology",
        ]

        found = []
        content_lower = content.lower()
        for term in philosophical_terms:
            if term in content_lower:
                found.append(term.title())

        return found

    def _extract_figures(
        self,
        content: Dict[str, Any],
        source: ResearchSource
    ) -> List[Dict[str, Any]]:
        """Extract philosopher figures from content."""
        figures = []

        raw_content = content.get("content", "")

        # Common philosopher names to detect
        philosopher_patterns = [
            r"Plato", r"Aristotle", r"Kant", r"Hegel", r"Nietzsche",
            r"Descartes", r"Hume", r"Locke", r"Spinoza", r"Leibniz",
            r"Husserl", r"Heidegger", r"Sartre", r"Wittgenstein",
            r"Buddha", r"Confucius", r"Laozi", r"Nagarjuna", r"Shankara",
        ]

        for pattern in philosopher_patterns:
            if re.search(pattern, raw_content, re.IGNORECASE):
                figures.append({
                    "name": pattern,
                    "source": source.value,
                })

        return figures

    # ========================================================================
    # KNOWLEDGE INTEGRATION
    # ========================================================================

    async def _integrate_results(self, result: ResearchResult) -> None:
        """Integrate research results into the knowledge base."""
        from .philosophical_consciousness_interface import (
            PhilosophicalConcept,
            PhilosophicalTradition,
            PhilosophicalDomain,
        )

        for concept_data in result.concepts_discovered:
            # Create concept object
            concept_id = self._slugify(concept_data["name"])

            # Skip if already exists
            if concept_id in self.interface.concept_index:
                continue

            concept = PhilosophicalConcept(
                concept_id=concept_id,
                name=concept_data["name"],
                tradition=PhilosophicalTradition.ANALYTIC,  # Default
                domain=PhilosophicalDomain.METAPHYSICS,  # Default
                definition=concept_data.get("definition", ""),
                sources=[{
                    "type": concept_data.get("source", "research"),
                    "url": concept_data.get("source_url", ""),
                    "accessed": datetime.now(timezone.utc).isoformat()
                }]
            )

            # Add to index
            await self.interface.add_concept(concept)

        logger.info(
            f"Integrated {len(result.concepts_discovered)} concepts from research task {result.task_id}"
        )

    # ========================================================================
    # KNOWLEDGE GAP DETECTION
    # ========================================================================

    async def detect_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Proactively identify gaps in philosophical knowledge."""
        from .philosophical_consciousness_interface import PhilosophicalTradition

        gaps = []
        gap_counter = 0

        # Check tradition coverage
        for tradition in PhilosophicalTradition:
            tradition_concepts = [
                c for c in self.interface.concept_index.values()
                if c.tradition == tradition
            ]

            if len(tradition_concepts) < 5:
                gap_counter += 1
                gaps.append(KnowledgeGap(
                    gap_id=f"gap_{gap_counter}",
                    gap_type="shallow_tradition",
                    description=f"Tradition {tradition.value} has only {len(tradition_concepts)} concepts",
                    tradition=tradition.value,
                    suggested_query=f"core concepts in {tradition.value}",
                    priority=7
                ))

        # Check for shallow concepts
        for concept in self.interface.concept_index.values():
            if concept.maturity_score < 0.3 and concept.research_depth < 2:
                gap_counter += 1
                gaps.append(KnowledgeGap(
                    gap_id=f"gap_{gap_counter}",
                    gap_type="shallow_concept",
                    description=f"Concept '{concept.name}' needs deeper research",
                    tradition=concept.tradition.value,
                    domain=concept.domain.value,
                    suggested_query=f"detailed analysis of {concept.name}",
                    priority=5
                ))

        # Sort by priority
        gaps.sort(key=lambda g: g.priority, reverse=True)

        return gaps[:20]  # Return top 20 gaps

    async def research_gaps(self, max_gaps: int = 5) -> List[str]:
        """Trigger research for detected knowledge gaps."""
        gaps = await self.detect_knowledge_gaps()
        task_ids = []

        for gap in gaps[:max_gaps]:
            task_id = await self.trigger_research(
                query=gap.suggested_query,
                priority=gap.priority,
                triggered_by="gap_detection"
            )
            task_ids.append(task_id)

        return task_ids

    # ========================================================================
    # TASK STATUS
    # ========================================================================

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a research task."""
        # Check completed
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": result.status.value,
                "concepts_discovered": len(result.concepts_discovered),
                "figures_discovered": len(result.figures_discovered),
                "sources_consulted": result.sources_consulted,
                "processing_time_seconds": result.processing_time_seconds,
            }

        # Check active
        if task_id in self.active_tasks:
            return {
                "task_id": task_id,
                "status": "in_progress",
            }

        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "priority": task.priority,
                }

        return None

    def get_all_tasks(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        return {
            "queued": len(self.task_queue),
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "queued_tasks": [t.task_id for t in self.task_queue],
            "active_tasks": list(self.active_tasks.keys()),
        }

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _generate_task_id(self, query: str) -> str:
        """Generate unique task ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_part = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:8]
        return f"research_{timestamp}_{hash_part}"

    def _slugify(self, text: str) -> str:
        """Convert text to slug."""
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9]+', '_', slug)
        slug = slug.strip('_')
        return slug

    def _deduplicate_concepts(self, concepts: List[Dict]) -> List[Dict]:
        """Deduplicate concept list."""
        seen = set()
        unique = []
        for concept in concepts:
            name = concept.get("name", "").lower()
            if name not in seen:
                seen.add(name)
                unique.append(concept)
        return unique

    def _deduplicate_figures(self, figures: List[Dict]) -> List[Dict]:
        """Deduplicate figure list."""
        seen = set()
        unique = []
        for figure in figures:
            name = figure.get("name", "").lower()
            if name not in seen:
                seen.add(name)
                unique.append(figure)
        return unique
