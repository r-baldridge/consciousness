"""
Memory Techniques

Methods for managing context, storing/retrieving information, and
extending the effective memory of language models.

=============================================================================
TECHNIQUES
=============================================================================

1. RAG (Retrieval-Augmented Generation)
   - Retrieve relevant documents from knowledge base
   - Include in prompt context for grounded generation

2. MemoryBank
   - Persistent storage of past interactions/facts
   - Query and update over time
   - Episodic, semantic, and working memory types

3. ContextCompression
   - Compress long contexts to fit model limits
   - Preserve essential information
   - Methods: summarization, selective attention, pruning

=============================================================================
MEMORY ARCHITECTURE
=============================================================================

                    ┌─────────────────────────────────────┐
                    │          Memory System              │
                    └─────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │    Working    │      │   Episodic    │      │   Semantic    │
    │    Memory     │      │    Memory     │      │    Memory     │
    │  (current     │      │   (specific   │      │   (general    │
    │   context)    │      │  experiences) │      │   knowledge)  │
    └───────────────┘      └───────────────┘      └───────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         Retrieval / Query           │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │       Context Composition           │
                    └─────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from enum import Enum
from abc import abstractmethod
import time
import hashlib

from .. import TechniqueBase, TechniqueResult, TechniqueConfig, TechniqueCategory

# Backend imports
if TYPE_CHECKING:
    from ...backends import LLMBackend

try:
    from ...backends import get_backend, LLMBackend as LLMBackendClass
    BACKENDS_AVAILABLE = True
except ImportError:
    BACKENDS_AVAILABLE = False
    LLMBackendClass = None


# =============================================================================
# RAG (Retrieval-Augmented Generation)
# =============================================================================

class RetrieverType(Enum):
    """Types of retrievers for RAG."""
    DENSE = "dense"      # Embedding-based (e.g., sentence transformers)
    SPARSE = "sparse"    # Keyword-based (e.g., BM25)
    HYBRID = "hybrid"    # Combination of dense and sparse


@dataclass
class Document:
    """A document in the knowledge base."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunks: List["DocumentChunk"] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """A chunk of a document."""
    chunk_id: str
    doc_id: str
    content: str
    start_idx: int
    end_idx: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    chunk: DocumentChunk
    score: float
    retriever: str


class RAG(TechniqueBase):
    """
    Retrieval-Augmented Generation.

    Paper: "Retrieval-Augmented Generation for Knowledge-Intensive
           NLP Tasks" (Lewis et al., 2020)
    https://arxiv.org/abs/2005.11401

    Retrieves relevant documents/passages from a knowledge base and
    includes them in the prompt context for grounded generation.

    Components:
        - Document store: Vector DB, BM25 index, or hybrid
        - Retriever: Finds relevant documents
        - Generator: LLM backend that uses retrieved context

    Configuration:
        backend: LLMBackend instance or name for generation
        embedding_backend: Optional separate backend for embeddings
        retriever_type: dense, sparse, or hybrid
        top_k: Number of documents to retrieve (default: 5)
        chunk_size: Size of document chunks (default: 512)
        chunk_overlap: Overlap between chunks (default: 50)
        rerank: Whether to rerank results (default: False)
        context_template: Template for including context

    Usage:
        from ml_research.backends import MockBackend

        rag = RAG(
            backend=MockBackend(),
            documents=my_documents,
            retriever_type=RetrieverType.HYBRID,
            top_k=5,
        )
        result = rag.run("What is the capital of France?")
    """

    TECHNIQUE_ID = "rag"
    CATEGORY = TechniqueCategory.MEMORY

    def __init__(
        self,
        backend: Optional[Any] = None,
        model: Optional[Any] = None,  # Deprecated, use backend
        embedding_backend: Optional[Any] = None,
        documents: Optional[List[Document]] = None,
        retriever_type: RetrieverType = RetrieverType.DENSE,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        rerank: bool = False,
        context_template: str = "Context:\n{context}\n\nQuestion: {question}\nAnswer:",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve backends
        self.backend = self._resolve_backend(backend, model)
        self.embedding_backend = embedding_backend or self.backend
        self.model = model  # Keep for backward compatibility

        self.documents = documents or []
        self.retriever_type = retriever_type
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank = rerank
        self.context_template = context_template

        # Index documents
        self._chunks: List[DocumentChunk] = []
        self._chunk_embeddings: Dict[str, List[float]] = {}
        self._index_documents()

    def _resolve_backend(self, backend: Optional[Any], model: Optional[Any]) -> Optional[Any]:
        """Resolve backend from various input types."""
        if backend is not None:
            if isinstance(backend, str):
                if BACKENDS_AVAILABLE:
                    return get_backend(backend)
                else:
                    raise ImportError("Backends module not available")
            return backend

        if model is not None:
            return model

        if BACKENDS_AVAILABLE:
            try:
                return get_backend()
            except (ValueError, KeyError):
                pass

        return None

    def _index_documents(self) -> None:
        """Index all documents into chunks."""
        for doc in self.documents:
            chunks = self._chunk_document(doc)
            doc.chunks = chunks
            self._chunks.extend(chunks)

    def _chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """Split document into chunks."""
        chunks = []
        content = doc.content
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            # Try to break at sentence boundary
            if end < len(content):
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    last_sep = content[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk_id = f"{doc.doc_id}_chunk_{chunk_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                content=content[start:end].strip(),
                start_idx=start,
                end_idx=end,
                metadata=doc.metadata.copy(),
            )
            chunks.append(chunk)

            start = end - self.chunk_overlap
            chunk_idx += 1

        return chunks

    def _embed_query(self, query: str) -> List[float]:
        """Get embedding for query using backend or fallback."""
        # Try to use embedding backend if available
        if self.embedding_backend is not None and hasattr(self.embedding_backend, 'embed'):
            try:
                return self.embedding_backend.embed(query)
            except (NotImplementedError, Exception):
                pass

        # Fallback: Simple hash-based pseudo-embedding
        h = hashlib.md5(query.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

    def _embed_chunk(self, chunk: DocumentChunk) -> List[float]:
        """Get embedding for chunk using backend or fallback."""
        if chunk.chunk_id in self._chunk_embeddings:
            return self._chunk_embeddings[chunk.chunk_id]

        # Try to use embedding backend if available
        if self.embedding_backend is not None and hasattr(self.embedding_backend, 'embed'):
            try:
                embedding = self.embedding_backend.embed(chunk.content)
                self._chunk_embeddings[chunk.chunk_id] = embedding
                return embedding
            except (NotImplementedError, Exception):
                pass

        # Fallback: Simple hash-based pseudo-embedding
        h = hashlib.md5(chunk.content.encode()).hexdigest()
        embedding = [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
        self._chunk_embeddings[chunk.chunk_id] = embedding
        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _retrieve_dense(self, query: str, k: int) -> List[RetrievalResult]:
        """Dense retrieval using embeddings."""
        query_embedding = self._embed_query(query)
        results = []

        for chunk in self._chunks:
            chunk_embedding = self._embed_chunk(chunk)
            score = self._cosine_similarity(query_embedding, chunk_embedding)
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                retriever="dense",
            ))

        results.sort(key=lambda x: -x.score)
        return results[:k]

    def _retrieve_sparse(self, query: str, k: int) -> List[RetrievalResult]:
        """Sparse retrieval using keyword matching (simple BM25-like)."""
        query_terms = set(query.lower().split())
        results = []

        for chunk in self._chunks:
            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms)
            score = overlap / max(len(query_terms), 1)
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                retriever="sparse",
            ))

        results.sort(key=lambda x: -x.score)
        return results[:k]

    def _retrieve_hybrid(self, query: str, k: int) -> List[RetrievalResult]:
        """Hybrid retrieval combining dense and sparse."""
        dense_results = self._retrieve_dense(query, k * 2)
        sparse_results = self._retrieve_sparse(query, k * 2)

        # Combine scores
        scores: Dict[str, float] = {}
        chunks: Dict[str, DocumentChunk] = {}

        for r in dense_results:
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0) + r.score * 0.7
            chunks[r.chunk.chunk_id] = r.chunk

        for r in sparse_results:
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0) + r.score * 0.3
            chunks[r.chunk.chunk_id] = r.chunk

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])
        return [
            RetrievalResult(chunk=chunks[cid], score=scores[cid], retriever="hybrid")
            for cid in sorted_ids[:k]
        ]

    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Rerank results using a reranker model (placeholder)."""
        # Real implementation uses cross-encoder reranker
        return results

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Build context string from retrieved results."""
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(f"[{i+1}] {r.chunk.content}")
        return "\n\n".join(context_parts)

    def _generate_response(self, prompt: str) -> str:
        """Generate response from backend or fallback to placeholder."""
        # Try to use backend if available
        if self.backend is not None:
            if hasattr(self.backend, 'generate'):
                return self.backend.generate(
                    prompt,
                    max_tokens=1024,
                    temperature=0.7,
                )
            elif callable(self.backend):
                return self.backend(prompt)

        # Placeholder response
        return "[Generated response based on retrieved context]"

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the knowledge base."""
        for doc in documents:
            chunks = self._chunk_document(doc)
            doc.chunks = chunks
            self._chunks.extend(chunks)
        self.documents.extend(documents)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        query = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        try:
            # Retrieve
            if self.retriever_type == RetrieverType.DENSE:
                results = self._retrieve_dense(query, self.top_k)
            elif self.retriever_type == RetrieverType.SPARSE:
                results = self._retrieve_sparse(query, self.top_k)
            else:
                results = self._retrieve_hybrid(query, self.top_k)

            trace.append({
                "action": "retrieve",
                "retriever": self.retriever_type.value,
                "num_results": len(results),
            })

            # Rerank if enabled
            if self.rerank:
                results = self._rerank_results(query, results)
                trace.append({"action": "rerank"})

            # Build context
            retrieved_context = self._build_context(results)
            trace.append({
                "action": "build_context",
                "context_length": len(retrieved_context),
            })

            # Format prompt
            prompt = self.context_template.format(
                context=retrieved_context,
                question=query,
            )

            # Generate
            response = self._generate_response(prompt)
            trace.append({
                "action": "generate",
                "response_length": len(response),
            })

            return TechniqueResult(
                success=True,
                output={
                    "response": response,
                    "retrieved_chunks": [
                        {"content": r.chunk.content[:100], "score": r.score}
                        for r in results
                    ],
                    "context": retrieved_context,
                },
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
            )

        except Exception as e:
            return TechniqueResult(
                success=False,
                output=None,
                technique_id=self.TECHNIQUE_ID,
                execution_time_ms=(time.time() - start) * 1000,
                intermediate_steps=trace,
                error=str(e),
            )


# =============================================================================
# MEMORY BANK
# =============================================================================

class MemoryType(Enum):
    """Types of memory."""
    EPISODIC = "episodic"    # Specific experiences/interactions
    SEMANTIC = "semantic"     # General knowledge/facts
    WORKING = "working"       # Current task context
    PROCEDURAL = "procedural" # How to do things


@dataclass
class MemoryEntry:
    """An entry in the memory bank."""
    memory_id: str
    memory_type: MemoryType
    content: str
    timestamp: float
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    associations: List[str] = field(default_factory=list)


class MemoryBank(TechniqueBase):
    """
    Memory Bank for persistent storage and retrieval.

    Maintains different types of memory:
        - Episodic: Specific past experiences/interactions
        - Semantic: General knowledge and facts
        - Working: Current task context
        - Procedural: Learned procedures/skills

    Features:
        - Importance-based retention
        - Recency-weighted retrieval
        - Associative linking
        - Memory consolidation

    Configuration:
        max_memories: Maximum memories to store
        memory_types: Which memory types to use
        retrieval_strategy: How to retrieve memories
        consolidation_interval: How often to consolidate

    Usage:
        memory = MemoryBank(max_memories=1000)
        memory.store("User prefers dark mode", MemoryType.SEMANTIC)
        relevant = memory.retrieve("display settings", k=3)
    """

    TECHNIQUE_ID = "memory_bank"
    CATEGORY = TechniqueCategory.MEMORY

    def __init__(
        self,
        model: Optional[Any] = None,
        max_memories: int = 1000,
        importance_threshold: float = 0.3,
        decay_rate: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.max_memories = max_memories
        self.importance_threshold = importance_threshold
        self.decay_rate = decay_rate

        self._memories: Dict[str, MemoryEntry] = {}
        self._memory_counter = 0

    def _generate_id(self) -> str:
        """Generate unique memory ID."""
        self._memory_counter += 1
        return f"mem_{self._memory_counter}_{int(time.time())}"

    def _compute_importance(self, content: str, memory_type: MemoryType) -> float:
        """Compute importance score for a memory (placeholder)."""
        # Real implementation could use LLM to assess importance
        base_importance = {
            MemoryType.EPISODIC: 0.5,
            MemoryType.SEMANTIC: 0.7,
            MemoryType.WORKING: 0.3,
            MemoryType.PROCEDURAL: 0.8,
        }.get(memory_type, 0.5)

        # Adjust based on content length (proxy for information)
        length_factor = min(len(content) / 500, 1.0) * 0.2
        return min(base_importance + length_factor, 1.0)

    def _embed_memory(self, content: str) -> List[float]:
        """Get embedding for memory content (placeholder)."""
        h = hashlib.md5(content.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a new memory."""
        memory_id = self._generate_id()
        now = time.time()

        if importance is None:
            importance = self._compute_importance(content, memory_type)

        entry = MemoryEntry(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            timestamp=now,
            importance=importance,
            last_accessed=now,
            metadata=metadata or {},
            embedding=self._embed_memory(content),
        )

        self._memories[memory_id] = entry

        # Evict if over capacity
        if len(self._memories) > self.max_memories:
            self._evict_memories()

        return memory_id

    def retrieve(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories."""
        query_embedding = self._embed_memory(query)
        now = time.time()

        candidates = []
        for memory in self._memories.values():
            # Filter by type
            if memory_types and memory.memory_type not in memory_types:
                continue

            # Filter by importance
            if memory.importance < min_importance:
                continue

            # Compute relevance score
            similarity = self._cosine_similarity(query_embedding, memory.embedding or [])

            # Recency boost
            age_hours = (now - memory.timestamp) / 3600
            recency_factor = 1.0 / (1.0 + age_hours * self.decay_rate)

            # Combined score
            score = similarity * 0.7 + memory.importance * 0.2 + recency_factor * 0.1

            candidates.append((memory, score))

        # Sort by score and return top k
        candidates.sort(key=lambda x: -x[1])

        # Update access counts
        results = []
        for memory, _ in candidates[:k]:
            memory.access_count += 1
            memory.last_accessed = now
            results.append(memory)

        return results

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing memory."""
        if memory_id not in self._memories:
            return False

        memory = self._memories[memory_id]

        if content is not None:
            memory.content = content
            memory.embedding = self._embed_memory(content)

        if importance is not None:
            memory.importance = importance

        if metadata is not None:
            memory.metadata.update(metadata)

        return True

    def forget(self, memory_id: str) -> bool:
        """Remove a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def _evict_memories(self) -> None:
        """Evict least important/accessed memories."""
        if len(self._memories) <= self.max_memories:
            return

        # Score memories for eviction
        now = time.time()
        scores = []
        for memory in self._memories.values():
            age = (now - memory.timestamp) / 3600
            score = (
                memory.importance * 0.5 +
                min(memory.access_count / 10, 1.0) * 0.3 +
                (1.0 / (1.0 + age * self.decay_rate)) * 0.2
            )
            scores.append((memory.memory_id, score))

        # Sort by score (ascending) and remove lowest
        scores.sort(key=lambda x: x[1])
        num_to_remove = len(self._memories) - self.max_memories

        for memory_id, _ in scores[:num_to_remove]:
            del self._memories[memory_id]

    def consolidate(self) -> None:
        """Consolidate memories (e.g., merge similar, summarize)."""
        # Placeholder for memory consolidation
        # Real implementation could merge similar memories,
        # create summary memories, update importance based on access patterns
        pass

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        """Run memory retrieval as a technique."""
        start = time.time()
        query = input_data if isinstance(input_data, str) else str(input_data)
        trace: List[Dict] = []

        # Retrieve relevant memories
        memories = self.retrieve(query, k=5)
        trace.append({
            "action": "retrieve",
            "num_results": len(memories),
        })

        # Format memories as context
        memory_context = "\n".join(
            f"- [{m.memory_type.value}] {m.content}"
            for m in memories
        )

        return TechniqueResult(
            success=True,
            output={
                "memories": [
                    {
                        "content": m.content,
                        "type": m.memory_type.value,
                        "importance": m.importance,
                    }
                    for m in memories
                ],
                "context": memory_context,
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# CONTEXT COMPRESSION
# =============================================================================

class CompressionMethod(Enum):
    """Methods for compressing context."""
    SUMMARIZE = "summarize"      # LLM summarization
    SELECTIVE = "selective"      # Keep only relevant parts
    PRUNE = "prune"              # Remove redundant tokens
    AUTOCOMPRESSOR = "autocompressor"  # Learned compression


class ContextCompression(TechniqueBase):
    """
    Context Compression for long inputs.

    Compresses long contexts to fit within model limits while
    preserving essential information.

    Methods:
        - Summarize: Use LLM to create summary
        - Selective: Keep query-relevant parts
        - Prune: Remove low-information tokens
        - AutoCompressor: Learned compression tokens

    Configuration:
        method: Compression method to use
        target_ratio: Target compression ratio (default: 0.5)
        preserve_keywords: Keywords to always preserve
        chunk_size: Size of chunks for processing

    Usage:
        compressor = ContextCompression(
            method=CompressionMethod.SELECTIVE,
            target_ratio=0.3,
        )
        compressed = compressor.run({
            "context": long_document,
            "query": "What is the main argument?"
        })
    """

    TECHNIQUE_ID = "context_compression"
    CATEGORY = TechniqueCategory.MEMORY

    def __init__(
        self,
        model: Optional[Any] = None,
        method: CompressionMethod = CompressionMethod.SELECTIVE,
        target_ratio: float = 0.5,
        preserve_keywords: Optional[List[str]] = None,
        min_length: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.method = method
        self.target_ratio = target_ratio
        self.preserve_keywords = preserve_keywords or []
        self.min_length = min_length

    def _summarize(self, text: str, target_length: int) -> str:
        """Summarize text to target length (placeholder)."""
        # Real implementation uses LLM
        words = text.split()
        if len(words) <= target_length:
            return text
        return " ".join(words[:target_length]) + "..."

    def _selective_compress(
        self,
        text: str,
        query: str,
        target_length: int,
    ) -> str:
        """Keep query-relevant parts (placeholder)."""
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', text)

        # Score sentences by query relevance
        query_terms = set(query.lower().split())
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_terms = set(sent.lower().split())
            overlap = len(query_terms & sent_terms)
            # Also boost sentences with preserve keywords
            keyword_boost = sum(
                1 for kw in self.preserve_keywords
                if kw.lower() in sent.lower()
            )
            score = overlap + keyword_boost * 2
            scored.append((sent, score))

        # Sort by score and take top sentences
        scored.sort(key=lambda x: -x[1])

        # Accumulate until target length
        result = []
        current_length = 0
        for sent, _ in scored:
            if current_length + len(sent.split()) > target_length:
                break
            result.append(sent)
            current_length += len(sent.split())

        return ". ".join(result) + "."

    def _prune(self, text: str, target_length: int) -> str:
        """Prune low-information tokens (placeholder)."""
        # Simple implementation: remove stop words and short words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "again", "further", "then", "once"}

        words = text.split()
        filtered = [w for w in words if w.lower() not in stop_words or len(w) > 3]

        if len(filtered) <= target_length:
            return " ".join(filtered)

        return " ".join(filtered[:target_length])

    def compress(
        self,
        text: str,
        query: Optional[str] = None,
        target_ratio: Optional[float] = None,
    ) -> str:
        """Compress text using configured method."""
        ratio = target_ratio or self.target_ratio
        target_length = max(int(len(text.split()) * ratio), self.min_length)

        if self.method == CompressionMethod.SUMMARIZE:
            return self._summarize(text, target_length)
        elif self.method == CompressionMethod.SELECTIVE:
            return self._selective_compress(text, query or "", target_length)
        elif self.method == CompressionMethod.PRUNE:
            return self._prune(text, target_length)
        else:
            return self._summarize(text, target_length)

    def run(self, input_data: Any, context: Optional[Dict] = None) -> TechniqueResult:
        start = time.time()
        trace: List[Dict] = []

        # Parse input
        if isinstance(input_data, dict):
            text = input_data.get("context", input_data.get("text", ""))
            query = input_data.get("query", "")
        else:
            text = str(input_data)
            query = ""

        original_length = len(text.split())
        trace.append({
            "action": "analyze",
            "original_length": original_length,
        })

        # Compress
        compressed = self.compress(text, query)
        compressed_length = len(compressed.split())

        trace.append({
            "action": "compress",
            "method": self.method.value,
            "compressed_length": compressed_length,
            "ratio": compressed_length / max(original_length, 1),
        })

        return TechniqueResult(
            success=True,
            output={
                "compressed": compressed,
                "original_length": original_length,
                "compressed_length": compressed_length,
                "compression_ratio": compressed_length / max(original_length, 1),
            },
            technique_id=self.TECHNIQUE_ID,
            execution_time_ms=(time.time() - start) * 1000,
            intermediate_steps=trace,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RetrieverType",
    "MemoryType",
    "CompressionMethod",
    # Data classes
    "Document",
    "DocumentChunk",
    "RetrievalResult",
    "MemoryEntry",
    # Techniques
    "RAG",
    "MemoryBank",
    "ContextCompression",
]
