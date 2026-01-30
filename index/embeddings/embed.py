#!/usr/bin/env python3
"""
Embedding generator for the 40-Form Consciousness Repository.

Reads chunks.jsonl produced by chunker.py, generates vector embeddings using
a pluggable backend, and outputs vectors.npy alongside updated metadata.json.

Supported backends:
    - openai:   Uses OpenAI's text-embedding-3-large (requires OPENAI_API_KEY)
    - sentence: Uses sentence-transformers (requires sentence-transformers package)
    - custom:   Placeholder for any custom embedding function

Usage:
    python3 embed.py --backend openai [--model text-embedding-3-large] [--batch-size 100]
    python3 embed.py --backend sentence [--model all-MiniLM-L6-v2]
    python3 embed.py --dry-run

Flags:
    --dry-run       Report chunk count and estimated API cost without generating embeddings.
    --backend       Embedding backend to use: openai, sentence, custom (default: openai).
    --model         Model name override for the chosen backend.
    --batch-size    Batch size for API-based models (default: 100).
    --data-dir      Directory containing chunks.jsonl and metadata.json
                    (default: same directory as this script).
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# numpy is available per the project constraints
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).parent
DEFAULT_BATCH_SIZE = 100

# Approximate pricing for OpenAI embeddings (per 1M tokens)
OPENAI_PRICING: dict[str, float] = {
    "text-embedding-3-large": 0.13,   # $0.13 per 1M tokens
    "text-embedding-3-small": 0.02,   # $0.02 per 1M tokens
    "text-embedding-ada-002": 0.10,   # $0.10 per 1M tokens (legacy)
}


# ---------------------------------------------------------------------------
# Embedding backend interface
# ---------------------------------------------------------------------------


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model name string."""
        ...

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings produced."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        ...

    def embed_all(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool = True,
    ) -> np.ndarray:
        """
        Embed all texts in batches.

        Args:
            texts: Complete list of strings to embed.
            batch_size: Number of texts per batch.
            progress: Whether to print progress to stderr.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        n = len(texts)
        dim = self.embedding_dim()
        vectors = np.zeros((n, dim), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = texts[start:end]

            if progress:
                print(
                    f"  Embedding batch {start // batch_size + 1}"
                    f" / {(n + batch_size - 1) // batch_size}"
                    f"  ({end}/{n} chunks)",
                    file=sys.stderr,
                )

            batch_vectors = self.embed_batch(batch)
            vectors[start:end] = batch_vectors

        return vectors


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


class OpenAIBackend(EmbeddingBackend):
    """Embedding backend using the OpenAI API."""

    # Default model and known dimensions
    DIMENSIONS: dict[str, int] = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-large") -> None:
        self._model = model
        self._dim = self.DIMENSIONS.get(model, 3072)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialise the OpenAI client."""
        if self._client is None:
            try:
                import openai  # type: ignore[import-untyped]
            except ImportError:
                print(
                    "Error: openai package not installed. "
                    "Install with: pip install openai",
                    file=sys.stderr,
                )
                sys.exit(1)

            self._client = openai.OpenAI()  # reads OPENAI_API_KEY from env
        return self._client

    def model_name(self) -> str:
        return self._model

    def embedding_dim(self) -> int:
        return self._dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self._model)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Sentence-Transformers backend
# ---------------------------------------------------------------------------


class SentenceTransformerBackend(EmbeddingBackend):
    """Embedding backend using a local sentence-transformers model."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._model: Any = None
        self._dim: int | None = None

    def _load_model(self) -> Any:
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            except ImportError:
                print(
                    "Error: sentence-transformers package not installed. "
                    "Install with: pip install sentence-transformers",
                    file=sys.stderr,
                )
                sys.exit(1)

            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def model_name(self) -> str:
        return self._model_name

    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load_model()
        assert self._dim is not None
        return self._dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Custom backend (placeholder)
# ---------------------------------------------------------------------------


class CustomBackend(EmbeddingBackend):
    """
    Placeholder backend for custom embedding models.

    To use: subclass or replace the embed_batch method with your own model.
    Currently generates random vectors for testing purposes.
    """

    def __init__(self, model: str = "custom-placeholder", dim: int = 768) -> None:
        self._model = model
        self._dim = dim

    def model_name(self) -> str:
        return self._model

    def embedding_dim(self) -> int:
        return self._dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        # Placeholder: returns random unit vectors for testing
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((len(texts), self._dim)).astype(np.float32)
        # Normalize to unit vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


BACKENDS: dict[str, type[EmbeddingBackend]] = {
    "openai": OpenAIBackend,
    "sentence": SentenceTransformerBackend,
    "custom": CustomBackend,
}


def get_backend(name: str, model: str | None = None) -> EmbeddingBackend:
    """
    Instantiate an embedding backend by name.

    Args:
        name: One of 'openai', 'sentence', or 'custom'.
        model: Optional model name override.

    Returns:
        An instance of the corresponding EmbeddingBackend.
    """
    cls = BACKENDS.get(name)
    if cls is None:
        print(f"Error: unknown backend '{name}'. Choose from: {list(BACKENDS.keys())}", file=sys.stderr)
        sys.exit(1)

    if model:
        return cls(model=model)  # type: ignore[call-arg]
    return cls()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_chunks(data_dir: Path) -> list[dict[str, Any]]:
    """Load chunk records from chunks.jsonl."""
    chunks_path = data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run chunker.py first.", file=sys.stderr)
        sys.exit(1)

    chunks: list[dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_metadata(data_dir: Path) -> dict[str, Any]:
    """Load metadata.json."""
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Run chunker.py first.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(data_dir: Path, metadata: dict[str, Any]) -> None:
    """Write updated metadata.json."""
    meta_path = data_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(data_dir: Path, backend_name: str, model: str | None) -> None:
    """Report chunk statistics and estimated cost without embedding."""
    chunks = load_chunks(data_dir)
    metadata = load_metadata(data_dir)

    total_chunks = len(chunks)
    total_chars = sum(c.get("char_count", 0) for c in chunks)
    total_tokens_est = sum(c.get("estimated_tokens", 0) for c in chunks)

    # Determine model name for cost estimation
    if model is None:
        if backend_name == "openai":
            model = "text-embedding-3-large"
        elif backend_name == "sentence":
            model = "all-MiniLM-L6-v2"
        else:
            model = "custom"

    print(f"Dry Run Report")
    print(f"==============")
    print(f"  Data directory:       {data_dir}")
    print(f"  Backend:              {backend_name}")
    print(f"  Model:                {model}")
    print(f"  Total chunks:         {total_chunks:,}")
    print(f"  Total characters:     {total_chars:,}")
    print(f"  Estimated tokens:     {total_tokens_est:,}")
    print(f"  Files chunked:        {metadata.get('total_files_chunked', 'N/A')}")
    print()

    # Cost estimation for OpenAI models
    if model in OPENAI_PRICING:
        cost_per_million = OPENAI_PRICING[model]
        estimated_cost = (total_tokens_est / 1_000_000) * cost_per_million
        print(f"  OpenAI pricing ({model}):")
        print(f"    Rate:               ${cost_per_million:.2f} / 1M tokens")
        print(f"    Estimated cost:     ${estimated_cost:.4f}")
    elif backend_name == "sentence":
        print(f"  Local model: no API cost (runs on your machine).")
    else:
        print(f"  Cost estimation not available for model '{model}'.")

    print()
    print("No embeddings generated (dry run).")


# ---------------------------------------------------------------------------
# Main embedding logic
# ---------------------------------------------------------------------------


def run_embed(
    data_dir: Path,
    backend_name: str,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Load chunks, generate embeddings, and save vectors.npy + updated metadata.json.
    """
    chunks = load_chunks(data_dir)
    metadata = load_metadata(data_dir)

    if not chunks:
        print("No chunks found. Run chunker.py first.", file=sys.stderr)
        sys.exit(1)

    backend = get_backend(backend_name, model)

    print(f"Embedding {len(chunks):,} chunks")
    print(f"  Backend: {backend_name}")
    print(f"  Model:   {backend.model_name()}")
    print(f"  Dim:     {backend.embedding_dim()}")
    print(f"  Batch:   {batch_size}")
    print()

    texts = [c["text"] for c in chunks]
    vectors = backend.embed_all(texts, batch_size=batch_size, progress=True)

    # Save vectors
    vectors_path = data_dir / "vectors.npy"
    np.save(vectors_path, vectors)
    print(f"\nSaved {vectors_path} ({vectors.shape[0]} x {vectors.shape[1]})")

    # Update metadata
    metadata["embedding_model"] = backend.model_name()
    metadata["embedding_dim"] = backend.embedding_dim()
    save_metadata(data_dir, metadata)
    print(f"Updated metadata.json with embedding info.")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for consciousness repository chunks.",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="openai",
        help="Embedding backend to use (default: openai).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override for the chosen backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for API-based models (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing chunks.jsonl and metadata.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report chunk count and estimated cost without generating embeddings.",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.data_dir, args.backend, args.model)
    else:
        run_embed(args.data_dir, args.backend, args.model, args.batch_size)


if __name__ == "__main__":
    main()
