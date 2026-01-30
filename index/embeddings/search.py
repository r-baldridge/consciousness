#!/usr/bin/env python3
"""
Semantic search over the 40-Form Consciousness Repository embeddings.

Loads pre-computed embeddings from vectors.npy and metadata from metadata.json,
embeds a query string using the same model, and returns the top-K most similar
chunks ranked by cosine similarity.

Usage:
    python3 search.py "What is visual binding?"
    python3 search.py "dream lucidity" --top-k 5
    python3 search.py "plant signaling" --form 31
    python3 search.py "altered states" --top-k 20 --form 27

Flags:
    --top-k     Number of results to return (default: 10).
    --form      Filter results to a specific Form ID (e.g. 1 for Visual).
    --backend   Embedding backend: openai, sentence, custom (default: from metadata).
    --model     Model name override (default: from metadata).
    --data-dir  Directory containing vectors.npy, chunks.jsonl, metadata.json.
    --no-text   Suppress chunk text preview in output.
    --preview   Max characters of text preview per result (default: 200).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).parent
DEFAULT_TOP_K = 10
DEFAULT_PREVIEW_LEN = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_vectors(data_dir: Path) -> np.ndarray:
    """Load the embedding vectors from vectors.npy."""
    vectors_path = data_dir / "vectors.npy"
    if not vectors_path.exists():
        print(
            f"Error: {vectors_path} not found. Run embed.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return np.load(vectors_path)


def load_chunks(data_dir: Path) -> list[dict[str, Any]]:
    """Load chunk records from chunks.jsonl."""
    chunks_path = data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(
            f"Error: {chunks_path} not found. Run chunker.py first.",
            file=sys.stderr,
        )
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
        print(
            f"Error: {meta_path} not found. Run chunker.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Embedding backend resolution (reuse from embed.py)
# ---------------------------------------------------------------------------


def get_query_embedding(
    query: str,
    backend_name: str,
    model_name: str | None,
) -> np.ndarray:
    """
    Embed a single query string using the specified backend.

    Returns a 1-D numpy array of shape (embedding_dim,).
    """
    # Import the backend factory from embed.py (sibling module)
    try:
        from embed import get_backend
    except ImportError:
        # If running from a different directory, try adding script dir to path
        script_dir = str(Path(__file__).parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from embed import get_backend

    backend = get_backend(backend_name, model_name)
    vectors = backend.embed_batch([query])
    return vectors[0]


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row of a matrix.

    Args:
        query_vec: 1-D array of shape (dim,).
        matrix: 2-D array of shape (n, dim).

    Returns:
        1-D array of shape (n,) with cosine similarities in [-1, 1].
    """
    # Normalize query
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    q = query_vec / query_norm

    # Normalize matrix rows
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    m = matrix / row_norms

    # Dot product gives cosine similarity for normalized vectors
    return m @ q


def search(
    query: str,
    data_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    form_filter: int | None = None,
    backend_name: str | None = None,
    model_name: str | None = None,
    show_text: bool = True,
    preview_len: int = DEFAULT_PREVIEW_LEN,
) -> list[dict[str, Any]]:
    """
    Search the embedding index for chunks most similar to *query*.

    Args:
        query: Natural language query string.
        data_dir: Directory containing vectors.npy, chunks.jsonl, metadata.json.
        top_k: Number of results to return.
        form_filter: If set, only return chunks from this Form ID.
        backend_name: Embedding backend name (auto-detected from metadata if None).
        model_name: Model name override (auto-detected from metadata if None).
        show_text: Whether to include text previews in results.
        preview_len: Max characters for text previews.

    Returns:
        List of result dicts with keys: rank, score, chunk_id, form_id,
        form_name, file_path, section, text_preview (optional).
    """
    vectors = load_vectors(data_dir)
    chunks = load_chunks(data_dir)
    metadata = load_metadata(data_dir)

    if len(vectors) != len(chunks):
        print(
            f"Error: vector count ({len(vectors)}) != chunk count ({len(chunks)}). "
            f"Re-run embed.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve backend from metadata if not specified
    if backend_name is None:
        stored_model = metadata.get("embedding_model")
        if stored_model is None:
            print(
                "Error: no embedding_model in metadata.json and no --backend specified.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Infer backend from model name
        if "embedding" in str(stored_model).lower() and "ada" in str(stored_model).lower():
            backend_name = "openai"
        elif "text-embedding" in str(stored_model).lower():
            backend_name = "openai"
        elif "minilm" in str(stored_model).lower() or "sentence" in str(stored_model).lower():
            backend_name = "sentence"
        else:
            backend_name = "custom"

    if model_name is None:
        model_name = metadata.get("embedding_model")

    # Embed the query
    query_vec = get_query_embedding(query, backend_name, model_name)

    # Compute similarities
    similarities = cosine_similarity(query_vec, vectors)

    # Apply form filter
    if form_filter is not None:
        for i, chunk in enumerate(chunks):
            if chunk.get("form_id") != form_filter:
                similarities[i] = -2.0  # effectively exclude

    # Get top-K indices
    if top_k >= len(similarities):
        top_indices = np.argsort(-similarities)
    else:
        # Use argpartition for efficiency on large arrays
        top_indices_unsorted = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = top_indices_unsorted[np.argsort(-similarities[top_indices_unsorted])]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, 1):
        score = float(similarities[idx])
        if score <= -1.5:
            # Filtered out by form filter
            continue

        chunk = chunks[idx]
        result: dict[str, Any] = {
            "rank": rank,
            "score": round(score, 4),
            "chunk_id": chunk["chunk_id"],
            "form_id": chunk["form_id"],
            "form_name": chunk["form_name"],
            "file_path": chunk["file_path"],
            "section": chunk.get("section", ""),
        }
        if show_text:
            text = chunk.get("text", "")
            preview = text[:preview_len].replace("\n", " ").strip()
            if len(text) > preview_len:
                preview += "..."
            result["text_preview"] = preview

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def display_results(
    query: str,
    results: list[dict[str, Any]],
    show_text: bool = True,
) -> None:
    """Pretty-print search results to stdout."""
    print(f'Search: "{query}"')
    print(f"Results: {len(results)}")
    print("=" * 80)

    for r in results:
        print(f"\n  #{r['rank']}  Score: {r['score']:.4f}")
        print(f"  Form {r['form_id']:2d}: {r['form_name']}")
        print(f"  File:    {r['file_path']}")
        print(f"  Section: {r['section']}")
        if show_text and "text_preview" in r:
            wrapped = textwrap.fill(
                r["text_preview"],
                width=76,
                initial_indent="  Preview: ",
                subsequent_indent="           ",
            )
            print(wrapped)
        print(f"  Chunk:   {r['chunk_id']}")
        print("  " + "-" * 76)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic search over consciousness repository embeddings.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural language query string.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--form",
        type=int,
        default=None,
        dest="form_filter",
        help="Filter results to a specific Form ID (e.g. 1 for Visual).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Embedding backend: openai, sentence, custom (auto-detected from metadata).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (auto-detected from metadata).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing vectors.npy, chunks.jsonl, metadata.json.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Suppress chunk text preview in output.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=DEFAULT_PREVIEW_LEN,
        help=f"Max characters of text preview per result (default: {DEFAULT_PREVIEW_LEN}).",
    )

    args = parser.parse_args()

    results = search(
        query=args.query,
        data_dir=args.data_dir,
        top_k=args.top_k,
        form_filter=args.form_filter,
        backend_name=args.backend,
        model_name=args.model,
        show_text=not args.no_text,
        preview_len=args.preview,
    )

    display_results(args.query, results, show_text=not args.no_text)


if __name__ == "__main__":
    main()
