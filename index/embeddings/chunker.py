#!/usr/bin/env python3
"""
Chunker for the 40-Form Consciousness Repository.

Walks all markdown files in the consciousness repository, splits them into
~500-token segments using section headers as natural boundaries, and outputs
chunks.jsonl and metadata.json for downstream embedding and search.

Usage:
    python3 chunker.py [--repo-root PATH] [--chunk-size TOKENS] [--overlap TOKENS]

Defaults:
    --repo-root   ./
    --chunk-size  500   (approximate tokens; 1 token ~ 4 chars)
    --overlap     64    (approximate overlap tokens between adjacent chunks)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REPO_ROOT = Path(".")
CHARS_PER_TOKEN = 4  # rough approximation

# Directories to skip (relative to repo root)
SKIP_DIRS: set[str] = {"index", "archives", "__pycache__", ".git", "logs"}

# Form directory name -> (form_id: int, form_name: str)
# We parse these dynamically from directory names.

# Mapping of form_id to human-readable name.  Built at runtime from directory
# names like "01-visual" -> (1, "Visual Consciousness") and
# "00_Info" -> (0, "Background Information").
FORM_NAME_OVERRIDES: dict[int, str] = {
    0: "Background Information",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert text to a lowercase slug suitable for chunk IDs."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _titleize(slug: str) -> str:
    """Convert a hyphen/underscore slug into Title Case words."""
    return " ".join(word.capitalize() for word in re.split(r"[-_]+", slug))


def parse_form_dir(dirname: str) -> tuple[int, str] | None:
    """
    Parse a form directory name like '01-visual' into (1, 'Visual Consciousness').

    Returns None if the directory name does not look like a form directory.
    """
    # Match patterns: "00_Info", "01-visual", "40-xenoconsciousness"
    m = re.match(r"^(\d{2})[-_](.+)$", dirname)
    if not m:
        return None
    form_id = int(m.group(1))
    slug = m.group(2)
    if form_id in FORM_NAME_OVERRIDES:
        return form_id, FORM_NAME_OVERRIDES[form_id]
    name = _titleize(slug)
    # Append "Consciousness" if not already present and form_id > 0
    if form_id > 0 and "consciousness" not in name.lower() and "intelligence" not in name.lower():
        name += " Consciousness"
    return form_id, name


def extract_sections(text: str) -> list[dict[str, str]]:
    """
    Split markdown text into sections based on ## and ### headers.

    Returns a list of dicts:
        {"header": "Section Title" or "", "body": "section text..."}

    The first section (before any header) has header="".
    """
    # Split on lines that start with ## or ### (but not # alone or ####+)
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

    sections: list[dict[str, str]] = []
    last_end = 0
    last_header = ""

    for match in header_pattern.finditer(text):
        # Capture text before this header as previous section body
        body = text[last_end : match.start()]
        if body.strip() or last_header:
            sections.append({"header": last_header, "body": body})
        last_header = match.group(2).strip()
        last_end = match.end()

    # Remaining text after last header
    body = text[last_end:]
    if body.strip() or last_header:
        sections.append({"header": last_header, "body": body})

    # If no sections found, return whole text as one section
    if not sections:
        sections.append({"header": "", "body": text})

    return sections


def chunk_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters with
    *overlap_chars* characters of overlap between consecutive chunks.

    Tries to break on paragraph boundaries (double newline) first, then on
    single newlines, then on spaces.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunk = text[start:]
            if chunk.strip():
                chunks.append(chunk)
            break

        # Try to find a good break point
        segment = text[start:end]
        break_point = _find_break(segment)

        chunk = text[start : start + break_point]
        if chunk.strip():
            chunks.append(chunk)

        # Advance with overlap
        start = start + break_point - overlap_chars
        if start < 0:
            start = 0

    return chunks


def _find_break(segment: str) -> int:
    """Find a good break point in *segment*, preferring paragraph then line then word."""
    length = len(segment)

    # Look for double newline in last 30% of segment
    search_start = int(length * 0.7)
    idx = segment.rfind("\n\n", search_start)
    if idx != -1:
        return idx + 2  # include the double newline

    # Look for single newline in last 30%
    idx = segment.rfind("\n", search_start)
    if idx != -1:
        return idx + 1

    # Look for space in last 20%
    search_start = int(length * 0.8)
    idx = segment.rfind(" ", search_start)
    if idx != -1:
        return idx + 1

    # Hard break
    return length


# ---------------------------------------------------------------------------
# Main chunking logic
# ---------------------------------------------------------------------------


def build_chunk_id(form_id: int, rel_path: str, section: str, seq: int) -> str:
    """
    Build a deterministic chunk ID like:
        form_01_visual_info_overview_001
    """
    # Extract filename stem + parent subdir from relative path
    parts = Path(rel_path).parts  # e.g. ('01-visual', 'info', 'overview.md')
    # Skip the form dir itself, take the rest
    path_slug = "_".join(_slugify(p.replace(".md", "")) for p in parts[1:])
    section_slug = _slugify(section) if section else "preamble"
    return f"form_{form_id:02d}_{path_slug}_{section_slug}_{seq:03d}"


def chunk_file(
    file_path: Path,
    repo_root: Path,
    form_id: int,
    form_name: str,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    """
    Read a markdown file, split it into overlapping chunks respecting section
    boundaries, and return a list of chunk records.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    rel_path = str(file_path.relative_to(repo_root))
    sections = extract_sections(text)
    records: list[dict[str, Any]] = []

    for section_info in sections:
        header = section_info["header"]
        body = section_info["body"]

        # If the section header line itself is useful context, prepend it
        if header:
            full_text = f"## {header}\n{body}"
        else:
            full_text = body

        sub_chunks = chunk_text(full_text, max_chars, overlap_chars)

        for i, chunk_text_str in enumerate(sub_chunks):
            char_count = len(chunk_text_str)
            estimated_tokens = char_count // CHARS_PER_TOKEN
            chunk_id = build_chunk_id(form_id, rel_path, header, i + 1)

            records.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "form_id": form_id,
                "form_name": form_name,
                "file_path": rel_path,
                "section": header if header else "(preamble)",
                "char_count": char_count,
                "estimated_tokens": estimated_tokens,
            })

    return records


def discover_markdown_files(repo_root: Path) -> list[tuple[Path, int, str]]:
    """
    Walk the repository and yield (file_path, form_id, form_name) for every
    markdown file that should be chunked.

    Skips:
      - index/ directory
      - archives/ directory
      - __pycache__/ directories
      - .py files (only .md files are included)
      - Root-level .md files that are not inside a form directory are assigned
        form_id=0 with name "Background Information" (treated as cross-cutting).
    """
    results: list[tuple[Path, int, str]] = []

    for dirpath_str, dirnames, filenames in os.walk(repo_root):
        dirpath = Path(dirpath_str)

        # Prune skipped directories
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in sorted(filenames):
            if not fname.endswith(".md"):
                continue

            fpath = dirpath / fname
            rel = fpath.relative_to(repo_root)
            parts = rel.parts  # e.g. ('01-visual', 'info', 'overview.md')

            if len(parts) < 2:
                # Root-level markdown file (e.g. README.md)
                # Assign to form 0
                results.append((fpath, 0, "Background Information"))
                continue

            # First directory component should be a form directory
            top_dir = parts[0]
            parsed = parse_form_dir(top_dir)
            if parsed is None:
                # Non-form directory (e.g. scripts/, tools/)
                # Still chunk it, assign to form 0
                results.append((fpath, 0, "Background Information"))
                continue

            form_id, form_name = parsed
            results.append((fpath, form_id, form_name))

    return results


def run_chunker(
    repo_root: Path,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 64,
) -> None:
    """
    Main entry point: discover files, chunk them, write chunks.jsonl and metadata.json.
    """
    max_chars = chunk_size_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    output_dir = repo_root / "index" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "chunks.jsonl"
    meta_path = output_dir / "metadata.json"

    print(f"Consciousness Repository Chunker")
    print(f"  Repo root:    {repo_root}")
    print(f"  Chunk size:   ~{chunk_size_tokens} tokens (~{max_chars} chars)")
    print(f"  Overlap:      ~{overlap_tokens} tokens (~{overlap_chars} chars)")
    print(f"  Output dir:   {output_dir}")
    print()

    # Discover files
    files = discover_markdown_files(repo_root)
    print(f"Discovered {len(files)} markdown files to chunk.")

    # Chunk all files
    all_chunks: list[dict[str, Any]] = []
    files_chunked = 0
    form_index: dict[str, dict[str, Any]] = {}

    for fpath, form_id, form_name in files:
        chunks = chunk_file(fpath, repo_root, form_id, form_name, max_chars, overlap_chars)
        if chunks:
            files_chunked += 1
            all_chunks.extend(chunks)

            fid_str = str(form_id)
            if fid_str not in form_index:
                form_index[fid_str] = {
                    "name": form_name,
                    "chunk_ids": [],
                    "total_chunks": 0,
                }
            form_index[fid_str]["chunk_ids"].extend(c["chunk_id"] for c in chunks)
            form_index[fid_str]["total_chunks"] += len(chunks)

    print(f"Generated {len(all_chunks)} chunks from {files_chunked} files.")

    # Write chunks.jsonl
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {chunks_path}")

    # Sort form_index by numeric key
    sorted_form_index = dict(sorted(form_index.items(), key=lambda kv: int(kv[0])))

    # Write metadata.json
    metadata: dict[str, Any] = {
        "version": "1.0",
        "generated": str(date.today()),
        "total_chunks": len(all_chunks),
        "chunk_size_tokens": chunk_size_tokens,
        "overlap_tokens": overlap_tokens,
        "total_files_chunked": files_chunked,
        "embedding_model": None,
        "embedding_dim": None,
        "form_index": sorted_form_index,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Wrote {meta_path}")

    # Summary
    print()
    print("Per-form summary:")
    for fid_str in sorted(sorted_form_index, key=lambda x: int(x)):
        entry = sorted_form_index[fid_str]
        print(f"  Form {int(fid_str):2d}: {entry['name']:<40s} {entry['total_chunks']:>5d} chunks")

    print()
    print(f"Total: {len(all_chunks)} chunks, {files_chunked} files")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chunk the 40-Form Consciousness Repository for embedding.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the consciousness repository root.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Approximate chunk size in tokens (default: 500).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Approximate overlap in tokens between adjacent chunks (default: 64).",
    )
    args = parser.parse_args()

    if not args.repo_root.is_dir():
        print(f"Error: repo root does not exist: {args.repo_root}", file=sys.stderr)
        sys.exit(1)

    run_chunker(args.repo_root, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
