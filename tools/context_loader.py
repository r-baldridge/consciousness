#!/usr/bin/env python3
"""Context Loader for the 40-Form Consciousness Repository.

Loads repository content scaled to any context window budget. Supports
query-based relevance matching, named budget profiles, explicit form
selection, and cluster-based loading.

Usage:
    # Query-based loading with token budget
    python3 context_loader.py -q "visual binding problem" -b 32000

    # Named profile
    python3 context_loader.py -q "plant intelligence" -p focused_128k

    # Explicit form IDs
    python3 context_loader.py -f 1,9,16 -b 128000

    # Cluster loading
    python3 context_loader.py -c sensory -b 200000

    # Output as file paths
    python3 context_loader.py -q "meditation" -b 32000 -o paths

    # Output as concatenated content
    python3 context_loader.py -q "meditation" -b 32000 -o content

Programmatic usage:
    from context_loader import ContextLoader
    loader = ContextLoader("/path/to/consciousness/")
    result = loader.load(query="visual binding", budget=32000)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4  # approximation: 1 token ~ 4 characters

# Default token budgets mapped from named profiles
PROFILE_BUDGETS: dict[str, int] = {
    "minimal_8k": 8_000,
    "standard_32k": 32_000,
    "focused_128k": 128_000,
    "deep_200k": 200_000,
    "rag_unlimited": 1_000_000,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FormEntry:
    """Parsed form entry from manifest.yaml."""

    id: int
    name: str
    slug: str
    desc: str
    files: int
    md: int
    py: int
    test: bool
    research: bool
    specs: int
    tags: list[str]
    xrefs: list[int]
    tier: int
    cluster: str = ""  # populated from topic_graph.json


@dataclass
class FileSelection:
    """A single file selected for loading."""

    path: str
    estimated_tokens: int
    reason: str


@dataclass
class LoadResult:
    """Result of a context loading operation."""

    files: list[FileSelection] = field(default_factory=list)
    total_estimated_tokens: int = 0
    budget: int = 0
    query: str = ""
    profile: str = ""
    forms_matched: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ContextLoader class
# ---------------------------------------------------------------------------

class ContextLoader:
    """Loads 40-Form Consciousness Repository content within a token budget.

    Parameters
    ----------
    repo_root : str | Path
        Absolute path to the ``consciousness/`` directory.
    """

    def __init__(self, repo_root: str | Path) -> None:
        self.root = Path(repo_root).resolve()
        self.index_dir = self.root / "index"

        # Loaded data
        self.manifest: dict[str, Any] = {}
        self.forms: dict[int, FormEntry] = {}
        self.info_entry: FormEntry | None = None
        self.profiles: dict[str, Any] = {}
        self.topic_graph: dict[str, Any] = {}
        self.clusters: dict[str, list[int]] = {}
        self.edges: list[dict[str, Any]] = []
        self.concept_edges: list[dict[str, Any]] = []
        self.hub_patterns: list[dict[str, Any]] = []

        self._load_index()

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Read and parse all index files on startup."""
        self._load_manifest()
        self._load_profiles()
        self._load_topic_graph()

    def _load_manifest(self) -> None:
        """Parse manifest.yaml into FormEntry objects."""
        manifest_path = self.index_dir / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.yaml not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

        # Parse info section (Form 0 / background)
        info = self.manifest.get("info", {})
        if info:
            self.info_entry = FormEntry(
                id=info.get("id", 0),
                name=info.get("name", "Background Information"),
                slug=info.get("slug", "00_Info"),
                desc=info.get("desc", ""),
                files=info.get("files", 0),
                md=info.get("md", 0),
                py=info.get("py", 0),
                test=info.get("test", False),
                research=info.get("research", False),
                specs=info.get("specs", 0),
                tags=info.get("tags", []),
                xrefs=info.get("xrefs", []),
                tier=info.get("tier", 3),
            )

        # Parse each form
        for entry in self.manifest.get("forms", []):
            form = FormEntry(
                id=entry["id"],
                name=entry["name"],
                slug=entry["slug"],
                desc=entry.get("desc", ""),
                files=entry.get("files", 0),
                md=entry.get("md", 0),
                py=entry.get("py", 0),
                test=entry.get("test", False),
                research=entry.get("research", False),
                specs=entry.get("specs", 0),
                tags=entry.get("tags", []),
                xrefs=entry.get("xrefs", []),
                tier=entry.get("tier", 1),
            )
            self.forms[form.id] = form

    def _load_profiles(self) -> None:
        """Parse token_budget_profiles.yaml."""
        profiles_path = self.index_dir / "token_budget_profiles.yaml"
        if profiles_path.exists():
            with open(profiles_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.profiles = data.get("profiles", {})

    def _load_topic_graph(self) -> None:
        """Parse topic_graph.json for clusters and edges."""
        graph_path = self.index_dir / "topic_graph.json"
        if not graph_path.exists():
            return

        with open(graph_path, "r", encoding="utf-8") as f:
            self.topic_graph = json.load(f)

        # Clusters
        self.clusters = self.topic_graph.get("clusters", {})

        # Assign cluster names back to forms
        for cluster_name, form_ids in self.clusters.items():
            for fid in form_ids:
                if fid in self.forms:
                    self.forms[fid].cluster = cluster_name

        # Edges
        edges_section = self.topic_graph.get("edges", {})
        self.edges = edges_section.get("form_to_form", [])
        self.concept_edges = edges_section.get("concept_to_form", [])
        self.hub_patterns = edges_section.get("hub_patterns", [])

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _estimate_tokens_for_file(self, path: Path) -> int:
        """Estimate token count for a file based on character length."""
        try:
            size = path.stat().st_size
            return max(1, size // CHARS_PER_TOKEN)
        except OSError:
            return 0

    def _estimate_tokens_for_dir(self, dir_path: Path) -> int:
        """Estimate total tokens for all files in a directory tree."""
        total = 0
        if not dir_path.exists():
            return 0
        for child in dir_path.rglob("*"):
            if child.is_file():
                total += self._estimate_tokens_for_file(child)
        return total

    # ------------------------------------------------------------------
    # Query matching
    # ------------------------------------------------------------------

    def _score_form(self, form: FormEntry, query_terms: list[str]) -> float:
        """Score a form's relevance to a set of query terms.

        Checks name, description, tags, and slug. Returns a float score
        where higher is more relevant. Simple keyword/substring matching.
        """
        score = 0.0
        name_lower = form.name.lower()
        desc_lower = form.desc.lower()
        slug_lower = form.slug.lower()
        tags_lower = [t.lower() for t in form.tags]

        for term in query_terms:
            term_l = term.lower()
            # Exact tag match: highest weight
            if term_l in tags_lower:
                score += 10.0
            # Tag substring match
            elif any(term_l in tag for tag in tags_lower):
                score += 7.0
            # Name match
            if term_l in name_lower:
                score += 8.0
            # Description match
            if term_l in desc_lower:
                score += 5.0
            # Slug match
            if term_l in slug_lower:
                score += 3.0

        return score

    def _find_related_forms(self, form_ids: list[int]) -> list[int]:
        """Find forms related to the given IDs via edges in the topic graph.

        Returns form IDs that are connected (as src or tgt) but not
        already in the input set.
        """
        related: set[int] = set()
        input_set = set(form_ids)

        # form-to-form edges
        for edge in self.edges:
            src, tgt = edge.get("src"), edge.get("tgt")
            if src in input_set:
                related.add(tgt)
            if tgt in input_set:
                related.add(src)

        # hub patterns
        for hub in self.hub_patterns:
            source = hub.get("source")
            targets = hub.get("targets", [])
            if source in input_set:
                related.update(targets)
            for t in targets:
                if t in input_set:
                    related.add(source)

        # concept-to-form edges
        for concept_edge in self.concept_edges:
            targets = concept_edge.get("targets", [])
            if input_set & set(targets):
                related.update(targets)

        # xrefs from manifest
        for fid in form_ids:
            if fid in self.forms:
                related.update(self.forms[fid].xrefs)

        # Remove forms already in the input set
        related -= input_set
        # Remove invalid IDs
        related = {fid for fid in related if fid in self.forms}
        return sorted(related)

    def _match_query(self, query: str) -> list[tuple[int, float]]:
        """Match a query string against all forms and return scored results.

        Returns list of (form_id, score) tuples, sorted by descending score.
        Only includes forms with score > 0.
        """
        terms = query.lower().split()
        scored: list[tuple[int, float]] = []
        for form in self.forms.values():
            s = self._score_form(form, terms)
            if s > 0:
                scored.append((form.id, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.yaml"

    def _overview_path(self) -> Path:
        return self.index_dir / "overview.md"

    def _topic_graph_path(self) -> Path:
        return self.index_dir / "topic_graph.json"

    def _form_summary_path(self, form: FormEntry) -> Path:
        """Derive the summary file path for a form.

        Summary files follow the pattern: form_XX_slug_part.md
        where the slug transforms hyphens to underscores and drops
        the leading number prefix.
        """
        # The slug is like '01-visual' or '28-philosophy'
        # The summary filename is like 'form_01_visual.md'
        parts = form.slug.split("-", 1)
        if len(parts) == 2:
            num_part, name_part = parts
            name_part = name_part.replace("-", "_")
            filename = f"form_{num_part}_{name_part}.md"
        else:
            filename = f"form_{form.slug.replace('-', '_')}.md"
        return self.index_dir / "form_summaries" / filename

    def _form_dir(self, form: FormEntry) -> Path:
        """Return the root directory for a form's full content."""
        return self.root / form.slug

    def _info_dir(self) -> Path:
        """Return path to the 00_Info background directory."""
        return self.root / "00_Info"

    def _collect_files_in_dir(self, dir_path: Path) -> list[Path]:
        """Collect all files in a directory, sorted by name."""
        if not dir_path.exists():
            return []
        files = [f for f in dir_path.rglob("*") if f.is_file()]
        files.sort(key=lambda p: str(p))
        return files

    # ------------------------------------------------------------------
    # Core loading logic
    # ------------------------------------------------------------------

    def load(
        self,
        query: str = "",
        budget: int = 32_000,
        profile: str = "",
        form_ids: list[int] | None = None,
        cluster: str = "",
        output: str = "json",
    ) -> LoadResult:
        """Load repository content within a token budget.

        Parameters
        ----------
        query : str
            Text query for relevance matching.
        budget : int
            Maximum token budget.
        profile : str
            Named profile from token_budget_profiles.yaml. Overrides
            budget if the profile defines one.
        form_ids : list[int] | None
            Explicit form IDs to load.
        cluster : str
            Cluster name to load (e.g., "sensory", "theoretical").
        output : str
            Output mode: "paths", "content", or "json".

        Returns
        -------
        LoadResult
            Structured result with selected files and metadata.
        """
        # Resolve profile to budget
        if profile:
            if profile in PROFILE_BUDGETS:
                budget = PROFILE_BUDGETS[profile]
            elif profile in self.profiles:
                # Try to extract estimated_tokens as a budget hint
                est = self.profiles[profile].get("estimated_tokens")
                if est:
                    budget = int(est)

        result = LoadResult(budget=budget, query=query, profile=profile)

        # Determine target forms
        target_form_ids: list[int] = []
        related_form_ids: list[int] = []

        if form_ids:
            target_form_ids = [fid for fid in form_ids if fid in self.forms]
        elif cluster:
            cluster_lower = cluster.lower()
            if cluster_lower in self.clusters:
                target_form_ids = list(self.clusters[cluster_lower])
        elif query:
            scored = self._match_query(query)
            target_form_ids = [fid for fid, _ in scored]

        if target_form_ids:
            related_form_ids = self._find_related_forms(target_form_ids)
            result.forms_matched = target_form_ids

        # Build the file selection within budget
        remaining = budget
        selections: list[FileSelection] = []

        def _add_file(path: Path, reason: str) -> bool:
            """Add a file if it fits in the remaining budget. Returns True if added."""
            nonlocal remaining
            if not path.exists():
                return False
            tokens = self._estimate_tokens_for_file(path)
            if tokens > remaining:
                return False
            selections.append(FileSelection(
                path=str(path),
                estimated_tokens=tokens,
                reason=reason,
            ))
            remaining -= tokens
            return True

        def _add_dir(dir_path: Path, reason: str) -> int:
            """Add all files in a directory. Returns count of files added."""
            nonlocal remaining
            count = 0
            for f in self._collect_files_in_dir(dir_path):
                tokens = self._estimate_tokens_for_file(f)
                if tokens > remaining:
                    break
                selections.append(FileSelection(
                    path=str(f),
                    estimated_tokens=tokens,
                    reason=reason,
                ))
                remaining -= tokens
                count += 1
            return count

        # ---- Priority 1: Always include manifest.yaml ----
        _add_file(self._manifest_path(), "Core manifest - form metadata and cross-references")

        # ---- Priority 2: Include overview.md if budget >= 8K ----
        if budget >= 8_000:
            _add_file(self._overview_path(), "Executive overview of all 40 Forms")

        # ---- Priority 3: Form summaries ----
        if budget >= 32_000:
            # Load target form summaries first, then related, then all
            loaded_summaries: set[int] = set()

            # Target form summaries
            for fid in target_form_ids:
                if fid in self.forms:
                    form = self.forms[fid]
                    path = self._form_summary_path(form)
                    if _add_file(path, f"Summary for target Form {fid}: {form.name}"):
                        loaded_summaries.add(fid)

            # Related form summaries
            for fid in related_form_ids:
                if fid in self.forms and fid not in loaded_summaries:
                    form = self.forms[fid]
                    path = self._form_summary_path(form)
                    if _add_file(path, f"Summary for related Form {fid}: {form.name}"):
                        loaded_summaries.add(fid)

            # Remaining form summaries (all others)
            for fid in sorted(self.forms.keys()):
                if fid not in loaded_summaries:
                    form = self.forms[fid]
                    path = self._form_summary_path(form)
                    if not _add_file(path, f"Summary for Form {fid}: {form.name}"):
                        break  # Out of budget

        # ---- Priority 4: Full content for target forms (budget >= 128K) ----
        if budget >= 128_000 and target_form_ids:
            # Load full content for the top target form(s)
            forms_to_load_full = target_form_ids[:1] if budget < 200_000 else target_form_ids[:6]
            for fid in forms_to_load_full:
                if fid in self.forms:
                    form = self.forms[fid]
                    form_dir = self._form_dir(form)
                    _add_dir(form_dir, f"Full content for Form {fid}: {form.name}")

        # ---- Priority 5: Background info and cluster content (budget >= 200K) ----
        if budget >= 200_000:
            # Include 00_Info background
            _add_dir(self._info_dir(), "Cross-cutting background reference material")

            # Include topic_graph.json
            _add_file(self._topic_graph_path(), "Topic relationship graph")

            # Load full content for related forms if we have remaining budget
            if target_form_ids:
                loaded_full: set[int] = set(target_form_ids[:6])
                for fid in related_form_ids:
                    if fid not in loaded_full and fid in self.forms:
                        form = self.forms[fid]
                        # Skip Form 27 unless explicitly targeted (440K outlier)
                        if fid == 27 and fid not in (form_ids or []):
                            continue
                        form_dir = self._form_dir(form)
                        before = remaining
                        _add_dir(form_dir, f"Full content for related Form {fid}: {form.name}")
                        if remaining <= 0:
                            break

        result.files = selections
        result.total_estimated_tokens = sum(s.estimated_tokens for s in selections)
        return result

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def format_paths(self, result: LoadResult) -> str:
        """Format result as one file path per line."""
        return "\n".join(sel.path for sel in result.files)

    def format_content(self, result: LoadResult) -> str:
        """Format result as concatenated file content with headers."""
        parts: list[str] = []
        for sel in result.files:
            path = Path(sel.path)
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                content = f"[Error: could not read {sel.path}]"
            parts.append(f"--- {sel.path} ({sel.estimated_tokens} tokens) ---")
            parts.append(content)
        return "\n\n".join(parts)

    def format_json(self, result: LoadResult) -> str:
        """Format result as structured JSON."""
        data = {
            "files": [
                {
                    "path": sel.path,
                    "estimated_tokens": sel.estimated_tokens,
                    "reason": sel.reason,
                }
                for sel in result.files
            ],
            "total_estimated_tokens": result.total_estimated_tokens,
            "budget": result.budget,
            "query": result.query,
            "profile": result.profile,
            "forms_matched": result.forms_matched,
        }
        return json.dumps(data, indent=2)

    def format_output(self, result: LoadResult, mode: str = "json") -> str:
        """Format result in the specified output mode.

        Parameters
        ----------
        result : LoadResult
            The loading result to format.
        mode : str
            One of "paths", "content", or "json".
        """
        if mode == "paths":
            return self.format_paths(result)
        elif mode == "content":
            return self.format_content(result)
        else:
            return self.format_json(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="context_loader",
        description=(
            "Load 40-Form Consciousness Repository content within a token budget. "
            "Supports query-based relevance matching, named profiles, explicit "
            "form selection, and cluster-based loading."
        ),
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default="",
        help="Text query for relevance matching (e.g., 'visual binding problem')",
    )
    parser.add_argument(
        "-b", "--budget",
        type=int,
        default=32_000,
        help="Context window budget in tokens (default: 32000)",
    )
    parser.add_argument(
        "-p", "--profile",
        type=str,
        default="",
        help="Named profile from token_budget_profiles.yaml (e.g., 'minimal_8k', 'standard_32k')",
    )
    parser.add_argument(
        "-f", "--forms",
        type=str,
        default="",
        help="Comma-separated Form IDs to explicitly load (e.g., '1,9,16')",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        choices=["paths", "content", "json"],
        default="json",
        help="Output mode: 'paths' (file list), 'content' (concatenated), 'json' (structured)",
    )
    parser.add_argument(
        "-c", "--cluster",
        type=str,
        default="",
        help="Cluster name to load (e.g., 'sensory', 'theoretical', 'ecological')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Repository root path (default: auto-detect from script location)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Determine repository root
    if args.root:
        repo_root = Path(args.root)
    else:
        # Auto-detect: script lives in consciousness/tools/
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent

    if not (repo_root / "index" / "manifest.yaml").exists():
        print(
            f"Error: Cannot find index/manifest.yaml under {repo_root}\n"
            f"Use --root to specify the consciousness/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse explicit form IDs
    form_ids: list[int] | None = None
    if args.forms:
        try:
            form_ids = [int(x.strip()) for x in args.forms.split(",") if x.strip()]
        except ValueError:
            print("Error: --forms must be comma-separated integers (e.g., '1,9,16')", file=sys.stderr)
            sys.exit(1)

    # Validate that at least one selection criterion is provided
    if not args.query and not args.forms and not args.cluster and not args.profile:
        print(
            "Error: Provide at least one of --query, --forms, --cluster, or --profile.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load and run
    loader = ContextLoader(repo_root)
    result = loader.load(
        query=args.query,
        budget=args.budget,
        profile=args.profile,
        form_ids=form_ids,
        cluster=args.cluster,
        output=args.output,
    )

    print(loader.format_output(result, args.output))


if __name__ == "__main__":
    main()
