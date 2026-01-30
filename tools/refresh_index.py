#!/usr/bin/env python3
"""
Refresh Index Tool for the 40-Form Consciousness Repository.

Detects which Forms have been modified since the last index generation,
audits file counts against manifest.yaml, verifies structural completeness,
and optionally auto-updates manifest counts.

Usage:
    python3 refresh_index.py                     # Report only
    python3 refresh_index.py --forms 1,9,16      # Check specific Forms
    python3 refresh_index.py --update-manifest    # Auto-update manifest counts
    python3 refresh_index.py --verbose            # Show all Forms, not just stale
    python3 refresh_index.py --json               # JSON output

Programmatic usage:
    from refresh_index import RefreshAuditor
    auditor = RefreshAuditor("/path/to/consciousness/")
    report = auditor.audit()
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Data classes for audit results
# ---------------------------------------------------------------------------

@dataclass
class FormFileCount:
    """Actual file counts for a Form directory."""
    total: int = 0
    md: int = 0
    py: int = 0
    has_test: bool = False
    has_research: bool = False
    specs_count: int = 0


@dataclass
class FormAuditResult:
    """Audit result for a single Form."""
    form_id: int
    name: str
    slug: str

    # Staleness
    content_mtime: float = 0.0
    summary_mtime: float = 0.0
    manifest_mtime: float = 0.0
    is_stale: bool = False

    # File counts (actual vs manifest)
    actual_counts: FormFileCount = field(default_factory=FormFileCount)
    manifest_counts: dict[str, Any] = field(default_factory=dict)
    count_mismatches: list[str] = field(default_factory=list)

    # Structural
    missing_structure: list[str] = field(default_factory=list)
    summary_exists: bool = True


@dataclass
class AuditReport:
    """Complete audit report for the repository."""
    timestamp: str = ""
    stale_forms: list[FormAuditResult] = field(default_factory=list)
    count_mismatches: list[FormAuditResult] = field(default_factory=list)
    missing_structure: list[FormAuditResult] = field(default_factory=list)
    missing_summaries: list[int] = field(default_factory=list)
    missing_index_files: list[str] = field(default_factory=list)
    all_results: list[FormAuditResult] = field(default_factory=list)

    # Token estimation
    estimated_total_tokens: int = 0
    budget_profile_tokens: dict[str, int] = field(default_factory=dict)

    # 00_Info check
    info_dir_exists: bool = True
    info_dir_has_content: bool = True


# ---------------------------------------------------------------------------
# Core auditor class
# ---------------------------------------------------------------------------

class RefreshAuditor:
    """Audits the 40-Form Consciousness Repository for staleness and completeness.

    Args:
        root: Path to the consciousness/ repository root directory.
    """

    EXPECTED_INDEX_FILES = [
        "manifest.yaml",
        "overview.md",
        "topic_graph.json",
        "token_budget_profiles.yaml",
    ]

    REQUIRED_SUBDIRS = ["info", "spec", "research"]
    # Some Forms use 'specs' instead of 'spec'; we accept either.

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.index_dir = self.root / "index"
        self.summaries_dir = self.index_dir / "form_summaries"
        self.manifest_path = self.index_dir / "manifest.yaml"

        self._manifest: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        """Load and cache manifest.yaml."""
        if self._manifest is not None:
            return self._manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest.yaml not found at {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self._manifest = yaml.safe_load(f)
        return self._manifest

    def _get_manifest_forms(self) -> list[dict[str, Any]]:
        """Return the list of form entries from manifest."""
        manifest = self._load_manifest()
        return manifest.get("forms", [])

    def _get_manifest_info(self) -> dict[str, Any] | None:
        """Return the info (00_Info) entry from manifest."""
        manifest = self._load_manifest()
        return manifest.get("info")

    # ------------------------------------------------------------------
    # File system helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _newest_mtime(directory: Path, extensions: tuple[str, ...] = (".md", ".py")) -> float:
        """Return the newest modification time of files with given extensions in directory tree.

        Returns 0.0 if no matching files found.
        """
        newest = 0.0
        if not directory.exists():
            return newest
        for root_dir, _dirs, files in os.walk(directory):
            for fname in files:
                if fname.startswith("."):
                    continue
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = Path(root_dir) / fname
                    try:
                        mt = fpath.stat().st_mtime
                        if mt > newest:
                            newest = mt
                    except OSError:
                        continue
        return newest

    @staticmethod
    def _count_files(directory: Path) -> FormFileCount:
        """Count .md and .py files in a Form directory, check for tests and research."""
        counts = FormFileCount()
        if not directory.exists():
            return counts

        for root_dir, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            rel = Path(root_dir).relative_to(directory)

            for fname in files:
                if fname.startswith("."):
                    continue
                if fname == "__init__.py":
                    # Don't count __init__.py in totals (they're structural)
                    # but DO count them as .py files since manifest includes them
                    counts.total += 1
                    counts.py += 1
                    continue
                if fname.endswith(".md"):
                    counts.total += 1
                    counts.md += 1
                elif fname.endswith(".py"):
                    counts.total += 1
                    counts.py += 1

            # Check for tests
            parent_name = Path(root_dir).name
            if parent_name == "tests" or parent_name == "test":
                for fname in files:
                    if fname.startswith("test_") and fname.endswith(".py"):
                        counts.has_test = True

            # Check for research
            if parent_name == "research":
                for fname in files:
                    if fname.endswith(".md"):
                        counts.has_research = True

        # Count spec files
        spec_dir = directory / "spec"
        specs_dir = directory / "specs"
        target = spec_dir if spec_dir.exists() else (specs_dir if specs_dir.exists() else None)
        if target is not None:
            for fname in os.listdir(target):
                if fname.endswith(".md") and not fname.startswith("."):
                    counts.specs_count += 1

        return counts

    def _summary_filename(self, form_entry: dict[str, Any]) -> str:
        """Derive the form summary filename from a manifest form entry.

        Pattern: form_{id:02d}_{slug_suffix_underscored}.md
        Example: slug='01-visual' -> 'form_01_visual.md'
                 slug='27-altered-state' -> 'form_27_altered_state.md'
        """
        form_id: int = form_entry["id"]
        slug: str = form_entry["slug"]

        # Strip the leading number prefix (e.g., '01-' or '27-')
        # Find the first '-' and take everything after it
        parts = slug.split("-", 1)
        if len(parts) == 2:
            suffix = parts[1].replace("-", "_")
        else:
            suffix = slug.replace("-", "_")

        return f"form_{form_id:02d}_{suffix}.md"

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def _check_structure(self, form_dir: Path) -> list[str]:
        """Check that a Form directory has the required structural elements."""
        missing = []
        if not form_dir.exists():
            missing.append(f"directory {form_dir.name} does not exist")
            return missing

        # Check info/ directory
        if not (form_dir / "info").exists():
            missing.append("info/ directory")

        # Check spec/ or specs/ directory
        if not (form_dir / "spec").exists() and not (form_dir / "specs").exists():
            missing.append("spec/ (or specs/) directory")

        # Check research/ directory
        if not (form_dir / "research").exists():
            missing.append("research/ directory")

        # Check tests/ directory with test file
        tests_dir = form_dir / "tests"
        test_dir = form_dir / "test"
        if tests_dir.exists():
            has_test_file = any(
                f.startswith("test_") and f.endswith(".py")
                for f in os.listdir(tests_dir) if not f.startswith(".")
            )
            if not has_test_file:
                missing.append("test file in tests/")
        elif test_dir.exists():
            has_test_file = any(
                f.startswith("test_") and f.endswith(".py")
                for f in os.listdir(test_dir) if not f.startswith(".")
            )
            if not has_test_file:
                missing.append("test file in test/")
        else:
            missing.append("tests/ directory")

        return missing

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self) -> int:
        """Estimate total tokens by scanning all .md files and dividing chars by 4."""
        total_chars = 0
        # Scan all Form directories
        for entry in self._get_manifest_forms():
            form_dir = self.root / entry["slug"]
            if not form_dir.exists():
                continue
            for root_dir, dirs, files in os.walk(form_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in files:
                    if fname.endswith(".md") and not fname.startswith("."):
                        fpath = Path(root_dir) / fname
                        try:
                            total_chars += fpath.stat().st_size
                        except OSError:
                            continue

        # Also scan 00_Info
        info_dir = self.root / "00_Info"
        if info_dir.exists():
            for root_dir, dirs, files in os.walk(info_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in files:
                    if fname.endswith(".md") and not fname.startswith("."):
                        fpath = Path(root_dir) / fname
                        try:
                            total_chars += fpath.stat().st_size
                        except OSError:
                            continue

        # Also scan index files
        for idx_file in self.EXPECTED_INDEX_FILES:
            fpath = self.index_dir / idx_file
            if fpath.exists():
                try:
                    total_chars += fpath.stat().st_size
                except OSError:
                    pass

        # Scan summaries
        if self.summaries_dir.exists():
            for fname in os.listdir(self.summaries_dir):
                if fname.endswith(".md"):
                    fpath = self.summaries_dir / fname
                    try:
                        total_chars += fpath.stat().st_size
                    except OSError:
                        pass

        return total_chars // 4

    def _get_budget_profile_tokens(self) -> dict[str, int]:
        """Read estimated_tokens from token_budget_profiles.yaml."""
        profiles_path = self.index_dir / "token_budget_profiles.yaml"
        if not profiles_path.exists():
            return {}
        with open(profiles_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        result: dict[str, int] = {}
        profiles = data.get("profiles", {})
        for profile_name, profile_data in profiles.items():
            if isinstance(profile_data, dict) and "estimated_tokens" in profile_data:
                result[profile_name] = profile_data["estimated_tokens"]
        return result

    # ------------------------------------------------------------------
    # Main audit
    # ------------------------------------------------------------------

    def audit(self, form_ids: list[int] | None = None) -> AuditReport:
        """Run a complete audit of the repository.

        Args:
            form_ids: If provided, only audit these Form IDs. Otherwise audit all.

        Returns:
            AuditReport with all findings.
        """
        report = AuditReport(
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        manifest = self._load_manifest()
        manifest_forms = self._get_manifest_forms()
        manifest_mtime = self.manifest_path.stat().st_mtime if self.manifest_path.exists() else 0.0

        # Filter to requested forms if specified
        if form_ids is not None:
            manifest_forms = [f for f in manifest_forms if f["id"] in form_ids]

        # Check each Form
        for entry in manifest_forms:
            form_id = entry["id"]
            slug = entry["slug"]
            name = entry.get("name", slug)
            form_dir = self.root / slug

            result = FormAuditResult(
                form_id=form_id,
                name=name,
                slug=slug,
                manifest_counts={
                    "files": entry.get("files", 0),
                    "md": entry.get("md", 0),
                    "py": entry.get("py", 0),
                    "test": entry.get("test", False),
                    "research": entry.get("research", False),
                    "specs": entry.get("specs", 0),
                },
            )

            # --- Staleness detection ---
            result.content_mtime = self._newest_mtime(form_dir)
            summary_file = self.summaries_dir / self._summary_filename(entry)
            if summary_file.exists():
                result.summary_mtime = summary_file.stat().st_mtime
                result.summary_exists = True
            else:
                result.summary_mtime = 0.0
                result.summary_exists = False

            result.manifest_mtime = manifest_mtime

            # A form is stale if content is newer than both summary and manifest
            if result.content_mtime > 0:
                if not result.summary_exists:
                    result.is_stale = True
                elif result.content_mtime > result.summary_mtime:
                    result.is_stale = True
                elif result.content_mtime > result.manifest_mtime:
                    result.is_stale = True

            # --- File count audit ---
            actual = self._count_files(form_dir)
            result.actual_counts = actual

            mc = result.manifest_counts
            if actual.total != mc["files"]:
                result.count_mismatches.append(
                    f"total files: manifest={mc['files']}, actual={actual.total}"
                )
            if actual.md != mc["md"]:
                result.count_mismatches.append(
                    f"markdown: manifest={mc['md']}, actual={actual.md}"
                )
            if actual.py != mc["py"]:
                result.count_mismatches.append(
                    f"python: manifest={mc['py']}, actual={actual.py}"
                )
            if actual.has_test != mc["test"]:
                result.count_mismatches.append(
                    f"test: manifest={mc['test']}, actual={actual.has_test}"
                )
            if actual.has_research != mc["research"]:
                result.count_mismatches.append(
                    f"research: manifest={mc['research']}, actual={actual.has_research}"
                )
            if actual.specs_count != mc["specs"]:
                result.count_mismatches.append(
                    f"specs: manifest={mc['specs']}, actual={actual.specs_count}"
                )

            # --- Structural completeness ---
            result.missing_structure = self._check_structure(form_dir)

            # Collect into report
            report.all_results.append(result)
            if result.is_stale:
                report.stale_forms.append(result)
            if result.count_mismatches:
                report.count_mismatches.append(result)
            if result.missing_structure:
                report.missing_structure.append(result)

        # --- Check all 40 summaries exist ---
        all_forms = self._get_manifest_forms()
        for entry in all_forms:
            summary_file = self.summaries_dir / self._summary_filename(entry)
            if not summary_file.exists():
                report.missing_summaries.append(entry["id"])

        # --- Check index files ---
        for idx_file in self.EXPECTED_INDEX_FILES:
            if not (self.index_dir / idx_file).exists():
                report.missing_index_files.append(idx_file)
        if not self.summaries_dir.exists():
            report.missing_index_files.append("form_summaries/")

        # --- Check 00_Info ---
        info_dir = self.root / "00_Info"
        report.info_dir_exists = info_dir.exists()
        if info_dir.exists():
            md_files = [f for f in os.listdir(info_dir) if f.endswith(".md") and not f.startswith(".")]
            report.info_dir_has_content = len(md_files) > 0
        else:
            report.info_dir_has_content = False

        # --- Token estimation ---
        report.estimated_total_tokens = self._estimate_tokens()
        report.budget_profile_tokens = self._get_budget_profile_tokens()

        return report

    # ------------------------------------------------------------------
    # Manifest updater
    # ------------------------------------------------------------------

    def update_manifest(self, form_ids: list[int] | None = None) -> list[str]:
        """Re-count files in each Form directory and update manifest.yaml.

        Args:
            form_ids: If provided, only update these Form IDs. Otherwise update all.

        Returns:
            List of changes made (human-readable strings).
        """
        manifest = self._load_manifest()
        forms_list = manifest.get("forms", [])
        changes: list[str] = []

        total_files = 0
        total_md = 0

        for entry in forms_list:
            form_id = entry["id"]
            slug = entry["slug"]
            form_dir = self.root / slug

            if form_ids is not None and form_id not in form_ids:
                # Keep existing counts for non-targeted forms
                total_files += entry.get("files", 0)
                total_md += entry.get("md", 0)
                continue

            actual = self._count_files(form_dir)

            # Track changes
            old_files = entry.get("files", 0)
            old_md = entry.get("md", 0)
            old_py = entry.get("py", 0)
            old_test = entry.get("test", False)
            old_research = entry.get("research", False)
            old_specs = entry.get("specs", 0)

            diffs = []
            if actual.total != old_files:
                diffs.append(f"files: {old_files}->{actual.total}")
            if actual.md != old_md:
                diffs.append(f"md: {old_md}->{actual.md}")
            if actual.py != old_py:
                diffs.append(f"py: {old_py}->{actual.py}")
            if actual.has_test != old_test:
                diffs.append(f"test: {old_test}->{actual.has_test}")
            if actual.has_research != old_research:
                diffs.append(f"research: {old_research}->{actual.has_research}")
            if actual.specs_count != old_specs:
                diffs.append(f"specs: {old_specs}->{actual.specs_count}")

            if diffs:
                changes.append(f"Form {form_id:02d} ({slug}): {', '.join(diffs)}")

            # Update entry
            entry["files"] = actual.total
            entry["md"] = actual.md
            entry["py"] = actual.py
            entry["test"] = actual.has_test
            entry["research"] = actual.has_research
            entry["specs"] = actual.specs_count

            total_files += actual.total
            total_md += actual.md

        # Also recount 00_Info for header totals
        info_entry = manifest.get("info")
        if info_entry:
            info_dir = self.root / "00_Info"
            info_actual = self._count_files(info_dir)
            info_entry["files"] = info_actual.total
            info_entry["md"] = info_actual.md
            info_entry["py"] = info_actual.py
            total_files += info_actual.total
            total_md += info_actual.md

        # Update header totals
        header = manifest.get("header", {})
        old_total = header.get("total_files", 0)
        old_total_md = header.get("total_markdown", 0)
        if total_files != old_total:
            changes.append(f"Header total_files: {old_total}->{total_files}")
        if total_md != old_total_md:
            changes.append(f"Header total_markdown: {old_total_md}->{total_md}")
        header["total_files"] = total_files
        header["total_markdown"] = total_md
        header["generated"] = datetime.now().strftime("%Y-%m-%d")

        # Write manifest back
        # We use a custom approach to preserve the compact flow-style for form entries
        self._write_manifest(manifest)

        # Clear cached manifest
        self._manifest = None

        return changes

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write manifest.yaml preserving the compact flow-style form entries."""
        lines: list[str] = []
        lines.append("# 40-Form Consciousness Repository Manifest")

        # Header
        header = manifest["header"]
        lines.append("header:")
        lines.append(f'  version: "{header["version"]}"')
        lines.append(f'  generated: "{header["generated"]}"')
        lines.append(f"  total_forms: {header['total_forms']}")
        lines.append(f"  total_files: {header['total_files']}")
        lines.append(f"  total_markdown: {header['total_markdown']}")
        lines.append("")

        # Info section
        info = manifest.get("info")
        if info:
            lines.append("info:")
            lines.append(f"  id: {info['id']}")
            lines.append(f"  name: {info['name']}")
            lines.append(f"  slug: {info['slug']}")
            lines.append(f"  desc: {info['desc']}")
            lines.append(f"  files: {info['files']}")
            lines.append(f"  md: {info['md']}")
            lines.append(f"  py: {info['py']}")
            lines.append(f"  test: {str(info['test']).lower()}")
            lines.append(f"  research: {str(info['research']).lower()}")
            lines.append(f"  specs: {info['specs']}")
            tags_str = "[" + ", ".join(info["tags"]) + "]"
            lines.append(f"  tags: {tags_str}")
            xrefs_str = "[" + ",".join(str(x) for x in info["xrefs"]) + "]"
            lines.append(f"  xrefs: {xrefs_str}")
            lines.append(f"  tier: {info['tier']}")
            lines.append("")

        # Forms
        lines.append("forms:")
        for entry in manifest["forms"]:
            tags_str = "[" + ", ".join(entry["tags"]) + "]"
            xrefs_str = "[" + ",".join(str(x) for x in entry["xrefs"]) + "]"
            test_str = "true" if entry["test"] else "false"
            research_str = "true" if entry["research"] else "false"

            # Escape description for YAML inline
            desc = entry["desc"]
            desc_escaped = desc.replace('"', '\\"')

            line = (
                f'- {{id: {entry["id"]}, '
                f'name: {entry["name"]}, '
                f'slug: {entry["slug"]}, '
                f'desc: "{desc_escaped}", '
                f'files: {entry["files"]}, '
                f'md: {entry["md"]}, '
                f'py: {entry["py"]}, '
                f'test: {test_str}, '
                f'research: {research_str}, '
                f'specs: {entry["specs"]}, '
                f'tags: {tags_str}, '
                f'xrefs: {xrefs_str}, '
                f'tier: {entry["tier"]}}}'
            )
            lines.append(line)

        lines.append("")  # trailing newline

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------

def _format_timestamp(mtime: float) -> str:
    """Format a modification time as human-readable string."""
    if mtime == 0.0:
        return "N/A"
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")


def format_text_report(report: AuditReport, verbose: bool = False) -> str:
    """Format the audit report as human-readable text."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  40-Form Consciousness Repository - Refresh Audit Report")
    lines.append(f"  Generated: {report.timestamp}")
    lines.append("=" * 72)
    lines.append("")

    # --- Summary ---
    total = len(report.all_results)
    lines.append(f"Forms audited: {total}")
    lines.append(f"Stale forms (content newer than index): {len(report.stale_forms)}")
    lines.append(f"File count mismatches: {len(report.count_mismatches)}")
    lines.append(f"Structural issues: {len(report.missing_structure)}")
    lines.append("")

    # --- Stale Forms ---
    if report.stale_forms:
        lines.append("-" * 72)
        lines.append("STALE FORMS (content newer than summary/manifest)")
        lines.append("-" * 72)
        for r in report.stale_forms:
            lines.append(f"  Form {r.form_id:02d}: {r.name} ({r.slug})")
            lines.append(f"    Content modified: {_format_timestamp(r.content_mtime)}")
            if r.summary_exists:
                lines.append(f"    Summary modified: {_format_timestamp(r.summary_mtime)}")
            else:
                lines.append("    Summary: MISSING")
            lines.append(f"    Manifest modified: {_format_timestamp(r.manifest_mtime)}")
        lines.append("")

    # --- File Count Mismatches ---
    if report.count_mismatches:
        lines.append("-" * 72)
        lines.append("FILE COUNT MISMATCHES (manifest vs. actual)")
        lines.append("-" * 72)
        for r in report.count_mismatches:
            lines.append(f"  Form {r.form_id:02d}: {r.name} ({r.slug})")
            for mismatch in r.count_mismatches:
                lines.append(f"    - {mismatch}")
        lines.append("")

    # --- Structural Issues ---
    if report.missing_structure:
        lines.append("-" * 72)
        lines.append("STRUCTURAL ISSUES (missing directories/files)")
        lines.append("-" * 72)
        for r in report.missing_structure:
            lines.append(f"  Form {r.form_id:02d}: {r.name} ({r.slug})")
            for issue in r.missing_structure:
                lines.append(f"    - Missing: {issue}")
        lines.append("")

    # --- Missing Summaries ---
    if report.missing_summaries:
        lines.append("-" * 72)
        lines.append("MISSING FORM SUMMARIES")
        lines.append("-" * 72)
        for fid in report.missing_summaries:
            lines.append(f"  Form {fid:02d}: no summary in index/form_summaries/")
        lines.append("")

    # --- Missing Index Files ---
    if report.missing_index_files:
        lines.append("-" * 72)
        lines.append("MISSING INDEX FILES")
        lines.append("-" * 72)
        for fname in report.missing_index_files:
            lines.append(f"  - {fname}")
        lines.append("")

    # --- 00_Info check ---
    if not report.info_dir_exists:
        lines.append("WARNING: 00_Info/ directory does not exist")
        lines.append("")
    elif not report.info_dir_has_content:
        lines.append("WARNING: 00_Info/ directory exists but has no .md files")
        lines.append("")

    # --- Token Estimation ---
    lines.append("-" * 72)
    lines.append("TOKEN ESTIMATION")
    lines.append("-" * 72)
    lines.append(f"  Estimated total tokens (all .md content): {report.estimated_total_tokens:,}")
    if report.budget_profile_tokens:
        lines.append("  Token budget profiles (from token_budget_profiles.yaml):")
        for profile, tokens in sorted(report.budget_profile_tokens.items()):
            lines.append(f"    {profile}: {tokens:,}")
    lines.append("")

    # --- Verbose: All Forms ---
    if verbose:
        lines.append("-" * 72)
        lines.append("ALL FORMS DETAIL")
        lines.append("-" * 72)
        for r in report.all_results:
            status_parts = []
            if r.is_stale:
                status_parts.append("STALE")
            if r.count_mismatches:
                status_parts.append("COUNT-MISMATCH")
            if r.missing_structure:
                status_parts.append("STRUCTURAL-ISSUE")
            if not r.summary_exists:
                status_parts.append("NO-SUMMARY")
            status = " | ".join(status_parts) if status_parts else "OK"

            lines.append(
                f"  Form {r.form_id:02d}: {r.name:<35s} [{status}]"
            )
            lines.append(
                f"    Files: {r.actual_counts.total} total, "
                f"{r.actual_counts.md} md, {r.actual_counts.py} py, "
                f"{r.actual_counts.specs_count} specs | "
                f"test={r.actual_counts.has_test}, research={r.actual_counts.has_research}"
            )
        lines.append("")

    # --- Clean bill of health ---
    if (
        not report.stale_forms
        and not report.count_mismatches
        and not report.missing_structure
        and not report.missing_summaries
        and not report.missing_index_files
    ):
        lines.append("All checks passed. Repository index is up to date.")
        lines.append("")

    return "\n".join(lines)


def format_json_report(report: AuditReport) -> str:
    """Format the audit report as JSON."""
    data: dict[str, Any] = {
        "timestamp": report.timestamp,
        "summary": {
            "forms_audited": len(report.all_results),
            "stale_forms": len(report.stale_forms),
            "count_mismatches": len(report.count_mismatches),
            "structural_issues": len(report.missing_structure),
            "missing_summaries": len(report.missing_summaries),
            "missing_index_files": len(report.missing_index_files),
        },
        "stale_forms": [
            {
                "id": r.form_id,
                "name": r.name,
                "slug": r.slug,
                "content_modified": _format_timestamp(r.content_mtime),
                "summary_modified": _format_timestamp(r.summary_mtime),
                "summary_exists": r.summary_exists,
            }
            for r in report.stale_forms
        ],
        "count_mismatches": [
            {
                "id": r.form_id,
                "name": r.name,
                "slug": r.slug,
                "mismatches": r.count_mismatches,
                "actual": {
                    "files": r.actual_counts.total,
                    "md": r.actual_counts.md,
                    "py": r.actual_counts.py,
                    "test": r.actual_counts.has_test,
                    "research": r.actual_counts.has_research,
                    "specs": r.actual_counts.specs_count,
                },
                "manifest": r.manifest_counts,
            }
            for r in report.count_mismatches
        ],
        "structural_issues": [
            {
                "id": r.form_id,
                "name": r.name,
                "slug": r.slug,
                "missing": r.missing_structure,
            }
            for r in report.missing_structure
        ],
        "missing_summaries": report.missing_summaries,
        "missing_index_files": report.missing_index_files,
        "info_dir": {
            "exists": report.info_dir_exists,
            "has_content": report.info_dir_has_content,
        },
        "token_estimation": {
            "estimated_total_tokens": report.estimated_total_tokens,
            "budget_profiles": report.budget_profile_tokens,
        },
        "all_forms": [
            {
                "id": r.form_id,
                "name": r.name,
                "slug": r.slug,
                "stale": r.is_stale,
                "summary_exists": r.summary_exists,
                "actual_files": r.actual_counts.total,
                "actual_md": r.actual_counts.md,
                "actual_py": r.actual_counts.py,
                "actual_specs": r.actual_counts.specs_count,
                "has_test": r.actual_counts.has_test,
                "has_research": r.actual_counts.has_research,
                "count_mismatches": r.count_mismatches,
                "missing_structure": r.missing_structure,
            }
            for r in report.all_results
        ],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Refresh audit tool for the 40-Form Consciousness Repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python3 refresh_index.py                     Report only (default)
  python3 refresh_index.py --forms 1,9,16      Check specific Forms
  python3 refresh_index.py --update-manifest    Auto-update manifest.yaml counts
  python3 refresh_index.py --verbose            Show all Forms in detail
  python3 refresh_index.py --json               JSON output for scripting
""",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to consciousness/ repository root (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default=None,
        help="Comma-separated Form IDs to check (e.g., 1,9,16)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Auto-update manifest.yaml with actual file counts",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all Forms in detail, not just those with issues",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output report as JSON",
    )

    args = parser.parse_args()

    # Determine root
    if args.root:
        root = Path(args.root)
    else:
        # Auto-detect: this script lives in consciousness/tools/
        root = Path(__file__).resolve().parent.parent

    if not root.exists():
        print(f"Error: root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    if not (root / "index" / "manifest.yaml").exists():
        print(f"Error: manifest.yaml not found in {root / 'index'}", file=sys.stderr)
        sys.exit(1)

    # Parse form IDs
    form_ids: list[int] | None = None
    if args.forms:
        try:
            form_ids = [int(x.strip()) for x in args.forms.split(",")]
        except ValueError:
            print(f"Error: invalid form IDs: {args.forms}", file=sys.stderr)
            sys.exit(1)

    auditor = RefreshAuditor(root)

    # Run manifest update if requested
    if args.update_manifest:
        print("Updating manifest.yaml...\n")
        changes = auditor.update_manifest(form_ids)
        if changes:
            print("Changes made:")
            for change in changes:
                print(f"  {change}")
        else:
            print("No changes needed — manifest counts are accurate.")
        print()

    # Run audit
    report = auditor.audit(form_ids)

    if args.json_output:
        print(format_json_report(report))
    else:
        print(format_text_report(report, verbose=args.verbose))


if __name__ == "__main__":
    main()
