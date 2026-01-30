# 40 Forms of Consciousness -- Project Status

**Last Updated:** 2026-01-29
**Repository:** `consciousness/`
**Scope:** 40 Forms of consciousness, fully specified with documentation, tests, and research

---

## Repository Overview

This repository implements a comprehensive model of consciousness across 40 distinct forms, spanning phenomenal/qualitative experience (Forms 1-7), levels of awareness (Forms 8-12), functional/theoretical sub-components (Forms 13-17), contextual/specialized states (Forms 18-27), and extended ecological/developmental forms (Forms 28-40). Each Form is documented with technical specifications, processing algorithms, data structures, interface definitions, and supporting research. The project also includes a tiered access infrastructure for AI/ML systems to efficiently navigate the repository across any context window size.

---

## Current State Summary

| Metric | Value |
|--------|-------|
| Total Forms | 40 (all with content) |
| Total markdown files | ~819 |
| Total Python files | ~6,754 |
| Repository size | ~861 MB |
| Estimated documentation tokens | ~800K+ |
| Forms with 4+ spec files | 40/40 |
| Forms with test files | 40/40 |
| Forms with research directories | 40/40 |
| Tiered access index | Complete (Phase 1) |

All 40 Forms have content across their `spec/`, `tests/`, and `research/` directories. The core specification files present in every Form include `interface-spec.md`, `processing-algorithms.md`, `data-structures.md`, and `technical-requirements.md`. Some Forms (e.g., 08, 12, 14) carry additional spec files such as `neural-mapping.md`, `qualia-generation.md`, or a `README.md`.

---

## Form Inventory

| Form # | Name | Spec Files | Has Tests | Has Research | Content Tier |
|--------|------|-----------|-----------|-------------|-------------|
| 01 | Visual Consciousness | 4 | Yes | Yes | Full depth |
| 02 | Auditory Consciousness | 4 | Yes | Yes | Full depth |
| 03 | Somatosensory Consciousness | 4 | Yes | Yes | Full depth |
| 04 | Olfactory Consciousness | 4 | Yes | Yes | Full depth |
| 05 | Gustatory Consciousness | 4 | Yes | Yes | Full depth |
| 06 | Interoceptive Consciousness | 4 | Yes | Yes | Full depth |
| 07 | Emotional Consciousness | 4 | Yes | Yes | Full depth |
| 08 | Arousal / Vigilance | 5 | Yes | Yes | Full depth |
| 09 | Perceptual Consciousness | 4 | Yes | Yes | Full depth |
| 10 | Self-Recognition | 4 | Yes | Yes | Full depth |
| 11 | Meta-Consciousness | 4 | Yes | Yes | Full depth |
| 12 | Narrative Consciousness | 5 | Yes | Yes | Full depth |
| 13 | Integrated Information | 4 | Yes | Yes | Full depth |
| 14 | Global Workspace Broadcasting | 5 | Yes | Yes | Full depth |
| 15 | Higher-Order Thought (HOT) | 4 | Yes | Yes | Full depth |
| 16 | Predictive Coding / Bayesian | 4 | Yes | Yes | Full depth |
| 17 | Recurrent Processing | 4 | Yes | Yes | Full depth |
| 18 | Primary Consciousness | 4 | Yes | Yes | Full depth |
| 19 | Reflective Consciousness | 4 | Yes | Yes | Full depth |
| 20 | Collective Consciousness | 4 | Yes | Yes | Full depth |
| 21 | Artificial Consciousness | 4 | Yes | Yes | Full depth |
| 22 | Dream Consciousness | 4 | Yes | Yes | Full depth |
| 23 | Lucid Dream Consciousness | 4 | Yes | Yes | Full depth |
| 24 | Locked-In Consciousness | 4 | Yes | Yes | Full depth |
| 25 | Blindsight Consciousness | 4 | Yes | Yes | Full depth |
| 26 | Split-Brain Consciousness | 4 | Yes | Yes | Full depth |
| 27 | Altered-State Consciousness | 4 | Yes | Yes | Full depth |
| 28 | Philosophy of Consciousness | 4 | Yes | Yes | Expanded |
| 29 | Folk Wisdom / Indigenous | 4 | Yes | Yes | Expanded |
| 30 | Animal Cognition | 4 | Yes | Yes | Expanded |
| 31 | Plant Intelligence | 4 | Yes | Yes | Expanded |
| 32 | Fungal Intelligence | 4 | Yes | Yes | Expanded |
| 33 | Swarm Intelligence | 4 | Yes | Yes | Expanded |
| 34 | Gaia Intelligence | 4 | Yes | Yes | Expanded |
| 35 | Developmental Consciousness | 4 | Yes | Yes | Expanded |
| 36 | Contemplative States | 4 | Yes | Yes | Expanded |
| 37 | Psychedelic Consciousness | 4 | Yes | Yes | Expanded |
| 38 | Neurodivergent Consciousness | 4 | Yes | Yes | Expanded |
| 39 | Trauma Consciousness | 4 | Yes | Yes | Expanded |
| 40 | Xenoconsciousness | 4 | Yes | Yes | Expanded |

**Content Tier key:**
- **Full depth** (Forms 1-27): Original 27 Forms with the deepest specification history, multiple spec files, system files, validation files, and extensive documentation.
- **Expanded** (Forms 28-40): Extended Forms added later with full spec coverage, tests, and research directories. Forms 31-40 also have dedicated research subdirectories.

---

## Tiered Access Infrastructure

A machine-readable index infrastructure exists in `consciousness/index/` to support AI/ML systems navigating this repository across different context window sizes.

| File | Purpose | Status |
|------|---------|--------|
| `index/manifest.yaml` | Machine-readable listing of all 40 Forms with metadata, file paths, token counts, and topic tags | Complete |
| `index/overview.md` | Executive overview (~8K tokens) with one paragraph per Form and a quick-reference table | Complete |
| `index/form_summaries/` | 40 individual Form summaries (~1,500 tokens each) for Tier 2 loading | Complete (40 files) |
| `index/topic_graph.json` | JSON relationship graph with nodes, typed edges, and pre-computed clusters | Complete |
| `index/token_budget_profiles.yaml` | Pre-computed loading plans for 8K, 32K, 128K, 200K, and RAG context windows | Complete |

The overall tiered access plan is tracked in `TIERED_ACCESS_PLAN.md` at the repository root.

---

## Historical Context

This project evolved through several major phases:

1. **Phase 1 -- Foundation (Sep 2025):** Established the first 27 Forms as the core consciousness model. Began with Form 8 (Arousal/Vigilance) as the foundational gating mechanism, then built out Forms 13 (Integrated Information), 14 (Global Workspace), and 9 (Perceptual Consciousness) as the integration layer.

2. **Phase 2 -- Sensory & Higher-Order (Sep-Oct 2025):** Completed sensory Forms (1-2, 7), then higher-order Forms (10-12 for self-recognition, meta-consciousness, and narrative consciousness). Originally structured as 15 files per Form across `info/`, `spec/`, `system/`, and `validation/` directories.

3. **Phase 3 -- Specialized Forms (Oct 2025+):** Implemented Forms 16-27 following a priority framework: critical theory forms first (Predictive Coding, Primary Consciousness), then integration forms (Recurrent Processing, Reflective, Artificial), extended capabilities (Dream, Collective), and specialized clinical cases (Lucid Dream, Locked-In, Blindsight, Split-Brain, Altered States).

4. **Phase 4 -- Expansion to 40 Forms (2025-2026):** Extended the model beyond the original 27 Forms to include philosophy of consciousness (28), folk/indigenous wisdom (29), animal cognition (30), ecological intelligence (31-34), developmental consciousness (35), contemplative and psychedelic states (36-37), neurodivergent and trauma consciousness (38-39), and xenoconsciousness (40).

5. **Phase 5 -- Tiered Access & Parity (Jan 2026):** Built machine-readable index infrastructure for AI/ML consumption. Brought all 40 Forms to spec parity (4+ spec files each), added test files and research directories to all Forms, and consolidated overlapping root documentation.

---

## Remaining Work (Phase 3 from TIERED_ACCESS_PLAN.md)

The following Phase 3 tasks from `TIERED_ACCESS_PLAN.md` remain pending. These are optional advanced access features:

### Task 3.1: Embeddings Index
- **Directory:** `consciousness/index/embeddings/`
- **Action:** Chunk all markdown content into ~500-token segments, generate embeddings, store as `chunks.jsonl` and `vectors.npy`, create `metadata.json` mapping chunk IDs to Form/file/section.
- **Purpose:** Enables RAG-based retrieval for any context window size.
- **Status:** Pending (requires decision on embedding model/infrastructure).

### Task 3.2: API/Tool Interface
- **File:** `consciousness/tools/context_loader.py`
- **Action:** Create a Python utility that reads `manifest.yaml` and `token_budget_profiles.yaml`, accepts a query + context window size, and returns the optimal set of documents to load.
- **Status:** Pending (requires Phase 1 completion -- now satisfied).

### Task 3.3: Automated Summary Refresh
- **File:** `consciousness/tools/refresh_index.py`
- **Action:** Create a script that detects modified Forms, regenerates affected Tier 2 summaries, updates manifest token counts, and updates the topic graph.
- **Status:** Pending (requires Phase 1 completion -- now satisfied).

---

*This file consolidates and supersedes the following legacy documents, now archived in `archives/legacy_docs/`: STATUS_REPORT.md, STATUS_REPORT_UPDATED.md, PROJECT_STATUS_2025-09-22.md, PROGRESS_CHECKLIST.md, TASKS.md, Form_16-27_Priorities.md, PRIORITY_ASSESSMENT_FORMS_16-27.md.*
