# Tiered Access Architecture & Content Completion Plan

## Purpose

Transform the 40-Form Consciousness repository into a maximally usable resource for AI/ML systems by building a tiered index and summary infrastructure that algorithmically compresses and expands content to fit any context window, while allowing full-detail expansion on demand.

## Current State (as of 2026-01-29)

### Repository Statistics (Post-Phase 3)
- **40 Forms** of consciousness, all at full parity
- **~850+ markdown files**, comprehensive Python tooling
- **~4.5M tokens** of documentation content (measured)
- **22,316 chunks** indexed for RAG retrieval
- **Tiered access system** fully operational across 5 context window profiles

### Content Depth (Uniform Across All Forms)
| Feature | Status |
|---------|--------|
| Spec files | 4+ per Form (all 40 Forms) |
| Research files | All 40 Forms |
| Test files | All 40 Forms |
| Info directories | All 40 Forms |
| Form summaries | All 40 Forms (~1,600 tokens each) |
| Embeddings chunks | 22,316 chunks across 846 files |

### Navigation Infrastructure
- `index/manifest.yaml` -- Machine-readable manifest (v2.0, all 40 Forms)
- `index/overview.md` -- Executive overview (~7,400 tokens)
- `index/form_summaries/` -- 40 Tier-2 summaries with Research Highlights
- `index/topic_graph.json` -- 55 nodes, 208 edges, 8 clusters
- `index/token_budget_profiles.yaml` -- 5 loading profiles (8K to unlimited)
- `index/embeddings/` -- Chunked content (22,316 chunks), embed/search scripts
- `tools/context_loader.py` -- Query-driven context loading tool
- `tools/refresh_index.py` -- Automated staleness detection and manifest updater

---

## Phase 1: Tiered Access Infrastructure -- COMPLETE

### Task 1.1: Create Machine-Readable Manifest
- **File:** `consciousness/index/manifest.yaml`
- **Status:** `complete` -- v2.0, all 40 Forms with accurate file counts, research flags, spec counts

### Task 1.2: Create Executive Overview (Tier 1)
- **File:** `consciousness/index/overview.md`
- **Status:** `complete` -- ~7,400 tokens, 40 Form paragraphs, quick-reference table

### Task 1.3: Generate 40 Form Summaries (Tier 2)
- **Directory:** `consciousness/index/form_summaries/`
- **Status:** `complete` -- 40 summaries, all with Research Highlights sections

### Task 1.4: Create Token Budget Profiles
- **File:** `consciousness/index/token_budget_profiles.yaml`
- **Status:** `complete` -- 5 profiles with measured token estimates

### Task 1.5: Create Topic Relationship Graph
- **File:** `consciousness/index/topic_graph.json`
- **Status:** `complete` -- 55 nodes (40 Forms + 15 concepts), 208 edges, 8 clusters

---

## Phase 2: Content Completion & Parity -- COMPLETE

### Task 2.1: Forms 29-40 Spec Parity
- **Status:** `complete` -- All Forms 29-40 now have 4 spec files each

### Task 2.2: Form 27 Test File
- **File:** `consciousness/27-altered-state/tests/test_form_27.py`
- **Status:** `complete`

### Task 2.3: Research Layer for Forms 1-30
- **Status:** `complete` -- All 30 research files created

### Task 2.4: Root Documentation Consolidation
- **Status:** `complete` -- Legacy docs archived to `archives/legacy_docs/`

### Post-Phase-2: Index Refresh
- **Status:** `complete` -- manifest.yaml v2.0, all 40 summaries updated, overview rewritten, topic graph v2.0, profiles recalculated

---

## Phase 3: Advanced Access -- COMPLETE

### Task 3.1: Embeddings Index
- **Directory:** `consciousness/index/embeddings/`
- **Status:** `complete`
- **Files created:**
  - `chunker.py` -- Chunks all markdown into ~500-token segments with section-aware splitting
  - `chunks.jsonl` -- 22,316 chunks from 846 files (27 MB)
  - `metadata.json` -- Full chunk-to-Form mapping with per-Form chunk counts
  - `embed.py` -- Pluggable embedding backend (OpenAI, sentence-transformers, custom); `--dry-run` estimates $0.67 for OpenAI
  - `search.py` -- Cosine similarity search with Form filtering and top-K retrieval
- **Note:** Vector embeddings (`vectors.npy`) not yet generated -- requires `pip install openai` or `sentence-transformers` and running `embed.py`

### Task 3.2: API/Tool Interface
- **File:** `consciousness/tools/context_loader.py`
- **Status:** `complete`
- **Capabilities:**
  - Query matching against Form names, descriptions, tags, and topic graph
  - Budget-aware document selection (8K to 200K+ tokens)
  - Cluster-based and explicit Form loading
  - Three output modes: `json`, `paths`, `content`
  - Programmatic `ContextLoader` class for import

### Task 3.3: Automated Summary Refresh
- **File:** `consciousness/tools/refresh_index.py`
- **Status:** `complete`
- **Capabilities:**
  - Staleness detection via file modification times
  - File count auditing against manifest.yaml
  - Structural completeness verification (info/, spec/, research/, tests/)
  - Token estimation with budget profile comparison
  - Auto-update manifest with `--update-manifest` flag
  - Programmatic `RefreshAuditor` class for import

---

## Verification Checklist

### Phase 1 Complete:
- [x] `consciousness/index/` directory exists
- [x] `manifest.yaml` lists all 40 Forms with valid metadata
- [x] `overview.md` exists and is under 8,000 tokens
- [x] 40 form summary files exist in `form_summaries/`
- [x] Each summary is under 2,000 tokens
- [x] `token_budget_profiles.yaml` exists with all 5 profiles
- [x] `topic_graph.json` exists with nodes for all 40 Forms
- [x] All YAML/JSON files parse without errors

### Phase 2 Complete:
- [x] Forms 29-40 each have 4+ spec files
- [x] `27-altered-state/tests/test_form_27.py` exists and passes
- [x] Forms 1-30 each have a `research/` subdirectory
- [x] Root documentation consolidated to single status file
- [x] Manifest and summaries re-generated to reflect new content

### Phase 3 Complete:
- [x] Embeddings chunking infrastructure built and operational (22,316 chunks)
- [x] `context_loader.py` functional and tested
- [x] `refresh_index.py` functional and tested
- [ ] Vector embeddings generated (requires embedding model installation)

---

## Usage Guide

### Quick Start for AI/ML Systems

```bash
# Load context for a query at 32K budget
python3 consciousness/tools/context_loader.py -q "visual binding problem" -b 32000 -o paths

# Load full content for a specific cluster
python3 consciousness/tools/context_loader.py -c ecological -b 200000 -o content

# Check repository health
python3 consciousness/tools/refresh_index.py

# Generate embeddings (after installing embedding library)
python3 consciousness/index/embeddings/embed.py --backend openai --dry-run
python3 consciousness/index/embeddings/embed.py --backend openai

# Search by semantic similarity (after embeddings generated)
python3 consciousness/index/embeddings/search.py "neural correlates of visual awareness" --top-k 5
```

### Tiered Loading Strategy

| Budget | What You Get | Tool Command |
|--------|-------------|--------------|
| 3K | Form names and metadata only | Load `index/manifest.yaml` |
| 11K | Architecture overview + metadata | `-p minimal_8k` |
| 76K | All 40 Form summaries | `-p standard_32k` |
| 125K | Summaries + 1 full Form | `-p focused_128k -q "your topic"` |
| 190K | Summaries + full cluster | `-p deep_200k -c cluster_name` |
| Any | Semantic search chunks | `search.py "your query"` |

---

*Plan created 2026-01-29 for the 40-Form Consciousness Repository*
*All three phases completed 2026-01-29*
*Repository is now maximally accessible to AI/ML systems across all context window sizes*
