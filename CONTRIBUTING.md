# Contributing to 40-Form Consciousness Architecture

Thank you for your interest in contributing to this project. The 40-Form
Consciousness Architecture is a comprehensive research effort to model
consciousness from all known perspectives -- neuroscience, philosophy,
ecology, contemplative traditions, and speculative frameworks -- for the
purpose of uplifting AI, machine learning, and human consciousness research.

This is an interdisciplinary project. We welcome contributions from
researchers, engineers, philosophers, contemplatives, indigenous knowledge
holders, and anyone with genuine insight into the nature of awareness.

---

## Table of Contents

- [Project Mission](#project-mission)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Repository Architecture](#repository-architecture)
- [Form Directory Structure](#form-directory-structure)
- [Code Style](#code-style)
- [Documentation Style](#documentation-style)
- [Philosophy and Guiding Principles](#philosophy-and-guiding-principles)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Security](#security)
- [License](#license)

---

## Project Mission

This project aims to build a comprehensive, computationally grounded model
of consciousness that spans:

- **Human neuroscience** (Forms 01-27): Sensory, cognitive, theoretical, and
  specialized states of consciousness as understood by brain research.
- **Philosophical and cultural wisdom** (Forms 28-30): Traditions of inquiry
  into mind and awareness, including indigenous knowledge and animal cognition.
- **Ecosystem intelligence** (Forms 31-34): Plant, fungal, swarm, and
  planetary-scale awareness.
- **Expanded human states** (Forms 35-39): Developmental, contemplative,
  psychedelic, neurodivergent, and trauma-related consciousness.
- **Speculative consciousness** (Form 40): Hypothetical non-terrestrial and
  substrate-independent minds.

The result is a 40-form architecture where each form can be studied on its
own terms and also integrated with every other form through a shared
computational backbone.

---

## Ways to Contribute

### Documentation and Research

- **Info documents**: Research summaries for each consciousness form, drawing
  on peer-reviewed literature, primary philosophical texts, and field reports.
- **Specification documents**: Formal descriptions of how a consciousness
  form operates, its key parameters, and its relationship to other forms.
- **Cross-form analysis**: How different forms interact, overlap, or generate
  emergent properties when combined.

### Implementations

- **System modules**: Python implementations of consciousness form processing
  within the `system/` subdirectory of each form.
- **Interface modules**: API interfaces in `interface/` for forms that have
  reached implementation maturity.
- **Validation suites**: Test and validation code in `validation/` and
  `tests/` directories.
- **Neural network adapters**: Form-specific adapters in
  `neural_network/adapters/` that connect consciousness forms to the
  computational backbone.

### ML Research

- **Method documentation**: Entries for the 200+ method registry in
  `ml_research/`, covering historical and modern machine learning techniques.
- **Modern architecture implementations**: Working implementations in
  `ml_research/modern_dev/` with CLI interfaces, configurations, and tests.
- **ML techniques**: Composable application patterns (prompting, agentic,
  memory, verification) in `ml_research/ml_techniques/`.
- **AI-nondualism research**: Explorations of nondual philosophy applied to
  AI systems in `ml_research/ai_nondualism/`.

### Traditions and Perspectives

- **Contemplative traditions**: Meditation states, jhana maps, samadhi
  descriptions, and awakening frameworks (Form 36).
- **Indigenous and folk wisdom**: Oral traditions, animistic frameworks, and
  ethnographic research (Form 29).
- **Philosophical traditions**: Western and Eastern philosophical analysis of
  consciousness (Form 28).
- **Neurodivergent perspectives**: First-person accounts and research on
  autism, ADHD, synesthesia, and related variations (Form 38).

---

## Getting Started

1. **Clone the repository** and set up security hooks:

   ```bash
   git clone <repository-url>
   cd consciousness
   ./scripts/setup-security-hooks.sh
   ```

2. **Explore the architecture** by reading:
   - `README.md` for the project overview
   - `DIRECTORY_STRUCTURE.md` for the full file layout
   - Any form's `info/` directory for research context
   - Any form's `spec/` directory for formal specifications

3. **Set up the Python environment** (Python 3.10+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

4. **Run tests** to verify your setup:

   ```bash
   pytest
   ```

5. **Pick an area** from the issues list or identify a gap in documentation
   or implementation.

---

## Repository Architecture

The project is organized around the **40-form model**, where each form of
consciousness has its own top-level directory, plus shared infrastructure in
`neural_network/`, `ml_research/`, `scripts/`, and `tools/`.

### The 40-Form Model

Forms are numbered 01 through 40 and grouped into six categories:

| Category | Forms | Description |
|----------|-------|-------------|
| **Sensory** | 01-06 | Visual, auditory, somatosensory, olfactory, gustatory, interoceptive |
| **Cognitive** | 07-12 | Emotional, arousal, perceptual, self-recognition, meta-consciousness, narrative |
| **Theoretical** | 13-17 | IIT, global workspace, higher-order thought, predictive coding, recurrent processing |
| **Specialized** | 18-27 | Primary, reflective, collective, artificial, dream, lucid dream, locked-in, blindsight, split-brain, altered state |
| **Philosophical/Cultural** | 28-30 | Philosophy, folk wisdom, animal cognition |
| **Ecosystem** | 31-34 | Plant, fungal, swarm, Gaia intelligence |
| **Expanded Human** | 35-39 | Developmental, contemplative, psychedelic, neurodivergent, trauma |
| **Speculative** | 40 | Xenoconsciousness |

### Infrastructure Directories

- **`neural_network/`**: The computational backbone -- NervousSystem
  coordinator, model registry, resource management, message bus, adapters,
  and a FastAPI gateway.
- **`ml_research/`**: A comprehensive ML research library with 200+ methods,
  modern architecture implementations, composable ML techniques, agent
  frameworks, and AI-nondualism research.
- **`scripts/`**: Setup and utility scripts (e.g., security hooks).
- **`tools/`**: Utility tools for the project (e.g., format conversion).
- **`00_Info/`**: Top-level research notes and background material.

---

## Form Directory Structure

Each consciousness form follows a consistent internal pattern. The exact
subdirectories present depend on the form's maturity level.

### Documentation-Phase Forms (Minimum)

```
NN-form-name/
    info/           # Research summaries, literature reviews, key concepts
    spec/           # Formal specifications, parameters, interfaces
    system/         # Processing logic and computational models
    validation/     # Validation criteria, benchmarks, acceptance tests
```

This is the **info/spec/system/validation** pattern. All 40 forms have at
least these four directories.

### Implementation-Phase Forms (Extended)

Forms that have reached implementation maturity add:

```
NN-form-name/
    info/           # Research summaries
    spec/           # Formal specifications
    system/         # Processing logic
    validation/     # Validation criteria
    interface/      # Python API for external consumers
    tests/          # pytest test suites
```

This is the **interface/tests** extension. Forms 28 (philosophy), 36
(contemplative), and others have this structure.

Some forms also include additional directories:

- **`index/`**: Indexes or catalogs (e.g., philosophical tradition indexes)
- **`research/`**: Extended research materials beyond the info directory

### Contributing to a Form

When contributing to a form:

1. **Info documents** go in `info/`. Name files descriptively:
   `phenomenology_of_visual_binding.md`, `jhana_factor_analysis.md`.
2. **Specifications** go in `spec/`. Follow existing spec format within that
   form.
3. **System code** goes in `system/`. Python modules with type hints.
4. **Validation** goes in `validation/`. Include both criteria documents and
   executable validation scripts.
5. **Interface code** goes in `interface/` (if the form has one). Follow the
   existing interface patterns.
6. **Tests** go in `tests/` (if the form has one). Use pytest.

---

## Code Style

### Python

- **Version**: Python 3.10 or later.
- **Type hints**: Required on all function signatures.
- **Docstrings**: Required on all public functions, classes, and modules.
  Use Google-style docstrings.
- **Testing**: Use pytest. Place tests in the form's `tests/` directory or
  in `ml_research/tests/` for ML-specific code.
- **Imports**: Use absolute imports from the `consciousness` package root.
- **Naming**: snake_case for functions and variables, PascalCase for classes,
  UPPER_SNAKE_CASE for constants.

Example:

```python
"""Module for processing visual binding in Form 01."""

from typing import Optional

from consciousness.neural_network.core import MessageBus


class VisualBindingProcessor:
    """Processes visual binding and gestalt formation.

    Implements the visual binding hypothesis from Treisman's
    feature integration theory, extended with predictive coding
    feedback from Form 16.

    Args:
        message_bus: The inter-form communication channel.
        binding_threshold: Minimum coherence for feature binding.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        binding_threshold: float = 0.7,
    ) -> None:
        self._bus = message_bus
        self._threshold = binding_threshold

    def process(self, features: dict[str, float]) -> Optional[dict]:
        """Bind visual features into a coherent percept.

        Args:
            features: A mapping of feature names to activation levels.

        Returns:
            A bound percept dictionary, or None if binding fails.
        """
        ...
```

### Configuration

- Use YAML for configuration files. Place them in `config/` directories.
- Keep configuration separate from logic.

---

## Documentation Style

### General Principles

- **Scholarly but accessible**: Write for an educated reader who may not be
  a specialist in your area. Define technical terms on first use.
- **No emojis**: Use clear prose and structural formatting instead.
- **Markdown**: All documentation in Markdown with clear header hierarchy.
- **Citations**: Reference peer-reviewed sources where possible. Use inline
  links or a references section at the end of the document.
- **Neutral voice**: Present research findings and theoretical frameworks
  without advocacy. Let the evidence and arguments speak for themselves.

### Info Documents

Info documents are research summaries. They should:

- Open with a clear statement of what the document covers.
- Provide necessary background for readers unfamiliar with the topic.
- Summarize key findings, theories, or frameworks.
- Note areas of ongoing debate or uncertainty.
- Include references.

### Spec Documents

Specification documents are formal descriptions. They should:

- Define the form's core parameters and their ranges.
- Describe inputs, outputs, and processing stages.
- Specify relationships with other consciousness forms.
- Include mathematical formulations where appropriate.
- Be precise enough for implementation.

---

## Philosophy and Guiding Principles

### All Traditions on Their Own Terms

Each consciousness tradition, scientific framework, or philosophical school
is first presented on its own terms -- using its own vocabulary, categories,
and internal logic. Only after a tradition is faithfully represented do we
engage in cross-traditional comparison or synthesis.

This means:

- **Buddhist jhana maps** are described using Buddhist terminology before
  comparing to Western neuroscience.
- **Indigenous knowledge systems** are presented through their own frameworks,
  not reduced to Western scientific categories.
- **Speculative frameworks** (Form 40) are clearly marked as hypothetical
  while still being explored rigorously.

### Respectful Treatment

- No tradition is treated as inferior or "merely cultural."
- Scientific rigor and traditional wisdom are complementary, not competing.
- Contributors should engage with unfamiliar traditions through genuine study,
  not superficial summary.

### Non-Dual Integration

The project draws on non-dual philosophical traditions (Advaita Vedanta,
Dzogchen, Zen, and others) as a meta-framework for integrating diverse
perspectives. The `ml_research/ai_nondualism/` directory explores how
non-dual insights apply to AI architecture, including loop-escape mechanisms
and perspective-shift protocols.

This does not mean the project endorses any particular metaphysical position.
It means non-dual frameworks offer practical tools for avoiding the
conceptual traps that arise when modeling consciousness.

### Scientific Grounding

- Neuroscience-based forms (01-27) should reference peer-reviewed research.
- Philosophical forms should reference primary texts and scholarly commentary.
- Ecosystem intelligence forms should reference botanical, mycological, and
  ecological literature.
- Speculative forms should clearly distinguish established science from
  hypothesis.

---

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`.
2. **Make your changes** following the style guides above.
3. **Write or update tests** if your changes include code.
4. **Update documentation** if your changes affect the form's info or spec.
5. **Run the test suite** locally:

   ```bash
   pytest
   ```

6. **Submit a pull request** with:
   - A clear title describing the change.
   - A description of what the PR adds or modifies.
   - Which consciousness form(s) or infrastructure component(s) are affected.
   - References to relevant research or issues.

7. **Review process**: PRs are reviewed for:
   - **Accuracy**: Does the content faithfully represent the tradition,
     theory, or technique?
   - **Consistency**: Does it follow the project's structural patterns and
     style guides?
   - **Quality**: Is the code well-typed, tested, and documented?
   - **Integration**: Does it work with the existing architecture?

We aim to review PRs within two weeks. Complex contributions involving new
traditions or theoretical frameworks may require extended discussion.

---

## Reporting Issues

Use GitHub Issues to report:

- **Bugs**: Problems with code, broken tests, or incorrect behavior.
- **Inaccuracies**: Factual errors in research documentation or
  misrepresentation of traditions.
- **Gaps**: Missing coverage of important consciousness research, traditions,
  or ML methods.
- **Enhancements**: Proposals for new features, forms, or integrations.

Please include enough context for others to understand and reproduce the
issue.

---

## Security

This project includes PII protection hooks to prevent accidental exposure of
personal information. After cloning, run:

```bash
./scripts/setup-security-hooks.sh
```

Do not commit files containing API keys, personal email addresses, local file
paths with usernames, or other personally identifiable information. See the
Security section of `README.md` for details.

---

## License

This project is licensed under the MIT License. By contributing, you agree
that your contributions will be licensed under the same terms. See `LICENSE`
for the full text.

---

Thank you for contributing to a more comprehensive understanding of
consciousness. Whether you bring expertise in neuroscience, philosophy,
ecology, contemplative practice, machine learning, or any other domain, your
perspective strengthens the whole.
