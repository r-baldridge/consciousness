# Processing Algorithms

## Overview

This document specifies the core algorithms used by Form 28 for philosophical reasoning, cross-tradition synthesis, knowledge gap detection, and wisdom integration. These algorithms enable nuanced philosophical processing while maintaining authenticity to source traditions.

---

## Semantic Retrieval Algorithm

### RAG-Based Concept Retrieval

```python
async def retrieve_philosophical_concepts(
    query: str,
    filters: QueryFilters,
    top_k: int = 10,
    min_similarity: float = 0.5
) -> List[ScoredConcept]:
    """
    Retrieve relevant philosophical concepts using RAG.

    Algorithm:
    1. Embed query using sentence-transformers
    2. Search vector store for similar embeddings
    3. Apply filters (tradition, domain, maturity)
    4. Expand results via knowledge graph
    5. Re-rank based on relevance and authority
    """

    # Step 1: Query embedding
    query_embedding = await embed_text(query)

    # Step 2: Vector similarity search
    candidates = await vector_search(
        collection="concepts",
        query_vector=query_embedding,
        top_k=top_k * 3,  # Over-fetch for filtering
        min_similarity=min_similarity * 0.8  # Slightly lower for filtering headroom
    )

    # Step 3: Apply filters
    filtered = apply_filters(candidates, filters)

    # Step 4: Knowledge graph expansion
    expanded = await expand_via_graph(
        concepts=filtered[:top_k],
        relationship_types=["RELATED_TO", "PREREQUISITE_OF"],
        max_expansion=5
    )

    # Step 5: Re-rank
    reranked = rerank_results(
        candidates=filtered + expanded,
        query=query,
        weights={
            "semantic_similarity": 0.4,
            "tradition_relevance": 0.2,
            "maturity_score": 0.2,
            "authority_score": 0.2
        }
    )

    return reranked[:top_k]
```

### Multi-Hop Reasoning

```python
async def multi_hop_query(
    question: str,
    max_hops: int = 3
) -> MultiHopResult:
    """
    Answer complex philosophical questions requiring multiple reasoning steps.

    Algorithm:
    1. Decompose question into sub-questions
    2. Answer each sub-question
    3. Synthesize answers into final response
    """

    # Step 1: Question decomposition
    sub_questions = decompose_question(question)

    # Step 2: Iterative answering
    context = []
    for hop, sub_q in enumerate(sub_questions):
        # Retrieve relevant concepts using accumulated context
        concepts = await retrieve_philosophical_concepts(
            query=sub_q + " " + " ".join(context),
            filters=QueryFilters(),
            top_k=5
        )

        # Extract answer from concepts
        answer = extract_answer(sub_q, concepts)
        context.append(answer)

        if hop >= max_hops:
            break

    # Step 3: Synthesize final answer
    final_answer = synthesize_answers(question, context)

    return MultiHopResult(
        question=question,
        reasoning_chain=context,
        final_answer=final_answer
    )
```

---

## Cross-Tradition Synthesis Algorithm

### Synthesis Pipeline

```python
async def synthesize_across_traditions(
    topic: str,
    traditions: List[PhilosophicalTradition],
    depth: SynthesisDepth = SynthesisDepth.MODERATE
) -> CrossTraditionSynthesis:
    """
    Synthesize philosophical insights across multiple traditions.

    Algorithm:
    1. Gather tradition-specific perspectives
    2. Identify convergent themes
    3. Map divergent positions
    4. Find complementary aspects
    5. Generate synthesis statement
    6. Assess fidelity to each tradition
    """

    # Step 1: Gather perspectives
    perspectives = {}
    for tradition in traditions:
        concepts = await retrieve_philosophical_concepts(
            query=topic,
            filters=QueryFilters(traditions=[tradition]),
            top_k=10 if depth == SynthesisDepth.DEEP else 5
        )
        perspectives[tradition] = extract_tradition_perspective(concepts, topic)

    # Step 2: Identify convergences
    convergent = identify_convergent_themes(perspectives)

    # Step 3: Map divergences
    divergent = map_divergent_positions(perspectives)

    # Step 4: Find complements
    complementary = find_complementary_aspects(perspectives, divergent)

    # Step 5: Generate synthesis
    synthesis_statement = generate_synthesis(
        topic=topic,
        convergent=convergent,
        divergent=divergent,
        complementary=complementary,
        preserve_nuance=True
    )

    # Step 6: Assess fidelity
    fidelity_scores = {}
    for tradition in traditions:
        fidelity_scores[tradition.value] = assess_fidelity(
            synthesis_statement,
            perspectives[tradition],
            tradition
        )

    return CrossTraditionSynthesis(
        synthesis_id=generate_id(),
        topic=topic,
        traditions=traditions,
        convergent_insights=convergent,
        divergent_positions=divergent,
        complementary_aspects=complementary,
        synthesis_statement=synthesis_statement,
        fidelity_scores=fidelity_scores,
        coherence_score=calculate_coherence(synthesis_statement)
    )
```

### Convergence Detection

```python
def identify_convergent_themes(
    perspectives: Dict[PhilosophicalTradition, TraditionPerspective]
) -> List[str]:
    """
    Identify themes that appear across multiple traditions.

    Uses semantic similarity to detect conceptually similar ideas
    even when expressed in different terminology.
    """

    # Extract all claims from all traditions
    all_claims = []
    for tradition, perspective in perspectives.items():
        for claim in perspective.key_claims:
            all_claims.append({
                "tradition": tradition,
                "claim": claim,
                "embedding": embed_text(claim)
            })

    # Cluster similar claims
    clusters = cluster_by_similarity(
        items=all_claims,
        similarity_threshold=0.75,
        min_cluster_size=2
    )

    # Identify multi-tradition clusters
    convergent_themes = []
    for cluster in clusters:
        traditions_in_cluster = set(item["tradition"] for item in cluster)
        if len(traditions_in_cluster) >= 2:
            theme = summarize_cluster(cluster)
            convergent_themes.append(theme)

    return convergent_themes
```

### Fidelity Assessment

```python
def assess_fidelity(
    synthesis: str,
    original_perspective: TraditionPerspective,
    tradition: PhilosophicalTradition
) -> float:
    """
    Assess how faithfully the synthesis represents a tradition.

    Checks:
    1. Core tenets are not contradicted
    2. Key terminology is used correctly
    3. Nuances are preserved
    4. Tradition is not misrepresented

    Returns: 0.0 (unfaithful) to 1.0 (perfectly faithful)
    """

    # Check core tenet preservation
    tenet_score = check_tenet_preservation(
        synthesis,
        original_perspective.core_tenets,
        tradition
    )

    # Check terminology usage
    terminology_score = check_terminology_accuracy(
        synthesis,
        tradition
    )

    # Check nuance preservation
    nuance_score = check_nuance_preservation(
        synthesis,
        original_perspective.nuances
    )

    # Check for misrepresentation
    misrepresentation_penalty = detect_misrepresentation(
        synthesis,
        tradition
    )

    return (
        tenet_score * 0.4 +
        terminology_score * 0.2 +
        nuance_score * 0.3 -
        misrepresentation_penalty * 0.5
    )
```

---

## Knowledge Gap Detection Algorithm

### Gap Analysis

```python
async def detect_knowledge_gaps() -> List[KnowledgeGap]:
    """
    Proactively identify areas where philosophical knowledge is shallow or missing.

    Algorithm:
    1. Analyze tradition coverage
    2. Analyze domain coverage
    3. Check for orphan concepts
    4. Identify shallow areas
    5. Prioritize gaps
    """

    gaps = []

    # Step 1: Tradition coverage analysis
    tradition_stats = get_tradition_statistics()
    for tradition in PhilosophicalTradition:
        stats = tradition_stats.get(tradition.value, {})

        if stats.get("concept_count", 0) < MINIMUM_TRADITION_CONCEPTS:
            gaps.append(KnowledgeGap(
                gap_type="shallow_tradition",
                description=f"Tradition {tradition.value} has insufficient concepts",
                tradition=tradition,
                suggested_research_query=f"core concepts in {tradition.value}",
                priority=calculate_tradition_priority(tradition)
            ))

    # Step 2: Domain coverage analysis
    for tradition in PhilosophicalTradition:
        missing_domains = identify_missing_domains(tradition)
        for domain in missing_domains:
            gaps.append(KnowledgeGap(
                gap_type="missing_domain_coverage",
                description=f"{tradition.value} lacks {domain.value} coverage",
                tradition=tradition,
                domain=domain,
                suggested_research_query=f"{domain.value} in {tradition.value}",
                priority=5
            ))

    # Step 3: Orphan concept detection
    orphans = find_orphan_concepts()
    for concept_id in orphans:
        gaps.append(KnowledgeGap(
            gap_type="orphan_concept",
            description=f"Concept {concept_id} lacks connections",
            suggested_research_query=f"related concepts to {concept_id}",
            priority=3
        ))

    # Step 4: Shallow area detection
    shallow_concepts = find_shallow_concepts(maturity_threshold=0.3)
    for concept in shallow_concepts:
        gaps.append(KnowledgeGap(
            gap_type="shallow_concept",
            description=f"Concept {concept.name} needs deeper research",
            tradition=concept.tradition,
            domain=concept.domain,
            suggested_research_query=f"detailed analysis of {concept.name}",
            priority=4
        ))

    # Step 5: Prioritize
    gaps.sort(key=lambda g: g.priority, reverse=True)

    return gaps
```

### Maturity Scoring

```python
def calculate_concept_maturity(concept: PhilosophicalConcept) -> float:
    """
    Calculate maturity score for a philosophical concept.

    Factors:
    1. Definition completeness
    2. Related concepts linked
    3. Key figures attributed
    4. Primary texts cited
    5. Arguments analyzed
    6. Source quality
    7. Research depth
    """

    scores = []

    # Definition completeness (0-1)
    definition_score = min(1.0, len(concept.definition) / 200)
    if concept.extended_description:
        definition_score = min(1.0, definition_score + 0.3)
    scores.append(("definition", definition_score, 0.2))

    # Related concepts (0-1)
    related_score = min(1.0, len(concept.related_concepts) / 5)
    scores.append(("relations", related_score, 0.15))

    # Attribution (0-1)
    attribution_score = min(1.0, (len(concept.key_figures) + len(concept.primary_texts)) / 4)
    scores.append(("attribution", attribution_score, 0.15))

    # Arguments (0-1)
    argument_score = min(1.0, (len(concept.key_arguments) + len(concept.counter_arguments)) / 4)
    scores.append(("arguments", argument_score, 0.2))

    # Source quality (0-1)
    source_score = calculate_source_quality(concept.sources)
    scores.append(("sources", source_score, 0.15))

    # Research depth (0-1)
    research_score = min(1.0, concept.research_depth / 5)
    scores.append(("research", research_score, 0.15))

    # Weighted average
    total = sum(score * weight for _, score, weight in scores)

    return total
```

---

## Philosophical Reasoning Algorithms

### Argument Analysis

```python
async def analyze_argument(
    argument: PhilosophicalArgument
) -> ArgumentAnalysis:
    """
    Analyze a philosophical argument for validity, soundness, and implications.
    """

    # Check logical validity
    validity = check_logical_validity(
        premises=argument.premises,
        conclusion=argument.conclusion,
        logical_form=argument.logical_form
    )

    # Assess premise plausibility
    premise_assessments = []
    for premise in argument.premises:
        plausibility = assess_premise_plausibility(premise, argument.tradition)
        premise_assessments.append({
            "premise": premise,
            "plausibility": plausibility,
            "supporting_concepts": find_supporting_concepts(premise),
            "challenging_concepts": find_challenging_concepts(premise)
        })

    # Identify assumptions
    assumptions = extract_hidden_assumptions(
        premises=argument.premises,
        conclusion=argument.conclusion,
        tradition=argument.tradition
    )

    # Find related arguments
    related = await find_related_arguments(argument)

    # Generate overall assessment
    soundness = calculate_soundness(validity, premise_assessments)

    return ArgumentAnalysis(
        argument_id=argument.argument_id,
        validity=validity,
        premise_assessments=premise_assessments,
        hidden_assumptions=assumptions,
        soundness_assessment=soundness,
        related_arguments=related,
        scholarly_notes=argument.scholarly_consensus
    )
```

### Dialectical Processing

```python
async def dialectical_synthesis(
    thesis: str,
    antithesis: str,
    context: Optional[Dict[str, Any]] = None
) -> DialecticalResult:
    """
    Perform Hegelian-style dialectical synthesis.

    Algorithm:
    1. Analyze thesis and antithesis
    2. Identify the contradiction
    3. Find the kernel of truth in each
    4. Generate synthesis that preserves both truths
    5. Identify what is negated/transcended
    """

    # Step 1: Analyze positions
    thesis_analysis = await analyze_position(thesis, context)
    antithesis_analysis = await analyze_position(antithesis, context)

    # Step 2: Identify contradiction
    contradiction = identify_contradiction(thesis_analysis, antithesis_analysis)

    # Step 3: Extract kernels of truth
    thesis_truth = extract_truth_kernel(thesis_analysis)
    antithesis_truth = extract_truth_kernel(antithesis_analysis)

    # Step 4: Generate synthesis
    synthesis = generate_dialectical_synthesis(
        thesis_truth=thesis_truth,
        antithesis_truth=antithesis_truth,
        contradiction=contradiction
    )

    # Step 5: Identify aufhebung (what is preserved/negated/transcended)
    aufhebung = analyze_aufhebung(
        thesis=thesis,
        antithesis=antithesis,
        synthesis=synthesis
    )

    return DialecticalResult(
        thesis=thesis,
        antithesis=antithesis,
        synthesis=synthesis,
        contradiction_identified=contradiction,
        thesis_truth_preserved=thesis_truth,
        antithesis_truth_preserved=antithesis_truth,
        aufhebung_analysis=aufhebung
    )
```

---

## Wisdom Integration Algorithm

### Context-Appropriate Wisdom Selection

```python
async def select_wisdom_for_context(
    engagement_context: EngagementContext,
    preferred_traditions: Optional[List[PhilosophicalTradition]] = None
) -> WisdomSelection:
    """
    Select appropriate philosophical wisdom for a given engagement context.

    Considers:
    1. Recipient capacity (beginner/intermediate/advanced)
    2. Emotional state
    3. Current need
    4. Cultural/spiritual orientation
    5. Harm risk assessment
    """

    # Determine appropriate traditions
    if preferred_traditions:
        traditions = preferred_traditions
    else:
        traditions = infer_appropriate_traditions(engagement_context)

    # Select wisdom based on capacity
    if engagement_context.recipient_capacity == "beginner":
        wisdom_level = "foundational"
        max_abstraction = 0.3
    elif engagement_context.recipient_capacity == "intermediate":
        wisdom_level = "developing"
        max_abstraction = 0.6
    else:
        wisdom_level = "advanced"
        max_abstraction = 1.0

    # Gather candidate teachings
    candidates = []
    for tradition in traditions:
        teachings = get_tradition_wisdom_teachings(tradition)
        for teaching in teachings:
            if teaching.abstraction_level <= max_abstraction:
                candidates.append({
                    "teaching": teaching,
                    "tradition": tradition,
                    "relevance": calculate_contextual_relevance(
                        teaching, engagement_context
                    )
                })

    # Filter by emotional state appropriateness
    if engagement_context.emotional_state == "distressed":
        candidates = [c for c in candidates if c["teaching"].is_comforting]
    elif engagement_context.emotional_state == "resistant":
        candidates = [c for c in candidates if c["teaching"].is_gentle]

    # Select top teachings
    candidates.sort(key=lambda c: c["relevance"], reverse=True)
    selected = candidates[:5]

    # Generate practical guidance
    practical = generate_practical_guidance(
        teachings=[c["teaching"] for c in selected],
        context=engagement_context
    )

    return WisdomSelection(
        teachings=[c["teaching"].content for c in selected],
        traditions_used=[c["tradition"] for c in selected],
        practical_guidance=practical,
        wisdom_level=wisdom_level
    )
```

### Maturation Tracking

```python
async def update_maturity_after_engagement(
    concepts_engaged: List[str],
    synthesis_performed: bool,
    research_triggered: bool
) -> MaturityUpdate:
    """
    Update philosophical maturity metrics after an engagement.

    Maturity grows through:
    1. Engaging with concepts (especially deep engagement)
    2. Performing cross-tradition synthesis
    3. Conducting research
    4. Successful wisdom application
    """

    maturity_state = await get_maturity_state()

    # Concept engagement bonus
    for concept_id in concepts_engaged:
        concept = await get_concept(concept_id)
        tradition_key = concept.tradition.value

        # Increment tradition depth
        current_depth = maturity_state.traditions_depth.get(tradition_key, 0.0)
        increment = 0.001  # Small increment per engagement
        maturity_state.traditions_depth[tradition_key] = min(1.0, current_depth + increment)

    # Synthesis bonus
    if synthesis_performed:
        maturity_state.cross_tradition_syntheses += 1
        # Larger maturity boost for synthesis
        for tradition in maturity_state.traditions_depth:
            current = maturity_state.traditions_depth[tradition]
            maturity_state.traditions_depth[tradition] = min(1.0, current + 0.005)

    # Research bonus
    if research_triggered:
        maturity_state.research_sessions_completed += 1

    # Save updated state
    await save_maturity_state(maturity_state)

    return MaturityUpdate(
        new_overall_maturity=maturity_state.get_overall_maturity(),
        new_level=maturity_state.get_maturity_level(),
        traditions_updated=list(maturity_state.traditions_depth.keys())
    )
```

---

## Research Agent Algorithms

### Autonomous Research Flow

```python
async def execute_research_task(task: ResearchTask) -> ResearchResult:
    """
    Execute an autonomous research task.

    Algorithm:
    1. Query external sources
    2. Extract concepts, figures, texts
    3. Validate extracted information
    4. Embed and integrate into index
    5. Update knowledge graph
    """

    results = ResearchResult(task_id=task.task_id)

    # Step 1: Query sources
    raw_data = {}
    for source in task.sources:
        if source == ResearchSource.STANFORD_ENCYCLOPEDIA:
            raw_data["sep"] = await fetch_sep_article(task.query)
        elif source == ResearchSource.PHILPAPERS:
            raw_data["philpapers"] = await search_philpapers(task.query)
        elif source == ResearchSource.WEB_SEARCH:
            raw_data["web"] = await philosophical_web_search(task.query)

    # Step 2: Extract entities
    for source_name, data in raw_data.items():
        concepts = extract_concepts_from_source(data, source_name)
        figures = extract_figures_from_source(data, source_name)
        texts = extract_texts_from_source(data, source_name)

        results.concepts_discovered.extend(concepts)
        results.figures_discovered.extend(figures)
        results.texts_discovered.extend(texts)

    # Step 3: Validate
    validated_concepts = []
    for concept in results.concepts_discovered:
        if validate_concept(concept):
            validated_concepts.append(concept)
    results.concepts_discovered = validated_concepts

    # Step 4: Embed and integrate
    for concept in results.concepts_discovered:
        concept.embedding = await embed_text(concept.to_embedding_text())
        await add_concept_to_index(concept)

    # Step 5: Update graph
    await update_knowledge_graph(results)

    results.status = "completed"
    return results
```

### Source-Specific Extractors

```python
async def extract_from_sep(html_content: str) -> ExtractionResult:
    """
    Extract philosophical knowledge from Stanford Encyclopedia article.

    SEP articles follow a consistent structure:
    - Preamble with summary
    - Numbered sections with content
    - Bibliography
    - Related entries
    """

    # Parse HTML structure
    soup = parse_html(html_content)

    # Extract main concept
    title = soup.find("h1").text
    preamble = soup.find("div", {"id": "preamble"}).text

    main_concept = PhilosophicalConcept(
        concept_id=slugify(title),
        name=title,
        definition=extract_definition(preamble),
        extended_description=preamble
    )

    # Extract related concepts from links
    related = []
    for link in soup.find_all("a", href=re.compile(r"/entries/")):
        related.append(slugify(link.text))
    main_concept.related_concepts = related[:10]

    # Extract key figures
    figures = extract_mentioned_philosophers(soup)
    main_concept.key_figures = figures

    # Extract bibliography as texts
    bib_section = soup.find("div", {"id": "bibliography"})
    texts = extract_bibliography_entries(bib_section)

    # Determine tradition and domain from content
    main_concept.tradition = infer_tradition(preamble + str(soup))
    main_concept.domain = infer_domain(preamble + str(soup))

    return ExtractionResult(
        main_concept=main_concept,
        related_concepts=related,
        figures=figures,
        texts=texts
    )
```
