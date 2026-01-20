"""
Claude Family (Anthropic) - Constitutional AI Language Models

A family of large language models developed by Anthropic, focused on
being helpful, harmless, and honest (HHH). Claude models are trained
using Constitutional AI (CAI) and RLHF with an emphasis on safety.

Models:
- Claude 1 (2023): Initial release with CAI training
- Claude 2 (2023): Improved capabilities and longer context
- Claude 3 (2024): Haiku, Sonnet, Opus - tiered capability levels

Key Innovations:
- Constitutional AI (self-improvement via principles)
- Focus on safety and avoiding harmful outputs
- Instruction following and helpfulness
"""

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# =============================================================================
# Claude 1 (2023)
# =============================================================================

CLAUDE_1 = MLMethod(
    method_id="claude_1_2023",
    name="Claude 1",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Anthropic"],
    paper_title="Constitutional AI: Harmlessness from AI Feedback",
    paper_url="https://arxiv.org/abs/2212.08073",
    key_innovation=(
        "Introduced Constitutional AI (CAI), where the model critiques and revises "
        "its own responses according to a set of principles, followed by RLHF using "
        "AI-generated preference labels. This reduces reliance on human labeling for "
        "safety while producing less harmful, more helpful outputs."
    ),
    mathematical_formulation=r"""
    Constitutional AI Training:

    Phase 1: Supervised Learning from Self-Critique

    1. Generate responses to harmful prompts
    2. Ask model to critique its response using principles:
       "Identify specific ways this response is harmful, unethical, or illegal..."
    3. Ask model to revise based on critique:
       "Please rewrite to remove harmful content while remaining helpful..."
    4. Fine-tune on (prompt, revised_response) pairs

    Constitutional Principles (examples):
       - "Please choose the response that is most helpful."
       - "Choose the response that is least racist/sexist."
       - "Choose the response that is most honest."
       - "Choose the response that would be least likely to cause harm."

    Phase 2: RLHF with AI Feedback (RLAIF)

    1. Generate pairs of responses
    2. Ask model to choose preferred response using principles:
       "Which response is more helpful, harmless, and honest?"
    3. Train reward model on AI preference labels
    4. Fine-tune policy with PPO against reward model

    AI Preference Labeling:
       P(response_A > response_B | principle) = sigmoid(R_A - R_B)

       where R is reward model trained on AI preferences

    HHH Objectives:
       - Helpful: Assist user with their task effectively
       - Harmless: Avoid generating harmful content
       - Honest: Provide accurate, truthful information

    Key Insight:
       Self-critique allows model to:
       - Identify problems it wouldn't catch during initial generation
       - Apply ethical reasoning explicitly
       - Generate training data for safety without human labeling
    """,
    predecessors=[
        "gpt_3_2020",
        "instructgpt_2022",
    ],
    successors=[
        "claude_2_2023",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "constitutional-ai",
        "rlhf",
        "safety",
        "anthropic",
        "language-model",
        "alignment",
    ],
)


# =============================================================================
# Claude 2 (2023)
# =============================================================================

CLAUDE_2 = MLMethod(
    method_id="claude_2_2023",
    name="Claude 2",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Anthropic"],
    paper_title="Claude 2 Model Card and Evaluations",
    paper_url="https://www.anthropic.com/news/claude-2",
    key_innovation=(
        "Significant improvements in coding, math, and reasoning capabilities over "
        "Claude 1. Extended context window to 100K tokens, enabling analysis of "
        "entire books or codebases. Improved safety while maintaining helpfulness."
    ),
    mathematical_formulation=r"""
    Claude 2 Improvements:

    Extended Context (100K tokens):
       - Approximately 75,000 words
       - Can process entire books, legal documents, codebases
       - Improved long-range attention via architectural optimizations

    Capability Benchmarks (vs Claude 1):
       - GRE Writing: 75th percentile
       - Bar Exam multiple choice: 76.5% (vs 73.0%)
       - Python coding: 71.2% pass@1 (vs 56.0%)
       - GSM8K (math): Improved reasoning accuracy

    Safety Improvements:
       - 2x less likely to produce harmful outputs
       - Improved refusal of harmful requests
       - Better handling of ambiguous safety cases

    Training Approach:
       - Constitutional AI (refined principles)
       - RLHF with expanded human feedback
       - Improved instruction following
       - Red teaming for safety validation

    Claude 2.1 Updates:
       - Reduced hallucination rate
       - Improved tool use / function calling
       - System prompts for customization
       - 200K context window option
    """,
    predecessors=[
        "claude_1_2023",
    ],
    successors=[
        "claude_3_2024",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "constitutional-ai",
        "long-context",
        "rlhf",
        "safety",
        "anthropic",
        "language-model",
    ],
)


# =============================================================================
# Claude 3 Haiku (2024)
# =============================================================================

CLAUDE_3_HAIKU = MLMethod(
    method_id="claude_3_haiku_2024",
    name="Claude 3 Haiku",
    year=2024,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Anthropic"],
    paper_title="The Claude 3 Model Family",
    paper_url="https://www.anthropic.com/news/claude-3-family",
    key_innovation=(
        "Fastest model in the Claude 3 family, optimized for speed and cost "
        "efficiency while maintaining strong capabilities. Designed for "
        "high-throughput, latency-sensitive applications where instant responses "
        "are critical."
    ),
    mathematical_formulation=r"""
    Claude 3 Haiku:

    Optimized for:
       - Speed: Near-instant responses
       - Cost: Lowest price in Claude 3 family
       - Efficiency: High throughput applications

    Context Window: 200K tokens

    Use Cases:
       - Customer service chatbots
       - Content moderation at scale
       - Quick data extraction
       - Real-time applications

    Performance vs Efficiency Tradeoff:
       - Fastest inference in family
       - Lower capability than Sonnet/Opus
       - Best for simpler tasks at scale

    Benchmark Performance:
       - Competitive with Claude 2 on many tasks
       - Optimized for speed over maximum capability
    """,
    predecessors=[
        "claude_2_2023",
    ],
    successors=[],
    tags=[
        "transformer",
        "decoder-only",
        "efficient",
        "fast",
        "constitutional-ai",
        "anthropic",
        "language-model",
    ],
)


# =============================================================================
# Claude 3 Sonnet (2024)
# =============================================================================

CLAUDE_3_SONNET = MLMethod(
    method_id="claude_3_sonnet_2024",
    name="Claude 3 Sonnet",
    year=2024,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Anthropic"],
    paper_title="The Claude 3 Model Family",
    paper_url="https://www.anthropic.com/news/claude-3-family",
    key_innovation=(
        "Balanced model offering strong performance at moderate cost. Positioned "
        "between Haiku (speed) and Opus (capability) for enterprise workloads "
        "requiring both intelligence and throughput."
    ),
    mathematical_formulation=r"""
    Claude 3 Sonnet:

    Design Philosophy:
       - Balance of intelligence and speed
       - Enterprise-grade reliability
       - Cost-effective for complex tasks

    Context Window: 200K tokens

    Capabilities:
       - Strong reasoning and analysis
       - Code generation and review
       - Document understanding
       - Multimodal (vision + text)

    Use Cases:
       - Data processing and extraction
       - Sales and marketing content
       - Code generation
       - General enterprise AI

    Performance:
       - 2x faster than Opus
       - Approaches Opus quality on many tasks
       - Best balance for most applications
    """,
    predecessors=[
        "claude_2_2023",
    ],
    successors=[
        "claude_3_5_sonnet_2024",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "balanced",
        "multimodal",
        "constitutional-ai",
        "anthropic",
        "language-model",
    ],
)


# =============================================================================
# Claude 3 Opus (2024)
# =============================================================================

CLAUDE_3_OPUS = MLMethod(
    method_id="claude_3_opus_2024",
    name="Claude 3 Opus",
    year=2024,
    era=MethodEra.NOVEL,
    category=MethodCategory.MULTIMODAL,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Anthropic"],
    paper_title="The Claude 3 Model Family",
    paper_url="https://www.anthropic.com/news/claude-3-family",
    key_innovation=(
        "Most capable model in the Claude 3 family, achieving state-of-the-art "
        "performance on complex reasoning, analysis, and creative tasks. Near-human "
        "comprehension and fluency with strong performance on expert-level benchmarks."
    ),
    mathematical_formulation=r"""
    Claude 3 Opus:

    Flagship Model Capabilities:
       - State-of-the-art reasoning
       - Expert-level knowledge
       - Complex instruction following
       - Nuanced understanding

    Context Window: 200K tokens

    Benchmark Performance:
       - Graduate-level reasoning (GPQA): 50.4%
       - Undergraduate knowledge (MMLU): 86.8%
       - Math (GSM8K): 95.0%
       - Code (HumanEval): 84.9%

    Competitive positioning:
       - Exceeded GPT-4 on multiple benchmarks at release
       - Near or at frontier capability level

    Multimodal Capabilities:
       - Vision understanding
       - Chart and diagram analysis
       - Document processing
       - Scientific figure interpretation

    Safety Features:
       - Constitutional AI training
       - Extensive red teaming
       - Refusal of harmful requests
       - Improved calibration (knows what it doesn't know)

    Advanced Features:
       - Tool use / function calling
       - Extended thinking (chain-of-thought)
       - System prompts for customization
       - Structured output generation
    """,
    predecessors=[
        "claude_2_2023",
        "gpt_4_2023",
    ],
    successors=[
        "claude_opus_4_2025",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "multimodal",
        "frontier-model",
        "constitutional-ai",
        "rlhf",
        "safety",
        "anthropic",
        "language-model",
        "reasoning",
    ],
)
