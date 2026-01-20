"""
LLaMA Family (Meta) - Large Language Model Meta AI

A family of open-weight foundation models from Meta that democratized
access to large language model research. LLaMA models are known for
efficient training and strong performance relative to model size.

Models:
- LLaMA (2023): Open foundation model, 7B-65B parameters
- LLaMA 2 (2023): RLHF fine-tuning, commercial license
- LLaMA 3 (2024): Extended to 405B, improved capabilities

Key Innovations:
- Efficient training on public data only
- Open weights enabling research community access
- Strong performance at smaller scales
"""

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# =============================================================================
# LLaMA (2023)
# =============================================================================

LLAMA = MLMethod(
    method_id="llama_2023",
    name="LLaMA (Large Language Model Meta AI)",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet",
             "Marie-Anne Lachaux", "Timothee Lacroix", "Baptiste Roziere",
             "Naman Goyal", "Eric Hambro", "Faisal Azhar", "et al."],
    paper_title="LLaMA: Open and Efficient Foundation Language Models",
    paper_url="https://arxiv.org/abs/2302.13971",
    key_innovation=(
        "Demonstrated that state-of-the-art performance can be achieved training "
        "exclusively on publicly available data, without resorting to proprietary "
        "datasets. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, showing "
        "the importance of training data quality and training tokens over raw model size."
    ),
    mathematical_formulation=r"""
    LLaMA Architecture:

    Decoder-only transformer with several modifications:

    1. Pre-normalization (GPT-3 style):
       - RMSNorm instead of LayerNorm
       - Applied before each sub-layer (pre-norm)

       RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    2. SwiGLU Activation (instead of ReLU):
       FFN(x) = SwiGLU(x) = (Swish(xW_1) * xV) W_2

       Swish(x) = x * sigmoid(beta * x)

       Note: Uses 3 matrices instead of 2, adjusted hidden dim accordingly
             Hidden dimension = 2/3 * 4d = 8/3 * d

    3. Rotary Positional Embeddings (RoPE):
       - Relative positional encoding through rotation
       - Applied to query and key at each attention layer

       RoPE(x, pos) = x * cos(pos * theta) + rotate(x) * sin(pos * theta)

       where theta_i = 10000^{-2i/d}

    Model Configurations:
        LLaMA-7B:  32 layers, 4096 hidden, 32 heads
        LLaMA-13B: 40 layers, 5120 hidden, 40 heads
        LLaMA-33B: 60 layers, 6656 hidden, 52 heads
        LLaMA-65B: 80 layers, 8192 hidden, 64 heads

    Training Data (1.4T tokens):
        - CommonCrawl: 67% (filtered, deduped)
        - C4: 15%
        - GitHub: 4.5%
        - Wikipedia: 4.5%
        - Books: 4.5%
        - ArXiv: 2.5%
        - StackExchange: 2%

    Training Details:
        - AdamW optimizer: beta1=0.9, beta2=0.95
        - Cosine learning rate schedule
        - Weight decay: 0.1
        - Gradient clipping: 1.0
        - Context length: 2048 tokens
        - Trained on ~1T tokens for smaller models, 1.4T for 65B
    """,
    predecessors=[
        "gpt_3_2020",
        "chinchilla_2022",
    ],
    successors=[
        "llama_2_2023",
        "alpaca_2023",
        "vicuna_2023",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "foundation-model",
        "open-weights",
        "efficient",
        "meta",
        "language-model",
    ],
)


# =============================================================================
# LLaMA 2 (2023)
# =============================================================================

LLAMA_2 = MLMethod(
    method_id="llama_2_2023",
    name="LLaMA 2",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Hugo Touvron", "Louis Martin", "Kevin Stone", "Peter Albert",
             "Amjad Almahairi", "Yasmine Babaei", "Nikolay Bashlykov",
             "Soumya Batra", "Prajjwal Bhargava", "Shruti Bhosale", "et al."],
    paper_title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
    paper_url="https://arxiv.org/abs/2307.09288",
    key_innovation=(
        "Extended LLaMA with 40% more training data (2T tokens), doubled context "
        "length (4096), and introduced LLaMA 2-Chat fine-tuned with RLHF. First "
        "open-weight model with commercial license, enabling widespread deployment. "
        "Introduced Ghost Attention for multi-turn dialogue consistency."
    ),
    mathematical_formulation=r"""
    LLaMA 2 Architecture:

    Same base architecture as LLaMA with improvements:

    1. Extended Context Length:
       - 4096 tokens (vs LLaMA's 2048)
       - RoPE extended via frequency interpolation

    2. Grouped-Query Attention (GQA) for 34B and 70B:
       - Key-Value heads shared across query heads
       - 70B: 64 query heads, 8 KV heads
       - Reduces KV cache size, faster inference

       GQA groups G query heads per KV head:
           Attention(Q, K, V) where K, V are shared within groups

    Model Configurations:
        LLaMA 2-7B:  32 layers, 4096 hidden, 32 heads
        LLaMA 2-13B: 40 layers, 5120 hidden, 40 heads
        LLaMA 2-70B: 80 layers, 8192 hidden, 64 query / 8 KV heads

    LLaMA 2-Chat (RLHF Training):

    1. Supervised Fine-Tuning (SFT):
       - High-quality dialogue data
       - Focus on helpfulness and safety

    2. Reward Model Training:
       - Human preference data collection
       - Separate models for helpfulness and safety

       R(x, y) = transformer(x, y) -> scalar reward

    3. RLHF via PPO:
       - Optimize policy against reward model
       - KL divergence penalty to SFT model

       L = E[R(x, y) - beta * KL(pi || pi_SFT)]

    4. Ghost Attention (GAtt):
       - Maintain instruction following across turns
       - Synthetically add system prompt to training examples
       - Helps model remember persona/instructions

    Safety Fine-Tuning:
       - Red teaming and adversarial testing
       - Safety-specific reward model
       - Rejection sampling for safe responses
    """,
    predecessors=[
        "llama_2023",
        "instructgpt_2022",
    ],
    successors=[
        "llama_3_2024",
        "code_llama_2023",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "foundation-model",
        "open-weights",
        "rlhf",
        "chat-model",
        "commercial-license",
        "meta",
        "language-model",
    ],
)


# =============================================================================
# LLaMA 3 (2024)
# =============================================================================

LLAMA_3 = MLMethod(
    method_id="llama_3_2024",
    name="LLaMA 3",
    year=2024,
    era=MethodEra.NOVEL,
    category=MethodCategory.MULTIMODAL,
    lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RL_LINE],
    authors=["Meta AI"],
    paper_title="The Llama 3 Herd of Models",
    paper_url="https://arxiv.org/abs/2407.21783",
    key_innovation=(
        "Scaled to 405B parameters with 15T+ training tokens, achieving performance "
        "competitive with GPT-4 and Claude 3. Introduced multimodal capabilities, "
        "128K context length, and extensive safety training. Represents the "
        "culmination of Meta's open foundation model efforts."
    ),
    mathematical_formulation=r"""
    LLaMA 3 Architecture:

    Model Configurations:
        LLaMA 3-8B:   32 layers, 4096 hidden, 32 heads, 8K context
        LLaMA 3-70B:  80 layers, 8192 hidden, 64 query / 8 KV heads, 8K context
        LLaMA 3-405B: 126 layers, 16384 hidden, 128 query / 8 KV heads, 128K context

    Key Architectural Changes:

    1. Extended Vocabulary:
       - 128K tokens (vs LLaMA 2's 32K)
       - BPE tokenizer with improved encoding efficiency
       - Better multilingual support

    2. Extended Context (up to 128K):
       - RoPE with position interpolation
       - Efficient attention via GQA

    3. Grouped-Query Attention (GQA) across all sizes:
       - 8B: 32 query heads, 8 KV heads
       - 70B/405B: 64/128 query heads, 8 KV heads

    Training:
       - 15T+ tokens for 405B model
       - Data: web text, code, math, multilingual
       - Quality filtering and data mixing

    405B Training Scale:
       - 16,000+ H100 GPUs
       - Custom training infrastructure
       - FP8 training for efficiency

    Post-Training:

    1. Supervised Fine-Tuning:
       - High-quality instruction data
       - Multi-turn conversation data

    2. Direct Preference Optimization (DPO):
       - More stable than PPO-based RLHF
       - Implicit reward model

       L_DPO = -E[log sigmoid(beta * (r(y_w) - r(y_l)))]

    3. Safety Training:
       - Extensive red teaming
       - Llama Guard for input/output filtering
       - System prompt adherence

    Multimodal (LLaMA 3.2):
       - Vision encoder integration
       - Image understanding capabilities
       - 11B and 90B vision-language models
    """,
    predecessors=[
        "llama_2_2023",
    ],
    successors=[],
    tags=[
        "transformer",
        "decoder-only",
        "foundation-model",
        "open-weights",
        "multimodal",
        "dpo",
        "meta",
        "language-model",
        "large-scale",
    ],
)
