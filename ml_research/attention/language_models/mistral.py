"""
Mistral Family - Efficient Open Language Models

A family of efficient language models from Mistral AI (French startup)
that achieved strong performance through architectural innovations like
sliding window attention and mixture of experts.

Models:
- Mistral 7B (2023): Sliding window attention, outperforms LLaMA 2 13B
- Mixtral 8x7B (2023): Sparse mixture of experts, 12B active parameters

Key Innovations:
- Sliding window attention for efficient long contexts
- Mixture of experts for parameter-efficient scaling
- Strong performance at smaller active parameter counts
"""

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# =============================================================================
# Mistral 7B (2023)
# =============================================================================

MISTRAL_7B = MLMethod(
    method_id="mistral_7b_2023",
    name="Mistral 7B",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Albert Q. Jiang", "Alexandre Sablayrolles", "Arthur Mensch",
             "Chris Bamford", "Devendra Singh Chaplot", "Diego de las Casas",
             "Florian Bressand", "Gianna Lengyel", "Guillaume Lample",
             "Lucile Saulnier", "et al."],
    paper_title="Mistral 7B",
    paper_url="https://arxiv.org/abs/2310.06825",
    key_innovation=(
        "Introduced sliding window attention (SWA) for efficient processing of long "
        "sequences, combined with grouped-query attention and rolling buffer KV cache. "
        "Outperforms LLaMA 2 13B on all benchmarks and approaches CodeLLaMA 7B on "
        "code tasks, despite having fewer parameters."
    ),
    mathematical_formulation=r"""
    Mistral 7B Architecture:

    32 layers, 4096 hidden dimensions, 32 query heads, 8 KV heads
    7.3B parameters total

    Key Innovations:

    1. Sliding Window Attention (SWA):
       - Each token attends to W previous tokens only
       - Window size W = 4096
       - Reduces attention complexity from O(n^2) to O(n * W)

       For position i:
           Attention(i) = softmax(Q_i * K[max(0,i-W):i]^T / sqrt(d))

       Information can still propagate beyond window via stacking:
       - Layer 1: token sees window W
       - Layer k: token can access up to k * W tokens
       - At layer 32: effective context = 32 * 4096 = 131K tokens

    2. Rolling Buffer KV Cache:
       - Only store last W key-value pairs per layer
       - Position i stored at cache index (i mod W)
       - Constant memory regardless of sequence length

       Memory: O(W * n_layers * d) instead of O(seq_len * n_layers * d)

    3. Pre-fill and Chunking:
       - Split prompt into chunks of size W
       - Pre-compute KV cache for each chunk
       - Enables parallel prompt processing

    4. Grouped-Query Attention (GQA):
       - 32 query heads, 8 KV heads (4 query heads per KV group)
       - Same as LLaMA 2 approach

    Architecture Details:
        - Vocabulary: 32,000 tokens (SentencePiece BPE)
        - RoPE positional embeddings
        - SwiGLU activation in FFN
        - RMSNorm pre-normalization

    Comparison to LLaMA 2:
        Mistral 7B vs LLaMA 2 13B:
        - MMLU: 60.1 vs 54.8
        - HellaSwag: 81.3 vs 80.7
        - ARC: 61.1 vs 57.4
        - TriviaQA: 62.5 vs 63.5
    """,
    predecessors=[
        "llama_2023",
        "longformer_2020",
    ],
    successors=[
        "mixtral_8x7b_2023",
        "mistral_large_2024",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "sliding-window-attention",
        "efficient",
        "open-weights",
        "mistral-ai",
        "language-model",
    ],
)


# =============================================================================
# Mixtral 8x7B (2023)
# =============================================================================

MIXTRAL_8X7B = MLMethod(
    method_id="mixtral_8x7b_2023",
    name="Mixtral 8x7B",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Albert Q. Jiang", "Alexandre Sablayrolles", "Antoine Roux",
             "Arthur Mensch", "Blanche Savary", "Chris Bamford",
             "Devendra Singh Chaplot", "Diego de las Casas", "Emma Bou Hanna",
             "Florian Bressand", "et al."],
    paper_title="Mixtral of Experts",
    paper_url="https://arxiv.org/abs/2401.04088",
    key_innovation=(
        "Sparse Mixture of Experts (MoE) language model with 8 expert FFNs per layer, "
        "where each token is routed to 2 experts. Total 46.7B parameters but only "
        "12.9B active per forward pass. Matches or exceeds LLaMA 2 70B and GPT-3.5 "
        "while using 5x less compute at inference."
    ),
    mathematical_formulation=r"""
    Mixtral 8x7B Architecture:

    32 layers, 4096 hidden dimensions
    - 46.7B total parameters
    - 12.9B active parameters per forward pass
    - 8 experts per MoE layer, top-2 routing

    Mixture of Experts Layer:

    Standard FFN replaced with MoE:

        y = sum_{i=1}^{n} G(x)_i * E_i(x)

    where:
        E_i = Expert FFN (SwiGLU network)
        G(x) = Router/Gating function

    Top-K Sparse Gating:

        G(x) = softmax(TopK(x * W_g))

        TopK keeps only top K values, sets others to -inf
        K = 2 for Mixtral (each token uses 2 experts)

    Router weights:
        W_g: (d_model, n_experts) = (4096, 8)

    Expert FFN (same as Mistral 7B):
        E_i(x) = SwiGLU(x) = (Swish(xW_1^i) * xW_V^i) W_2^i

        FFN dimensions: 4096 -> 14336 -> 4096

    Load Balancing (during training):
        Auxiliary loss to encourage uniform expert usage:

        L_aux = alpha * n_experts * sum_i(f_i * P_i)

        where:
            f_i = fraction of tokens routed to expert i
            P_i = average routing probability to expert i

    Architecture Details:
        - Sliding Window Attention (W = 4096)
        - 32 attention heads, 8 KV heads (GQA)
        - RoPE positional embeddings
        - 32,000 token vocabulary

    Compute Efficiency:
        - Active params: 12.9B (2 experts * 6.5B params/expert)
        - Inference FLOPS: ~same as 12B dense model
        - Memory: 46.7B params must be loaded

    Performance vs Dense Models:
        Mixtral 8x7B vs LLaMA 2 70B:
        - MMLU: 70.6 vs 69.8
        - HellaSwag: 84.4 vs 85.3
        - ARC: 66.4 vs 67.3
        - 6x faster inference
    """,
    predecessors=[
        "mistral_7b_2023",
        "gshard_2020",
        "switch_transformer_2022",
    ],
    successors=[
        "mixtral_8x22b_2024",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "mixture-of-experts",
        "sparse",
        "efficient",
        "open-weights",
        "mistral-ai",
        "language-model",
    ],
)
