"""
Attention Era Core Modules

This module contains implementations and documentation of key attention mechanisms
that revolutionized sequence modeling and led to the Transformer architecture.

Timeline:
    2014 - Bahdanau Attention (additive attention for NMT)
    2015 - Luong Attention variants (dot-product, general, concat)
    2017 - Transformer (self-attention, multi-head attention)
    2020 - Efficient Attention (Longformer, BigBird, Linformer, Performer)
    2021 - CLIP, RoPE, ALiBi (multimodal + advanced positional encodings)
    2022 - Flamingo, Flash Attention (multimodal VLMs + IO-aware attention)
    2023 - LLaVA, GPT-4V, Mamba (visual instruction tuning + selective SSM)

Modules:
    attention_mechanism: Bahdanau (additive) attention
    self_attention: Self-attention and multi-head attention
    transformer: Full Transformer architecture
    positional_encoding: Positional encoding schemes (sinusoidal, RoPE, ALiBi)
    multimodal: Vision-language models (CLIP, Flamingo, LLaVA, GPT-4V)
    efficient_attention: Efficient attention (sparse, linear, Flash, Mamba)
"""

from consciousness.ml_research.attention.attention_mechanism import (
    BAHDANAU_ATTENTION,
    bahdanau_attention,
    compute_attention_weights,
    compute_context_vector,
)

from consciousness.ml_research.attention.self_attention import (
    SELF_ATTENTION,
    MULTI_HEAD_ATTENTION,
    scaled_dot_product_attention,
    multi_head_attention,
)

from consciousness.ml_research.attention.transformer import (
    TRANSFORMER,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    position_wise_ffn,
    layer_norm,
)

from consciousness.ml_research.attention.positional_encoding import (
    SINUSOIDAL_POSITIONAL_ENCODING,
    ROTARY_POSITION_EMBEDDING,
    ALIBI,
    sinusoidal_encoding,
    rotary_position_embedding,
    alibi_bias,
)

# Multimodal attention methods
from consciousness.ml_research.attention.multimodal import (
    get_clip_info,
    get_flamingo_info,
    get_llava_info,
    get_gpt4v_info,
)

# Efficient attention methods
from consciousness.ml_research.attention.efficient_attention import (
    get_longformer_info,
    get_bigbird_info,
    get_linformer_info,
    get_performer_info,
    get_flash_attention_info,
    get_mamba_info,
)

__all__ = [
    # Attention Mechanism (Bahdanau)
    "BAHDANAU_ATTENTION",
    "bahdanau_attention",
    "compute_attention_weights",
    "compute_context_vector",
    # Self-Attention
    "SELF_ATTENTION",
    "MULTI_HEAD_ATTENTION",
    "scaled_dot_product_attention",
    "multi_head_attention",
    # Transformer
    "TRANSFORMER",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "position_wise_ffn",
    "layer_norm",
    # Positional Encodings
    "SINUSOIDAL_POSITIONAL_ENCODING",
    "ROTARY_POSITION_EMBEDDING",
    "ALIBI",
    "sinusoidal_encoding",
    "rotary_position_embedding",
    "alibi_bias",
    # Multimodal Methods
    "get_clip_info",
    "get_flamingo_info",
    "get_llava_info",
    "get_gpt4v_info",
    # Efficient Attention Methods
    "get_longformer_info",
    "get_bigbird_info",
    "get_linformer_info",
    "get_performer_info",
    "get_flash_attention_info",
    "get_mamba_info",
]
