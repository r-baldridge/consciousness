"""
Language Models Module

Research index entries for major language model families, documenting
the evolution from GPT-1 to modern foundation models.

Families included:
- GPT Family (OpenAI): GPT-1, GPT-2, GPT-3, GPT-4
- BERT Family (Google): BERT, RoBERTa, ALBERT, DistilBERT
- T5 (Google): Text-to-Text Transfer Transformer
- LLaMA Family (Meta): LLaMA, LLaMA 2, LLaMA 3
- Mistral Family: Mistral 7B, Mixtral 8x7B
- Claude Family (Anthropic): Claude 1, 2, 3 series
"""

# GPT Family
from consciousness.ml_research.attention.language_models.gpt_family import (
    GPT_1,
    GPT_2,
    GPT_3,
    GPT_4,
)

# BERT Family
from consciousness.ml_research.attention.language_models.bert_family import (
    BERT,
    ROBERTA,
    ALBERT,
    DISTILBERT,
)

# T5
from consciousness.ml_research.attention.language_models.t5 import T5

# LLaMA Family
from consciousness.ml_research.attention.language_models.llama_family import (
    LLAMA,
    LLAMA_2,
    LLAMA_3,
)

# Mistral Family
from consciousness.ml_research.attention.language_models.mistral import (
    MISTRAL_7B,
    MIXTRAL_8X7B,
)

# Claude Family
from consciousness.ml_research.attention.language_models.claude import (
    CLAUDE_1,
    CLAUDE_2,
    CLAUDE_3_HAIKU,
    CLAUDE_3_SONNET,
    CLAUDE_3_OPUS,
)


__all__ = [
    # GPT Family
    "GPT_1",
    "GPT_2",
    "GPT_3",
    "GPT_4",
    # BERT Family
    "BERT",
    "ROBERTA",
    "ALBERT",
    "DISTILBERT",
    # T5
    "T5",
    # LLaMA Family
    "LLAMA",
    "LLAMA_2",
    "LLAMA_3",
    # Mistral Family
    "MISTRAL_7B",
    "MIXTRAL_8X7B",
    # Claude Family
    "CLAUDE_1",
    "CLAUDE_2",
    "CLAUDE_3_HAIKU",
    "CLAUDE_3_SONNET",
    "CLAUDE_3_OPUS",
]
