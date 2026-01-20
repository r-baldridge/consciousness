"""
T5 - Text-to-Text Transfer Transformer (2019)

A unified text-to-text framework that converts all NLP tasks into a
sequence-to-sequence format. T5 systematically studied transfer learning
approaches and established best practices for pre-training at scale.

Authors: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee,
         Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

Key Innovations:
- Unified text-to-text format for all NLP tasks
- Systematic study of pre-training objectives, architectures, and datasets
- C4 dataset (Colossal Clean Crawled Corpus)
- Span corruption pre-training objective
"""

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# =============================================================================
# T5 (2019)
# =============================================================================

T5 = MLMethod(
    method_id="t5_2019",
    name="T5 (Text-to-Text Transfer Transformer)",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Colin Raffel", "Noam Shazeer", "Adam Roberts", "Katherine Lee",
             "Sharan Narang", "Michael Matena", "Yanqi Zhou", "Wei Li", "Peter J. Liu"],
    paper_title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    paper_url="https://arxiv.org/abs/1910.10683",
    key_innovation=(
        "Proposed a unified text-to-text framework where every NLP task is cast as "
        "generating output text conditioned on input text. Conducted a systematic "
        "empirical study comparing pre-training objectives, architectures, data, "
        "and transfer approaches. Introduced the C4 dataset and span corruption objective."
    ),
    mathematical_formulation=r"""
    T5 Architecture:

    Encoder-Decoder Transformer (following original Transformer)

    Model Sizes:
        T5-Small:  60M params   (6 layers, 512 hidden, 8 heads)
        T5-Base:   220M params  (12 layers, 768 hidden, 12 heads)
        T5-Large:  770M params  (24 layers, 1024 hidden, 16 heads)
        T5-3B:     3B params    (24 layers, 1024 hidden, 32 heads)
        T5-11B:    11B params   (24 layers, 1024 hidden, 128 heads)

    Text-to-Text Framework:

    All tasks converted to text generation:

    Classification:
        Input:  "mnli premise: I love cats. hypothesis: I adore felines."
        Output: "entailment"

    Translation:
        Input:  "translate English to German: Hello, how are you?"
        Output: "Hallo, wie geht es dir?"

    Summarization:
        Input:  "summarize: [long article text...]"
        Output: "[summary]"

    Question Answering:
        Input:  "question: What is the capital? context: Paris is the capital of France."
        Output: "Paris"

    Pre-training Objective - Span Corruption:

    1. Randomly select spans of consecutive tokens
       - Span lengths sampled from geometric distribution (mean ~3)
       - 15% of tokens corrupted on average

    2. Replace spans with sentinel tokens <extra_id_0>, <extra_id_1>, ...

    3. Target: predict original spans with corresponding sentinels

    Example:
        Original: "The quick brown fox jumps over the lazy dog"
        Input:    "The <X> brown fox <Y> the lazy dog"
        Target:   "<X> quick <Y> jumps over <Z>"

    Training Details:
        - Optimizer: Adafactor (Adam variant for large models)
        - Learning rate: inverse square root decay
        - Maximum sequence length: 512
        - Pre-trained on C4 for 1M steps, then fine-tuned

    C4 Dataset (Colossal Clean Crawled Corpus):
        - 750GB of cleaned English web text
        - Filtered from Common Crawl April 2019
        - Deduplication, language filtering, profanity removal
    """,
    predecessors=[
        "transformer_2017",
        "bert_2018",
        "gpt_2_2019",
    ],
    successors=[
        "t5_1_1_2020",
        "mt5_2020",
        "flan_t5_2022",
        "ul2_2022",
    ],
    tags=[
        "transformer",
        "encoder-decoder",
        "text-to-text",
        "pre-training",
        "transfer-learning",
        "google",
        "nlp",
        "unified-framework",
        "seq2seq",
    ],
)
