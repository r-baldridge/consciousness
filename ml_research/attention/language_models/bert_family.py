"""
BERT Family - Bidirectional Encoder Representations from Transformers

A family of encoder-only transformer models that revolutionized NLP through
bidirectional pre-training. BERT and its variants became the foundation for
transfer learning in natural language understanding tasks.

Models:
- BERT (2018): Original bidirectional encoder with MLM and NSP
- RoBERTa (2019): Robustly optimized BERT pre-training
- ALBERT (2019): Parameter-efficient BERT with cross-layer sharing
- DistilBERT (2019): Distilled smaller BERT via knowledge distillation

Key Innovations:
- Masked Language Modeling (MLM) for bidirectional context
- Pre-training/fine-tuning paradigm for NLU tasks
- [CLS] token for sentence-level representations
"""

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# =============================================================================
# BERT (2018)
# =============================================================================

BERT = MLMethod(
    method_id="bert_2018",
    name="BERT (Bidirectional Encoder Representations from Transformers)",
    year=2018,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
    paper_title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    paper_url="https://arxiv.org/abs/1810.04805",
    key_innovation=(
        "Introduced bidirectional pre-training using Masked Language Modeling (MLM), "
        "where random tokens are masked and the model predicts them using both left "
        "and right context. Combined with Next Sentence Prediction (NSP) for "
        "sentence-level understanding. Achieved SOTA on 11 NLP tasks."
    ),
    mathematical_formulation=r"""
    BERT Architecture:

    Encoder-only transformer with bidirectional self-attention.

    BERT-Base: 12 layers, 768 hidden, 12 heads, 110M params
    BERT-Large: 24 layers, 1024 hidden, 16 heads, 340M params

    Pre-training Objectives:

    1. Masked Language Modeling (MLM):
       - Randomly mask 15% of tokens
       - Of masked tokens: 80% [MASK], 10% random, 10% unchanged
       - Predict original tokens using cross-entropy loss

       L_MLM = -sum_{i in masked} log P(x_i | x_{\masked})

    2. Next Sentence Prediction (NSP):
       - Binary classification: Is sentence B the actual next sentence after A?
       - 50% positive pairs, 50% random pairs

       L_NSP = -log P(IsNext | [CLS] representation)

    Total pre-training loss:
       L = L_MLM + L_NSP

    Input representation:
       E = E_token + E_segment + E_position

       Token: WordPiece embeddings (30,000 vocab)
       Segment: Sentence A or B indicator
       Position: Learned positional embeddings (max 512)

    Special tokens:
       [CLS] - Classification token (first position)
       [SEP] - Separator between sentences
       [MASK] - Masked token placeholder

    Fine-tuning:
       Add task-specific output layer on top of [CLS] or token embeddings
       Fine-tune entire model end-to-end on task
    """,
    predecessors=[
        "transformer_2017",
        "elmo_2018",
        "gpt_1_2018",
    ],
    successors=[
        "roberta_2019",
        "albert_2019",
        "distilbert_2019",
        "spanbert_2019",
        "electra_2020",
    ],
    tags=[
        "transformer",
        "encoder-only",
        "bidirectional",
        "masked-language-model",
        "pre-training",
        "transfer-learning",
        "google",
        "nlp",
    ],
)


# =============================================================================
# RoBERTa (2019)
# =============================================================================

ROBERTA = MLMethod(
    method_id="roberta_2019",
    name="RoBERTa (Robustly Optimized BERT Approach)",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Yinhan Liu", "Myle Ott", "Naman Goyal", "Jingfei Du", "Mandar Joshi",
             "Danqi Chen", "Omer Levy", "Mike Lewis", "Luke Zettlemoyer", "Veselin Stoyanov"],
    paper_title="RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    paper_url="https://arxiv.org/abs/1907.11692",
    key_innovation=(
        "Showed that BERT was significantly undertrained and proposed optimized "
        "pre-training recipe: (1) longer training with bigger batches, (2) removing NSP, "
        "(3) training on longer sequences, (4) dynamically changing masking pattern. "
        "Matched or exceeded all post-BERT models on GLUE and SQuAD."
    ),
    mathematical_formulation=r"""
    RoBERTa Architecture:

    Same architecture as BERT-Large (24 layers, 1024 hidden, 16 heads, 355M params)

    Key Training Modifications:

    1. Dynamic Masking:
       - Generate masking pattern anew for each sequence each epoch
       - vs BERT's static masking applied once during preprocessing

    2. Remove Next Sentence Prediction (NSP):
       - NSP found to hurt performance on some tasks
       - Use only MLM objective

    3. Larger Batches:
       - 8,000 sequences per batch (vs BERT's 256)
       - Improves perplexity and downstream task performance

    4. More Data:
       - 160GB of text (vs BERT's ~16GB)
       - CC-News, OpenWebText, Stories datasets added

    5. Byte-Pair Encoding:
       - 50,000 subword units (vs BERT's 30,000 WordPiece)
       - Trained on combined datasets

    6. Longer Training:
       - 500K update steps with batch size 8K
       - vs BERT's 1M steps with batch size 256

    Training Configuration:
       - Peak learning rate: 4e-4
       - Adam: beta1=0.9, beta2=0.98
       - Weight decay: 0.01
       - Warmup: 30K steps
       - Max sequence length: 512
    """,
    predecessors=[
        "bert_2018",
    ],
    successors=[
        "xlm_roberta_2019",
        "longformer_2020",
    ],
    tags=[
        "transformer",
        "encoder-only",
        "bidirectional",
        "masked-language-model",
        "pre-training",
        "facebook",
        "nlp",
        "optimization",
    ],
)


# =============================================================================
# ALBERT (2019)
# =============================================================================

ALBERT = MLMethod(
    method_id="albert_2019",
    name="ALBERT (A Lite BERT)",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Zhenzhong Lan", "Mingda Chen", "Sebastian Goodman", "Kevin Gimpel",
             "Piyush Sharma", "Radu Soricut"],
    paper_title="ALBERT: A Lite BERT for Self-supervised Learning of Language Representations",
    paper_url="https://arxiv.org/abs/1909.11942",
    key_innovation=(
        "Introduced two parameter reduction techniques: (1) factorized embedding "
        "parameterization to decouple embedding size from hidden size, and "
        "(2) cross-layer parameter sharing. Also replaced NSP with Sentence Order "
        "Prediction (SOP). Achieved BERT-like performance with 18x fewer parameters."
    ),
    mathematical_formulation=r"""
    ALBERT Architecture:

    Parameter Reduction Techniques:

    1. Factorized Embedding Parameterization:
       - Decouple embedding dimension E from hidden dimension H
       - Original BERT: V x H parameters (30K x 768 = 23M)
       - ALBERT: V x E + E x H parameters (30K x 128 + 128 x 768 = 3.9M)

       Embedding lookup:
           E_word -> Project to H dimensions

       For E << H, significant parameter savings:
           O(V x H) -> O(V x E + E x H)

    2. Cross-Layer Parameter Sharing:
       - Share all parameters across transformer layers
       - Options: share attention only, FFN only, or all
       - ALBERT-xxlarge: 12M unique params vs 223M repeated

       Layer l computation:
           h_l = Transformer(h_{l-1}; theta)  # Same theta for all l

    3. Sentence Order Prediction (SOP):
       - Replace NSP with harder task
       - Predict if sentence order is correct or swapped
       - Forces model to learn discourse coherence

       L_SOP = -log P(IsCorrectOrder | [CLS])

    ALBERT Configurations:
       Base:    E=128, H=768,  L=12, A=12,  ~12M params
       Large:   E=128, H=1024, L=24, A=16,  ~18M params
       XLarge:  E=128, H=2048, L=24, A=32,  ~60M params
       XXLarge: E=128, H=4096, L=12, A=64,  ~235M params

    Note: XXLarge uses 12 layers but wider hidden size, still competitive
    with deeper models due to parameter sharing efficiency.
    """,
    predecessors=[
        "bert_2018",
    ],
    successors=[],
    tags=[
        "transformer",
        "encoder-only",
        "bidirectional",
        "parameter-efficient",
        "pre-training",
        "google",
        "nlp",
        "efficiency",
    ],
)


# =============================================================================
# DistilBERT (2019)
# =============================================================================

DISTILBERT = MLMethod(
    method_id="distilbert_2019",
    name="DistilBERT",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Victor Sanh", "Lysandre Debut", "Julien Chaumond", "Thomas Wolf"],
    paper_title="DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    paper_url="https://arxiv.org/abs/1910.01108",
    key_innovation=(
        "Applied knowledge distillation to compress BERT into a smaller, faster model. "
        "DistilBERT retains 97% of BERT's performance while being 40% smaller and "
        "60% faster. Uses a combination of language modeling, distillation, and "
        "cosine embedding loss during pre-training distillation."
    ),
    mathematical_formulation=r"""
    DistilBERT Architecture:

    6-layer transformer (vs BERT-base's 12 layers)
    - 768 hidden dimensions (same as BERT-base)
    - 12 attention heads (same as BERT-base)
    - 66M parameters (vs BERT-base's 110M, 40% reduction)

    Knowledge Distillation Training:

    Triple loss function:

    1. Distillation Loss (soft targets from teacher):
       L_ce = -sum_i t_i * log(s_i)

       where:
       t_i = softmax(z_i^{teacher} / T)  # Teacher soft labels
       s_i = softmax(z_i^{student} / T)  # Student predictions
       T = temperature (typically 2-4)

    2. Masked Language Model Loss (hard targets):
       L_mlm = -sum_{i in masked} log P(x_i | x_{\masked}; student)

    3. Cosine Embedding Loss (hidden state alignment):
       L_cos = 1 - cos(h_s, h_t)

       Aligns student hidden states with teacher's

    Total Loss:
       L = alpha * L_ce + beta * L_mlm + gamma * L_cos

    Distillation Process:
       - Initialize student with every other layer from teacher
       - Remove token type embeddings and pooler
       - Train on same data as BERT
       - Temperature T=8 for soft targets

    Performance (relative to BERT-base):
       - Parameters: 66M (60% of BERT)
       - Speed: 60% faster
       - GLUE: 97% of BERT performance
    """,
    predecessors=[
        "bert_2018",
    ],
    successors=[
        "tinybert_2020",
        "mobilebert_2020",
    ],
    tags=[
        "transformer",
        "encoder-only",
        "knowledge-distillation",
        "compression",
        "efficiency",
        "huggingface",
        "nlp",
    ],
)
