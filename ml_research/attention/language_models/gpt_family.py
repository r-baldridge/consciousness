"""
GPT Family (OpenAI) - Generative Pre-trained Transformers

A series of decoder-only autoregressive language models that pioneered
large-scale unsupervised pre-training followed by task-specific fine-tuning,
and later demonstrated emergent capabilities through scale.

Models:
- GPT-1 (2018): First GPT, demonstrated transfer learning in NLP
- GPT-2 (2019): 1.5B params, zero-shot task generalization
- GPT-3 (2020): 175B params, few-shot in-context learning
- GPT-4 (2023): Multimodal, advanced reasoning capabilities

Key Innovations:
- Decoder-only transformer architecture for language modeling
- Unsupervised pre-training on large text corpora
- In-context learning without gradient updates
- Scaling laws for neural language models
"""

from consciousness.ml_research.core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# =============================================================================
# GPT-1 (2018)
# =============================================================================

GPT_1 = MLMethod(
    method_id="gpt_1_2018",
    name="GPT-1 (Generative Pre-trained Transformer)",
    year=2018,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Alec Radford", "Karthik Narasimhan", "Tim Salimans", "Ilya Sutskever"],
    paper_title="Improving Language Understanding by Generative Pre-Training",
    paper_url="https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
    key_innovation=(
        "Demonstrated that generative pre-training of a language model on diverse "
        "unlabeled text, followed by discriminative fine-tuning on specific tasks, "
        "achieves large gains on many NLP benchmarks. First successful application "
        "of transformer decoder for transfer learning in NLP."
    ),
    mathematical_formulation=r"""
    GPT-1 Architecture:

    12-layer decoder-only transformer with:
    - 768 hidden dimensions
    - 12 attention heads
    - 117M parameters total

    Pre-training Objective (Causal Language Modeling):
        L_1(U) = sum_i log P(u_i | u_{i-k}, ..., u_{i-1}; Theta)

    where U = {u_1, ..., u_n} is the corpus of tokens.

    Fine-tuning adds task-specific linear output layer:
        P(y|x_1, ..., x_m) = softmax(h_l^m * W_y)

    Combined objective during fine-tuning:
        L_3(C) = L_2(C) + lambda * L_1(C)

    where L_2 is task-specific loss and L_1 is language modeling auxiliary loss.

    Input representation uses special tokens:
        [START] text [DELIM] hypothesis [EXTRACT]
    """,
    predecessors=[
        "transformer_2017",
        "elmo_2018",
    ],
    successors=[
        "gpt_2_2019",
        "bert_2018",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "autoregressive",
        "pre-training",
        "transfer-learning",
        "language-model",
        "openai",
    ],
)


# =============================================================================
# GPT-2 (2019)
# =============================================================================

GPT_2 = MLMethod(
    method_id="gpt_2_2019",
    name="GPT-2",
    year=2019,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["Alec Radford", "Jeffrey Wu", "Rewon Child", "David Luan", "Dario Amodei", "Ilya Sutskever"],
    paper_title="Language Models are Unsupervised Multitask Learners",
    paper_url="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    key_innovation=(
        "Demonstrated that language models can perform downstream tasks in a zero-shot "
        "setting without any parameter updates or architecture modifications. Trained "
        "on WebText (40GB), showed emergent task-solving abilities purely from "
        "unsupervised pre-training at larger scale."
    ),
    mathematical_formulation=r"""
    GPT-2 Architecture (1.5B version):

    48-layer decoder-only transformer with:
    - 1600 hidden dimensions
    - 25 attention heads
    - 1.5B parameters total

    Model sizes: 117M, 345M, 762M, 1.5B

    Key architectural changes from GPT-1:
    - Layer normalization moved to input of each sub-block
    - Additional layer normalization after final self-attention
    - Vocabulary expanded to 50,257 tokens (BPE)
    - Context length increased to 1024 tokens
    - Batch size of 512, trained for 250k iterations

    Zero-shot task formulation:
        Task framed as conditional probability:
        P(output | input, task_description)

        Example for translation:
        P("french sentence" | "English sentence", "translate to French")

        No fine-tuning required - same model weights for all tasks.
    """,
    predecessors=[
        "gpt_1_2018",
    ],
    successors=[
        "gpt_3_2020",
        "ctrl_2019",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "autoregressive",
        "zero-shot",
        "multitask",
        "language-model",
        "openai",
        "scaling",
    ],
)


# =============================================================================
# GPT-3 (2020)
# =============================================================================

GPT_3 = MLMethod(
    method_id="gpt_3_2020",
    name="GPT-3",
    year=2020,
    era=MethodEra.ATTENTION,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=[
        "Tom Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah",
        "Jared Kaplan", "Prafulla Dhariwal", "Arvind Neelakantan",
        "Pranav Shyam", "Girish Sastry", "Amanda Askell", "et al."
    ],
    paper_title="Language Models are Few-Shot Learners",
    paper_url="https://arxiv.org/abs/2005.14165",
    key_innovation=(
        "Demonstrated that scaling language models to 175B parameters enables "
        "strong few-shot learning capabilities through in-context learning, where "
        "the model learns tasks from a few examples provided in the prompt without "
        "any gradient updates. Established scaling laws and emergent capabilities."
    ),
    mathematical_formulation=r"""
    GPT-3 Architecture (175B version):

    96-layer decoder-only transformer with:
    - 12,288 hidden dimensions
    - 96 attention heads
    - 175B parameters total

    Model sizes: 125M to 175B (8 different scales)

    In-Context Learning paradigms:

    1. Zero-shot:
       Input: "Translate English to French: cheese =>"
       Output: "fromage"

    2. One-shot:
       Input: "Translate English to French:
               sea otter => loutre de mer
               cheese =>"
       Output: "fromage"

    3. Few-shot:
       Input: "Translate English to French:
               sea otter => loutre de mer
               peppermint => menthe poivree
               plush giraffe => girafe peluche
               cheese =>"
       Output: "fromage"

    Training data: ~570GB filtered text (Common Crawl, WebText2, Books, Wikipedia)
    Context length: 2048 tokens
    Trained on ~300B tokens

    Scaling law (from Kaplan et al.):
        L(N) ~ N^{-0.076}

    where L is loss and N is number of parameters.
    """,
    predecessors=[
        "gpt_2_2019",
    ],
    successors=[
        "gpt_4_2023",
        "instructgpt_2022",
        "codex_2021",
    ],
    tags=[
        "transformer",
        "decoder-only",
        "autoregressive",
        "few-shot",
        "in-context-learning",
        "foundation-model",
        "language-model",
        "openai",
        "scaling",
        "emergent-capabilities",
    ],
)


# =============================================================================
# GPT-4 (2023)
# =============================================================================

GPT_4 = MLMethod(
    method_id="gpt_4_2023",
    name="GPT-4",
    year=2023,
    era=MethodEra.NOVEL,
    category=MethodCategory.MULTIMODAL,
    lineages=[MethodLineage.ATTENTION_LINE],
    authors=["OpenAI"],
    paper_title="GPT-4 Technical Report",
    paper_url="https://arxiv.org/abs/2303.08774",
    key_innovation=(
        "Multimodal large language model accepting both image and text inputs, "
        "producing text outputs. Exhibits human-level performance on various "
        "professional and academic benchmarks. Demonstrates significantly improved "
        "reasoning, factuality, and safety compared to predecessors."
    ),
    mathematical_formulation=r"""
    GPT-4 Architecture:

    (Limited details disclosed in technical report)

    Key capabilities demonstrated:
    - Multimodal: Processes images + text (vision encoder + LLM)
    - Extended context: Up to 32k tokens (later 128k)
    - Improved calibration and factuality

    Performance benchmarks:
    - Bar Exam: 90th percentile (vs GPT-3.5 at 10th)
    - SAT Math: 700/800
    - GRE Verbal: 99th percentile
    - LSAT: 88th percentile

    Safety improvements:
    - RLHF with human feedback
    - Red teaming and adversarial testing
    - 82% less likely to respond to disallowed content
    - 40% more factual than GPT-3.5

    Training:
    - Predictable scaling using smaller models
    - Infrastructure optimizations for training stability
    - Mixed precision training

    Note: Parameter count, architecture details, training data,
    and compute requirements not disclosed.
    """,
    predecessors=[
        "gpt_3_2020",
        "instructgpt_2022",
        "vision_transformer_2020",
    ],
    successors=[
        "gpt_4o_2024",
        "gpt_4_turbo_2024",
    ],
    tags=[
        "transformer",
        "multimodal",
        "vision-language",
        "foundation-model",
        "language-model",
        "openai",
        "rlhf",
        "safety",
        "reasoning",
    ],
)
