"""
Flamingo - Visual Language Model for Few-Shot Learning (2022)

Research index entry for Flamingo, DeepMind's visual language model that
achieves strong few-shot learning performance on vision-language tasks.

Key contributions:
- Perceiver Resampler for fixed-size visual representations
- Gated cross-attention for visual-language fusion
- Interleaved image-text training
- State-of-the-art few-shot visual question answering
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Flamingo."""
    return MLMethod(
        method_id="flamingo_2022",
        name="Flamingo",
        year=2022,
        era=MethodEra.NOVEL,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Jean-Baptiste Alayrac",
            "Jeff Donahue",
            "Pauline Luc",
            "Antoine Miech",
            "Iain Barr",
            "Yana Hasson",
            "Karel Lenc",
            "Arthur Mensch",
            "Katie Millican",
            "Malcolm Reynolds",
            "Roman Ring",
            "Eliza Rutherford",
            "Serkan Cabi",
            "Tengda Han",
            "Zhitao Gong",
            "Sina Samangooei",
            "Marianne Monteiro",
            "Jacob Menick",
            "Sebastian Borgeaud",
            "Andrew Brock",
            "Aida Nematzadeh",
            "Sahand Sharifzadeh",
            "Mikolaj Binkowski",
            "Ricardo Barreira",
            "Oriol Vinyals",
            "Andrew Zisserman",
            "Karen Simonyan",
        ],
        paper_title="Flamingo: a Visual Language Model for Few-Shot Learning",
        paper_url="https://arxiv.org/abs/2204.14198",
        key_innovation="""
        Flamingo bridges powerful pre-trained vision-only and language-only
        models using two key architectural innovations:

        1. Perceiver Resampler: Transforms variable-sized visual features into
           a fixed number of visual tokens (64), regardless of image resolution
           or number of video frames.

        2. Gated Cross-Attention: Inserts cross-attention layers into a frozen
           LLM that attend to visual tokens. A learnable gating mechanism
           (initialized to zero) allows gradual integration of visual information.

        The model handles arbitrarily interleaved sequences of images, video,
        and text, enabling flexible few-shot prompting with visual examples.
        """,
        mathematical_formulation="""
        PERCEIVER RESAMPLER
        ===================
        Input: Visual features X in R^(T x H x W x d) from vision encoder
        Output: Fixed visual tokens V in R^(N x d) where N=64

        Learned latent queries: Q in R^(N x d)

        For each cross-attention layer:
            V = V + CrossAttn(Q=V, K=flatten(X), V=flatten(X))
            V = V + FFN(V)

        GATED CROSS-ATTENTION
        =====================
        Inserted after every self-attention block in frozen LLM:

        Let x be text hidden states, v be visual tokens from Perceiver

        Standard cross-attention:
            y = CrossAttn(Q=x, K=v, V=v)

        Gated integration:
            x' = x + tanh(alpha) * y

        where alpha is initialized to 0, so tanh(alpha) = 0 initially
        (model starts as pure LLM, gradually incorporates vision)

        FULL FORWARD PASS
        =================
        1. Encode images/video with frozen vision encoder (e.g., NFNet)
        2. Apply Perceiver Resampler to get visual tokens
        3. Process interleaved sequence through LLM with gated cross-attention
        4. Generate text output autoregressively

        TRAINING OBJECTIVE
        ==================
        Standard language modeling loss on text tokens only:

        L = -sum_t log p(w_t | w_<t, images)

        Only train:
        - Perceiver Resampler
        - Gated cross-attention layers
        - Input/output embeddings for <image> tokens

        Freeze: Vision encoder, LLM layers
        """,
        predecessors=["clip_2021", "perceiver_2021", "chinchilla_2022"],
        successors=["llava_2023", "gpt4v_2023"],
        tags=[
            "multimodal",
            "few_shot",
            "visual_language_model",
            "perceiver",
            "cross_attention",
            "foundation_model",
        ],
        notes="""
        Flamingo demonstrated that frozen pre-trained models can be efficiently
        adapted for multimodal tasks with minimal additional parameters. The
        few-shot prompting capability enables rapid adaptation to new tasks
        without fine-tuning. Key design choice: keeping both vision and language
        backbones frozen preserves their individual capabilities.

        Flamingo models range from 3B to 80B parameters. The 80B model achieves
        state-of-the-art on numerous VQA benchmarks with just 4-32 shots.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for Flamingo architecture and inference."""
    return """
    FLAMINGO ARCHITECTURE
    =====================

    class PerceiverResampler:
        def __init__(self, num_latents=64, dim=1024, depth=6):
            self.latents = learnable_parameter(num_latents, dim)
            self.cross_attn_layers = [CrossAttention(dim) for _ in range(depth)]
            self.ffn_layers = [FFN(dim) for _ in range(depth)]

        def forward(self, visual_features):
            # visual_features: [B, T, H, W, D] -> flatten -> [B, T*H*W, D]
            x = flatten(visual_features)

            # Initialize with learned latents
            latents = self.latents.expand(batch_size, -1, -1)  # [B, N, D]

            # Iteratively refine through cross-attention
            for cross_attn, ffn in zip(self.cross_attn_layers, self.ffn_layers):
                latents = latents + cross_attn(query=latents, key=x, value=x)
                latents = latents + ffn(latents)

            return latents  # [B, N, D] where N=64


    class GatedCrossAttentionBlock:
        def __init__(self, dim):
            self.cross_attn = CrossAttention(dim)
            self.alpha = learnable_parameter(init=0.0)  # Tanh gate

        def forward(self, x, visual_tokens):
            # x: text hidden states [B, T, D]
            # visual_tokens: [B, N, D] from Perceiver

            y = self.cross_attn(query=x, key=visual_tokens, value=visual_tokens)
            return x + tanh(self.alpha) * y  # Gated residual


    class Flamingo:
        def __init__(self, vision_encoder, llm, perceiver_depth=6):
            self.vision_encoder = freeze(vision_encoder)  # e.g., NFNet
            self.llm = freeze(llm)  # e.g., Chinchilla
            self.perceiver = PerceiverResampler(depth=perceiver_depth)

            # Insert gated cross-attention after each LLM self-attention
            self.gated_xattn = [GatedCrossAttentionBlock(llm.dim)
                                for _ in range(llm.num_layers)]

        def forward(self, images, text):
            # Encode images
            visual_features = self.vision_encoder(images)
            visual_tokens = self.perceiver(visual_features)

            # Process through LLM with visual cross-attention
            x = self.llm.embed(text)

            for i, layer in enumerate(self.llm.layers):
                x = layer.self_attention(x)
                x = self.gated_xattn[i](x, visual_tokens)  # Visual fusion
                x = layer.ffn(x)

            return self.llm.output(x)


    FEW-SHOT PROMPTING
    ==================

    # Example: 4-shot VQA
    prompt = '''
    <image1>Question: What color is the car? Answer: Red
    <image2>Question: How many people? Answer: Three
    <image3>Question: What animal is this? Answer: Dog
    <image4>Question: Is it raining? Answer: Yes
    <query_image>Question: {question} Answer:
    '''

    # Model generates answer conditioned on examples
    answer = flamingo.generate(prompt, images=[img1, img2, img3, img4, query_img])
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for Flamingo."""
    return {
        "perceiver_cross_attn": "V = V + softmax(QK^T/sqrt(d))V",
        "gated_fusion": "x' = x + tanh(alpha) * CrossAttn(x, v, v)",
        "tanh_gate_init": "alpha = 0 => tanh(alpha) = 0 (starts as pure LLM)",
        "training_loss": "L = -sum_t log p(w_t | w_<t, I_1, ..., I_k)",
        "resampler_output": "V in R^(N x d), N = 64 (fixed)",
    }


def get_model_variants() -> List[Dict[str, str]]:
    """Return Flamingo model variants and their configurations."""
    return [
        {
            "name": "Flamingo-3B",
            "llm_backbone": "Chinchilla 1.4B",
            "vision_encoder": "NFNet-F0",
            "total_params": "3B",
            "trainable_params": "1.6B",
        },
        {
            "name": "Flamingo-9B",
            "llm_backbone": "Chinchilla 7B",
            "vision_encoder": "NFNet-F6",
            "total_params": "9B",
            "trainable_params": "2B",
        },
        {
            "name": "Flamingo-80B",
            "llm_backbone": "Chinchilla 70B",
            "vision_encoder": "NFNet-F6",
            "total_params": "80B",
            "trainable_params": "10B",
        },
    ]


def get_training_data() -> Dict[str, str]:
    """Return information about Flamingo's training data."""
    return {
        "M3W": "Interleaved image-text from 43M web pages",
        "ALIGN": "1.8B image-alt-text pairs",
        "LTIP": "Long Text & Image Pairs (312M)",
        "VTP": "Video & Text Pairs (27M short videos)",
        "total_images": "~2.1B images",
        "total_videos": "~27M videos",
        "training_objective": "Next token prediction on text only",
    }
