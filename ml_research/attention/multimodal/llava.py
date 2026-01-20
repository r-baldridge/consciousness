"""
LLaVA - Large Language and Vision Assistant (2023)

Research index entry for LLaVA, a multimodal model that combines a vision
encoder with an LLM for visual instruction following.

Key contributions:
- Simple yet effective vision-language connector
- Visual instruction tuning methodology
- GPT-4 generated instruction-following data
- Strong multimodal conversation capabilities
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for LLaVA."""
    return MLMethod(
        method_id="llava_2023",
        name="LLaVA (Large Language and Vision Assistant)",
        year=2023,
        era=MethodEra.NOVEL,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Haotian Liu",
            "Chunyuan Li",
            "Qingyang Wu",
            "Yong Jae Lee",
        ],
        paper_title="Visual Instruction Tuning",
        paper_url="https://arxiv.org/abs/2304.08485",
        key_innovation="""
        LLaVA demonstrates that connecting a pre-trained vision encoder (CLIP)
        to a pre-trained LLM (LLaMA/Vicuna) with a simple linear projection
        layer, followed by visual instruction tuning, produces a powerful
        multimodal assistant.

        Key innovations:
        1. Simplicity: Single linear layer (or 2-layer MLP) to project visual
           features into the LLM's embedding space

        2. Visual Instruction Tuning: Fine-tune on instruction-following data
           that includes images, using GPT-4 to generate training conversations

        3. Two-stage training:
           - Stage 1: Pre-train projection on image-caption pairs (frozen encoders)
           - Stage 2: End-to-end fine-tune on visual instruction data

        4. Data Generation: Use GPT-4 to create diverse instruction-following
           examples from COCO images with captions and bounding boxes
        """,
        mathematical_formulation="""
        ARCHITECTURE
        ============
        Components:
            Vision Encoder: CLIP ViT-L/14 (frozen in Stage 1)
            Language Model: LLaMA/Vicuna (frozen in Stage 1)
            Projection: W in R^(d_v x d_l) linear layer

        Visual feature extraction:
            Given image I:
            Z_v = CLIP_visual(I)  # [H x W, d_v] patch features

        Projection to language space:
            H_v = Z_v @ W  # [H x W, d_l] visual tokens

        For LLaVA-1.5, use 2-layer MLP:
            H_v = MLP(Z_v) = W_2 * GELU(W_1 * Z_v + b_1) + b_2

        SEQUENCE CONSTRUCTION
        =====================
        Combine visual and text tokens:
            X = [H_v; Embed(text_tokens)]

            Example sequence:
            [<image_tokens>; "USER: What is in this image?"; "ASSISTANT:"]

        TRAINING
        ========
        Stage 1 - Feature Alignment (freeze vision + LLM):
            Train only projection W on CC3M captions
            L = -sum_t log p(w_t | X_v, w_<t)

        Stage 2 - Visual Instruction Tuning (freeze vision only):
            Train projection W and LLM on instruction data
            L = -sum_t log p(w_t | X_v, X_instruct, w_<t)

            Only compute loss on assistant responses, not user turns

        INFERENCE
        =========
        1. Extract CLIP visual features
        2. Project to LLM space
        3. Prepend to tokenized conversation
        4. Autoregressively generate response
        """,
        predecessors=["clip_2021", "llama_2023", "vicuna_2023", "flamingo_2022"],
        successors=["llava_1_5_2023", "llava_next_2024"],
        tags=[
            "multimodal",
            "visual_instruction_tuning",
            "vision_language",
            "foundation_model",
            "instruction_following",
            "open_source",
        ],
        notes="""
        LLaVA's significance lies in demonstrating that effective multimodal
        models can be built with minimal architectural changes to existing
        models. The visual instruction tuning paradigm showed that the key
        bottleneck was training data quality, not architectural complexity.

        LLaVA-1.5 improved upon the original with:
        - Higher resolution (336px)
        - 2-layer MLP projector instead of linear
        - More diverse training data
        - Academic task-specific fine-tuning data

        The open-source nature enabled rapid community iteration and research.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for LLaVA architecture and training."""
    return """
    LLAVA ARCHITECTURE
    ==================

    class LLaVA:
        def __init__(self):
            # Pre-trained components
            self.vision_encoder = CLIP_ViT_L14(pretrained=True)
            self.llm = Vicuna_13B(pretrained=True)

            # Trainable projection (LLaVA-1.5 uses MLP)
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim)
            )

        def encode_image(self, image):
            # Extract CLIP features (before final pooling)
            features = self.vision_encoder.encode_image(image)
            # features shape: [B, num_patches, vision_dim]

            # Project to language model dimension
            visual_tokens = self.projector(features)
            # visual_tokens shape: [B, num_patches, llm_dim]

            return visual_tokens

        def forward(self, image, text):
            # Get visual tokens
            visual_tokens = self.encode_image(image)

            # Tokenize and embed text
            text_tokens = self.llm.tokenize(text)
            text_embeds = self.llm.embed(text_tokens)

            # Concatenate visual and text embeddings
            # Replace <image> placeholder with visual tokens
            combined = insert_visual_tokens(text_embeds, visual_tokens)

            # Forward through LLM
            output = self.llm(combined)

            return output


    TRAINING STAGE 1: FEATURE ALIGNMENT
    ====================================

    # Freeze vision encoder and LLM
    freeze(model.vision_encoder)
    freeze(model.llm)

    # Only train projection layer
    optimizer = AdamW(model.projector.parameters(), lr=1e-3)

    for image, caption in cc3m_dataset:
        # Construct prompt: "<image>\n{caption}"
        prompt = f"<image>\n{caption}"

        # Forward pass
        logits = model(image, prompt)

        # Language modeling loss on caption tokens
        loss = cross_entropy(logits, caption_tokens)
        loss.backward()
        optimizer.step()


    TRAINING STAGE 2: VISUAL INSTRUCTION TUNING
    ============================================

    # Freeze only vision encoder
    freeze(model.vision_encoder)
    unfreeze(model.projector)
    unfreeze(model.llm)

    optimizer = AdamW([
        {'params': model.projector.parameters(), 'lr': 2e-5},
        {'params': model.llm.parameters(), 'lr': 2e-5}
    ])

    for image, conversation in instruction_dataset:
        # conversation = [
        #   {"role": "user", "content": "<image>\nDescribe this image."},
        #   {"role": "assistant", "content": "The image shows..."}
        # ]

        # Format as multi-turn dialogue
        prompt = format_conversation(conversation)

        # Forward pass
        logits = model(image, prompt)

        # Loss only on assistant responses (masked)
        loss = masked_cross_entropy(logits, targets, assistant_mask)
        loss.backward()
        optimizer.step()


    GPT-4 DATA GENERATION
    =====================

    def generate_instruction_data(image_caption, bounding_boxes):
        '''
        Use GPT-4 (text-only) to generate instruction-following examples
        '''
        context = f'''
        Image caption: {image_caption}
        Objects in image: {bounding_boxes}

        Generate a conversation between a user and an AI assistant
        about this image. The conversation should include:
        1. A question about the image content
        2. Detailed description request
        3. Complex reasoning question
        '''

        conversation = gpt4.generate(context)
        return conversation
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for LLaVA."""
    return {
        "visual_projection": "H_v = Z_v @ W (linear) or MLP(Z_v)",
        "mlp_projection": "H_v = W_2 * GELU(W_1 * Z_v + b_1) + b_2",
        "sequence_construction": "X = [H_v; Embed(text)]",
        "stage1_loss": "L = -sum_t log p(w_t | H_v, w_<t)",
        "stage2_loss": "L = -sum_{t in assistant} log p(w_t | H_v, X_conv, w_<t)",
        "num_visual_tokens": "N = (H/14) * (W/14) = 256 for 224px, 576 for 336px",
    }


def get_model_variants() -> List[Dict[str, str]]:
    """Return LLaVA model variants."""
    return [
        {
            "name": "LLaVA-7B",
            "vision_encoder": "CLIP ViT-L/14",
            "llm": "LLaMA-7B / Vicuna-7B",
            "projector": "Linear",
            "resolution": "224x224",
            "visual_tokens": 256,
        },
        {
            "name": "LLaVA-13B",
            "vision_encoder": "CLIP ViT-L/14",
            "llm": "LLaMA-13B / Vicuna-13B",
            "projector": "Linear",
            "resolution": "224x224",
            "visual_tokens": 256,
        },
        {
            "name": "LLaVA-1.5-7B",
            "vision_encoder": "CLIP ViT-L/14@336px",
            "llm": "Vicuna-7B",
            "projector": "2-layer MLP",
            "resolution": "336x336",
            "visual_tokens": 576,
        },
        {
            "name": "LLaVA-1.5-13B",
            "vision_encoder": "CLIP ViT-L/14@336px",
            "llm": "Vicuna-13B",
            "projector": "2-layer MLP",
            "resolution": "336x336",
            "visual_tokens": 576,
        },
    ]


def get_training_data_types() -> List[Dict[str, str]]:
    """Return types of training data used in LLaVA."""
    return [
        {
            "type": "Conversation",
            "description": "Multi-turn dialogues about images",
            "example": "Q: What's in the image? A: I see a dog... Q: What breed?",
            "source": "GPT-4 generated from COCO",
        },
        {
            "type": "Detail Description",
            "description": "Comprehensive image descriptions",
            "example": "Describe this image in detail.",
            "source": "GPT-4 generated from COCO",
        },
        {
            "type": "Complex Reasoning",
            "description": "Questions requiring multi-step reasoning",
            "example": "What might happen next in this scene?",
            "source": "GPT-4 generated from COCO",
        },
        {
            "type": "Academic VQA",
            "description": "Standard VQA benchmarks (LLaVA-1.5)",
            "example": "VQAv2, GQA, OCR-VQA, TextVQA",
            "source": "Existing datasets reformatted",
        },
    ]
