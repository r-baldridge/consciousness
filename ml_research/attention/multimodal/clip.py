"""
CLIP - Contrastive Language-Image Pre-training (2021)

Research index entry for CLIP, which learns visual concepts from natural
language supervision using contrastive learning on image-text pairs.

Key contributions:
- Contrastive pre-training on 400M image-text pairs
- Zero-shot transfer to downstream tasks
- Natural language as flexible class specification
- Emergent multimodal understanding
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for CLIP."""
    return MLMethod(
        method_id="clip_2021",
        name="CLIP (Contrastive Language-Image Pre-training)",
        year=2021,
        era=MethodEra.NOVEL,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Alec Radford",
            "Jong Wook Kim",
            "Chris Hallacy",
            "Aditya Ramesh",
            "Gabriel Goh",
            "Sandhini Agarwal",
            "Girish Sastry",
            "Amanda Askell",
            "Pamela Mishkin",
            "Jack Clark",
            "Gretchen Krueger",
            "Ilya Sutskever",
        ],
        paper_title="Learning Transferable Visual Models From Natural Language Supervision",
        paper_url="https://arxiv.org/abs/2103.00020",
        key_innovation="""
        CLIP learns visual representations by predicting which caption goes with
        which image using contrastive learning. Instead of predicting exact words,
        CLIP learns a joint embedding space where matching image-text pairs are
        close together and non-matching pairs are far apart.

        Key innovations:
        1. Scale: Pre-trained on 400 million image-text pairs from the internet
        2. Efficiency: Contrastive objective more efficient than generative
        3. Zero-shot: Natural language specifies visual concepts at test time
        4. Robustness: Strong performance on distribution shift benchmarks

        CLIP enables zero-shot image classification by comparing image embeddings
        to text embeddings of class descriptions (e.g., "a photo of a {class}").
        """,
        mathematical_formulation="""
        Architecture:
            Image Encoder: f_image(x) -> z_image in R^d  (ViT or ResNet)
            Text Encoder: f_text(t) -> z_text in R^d    (Transformer)

        Embedding normalization:
            z_image = f_image(x) / ||f_image(x)||_2
            z_text = f_text(t) / ||f_text(t)||_2

        Cosine similarity:
            s(x, t) = z_image^T * z_text * exp(tau)
            where tau is a learned temperature parameter

        Contrastive loss (InfoNCE / symmetric cross-entropy):
            For batch of N (image, text) pairs:

            Image-to-text loss:
            L_i2t = -(1/N) * sum_i log(exp(s(x_i, t_i)) / sum_j exp(s(x_i, t_j)))

            Text-to-image loss:
            L_t2i = -(1/N) * sum_i log(exp(s(x_i, t_i)) / sum_j exp(s(x_j, t_i)))

            Total loss:
            L = (L_i2t + L_t2i) / 2

        Zero-shot classification:
            For image x and classes C = {c_1, ..., c_K}:
            1. Create text prompts: t_k = "a photo of a {c_k}"
            2. Compute: p(c_k | x) = exp(s(x, t_k)) / sum_j exp(s(x, t_j))
            3. Predict: c* = argmax_k p(c_k | x)
        """,
        predecessors=["vit_2020", "gpt2_2019", "transformer_2017"],
        successors=["dalle_2021", "stable_diffusion_2022", "llava_2023"],
        tags=[
            "multimodal",
            "contrastive_learning",
            "zero_shot",
            "vision_language",
            "foundation_model",
            "transfer_learning",
        ],
        notes="""
        CLIP demonstrated that natural language supervision can scale to learn
        robust visual representations. Its zero-shot capabilities challenged
        the standard paradigm of task-specific fine-tuning. CLIP embeddings
        became widely used for image-text retrieval, generation guidance
        (DALL-E, Stable Diffusion), and multimodal understanding.

        Limitations include poor fine-grained classification, counting,
        and abstract/systematic tasks. The model inherits biases from
        internet-scraped training data.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for CLIP training and inference."""
    return """
    CLIP TRAINING ALGORITHM
    =======================

    # Initialize encoders
    image_encoder = VisionTransformer(patch_size=16, width=768, layers=12)
    text_encoder = Transformer(vocab_size=49152, width=512, layers=12)
    temperature = learnable_parameter(init=0.07)

    for batch in data_loader:
        images, texts = batch  # N image-text pairs

        # Encode both modalities
        image_features = image_encoder(images)  # [N, d]
        text_features = text_encoder(texts)     # [N, d]

        # L2 normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logits = (image_features @ text_features.T) * exp(temperature)  # [N, N]

        # Symmetric cross-entropy loss
        labels = arange(N)  # diagonal elements are positive pairs
        loss_i2t = cross_entropy(logits, labels)
        loss_t2i = cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        # Update all parameters
        loss.backward()
        optimizer.step()


    ZERO-SHOT CLASSIFICATION
    ========================

    def zero_shot_classify(image, class_names, prompt_template="a photo of a {}"):
        # Encode image
        image_features = image_encoder(image)
        image_features = image_features / image_features.norm()

        # Encode class descriptions
        text_prompts = [prompt_template.format(c) for c in class_names]
        text_features = text_encoder(text_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = image_features @ text_features.T

        # Return most similar class
        return class_names[similarities.argmax()]
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for CLIP."""
    return {
        "image_embedding": "z_I = f_I(x) / ||f_I(x)||_2",
        "text_embedding": "z_T = f_T(t) / ||f_T(t)||_2",
        "cosine_similarity": "s(x, t) = z_I^T z_T * exp(tau)",
        "info_nce": "L = -log(exp(s(x_i, t_i)) / sum_j exp(s(x_i, t_j)))",
        "symmetric_loss": "L = (L_i2t + L_t2i) / 2",
        "zero_shot_prob": "p(c|x) = softmax(s(x, t_c))",
    }


def get_architecture_details() -> Dict[str, Dict]:
    """Return architecture configurations for CLIP variants."""
    return {
        "ViT-B/32": {
            "image_encoder": "ViT-Base with 32x32 patches",
            "text_encoder": "Transformer (63M params)",
            "embedding_dim": 512,
            "image_resolution": 224,
            "params": "151M",
        },
        "ViT-B/16": {
            "image_encoder": "ViT-Base with 16x16 patches",
            "text_encoder": "Transformer (63M params)",
            "embedding_dim": 512,
            "image_resolution": 224,
            "params": "150M",
        },
        "ViT-L/14": {
            "image_encoder": "ViT-Large with 14x14 patches",
            "text_encoder": "Transformer (123M params)",
            "embedding_dim": 768,
            "image_resolution": 224,
            "params": "428M",
        },
        "ViT-L/14@336px": {
            "image_encoder": "ViT-Large with 14x14 patches",
            "text_encoder": "Transformer (123M params)",
            "embedding_dim": 768,
            "image_resolution": 336,
            "params": "428M",
        },
        "RN50": {
            "image_encoder": "ResNet-50 (modified)",
            "text_encoder": "Transformer (63M params)",
            "embedding_dim": 1024,
            "image_resolution": 224,
            "params": "102M",
        },
    }


def get_prompt_engineering_tips() -> List[Dict[str, str]]:
    """Return prompt engineering strategies for CLIP zero-shot."""
    return [
        {
            "strategy": "Context prompt",
            "template": "a photo of a {class}",
            "rationale": "Specifies the domain (photographs)",
        },
        {
            "strategy": "Domain-specific",
            "template": "a satellite photo of {class}",
            "rationale": "Match training distribution for specific domains",
        },
        {
            "strategy": "Ensemble prompts",
            "templates": [
                "a photo of a {class}",
                "a photo of the {class}",
                "an image of a {class}",
            ],
            "rationale": "Average embeddings reduce variance",
        },
        {
            "strategy": "Fine-grained description",
            "template": "a photo of a {class}, a type of {category}",
            "rationale": "Add hierarchical context for fine-grained classes",
        },
    ]
