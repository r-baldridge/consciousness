"""
Genie - Bruce et al., Google DeepMind (2024)

Generative Interactive Environments from unlabeled video.
Genie learns to generate playable 2D worlds from video-only data,
discovering actions and dynamics without any action labels.

Paper: "Genie: Generative Interactive Environments"
arXiv: 2402.15391

Key Innovation:
    Learn action-controllable world models from video without action labels.
    Uses a latent action model to infer actions between frames, then
    conditions a dynamics model on these latent actions for generation.

Architecture:
    1. Video Tokenizer: Spatiotemporal VQ encoding
    2. Latent Action Model: Infer actions between frames
    3. Dynamics Model: Predict next tokens given history and action
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

GENIE = MLMethod(
    method_id="genie_2024",
    name="Genie",
    year=2024,

    era=MethodEra.NOVEL,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.ATTENTION_LINE],

    authors=[
        "Jake Bruce", "Michael Dennis", "Ashley Edwards", "Jack Parker-Holder",
        "Yuge Shi", "Edward Hughes", "Matthew Lai", "Aditi Mavalankar",
        "Richie Steiber", "Chris Apps", "Yusuf Aytar", "Sarah Bechtle",
        "Feryal Behbahani", "Stephanie Chan", "Nicolas Heess", "Lucy Gonzalez",
        "Simon Osindero", "Sherjil Ozair", "Scott Reed", "Jingwei Zhang",
        "Konrad Zolna", "Jeff Clune", "Nando de Freitas", "Satinder Singh",
        "Tim Rocktaschel"
    ],
    paper_title="Genie: Generative Interactive Environments",
    paper_url="https://arxiv.org/abs/2402.15391",

    key_innovation=(
        "First large-scale generative model that creates playable, action-controllable "
        "environments from unlabeled video. Key insight: use a Latent Action Model (LAM) "
        "to discover actions from frame transitions, then use these latent actions to "
        "condition a dynamics model. Enables generation of diverse 2D platformer worlds "
        "from a single image prompt."
    ),

    mathematical_formulation=r"""
Video Tokenizer (ST-ViViT):
    Input: Video frames x_{1:T} in R^{T x H x W x C}
    Output: Discrete tokens z_{1:T} in {1,...,K}^{T x h x w}

    Encoder: z_t = VQ(Encoder(x_t), codebook)
    Decoder: x_hat_t = Decoder(z_t)

    Spatial-temporal factorization:
    - Spatial encoder per frame
    - Temporal transformer across frames
    - VQ bottleneck with K codes

Latent Action Model (LAM):
    Input: Adjacent frame tokens (z_t, z_{t+1})
    Output: Latent action a_t in {1,...,A}

    a_t = VQ(LAM_encoder(z_t, z_{t+1}), action_codebook)

    Where A = 8 (8 discrete latent actions discovered)

Dynamics Model (MaskGIT-style):
    Input: History z_{1:t}, action a_t
    Output: Next frame tokens z_{t+1}

    P(z_{t+1} | z_{1:t}, a_t) = prod_i P(z_{t+1,i} | z_{1:t}, a_t, z_{t+1,<i})

    Using masked prediction with parallel decoding

Full Generation Loop:
    1. Given initial frame x_1, encode: z_1 = Tokenizer(x_1)
    2. User provides action: a_1
    3. Dynamics predicts: z_2 = Dynamics(z_1, a_1)
    4. Decode: x_2 = Decoder(z_2)
    5. Repeat for interactive generation
""",

    predecessors=["vq_vae_2017", "maskgit_2022", "dreamer_v3_2023"],
    successors=[],

    tags=[
        "world-model", "generative", "video-tokenizer", "latent-actions",
        "interactive-environments", "unsupervised", "foundation-model"
    ],
    notes=(
        "Trained on 200K hours of 2D platformer videos from the internet. "
        "The model has 11B parameters. Despite no action labels in training, "
        "the LAM discovers meaningful actions (left, right, jump, etc.) that "
        "correspond to human-interpretable controls. Enables 'imagination as "
        "data' - generating training environments for RL agents."
    )
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Genie."""
    return GENIE


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class GenieArchitecture:
    """Reference architecture for Genie."""

    # Video tokenizer
    frame_size: tuple = (64, 64)
    patch_size: int = 4
    token_grid: tuple = (16, 16)  # 64/4 = 16
    codebook_size: int = 1024

    # Latent action model
    num_latent_actions: int = 8

    # Dynamics model
    context_frames: int = 16
    transformer_layers: int = 18
    hidden_dim: int = 2048
    num_heads: int = 16

    @staticmethod
    def video_tokenizer_structure() -> str:
        """ST-ViViT tokenizer structure."""
        return """
Video Tokenizer (Spatiotemporal VQ-ViViT):

Encoder:
    Input: Frame x_t [H, W, C] = [64, 64, 3]

    # Patchify
    patches = split_into_patches(x_t, patch_size=4)  # [16, 16, 48]

    # Spatial Transformer
    spatial_tokens = LinearEmbed(patches) + pos_embed  # [256, D]
    spatial_out = SpatialTransformer(spatial_tokens)   # [256, D]

    # Temporal Transformer (across frames)
    for each position i in [256]:
        temporal_input = [spatial_out_1[i], spatial_out_2[i], ...]
        temporal_out[i] = TemporalTransformer(temporal_input)

    # Vector Quantization
    z_t = argmin_k ||temporal_out - codebook[k]||^2
    z_t in {1,...,1024}^{16x16}

Decoder:
    # Lookup embeddings
    embeddings = codebook[z_t]  # [256, D]

    # Transformer decoder
    decoded = TransformerDecoder(embeddings)

    # Unpatchify
    x_hat_t = reshape_to_image(decoded)  # [64, 64, 3]

Training Loss:
    L = ||x_t - x_hat_t||^2 + ||sg(z_e) - e||^2 + beta * ||z_e - sg(e)||^2

    Where z_e = encoder output, e = quantized embedding
"""

    @staticmethod
    def latent_action_model_structure() -> str:
        """Latent Action Model structure."""
        return """
Latent Action Model (LAM):

Purpose: Discover actions from frame transitions without labels

Input:
    z_t: Current frame tokens [h, w] = [16, 16]
    z_{t+1}: Next frame tokens [h, w] = [16, 16]

Architecture:
    # Encode frames
    h_t = FrameEncoder(z_t)      # [h*w, D]
    h_{t+1} = FrameEncoder(z_{t+1})  # [h*w, D]

    # Cross-attention to find differences
    diff = CrossAttention(h_{t+1}, h_t)

    # Pool to single action representation
    action_repr = GlobalPool(diff)  # [D]

    # Quantize to discrete action
    a_t = VQ(action_repr, action_codebook)  # in {1,...,8}

Training:
    The LAM is trained jointly with dynamics model.
    Objective: Enable dynamics model to predict z_{t+1} from (z_t, a_t)

    This incentivizes the LAM to capture action-relevant information
    in the transition, discovering meaningful latent actions.

Discovered Actions (emergent):
    Despite only 8 codes, LAM learns:
    - Directional movement (left, right, up, down)
    - Jump actions
    - Interaction actions
    These emerge from video structure without any labels!
"""

    @staticmethod
    def dynamics_model_structure() -> str:
        """Dynamics model structure."""
        return """
Dynamics Model (MaskGIT-style Transformer):

Purpose: Predict next frame tokens given history and action

Input:
    z_{1:t}: History of token grids [t, h, w]
    a_t: Latent action (discrete, 1 of 8)

Output:
    z_{t+1}: Predicted next frame tokens [h, w]

Architecture:
    # Flatten spatial tokens with temporal position
    tokens = []
    for tau in range(t):
        for (i,j) in positions:
            tokens.append(embed(z_tau[i,j]) + pos(tau,i,j))

    # Add action conditioning
    action_embed = ActionEmbed(a_t)
    tokens = [action_embed] + tokens

    # Causal Transformer
    # Attends to past frames and current action
    out = CausalTransformer(tokens)

    # Predict next frame tokens
    logits = OutputHead(out[-h*w:])  # [h*w, vocab_size]

MaskGIT Decoding (parallel generation):
    1. Initialize z_{t+1} as all [MASK]
    2. For iter in range(T):
        - Predict logits for masked positions
        - Sample highest confidence positions
        - Unmask top-k positions
    3. After T iterations, all positions filled

    Much faster than autoregressive (256 steps -> ~10 steps)

Training Loss:
    L = CrossEntropy(logits, z_{t+1}_target)

    With masking: randomly mask subset of z_{t+1},
    predict masked from unmasked + history
"""

    @staticmethod
    def generation_procedure() -> str:
        """Interactive generation procedure."""
        return """
Interactive Generation:

Given: Initial image x_1 (user-provided prompt)

Setup:
    z_1 = VideoTokenizer.encode(x_1)
    history = [z_1]

Interactive Loop:
    while user_playing:
        # Get user action (mapped to latent action)
        user_input = get_keyboard_input()  # e.g., "right arrow"
        a_t = map_to_latent_action(user_input)  # e.g., a_t = 3

        # Generate next frame
        z_{t+1} = DynamicsModel.generate(history, a_t)

        # Decode and display
        x_{t+1} = VideoTokenizer.decode(z_{t+1})
        display(x_{t+1})

        # Update history
        history.append(z_{t+1})
        if len(history) > context_length:
            history = history[-context_length:]

Key Properties:
    - Real-time generation at playable framerates
    - Coherent multi-step generation (not just frame-to-frame)
    - Novel environments from single image prompt
    - Actions transfer across different visual styles
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def maskgit_sampling(model, history, action, num_iterations=10):
    """
    MaskGIT-style parallel decoding for dynamics model.

    Instead of autoregressive token-by-token generation,
    iteratively unmask tokens in parallel based on confidence.

    Args:
        model: Dynamics model
        history: Past frame tokens [t, h, w]
        action: Latent action
        num_iterations: Number of unmasking iterations

    Returns:
        Generated frame tokens [h, w]
    """
    return {
        "algorithm": """
            1. z = [MASK] * num_tokens  # Start fully masked
            2. for iter in range(num_iterations):
                   # Predict all positions
                   logits = model(history, action, z)
                   probs = softmax(logits)

                   # Sample tokens
                   samples = sample_from(probs)
                   confidences = max(probs, dim=-1)

                   # Determine how many to unmask
                   ratio = 1 - (iter + 1) / num_iterations
                   num_mask = int(ratio * num_tokens)

                   # Keep highest confidence, remask lowest
                   sorted_idx = argsort(confidences)
                   unmask_idx = sorted_idx[num_mask:]
                   z[unmask_idx] = samples[unmask_idx]

            3. return z
        """,
        "complexity": "O(num_iterations * Transformer) vs O(num_tokens * Transformer)",
        "speedup": "Typically 10-25x faster than autoregressive"
    }


def latent_action_discovery(frame_pairs, action_codebook_size=8):
    """
    Learn latent actions from frame transition pairs.

    The LAM learns to compress (z_t, z_{t+1}) -> a_t such that
    a_t captures the action-relevant information for prediction.

    Args:
        frame_pairs: Pairs of consecutive frame tokens
        action_codebook_size: Number of discrete actions to discover

    Returns:
        Latent action model and codebook
    """
    return {
        "objective": """
            Learn a_t = LAM(z_t, z_{t+1}) such that:
            z_{t+1} = Dynamics(z_t, a_t)

            The information bottleneck (8 codes) forces LAM to
            capture only action-relevant differences.
        """,
        "training": """
            L = CrossEntropy(Dynamics(z_t, LAM(z_t, z_{t+1})), z_{t+1})
              + VQ_commitment_loss
        """,
        "emergence": """
            With 8 codes and platformer videos:
            - Code 0-1: Horizontal movement (left/right)
            - Code 2-3: Vertical movement (up/down)
            - Code 4-5: Jump variations
            - Code 6-7: Interaction/special actions

            These emerge from data statistics, not labels!
        """
    }


# =============================================================================
# Key Insights and Applications
# =============================================================================

GENIE_INSIGHTS = {
    "action_discovery": """
        Latent Action Discovery:
        - Actions emerge from frame differences
        - Information bottleneck (8 codes) forces compression
        - Learned actions are human-interpretable
        - Same architecture discovers different action sets for different domains
    """,

    "video_as_world_model": """
        Internet Video as World Knowledge:
        - 200K hours of platformer videos
        - Captures physics, object interactions, game mechanics
        - No environment interaction needed for learning
        - Scales with more video data
    """,

    "imagination_as_data": """
        Generated Worlds as Training Data:
        - Generate novel environments from image prompts
        - Train RL agents in imagined worlds
        - Infinite training environment diversity
        - No human level design needed
    """,

    "controllable_generation": """
        Controllability Without Labels:
        - Learn to respond to latent actions
        - Map keyboard to latent actions at inference
        - Enables interactive play
        - Actions generalize across visual styles
    """
}


GENIE_APPLICATIONS = {
    "game_generation": {
        "description": "Generate playable games from single image",
        "input": "Image prompt (hand-drawn or real)",
        "output": "Interactive 2D platformer",
        "examples": "Sketch a character -> play as that character"
    },

    "rl_environment_synthesis": {
        "description": "Create training environments for RL",
        "benefit": "Unlimited diverse environments",
        "approach": "Generate from image prompts, train agent in generated world",
        "paper_result": "Agents trained in Genie worlds transfer to real games"
    },

    "creative_tools": {
        "description": "Tools for game designers",
        "workflow": "Sketch concept art -> Generate playable prototype",
        "iteration": "Modify prompt -> New environment variant"
    },

    "research_platform": {
        "description": "Study emergent behavior and world models",
        "questions": [
            "What physics does the model learn?",
            "How do latent actions relate to human concepts?",
            "Can we improve transfer to real environments?"
        ]
    }
}


GENIE_LIMITATIONS = {
    "resolution": "Currently 64x64, lower than modern games",
    "domains": "Trained on 2D platformers, not general video",
    "temporal": "Limited context window for long-term coherence",
    "physics": "Approximate physics, not simulation-accurate",
    "training_cost": "11B parameters, significant compute"
}
