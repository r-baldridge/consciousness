"""
Test-Time Training (TTT) Layers - Stanford/NVIDIA, 2024

A revolutionary sequence modeling architecture that replaces attention with
learnable hidden states that update via gradient descent during inference.
Linear complexity in sequence length with true long-context capability.

Paper: "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
arXiv: https://arxiv.org/abs/2407.04620
GitHub: https://github.com/test-time-training/ttt-lm-pytorch
Authors: Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram,
         Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang,
         Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin
Organization: Stanford University, NVIDIA

=============================================================================
KEY INNOVATION: THE SELF-SUPERVISED HIDDEN STATE
=============================================================================

The Core Insight:
    Traditional RNNs: Hidden state is a fixed-size VECTOR
    TTT Layers: Hidden state is a NEURAL NETWORK WITH WEIGHTS

During Inference:
    - The hidden state (a model's weights) is TRAINED on each input token
    - Learning happens IN REAL-TIME during the forward pass
    - Each token's representation is improved by self-supervised learning
    - Equivalent to doing "test-time training" at every timestep

Why This Matters:
    - Linear complexity: O(n) vs O(n^2) for attention
    - Unbounded context: No fixed context window
    - Adaptive: Hidden state specializes to current context
    - Parallelizable: Mini-batch gradient descent enables GPU efficiency

=============================================================================
ARCHITECTURE COMPARISON
=============================================================================

Standard Transformer:
    Token_1 ──┬── Attention ──┬── Output_1
    Token_2 ──┤    (O(n^2))   ├── Output_2
    Token_n ──┴───────────────┴── Output_n

Traditional RNN:
    Token_1 → [h_1] → Token_2 → [h_2] → ... → Token_n → [h_n]
                ↓                 ↓                       ↓
            Output_1          Output_2               Output_n
    Hidden state: fixed-size vector (compression bottleneck)

TTT Layer:
    Token_1 → [W_1] → Token_2 → [W_2] → ... → Token_n → [W_n]
                ↓                 ↓                       ↓
            Output_1          Output_2               Output_n

    Hidden state W_t = neural network weights
    W_t updated by gradient descent: W_{t+1} = W_t - η∇L(x_t; W_t)

=============================================================================
TTT VARIANTS
=============================================================================

TTT-Linear:
    - Hidden state: Single linear layer (W ∈ R^{d×d})
    - Self-supervised task: Reconstruct input from projection
    - Fastest variant, strong baseline performance
    - Memory: O(d^2) per position

TTT-MLP:
    - Hidden state: Two-layer MLP
    - More expressive but slightly slower
    - Better for complex dependencies
    - Memory: O(d×d_hidden + d_hidden×d)

TTT-Base vs TTT-Mini:
    - TTT-Base: Replaces attention in standard architecture
    - TTT-Mini: Mini-batch TTT for improved parallelism

=============================================================================
MATHEMATICAL FORMULATION
=============================================================================

Self-Supervised Learning Objective:
    L(x; W) = ||f(xW_K; W) - xW_V||^2

    where:
        x = input token embedding
        W_K = key projection (learned, fixed at test time)
        W_V = value projection (learned, fixed at test time)
        W = hidden state weights (updated at test time)
        f(·; W) = TTT model (linear or MLP)

Hidden State Update:
    W_{t+1} = W_t - η∇_W L(x_t; W_t)

    Single gradient step per token (can be multiple)

Output Computation:
    z_t = f(x_t W_K; W_t) + x_t  (with residual connection)
    output_t = z_t W_O           (output projection)

Mini-Batch TTT (for parallelism):
    - Process tokens in mini-batches
    - Batch gradient descent instead of online SGD
    - Enables parallel processing on GPUs
    - Trade-off: Less fine-grained adaptation

Dual Form (Equivalent Perspective):
    TTT can be viewed as:
        z_t = Σ_{i<t} K(x_t, x_i) · x_i W_V

    where K is an implicit kernel induced by gradient descent
    This connects TTT to attention mechanisms theoretically

=============================================================================
BENCHMARKS
=============================================================================

Language Modeling (Perplexity, lower is better):

    Model                   Pile (val)    Books       ArXiv       Code
    ───────────────────────────────────────────────────────────────────
    Mamba 1.4B             8.85          12.20       8.16        9.48
    TTT-Linear 1.3B        8.71          11.23       7.91        9.33
    TTT-MLP 1.3B           8.52          10.87       7.72        9.08
    Transformer 1.3B       8.44          10.71       7.58        8.95

Long Context (32K-128K tokens):
    - TTT maintains quality at 128K context
    - Linear scaling vs quadratic for attention
    - Memory efficient: O(n) vs O(n^2)

Throughput (tokens/second):
    - TTT-Linear: ~90% of Mamba throughput
    - TTT-MLP: ~70% of Mamba throughput
    - Both faster than Transformer at long sequences

=============================================================================
IMPLEMENTATION REQUIREMENTS
=============================================================================

Dependencies:
    - PyTorch >= 2.0
    - Triton >= 2.0 (for custom kernels)
    - einops
    - flash-attn (optional, for hybrid models)

Core Components to Implement:
    1. TTTLinear: Linear hidden state layer
    2. TTT_MLP: MLP hidden state layer
    3. Mini-batch gradient computation
    4. Efficient memory management for weights
    5. Triton kernels for TTT operations

Hardware Considerations:
    - GPU strongly recommended (parallel gradient computation)
    - VRAM scales with d^2 (hidden state size)
    - Benefits from tensor cores (matmul-heavy)

Key Optimization: Parallel Gradient Computation
    - Compute gradients for mini-batch simultaneously
    - Use associative scan for cumulative weight updates
    - Custom Triton kernels for fused operations

=============================================================================
COMPARISON WITH OTHER ARCHITECTURES
=============================================================================

                    TTT         Transformer     Mamba       RWKV
Context Length      Unlimited   Limited*        Unlimited   Unlimited
Complexity          O(n)        O(n^2)          O(n)        O(n)
Parallel Training   Yes         Yes             Yes         Yes
Parallel Inference  Mini-batch  Chunked         Sequential  Sequential
Expressivity        High        Highest         Medium      Medium
Memory (inference)  O(d^2)      O(n·d)          O(d)        O(d)

*With attention modifications like RoPE, context can extend but quality degrades

=============================================================================
DEVELOPMENT ROADMAP
=============================================================================

Phase 1: Core Implementation
    - [ ] TTTLinear layer
    - [ ] TTT_MLP layer
    - [ ] Basic forward pass with online learning
    - [ ] Gradient computation infrastructure

Phase 2: Optimization
    - [ ] Mini-batch TTT implementation
    - [ ] Triton kernels for efficient gradient computation
    - [ ] Memory-efficient weight accumulation
    - [ ] Associative scan for parallel updates

Phase 3: Model Integration
    - [ ] TTT language model (GPT-style)
    - [ ] Hybrid TTT-Attention models
    - [ ] Pre-trained model loading (from official checkpoints)

Phase 4: Training Infrastructure
    - [ ] Training loop with curriculum learning
    - [ ] Long-context training strategies
    - [ ] Evaluation on standard benchmarks

Phase 5: CLI Tools
    - [ ] ttt-train: Training script
    - [ ] ttt-infer: Inference with streaming output
    - [ ] ttt-benchmark: Speed and quality benchmarks
    - [ ] ttt-analyze: Hidden state visualization

=============================================================================
CONNECTIONS TO CONSCIOUSNESS RESEARCH
=============================================================================

Relevance to Working Memory:
    - TTT hidden states = adaptive working memory
    - Weights update in real-time = memory consolidation
    - Self-supervised objective = predictive processing

Potential for Metacognition:
    - Hidden state can be inspected mid-inference
    - Learning dynamics reveal "thinking process"
    - Could implement "thinking about thinking"

Integration Points:
    - TTT as memory substrate for cognitive architectures
    - Combine with CTM for adaptive temporal processing
    - Use TTT hidden state as JEPA-style world model
"""

__version__ = "0.1.0"
__status__ = "indexed"

# Architecture metadata
ARCHITECTURE_INFO = {
    "name": "Test-Time Training",
    "abbreviation": "TTT",
    "year": 2024,
    "organization": "Stanford University / NVIDIA",
    "paper_url": "https://arxiv.org/abs/2407.04620",
    "github_url": "https://github.com/test-time-training/ttt-lm-pytorch",
    "authors": [
        "Yu Sun", "Xinhao Li", "Karan Dalal", "Jiarui Xu",
        "Arjun Vikram", "Genghan Zhang", "Yann Dubois",
        "Xinlei Chen", "Xiaolong Wang", "Sanmi Koyejo",
        "Tatsunori Hashimoto", "Carlos Guestrin"
    ],
    "key_innovation": "Learnable hidden states updated via gradient descent at test time",
}

# TTT Variants
VARIANTS = {
    "ttt_linear": {
        "name": "TTT-Linear",
        "hidden_state": "Linear layer (W in R^{d x d})",
        "complexity": "O(d^2) per token",
        "speed": "Fast",
        "expressivity": "Good baseline",
    },
    "ttt_mlp": {
        "name": "TTT-MLP",
        "hidden_state": "Two-layer MLP",
        "complexity": "O(d * d_hidden) per token",
        "speed": "Moderate",
        "expressivity": "Higher",
    },
}

# Mathematical formulation
FORMULATION = """
Self-Supervised Objective:
    L(x; W) = ||f(xW_K; W) - xW_V||^2

    where:
        x = input token embedding (1 x d)
        W_K = key projection (d x d_k), fixed after training
        W_V = value projection (d x d_v), fixed after training
        W = hidden state weights, updated at test time
        f(.; W) = TTT model (linear: f(z; W) = zW, or MLP)

Hidden State Update (Online Learning):
    W_{t+1} = W_t - eta * nabla_W L(x_t; W_t)

    Gradient for TTT-Linear:
        nabla_W L = (f(x_t W_K; W_t) - x_t W_V)^T * (x_t W_K)
                  = (x_t W_K W_t - x_t W_V)^T * (x_t W_K)

Output Computation:
    z_t = f(x_t W_K; W_t) + x_t     # with residual
    output_t = z_t W_O              # output projection

Mini-Batch TTT:
    For batch [x_1, ..., x_B]:
        Cumulative W_t = W_0 - eta * sum_{i<t} nabla_W L(x_i; W_{i-1})

    Computed efficiently via associative scan
"""

# Default configurations
TTT_LINEAR_CONFIG = {
    "hidden_dim": 768,
    "num_layers": 12,
    "num_heads": 12,  # for multi-head TTT
    "learning_rate": 1.0,  # TTT learning rate (often 1.0 works well)
    "mini_batch_size": 16,
    "rope_base": 10000,
    "rope_scaling": None,
    "layer_norm_eps": 1e-5,
    "initializer_range": 0.02,
}

TTT_MLP_CONFIG = {
    "hidden_dim": 768,
    "mlp_hidden_dim": 2048,  # hidden dimension in TTT-MLP
    "num_layers": 12,
    "num_heads": 12,
    "learning_rate": 1.0,
    "mini_batch_size": 16,
    "activation": "gelu",
    "rope_base": 10000,
    "layer_norm_eps": 1e-5,
}

# Benchmark results
BENCHMARKS = {
    "pile_validation": {
        "mamba_1.4B": 8.85,
        "ttt_linear_1.3B": 8.71,
        "ttt_mlp_1.3B": 8.52,
        "transformer_1.3B": 8.44,
    },
    "books": {
        "mamba_1.4B": 12.20,
        "ttt_linear_1.3B": 11.23,
        "ttt_mlp_1.3B": 10.87,
        "transformer_1.3B": 10.71,
    },
    "long_context_capability": "128K+ tokens with maintained quality",
    "throughput_vs_transformer": "Faster at long sequences (> 4K tokens)",
}

# Placeholder imports - will be implemented
# from .src.ttt_linear import TTTLinear, TTTLinearConfig
# from .src.ttt_mlp import TTTMLP, TTTMLPConfig
# from .src.model import TTTLanguageModel
# from .cli.train import main as train
# from .cli.infer import main as infer
