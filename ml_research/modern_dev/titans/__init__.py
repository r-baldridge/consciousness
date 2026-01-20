"""
Titans: Learning to Memorize at Test Time - Google, 2025

A neural architecture that introduces learnable memory modules capable of
memorizing at test time. Unlike traditional attention which stores context
in fixed KV caches, Titans learns HOW to memorize, enabling dynamic and
adaptive long-term memory.

Paper: "Titans: Learning to Memorize at Test Time"
arXiv: https://arxiv.org/abs/2501.00663
Authors: Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)

=============================================================================
KEY INNOVATIONS
=============================================================================

1. TEST-TIME LEARNING FOR MEMORY
   - Memory module updates its parameters during inference
   - Learns what to store based on incoming data
   - Surprise-based gating for memory writes
   - No static KV cache - dynamic learned memory

2. NEURAL LONG-TERM MEMORY MODULE
   - Separate memory network (small MLP)
   - Updated via gradient descent at test time
   - Stores compressed representations of past context
   - Read via forward pass, write via gradient update

3. SURPRISE-BASED MEMORY GATING
   - "Surprise" = prediction error on current input
   - High surprise triggers memory write
   - Low surprise uses cached representation
   - Efficient - only updates when necessary

=============================================================================
ARCHITECTURE
=============================================================================

Traditional Transformer:
    Input -> [Attention (KV cache)] -> Output
             |
             KV cache grows linearly with context
             Fixed storage mechanism

Titans Architecture:
    Input -> [Core Module + Memory Module] -> Output
             |
             Core: Short-term via attention
             Memory: Long-term via learnable module

    Memory Module:
        ┌─────────────────────────────────────┐
        │  Memory Network M (small MLP)       │
        │                                     │
        │  Read:  m_t = M(query)              │
        │  Write: M = M - lr * grad(loss)     │
        │                                     │
        │  Loss = surprise(x_t, prediction)   │
        └─────────────────────────────────────┘

    Three Variants:
    1. MAC (Memory as Context): Memory read added to attention
    2. MAG (Memory as Gate): Memory gates attention output
    3. MAL (Memory as Layer): Memory as separate processing layer

=============================================================================
MEMORY MODULE DETAILS
=============================================================================

Architecture of Memory Network:
    - 1-3 layer MLP
    - Input: query (current hidden state)
    - Output: retrieved memory representation
    - Small relative to main model (0.1-1% params)

Test-Time Learning Process:

    For each token x_t:
        1. Query memory: m_t = Memory(h_t)
        2. Compute surprise: s_t = ||x_t - prediction||^2
        3. If s_t > threshold:
            # Update memory with gradient descent
            grad = nabla_Memory(s_t)
            Memory = Memory - lr * grad
        4. Use m_t in attention/output

    Key Properties:
        - Memory learns associations online
        - Recent surprising events stored
        - Gradual forgetting of old info
        - No explicit memory capacity limit

=============================================================================
SURPRISE-BASED GATING
=============================================================================

Surprise Signal:
    s_t = ||f(h_t) - embed(x_{t+1})||^2

    where:
        h_t = hidden state at time t
        f = linear predictor
        embed = token embedding

Gate Application:
    gate_t = sigmoid(linear(s_t))

    If gate_t high (surprised):
        - Write to memory more strongly
        - Pay more attention to current input

    If gate_t low (expected):
        - Rely more on cached memory
        - Fast-path processing

Benefits:
    - Computational savings on predictable sequences
    - Focus resources on novel/important info
    - Natural curriculum: easy parts processed fast

=============================================================================
THREE ARCHITECTURAL VARIANTS
=============================================================================

1. MAC (Memory as Context):
   ─────────────────────────
   Memory read concatenated with attention context

   attn_out = Attention(Q, [K; K_mem], [V; V_mem])

   Best for: Tasks requiring explicit memory retrieval

2. MAG (Memory as Gate):
   ─────────────────────────
   Memory modulates attention output

   gate = sigmoid(Memory(h))
   out = gate * attn_out + (1 - gate) * memory_out

   Best for: Adaptive blending of short/long-term

3. MAL (Memory as Layer):
   ─────────────────────────
   Memory as separate processing step

   out = FFN(Attention(x) + Memory(x))

   Best for: Deep memory integration

=============================================================================
COMPARISON WITH RELATED APPROACHES
=============================================================================

                Titans      TTT-Linear    Mamba       Attention
Memory Type     Learned     Learned       Fixed SSM   KV Cache
Test-Time       Gradient    Gradient      None        None
  Learning      descent     descent
Context         Unlimited   Unlimited     Unlimited   Limited
Complexity      O(n)        O(n)          O(n)        O(n^2)
Interpretable   Yes         Partial       No          Yes

Key Differences from TTT:
    - TTT: Hidden state IS the model being trained
    - Titans: Separate memory module trained alongside main model
    - Titans: Surprise-based gating for efficiency
    - Titans: Multiple integration variants (MAC/MAG/MAL)

=============================================================================
PERFORMANCE BENCHMARKS
=============================================================================

Language Modeling (from paper):
    - Competitive with Transformers at standard contexts
    - Superior at very long contexts (100K+)
    - Memory module adds minimal overhead

Needle-in-Haystack:
    - Near-perfect retrieval at 2M+ context
    - Attention degrades after 128K
    - Mamba degrades after 64K

Efficiency:
    - 1.5-2x faster than full attention at long contexts
    - Memory module: <5% compute overhead
    - Constant memory usage regardless of history

=============================================================================
IMPLEMENTATION REQUIREMENTS
=============================================================================

Dependencies:
    - PyTorch >= 2.0
    - einops
    - Optional: Flash Attention for core attention

Hardware:
    - Single GPU sufficient (no distributed requirement)
    - Memory: Standard transformer requirements
    - Compute: Slight overhead for memory gradients

Key Components to Implement:
    1. MemoryModule (learnable MLP)
    2. SurpriseComputer (prediction error)
    3. MemoryGate (write decision)
    4. TestTimeLearner (gradient updates)
    5. TitansLayer (integrated attention + memory)
    6. TitansModel (full architecture)

=============================================================================
TRAINING CONSIDERATIONS
=============================================================================

Two-Stage Training:
    Stage 1: Train core model (attention + FFN)
    Stage 2: Train memory module integration

Memory Module Initialization:
    - Random init for memory network
    - Pre-train on reconstruction task
    - Fine-tune with full model

Meta-Learning Aspect:
    - Model learns HOW to learn at test time
    - Training teaches memory module what to store
    - Test-time learning applies learned strategy

Loss Function:
    L = L_main + lambda * L_memory

    where L_memory encourages:
        - Good memory retrieval
        - Appropriate surprise detection
        - Stable test-time updates

=============================================================================
DEVELOPMENT ROADMAP
=============================================================================

Phase 1: Core Memory Module
    - [ ] Implement MemoryModule MLP
    - [ ] Implement surprise computation
    - [ ] Test-time gradient updates
    - [ ] Basic forward/backward pass

Phase 2: Integration Variants
    - [ ] MAC (Memory as Context)
    - [ ] MAG (Memory as Gate)
    - [ ] MAL (Memory as Layer)
    - [ ] Variant comparison benchmarks

Phase 3: Training Infrastructure
    - [ ] Two-stage training pipeline
    - [ ] Memory initialization strategies
    - [ ] Hyperparameter tuning
    - [ ] Long-context data loading

Phase 4: CLI Tools
    - [ ] titans-train: Training script
    - [ ] titans-infer: Long-context inference
    - [ ] titans-analyze: Memory visualization
    - [ ] titans-bench: Performance benchmarks
"""

__version__ = "0.1.0"
__status__ = "indexed"

# Architecture metadata
ARCHITECTURE_INFO = {
    "name": "Titans",
    "abbreviation": "Titans",
    "year": 2025,
    "organization": "Google Research",
    "paper_url": "https://arxiv.org/abs/2501.00663",
    "github_url": None,  # No official implementation yet
    "authors": ["Ali Behrouz", "Peilin Zhong", "Vahab Mirrokni"],
    "key_contribution": "Test-time learning for adaptive memory with surprise-based gating",
}

# Mathematical formulation
FORMULATION = """
Titans Memory System:

Memory Module M with parameters theta_M:
    m_t = M_theta(h_t)  # Memory read (forward pass)

Surprise Signal:
    s_t = ||f(h_t) - embed(x_{t+1})||^2

    where f is a learned linear predictor

Test-Time Memory Update:
    If s_t > tau (surprise threshold):
        L_memory = s_t  # or reconstruction loss
        theta_M = theta_M - alpha * nabla_{theta_M} L_memory

    This is ONE gradient step, not full training

Memory Gating (MAG variant):
    g_t = sigmoid(W_g * [h_t; m_t] + b_g)
    o_t = g_t * attention_out + (1 - g_t) * m_t

Full Titans Layer:
    h = LayerNorm(x)
    a = Attention(h)  # Short-term
    m = Memory(h)     # Long-term (with test-time update)
    o = Gate(a, m)    # Combine
    out = x + FFN(o)

Key Property:
    - Training: Learn theta_M initialization and update rule
    - Inference: theta_M adapts to specific context via gradient
    - Memory capacity grows with context (stored in weights)
"""

# Default configuration
DEFAULT_CONFIG = {
    "hidden_dim": 2048,
    "num_layers": 24,
    "num_heads": 16,
    "head_dim": 128,

    # Memory module config
    "memory_dim": 256,
    "memory_layers": 2,
    "memory_lr": 0.01,  # Learning rate for test-time updates
    "surprise_threshold": 0.1,

    # Integration variant
    "variant": "MAG",  # MAC, MAG, or MAL

    # Training config
    "base_lr": 1e-4,
    "memory_loss_weight": 0.1,
}

# Variant configurations
VARIANT_CONFIGS = {
    "MAC": {
        "description": "Memory as Context - concatenate memory with attention KV",
        "memory_key_dim": 64,
        "memory_value_dim": 64,
        "num_memory_slots": 64,
    },
    "MAG": {
        "description": "Memory as Gate - memory modulates attention output",
        "gate_activation": "sigmoid",
        "gate_init_bias": -2.0,  # Start with attention-dominant
    },
    "MAL": {
        "description": "Memory as Layer - separate memory processing",
        "memory_ffn_dim": 512,
        "residual_memory": True,
    },
}

# Comparison with related architectures
COMPARISON = {
    "vs_attention": {
        "advantage": "Unlimited context with O(n) complexity",
        "disadvantage": "Additional memory module overhead",
    },
    "vs_ttt": {
        "advantage": "Separate memory module, surprise-based efficiency",
        "disadvantage": "Slightly more complex architecture",
    },
    "vs_mamba": {
        "advantage": "Adaptive memory via learning, more interpretable",
        "disadvantage": "Test-time gradient computation",
    },
    "vs_rwkv": {
        "advantage": "Learned memory strategy, surprise gating",
        "disadvantage": "Not fully recurrent, requires backprop",
    },
}

# Placeholder imports
# from .src.memory_module import MemoryModule
# from .src.surprise import SurpriseComputer
# from .src.titans_layer import TitansLayer, MAC, MAG, MAL
# from .src.model import TitansModel
# from .cli.train import main as train
# from .cli.infer import main as infer
