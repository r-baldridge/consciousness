"""
Mamba - Linear-Time Sequence Modeling with Selective State Spaces (2023)

Research index entry for Mamba, which introduces selective state space models
(SSMs) as an alternative to attention with linear scaling and fast inference.

Key contributions:
- Selective state space mechanism (input-dependent dynamics)
- Hardware-aware parallel scan algorithm
- Linear-time training, constant-time inference per token
- Competitive with Transformers at scale
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Mamba."""
    return MLMethod(
        method_id="mamba_2023",
        name="Mamba (Selective State Space Model)",
        year=2023,
        era=MethodEra.NOVEL,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.RNN_LINE],
        authors=[
            "Albert Gu",
            "Tri Dao",
        ],
        paper_title="Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
        paper_url="https://arxiv.org/abs/2312.00752",
        key_innovation="""
        Mamba introduces a selective state space model that combines the
        efficiency of RNNs (linear complexity) with the expressiveness of
        Transformers (content-based reasoning).

        Key insight: Standard SSMs use fixed, input-independent parameters.
        This limits their ability to perform content-based reasoning. Mamba
        makes SSM parameters (A, B, C) input-dependent, enabling selection
        of relevant information from context.

        Innovations:

        1. Selective Mechanism: Parameters B, C, and Delta (discretization step)
           are functions of the input, allowing the model to selectively
           remember or forget information based on content.

        2. Hardware-Aware Algorithm: Custom CUDA kernels with parallel scan
           for efficient training. Avoids materializing large state matrices.

        3. Simplified Architecture: No attention, no MLP - just stacked
           Mamba blocks with linear projections and convolutions.

        4. Linear Scaling: O(N) training complexity, O(1) inference per token
           (compared to O(N^2) for attention, O(N) for attention inference).

        Mamba matches or exceeds Transformer performance at various scales
        while being 5x faster at inference for long sequences.
        """,
        mathematical_formulation="""
        CONTINUOUS-TIME STATE SPACE MODEL
        ==================================
        Standard linear SSM:
            h'(t) = A h(t) + B x(t)     (state dynamics)
            y(t) = C h(t)                (output)

        Where:
            x(t) in R^D: input
            h(t) in R^N: hidden state
            y(t) in R^D: output
            A in R^(N x N): state matrix
            B in R^(N x D): input matrix
            C in R^(D x N): output matrix

        DISCRETIZATION
        ==============
        Convert continuous SSM to discrete for sequence processing:

        Using zero-order hold with step size Delta:
            A_bar = exp(Delta * A)
            B_bar = (Delta * A)^(-1) * (exp(Delta * A) - I) * Delta * B
                  ≈ Delta * B  (first-order approximation)

        Discrete recurrence:
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C * h_t

        SELECTIVE STATE SPACE (MAMBA)
        ==============================
        Make parameters input-dependent:

        For input x in R^(B x L x D):
            B = Linear_B(x)         # R^(B x L x N)
            C = Linear_C(x)         # R^(B x L x N)
            Delta = softplus(Linear_Delta(x))  # R^(B x L x D)

        A remains fixed (learned, structured as diagonal)

        Discretization per position:
            A_bar_t = exp(Delta_t * A)  # Input-dependent!
            B_bar_t = Delta_t * B_t

        Recurrence:
            h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            y_t = C_t * h_t

        PARALLEL SCAN FOR TRAINING
        ==========================
        The recurrence h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        can be computed in O(log L) parallel steps using associative scan.

        Define: (a_i, b_i) = (A_bar_i, B_bar_i * x_i)

        Associative operator:
            (a_1, b_1) * (a_2, b_2) = (a_1 * a_2, a_2 * b_1 + b_2)

        Parallel prefix sum computes all h_t in O(L log L) work, O(log L) depth

        MAMBA BLOCK
        ===========
        Input: x in R^(B x L x D)

        1. Linear expansion: x -> [z, x'] where z, x' in R^(B x L x E), E = expand * D

        2. Convolution: x' = Conv1D(x')  (short conv, kernel size 4)

        3. Activation: x' = SiLU(x')

        4. Selective SSM:
           - Project: B = Linear(x'), C = Linear(x'), Delta = Linear(x')
           - Discretize: A_bar, B_bar from Delta
           - Scan: y = SSM_scan(A_bar, B_bar, C, x')

        5. Gating: y = y * SiLU(z)

        6. Output projection: output = Linear(y)

        COMPLEXITY
        ==========
        Training:
            Standard attention: O(L^2 * D)
            Mamba: O(L * D * N) = O(L) when N is constant

        Inference (per token):
            Attention: O(L * D) - must attend to all previous tokens
            Mamba: O(D * N) = O(1) - just update state

        Memory:
            Attention: O(L^2) or O(L) with Flash Attention
            Mamba: O(L * D * N) training, O(D * N) inference
        """,
        predecessors=["s4_2021", "h3_2022", "flash_attention_2022"],
        successors=["mamba2_2024", "jamba_2024"],
        tags=[
            "state_space_model",
            "selective_ssm",
            "linear_attention",
            "efficient_sequence_model",
            "recurrent",
            "hardware_aware",
        ],
        notes="""
        Mamba represents a paradigm shift, showing that attention may not be
        necessary for strong language modeling. The selective mechanism
        addresses the key limitation of prior SSMs (fixed dynamics) while
        maintaining linear complexity.

        Key observations:
        - Selection is crucial: fixed SSMs cannot match Transformers
        - Hardware matters: naive selective SSM is slow; parallel scan +
          kernel fusion makes it practical
        - Scales well: Mamba-3B matches Transformer-3B quality

        Limitations:
        - Cannot do arbitrary pairwise comparisons (no N^2 computation)
        - May struggle on tasks requiring precise copying/recall
        - Less interpretable than attention patterns

        Follow-ups:
        - Mamba-2 (2024): Improved architecture, better scaling
        - Jamba (2024): Hybrid Mamba-Transformer (AI21)
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for Mamba."""
    return """
    MAMBA ARCHITECTURE
    ==================

    class MambaBlock:
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            '''
            d_model: model dimension
            d_state: SSM state dimension (N)
            d_conv: convolution kernel size
            expand: expansion factor for inner dimension
            '''
            self.d_inner = expand * d_model

            # Input projection (expands dimension)
            self.in_proj = Linear(d_model, 2 * self.d_inner)  # for x and z

            # Short convolution
            self.conv1d = Conv1D(self.d_inner, kernel_size=d_conv, groups=self.d_inner)

            # SSM parameters
            self.A_log = Parameter(log_A_init(d_state, self.d_inner))  # Log of A (for stability)

            # Input-dependent projections
            self.x_proj = Linear(self.d_inner, d_state * 2 + 1)  # B, C, Delta
            self.dt_proj = Linear(1, self.d_inner)

            # Output projection
            self.out_proj = Linear(self.d_inner, d_model)

        def forward(self, x):
            '''
            x: [batch, seq_len, d_model]
            '''
            batch, seq_len, _ = x.shape

            # 1. Input projection and split
            xz = self.in_proj(x)  # [batch, seq_len, 2 * d_inner]
            x, z = xz.chunk(2, dim=-1)  # Each [batch, seq_len, d_inner]

            # 2. Convolution
            x = self.conv1d(x)  # [batch, seq_len, d_inner]
            x = silu(x)

            # 3. Selective SSM
            y = self.selective_ssm(x)

            # 4. Gating
            y = y * silu(z)

            # 5. Output projection
            return self.out_proj(y)

        def selective_ssm(self, x):
            '''
            Selective state space model with input-dependent parameters
            '''
            batch, seq_len, d_inner = x.shape

            # Get A (fixed, from learned log)
            A = -exp(self.A_log)  # [d_inner, d_state], negative for stability

            # Project x to get B, C, Delta
            x_dbl = self.x_proj(x)  # [batch, seq_len, d_state*2 + 1]
            B, C, dt = x_dbl.split([d_state, d_state, 1], dim=-1)

            # Delta (discretization step size)
            dt = softplus(self.dt_proj(dt))  # [batch, seq_len, d_inner]

            # Discretize A and B
            # A_bar = exp(dt * A)
            # B_bar = dt * B
            A_bar = exp(einsum('bld,dn->bldn', dt, A))  # [batch, seq_len, d_inner, d_state]
            B_bar = einsum('bld,bln->bldn', dt, B)       # [batch, seq_len, d_inner, d_state]

            # Run SSM (parallel scan or sequential)
            y = ssm_scan(A_bar, B_bar, C, x)  # [batch, seq_len, d_inner]

            return y


    PARALLEL SCAN (ASSOCIATIVE SCAN)
    =================================

    def ssm_scan(A_bar, B_bar, C, x):
        '''
        Parallel scan for SSM computation

        Recurrence: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        Output: y_t = C_t * h_t

        Uses associative scan for O(log L) parallel depth
        '''
        batch, seq_len, d_inner, d_state = A_bar.shape

        # Prepare scan inputs
        # Each element: (A_bar_t, B_bar_t * x_t)
        Bx = einsum('bldn,bld->bldn', B_bar, x)  # [batch, seq_len, d_inner, d_state]

        # Associative operator: (a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)
        def associative_op(elem1, elem2):
            a1, b1 = elem1
            a2, b2 = elem2
            return (a1 * a2, a2 * b1 + b2)

        # Parallel prefix scan
        _, h = parallel_scan(associative_op, (A_bar, Bx))
        # h: [batch, seq_len, d_inner, d_state]

        # Compute output
        y = einsum('bln,bldn->bld', C, h)  # [batch, seq_len, d_inner]

        return y


    SEQUENTIAL SCAN (FOR INFERENCE)
    ================================

    def ssm_scan_sequential(A_bar, B_bar, C, x, h_prev=None):
        '''
        Sequential SSM for autoregressive inference
        O(1) per token (just update state)
        '''
        batch, _, d_inner, d_state = A_bar.shape

        if h_prev is None:
            h = zeros(batch, d_inner, d_state)
        else:
            h = h_prev

        # Single step update
        # h = A_bar * h + B_bar * x
        h = A_bar[:, 0] * h + B_bar[:, 0] * x[:, 0].unsqueeze(-1)

        # Output
        y = einsum('bn,bdn->bd', C[:, 0], h)

        return y, h  # Return state for next step


    FULL MAMBA MODEL
    =================

    class Mamba:
        def __init__(self, d_model, n_layers, vocab_size):
            self.embedding = Embedding(vocab_size, d_model)
            self.layers = ModuleList([
                MambaBlock(d_model) for _ in range(n_layers)
            ])
            self.norm = RMSNorm(d_model)
            self.lm_head = Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)

            for layer in self.layers:
                x = x + layer(x)  # Residual connection

            x = self.norm(x)
            logits = self.lm_head(x)
            return logits

        def generate(self, input_ids, max_length):
            '''
            Efficient autoregressive generation
            Maintains state for O(1) per-token cost
            '''
            # Process prefix
            x = self.embedding(input_ids)
            states = [None] * len(self.layers)

            for i, layer in enumerate(self.layers):
                x, states[i] = layer.forward_with_state(x)

            # Generate tokens
            for _ in range(max_length):
                logits = self.lm_head(self.norm(x[:, -1:]))
                next_token = sample(logits)

                # Process new token with cached states (O(1))
                x = self.embedding(next_token)
                for i, layer in enumerate(self.layers):
                    x, states[i] = layer.step(x, states[i])

                yield next_token
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for Mamba."""
    return {
        # Continuous SSM
        "state_dynamics": "h'(t) = A h(t) + B x(t)",
        "output": "y(t) = C h(t)",

        # Discretization
        "discretize_A": "A_bar = exp(Delta * A)",
        "discretize_B": "B_bar ≈ Delta * B",
        "discrete_recurrence": "h_t = A_bar h_{t-1} + B_bar x_t",

        # Selective mechanism
        "selective_B": "B = Linear_B(x)",
        "selective_C": "C = Linear_C(x)",
        "selective_delta": "Delta = softplus(Linear_Delta(x))",

        # Complexity
        "training_complexity": "O(L * D * N) = O(L)",
        "inference_per_token": "O(D * N) = O(1)",
        "attention_comparison": "Attention: O(L^2) train, O(L) inference per token",

        # Parallel scan
        "associative_op": "(a1, b1) * (a2, b2) = (a1*a2, a2*b1 + b2)",
        "scan_depth": "O(log L) parallel depth",
    }


def get_model_variants() -> List[Dict[str, str]]:
    """Return Mamba model variants."""
    return [
        {
            "name": "Mamba-130M",
            "d_model": 768,
            "n_layers": 24,
            "params": "130M",
            "notes": "Matches Transformer-130M",
        },
        {
            "name": "Mamba-370M",
            "d_model": 1024,
            "n_layers": 48,
            "params": "370M",
            "notes": "Matches Transformer-370M",
        },
        {
            "name": "Mamba-790M",
            "d_model": 1536,
            "n_layers": 48,
            "params": "790M",
            "notes": "Matches Transformer-790M",
        },
        {
            "name": "Mamba-1.4B",
            "d_model": 2048,
            "n_layers": 48,
            "params": "1.4B",
            "notes": "Matches Transformer-1.4B",
        },
        {
            "name": "Mamba-2.8B",
            "d_model": 2560,
            "n_layers": 64,
            "params": "2.8B",
            "notes": "Matches Transformer-2.8B",
        },
    ]


def compare_with_attention() -> Dict[str, Dict[str, str]]:
    """Compare Mamba with attention-based models."""
    return {
        "Training Speed": {
            "Transformer": "O(L^2) attention, parallelizable",
            "Mamba": "O(L) selective SSM, parallelizable via scan",
            "Winner": "Mamba (linear vs quadratic)",
        },
        "Inference Speed": {
            "Transformer": "O(L * D) per token (KV cache)",
            "Mamba": "O(D * N) per token (state update)",
            "Winner": "Mamba (~5x faster for long sequences)",
        },
        "Memory (Training)": {
            "Transformer": "O(L^2) or O(L) with Flash Attention",
            "Mamba": "O(L * N * D)",
            "Winner": "Comparable with Flash Attention",
        },
        "Memory (Inference)": {
            "Transformer": "O(L * D) KV cache grows with context",
            "Mamba": "O(N * D) fixed state size",
            "Winner": "Mamba (constant vs linear)",
        },
        "Quality": {
            "Transformer": "Strong on all tasks",
            "Mamba": "Matches on most, may lag on precise recall",
            "Winner": "Comparable at scale",
        },
        "Interpretability": {
            "Transformer": "Attention patterns are interpretable",
            "Mamba": "State evolution less interpretable",
            "Winner": "Transformer",
        },
    }


def get_hardware_optimizations() -> Dict[str, str]:
    """Return hardware optimizations in Mamba."""
    return {
        "kernel_fusion": "Fuse discretization, scan, and output in one kernel",
        "recomputation": "Recompute states in backward (like Flash Attention)",
        "parallel_scan": "Use GPU-efficient parallel prefix sum",
        "memory_layout": "Optimize tensor layouts for coalesced memory access",
        "warp_primitives": "Use warp-level primitives for scan operations",
        "no_materialization": "Never materialize O(L * N * D) intermediate states",
    }
