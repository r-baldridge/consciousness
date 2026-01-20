"""
Linear Attention Methods - Linformer and Performer (2020)

Research index entries for linear attention mechanisms that approximate
full attention with O(n) complexity using low-rank or kernel approximations.

Methods included:
- Linformer: Low-rank projection of K, V (Wang et al., 2020)
- Performer: FAVOR+ random feature approximation (Choromanski et al., 2020)
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_linformer_info() -> MLMethod:
    """Return the MLMethod entry for Linformer."""
    return MLMethod(
        method_id="linformer_2020",
        name="Linformer",
        year=2020,
        era=MethodEra.ATTENTION,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Sinong Wang",
            "Belinda Z. Li",
            "Madian Khabsa",
            "Han Fang",
            "Hao Ma",
        ],
        paper_title="Linformer: Self-Attention with Linear Complexity",
        paper_url="https://arxiv.org/abs/2006.04768",
        key_innovation="""
        Linformer approximates self-attention with linear complexity by
        projecting the key and value matrices to a lower dimension k << n
        before computing attention.

        Key insight: Self-attention matrices are often low-rank. The attention
        matrix A = softmax(QK^T) can be well-approximated by a low-rank matrix.
        Instead of computing full n x n attention, project K and V to k
        dimensions, resulting in n x k attention.

        Innovations:
        1. Low-rank projection: K' = E_K @ K, V' = E_V @ V
           where E_K, E_V are n -> k projection matrices

        2. Theoretical justification: With k = O(d/epsilon^2), the
           approximation error is bounded by epsilon

        3. Parameter sharing: Share projection matrices across layers
           and heads to reduce parameters
        """,
        mathematical_formulation="""
        STANDARD ATTENTION
        ==================
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

        Q, K, V in R^(n x d)
        QK^T in R^(n x n)  <- bottleneck!
        Complexity: O(n^2 * d)

        LINFORMER
        =========
        Project K and V to lower dimension k:

        E_K in R^(k x n)  (learned projection for keys)
        E_V in R^(k x n)  (learned projection for values)

        K' = E_K @ K  in R^(k x d)
        V' = E_V @ V  in R^(k x d)

        LinformerAttn(Q, K, V) = softmax(Q @ K'^T / sqrt(d)) @ V'

        Dimensions:
        Q @ K'^T: [n x d] @ [d x k] = [n x k]
        softmax(.) @ V': [n x k] @ [k x d] = [n x d]

        Complexity: O(n * k * d) = O(n) when k is constant

        THEORETICAL GUARANTEE
        =====================
        Theorem: For attention matrix A and any row A_i:

        If k = O(d * log(d) / epsilon^2), then with high probability:
            ||A_i - A'_i|| <= epsilon * ||A_i||

        where A' is the Linformer approximation.

        In practice, k = 256 works well for n = 512 to 4096.

        PROJECTION OPTIONS
        ==================
        1. Learnable projections (E_K, E_V trainable)
        2. Fixed random projections (Johnson-Lindenstrauss)
        3. Shared across layers (reduce parameters)
        4. Shared across heads (further parameter reduction)
        5. Different projections per head (most expressive)

        MULTI-HEAD LINFORMER
        ====================
        For h heads with projections shared across heads:

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

        head_i = LinformerAttn(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)

        Using same E_K, E_V for all heads saves parameters.
        """,
        predecessors=["transformer_2017", "bert_2018"],
        successors=["performer_2020", "linear_transformer_2020"],
        tags=[
            "efficient_attention",
            "linear_attention",
            "low_rank",
            "projection",
            "approximation",
        ],
        notes="""
        Linformer demonstrates that self-attention matrices are inherently
        low-rank, enabling effective linear-complexity approximations.
        The method works well for encoder models but faces challenges for
        autoregressive generation due to the global projection.

        Strengths: Simple implementation, strong empirical results, theoretical
        guarantees on approximation quality.

        Limitations: Fixed k may not adapt to varying sequence lengths;
        projection is global (not causal), making autoregressive use complex.
        """,
    )


def get_performer_info() -> MLMethod:
    """Return the MLMethod entry for Performer."""
    return MLMethod(
        method_id="performer_2020",
        name="Performer (FAVOR+)",
        year=2020,
        era=MethodEra.ATTENTION,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Krzysztof Choromanski",
            "Valerii Likhosherstov",
            "David Dohan",
            "Xingyou Song",
            "Andreea Gane",
            "Tamas Sarlos",
            "Peter Hawkins",
            "Jared Davis",
            "Afroz Mohiuddin",
            "Lukasz Kaiser",
            "David Belanger",
            "Lucy Colwell",
            "Adrian Weller",
        ],
        paper_title="Rethinking Attention with Performers",
        paper_url="https://arxiv.org/abs/2009.14794",
        key_innovation="""
        Performer introduces FAVOR+ (Fast Attention Via positive Orthogonal
        Random features), a method to approximate softmax attention using
        random feature maps that enable linear-time computation.

        Key insight: Softmax attention can be decomposed using kernel methods.
        By approximating the softmax kernel with random features, attention
        can be computed without explicitly forming the n x n attention matrix.

        FAVOR+ innovations:
        1. Positive random features: Ensure non-negative attention weights
        2. Orthogonal random features: Reduce variance of approximation
        3. Causal compatibility: Supports autoregressive generation
        4. Kernel approximation: exp(q^T k) approx phi(q)^T phi(k)
        """,
        mathematical_formulation="""
        KERNEL VIEW OF ATTENTION
        ========================
        Softmax attention can be written as:

        Attn(Q, K, V)_i = sum_j K(q_i, k_j) * v_j / sum_j K(q_i, k_j)

        where K(q, k) = exp(q^T k / sqrt(d)) is the softmax kernel.

        RANDOM FEATURE APPROXIMATION
        ============================
        Approximate kernel K using random features:

        K(q, k) approx phi(q)^T phi(k)

        where phi: R^d -> R^m is a random feature map.

        For softmax kernel (FAVOR+):
        phi(x) = (1/sqrt(m)) * exp(-||x||^2/2) * [exp(w_1^T x), ..., exp(w_m^T x)]

        where w_1, ..., w_m are random vectors.

        POSITIVE ORTHOGONAL RANDOM FEATURES (FAVOR+)
        ============================================
        To ensure positive attention and reduce variance:

        1. Use positive features:
           phi+(x) = (1/sqrt(m)) * exp(-||x||^2/2) * [exp(w_1^T x), ..., exp(w_m^T x)]

           All components are positive -> attention weights non-negative

        2. Use orthogonal random features:
           Draw W = [w_1, ..., w_m] such that w_i^T w_j = 0 for i != j
           (via Gram-Schmidt or structured orthogonal matrices)

           Reduces variance compared to i.i.d. random features

        LINEAR ATTENTION COMPUTATION
        ============================
        Let phi(Q) in R^(n x m), phi(K) in R^(n x m)

        Standard softmax attention:
        Attn = softmax(QK^T / sqrt(d)) @ V  # O(n^2)

        Performer attention:
        Attn = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1_n)

        Computation order (right to left):
        1. phi(K)^T @ V: [m x n] @ [n x d] = [m x d]  O(n*m*d)
        2. phi(Q) @ (...): [n x m] @ [m x d] = [n x d]  O(n*m*d)
        3. Normalization: O(n*m)

        Total: O(n * m * d) = O(n) when m = O(d)

        CAUSAL (AUTOREGRESSIVE) ATTENTION
        ==================================
        For causal attention, maintain running sums:

        S_i = sum_{j<=i} phi(k_j) v_j^T  (cumulative KV)
        z_i = sum_{j<=i} phi(k_j)         (cumulative normalization)

        Attn_i = (phi(q_i)^T @ S_i) / (phi(q_i)^T @ z_i)

        Update incrementally: S_i = S_{i-1} + phi(k_i) v_i^T
        Enables O(n) autoregressive generation!
        """,
        predecessors=["transformer_2017", "random_features_2007"],
        successors=["linear_transformer_2020", "rfa_2021"],
        tags=[
            "efficient_attention",
            "linear_attention",
            "random_features",
            "kernel_methods",
            "causal_attention",
            "FAVOR+",
        ],
        notes="""
        Performer bridges kernel methods and attention, showing that softmax
        attention is a kernel smoother that can be efficiently approximated.
        The FAVOR+ mechanism enables both bidirectional and causal attention
        with linear complexity.

        Strengths: Theoretically principled, supports causal attention,
        good for very long sequences, enables protein modeling at scale.

        Limitations: Approximation quality depends on number of random
        features; may underperform exact attention on some tasks; requires
        careful hyperparameter tuning (number of features m).

        Applications: Protein sequence modeling (demonstrated on 280,000
        length sequences), long document processing.
        """,
    )


def pseudocode_linformer() -> str:
    """Return pseudocode for Linformer attention."""
    return """
    LINFORMER ATTENTION ALGORITHM
    =============================

    class LinformerAttention:
        def __init__(self, d_model, n_heads, seq_len, k=256):
            '''
            k: projected dimension (k << seq_len)
            '''
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.k = k

            # Standard attention projections
            self.W_Q = Linear(d_model, d_model)
            self.W_K = Linear(d_model, d_model)
            self.W_V = Linear(d_model, d_model)
            self.W_O = Linear(d_model, d_model)

            # Low-rank projection matrices (shared across heads)
            self.E_K = Linear(seq_len, k, bias=False)  # Project K
            self.E_V = Linear(seq_len, k, bias=False)  # Project V

        def forward(self, Q, K, V):
            batch_size, seq_len, _ = Q.shape

            # Project to Q, K, V
            Q = self.W_Q(Q).view(batch_size, seq_len, self.n_heads, self.d_k)
            K = self.W_K(K).view(batch_size, seq_len, self.n_heads, self.d_k)
            V = self.W_V(V).view(batch_size, seq_len, self.n_heads, self.d_k)

            # Transpose for attention: [batch, heads, seq, d_k]
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Low-rank projection of K and V
            # E projects along sequence dimension: [batch, heads, seq, d_k] -> [batch, heads, k, d_k]
            K_proj = self.E_K(K.transpose(-2, -1)).transpose(-2, -1)  # [batch, heads, k, d_k]
            V_proj = self.E_V(V.transpose(-2, -1)).transpose(-2, -1)  # [batch, heads, k, d_k]

            # Compute attention with projected K, V
            # [batch, heads, seq, d_k] @ [batch, heads, d_k, k] = [batch, heads, seq, k]
            scores = Q @ K_proj.transpose(-2, -1) / sqrt(self.d_k)
            attn_weights = softmax(scores, dim=-1)

            # [batch, heads, seq, k] @ [batch, heads, k, d_k] = [batch, heads, seq, d_k]
            output = attn_weights @ V_proj

            # Reshape and project output
            output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            return self.W_O(output)


    PARAMETER SHARING VARIANTS
    ==========================

    # Variant 1: Share projections across layers
    class LinformerSharedLayers:
        def __init__(self, num_layers, ...):
            self.E_K = Linear(seq_len, k)  # Shared
            self.E_V = Linear(seq_len, k)  # Shared
            self.layers = [LinformerLayer(self.E_K, self.E_V) for _ in range(num_layers)]

    # Variant 2: Share projections across heads and layers
    class LinformerMaxSharing:
        # Same E_K, E_V for all heads and all layers
        # Most parameter efficient

    # Variant 3: Headwise projections (most expressive)
    class LinformerHeadwise:
        def __init__(self, ...):
            self.E_K = nn.ModuleList([Linear(seq_len, k) for _ in range(n_heads)])
            self.E_V = nn.ModuleList([Linear(seq_len, k) for _ in range(n_heads)])
    """


def pseudocode_performer() -> str:
    """Return pseudocode for Performer attention."""
    return """
    PERFORMER ATTENTION ALGORITHM (FAVOR+)
    ======================================

    def random_feature_map(x, W, normalize=True):
        '''
        Compute FAVOR+ positive random features

        x: [batch, seq, d]
        W: [m, d] orthogonal random matrix
        '''
        # Project x using random features
        # x @ W^T: [batch, seq, m]
        proj = x @ W.T

        if normalize:
            # Normalize by ||x||^2 for numerical stability
            norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [batch, seq, 1]
            proj = proj - norm_sq / 2

        # Positive features: exp(w^T x)
        return exp(proj) / sqrt(m)


    def generate_orthogonal_random_features(d, m):
        '''
        Generate orthogonal random feature matrix
        '''
        # Start with random Gaussian matrix
        G = randn(m, d)

        # Apply Gram-Schmidt orthogonalization
        Q, R = qr(G)

        # Scale to preserve expected norms
        Q = Q * sqrt(d)

        return Q


    class PerformerAttention:
        def __init__(self, d_model, n_heads, num_features=256):
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.m = num_features  # Number of random features

            # Standard projections
            self.W_Q = Linear(d_model, d_model)
            self.W_K = Linear(d_model, d_model)
            self.W_V = Linear(d_model, d_model)
            self.W_O = Linear(d_model, d_model)

            # Orthogonal random features (fixed or resampled)
            self.W_rand = generate_orthogonal_random_features(self.d_k, self.m)

        def forward(self, Q, K, V, causal=False):
            batch_size, seq_len, _ = Q.shape

            # Project to Q, K, V
            Q = self.W_Q(Q).view(batch_size, seq_len, self.n_heads, self.d_k)
            K = self.W_K(K).view(batch_size, seq_len, self.n_heads, self.d_k)
            V = self.W_V(V).view(batch_size, seq_len, self.n_heads, self.d_k)

            # Apply random feature map: [batch, seq, heads, d_k] -> [batch, seq, heads, m]
            phi_Q = random_feature_map(Q, self.W_rand)  # [batch, seq, heads, m]
            phi_K = random_feature_map(K, self.W_rand)  # [batch, seq, heads, m]

            if causal:
                output = causal_linear_attention(phi_Q, phi_K, V)
            else:
                output = bidirectional_linear_attention(phi_Q, phi_K, V)

            output = output.reshape(batch_size, seq_len, self.d_model)
            return self.W_O(output)


    def bidirectional_linear_attention(phi_Q, phi_K, V):
        '''
        Non-causal linear attention
        phi_Q, phi_K: [batch, seq, heads, m]
        V: [batch, seq, heads, d_v]
        '''
        # Compute KV product: [batch, heads, m, d_v]
        # phi_K^T @ V: [batch, heads, m, seq] @ [batch, heads, seq, d_v]
        KV = einsum('bshm,bshd->bhmd', phi_K, V)

        # Compute normalization: sum of phi_K
        # [batch, heads, m]
        K_sum = phi_K.sum(dim=1).transpose(1, 2)

        # Compute output: phi_Q @ KV
        # [batch, seq, heads, m] @ [batch, heads, m, d_v] -> [batch, seq, heads, d_v]
        numerator = einsum('bshm,bhmd->bshd', phi_Q, KV)

        # Normalize
        denominator = einsum('bshm,bhm->bsh', phi_Q, K_sum).unsqueeze(-1)
        output = numerator / (denominator + eps)

        return output


    def causal_linear_attention(phi_Q, phi_K, V):
        '''
        Causal (autoregressive) linear attention using cumulative sums
        '''
        batch, seq, heads, m = phi_Q.shape
        d_v = V.shape[-1]

        output = zeros(batch, seq, heads, d_v)

        # Cumulative KV and K sums
        S = zeros(batch, heads, m, d_v)  # Cumulative KV
        z = zeros(batch, heads, m)        # Cumulative K sum

        for i in range(seq):
            # Update cumulative sums
            S = S + einsum('bhm,bhd->bhmd', phi_K[:, i], V[:, i])
            z = z + phi_K[:, i].transpose(1, 2)

            # Compute output for position i
            numerator = einsum('bhm,bhmd->bhd', phi_Q[:, i].transpose(1, 2), S)
            denominator = einsum('bhm,bhm->bh', phi_Q[:, i].transpose(1, 2), z)

            output[:, i] = numerator / (denominator.unsqueeze(-1) + eps)

        return output
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for linear attention methods."""
    return {
        # Standard attention
        "standard_attention": "Attn = softmax(QK^T / sqrt(d)) V, O(n^2)",

        # Linformer
        "linformer_projection": "K' = E_K @ K, V' = E_V @ V (project n -> k)",
        "linformer_attention": "Attn = softmax(Q @ K'^T / sqrt(d)) @ V'",
        "linformer_complexity": "O(n * k * d) = O(n) for constant k",
        "linformer_k_bound": "k = O(d * log(d) / epsilon^2) for epsilon error",

        # Performer
        "kernel_attention": "Attn_i = sum_j K(q_i, k_j) v_j / sum_j K(q_i, k_j)",
        "softmax_kernel": "K(q, k) = exp(q^T k / sqrt(d))",
        "random_feature_approx": "K(q, k) approx phi(q)^T phi(k)",
        "favor_features": "phi(x) = exp(-||x||^2/2) * [exp(w_1^T x), ..., exp(w_m^T x)] / sqrt(m)",
        "performer_attention": "Attn = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)",
        "performer_complexity": "O(n * m * d) = O(n) for constant m",
        "causal_cumsum": "S_i = S_{i-1} + phi(k_i) v_i^T (incremental update)",
    }


def compare_methods() -> Dict[str, Dict[str, str]]:
    """Compare Linformer and Performer."""
    return {
        "Linformer": {
            "approach": "Low-rank projection of K, V",
            "complexity": "O(n * k * d)",
            "approximation": "Direct matrix rank reduction",
            "causal_support": "Complex (requires modifications)",
            "hyperparameters": "k (projection dimension)",
            "strengths": "Simple, good for encoders",
            "weaknesses": "Global projection, fixed k",
        },
        "Performer": {
            "approach": "Random feature kernel approximation",
            "complexity": "O(n * m * d)",
            "approximation": "Softmax kernel via random features",
            "causal_support": "Native (via cumulative sums)",
            "hyperparameters": "m (number of random features)",
            "strengths": "Theoretically principled, causal support",
            "weaknesses": "Approximation variance, tuning m",
        },
    }
