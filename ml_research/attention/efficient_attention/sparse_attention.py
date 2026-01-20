"""
Sparse Attention Methods - Longformer and BigBird (2020)

Research index entries for sparse attention mechanisms that reduce
the quadratic complexity of self-attention to linear or near-linear.

Methods included:
- Longformer: Sliding window + global attention (Beltagy et al., 2020)
- BigBird: Random + window + global attention (Zaheer et al., 2020)
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_longformer_info() -> MLMethod:
    """Return the MLMethod entry for Longformer."""
    return MLMethod(
        method_id="longformer_2020",
        name="Longformer",
        year=2020,
        era=MethodEra.ATTENTION,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Iz Beltagy",
            "Matthew E. Peters",
            "Arman Cohan",
        ],
        paper_title="Longformer: The Long-Document Transformer",
        paper_url="https://arxiv.org/abs/2004.05150",
        key_innovation="""
        Longformer introduces a sparse attention pattern that scales linearly
        with sequence length, enabling processing of documents up to 4,096
        tokens (vs. 512 for BERT).

        Key innovations:

        1. Sliding Window Attention: Each token attends only to w/2 tokens
           on each side, giving O(n * w) complexity instead of O(n^2)

        2. Dilated Sliding Window: Gaps in the sliding window increase
           receptive field without increasing computation. With dilation d,
           receptive field grows to w * d per layer.

        3. Global Attention: Selected tokens (e.g., [CLS], question tokens)
           attend to all positions and are attended to by all positions.
           Enables information aggregation for downstream tasks.

        Combined, these patterns allow long-range dependencies through
        stacked layers while maintaining linear complexity.
        """,
        mathematical_formulation="""
        STANDARD ATTENTION (for comparison)
        ====================================
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

        Full attention: A[i,j] = 1 for all i,j
        Complexity: O(n^2 * d)

        SLIDING WINDOW ATTENTION
        ========================
        For window size w:
            A[i,j] = 1 if |i - j| <= w/2
            A[i,j] = 0 otherwise

        Each token attends to w tokens (w/2 on each side)
        Complexity: O(n * w * d)

        With L layers, receptive field = L * w

        DILATED SLIDING WINDOW
        ======================
        For window w and dilation d:
            A[i,j] = 1 if (|i - j| mod d == 0) and (|i - j| <= w * d / 2)

        Effective receptive field grows to w * d per layer
        Useful for higher layers that need broader context

        GLOBAL ATTENTION
        ================
        Let G be the set of global tokens (e.g., [CLS], question tokens)

        For global token g in G:
            A[g, :] = 1  (g attends to all tokens)
            A[:, g] = 1  (all tokens attend to g)

        Global tokens use separate Q, K, V projections:
            Q_g, K_g, V_g for global attention

        COMBINED ATTENTION PATTERN
        ==========================
        Final attention mask:
            A = SlidingWindow + GlobalAttention

        For position i:
            If i in G:
                attend to all positions (global)
            Else:
                attend to positions in sliding window
                + attend to all global positions

        Complexity: O(n * w + n * |G|) = O(n) when w and |G| are constant
        """,
        predecessors=["transformer_2017", "bert_2018"],
        successors=["bigbird_2020", "led_2020"],
        tags=[
            "efficient_attention",
            "sparse_attention",
            "long_document",
            "linear_complexity",
            "sliding_window",
        ],
        notes="""
        Longformer showed that carefully designed sparse attention patterns
        can match or exceed full attention performance on long document tasks
        while being much more efficient. The combination of local (sliding
        window) and global attention provides both fine-grained local context
        and document-level understanding.

        Key applications: Document classification, long-form QA, summarization.
        Longformer-Encoder-Decoder (LED) extends this to seq2seq tasks.
        """,
    )


def get_bigbird_info() -> MLMethod:
    """Return the MLMethod entry for BigBird."""
    return MLMethod(
        method_id="bigbird_2020",
        name="BigBird",
        year=2020,
        era=MethodEra.ATTENTION,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Manzil Zaheer",
            "Guru Guruganesh",
            "Kumar Avinava Dubey",
            "Joshua Ainslie",
            "Chris Alberti",
            "Santiago Ontanon",
            "Philip Pham",
            "Anirudh Ravula",
            "Qifan Wang",
            "Li Yang",
            "Amr Ahmed",
        ],
        paper_title="Big Bird: Transformers for Longer Sequences",
        paper_url="https://arxiv.org/abs/2007.14062",
        key_innovation="""
        BigBird introduces a sparse attention mechanism combining three
        patterns: random, window (local), and global attention. The key
        theoretical contribution is proving that this combination is a
        universal approximator and Turing complete.

        Key innovations:

        1. Random Attention: Each query attends to r random keys.
           Provides connectivity for graph-theoretic expressive power.

        2. Window (Local) Attention: Each token attends to w/2 neighbors
           on each side. Captures local context efficiently.

        3. Global Tokens: Special tokens that attend to and are attended
           by all other tokens. Can be existing tokens (ITC) or added
           tokens (ETC - Extended Transformer Construction).

        Theoretical Results:
        - BigBird attention is a universal approximator of sequence functions
        - BigBird is Turing complete
        - Sparse patterns preserve expressive power of full attention
        """,
        mathematical_formulation="""
        BIGBIRD ATTENTION PATTERN
        =========================
        Three components combined:

        1. RANDOM ATTENTION
        -------------------
        For each query i, sample r random positions to attend to:
            R_i = random_sample({1, ..., n}, r)
            A_random[i, j] = 1 if j in R_i

        Different random patterns can be used per layer
        Complexity contribution: O(n * r)

        2. WINDOW (LOCAL) ATTENTION
        ---------------------------
        Sliding window of size w:
            A_window[i, j] = 1 if |i - j| <= w/2

        Complexity contribution: O(n * w)

        3. GLOBAL ATTENTION
        -------------------
        Let G be global token positions:

        Option A - Internal Transformer Construction (ITC):
            Use first g tokens as global tokens
            G = {1, 2, ..., g}

        Option B - Extended Transformer Construction (ETC):
            Add g extra global tokens
            G = {CLS, SEP, added_globals}

        A_global[i, j] = 1 if i in G or j in G

        COMBINED PATTERN
        ================
        A = A_random OR A_window OR A_global

        Full attention for row i:
            Attend to:
            - r random positions
            - w neighboring positions
            - all global positions
            - if i is global: all positions

        Complexity: O(n * (r + w + g)) = O(n) for constant r, w, g

        THEORETICAL PROPERTIES
        ======================
        Theorem 1 (Universal Approximation):
            BigBird can approximate any continuous sequence-to-sequence
            function with bounded inputs.

        Theorem 2 (Turing Completeness):
            BigBird with O(1) global tokens is Turing complete,
            meaning it can simulate any Turing machine.

        Key insight: Random edges ensure the attention graph remains
        connected (small-world property), enabling information flow
        across the entire sequence.
        """,
        predecessors=["transformer_2017", "longformer_2020"],
        successors=["etc_2021"],
        tags=[
            "efficient_attention",
            "sparse_attention",
            "random_attention",
            "universal_approximator",
            "turing_complete",
            "long_sequences",
        ],
        notes="""
        BigBird's theoretical contributions are significant: it proves that
        sparse attention can match the expressive power of full attention.
        The random attention component, inspired by random graphs, ensures
        the attention pattern remains well-connected despite sparsity.

        In practice, BigBird handles sequences up to 4,096 tokens and
        achieves strong results on QA (Natural Questions, TriviaQA),
        summarization (arXiv, PubMed), and genomics applications.

        The ETC (Extended Transformer Construction) variant adds extra
        global tokens that can aggregate information, useful for tasks
        requiring document-level understanding.
        """,
    )


def pseudocode_longformer() -> str:
    """Return pseudocode for Longformer attention."""
    return """
    LONGFORMER ATTENTION ALGORITHM
    ==============================

    def sliding_window_attention(Q, K, V, window_size):
        '''
        Efficient sliding window attention using chunking
        '''
        n, d = Q.shape
        w = window_size

        output = zeros(n, d)

        for i in range(n):
            # Define window bounds
            start = max(0, i - w // 2)
            end = min(n, i + w // 2 + 1)

            # Compute attention for this position
            q_i = Q[i]  # [d]
            K_window = K[start:end]  # [w, d]
            V_window = V[start:end]  # [w, d]

            scores = q_i @ K_window.T / sqrt(d)  # [w]
            weights = softmax(scores)
            output[i] = weights @ V_window

        return output


    def longformer_attention(Q, K, V, Q_g, K_g, V_g, window_size, global_mask):
        '''
        Full Longformer attention with sliding window + global
        '''
        n, d = Q.shape

        # Identify global and local tokens
        global_indices = where(global_mask)
        local_indices = where(~global_mask)

        output = zeros(n, d)

        # Local (sliding window) attention for non-global tokens
        for i in local_indices:
            # Window attention
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)

            q_i = Q[i]

            # Attend to window + global tokens
            K_attend = concat(K[start:end], K_g[global_indices])
            V_attend = concat(V[start:end], V_g[global_indices])

            scores = q_i @ K_attend.T / sqrt(d)
            weights = softmax(scores)
            output[i] = weights @ V_attend

        # Global attention for global tokens
        for g in global_indices:
            # Global tokens attend to ALL positions
            q_g = Q_g[g]

            scores = q_g @ K.T / sqrt(d)  # Attend to all
            weights = softmax(scores)
            output[g] = weights @ V

        return output


    EFFICIENT IMPLEMENTATION
    ========================

    # Use chunked matrix operations for efficiency
    def chunked_attention(Q, K, V, window_size, chunk_size=256):
        '''
        Process in chunks for memory efficiency
        '''
        n = Q.shape[0]
        chunks = n // chunk_size

        outputs = []
        for chunk_idx in range(chunks):
            start = chunk_idx * chunk_size
            end = (chunk_idx + 1) * chunk_size

            # Include overlap for window attention
            k_start = max(0, start - window_size // 2)
            k_end = min(n, end + window_size // 2)

            Q_chunk = Q[start:end]
            K_chunk = K[k_start:k_end]
            V_chunk = V[k_start:k_end]

            # Build attention mask for chunk
            mask = build_window_mask(chunk_size, k_end - k_start, window_size)

            # Compute attention with mask
            scores = Q_chunk @ K_chunk.T / sqrt(d)
            scores = scores.masked_fill(~mask, -inf)
            weights = softmax(scores, dim=-1)
            output = weights @ V_chunk

            outputs.append(output)

        return concat(outputs)
    """


def pseudocode_bigbird() -> str:
    """Return pseudocode for BigBird attention."""
    return """
    BIGBIRD ATTENTION ALGORITHM
    ===========================

    def bigbird_attention_mask(n, num_random, window_size, global_indices):
        '''
        Generate BigBird attention mask combining three patterns
        '''
        mask = zeros(n, n, dtype=bool)

        # 1. Window (local) attention
        for i in range(n):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2 + 1)
            mask[i, start:end] = True

        # 2. Random attention
        for i in range(n):
            random_indices = random_sample(range(n), num_random)
            mask[i, random_indices] = True

        # 3. Global attention
        for g in global_indices:
            mask[g, :] = True  # global attends to all
            mask[:, g] = True  # all attend to global

        return mask


    def bigbird_attention(Q, K, V, num_random, window_size, num_global):
        '''
        BigBird sparse attention implementation
        '''
        n, d = Q.shape

        # Designate first num_global tokens as global (ITC variant)
        global_indices = list(range(num_global))

        # Generate attention mask
        mask = bigbird_attention_mask(n, num_random, window_size, global_indices)

        # Compute full attention with mask
        scores = Q @ K.T / sqrt(d)  # [n, n]
        scores = scores.masked_fill(~mask, -inf)
        weights = softmax(scores, dim=-1)
        output = weights @ V

        return output


    EFFICIENT SPARSE IMPLEMENTATION
    ===============================

    def sparse_bigbird_attention(Q, K, V, attention_pattern):
        '''
        Memory-efficient sparse attention using block-sparse operations
        '''
        n, d = Q.shape

        # attention_pattern contains indices for each query
        # pattern[i] = list of key indices that query i attends to

        output = zeros(n, d)

        for i in range(n):
            # Get indices this query attends to
            attend_indices = attention_pattern[i]

            q_i = Q[i]  # [d]
            K_sparse = K[attend_indices]  # [num_attend, d]
            V_sparse = V[attend_indices]  # [num_attend, d]

            scores = q_i @ K_sparse.T / sqrt(d)  # [num_attend]
            weights = softmax(scores)
            output[i] = weights @ V_sparse  # [d]

        return output


    ETC VARIANT (EXTENDED TRANSFORMER CONSTRUCTION)
    ===============================================

    def etc_attention(Q, K, V, num_extra_globals):
        '''
        ETC adds extra global tokens rather than using input tokens
        '''
        n, d = Q.shape

        # Add learnable global tokens
        global_tokens = learnable_parameter(num_extra_globals, d)

        # Expand Q, K, V
        Q_ext = concat(global_tokens, Q)  # [g + n, d]
        K_ext = concat(global_tokens, K)
        V_ext = concat(global_tokens, V)

        # Global tokens are first g positions
        global_indices = list(range(num_extra_globals))

        # Apply BigBird attention to extended sequence
        output_ext = bigbird_attention(Q_ext, K_ext, V_ext, ...)

        # Return only non-global outputs (or use globals for classification)
        return output_ext[num_extra_globals:], output_ext[:num_extra_globals]
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for sparse attention methods."""
    return {
        "full_attention_complexity": "O(n^2 * d)",
        "sliding_window_complexity": "O(n * w * d)",
        "longformer_complexity": "O(n * (w + g) * d) = O(n)",
        "bigbird_complexity": "O(n * (r + w + g) * d) = O(n)",
        "receptive_field_stacked": "L * w (for L layers with window w)",
        "dilated_receptive_field": "L * w * d (with dilation d)",
        "global_attention": "A[g,:] = A[:,g] = 1 for global token g",
        "sliding_window": "A[i,j] = 1 if |i-j| <= w/2",
        "random_attention": "A[i,j] = 1 if j in RandomSample(r)",
    }


def compare_methods() -> Dict[str, Dict[str, str]]:
    """Compare Longformer and BigBird."""
    return {
        "Longformer": {
            "patterns": "Sliding window + dilated + global",
            "complexity": "O(n * w)",
            "theoretical_guarantee": "Empirical effectiveness",
            "global_strategy": "Task-specific tokens ([CLS], question)",
            "strengths": "Simple, efficient, good for QA",
        },
        "BigBird": {
            "patterns": "Random + window + global",
            "complexity": "O(n * (r + w))",
            "theoretical_guarantee": "Universal approximator, Turing complete",
            "global_strategy": "ITC (existing tokens) or ETC (extra tokens)",
            "strengths": "Theoretical foundations, genomics applications",
        },
    }
