"""
Flash Attention - Fast and Memory-Efficient Exact Attention (2022)

Research index entry for Flash Attention, which computes exact attention
with O(N) memory and significant speedups through IO-aware algorithms.

Key contributions:
- IO-aware attention algorithm
- Tiling for GPU memory hierarchy
- Recomputation strategy for backward pass
- Exact attention (not approximation)
"""

from typing import Dict, List

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Flash Attention."""
    return MLMethod(
        method_id="flash_attention_2022",
        name="Flash Attention",
        year=2022,
        era=MethodEra.ATTENTION,
        category=MethodCategory.ATTENTION,
        lineages=[MethodLineage.ATTENTION_LINE, MethodLineage.ATTENTION_LINE],
        authors=[
            "Tri Dao",
            "Daniel Y. Fu",
            "Stefano Ermon",
            "Atri Rudra",
            "Christopher Re",
        ],
        paper_title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        paper_url="https://arxiv.org/abs/2205.14135",
        key_innovation="""
        Flash Attention computes EXACT self-attention (not an approximation)
        while being 2-4x faster and using O(N) memory instead of O(N^2).

        Key insight: Standard attention implementations are memory-bound, not
        compute-bound. The bottleneck is reading/writing the N x N attention
        matrix to GPU high-bandwidth memory (HBM), not the actual computation.

        Innovations:

        1. IO-Awareness: Account for GPU memory hierarchy (HBM vs SRAM).
           Minimize reads/writes to slow HBM by keeping data in fast SRAM.

        2. Tiling: Compute attention in blocks that fit in SRAM. Never
           materialize the full N x N attention matrix.

        3. Recomputation: In backward pass, recompute attention from Q, K, V
           rather than storing the attention matrix. Trades FLOPs for memory.

        4. Kernel fusion: Fuse multiple operations (matmul, softmax, dropout)
           into a single GPU kernel to minimize memory round-trips.

        Flash Attention enables training with much longer sequences (16K+
        tokens) by removing the O(N^2) memory bottleneck.
        """,
        mathematical_formulation="""
        STANDARD ATTENTION (memory analysis)
        ====================================
        S = QK^T / sqrt(d)    # O(N^2) memory
        P = softmax(S)        # O(N^2) memory
        O = PV                # O(N*d) memory

        Memory: O(N^2) for storing S and P
        HBM accesses: O(N^2 * d + N^2)  (read Q,K,V, write S,P,O)

        FLASH ATTENTION (tiled computation)
        ===================================
        Key idea: Compute output O in blocks without materializing full S or P

        Block sizes: B_r, B_c (chosen to fit in SRAM)
        Typically B_r = B_c = sqrt(SRAM_size / d)

        For output block O[i]:
            Initialize: m_i = -inf, l_i = 0, O_i = 0

            For each K, V block j:
                S_ij = Q_i @ K_j^T / sqrt(d)    # [B_r x B_c] in SRAM
                m_new = max(m_i, rowmax(S_ij))
                P_ij = exp(S_ij - m_new)        # Local softmax
                l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)

                # Update output with rescaling
                O_i = (l_i * exp(m_i - m_new) * O_i + P_ij @ V_j) / l_new

                m_i = m_new
                l_i = l_new

        The rescaling ensures correct softmax normalization despite block-wise
        computation (online softmax trick).

        MEMORY COMPLEXITY
        =================
        Standard: O(N^2) for attention matrix
        Flash:    O(N) - only store Q, K, V, O (no intermediate attention)

        HBM ACCESS COMPLEXITY
        =====================
        Standard: O(N^2 * d + N^2) memory accesses
        Flash:    O(N^2 * d^2 / SRAM) accesses

        Since SRAM >> d, Flash Attention is much faster despite doing
        the same number of FLOPs.

        BACKWARD PASS (RECOMPUTATION)
        =============================
        Instead of storing P for backward:
        1. Store only Q, K, V, O, and the logsumexp statistics (l, m)
        2. Recompute S and P during backward using same tiling
        3. Trade O(N^2) memory for O(N^2) extra FLOPs

        This is worthwhile because memory is the bottleneck.

        Backward gradients:
        dV = P^T @ dO
        dP = dO @ V^T
        dS = P * (dP - rowsum(dP * P))  # Softmax backward
        dQ = dS @ K / sqrt(d)
        dK = dS^T @ Q / sqrt(d)
        """,
        predecessors=["transformer_2017", "memory_efficient_attention_2021"],
        successors=["flash_attention_2_2023", "flash_decoding_2023"],
        tags=[
            "efficient_attention",
            "exact_attention",
            "io_aware",
            "memory_efficient",
            "gpu_optimization",
            "tiling",
            "kernel_fusion",
        ],
        notes="""
        Flash Attention is a pivotal contribution showing that algorithmic
        improvements accounting for hardware realities can dramatically
        improve efficiency without sacrificing exactness.

        Flash Attention 2 (2023) further optimizes:
        - Better work partitioning across GPU threads
        - Reduced non-matmul FLOPs
        - 2x speedup over Flash Attention 1

        Flash Attention has been integrated into PyTorch (torch.nn.functional.
        scaled_dot_product_attention), HuggingFace Transformers, and most
        modern transformer libraries.

        The IO-aware perspective has influenced other algorithm design,
        recognizing that memory access patterns often matter more than
        FLOP counts on modern hardware.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for Flash Attention."""
    return """
    FLASH ATTENTION FORWARD PASS
    ============================

    def flash_attention_forward(Q, K, V, block_size):
        '''
        Q, K, V: [batch, seq_len, d]
        Returns O: [batch, seq_len, d]

        Computes exact attention without materializing N x N matrix
        '''
        batch, N, d = Q.shape
        B_r = B_c = block_size  # Block sizes for Q and K/V

        # Output and running statistics
        O = zeros(batch, N, d)
        l = zeros(batch, N)  # Softmax denominators
        m = full(batch, N, -inf)  # Running max for numerical stability

        # Number of blocks
        T_r = ceil(N / B_r)  # Number of query blocks
        T_c = ceil(N / B_c)  # Number of key/value blocks

        # Iterate over query blocks
        for i in range(T_r):
            # Load Q block to SRAM
            Q_i = Q[:, i*B_r : (i+1)*B_r, :]  # [batch, B_r, d]

            # Initialize block outputs
            O_i = zeros(batch, B_r, d)
            l_i = zeros(batch, B_r)
            m_i = full(batch, B_r, -inf)

            # Iterate over key/value blocks
            for j in range(T_c):
                # Load K, V blocks to SRAM
                K_j = K[:, j*B_c : (j+1)*B_c, :]  # [batch, B_c, d]
                V_j = V[:, j*B_c : (j+1)*B_c, :]  # [batch, B_c, d]

                # Compute attention scores for this block
                S_ij = Q_i @ K_j.transpose(-1, -2) / sqrt(d)  # [batch, B_r, B_c]

                # Online softmax: update running max
                m_ij = S_ij.max(dim=-1)  # [batch, B_r]
                m_new = maximum(m_i, m_ij)

                # Compute local softmax with updated max
                P_ij = exp(S_ij - m_new.unsqueeze(-1))  # [batch, B_r, B_c]

                # Update running sum
                l_new = exp(m_i - m_new) * l_i + P_ij.sum(dim=-1)  # [batch, B_r]

                # Update output with rescaling
                O_i = (l_i.unsqueeze(-1) * exp(m_i - m_new).unsqueeze(-1) * O_i
                       + P_ij @ V_j) / l_new.unsqueeze(-1)

                # Update statistics
                m_i = m_new
                l_i = l_new

            # Write output block back to HBM
            O[:, i*B_r : (i+1)*B_r, :] = O_i
            l[:, i*B_r : (i+1)*B_r] = l_i
            m[:, i*B_r : (i+1)*B_r] = m_i

        return O, l, m  # l, m saved for backward


    FLASH ATTENTION BACKWARD PASS
    =============================

    def flash_attention_backward(Q, K, V, O, dO, l, m, block_size):
        '''
        Backward pass: compute dQ, dK, dV
        Recomputes attention matrix on-the-fly (memory efficient)
        '''
        batch, N, d = Q.shape
        B_r = B_c = block_size

        dQ = zeros(batch, N, d)
        dK = zeros(batch, N, d)
        dV = zeros(batch, N, d)

        # Iterate over query blocks
        for i in range(T_r):
            Q_i = Q[:, i*B_r : (i+1)*B_r, :]
            O_i = O[:, i*B_r : (i+1)*B_r, :]
            dO_i = dO[:, i*B_r : (i+1)*B_r, :]
            l_i = l[:, i*B_r : (i+1)*B_r]
            m_i = m[:, i*B_r : (i+1)*B_r]

            dQ_i = zeros(batch, B_r, d)

            for j in range(T_c):
                K_j = K[:, j*B_c : (j+1)*B_c, :]
                V_j = V[:, j*B_c : (j+1)*B_c, :]

                # Recompute attention (not stored!)
                S_ij = Q_i @ K_j.transpose(-1, -2) / sqrt(d)
                P_ij = exp(S_ij - m_i.unsqueeze(-1)) / l_i.unsqueeze(-1)

                # Compute gradients
                dV_j = P_ij.transpose(-1, -2) @ dO_i  # [batch, B_c, d]
                dP_ij = dO_i @ V_j.transpose(-1, -2)  # [batch, B_r, B_c]

                # Softmax backward
                D_i = (dO_i * O_i).sum(dim=-1)  # [batch, B_r]
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # [batch, B_r, B_c]

                # Accumulate gradients
                dQ_i += dS_ij @ K_j / sqrt(d)
                dK_j = dS_ij.transpose(-1, -2) @ Q_i / sqrt(d)

                dK[:, j*B_c : (j+1)*B_c, :] += dK_j
                dV[:, j*B_c : (j+1)*B_c, :] += dV_j

            dQ[:, i*B_r : (i+1)*B_r, :] = dQ_i

        return dQ, dK, dV


    CUDA KERNEL (CONCEPTUAL)
    ========================

    __global__ void flash_attention_kernel(Q, K, V, O, l, m, N, d) {
        // Each thread block handles one query block

        // Shared memory (SRAM) for blocks
        __shared__ float Q_shared[B_r][d];
        __shared__ float K_shared[B_c][d];
        __shared__ float V_shared[B_c][d];
        __shared__ float S_shared[B_r][B_c];

        // Load Q block to shared memory
        load_to_shared(Q, Q_shared, block_idx);

        // Initialize accumulators
        float O_local[d] = {0};
        float m_local = -INFINITY;
        float l_local = 0;

        // Iterate over K, V blocks
        for (int j = 0; j < num_kv_blocks; j++) {
            // Load K, V to shared memory
            load_to_shared(K, K_shared, j);
            load_to_shared(V, V_shared, j);
            __syncthreads();

            // Compute S = Q @ K^T (in shared memory)
            compute_matmul(Q_shared, K_shared, S_shared);

            // Online softmax and output update
            update_output(S_shared, V_shared, O_local, m_local, l_local);
            __syncthreads();
        }

        // Write final output to global memory
        write_to_global(O, O_local, block_idx);
    }
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for Flash Attention."""
    return {
        # Standard attention
        "standard_attention": "O = softmax(QK^T / sqrt(d)) V",
        "standard_memory": "O(N^2) for attention matrix",
        "standard_hbm_access": "O(N^2 * d + N^2)",

        # Flash attention
        "flash_memory": "O(N) - no attention matrix stored",
        "flash_hbm_access": "O(N^2 * d^2 / M) where M = SRAM size",
        "block_size": "B = sqrt(M / d) (maximize SRAM utilization)",

        # Online softmax
        "online_max_update": "m_new = max(m_old, max(S_block))",
        "online_sum_update": "l_new = exp(m_old - m_new) * l_old + sum(exp(S - m_new))",
        "output_rescale": "O_new = (l_old * exp(m_old - m_new) * O_old + P @ V) / l_new",

        # Backward recomputation
        "softmax_backward": "dS = P * (dP - sum(dP * P, dim=-1))",
        "backward_flops": "O(N^2 * d) - recompute attention",
        "backward_memory": "O(N) - only Q,K,V,O,l,m stored",
    }


def get_speedup_analysis() -> Dict[str, str]:
    """Return speedup analysis for Flash Attention."""
    return {
        "vs_pytorch_baseline": "2-4x speedup on A100 GPU",
        "memory_reduction": "5-20x reduction in memory usage",
        "sequence_length": "Enables 4x longer sequences at same memory budget",
        "bottleneck_shift": "From memory-bound to compute-bound",
        "flop_count": "Same as standard attention (not fewer FLOPs)",
        "io_reduction": "Key insight - reduce memory accesses, not FLOPs",
    }


def get_flash_attention_2_improvements() -> Dict[str, str]:
    """Return improvements in Flash Attention 2."""
    return {
        "parallelism": "Better work partitioning across thread blocks",
        "warp_specialization": "Separate warps for loading vs computing",
        "non_matmul_ops": "Reduced overhead from non-matmul operations",
        "causal_masking": "More efficient causal attention implementation",
        "speedup": "~2x faster than Flash Attention 1",
        "triton_support": "Available in Triton for easier customization",
    }


def get_hardware_considerations() -> Dict[str, str]:
    """Return hardware considerations for Flash Attention."""
    return {
        "sram_size": "A100: 192KB shared memory per SM",
        "hbm_bandwidth": "A100: 2TB/s HBM bandwidth",
        "compute_throughput": "A100: 312 TFLOPS (FP16 Tensor Cores)",
        "arithmetic_intensity": "Flash Attention increases arithmetic intensity",
        "block_size_a100": "Typical: B_r = B_c = 64-128",
        "tile_shape": "Tiles must fit: B_r * B_c * sizeof(float) < SRAM",
    }


def compare_attention_methods() -> List[Dict[str, str]]:
    """Compare Flash Attention with other efficient attention methods."""
    return [
        {
            "method": "Standard Attention",
            "memory": "O(N^2)",
            "flops": "O(N^2 * d)",
            "exact": "Yes",
            "note": "Memory bottleneck",
        },
        {
            "method": "Flash Attention",
            "memory": "O(N)",
            "flops": "O(N^2 * d)",
            "exact": "Yes",
            "note": "IO-aware, same FLOPs but faster",
        },
        {
            "method": "Linformer",
            "memory": "O(N)",
            "flops": "O(N * k * d)",
            "exact": "No (approximation)",
            "note": "Low-rank projection",
        },
        {
            "method": "Performer",
            "memory": "O(N)",
            "flops": "O(N * m * d)",
            "exact": "No (approximation)",
            "note": "Random features",
        },
        {
            "method": "Sparse (Longformer)",
            "memory": "O(N * w)",
            "flops": "O(N * w * d)",
            "exact": "No (sparse pattern)",
            "note": "Sliding window",
        },
    ]
