"""
Restricted Boltzmann Machine (RBM)

A simplification of the Boltzmann Machine with bipartite graph structure
(no intra-layer connections), enabling efficient inference and learning.

Key Innovation:
    - Removed lateral connections within layers (visible-visible, hidden-hidden)
    - Bipartite structure allows parallel Gibbs sampling
    - Contrastive Divergence (CD) training avoids expensive equilibrium sampling
    - Foundation for Deep Belief Networks

Mathematical Formulation:
    Energy function (simplified from full Boltzmann Machine):
        E(v,h) = -sum_i(a_i * v_i) - sum_j(b_j * h_j) - sum_ij(v_i * w_ij * h_j)

    Joint probability:
        P(v,h) = exp(-E(v,h)) / Z

    Marginal over hidden units:
        P(v) = sum_h exp(-E(v,h)) / Z

    Conditional independence (key property):
        P(h|v) = product_j P(h_j|v)
        P(v|h) = product_i P(v_i|h)

    For binary units:
        P(h_j = 1 | v) = sigmoid(sum_i(w_ij * v_i) + b_j)
        P(v_i = 1 | h) = sigmoid(sum_j(w_ij * h_j) + a_i)

    Free energy (integrating out hidden units):
        F(v) = -sum_i(a_i * v_i) - sum_j log(1 + exp(sum_i(w_ij * v_i) + b_j))

Contrastive Divergence (CD-k) Algorithm:
    Instead of running Gibbs sampling to equilibrium, use k steps
    starting from the data distribution.

    ```
    def CD_k(v_data, k=1):
        # Positive phase: data-driven
        h_pos = sample_hidden(v_data)
        pos_associations = outer(v_data, h_pos)

        # Negative phase: k steps of Gibbs sampling
        v_neg = v_data
        for _ in range(k):
            h_neg = sample_hidden(v_neg)
            v_neg = sample_visible(h_neg)
        h_neg = sample_hidden(v_neg)  # Final hidden for associations
        neg_associations = outer(v_neg, h_neg)

        # Weight update
        delta_W = learning_rate * (pos_associations - neg_associations)
        delta_a = learning_rate * (v_data - v_neg)
        delta_b = learning_rate * (h_pos - h_neg)

        return delta_W, delta_a, delta_b
    ```

Pseudocode - Full Training:
    ```
    def train_rbm(data, num_hidden, num_epochs, learning_rate, k=1):
        num_visible = data.shape[1]

        # Initialize weights (small random, zero biases)
        W = 0.01 * randn(num_visible, num_hidden)
        a = zeros(num_visible)
        b = zeros(num_hidden)

        for epoch in range(num_epochs):
            for batch in get_batches(data):
                # Positive phase
                v0 = batch
                p_h0 = sigmoid(v0 @ W + b)  # P(h=1|v)
                h0 = (random(p_h0.shape) < p_h0).astype(float)

                # Negative phase (CD-k)
                vk = v0
                for _ in range(k):
                    p_hk = sigmoid(vk @ W + b)
                    hk = (random(p_hk.shape) < p_hk).astype(float)
                    p_vk = sigmoid(hk @ W.T + a)
                    vk = (random(p_vk.shape) < p_vk).astype(float)

                # Final hidden probabilities (not samples) for update
                p_hk = sigmoid(vk @ W + b)

                # Update parameters
                W += learning_rate * (v0.T @ p_h0 - vk.T @ p_hk) / batch_size
                a += learning_rate * mean(v0 - vk, axis=0)
                b += learning_rate * mean(p_h0 - p_hk, axis=0)

        return W, a, b
    ```

Variants:
    - Gaussian-Bernoulli RBM: Real-valued visible units
    - Softmax RBM: Multinomial visible units
    - Spike-and-Slab RBM: Continuous hidden units
    - Convolutional RBM: Weight sharing for images

Historical Significance:
    - Enabled practical training of energy-based models
    - CD algorithm was breakthrough for approximate inference
    - Building block for Deep Belief Networks (2006)
    - Precursor to modern generative models

Relationship to Other Methods:
    - Simplification of Boltzmann Machine (removed lateral connections)
    - Building block for Deep Belief Networks
    - Related to autoencoders (RBM learns similar representations)
    - Connection to variational autoencoders (both learn latent representations)
"""

from typing import Tuple, Optional
import numpy as np

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entry
RESTRICTED_BOLTZMANN_MACHINE = MLMethod(
    method_id="restricted_boltzmann_machine",
    name="Restricted Boltzmann Machine",
    year=1986,  # Originally proposed, popularized in 2000s
    era=MethodEra.CLASSICAL,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE],
    authors=["Paul Smolensky"],  # "Harmonium", later developed by Hinton
    paper_title="Information Processing in Dynamical Systems: Foundations of Harmony Theory",
    paper_url=None,
    key_innovation=(
        "Bipartite graph structure with no intra-layer connections enables "
        "efficient block Gibbs sampling. Contrastive Divergence training "
        "avoids need for equilibrium sampling."
    ),
    mathematical_formulation="""
    Energy: E(v,h) = -a^T*v - b^T*h - v^T*W*h

    Joint: P(v,h) = exp(-E(v,h)) / Z

    Conditionals (factorize due to bipartite structure):
        P(h_j=1|v) = sigmoid(W[:,j]^T*v + b_j)
        P(v_i=1|h) = sigmoid(W[i,:]^T*h + a_i)

    Free energy: F(v) = -a^T*v - sum_j log(1 + exp(W[:,j]^T*v + b_j))
    """,
    predecessors=["boltzmann_machine_1985"],
    successors=["deep_belief_network_2006", "gaussian_rbm", "convolutional_rbm"],
    tags=[
        "energy-based",
        "probabilistic",
        "generative",
        "bipartite",
        "contrastive-divergence",
        "unsupervised",
        "feature-learning",
    ],
)


def rbm_energy(
    visible: np.ndarray,
    hidden: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute energy of an RBM configuration.

    E(v,h) = -a^T*v - b^T*h - v^T*W*h

    Args:
        visible: Binary visible unit states (n_visible,) or (batch, n_visible)
        hidden: Binary hidden unit states (n_hidden,) or (batch, n_hidden)
        W: Weight matrix (n_visible, n_hidden)
        a: Visible unit biases (n_visible,)
        b: Hidden unit biases (n_hidden,)

    Returns:
        Energy of the configuration(s)
    """
    # Handle both single sample and batch
    if visible.ndim == 1:
        visible = visible.reshape(1, -1)
        hidden = hidden.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    # E(v,h) = -a^T*v - b^T*h - v^T*W*h
    visible_term = -np.dot(visible, a)
    hidden_term = -np.dot(hidden, b)
    interaction_term = -np.sum(visible @ W * hidden, axis=1)

    energy = visible_term + hidden_term + interaction_term

    if squeeze:
        return float(energy[0])
    return energy


def sample_hidden(
    visible: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample hidden units given visible units.

    P(h_j = 1 | v) = sigmoid(W[:,j]^T * v + b_j)

    Args:
        visible: Visible unit states (n_visible,) or (batch, n_visible)
        W: Weight matrix (n_visible, n_hidden)
        b: Hidden unit biases (n_hidden,)
        sample: If True, return binary samples; if False, return probabilities

    Returns:
        Tuple of (hidden_probabilities, hidden_samples or hidden_probabilities)
    """
    # Handle both single sample and batch
    squeeze = False
    if visible.ndim == 1:
        visible = visible.reshape(1, -1)
        squeeze = True

    # P(h_j = 1 | v) = sigmoid(v @ W + b)
    pre_sigmoid = visible @ W + b
    probs = 1.0 / (1.0 + np.exp(-pre_sigmoid))

    if sample:
        samples = (np.random.random(probs.shape) < probs).astype(np.float64)
    else:
        samples = probs

    if squeeze:
        return probs.squeeze(0), samples.squeeze(0)
    return probs, samples


def sample_visible(
    hidden: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample visible units given hidden units.

    P(v_i = 1 | h) = sigmoid(W[i,:] * h + a_i)

    Args:
        hidden: Hidden unit states (n_hidden,) or (batch, n_hidden)
        W: Weight matrix (n_visible, n_hidden)
        a: Visible unit biases (n_visible,)
        sample: If True, return binary samples; if False, return probabilities

    Returns:
        Tuple of (visible_probabilities, visible_samples or visible_probabilities)
    """
    # Handle both single sample and batch
    squeeze = False
    if hidden.ndim == 1:
        hidden = hidden.reshape(1, -1)
        squeeze = True

    # P(v_i = 1 | h) = sigmoid(h @ W.T + a)
    pre_sigmoid = hidden @ W.T + a
    probs = 1.0 / (1.0 + np.exp(-pre_sigmoid))

    if sample:
        samples = (np.random.random(probs.shape) < probs).astype(np.float64)
    else:
        samples = probs

    if squeeze:
        return probs.squeeze(0), samples.squeeze(0)
    return probs, samples


def contrastive_divergence_update(
    visible_data: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    k: int = 1,
    learning_rate: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Contrastive Divergence (CD-k) update step.

    CD-k approximates the gradient of the log-likelihood by running
    k steps of Gibbs sampling starting from the data.

    Args:
        visible_data: Training data batch (batch_size, n_visible)
        W: Current weight matrix (n_visible, n_hidden)
        a: Current visible biases (n_visible,)
        b: Current hidden biases (n_hidden,)
        k: Number of Gibbs sampling steps
        learning_rate: Learning rate for parameter updates

    Returns:
        Tuple of (delta_W, delta_a, delta_b) - gradients to add to parameters
    """
    batch_size = visible_data.shape[0]

    # Positive phase: compute P(h|v) from data
    v0 = visible_data
    p_h0, h0 = sample_hidden(v0, W, b, sample=True)

    # Negative phase: k steps of Gibbs sampling
    vk = v0
    for _ in range(k):
        _, hk = sample_hidden(vk, W, b, sample=True)
        _, vk = sample_visible(hk, W, a, sample=True)

    # Final hidden probabilities (use probabilities, not samples, for gradient)
    p_hk, _ = sample_hidden(vk, W, b, sample=False)

    # Compute gradients
    # Positive gradient: <v_i * h_j>_data
    positive_grad = v0.T @ p_h0

    # Negative gradient: <v_i * h_j>_model (after k steps)
    negative_grad = vk.T @ p_hk

    # Parameter updates
    delta_W = learning_rate * (positive_grad - negative_grad) / batch_size
    delta_a = learning_rate * np.mean(v0 - vk, axis=0)
    delta_b = learning_rate * np.mean(p_h0 - p_hk, axis=0)

    return delta_W, delta_a, delta_b


def free_energy(
    visible: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute free energy of visible configuration (marginalized over hidden).

    F(v) = -a^T*v - sum_j log(1 + exp(W[:,j]^T*v + b_j))

    The free energy is useful for:
        - Computing pseudo-likelihood during training
        - Model comparison
        - Anomaly detection

    Args:
        visible: Visible unit states (n_visible,) or (batch, n_visible)
        W: Weight matrix (n_visible, n_hidden)
        a: Visible unit biases (n_visible,)
        b: Hidden unit biases (n_hidden,)

    Returns:
        Free energy value(s)
    """
    squeeze = False
    if visible.ndim == 1:
        visible = visible.reshape(1, -1)
        squeeze = True

    # -a^T * v
    visible_term = -visible @ a

    # -sum_j log(1 + exp(v @ W[:,j] + b_j))
    wx_b = visible @ W + b
    hidden_term = -np.sum(np.log(1 + np.exp(wx_b)), axis=1)

    free_energy_val = visible_term + hidden_term

    if squeeze:
        return float(free_energy_val[0])
    return free_energy_val
