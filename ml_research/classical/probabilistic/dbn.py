"""
Deep Belief Networks (2006)

Author: Geoffrey Hinton
Paper: "A Fast Learning Algorithm for Deep Belief Nets"
       Neural Computation 18(7):1527-1554, 2006

A generative graphical model composed of multiple layers of latent variables,
trained using greedy layer-wise pretraining of Restricted Boltzmann Machines.

Key Innovation:
    - Solved the vanishing gradient problem through unsupervised pretraining
    - Greedy layer-wise training enables deep architecture learning
    - Each layer learns increasingly abstract representations
    - Sparked the "deep learning revolution" of 2006+

Mathematical Formulation:
    A DBN with L layers has the joint distribution:

        P(v, h^1, h^2, ..., h^L) = P(h^{L-1}, h^L) * prod_{k=0}^{L-2} P(h^k | h^{k+1})

    Where:
        - v = h^0 is the visible layer
        - h^k are hidden layers
        - P(h^{L-1}, h^L) is modeled by an RBM (undirected)
        - P(h^k | h^{k+1}) are directed sigmoid belief networks

    After pretraining, the top two layers form an RBM:
        P(h^{L-1}, h^L) = exp(-E(h^{L-1}, h^L)) / Z

    Lower layers are directed generative:
        P(h^k_j = 1 | h^{k+1}) = sigmoid(W^{k+1} * h^{k+1} + b^k)

Greedy Layer-wise Pretraining Algorithm:
    ```
    def pretrain_dbn(data, layer_sizes, num_epochs, learning_rate):
        '''
        Train DBN by stacking RBMs trained with Contrastive Divergence.

        Args:
            data: Training data (batch_size, input_dim)
            layer_sizes: List of hidden layer sizes [h1, h2, ..., hL]
            num_epochs: Epochs per layer
            learning_rate: Learning rate for CD

        Returns:
            List of (W, a, b) tuples for each layer
        '''
        weights = []
        current_input = data

        for layer_idx, num_hidden in enumerate(layer_sizes):
            print(f"Training layer {layer_idx + 1}: {current_input.shape[1]} -> {num_hidden}")

            # Train an RBM for this layer
            W, a, b = train_rbm(
                current_input,
                num_hidden,
                num_epochs,
                learning_rate
            )
            weights.append((W, a, b))

            # Transform data through this layer for next layer's input
            # Use deterministic (mean) activation for stability
            p_hidden = sigmoid(current_input @ W + b)
            current_input = p_hidden

        return weights
    ```

Fine-tuning Options:
    1. **Discriminative Fine-tuning** (for classification):
       - Add softmax output layer
       - Backpropagate through entire network
       - Pretrained weights provide good initialization

    2. **Generative Fine-tuning** (wake-sleep):
       - Wake phase: bottom-up pass, update generative weights
       - Sleep phase: top-down pass, update recognition weights
       - Contrastive wake-sleep algorithm

    3. **Hybrid**: Use both supervised and unsupervised objectives

Pseudocode - Complete Training Pipeline:
    ```
    def train_dbn_classifier(X_train, y_train, layer_sizes, num_epochs_pretrain,
                             num_epochs_finetune, learning_rate):
        # Phase 1: Greedy layer-wise pretraining (unsupervised)
        pretrained_weights = pretrain_dbn(
            X_train, layer_sizes, num_epochs_pretrain, learning_rate
        )

        # Phase 2: Add classification layer
        num_classes = len(unique(y_train))
        classification_weights = initialize_weights(layer_sizes[-1], num_classes)

        # Phase 3: Fine-tune with backpropagation
        all_weights = pretrained_weights + [classification_weights]

        for epoch in range(num_epochs_finetune):
            for batch_x, batch_y in get_batches(X_train, y_train):
                # Forward pass
                activations = forward_pass(batch_x, all_weights)
                predictions = softmax(activations[-1])

                # Backward pass
                gradients = backprop(batch_y, predictions, activations, all_weights)

                # Update weights
                for i, (w, grad) in enumerate(zip(all_weights, gradients)):
                    all_weights[i] = (
                        w[0] - learning_rate * grad[0],  # W
                        w[1] - learning_rate * grad[1],  # a
                        w[2] - learning_rate * grad[2],  # b
                    )

        return all_weights

    def forward_pass(x, weights):
        '''Forward pass through DBN.'''
        activations = [x]
        current = x

        for W, a, b in weights:
            pre_activation = current @ W + b
            current = sigmoid(pre_activation)  # Or softmax for final layer
            activations.append(current)

        return activations

    def generate_samples(weights, num_samples, gibbs_steps=1000):
        '''Generate samples from trained DBN (top-down).'''
        # Start with random activations at top layer
        top_hidden = random_binary(num_samples, weights[-1][0].shape[1])

        # Gibbs sample in top RBM
        for _ in range(gibbs_steps):
            p_below = sigmoid(top_hidden @ weights[-1][0].T + weights[-1][1])
            below = sample_binary(p_below)
            p_top = sigmoid(below @ weights[-1][0] + weights[-1][2])
            top_hidden = sample_binary(p_top)

        # Propagate down through directed layers
        current = sigmoid(top_hidden @ weights[-1][0].T + weights[-1][1])

        for W, a, b in reversed(weights[:-1]):
            current = sigmoid(current @ W.T + a)

        return current  # Generated visible samples
    ```

Architecture Variants:
    - **Standard DBN**: All binary hidden units
    - **Gaussian-Binary DBN**: Real-valued inputs, binary hidden
    - **Deep Boltzmann Machine**: Undirected connections at all levels
    - **Convolutional DBN**: Weight sharing for image data

Historical Significance:
    - Reignited interest in deep neural networks
    - Demonstrated that deep architectures could be trained effectively
    - Pretraining idea influenced development of autoencoders, VAEs
    - Led to breakthroughs in speech recognition (2009-2012)
    - Laid groundwork for modern deep learning

Relationship to Modern Methods:
    - Pretraining concept influenced BERT, GPT (masked/autoregressive pretraining)
    - Layer-wise ideas in progressive growing of GANs
    - Feature hierarchy concept fundamental to all deep learning
    - Replaced by end-to-end training with better optimizers (Adam, batch norm)
"""

from typing import List, Tuple, Optional
import numpy as np

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entry
DEEP_BELIEF_NETWORK = MLMethod(
    method_id="deep_belief_network_2006",
    name="Deep Belief Network",
    year=2006,
    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE],
    authors=["Geoffrey Hinton", "Simon Osindero", "Yee-Whye Teh"],
    paper_title="A Fast Learning Algorithm for Deep Belief Nets",
    paper_url="https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf",
    key_innovation=(
        "Greedy layer-wise pretraining using RBMs solves vanishing gradient problem. "
        "Demonstrated that deep architectures can be effectively trained, "
        "sparking the deep learning revolution."
    ),
    mathematical_formulation="""
    Joint distribution for L-layer DBN:
        P(v, h^1, ..., h^L) = P(h^{L-1}, h^L) * prod_{k=0}^{L-2} P(h^k | h^{k+1})

    Top RBM (undirected):
        P(h^{L-1}, h^L) = exp(-E(h^{L-1}, h^L)) / Z

    Directed generative layers:
        P(h^k_j = 1 | h^{k+1}) = sigmoid(W^{k+1} * h^{k+1} + b^k)

    Training: Maximize variational lower bound via greedy layer-wise CD
    """,
    predecessors=["restricted_boltzmann_machine", "multilayer_perceptron"],
    successors=[
        "deep_boltzmann_machine",
        "variational_autoencoder",
        "stacked_autoencoder",
    ],
    tags=[
        "deep-learning",
        "pretraining",
        "generative",
        "layer-wise",
        "unsupervised",
        "breakthrough",
        "feature-hierarchy",
    ],
)


def greedy_layerwise_pretrain(
    data: np.ndarray,
    layer_sizes: List[int],
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    cd_k: int = 1,
    batch_size: int = 32,
    verbose: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Greedy layer-wise pretraining of a Deep Belief Network.

    Trains the DBN by stacking RBMs, where each RBM is trained with
    Contrastive Divergence on the output of the previous layer.

    Args:
        data: Training data (num_samples, input_dim)
        layer_sizes: List of hidden layer sizes [h1, h2, ..., hL]
        num_epochs: Number of epochs to train each RBM
        learning_rate: Learning rate for CD updates
        cd_k: Number of Gibbs sampling steps for CD
        batch_size: Mini-batch size
        verbose: Print progress information

    Returns:
        List of (W, a, b) tuples for each layer, where:
            W: Weight matrix (n_lower, n_upper)
            a: Visible/lower layer biases
            b: Hidden/upper layer biases
    """

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def train_rbm_layer(
        input_data: np.ndarray,
        num_hidden: int,
        epochs: int,
        lr: float,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train a single RBM layer with CD-k."""
        num_visible = input_data.shape[1]
        num_samples = input_data.shape[0]

        # Initialize weights (small random values, zero biases)
        W = 0.01 * np.random.randn(num_visible, num_hidden)
        a = np.zeros(num_visible)
        b = np.zeros(num_hidden)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            total_error = 0.0

            for batch_start in range(0, num_samples, batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                v0 = input_data[batch_indices]
                current_batch_size = v0.shape[0]

                # Positive phase
                p_h0 = sigmoid(v0 @ W + b)
                h0 = (np.random.random(p_h0.shape) < p_h0).astype(np.float64)

                # Negative phase (CD-k)
                vk = v0
                for _ in range(k):
                    p_hk = sigmoid(vk @ W + b)
                    hk = (np.random.random(p_hk.shape) < p_hk).astype(np.float64)
                    p_vk = sigmoid(hk @ W.T + a)
                    vk = (np.random.random(p_vk.shape) < p_vk).astype(np.float64)

                # Final hidden probabilities
                p_hk = sigmoid(vk @ W + b)

                # Update parameters
                W += lr * (v0.T @ p_h0 - vk.T @ p_hk) / current_batch_size
                a += lr * np.mean(v0 - vk, axis=0)
                b += lr * np.mean(p_h0 - p_hk, axis=0)

                # Track reconstruction error
                total_error += np.sum((v0 - p_vk) ** 2)

            if verbose:
                avg_error = total_error / num_samples
                print(f"    Epoch {epoch + 1}/{epochs}, Reconstruction Error: {avg_error:.4f}")

        return W, a, b

    # Greedy layer-wise training
    weights = []
    current_input = data.astype(np.float64)

    for layer_idx, num_hidden in enumerate(layer_sizes):
        if verbose:
            print(
                f"Training layer {layer_idx + 1}/{len(layer_sizes)}: "
                f"{current_input.shape[1]} -> {num_hidden}"
            )

        # Train RBM for this layer
        W, a, b = train_rbm_layer(
            current_input, num_hidden, num_epochs, learning_rate, cd_k
        )
        weights.append((W, a, b))

        # Transform data through this layer for next layer's input
        # Use probabilities (mean activation) for stability
        current_input = sigmoid(current_input @ W + b)

    return weights


def dbn_forward_pass(
    x: np.ndarray,
    weights: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    use_probabilities: bool = True,
) -> List[np.ndarray]:
    """
    Forward pass through DBN, returning activations at each layer.

    Args:
        x: Input data (batch_size, input_dim)
        weights: List of (W, a, b) tuples from pretraining
        use_probabilities: If True, use sigmoid probabilities; if False, sample

    Returns:
        List of activations [input, h1, h2, ..., hL]
    """

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    activations = [x]
    current = x

    for W, a, b in weights:
        pre_activation = current @ W + b
        prob = sigmoid(pre_activation)

        if use_probabilities:
            current = prob
        else:
            current = (np.random.random(prob.shape) < prob).astype(np.float64)

        activations.append(current)

    return activations


def dbn_generate(
    weights: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_samples: int = 1,
    gibbs_steps: int = 1000,
) -> np.ndarray:
    """
    Generate samples from a trained DBN.

    Starts by Gibbs sampling in the top-level RBM, then propagates
    down through the directed generative layers.

    Args:
        weights: List of (W, a, b) tuples from pretraining
        num_samples: Number of samples to generate
        gibbs_steps: Number of Gibbs sampling steps in top RBM

    Returns:
        Generated visible samples (num_samples, input_dim)
    """

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # Get top layer dimensions
    top_W, top_a, top_b = weights[-1]
    num_top_hidden = top_W.shape[1]

    # Initialize with random binary activations
    top_hidden = (np.random.random((num_samples, num_top_hidden)) < 0.5).astype(
        np.float64
    )

    # Gibbs sampling in top RBM
    for _ in range(gibbs_steps):
        # Sample layer below top
        p_below = sigmoid(top_hidden @ top_W.T + top_a)
        below = (np.random.random(p_below.shape) < p_below).astype(np.float64)

        # Sample top layer
        p_top = sigmoid(below @ top_W + top_b)
        top_hidden = (np.random.random(p_top.shape) < p_top).astype(np.float64)

    # Get final sample from layer below top
    current = sigmoid(top_hidden @ top_W.T + top_a)

    # Propagate down through remaining layers (directed, generative)
    for W, a, b in reversed(weights[:-1]):
        p_visible = sigmoid(current @ W.T + a)
        current = p_visible  # Use probabilities for cleaner samples

    return current


def dbn_extract_features(
    data: np.ndarray,
    weights: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    layer_index: int = -1,
) -> np.ndarray:
    """
    Extract learned features from a specific layer of the DBN.

    Args:
        data: Input data (num_samples, input_dim)
        weights: List of (W, a, b) tuples from pretraining
        layer_index: Which layer's activations to return (-1 for top)

    Returns:
        Features from specified layer (num_samples, layer_dim)
    """
    activations = dbn_forward_pass(data, weights, use_probabilities=True)
    return activations[layer_index + 1]  # +1 because activations[0] is input
