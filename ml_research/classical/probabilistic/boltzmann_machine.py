"""
Boltzmann Machine (1985)

Authors: Geoffrey Hinton & Terrence Sejnowski
Paper: "A Learning Algorithm for Boltzmann Machines"
       Cognitive Science 9(1):147-169, 1985

A stochastic recurrent neural network that can learn arbitrary probability
distributions over binary vectors. The first energy-based model to use
simulated annealing for inference.

Key Innovation:
    - First successful learning algorithm for neural networks with hidden units
    - Introduced energy-based probabilistic interpretation to neural networks
    - Uses simulated annealing (Gibbs sampling) for inference

Mathematical Formulation:
    Energy function:
        E(v,h) = -sum_i(a_i * v_i) - sum_j(b_j * h_j) - sum_ij(v_i * w_ij * h_j)
                 - sum_ij(v_i * L_ij * v_j) - sum_ij(h_i * J_ij * h_j)

    Where:
        v = visible units
        h = hidden units
        a = visible biases
        b = hidden biases
        w = visible-hidden weights
        L = visible-visible lateral weights
        J = hidden-hidden lateral weights

    Probability distribution:
        P(v,h) = exp(-E(v,h)) / Z

    Where Z is the partition function:
        Z = sum_{v,h} exp(-E(v,h))

    Update probability (at temperature T):
        P(s_i = 1) = sigma((sum_j(w_ij * s_j) + b_i) / T)

    Where sigma is the sigmoid function.

Training Algorithm:
    Uses contrastive Hebbian learning:
    delta_w_ij = eta * (<v_i * h_j>_data - <v_i * h_j>_model)

    Where:
        <...>_data = expectation with visible units clamped to data
        <...>_model = expectation under free-running model

Pseudocode:
    ```
    def train_boltzmann_machine(data, num_epochs, learning_rate, T_start, T_end):
        # Initialize weights randomly
        W = random_init()
        L = random_init()  # visible-visible
        J = random_init()  # hidden-hidden
        a = zeros(num_visible)
        b = zeros(num_hidden)

        for epoch in range(num_epochs):
            for sample in data:
                # Positive phase: clamp visible to data
                v = sample
                # Anneal to equilibrium with visible clamped
                h = simulated_annealing(v, clamped=True, T_start, T_end)
                pos_stats = compute_statistics(v, h)

                # Negative phase: free-running
                v, h = simulated_annealing(T_start, T_end)
                neg_stats = compute_statistics(v, h)

                # Update weights
                W += learning_rate * (pos_stats['vh'] - neg_stats['vh'])
                L += learning_rate * (pos_stats['vv'] - neg_stats['vv'])
                J += learning_rate * (pos_stats['hh'] - neg_stats['hh'])
                a += learning_rate * (pos_stats['v'] - neg_stats['v'])
                b += learning_rate * (pos_stats['h'] - neg_stats['h'])

        return W, L, J, a, b

    def simulated_annealing(T_start, T_end, num_steps):
        for t in range(num_steps):
            T = T_start * (T_end / T_start) ** (t / num_steps)
            for unit in random_order(all_units):
                p = sigmoid(input_to_unit(unit) / T)
                unit.state = 1 if random() < p else 0
        return states
    ```

Historical Significance:
    - Bridged statistical mechanics and neural networks
    - Introduced concept of energy landscapes to ML
    - Foundation for Restricted Boltzmann Machines and Deep Belief Networks
    - Demonstrated hidden units could learn useful representations

Limitations:
    - Extremely slow due to need for equilibrium in both phases
    - Partition function Z is intractable for large networks
    - Lateral connections make inference expensive (O(n^2) per step)
"""

from typing import List, Optional, Tuple
import numpy as np

from ...core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


# Research index entry
BOLTZMANN_MACHINE = MLMethod(
    method_id="boltzmann_machine_1985",
    name="Boltzmann Machine",
    year=1985,
    era=MethodEra.CLASSICAL,
    category=MethodCategory.GENERATIVE,
    lineages=[MethodLineage.GENERATIVE_LINE],
    authors=["Geoffrey Hinton", "Terrence Sejnowski"],
    paper_title="A Learning Algorithm for Boltzmann Machines",
    paper_url="https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf",
    key_innovation=(
        "First learning algorithm for neural networks with hidden units. "
        "Introduced energy-based probabilistic models using simulated annealing "
        "for inference in stochastic recurrent networks."
    ),
    mathematical_formulation="""
    Energy: E(v,h) = -sum_i(a_i*v_i) - sum_j(b_j*h_j) - sum_ij(v_i*w_ij*h_j)
                     - sum_ij(v_i*L_ij*v_j) - sum_ij(h_i*J_ij*h_j)

    Probability: P(v,h) = exp(-E(v,h)) / Z

    Update rule: P(s_i=1) = sigmoid((sum_j(w_ij*s_j) + b_i) / T)

    Learning: delta_w_ij = eta * (<v_i*h_j>_data - <v_i*h_j>_model)
    """,
    predecessors=["hopfield_network_1982"],
    successors=["restricted_boltzmann_machine", "mean_field_boltzmann"],
    tags=[
        "energy-based",
        "probabilistic",
        "stochastic",
        "recurrent",
        "simulated-annealing",
        "connectionist",
        "generative",
    ],
)


def boltzmann_energy(
    visible: np.ndarray,
    hidden: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    J: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute energy of a Boltzmann Machine configuration.

    E(v,h) = -sum_i(a_i*v_i) - sum_j(b_j*h_j) - sum_ij(v_i*w_ij*h_j)
             - sum_ij(v_i*L_ij*v_j) - sum_ij(h_i*J_ij*h_j)

    Args:
        visible: Binary visible unit states (n_visible,)
        hidden: Binary hidden unit states (n_hidden,)
        W: Visible-hidden weight matrix (n_visible, n_hidden)
        L: Visible-visible lateral weights (n_visible, n_visible)
        J: Hidden-hidden lateral weights (n_hidden, n_hidden)
        a: Visible unit biases (n_visible,)
        b: Hidden unit biases (n_hidden,)

    Returns:
        Energy of the configuration (lower is more probable)
    """
    # Bias terms
    visible_bias_term = -np.dot(a, visible)
    hidden_bias_term = -np.dot(b, hidden)

    # Interaction terms
    visible_hidden_term = -np.dot(visible, np.dot(W, hidden))
    visible_lateral_term = -0.5 * np.dot(visible, np.dot(L, visible))
    hidden_lateral_term = -0.5 * np.dot(hidden, np.dot(J, hidden))

    energy = (
        visible_bias_term
        + hidden_bias_term
        + visible_hidden_term
        + visible_lateral_term
        + hidden_lateral_term
    )

    return float(energy)


def gibbs_sampling_step(
    visible: np.ndarray,
    hidden: np.ndarray,
    W: np.ndarray,
    L: np.ndarray,
    J: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    temperature: float = 1.0,
    clamp_visible: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one Gibbs sampling step for Boltzmann Machine.

    Updates each unit stochastically based on sigmoid probability:
        P(s_i = 1) = sigmoid((sum_j(w_ij * s_j) + b_i) / T)

    Args:
        visible: Current visible unit states (n_visible,)
        hidden: Current hidden unit states (n_hidden,)
        W: Visible-hidden weight matrix (n_visible, n_hidden)
        L: Visible-visible lateral weights (n_visible, n_visible)
        J: Hidden-hidden lateral weights (n_hidden, n_hidden)
        a: Visible unit biases (n_visible,)
        b: Hidden unit biases (n_hidden,)
        temperature: Sampling temperature (higher = more random)
        clamp_visible: If True, don't update visible units

    Returns:
        Tuple of (new_visible, new_hidden) states
    """
    n_visible = len(visible)
    n_hidden = len(hidden)

    new_visible = visible.copy()
    new_hidden = hidden.copy()

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    # Update hidden units
    for j in range(n_hidden):
        # Input from visible units, other hidden units, and bias
        input_j = (
            np.dot(W[:, j], new_visible)
            + np.dot(J[j, :], new_hidden)
            - J[j, j] * new_hidden[j]  # Exclude self-connection
            + b[j]
        )
        prob = sigmoid(input_j / temperature)
        new_hidden[j] = 1 if np.random.random() < prob else 0

    # Update visible units (unless clamped)
    if not clamp_visible:
        for i in range(n_visible):
            # Input from hidden units, other visible units, and bias
            input_i = (
                np.dot(W[i, :], new_hidden)
                + np.dot(L[i, :], new_visible)
                - L[i, i] * new_visible[i]  # Exclude self-connection
                + a[i]
            )
            prob = sigmoid(input_i / temperature)
            new_visible[i] = 1 if np.random.random() < prob else 0

    return new_visible, new_hidden
