"""
Radial Basis Function Networks (RBF Networks)

Research index entry for RBF networks, which use Gaussian (radial)
activation functions for local, rather than distributed, representations.
Popular in the late 1980s and 1990s as an alternative to MLPs.

Key papers:
- Broomhead & Lowe (1988) "Multivariable Functional Interpolation and Adaptive Networks"
- Moody & Darken (1989) "Fast Learning in Networks of Locally-Tuned Processing Units"

Key contributions:
- Local (vs distributed) representations
- Gaussian activation functions
- Two-stage training (centers then weights)
- Universal approximation with different properties than MLPs
"""

from typing import Dict, List

from ..core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for RBF Networks."""
    return MLMethod(
        method_id="rbf_network_1988",
        name="Radial Basis Function Network",
        year=1988,
        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE],
        authors=[
            "David Broomhead",
            "David Lowe",
            "John Moody",
            "Christian Darken",
        ],
        paper_title="Multivariable Functional Interpolation and Adaptive Networks",
        paper_url="https://www.complex-systems.com/abstracts/v02_i03_a05/",
        key_innovation="""
        RBF networks introduced LOCAL representation learning:

        1. GAUSSIAN ACTIVATION: Each hidden unit responds to inputs near
           its "center" with response decreasing based on distance.
           phi(x) = exp(-||x - c||^2 / (2 * sigma^2))

        2. LOCAL REPRESENTATION: Each hidden unit is responsible for a
           specific region of input space, unlike MLPs where all units
           contribute to all outputs (distributed representation).

        3. TWO-STAGE TRAINING: Centers can be determined by unsupervised
           learning (k-means, random selection), then output weights
           learned via linear regression - faster than backprop.

        4. INTERPOLATION PROPERTIES: With enough centers, RBF networks
           can exactly interpolate training data (but may overfit).

        RBF networks bridge neural networks and kernel methods, and
        were influential in the development of SVMs.
        """,
        mathematical_formulation="""
        NETWORK ARCHITECTURE:
            Input: x in R^d
            Hidden layer: K radial basis functions
            Output: Linear combination

        RADIAL BASIS FUNCTION:
            phi_k(x) = exp(-||x - c_k||^2 / (2 * sigma_k^2))

        Where:
            c_k = center of the k-th basis function
            sigma_k = width parameter

        NETWORK OUTPUT:
            f(x) = sum_{k=1}^{K} w_k * phi_k(x) + w_0

            = sum_{k=1}^{K} w_k * exp(-||x - c_k||^2 / (2 * sigma_k^2)) + w_0

        MATRIX FORM:
            y = Phi * w
        Where:
            Phi_ij = phi_j(x_i) = exp(-||x_i - c_j||^2 / (2 * sigma_j^2))

        EXACT INTERPOLATION (when K = N):
            w = Phi^(-1) * y   (solve linear system)
        """,
        predecessors=["perceptron_1958", "potential_functions"],
        successors=["svm_1995", "gaussian_processes"],
        tags=[
            "radial_basis_function",
            "gaussian",
            "local_representation",
            "kernel",
            "interpolation",
            "two_stage_training",
        ],
        notes="""
        RBF networks were particularly popular in the 1990s before SVMs
        became dominant. They offered faster training than MLPs (no
        backpropagation needed if centers are fixed), interpretable
        local representations, and theoretical connections to
        regularization theory.

        The choice between local (RBF) and distributed (MLP) representations
        depends on the problem: RBF networks may need many more hidden units
        for high-dimensional inputs due to the "curse of dimensionality"
        affecting local methods.

        Modern usage is limited but RBF networks remain important for
        understanding the relationship between neural networks and
        kernel methods.
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for RBF network training and inference."""
    return """
    RBF NETWORK ALGORITHM
    =====================

    ARCHITECTURE:
        - Input layer: d dimensions
        - Hidden layer: K RBF units with centers c_1, ..., c_K
        - Output layer: Linear combination with weights w

    TRAINING (Two-Stage):

        STAGE 1: Determine Centers (Unsupervised)

        Option A: Random Selection
            centers = random_sample(training_data, K)

        Option B: K-Means Clustering
            centers, _ = kmeans(training_data, K)

        Option C: Orthogonal Least Squares
            # Greedily select centers that most reduce error
            for k in 1 to K:
                best_center = argmax_x (variance_explained(x))
                centers[k] = best_center

        STAGE 2: Determine Widths
            Option A: Fixed width
                sigma = average_distance_between_centers / sqrt(2*K)

            Option B: K-nearest neighbors
                sigma_k = (1/k) * sum(distance to k nearest centers)

            Option C: Learned via gradient descent

        STAGE 3: Learn Output Weights (Linear Regression)
            # Compute design matrix
            for i in 1 to N:
                for k in 1 to K:
                    Phi[i, k] = exp(-||x_i - c_k||^2 / (2 * sigma_k^2))

            # Solve linear system (closed form)
            w = (Phi^T * Phi)^(-1) * Phi^T * y    # pseudoinverse

            # Or with regularization
            w = (Phi^T * Phi + lambda * I)^(-1) * Phi^T * y

    INFERENCE:
        def predict(x):
            # Compute activations
            phi = []
            for k in 1 to K:
                phi[k] = exp(-||x - c_k||^2 / (2 * sigma_k^2))

            # Linear combination
            return sum(w[k] * phi[k] for k in 1 to K) + bias

    FULL GRADIENT TRAINING (Alternative):
        # If learning centers and widths too
        for epoch in 1 to max_epochs:
            for (x, y) in training_data:
                y_hat = predict(x)
                loss = (y - y_hat)^2

                # Gradients w.r.t. weights
                dL/dw_k = -2 * (y - y_hat) * phi_k(x)

                # Gradients w.r.t. centers
                dL/dc_k = 2 * (y - y_hat) * w_k * phi_k(x) * (x - c_k) / sigma_k^2

                # Gradients w.r.t. widths
                dL/dsigma_k = 2 * (y - y_hat) * w_k * phi_k(x) * ||x - c_k||^2 / sigma_k^3

                # Update all parameters
                w, c, sigma = w - lr * dL/dw, c - lr * dL/dc, sigma - lr * dL/dsigma
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for RBF networks."""
    return {
        # Gaussian RBF
        "gaussian_rbf": "phi(x, c) = exp(-||x - c||^2 / (2 * sigma^2))",
        "gaussian_normalized": "phi(x, c) = exp(-||x - c||^2 / r^2) where r = sqrt(2) * sigma",
        # Other RBF types
        "multiquadric": "phi(r) = sqrt(r^2 + c^2)",
        "inverse_multiquadric": "phi(r) = 1 / sqrt(r^2 + c^2)",
        "thin_plate_spline": "phi(r) = r^2 * log(r)",
        "polyharmonic": "phi(r) = r^k where k is odd, or r^k * log(r) where k is even",
        # Network output
        "network_output": "f(x) = sum_{k=1}^{K} w_k * phi(||x - c_k||) + w_0",
        "matrix_form": "y = Phi * w where Phi_ij = phi(||x_i - c_j||)",
        # Training
        "weight_solution": "w = (Phi^T Phi)^(-1) Phi^T y  (pseudoinverse)",
        "regularized_solution": "w = (Phi^T Phi + lambda I)^(-1) Phi^T y",
        # Width heuristics
        "width_heuristic_1": "sigma = d_max / sqrt(2K) where d_max = max distance between centers",
        "width_heuristic_2": "sigma_k = (1/P) * sum_{p=1}^{P} ||c_k - c_{kp}|| (P nearest centers)",
        # Interpolation condition
        "exact_interpolation": "Phi * w = y has unique solution when Phi is invertible",
        # Universal approximation
        "universal_approx": "For continuous f on compact K, exists RBF net g: ||f - g||_inf < epsilon",
        # Gradient (for backprop training)
        "gradient_center": "dL/dc_k = sum_i delta_i * w_k * phi_k(x_i) * (x_i - c_k) / sigma_k^2",
        "gradient_width": "dL/dsigma_k = sum_i delta_i * w_k * phi_k(x_i) * ||x_i - c_k||^2 / sigma_k^3",
    }


def get_local_vs_distributed() -> Dict[str, any]:
    """Compare local (RBF) and distributed (MLP) representations."""
    return {
        "local_representation": {
            "description": """
                Each hidden unit is 'responsible' for a specific region
                of input space. The unit's response decreases with distance
                from its center.
            """,
            "example": "RBF Networks, K-Nearest Neighbors",
            "properties": [
                "Interpretable: can identify which 'prototype' an input is near",
                "Sparse activation: only nearby units respond strongly",
                "May need many units for high-dimensional spaces",
                "Smooth interpolation between known points",
            ],
            "curse_of_dimensionality": """
                To cover a d-dimensional hypercube with local units of
                radius r, need O((1/r)^d) units. Exponential in dimension.
            """,
        },
        "distributed_representation": {
            "description": """
                Information is encoded across all hidden units. Each unit
                contributes to representing all inputs, and each input
                activates many units.
            """,
            "example": "MLP, Autoencoders, Word Embeddings",
            "properties": [
                "Efficient: can represent exponentially many patterns",
                "Dense activation: all units participate",
                "Better generalization in high dimensions",
                "Less interpretable individual units",
            ],
            "advantage": """
                A distributed code with n binary features can represent
                2^n distinct patterns, while n local units represent n patterns.
            """,
        },
        "hybrid_approaches": [
            "Gaussian Mixture Models (local in latent space)",
            "Self-Organizing Maps (local but learned topology)",
            "Attention mechanisms (dynamic local-like weighting)",
        ],
    }


def get_rbf_variants() -> List[Dict[str, str]]:
    """Return variants and extensions of RBF networks."""
    return [
        {
            "name": "Normalized RBF Network",
            "description": """
                Normalize the RBF activations to sum to 1:
                phi_k_norm(x) = phi_k(x) / sum_j phi_j(x)

                This ensures outputs are weighted averages of the weight
                values, bounded by min/max of weights.
            """,
            "benefit": "More stable extrapolation, smoother output surface.",
        },
        {
            "name": "Generalized RBF Network",
            "description": """
                Use elliptical Gaussians instead of spherical:
                phi(x) = exp(-0.5 * (x-c)^T Sigma^(-1) (x-c))

                Each unit has its own covariance matrix Sigma.
            """,
            "benefit": "Can model anisotropic input distributions.",
        },
        {
            "name": "Growing RBF Network",
            "description": """
                Start with few centers and add new ones during training
                in regions with high error.
            """,
            "benefit": "Automatic structure selection.",
        },
        {
            "name": "RBF-SVM Connection",
            "description": """
                SVM with Gaussian kernel is equivalent to an RBF network
                where support vectors are the centers and dual coefficients
                are the weights.
            """,
            "benefit": "Principled selection of centers (support vectors).",
        },
        {
            "name": "Probabilistic RBF (Mixture of Gaussians)",
            "description": """
                Interpret as a mixture density model where each RBF unit
                represents a Gaussian component with mixing weight pi_k.
            """,
            "benefit": "Probabilistic interpretation, density estimation.",
        },
    ]


def get_training_methods() -> Dict[str, str]:
    """Return different training methods for RBF networks."""
    return {
        "random_centers": """
            Simply select K training points randomly as centers.
            Fast but may not cover input space well.
        """,
        "kmeans_centers": """
            Use k-means clustering to find centers that represent
            the data distribution. More principled than random.
        """,
        "orthogonal_least_squares": """
            Greedily select centers that maximally reduce output
            error. Efficient and gives compact networks.
        """,
        "gradient_descent_all": """
            Learn centers, widths, and weights jointly via gradient
            descent. Most flexible but slow.
        """,
        "em_algorithm": """
            If using Gaussian mixture interpretation, use EM to
            learn centers (means) and widths (covariances).
        """,
        "regularization": """
            Add penalty term lambda * ||w||^2 to prevent overfitting
            when number of centers approaches number of training points.
        """,
    }
