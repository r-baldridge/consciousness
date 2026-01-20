"""
Support Vector Machines (SVM) - Cortes & Vapnik (1995)

Research index entry for Support Vector Machines, which introduced
the maximum margin principle and kernel trick for classification,
dominating machine learning in the late 1990s and early 2000s.

Paper: "Support-Vector Networks"
Machine Learning, 20(3), 273-297

Key contributions:
- Maximum margin classification
- Kernel trick for nonlinear boundaries
- Convex optimization (global optimum guaranteed)
- Strong theoretical foundations (VC dimension, generalization bounds)
"""

from typing import Dict, List

from ..core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Support Vector Machines."""
    return MLMethod(
        method_id="svm_1995",
        name="Support Vector Machine",
        year=1995,
        era=MethodEra.CLASSICAL,
        category=MethodCategory.ARCHITECTURE,
        lineages=[MethodLineage.PERCEPTRON_LINE],
        authors=[
            "Corinna Cortes",
            "Vladimir Vapnik",
        ],
        paper_title="Support-Vector Networks",
        paper_url="https://link.springer.com/article/10.1007/BF00994018",
        key_innovation="""
        SVMs introduced several powerful ideas:

        1. MAXIMUM MARGIN: Instead of finding any separating hyperplane,
           find the one that maximizes the margin (distance to nearest
           points). This provides better generalization.

        2. SUPPORT VECTORS: The solution depends only on the training
           points closest to the boundary (support vectors), making
           the model sparse and efficient.

        3. KERNEL TRICK: By using kernel functions K(x, x'), we can
           implicitly compute dot products in high-dimensional feature
           spaces, enabling nonlinear decision boundaries without
           explicitly computing the transformation.

        4. CONVEX OPTIMIZATION: The SVM optimization problem is convex
           (quadratic programming), guaranteeing a global optimum and
           avoiding local minima issues of neural networks.

        5. STRONG THEORY: VC dimension and structural risk minimization
           provide theoretical bounds on generalization error.

        SVMs dominated machine learning before the deep learning revolution
        and remain important for small/medium datasets.
        """,
        mathematical_formulation="""
        HARD MARGIN SVM (linearly separable case):
            Maximize margin: max_{w,b} 2/||w||
            Subject to: y_i(w^T x_i + b) >= 1 for all i

        Equivalent (minimize ||w||^2):
            min_{w,b} (1/2) ||w||^2
            s.t. y_i(w^T x_i + b) >= 1

        SOFT MARGIN SVM (allow misclassification):
            min_{w,b,xi} (1/2) ||w||^2 + C * sum_i xi_i
            s.t. y_i(w^T x_i + b) >= 1 - xi_i
                 xi_i >= 0

        DUAL FORMULATION:
            max_alpha sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j (x_i^T x_j)
            s.t. 0 <= alpha_i <= C
                 sum_i alpha_i y_i = 0

        KERNEL SVM:
            Replace x_i^T x_j with K(x_i, x_j) = phi(x_i)^T phi(x_j)

        DECISION FUNCTION:
            f(x) = sign(sum_i alpha_i y_i K(x_i, x) + b)

        Only support vectors (alpha_i > 0) contribute to the sum.
        """,
        predecessors=["optimal_hyperplane_1963", "rbf_network_1988"],
        successors=["kernel_pca", "svr", "structured_svm"],
        tags=[
            "maximum_margin",
            "kernel_methods",
            "convex_optimization",
            "support_vectors",
            "classification",
            "vc_dimension",
        ],
        notes="""
        Historical context: The theory of optimal hyperplanes dates to
        Vapnik & Chervonenkis (1963), but practical SVMs with the kernel
        trick and soft margin emerged in the 1990s.

        SVMs were the dominant method for many applications (text
        classification, image recognition, bioinformatics) from the
        late 1990s until deep learning took over around 2012.

        Key advantages over neural networks of that era:
        - Global optimum (convex problem)
        - Strong theoretical guarantees
        - Work well with small datasets
        - No need for architecture selection

        Key limitations:
        - Doesn't scale well to very large datasets (O(n^2) or O(n^3))
        - Kernel selection can be tricky
        - Less effective for very high-dimensional feature learning
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for SVM training and prediction."""
    return """
    SUPPORT VECTOR MACHINE ALGORITHM
    =================================

    PRIMAL OPTIMIZATION (for linear SVM):
        Objective: min_{w,b,xi} (1/2) ||w||^2 + C * sum_i xi_i
        Subject to:
            y_i (w^T x_i + b) >= 1 - xi_i  for all i
            xi_i >= 0  for all i

        This is a quadratic program (QP) - can use standard QP solvers.

    DUAL OPTIMIZATION (more efficient, enables kernels):
        Objective: max_alpha sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)
        Subject to:
            0 <= alpha_i <= C  for all i
            sum_i alpha_i y_i = 0

        This is also a QP with box constraints.

    SEQUENTIAL MINIMAL OPTIMIZATION (SMO) - Platt 1998:
        # Efficient algorithm that solves QP by iteratively optimizing pairs

        Initialize: alpha = 0

        while not converged:
            # Select working set (two alphas to optimize)
            (i, j) = select_working_set()

            # Analytically solve 2-variable subproblem
            # (closed-form solution for alpha_i, alpha_j)

            # Clip to box constraints [0, C]
            alpha_i_new, alpha_j_new = solve_and_clip(i, j)

            # Update alphas
            alpha[i], alpha[j] = alpha_i_new, alpha_j_new

        # Compute bias from support vectors
        b = compute_bias(alpha, X, y)

        return alpha, b

    PREDICTION:
        def predict(x_new):
            # Sum over support vectors only (where alpha_i > 0)
            score = sum(alpha_i * y_i * K(x_i, x_new) for i in support_vectors) + b
            return sign(score)

    KERNEL COMPUTATION:
        def linear_kernel(x, x'):
            return x^T x'

        def polynomial_kernel(x, x', degree=3, c=1):
            return (x^T x' + c)^degree

        def rbf_kernel(x, x', gamma=1.0):
            return exp(-gamma * ||x - x'||^2)

        def sigmoid_kernel(x, x', kappa=1, theta=0):
            return tanh(kappa * x^T x' + theta)

    MULTICLASS EXTENSION:
        Option 1: One-vs-Rest (OvR)
            Train K binary classifiers (class k vs all others)
            Predict: argmax_k f_k(x)

        Option 2: One-vs-One (OvO)
            Train K(K-1)/2 binary classifiers (each pair)
            Predict: majority voting
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for SVMs."""
    return {
        # Margin
        "margin": "gamma = 2 / ||w|| (distance between parallel hyperplanes)",
        "functional_margin": "y_i (w^T x_i + b)",
        "geometric_margin": "y_i (w^T x_i + b) / ||w||",
        # Primal formulation
        "hard_margin_primal": "min (1/2)||w||^2  s.t. y_i(w^T x_i + b) >= 1",
        "soft_margin_primal": "min (1/2)||w||^2 + C*sum(xi_i)  s.t. y_i(w^T x_i + b) >= 1 - xi_i, xi_i >= 0",
        # Dual formulation
        "dual_objective": "max sum(alpha_i) - (1/2) sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j)",
        "dual_constraints": "0 <= alpha_i <= C,  sum(alpha_i y_i) = 0",
        # KKT conditions
        "kkt_complementarity": "alpha_i [y_i(w^T x_i + b) - 1 + xi_i] = 0",
        "kkt_slack": "(C - alpha_i) xi_i = 0",
        # Decision function
        "decision_function": "f(x) = sign(sum_i alpha_i y_i K(x_i, x) + b)",
        "primal_from_dual": "w = sum_i alpha_i y_i x_i (only linear kernel)",
        # Kernels
        "linear_kernel": "K(x, x') = x^T x'",
        "polynomial_kernel": "K(x, x') = (x^T x' + c)^d",
        "rbf_kernel": "K(x, x') = exp(-gamma ||x - x'||^2)",
        "kernel_matrix": "K_ij = K(x_i, x_j) must be positive semi-definite",
        # Mercer's condition
        "mercer": "K is valid kernel iff K is positive semi-definite",
        # Hinge loss
        "hinge_loss": "L(y, f(x)) = max(0, 1 - y*f(x))",
        "svm_as_regularized_loss": "min (1/n) sum max(0, 1 - y_i f(x_i)) + lambda ||w||^2",
        # Generalization
        "vc_bound": "R <= R_emp + sqrt((h(log(2n/h) + 1) - log(eta/4)) / n)",
        "margin_bound": "R <= O(R^2 / (n * gamma^2)) where R = data radius, gamma = margin",
    }


def get_common_kernels() -> List[Dict[str, str]]:
    """Return list of common kernel functions with properties."""
    return [
        {
            "name": "Linear Kernel",
            "formula": "K(x, x') = x^T x'",
            "parameters": "None",
            "use_case": "Linearly separable data, high-dimensional sparse data (text)",
            "implicit_dim": "Same as input dimension",
        },
        {
            "name": "Polynomial Kernel",
            "formula": "K(x, x') = (gamma * x^T x' + c)^d",
            "parameters": "degree d, coefficient c, scale gamma",
            "use_case": "Polynomial decision boundaries, image classification",
            "implicit_dim": "O(n^d) for degree d",
        },
        {
            "name": "RBF (Gaussian) Kernel",
            "formula": "K(x, x') = exp(-gamma ||x - x'||^2)",
            "parameters": "gamma (inverse bandwidth)",
            "use_case": "General nonlinear problems, default choice",
            "implicit_dim": "Infinite (Gaussian RKHS)",
        },
        {
            "name": "Sigmoid Kernel",
            "formula": "K(x, x') = tanh(gamma * x^T x' + c)",
            "parameters": "scale gamma, offset c",
            "use_case": "Neural network-like behavior",
            "note": "Not always positive semi-definite!",
        },
        {
            "name": "Laplacian Kernel",
            "formula": "K(x, x') = exp(-gamma ||x - x'||_1)",
            "parameters": "gamma",
            "use_case": "Less sensitive to outliers than RBF",
            "implicit_dim": "Infinite",
        },
        {
            "name": "String Kernels",
            "formula": "Various (subsequence, spectrum, etc.)",
            "parameters": "Depends on specific kernel",
            "use_case": "Text, DNA sequences, structured data",
            "implicit_dim": "Depends on kernel",
        },
    ]


def get_kernel_trick_explanation() -> Dict[str, str]:
    """Return detailed explanation of the kernel trick."""
    return {
        "motivation": """
            To create nonlinear decision boundaries, we can map inputs
            to a higher-dimensional feature space where they become
            linearly separable. However, explicit computation of this
            mapping phi(x) can be expensive or impossible.
        """,
        "key_insight": """
            The SVM solution (both in training and prediction) only
            depends on dot products between data points, never on
            the individual feature vectors:

            Dual: sum_ij alpha_i alpha_j y_i y_j (x_i^T x_j)
            Prediction: sum_i alpha_i y_i (x_i^T x_new)
        """,
        "the_trick": """
            If we can compute K(x, x') = phi(x)^T phi(x') efficiently
            without explicitly computing phi(x), we get the benefits
            of high-dimensional features at low computational cost.

            Example (polynomial degree 2, 2D input):
            phi([x1, x2]) = [x1^2, sqrt(2)*x1*x2, x2^2]
            K(x, x') = (x^T x')^2 = phi(x)^T phi(x')
        """,
        "infinite_dimensions": """
            The RBF kernel corresponds to an infinite-dimensional
            feature space! The kernel implicitly computes the dot
            product in this space without ever constructing it.
        """,
        "mercer_theorem": """
            A function K(x, x') is a valid kernel (can be written as
            phi(x)^T phi(x') for some phi) if and only if the kernel
            matrix K_ij = K(x_i, x_j) is positive semi-definite for
            any choice of points.
        """,
    }


def get_svm_variants() -> List[Dict[str, str]]:
    """Return SVM variants and extensions."""
    return [
        {
            "name": "nu-SVM",
            "description": """
                Reparameterization where nu in (0, 1] controls the fraction
                of support vectors and margin errors. More intuitive than C.
            """,
            "formulation": "min (1/2)||w||^2 - nu*rho + (1/n)*sum(xi_i)  s.t. y_i(w^Tx_i + b) >= rho - xi_i",
        },
        {
            "name": "Support Vector Regression (SVR)",
            "description": """
                Adaptation for regression using epsilon-insensitive loss.
                Points within epsilon of the prediction incur no loss.
            """,
            "formulation": "min (1/2)||w||^2 + C*sum(xi_i + xi_i*)  s.t. |y_i - (w^Tx_i + b)| <= epsilon + xi_i",
        },
        {
            "name": "One-Class SVM",
            "description": """
                For novelty/anomaly detection. Finds a hyperplane separating
                data from the origin with maximum margin.
            """,
            "use_case": "Outlier detection, density estimation.",
        },
        {
            "name": "Structured SVM",
            "description": """
                Extension to structured output spaces (sequences, trees, etc.)
                using structured loss functions and inference.
            """,
            "use_case": "POS tagging, parsing, image segmentation.",
        },
        {
            "name": "Least Squares SVM (LS-SVM)",
            "description": """
                Uses squared loss instead of hinge loss, leading to a
                system of linear equations instead of QP.
            """,
            "formulation": "min (1/2)||w||^2 + C*sum(e_i^2)  where e_i = y_i - (w^Tx_i + b)",
        },
        {
            "name": "Multi-Kernel Learning",
            "description": """
                Learn a weighted combination of multiple kernels:
                K(x, x') = sum_m beta_m K_m(x, x')
            """,
            "benefit": "Automatically combine different feature types.",
        },
    ]


def get_computational_aspects() -> Dict[str, str]:
    """Return computational complexity and scaling information."""
    return {
        "training_complexity": """
            Naive QP: O(n^3) time, O(n^2) space for kernel matrix
            SMO: O(n^2) typical, up to O(n^3) worst case
            LibSVM: Highly optimized, practical for n ~ 100,000
        """,
        "prediction_complexity": """
            O(n_sv * d) for linear kernel (n_sv = number of support vectors)
            O(n_sv * kernel_cost) for nonlinear kernels
        """,
        "scaling_approaches": [
            "Chunking: solve QP on subsets",
            "SMO: optimize pairs of variables",
            "Core Vector Machine: approximate with coresets",
            "Random Fourier Features: approximate RBF kernel",
            "Nystrom approximation: low-rank kernel matrix",
            "SGD for primal: works for very large datasets",
        ],
        "linear_svm_at_scale": """
            For linear kernels, can use stochastic gradient descent
            on the primal hinge loss formulation:
            LibLinear achieves O(n) training for sparse data
        """,
    }
