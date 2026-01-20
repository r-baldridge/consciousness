"""
Differentiable Programming - Concepts and Techniques

The paradigm of making traditionally discrete, non-differentiable operations
differentiable so they can be learned end-to-end with gradient descent.
Key techniques include soft attention, Gumbel-softmax, and relaxations
of discrete structures.

Key Concepts:
    - Soft attention: Differentiable selection via weighted sums
    - Gumbel-softmax: Differentiable sampling from categorical distributions
    - Neural Module Networks: Composing neural modules based on input
    - Program Synthesis: Learning to generate programs from examples

Mathematical Formulation:
    Core idea: Replace argmax with softmax

    Hard selection: y = x_{argmax(s)}
    Soft selection: y = sum_i softmax(s)_i * x_i

    Gumbel-softmax (for sampling):
        y_i = exp((log(p_i) + g_i) / tau) / sum_j exp((log(p_j) + g_j) / tau)
        where g_i ~ Gumbel(0, 1)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

DIFFERENTIABLE_PROGRAMMING = MLMethod(
    method_id="differentiable_programming",
    name="Differentiable Programming",
    year=2015,  # Approximate, spans multiple works

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.OPTIMIZATION,
    lineages=[MethodLineage.PERCEPTRON_LINE, MethodLineage.ATTENTION_LINE],

    authors=["Various"],  # Multiple contributors
    paper_title="Various - Key papers: Gumbel-Softmax (2016), Soft Attention (Bahdanau 2014)",
    paper_url="https://arxiv.org/abs/1611.01144",  # Gumbel-softmax paper

    key_innovation=(
        "Making traditionally non-differentiable operations (selection, sampling, "
        "discrete choices) differentiable to enable end-to-end gradient-based "
        "learning. Key techniques include soft attention (weighted sums instead "
        "of hard selection), Gumbel-softmax (differentiable categorical sampling), "
        "and continuous relaxations of discrete structures."
    ),

    mathematical_formulation=r"""
Core Principle:
    Replace discrete/non-differentiable operations with continuous approximations

Hard Selection vs Soft Selection:
    Hard (non-differentiable):
        y = e_{argmax_i(s_i)}  (one-hot selection)
        Gradient: undefined (argmax not differentiable)

    Soft (differentiable):
        y = sum_i softmax(s / tau)_i * x_i
        Gradient: d(y)/d(s) exists and is non-zero

Temperature tau controls hardness:
    - tau -> 0: Approaches hard selection
    - tau -> inf: Uniform weighting

Straight-Through Estimator:
    Forward: Use hard (discrete) output
    Backward: Use soft gradient

    y = one_hot(argmax(s))  (forward)
    d(L)/d(s) = d(L)/d(y) * d(softmax(s))/d(s)  (backward)

Gumbel-Softmax:
    Sample from categorical without argmax:

    y_i = exp((log(p_i) + g_i) / tau) / sum_j exp((log(p_j) + g_j) / tau)

    Where g_i = -log(-log(u_i)), u_i ~ Uniform(0, 1)

    Properties:
        - E[y] -> p as tau -> inf
        - y -> one_hot(argmax) as tau -> 0
        - Differentiable for tau > 0
""",

    predecessors=["backpropagation", "attention_mechanism"],
    successors=["neural_architecture_search", "neural_program_synthesis"],

    tags=["differentiable", "relaxation", "soft-attention", "program-synthesis"]
)

# Note: Differentiable programming is a paradigm shift: instead of writing
# programs, we write program templates with learnable parameters and
# optimize end-to-end. This enables learning of complex structured
# computations that would be intractable to specify by hand.


# =============================================================================
# Soft Attention Mechanisms
# =============================================================================

def soft_attention_mechanisms() -> Dict[str, Dict]:
    """
    Soft attention as differentiable selection.

    Returns:
        Dictionary describing soft attention variants
    """
    return {
        "bahdanau_attention": {
            "year": 2014,
            "paper": "Neural Machine Translation by Jointly Learning to Align and Translate",
            "formulation": """
Score: e_{ij} = v^T tanh(W_h h_i + W_s s_j)
Attention: alpha_{ij} = softmax_i(e_{ij})
Context: c_j = sum_i alpha_{ij} * h_i

This is a differentiable selection mechanism:
    - Without attention: Select one h_i (non-differentiable)
    - With attention: Weighted sum of all h_i (differentiable)
""",
            "key_insight": "Soft alignment replaces hard alignment in translation"
        },
        "luong_attention": {
            "year": 2015,
            "paper": "Effective Approaches to Attention-based Neural Machine Translation",
            "variants": {
                "dot": "e_ij = h_i^T s_j",
                "general": "e_ij = h_i^T W s_j",
                "concat": "e_ij = v^T tanh(W [h_i; s_j])"
            }
        },
        "hard_vs_soft": {
            "hard_attention": """
Hard Attention:
    z_j = h_{argmax_i(alpha_{ij})}  (select single location)

    Problems:
        - argmax not differentiable
        - Requires REINFORCE for training (high variance)
        - Used in visual attention models (Xu et al. 2015)
""",
            "soft_attention": """
Soft Attention:
    z_j = sum_i alpha_{ij} * h_i  (weighted sum)

    Advantages:
        - Fully differentiable
        - Can be trained with backpropagation
        - Lower variance than hard attention
"""
        },
        "differentiable_selection_examples": [
            "Memory access (Neural Turing Machine)",
            "Pointer Networks (selecting from input)",
            "Copy mechanisms (copying vs generating)",
            "Attention over vocabulary (output distribution)"
        ]
    }


# =============================================================================
# Gumbel-Softmax
# =============================================================================

def gumbel_softmax() -> Dict[str, str]:
    """
    Gumbel-Softmax trick for differentiable categorical sampling.

    Returns:
        Dictionary describing Gumbel-softmax technique
    """
    return {
        "problem": """
Problem: Sampling from categorical distribution is non-differentiable

    z ~ Categorical(p_1, ..., p_k)  (one-hot sample)

    d(z)/d(p) is undefined - can't backprop through sampling

Need: Reparameterization trick for categorical (like VAE's for Gaussian)
""",
        "gumbel_max_trick": """
Gumbel-Max Trick (non-differentiable):
    z = one_hot(argmax_i(log(p_i) + g_i))

    Where g_i = -log(-log(u_i)), u_i ~ Uniform(0, 1)

    This produces exact categorical samples, but argmax is still not differentiable.
""",
        "gumbel_softmax": """
Gumbel-Softmax (differentiable):
    y_i = exp((log(p_i) + g_i) / tau) / sum_j exp((log(p_j) + g_j) / tau)

    Properties:
        - Differentiable w.r.t. parameters of p
        - As tau -> 0: y approaches one-hot (hard)
        - As tau -> inf: y approaches uniform
        - y_i always in (0, 1) for tau > 0

Papers:
    - "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2016)
    - "The Concrete Distribution" (Maddison et al., 2016)
""",
        "straight_through_gumbel": """
Straight-Through Gumbel-Softmax:
    Forward: y = one_hot(argmax(log(p) + g))  (hard sample)
    Backward: Gradient as if y = softmax((log(p) + g) / tau)

    Code sketch:
        y_soft = gumbel_softmax(logits, tau)
        y_hard = one_hot(argmax(y_soft))
        y = y_hard - y_soft.detach() + y_soft  # Straight-through

    Benefit: Hard samples during forward, smooth gradients during backward
""",
        "temperature_annealing": """
Temperature Annealing:
    Start with high tau (soft), anneal to low tau (hard)

    tau_t = max(tau_min, tau_0 * exp(-anneal_rate * t))

    Rationale:
        - High tau: Easy optimization, but biased samples
        - Low tau: Accurate samples, but high variance gradients
        - Annealing: Get best of both
""",
        "applications": [
            "Variational inference with discrete latents",
            "Neural architecture search (operation selection)",
            "Discrete communication in multi-agent systems",
            "Hard attention mechanisms",
            "Structured prediction"
        ]
    }


# =============================================================================
# Neural Module Networks
# =============================================================================

def neural_module_networks() -> Dict[str, str]:
    """
    Neural Module Networks for compositional reasoning.

    Returns:
        Dictionary describing Neural Module Networks
    """
    return {
        "concept": """
Neural Module Networks (NMN):
    Compose neural network modules dynamically based on input

    Key idea:
        - Parse input (e.g., question) into program structure
        - Execute program using neural modules
        - Modules are reusable and composable

    Example (Visual QA):
        Question: "What color is the object to the left of the dog?"
        Program: color(left_of(find(dog)))
        Execution: Neural modules for find, left_of, color composed together
""",
        "paper": """
Papers:
    - "Neural Module Networks" (Andreas et al., 2016)
    - "Learning to Compose Neural Networks for Question Answering" (2016)
    - "Inferring and Executing Programs for Visual Reasoning" (Johnson et al., 2017)
""",
        "modules": """
Example Modules:
    find[X]: Attend to instances of X in image
        Input: Image features (H x W x C)
        Output: Attention map (H x W)

    filter[X]: Filter attention by property X
        Input: Attention map, image features
        Output: Filtered attention map

    relate[R]: Find objects in relation R to attended region
        Input: Attention map, image features
        Output: Attention map of related objects

    count: Count attended objects
        Input: Attention map
        Output: Scalar count

    query[A]: Query attribute A of attended object
        Input: Attention map, image features
        Output: Attribute value/distribution
""",
        "program_generation": """
Program Generation:
    Option 1: Parse question with semantic parser
        - Pre-defined grammar
        - Question -> symbolic program
        - Modules instantiated from program

    Option 2: Learn program layout
        - Policy network generates program
        - Trained end-to-end or with RL
        - More flexible but harder to train

    Option 3: Soft module selection (differentiable)
        - All modules computed in parallel
        - Output is weighted combination
        - Fully differentiable
""",
        "differentiable_composition": """
Making Module Composition Differentiable:

Problem: Discrete program structure is non-differentiable

Solutions:
    1. Pre-compute all possible structures, weight outputs
       y = sum_{prog p} p(prog|question) * execute(prog, image)

    2. Soft module selection at each step
       output = sum_m alpha_m * module_m(input)

    3. End-to-end with RL (REINFORCE)
       Program sampled, reward based on answer correctness

    4. Neuro-symbolic with differentiable rendering
       Symbolic execution with differentiable primitives
"""
    }


# =============================================================================
# Program Synthesis
# =============================================================================

def program_synthesis() -> Dict[str, Dict]:
    """
    Neural program synthesis approaches.

    Returns:
        Dictionary describing program synthesis methods
    """
    return {
        "problem_definition": {
            "description": """
Program Synthesis: Learn to generate programs from specifications

Specifications can be:
    - Input-output examples
    - Natural language descriptions
    - Formal constraints

Challenge: Programs are discrete, structured objects
Solution: Make program generation differentiable
"""
        },
        "neural_programmer": {
            "year": 2015,
            "paper": "Neural Programmer: Inducing Latent Programs with Gradient Descent",
            "approach": """
Neural Programmer:
    - Fixed set of operations (arithmetic, logic, selection)
    - Learn soft selection over operations at each step
    - Output = weighted sum of all operation outputs

    Differentiable because:
        - Each operation is differentiable
        - Operation selection via softmax is differentiable
        - Composition through time
""",
            "limitation": "Operations must all be differentiable"
        },
        "neural_programmer_interpreter": {
            "year": 2015,
            "paper": "Neural Programmer-Interpreters",
            "approach": """
Neural Programmer-Interpreter (NPI):
    - Learn to trace program execution
    - Core LSTM + environment-specific encoders
    - Key-value program memory

    Program = (name, arguments)
    At each step: Generate next (program, arguments) or terminate

    Trained on execution traces (supervised)
""",
            "features": ["Compositional", "Recursive", "Domain-specific"]
        },
        "differentiable_forth": {
            "approach": """
Differentiable Forth (diff4th):
    - Stack-based programming language
    - Make stack operations differentiable
    - Program counter via soft attention over instructions
    - Stack entries via soft attention over positions

    Enables: Learning stack-based algorithms end-to-end
"""
        },
        "neuro_symbolic_synthesis": {
            "approach": """
Neuro-Symbolic Program Synthesis:
    1. Neural network proposes program sketch
    2. Symbolic search fills in details
    3. Program executed to verify

    Example: DreamCoder
        - Learns library of reusable components
        - Compression-based objective
        - Wake-sleep training
""",
            "papers": [
                "DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning",
                "Learning to Infer Program Sketches"
            ]
        }
    }


# =============================================================================
# Advanced Techniques
# =============================================================================

ADVANCED_TECHNIQUES = {
    "relaxed_discrete_structures": {
        "graphs": """
Differentiable Graph Generation:
    - Adjacency as continuous [0, 1] values
    - Edge existence via sigmoid/softmax
    - Examples: Neural relational inference, graph VAE
""",
        "trees": """
Differentiable Trees:
    - Soft tree structure via attention
    - Path probability through soft gating
    - Examples: Neural Decision Trees, NALU
""",
        "sequences": """
Differentiable Sequence Operations:
    - Sorting: Sinkhorn operator for soft permutation
    - Top-k: Differentiable top-k selection
    - Masking: Soft masks via sigmoid
"""
    },
    "implicit_differentiation": {
        "description": """
Implicit Differentiation:
    For optimization problems y* = argmin f(x, y):

    dy*/dx can be computed without unrolling optimization

    By implicit function theorem:
        dy*/dx = -(d^2f/dy^2)^{-1} * (d^2f/dxdy)

    Enables: Differentiating through optimization (meta-learning, bilevel)
""",
        "examples": [
            "DEQ (Deep Equilibrium Models): Differentiate through fixed point",
            "OptNet: Differentiate through QP solver",
            "Meta-learning: Differentiate through inner loop optimization"
        ]
    },
    "score_function_gradient": {
        "description": """
Score Function Gradient (REINFORCE):
    When sampling is truly discrete and can't be relaxed:

    E_{z~p(z|theta)}[f(z)]

    Gradient: E[f(z) * grad_theta log p(z|theta)]

    Used when:
        - Gumbel-softmax too biased
        - Need exact discrete samples
        - f is non-differentiable

    Variance reduction: baselines, control variates
"""
    }
}
