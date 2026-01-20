"""
Liquid Neural Networks - Hasani et al., MIT (2021)

Continuous-time neural networks with variable time constants
that adapt based on input, enabling compact models with strong
temporal reasoning capabilities.

Paper: "Liquid Time-constant Networks"
arXiv: 2006.04439

Mathematical Formulation:
    dx/dt = -[1/tau(x,I) + f(x,I)] * x + g(x,I)

    Where:
        - x: Hidden state
        - I: Input
        - tau(x,I): Variable time constant (input-dependent)
        - f,g: Neural network functions

Key Innovation:
    Time constants that vary based on input and state,
    allowing the network to adaptively process signals
    at different timescales.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ...core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

LIQUID_NETWORK = MLMethod(
    method_id="ltc_2021",
    name="Liquid Time-constant Networks",
    year=2021,

    era=MethodEra.DEEP_LEARNING,
    category=MethodCategory.ARCHITECTURE,
    lineages=[MethodLineage.RNN_LINE],

    authors=[
        "Ramin Hasani", "Mathias Lechner", "Alexander Amini",
        "Daniela Rus", "Radu Grosu"
    ],
    paper_title="Liquid Time-constant Networks",
    paper_url="https://arxiv.org/abs/2006.04439",

    key_innovation=(
        "Introduced neural networks with input-dependent time constants, inspired by "
        "biological neurons. Unlike standard RNNs with fixed dynamics, LTCs adapt their "
        "temporal behavior based on the input signal. This enables compact models (as few "
        "as 19 neurons) to outperform much larger LSTMs on temporal tasks. The continuous-time "
        "formulation also provides theoretical stability guarantees."
    ),

    mathematical_formulation=r"""
Liquid Time-constant (LTC) Cell:
    dx/dt = -[1/tau(x,I,t) + f(x,I,t)] * x + g(x,I,t)

    Where:
        x in R^n: Hidden state
        I in R^m: Input
        t: Time

Time Constant (input-dependent):
    tau(x,I,t) = tau_base * sigma(W_tau * [x; I; bias])

    Where:
        tau_base: Base time constant
        sigma: Sigmoid (ensures tau > 0)
        W_tau: Learnable weights

Neural ODE Functions:
    f(x,I,t) = sigma(W_f * [x; I; bias])
    g(x,I,t) = W_g * [x; I; bias]

    Or with nonlinearity:
    g(x,I,t) = sigma(W_g * [x; I; bias])

Discretization (for implementation):
    Using Euler: x_{t+1} = x_t + dt * dx/dt
    Using ODE solver: x_{t+1} = ODESolve(x_t, t, t+dt, dx/dt)

Closed-form solution (special case):
    For linear f,g:
    x(t) = x_0 * exp(-integral_0^t (1/tau + f) ds) + integral terms

Stability Condition:
    If 1/tau(x,I,t) + f(x,I,t) > 0 for all x,I,t:
        System is exponentially stable
        States remain bounded
""",

    predecessors=["lstm_1997", "neural_ode_2018", "gru_2014"],
    successors=["cfc_2022"],

    tags=[
        "continuous-time", "liquid-networks", "variable-timescale",
        "neural-ode", "compact-models", "temporal-reasoning"
    ],
    notes=(
        "LTCs achieve remarkable efficiency: 19 neurons can match performance of "
        "LSTMs with 64-128 units on autonomous driving tasks. The input-dependent "
        "time constants allow the network to 'speed up' or 'slow down' based on "
        "input complexity. Connection to Closed-form Continuous-depth networks (CfC) "
        "provides analytical solutions and faster inference."
    )
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Liquid Networks."""
    return LIQUID_NETWORK


# =============================================================================
# Architecture Reference
# =============================================================================

@dataclass
class LTCArchitecture:
    """Reference architecture for Liquid Time-constant Networks."""

    # Network size
    state_dim: int = 19  # Can be very small!
    input_dim: int = 96

    # Time constant parameters
    tau_min: float = 0.001
    tau_max: float = 10.0

    # ODE solver parameters
    solver: str = "euler"  # or "rk4", "dopri5"
    dt: float = 0.01

    @staticmethod
    def ltc_cell_structure() -> str:
        """LTC cell structure."""
        return """
LTC Cell:

State Update (continuous-time):
    dx/dt = -[1/tau(x,I) + f(x,I)] * x + g(x,I)

Implementation Components:

1. Time Constant Network:
    Input: [x, I]  # concatenated state and input
    tau_raw = Linear([x, I]) -> tau_raw
    tau = tau_min + (tau_max - tau_min) * sigmoid(tau_raw)

    tau is input-dependent, always positive and bounded

2. Decay Function f:
    f_raw = Linear([x, I])
    f = sigmoid(f_raw)  # bounded [0, 1]

3. Input Function g:
    g = Linear([x, I])  # or with nonlinearity

4. State Update:
    decay_rate = 1/tau + f
    dx_dt = -decay_rate * x + g

5. Integration:
    Euler: x_new = x + dt * dx_dt
    RK4: x_new = rk4_step(x, dx_dt_fn, dt)

Intuition:
    - Large tau (slow dynamics): state changes slowly, long memory
    - Small tau (fast dynamics): state responds quickly to input
    - tau adapts based on input, enabling multi-timescale processing
"""

    @staticmethod
    def wiring_patterns() -> str:
        """Wiring patterns for LTC networks."""
        return """
Wiring Patterns (Neural Circuit Policies):

1. Fully Connected:
    Every neuron connected to every other
    Standard but computationally expensive

2. Random Sparse:
    Random subset of connections
    Reduces parameters while maintaining expressivity

3. Neural Circuit Policy (NCP):
    Biologically-inspired structure:

    Input Layer:
        Sensory neurons receive input
        Project to interneurons

    Inter Layer:
        Interneurons process information
        Recurrent connections within layer

    Command Layer:
        Command neurons aggregate
        Project to motor neurons

    Motor Layer:
        Output neurons produce actions
        Feedback to inter/command layers

    Structure: Input -> Inter <-> Command -> Motor
                           ^          |
                           |__________|

4. Advantages of structured wiring:
    - Fewer parameters (sparse)
    - Better interpretability
    - Matches biological priors
    - Can be very compact (19 neurons sufficient!)
"""

    @staticmethod
    def closed_form_variant() -> str:
        """Closed-form Continuous-depth (CfC) networks."""
        return """
Closed-form Continuous-depth Networks (CfC):

Problem with Neural ODEs:
    Require numerical integration (slow)
    Memory issues for long sequences

CfC Solution:
    Derive closed-form solution for LTC equations
    No numerical integration needed!

For LTC with specific functional forms:
    dx/dt = -[1/tau + f] * x + g

Solution (time-invariant case):
    x(t+Delta_t) = x(t) * exp(-[1/tau + f] * Delta_t)
                 + g * tau_eff * (1 - exp(-[1/tau + f] * Delta_t))

Where:
    tau_eff = 1 / (1/tau + f)

CfC Cell (one-step update):
    sigma_t = sigmoid(time_linear(x, I))
    tau_eff = tau_base * sigma_t
    f_val = sigmoid(f_linear(x, I))

    # Closed-form update
    exp_term = exp(-Delta_t / tau_eff)
    x_new = x * exp_term + (1 - exp_term) * f_val * g(x, I)

Benefits:
    - No ODE solver needed
    - Constant memory (no backprop through solver)
    - Much faster training and inference
    - Same expressive power as LTC
"""


# =============================================================================
# Mathematical Functions (Reference)
# =============================================================================

def ltc_dynamics(x, I, tau_net, f_net, g_net):
    """
    Compute LTC dynamics.

    dx/dt = -[1/tau(x,I) + f(x,I)] * x + g(x,I)

    Args:
        x: Current state [batch, state_dim]
        I: Input [batch, input_dim]
        tau_net: Network computing time constant
        f_net: Network computing decay rate
        g_net: Network computing input contribution

    Returns:
        dx/dt: State derivative
    """
    return {
        "formula": """
            tau = tau_net([x, I])  # Input-dependent time constant
            f = f_net([x, I])      # Additional decay
            g = g_net([x, I])      # Input contribution

            dx_dt = -(1/tau + f) * x + g
        """,
        "interpretation": """
            - (1/tau + f) * x: Exponential decay term
            - g: Driving input
            - Large tau -> slow decay -> long memory
            - Small tau -> fast decay -> responsive to input
        """,
        "stability": "System stable if 1/tau + f > 0"
    }


def variable_timescale_intuition():
    """
    Intuition for variable time constants.

    The key insight is that different inputs may require
    processing at different timescales.
    """
    return {
        "example_autonomous_driving": """
            Scenario 1: Highway, steady driving
                - tau is large (slow dynamics)
                - Network maintains stable internal state
                - Smooths over minor fluctuations

            Scenario 2: Intersection, complex activity
                - tau is small (fast dynamics)
                - Network responds quickly to changes
                - Processes rapid sequences of events
        """,
        "comparison_to_lstm": """
            LSTM: Fixed forget gate dynamics
                  Can learn when to remember/forget
                  But timescale is data-independent at inference

            LTC: Variable time constant
                 Timescale adapts to input in real-time
                 More flexible temporal processing
        """,
        "biological_analogy": """
            Biological neurons have variable time constants:
            - Adaptation: repeated stimulus -> longer tau
            - Neuromodulation: dopamine/ACh change dynamics
            - Dendritic properties: vary across neuron

            LTCs capture this input-dependent temporal flexibility.
        """
    }


def ode_solver_comparison():
    """
    Comparison of ODE solvers for neural ODEs / LTCs.
    """
    return {
        "euler": {
            "formula": "x_{t+1} = x_t + dt * f(x_t, t)",
            "order": 1,
            "pros": "Fast, simple",
            "cons": "Requires small dt for accuracy"
        },
        "rk4": {
            "formula": """
                k1 = f(x, t)
                k2 = f(x + dt/2 * k1, t + dt/2)
                k3 = f(x + dt/2 * k2, t + dt/2)
                k4 = f(x + dt * k3, t + dt)
                x_{t+1} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            """,
            "order": 4,
            "pros": "More accurate per step",
            "cons": "4x function evaluations"
        },
        "dopri5": {
            "description": "Dormand-Prince adaptive step size",
            "order": 5,
            "pros": "Adaptive dt, handles stiff systems",
            "cons": "Complex, memory for backprop"
        },
        "cfc_closed_form": {
            "description": "Closed-form solution (no solver)",
            "pros": "Fast, constant memory, exact",
            "cons": "Requires specific functional form"
        }
    }


# =============================================================================
# Key Insights and Applications
# =============================================================================

LIQUID_NETWORK_INSIGHTS = {
    "compactness": """
        LTCs achieve remarkable compactness:
        - 19 neurons can control autonomous vehicle
        - Outperforms LSTM with 64-128 units
        - Entire network can be visualized and understood

        Why? Variable time constants provide more expressivity
        per neuron than fixed-dynamics alternatives.
    """,

    "interpretability": """
        Small LTCs are interpretable:
        - Can visualize all neuron activities
        - Can trace information flow
        - Can identify what each neuron represents

        This enables verification for safety-critical applications.
    """,

    "continuous_time": """
        Continuous-time formulation has advantages:
        - Natural for irregular time series
        - Theoretical stability guarantees
        - No need to choose fixed time step
        - Elegant mathematical framework
    """,

    "biological_plausibility": """
        More biologically plausible than LSTMs:
        - Time constants exist in real neurons
        - Input-dependent dynamics observed in biology
        - Sparse, structured connectivity like neural circuits
    """
}


LIQUID_NETWORK_APPLICATIONS = {
    "autonomous_driving": {
        "task": "Steering angle prediction from camera",
        "advantage": "Compact, interpretable, safe",
        "result": "19-neuron LTC matches 128-unit LSTM"
    },
    "medical_time_series": {
        "task": "ICU patient monitoring, early warning",
        "advantage": "Handles irregular sampling naturally",
        "result": "Strong performance on PhysioNet tasks"
    },
    "robotics": {
        "task": "Continuous control tasks",
        "advantage": "Real-time execution on embedded systems",
        "result": "Efficient inference, interpretable policies"
    },
    "natural_language": {
        "task": "Sentiment, classification",
        "advantage": "Variable-speed processing of text",
        "note": "CfC variant shows competitive results"
    }
}


LIQUID_VS_OTHER_RNNS = {
    "vs_lstm": {
        "dynamics": "Continuous-time vs discrete gates",
        "timescale": "Variable vs learned-but-fixed",
        "size": "Much smaller for same performance",
        "theory": "Stability guarantees vs empirical"
    },
    "vs_gru": {
        "dynamics": "Continuous vs discrete",
        "gates": "Time constant vs reset/update",
        "efficiency": "Similar parameter count, different tradeoffs"
    },
    "vs_neural_ode": {
        "time_constant": "Variable vs fixed/none",
        "closed_form": "CfC variant vs numerical integration",
        "structure": "NCP wiring vs arbitrary"
    }
}
