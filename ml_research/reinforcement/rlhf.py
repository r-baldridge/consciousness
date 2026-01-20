"""
Reinforcement Learning from Human Feedback (RLHF) - 2017/2022

Framework for aligning AI systems with human preferences using RL.
Trains a reward model from human preference data, then optimizes
the policy to maximize learned reward while staying close to base model.

Papers:
    "Deep Reinforcement Learning from Human Preferences" (Christiano et al., 2017)
    "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)

Mathematical Formulation:
    Reward Model:
        r(x, y) trained to predict human preferences
        Loss: -log(sigma(r(x, y_w) - r(x, y_l)))  (Bradley-Terry model)

    PPO with KL Penalty:
        maximize E[r(x,y)] - beta * KL(pi || pi_ref)

Key Applications:
    - InstructGPT, ChatGPT, Claude
    - LLM fine-tuning for helpfulness, harmlessness, honesty
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from ..core.taxonomy import MLMethod, MethodEra, MethodCategory, MethodLineage


# =============================================================================
# MLMethod Research Index Entry
# =============================================================================

RLHF = MLMethod(
    method_id="rlhf_2017",
    name="Reinforcement Learning from Human Feedback",
    year=2017,

    era=MethodEra.ATTENTION,
    category=MethodCategory.REINFORCEMENT,
    lineages=[MethodLineage.RL_LINE],

    authors=["Paul Christiano", "Jan Leike", "Tom Brown", "Miljan Martic",
             "Shane Legg", "Dario Amodei"],
    paper_title="Deep Reinforcement Learning from Human Preferences",
    paper_url="https://arxiv.org/abs/1706.03741",

    key_innovation=(
        "Enabled learning from human preferences rather than hand-crafted reward "
        "functions. A reward model is trained from human comparisons, then used to "
        "guide RL. Applied to LLMs, this became the key technique for aligning "
        "language models (InstructGPT, ChatGPT, Claude)."
    ),

    mathematical_formulation=r"""
RLHF Three-Stage Process:

Stage 1: Supervised Fine-tuning (SFT)
    Train policy pi_SFT on high-quality demonstrations
    L_SFT = -E[log pi(y|x)]
    (Standard language modeling on curated data)

Stage 2: Reward Model Training
    Collect preferences: human prefers y_w over y_l for prompt x
    Bradley-Terry preference model:
        P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))

    Reward model loss:
        L_RM = -E[log sigma(r(x, y_w) - r(x, y_l))]

    Often initialized from SFT model with new head

Stage 3: RL Fine-tuning (PPO)
    Objective:
        maximize E_{x~D, y~pi}[r(x, y)] - beta * KL(pi || pi_ref)

    Per-token formulation:
        R_t = { 0                           if t < T
              { r(x, y) - beta * KL_token   if t = T

    KL penalty prevents:
        - Reward hacking (exploiting reward model)
        - Catastrophic forgetting
        - Distribution shift

PPO Objective for RLHF:
    L = E[min(r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t)]

    Where advantage A uses learned reward r(x, y)

Alternative: DPO (Direct Preference Optimization)
    Skip reward model, optimize directly:
    L_DPO = -E[log sigma(beta * (log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)))]
""",

    predecessors=["ppo_2017", "gpt_2018", "preference_learning"],
    successors=["instructgpt", "chatgpt", "constitutional_ai", "dpo"],

    tags=["rlhf", "alignment", "llm", "human-feedback", "preference-learning", "policy-optimization"]
)

# Historical Note:
# RLHF transformed language model capabilities from next-token prediction to
# following human intent. InstructGPT (2022) demonstrated dramatic improvements
# in helpfulness and safety over GPT-3. The approach has limitations: reward
# hacking, preference inconsistency, and specification gaming. Alternatives like
# DPO and Constitutional AI address some issues.


# =============================================================================
# Detailed Pipeline
# =============================================================================

def reward_model_loss(preferred_rewards: List[float],
                      rejected_rewards: List[float]) -> Dict:
    """
    Bradley-Terry pairwise preference loss for reward model.

    L = -log(sigma(r_w - r_l))
      = -log(1 / (1 + exp(-(r_w - r_l))))
      = log(1 + exp(r_l - r_w))

    Args:
        preferred_rewards: r(x, y_w) for preferred completions
        rejected_rewards: r(x, y_l) for rejected completions

    Returns:
        Reward model loss
    """
    return {
        "formula": "L = -log(sigma(r_w - r_l))",
        "simplified": "L = log(1 + exp(r_l - r_w))",
        "gradient": "Increase r_w, decrease r_l until well-separated",
        "margin": "Effective margin learned from data"
    }


def rlhf_objective(reward: float, kl_divergence: float,
                   beta: float = 0.1) -> Dict:
    """
    RLHF objective combining reward and KL penalty.

    J = E[r(x, y)] - beta * KL(pi || pi_ref)

    Args:
        reward: Learned reward r(x, y)
        kl_divergence: KL from reference policy
        beta: KL penalty coefficient

    Returns:
        Objective value and interpretation
    """
    return {
        "formula": "J = E[r(x, y)] - beta * KL(pi || pi_ref)",
        "reward_term": "Maximize learned human preference",
        "kl_term": "Stay close to reference (prevent reward hacking)",
        "beta_interpretation": {
            "high_beta": "Conservative, closer to reference",
            "low_beta": "More reward optimization, risk of hacking"
        }
    }


def dpo_loss(pi_w: float, pi_l: float, ref_w: float, ref_l: float,
             beta: float = 0.1) -> Dict:
    """
    Direct Preference Optimization loss (alternative to RLHF).

    L_DPO = -log sigma(beta * (log(pi_w/ref_w) - log(pi_l/ref_l)))

    This implicitly optimizes the RLHF objective without explicit reward model.

    Args:
        pi_w: pi(y_w | x) probability of preferred under new policy
        pi_l: pi(y_l | x) probability of rejected under new policy
        ref_w: pi_ref(y_w | x) probability under reference
        ref_l: pi_ref(y_l | x) probability under reference
        beta: Temperature parameter

    Returns:
        DPO loss
    """
    return {
        "formula": "L = -log sigma(beta * (log(pi_w/ref_w) - log(pi_l/ref_l)))",
        "implicit_reward": "r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))",
        "advantage": "No reward model training needed",
        "disadvantage": "Less flexible, reward not explicit"
    }


# =============================================================================
# Algorithm Reference
# =============================================================================

@dataclass
class RLHFPipeline:
    """Reference implementation structure for RLHF."""

    # Stage 1: SFT
    sft_epochs: int = 3
    sft_lr: float = 1e-5

    # Stage 2: Reward Model
    rm_epochs: int = 1
    rm_lr: float = 1e-5
    comparison_batch_size: int = 64

    # Stage 3: PPO
    ppo_epochs: int = 4
    ppo_lr: float = 1e-6
    kl_coef: float = 0.1
    clip_eps: float = 0.2
    batch_size: int = 512
    mini_batch_size: int = 64

    @staticmethod
    def full_pipeline() -> str:
        """Complete RLHF training pipeline."""
        return """
RLHF Training Pipeline:

Stage 1: Supervised Fine-Tuning (SFT)
    Input: Base pretrained LLM, demonstration dataset

    for epoch in range(sft_epochs):
        for (prompt, response) in demonstrations:
            loss = -log P(response | prompt; theta)
            theta <- theta - lr * nabla loss

    Output: pi_SFT (used as initialization and reference)

Stage 2: Reward Model Training
    Input: pi_SFT, comparison dataset
    Initialize: rm_theta from pi_SFT with reward head

    for epoch in range(rm_epochs):
        for (prompt, y_w, y_l) in comparisons:
            r_w = reward_model(prompt, y_w)
            r_l = reward_model(prompt, y_l)
            loss = -log(sigmoid(r_w - r_l))
            rm_theta <- rm_theta - lr * nabla loss

    Output: r(x, y) reward model

Stage 3: PPO Fine-Tuning
    Input: pi_SFT (as pi_ref), reward model r, prompt distribution
    Initialize: pi = pi_SFT

    for iteration in range(max_iters):
        # Collect rollouts
        for prompt in batch:
            response ~ pi(. | prompt)
            reward = r(prompt, response)
            kl = KL(pi(.|prompt) || pi_ref(.|prompt))
            total_reward = reward - beta * kl

        # Compute advantages (GAE)
        advantages = compute_gae(rewards, values)

        # PPO update
        for epoch in range(ppo_epochs):
            for mini_batch in shuffle(batch):
                ratio = pi(a|s) / pi_old(a|s)
                L_clip = min(ratio * A, clip(ratio) * A)
                L_value = (V - R)^2
                loss = -L_clip + vf_coef * L_value
                theta <- theta - lr * nabla loss

    Output: Aligned policy pi
"""

    @staticmethod
    def reward_model_architecture() -> str:
        """Reward model architecture."""
        return """
Reward Model Architecture:

Option 1: Shared backbone with reward head
    base_model = pretrained_llm  # Same as SFT
    reward_head = Linear(hidden_dim, 1)

    def forward(prompt, response):
        tokens = tokenize(prompt + response)
        hidden = base_model(tokens)
        last_hidden = hidden[:, -1, :]  # Last token
        reward = reward_head(last_hidden)
        return reward.squeeze()

Option 2: Separate reward model (larger scale)
    Dedicated model trained from scratch on preferences
    Often 6B-13B parameters

Training Tips:
    - Use dropout for regularization
    - Train for 1 epoch (overfitting is common)
    - Normalize rewards during PPO
    - Use reward model ensemble for robustness
"""


# =============================================================================
# Key Papers and Timeline
# =============================================================================

RLHF_TIMELINE = {
    "2017": {
        "paper": "Deep RL from Human Preferences",
        "authors": "Christiano et al.",
        "contribution": "RLHF framework for MuJoCo, Atari",
        "key_idea": "Learn reward from comparisons, optimize with RL"
    },
    "2019": {
        "paper": "Fine-Tuning Language Models from Human Preferences",
        "authors": "Ziegler et al.",
        "contribution": "First application to language models (GPT-2)",
        "task": "Summarization, continuation"
    },
    "2020": {
        "paper": "Learning to Summarize from Human Feedback",
        "authors": "Stiennon et al.",
        "contribution": "Improved summarization with RLHF",
        "result": "Human-preferred summaries over supervised"
    },
    "2022": {
        "paper": "Training language models to follow instructions",
        "authors": "Ouyang et al.",
        "contribution": "InstructGPT - RLHF at scale",
        "result": "GPT-3 + RLHF >> GPT-3",
        "model": "InstructGPT (precursor to ChatGPT)"
    },
    "2023": {
        "developments": [
            "ChatGPT (OpenAI) - RLHF-trained GPT-3.5/4",
            "Claude (Anthropic) - Constitutional AI + RLHF",
            "DPO - Direct optimization without reward model"
        ]
    }
}


# =============================================================================
# Challenges and Alternatives
# =============================================================================

RLHF_CHALLENGES = {
    "reward_hacking": {
        "problem": "Policy exploits reward model imperfections",
        "examples": [
            "Verbose but unhelpful responses",
            "Sycophantic agreement",
            "Exploiting formatting preferences"
        ],
        "mitigations": [
            "KL penalty to reference",
            "Reward model ensembles",
            "Ongoing human evaluation"
        ]
    },
    "distribution_shift": {
        "problem": "Policy generates OOD text, reward model unreliable",
        "mitigation": "KL constraint, iterative training"
    },
    "preference_inconsistency": {
        "problem": "Human preferences can be inconsistent, noisy",
        "mitigation": "Multiple annotators, consensus, uncertainty modeling"
    },
    "scalability": {
        "problem": "Human feedback is expensive",
        "solutions": [
            "AI feedback (Constitutional AI)",
            "Synthetic preferences",
            "Efficient comparison formats"
        ]
    }
}


RLHF_ALTERNATIVES = {
    "dpo": {
        "name": "Direct Preference Optimization",
        "year": 2023,
        "idea": "Optimize preference objective directly without RM",
        "advantage": "Simpler, no reward model",
        "disadvantage": "Less interpretable, no explicit reward"
    },
    "rrhf": {
        "name": "Rank Responses to align Human Feedback",
        "idea": "Use ranking loss directly"
    },
    "constitutional_ai": {
        "name": "Constitutional AI",
        "year": 2022,
        "authors": "Anthropic",
        "idea": "AI provides feedback based on constitution/principles",
        "benefit": "Scalable, consistent feedback"
    },
    "ipo": {
        "name": "Identity Preference Optimization",
        "improvement": "Better theoretical grounding than DPO"
    },
    "kto": {
        "name": "Kahneman-Tversky Optimization",
        "idea": "Model human decision-making biases"
    }
}
