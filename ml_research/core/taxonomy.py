"""
Taxonomy module for ML Research.

Defines all core enums and dataclasses used throughout the ml_research module
for categorizing, tracking, and documenting machine learning methods.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class MethodEra(Enum):
    """
    Historical eras of machine learning development.

    Each era represents a distinct period characterized by specific
    breakthroughs and dominant paradigms in ML research.
    """
    FOUNDATIONAL = "foundational"      # 1943-1980: McCulloch-Pitts, Perceptron, Backprop origins
    CLASSICAL = "classical"            # 1980-2006: SVMs, Decision Trees, Ensemble methods
    DEEP_LEARNING = "deep_learning"    # 2006-2017: Deep belief nets, CNNs, RNNs, AlexNet
    ATTENTION = "attention"            # 2017-present: Transformers, BERT, GPT
    NOVEL = "novel"                    # 2023+: Emerging methods, multimodal, agents


class MethodCategory(Enum):
    """
    Functional categories of ML methods.

    Classifies methods by their primary function or role in the
    machine learning pipeline.
    """
    NEURON_MODEL = "neuron_model"          # Basic computational units (perceptron, ReLU, etc.)
    LEARNING_RULE = "learning_rule"        # How weights are updated (backprop, Hebbian, etc.)
    ARCHITECTURE = "architecture"          # Network structures (CNN, RNN, Transformer, etc.)
    OPTIMIZATION = "optimization"          # Training algorithms (SGD, Adam, etc.)
    REGULARIZATION = "regularization"      # Preventing overfitting (dropout, batch norm, etc.)
    ATTENTION = "attention"                # Attention mechanisms (self-attention, cross-attention)
    GENERATIVE = "generative"              # Generative models (GANs, VAEs, diffusion, etc.)
    REINFORCEMENT = "reinforcement"        # RL methods (Q-learning, policy gradient, etc.)
    MULTIMODAL = "multimodal"              # Multi-modal learning (CLIP, Flamingo, etc.)


class MethodLineage(Enum):
    """
    Evolutionary lineages of ML methods.

    Tracks the intellectual ancestry and descent of methods,
    grouping related innovations into coherent lines of development.
    """
    PERCEPTRON_LINE = "perceptron"         # Perceptron -> MLP -> Deep networks
    CNN_LINE = "cnn"                       # Neocognitron -> LeNet -> AlexNet -> ResNet
    RNN_LINE = "rnn"                       # Simple RNN -> LSTM -> GRU -> Seq2Seq
    ATTENTION_LINE = "attention"           # Attention -> Self-attention -> Transformer
    GENERATIVE_LINE = "generative"         # RBM -> VAE -> GAN -> Diffusion
    RL_LINE = "reinforcement"              # Q-learning -> DQN -> PPO -> RLHF


@dataclass
class MLMethod:
    """
    Comprehensive representation of a machine learning method.

    Captures all relevant information about an ML method including
    its historical context, mathematical formulation, relationships
    to other methods, and practical implementations.

    Attributes:
        method_id: Unique identifier for the method (e.g., "transformer_2017")
        name: Human-readable name (e.g., "Transformer")
        year: Year of introduction
        era: Historical era when the method was introduced
        category: Functional category of the method
        lineages: List of evolutionary lineages this method belongs to
        authors: List of author names
        paper_title: Title of the original paper
        paper_url: URL to the paper (arXiv, DOI, etc.)
        key_innovation: Brief description of what made this method novel
        mathematical_formulation: LaTeX or plain text mathematical description
        pseudocode: Optional pseudocode implementation
        predecessors: List of method_ids this method builds upon
        successors: List of method_ids that build upon this method
        implementations: Dict mapping framework names to code/URLs
        benchmarks: Dict mapping benchmark names to scores
        tags: List of searchable tags
    """
    method_id: str
    name: str
    year: int
    era: MethodEra
    category: MethodCategory
    lineages: List[MethodLineage]
    authors: List[str]
    paper_title: str
    paper_url: Optional[str]
    key_innovation: str
    mathematical_formulation: str
    pseudocode: Optional[str] = None
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    implementations: Dict[str, str] = field(default_factory=dict)
    benchmarks: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate method data after initialization."""
        if not self.method_id:
            raise ValueError("method_id cannot be empty")
        if self.year < 1943:
            raise ValueError("year cannot be before 1943 (McCulloch-Pitts neuron)")
        if not self.lineages:
            raise ValueError("method must belong to at least one lineage")


@dataclass
class Paper:
    """
    Representation of an academic paper in machine learning.

    Stores metadata about papers that introduced or significantly
    advanced ML methods.

    Attributes:
        paper_id: Unique identifier (e.g., "vaswani2017attention")
        title: Full paper title
        authors: List of author names
        year: Publication year
        venue: Publication venue (conference, journal, arXiv)
        url: URL to access the paper
        citations: Current citation count
        methods_introduced: List of method_ids introduced in this paper
        abstract: Paper abstract
    """
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    url: str
    citations: int
    methods_introduced: List[str]
    abstract: str

    def __post_init__(self) -> None:
        """Validate paper data after initialization."""
        if not self.paper_id:
            raise ValueError("paper_id cannot be empty")
        if not self.title:
            raise ValueError("title cannot be empty")
        if self.citations < 0:
            raise ValueError("citations cannot be negative")


@dataclass
class Benchmark:
    """
    Representation of an ML benchmark for tracking state-of-the-art.

    Tracks performance metrics and historical progress on standard
    ML benchmarks and datasets.

    Attributes:
        benchmark_id: Unique identifier (e.g., "imagenet_top1")
        name: Human-readable name (e.g., "ImageNet Top-1 Accuracy")
        domain: Problem domain (e.g., "image_classification")
        dataset: Dataset name (e.g., "ImageNet-1K")
        metric: Metric being measured (e.g., "accuracy", "perplexity")
        sota_method: method_id of current state-of-the-art
        sota_score: Current best score
        history: List of (date, method_id, score) tuples tracking progress
    """
    benchmark_id: str
    name: str
    domain: str
    dataset: str
    metric: str
    sota_method: str
    sota_score: float
    history: List[Tuple[str, str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate benchmark data after initialization."""
        if not self.benchmark_id:
            raise ValueError("benchmark_id cannot be empty")
        if not self.dataset:
            raise ValueError("dataset cannot be empty")
        if not self.metric:
            raise ValueError("metric cannot be empty")
