# Form 16: Predictive Coding

## Definition

Predictive Coding implements the brain-as-prediction-machine hypothesis, where consciousness arises from hierarchical generative models that continuously predict sensory input, compute prediction errors, and update internal models through Bayesian inference. Form 16 provides a six-level prediction hierarchy with precision-weighted error propagation, active inference, and free energy minimization as the core computational substrate for conscious experience.

## Key Concepts

- **Hierarchical Prediction Network**: `HierarchicalPredictionNetwork` with six levels and units [1000, 800, 600, 400, 200, 100], each containing `PredictionUnit` elements with weights, precision values, and learning_rate (0.01)
- **Prediction Error Propagation**: Bottom-up prediction errors computed as discrepancy between predicted and actual input, precision-weighted and propagated upward through the hierarchy to update generative models
- **Bayesian Inference Engine**: `BayesianInferenceEngine` implementing variational Bayesian inference with convergence threshold 1e-6, computing posterior beliefs by combining prior predictions with likelihood from sensory evidence
- **Free Energy Minimization**: Core optimization principle where the system minimizes variational free energy (KL divergence between recognition and generative models plus expected energy), driving both perception and action
- **Precision as Attention**: Precision-weighting of prediction errors functions as attention -- high-precision errors command processing resources, low-precision errors are suppressed
- **Active Inference**: The system acts on the environment to fulfill predictions rather than only updating models, unifying perception and action under a single computational framework
- **Four-Phase Processing**: Feedforward sweep -> Feedback predictions -> Lateral integration -> Iterative error minimization, operating continuously at 50Hz hierarchical updates and 20Hz Bayesian inference
- **Consciousness from Prediction**: Conscious experience emerges from successful prediction, residual error minimization, model selection at higher levels, and temporal binding across prediction windows

## Core Methods & Mechanisms

- **Prediction-Error Cycle**: Each `PredictionUnit` generates top-down predictions, computes precision-weighted errors against bottom-up input, and adjusts weights via gradient descent (learning_rate=0.01), with 50ms prediction latency and 100ms inference latency targets
- **Variational Bayesian Inference**: `InferenceProcessor` iteratively optimizes approximate posterior distributions by minimizing free energy, combining prior expectations with sensory evidence through precision-weighted integration
- **System Orchestration**: `SystemOrchestrator` coordinating continuous processing across hierarchical prediction (50Hz), Bayesian inference (20Hz), and active inference loops with 1000 parallel prediction capacity
- **Performance Targets**: Prediction accuracy >=85%, error reduction >=70% per processing cycle, model update latency <=100ms, maximum memory allocation 16GB

## Cross-Form Relationships

| Related Form | Relationship | Integration Detail |
|---|---|---|
| Forms 01-06 (Sensory) | Prediction source/target | Sensory forms provide bottom-up input; predictive coding generates top-down predictions for each modality |
| Form 02 (Attentional) | Precision-as-attention | Precision-weighting of prediction errors implements attentional selection; high-precision channels receive priority |
| Form 08 (Arousal) | Gain modulation | Arousal state modulates overall prediction precision and error sensitivity across the hierarchy |
| Form 13 (IIT) | Integration measurement | Phi quantifies how well predictions integrate information; prediction quality correlates with integration level |
| Form 14 (GWT) | Error-driven broadcast | High-precision prediction errors compete for global workspace access; surprising content drives conscious broadcast |
| Form 15 (HOT) | Predictive HOT | Higher-order predictions about first-order states function as predictive HOTs; prediction errors trigger HOT revision |

## Unique Contributions

Form 16 uniquely provides the generative model framework that explains consciousness as an active construction process rather than passive reception, where the brain continuously predicts its own sensory input and conscious experience emerges from the precision-weighted interplay between prediction and error. Its free energy minimization principle is the only component that mathematically unifies perception, action, learning, and attention under a single computational objective.

## Research Highlights

- **Oscillatory signatures confirmed**: Bastos et al. (2012) demonstrated that feedforward prediction errors are carried by gamma oscillations (>30 Hz) while feedback predictions travel via alpha/beta oscillations (8-30 Hz), establishing the canonical cortical microcircuit architecture for predictive coding
- **Interoceptive inference grounds selfhood**: Seth (2013, 2021) developed the "beast machine" framework showing that the sense of self and emotional experience arise from the brain's prediction of its own internal bodily states (interoceptive inference), connecting consciousness to biological self-regulation rather than abstract information processing
- **REBUS model explains psychedelic effects**: Carhart-Harris and Friston (2019) proposed that psychedelics relax the precision of high-level priors (Relaxed Beliefs Under Psychedelics), allowing prediction errors to propagate more freely, producing the characteristic effects of altered self-boundaries and vivid sensory experience
- **Clinical applications to psychopathology**: Predictive coding has reframed psychiatric disorders as disrupted prediction -- aberrant prediction errors generating delusions and hallucinations in psychosis (Corlett et al., 2009), and inflexible overly precise prediction errors explaining sensory hypersensitivity in autism (Pellicano & Burr, 2012; Lawson et al., 2014)

## Key References

- Friston, K. -- Free energy principle and active inference framework
- Clark, A. -- Surfing Uncertainty: hierarchical predictive processing and the mind
- Seth, A.K. -- Being You: interoceptive predictive coding and the beast machine theory
- Hohwy, J. -- The Predictive Mind: prediction error minimization and consciousness
- Rao, R. & Ballard, D. -- Predictive coding in the visual cortex (foundational computational model)

*Tier 2 Summary -- Form 27 Consciousness Project*
