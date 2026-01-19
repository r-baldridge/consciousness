"""
Adapters Module - Form-specific model adapters
Part of the Neural Network module for the Consciousness system.
"""

from .base_adapter import (
    FormAdapter,
    SensoryAdapter,
    CognitiveAdapter,
    TheoreticalAdapter,
    SpecializedAdapter,
    ArousalAdapterInterface,
    IITAdapterInterface,
    GlobalWorkspaceAdapterInterface,
)

# Sensory Adapters (Forms 01-06)
from .sensory import (
    VisualAdapter,
    AuditoryAdapter,
    TactileAdapter,
    OlfactoryAdapter,
    GustatoryAdapter,
    ProprioceptiveAdapter,
)

# Cognitive Adapters (Forms 07-12)
from .cognitive import (
    AttentionAdapter,
    ArousalAdapter,
    MemorySTMAdapter,
    MemoryLTMAdapter,
    EmotionAdapter,
    ExecutiveAdapter,
)

# Theoretical Adapters (Forms 13-17)
from .theoretical import (
    IITAdapter,
    GlobalWorkspaceAdapter,
    HOTAdapter,
    PredictiveAdapter,
    RecurrentAdapter,
)

# Specialized Adapters (Forms 18-27)
from .specialized import (
    PrimaryAdapter,
    ReflectiveAdapter,
    SocialAdapter,
    ArtificialAdapter,
    DreamAdapter,
    MeditationAdapter,
    FlowAdapter,
    MysticalAdapter,
    ThresholdAdapter,
    AlteredAdapter,
)

# Extended Adapters (Forms 28-40)
from .extended import (
    # Forms 28-30
    PhilosophyAdapter,
    FolkWisdomAdapter,
    AnimalCognitionAdapter,
    # Forms 31-34 (Ecosystem)
    PlantIntelligenceAdapter,
    FungalIntelligenceAdapter,
    SwarmIntelligenceAdapter,
    GaiaIntelligenceAdapter,
    # Forms 35-40 (Expanded)
    DevelopmentalConsciousnessAdapter,
    ContemplativeStatesAdapter,
    PsychedelicConsciousnessAdapter,
    NeurodivergentConsciousnessAdapter,
    TraumaConsciousnessAdapter,
    XenoconsciousnessAdapter,
)

__all__ = [
    # Base
    'FormAdapter',
    'SensoryAdapter',
    'CognitiveAdapter',
    'TheoreticalAdapter',
    'SpecializedAdapter',
    'ArousalAdapterInterface',
    'IITAdapterInterface',
    'GlobalWorkspaceAdapterInterface',
    # Sensory (01-06)
    'VisualAdapter',
    'AuditoryAdapter',
    'TactileAdapter',
    'OlfactoryAdapter',
    'GustatoryAdapter',
    'ProprioceptiveAdapter',
    # Cognitive (07-12)
    'AttentionAdapter',
    'ArousalAdapter',
    'MemorySTMAdapter',
    'MemoryLTMAdapter',
    'EmotionAdapter',
    'ExecutiveAdapter',
    # Theoretical (13-17)
    'IITAdapter',
    'GlobalWorkspaceAdapter',
    'HOTAdapter',
    'PredictiveAdapter',
    'RecurrentAdapter',
    # Specialized (18-27)
    'PrimaryAdapter',
    'ReflectiveAdapter',
    'SocialAdapter',
    'ArtificialAdapter',
    'DreamAdapter',
    'MeditationAdapter',
    'FlowAdapter',
    'MysticalAdapter',
    'ThresholdAdapter',
    'AlteredAdapter',
    # Extended (28-40)
    'PhilosophyAdapter',
    'FolkWisdomAdapter',
    'AnimalCognitionAdapter',
    # Ecosystem (31-34)
    'PlantIntelligenceAdapter',
    'FungalIntelligenceAdapter',
    'SwarmIntelligenceAdapter',
    'GaiaIntelligenceAdapter',
    # Expanded (35-40)
    'DevelopmentalConsciousnessAdapter',
    'ContemplativeStatesAdapter',
    'PsychedelicConsciousnessAdapter',
    'NeurodivergentConsciousnessAdapter',
    'TraumaConsciousnessAdapter',
    'XenoconsciousnessAdapter',
]
