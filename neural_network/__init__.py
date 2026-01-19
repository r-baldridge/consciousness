"""
Neural Network Module - Consciousness System Nervous System

This module coordinates local specialized AI models for each of the 40 consciousness
forms, functioning as the "nervous system" that enables inter-form communication
and resource management.

Main Components:
- NervousSystem: Central coordinator connecting all forms
- ModelRegistry: Model lifecycle management (loading, unloading, versioning)
- ResourceManager: Dynamic GPU/CPU/memory allocation with arousal gating
- MessageBus: Inter-form communication with priority queues
- FormAdapter: Base adapter interface for all consciousness forms
- ModelLoader: Unified model loading with quantization support

Usage:
    from consciousness.neural_network import NervousSystem

    # Initialize the nervous system
    ns = NervousSystem()
    await ns.initialize()
    await ns.start()

    # Run inference
    result = await ns.inference(InferenceRequest(
        form_id="01-visual",
        input_data=image_tensor,
        priority=Priority.NORMAL,
    ))

    # Get consciousness state
    state = ns.get_consciousness_state()

API Server:
    Run with: uvicorn consciousness.neural_network.gateway.api_gateway:app --reload
"""

__version__ = "1.0.0"

# Core Components
from .core import (
    NervousSystem,
    ConsciousnessState,
    InferenceRequest,
    InferenceResult,
    ModelRegistry,
    ModelConfig,
    LoadedModel,
    ModelState,
    Priority,
    ResourceManager,
    ResourceRequest,
    Allocation,
    ResourceUsage,
    ArousalState,
    ArousalGatingConfig,
    MessageBus,
    FormMessage,
    MessageHeader,
    MessageFooter,
    MessageType,
)

# Base Adapter
from .adapters.base_adapter import (
    FormAdapter,
    SensoryAdapter,
    CognitiveAdapter,
    TheoreticalAdapter,
    SpecializedAdapter,
)

# Extended Adapters (Forms 28-40)
from .adapters.extended import (
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

# Model Utilities
from .models import (
    ModelLoader,
    LoadedModelInfo,
    QuantizationType,
    ModelQuantizer,
    QuantizationMethod,
)

__all__ = [
    # Version
    '__version__',
    # Core - Nervous System
    'NervousSystem',
    'ConsciousnessState',
    'InferenceRequest',
    'InferenceResult',
    # Core - Model Registry
    'ModelRegistry',
    'ModelConfig',
    'LoadedModel',
    'ModelState',
    'Priority',
    # Core - Resource Manager
    'ResourceManager',
    'ResourceRequest',
    'Allocation',
    'ResourceUsage',
    'ArousalState',
    'ArousalGatingConfig',
    # Core - Message Bus
    'MessageBus',
    'FormMessage',
    'MessageHeader',
    'MessageFooter',
    'MessageType',
    # Adapters
    'FormAdapter',
    'SensoryAdapter',
    'CognitiveAdapter',
    'TheoreticalAdapter',
    'SpecializedAdapter',
    # Extended Adapters (Forms 28-40)
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
    # Models
    'ModelLoader',
    'LoadedModelInfo',
    'QuantizationType',
    'ModelQuantizer',
    'QuantizationMethod',
]
