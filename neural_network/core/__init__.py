"""
Core Module - Central coordination and management components
Part of the Neural Network module for the Consciousness system.
"""

from .model_registry import (
    ModelRegistry,
    ModelConfig,
    LoadedModel,
    ModelState,
    Priority,
)
from .resource_manager import (
    ResourceManager,
    ResourceRequest,
    Allocation,
    ResourceUsage,
    ArousalState,
    ArousalGatingConfig,
)
from .message_bus import (
    MessageBus,
    FormMessage,
    MessageHeader,
    MessageFooter,
    MessageType,
    MessageQueue,
)
from .nervous_system import (
    NervousSystem,
    ConsciousnessState,
    InferenceRequest,
    InferenceResult,
)

__all__ = [
    # Model Registry
    'ModelRegistry',
    'ModelConfig',
    'LoadedModel',
    'ModelState',
    'Priority',
    # Resource Manager
    'ResourceManager',
    'ResourceRequest',
    'Allocation',
    'ResourceUsage',
    'ArousalState',
    'ArousalGatingConfig',
    # Message Bus
    'MessageBus',
    'FormMessage',
    'MessageHeader',
    'MessageFooter',
    'MessageType',
    'MessageQueue',
    # Nervous System
    'NervousSystem',
    'ConsciousnessState',
    'InferenceRequest',
    'InferenceResult',
]
