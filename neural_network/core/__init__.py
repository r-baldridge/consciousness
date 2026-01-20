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
from .adapter_factory import (
    AdapterFactory,
    FORM_ADAPTER_REGISTRY,
    FORM_CATEGORIES,
    get_factory,
    create_adapter,
    get_adapter,
    register_all_adapters,
    register_critical_adapters,
    get_adapter_info,
    list_all_adapters,
)
from .message_handlers import (
    # Core classes
    HandlerResult,
    HandlerResponse,
    MessageHandlerContext,
    BaseMessageHandler,
    QueryResponseHandler,
    BroadcastHandler,
    MessageHandlerRegistry,
    MessageHandlerCoordinator,
    MessageHandlerFactory,
    # Convenience functions
    create_coordinator,
    get_handled_message_types,
    # Concrete handlers
    ArousalUpdateHandler,
    WorkspaceBroadcastHandler,
    PhiUpdateHandler,
    EmergencyHandler,
    SensoryInputHandler,
    AttentionRequestHandler,
    MemoryQueryHandler,
    PhilosophicalQueryHandler,
    CrossFormSynthesisHandler,
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
    # Adapter Factory
    'AdapterFactory',
    'FORM_ADAPTER_REGISTRY',
    'FORM_CATEGORIES',
    'get_factory',
    'create_adapter',
    'get_adapter',
    'register_all_adapters',
    'register_critical_adapters',
    'get_adapter_info',
    'list_all_adapters',
    # Message Handlers
    'HandlerResult',
    'HandlerResponse',
    'MessageHandlerContext',
    'BaseMessageHandler',
    'QueryResponseHandler',
    'BroadcastHandler',
    'MessageHandlerRegistry',
    'MessageHandlerCoordinator',
    'MessageHandlerFactory',
    'create_coordinator',
    'get_handled_message_types',
    'ArousalUpdateHandler',
    'WorkspaceBroadcastHandler',
    'PhiUpdateHandler',
    'EmergencyHandler',
    'SensoryInputHandler',
    'AttentionRequestHandler',
    'MemoryQueryHandler',
    'PhilosophicalQueryHandler',
    'CrossFormSynthesisHandler',
]
