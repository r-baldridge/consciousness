"""
Gateway Module - FastAPI application and routes
Part of the Neural Network module for the Consciousness system.
"""

from .api_gateway import (
    app,
    get_nervous_system,
    ConnectionManager,
    # Request/Response Models
    HealthResponse,
    ResourceUsageResponse,
    FormStatus,
    FormListResponse,
    InferenceRequestModel,
    InferenceResponseModel,
    BatchInferenceRequest,
    ConsciousnessStateResponse,
)

__all__ = [
    'app',
    'get_nervous_system',
    'ConnectionManager',
    # Request/Response Models
    'HealthResponse',
    'ResourceUsageResponse',
    'FormStatus',
    'FormListResponse',
    'InferenceRequestModel',
    'InferenceResponseModel',
    'BatchInferenceRequest',
    'ConsciousnessStateResponse',
]
