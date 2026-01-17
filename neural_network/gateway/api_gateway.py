"""
API Gateway - FastAPI application for the Neural Network module
Part of the Neural Network module for the Consciousness system.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.nervous_system import NervousSystem, InferenceRequest, ConsciousnessState
from ..core.model_registry import Priority

logger = logging.getLogger(__name__)

# Global nervous system instance
_nervous_system: Optional[NervousSystem] = None


def get_nervous_system() -> NervousSystem:
    """Get the global nervous system instance."""
    global _nervous_system
    if _nervous_system is None:
        raise RuntimeError("Nervous system not initialized")
    return _nervous_system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    global _nervous_system

    # Startup
    logger.info("Starting Neural Network API Gateway...")
    _nervous_system = NervousSystem()
    await _nervous_system.initialize()
    await _nervous_system.start()
    logger.info("Neural Network API Gateway started")

    yield

    # Shutdown
    logger.info("Shutting down Neural Network API Gateway...")
    if _nervous_system:
        await _nervous_system.stop()
    logger.info("Neural Network API Gateway stopped")


# Create FastAPI app
app = FastAPI(
    title="Neural Network Consciousness API",
    description="API for coordinating local AI models across 27 consciousness forms",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================
# Pydantic Models
# =============================================

class HealthResponse(BaseModel):
    status: str
    loaded_forms: int
    gpu_utilization: float
    message_queue_depth: int
    timestamp: str


class ResourceUsageResponse(BaseModel):
    gpu_used_mb: int
    gpu_total_mb: int
    gpu_percent: float
    arousal_state: str
    active_forms: List[str]


class FormStatus(BaseModel):
    form_id: str
    name: str
    loaded: bool
    priority: str
    vram_mb: int
    inference_count: int
    avg_inference_ms: float


class FormListResponse(BaseModel):
    total_forms: int
    loaded_forms: int
    forms: List[FormStatus]


class InferenceRequestModel(BaseModel):
    input: Any
    priority: str = "normal"
    timeout_ms: int = 100


class InferenceResponseModel(BaseModel):
    form_id: str
    output: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None


class BatchInferenceRequest(BaseModel):
    requests: List[Dict[str, Any]]


class ConsciousnessStateResponse(BaseModel):
    arousal_level: float
    arousal_state: str
    phi_value: float
    global_workspace_contents: List[str]
    active_forms: List[str]
    processing_rate: float
    timestamp: str


# =============================================
# System Health Endpoints
# =============================================

@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """System health check."""
    ns = get_nervous_system()
    status = ns.get_status()

    return HealthResponse(
        status="healthy" if status['running'] else "degraded",
        loaded_forms=status['registry']['loaded_forms'],
        gpu_utilization=status['resources']['gpu']['utilization_percent'],
        message_queue_depth=status['message_bus']['total_queue_depth'],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/v1/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    ns = get_nervous_system()
    status = ns.get_status()

    # Format as Prometheus metrics
    lines = [
        "# HELP consciousness_loaded_forms Number of loaded consciousness forms",
        "# TYPE consciousness_loaded_forms gauge",
        f"consciousness_loaded_forms {status['registry']['loaded_forms']}",
        "",
        "# HELP consciousness_gpu_utilization GPU utilization percentage",
        "# TYPE consciousness_gpu_utilization gauge",
        f"consciousness_gpu_utilization {status['resources']['gpu']['utilization_percent']}",
        "",
        "# HELP consciousness_arousal_level Current arousal level",
        "# TYPE consciousness_arousal_level gauge",
        f"consciousness_arousal_level {status['consciousness_state']['arousal_level']}",
        "",
        "# HELP consciousness_phi_value Current phi (integration) value",
        "# TYPE consciousness_phi_value gauge",
        f"consciousness_phi_value {status['consciousness_state']['phi_value']}",
        "",
        "# HELP consciousness_inference_count Total inference count",
        "# TYPE consciousness_inference_count counter",
        f"consciousness_inference_count {status['metrics']['inference_count']}",
        "",
        "# HELP consciousness_cycle_count Processing cycle count",
        "# TYPE consciousness_cycle_count counter",
        f"consciousness_cycle_count {status['metrics']['cycle_count']}",
        "",
        "# HELP consciousness_message_queue_depth Message queue depth",
        "# TYPE consciousness_message_queue_depth gauge",
        f"consciousness_message_queue_depth {status['message_bus']['total_queue_depth']}",
    ]

    return "\n".join(lines)


@app.get("/api/v1/resources", response_model=ResourceUsageResponse)
async def resources():
    """Current resource utilization."""
    ns = get_nervous_system()
    status = ns.get_status()
    res = status['resources']

    return ResourceUsageResponse(
        gpu_used_mb=res['gpu']['used_mb'],
        gpu_total_mb=res['gpu']['total_mb'],
        gpu_percent=res['gpu']['utilization_percent'],
        arousal_state=res['arousal']['state'],
        active_forms=res['allocations']['forms'],
    )


# =============================================
# Form Management Endpoints
# =============================================

@app.get("/api/v1/forms", response_model=FormListResponse)
async def list_forms():
    """List all forms with status."""
    ns = get_nervous_system()
    registry = ns.registry
    status = registry.get_status()

    forms = []
    for form_id, config in registry.get_all_configs().items():
        loaded_model = await registry.get_model(form_id)
        is_loaded = loaded_model is not None

        forms.append(FormStatus(
            form_id=form_id,
            name=config.name,
            loaded=is_loaded,
            priority=config.priority.name,
            vram_mb=config.vram_mb,
            inference_count=loaded_model.inference_count if loaded_model else 0,
            avg_inference_ms=loaded_model.avg_inference_time_ms if loaded_model else 0,
        ))

    return FormListResponse(
        total_forms=len(forms),
        loaded_forms=status['loaded_forms'],
        forms=forms,
    )


@app.get("/api/v1/forms/{form_id}")
async def get_form(form_id: str):
    """Get form details and status."""
    ns = get_nervous_system()
    registry = ns.registry

    config = registry.get_config(form_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Form {form_id} not found")

    loaded_model = await registry.get_model(form_id)
    adapter = ns.adapters.get(form_id)

    return {
        'form_id': form_id,
        'name': config.name,
        'model_type': config.model_type,
        'model_name': config.model_name,
        'loaded': loaded_model is not None,
        'priority': config.priority.name,
        'preemptible': config.preemptible,
        'critical': config.critical,
        'vram_mb': config.vram_mb,
        'size_mb': config.size_mb,
        'quantization': config.quantization,
        'timeout_ms': config.timeout_ms,
        'input_spec': config.input_spec,
        'output_spec': config.output_spec,
        'adapter_registered': adapter is not None,
        'model_stats': {
            'inference_count': loaded_model.inference_count if loaded_model else 0,
            'avg_inference_ms': loaded_model.avg_inference_time_ms if loaded_model else 0,
            'error_count': loaded_model.error_count if loaded_model else 0,
            'loaded_at': loaded_model.loaded_at.isoformat() if loaded_model and loaded_model.loaded_at else None,
        } if loaded_model else None,
    }


@app.post("/api/v1/forms/{form_id}/load")
async def load_form(form_id: str):
    """Load form model into memory."""
    ns = get_nervous_system()

    success = await ns.load_form(form_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load form {form_id}"
        )

    return {
        'status': 'loaded',
        'form_id': form_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/forms/{form_id}/unload")
async def unload_form(form_id: str):
    """Unload form model from memory."""
    ns = get_nervous_system()

    success = await ns.unload_form(form_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot unload form {form_id} (may be critical or not loaded)"
        )

    return {
        'status': 'unloaded',
        'form_id': form_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


# =============================================
# Inference Endpoints
# =============================================

@app.post("/api/v1/inference/{form_id}", response_model=InferenceResponseModel)
async def inference(form_id: str, request: InferenceRequestModel):
    """Single form inference."""
    ns = get_nervous_system()

    # Map priority string to enum
    try:
        priority = Priority[request.priority.upper()]
    except KeyError:
        priority = Priority.NORMAL

    inference_request = InferenceRequest(
        form_id=form_id,
        input_data=request.input,
        priority=priority,
        timeout_ms=request.timeout_ms,
    )

    result = await ns.inference(inference_request)

    return InferenceResponseModel(
        form_id=result.form_id,
        output=result.output,
        latency_ms=result.latency_ms,
        success=result.success,
        error=result.error,
    )


@app.post("/api/v1/inference/batch")
async def batch_inference(request: BatchInferenceRequest):
    """Multi-form batch inference."""
    ns = get_nervous_system()
    results = []

    # Process requests concurrently
    async def process_request(req: Dict[str, Any]):
        form_id = req.get('form_id')
        input_data = req.get('input')
        priority_str = req.get('priority', 'normal')
        timeout_ms = req.get('timeout_ms', 100)

        try:
            priority = Priority[priority_str.upper()]
        except KeyError:
            priority = Priority.NORMAL

        inference_request = InferenceRequest(
            form_id=form_id,
            input_data=input_data,
            priority=priority,
            timeout_ms=timeout_ms,
        )

        return await ns.inference(inference_request)

    tasks = [process_request(req) for req in request.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        'results': [
            {
                'form_id': r.form_id if hasattr(r, 'form_id') else None,
                'output': r.output if hasattr(r, 'output') else None,
                'latency_ms': r.latency_ms if hasattr(r, 'latency_ms') else 0,
                'success': r.success if hasattr(r, 'success') else False,
                'error': str(r) if isinstance(r, Exception) else (r.error if hasattr(r, 'error') else None),
            }
            for r in results
        ],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


# =============================================
# Consciousness State Endpoints
# =============================================

@app.get("/api/v1/consciousness", response_model=ConsciousnessStateResponse)
async def get_consciousness_state():
    """Get current consciousness state."""
    ns = get_nervous_system()
    state = ns.get_consciousness_state()

    return ConsciousnessStateResponse(
        arousal_level=state.arousal_level,
        arousal_state=state.arousal_state.value,
        phi_value=state.phi_value,
        global_workspace_contents=state.global_workspace_contents,
        active_forms=state.active_forms,
        processing_rate=state.processing_rate,
        timestamp=state.timestamp.isoformat(),
    )


@app.get("/api/v1/status")
async def get_system_status():
    """Get comprehensive system status."""
    ns = get_nervous_system()
    return ns.get_status()


# =============================================
# WebSocket Streaming Endpoints
# =============================================

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections:
            self.active_connections[channel].remove(websocket)

    async def broadcast(self, message: dict, channel: str):
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@app.websocket("/api/v1/stream/consciousness")
async def consciousness_stream(websocket: WebSocket):
    """Real-time consciousness state stream."""
    await manager.connect(websocket, "consciousness")

    ns = get_nervous_system()

    # Register callback for state updates
    async def on_state_change(state: ConsciousnessState):
        await manager.broadcast(state.to_dict(), "consciousness")

    ns.on_state_change(on_state_change)

    try:
        while True:
            # Keep connection alive, send periodic updates
            await asyncio.sleep(0.1)  # 10Hz update rate
            state = ns.get_consciousness_state()
            await websocket.send_json(state.to_dict())
    except WebSocketDisconnect:
        manager.disconnect(websocket, "consciousness")


@app.websocket("/api/v1/stream/forms/{form_id}")
async def form_stream(websocket: WebSocket, form_id: str):
    """Form-specific real-time updates."""
    await manager.connect(websocket, f"form_{form_id}")

    ns = get_nervous_system()
    adapter = ns.adapters.get(form_id)

    try:
        while True:
            await asyncio.sleep(0.05)  # 20Hz update rate

            if adapter:
                health = await adapter.health_check()
                await websocket.send_json(health)
            else:
                await websocket.send_json({
                    'form_id': form_id,
                    'status': 'adapter_not_registered',
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, f"form_{form_id}")


@app.websocket("/api/v1/stream/global-workspace")
async def global_workspace_stream(websocket: WebSocket):
    """Global workspace state stream."""
    await manager.connect(websocket, "global_workspace")

    ns = get_nervous_system()
    gw_adapter = ns.adapters.get("14-global-workspace")

    try:
        while True:
            await asyncio.sleep(0.05)  # 20Hz update rate

            if gw_adapter:
                state = await gw_adapter.get_workspace_state()
                await websocket.send_json(state)
            else:
                await websocket.send_json({
                    'status': 'global_workspace_adapter_not_registered',
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, "global_workspace")


# =============================================
# Message Bus Endpoints
# =============================================

@app.post("/api/v1/messages/publish")
async def publish_message(message: Dict[str, Any]):
    """Publish message to the message bus."""
    ns = get_nervous_system()
    bus = ns.bus

    from ..core.message_bus import MessageType

    try:
        message_type = MessageType(message.get('type', 'coordination'))
    except ValueError:
        message_type = MessageType.COORDINATION

    msg = bus.create_message(
        source_form=message.get('source', 'api'),
        target_form=message.get('target'),
        message_type=message_type,
        body=message.get('body', {}),
        priority=Priority.NORMAL,
    )

    await bus.publish(msg)

    return {
        'status': 'published',
        'message_id': msg.header.message_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/messages/status")
async def message_bus_status():
    """Get message bus status."""
    ns = get_nervous_system()
    return ns.bus.get_status()


# Run with: uvicorn consciousness.neural_network.gateway.api_gateway:app --reload
