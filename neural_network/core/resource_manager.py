"""
Resource Manager - Dynamic GPU/CPU/memory allocation with preemption
Part of the Neural Network module for the Consciousness system.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from .model_registry import Priority

logger = logging.getLogger(__name__)


class ArousalState(Enum):
    """Arousal states that affect resource allocation."""
    SLEEP = "sleep"
    DROWSY = "drowsy"
    RELAXED = "relaxed"
    ALERT = "alert"
    FOCUSED = "focused"
    HYPERAROUSED = "hyperaroused"


@dataclass
class ResourceRequest:
    """Request for resource allocation."""
    form_id: str
    vram_mb: int
    cpu_percent: float = 0.0
    memory_mb: int = 0
    priority: Priority = Priority.NORMAL
    timeout_ms: int = 100
    preemptible: bool = True


@dataclass
class Allocation:
    """Represents an active resource allocation."""
    form_id: str
    vram_mb: int
    cpu_percent: float
    memory_mb: int
    priority: Priority
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    preemptible: bool = True

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used = datetime.now(timezone.utc)


@dataclass
class ResourceUsage:
    """Current resource utilization snapshot."""
    gpu_used_mb: int
    gpu_total_mb: int
    gpu_percent: float
    cpu_percent: float
    memory_used_mb: int
    memory_total_mb: int
    memory_percent: float
    active_allocations: int
    forms_loaded: List[str]
    arousal_state: ArousalState
    gated_tiers: List[str]


@dataclass
class ArousalGatingConfig:
    """Configuration for arousal-based resource gating."""
    min_level: float
    max_level: float
    active_tiers: List[str]
    max_concurrent_forms: int
    processing_rate: float


class ResourceManager:
    """
    Dynamic GPU/CPU/memory allocation with preemption.

    Manages resource allocation across all consciousness forms,
    implementing arousal-based gating and priority-based preemption.
    """

    TOTAL_GPU_MEMORY_MB = 16 * 1024  # 16GB default

    def __init__(self, config_path: str):
        """
        Initialize the resource manager.

        Args:
            config_path: Path to resource_budgets.yaml
        """
        self.config_path = Path(config_path)
        self.allocations: Dict[str, Allocation] = {}
        self._lock = asyncio.Lock()
        self._arousal_level: float = 0.5
        self._arousal_state: ArousalState = ArousalState.ALERT

        # Resource limits
        self.total_gpu_mb = self.TOTAL_GPU_MEMORY_MB
        self.total_cpu_cores = 8
        self.total_memory_mb = 32 * 1024  # 32GB

        # Priority tier configurations
        self.priority_tiers: Dict[str, Dict[str, Any]] = {}
        self.arousal_gating: Dict[str, ArousalGatingConfig] = {}

        # Reserved resources per tier
        self.reserved_vram: Dict[str, int] = {}

        # Preemption tracking
        self.preemption_count = 0
        self.last_preemption_time: Optional[datetime] = None
        self.preemption_cooldown_ms = 5000

        self._load_config()

    def _load_config(self) -> None:
        """Load resource budget configurations from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load resource totals
        resources = config.get('resources', {})
        gpu = resources.get('gpu', {})
        self.total_gpu_mb = gpu.get('total_memory_mb', self.TOTAL_GPU_MEMORY_MB)

        cpu = resources.get('cpu', {})
        self.total_cpu_cores = cpu.get('total_cores', 8)

        memory = resources.get('memory', {})
        self.total_memory_mb = memory.get('total_ram_mb', 32 * 1024)

        # Load priority tiers
        self.priority_tiers = config.get('priority_tiers', {})
        for tier_name, tier_config in self.priority_tiers.items():
            self.reserved_vram[tier_name] = tier_config.get('reserved_vram_mb', 0)

        # Load arousal gating configurations
        arousal_config = config.get('arousal_gating', {})
        if arousal_config.get('enabled', True):
            thresholds = arousal_config.get('thresholds', {})
            for state_name, state_config in thresholds.items():
                range_vals = state_config.get('range', [0.0, 1.0])
                self.arousal_gating[state_name] = ArousalGatingConfig(
                    min_level=range_vals[0],
                    max_level=range_vals[1],
                    active_tiers=state_config.get('active_tiers', []),
                    max_concurrent_forms=state_config.get('max_concurrent_forms', 10),
                    processing_rate=state_config.get('processing_rate', 1.0),
                )

        # Load preemption settings
        preemption = config.get('preemption', {})
        self.preemption_cooldown_ms = preemption.get('preemption_cooldown_ms', 5000)

        logger.info(
            f"ResourceManager initialized: "
            f"GPU={self.total_gpu_mb}MB, "
            f"CPU={self.total_cpu_cores} cores, "
            f"RAM={self.total_memory_mb}MB"
        )

    def get_tier_for_form(self, form_id: str) -> Optional[str]:
        """Get the priority tier name for a form."""
        for tier_name, tier_config in self.priority_tiers.items():
            forms = tier_config.get('forms', [])
            if form_id in forms:
                return tier_name
        return None

    def set_arousal_level(self, level: float) -> None:
        """
        Update the current arousal level.

        Args:
            level: Arousal level from 0.0 to 1.0
        """
        self._arousal_level = max(0.0, min(1.0, level))

        # Determine arousal state from level
        for state_name, config in self.arousal_gating.items():
            if config.min_level <= self._arousal_level < config.max_level:
                self._arousal_state = ArousalState(state_name)
                break

        logger.debug(f"Arousal level set to {level:.2f} ({self._arousal_state.value})")

    @property
    def arousal_level(self) -> float:
        """Get current arousal level."""
        return self._arousal_level

    @property
    def arousal_state(self) -> ArousalState:
        """Get current arousal state."""
        return self._arousal_state

    def is_form_allowed(self, form_id: str) -> bool:
        """
        Check if a form is allowed to run based on current arousal state.

        Args:
            form_id: The form ID to check

        Returns:
            True if the form can run, False if gated
        """
        tier = self.get_tier_for_form(form_id)
        if not tier:
            return True  # Unknown forms default to allowed

        gating_config = self.arousal_gating.get(self._arousal_state.value)
        if not gating_config:
            return True  # No gating config means allowed

        return tier in gating_config.active_tiers

    def get_max_concurrent_forms(self) -> int:
        """Get the maximum allowed concurrent forms for current arousal state."""
        gating_config = self.arousal_gating.get(self._arousal_state.value)
        if gating_config:
            return gating_config.max_concurrent_forms
        return 15  # Default

    def get_processing_rate(self) -> float:
        """Get the processing rate modifier for current arousal state."""
        gating_config = self.arousal_gating.get(self._arousal_state.value)
        if gating_config:
            return gating_config.processing_rate
        return 1.0  # Default full rate

    async def allocate(
        self,
        request: ResourceRequest
    ) -> Optional[Allocation]:
        """
        Allocate resources for a form, preempting if necessary.

        Args:
            request: The resource allocation request

        Returns:
            Allocation if successful, None if resources unavailable
        """
        async with self._lock:
            # Check arousal gating
            if not self.is_form_allowed(request.form_id):
                logger.info(
                    f"Form {request.form_id} gated by arousal state "
                    f"({self._arousal_state.value})"
                )
                return None

            # Check concurrent forms limit
            if len(self.allocations) >= self.get_max_concurrent_forms():
                if not await self._try_preempt_for_limit(request):
                    logger.warning(
                        f"Max concurrent forms reached "
                        f"({self.get_max_concurrent_forms()})"
                    )
                    return None

            # Check available GPU memory
            used_vram = self._get_used_vram()
            available_vram = self.total_gpu_mb - used_vram

            if request.vram_mb > available_vram:
                # Try preemption
                if not await self._try_preempt_for_memory(
                    request.form_id,
                    request.priority,
                    request.vram_mb - available_vram
                ):
                    logger.warning(
                        f"Insufficient VRAM for {request.form_id}: "
                        f"need {request.vram_mb}MB, have {available_vram}MB"
                    )
                    return None

            # Create allocation
            allocation = Allocation(
                form_id=request.form_id,
                vram_mb=request.vram_mb,
                cpu_percent=request.cpu_percent,
                memory_mb=request.memory_mb,
                priority=request.priority,
                preemptible=request.preemptible,
            )

            self.allocations[request.form_id] = allocation
            logger.debug(f"Allocated resources for {request.form_id}: {request.vram_mb}MB VRAM")

            return allocation

    async def release(self, form_id: str) -> None:
        """
        Release all resources for a form.

        Args:
            form_id: The form ID to release resources for
        """
        async with self._lock:
            if form_id in self.allocations:
                allocation = self.allocations.pop(form_id)
                logger.debug(
                    f"Released resources for {form_id}: "
                    f"{allocation.vram_mb}MB VRAM"
                )

    async def touch(self, form_id: str) -> None:
        """Update last used timestamp for a form."""
        if form_id in self.allocations:
            self.allocations[form_id].touch()

    def _get_used_vram(self) -> int:
        """Get total VRAM currently allocated."""
        return sum(a.vram_mb for a in self.allocations.values())

    async def _try_preempt_for_memory(
        self,
        form_id: str,
        priority: Priority,
        needed_mb: int
    ) -> bool:
        """
        Try to preempt lower-priority forms to free memory.

        Returns True if enough memory was freed.
        """
        # Check cooldown
        if self.last_preemption_time:
            elapsed = (datetime.now(timezone.utc) - self.last_preemption_time).total_seconds() * 1000
            if elapsed < self.preemption_cooldown_ms:
                return False

        # Find preemptible allocations with lower priority
        candidates = [
            (fid, alloc)
            for fid, alloc in self.allocations.items()
            if alloc.preemptible and alloc.priority.value < priority.value
        ]

        if not candidates:
            return False

        # Sort by priority (lowest first), then by last used (oldest first)
        candidates.sort(key=lambda x: (x[1].priority.value, x[1].last_used))

        freed = 0
        to_preempt = []

        for fid, alloc in candidates:
            if freed >= needed_mb:
                break
            freed += alloc.vram_mb
            to_preempt.append(fid)

        if freed < needed_mb:
            return False

        # Perform preemption
        for fid in to_preempt:
            logger.info(f"Preempting {fid} for {form_id}")
            del self.allocations[fid]
            self.preemption_count += 1

        self.last_preemption_time = datetime.now(timezone.utc)
        return True

    async def _try_preempt_for_limit(self, request: ResourceRequest) -> bool:
        """
        Try to preempt a form to make room for a new allocation.
        """
        # Find lowest priority preemptible allocation
        candidates = [
            (fid, alloc)
            for fid, alloc in self.allocations.items()
            if alloc.preemptible and alloc.priority.value < request.priority.value
        ]

        if not candidates:
            return False

        # Sort by priority, then last used
        candidates.sort(key=lambda x: (x[1].priority.value, x[1].last_used))

        # Preempt the first one
        fid, _ = candidates[0]
        logger.info(f"Preempting {fid} for concurrent limit")
        del self.allocations[fid]
        self.preemption_count += 1
        self.last_preemption_time = datetime.now(timezone.utc)

        return True

    def can_preempt(self, source_form: str, target_form: str) -> bool:
        """
        Check if source_form can preempt target_form based on priority.

        Args:
            source_form: The form requesting resources
            target_form: The form that might be preempted

        Returns:
            True if preemption is allowed
        """
        target_alloc = self.allocations.get(target_form)
        if not target_alloc:
            return False

        if not target_alloc.preemptible:
            return False

        source_tier = self.get_tier_for_form(source_form)
        target_tier = self.get_tier_for_form(target_form)

        if source_tier == 'critical':
            return target_tier not in ['critical']

        tier_config = self.priority_tiers.get(source_tier, {})
        can_preempt_tiers = tier_config.get('can_preempt', [])

        return target_tier in can_preempt_tiers

    async def get_usage(self) -> ResourceUsage:
        """Get current resource utilization snapshot."""
        used_vram = self._get_used_vram()
        used_cpu = sum(a.cpu_percent for a in self.allocations.values())
        used_memory = sum(a.memory_mb for a in self.allocations.values())

        gating_config = self.arousal_gating.get(self._arousal_state.value)
        gated_tiers = gating_config.active_tiers if gating_config else []

        return ResourceUsage(
            gpu_used_mb=used_vram,
            gpu_total_mb=self.total_gpu_mb,
            gpu_percent=100.0 * used_vram / self.total_gpu_mb if self.total_gpu_mb else 0,
            cpu_percent=used_cpu,
            memory_used_mb=used_memory,
            memory_total_mb=self.total_memory_mb,
            memory_percent=100.0 * used_memory / self.total_memory_mb if self.total_memory_mb else 0,
            active_allocations=len(self.allocations),
            forms_loaded=list(self.allocations.keys()),
            arousal_state=self._arousal_state,
            gated_tiers=gated_tiers,
        )

    @property
    def gpu_percent(self) -> float:
        """Get GPU utilization percentage."""
        used = self._get_used_vram()
        return 100.0 * used / self.total_gpu_mb if self.total_gpu_mb else 0

    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status."""
        used_vram = self._get_used_vram()

        return {
            'gpu': {
                'used_mb': used_vram,
                'total_mb': self.total_gpu_mb,
                'available_mb': self.total_gpu_mb - used_vram,
                'utilization_percent': round(100 * used_vram / self.total_gpu_mb, 1),
            },
            'arousal': {
                'level': round(self._arousal_level, 3),
                'state': self._arousal_state.value,
                'processing_rate': self.get_processing_rate(),
                'max_concurrent': self.get_max_concurrent_forms(),
            },
            'allocations': {
                'count': len(self.allocations),
                'forms': list(self.allocations.keys()),
            },
            'preemption': {
                'total_count': self.preemption_count,
                'last_time': (
                    self.last_preemption_time.isoformat()
                    if self.last_preemption_time else None
                ),
            },
        }
