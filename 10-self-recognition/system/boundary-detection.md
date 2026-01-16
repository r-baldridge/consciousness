# Form 10: Self-Recognition - Boundary Detection System

## Process Boundary Monitor

```python
import psutil
import os
import threading
import asyncio
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import logging

class ProcessBoundaryMonitor:
    """
    Monitor for detecting and tracking process-level boundaries.

    Identifies which processes belong to self vs. other, tracking
    process creation, termination, and resource usage patterns.
    """

    def __init__(self, config: 'ProcessBoundaryConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ProcessBoundary")

        # Process tracking
        self._own_processes = set()
        self._process_hierarchy = {}
        self._process_metadata = {}

        # Resource monitoring
        self._resource_tracker = ResourceTracker()
        self._execution_tracer = ExecutionTracer()

        # Change detection
        self._process_changes = deque(maxlen=config.change_history_size)
        self._last_scan_time = 0

        # Performance optimization
        self._scan_cache = {}
        self._cache_timeout = config.cache_timeout

    async def initialize(self):
        """Initialize the process boundary monitor."""
        self.logger.info("Initializing process boundary monitor")

        # Identify initial owned processes
        await self._identify_initial_processes()

        # Start resource tracking
        await self._resource_tracker.start()

        # Begin execution tracing
        await self._execution_tracer.start()

        self.logger.info("Process boundary monitor initialized")

    async def detect_boundaries(self, process_data: 'ProcessData') -> 'ProcessBoundaries':
        """Detect current process boundaries."""
        detection_start = time.time()

        # Get current process information
        current_processes = await self._scan_current_processes()

        # Update ownership classification
        ownership_map = await self._classify_process_ownership(current_processes)

        # Detect boundary changes
        boundary_changes = self._detect_boundary_changes(ownership_map)

        # Analyze resource allocations
        resource_boundaries = await self._analyze_resource_boundaries(ownership_map)

        # Build process hierarchy
        process_hierarchy = self._build_process_hierarchy(current_processes)

        # Calculate confidence scores
        boundary_confidence = self._calculate_boundary_confidence(
            ownership_map, resource_boundaries, process_hierarchy
        )

        return ProcessBoundaries(
            timestamp=time.time(),
            owned_processes=ownership_map['owned'],
            other_processes=ownership_map['other'],
            uncertain_processes=ownership_map['uncertain'],
            process_hierarchy=process_hierarchy,
            resource_boundaries=resource_boundaries,
            boundary_changes=boundary_changes,
            confidence=boundary_confidence,
            detection_time=time.time() - detection_start
        )

    async def _scan_current_processes(self) -> Dict[int, 'ProcessInfo']:
        """Scan all current processes on the system."""
        processes = {}

        try:
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'exe', 'cmdline',
                                           'create_time', 'cpu_percent', 'memory_percent',
                                           'num_threads', 'status']):
                try:
                    pinfo = proc.info
                    if pinfo['pid'] is not None:
                        processes[pinfo['pid']] = ProcessInfo(
                            pid=pinfo['pid'],
                            ppid=pinfo['ppid'],
                            name=pinfo['name'],
                            exe=pinfo['exe'],
                            cmdline=pinfo['cmdline'],
                            create_time=pinfo['create_time'],
                            cpu_percent=pinfo['cpu_percent'],
                            memory_percent=pinfo['memory_percent'],
                            num_threads=pinfo['num_threads'],
                            status=pinfo['status']
                        )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except Exception as e:
            self.logger.error(f"Error scanning processes: {e}")

        return processes

    async def _classify_process_ownership(
        self,
        processes: Dict[int, 'ProcessInfo']
    ) -> Dict[str, Set[int]]:
        """Classify processes as owned, other, or uncertain."""
        ownership = {
            'owned': set(),
            'other': set(),
            'uncertain': set()
        }

        current_pid = os.getpid()
        current_process = processes.get(current_pid)

        if current_process:
            ownership['owned'].add(current_pid)

            # Add parent and child processes
            await self._add_related_processes(
                current_process, processes, ownership['owned']
            )

        # Classify remaining processes
        for pid, proc_info in processes.items():
            if pid not in ownership['owned']:
                confidence = self._calculate_ownership_confidence(proc_info)

                if confidence > self.config.ownership_threshold:
                    ownership['owned'].add(pid)
                elif confidence < self.config.other_threshold:
                    ownership['other'].add(pid)
                else:
                    ownership['uncertain'].add(pid)

        return ownership

    def _calculate_ownership_confidence(self, proc_info: 'ProcessInfo') -> float:
        """Calculate confidence that a process belongs to self."""
        confidence_factors = []

        # Check if process name matches known patterns
        name_confidence = self._check_name_patterns(proc_info.name)
        confidence_factors.append(name_confidence * 0.3)

        # Check executable path
        exe_confidence = self._check_executable_path(proc_info.exe)
        confidence_factors.append(exe_confidence * 0.2)

        # Check command line arguments
        cmdline_confidence = self._check_command_line(proc_info.cmdline)
        confidence_factors.append(cmdline_confidence * 0.2)

        # Check resource usage patterns
        resource_confidence = self._check_resource_patterns(proc_info)
        confidence_factors.append(resource_confidence * 0.3)

        return sum(confidence_factors)


class MemoryBoundaryMonitor:
    """
    Monitor for detecting and tracking memory-level boundaries.

    Tracks memory allocations, shared memory segments, and
    memory access patterns to distinguish self from other.
    """

    def __init__(self, config: 'MemoryBoundaryConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MemoryBoundary")

        # Memory tracking
        self._allocated_regions = {}
        self._shared_segments = {}
        self._access_patterns = defaultdict(list)

        # Memory mapping
        self._memory_mapper = MemoryMapper()
        self._allocation_tracker = AllocationTracker()

        # Pattern analysis
        self._pattern_analyzer = MemoryPatternAnalyzer()

    async def initialize(self):
        """Initialize the memory boundary monitor."""
        self.logger.info("Initializing memory boundary monitor")

        await self._memory_mapper.initialize()
        await self._allocation_tracker.start()

        # Map initial memory layout
        await self._map_initial_memory()

        self.logger.info("Memory boundary monitor initialized")

    async def detect_boundaries(self, memory_data: 'MemoryData') -> 'MemoryBoundaries':
        """Detect current memory boundaries."""
        detection_start = time.time()

        # Map current memory regions
        memory_regions = await self._memory_mapper.map_regions()

        # Classify memory ownership
        ownership_map = await self._classify_memory_ownership(memory_regions)

        # Analyze allocation patterns
        allocation_analysis = await self._analyze_allocation_patterns()

        # Detect shared memory usage
        shared_memory_analysis = await self._analyze_shared_memory()

        # Calculate memory boundary confidence
        confidence = self._calculate_memory_confidence(
            ownership_map, allocation_analysis, shared_memory_analysis
        )

        return MemoryBoundaries(
            timestamp=time.time(),
            owned_regions=ownership_map['owned'],
            shared_regions=ownership_map['shared'],
            other_regions=ownership_map['other'],
            allocation_patterns=allocation_analysis,
            shared_memory_info=shared_memory_analysis,
            confidence=confidence,
            detection_time=time.time() - detection_start
        )

    async def _map_initial_memory(self):
        """Map the initial memory layout of the process."""
        try:
            process = psutil.Process()
            memory_maps = process.memory_maps()

            for mmap in memory_maps:
                region = MemoryRegion(
                    start_address=int(mmap.addr.split('-')[0], 16),
                    end_address=int(mmap.addr.split('-')[1], 16),
                    permissions=mmap.perms,
                    path=mmap.path,
                    rss=mmap.rss,
                    size=mmap.size,
                    pss=mmap.pss
                )

                self._allocated_regions[region.start_address] = region

        except Exception as e:
            self.logger.error(f"Error mapping initial memory: {e}")


class NetworkBoundaryMonitor:
    """
    Monitor for detecting and tracking network-level boundaries.

    Monitors network connections, traffic patterns, and communication
    endpoints to distinguish self-initiated vs. external network activity.
    """

    def __init__(self, config: 'NetworkBoundaryConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NetworkBoundary")

        # Connection tracking
        self._active_connections = {}
        self._listening_ports = set()
        self._traffic_patterns = defaultdict(list)

        # Network monitoring
        self._connection_monitor = ConnectionMonitor()
        self._traffic_analyzer = TrafficAnalyzer()

        # Pattern recognition
        self._pattern_recognizer = NetworkPatternRecognizer()

    async def initialize(self):
        """Initialize the network boundary monitor."""
        self.logger.info("Initializing network boundary monitor")

        await self._connection_monitor.start()
        await self._traffic_analyzer.start()

        # Establish baseline network state
        await self._establish_network_baseline()

        self.logger.info("Network boundary monitor initialized")

    async def detect_boundaries(self, network_data: 'NetworkData') -> 'NetworkBoundaries':
        """Detect current network boundaries."""
        detection_start = time.time()

        # Scan current connections
        current_connections = await self._scan_network_connections()

        # Classify connection ownership
        connection_ownership = await self._classify_connection_ownership(
            current_connections
        )

        # Analyze traffic patterns
        traffic_analysis = await self._analyze_traffic_patterns()

        # Detect listening services
        listening_analysis = await self._analyze_listening_ports()

        # Calculate network boundary confidence
        confidence = self._calculate_network_confidence(
            connection_ownership, traffic_analysis, listening_analysis
        )

        return NetworkBoundaries(
            timestamp=time.time(),
            owned_connections=connection_ownership['owned'],
            other_connections=connection_ownership['other'],
            listening_ports=listening_analysis,
            traffic_patterns=traffic_analysis,
            confidence=confidence,
            detection_time=time.time() - detection_start
        )

    async def _scan_network_connections(self) -> List['NetworkConnection']:
        """Scan all current network connections."""
        connections = []

        try:
            # Get connections for current process
            current_process = psutil.Process()
            process_connections = current_process.connections()

            for conn in process_connections:
                if conn.laddr:
                    connection = NetworkConnection(
                        local_addr=conn.laddr.ip if conn.laddr else None,
                        local_port=conn.laddr.port if conn.laddr else None,
                        remote_addr=conn.raddr.ip if conn.raddr else None,
                        remote_port=conn.raddr.port if conn.raddr else None,
                        status=conn.status,
                        family=conn.family,
                        type=conn.type,
                        pid=current_process.pid
                    )
                    connections.append(connection)

            # Get system-wide connections for comparison
            system_connections = psutil.net_connections()

            for conn in system_connections:
                if conn.pid != current_process.pid and conn.laddr:
                    connection = NetworkConnection(
                        local_addr=conn.laddr.ip if conn.laddr else None,
                        local_port=conn.laddr.port if conn.laddr else None,
                        remote_addr=conn.raddr.ip if conn.raddr else None,
                        remote_port=conn.raddr.port if conn.raddr else None,
                        status=conn.status,
                        family=conn.family,
                        type=conn.type,
                        pid=conn.pid
                    )
                    connections.append(connection)

        except Exception as e:
            self.logger.error(f"Error scanning network connections: {e}")

        return connections


class BoundaryIntegrator:
    """
    Integrates boundary information from different monitors to provide
    unified boundary detection results.
    """

    def __init__(self, config: 'BoundaryIntegratorConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BoundaryIntegrator")

        # Integration weights
        self.integration_weights = {
            'process': config.process_weight,
            'memory': config.memory_weight,
            'network': config.network_weight
        }

        # Consistency checker
        self._consistency_checker = BoundaryConsistencyChecker()

        # Conflict resolver
        self._conflict_resolver = BoundaryConflictResolver()

    async def integrate_boundaries(
        self,
        process_boundaries: 'ProcessBoundaries',
        memory_boundaries: 'MemoryBoundaries',
        network_boundaries: 'NetworkBoundaries'
    ) -> 'IntegratedBoundaries':
        """Integrate boundaries from different monitors."""
        integration_start = time.time()

        # Check consistency across boundary types
        consistency_analysis = await self._consistency_checker.check_consistency(
            process_boundaries, memory_boundaries, network_boundaries
        )

        # Resolve conflicts if any
        if consistency_analysis.has_conflicts:
            resolved_boundaries = await self._conflict_resolver.resolve_conflicts(
                process_boundaries, memory_boundaries, network_boundaries,
                consistency_analysis.conflicts
            )
            process_boundaries, memory_boundaries, network_boundaries = resolved_boundaries

        # Compute integrated confidence
        integrated_confidence = self._compute_integrated_confidence(
            process_boundaries.confidence,
            memory_boundaries.confidence,
            network_boundaries.confidence
        )

        # Create unified boundary map
        unified_map = await self._create_unified_boundary_map(
            process_boundaries, memory_boundaries, network_boundaries
        )

        # Generate boundary summary
        boundary_summary = self._generate_boundary_summary(
            process_boundaries, memory_boundaries, network_boundaries,
            integrated_confidence
        )

        return IntegratedBoundaries(
            timestamp=time.time(),
            process_boundaries=process_boundaries,
            memory_boundaries=memory_boundaries,
            network_boundaries=network_boundaries,
            unified_boundary_map=unified_map,
            integrated_confidence=integrated_confidence,
            consistency_analysis=consistency_analysis,
            boundary_summary=boundary_summary,
            integration_time=time.time() - integration_start
        )

    def _compute_integrated_confidence(
        self,
        process_conf: float,
        memory_conf: float,
        network_conf: float
    ) -> float:
        """Compute integrated confidence score."""
        weights = self.integration_weights

        weighted_sum = (
            process_conf * weights['process'] +
            memory_conf * weights['memory'] +
            network_conf * weights['network']
        )

        weight_total = sum(weights.values())

        return weighted_sum / weight_total


# Data structures for boundary detection
@dataclass
class ProcessBoundaries:
    """Result of process boundary detection."""
    timestamp: float
    owned_processes: Set[int]
    other_processes: Set[int]
    uncertain_processes: Set[int]
    process_hierarchy: Dict[int, List[int]]
    resource_boundaries: 'ResourceBoundaries'
    boundary_changes: List['BoundaryChange']
    confidence: float
    detection_time: float


@dataclass
class MemoryBoundaries:
    """Result of memory boundary detection."""
    timestamp: float
    owned_regions: List['MemoryRegion']
    shared_regions: List['SharedMemoryRegion']
    other_regions: List['MemoryRegion']
    allocation_patterns: 'AllocationPatterns'
    shared_memory_info: 'SharedMemoryInfo'
    confidence: float
    detection_time: float


@dataclass
class NetworkBoundaries:
    """Result of network boundary detection."""
    timestamp: float
    owned_connections: List['NetworkConnection']
    other_connections: List['NetworkConnection']
    listening_ports: List['ListeningPort']
    traffic_patterns: 'TrafficPatterns'
    confidence: float
    detection_time: float


@dataclass
class IntegratedBoundaries:
    """Integrated boundary detection result."""
    timestamp: float
    process_boundaries: ProcessBoundaries
    memory_boundaries: MemoryBoundaries
    network_boundaries: NetworkBoundaries
    unified_boundary_map: 'UnifiedBoundaryMap'
    integrated_confidence: float
    consistency_analysis: 'ConsistencyAnalysis'
    boundary_summary: 'BoundarySummary'
    integration_time: float


class BoundaryViolationDetector:
    """
    Detects violations of established boundaries in real-time.
    """

    def __init__(self, config: 'ViolationDetectorConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BoundaryViolation")

        # Violation patterns
        self._violation_patterns = {
            'unauthorized_process': self._detect_unauthorized_process,
            'memory_intrusion': self._detect_memory_intrusion,
            'network_breach': self._detect_network_breach,
            'resource_theft': self._detect_resource_theft
        }

        # Alert system
        self._alert_system = BoundaryAlertSystem()

    async def detect_violations(
        self,
        current_boundaries: 'IntegratedBoundaries',
        baseline_boundaries: 'IntegratedBoundaries'
    ) -> List['BoundaryViolation']:
        """Detect boundary violations by comparing current state to baseline."""
        violations = []

        for pattern_name, detector in self._violation_patterns.items():
            pattern_violations = await detector(current_boundaries, baseline_boundaries)
            violations.extend(pattern_violations)

        # Filter and prioritize violations
        filtered_violations = self._filter_violations(violations)

        # Send alerts for critical violations
        await self._alert_system.process_violations(filtered_violations)

        return filtered_violations

    async def _detect_unauthorized_process(
        self,
        current: 'IntegratedBoundaries',
        baseline: 'IntegratedBoundaries'
    ) -> List['BoundaryViolation']:
        """Detect unauthorized processes accessing self resources."""
        violations = []

        # Check for new processes accessing owned resources
        current_processes = current.process_boundaries.owned_processes
        baseline_processes = baseline.process_boundaries.owned_processes

        new_processes = current_processes - baseline_processes

        for pid in new_processes:
            # Analyze if this process has legitimate access
            legitimacy_score = await self._assess_process_legitimacy(pid, current)

            if legitimacy_score < self.config.legitimacy_threshold:
                violation = BoundaryViolation(
                    type='unauthorized_process',
                    severity='high',
                    timestamp=time.time(),
                    details={
                        'pid': pid,
                        'legitimacy_score': legitimacy_score
                    },
                    recommended_actions=['investigate_process', 'terminate_if_malicious']
                )
                violations.append(violation)

        return violations
```

This boundary detection system provides comprehensive monitoring across process, memory, and network levels to establish and maintain clear self-other boundaries in the computational environment.