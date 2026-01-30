# Contemplative States Technical Requirements

## Overview

This document specifies the technical requirements for Form 36 (Contemplative & Meditative States), covering performance, integration, reliability, security, monitoring, and testing. These requirements ensure the system can reliably process multi-modal contemplative data -- from real-time neural recordings to phenomenological reports -- while maintaining scientific rigor and practitioner privacy.

---

## Performance Requirements

### Latency

| Operation | Target Latency | Maximum Latency | Notes |
|-----------|---------------|-----------------|-------|
| State detection (single EEG window) | < 50 ms | 100 ms | Real-time feedback requires sub-100ms |
| Phenomenological report ingestion | < 200 ms | 500 ms | Report parsing and classification |
| Cross-tradition state mapping lookup | < 100 ms | 300 ms | Cached mapping table queries |
| Neural recording preprocessing | < 2 s per 30s window | 5 s | Artifact rejection, band decomposition |
| Session summary generation | < 5 s | 15 s | Post-session analytics pipeline |
| Full developmental map query | < 500 ms | 2 s | Tradition-wide stage retrieval |
| State transition probability calculation | < 100 ms | 250 ms | Markov model inference |

### Throughput

| Metric | Minimum | Target | Notes |
|--------|---------|--------|-------|
| Concurrent meditation sessions monitored | 50 | 200 | Real-time neural + physiological streams |
| Phenomenological reports processed/minute | 100 | 500 | Structured and free-text reports |
| EEG channel processing rate | 64 channels at 256 Hz | 128 channels at 512 Hz | Per-session stream |
| State detection inferences/second | 10 | 50 | Across all active sessions |
| Cross-tradition mapping requests/second | 100 | 500 | API-level throughput |
| Batch session analysis (sessions/hour) | 500 | 2,000 | Offline retrospective analysis |

### Memory

| Component | Maximum Memory | Notes |
|-----------|---------------|-------|
| State detection model (per session) | 128 MB | Loaded calibration + active model |
| EEG buffer (per session) | 256 MB | Rolling 5-minute high-resolution window |
| Tradition knowledge graph (in-memory) | 2 GB | Full cross-tradition mapping cache |
| Phenomenological NLP model | 1 GB | Language model for report analysis |
| Session history index | 512 MB | Hot index for active practitioner lookups |
| Total system working set | 16 GB | All components active |

---

## Integration Requirements

### APIs

#### Contemplative Session API

```
POST   /api/v1/sessions                     # Create new session
GET    /api/v1/sessions/{session_id}         # Retrieve session details
PUT    /api/v1/sessions/{session_id}/state   # Update detected state
POST   /api/v1/sessions/{session_id}/report  # Submit phenomenological report
DELETE /api/v1/sessions/{session_id}         # End and archive session
GET    /api/v1/sessions/{session_id}/timeline  # Get state transition timeline
```

#### State Mapping API

```
GET    /api/v1/states                        # List all canonical states
GET    /api/v1/states/{state_id}             # Get state profile
GET    /api/v1/states/{state_id}/equivalences  # Cross-tradition equivalences
POST   /api/v1/states/detect                 # Submit data for state detection
GET    /api/v1/states/transitions/{from}/{to}  # Transition probability and path
```

#### Tradition API

```
GET    /api/v1/traditions                    # List all traditions
GET    /api/v1/traditions/{tradition_id}     # Full tradition profile
GET    /api/v1/traditions/{tradition_id}/map  # Developmental map
GET    /api/v1/traditions/{tradition_id}/practices  # Practice catalog
```

#### Practitioner API

```
POST   /api/v1/practitioners                 # Register practitioner (anonymized)
GET    /api/v1/practitioners/{id}/profile    # Get practitioner profile
PUT    /api/v1/practitioners/{id}/calibrate  # Update calibration data
GET    /api/v1/practitioners/{id}/history    # Session history summary
```

### Cross-Form Interfaces

#### Form 39 (Trauma Consciousness) Interface

- **Protocol**: Asynchronous event-driven messaging
- **Exchange format**: `ContemplativeTraumaInterface` data structure
- **Events emitted**:
  - `contemplative.depth_warning`: When absorption depth approaches trauma-sensitive thresholds
  - `contemplative.dissociative_pattern`: When state patterns suggest dissociation rather than meditation
  - `contemplative.grounding_needed`: When practitioner needs grounding intervention
- **Events consumed**:
  - `trauma.risk_assessment_updated`: Updated trauma risk profile for a practitioner
  - `trauma.contraindication_added`: New practice contraindication identified
- **Latency requirement**: Event delivery within 500 ms for safety-critical alerts
- **Retry policy**: At-least-once delivery with idempotent handlers

#### Form 40 (Xenoconsciousness) Interface

- **Protocol**: Request-response with optional streaming
- **Exchange format**: `ContemplativeXenoInterface` data structure
- **Operations**:
  - Provide contemplative state phenomenology for cross-species comparison
  - Receive substrate-independence assessments for contemplative states
  - Exchange universal consciousness marker data
- **Latency requirement**: < 2 s for standard queries
- **Caching**: Cross-species analogue mappings cached with 1-hour TTL

### External System Integration

| System | Protocol | Purpose |
|--------|----------|---------|
| EEG devices (OpenBCI, Muse, etc.) | LSL (Lab Streaming Layer) | Real-time neural data ingestion |
| fMRI data repositories | BIDS format over HTTPS | Import neuroimaging datasets |
| Research databases (PubMed) | REST API | Neural finding cross-referencing |
| Phenomenological survey tools | Webhook / REST | Report ingestion from external surveys |
| Timer/meditation apps | MQTT / WebSocket | Session event synchronization |

---

## Reliability Requirements

### Error Handling

| Error Category | Handling Strategy | Recovery Time |
|---------------|-------------------|---------------|
| EEG signal dropout (< 5 seconds) | Interpolation from surrounding channels | Immediate |
| EEG signal dropout (>= 5 seconds) | Mark gap, pause state detection, alert | < 1 s to resume after signal returns |
| Phenomenological report parse failure | Fall back to raw text storage, queue for manual review | Immediate (degraded mode) |
| State detection model failure | Revert to rule-based heuristic detector | < 2 s |
| Cross-tradition mapping not found | Return partial mapping with confidence annotation | Immediate |
| Database write failure | Queue to local write-ahead log, retry with exponential backoff | < 30 s for retry |
| Neural recording corruption | Discard corrupted segment, annotate timeline gap | Immediate |

### Fault Tolerance

- **State detection pipeline**: Dual-model hot standby. If the primary ML model fails, a simpler rule-based detector activates within 2 seconds.
- **Session data persistence**: All session data written to a local write-ahead log before network storage. No data loss on network partition.
- **Graceful degradation hierarchy**:
  1. Full operation: ML state detection + neural + phenomenological integration
  2. Degraded level 1: Rule-based state detection + phenomenological only
  3. Degraded level 2: Session recording only (no real-time detection)
  4. Minimum viable: Timestamped raw data capture to local storage
- **Recovery point objective (RPO)**: Zero data loss for active sessions
- **Recovery time objective (RTO)**: < 30 seconds to degraded level 1, < 5 minutes to full operation

### Availability

- **Target availability**: 99.5% uptime for session management and recording
- **Planned maintenance windows**: Off-peak hours only, with 48-hour advance notice
- **State detection availability**: 99.0% (lower due to model complexity)
- **Data query availability**: 99.9% for read-only historical queries

---

## Security and Privacy

### Data Classification

| Data Type | Classification | Retention |
|-----------|---------------|-----------|
| Practitioner identity | Highly Sensitive | Encrypted at rest, access-controlled |
| Neural recordings (raw) | Sensitive | 5 years or per consent agreement |
| Phenomenological reports | Sensitive | Anonymized after research use |
| Session metadata | Internal | 10 years |
| Tradition profiles | Public | Indefinite |
| State taxonomy | Public | Indefinite |
| Cross-tradition mappings | Public | Indefinite |

### Privacy Requirements

- **Anonymization**: All practitioner data must be anonymized using irreversible hashing before inclusion in any research dataset or cross-form exchange.
- **Consent management**: The system must track granular consent for each data type (neural, phenomenological, physiological) independently.
- **Right to deletion**: Practitioner data must be fully deletable within 30 days of request, including all derived features and model contributions.
- **Data minimization**: Only collect the minimum data required for the active session's configured analysis level.
- **Sacred content protection**: Certain tradition-specific teachings flagged as restricted must not be exposed via public APIs without appropriate authorization from tradition representatives.

### Access Control

- **Role-based access**: Practitioner, Researcher, Tradition Authority, System Administrator
- **Practitioner**: Access to own session data and general state/tradition information only
- **Researcher**: Access to anonymized aggregate data and consented individual datasets
- **Tradition Authority**: Read/write access to their tradition's profiles and developmental maps
- **Audit logging**: All data access logged with accessor identity, timestamp, data accessed, and purpose

### Encryption

- **At rest**: AES-256 for all sensitive and highly sensitive data
- **In transit**: TLS 1.3 minimum for all API communications
- **Neural data streams**: End-to-end encryption for real-time EEG/physiological streams

---

## Monitoring and Observability

### Key Metrics

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| State detection accuracy (validated sessions) | Gauge | < 75% triggers review |
| State detection latency (p95) | Histogram | > 100 ms |
| Active session count | Gauge | > 90% capacity |
| EEG signal quality (active sessions) | Gauge | < 60% quality score |
| Phenomenological report parse success rate | Counter/Rate | < 90% |
| Cross-form event delivery latency | Histogram | > 500 ms for safety events |
| Model inference error rate | Counter/Rate | > 1% |
| Session data persistence lag | Gauge | > 5 s |
| Memory utilization per session | Gauge | > 200 MB |

### Logging Requirements

- **Structured logging**: JSON format with correlation IDs linking session, practitioner (anonymized), and request context
- **Log levels**: DEBUG (development only), INFO (session lifecycle events), WARN (degraded operation), ERROR (failures), CRITICAL (safety-related failures)
- **Retention**: 30 days hot storage, 1 year cold storage, 7 years for audit logs
- **Safety event logging**: All contemplative depth warnings, dissociative pattern detections, and grounding interventions logged at WARN level with full context

### Dashboards

- **Operational dashboard**: Active sessions, system health, model performance, queue depths
- **Research dashboard**: Aggregate state distributions, practice patterns, neural finding correlations
- **Safety dashboard**: Trauma-related alerts, dissociative pattern frequency, intervention rates

### Tracing

- **Distributed tracing**: OpenTelemetry-compatible traces for all cross-form communications
- **Trace sampling**: 100% for safety-related events, 10% for routine operations
- **Span requirements**: State detection pipeline, report processing, cross-form exchange, and session lifecycle must each be traced as distinct spans

---

## Testing Requirements

### Unit Testing

- **Coverage target**: >= 90% line coverage for state detection logic, report parsing, and data model validation
- **State detection unit tests**: Each canonical contemplative state must have at least 5 synthetic test inputs verifying correct classification
- **Enumeration completeness**: All enum types must have tests verifying exhaustive handling in switch/match statements
- **Data validation tests**: All dataclass fields with constraints must have boundary-value tests

### Integration Testing

- **Cross-form integration**: End-to-end tests for Form 36 to Form 39 safety event flow (depth warning to trauma response)
- **Cross-form integration**: End-to-end tests for Form 36 to Form 40 state comparison queries
- **EEG pipeline integration**: Simulated EEG stream from ingestion through state detection to session timeline output
- **Report pipeline integration**: Phenomenological report submission through NLP analysis to state annotation
- **Database integration**: Full CRUD cycle for all core data models with referential integrity verification

### Performance Testing

- **Load testing**: Simulate 200 concurrent sessions with mixed neural and phenomenological inputs for 1 hour
- **Latency benchmarks**: All latency targets from the Performance Requirements section must be verified under load
- **Memory leak detection**: 24-hour soak test with continuous session creation and teardown
- **Throughput verification**: Batch processing pipeline must sustain target throughput for 10,000 session analyses

### Specialized Testing

- **State detection accuracy**: Validated against a curated dataset of expert-annotated meditation sessions (minimum 500 sessions across 10 traditions)
- **Cross-tradition mapping validation**: Reviewed by at least two tradition scholars per mapping for accuracy
- **Phenomenological NLP validation**: Report classification accuracy tested against human-coded ground truth (target: >= 85% agreement)
- **Safety-critical path testing**: All trauma-sensitive alert paths tested with fault injection (network delays, model failures) to verify timely degraded-mode alerting
- **Temporal consistency testing**: State timeline generation verified for monotonicity, gap detection, and transition coherence

### Regression Testing

- **Automated regression suite**: Full suite runs on every code change to state detection models, report parsers, or cross-form interfaces
- **Model regression**: New model versions must match or exceed accuracy benchmarks on the canonical validation dataset before deployment
- **API contract testing**: Consumer-driven contract tests for all cross-form interfaces
- **Performance regression**: Latency and throughput benchmarks tracked over time with automatic alerts on > 10% degradation
