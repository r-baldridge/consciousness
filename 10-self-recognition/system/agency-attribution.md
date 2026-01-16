# Form 10: Self-Recognition - Agency Attribution System

## Predictive Agency Attribution Engine

```python
import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import threading
from abc import ABC, abstractmethod

class PredictionSystem:
    """
    System for creating and evaluating predictions about self-initiated actions.

    This system maintains forward models that predict the outcomes of
    intended actions, enabling accurate attribution of agency based on
    prediction-outcome matching.
    """

    def __init__(self, config: 'PredictionConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PredictionSystem")

        # Forward models for different action types
        self._forward_models = {}
        self._model_factory = ForwardModelFactory()

        # Intention tracking
        self._intention_tracker = IntentionTracker()
        self._active_predictions = {}

        # Prediction evaluation
        self._evaluation_history = deque(maxlen=config.evaluation_history_size)
        self._accuracy_metrics = AccuracyMetrics()

        # Model learning and adaptation
        self._model_learner = ModelLearner()
        self._adaptation_scheduler = AdaptationScheduler()

    async def initialize(self):
        """Initialize the prediction system."""
        self.logger.info("Initializing prediction system")

        # Initialize model factory
        await self._model_factory.initialize()

        # Load or create initial forward models
        await self._initialize_forward_models()

        # Start intention tracking
        await self._intention_tracker.start()

        # Begin model adaptation
        self._adaptation_scheduler.start()

        self.logger.info("Prediction system initialized")

    async def create_prediction(
        self,
        intention: 'Intention',
        context: 'PredictionContext'
    ) -> 'AgencyPrediction':
        """Create a prediction for a self-initiated action."""
        prediction_start = time.time()

        # Get appropriate forward model
        model = await self._get_forward_model(intention.action_type)

        # Create prediction using the model
        predicted_outcome = await model.predict(intention, context)

        # Generate monitoring criteria
        monitoring_criteria = self._generate_monitoring_criteria(
            intention, predicted_outcome
        )

        # Create prediction object
        prediction = AgencyPrediction(
            prediction_id=self._generate_prediction_id(),
            intention_id=intention.id,
            action_type=intention.action_type,
            predicted_outcome=predicted_outcome,
            prediction_timestamp=time.time(),
            confidence=predicted_outcome.confidence,
            model_id=model.model_id,
            context_snapshot=context.snapshot(),
            monitoring_criteria=monitoring_criteria,
            creation_time=time.time() - prediction_start
        )

        # Register for monitoring
        self._active_predictions[prediction.prediction_id] = prediction

        # Track intention
        self._intention_tracker.track_intention(intention, prediction)

        return prediction

    async def check_prediction_match(
        self,
        event: 'Event'
    ) -> 'PredictionMatchResult':
        """Check if an event matches any active predictions."""
        best_match = None
        best_score = 0.0

        for prediction in self._active_predictions.values():
            # Check if event could match this prediction
            if self._could_match_prediction(event, prediction):
                match_score = await self._calculate_match_score(event, prediction)

                if match_score > best_score:
                    best_score = match_score
                    best_match = prediction

        if best_match and best_score > self.config.match_threshold:
            # Create match result
            match_result = PredictionMatchResult(
                prediction=best_match,
                event=event,
                match_score=best_score,
                match_quality=self._assess_match_quality(event, best_match),
                temporal_accuracy=self._calculate_temporal_accuracy(event, best_match),
                outcome_accuracy=self._calculate_outcome_accuracy(event, best_match)
            )

            # Record evaluation for learning
            await self._record_prediction_evaluation(match_result)

            return match_result

        # No matching prediction found
        return PredictionMatchResult(
            prediction=None,
            event=event,
            match_score=0.0,
            match_quality=MatchQuality.NO_MATCH,
            temporal_accuracy=0.0,
            outcome_accuracy=0.0
        )

    async def _get_forward_model(self, action_type: str) -> 'ForwardModel':
        """Get or create forward model for action type."""
        if action_type not in self._forward_models:
            model = await self._model_factory.create_model(action_type)
            self._forward_models[action_type] = model

        return self._forward_models[action_type]

    async def _calculate_match_score(
        self,
        event: 'Event',
        prediction: 'AgencyPrediction'
    ) -> float:
        """Calculate how well an event matches a prediction."""
        score_components = []

        # Temporal matching
        temporal_score = self._calculate_temporal_match(event, prediction)
        score_components.append(temporal_score * self.config.temporal_weight)

        # Outcome matching
        outcome_score = self._calculate_outcome_match(event, prediction)
        score_components.append(outcome_score * self.config.outcome_weight)

        # Context matching
        context_score = self._calculate_context_match(event, prediction)
        score_components.append(context_score * self.config.context_weight)

        # Signature matching
        signature_score = self._calculate_signature_match(event, prediction)
        score_components.append(signature_score * self.config.signature_weight)

        return sum(score_components)


class CorrelationTracker:
    """
    Tracks temporal correlations between intentions and observed outcomes.

    Analyzes the timing relationships between self-initiated actions
    and their observed effects to support agency attribution.
    """

    def __init__(self, config: 'CorrelationConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CorrelationTracker")

        # Correlation analysis
        self._correlation_analyzer = CorrelationAnalyzer()
        self._temporal_window_manager = TemporalWindowManager()

        # Event tracking
        self._event_history = deque(maxlen=config.event_history_size)
        self._intention_history = deque(maxlen=config.intention_history_size)

        # Correlation patterns
        self._correlation_patterns = {}
        self._pattern_learner = PatternLearner()

    async def initialize(self):
        """Initialize the correlation tracker."""
        self.logger.info("Initializing correlation tracker")

        await self._correlation_analyzer.initialize()
        await self._pattern_learner.initialize()

        # Load existing correlation patterns
        await self._load_correlation_patterns()

        self.logger.info("Correlation tracker initialized")

    async def calculate_correlation(
        self,
        event: 'Event',
        temporal_window: 'TemporalWindow'
    ) -> 'TemporalCorrelationResult':
        """Calculate temporal correlation for an event."""
        correlation_start = time.time()

        # Get relevant intentions within temporal window
        relevant_intentions = self._get_relevant_intentions(event, temporal_window)

        # Calculate correlations for each intention
        intention_correlations = []
        for intention in relevant_intentions:
            correlation = await self._calculate_intention_correlation(
                event, intention
            )
            intention_correlations.append(correlation)

        # Find best correlation
        best_correlation = max(
            intention_correlations,
            key=lambda c: c.correlation_strength,
            default=None
        )

        # Analyze correlation patterns
        pattern_analysis = await self._analyze_correlation_patterns(
            event, relevant_intentions, intention_correlations
        )

        return TemporalCorrelationResult(
            event=event,
            relevant_intentions=relevant_intentions,
            intention_correlations=intention_correlations,
            best_correlation=best_correlation,
            pattern_analysis=pattern_analysis,
            overall_correlation_score=best_correlation.correlation_strength if best_correlation else 0.0,
            calculation_time=time.time() - correlation_start
        )

    async def _calculate_intention_correlation(
        self,
        event: 'Event',
        intention: 'Intention'
    ) -> 'IntentionCorrelation':
        """Calculate correlation between an event and an intention."""
        # Temporal correlation
        temporal_correlation = self._calculate_temporal_correlation(event, intention)

        # Causal correlation
        causal_correlation = self._calculate_causal_correlation(event, intention)

        # Contextual correlation
        contextual_correlation = self._calculate_contextual_correlation(event, intention)

        # Feature correlation
        feature_correlation = self._calculate_feature_correlation(event, intention)

        # Combine correlations
        overall_correlation = self._combine_correlations(
            temporal_correlation, causal_correlation,
            contextual_correlation, feature_correlation
        )

        return IntentionCorrelation(
            intention=intention,
            temporal_correlation=temporal_correlation,
            causal_correlation=causal_correlation,
            contextual_correlation=contextual_correlation,
            feature_correlation=feature_correlation,
            correlation_strength=overall_correlation,
            confidence=self._calculate_correlation_confidence(overall_correlation)
        )

    def _calculate_temporal_correlation(
        self,
        event: 'Event',
        intention: 'Intention'
    ) -> float:
        """Calculate temporal correlation between event and intention."""
        time_diff = abs(event.timestamp - intention.timestamp)
        max_delay = self.config.max_correlation_delay

        if time_diff > max_delay:
            return 0.0

        # Exponential decay based on time difference
        correlation = np.exp(-time_diff / self.config.temporal_decay_constant)

        # Adjust for expected delay patterns
        expected_delay = self._get_expected_delay(intention.action_type)
        delay_factor = self._calculate_delay_factor(time_diff, expected_delay)

        return correlation * delay_factor


class CausalAnalyzer:
    """
    Analyzes causal relationships to support agency attribution.

    Determines the likelihood that observed events are caused by
    self-initiated actions through causal reasoning and inference.
    """

    def __init__(self, config: 'CausalConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CausalAnalyzer")

        # Causal models
        self._causal_models = {}
        self._causal_graph_builder = CausalGraphBuilder()

        # Inference engines
        self._bayesian_inferencer = BayesianInferencer()
        self._causal_inferencer = CausalInferencer()

        # Evidence tracking
        self._causal_evidence = defaultdict(list)
        self._intervention_tracker = InterventionTracker()

    async def initialize(self):
        """Initialize the causal analyzer."""
        self.logger.info("Initializing causal analyzer")

        await self._causal_graph_builder.initialize()
        await self._bayesian_inferencer.initialize()

        # Build initial causal models
        await self._build_initial_causal_models()

        self.logger.info("Causal analyzer initialized")

    async def analyze_causality(
        self,
        event: 'Event',
        context: 'CausalContext'
    ) -> 'CausalAnalysisResult':
        """Analyze causal relationships for an event."""
        analysis_start = time.time()

        # Get relevant causal model
        causal_model = await self._get_causal_model(event.event_type)

        # Perform causal inference
        causal_inference = await self._perform_causal_inference(
            event, context, causal_model
        )

        # Calculate causal strength
        causal_strength = self._calculate_causal_strength(
            causal_inference, context
        )

        # Assess alternative explanations
        alternative_analysis = await self._assess_alternative_explanations(
            event, context, causal_model
        )

        # Generate causal explanation
        causal_explanation = self._generate_causal_explanation(
            causal_inference, alternative_analysis
        )

        return CausalAnalysisResult(
            event=event,
            causal_model_used=causal_model.model_id,
            causal_inference=causal_inference,
            causal_strength=causal_strength,
            alternative_explanations=alternative_analysis,
            causal_explanation=causal_explanation,
            confidence=self._calculate_causal_confidence(causal_strength),
            analysis_time=time.time() - analysis_start
        )

    async def _perform_causal_inference(
        self,
        event: 'Event',
        context: 'CausalContext',
        causal_model: 'CausalModel'
    ) -> 'CausalInference':
        """Perform causal inference using the specified model."""
        # Collect evidence
        evidence = await self._collect_causal_evidence(event, context)

        # Apply causal model
        inference_result = await causal_model.infer(evidence)

        # Calculate intervention likelihood
        intervention_likelihood = self._calculate_intervention_likelihood(
            event, context, evidence
        )

        # Assess causal necessity and sufficiency
        necessity_score = self._assess_causal_necessity(event, evidence)
        sufficiency_score = self._assess_causal_sufficiency(event, evidence)

        return CausalInference(
            evidence=evidence,
            inference_result=inference_result,
            intervention_likelihood=intervention_likelihood,
            necessity_score=necessity_score,
            sufficiency_score=sufficiency_score,
            causal_pathway=inference_result.dominant_pathway
        )


class ConfidenceCalibrator:
    """
    Calibrates confidence scores for agency attribution decisions.

    Adjusts raw attribution scores based on historical accuracy,
    context factors, and uncertainty quantification.
    """

    def __init__(self, config: 'ConfidenceConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ConfidenceCalibrator")

        # Calibration models
        self._calibration_model = CalibrationModel()
        self._uncertainty_quantifier = UncertaintyQuantifier()

        # Historical accuracy tracking
        self._accuracy_history = deque(maxlen=config.history_size)
        self._calibration_data = CalibrationData()

        # Context-dependent calibration
        self._context_calibrators = {}
        self._meta_calibrator = MetaCalibrator()

    async def initialize(self):
        """Initialize the confidence calibrator."""
        self.logger.info("Initializing confidence calibrator")

        await self._calibration_model.initialize()
        await self._uncertainty_quantifier.initialize()

        # Load historical calibration data
        await self._load_calibration_data()

        # Train initial calibration model
        await self._train_calibration_model()

        self.logger.info("Confidence calibrator initialized")

    async def calibrate(
        self,
        raw_score: float,
        context: 'AgencyContext',
        event: 'Event'
    ) -> 'CalibratedConfidence':
        """Calibrate raw agency score to produce calibrated confidence."""
        calibration_start = time.time()

        # Apply base calibration
        base_calibrated = await self._calibration_model.calibrate(raw_score)

        # Apply context-specific calibration
        context_adjustment = await self._apply_context_calibration(
            base_calibrated, context
        )

        # Quantify uncertainty
        uncertainty_estimate = await self._uncertainty_quantifier.estimate(
            raw_score, context, event
        )

        # Apply meta-calibration
        meta_calibrated = await self._meta_calibrator.calibrate(
            context_adjustment, uncertainty_estimate, context
        )

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            meta_calibrated, uncertainty_estimate
        )

        return CalibratedConfidence(
            raw_score=raw_score,
            calibrated_confidence=meta_calibrated,
            uncertainty_estimate=uncertainty_estimate,
            confidence_intervals=confidence_intervals,
            calibration_factors={
                'base_adjustment': base_calibrated - raw_score,
                'context_adjustment': context_adjustment - base_calibrated,
                'meta_adjustment': meta_calibrated - context_adjustment
            },
            calibration_time=time.time() - calibration_start
        )

    async def update_calibration(
        self,
        attribution_result: 'AgencyAttributionResult',
        ground_truth: 'GroundTruth'
    ):
        """Update calibration based on ground truth feedback."""
        # Record accuracy
        accuracy_record = AccuracyRecord(
            predicted_confidence=attribution_result.confidence,
            actual_outcome=ground_truth.actual_agency,
            context=attribution_result.context,
            timestamp=time.time()
        )

        self._accuracy_history.append(accuracy_record)

        # Update calibration data
        self._calibration_data.add_record(accuracy_record)

        # Retrain calibration model if needed
        if len(self._accuracy_history) % self.config.retrain_frequency == 0:
            await self._retrain_calibration_model()


# Data structures for agency attribution
@dataclass
class AgencyPrediction:
    """Prediction about the outcome of a self-initiated action."""
    prediction_id: str
    intention_id: str
    action_type: str
    predicted_outcome: 'PredictedOutcome'
    prediction_timestamp: float
    confidence: float
    model_id: str
    context_snapshot: Dict[str, Any]
    monitoring_criteria: List['MonitoringCriterion']
    creation_time: float


@dataclass
class PredictionMatchResult:
    """Result of matching an event to predictions."""
    prediction: Optional[AgencyPrediction]
    event: 'Event'
    match_score: float
    match_quality: 'MatchQuality'
    temporal_accuracy: float
    outcome_accuracy: float


@dataclass
class TemporalCorrelationResult:
    """Result of temporal correlation analysis."""
    event: 'Event'
    relevant_intentions: List['Intention']
    intention_correlations: List['IntentionCorrelation']
    best_correlation: Optional['IntentionCorrelation']
    pattern_analysis: 'PatternAnalysis'
    overall_correlation_score: float
    calculation_time: float


@dataclass
class IntentionCorrelation:
    """Correlation between an event and an intention."""
    intention: 'Intention'
    temporal_correlation: float
    causal_correlation: float
    contextual_correlation: float
    feature_correlation: float
    correlation_strength: float
    confidence: float


@dataclass
class CausalAnalysisResult:
    """Result of causal analysis."""
    event: 'Event'
    causal_model_used: str
    causal_inference: 'CausalInference'
    causal_strength: float
    alternative_explanations: List['AlternativeExplanation']
    causal_explanation: str
    confidence: float
    analysis_time: float


@dataclass
class CalibratedConfidence:
    """Calibrated confidence result."""
    raw_score: float
    calibrated_confidence: float
    uncertainty_estimate: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    calibration_factors: Dict[str, float]
    calibration_time: float


class AgencyAttributionOrchestrator:
    """
    Orchestrates the complete agency attribution process.

    Coordinates prediction matching, correlation analysis, causal inference,
    and confidence calibration to produce final agency attribution decisions.
    """

    def __init__(self, config: 'AttributionConfig'):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AttributionOrchestrator")

        # Component systems
        self.prediction_system = PredictionSystem(config.prediction_config)
        self.correlation_tracker = CorrelationTracker(config.correlation_config)
        self.causal_analyzer = CausalAnalyzer(config.causal_config)
        self.confidence_calibrator = ConfidenceCalibrator(config.confidence_config)

        # Integration and decision making
        self.evidence_integrator = EvidenceIntegrator()
        self.decision_engine = AttributionDecisionEngine()

        # Performance monitoring
        self.performance_monitor = AttributionPerformanceMonitor()

    async def initialize(self):
        """Initialize the agency attribution orchestrator."""
        self.logger.info("Initializing agency attribution orchestrator")

        # Initialize all component systems
        await asyncio.gather(
            self.prediction_system.initialize(),
            self.correlation_tracker.initialize(),
            self.causal_analyzer.initialize(),
            self.confidence_calibrator.initialize()
        )

        await self.evidence_integrator.initialize()
        await self.decision_engine.initialize()

        self.logger.info("Agency attribution orchestrator initialized")

    async def attribute_agency(
        self,
        event: 'Event',
        context: 'AgencyContext'
    ) -> 'AgencyAttributionResult':
        """Perform complete agency attribution for an event."""
        attribution_start = time.time()

        try:
            # Run all analysis components in parallel
            prediction_task = self.prediction_system.check_prediction_match(event)
            correlation_task = self.correlation_tracker.calculate_correlation(
                event, context.temporal_window
            )
            causal_task = self.causal_analyzer.analyze_causality(
                event, context.causal_context
            )

            # Wait for all analyses to complete
            prediction_result, correlation_result, causal_result = await asyncio.gather(
                prediction_task, correlation_task, causal_task
            )

            # Integrate evidence
            integrated_evidence = await self.evidence_integrator.integrate(
                prediction_result, correlation_result, causal_result
            )

            # Make attribution decision
            raw_attribution = await self.decision_engine.decide(
                integrated_evidence, context
            )

            # Calibrate confidence
            calibrated_confidence = await self.confidence_calibrator.calibrate(
                raw_attribution.score, context, event
            )

            # Create final result
            attribution_result = AgencyAttributionResult(
                event_id=event.id,
                attribution_timestamp=time.time(),
                agency_score=raw_attribution.score,
                calibrated_confidence=calibrated_confidence,
                prediction_result=prediction_result,
                correlation_result=correlation_result,
                causal_result=causal_result,
                integrated_evidence=integrated_evidence,
                attribution_decision=raw_attribution.decision,
                processing_time=time.time() - attribution_start
            )

            # Monitor performance
            await self.performance_monitor.record_attribution(attribution_result)

            return attribution_result

        except Exception as e:
            self.logger.error(f"Error in agency attribution: {e}")
            raise AgencyAttributionError(f"Attribution failed: {e}")
```

This agency attribution system provides sophisticated analysis of causal relationships, temporal correlations, and predictive matching to accurately determine whether observed events are self-generated or externally caused.