# Form 24: Locked-in Syndrome Consciousness - Behavioral Indicators

## Observable Behaviors Indicating Preserved Consciousness in Locked-in Syndrome

Behavioral indicators for locked-in syndrome consciousness present unique challenges due to the severe motor impairment that characterizes this condition. Unlike other consciousness assessments that rely on observable motor responses, locked-in syndrome requires detection of subtle, preserved behaviors and alternative communication methods to identify consciousness.

### Core Behavioral Signatures

#### 1. Preserved Voluntary Eye Movements

**Indicator**: Systematic, purposeful eye movements that respond to commands or stimuli.

**Observable Behaviors**:
- Vertical eye movements following commands
- Horizontal eye movements (if preserved in incomplete LIS)
- Systematic eye blinking patterns
- Eye tracking of moving objects or people
- Appropriate eye movement responses to questions

**Measurement Criteria**:
```python
@dataclass
class EyeMovementIndicators:
    voluntary_vertical_movement: float  # 0.0-1.0 control level
    command_following_accuracy: float  # Percentage of correct responses
    spontaneous_tracking: float  # Natural tracking behavior score
    blink_pattern_communication: float  # Use of blinks for communication
    contextual_eye_responses: float  # Appropriate responses to context

    def assess_eye_movement_consciousness(self) -> float:
        """Assess consciousness based on eye movement indicators."""
        if self.voluntary_vertical_movement < 0.3:
            return 0.2  # Minimal voluntary control detected

        # Weight different aspects of eye movement control
        command_weight = 0.35
        tracking_weight = 0.25
        communication_weight = 0.25
        contextual_weight = 0.15

        consciousness_score = (
            self.command_following_accuracy * command_weight +
            self.spontaneous_tracking * tracking_weight +
            self.blink_pattern_communication * communication_weight +
            self.contextual_eye_responses * contextual_weight
        )

        return min(1.0, consciousness_score)
```

**Detection Methods**:
- Structured eye movement command protocols
- Gaze tracking during natural interactions
- Systematic blinking pattern analysis
- Response to visual stimuli assessments

#### 2. Communication Attempts Through Alternative Modalities

**Indicator**: Deliberate attempts to communicate using available motor capabilities or brain-computer interfaces.

**Observable Behaviors**:
- Consistent yes/no responses using eye movements
- Spelling attempts through eye-tracking systems
- Use of brain-computer interface paradigms
- Appropriate emotional expressions through facial muscles (if preserved)
- Systematic use of minimal residual movements

**Measurement Criteria**:
```python
@dataclass
class CommunicationAttemptIndicators:
    yes_no_consistency: float  # Consistency in yes/no responses
    spelling_accuracy: float  # Accuracy in spelling attempts
    bci_engagement: float  # Engagement with BCI systems
    emotional_expression: float  # Appropriate emotional responses
    message_coherence: float  # Coherence of communicated messages

    def calculate_communication_authenticity(self) -> float:
        """Calculate authenticity of communication attempts."""
        # Check for minimum communication capability
        if self.yes_no_consistency < 0.6:
            return 0.3  # Insufficient consistency for reliable communication

        # Advanced communication capabilities
        advanced_score = (
            self.spelling_accuracy * 0.3 +
            self.bci_engagement * 0.25 +
            self.message_coherence * 0.25 +
            self.emotional_expression * 0.2
        )

        # Combine basic and advanced communication
        basic_score = self.yes_no_consistency * 0.6
        return basic_score + (advanced_score * 0.4)
```

#### 3. Cognitive Task Performance Through Alternative Responses

**Indicator**: Performance on cognitive tasks adapted for limited motor output.

**Observable Behaviors**:
- Problem-solving using eye movements or BCI
- Memory recall demonstrated through communication systems
- Attention and concentration during tasks
- Learning and adaptation to new communication methods
- Abstract reasoning demonstrated through adapted tests

**Assessment Framework**:
```python
class CognitiveTaskAssessment:
    def __init__(self):
        self.working_memory_tasks = WorkingMemoryTasks()
        self.attention_tasks = AttentionTasks()
        self.executive_function_tasks = ExecutiveFunctionTasks()
        self.learning_tasks = LearningTasks()

    async def assess_cognitive_performance(self, patient_id: str,
                                         communication_system: CommunicationSystem) -> CognitivePerformanceResult:
        """Assess cognitive performance through alternative response modalities."""

        # Working memory assessment
        wm_result = await self.working_memory_tasks.assess_via_communication(
            patient_id, communication_system
        )

        # Attention assessment
        attention_result = await self.attention_tasks.assess_via_eyetracking(
            patient_id, communication_system
        )

        # Executive function assessment
        exec_result = await self.executive_function_tasks.assess_via_bci(
            patient_id, communication_system
        )

        # Learning assessment
        learning_result = await self.learning_tasks.assess_adaptation(
            patient_id, communication_system
        )

        return CognitivePerformanceResult(
            working_memory_score=wm_result.performance_score,
            attention_score=attention_result.performance_score,
            executive_function_score=exec_result.performance_score,
            learning_score=learning_result.performance_score,
            overall_cognitive_preservation=self.calculate_overall_preservation([
                wm_result, attention_result, exec_result, learning_result
            ])
        )

    async def assess_working_memory_via_nback_task(self, patient_id: str,
                                                 communication_system: CommunicationSystem) -> TaskResult:
        """Assess working memory using adapted n-back task."""
        # Present sequence of stimuli, require identification of repeats
        stimulus_sequence = self.generate_nback_sequence(n=2, length=20)
        responses = []

        for stimulus in stimulus_sequence:
            # Present stimulus
            await communication_system.present_stimulus(stimulus)

            # Collect response via alternative communication
            response = await communication_system.collect_yes_no_response(
                question="Was this stimulus presented 2 items ago?",
                timeout_seconds=10
            )

            responses.append(response)

        # Calculate accuracy
        correct_responses = sum(1 for i, response in enumerate(responses)
                              if response.answer == self.get_correct_answer(i, stimulus_sequence))

        accuracy = correct_responses / len(responses)

        return TaskResult(
            task_name="n_back_working_memory",
            performance_score=accuracy,
            response_times=response.response_times,
            confidence_scores=response.confidence_scores
        )
```

#### 4. Emotional and Social Responsiveness

**Indicator**: Appropriate emotional responses and social engagement despite motor limitations.

**Observable Behaviors**:
- Emotional reactions to personal topics or family interactions
- Appropriate responses to humor or emotional content
- Social engagement through available communication methods
- Recognition of familiar people and places
- Emotional expression through preserved facial muscles or eye expressions

**Measurement Framework**:
```python
@dataclass
class EmotionalSocialIndicators:
    emotional_appropriateness: float  # Appropriateness of emotional responses
    social_engagement_level: float  # Level of social interaction
    recognition_responses: float  # Recognition of familiar people/places
    humor_appreciation: float  # Responses to humorous content
    empathic_responses: float  # Responses to others' emotions

    def assess_emotional_social_consciousness(self) -> float:
        """Assess emotional and social consciousness indicators."""
        # Emotional appropriateness is critical
        if self.emotional_appropriateness < 0.5:
            return 0.3  # Inappropriate responses suggest altered consciousness

        # Calculate weighted score
        weights = {
            'emotional_appropriateness': 0.30,
            'social_engagement_level': 0.25,
            'recognition_responses': 0.20,
            'humor_appreciation': 0.15,
            'empathic_responses': 0.10
        }

        scores = [
            self.emotional_appropriateness,
            self.social_engagement_level,
            self.recognition_responses,
            self.humor_appreciation,
            self.empathic_responses
        ]

        return sum(score * weight for score, weight in zip(scores, weights.values()))

class EmotionalResponseAssessment:
    async def assess_emotional_responses(self, patient_id: str,
                                       communication_system: CommunicationSystem) -> EmotionalAssessmentResult:
        """Assess emotional responsiveness through various stimuli."""

        # Personal emotional stimuli
        personal_responses = await self.assess_personal_emotional_responses(
            patient_id, communication_system
        )

        # Social emotional stimuli
        social_responses = await self.assess_social_emotional_responses(
            patient_id, communication_system
        )

        # Humor appreciation
        humor_responses = await self.assess_humor_appreciation(
            patient_id, communication_system
        )

        return EmotionalAssessmentResult(
            personal_emotional_score=personal_responses.appropriateness_score,
            social_emotional_score=social_responses.appropriateness_score,
            humor_appreciation_score=humor_responses.appropriateness_score,
            overall_emotional_preservation=self.calculate_emotional_preservation([
                personal_responses, social_responses, humor_responses
            ])
        )
```

#### 5. Learning and Adaptation Behaviors

**Indicator**: Ability to learn new communication methods and adapt to system changes.

**Observable Behaviors**:
- Improvement in BCI performance over training sessions
- Adaptation to new eye-tracking interfaces
- Learning of new communication vocabulary or symbols
- Adjustment to changes in communication system parameters
- Development of personal communication strategies

**Learning Assessment Protocol**:
```python
class LearningAdaptationAssessment:
    def __init__(self):
        self.learning_curve_analyzer = LearningCurveAnalyzer()
        self.adaptation_tracker = AdaptationTracker()
        self.strategy_identifier = CommunicationStrategyIdentifier()

    async def assess_learning_adaptation(self, patient_id: str,
                                       training_sessions: List[TrainingSession]) -> LearningAssessmentResult:
        """Assess learning and adaptation capabilities."""

        # Analyze learning curves across different modalities
        bci_learning = await self.learning_curve_analyzer.analyze_bci_learning(
            patient_id, training_sessions
        )

        eyetracking_learning = await self.learning_curve_analyzer.analyze_eyetracking_learning(
            patient_id, training_sessions
        )

        # Assess adaptation to system changes
        adaptation_results = await self.adaptation_tracker.assess_adaptations(
            patient_id, training_sessions
        )

        # Identify personal communication strategies
        strategy_development = await self.strategy_identifier.identify_strategies(
            patient_id, training_sessions
        )

        return LearningAssessmentResult(
            bci_learning_rate=bci_learning.learning_rate,
            eyetracking_learning_rate=eyetracking_learning.learning_rate,
            adaptation_flexibility=adaptation_results.flexibility_score,
            strategy_development=strategy_development.development_score,
            overall_learning_capacity=self.calculate_learning_capacity([
                bci_learning, eyetracking_learning, adaptation_results, strategy_development
            ])
        )

    async def analyze_bci_learning_progression(self, patient_id: str,
                                             bci_sessions: List[BCISession]) -> BCILearningResult:
        """Analyze BCI learning progression over time."""
        # Extract performance metrics over time
        performance_timeline = []

        for session in sorted(bci_sessions, key=lambda x: x.timestamp):
            performance_timeline.append({
                'timestamp': session.timestamp,
                'accuracy': session.accuracy,
                'speed': session.communication_speed,
                'session_number': session.session_number
            })

        # Calculate learning rate
        learning_rate = self.calculate_learning_rate(performance_timeline)

        # Identify learning plateaus
        plateaus = self.identify_learning_plateaus(performance_timeline)

        # Assess learning consistency
        consistency = self.assess_learning_consistency(performance_timeline)

        return BCILearningResult(
            learning_rate=learning_rate,
            plateau_points=plateaus,
            learning_consistency=consistency,
            final_performance_level=performance_timeline[-1]['accuracy'],
            total_improvement=performance_timeline[-1]['accuracy'] - performance_timeline[0]['accuracy']
        )
```

### Advanced Behavioral Indicators

#### 6. Metacognitive Awareness

**Indicator**: Awareness of own cognitive processes and communication abilities.

**Observable Behaviors**:
- Accurate self-assessment of communication accuracy
- Recognition of own fatigue levels
- Awareness of optimal communication times
- Understanding of own learning progress
- Ability to report on subjective experiences

**Assessment Methods**:
```python
class MetacognitiveAwarenessAssessment:
    async def assess_metacognitive_awareness(self, patient_id: str,
                                           communication_system: CommunicationSystem) -> MetacognitiveResult:
        """Assess metacognitive awareness through self-reporting."""

        # Self-assessment of communication accuracy
        accuracy_awareness = await self.assess_accuracy_awareness(
            patient_id, communication_system
        )

        # Fatigue self-monitoring
        fatigue_awareness = await self.assess_fatigue_awareness(
            patient_id, communication_system
        )

        # Learning progress awareness
        progress_awareness = await self.assess_progress_awareness(
            patient_id, communication_system
        )

        return MetacognitiveResult(
            accuracy_self_assessment=accuracy_awareness.correlation_with_actual,
            fatigue_awareness=fatigue_awareness.self_monitoring_accuracy,
            progress_awareness=progress_awareness.insight_level,
            overall_metacognitive_score=self.calculate_metacognitive_score([
                accuracy_awareness, fatigue_awareness, progress_awareness
            ])
        )

    async def assess_communication_confidence_calibration(self, patient_id: str,
                                                        communication_system: CommunicationSystem) -> ConfidenceCalibrationResult:
        """Assess how well patient's confidence correlates with actual performance."""

        communication_trials = []

        # Conduct series of communication trials with confidence ratings
        for trial_number in range(20):
            # Present communication task
            task = self.generate_communication_task()

            # Collect response
            response = await communication_system.collect_response(task)

            # Collect confidence rating
            confidence = await communication_system.collect_confidence_rating(
                question="How confident are you in your response?",
                scale="1-7"
            )

            # Assess accuracy
            accuracy = self.assess_response_accuracy(response, task.correct_answer)

            communication_trials.append({
                'trial_number': trial_number,
                'confidence': confidence.rating,
                'accuracy': accuracy,
                'response_time': response.response_time
            })

        # Calculate confidence calibration
        calibration_score = self.calculate_confidence_calibration(communication_trials)

        return ConfidenceCalibrationResult(
            calibration_score=calibration_score,
            overconfidence_bias=self.calculate_overconfidence(communication_trials),
            underconfidence_bias=self.calculate_underconfidence(communication_trials),
            trials_data=communication_trials
        )
```

#### 7. Temporal Consciousness Integration

**Indicator**: Awareness of time passage and integration of experiences across time.

**Observable Behaviors**:
- Reference to past events and experiences
- Planning and goal-setting for future activities
- Understanding of temporal sequences
- Recognition of personal history continuity
- Appropriate temporal context in communications

**Temporal Assessment Framework**:
```python
class TemporalConsciousnessAssessment:
    async def assess_temporal_integration(self, patient_id: str,
                                        communication_system: CommunicationSystem) -> TemporalIntegrationResult:
        """Assess temporal consciousness integration."""

        # Past integration assessment
        past_integration = await self.assess_past_integration(
            patient_id, communication_system
        )

        # Future planning assessment
        future_planning = await self.assess_future_planning(
            patient_id, communication_system
        )

        # Temporal sequencing assessment
        temporal_sequencing = await self.assess_temporal_sequencing(
            patient_id, communication_system
        )

        # Personal history continuity
        history_continuity = await self.assess_history_continuity(
            patient_id, communication_system
        )

        return TemporalIntegrationResult(
            past_integration_score=past_integration.integration_score,
            future_planning_score=future_planning.planning_score,
            temporal_sequencing_score=temporal_sequencing.sequencing_score,
            history_continuity_score=history_continuity.continuity_score,
            overall_temporal_consciousness=self.calculate_temporal_consciousness([
                past_integration, future_planning, temporal_sequencing, history_continuity
            ])
        )
```

#### 8. Consistency Across Assessment Sessions

**Indicator**: Consistent behavioral indicators across multiple assessment sessions and contexts.

**Observable Behaviors**:
- Stable communication patterns across sessions
- Consistent personality expression
- Reliable cognitive performance patterns
- Maintained learning progress
- Consistent emotional responsiveness

**Consistency Analysis**:
```python
class ConsistencyAnalysis:
    def __init__(self):
        self.session_comparator = SessionComparator()
        self.pattern_analyzer = PatternAnalyzer()
        self.stability_assessor = StabilityAssessor()

    async def analyze_cross_session_consistency(self, patient_id: str,
                                              assessment_sessions: List[AssessmentSession]) -> ConsistencyResult:
        """Analyze consistency across multiple assessment sessions."""

        # Communication pattern consistency
        communication_consistency = await self.session_comparator.compare_communication_patterns(
            assessment_sessions
        )

        # Cognitive performance consistency
        cognitive_consistency = await self.session_comparator.compare_cognitive_performance(
            assessment_sessions
        )

        # Emotional response consistency
        emotional_consistency = await self.session_comparator.compare_emotional_responses(
            assessment_sessions
        )

        # Personality expression consistency
        personality_consistency = await self.session_comparator.compare_personality_expression(
            assessment_sessions
        )

        return ConsistencyResult(
            communication_consistency=communication_consistency.consistency_score,
            cognitive_consistency=cognitive_consistency.consistency_score,
            emotional_consistency=emotional_consistency.consistency_score,
            personality_consistency=personality_consistency.consistency_score,
            overall_consistency=self.calculate_overall_consistency([
                communication_consistency, cognitive_consistency,
                emotional_consistency, personality_consistency
            ]),
            consistency_confidence=self.calculate_consistency_confidence(assessment_sessions)
        )
```

## Comprehensive Behavioral Assessment Protocol

### Integrated Assessment Framework

```python
class LISBehavioralAssessmentProtocol:
    """Comprehensive behavioral assessment protocol for locked-in syndrome consciousness."""

    def __init__(self):
        self.eye_movement_assessor = EyeMovementAssessor()
        self.communication_assessor = CommunicationAssessor()
        self.cognitive_assessor = CognitiveAssessor()
        self.emotional_assessor = EmotionalAssessor()
        self.learning_assessor = LearningAssessor()
        self.metacognitive_assessor = MetacognitiveAssessor()
        self.temporal_assessor = TemporalAssessor()
        self.consistency_analyzer = ConsistencyAnalyzer()

    async def conduct_comprehensive_assessment(self, patient_id: str,
                                             communication_system: CommunicationSystem,
                                             assessment_config: AssessmentConfig) -> ComprehensiveBehavioralResult:
        """Conduct comprehensive behavioral assessment for consciousness detection."""

        assessment_results = {}

        # Core behavioral assessments
        eye_movement_result = await self.eye_movement_assessor.assess(
            patient_id, communication_system
        )
        assessment_results['eye_movements'] = eye_movement_result

        communication_result = await self.communication_assessor.assess(
            patient_id, communication_system
        )
        assessment_results['communication'] = communication_result

        cognitive_result = await self.cognitive_assessor.assess(
            patient_id, communication_system
        )
        assessment_results['cognitive'] = cognitive_result

        emotional_result = await self.emotional_assessor.assess(
            patient_id, communication_system
        )
        assessment_results['emotional'] = emotional_result

        # Advanced assessments
        if assessment_config.include_advanced_assessments:
            learning_result = await self.learning_assessor.assess(
                patient_id, communication_system
            )
            assessment_results['learning'] = learning_result

            metacognitive_result = await self.metacognitive_assessor.assess(
                patient_id, communication_system
            )
            assessment_results['metacognitive'] = metacognitive_result

            temporal_result = await self.temporal_assessor.assess(
                patient_id, communication_system
            )
            assessment_results['temporal'] = temporal_result

        # Calculate overall consciousness likelihood
        consciousness_likelihood = await self.calculate_consciousness_likelihood(
            assessment_results
        )

        # Generate assessment summary
        assessment_summary = await self.generate_assessment_summary(
            assessment_results, consciousness_likelihood
        )

        return ComprehensiveBehavioralResult(
            patient_id=patient_id,
            assessment_timestamp=time.time(),
            individual_assessments=assessment_results,
            consciousness_likelihood=consciousness_likelihood,
            assessment_summary=assessment_summary,
            recommendations=self.generate_recommendations(assessment_results),
            confidence_level=self.calculate_assessment_confidence(assessment_results)
        )

    async def calculate_consciousness_likelihood(self, assessment_results: Dict[str, Any]) -> float:
        """Calculate overall consciousness likelihood from behavioral indicators."""

        # Define weights for different assessment domains
        domain_weights = {
            'eye_movements': 0.25,
            'communication': 0.30,
            'cognitive': 0.20,
            'emotional': 0.15,
            'learning': 0.05,
            'metacognitive': 0.03,
            'temporal': 0.02
        }

        weighted_score = 0.0
        total_weight = 0.0

        for domain, weight in domain_weights.items():
            if domain in assessment_results:
                domain_score = assessment_results[domain].overall_score
                weighted_score += weight * domain_score
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def generate_recommendations(self, assessment_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []

        # Analyze each domain for specific recommendations
        for domain, result in assessment_results.items():
            domain_recommendations = self.generate_domain_recommendations(domain, result)
            recommendations.extend(domain_recommendations)

        return recommendations
```

This comprehensive behavioral indicators framework provides detailed methods for detecting and validating consciousness in locked-in syndrome patients through observable behaviors adapted to their unique capabilities and limitations.