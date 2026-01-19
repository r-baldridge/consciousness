# Contemplative States Input/Output Interface Specification

## Overview
This document specifies the comprehensive input/output interface design for computational modeling of contemplative and meditative states. The interface bridges practitioner inputs, phenomenological data, neural correlates, and system outputs for state detection, modeling, and analysis.

## Input Interface Architecture

### Multi-Modal Input Processing Framework
```python
class ContemplativeInputInterface:
    def __init__(self):
        self.input_levels = {
            'phenomenological_input': PhenomenologicalInputLevel(
                first_person_reports=True,
                structured_questionnaires=True,
                experience_sampling=True,
                retrospective_narratives=True
            ),
            'neural_input': NeuralInputLevel(
                eeg_signals=True,
                fmri_data=True,
                autonomic_measures=True,
                physiological_markers=True
            ),
            'practice_context': PracticeContextLevel(
                tradition_specification=True,
                practice_type=True,
                session_parameters=True,
                practitioner_profile=True
            ),
            'temporal_context': TemporalContextLevel(
                session_timeline=True,
                state_transitions=True,
                practice_history=True
            )
        }

        self.input_modalities = {
            'verbal_reports': VerbalReportChannel(),
            'neural_signals': NeuralSignalChannel(),
            'physiological_signals': PhysiologicalChannel(),
            'contextual_metadata': MetadataChannel()
        }

    def process_contemplative_input(self, raw_input):
        """
        Process multi-modal contemplative input through interface levels
        """
        # Phenomenological processing
        phenomenological_processing = self.input_levels['phenomenological_input'].process(
            raw_input.phenomenological_data
        )

        # Neural processing
        neural_processing = self.input_levels['neural_input'].process(
            raw_input.neural_data
        )

        # Context processing
        context_processing = self.input_levels['practice_context'].process(
            raw_input.context_data
        )

        # Temporal processing
        temporal_processing = self.input_levels['temporal_context'].process(
            raw_input.temporal_data
        )

        # Cross-modal integration
        integrated_input = self.integrate_modalities(
            phenomenological_processing,
            neural_processing,
            context_processing,
            temporal_processing
        )

        return ContemplativeInputResult(
            phenomenological=phenomenological_processing,
            neural=neural_processing,
            context=context_processing,
            temporal=temporal_processing,
            integrated=integrated_input,
            state_readiness=self.assess_state_readiness(integrated_input)
        )


class PhenomenologicalInputLevel:
    def __init__(self):
        self.report_processors = {
            'free_narrative': FreeNarrativeProcessor(
                nlp_extraction=True,
                phenomenological_coding=True,
                quality_dimensions_extraction=True
            ),
            'structured_questionnaire': StructuredQuestionnaireProcessor(
                likert_processing=True,
                categorical_processing=True,
                composite_score_calculation=True
            ),
            'experience_sampling': ExperienceSamplingProcessor(
                momentary_assessment=True,
                time_series_construction=True,
                variability_analysis=True
            ),
            'micro_phenomenological': MicroPhenomenologicalProcessor(
                detailed_interview_coding=True,
                temporal_structure_extraction=True,
                quality_gradient_mapping=True
            )
        }

        self.phenomenological_dimensions = {
            'attention_qualities': AttentionQualityExtractor(
                focus_type=['narrow', 'broad', 'choiceless'],
                stability=['unstable', 'stable', 'absorbed'],
                effort_level=['effortful', 'effortless'],
                meta_awareness=['present', 'absent']
            ),
            'self_experience': SelfExperienceExtractor(
                self_referencing=['normal', 'reduced', 'absent'],
                witness_quality=['identified', 'observing', 'dissolved'],
                boundaries=['defined', 'permeable', 'absent']
            ),
            'temporal_experience': TemporalExperienceExtractor(
                time_sense=['normal', 'dilated', 'absent'],
                present_moment=['divided', 'unified', 'eternal']
            ),
            'affective_qualities': AffectiveQualityExtractor(
                valence=['negative', 'neutral', 'positive', 'beyond_valence'],
                specific_states=['bliss', 'peace', 'equanimity', 'fear'],
                intensity=['subtle', 'moderate', 'intense']
            ),
            'cognitive_qualities': CognitiveQualityExtractor(
                thought_activity=['active', 'reduced', 'absent'],
                clarity=['foggy', 'clear', 'luminous']
            )
        }

    def process(self, phenomenological_data):
        """
        Process phenomenological input data
        """
        # Extract from appropriate processor
        if phenomenological_data.type == 'free_narrative':
            extracted = self.report_processors['free_narrative'].extract(
                phenomenological_data.content
            )
        elif phenomenological_data.type == 'structured':
            extracted = self.report_processors['structured_questionnaire'].process(
                phenomenological_data.content
            )
        else:
            extracted = self.report_processors['experience_sampling'].process(
                phenomenological_data.content
            )

        # Map to phenomenological dimensions
        dimension_mapping = {}
        for dim_name, extractor in self.phenomenological_dimensions.items():
            dimension_mapping[dim_name] = extractor.extract(extracted)

        return PhenomenologicalProcessingResult(
            raw_extraction=extracted,
            dimension_mapping=dimension_mapping,
            quality_scores=self.calculate_quality_scores(dimension_mapping),
            state_indicators=self.identify_state_indicators(dimension_mapping)
        )
```

### Neural Input Interface
```python
class NeuralInputLevel:
    def __init__(self):
        self.signal_processors = {
            'eeg_processor': EEGProcessor(
                preprocessing=EEGPreprocessing(
                    filtering=True,
                    artifact_removal=True,
                    reference_correction=True
                ),
                feature_extraction=EEGFeatureExtraction(
                    power_spectral_density=True,
                    band_powers=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    coherence_measures=True,
                    asymmetry_indices=True
                ),
                contemplative_features=ContemplativeEEGFeatures(
                    frontal_midline_theta=True,
                    gamma_amplitude=True,
                    alpha_coherence=True,
                    dmn_markers=True
                )
            ),
            'fmri_processor': fMRIProcessor(
                preprocessing=fMRIPreprocessing(
                    motion_correction=True,
                    normalization=True,
                    smoothing=True
                ),
                feature_extraction=fMRIFeatureExtraction(
                    roi_activation=True,
                    connectivity_matrices=True,
                    network_measures=True
                ),
                contemplative_features=ContemplativefMRIFeatures(
                    dmn_activation=True,
                    acc_activation=True,
                    insula_activation=True,
                    parietal_changes=True
                )
            ),
            'physiological_processor': PhysiologicalProcessor(
                hrv_analysis=HRVAnalysis(
                    time_domain=True,
                    frequency_domain=True,
                    nonlinear_measures=True
                ),
                respiration_analysis=RespirationAnalysis(
                    rate=True,
                    depth=True,
                    variability=True
                ),
                skin_conductance=SkinConductanceAnalysis(
                    tonic_level=True,
                    phasic_responses=True
                )
            )
        }

    def process(self, neural_data):
        """
        Process neural input signals
        """
        processed_signals = {}

        # Process EEG if available
        if neural_data.has_eeg:
            processed_signals['eeg'] = self.signal_processors['eeg_processor'].process(
                neural_data.eeg_data
            )

        # Process fMRI if available
        if neural_data.has_fmri:
            processed_signals['fmri'] = self.signal_processors['fmri_processor'].process(
                neural_data.fmri_data
            )

        # Process physiological signals
        if neural_data.has_physiological:
            processed_signals['physiological'] = self.signal_processors[
                'physiological_processor'
            ].process(neural_data.physiological_data)

        # Extract contemplative-specific features
        contemplative_features = self.extract_contemplative_features(processed_signals)

        return NeuralProcessingResult(
            processed_signals=processed_signals,
            contemplative_features=contemplative_features,
            state_markers=self.identify_state_markers(contemplative_features),
            quality_metrics=self.calculate_signal_quality(processed_signals)
        )

    def extract_contemplative_features(self, processed_signals):
        """
        Extract features specific to contemplative state detection
        """
        features = {}

        if 'eeg' in processed_signals:
            features['gamma_power'] = processed_signals['eeg'].band_powers['gamma']
            features['theta_power'] = processed_signals['eeg'].band_powers['theta']
            features['alpha_coherence'] = processed_signals['eeg'].coherence['alpha']
            features['frontal_midline_theta'] = processed_signals['eeg'].fmtheta

        if 'fmri' in processed_signals:
            features['dmn_activation'] = processed_signals['fmri'].roi_activation['dmn']
            features['acc_activation'] = processed_signals['fmri'].roi_activation['acc']
            features['insula_activation'] = processed_signals['fmri'].roi_activation['insula']

        if 'physiological' in processed_signals:
            features['hrv_hf'] = processed_signals['physiological'].hrv['hf_power']
            features['respiration_rate'] = processed_signals['physiological'].respiration['rate']

        return ContemplativeFeatureSet(features=features)
```

## Output Interface Architecture

### State Classification Output
```python
class ContemplativeOutputInterface:
    def __init__(self):
        self.output_modules = {
            'state_classification': StateClassificationOutput(
                primary_state_classification=True,
                confidence_scoring=True,
                alternative_hypotheses=True,
                tradition_specific_mapping=True
            ),
            'phenomenological_report': PhenomenologicalReportOutput(
                quality_profile_generation=True,
                dimensional_scoring=True,
                narrative_synthesis=True
            ),
            'neural_summary': NeuralSummaryOutput(
                marker_summary=True,
                comparison_to_norms=True,
                trajectory_analysis=True
            ),
            'guidance_output': GuidanceOutput(
                practice_recommendations=True,
                progression_suggestions=True,
                safety_alerts=True
            )
        }

    def generate_output(self, processing_result):
        """
        Generate comprehensive output from processing result
        """
        # State classification
        state_classification = self.output_modules['state_classification'].classify(
            processing_result
        )

        # Phenomenological report
        phenomenological_report = self.output_modules['phenomenological_report'].generate(
            processing_result
        )

        # Neural summary
        neural_summary = self.output_modules['neural_summary'].summarize(
            processing_result
        )

        # Guidance output
        guidance = self.output_modules['guidance_output'].generate(
            processing_result,
            state_classification
        )

        return ContemplativeOutputResult(
            state_classification=state_classification,
            phenomenological_report=phenomenological_report,
            neural_summary=neural_summary,
            guidance=guidance,
            confidence_metrics=self.calculate_confidence(processing_result)
        )


class StateClassificationOutput:
    def __init__(self):
        self.state_classifiers = {
            'concentration_classifier': ConcentrationStateClassifier(
                states=['ordinary', 'access', 'jhana_1', 'jhana_2', 'jhana_3', 'jhana_4'],
                formless_states=['space', 'consciousness', 'nothingness', 'neither_perception']
            ),
            'insight_classifier': InsightStateClassifier(
                stages=['mind_body', 'cause_effect', 'arising_passing', 'dissolution',
                       'fear', 'misery', 'disgust', 'desire_deliverance', 're_observation',
                       'equanimity', 'path', 'fruition', 'review']
            ),
            'non_dual_classifier': NonDualStateClassifier(
                states=['witness', 'non_dual', 'rigpa', 'turiya']
            ),
            'cessation_classifier': CessationClassifier(
                states=['pre_cessation', 'cessation', 'post_cessation']
            ),
            'tradition_mapper': TraditionMapper(
                traditions=['theravada', 'zen', 'tibetan', 'hindu_yoga', 'advaita',
                           'christian', 'sufi', 'secular']
            )
        }

    def classify(self, processing_result):
        """
        Classify contemplative state from processing result
        """
        # Run all classifiers
        concentration_result = self.state_classifiers['concentration_classifier'].classify(
            processing_result
        )
        insight_result = self.state_classifiers['insight_classifier'].classify(
            processing_result
        )
        non_dual_result = self.state_classifiers['non_dual_classifier'].classify(
            processing_result
        )
        cessation_result = self.state_classifiers['cessation_classifier'].classify(
            processing_result
        )

        # Integrate classifications
        integrated_classification = self.integrate_classifications(
            concentration_result,
            insight_result,
            non_dual_result,
            cessation_result
        )

        # Map to tradition-specific terminology
        tradition_mapping = self.state_classifiers['tradition_mapper'].map(
            integrated_classification,
            processing_result.context.tradition
        )

        return StateClassificationResult(
            primary_state=integrated_classification.primary,
            confidence=integrated_classification.confidence,
            secondary_hypotheses=integrated_classification.alternatives,
            tradition_mapping=tradition_mapping,
            feature_contributions=self.explain_classification(
                processing_result, integrated_classification
            )
        )

    def integrate_classifications(self, concentration, insight, non_dual, cessation):
        """
        Integrate multiple classifier outputs into unified classification
        """
        # Priority handling
        if cessation.detected:
            return IntegratedClassification(
                primary='cessation',
                confidence=cessation.confidence,
                alternatives=[]
            )

        if non_dual.detected and non_dual.confidence > 0.7:
            return IntegratedClassification(
                primary=non_dual.state,
                confidence=non_dual.confidence,
                alternatives=[concentration.state, insight.state]
            )

        # Compare concentration vs insight
        if concentration.confidence > insight.confidence:
            return IntegratedClassification(
                primary=concentration.state,
                confidence=concentration.confidence,
                alternatives=[insight.state] if insight.confidence > 0.3 else []
            )
        else:
            return IntegratedClassification(
                primary=insight.state,
                confidence=insight.confidence,
                alternatives=[concentration.state] if concentration.confidence > 0.3 else []
            )
```

### Phenomenological Report Output
```python
class PhenomenologicalReportOutput:
    def __init__(self):
        self.report_generators = {
            'quality_profile': QualityProfileGenerator(
                dimensions=[
                    'attention', 'self_experience', 'temporal',
                    'spatial', 'affective', 'cognitive'
                ],
                scale='0_to_1',
                visualization=True
            ),
            'state_description': StateDescriptionGenerator(
                template_based=True,
                tradition_aware=True,
                phenomenological_accuracy=True
            ),
            'comparison_generator': ComparisonGenerator(
                historical_comparison=True,
                norm_comparison=True,
                progression_tracking=True
            )
        }

    def generate(self, processing_result):
        """
        Generate phenomenological report from processing result
        """
        # Generate quality profile
        quality_profile = self.report_generators['quality_profile'].generate(
            processing_result.phenomenological
        )

        # Generate state description
        state_description = self.report_generators['state_description'].generate(
            processing_result,
            quality_profile
        )

        # Generate comparisons
        comparisons = self.report_generators['comparison_generator'].generate(
            processing_result,
            quality_profile
        )

        return PhenomenologicalReport(
            quality_profile=quality_profile,
            state_description=state_description,
            comparisons=comparisons,
            summary=self.generate_summary(quality_profile, state_description)
        )


class GuidanceOutput:
    def __init__(self):
        self.guidance_generators = {
            'practice_recommender': PracticeRecommender(
                tradition_aware=True,
                progression_based=True,
                safety_conscious=True
            ),
            'progression_advisor': ProgressionAdvisor(
                stage_identification=True,
                next_steps=True,
                obstacles_identification=True
            ),
            'safety_monitor': SafetyMonitor(
                adverse_effect_detection=True,
                alert_generation=True,
                referral_triggers=True
            )
        }

    def generate(self, processing_result, state_classification):
        """
        Generate guidance based on current state
        """
        # Check safety first
        safety_check = self.guidance_generators['safety_monitor'].check(
            processing_result,
            state_classification
        )

        if safety_check.alert:
            return GuidanceResult(
                safety_alert=safety_check,
                recommendations=safety_check.recommended_actions,
                progression_advice=None
            )

        # Generate practice recommendations
        practice_recommendations = self.guidance_generators['practice_recommender'].recommend(
            processing_result,
            state_classification
        )

        # Generate progression advice
        progression_advice = self.guidance_generators['progression_advisor'].advise(
            processing_result,
            state_classification
        )

        return GuidanceResult(
            safety_alert=None,
            recommendations=practice_recommendations,
            progression_advice=progression_advice
        )
```

## Data Exchange Formats

### Standard Data Structures
```python
class ContemplativeDataStructures:
    def __init__(self):
        self.data_formats = {
            'session_record': SessionRecord(
                session_id='string',
                practitioner_id='string',
                timestamp_start='datetime',
                timestamp_end='datetime',
                tradition='ContemplativeTradition',
                practice_type='PracticeType',
                duration_minutes='int',
                states_recorded='List[StateRecord]',
                phenomenological_reports='List[PhenomenologicalReport]',
                neural_data_ids='List[string]',
                quality_metrics='QualityMetrics'
            ),
            'state_record': StateRecord(
                state_id='string',
                session_id='string',
                state_type='ContemplativeState',
                confidence='float',
                timestamp_start='datetime',
                timestamp_end='datetime',
                phenomenological_profile='PhenomenologicalProfile',
                neural_markers='NeuralMarkers'
            ),
            'phenomenological_profile': PhenomenologicalProfile(
                attention_qualities='AttentionProfile',
                self_experience='SelfExperienceProfile',
                temporal_experience='TemporalProfile',
                spatial_experience='SpatialProfile',
                affective_qualities='AffectiveProfile',
                cognitive_qualities='CognitiveProfile'
            ),
            'neural_markers': NeuralMarkers(
                eeg_markers='EEGMarkers',
                fmri_markers='fMRIMarkers',
                physiological_markers='PhysiologicalMarkers'
            )
        }

    def validate_data(self, data, data_type):
        """
        Validate data against schema
        """
        schema = self.data_formats[data_type]
        validation_result = schema.validate(data)
        return validation_result


class APIEndpoints:
    def __init__(self):
        self.endpoints = {
            'session': SessionEndpoints(
                create_session='/api/v1/sessions',
                get_session='/api/v1/sessions/{session_id}',
                update_session='/api/v1/sessions/{session_id}',
                list_sessions='/api/v1/sessions'
            ),
            'state': StateEndpoints(
                classify_state='/api/v1/states/classify',
                get_state='/api/v1/states/{state_id}',
                compare_states='/api/v1/states/compare'
            ),
            'practitioner': PractitionerEndpoints(
                create_practitioner='/api/v1/practitioners',
                get_practitioner='/api/v1/practitioners/{practitioner_id}',
                get_progress='/api/v1/practitioners/{practitioner_id}/progress'
            ),
            'analysis': AnalysisEndpoints(
                analyze_session='/api/v1/analysis/session/{session_id}',
                cross_session_analysis='/api/v1/analysis/cross-session',
                tradition_comparison='/api/v1/analysis/tradition-comparison'
            )
        }
```

## Error Handling and Validation

### Input Validation
```python
class InputValidation:
    def __init__(self):
        self.validators = {
            'phenomenological_validator': PhenomenologicalValidator(
                required_fields=['report_type', 'content'],
                content_validation=True,
                range_checking=True
            ),
            'neural_validator': NeuralValidator(
                signal_quality_threshold=0.7,
                artifact_detection=True,
                sampling_rate_validation=True
            ),
            'context_validator': ContextValidator(
                tradition_validation=True,
                practice_type_validation=True,
                temporal_consistency=True
            )
        }

    def validate_input(self, input_data):
        """
        Validate all input data
        """
        validation_results = {}

        for validator_name, validator in self.validators.items():
            result = validator.validate(input_data)
            validation_results[validator_name] = result

        overall_valid = all(r.is_valid for r in validation_results.values())

        return ValidationResult(
            is_valid=overall_valid,
            component_results=validation_results,
            errors=self.collect_errors(validation_results),
            warnings=self.collect_warnings(validation_results)
        )


class ErrorHandling:
    def __init__(self):
        self.error_handlers = {
            'input_error': InputErrorHandler(
                missing_data_handling='request_resubmission',
                invalid_data_handling='return_validation_error',
                partial_data_handling='process_available'
            ),
            'processing_error': ProcessingErrorHandler(
                classification_failure='return_uncertain',
                neural_processing_failure='fallback_to_phenomenological',
                integration_failure='return_partial_results'
            ),
            'system_error': SystemErrorHandler(
                database_error='retry_with_backoff',
                api_error='graceful_degradation',
                timeout_error='return_partial'
            )
        }

    def handle_error(self, error, context):
        """
        Handle errors appropriately based on type and context
        """
        error_type = self.classify_error(error)
        handler = self.error_handlers.get(error_type)

        if handler:
            return handler.handle(error, context)
        else:
            return self.default_error_handling(error, context)
```

## References

### Interface Design Standards
1. OpenAPI Specification 3.0
2. JSON Schema Draft 7
3. REST API Design Guidelines

### Data Standards
1. BIDS (Brain Imaging Data Structure)
2. EDF+ (European Data Format for EEG)
3. NIfTI (Neuroimaging Informatics Technology Initiative)

---

*Document compiled for Form 36: Contemplative & Meditative States*
*Last updated: 2026-01-18*
