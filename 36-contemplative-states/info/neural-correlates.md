# Neural Correlates of Contemplative States

## Overview
This document details the neural correlates of meditation, mindfulness, and contemplative states, synthesizing findings from neuroimaging, electrophysiology, and structural brain research. Understanding these neural signatures is essential for developing computational models that can simulate or detect contemplative states.

## Electrophysiological Correlates

### Gamma Wave Activity in Meditation
```python
class GammaWaveCorrelates:
    def __init__(self):
        self.gamma_findings = {
            'lutz_2004_landmark': LandmarkGammaStudy(
                researchers=['Lutz', 'Greischar', 'Rawlings', 'Ricard', 'Davidson'],
                sample_size=8,
                practitioner_type='Tibetan Buddhist monks',
                practice_hours='10,000-50,000 hours',
                meditation_type='Non-referential compassion meditation',
                key_findings=GammaFindings(
                    frequency_range='25-42 Hz',
                    amplitude='Highest non-pathological levels recorded',
                    location='Lateral frontoparietal electrodes',
                    synchrony='Long-distance phase synchrony',
                    baseline_differences='Elevated gamma at rest in experts',
                    practice_correlation='Hours predict baseline gamma'
                )
            ),
            'cross_tradition_gamma': CrossTraditionGamma(
                traditions=['Theravada', 'Tibetan', 'Zen'],
                frequency_range='60-110 Hz (high gamma)',
                location='Parieto-occipital',
                consistency='Present across meditation types',
                correlation='Positive with experience level'
            ),
            'gamma_during_jhana': JhanaGamma(
                state_specificity=True,
                patterns_by_jhana={
                    'first_jhana': 'Moderate gamma increase',
                    'second_jhana': 'Sustained gamma',
                    'third_jhana': 'Gamma with reduced arousal',
                    'fourth_jhana': 'Variable patterns'
                }
            )
        }

    def analyze_gamma_significance(self):
        """
        Analyze the significance of gamma activity in meditation
        """
        significance = {}

        significance['cognitive_binding'] = {
            'theory': 'Gamma binds distributed neural populations',
            'meditation_relevance': 'Unified conscious experience',
            'evidence': 'Increased binding during absorption'
        }

        significance['conscious_access'] = {
            'theory': 'Gamma relates to conscious awareness',
            'meditation_relevance': 'Enhanced metacognitive awareness',
            'evidence': 'Gamma correlates with reported clarity'
        }

        significance['neuroplasticity_marker'] = {
            'theory': 'Baseline gamma reflects trait changes',
            'meditation_relevance': 'Long-term practice effects',
            'evidence': 'Practice hours predict resting gamma'
        }

        return GammaSignificanceAnalysis(
            cognitive_role=significance['cognitive_binding'],
            consciousness_role=significance['conscious_access'],
            plasticity_role=significance['neuroplasticity_marker']
        )


class AlphaThetaCorrelates:
    def __init__(self):
        self.alpha_findings = {
            'frontal_alpha': FrontalAlphaFindings(
                frequency='8-12 Hz',
                typical_change='Increased power and coherence',
                meditation_types=['Focused attention', 'TM'],
                interpretation='Relaxed alertness, reduced external processing',
                reliability='Moderate across studies'
            ),
            'posterior_alpha': PosteriorAlphaFindings(
                frequency='8-12 Hz',
                typical_change='Variable (increase or decrease)',
                meditation_types=['Open monitoring', 'Vipassana'],
                interpretation='Depends on attention type',
                reliability='Lower, context dependent'
            ),
            'alpha_asymmetry': AlphaAsymmetryFindings(
                finding='Reduced left-right asymmetry',
                meditation_types=['Loving-kindness', 'Compassion'],
                interpretation='Balanced emotional processing',
                reliability='Moderate'
            )
        }

        self.theta_findings = {
            'frontal_midline_theta': FrontalMidlineTheta(
                frequency='4-8 Hz',
                location='Frontal midline (Fz, FCz)',
                typical_change='Increased during meditation',
                meditation_types=['Most types'],
                interpretation='Internalized attention, working memory',
                reliability='High across studies'
            ),
            'theta_coherence': ThetaCoherence(
                finding='Increased inter-hemispheric theta coherence',
                meditation_types=['TM', 'Concentrative meditation'],
                interpretation='Enhanced integration',
                reliability='Moderate'
            )
        }

    def map_frequency_to_state(self):
        """
        Map EEG frequency patterns to contemplative states
        """
        frequency_state_map = {}

        frequency_state_map['access_concentration'] = {
            'alpha': 'Increased posterior alpha',
            'theta': 'Emerging frontal theta',
            'gamma': 'Normal to slightly elevated',
            'interpretation': 'Calming of mind, preparatory state'
        }

        frequency_state_map['jhana_states'] = {
            'alpha': 'High alpha coherence',
            'theta': 'Elevated frontal midline theta',
            'gamma': 'Increased, state-specific patterns',
            'interpretation': 'Deep absorption, reduced external processing'
        }

        frequency_state_map['insight_states'] = {
            'alpha': 'Variable, often suppressed',
            'theta': 'Elevated during insight moments',
            'gamma': 'Bursts during insights',
            'interpretation': 'Active investigation, clarity'
        }

        frequency_state_map['non_dual_states'] = {
            'alpha': 'High coherence',
            'theta': 'Elevated',
            'gamma': 'Sustained high gamma',
            'interpretation': 'Unified consciousness, witness awareness'
        }

        return FrequencyStateMapping(
            state_map=frequency_state_map,
            detection_algorithms=self.design_detection_algorithms(),
            validation_criteria=self.establish_validation_criteria()
        )
```

## Functional Neuroimaging Correlates

### Default Mode Network Changes
```python
class DefaultModeNetworkCorrelates:
    def __init__(self):
        self.dmn_regions = DMNRegions(
            medial_prefrontal_cortex=mPFC(
                function='Self-referential processing',
                meditation_change='Reduced activation'
            ),
            posterior_cingulate_cortex=PCC(
                function='Self-awareness, episodic memory',
                meditation_change='Reduced activation'
            ),
            lateral_temporal_cortex=LTC(
                function='Semantic processing',
                meditation_change='Variable'
            ),
            angular_gyrus=AG(
                function='Episodic memory, attention',
                meditation_change='Reduced activation'
            )
        )

        self.dmn_findings = {
            'brewer_2011': Brewer2011Study(
                sample='12 experienced meditators (mean 9,676 hours)',
                methodology='fMRI during meditation',
                key_findings=DMNFindings(
                    activation='Decreased DMN across meditation types',
                    connectivity='Stronger PCC-dACC-dlPFC coupling',
                    mind_wandering='Reduced correlates with DMN decrease',
                    generalization='Consistent across FA, OM, LKM'
                ),
                trait_effects=TraitEffects(
                    baseline_changes=True,
                    experience_correlation=True,
                    permanent_alteration=True
                )
            ),
            'hasenkamp_2012': Hasenkamp2012Study(
                methodology='Real-time fMRI with experience sampling',
                finding='DMN active during mind-wandering, suppressed during focus',
                insight='Meditation involves repeated DMN regulation'
            )
        }

    def analyze_dmn_meditation_relationship(self):
        """
        Analyze the relationship between DMN and meditation
        """
        analysis = {}

        analysis['state_effects'] = {
            'during_meditation': 'DMN suppression during focused practice',
            'mind_wandering_correlation': 'DMN activity predicts mind-wandering',
            'self_referential_reduction': 'Reduced self-focused processing'
        }

        analysis['trait_effects'] = {
            'baseline_changes': 'Altered DMN dynamics at rest',
            'connectivity_changes': 'Enhanced regulatory connections',
            'microstate_duration': 'Shorter DMN microstates in meditators'
        }

        analysis['theoretical_significance'] = {
            'self_model': 'DMN as neural substrate of narrative self',
            'meditation_target': 'Meditation reduces automatic self-modeling',
            'consciousness_implications': 'Self-consciousness is constructed, not given'
        }

        return DMNAnalysis(
            state_effects=analysis['state_effects'],
            trait_effects=analysis['trait_effects'],
            theory=analysis['theoretical_significance']
        )


class AttentionNetworkCorrelates:
    def __init__(self):
        self.attention_networks = {
            'dorsal_attention_network': DorsalAttentionNetwork(
                regions=['Frontal eye fields', 'Intraparietal sulcus'],
                function='Top-down, voluntary attention',
                meditation_change='Enhanced during focused attention',
                practice_types=['Samatha', 'Focused attention meditation']
            ),
            'ventral_attention_network': VentralAttentionNetwork(
                regions=['Temporoparietal junction', 'Ventral frontal cortex'],
                function='Bottom-up, stimulus-driven attention',
                meditation_change='Variable',
                practice_types=['Open monitoring may modulate']
            ),
            'salience_network': SalienceNetwork(
                regions=['Anterior insula', 'Dorsal ACC'],
                function='Detecting relevant stimuli, switching networks',
                meditation_change='Enhanced connectivity and function',
                practice_types=['Most meditation types']
            )
        }

        self.acc_findings = ACCFindings(
            subregions={
                'dorsal_acc': DorsalACC(
                    function='Cognitive control, conflict monitoring',
                    meditation_change='Enhanced activation'
                ),
                'rostral_acc': RostralACC(
                    function='Emotional regulation',
                    meditation_change='Enhanced activation'
                ),
                'subgenual_acc': SubgenualACC(
                    function='Affective processing',
                    meditation_change='Increased blood flow'
                )
            },
            tang_2015_study=Tang2015Study(
                finding='5 days IBMT increases ACC blood flow',
                mechanism='Enhanced self-regulation capacity'
            )
        )

    def map_attention_meditation_types(self):
        """
        Map attention network changes to meditation types
        """
        attention_map = {}

        attention_map['focused_attention'] = {
            'primary_network': 'Dorsal attention network',
            'changes': 'Enhanced sustained attention',
            'acc_involvement': 'Conflict monitoring during distraction',
            'neural_efficiency': 'Less effort for sustained focus over time'
        }

        attention_map['open_monitoring'] = {
            'primary_network': 'Salience network',
            'changes': 'Enhanced meta-awareness',
            'acc_involvement': 'Detecting arising phenomena',
            'neural_pattern': 'Broad, receptive attention'
        }

        attention_map['loving_kindness'] = {
            'primary_network': 'Limbic-salience integration',
            'changes': 'Enhanced emotional awareness',
            'acc_involvement': 'Emotional regulation',
            'neural_pattern': 'Affective-cognitive integration'
        }

        return AttentionMeditationMapping(
            practice_map=attention_map,
            network_interactions=self.analyze_network_interactions(),
            development_trajectory=self.model_attention_development()
        )
```

## Structural Brain Changes

### Gray Matter Changes
```python
class GrayMatterCorrelates:
    def __init__(self):
        self.structural_findings = {
            'lazar_2005': Lazar2005Study(
                sample='20 experienced meditators vs 15 controls',
                methodology='MRI cortical thickness',
                key_findings=ThicknessFindings(
                    regions_increased=[
                        'Right anterior insula',
                        'Right prefrontal cortex'
                    ],
                    experience_correlation=True,
                    age_interaction='Offset of age-related thinning'
                )
            ),
            'luders_2009': Luders2009Study(
                finding='Increased gray matter in multiple regions',
                regions=['Hippocampus', 'Frontal cortex', 'Right insula'],
                methodology='VBM morphometry'
            ),
            'holzel_2011': Holzel2011Study(
                design='Longitudinal, 8-week MBSR',
                findings=MBSRFindings(
                    increased_gm=['Left hippocampus', 'PCC', 'TPJ', 'Cerebellum'],
                    decreased_gm=['Right amygdala'],
                    correlation='Stress reduction correlated with amygdala decrease'
                )
            )
        }

        self.region_specific_changes = RegionSpecificChanges(
            prefrontal_cortex=PFCChanges(
                change='Increased thickness/volume',
                function='Executive function, attention regulation',
                meditation_relevance='Enhanced cognitive control'
            ),
            insula=InsulaChanges(
                change='Increased thickness',
                function='Interoception, emotional awareness',
                meditation_relevance='Enhanced body awareness'
            ),
            hippocampus=HippocampusChanges(
                change='Increased volume',
                function='Memory, learning',
                meditation_relevance='Enhanced learning capacity'
            ),
            amygdala=AmygdalaChanges(
                change='Decreased volume/activity',
                function='Threat detection, emotional reactivity',
                meditation_relevance='Reduced emotional reactivity'
            )
        )

    def synthesize_structural_findings(self):
        """
        Synthesize structural brain changes from meditation
        """
        synthesis = {}

        synthesis['attention_regions'] = {
            'regions': ['PFC', 'ACC', 'Parietal cortex'],
            'changes': 'Increased thickness/volume',
            'mechanism': 'Use-dependent plasticity',
            'functional_correlation': 'Enhanced attention capacity'
        }

        synthesis['interoception_regions'] = {
            'regions': ['Insula', 'Somatosensory cortex'],
            'changes': 'Increased thickness',
            'mechanism': 'Enhanced body awareness practice',
            'functional_correlation': 'Improved interoceptive accuracy'
        }

        synthesis['emotional_regions'] = {
            'regions': ['Amygdala', 'Hippocampus', 'PFC'],
            'changes': 'Decreased amygdala, enhanced regulation regions',
            'mechanism': 'Repeated emotional regulation practice',
            'functional_correlation': 'Improved emotional regulation'
        }

        return StructuralSynthesis(
            regional_changes=synthesis,
            neuroplasticity_mechanisms=self.analyze_mechanisms(),
            practice_duration_effects=self.model_dose_response()
        )


class WhiteMatterCorrelates:
    def __init__(self):
        self.white_matter_findings = {
            'tang_2010': Tang2010Study(
                methodology='DTI (diffusion tensor imaging)',
                intervention='11 hours IBMT over 4 weeks',
                finding='Increased FA in corona radiata',
                interpretation='Enhanced white matter integrity'
            ),
            'luders_2011': Luders2011Study(
                sample='Long-term meditators vs controls',
                finding='Widespread increased FA',
                regions=['Corpus callosum', 'Corona radiata', 'Cingulum'],
                interpretation='Enhanced structural connectivity'
            ),
            'holzel_2011_wm': Holzel2011WMStudy(
                design='8-week MBSR',
                finding='FA changes in multiple tracts',
                correlation='Changes correlated with practice hours'
            )
        }

    def analyze_connectivity_implications(self):
        """
        Analyze implications of white matter changes
        """
        implications = {}

        implications['interhemispheric'] = {
            'tract': 'Corpus callosum',
            'change': 'Increased FA',
            'implication': 'Enhanced bilateral integration'
        }

        implications['attention_regulation'] = {
            'tract': 'Cingulum',
            'change': 'Increased FA',
            'implication': 'Enhanced attention-emotion integration'
        }

        implications['executive_function'] = {
            'tract': 'Corona radiata',
            'change': 'Increased FA',
            'implication': 'Enhanced top-down control'
        }

        return WhiteMatterImplications(
            connectivity_changes=implications,
            functional_correlations=self.correlate_with_function(),
            development_timeline=self.model_timeline()
        )
```

## State-Specific Neural Correlates

### Jhana and Absorption State Correlates
```python
class JhanaNeuralCorrelates:
    def __init__(self):
        self.jhana_studies = {
            'mgh_7t_study': MGH7TStudy(
                methodology='Ultra-high field 7T fMRI',
                practitioner='25+ years experience',
                sessions=27,
                findings=JhanaNeuralFindings(
                    cortical_patterns={
                        'somatomotor': 'Distinctive activation patterns',
                        'limbic': 'State-specific changes',
                        'dmn': 'Progressive deactivation',
                        'control_networks': 'Enhanced regulation'
                    },
                    subcortical_patterns={
                        'thalamus': 'Gating function changes',
                        'nucleus_accumbens': 'Reward system activation',
                        'brainstem': 'Arousal regulation changes'
                    },
                    phenomenology_correlation={
                        'bliss_with_reward': 'NAcc activation correlates with piti/sukha',
                        'stability_with_dmn': 'DMN suppression correlates with stability',
                        'depth_with_hierarchy': 'Deeper states show more hierarchy disruption'
                    }
                )
            ),
            'hagerty_2013': Hagerty2013Study(
                finding='Reward circuitry activation in jhana',
                regions=['Nucleus accumbens', 'Medial OFC'],
                interpretation='Jhana bliss has neural basis in reward system'
            )
        }

    def map_jhana_neural_progression(self):
        """
        Map neural changes across jhana progression
        """
        jhana_neural_map = {}

        jhana_neural_map['first_jhana'] = {
            'attention_networks': 'Strongly engaged',
            'dmn': 'Partially suppressed',
            'reward': 'Activated (piti/sukha)',
            'sensory': 'Attenuated external, enhanced internal',
            'verbal': 'Still active (vitakka/vicara)'
        }

        jhana_neural_map['second_jhana'] = {
            'attention_networks': 'Sustained engagement',
            'dmn': 'Further suppressed',
            'reward': 'Sustained activation',
            'sensory': 'Further attenuated',
            'verbal': 'Suppressed'
        }

        jhana_neural_map['third_jhana'] = {
            'attention_networks': 'Effortless engagement',
            'dmn': 'Minimal activity',
            'reward': 'Modulated (sukha without piti)',
            'sensory': 'Minimal',
            'emotional': 'Equanimity networks active'
        }

        jhana_neural_map['fourth_jhana'] = {
            'attention_networks': 'Minimal effortful activity',
            'dmn': 'Deeply suppressed',
            'reward': 'Baseline',
            'sensory': 'Minimal',
            'overall': 'Profound stillness signature'
        }

        jhana_neural_map['formless_jhanas'] = {
            'progression': 'Progressive abstraction',
            'space_perception': 'Altered spatial processing',
            'self_representation': 'Progressively reduced',
            'conceptual': 'Minimal conceptual activity'
        }

        return JhanaNeuralMapping(
            state_map=jhana_neural_map,
            detection_features=self.extract_detection_features(),
            validation_criteria=self.establish_validation_criteria()
        )


class CessationNeuralCorrelates:
    def __init__(self):
        self.cessation_predictions = CessationPredictions(
            theoretical_framework='Laukkonen et al. 2023',
            predicted_pattern=PredictedPattern(
                cortical='Complete suppression of cortical activity',
                subcortical='Minimal subcortical activity',
                brainstem='Possible brainstem involvement in transition',
                phenomenology='No experience during state'
            ),
            research_challenges=Challenges(
                timing='Cannot predict when cessation occurs',
                measurement='No consciousness to report during',
                rarity='Few practitioners achieve state',
                verification='Relies on post-hoc reports'
            )
        )

    def model_cessation_transition(self):
        """
        Model the neural transition into and out of cessation
        """
        transition_model = {}

        transition_model['pre_cessation'] = {
            'eighth_jhana': 'Extremely subtle activity',
            'transition_marker': 'Unknown - research needed',
            'phenomenology': 'Approaching threshold of no experience'
        }

        transition_model['during_cessation'] = {
            'neural_activity': 'Predicted minimal to absent',
            'phenomenology': 'No experience',
            'duration': 'Milliseconds to days (traditional reports)'
        }

        transition_model['post_cessation'] = {
            'emergence': 'Rapid return of neural activity',
            'phenomenology': 'Profound clarity, peace',
            'neural_reset': 'Possible reconfiguration of connectivity'
        }

        return CessationTransitionModel(
            transition_stages=transition_model,
            research_questions=self.identify_research_questions(),
            experimental_approaches=self.propose_experiments()
        )
```

### Non-Dual State Correlates
```python
class NonDualStateCorrelates:
    def __init__(self):
        self.non_dual_findings = {
            'newberg_studies': NewbergStudies(
                methodology='SPECT imaging',
                practitioners=['Buddhist meditators', 'Franciscan nuns'],
                key_finding=NewbergFinding(
                    parietal_changes='Decreased posterior parietal activity',
                    interpretation='Altered spatial self-boundaries',
                    phenomenology_correlation='Unity experience correlates with parietal decrease'
                )
            ),
            'josipovic_2012': Josipovic2012Study(
                methodology='fMRI',
                finding='Reduced anticorrelation between DMN and task-positive networks',
                interpretation='Dissolution of typical network opposition',
                relevance='May underlie subject-object dissolution'
            )
        }

        self.neural_signatures = NonDualSignatures(
            parietal_deactivation=ParietalDeactivation(
                regions=['Posterior parietal cortex', 'TPJ'],
                function='Self-other distinction, spatial orientation',
                change='Reduced activity',
                phenomenology='Dissolved boundaries, unity'
            ),
            network_integration=NetworkIntegration(
                finding='Reduced antagonism between networks',
                interpretation='Integration of normally opposing processes',
                phenomenology='Non-dual awareness'
            ),
            prefrontal_changes=PrefrontalChanges(
                regions=['mPFC', 'dlPFC'],
                change='Variable patterns',
                interpretation='Altered self-modeling'
            )
        )

    def model_non_dual_emergence(self):
        """
        Model neural conditions for non-dual state emergence
        """
        emergence_model = {}

        emergence_model['prerequisites'] = {
            'attention_stability': 'Sustained attention capacity',
            'dmn_regulation': 'Ability to modulate DMN',
            'meta_awareness': 'Clear awareness of awareness'
        }

        emergence_model['transition'] = {
            'parietal_shift': 'Reduced self-location processing',
            'network_reconfiguration': 'Reduced network antagonism',
            'self_model_shift': 'Shift from narrative to minimal self'
        }

        emergence_model['maintenance'] = {
            'effortless_attention': 'Attention without attender',
            'stable_recognition': 'Sustained non-dual awareness',
            'integration': 'Awareness pervades all experience'
        }

        return NonDualEmergenceModel(
            conditions=emergence_model,
            neural_markers=self.identify_markers(),
            detection_approach=self.design_detection_approach()
        )
```

## Neural Correlates Summary Table

```python
class NeuralCorrelatesSummary:
    def __init__(self):
        self.correlate_summary = {
            'gamma_waves': NeuralCorrelate(
                measure='25-100 Hz oscillations',
                states=['Absorption', 'Compassion', 'Insight'],
                change_direction='Increase',
                reliability='High',
                mechanism='Cognitive binding, conscious access'
            ),
            'dmn_deactivation': NeuralCorrelate(
                measure='mPFC, PCC activity',
                states=['Most meditation types'],
                change_direction='Decrease',
                reliability='High',
                mechanism='Reduced self-referential processing'
            ),
            'acc_activation': NeuralCorrelate(
                measure='Dorsal/rostral ACC activity',
                states=['Focused attention', 'Emotional regulation'],
                change_direction='Increase',
                reliability='High',
                mechanism='Enhanced cognitive control'
            ),
            'insula_activation': NeuralCorrelate(
                measure='Anterior insula activity',
                states=['Body awareness', 'Emotional awareness'],
                change_direction='Increase',
                reliability='Moderate',
                mechanism='Enhanced interoception'
            ),
            'frontal_theta': NeuralCorrelate(
                measure='4-8 Hz frontal midline',
                states=['Most meditation types'],
                change_direction='Increase',
                reliability='High',
                mechanism='Internalized attention'
            ),
            'alpha_coherence': NeuralCorrelate(
                measure='8-12 Hz synchrony',
                states=['Relaxed awareness', 'TM'],
                change_direction='Increase',
                reliability='Moderate',
                mechanism='Relaxed alertness'
            ),
            'parietal_changes': NeuralCorrelate(
                measure='Posterior parietal activity',
                states=['Non-dual states', 'Unity experiences'],
                change_direction='Decrease',
                reliability='Moderate',
                mechanism='Altered self-boundaries'
            ),
            'reward_activation': NeuralCorrelate(
                measure='NAcc, mOFC activity',
                states=['Jhana', 'Blissful states'],
                change_direction='Increase',
                reliability='Moderate',
                mechanism='Neural basis of meditative bliss'
            )
        }

    def generate_detection_feature_set(self):
        """
        Generate feature set for computational state detection
        """
        features = {}

        features['eeg_features'] = [
            'gamma_power', 'gamma_coherence',
            'alpha_power', 'alpha_coherence',
            'theta_power', 'theta_coherence',
            'alpha_theta_ratio', 'gamma_theta_ratio'
        ]

        features['fmri_features'] = [
            'dmn_activation', 'dmn_connectivity',
            'acc_activation', 'insula_activation',
            'parietal_activation', 'reward_activation'
        ]

        features['derived_features'] = [
            'network_integration_index',
            'self_referential_processing_index',
            'attention_stability_index'
        ]

        return DetectionFeatureSet(
            feature_list=features,
            extraction_methods=self.define_extraction_methods(),
            normalization_approach=self.define_normalization()
        )
```

## References

### Key Neural Correlates Studies
1. Lutz, A., et al. (2004). Long-term meditators self-induce high-amplitude gamma synchrony. PNAS.
2. Brewer, J.A., et al. (2011). Meditation experience is associated with differences in DMN. PNAS.
3. Lazar, S.W., et al. (2005). Meditation experience is associated with increased cortical thickness. NeuroReport.
4. Tang, Y.Y., et al. (2015). Short-term meditation increases blood flow in ACC and insula. Frontiers in Psychology.
5. Newberg, A., et al. (2001). The measurement of regional cerebral blood flow during glossolalia. Psychiatry Research.
6. Hagerty, M.R., et al. (2013). Case study of ecstatic meditation: fMRI and EEG evidence. Neural Plasticity.
7. Josipovic, Z. (2012). Neural correlates of nondual awareness in meditation. ANYAS.

---

*Document compiled for Form 36: Contemplative & Meditative States*
*Last updated: 2026-01-18*
