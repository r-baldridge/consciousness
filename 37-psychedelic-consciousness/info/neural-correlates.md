# Neural Correlates of Psychedelic Consciousness
**Form 37: Psychedelic/Entheogenic Consciousness**
**Task 37.A.2: Neural Mechanisms of Psychedelic Experiences**
**Date:** January 18, 2026

## Overview

This document details the neural mechanisms underlying psychedelic-induced alterations in consciousness, including receptor pharmacology, network dynamics, neuroplasticity mechanisms, and the neural correlates of specific phenomenological experiences such as ego dissolution, mystical experiences, and visual phenomena.

## Receptor Pharmacology

### Primary Serotonergic Mechanisms

```python
class SerotoninReceptorPharmacology:
    """
    Neural mechanisms involving serotonin receptor activation by classical psychedelics.
    """
    def __init__(self):
        self.primary_target = {
            'receptor': '5-HT2A',
            'location': ReceptorLocation(
                cortical_layer='Layer V pyramidal neurons',
                regions=['Prefrontal cortex', 'Visual cortex', 'Posterior cingulate'],
                density_distribution='High in association cortices'
            ),
            'mechanism': '5-HT2A_ActivationMechanism',
            'downstream_effects': [
                'Glutamate release from thalamo-cortical afferents',
                'Increased cortical excitation',
                'Enhanced sensory processing',
                'Disruption of predictive coding'
            ]
        }

        self.activation_cascade = ActivationCascade(
            steps=[
                CascadeStep(
                    step=1,
                    event="5-HT2A receptor binding",
                    location="Layer V apical dendrites",
                    duration_ms=100
                ),
                CascadeStep(
                    step=2,
                    event="Gq protein activation",
                    effect="Phospholipase C activation",
                    duration_ms=200
                ),
                CascadeStep(
                    step=3,
                    event="IP3 and DAG production",
                    effect="Calcium release, PKC activation",
                    duration_ms=500
                ),
                CascadeStep(
                    step=4,
                    event="Enhanced glutamate transmission",
                    effect="AMPA/NMDA receptor activation",
                    duration_ms=1000
                ),
                CascadeStep(
                    step=5,
                    event="Gene expression changes",
                    effect="BDNF, Arc, c-Fos upregulation",
                    duration_hours=1
                )
            ]
        )

    def model_receptor_binding(self, substance: str, dose: float) -> ReceptorOccupancy:
        """Model receptor binding for given substance and dose."""
        binding_profiles = {
            'psilocin': ReceptorProfile(
                five_ht2a_ki=6.0,  # nM
                five_ht2c_ki=2.0,
                five_ht1a_ki=190.0,
                selectivity='5-HT2C > 5-HT2A > 5-HT1A'
            ),
            'lsd': ReceptorProfile(
                five_ht2a_ki=3.5,
                five_ht2c_ki=5.5,
                five_ht1a_ki=1.1,
                d2_ki=3.5,
                selectivity='Broad serotonergic and dopaminergic'
            ),
            'dmt': ReceptorProfile(
                five_ht2a_ki=127.0,
                five_ht2c_ki=360.0,
                sigma1_ki=14.0,
                selectivity='5-HT2A with sigma-1 activity'
            ),
            'mescaline': ReceptorProfile(
                five_ht2a_ki=6000.0,
                five_ht2b_ki=900.0,
                selectivity='Lower potency, broader profile'
            )
        }

        profile = binding_profiles.get(substance, binding_profiles['psilocin'])
        occupancy = self._calculate_occupancy(profile, dose)

        return ReceptorOccupancy(
            five_ht2a_occupancy=occupancy.five_ht2a,
            five_ht2c_occupancy=occupancy.five_ht2c,
            threshold_for_effects=0.60,  # ~60% occupancy for subjective effects
            saturating_occupancy=0.90
        )

class NonSerotoninergicMechanisms:
    """
    Neural mechanisms for non-classical psychedelics and additional targets.
    """
    def __init__(self):
        self.ketamine_mechanism = DissociativeMechanism(
            primary_target='NMDA receptor antagonist',
            binding_site='PCP site within NMDA channel',
            downstream_effects=[
                'Reduced NMDA-mediated inhibition',
                'Disinhibition of glutamate release',
                'AMPA receptor activation surge',
                'BDNF release and mTOR activation'
            ],
            unique_features=[
                'Rapid antidepressant effects',
                'Dissociative phenomenology',
                'Anesthetic properties at high doses'
            ]
        )

        self.salvia_mechanism = KappaOpioidMechanism(
            primary_target='Kappa opioid receptor agonist',
            unique_properties=[
                'Non-serotonergic psychedelic',
                'Dysphoric potential',
                'Unique phenomenology (reality shattering)',
                'Very short duration'
            ],
            neural_effects=[
                'Kappa-mediated dopamine suppression',
                'Altered claustrum function hypothesis',
                'Disruption of reality monitoring'
            ]
        )

        self.ibogaine_mechanism = ComplexMechanism(
            targets=[
                'NMDA antagonist',
                'Kappa opioid agonist',
                'Sigma-1 receptor',
                'Serotonin transporter',
                'Dopamine transporter'
            ],
            unique_properties=[
                'Opioid withdrawal interruption',
                'Long duration (24-36 hours)',
                'Cardiac risk (QT prolongation)',
                'Visionary life review experiences'
            ]
        )
```

### Glutamate System Integration

```python
class GlutamateSystemEffects:
    """
    Neural correlates involving glutamatergic transmission.
    """
    def __init__(self):
        self.thalamocortical_effects = ThalamoCorticaLoop(
            normal_function="Gating of sensory information",
            psychedelic_effect="Reduced filtering, enhanced throughput",
            mechanism="5-HT2A activation on thalamic neurons",
            consequence="Sensory flooding, enhanced perception"
        )

        self.cortical_effects = CorticalGlutamateEffects(
            regions_affected=[
                CorticalRegion(
                    name="Prefrontal cortex",
                    effect="Increased glutamate release",
                    consequence="Enhanced cognitive flexibility, disrupted executive control"
                ),
                CorticalRegion(
                    name="Visual cortex",
                    effect="Enhanced V1/V2 activity",
                    consequence="Visual hallucinations, form constants"
                ),
                CorticalRegion(
                    name="Temporal cortex",
                    effect="Altered auditory processing",
                    consequence="Music enhancement, auditory distortions"
                ),
                CorticalRegion(
                    name="Parietal cortex",
                    effect="Disrupted spatial processing",
                    consequence="Body distortions, boundary dissolution"
                )
            ]
        )

        self.vollenweider_model = VollenweiderModel(
            hypothesis="Glutamate hypothesis of psychedelic action",
            key_points=[
                "5-HT2A activation triggers glutamate release",
                "Glutamate acts on AMPA receptors",
                "Creates cortical excitation pattern",
                "Explains sensory and cognitive effects"
            ],
            supporting_evidence=[
                "Lamotrigine (glutamate inhibitor) blocks some LSD effects",
                "NMDA antagonists have partial psychedelic overlap",
                "Cortical glutamate increases under psilocybin"
            ]
        )

    def model_cortical_excitation(self, substance: str, dose: float) -> CorticalExcitationPattern:
        """Model cortical excitation patterns."""
        return CorticalExcitationPattern(
            prefrontal_activation=self._calculate_regional_activation('pfc', substance, dose),
            visual_activation=self._calculate_regional_activation('visual', substance, dose),
            temporal_activation=self._calculate_regional_activation('temporal', substance, dose),
            thalamic_gating=self._calculate_thalamic_effects(substance, dose),
            overall_entropy=self._calculate_entropy_increase(substance, dose)
        )
```

## Network-Level Dynamics

### Default Mode Network Effects

```python
class DefaultModeNetworkEffects:
    """
    Neural correlates of DMN changes during psychedelic states.
    """
    def __init__(self):
        self.dmn_components = {
            'medial_prefrontal_cortex': DMNNode(
                function="Self-referential processing, autobiographical memory",
                psychedelic_effect="Decreased activity",
                ego_dissolution_correlation=0.72
            ),
            'posterior_cingulate_cortex': DMNNode(
                function="Autobiographical memory, self-reflection",
                psychedelic_effect="Decreased activity and connectivity",
                ego_dissolution_correlation=0.81
            ),
            'angular_gyrus': DMNNode(
                function="Theory of mind, semantic processing",
                psychedelic_effect="Reduced connectivity to DMN hub",
                ego_dissolution_correlation=0.65
            ),
            'hippocampus': DMNNode(
                function="Memory encoding and retrieval",
                psychedelic_effect="Altered coupling with cortex",
                ego_dissolution_correlation=0.58
            )
        }

        self.dmn_disruption_findings = {
            'psilocybin_fmri_2012': ImagingFinding(
                study="Carhart-Harris et al., 2012",
                modality="fMRI",
                finding="Decreased DMN activity correlates with subjective effects",
                significance="First demonstration of DMN-psychedelic link"
            ),
            'lsd_connectivity_2016': ImagingFinding(
                study="Carhart-Harris et al., 2016",
                modality="fMRI",
                finding="Reduced DMN integrity under LSD",
                significance="DMN disintegration predicts ego dissolution"
            ),
            'entropy_increase': ImagingFinding(
                study="Tagliazucchi et al., 2014",
                modality="fMRI entropy analysis",
                finding="Increased entropy in DMN regions",
                significance="Supports entropic brain hypothesis"
            )
        }

    def model_dmn_disruption(
        self,
        substance: str,
        dose: float,
        baseline_dmn_activity: float
    ) -> DMNDisruptionState:
        """Model DMN disruption for given parameters."""
        disruption_factors = {
            'psilocybin': 0.35,
            'lsd': 0.40,
            'dmt': 0.45,
            'mdma': 0.15,
            'ketamine': 0.30
        }

        base_factor = disruption_factors.get(substance, 0.30)
        dose_adjusted = base_factor * (1 + np.log1p(dose / 10))

        return DMNDisruptionState(
            mpfc_activity_reduction=dose_adjusted * 0.40,
            pcc_activity_reduction=dose_adjusted * 0.45,
            connectivity_reduction=dose_adjusted * 0.50,
            ego_dissolution_prediction=self._predict_ego_dissolution(dose_adjusted),
            therapeutic_potential=self._assess_therapeutic_potential(dose_adjusted)
        )

class GlobalConnectivityChanges:
    """
    Neural correlates of increased global brain connectivity.
    """
    def __init__(self):
        self.connectivity_patterns = {
            'increased_global_connectivity': ConnectivityPattern(
                description="More connections between normally segregated regions",
                mechanism="Reduced hierarchical constraints",
                functional_consequence="Novel information integration patterns",
                measurement="Functional connectivity matrices, graph theory metrics"
            ),
            'reduced_network_segregation': ConnectivityPattern(
                description="Decreased modularity of functional networks",
                mechanism="DMN-TPN boundary dissolution",
                functional_consequence="Mixed self-referential and task processing",
                measurement="Modularity index, network segregation metrics"
            ),
            'enhanced_thalamocortical_connectivity': ConnectivityPattern(
                description="Increased information flow from thalamus to cortex",
                mechanism="5-HT2A effects on thalamic gating",
                functional_consequence="Sensory flooding, enhanced perception",
                measurement="Dynamic causal modeling, transfer entropy"
            )
        }

        self.homological_scaffolds = HomologicalAnalysis(
            study="Petri et al., 2014",
            method="Persistent homology on fMRI data",
            finding="New topological features emerge under psilocybin",
            interpretation="Novel patterns of brain organization",
            clinical_relevance="May underlie cognitive flexibility increase"
        )

    def compute_connectivity_metrics(
        self,
        fmri_data: np.ndarray,
        condition: str
    ) -> ConnectivityMetrics:
        """Compute connectivity metrics from fMRI data."""
        return ConnectivityMetrics(
            global_efficiency=self._compute_global_efficiency(fmri_data),
            modularity=self._compute_modularity(fmri_data),
            participation_coefficient=self._compute_participation(fmri_data),
            integration_segregation_balance=self._compute_balance(fmri_data),
            entropy=self._compute_signal_entropy(fmri_data)
        )
```

## Neuroplasticity Mechanisms

### Structural and Functional Plasticity

```python
class PsychedelicNeuroplasticity:
    """
    Neural correlates of psychedelic-induced neuroplasticity.
    """
    def __init__(self):
        self.plasticity_mechanisms = {
            'dendritic_spine_growth': PlasticityMechanism(
                level="Structural",
                effect="Increased dendritic spine density",
                timeline="Hours to days post-administration",
                evidence=[
                    "Ly et al. (2018) - in vitro and in vivo",
                    "Shao et al. (2021) - psilocybin in mice"
                ],
                molecular_pathway=[
                    "5-HT2A activation",
                    "TrkB receptor activation",
                    "mTOR signaling",
                    "Protein synthesis",
                    "Spine morphogenesis"
                ]
            ),
            'bdnf_upregulation': PlasticityMechanism(
                level="Molecular",
                effect="Increased BDNF expression and release",
                timeline="Hours post-administration",
                evidence=[
                    "Multiple studies show BDNF increase",
                    "Correlates with antidepressant effects"
                ],
                molecular_pathway=[
                    "5-HT2A/AMPA activation",
                    "Calcium influx",
                    "CREB phosphorylation",
                    "BDNF transcription",
                    "TrkB signaling cascade"
                ]
            ),
            'synaptic_strengthening': PlasticityMechanism(
                level="Functional",
                effect="Enhanced synaptic transmission",
                timeline="Acute to sustained",
                evidence=[
                    "Enhanced LTP in hippocampus",
                    "Increased AMPA receptor expression"
                ],
                molecular_pathway=[
                    "Glutamate release",
                    "AMPA receptor trafficking",
                    "Postsynaptic density remodeling"
                ]
            )
        }

        self.critical_period_reopening = CriticalPeriodHypothesis(
            hypothesis="Psychedelics may reopen critical periods for learning",
            evidence=[
                "Enhanced ocular dominance plasticity (Grieco et al., 2024)",
                "Increased learning of new associations",
                "Reduced habitual/rigid behaviors"
            ],
            therapeutic_implication="May enable 'rewriting' of maladaptive patterns",
            proposed_mechanism="5-HT2A modulation of PNN and plasticity brakes"
        )

    def model_plasticity_timeline(
        self,
        substance: str,
        dose: float
    ) -> PlasticityTimeline:
        """Model the timeline of plasticity effects."""
        return PlasticityTimeline(
            acute_phase=PlasticityPhase(
                duration_hours=6,
                effects=["Receptor activation", "Glutamate surge", "Gene induction"],
                reversible=True
            ),
            early_consolidation=PlasticityPhase(
                duration_hours=24,
                effects=["BDNF synthesis", "Early spine changes", "mTOR activation"],
                reversible="Partially"
            ),
            late_consolidation=PlasticityPhase(
                duration_days=7,
                effects=["Spine maturation", "Synaptic strengthening", "Network reorganization"],
                reversible="Partially"
            ),
            maintenance=PlasticityPhase(
                duration_weeks=4,
                effects=["Sustained connectivity changes", "Behavioral plasticity"],
                reversible="Experience-dependent"
            )
        )

class KetamineSpecificPlasticity:
    """
    Neural plasticity mechanisms specific to ketamine.
    """
    def __init__(self):
        self.glutamate_surge_hypothesis = GlutamateSurgeModel(
            mechanism=[
                "NMDA blockade on GABAergic interneurons",
                "Disinhibition of glutamatergic pyramidal neurons",
                "Glutamate surge in prefrontal cortex",
                "AMPA receptor activation",
                "BDNF release and mTOR activation"
            ],
            therapeutic_relevance="Explains rapid antidepressant effects",
            supporting_evidence=[
                "AMPA antagonists block ketamine's antidepressant effects",
                "mTOR inhibitors block ketamine's effects",
                "Glutamate increase observed in vivo"
            ]
        )

        self.rapid_synaptic_effects = RapidSynapticChanges(
            timeline_hours=2,
            effects=[
                "Increased synaptic protein synthesis",
                "Enhanced spine density in PFC",
                "Restored synaptic deficits in stress models"
            ],
            persistence="Days to weeks without further dosing"
        )
```

## Neural Correlates of Specific Experiences

### Ego Dissolution Neural Correlates

```python
class EgoDissolutionNeuralCorrelates:
    """
    Neural correlates specifically associated with ego dissolution experiences.
    """
    def __init__(self):
        self.neural_signatures = {
            'dmn_pcc_disruption': NeuralSignature(
                region="Posterior cingulate cortex",
                effect="Decreased activity and connectivity",
                correlation_with_edi=0.81,
                interpretation="PCC is key hub for self-referential processing"
            ),
            'dmn_mpfc_disruption': NeuralSignature(
                region="Medial prefrontal cortex",
                effect="Decreased activity",
                correlation_with_edi=0.72,
                interpretation="mPFC involved in self-reflection and narrative self"
            ),
            'parietal_disconnection': NeuralSignature(
                region="Inferior parietal lobule",
                effect="Altered body representation",
                correlation_with_edi=0.65,
                interpretation="Body boundary dissolution"
            ),
            'insula_changes': NeuralSignature(
                region="Insula",
                effect="Altered interoceptive processing",
                correlation_with_edi=0.60,
                interpretation="Changed sense of embodied self"
            ),
            'claustrum_hypothesis': NeuralSignature(
                region="Claustrum",
                effect="Potential disruption of binding",
                correlation_with_edi="Theoretical",
                interpretation="Consciousness binding disruption (Crick hypothesis)"
            )
        }

        self.network_level_changes = {
            'dmn_tpn_coupling': NetworkChange(
                networks=["Default Mode Network", "Task Positive Network"],
                normal_state="Anti-correlated (segregated)",
                psychedelic_state="Reduced anti-correlation (merged)",
                consequence="Blurring of self-world boundary",
                ego_dissolution_correlation=0.75
            ),
            'salience_network_changes': NetworkChange(
                networks=["Salience Network"],
                effect="Altered switching between DMN and TPN",
                consequence="Disrupted self-monitoring",
                ego_dissolution_correlation=0.62
            )
        }

    def predict_ego_dissolution(
        self,
        dmn_connectivity: float,
        pcc_activity: float,
        dose: float
    ) -> EgoDissolutionPrediction:
        """Predict ego dissolution from neural measures."""
        # Model based on empirical correlations
        weighted_score = (
            (1 - dmn_connectivity) * 0.40 +
            (1 - pcc_activity) * 0.35 +
            np.log1p(dose) * 0.25
        )

        return EgoDissolutionPrediction(
            predicted_edi_score=weighted_score * 10,
            confidence_interval=(weighted_score * 8, weighted_score * 12),
            contributing_factors={
                'dmn_disruption': (1 - dmn_connectivity) * 0.40,
                'pcc_reduction': (1 - pcc_activity) * 0.35,
                'dose_effect': np.log1p(dose) * 0.25
            }
        )

### Visual Phenomena Neural Correlates

class VisualPhenomenaNeuralCorrelates:
    """
    Neural correlates of psychedelic visual experiences.
    """
    def __init__(self):
        self.visual_system_effects = {
            'v1_hyperactivation': VisualCortexEffect(
                region="Primary visual cortex (V1)",
                effect="Increased spontaneous activity",
                phenomenology="Basic form constants, geometry",
                mechanism="5-HT2A receptors in V1, reduced thalamic gating"
            ),
            'v4_color_processing': VisualCortexEffect(
                region="V4 color area",
                effect="Enhanced color processing",
                phenomenology="Color intensification, novel colors",
                mechanism="Serotonergic modulation of color channels"
            ),
            'mt_motion_processing': VisualCortexEffect(
                region="MT/V5 motion area",
                effect="Altered motion processing",
                phenomenology="Drifting, flowing, breathing surfaces",
                mechanism="Temporal processing changes"
            ),
            'higher_visual_areas': VisualCortexEffect(
                region="Inferotemporal cortex, fusiform",
                effect="Pattern completion, face pareidolia",
                phenomenology="Complex imagery, faces in patterns",
                mechanism="Top-down pattern matching disinhibition"
            )
        }

        self.form_constants_neural_basis = FormConstantsModel(
            hypothesis="Ermentrout-Cowan neural field theory",
            mechanism=[
                "Spontaneous pattern formation in V1",
                "Symmetry breaking in neural activity",
                "Lateral inhibition creating stripe/spot patterns",
                "Cortical magnification maps patterns to visual field"
            ],
            pattern_types={
                'tunnels': "Concentric activity patterns",
                'spirals': "Rotating wave patterns",
                'cobwebs': "Hexagonal lattice activity",
                'gratings': "Striped activity patterns"
            }
        )

    def model_visual_intensity(
        self,
        substance: str,
        dose: float,
        eyes_closed: bool
    ) -> VisualIntensityPrediction:
        """Model visual intensity from neural factors."""
        substance_factors = {
            'dmt': 1.0,
            'lsd': 0.85,
            'psilocybin': 0.75,
            'mescaline': 0.70,
            'ketamine': 0.40
        }

        base_factor = substance_factors.get(substance, 0.5)
        closed_eye_boost = 1.3 if eyes_closed else 1.0

        intensity = base_factor * np.log1p(dose / 5) * closed_eye_boost

        return VisualIntensityPrediction(
            form_constants_intensity=intensity * 0.8,
            complex_imagery_intensity=intensity * 0.6,
            color_enhancement=intensity * 0.9,
            motion_effects=intensity * 0.7
        )
```

### Mystical Experience Neural Correlates

```python
class MysticalExperienceNeuralCorrelates:
    """
    Neural correlates of complete mystical experiences.
    """
    def __init__(self):
        self.neural_signatures = {
            'unity_experience': NeuralSignature(
                networks_involved=["DMN", "Salience Network", "Visual Network"],
                signature="Global network integration",
                mechanism="Reduced network segregation, enhanced global connectivity",
                phenomenology="Internal and external unity, boundary dissolution"
            ),
            'transcendence': NeuralSignature(
                networks_involved=["DMN", "Temporal-parietal junction"],
                signature="Altered temporal processing, parietal changes",
                mechanism="Disrupted time estimation circuits",
                phenomenology="Timelessness, spacelessness"
            ),
            'noetic_quality': NeuralSignature(
                networks_involved=["Prefrontal cortex", "Language networks"],
                signature="Enhanced prefrontal activity with DMN suppression",
                mechanism="Direct knowing bypassing narrative processing",
                phenomenology="Sense of profound truth and meaning"
            ),
            'sacredness': NeuralSignature(
                networks_involved=["Limbic system", "Insula", "mPFC"],
                signature="Emotional limbic activation with cortical integration",
                mechanism="Enhanced emotional processing with meaning attribution",
                phenomenology="Sense of the holy, reverence"
            )
        }

        self.predictive_model = MysticalExperiencePredictiveModel(
            predictors=[
                Predictor(
                    name="DMN disruption magnitude",
                    weight=0.35,
                    threshold=0.40
                ),
                Predictor(
                    name="5-HT2A occupancy",
                    weight=0.25,
                    threshold=0.60
                ),
                Predictor(
                    name="Global connectivity increase",
                    weight=0.25,
                    threshold=0.30
                ),
                Predictor(
                    name="Set and setting factors",
                    weight=0.15,
                    threshold="Supportive environment"
                )
            ],
            outcome_correlation_with_meq30=0.78
        )

    def assess_mystical_potential(
        self,
        neural_state: NeuralState,
        set_setting: SetSettingFactors
    ) -> MysticalPotentialAssessment:
        """Assess potential for mystical experience from neural state."""
        neural_score = (
            neural_state.dmn_disruption * 0.35 +
            neural_state.receptor_occupancy * 0.25 +
            neural_state.connectivity_increase * 0.25
        )

        setting_score = set_setting.compute_supportiveness() * 0.15

        total_score = neural_score + setting_score

        return MysticalPotentialAssessment(
            predicted_meq30=total_score * 100,
            unity_probability=self._sigmoid(total_score, 0.6),
            transcendence_probability=self._sigmoid(total_score, 0.7),
            therapeutic_outcome_prediction=total_score * 0.8
        )
```

## Entropy and Information Processing

```python
class EntropicBrainMechanisms:
    """
    Neural mechanisms underlying the Entropic Brain Hypothesis.
    """
    def __init__(self):
        self.entropy_measures = {
            'lempel_ziv_complexity': EntropyMeasure(
                name="Lempel-Ziv Complexity",
                description="Compressibility of neural signals",
                psychedelic_effect="Increased complexity",
                interpretation="More unpredictable brain activity"
            ),
            'sample_entropy': EntropyMeasure(
                name="Sample Entropy",
                description="Irregularity of time series",
                psychedelic_effect="Increased entropy",
                interpretation="Less predictable dynamics"
            ),
            'integration_information': EntropyMeasure(
                name="Integration/Information balance",
                description="PHI-like measures",
                psychedelic_effect="Altered integration patterns",
                interpretation="Changed consciousness integration"
            )
        }

        self.rebus_model = REBUSModel(
            name="Relaxed Beliefs Under Psychedelics",
            authors="Carhart-Harris & Friston, 2019",
            framework="Predictive processing / Free energy",
            key_principles=[
                "Brain as hierarchical prediction machine",
                "Psychedelics relax high-level priors",
                "Bottom-up signals gain influence",
                "Enables revision of maladaptive beliefs"
            ],
            therapeutic_mechanism=[
                "Overly rigid priors underlie disorders",
                "Depression: negative self-priors",
                "Addiction: habitual behavioral priors",
                "Psychedelics allow prior revision"
            ]
        )

    def compute_entropy_state(
        self,
        eeg_data: np.ndarray,
        condition: str
    ) -> EntropyState:
        """Compute entropy measures from EEG data."""
        return EntropyState(
            lz_complexity=self._compute_lz_complexity(eeg_data),
            sample_entropy=self._compute_sample_entropy(eeg_data),
            spectral_entropy=self._compute_spectral_entropy(eeg_data),
            criticality_index=self._assess_criticality(eeg_data),
            consciousness_level_prediction=self._predict_consciousness_level(eeg_data)
        )
```

## Conclusion

The neural correlates of psychedelic consciousness involve:

1. **Receptor Mechanisms**: 5-HT2A activation as primary driver, with glutamate cascade downstream
2. **Network Dynamics**: DMN disruption, increased global connectivity, reduced network segregation
3. **Neuroplasticity**: Rapid structural changes, BDNF upregulation, potential critical period reopening
4. **Experience-Specific Correlates**: Distinct neural signatures for ego dissolution, visual phenomena, and mystical experiences
5. **Information Processing**: Increased entropy, relaxed priors, enhanced flexibility

These mechanisms collectively explain both the acute experiential effects and the lasting therapeutic benefits of psychedelic-assisted interventions.
