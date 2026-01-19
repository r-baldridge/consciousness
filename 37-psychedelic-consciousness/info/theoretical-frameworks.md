# Theoretical Frameworks of Psychedelic Consciousness
**Form 37: Psychedelic/Entheogenic Consciousness**
**Task 37.A.3: Theories of Psychedelic Consciousness**
**Date:** January 18, 2026

## Overview

This document examines the major theoretical frameworks for understanding psychedelic-induced alterations in consciousness, including the Entropic Brain Hypothesis, REBUS model, predictive processing accounts, Global Workspace Theory applications, Integrated Information Theory perspectives, and traditional/indigenous cosmologies.

## The Entropic Brain Hypothesis

### Core Theory

```python
class EntropicBrainHypothesis:
    """
    The Entropic Brain Hypothesis proposed by Robin Carhart-Harris (2014).
    """
    def __init__(self):
        self.author = "Robin Carhart-Harris"
        self.year_proposed = 2014
        self.journal = "Frontiers in Human Neuroscience"

        self.core_principles = EntropicPrinciples(
            entropy_spectrum=EntropySpectrum(
                low_entropy_states=[
                    "Deep sleep", "Anesthesia", "Sedation",
                    "Rigid depression", "Obsessive states"
                ],
                normal_waking="Constrained, moderate entropy",
                high_entropy_states=[
                    "Psychedelic states", "Dreaming (REM)",
                    "Early psychosis onset", "Meditation peaks"
                ]
            ),
            consciousness_as_entropy=ConsciousnessEntropyRelation(
                hypothesis="Consciousness correlates with entropy within bounds",
                optimal_range="Criticality - between order and disorder",
                too_low="Unconsciousness, rigidity",
                too_high="Disorganization, potential psychosis"
            )
        )

        self.primary_secondary_consciousness = PrimarySecondaryModel(
            primary_consciousness=ConsciousnessMode(
                name="Primary Consciousness",
                characteristics=[
                    "Unconstrained cognition",
                    "Present-moment awareness",
                    "Infant-like or animal-like",
                    "High entropy",
                    "Magical thinking",
                    "Ego dissolution"
                ],
                neural_correlates="Reduced DMN constraint, high connectivity",
                examples=["Psychedelic peak", "Dreaming", "Early infancy"]
            ),
            secondary_consciousness=ConsciousnessMode(
                name="Secondary Consciousness",
                characteristics=[
                    "Constrained cognition",
                    "Reality-tested",
                    "Ego-bound",
                    "Low entropy",
                    "Narrative self",
                    "Metacognition"
                ],
                neural_correlates="Strong DMN activity, network segregation",
                examples=["Normal waking", "Focused attention", "Adult cognition"]
            )
        )

    def model_entropy_shift(
        self,
        baseline_entropy: float,
        substance: str,
        dose: float
    ) -> EntropyShiftResult:
        """Model the entropy shift induced by psychedelics."""
        substance_factors = {
            'psilocybin': 0.35,
            'lsd': 0.40,
            'dmt': 0.50,
            'ayahuasca': 0.38,
            'mescaline': 0.30,
            'ketamine': 0.25
        }

        max_shift = substance_factors.get(substance, 0.30)
        dose_factor = 1 - np.exp(-dose / 20)  # Saturating dose-response
        entropy_increase = max_shift * dose_factor

        new_entropy = baseline_entropy + entropy_increase

        return EntropyShiftResult(
            baseline_entropy=baseline_entropy,
            peak_entropy=new_entropy,
            entropy_increase=entropy_increase,
            consciousness_mode=self._determine_mode(new_entropy),
            therapeutic_window=self._assess_therapeutic_window(new_entropy),
            risk_assessment=self._assess_entropy_risk(new_entropy)
        )

    def _determine_mode(self, entropy: float) -> str:
        """Determine consciousness mode from entropy level."""
        if entropy < 0.3:
            return "Severely constrained / unconscious"
        elif entropy < 0.5:
            return "Secondary consciousness (normal)"
        elif entropy < 0.7:
            return "Enhanced flexibility / mild primary"
        elif entropy < 0.85:
            return "Primary consciousness / psychedelic"
        else:
            return "Extreme primary / disorganization risk"
```

### Criticisms and Refinements

```python
class EntropicBrainCritiques:
    """
    Critical evaluation and refinements of the Entropic Brain Hypothesis.
    """
    def __init__(self):
        self.methodological_critiques = [
            Critique(
                issue="Entropy measurement validity",
                description="Different entropy measures yield different results",
                response="Multiple converging measures support hypothesis",
                resolution_status="Ongoing refinement"
            ),
            Critique(
                issue="Correlation vs causation",
                description="Entropy increase may be epiphenomenal",
                response="Causal models being developed",
                resolution_status="Active research"
            ),
            Critique(
                issue="Not all high-entropy states are therapeutic",
                description="Psychosis is high-entropy but not beneficial",
                response="Context and reversibility matter",
                resolution_status="Addressed in REBUS model"
            )
        ]

        self.theoretical_refinements = [
            Refinement(
                name="Optimal entropy range",
                contribution="Therapeutic effects require bounded entropy increase",
                implication="Set and setting maintain beneficial range"
            ),
            Refinement(
                name="Controlled entropy",
                contribution="Reversible, supported entropy increase is key",
                implication="Distinguishes therapy from pathology"
            ),
            Refinement(
                name="Integration of REBUS",
                contribution="Entropy connected to belief revision",
                implication="Mechanistic link to therapeutic change"
            )
        ]
```

## REBUS Model (Relaxed Beliefs Under Psychedelics)

### Predictive Processing Framework

```python
class REBUSModel:
    """
    The REBUS (Relaxed Beliefs Under Psychedelics) model.
    Carhart-Harris & Friston, 2019.
    """
    def __init__(self):
        self.authors = ["Robin Carhart-Harris", "Karl Friston"]
        self.year = 2019
        self.journal = "Pharmacological Reviews"
        self.framework = "Predictive Processing / Free Energy Principle"

        self.predictive_processing_basis = PredictiveProcessingModel(
            core_idea="Brain as hierarchical prediction machine",
            key_concepts={
                'priors': "Top-down predictions/beliefs about world",
                'prediction_errors': "Bottom-up signals when predictions fail",
                'precision': "Confidence weighting of predictions vs errors",
                'hierarchy': "Multiple levels from sensory to abstract"
            },
            normal_function="Minimize prediction error via updating beliefs or acting"
        )

        self.rebus_mechanism = REBUSMechanism(
            psychedelic_effect="Relaxation of high-level priors",
            mechanism_steps=[
                MechanismStep(
                    step=1,
                    process="5-HT2A activation in deep cortical layers",
                    effect="Reduced precision of high-level predictions"
                ),
                MechanismStep(
                    step=2,
                    process="Weakened top-down constraints",
                    effect="Bottom-up signals gain relative influence"
                ),
                MechanismStep(
                    step=3,
                    process="Prior beliefs become revisable",
                    effect="Maladaptive beliefs can be updated"
                ),
                MechanismStep(
                    step=4,
                    process="New patterns of prediction established",
                    effect="Therapeutic restructuring of beliefs"
                )
            ],
            neural_implementation={
                'layer_v_pyramidal': "Site of 5-HT2A activation",
                'reduced_alpha': "Marker of relaxed priors",
                'increased_entropy': "Reflects prediction uncertainty"
            }
        )

    def model_belief_revision(
        self,
        prior_strength: float,
        prediction_error: float,
        psychedelic_effect: float
    ) -> BeliefRevisionResult:
        """Model belief revision under psychedelic influence."""
        # Standard Bayesian update
        normal_posterior = self._bayesian_update(prior_strength, prediction_error, precision=1.0)

        # REBUS: reduced prior precision
        relaxed_precision = 1.0 - (psychedelic_effect * 0.6)
        rebus_posterior = self._bayesian_update(
            prior_strength,
            prediction_error,
            precision=relaxed_precision
        )

        return BeliefRevisionResult(
            prior_belief=prior_strength,
            prediction_error=prediction_error,
            normal_posterior=normal_posterior,
            rebus_posterior=rebus_posterior,
            revision_magnitude=abs(rebus_posterior - normal_posterior),
            therapeutic_potential=self._assess_revision_benefit(
                prior_strength, rebus_posterior
            )
        )

class REBUSClinicalApplications:
    """
    Clinical applications of the REBUS framework.
    """
    def __init__(self):
        self.disorder_models = {
            'depression': DisorderModel(
                maladaptive_priors=[
                    "I am worthless",
                    "The future is hopeless",
                    "I cannot change"
                ],
                prior_rigidity="High - resistant to contradicting evidence",
                rebus_intervention=[
                    "Relax negative self-priors",
                    "Allow positive evidence to update beliefs",
                    "Enable new self-narratives"
                ],
                clinical_evidence="Psilocybin trials show belief revision"
            ),
            'addiction': DisorderModel(
                maladaptive_priors=[
                    "I need the substance",
                    "I cannot cope without it",
                    "Habitual behavioral patterns"
                ],
                prior_rigidity="High - compulsive, automatic",
                rebus_intervention=[
                    "Relax habitual priors",
                    "Enable consideration of alternatives",
                    "Update identity beliefs"
                ],
                clinical_evidence="Smoking cessation, alcohol use studies"
            ),
            'ptsd': DisorderModel(
                maladaptive_priors=[
                    "The world is dangerous",
                    "I am damaged",
                    "Trauma memories are overwhelming"
                ],
                prior_rigidity="High - fear-conditioned",
                rebus_intervention=[
                    "Reduce threat priors temporarily",
                    "Enable trauma memory reprocessing",
                    "Update safety beliefs"
                ],
                clinical_evidence="MDMA-AT trials (related mechanism)"
            ),
            'ocd': DisorderModel(
                maladaptive_priors=[
                    "Contamination is everywhere",
                    "Something terrible will happen",
                    "I must perform rituals"
                ],
                prior_rigidity="Extremely high - crystallized",
                rebus_intervention=[
                    "Relax obsessional priors",
                    "Reduce compulsion necessity beliefs",
                    "Enable flexible responses"
                ],
                clinical_evidence="Early psilocybin OCD studies"
            )
        }

    def predict_therapeutic_response(
        self,
        disorder: str,
        prior_rigidity: float,
        substance_effect: float,
        set_setting_quality: float
    ) -> TherapeuticPrediction:
        """Predict therapeutic response from REBUS parameters."""
        disorder_model = self.disorder_models.get(disorder)

        revision_potential = substance_effect * (1 - prior_rigidity * 0.5)
        integration_factor = set_setting_quality * 0.3

        therapeutic_score = revision_potential + integration_factor

        return TherapeuticPrediction(
            predicted_response=therapeutic_score,
            belief_revision_likelihood=revision_potential,
            integration_importance=integration_factor,
            recommendations=self._generate_recommendations(therapeutic_score, disorder)
        )
```

## Global Workspace Theory Applications

```python
class GlobalWorkspaceTheoryPsychedelics:
    """
    Application of Global Workspace Theory to psychedelic consciousness.
    """
    def __init__(self):
        self.gwt_basics = GlobalWorkspaceBasics(
            author="Bernard Baars",
            core_concept="Consciousness as global broadcasting in workspace",
            workspace_function="Integrate and broadcast information widely",
            access_consciousness="Information available to multiple systems"
        )

        self.psychedelic_gwt_model = PsychedelicGWTModel(
            normal_state=WorkspaceState(
                name="Normal waking",
                workspace_access="Selective, attention-gated",
                broadcasting="Focused, hierarchical",
                contents="Reality-tested, coherent"
            ),
            psychedelic_state=WorkspaceState(
                name="Psychedelic",
                workspace_access="Expanded, reduced gating",
                broadcasting="Widespread, non-hierarchical",
                contents="Diverse, including normally unconscious"
            ),
            transition_mechanism=[
                "5-HT2A effects reduce workspace selectivity",
                "Normally excluded information gains access",
                "Broadcasting becomes more democratic",
                "Contents become more varied and novel"
            ]
        )

        self.clinical_implications = GWTClinicalImplications(
            therapeutic_mechanism="Expanded workspace access",
            benefits=[
                "Unconscious material becomes conscious",
                "Novel information integration",
                "Breaking habitual thought patterns"
            ],
            requirements=[
                "Sufficient workspace capacity maintained",
                "Integration support post-experience",
                "Appropriate content for therapeutic focus"
            ]
        )

    def model_workspace_dynamics(
        self,
        baseline_selectivity: float,
        psychedelic_effect: float
    ) -> WorkspaceDynamics:
        """Model workspace dynamics under psychedelics."""
        reduced_selectivity = baseline_selectivity * (1 - psychedelic_effect * 0.5)
        expanded_access = 1 - reduced_selectivity

        return WorkspaceDynamics(
            selectivity=reduced_selectivity,
            access_breadth=expanded_access,
            information_diversity=expanded_access * 0.8,
            coherence=1 - expanded_access * 0.3,
            therapeutic_window=self._assess_therapeutic_window(
                expanded_access, coherence=1 - expanded_access * 0.3
            )
        )
```

## Integrated Information Theory Perspectives

```python
class IITPsychedelicPerspectives:
    """
    Integrated Information Theory perspectives on psychedelic consciousness.
    """
    def __init__(self):
        self.iit_basics = IITBasics(
            authors=["Giulio Tononi", "Christof Koch"],
            core_concept="Consciousness = Integrated Information (Phi)",
            phi_definition="Information generated by a system above its parts",
            key_axioms=["Intrinsic existence", "Composition", "Information",
                       "Integration", "Exclusion"]
        )

        self.psychedelic_iit_predictions = PsychedelicIITPredictions(
            phi_under_psychedelics=PhiPrediction(
                prediction="Complex changes in Phi",
                rationale="Increased connectivity may increase integration",
                complication="But reduced differentiation may decrease Phi",
                empirical_status="Limited direct measurements"
            ),
            quality_of_experience=QualitativePrediction(
                prediction="Altered qualia structure",
                rationale="Different information geometry",
                phenomenological_mapping="Explains ineffability, novel qualia"
            )
        )

        self.consciousness_level_analysis = ConsciousnessLevelAnalysis(
            perturbational_complexity_index=PCIAnalysis(
                measure="Response complexity to TMS perturbation",
                psychedelic_finding="Generally preserved or increased",
                interpretation="Consciousness maintained, content altered"
            ),
            integration_segregation_balance=ISBalance(
                normal="Balance between integration and segregation",
                psychedelic="Shift toward integration",
                implication="Changed but not diminished consciousness"
            )
        )

    def analyze_phi_changes(
        self,
        baseline_connectivity: np.ndarray,
        psychedelic_connectivity: np.ndarray
    ) -> PhiChangeAnalysis:
        """Analyze changes in integrated information proxies."""
        baseline_integration = self._compute_integration_proxy(baseline_connectivity)
        psychedelic_integration = self._compute_integration_proxy(psychedelic_connectivity)

        baseline_differentiation = self._compute_differentiation(baseline_connectivity)
        psychedelic_differentiation = self._compute_differentiation(psychedelic_connectivity)

        return PhiChangeAnalysis(
            integration_change=psychedelic_integration - baseline_integration,
            differentiation_change=psychedelic_differentiation - baseline_differentiation,
            phi_proxy_change=self._compute_phi_proxy_change(
                baseline_connectivity, psychedelic_connectivity
            ),
            consciousness_prediction=self._predict_consciousness_quality(
                psychedelic_integration, psychedelic_differentiation
            )
        )
```

## Indigenous and Traditional Cosmologies

```python
class TraditionalCosmologies:
    """
    Traditional and indigenous frameworks for understanding entheogenic consciousness.
    """
    def __init__(self):
        self.amazonian_cosmology = AmazonianCosmology(
            tradition="Ayahuasca traditions",
            core_concepts={
                'plant_teachers': PlantTeacherConcept(
                    description="Plants as conscious, teaching entities",
                    relationship="Reciprocal, respectful communion",
                    mechanism="Direct spirit communication"
                ),
                'healing_paradigm': HealingParadigm(
                    diagnosis="Spiritual/energetic imbalance",
                    treatment="Spirit intervention, energy extraction",
                    role_of_visions="Diagnostic and healing tools"
                ),
                'cosmological_levels': CosmologicalLevels(
                    worlds=["Upper world", "Middle world", "Lower world"],
                    access_method="Plant-mediated journeying",
                    entities="Spirits, ancestors, plant beings"
                )
            }
        )

        self.mesoamerican_cosmology = MesoamericanCosmology(
            tradition="Mazatec mushroom tradition",
            core_concepts={
                'mushrooms_as_saints': SaintsConcept(
                    name="Nti si tho - Little Saints",
                    role="Intermediaries with divine",
                    respect_required="Ritual protocols essential"
                ),
                'velada_structure': VeladaStructure(
                    timing="Nighttime ceremony",
                    purpose="Healing, divination, problem-solving",
                    guide_role="Curandera/o as intermediary"
                )
            }
        )

        self.native_american_peyote = NativeAmericanPeyote(
            tradition="Native American Church",
            core_concepts={
                'peyote_as_medicine': MedicineConcept(
                    nature="Holy sacrament",
                    relationship="Prayer and communion",
                    healing="Physical, emotional, spiritual"
                ),
                'ceremonial_structure': CeremonialStructure(
                    elements=["Tipi", "Fire", "Altar", "Drum"],
                    roles=["Road Chief", "Drum Chief", "Cedar Chief", "Fire Chief"],
                    integration="Christian and indigenous synthesis"
                )
            }
        )

        self.african_iboga = AfricanIboga(
            tradition="Bwiti (Gabon, Cameroon)",
            core_concepts={
                'iboga_purpose': IbogaPurpose(
                    initiation="Death and rebirth ritual",
                    ancestor_contact="Communication with lineage",
                    life_review="Seeing one's entire life"
                ),
                'cosmological_journey': CosmologicalJourney(
                    destination="Village of the ancestors",
                    purpose="Receive teachings, resolve issues",
                    return="Transformed, with knowledge"
                )
            }
        )

    def map_to_scientific_concepts(
        self,
        tradition: str
    ) -> TraditionalScientificMapping:
        """Map traditional concepts to scientific frameworks."""
        mappings = {
            'plant_teachers': ScientificMapping(
                traditional="Plants as conscious teachers",
                scientific="5-HT2A activation enabling insight",
                integration="Intentional, respectful approach enhances outcomes"
            ),
            'entity_encounters': ScientificMapping(
                traditional="Spirit contact",
                scientific="DMN disruption, pattern completion",
                integration="Phenomenologically real, therapeutically meaningful"
            ),
            'healing_visions': ScientificMapping(
                traditional="Diagnostic visions from spirits",
                scientific="Enhanced interoception, unconscious access",
                integration="Valuable therapeutic content regardless of ontology"
            ),
            'death_rebirth': ScientificMapping(
                traditional="Ego death as spiritual transformation",
                scientific="DMN disruption, prior relaxation",
                integration="Transformative experience central to healing"
            )
        }

        return TraditionalScientificMapping(
            tradition=tradition,
            mappings=mappings,
            integration_principles=[
                "Both frameworks offer valid insights",
                "Scientific mechanism does not negate meaning",
                "Traditional wisdom informs therapeutic application",
                "Respectful integration of perspectives"
            ]
        )
```

## Phenomenological Frameworks

```python
class PhenomenologicalFrameworks:
    """
    Phenomenological approaches to understanding psychedelic consciousness.
    """
    def __init__(self):
        self.grof_cartography = GrofCartography(
            author="Stanislav Grof",
            framework="Cartography of the Psyche",
            levels={
                'sensory_aesthetic': PsycheLevelConcept(
                    name="Sensory/Aesthetic",
                    content="Enhanced perception, synesthesia",
                    neural_correlate="Sensory cortex changes"
                ),
                'biographical_recollective': PsycheLevelConcept(
                    name="Biographical/Recollective",
                    content="Personal memories, emotional processing",
                    neural_correlate="Hippocampal-cortical interaction"
                ),
                'perinatal': PsycheLevelConcept(
                    name="Perinatal",
                    content="Birth process, death-rebirth",
                    neural_correlate="Deep limbic activation",
                    matrices={
                        'bpm_i': "Oceanic unity (pre-birth)",
                        'bpm_ii': "Cosmic engulfment (contractions begin)",
                        'bpm_iii': "Death-rebirth struggle (birth canal)",
                        'bpm_iv': "Death-rebirth (birth completion)"
                    }
                ),
                'transpersonal': PsycheLevelConcept(
                    name="Transpersonal",
                    content="Beyond individual biography",
                    experiences=["Past lives", "Collective unconscious",
                               "Archetypal", "Cosmic consciousness"],
                    neural_correlate="Global brain integration"
                )
            }
        )

        self.stace_mystical_criteria = StaceMysticalCriteria(
            author="Walter Stace",
            criteria=[
                MysticalCriterion(
                    name="Unity",
                    description="Internal and/or external",
                    central=True
                ),
                MysticalCriterion(
                    name="Transcendence of time and space",
                    description="Timelessness, spacelessness",
                    central=True
                ),
                MysticalCriterion(
                    name="Deeply felt positive mood",
                    description="Joy, blessedness, peace",
                    central=True
                ),
                MysticalCriterion(
                    name="Sense of sacredness",
                    description="Holy, divine quality",
                    central=True
                ),
                MysticalCriterion(
                    name="Noetic quality",
                    description="Direct knowing, insight",
                    central=True
                ),
                MysticalCriterion(
                    name="Paradoxicality",
                    description="Transcends logic",
                    secondary=True
                ),
                MysticalCriterion(
                    name="Ineffability",
                    description="Cannot be adequately expressed",
                    secondary=True
                )
            ]
        )

    def assess_experience_depth(
        self,
        experience_report: ExperienceReport
    ) -> ExperienceDepthAssessment:
        """Assess depth of experience using phenomenological frameworks."""
        grof_level = self._assess_grof_level(experience_report)
        mystical_score = self._assess_stace_criteria(experience_report)
        mckenna_level = self._assess_mckenna_level(experience_report)

        return ExperienceDepthAssessment(
            grof_level=grof_level,
            mystical_completeness=mystical_score,
            mckenna_intensity=mckenna_level,
            therapeutic_significance=self._assess_therapeutic_significance(
                grof_level, mystical_score
            ),
            integration_needs=self._determine_integration_needs(
                grof_level, mystical_score
            )
        )
```

## Synthesis and Integration

```python
class TheoreticalSynthesis:
    """
    Integration of multiple theoretical frameworks.
    """
    def __init__(self):
        self.framework_convergence = FrameworkConvergence(
            common_themes=[
                ConvergentTheme(
                    theme="Constraint relaxation",
                    entropic_brain="Increased entropy",
                    rebus="Relaxed priors",
                    gwt="Expanded workspace access",
                    traditional="Boundary dissolution"
                ),
                ConvergentTheme(
                    theme="Novel integration",
                    entropic_brain="New connectivity patterns",
                    rebus="Belief revision",
                    gwt="Unusual information combinations",
                    traditional="Spirit teachings, visions"
                ),
                ConvergentTheme(
                    theme="Therapeutic transformation",
                    entropic_brain="Escape from rigid states",
                    rebus="Maladaptive prior correction",
                    gwt="Access to healing content",
                    traditional="Spiritual healing, renewal"
                )
            ]
        )

        self.integrated_model = IntegratedTheoreticalModel(
            name="Comprehensive Psychedelic Consciousness Model",
            levels=[
                ModelLevel(
                    level="Molecular",
                    description="5-HT2A activation, glutamate cascade",
                    framework_connection="Mechanistic basis for all theories"
                ),
                ModelLevel(
                    level="Network",
                    description="DMN disruption, connectivity changes",
                    framework_connection="Implements entropic increase, prior relaxation"
                ),
                ModelLevel(
                    level="Computational",
                    description="Precision reduction, belief updating",
                    framework_connection="REBUS mechanism"
                ),
                ModelLevel(
                    level="Phenomenological",
                    description="Ego dissolution, mystical experience",
                    framework_connection="Grof/Stace/traditional maps"
                ),
                ModelLevel(
                    level="Therapeutic",
                    description="Belief revision, insight, healing",
                    framework_connection="Clinical application of all levels"
                )
            ]
        )

    def apply_integrated_model(
        self,
        clinical_case: ClinicalCase
    ) -> IntegratedAssessment:
        """Apply integrated theoretical model to clinical case."""
        return IntegratedAssessment(
            molecular_factors=self._assess_molecular(clinical_case),
            network_predictions=self._predict_network_changes(clinical_case),
            rebus_analysis=self._apply_rebus(clinical_case),
            phenomenological_map=self._map_phenomenology(clinical_case),
            therapeutic_recommendations=self._generate_recommendations(clinical_case),
            integration_protocol=self._design_integration(clinical_case)
        )
```

## Conclusion

The theoretical frameworks for psychedelic consciousness converge on several key principles:

1. **Constraint Relaxation**: All frameworks describe reduction of normal cognitive constraints
2. **Novel Integration**: Psychedelics enable new patterns of information processing
3. **Therapeutic Potential**: The alteration of rigid patterns underlies clinical benefit
4. **Multi-Level Understanding**: Complete picture requires molecular through phenomenological levels
5. **Traditional-Scientific Integration**: Indigenous wisdom and neuroscience offer complementary insights

These frameworks guide both research design and clinical application, providing a comprehensive understanding of how psychedelics alter consciousness and why this matters therapeutically.
