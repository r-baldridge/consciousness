# Form 12: Narrative Consciousness - Data Models

## Core Data Models

### 1. Autobiographical Memory Models

```python
@dataclass
class AutobiographicalMemory:
    """
    Comprehensive representation of autobiographical memory.

    Integrates hierarchical organization, thematic indexing,
    and narrative significance for coherent life story construction.
    """
    memory_id: str
    creation_timestamp: float
    last_updated: float

    # Core experience data
    experience: ExperienceData
    emotional_content: EmotionalContent
    sensory_details: SensoryDetails
    contextual_information: ContextualInformation

    # Hierarchical organization
    lifetime_period: LifetimePeriod
    general_event: Optional[GeneralEvent]
    episodic_specifics: EpisodicSpecifics

    # Thematic classification
    life_themes: List[LifeTheme]
    value_themes: List[ValueTheme]
    relationship_themes: List[RelationshipTheme]
    growth_themes: List[GrowthTheme]

    # Temporal organization
    temporal_context: TemporalContext
    temporal_relationships: List[TemporalRelationship]

    # Narrative significance
    story_role: NarrativeRole
    significance_scores: Dict[str, float]
    character_impact: CharacterDevelopmentImpact

@dataclass
class LifetimePeriod:
    """Major life phases with characteristic themes and patterns."""
    period_id: str
    period_name: str
    start_time: float
    end_time: Optional[float]

    # Period characteristics
    dominant_themes: List[str]
    characteristic_goals: List[str]
    typical_relationships: List[str]
    major_achievements: List[str]
    significant_challenges: List[str]

    # Period transitions
    transition_triggers: List[TransitionTrigger]
    transition_narratives: List[TransitionNarrative]

@dataclass
class GeneralEvent:
    """Repeated or extended events that span multiple episodes."""
    event_id: str
    event_type: str
    event_name: str

    # Temporal span
    first_occurrence: float
    last_occurrence: Optional[float]
    typical_frequency: str
    duration_pattern: str

    # Event characteristics
    typical_participants: List[str]
    typical_settings: List[str]
    characteristic_patterns: List[str]
    emotional_patterns: List[EmotionalPattern]

    # Narrative significance
    story_function: str
    thematic_contributions: List[str]

@dataclass
class EpisodicSpecifics:
    """Specific details of individual episodic memory."""
    episode_id: str
    precise_timestamp: float
    duration: float

    # Specific details
    exact_location: Location
    specific_participants: List[Person]
    specific_actions: List[Action]
    specific_dialogue: List[Dialogue]
    unique_details: List[UniqueDetail]

    # Verification information
    confidence_level: float
    source_information: List[SourceInfo]
    corroboration: List[Corroboration]

@dataclass
class ThematicIndex:
    """Cross-cutting thematic organization of memories."""
    theme_id: str
    theme_name: str
    theme_type: str  # life, value, relationship, growth

    # Theme characteristics
    theme_description: str
    emergence_time: float
    evolution_timeline: List[ThemeEvolution]
    current_status: str

    # Associated memories
    core_memories: List[str]  # Memory IDs
    supporting_memories: List[str]
    contradictory_memories: List[str]

    # Theme relationships
    parent_themes: List[str]
    child_themes: List[str]
    related_themes: List[str]

    # Narrative integration
    story_manifestations: List[StoryManifestation]
    character_development_role: str
```

### 2. Narrative Structure Models

```python
@dataclass
class NarrativeStructure:
    """Multi-scale narrative representation."""
    narrative_id: str
    narrative_scale: str  # micro, meso, macro, meta
    narrative_type: str
    creation_timestamp: float
    last_revision: float

    # Story elements
    title: str
    protagonists: List[Character]
    supporting_characters: List[Character]
    antagonists: List[Character]

    setting: Setting
    plot_structure: PlotStructure
    themes: List[NarrativeTheme]

    # Narrative quality
    coherence_scores: CoherenceScores
    authenticity_scores: AuthenticityScores
    emotional_resonance: EmotionalResonance

    # Source information
    source_memories: List[str]
    inspiration_events: List[str]
    construction_process: ConstructionProcess

@dataclass
class Character:
    """Character representation in narrative."""
    character_id: str
    character_name: str
    character_type: str  # self, family, friend, mentor, etc.

    # Character description
    physical_description: PhysicalDescription
    personality_traits: List[PersonalityTrait]
    characteristic_behaviors: List[Behavior]
    speech_patterns: List[SpeechPattern]

    # Character development
    initial_state: CharacterState
    development_arc: List[CharacterDevelopment]
    final_state: CharacterState
    growth_themes: List[str]

    # Relationship dynamics
    relationships: Dict[str, Relationship]
    relationship_evolution: List[RelationshipChange]

@dataclass
class PlotStructure:
    """Narrative plot organization."""
    structure_type: str  # hero's journey, tragedy, comedy, etc.

    # Plot elements
    exposition: Exposition
    inciting_incident: IncitingIncident
    rising_action: List[PlotPoint]
    climax: Climax
    falling_action: List[PlotPoint]
    resolution: Resolution

    # Plot dynamics
    conflicts: List[Conflict]
    turning_points: List[TurningPoint]
    reversals: List[Reversal]
    discoveries: List[Discovery]

    # Thematic elements
    moral_lessons: List[MoralLesson]
    wisdom_gained: List[Wisdom]
    character_insights: List[CharacterInsight]

@dataclass
class Setting:
    """Narrative setting information."""
    temporal_setting: TemporalSetting
    physical_setting: PhysicalSetting
    social_setting: SocialSetting
    cultural_setting: CulturalSetting
    psychological_setting: PsychologicalSetting

    # Setting significance
    thematic_significance: List[str]
    symbolic_meaning: List[str]
    emotional_associations: List[str]

@dataclass
class CoherenceScores:
    """Narrative coherence measurements."""
    overall_coherence: float
    temporal_coherence: float
    causal_coherence: float
    character_coherence: float
    thematic_coherence: float
    emotional_coherence: float

    # Consistency measures
    internal_consistency: float
    external_consistency: float
    cross_narrative_consistency: float

    # Quality indicators
    narrative_flow: float
    logical_progression: float
    believability: float
```

### 3. Temporal Self-Models

```python
@dataclass
class TemporalSelfState:
    """Complete self-state at specific time point."""
    timestamp: float
    self_state_id: str
    confidence_level: float

    # Identity components
    core_identity: CoreIdentityState
    social_identity: SocialIdentityState
    professional_identity: ProfessionalIdentityState
    personal_identity: PersonalIdentityState

    # Psychological state
    values: List[ValueState]
    beliefs: List[BeliefState]
    attitudes: List[AttitudeState]
    goals: List[GoalState]
    motivations: List[MotivationState]

    # Capabilities and skills
    cognitive_abilities: Dict[str, AbilityLevel]
    physical_abilities: Dict[str, AbilityLevel]
    social_skills: Dict[str, SkillLevel]
    professional_skills: Dict[str, SkillLevel]
    creative_abilities: Dict[str, AbilityLevel]

    # Relationships
    family_relationships: Dict[str, RelationshipState]
    friend_relationships: Dict[str, RelationshipState]
    professional_relationships: Dict[str, RelationshipState]
    romantic_relationships: Dict[str, RelationshipState]

    # Life context
    life_circumstances: LifeCircumstances
    environmental_context: EnvironmentalContext
    health_status: HealthStatus

@dataclass
class SelfContinuityMap:
    """Mapping of self-continuity across time."""
    continuity_id: str
    time_span: TimeSpan

    # Identity threads
    persistent_identity_elements: List[PersistentElement]
    evolving_identity_elements: List[EvolvingElement]
    discontinued_elements: List[DiscontinuedElement]
    emerged_elements: List[EmergedElement]

    # Continuity measures
    overall_continuity_score: float
    identity_stability_score: float
    growth_integration_score: float
    change_coherence_score: float

    # Change analysis
    major_transitions: List[IdentityTransition]
    gradual_changes: List[GradualChange]
    crisis_points: List[CrisisPoint]
    breakthrough_moments: List[BreakthroughMoment]

@dataclass
class FutureSelfProjection:
    """Projection of future self-states."""
    projection_id: str
    projection_timestamp: float
    time_horizon: float
    confidence_level: float

    # Projected states
    most_likely_self: TemporalSelfState
    best_case_self: TemporalSelfState
    worst_case_self: TemporalSelfState
    alternative_selves: List[AlternativeSelf]

    # Projection basis
    current_trajectory: Trajectory
    influencing_factors: List[InfluencingFactor]
    assumptions: List[Assumption]
    uncertainties: List[Uncertainty]

    # Goal alignment
    goal_achievement_likelihood: Dict[str, float]
    value_fulfillment_potential: Dict[str, float]
    life_satisfaction_projection: float

@dataclass
class IdentityTransition:
    """Major identity transition event."""
    transition_id: str
    transition_name: str
    start_time: float
    end_time: Optional[float]
    transition_type: str

    # Transition characteristics
    triggering_events: List[str]
    transition_process: List[TransitionPhase]
    challenges_faced: List[Challenge]
    resources_utilized: List[Resource]

    # Identity changes
    identity_before: TemporalSelfState
    identity_after: TemporalSelfState
    change_dimensions: List[ChangeDimension]

    # Narrative significance
    transition_story: NarrativeStructure
    lessons_learned: List[str]
    wisdom_gained: List[str]
```

### 4. Meaning-Making Models

```python
@dataclass
class MeaningMakingResult:
    """Result of meaning attribution process."""
    meaning_id: str
    creation_timestamp: float
    experience_reference: str

    # Multi-dimensional significance
    significance_analysis: SignificanceAnalysis
    personal_meaning: PersonalMeaning
    relational_meaning: RelationalMeaning
    existential_meaning: ExistentialMeaning

    # Life theme integration
    theme_connections: List[ThemeConnection]
    theme_evolution: List[ThemeEvolution]
    new_themes_emerged: List[NewTheme]

    # Growth integration
    growth_dimensions: List[GrowthDimension]
    learning_outcomes: List[LearningOutcome]
    character_development: CharacterDevelopment

    # Purpose alignment
    purpose_relevance: float
    value_alignment: Dict[str, float]
    goal_contribution: Dict[str, float]

@dataclass
class SignificanceAnalysis:
    """Multi-dimensional significance assessment."""
    overall_significance: float
    confidence_level: float

    # Dimension scores
    personal_growth_significance: float
    relationship_significance: float
    achievement_significance: float
    learning_significance: float
    emotional_significance: float
    spiritual_significance: float

    # Temporal significance
    immediate_significance: float
    short_term_significance: float
    long_term_significance: float

    # Context factors
    life_context_factors: List[ContextFactor]
    situational_factors: List[SituationalFactor]
    cultural_factors: List[CulturalFactor]

@dataclass
class LifeThemeTracker:
    """Tracking of life themes over time."""
    theme_id: str
    theme_name: str
    theme_category: str

    # Theme development
    emergence_point: EmergencePoint
    development_timeline: List[ThemeDevelopment]
    current_status: ThemeStatus
    projected_evolution: List[ProjectedEvolution]

    # Theme manifestations
    memory_manifestations: List[str]
    narrative_manifestations: List[str]
    behavioral_manifestations: List[str]
    goal_manifestations: List[str]

    # Theme relationships
    parent_themes: List[str]
    child_themes: List[str]
    conflicting_themes: List[str]
    supporting_themes: List[str]

    # Theme dynamics
    strength_over_time: List[ThemeStrength]
    coherence_score: float
    integration_quality: float

@dataclass
class GrowthIntegration:
    """Integration of growth experiences into life narrative."""
    integration_id: str
    growth_event: str
    integration_timestamp: float

    # Growth characteristics
    growth_type: str  # cognitive, emotional, social, spiritual
    growth_magnitude: float
    growth_duration: float
    growth_permanence: float

    # Integration process
    initial_impact: ImpactAssessment
    processing_phases: List[ProcessingPhase]
    integration_challenges: List[IntegrationChallenge]
    integration_supports: List[IntegrationSupport]

    # Narrative integration
    story_incorporation: StoryIncorporation
    character_development_impact: CharacterImpact
    theme_updates: List[ThemeUpdate]

    # Long-term effects
    behavior_changes: List[BehaviorChange]
    perspective_shifts: List[PerspectiveShift]
    relationship_impacts: List[RelationshipImpact]
```

### 5. Integration and Communication Models

```python
@dataclass
class ConsciousnessIntegrationState:
    """State of integration with other consciousness forms."""
    integration_timestamp: float

    # Self-recognition integration
    identity_narrative_alignment: float
    self_model_consistency: float
    boundary_narrative_coherence: float

    # Meta-consciousness integration
    narrative_self_awareness: float
    story_construction_awareness: float
    meta_narrative_capabilities: float

    # Intentional consciousness integration
    goal_narrative_alignment: float
    aspiration_story_integration: float
    purpose_coherence: float

    # Memory system integration
    episodic_narrative_integration: float
    semantic_story_coherence: float
    procedural_narrative_consistency: float

    # Emotional system integration
    affective_narrative_authenticity: float
    emotional_story_coherence: float
    feeling_meaning_integration: float

@dataclass
class NarrativeCommunication:
    """Communication of narratives to other systems or agents."""
    communication_id: str
    timestamp: float
    recipient: str
    communication_type: str

    # Narrative content
    shared_narratives: List[NarrativeStructure]
    narrative_summaries: List[NarrativeSummary]
    key_themes: List[str]
    significant_insights: List[str]

    # Communication context
    purpose: str
    audience_adaptation: AudienceAdaptation
    privacy_controls: PrivacyControls
    cultural_adaptation: CulturalAdaptation

    # Feedback and response
    recipient_response: Optional[Response]
    understanding_verification: UnderstandingVerification
    narrative_impact: NarrativeImpact

@dataclass
class NarrativeUpdate:
    """Update to existing narrative structures."""
    update_id: str
    update_timestamp: float
    narrative_id: str
    update_type: str  # revision, extension, correction, integration

    # Update content
    changes_made: List[Change]
    additions: List[Addition]
    modifications: List[Modification]
    deletions: List[Deletion]

    # Update justification
    triggering_events: List[str]
    new_information: List[str]
    consistency_requirements: List[str]
    coherence_improvements: List[str]

    # Update validation
    coherence_check: CoherenceCheck
    consistency_verification: ConsistencyVerification
    authenticity_assessment: AuthenticityAssessment

@dataclass
class NarrativeMetrics:
    """Comprehensive metrics for narrative quality assessment."""
    metrics_timestamp: float

    # Quality metrics
    coherence_metrics: CoherenceMetrics
    authenticity_metrics: AuthenticityMetrics
    completeness_metrics: CompletenessMetrics
    consistency_metrics: ConsistencyMetrics

    # Performance metrics
    construction_speed: float
    retrieval_efficiency: float
    update_responsiveness: float
    integration_effectiveness: float

    # Psychological metrics
    identity_support: float
    meaning_provision: float
    growth_facilitation: float
    resilience_contribution: float

    # System metrics
    memory_utilization: float
    computational_efficiency: float
    storage_optimization: float
    scalability_indicators: ScalabilityIndicators
```

### 6. Privacy and Security Models

```python
@dataclass
class NarrativePrivacyControls:
    """Privacy controls for autobiographical narratives."""
    privacy_id: str
    owner_id: str
    creation_timestamp: float

    # Content classification
    public_narratives: List[str]
    private_narratives: List[str]
    confidential_narratives: List[str]
    restricted_narratives: List[str]

    # Access controls
    sharing_permissions: Dict[str, SharingPermission]
    viewing_restrictions: Dict[str, ViewingRestriction]
    modification_controls: Dict[str, ModificationControl]

    # Anonymization options
    anonymization_levels: Dict[str, AnonymizationLevel]
    pseudonym_mappings: Dict[str, str]
    sanitization_rules: List[SanitizationRule]

@dataclass
class NarrativeIntegrity:
    """Integrity verification for narrative authenticity."""
    integrity_id: str
    verification_timestamp: float

    # Source verification
    memory_source_verification: List[SourceVerification]
    corroboration_evidence: List[CorroborationEvidence]
    consistency_checks: List[ConsistencyCheck]

    # Authenticity indicators
    emotional_authenticity: float
    behavioral_consistency: float
    temporal_plausibility: float
    causal_coherence: float

    # Integrity threats
    detected_inconsistencies: List[Inconsistency]
    potential_fabrications: List[FabricationAlert]
    memory_distortions: List[DistortionIndicator]
```

These comprehensive data models provide the structured foundation needed to represent all aspects of narrative consciousness, from individual autobiographical memories to complex multi-scale life stories, while supporting integration with other consciousness forms and maintaining privacy, security, and authenticity requirements.