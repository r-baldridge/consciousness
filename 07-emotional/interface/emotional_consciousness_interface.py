#!/usr/bin/env python3
"""
Emotional Consciousness Interface

Form 07: Emotional Consciousness processes affective states, moods,
and emotional experiences. It handles emotion detection, appraisal
processing, emotion regulation, and mood tracking. Emotional consciousness
provides the valence and arousal dimensions that color all conscious
experience, linking bodily feelings with cognitive evaluations.

This form works closely with Form 06 (Interoceptive) for bodily signals
and Form 08 (Arousal) for activation levels.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EmotionCategory(Enum):
    """Primary emotion categories based on Plutchik's wheel and basic emotions."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    CONTEMPT = "contempt"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class EmotionalValence(Enum):
    """Valence dimension of emotional experience."""
    VERY_NEGATIVE = "very_negative"      # -1.0 to -0.6
    NEGATIVE = "negative"                # -0.6 to -0.2
    NEUTRAL = "neutral"                  # -0.2 to 0.2
    POSITIVE = "positive"                # 0.2 to 0.6
    VERY_POSITIVE = "very_positive"      # 0.6 to 1.0


class AffectiveState(Enum):
    """Core affect states combining valence and arousal."""
    EXCITED = "excited"          # positive valence, high arousal
    HAPPY = "happy"              # positive valence, moderate arousal
    CONTENT = "content"          # positive valence, low arousal
    RELAXED = "relaxed"          # neutral valence, low arousal
    BORED = "bored"              # negative valence, low arousal
    SAD = "sad"                  # negative valence, moderate arousal
    DISTRESSED = "distressed"    # negative valence, high arousal
    ALERT = "alert"              # neutral valence, high arousal


class MoodState(Enum):
    """Longer-duration mood states."""
    ELEVATED = "elevated"
    EUTHYMIC = "euthymic"       # Normal baseline mood
    IRRITABLE = "irritable"
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    APATHETIC = "apathetic"
    EUPHORIC = "euphoric"


class EmotionalRegulationStrategy(Enum):
    """Strategies for emotional regulation."""
    REAPPRAISAL = "reappraisal"               # Cognitive reinterpretation
    SUPPRESSION = "suppression"               # Expressive suppression
    DISTRACTION = "distraction"               # Attentional deployment
    ACCEPTANCE = "acceptance"                 # Mindful acceptance
    PROBLEM_SOLVING = "problem_solving"       # Situation modification
    SITUATION_SELECTION = "situation_selection"  # Avoidance/approach


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class EmotionalStimulus:
    """An incoming stimulus to be emotionally appraised."""
    stimulus_id: str
    stimulus_type: str           # "event", "memory", "thought", "percept", "social"
    content_description: str
    intensity: float             # 0.0-1.0
    novelty: float               # 0.0-1.0
    personal_relevance: float    # 0.0-1.0
    source: str = "external"     # "external", "internal", "memory"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "stimulus_type": self.stimulus_type,
            "content_description": self.content_description,
            "intensity": self.intensity,
            "novelty": self.novelty,
            "personal_relevance": self.personal_relevance,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AppraisalInput:
    """Cognitive appraisal of an emotional stimulus."""
    stimulus_id: str
    goal_relevance: float        # 0.0-1.0
    goal_congruence: float       # -1.0 to 1.0 (hinders to helps)
    coping_potential: float      # 0.0-1.0
    norm_compatibility: float    # 0.0-1.0
    certainty: float             # 0.0-1.0 (predictability)
    agency: str = "self"         # "self", "other", "circumstance"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BodilySignalInput:
    """Interoceptive/bodily signals contributing to emotional experience."""
    heart_rate_change: float     # -1.0 to 1.0 (deceleration to acceleration)
    skin_conductance: float      # 0.0-1.0
    muscle_tension: float        # 0.0-1.0
    breathing_rate_change: float # -1.0 to 1.0
    gut_feeling: float           # -1.0 to 1.0
    temperature_change: float    # -1.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EmotionalInput:
    """Complete input for emotional processing."""
    stimulus: Optional[EmotionalStimulus] = None
    appraisal: Optional[AppraisalInput] = None
    bodily_signals: Optional[BodilySignalInput] = None
    social_context: Optional[str] = None     # "alone", "group", "intimate", "public"
    current_goals: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class EmotionIdentification:
    """Identified emotion with confidence."""
    category: EmotionCategory
    intensity: float             # 0.0-1.0
    confidence: float            # 0.0-1.0
    secondary_emotions: List[Tuple[EmotionCategory, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "intensity": round(self.intensity, 4),
            "confidence": round(self.confidence, 4),
            "secondary_emotions": [
                {"category": e.value, "intensity": round(i, 4)}
                for e, i in self.secondary_emotions
            ],
        }


@dataclass
class EmotionalOutput:
    """Complete output from emotional processing."""
    emotion: EmotionIdentification
    valence: float                           # -1.0 to 1.0
    valence_category: EmotionalValence
    arousal: float                           # 0.0-1.0
    affective_state: AffectiveState
    action_tendency: str                     # "approach", "avoid", "freeze", "attend"
    regulation_suggestion: Optional[EmotionalRegulationStrategy] = None
    appraisal_summary: Optional[str] = None
    confidence: float = 0.8
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "valence": round(self.valence, 4),
            "valence_category": self.valence_category.value,
            "arousal": round(self.arousal, 4),
            "affective_state": self.affective_state.value,
            "action_tendency": self.action_tendency,
            "regulation_suggestion": self.regulation_suggestion.value if self.regulation_suggestion else None,
            "appraisal_summary": self.appraisal_summary,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MoodReport:
    """Current mood state report."""
    mood_state: MoodState
    stability: float             # 0.0-1.0 (volatile to stable)
    duration_minutes: float
    dominant_valence: float      # -1.0 to 1.0
    contributing_emotions: List[EmotionCategory] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mood_state": self.mood_state.value,
            "stability": round(self.stability, 4),
            "duration_minutes": round(self.duration_minutes, 2),
            "dominant_valence": round(self.dominant_valence, 4),
            "contributing_emotions": [e.value for e in self.contributing_emotions],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmotionalSystemStatus:
    """Complete emotional system status."""
    current_emotion: Optional[EmotionIdentification]
    current_mood: MoodReport
    emotional_complexity: float      # 0.0-1.0 (mixed emotions)
    regulation_active: bool
    regulation_strategy: Optional[EmotionalRegulationStrategy]
    system_health: float             # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# EMOTION PROCESSING ENGINE
# ============================================================================

class EmotionProcessingEngine:
    """
    Core engine for emotion detection and appraisal processing.

    Implements appraisal theory of emotion: stimuli are evaluated
    against goals and coping resources to produce emotional responses.
    """

    # Appraisal-to-emotion mappings
    APPRAISAL_PATTERNS = {
        EmotionCategory.JOY: {"goal_congruence_min": 0.3, "certainty_min": 0.4},
        EmotionCategory.SADNESS: {"goal_congruence_max": -0.3, "coping_max": 0.4},
        EmotionCategory.ANGER: {"goal_congruence_max": -0.2, "agency": "other"},
        EmotionCategory.FEAR: {"goal_congruence_max": -0.1, "certainty_max": 0.4, "coping_max": 0.5},
        EmotionCategory.DISGUST: {"norm_compatibility_max": 0.3},
        EmotionCategory.SURPRISE: {"certainty_max": 0.3},
        EmotionCategory.TRUST: {"goal_congruence_min": 0.2, "certainty_min": 0.5},
        EmotionCategory.ANTICIPATION: {"goal_relevance_min": 0.5, "certainty_max": 0.5},
    }

    EMOTION_VALENCE = {
        EmotionCategory.JOY: 0.8,
        EmotionCategory.SADNESS: -0.7,
        EmotionCategory.ANGER: -0.6,
        EmotionCategory.FEAR: -0.8,
        EmotionCategory.DISGUST: -0.6,
        EmotionCategory.SURPRISE: 0.0,
        EmotionCategory.CONTEMPT: -0.4,
        EmotionCategory.TRUST: 0.5,
        EmotionCategory.ANTICIPATION: 0.3,
    }

    EMOTION_AROUSAL = {
        EmotionCategory.JOY: 0.7,
        EmotionCategory.SADNESS: 0.3,
        EmotionCategory.ANGER: 0.8,
        EmotionCategory.FEAR: 0.9,
        EmotionCategory.DISGUST: 0.5,
        EmotionCategory.SURPRISE: 0.8,
        EmotionCategory.CONTEMPT: 0.3,
        EmotionCategory.TRUST: 0.4,
        EmotionCategory.ANTICIPATION: 0.6,
    }

    ACTION_TENDENCIES = {
        EmotionCategory.JOY: "approach",
        EmotionCategory.SADNESS: "withdraw",
        EmotionCategory.ANGER: "approach",
        EmotionCategory.FEAR: "avoid",
        EmotionCategory.DISGUST: "avoid",
        EmotionCategory.SURPRISE: "attend",
        EmotionCategory.CONTEMPT: "avoid",
        EmotionCategory.TRUST: "approach",
        EmotionCategory.ANTICIPATION: "attend",
    }

    def __init__(self):
        self._emotion_history: List[EmotionIdentification] = []
        self._max_history = 50

    def detect_emotion(self, emotional_input: EmotionalInput) -> EmotionIdentification:
        """Detect the primary emotion from input signals."""
        scores: Dict[EmotionCategory, float] = {}

        # Appraisal-based detection
        if emotional_input.appraisal:
            appraisal = emotional_input.appraisal
            scores = self._appraisal_scores(appraisal)

        # Bodily signal contribution
        if emotional_input.bodily_signals:
            body_scores = self._body_signal_scores(emotional_input.bodily_signals)
            for cat, score in body_scores.items():
                scores[cat] = scores.get(cat, 0.0) + score * 0.3

        # Stimulus-based fallback
        if emotional_input.stimulus and not scores:
            scores = self._stimulus_scores(emotional_input.stimulus)

        # Default if no signals
        if not scores:
            return EmotionIdentification(
                category=EmotionCategory.SURPRISE,
                intensity=0.3,
                confidence=0.4,
            )

        # Find primary emotion
        primary_cat = max(scores, key=scores.get)
        primary_score = min(1.0, max(0.0, scores[primary_cat]))

        # Find secondary emotions
        secondary = [
            (cat, min(1.0, max(0.0, score)))
            for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if cat != primary_cat and score > 0.2
        ][:3]

        confidence = min(1.0, primary_score / max(0.01, sum(scores.values())) + 0.3)

        identification = EmotionIdentification(
            category=primary_cat,
            intensity=primary_score,
            confidence=confidence,
            secondary_emotions=secondary,
        )

        self._emotion_history.append(identification)
        if len(self._emotion_history) > self._max_history:
            self._emotion_history.pop(0)

        return identification

    def compute_valence(self, emotion: EmotionIdentification) -> float:
        """Compute overall emotional valence."""
        base_valence = self.EMOTION_VALENCE.get(emotion.category, 0.0)
        return max(-1.0, min(1.0, base_valence * emotion.intensity))

    def compute_arousal(self, emotion: EmotionIdentification) -> float:
        """Compute emotional arousal level."""
        base_arousal = self.EMOTION_AROUSAL.get(emotion.category, 0.5)
        return max(0.0, min(1.0, base_arousal * emotion.intensity))

    def get_action_tendency(self, emotion: EmotionIdentification) -> str:
        """Get the action tendency for the identified emotion."""
        return self.ACTION_TENDENCIES.get(emotion.category, "attend")

    def classify_valence(self, valence: float) -> EmotionalValence:
        """Classify a valence value into a category."""
        if valence < -0.6:
            return EmotionalValence.VERY_NEGATIVE
        elif valence < -0.2:
            return EmotionalValence.NEGATIVE
        elif valence < 0.2:
            return EmotionalValence.NEUTRAL
        elif valence < 0.6:
            return EmotionalValence.POSITIVE
        else:
            return EmotionalValence.VERY_POSITIVE

    def classify_affect(self, valence: float, arousal: float) -> AffectiveState:
        """Classify core affect from valence and arousal dimensions."""
        if arousal > 0.6:
            if valence > 0.2:
                return AffectiveState.EXCITED
            elif valence < -0.2:
                return AffectiveState.DISTRESSED
            else:
                return AffectiveState.ALERT
        elif arousal < 0.4:
            if valence > 0.2:
                return AffectiveState.CONTENT
            elif valence < -0.2:
                return AffectiveState.BORED
            else:
                return AffectiveState.RELAXED
        else:
            if valence > 0.2:
                return AffectiveState.HAPPY
            else:
                return AffectiveState.SAD

    def _appraisal_scores(self, appraisal: AppraisalInput) -> Dict[EmotionCategory, float]:
        """Score emotions based on appraisal dimensions."""
        scores = {}
        gc = appraisal.goal_congruence
        cp = appraisal.coping_potential
        cert = appraisal.certainty
        gr = appraisal.goal_relevance
        nc = appraisal.norm_compatibility

        # Joy: goal-congruent, certain
        scores[EmotionCategory.JOY] = max(0.0, gc * 0.6 + cert * 0.2 + gr * 0.2)
        # Sadness: goal-incongruent, low coping
        scores[EmotionCategory.SADNESS] = max(0.0, -gc * 0.5 + (1 - cp) * 0.3 + gr * 0.2)
        # Anger: goal-incongruent, other-caused
        anger_agency = 0.3 if appraisal.agency == "other" else 0.0
        scores[EmotionCategory.ANGER] = max(0.0, -gc * 0.4 + cp * 0.2 + anger_agency)
        # Fear: uncertain, low coping, goal-relevant threat
        scores[EmotionCategory.FEAR] = max(0.0, -gc * 0.3 + (1 - cert) * 0.3 + (1 - cp) * 0.3 + gr * 0.1)
        # Disgust: norm violation
        scores[EmotionCategory.DISGUST] = max(0.0, (1 - nc) * 0.7 + gr * 0.3)
        # Surprise: low certainty
        scores[EmotionCategory.SURPRISE] = max(0.0, (1 - cert) * 0.7 + gr * 0.3)
        # Trust: congruent, certain
        scores[EmotionCategory.TRUST] = max(0.0, gc * 0.4 + cert * 0.4 + nc * 0.2)
        # Anticipation: relevant, uncertain
        scores[EmotionCategory.ANTICIPATION] = max(0.0, gr * 0.5 + (1 - cert) * 0.3 + gc * 0.2)

        return scores

    def _body_signal_scores(self, signals: BodilySignalInput) -> Dict[EmotionCategory, float]:
        """Score emotions based on bodily signals."""
        scores = {}
        scores[EmotionCategory.FEAR] = max(0.0, signals.heart_rate_change * 0.4 + signals.skin_conductance * 0.3 + signals.muscle_tension * 0.3)
        scores[EmotionCategory.ANGER] = max(0.0, signals.heart_rate_change * 0.3 + signals.muscle_tension * 0.4 + signals.temperature_change * 0.3)
        scores[EmotionCategory.SADNESS] = max(0.0, -signals.heart_rate_change * 0.3 + signals.gut_feeling * -0.3)
        scores[EmotionCategory.DISGUST] = max(0.0, -signals.gut_feeling * 0.5 + signals.skin_conductance * 0.2)
        scores[EmotionCategory.JOY] = max(0.0, -signals.muscle_tension * 0.3 + signals.temperature_change * 0.3)
        return scores

    def _stimulus_scores(self, stimulus: EmotionalStimulus) -> Dict[EmotionCategory, float]:
        """Generate basic scores from stimulus properties alone."""
        base = stimulus.intensity * stimulus.personal_relevance
        scores = {}
        if stimulus.novelty > 0.6:
            scores[EmotionCategory.SURPRISE] = base * 0.7
        scores[EmotionCategory.ANTICIPATION] = base * stimulus.novelty * 0.5
        scores[EmotionCategory.TRUST] = base * (1 - stimulus.novelty) * 0.4
        return scores


# ============================================================================
# MOOD TRACKING ENGINE
# ============================================================================

class MoodTrackingEngine:
    """
    Tracks mood state over time based on accumulated emotional experiences.

    Mood is a longer-duration affective state influenced by the
    pattern of recent emotions rather than a single event.
    """

    def __init__(self):
        self._current_mood = MoodState.EUTHYMIC
        self._mood_start = datetime.now(timezone.utc)
        self._valence_accumulator: List[float] = []
        self._max_accumulator = 100

    def update_mood(self, emotion: EmotionIdentification, valence: float) -> MoodReport:
        """Update mood based on new emotional experience."""
        self._valence_accumulator.append(valence)
        if len(self._valence_accumulator) > self._max_accumulator:
            self._valence_accumulator.pop(0)

        avg_valence = sum(self._valence_accumulator) / len(self._valence_accumulator)
        stability = self._compute_stability()
        new_mood = self._classify_mood(avg_valence, stability, emotion)

        if new_mood != self._current_mood:
            self._current_mood = new_mood
            self._mood_start = datetime.now(timezone.utc)

        elapsed = (datetime.now(timezone.utc) - self._mood_start).total_seconds() / 60.0

        recent_cats = []
        return MoodReport(
            mood_state=self._current_mood,
            stability=stability,
            duration_minutes=elapsed,
            dominant_valence=avg_valence,
            contributing_emotions=recent_cats,
        )

    def get_current_mood(self) -> MoodReport:
        """Get the current mood state."""
        avg_valence = 0.0
        if self._valence_accumulator:
            avg_valence = sum(self._valence_accumulator) / len(self._valence_accumulator)
        elapsed = (datetime.now(timezone.utc) - self._mood_start).total_seconds() / 60.0

        return MoodReport(
            mood_state=self._current_mood,
            stability=self._compute_stability(),
            duration_minutes=elapsed,
            dominant_valence=avg_valence,
        )

    def _compute_stability(self) -> float:
        """Compute mood stability from valence history."""
        if len(self._valence_accumulator) < 3:
            return 1.0
        recent = self._valence_accumulator[-20:]
        mean = sum(recent) / len(recent)
        variance = sum((v - mean) ** 2 for v in recent) / len(recent)
        return max(0.0, 1.0 - variance * 4)

    def _classify_mood(
        self, avg_valence: float, stability: float, emotion: EmotionIdentification
    ) -> MoodState:
        """Classify mood from accumulated signals."""
        if avg_valence > 0.6:
            return MoodState.EUPHORIC
        elif avg_valence > 0.2:
            return MoodState.ELEVATED
        elif avg_valence < -0.5:
            return MoodState.DEPRESSED
        elif avg_valence < -0.2 and emotion.category == EmotionCategory.FEAR:
            return MoodState.ANXIOUS
        elif avg_valence < -0.2 and emotion.category == EmotionCategory.ANGER:
            return MoodState.IRRITABLE
        elif stability < 0.3:
            return MoodState.IRRITABLE
        else:
            return MoodState.EUTHYMIC


# ============================================================================
# EMOTION REGULATION ENGINE
# ============================================================================

class EmotionRegulationEngine:
    """
    Suggests and applies emotional regulation strategies.

    Based on Gross's process model of emotion regulation.
    """

    STRATEGY_EFFECTIVENESS = {
        EmotionalRegulationStrategy.REAPPRAISAL: 0.8,
        EmotionalRegulationStrategy.ACCEPTANCE: 0.7,
        EmotionalRegulationStrategy.PROBLEM_SOLVING: 0.75,
        EmotionalRegulationStrategy.DISTRACTION: 0.5,
        EmotionalRegulationStrategy.SUPPRESSION: 0.3,
        EmotionalRegulationStrategy.SITUATION_SELECTION: 0.6,
    }

    def suggest_strategy(
        self, emotion: EmotionIdentification, context: Optional[str] = None
    ) -> Optional[EmotionalRegulationStrategy]:
        """Suggest a regulation strategy if needed."""
        if emotion.intensity < 0.5:
            return None

        if emotion.category in (EmotionCategory.FEAR, EmotionCategory.ANGER):
            return EmotionalRegulationStrategy.REAPPRAISAL
        elif emotion.category == EmotionCategory.SADNESS:
            return EmotionalRegulationStrategy.ACCEPTANCE
        elif emotion.category == EmotionCategory.DISGUST:
            return EmotionalRegulationStrategy.DISTRACTION
        elif emotion.intensity > 0.8:
            return EmotionalRegulationStrategy.REAPPRAISAL
        return None

    def apply_regulation(
        self, emotion: EmotionIdentification, strategy: EmotionalRegulationStrategy
    ) -> EmotionIdentification:
        """Apply a regulation strategy to reduce emotional intensity."""
        effectiveness = self.STRATEGY_EFFECTIVENESS.get(strategy, 0.5)
        regulated_intensity = emotion.intensity * (1.0 - effectiveness * 0.5)
        return EmotionIdentification(
            category=emotion.category,
            intensity=max(0.1, regulated_intensity),
            confidence=emotion.confidence,
            secondary_emotions=emotion.secondary_emotions,
        )


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class EmotionalConsciousnessInterface:
    """
    Main interface for Form 07: Emotional Consciousness.

    Processes emotional stimuli through appraisal, detects emotions,
    tracks mood, and manages emotion regulation.
    """

    FORM_ID = "07-emotional"
    FORM_NAME = "Emotional Consciousness"

    def __init__(self):
        """Initialize the emotional consciousness interface."""
        self.processing_engine = EmotionProcessingEngine()
        self.mood_engine = MoodTrackingEngine()
        self.regulation_engine = EmotionRegulationEngine()

        self._current_emotion: Optional[EmotionIdentification] = None
        self._current_output: Optional[EmotionalOutput] = None
        self._regulation_active: bool = False
        self._active_strategy: Optional[EmotionalRegulationStrategy] = None
        self._initialized: bool = False

        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the emotional consciousness system."""
        self._initialized = True
        logger.info(f"{self.FORM_NAME} initialized")

    async def process_emotion(self, emotional_input: EmotionalInput) -> EmotionalOutput:
        """
        Process emotional input and generate emotional output.

        This is the main entry point for emotional processing.
        """
        # Detect emotion
        emotion = self.processing_engine.detect_emotion(emotional_input)
        self._current_emotion = emotion

        # Compute valence and arousal
        valence = self.processing_engine.compute_valence(emotion)
        arousal = self.processing_engine.compute_arousal(emotion)
        valence_category = self.processing_engine.classify_valence(valence)
        affective_state = self.processing_engine.classify_affect(valence, arousal)
        action_tendency = self.processing_engine.get_action_tendency(emotion)

        # Check regulation
        regulation = self.regulation_engine.suggest_strategy(emotion)
        if regulation and emotion.intensity > 0.7:
            self._regulation_active = True
            self._active_strategy = regulation
            emotion = self.regulation_engine.apply_regulation(emotion, regulation)
            self._current_emotion = emotion
            valence = self.processing_engine.compute_valence(emotion)
            arousal = self.processing_engine.compute_arousal(emotion)
        else:
            self._regulation_active = False
            self._active_strategy = None

        # Update mood
        self.mood_engine.update_mood(emotion, valence)

        # Build output
        output = EmotionalOutput(
            emotion=emotion,
            valence=valence,
            valence_category=valence_category,
            arousal=arousal,
            affective_state=affective_state,
            action_tendency=action_tendency,
            regulation_suggestion=regulation,
            confidence=emotion.confidence,
        )
        self._current_output = output
        return output

    async def regulate_emotion(
        self, strategy: EmotionalRegulationStrategy
    ) -> Optional[EmotionIdentification]:
        """Manually apply an emotion regulation strategy."""
        if not self._current_emotion:
            return None
        self._regulation_active = True
        self._active_strategy = strategy
        regulated = self.regulation_engine.apply_regulation(self._current_emotion, strategy)
        self._current_emotion = regulated
        return regulated

    def get_current_emotion(self) -> Optional[EmotionIdentification]:
        """Get the current identified emotion."""
        return self._current_emotion

    def get_current_mood(self) -> MoodReport:
        """Get the current mood report."""
        return self.mood_engine.get_current_mood()

    def get_status(self) -> EmotionalSystemStatus:
        """Get complete emotional system status."""
        mood = self.mood_engine.get_current_mood()
        complexity = 0.0
        if self._current_emotion and self._current_emotion.secondary_emotions:
            complexity = min(1.0, len(self._current_emotion.secondary_emotions) * 0.3)

        return EmotionalSystemStatus(
            current_emotion=self._current_emotion,
            current_mood=mood,
            emotional_complexity=complexity,
            regulation_active=self._regulation_active,
            regulation_strategy=self._active_strategy,
            system_health=self._compute_health(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "current_emotion": self._current_emotion.to_dict() if self._current_emotion else None,
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "current_mood": self.mood_engine.get_current_mood().to_dict(),
            "regulation_active": self._regulation_active,
            "active_strategy": self._active_strategy.value if self._active_strategy else None,
            "initialized": self._initialized,
        }

    def _compute_health(self) -> float:
        """Compute emotional system health."""
        mood = self.mood_engine.get_current_mood()
        stability = mood.stability
        if self._current_emotion:
            intensity_penalty = max(0.0, self._current_emotion.intensity - 0.8) * 2
            return max(0.0, min(1.0, stability - intensity_penalty))
        return stability


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_emotional_interface() -> EmotionalConsciousnessInterface:
    """Create and return an emotional consciousness interface."""
    return EmotionalConsciousnessInterface()


__all__ = [
    # Enums
    "EmotionCategory",
    "EmotionalValence",
    "AffectiveState",
    "MoodState",
    "EmotionalRegulationStrategy",
    # Input dataclasses
    "EmotionalStimulus",
    "AppraisalInput",
    "BodilySignalInput",
    "EmotionalInput",
    # Output dataclasses
    "EmotionIdentification",
    "EmotionalOutput",
    "MoodReport",
    "EmotionalSystemStatus",
    # Engines
    "EmotionProcessingEngine",
    "MoodTrackingEngine",
    "EmotionRegulationEngine",
    # Main interface
    "EmotionalConsciousnessInterface",
    # Convenience
    "create_emotional_interface",
]
