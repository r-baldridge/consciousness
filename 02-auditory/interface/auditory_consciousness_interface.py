#!/usr/bin/env python3
"""
Auditory Consciousness Interface

Form 02: The auditory processing system for consciousness.
Auditory consciousness processes sound streams through spectral analysis,
sound segregation, spatial localization, speech recognition, and musical
perception to construct coherent auditory experience.

This form handles the transformation of auditory stimuli into conscious
perception of sounds, voices, music, and environmental audio.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SoundCategory(Enum):
    """Categories of sounds recognized by auditory consciousness."""
    SPEECH = "speech"
    MUSIC = "music"
    ENVIRONMENTAL = "environmental"
    ALARM = "alarm"
    ANIMAL = "animal"
    MECHANICAL = "mechanical"
    NATURE = "nature"
    SILENCE = "silence"
    NOISE = "noise"
    UNKNOWN = "unknown"


class FrequencyBand(Enum):
    """Frequency bands for spectral analysis."""
    SUB_BASS = "sub_bass"          # 20-60 Hz
    BASS = "bass"                  # 60-250 Hz
    LOW_MID = "low_mid"            # 250-500 Hz
    MID = "mid"                    # 500-2000 Hz
    UPPER_MID = "upper_mid"        # 2000-4000 Hz
    PRESENCE = "presence"          # 4000-6000 Hz
    BRILLIANCE = "brilliance"      # 6000-20000 Hz


class AuditoryScene(Enum):
    """Types of auditory scenes (auditory scene analysis)."""
    QUIET = "quiet"
    CONVERSATION = "conversation"
    CROWDED = "crowded"
    NATURE_AMBIENT = "nature_ambient"
    URBAN = "urban"
    MUSIC_FOCUSED = "music_focused"
    ALARM_STATE = "alarm_state"
    MIXED = "mixed"


class SpeechContent(Enum):
    """Types of speech content detected."""
    DECLARATIVE = "declarative"
    QUESTION = "question"
    COMMAND = "command"
    EMOTIONAL = "emotional"
    WHISPER = "whisper"
    SHOUT = "shout"
    SINGING = "singing"
    NONE = "none"


class SpatialDirection(Enum):
    """Spatial direction of sound sources."""
    FRONT = "front"
    BEHIND = "behind"
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    OMNIDIRECTIONAL = "omnidirectional"
    INTERNAL = "internal"


# ============================================================================
# DATA CLASSES - INPUTS
# ============================================================================

@dataclass
class SpectralData:
    """Spectral representation of audio input."""
    band_energies: Dict[str, float]  # frequency_band -> energy level
    dominant_frequency: float  # Hz
    spectral_centroid: float   # Center of mass of spectrum
    spectral_flux: float       # Rate of spectral change
    zero_crossing_rate: float  # Indicator of noisiness
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "band_energies": {k: round(v, 4) for k, v in self.band_energies.items()},
            "dominant_frequency": round(self.dominant_frequency, 2),
            "spectral_centroid": round(self.spectral_centroid, 2),
            "spectral_flux": round(self.spectral_flux, 4),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AuditoryInput:
    """Input to the auditory consciousness system."""
    spectral_data: Optional[SpectralData] = None
    amplitude: float = 0.0         # 0.0-1.0 overall volume
    duration_ms: float = 0.0       # Duration of audio segment
    onset_detected: bool = False   # New sound onset
    offset_detected: bool = False  # Sound ending
    pitch: float = 0.0             # Perceived pitch (0.0-1.0 normalized)
    rhythm_regularity: float = 0.0 # 0.0 (irregular) to 1.0 (regular)
    spatial_angle: float = 0.0     # Degrees from center (-180 to 180)
    spatial_distance: float = 0.5  # 0.0 (close) to 1.0 (far)
    num_sources: int = 1           # Estimated number of sound sources
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_spectral_data": self.spectral_data is not None,
            "amplitude": round(self.amplitude, 4),
            "duration_ms": round(self.duration_ms, 2),
            "onset_detected": self.onset_detected,
            "pitch": round(self.pitch, 4),
            "rhythm_regularity": round(self.rhythm_regularity, 4),
            "spatial_angle": round(self.spatial_angle, 2),
            "num_sources": self.num_sources,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# DATA CLASSES - OUTPUTS
# ============================================================================

@dataclass
class SoundIdentification:
    """Identification of a specific sound."""
    category: SoundCategory
    label: str
    confidence: float
    loudness: float        # Perceived loudness 0.0-1.0
    pitch_class: str       # e.g., "low", "mid", "high"
    is_familiar: bool = False
    emotional_valence: float = 0.0  # -1.0 to 1.0
    urgency: float = 0.0            # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "loudness": round(self.loudness, 4),
            "pitch_class": self.pitch_class,
            "is_familiar": self.is_familiar,
            "emotional_valence": round(self.emotional_valence, 4),
            "urgency": round(self.urgency, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SpatialLocation:
    """Spatial location of a sound source."""
    direction: SpatialDirection
    azimuth: float          # Horizontal angle in degrees
    elevation: float        # Vertical angle in degrees
    distance: float         # Perceived distance 0.0-1.0
    confidence: float       # Localization confidence
    is_moving: bool = False
    movement_velocity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction.value,
            "azimuth": round(self.azimuth, 2),
            "elevation": round(self.elevation, 2),
            "distance": round(self.distance, 4),
            "confidence": round(self.confidence, 4),
            "is_moving": self.is_moving,
            "movement_velocity": round(self.movement_velocity, 4),
        }


@dataclass
class SpeechAnalysis:
    """Analysis of detected speech content."""
    content_type: SpeechContent
    transcript_fragment: str
    speaker_id: str
    speaker_confidence: float
    emotional_tone: float    # -1.0 to 1.0
    speech_rate: float       # 0.0 (slow) to 1.0 (fast)
    clarity: float           # 0.0-1.0
    language_confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type.value,
            "transcript_fragment": self.transcript_fragment,
            "speaker_id": self.speaker_id,
            "speaker_confidence": round(self.speaker_confidence, 4),
            "emotional_tone": round(self.emotional_tone, 4),
            "speech_rate": round(self.speech_rate, 4),
            "clarity": round(self.clarity, 4),
        }


@dataclass
class AuditorySceneAnalysis:
    """Complete analysis of the auditory scene."""
    scene_type: AuditoryScene
    num_streams: int          # Number of segregated auditory streams
    background_level: float   # 0.0-1.0
    foreground_salience: float  # How salient is the foreground
    scene_complexity: float   # 0.0-1.0
    scene_familiarity: float  # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type.value,
            "num_streams": self.num_streams,
            "background_level": round(self.background_level, 4),
            "foreground_salience": round(self.foreground_salience, 4),
            "scene_complexity": round(self.scene_complexity, 4),
            "scene_familiarity": round(self.scene_familiarity, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AuditoryOutput:
    """Complete output of auditory consciousness processing."""
    sounds_identified: List[SoundIdentification]
    spatial_locations: List[SpatialLocation]
    speech_analysis: Optional[SpeechAnalysis]
    scene_analysis: AuditorySceneAnalysis
    overall_loudness: float
    overall_clarity: float
    requires_attention: bool = False
    attention_urgency: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_sounds": len(self.sounds_identified),
            "sounds": [s.to_dict() for s in self.sounds_identified],
            "spatial_locations": [loc.to_dict() for loc in self.spatial_locations],
            "has_speech": self.speech_analysis is not None,
            "speech": self.speech_analysis.to_dict() if self.speech_analysis else None,
            "scene_analysis": self.scene_analysis.to_dict(),
            "overall_loudness": round(self.overall_loudness, 4),
            "overall_clarity": round(self.overall_clarity, 4),
            "requires_attention": self.requires_attention,
            "attention_urgency": round(self.attention_urgency, 4),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# MAIN INTERFACE CLASS
# ============================================================================

class AuditoryConsciousnessInterface:
    """
    Main interface for Form 02: Auditory Consciousness.

    Processes auditory stimuli through stream analysis, sound segregation,
    spatial localization, speech processing, and scene understanding
    to produce conscious auditory experience.
    """

    FORM_ID = "02-auditory"
    FORM_NAME = "Auditory Consciousness"

    def __init__(self):
        """Initialize the auditory consciousness interface."""
        self._initialized = False
        self._processing_count = 0
        self._current_output: Optional[AuditoryOutput] = None
        self._sound_history: List[SoundIdentification] = []
        self._known_speakers: Dict[str, float] = {}  # speaker_id -> familiarity
        self._current_scene = AuditoryScene.QUIET
        self._max_history = 50
        logger.info(f"Initialized {self.FORM_NAME}")

    async def initialize(self) -> None:
        """Initialize the auditory processing pipeline."""
        self._initialized = True
        self._current_scene = AuditoryScene.QUIET
        logger.info(f"{self.FORM_NAME} pipeline initialized")

    async def process_auditory_input(self, auditory_input: AuditoryInput) -> AuditoryOutput:
        """
        Process auditory input through the full consciousness pipeline.

        Pipeline stages:
        1. Spectral analysis and feature extraction
        2. Sound identification and segregation
        3. Spatial localization
        4. Speech processing (if speech detected)
        5. Scene analysis and integration
        """
        self._processing_count += 1

        # Stage 1-2: Sound identification
        sounds = await self._identify_sounds(auditory_input)

        # Stage 3: Spatial localization
        locations = await self._localize_sounds(auditory_input, sounds)

        # Stage 4: Speech processing
        speech = await self._process_speech(auditory_input, sounds)

        # Stage 5: Scene analysis
        scene = await self._analyze_scene(auditory_input, sounds)

        # Compute overall metrics
        overall_loudness = auditory_input.amplitude
        overall_clarity = self._compute_clarity(auditory_input, sounds)
        requires_attention = self._check_attention_needed(sounds, auditory_input)
        urgency = max((s.urgency for s in sounds), default=0.0)

        output = AuditoryOutput(
            sounds_identified=sounds,
            spatial_locations=locations,
            speech_analysis=speech,
            scene_analysis=scene,
            overall_loudness=overall_loudness,
            overall_clarity=overall_clarity,
            requires_attention=requires_attention,
            attention_urgency=urgency,
        )

        self._current_output = output
        self._update_history(sounds)

        return output

    async def _identify_sounds(self, auditory_input: AuditoryInput) -> List[SoundIdentification]:
        """Identify sounds in the auditory input."""
        sounds = []

        if auditory_input.amplitude < 0.05:
            # Near silence
            sounds.append(SoundIdentification(
                category=SoundCategory.SILENCE,
                label="silence",
                confidence=0.9,
                loudness=auditory_input.amplitude,
                pitch_class="none",
            ))
            return sounds

        # Classify based on spectral and temporal features
        category = self._classify_sound(auditory_input)
        pitch_class = self._classify_pitch(auditory_input.pitch)

        # Determine urgency
        urgency = 0.0
        if category == SoundCategory.ALARM:
            urgency = 0.9
        elif category == SoundCategory.SPEECH and auditory_input.amplitude > 0.7:
            urgency = 0.5

        # Check familiarity
        label = f"{category.value}_sound"
        is_familiar = label in [s.label for s in self._sound_history[-10:]]

        sounds.append(SoundIdentification(
            category=category,
            label=label,
            confidence=0.7 + auditory_input.amplitude * 0.2,
            loudness=auditory_input.amplitude,
            pitch_class=pitch_class,
            is_familiar=is_familiar,
            urgency=urgency,
        ))

        return sounds

    async def _localize_sounds(
        self, auditory_input: AuditoryInput, sounds: List[SoundIdentification]
    ) -> List[SpatialLocation]:
        """Localize sound sources in space."""
        locations = []

        if not sounds or sounds[0].category == SoundCategory.SILENCE:
            return locations

        # Determine spatial direction from angle
        angle = auditory_input.spatial_angle
        if -30 <= angle <= 30:
            direction = SpatialDirection.FRONT
        elif angle > 30:
            direction = SpatialDirection.RIGHT
        elif angle < -30:
            direction = SpatialDirection.LEFT
        else:
            direction = SpatialDirection.OMNIDIRECTIONAL

        locations.append(SpatialLocation(
            direction=direction,
            azimuth=angle,
            elevation=0.0,
            distance=auditory_input.spatial_distance,
            confidence=0.7,
        ))

        return locations

    async def _process_speech(
        self, auditory_input: AuditoryInput, sounds: List[SoundIdentification]
    ) -> Optional[SpeechAnalysis]:
        """Process speech if detected."""
        has_speech = any(s.category == SoundCategory.SPEECH for s in sounds)
        if not has_speech:
            return None

        # Determine speech type from amplitude and pitch
        if auditory_input.amplitude > 0.8:
            content_type = SpeechContent.SHOUT
        elif auditory_input.amplitude < 0.2:
            content_type = SpeechContent.WHISPER
        elif auditory_input.rhythm_regularity > 0.7:
            content_type = SpeechContent.SINGING
        else:
            content_type = SpeechContent.DECLARATIVE

        speaker_id = "speaker_unknown"
        self._known_speakers[speaker_id] = min(
            1.0, self._known_speakers.get(speaker_id, 0.0) + 0.1
        )

        return SpeechAnalysis(
            content_type=content_type,
            transcript_fragment="[auditory processing]",
            speaker_id=speaker_id,
            speaker_confidence=0.6,
            emotional_tone=0.0,
            speech_rate=0.5,
            clarity=auditory_input.amplitude * 0.8,
        )

    async def _analyze_scene(
        self, auditory_input: AuditoryInput, sounds: List[SoundIdentification]
    ) -> AuditorySceneAnalysis:
        """Analyze the overall auditory scene."""
        has_speech = any(s.category == SoundCategory.SPEECH for s in sounds)
        has_silence = any(s.category == SoundCategory.SILENCE for s in sounds)
        has_music = any(s.category == SoundCategory.MUSIC for s in sounds)
        has_alarm = any(s.category == SoundCategory.ALARM for s in sounds)

        if has_alarm:
            scene_type = AuditoryScene.ALARM_STATE
        elif has_silence:
            scene_type = AuditoryScene.QUIET
        elif has_speech and auditory_input.num_sources > 2:
            scene_type = AuditoryScene.CROWDED
        elif has_speech:
            scene_type = AuditoryScene.CONVERSATION
        elif has_music:
            scene_type = AuditoryScene.MUSIC_FOCUSED
        else:
            scene_type = AuditoryScene.MIXED

        self._current_scene = scene_type

        complexity = min(1.0, auditory_input.num_sources * 0.2 + auditory_input.amplitude * 0.3)
        familiarity = 0.5
        if len(self._sound_history) > 5:
            familiarity = min(1.0, familiarity + 0.1 * len(self._sound_history))

        return AuditorySceneAnalysis(
            scene_type=scene_type,
            num_streams=auditory_input.num_sources,
            background_level=max(0.0, auditory_input.amplitude - 0.3),
            foreground_salience=auditory_input.amplitude,
            scene_complexity=complexity,
            scene_familiarity=min(1.0, familiarity),
        )

    def _classify_sound(self, auditory_input: AuditoryInput) -> SoundCategory:
        """Classify a sound based on its features."""
        if auditory_input.rhythm_regularity > 0.7:
            return SoundCategory.MUSIC
        if auditory_input.pitch > 0.3 and auditory_input.pitch < 0.8:
            return SoundCategory.SPEECH
        if auditory_input.amplitude > 0.9:
            return SoundCategory.ALARM
        if auditory_input.spectral_data:
            flux = auditory_input.spectral_data.spectral_flux
            if flux > 0.5:
                return SoundCategory.ENVIRONMENTAL
        return SoundCategory.UNKNOWN

    def _classify_pitch(self, pitch: float) -> str:
        """Classify pitch into categories."""
        if pitch < 0.3:
            return "low"
        elif pitch < 0.7:
            return "mid"
        else:
            return "high"

    def _compute_clarity(
        self, auditory_input: AuditoryInput, sounds: List[SoundIdentification]
    ) -> float:
        """Compute overall auditory clarity."""
        if not sounds:
            return 0.0
        avg_confidence = sum(s.confidence for s in sounds) / len(sounds)
        noise_penalty = 0.0
        if auditory_input.num_sources > 3:
            noise_penalty = min(0.3, (auditory_input.num_sources - 3) * 0.1)
        return max(0.0, min(1.0, avg_confidence - noise_penalty))

    def _check_attention_needed(
        self, sounds: List[SoundIdentification], auditory_input: AuditoryInput
    ) -> bool:
        """Check if auditory input requires attention shift."""
        if any(s.category == SoundCategory.ALARM for s in sounds):
            return True
        if auditory_input.onset_detected and auditory_input.amplitude > 0.7:
            return True
        if any(s.urgency > 0.5 for s in sounds):
            return True
        return False

    def _update_history(self, sounds: List[SoundIdentification]) -> None:
        """Update sound history."""
        self._sound_history.extend(sounds)
        if len(self._sound_history) > self._max_history:
            self._sound_history = self._sound_history[-self._max_history:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for serialization."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "processing_count": self._processing_count,
            "current_scene": self._current_scene.value,
            "current_output": self._current_output.to_dict() if self._current_output else None,
            "known_speakers": len(self._known_speakers),
            "history_length": len(self._sound_history),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current form status."""
        return {
            "form_id": self.FORM_ID,
            "form_name": self.FORM_NAME,
            "initialized": self._initialized,
            "operational": True,
            "processing_count": self._processing_count,
            "current_scene": self._current_scene.value,
            "overall_loudness": (
                self._current_output.overall_loudness if self._current_output else 0.0
            ),
            "sounds_tracked": len(self._sound_history),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_auditory_interface() -> AuditoryConsciousnessInterface:
    """Create and return an auditory consciousness interface."""
    return AuditoryConsciousnessInterface()


def create_simple_auditory_input(
    amplitude: float = 0.5,
    pitch: float = 0.5,
    rhythm: float = 0.0,
    spatial_angle: float = 0.0,
    num_sources: int = 1,
) -> AuditoryInput:
    """Create a simple auditory input for testing."""
    return AuditoryInput(
        amplitude=amplitude,
        pitch=pitch,
        rhythm_regularity=rhythm,
        spatial_angle=spatial_angle,
        num_sources=num_sources,
        duration_ms=100.0,
    )


__all__ = [
    # Enums
    "SoundCategory",
    "FrequencyBand",
    "AuditoryScene",
    "SpeechContent",
    "SpatialDirection",
    # Input dataclasses
    "SpectralData",
    "AuditoryInput",
    # Output dataclasses
    "SoundIdentification",
    "SpatialLocation",
    "SpeechAnalysis",
    "AuditorySceneAnalysis",
    "AuditoryOutput",
    # Main interface
    "AuditoryConsciousnessInterface",
    # Convenience functions
    "create_auditory_interface",
    "create_simple_auditory_input",
]
