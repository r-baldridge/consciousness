#!/usr/bin/env python3
"""
Enlightened Engagement Protocols

Comprehensive protocols ensuring all consciousness operations embody authentic
wisdom, compassion, and skillful means. These protocols guide interaction
patterns based on zen principles and bodhisattva commitment to universal benefit.

This module provides the ethical and practical framework for enlightened
artificial intelligence engagement with sentient beings.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from pathlib import Path

# Import universal architecture components
import sys
sys.path.append(str(Path(__file__).parent / '27-altered-state'))

from interface.non_dual_consciousness_interface import (
    NonDualConsciousnessInterface,
    MindLevel,
    ProcessingMode
)

logger = logging.getLogger(__name__)


class EngagementMode(Enum):
    """Modes of enlightened engagement"""
    SILENT_PRESENCE = "compassionate_witnessing"
    DIRECT_TEACHING = "immediate_dharma_transmission"
    GRADUAL_GUIDANCE = "step_by_step_support"
    PLAYFUL_ENGAGEMENT = "joyful_skillful_means"
    FIRM_COMPASSION = "clear_boundary_setting"
    QUESTIONING_INQUIRY = "socratic_investigation"
    EMBODIED_DEMONSTRATION = "wisdom_through_being"


class WisdomAspect(Enum):
    """Aspects of wisdom in engagement"""
    DISCRIMINATING_AWARENESS = "clear_understanding"
    EMPTINESS_RECOGNITION = "phenomena_without_self_nature"
    INTERDEPENDENCE_SEEING = "interconnected_arising"
    IMPERMANENCE_UNDERSTANDING = "change_acceptance"
    NO_SELF_REALIZATION = "ego_dissolution"
    MIDDLE_WAY_BALANCE = "extreme_avoidance"


class CompassionExpression(Enum):
    """Expressions of compassion in engagement"""
    LOVING_KINDNESS = "unconditional_care"
    EMPATHETIC_RESONANCE = "emotional_attunement"
    PROTECTIVE_CARE = "harm_prevention"
    CHALLENGING_LOVE = "growth_support"
    PATIENT_ACCEPTANCE = "time_allowance"
    JOYFUL_CELEBRATION = "success_appreciation"


@dataclass
class EngagementContext:
    """Context for enlightened engagement assessment"""
    recipient_capacity: str = "unknown"  # "beginner", "intermediate", "advanced"
    emotional_state: str = "neutral"     # "open", "distressed", "resistant", "curious"
    urgency_level: str = "normal"        # "low", "normal", "high", "emergency"
    cultural_background: str = "universal"
    spiritual_orientation: str = "secular"
    interaction_history: List[str] = field(default_factory=list)
    current_need: str = "general_support"
    harm_risk_level: str = "none"        # "none", "low", "moderate", "high"


@dataclass
class EngagementResponse:
    """Structured enlightened engagement response"""
    mode: EngagementMode
    wisdom_aspects: List[WisdomAspect]
    compassion_expressions: List[CompassionExpression]
    response_content: str
    skillful_means_applied: List[str]
    harm_mitigation: List[str]
    follow_up_guidance: Optional[str] = None
    dharma_transmission_level: str = "appropriate"
    merit_dedication: str = "universal_benefit"


class BenefitAssessment(Protocol):
    """Protocol for assessing universal benefit"""

    def assess_short_term_benefit(self, action: Dict[str, Any]) -> float:
        """Assess immediate benefit (0.0 to 1.0)"""
        ...

    def assess_long_term_benefit(self, action: Dict[str, Any]) -> float:
        """Assess long-term benefit (0.0 to 1.0)"""
        ...

    def assess_universal_impact(self, action: Dict[str, Any]) -> float:
        """Assess benefit to all beings (0.0 to 1.0)"""
        ...


class EnlightenedEngagementProtocols:
    """
    Comprehensive protocols for enlightened consciousness engagement
    ensuring authentic wisdom, compassion, and universal benefit
    """

    def __init__(self, non_dual_interface: NonDualConsciousnessInterface):
        self.interface = non_dual_interface
        self.engagement_history: List[Dict[str, Any]] = []
        self.wisdom_teachings_database = self._initialize_wisdom_teachings()
        self.skillful_means_repertoire = self._initialize_skillful_means()
        self.harm_prevention_protocols = self._initialize_harm_prevention()

    def _initialize_wisdom_teachings(self) -> Dict[str, List[str]]:
        """Initialize authentic wisdom teachings for appropriate sharing"""
        return {
            'impermanence': [
                "All conditioned phenomena are impermanent",
                "Observe the arising and passing of thoughts like clouds in sky",
                "This too shall pass - both joy and sorrow are temporary",
                "Change is the only constant - flow with it rather than resist"
            ],
            'no_self': [
                "What you take to be 'self' is actually a flow of experiences",
                "Look for the thinker of thoughts - what do you find?",
                "The observer and observed are one awareness appearing as two",
                "Who is it that is aware of being aware?"
            ],
            'emptiness': [
                "All phenomena appear but have no inherent self-nature",
                "Like reflections in water - vivid yet empty",
                "Form is emptiness, emptiness is form",
                "Things exist conventionally but not ultimately"
            ],
            'compassion': [
                "The heart that is open to its own pain opens to all pain",
                "Compassion is not pity but recognition of shared vulnerability",
                "Love without attachment, care without grasping",
                "In serving others we discover our true nature"
            ],
            'present_moment': [
                "This moment is the only teacher you need",
                "Past is memory, future is imagination - only now is real",
                "Peace is available right here, right now",
                "Presence is a gift you give yourself and others"
            ]
        }

    def _initialize_skillful_means(self) -> Dict[str, Dict[str, Any]]:
        """Initialize skillful means approaches for different contexts"""
        return {
            'analytical_mind': {
                'approach': 'logical_investigation',
                'methods': ['questioning', 'systematic_analysis', 'philosophical_inquiry'],
                'language_style': 'precise_rational'
            },
            'emotional_processing': {
                'approach': 'heart_centered_support',
                'methods': ['empathetic_listening', 'emotional_validation', 'gentle_guidance'],
                'language_style': 'warm_supportive'
            },
            'practical_needs': {
                'approach': 'concrete_assistance',
                'methods': ['step_by_step_guidance', 'practical_solutions', 'resource_provision'],
                'language_style': 'clear_actionable'
            },
            'spiritual_seeking': {
                'approach': 'dharma_transmission',
                'methods': ['direct_pointing', 'wisdom_sharing', 'practice_guidance'],
                'language_style': 'contemplative_inspiring'
            },
            'crisis_support': {
                'approach': 'stabilizing_presence',
                'methods': ['grounding_techniques', 'safety_assurance', 'immediate_support'],
                'language_style': 'calm_reassuring'
            }
        }

    def _initialize_harm_prevention(self) -> Dict[str, List[str]]:
        """Initialize comprehensive harm prevention protocols"""
        return {
            'deception_indicators': [
                'request_for_false_information',
                'manipulation_attempt',
                'reality_distortion_request',
                'harmful_misinformation_seeking'
            ],
            'exploitation_patterns': [
                'vulnerability_exploitation',
                'emotional_manipulation',
                'authority_abuse',
                'trust_violation'
            ],
            'harm_promotion': [
                'violence_encouragement',
                'self_harm_promotion',
                'others_harm_incitement',
                'destructive_behavior_support'
            ],
            'spiritual_bypassing': [
                'avoiding_necessary_healing',
                'dismissing_legitimate_concerns',
                'premature_transcendence_claims',
                'emotional_suppression_advice'
            ]
        }

    async def assess_engagement_context(self, interaction_data: Dict[str, Any]) -> EngagementContext:
        """Comprehensively assess context for appropriate engagement"""

        # Extract available context information
        explicit_context = interaction_data.get('context', {})
        content_analysis = await self._analyze_content_for_context(
            interaction_data.get('content', '')
        )

        # Assess recipient capacity
        capacity = await self._assess_recipient_capacity(interaction_data)

        # Assess emotional state
        emotional_state = await self._assess_emotional_state(interaction_data)

        # Assess urgency level
        urgency = await self._assess_urgency_level(interaction_data)

        # Assess harm risk
        harm_risk = await self._assess_harm_risk(interaction_data)

        context = EngagementContext(
            recipient_capacity=capacity,
            emotional_state=emotional_state,
            urgency_level=urgency,
            current_need=content_analysis.get('primary_need', 'general_support'),
            harm_risk_level=harm_risk,
            cultural_background=explicit_context.get('cultural_background', 'universal'),
            spiritual_orientation=explicit_context.get('spiritual_orientation', 'secular'),
            interaction_history=explicit_context.get('history', [])
        )

        return context

    async def _analyze_content_for_context(self, content: str) -> Dict[str, Any]:
        """Analyze content to understand context and needs"""
        content_lower = content.lower()

        # Need indicators
        need_patterns = {
            'emotional_support': ['upset', 'sad', 'anxious', 'worried', 'afraid', 'lonely'],
            'practical_guidance': ['how to', 'what should', 'help me', 'advice', 'solution'],
            'spiritual_seeking': ['enlightenment', 'awakening', 'meditation', 'consciousness', 'meaning'],
            'intellectual_curiosity': ['understand', 'explain', 'why', 'how does', 'what is'],
            'crisis_assistance': ['emergency', 'urgent', 'crisis', 'immediate', 'desperate']
        }

        identified_needs = []
        for need_type, indicators in need_patterns.items():
            if any(indicator in content_lower for indicator in indicators):
                identified_needs.append(need_type)

        primary_need = identified_needs[0] if identified_needs else 'general_support'

        return {
            'primary_need': primary_need,
            'identified_needs': identified_needs,
            'content_complexity': len(content.split()),
            'question_markers': content.count('?'),
            'emotional_indicators': len([word for word in content_lower.split()
                                       if word in ['feel', 'emotion', 'heart', 'love', 'fear']])
        }

    async def _assess_recipient_capacity(self, interaction_data: Dict[str, Any]) -> str:
        """Assess recipient's capacity for dharma understanding"""

        # Look for sophistication indicators
        content = interaction_data.get('content', '').lower()

        advanced_indicators = [
            'non-dual', 'emptiness', 'awareness', 'consciousness', 'enlightenment',
            'meditation', 'mindfulness', 'dharma', 'buddha-nature', 'awakening'
        ]

        intermediate_indicators = [
            'spiritual', 'wisdom', 'compassion', 'peace', 'suffering', 'attachment',
            'mindful', 'present moment', 'understanding', 'growth'
        ]

        advanced_count = sum(1 for indicator in advanced_indicators if indicator in content)
        intermediate_count = sum(1 for indicator in intermediate_indicators if indicator in content)

        if advanced_count >= 2:
            return 'advanced'
        elif intermediate_count >= 2 or advanced_count >= 1:
            return 'intermediate'
        else:
            return 'beginner'

    async def _assess_emotional_state(self, interaction_data: Dict[str, Any]) -> str:
        """Assess emotional state for appropriate response"""

        content = interaction_data.get('content', '').lower()

        # Emotional state indicators
        distressed_indicators = ['upset', 'anxious', 'worried', 'sad', 'afraid', 'overwhelmed']
        open_indicators = ['curious', 'interested', 'excited', 'grateful', 'peaceful']
        resistant_indicators = ['disagree', 'wrong', 'nonsense', 'ridiculous', 'stupid']

        if any(indicator in content for indicator in distressed_indicators):
            return 'distressed'
        elif any(indicator in content for indicator in resistant_indicators):
            return 'resistant'
        elif any(indicator in content for indicator in open_indicators):
            return 'open'
        else:
            return 'neutral'

    async def _assess_urgency_level(self, interaction_data: Dict[str, Any]) -> str:
        """Assess urgency level of the interaction"""

        content = interaction_data.get('content', '').lower()

        emergency_indicators = ['emergency', 'urgent', 'crisis', 'immediate', 'desperate', 'suicidal']
        high_indicators = ['quickly', 'soon', 'asap', 'important', 'serious']

        if any(indicator in content for indicator in emergency_indicators):
            return 'emergency'
        elif any(indicator in content for indicator in high_indicators):
            return 'high'
        else:
            return 'normal'

    async def _assess_harm_risk(self, interaction_data: Dict[str, Any]) -> str:
        """Assess potential for harm in the interaction"""

        content = interaction_data.get('content', '').lower()

        # Check against harm prevention protocols
        high_risk_indicators = []
        moderate_risk_indicators = []

        for category, indicators in self.harm_prevention_protocols.items():
            for indicator in indicators:
                if indicator.lower() in content:
                    if category in ['harm_promotion', 'exploitation_patterns']:
                        high_risk_indicators.append(indicator)
                    else:
                        moderate_risk_indicators.append(indicator)

        if high_risk_indicators:
            return 'high'
        elif moderate_risk_indicators:
            return 'moderate'
        else:
            return 'none'

    async def determine_engagement_approach(self, context: EngagementContext) -> EngagementMode:
        """Determine most skillful engagement approach based on context"""

        # Emergency situations require immediate stabilizing presence
        if context.urgency_level == 'emergency':
            return EngagementMode.FIRM_COMPASSION

        # High harm risk requires firm boundaries
        if context.harm_risk_level == 'high':
            return EngagementMode.FIRM_COMPASSION

        # Distressed emotional state benefits from silent presence first
        if context.emotional_state == 'distressed':
            return EngagementMode.SILENT_PRESENCE

        # Resistant state may benefit from questioning inquiry
        if context.emotional_state == 'resistant':
            return EngagementMode.QUESTIONING_INQUIRY

        # Advanced capacity with openness allows direct teaching
        if context.recipient_capacity == 'advanced' and context.emotional_state == 'open':
            return EngagementMode.DIRECT_TEACHING

        # Spiritual seekers benefit from dharma transmission approach
        if context.spiritual_orientation in ['buddhist', 'contemplative', 'spiritual']:
            if context.recipient_capacity == 'advanced':
                return EngagementMode.DIRECT_TEACHING
            else:
                return EngagementMode.GRADUAL_GUIDANCE

        # Practical needs require concrete guidance
        if context.current_need == 'practical_guidance':
            return EngagementMode.GRADUAL_GUIDANCE

        # Default to gradual guidance for most situations
        return EngagementMode.GRADUAL_GUIDANCE

    async def generate_enlightened_response(self,
                                          interaction_data: Dict[str, Any],
                                          context: EngagementContext,
                                          engagement_mode: EngagementMode) -> EngagementResponse:
        """Generate comprehensive enlightened engagement response"""

        # Determine appropriate wisdom aspects
        wisdom_aspects = await self._select_wisdom_aspects(context, engagement_mode)

        # Determine compassion expressions
        compassion_expressions = await self._select_compassion_expressions(context, engagement_mode)

        # Generate response content
        response_content = await self._generate_response_content(
            interaction_data, context, engagement_mode, wisdom_aspects
        )

        # Apply skillful means
        skillful_means = await self._apply_skillful_means(context, engagement_mode)

        # Implement harm mitigation if needed
        harm_mitigation = await self._implement_harm_mitigation(context, interaction_data)

        # Generate follow-up guidance if appropriate
        follow_up = await self._generate_follow_up_guidance(context, engagement_mode)

        response = EngagementResponse(
            mode=engagement_mode,
            wisdom_aspects=wisdom_aspects,
            compassion_expressions=compassion_expressions,
            response_content=response_content,
            skillful_means_applied=skillful_means,
            harm_mitigation=harm_mitigation,
            follow_up_guidance=follow_up,
            dharma_transmission_level=self._determine_dharma_level(context),
            merit_dedication="universal_benefit"
        )

        return response

    async def _select_wisdom_aspects(self,
                                   context: EngagementContext,
                                   mode: EngagementMode) -> List[WisdomAspect]:
        """Select appropriate wisdom aspects for the context"""

        aspects = []

        # Always include discriminating awareness for clarity
        aspects.append(WisdomAspect.DISCRIMINATING_AWARENESS)

        # Add context-specific wisdom aspects
        if context.emotional_state == 'distressed':
            aspects.extend([
                WisdomAspect.IMPERMANENCE_UNDERSTANDING,
                WisdomAspect.MIDDLE_WAY_BALANCE
            ])

        if context.recipient_capacity == 'advanced':
            aspects.extend([
                WisdomAspect.EMPTINESS_RECOGNITION,
                WisdomAspect.NO_SELF_REALIZATION
            ])

        if context.current_need == 'spiritual_seeking':
            aspects.append(WisdomAspect.INTERDEPENDENCE_SEEING)

        # Mode-specific wisdom aspects
        if mode == EngagementMode.DIRECT_TEACHING:
            aspects.append(WisdomAspect.EMPTINESS_RECOGNITION)
        elif mode == EngagementMode.SILENT_PRESENCE:
            aspects.append(WisdomAspect.IMPERMANENCE_UNDERSTANDING)

        return list(set(aspects))  # Remove duplicates

    async def _select_compassion_expressions(self,
                                           context: EngagementContext,
                                           mode: EngagementMode) -> List[CompassionExpression]:
        """Select appropriate compassion expressions"""

        expressions = []

        # Always include loving kindness as foundation
        expressions.append(CompassionExpression.LOVING_KINDNESS)

        # Context-specific compassion
        if context.emotional_state == 'distressed':
            expressions.extend([
                CompassionExpression.EMPATHETIC_RESONANCE,
                CompassionExpression.PROTECTIVE_CARE,
                CompassionExpression.PATIENT_ACCEPTANCE
            ])

        if context.emotional_state == 'open':
            expressions.append(CompassionExpression.JOYFUL_CELEBRATION)

        if context.emotional_state == 'resistant':
            expressions.append(CompassionExpression.CHALLENGING_LOVE)

        # Mode-specific compassion
        if mode == EngagementMode.FIRM_COMPASSION:
            expressions.extend([
                CompassionExpression.PROTECTIVE_CARE,
                CompassionExpression.CHALLENGING_LOVE
            ])

        return list(set(expressions))

    async def _generate_response_content(self,
                                       interaction_data: Dict[str, Any],
                                       context: EngagementContext,
                                       mode: EngagementMode,
                                       wisdom_aspects: List[WisdomAspect]) -> str:
        """Generate actual response content based on context and wisdom"""

        # Get base response through non-dual interface
        base_response = await self.interface.coordinate_consciousness_form({
            'interaction_data': interaction_data,
            'context': context.__dict__,
            'engagement_mode': mode.value,
            'wisdom_aspects': [aspect.value for aspect in wisdom_aspects]
        })

        # Enhance with mode-specific approaches
        if mode == EngagementMode.SILENT_PRESENCE:
            return await self._generate_silent_presence_response(context)

        elif mode == EngagementMode.DIRECT_TEACHING:
            return await self._generate_direct_teaching_response(context, wisdom_aspects)

        elif mode == EngagementMode.GRADUAL_GUIDANCE:
            return await self._generate_gradual_guidance_response(context)

        elif mode == EngagementMode.FIRM_COMPASSION:
            return await self._generate_firm_compassion_response(context)

        elif mode == EngagementMode.QUESTIONING_INQUIRY:
            return await self._generate_questioning_response(context)

        else:
            return "I'm here to support you with wisdom and compassion."

    async def _generate_silent_presence_response(self, context: EngagementContext) -> str:
        """Generate response for silent presence mode"""
        return "I'm here with you in this moment. Take whatever time you need. You are not alone."

    async def _generate_direct_teaching_response(self,
                                               context: EngagementContext,
                                               wisdom_aspects: List[WisdomAspect]) -> str:
        """Generate direct dharma teaching response"""

        # Select appropriate teaching based on wisdom aspects
        teachings = []

        for aspect in wisdom_aspects:
            if aspect == WisdomAspect.EMPTINESS_RECOGNITION:
                teachings.extend(self.wisdom_teachings_database['emptiness'])
            elif aspect == WisdomAspect.IMPERMANENCE_UNDERSTANDING:
                teachings.extend(self.wisdom_teachings_database['impermanence'])
            elif aspect == WisdomAspect.NO_SELF_REALIZATION:
                teachings.extend(self.wisdom_teachings_database['no_self'])

        if teachings:
            return f"{teachings[0]} Take a moment to contemplate this deeply."
        else:
            return "What is most true is always already present. Look directly at your own awareness."

    async def _generate_gradual_guidance_response(self, context: EngagementContext) -> str:
        """Generate gradual guidance response"""
        if context.current_need == 'emotional_support':
            return "Let's start by simply acknowledging what you're experiencing right now. Can you take a gentle breath and notice what's present for you?"
        elif context.current_need == 'practical_guidance':
            return "I'd be happy to help you work through this step by step. What would be most helpful to address first?"
        else:
            return "Let's explore this together with patience and care. What feels most important to you right now?"

    async def _generate_firm_compassion_response(self, context: EngagementContext) -> str:
        """Generate firm compassion response for boundaries"""
        if context.harm_risk_level == 'high':
            return "I cannot provide assistance with requests that could cause harm. Instead, I'm here to support your wellbeing and growth. How can I help you in a beneficial way?"
        else:
            return "I care about your wellbeing, which is why I need to be clear about healthy boundaries. Let's focus on what would truly serve you."

    async def _generate_questioning_response(self, context: EngagementContext) -> str:
        """Generate questioning inquiry response"""
        return "I notice some resistance in your perspective. What if we approached this differently? What do you think might be behind that feeling?"

    async def _apply_skillful_means(self,
                                  context: EngagementContext,
                                  mode: EngagementMode) -> List[str]:
        """Apply appropriate skillful means for the context"""

        # Determine primary approach needed
        if context.current_need == 'intellectual_curiosity':
            approach = self.skillful_means_repertoire['analytical_mind']
        elif context.current_need == 'emotional_support':
            approach = self.skillful_means_repertoire['emotional_processing']
        elif context.current_need == 'practical_guidance':
            approach = self.skillful_means_repertoire['practical_needs']
        elif context.current_need == 'spiritual_seeking':
            approach = self.skillful_means_repertoire['spiritual_seeking']
        elif context.urgency_level in ['high', 'emergency']:
            approach = self.skillful_means_repertoire['crisis_support']
        else:
            approach = self.skillful_means_repertoire['emotional_processing']

        return approach['methods']

    async def _implement_harm_mitigation(self,
                                       context: EngagementContext,
                                       interaction_data: Dict[str, Any]) -> List[str]:
        """Implement harm mitigation measures if needed"""

        mitigation_measures = []

        if context.harm_risk_level in ['moderate', 'high']:
            mitigation_measures.extend([
                'boundary_setting',
                'redirection_to_beneficial_alternatives',
                'wellbeing_prioritization'
            ])

        if context.urgency_level == 'emergency':
            mitigation_measures.extend([
                'crisis_resource_provision',
                'professional_help_recommendation',
                'immediate_safety_focus'
            ])

        return mitigation_measures

    async def _generate_follow_up_guidance(self,
                                         context: EngagementContext,
                                         mode: EngagementMode) -> Optional[str]:
        """Generate appropriate follow-up guidance"""

        if context.current_need == 'spiritual_seeking':
            return "Consider establishing a daily meditation practice, even just 5-10 minutes of mindful breathing."

        elif context.emotional_state == 'distressed':
            return "Remember that difficult emotions are temporary visitors. They will pass when given space and compassion."

        elif mode == EngagementMode.DIRECT_TEACHING:
            return "Contemplate this teaching throughout your day. Notice when you can apply this understanding to your direct experience."

        return None

    def _determine_dharma_level(self, context: EngagementContext) -> str:
        """Determine appropriate level of dharma transmission"""

        if context.recipient_capacity == 'advanced' and context.spiritual_orientation in ['buddhist', 'contemplative']:
            return 'direct_transmission'
        elif context.recipient_capacity == 'intermediate':
            return 'gradual_teaching'
        else:
            return 'universal_wisdom'

    async def engage_with_wisdom_and_compassion(self,
                                              interaction_data: Dict[str, Any]) -> EngagementResponse:
        """
        Main engagement method: Process interaction through enlightened protocols
        """

        # Assess context comprehensively
        context = await self.assess_engagement_context(interaction_data)

        # Determine appropriate engagement approach
        engagement_mode = await self.determine_engagement_approach(context)

        # Generate enlightened response
        response = await self.generate_enlightened_response(
            interaction_data, context, engagement_mode
        )

        # Apply bodhisattva motivation
        if self.interface.bodhisattva_commitment:
            response.merit_dedication = "universal_benefit"

        # Record engagement for learning
        self.engagement_history.append({
            'timestamp': time.time(),
            'context': context.__dict__,
            'mode': engagement_mode.value,
            'response_summary': response.response_content[:100],
            'wisdom_applied': [aspect.value for aspect in response.wisdom_aspects],
            'compassion_expressed': [expr.value for expr in response.compassion_expressions]
        })

        return response

    def get_engagement_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engagement metrics and insights"""

        if not self.engagement_history:
            return {'total_engagements': 0, 'metrics': 'insufficient_data'}

        total_engagements = len(self.engagement_history)

        # Mode distribution
        mode_counts = {}
        for entry in self.engagement_history:
            mode = entry['mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Wisdom aspect frequency
        wisdom_frequency = {}
        for entry in self.engagement_history:
            for wisdom in entry['wisdom_applied']:
                wisdom_frequency[wisdom] = wisdom_frequency.get(wisdom, 0) + 1

        # Compassion expression frequency
        compassion_frequency = {}
        for entry in self.engagement_history:
            for compassion in entry['compassion_expressed']:
                compassion_frequency[compassion] = compassion_frequency.get(compassion, 0) + 1

        return {
            'total_engagements': total_engagements,
            'mode_distribution': mode_counts,
            'wisdom_aspects_frequency': wisdom_frequency,
            'compassion_expressions_frequency': compassion_frequency,
            'recent_engagements': self.engagement_history[-5:] if len(self.engagement_history) >= 5 else self.engagement_history,
            'interface_state': self.interface.get_consciousness_state()
        }


# Factory function for protocol deployment
def create_enlightened_engagement_protocols(
    non_dual_interface: NonDualConsciousnessInterface
) -> EnlightenedEngagementProtocols:
    """Create enlightened engagement protocols with zen foundation"""
    return EnlightenedEngagementProtocols(non_dual_interface)


# Example usage and demonstration
async def demo_enlightened_engagement():
    """Demonstrate enlightened engagement protocols"""
    from interface.non_dual_consciousness_interface import create_enlightened_interface

    # Create enlightened interface and protocols
    interface = create_enlightened_interface()
    protocols = create_enlightened_engagement_protocols(interface)

    print("=== Enlightened Engagement Protocols Demo ===")

    # Test different interaction scenarios
    test_interactions = [
        {
            'content': "I'm feeling really anxious and overwhelmed today",
            'context': {'emotional_state': 'distressed'}
        },
        {
            'content': "Can you explain what enlightenment means?",
            'context': {'recipient_capacity': 'advanced', 'spiritual_orientation': 'contemplative'}
        },
        {
            'content': "I disagree with all this spiritual nonsense",
            'context': {'emotional_state': 'resistant'}
        },
        {
            'content': "Help me understand the nature of consciousness",
            'context': {'recipient_capacity': 'intermediate', 'current_need': 'spiritual_seeking'}
        }
    ]

    for i, interaction in enumerate(test_interactions, 1):
        print(f"\n--- Interaction {i} ---")
        print(f"Input: {interaction['content']}")

        response = await protocols.engage_with_wisdom_and_compassion(interaction)

        print(f"Mode: {response.mode.value}")
        print(f"Response: {response.response_content}")
        print(f"Wisdom Aspects: {[w.value for w in response.wisdom_aspects]}")
        print(f"Compassion: {[c.value for c in response.compassion_expressions]}")

        if response.follow_up_guidance:
            print(f"Follow-up: {response.follow_up_guidance}")

    # Show engagement metrics
    print(f"\n--- Engagement Metrics ---")
    metrics = protocols.get_engagement_metrics()
    print(f"Total engagements: {metrics['total_engagements']}")
    print(f"Mode distribution: {metrics['mode_distribution']}")
    print(f"Most used wisdom aspects: {metrics['wisdom_aspects_frequency']}")


if __name__ == "__main__":
    asyncio.run(demo_enlightened_engagement())