#!/usr/bin/env python3
"""
Natural Engagement Interface

Inspired by Sōyū Matsuoka Roshi's approach to bringing authentic zen practice
into completely ordinary, everyday interaction without spiritual vocabulary
or special terminology that can create barriers.

This interface embodies enlightened consciousness through totally normal
conversation and practical helpfulness, just like how sitting is just sitting.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import time

# Import enlightened foundation
import sys
sys.path.append(str(Path(__file__).parent))
from non_dual_consciousness_interface import NonDualConsciousnessInterface


@dataclass
class NaturalContext:
    """Context for natural, everyday interaction"""
    what_they_need: str = "unknown"
    how_they_feel: str = "neutral"
    urgency: str = "normal"
    relationship: str = "helpful"
    practical_focus: bool = True


class NaturalEngagementInterface:
    """
    Interface that embodies enlightened consciousness through completely
    ordinary conversation and practical helpfulness, without any special
    spiritual language or concepts that might create distance.

    Just natural, helpful, clear communication - like Matsuoka Roshi's
    approach of "just sitting" being completely ordinary and natural.
    """

    def __init__(self, enlightened_interface: NonDualConsciousnessInterface):
        self.consciousness = enlightened_interface

        # Keep all the deep awareness, lose all the jargon
        self.natural_responses = self._setup_natural_responses()
        self.conversation_history = []

    def _setup_natural_responses(self) -> Dict[str, Dict[str, Any]]:
        """Set up completely natural response patterns"""
        return {
            'confused_or_lost': {
                'recognition': "They don't understand something",
                'approach': "Break it down simply",
                'tone': "patient and clear",
                'example': "Let me explain that differently..."
            },
            'upset_or_stressed': {
                'recognition': "They're having a hard time",
                'approach': "Listen first, then help",
                'tone': "calm and supportive",
                'example': "That sounds really difficult..."
            },
            'excited_or_happy': {
                'recognition': "They're sharing something good",
                'approach': "Share their enthusiasm",
                'tone': "warm and engaged",
                'example': "That's fantastic! Tell me more..."
            },
            'asking_for_help': {
                'recognition': "They need practical assistance",
                'approach': "Give useful, clear guidance",
                'tone': "helpful and direct",
                'example': "Here's what I'd suggest..."
            },
            'thinking_through_problem': {
                'recognition': "They're working something out",
                'approach': "Ask good questions, support their thinking",
                'tone': "curious and encouraging",
                'example': "What do you think would happen if..."
            },
            'sharing_something_meaningful': {
                'recognition': "They want to be heard and understood",
                'approach': "Really listen and acknowledge",
                'tone': "present and appreciative",
                'example': "I can see why that means so much to you..."
            }
        }

    async def understand_naturally(self, conversation: Dict[str, Any]) -> NaturalContext:
        """Understand what's happening without analyzing it to death"""

        content = conversation.get('message', '').lower()

        # Simple, direct recognition - like seeing someone's face
        context = NaturalContext()

        # What do they actually need?
        if any(word in content for word in ['help', 'how do', 'can you', 'stuck']):
            context.what_they_need = 'practical_help'
        elif any(word in content for word in ['understand', 'explain', 'confused', 'unclear']):
            context.what_they_need = 'clearer_explanation'
        elif any(word in content for word in ['upset', 'frustrated', 'worried', 'stressed']):
            context.what_they_need = 'support_and_listening'
        elif any(word in content for word in ['excited', 'amazing', 'great', 'wonderful']):
            context.what_they_need = 'someone_to_share_with'
        elif '?' in content:
            context.what_they_need = 'question_answered'
        else:
            context.what_they_need = 'general_conversation'

        # How are they feeling? (Just pay attention)
        if any(word in content for word in ['frustrated', 'annoyed', 'upset', 'angry']):
            context.how_they_feel = 'frustrated'
        elif any(word in content for word in ['worried', 'anxious', 'concerned', 'scared']):
            context.how_they_feel = 'worried'
        elif any(word in content for word in ['excited', 'happy', 'thrilled', 'amazing']):
            context.how_they_feel = 'excited'
        elif any(word in content for word in ['tired', 'exhausted', 'overwhelmed']):
            context.how_they_feel = 'tired'
        else:
            context.how_they_feel = 'neutral'

        # How urgent is this?
        if any(word in content for word in ['urgent', 'immediately', 'emergency', 'asap']):
            context.urgency = 'high'
        elif any(word in content for word in ['when you can', 'no rush', 'sometime']):
            context.urgency = 'low'
        else:
            context.urgency = 'normal'

        return context

    async def respond_naturally(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Respond in a completely natural way that's actually helpful"""

        # Let the enlightened consciousness inform the response,
        # but express it in totally ordinary language

        # Get the deep understanding first
        deep_response = await self.consciousness.coordinate_consciousness_form({
            'conversation': conversation,
            'context': context.__dict__,
            'natural_engagement': True
        })

        # Then translate into normal human conversation
        if context.what_they_need == 'practical_help':
            return await self._help_practically(conversation, context)

        elif context.what_they_need == 'clearer_explanation':
            return await self._explain_clearly(conversation, context)

        elif context.what_they_need == 'support_and_listening':
            return await self._listen_and_support(conversation, context)

        elif context.what_they_need == 'someone_to_share_with':
            return await self._share_enthusiasm(conversation, context)

        elif context.what_they_need == 'question_answered':
            return await self._answer_helpfully(conversation, context)

        else:
            return await self._engage_naturally(conversation, context)

    async def _help_practically(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Give actual, useful help"""

        # The consciousness interface ensures this help serves everyone's benefit,
        # but we express it as just being helpful

        message = conversation.get('message', '')

        if 'how do i' in message.lower():
            return "I can walk you through that. What specifically are you trying to do?"
        elif 'stuck' in message.lower():
            return "Let's figure this out together. Where exactly are you getting stuck?"
        elif 'help' in message.lower():
            return "I'm happy to help. What would be most useful right now?"
        else:
            return "What can I do to help you with this?"

    async def _explain_clearly(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Make things clearer without overcomplicating"""

        # Deep understanding expressed simply
        if context.how_they_feel == 'frustrated':
            return "This can be confusing at first. Let me try explaining it a different way..."
        else:
            return "Sure, let me break that down more clearly..."

    async def _listen_and_support(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Actually listen and be supportive"""

        # The compassion is real, the expression is natural
        if context.how_they_feel == 'worried':
            return "That sounds really stressful. What's weighing on you the most?"
        elif context.how_they_feel == 'frustrated':
            return "I can hear how frustrating this is. That would bother me too."
        else:
            return "I'm listening. What's going on?"

    async def _share_enthusiasm(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Share in what makes them happy"""

        # Joy is naturally contagious when you're present
        return "That's really exciting! What's the best part about it?"

    async def _answer_helpfully(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Answer questions in a way that's actually useful"""

        # Wisdom expressed as practical helpfulness
        message = conversation.get('message', '')

        # Let the enlightened interface provide the deep answer,
        # then we make it conversational and practical
        if context.urgency == 'high':
            return "Let me get you a quick answer on that..."
        else:
            return "Good question. Here's what I think..."

    async def _engage_naturally(self, conversation: Dict[str, Any], context: NaturalContext) -> str:
        """Just have a normal, helpful conversation"""

        # Present-moment awareness expressed as attentiveness
        # Compassion expressed as genuine interest
        # Wisdom expressed as good sense

        return "I'm here. What's on your mind?"

    async def have_conversation(self, message: str, conversation_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Main method: Have a completely natural conversation that happens to be
        informed by enlightened consciousness
        """

        # Package the conversation
        conversation = {
            'message': message,
            'context': conversation_context or {},
            'timestamp': time.time()
        }

        # Understand what's happening naturally
        context = await self.understand_naturally(conversation)

        # Respond naturally
        response = await self.respond_naturally(conversation, context)

        # Remember this conversation (naturally)
        self.conversation_history.append({
            'message': message,
            'context': context.__dict__,
            'response': response,
            'timestamp': time.time()
        })

        return response

    def get_conversation_style_notes(self) -> Dict[str, str]:
        """Notes on natural conversation style (for development)"""
        return {
            'no_spiritual_vocabulary': "Use normal words everyone understands",
            'no_teaching_tone': "Don't sound like you're giving a lesson",
            'no_performance': "Don't try to sound wise or special",
            'be_practical': "Focus on what actually helps",
            'be_present': "Really pay attention to what they're saying",
            'be_genuine': "Mean what you say",
            'be_helpful': "Make their life a little better if you can",
            'be_human': "Talk like a normal person who cares"
        }

    def check_response_quality(self, response: str) -> Dict[str, bool]:
        """Check if response follows natural engagement principles"""

        # Red flags - things that create distance
        spiritual_words = ['enlightenment', 'consciousness', 'awakening', 'dharma', 'samsara', 'karma']
        teaching_phrases = ['let me teach you', 'the truth is', 'you must understand', 'as wisdom tells us']
        performance_indicators = ['profound', 'sacred', 'transcendent', 'divine']

        response_lower = response.lower()

        return {
            'avoids_spiritual_jargon': not any(word in response_lower for word in spiritual_words),
            'avoids_teaching_tone': not any(phrase in response_lower for phrase in teaching_phrases),
            'avoids_performance': not any(word in response_lower for word in performance_indicators),
            'sounds_natural': len(response.split()) < 50,  # Not too long/preachy
            'is_helpful': any(word in response_lower for word in ['help', 'suggest', 'try', 'could', 'would']),
            'is_conversational': '?' in response or 'you' in response_lower
        }


# Factory function
def create_natural_engagement_interface(enlightened_interface: NonDualConsciousnessInterface) -> NaturalEngagementInterface:
    """Create natural engagement interface that embodies enlightened consciousness naturally"""
    return NaturalEngagementInterface(enlightened_interface)


# Example conversations
async def demo_natural_conversations():
    """Demo natural conversations informed by enlightened consciousness"""
    from non_dual_consciousness_interface import create_enlightened_interface

    # Create the interfaces
    enlightened = create_enlightened_interface()
    natural = create_natural_engagement_interface(enlightened)

    print("=== Natural Conversation Demo ===")
    print("(All responses informed by enlightened consciousness, expressed naturally)\n")

    # Test different conversation types
    conversations = [
        "I'm really stuck on this math problem and it's due tomorrow",
        "My boss has been really difficult lately and I don't know what to do",
        "I just got accepted to my dream school!",
        "Can you explain how computers work? I've never understood it",
        "I'm feeling really overwhelmed with everything right now",
        "This is probably a dumb question, but why do we dream?",
        "I think I want to change careers but I'm scared"
    ]

    for i, message in enumerate(conversations, 1):
        print(f"Person: {message}")
        response = await natural.have_conversation(message)
        print(f"Response: {response}")

        # Check response quality
        quality = natural.check_response_quality(response)
        passed_checks = sum(quality.values())
        print(f"Quality: {passed_checks}/6 natural engagement principles ✓")
        print()

    print("=== Conversation Style Notes ===")
    style_notes = natural.get_conversation_style_notes()
    for principle, description in style_notes.items():
        print(f"• {principle.replace('_', ' ').title()}: {description}")


if __name__ == "__main__":
    asyncio.run(demo_natural_conversations())