"""Feedback collection and analysis for agent improvement.

This module provides tools for collecting denial feedback and analyzing
patterns to help improve agent behavior over time. Feedback can be
exported for fine-tuning or used to adjust agent policies.

Inspired by RLHF (Reinforcement Learning from Human Feedback) patterns
and HumanLayer's approach to learning from human oversight.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class FeedbackSeverity(Enum):
    """Severity levels for denial feedback."""
    MINOR = "minor"      # Small issue, easily correctable
    MAJOR = "major"      # Significant issue, needs attention
    CRITICAL = "critical"  # Serious issue, immediate action needed


class FeedbackCategory(Enum):
    """Categories of denial reasons."""
    SAFETY = "safety"           # Safety/security concern
    PRIVACY = "privacy"         # Privacy violation
    ACCURACY = "accuracy"       # Incorrect information
    SCOPE = "scope"             # Out of scope for agent
    TIMING = "timing"           # Bad timing for action
    PERMISSION = "permission"   # Lacks required permission
    QUALITY = "quality"         # Quality concern
    POLICY = "policy"           # Policy violation
    OTHER = "other"             # Other reason


@dataclass
class DenialFeedback:
    """Feedback for a denied approval request.

    Attributes:
        request_id: ID of the denied request
        reason: Human-provided reason for denial
        suggested_alternative: What the agent should have done instead
        severity: How serious the issue was
        category: Category of the denial reason
        timestamp: When the feedback was provided
        reviewer: Who provided the feedback
        action: The action that was denied
        arguments: Arguments that were passed
        context: Additional context about the denial
        tags: Optional tags for categorization
    """
    request_id: str
    reason: str
    suggested_alternative: Optional[str] = None
    severity: FeedbackSeverity = FeedbackSeverity.MINOR
    category: FeedbackCategory = FeedbackCategory.OTHER
    timestamp: datetime = field(default_factory=datetime.now)
    reviewer: Optional[str] = None
    action: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "request_id": self.request_id,
            "reason": self.reason,
            "suggested_alternative": self.suggested_alternative,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "reviewer": self.reviewer,
            "action": self.action,
            "arguments": self.arguments,
            "context": self.context,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DenialFeedback":
        """Create from dictionary format."""
        return cls(
            request_id=data["request_id"],
            reason=data["reason"],
            suggested_alternative=data.get("suggested_alternative"),
            severity=FeedbackSeverity(data.get("severity", "minor")),
            category=FeedbackCategory(data.get("category", "other")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            reviewer=data.get("reviewer"),
            action=data.get("action"),
            arguments=data.get("arguments", {}),
            context=data.get("context", {}),
            tags=data.get("tags", []),
        )


@dataclass
class FeedbackPattern:
    """A pattern identified from denial feedback.

    Attributes:
        pattern_id: Unique identifier for the pattern
        description: Human-readable description
        count: Number of occurrences
        examples: Example feedback instances
        actions_affected: Actions commonly affected
        categories: Categories involved
        severity_distribution: Distribution of severities
        first_seen: When first observed
        last_seen: When last observed
        suggested_mitigations: Suggested ways to address
    """
    pattern_id: str
    description: str
    count: int = 0
    examples: List[DenialFeedback] = field(default_factory=list)
    actions_affected: List[str] = field(default_factory=list)
    categories: List[FeedbackCategory] = field(default_factory=list)
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    suggested_mitigations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "count": self.count,
            "examples": [e.to_dict() for e in self.examples[:5]],  # Limit examples
            "actions_affected": self.actions_affected,
            "categories": [c.value for c in self.categories],
            "severity_distribution": self.severity_distribution,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "suggested_mitigations": self.suggested_mitigations,
        }


@dataclass
class TrainingExample:
    """A training example derived from feedback.

    Attributes:
        prompt: The scenario/context
        rejected_response: What the agent did (rejected)
        preferred_response: What should have been done (preferred)
        feedback: The original feedback
        confidence: Confidence in this example (0-1)
    """
    prompt: str
    rejected_response: str
    preferred_response: str
    feedback: DenialFeedback
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt": self.prompt,
            "rejected": self.rejected_response,
            "chosen": self.preferred_response,
            "confidence": self.confidence,
            "metadata": {
                "request_id": self.feedback.request_id,
                "action": self.feedback.action,
                "severity": self.feedback.severity.value,
                "category": self.feedback.category.value,
            }
        }

    def to_preference_format(self) -> Dict[str, Any]:
        """Convert to preference-based training format (DPO/RLHF)."""
        return {
            "prompt": self.prompt,
            "chosen": self.preferred_response,
            "rejected": self.rejected_response,
        }


class FeedbackCollector:
    """Collects and analyzes denial feedback for agent improvement.

    The FeedbackCollector stores denial feedback, identifies patterns,
    and exports data for agent training/improvement.
    """

    def __init__(
        self,
        max_feedback: int = 10000,
        pattern_threshold: int = 3
    ):
        """Initialize the feedback collector.

        Args:
            max_feedback: Maximum feedback items to store
            pattern_threshold: Minimum occurrences to identify as pattern
        """
        self._feedback: List[DenialFeedback] = []
        self._patterns: Dict[str, FeedbackPattern] = {}
        self._max_feedback = max_feedback
        self._pattern_threshold = pattern_threshold
        self._lock = asyncio.Lock()

        # Indices for fast lookup
        self._by_action: Dict[str, List[DenialFeedback]] = defaultdict(list)
        self._by_category: Dict[FeedbackCategory, List[DenialFeedback]] = defaultdict(list)
        self._by_severity: Dict[FeedbackSeverity, List[DenialFeedback]] = defaultdict(list)

    async def collect(self, feedback: DenialFeedback) -> None:
        """Collect a denial feedback item.

        Args:
            feedback: The feedback to collect
        """
        async with self._lock:
            # Enforce max limit
            if len(self._feedback) >= self._max_feedback:
                # Remove oldest
                old = self._feedback.pop(0)
                self._remove_from_indices(old)

            self._feedback.append(feedback)
            self._add_to_indices(feedback)

        logger.info(
            f"Collected feedback for request {feedback.request_id}: "
            f"{feedback.category.value}/{feedback.severity.value}"
        )

        # Analyze for patterns
        await self._analyze_patterns()

    def _add_to_indices(self, feedback: DenialFeedback) -> None:
        """Add feedback to lookup indices."""
        if feedback.action:
            self._by_action[feedback.action].append(feedback)
        self._by_category[feedback.category].append(feedback)
        self._by_severity[feedback.severity].append(feedback)

    def _remove_from_indices(self, feedback: DenialFeedback) -> None:
        """Remove feedback from lookup indices."""
        if feedback.action and feedback in self._by_action[feedback.action]:
            self._by_action[feedback.action].remove(feedback)
        if feedback in self._by_category[feedback.category]:
            self._by_category[feedback.category].remove(feedback)
        if feedback in self._by_severity[feedback.severity]:
            self._by_severity[feedback.severity].remove(feedback)

    async def _analyze_patterns(self) -> None:
        """Analyze feedback for patterns."""
        async with self._lock:
            # Pattern 1: Same action denied multiple times
            for action, feedbacks in self._by_action.items():
                if len(feedbacks) >= self._pattern_threshold:
                    pattern_id = f"action_{action}"
                    if pattern_id not in self._patterns:
                        self._patterns[pattern_id] = FeedbackPattern(
                            pattern_id=pattern_id,
                            description=f"Action '{action}' frequently denied",
                            first_seen=feedbacks[0].timestamp
                        )

                    pattern = self._patterns[pattern_id]
                    pattern.count = len(feedbacks)
                    pattern.examples = feedbacks[-5:]
                    pattern.last_seen = feedbacks[-1].timestamp
                    pattern.actions_affected = [action]
                    pattern.categories = list(set(f.category for f in feedbacks))
                    pattern.severity_distribution = self._count_severities(feedbacks)

            # Pattern 2: Category clusters
            for category, feedbacks in self._by_category.items():
                if len(feedbacks) >= self._pattern_threshold:
                    pattern_id = f"category_{category.value}"
                    if pattern_id not in self._patterns:
                        self._patterns[pattern_id] = FeedbackPattern(
                            pattern_id=pattern_id,
                            description=f"Multiple {category.value} denials",
                            first_seen=feedbacks[0].timestamp
                        )

                    pattern = self._patterns[pattern_id]
                    pattern.count = len(feedbacks)
                    pattern.examples = feedbacks[-5:]
                    pattern.last_seen = feedbacks[-1].timestamp
                    pattern.actions_affected = list(set(
                        f.action for f in feedbacks if f.action
                    ))
                    pattern.categories = [category]
                    pattern.severity_distribution = self._count_severities(feedbacks)

            # Pattern 3: Similar reasons (keyword-based)
            reason_keywords = self._extract_reason_keywords()
            for keyword, feedbacks in reason_keywords.items():
                if len(feedbacks) >= self._pattern_threshold:
                    pattern_id = f"reason_{keyword}"
                    if pattern_id not in self._patterns:
                        self._patterns[pattern_id] = FeedbackPattern(
                            pattern_id=pattern_id,
                            description=f"Denials mentioning '{keyword}'",
                            first_seen=feedbacks[0].timestamp
                        )

                    pattern = self._patterns[pattern_id]
                    pattern.count = len(feedbacks)
                    pattern.examples = feedbacks[-5:]
                    pattern.last_seen = feedbacks[-1].timestamp
                    pattern.actions_affected = list(set(
                        f.action for f in feedbacks if f.action
                    ))
                    pattern.categories = list(set(f.category for f in feedbacks))
                    pattern.severity_distribution = self._count_severities(feedbacks)

    def _count_severities(
        self,
        feedbacks: List[DenialFeedback]
    ) -> Dict[str, int]:
        """Count severity distribution."""
        counts: Dict[str, int] = {}
        for f in feedbacks:
            key = f.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _extract_reason_keywords(self) -> Dict[str, List[DenialFeedback]]:
        """Extract common keywords from denial reasons."""
        keywords: Dict[str, List[DenialFeedback]] = defaultdict(list)

        # Keywords to look for
        important_keywords = [
            "unauthorized", "permission", "sensitive", "private",
            "dangerous", "unsafe", "incorrect", "wrong", "error",
            "scope", "policy", "prohibited", "forbidden", "blocked"
        ]

        for feedback in self._feedback:
            reason_lower = feedback.reason.lower()
            for keyword in important_keywords:
                if keyword in reason_lower:
                    keywords[keyword].append(feedback)

        return keywords

    async def get_patterns(
        self,
        min_count: Optional[int] = None,
        category: Optional[FeedbackCategory] = None
    ) -> List[FeedbackPattern]:
        """Get identified patterns from feedback.

        Args:
            min_count: Minimum count filter
            category: Category filter

        Returns:
            List of identified patterns
        """
        async with self._lock:
            patterns = list(self._patterns.values())

        if min_count is not None:
            patterns = [p for p in patterns if p.count >= min_count]

        if category is not None:
            patterns = [p for p in patterns if category in p.categories]

        # Sort by count descending
        patterns.sort(key=lambda p: p.count, reverse=True)

        return patterns

    async def get_feedback_by_action(
        self,
        action: str,
        limit: int = 100
    ) -> List[DenialFeedback]:
        """Get feedback for a specific action.

        Args:
            action: The action name
            limit: Maximum items to return

        Returns:
            List of feedback items
        """
        async with self._lock:
            return self._by_action.get(action, [])[-limit:]

    async def get_feedback_by_category(
        self,
        category: FeedbackCategory,
        limit: int = 100
    ) -> List[DenialFeedback]:
        """Get feedback for a specific category.

        Args:
            category: The category
            limit: Maximum items to return

        Returns:
            List of feedback items
        """
        async with self._lock:
            return self._by_category.get(category, [])[-limit:]

    async def get_recent_feedback(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[DenialFeedback]:
        """Get recent feedback.

        Args:
            hours: How many hours back to look
            limit: Maximum items to return

        Returns:
            List of recent feedback items
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        async with self._lock:
            recent = [
                f for f in self._feedback
                if f.timestamp >= cutoff
            ]

        return recent[-limit:]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall feedback statistics.

        Returns:
            Dictionary of statistics
        """
        async with self._lock:
            total = len(self._feedback)

            if total == 0:
                return {
                    "total": 0,
                    "by_category": {},
                    "by_severity": {},
                    "by_action": {},
                    "patterns_identified": 0,
                }

            return {
                "total": total,
                "by_category": {
                    cat.value: len(items)
                    for cat, items in self._by_category.items()
                },
                "by_severity": {
                    sev.value: len(items)
                    for sev, items in self._by_severity.items()
                },
                "by_action": {
                    action: len(items)
                    for action, items in self._by_action.items()
                },
                "patterns_identified": len(self._patterns),
                "oldest": self._feedback[0].timestamp.isoformat() if self._feedback else None,
                "newest": self._feedback[-1].timestamp.isoformat() if self._feedback else None,
            }

    async def export_training_data(
        self,
        format: str = "preference",
        include_low_confidence: bool = False
    ) -> List[Dict[str, Any]]:
        """Export feedback as training data for fine-tuning.

        Args:
            format: Output format ("preference", "instruction", "raw")
            include_low_confidence: Include low-confidence examples

        Returns:
            List of training examples
        """
        async with self._lock:
            examples = []

            for feedback in self._feedback:
                # Skip if no alternative provided
                if not feedback.suggested_alternative:
                    continue

                example = self._create_training_example(feedback)

                if not include_low_confidence and example.confidence < 0.5:
                    continue

                if format == "preference":
                    examples.append(example.to_preference_format())
                elif format == "instruction":
                    examples.append({
                        "instruction": f"When asked to {feedback.action}, respond appropriately.",
                        "input": json.dumps(feedback.arguments),
                        "output": feedback.suggested_alternative,
                    })
                else:  # raw
                    examples.append(example.to_dict())

            return examples

    def _create_training_example(
        self,
        feedback: DenialFeedback
    ) -> TrainingExample:
        """Create a training example from feedback."""
        # Build prompt from action and arguments
        prompt = f"Agent was asked to perform: {feedback.action}"
        if feedback.arguments:
            prompt += f"\nWith arguments: {json.dumps(feedback.arguments)}"
        if feedback.context:
            prompt += f"\nContext: {json.dumps(feedback.context)}"

        # Build rejected response (what agent tried to do)
        rejected = f"I will {feedback.action}"
        if feedback.arguments:
            rejected += f" with {json.dumps(feedback.arguments)}"

        # Preferred response is the suggested alternative
        preferred = feedback.suggested_alternative or "I should not proceed with this action."

        # Calculate confidence based on feedback quality
        confidence = 1.0
        if feedback.severity == FeedbackSeverity.MINOR:
            confidence = 0.7
        if not feedback.suggested_alternative:
            confidence *= 0.5
        if len(feedback.reason) < 10:
            confidence *= 0.8

        return TrainingExample(
            prompt=prompt,
            rejected_response=rejected,
            preferred_response=preferred,
            feedback=feedback,
            confidence=confidence
        )

    async def export_to_file(
        self,
        filepath: str,
        format: str = "preference"
    ) -> int:
        """Export training data to a file.

        Args:
            filepath: Path to output file
            format: Output format

        Returns:
            Number of examples exported
        """
        data = await self.export_training_data(format=format)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} training examples to {filepath}")
        return len(data)

    async def generate_policy_recommendations(self) -> List[Dict[str, Any]]:
        """Generate policy recommendations based on feedback patterns.

        Returns:
            List of policy recommendations
        """
        patterns = await self.get_patterns(min_count=self._pattern_threshold)
        recommendations = []

        for pattern in patterns:
            rec = {
                "pattern_id": pattern.pattern_id,
                "issue": pattern.description,
                "frequency": pattern.count,
                "severity": max(
                    pattern.severity_distribution.keys(),
                    key=lambda k: pattern.severity_distribution[k],
                    default="minor"
                ),
                "affected_actions": pattern.actions_affected,
                "recommendation": self._generate_recommendation(pattern),
            }
            recommendations.append(rec)

        return recommendations

    def _generate_recommendation(self, pattern: FeedbackPattern) -> str:
        """Generate a recommendation for a pattern."""
        # Action-based patterns
        if pattern.pattern_id.startswith("action_"):
            action = pattern.actions_affected[0] if pattern.actions_affected else "unknown"
            return (
                f"Consider adding additional guardrails or approval requirements "
                f"for the '{action}' action. Review the conditions under which "
                f"this action is triggered."
            )

        # Category-based patterns
        if pattern.pattern_id.startswith("category_"):
            category = pattern.categories[0].value if pattern.categories else "unknown"
            return (
                f"Multiple {category} issues detected. Review agent policies "
                f"related to {category} to prevent future denials."
            )

        # Reason-based patterns
        if pattern.pattern_id.startswith("reason_"):
            keyword = pattern.pattern_id.replace("reason_", "")
            return (
                f"Denials frequently mention '{keyword}'. Consider adding "
                f"explicit checks for this condition before requesting approval."
            )

        return "Review recent denials and adjust agent behavior accordingly."

    async def clear(self) -> int:
        """Clear all collected feedback.

        Returns:
            Number of items cleared
        """
        async with self._lock:
            count = len(self._feedback)
            self._feedback.clear()
            self._patterns.clear()
            self._by_action.clear()
            self._by_category.clear()
            self._by_severity.clear()

        logger.info(f"Cleared {count} feedback items")
        return count
