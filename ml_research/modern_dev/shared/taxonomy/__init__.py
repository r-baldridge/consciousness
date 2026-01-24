"""
Bug taxonomy for code repair models.
"""

from .bug_types import (
    BugType,
    BugCategory,
    BugInfo,
    BUG_TAXONOMY,
    get_bug_info,
    get_bugs_by_category,
    get_bugs_by_difficulty,
)

__all__ = [
    'BugType',
    'BugCategory',
    'BugInfo',
    'BUG_TAXONOMY',
    'get_bug_info',
    'get_bugs_by_category',
    'get_bugs_by_difficulty',
]
