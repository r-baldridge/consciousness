"""
Form 14: Global Workspace Theory (GWT) Consciousness Interface

Implements Baars/Dehaene's GWT for conscious access and broadcasting.
"""

from .global_workspace_interface import (
    # Enums
    WorkspaceState,
    ContentType,
    ProcessorType,
    BroadcastStrength,
    CompetitionOutcome,
    # Input dataclasses
    WorkspaceContent,
    ProcessorRegistration,
    # Output dataclasses
    BroadcastEvent,
    CompetitionResult,
    WorkspaceSnapshot,
    GWTSystemStatus,
    # Engines
    WorkspaceCompetitionEngine,
    BroadcastEngine,
    # Main interface
    GlobalWorkspaceInterface,
    # Convenience functions
    create_global_workspace_interface,
    create_workspace_content,
)

__all__ = [
    # Enums
    "WorkspaceState",
    "ContentType",
    "ProcessorType",
    "BroadcastStrength",
    "CompetitionOutcome",
    # Input dataclasses
    "WorkspaceContent",
    "ProcessorRegistration",
    # Output dataclasses
    "BroadcastEvent",
    "CompetitionResult",
    "WorkspaceSnapshot",
    "GWTSystemStatus",
    # Engines
    "WorkspaceCompetitionEngine",
    "BroadcastEngine",
    # Main interface
    "GlobalWorkspaceInterface",
    # Convenience functions
    "create_global_workspace_interface",
    "create_workspace_content",
]
