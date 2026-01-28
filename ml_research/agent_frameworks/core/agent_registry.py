"""
Framework and agent registry for managing agent types and configurations.

This module provides a singleton registry pattern for registering and
retrieving agent classes and framework configurations.
"""

from typing import Dict, Type, List, Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .base_agent import AgentBase


@dataclass
class FrameworkConfig:
    """Configuration for an agent framework.

    Attributes:
        name: Unique name of the framework
        version: Framework version string
        description: Human-readable description
        default_agent: Default agent ID to use
        settings: Framework-specific settings
        metadata: Additional metadata
        registered_at: When the framework was registered
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    default_agent: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "default_agent": self.default_agent,
            "settings": self.settings,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat()
        }


@dataclass
class AgentRegistration:
    """Registration record for an agent class.

    Attributes:
        agent_id: Unique identifier for the agent
        agent_class: The agent class itself
        framework: Name of the framework this agent belongs to
        description: Human-readable description
        tags: List of tags for categorization
        metadata: Additional metadata
        registered_at: When the agent was registered
    """
    agent_id: str
    agent_class: Type[AgentBase]
    framework: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_class": self.agent_class.__name__,
            "framework": self.framework,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat()
        }


class AgentRegistry:
    """Singleton registry for agents and frameworks.

    This class provides a centralized registry for managing agent classes
    and framework configurations. It uses the singleton pattern to ensure
    a single global registry instance.

    Usage:
        registry = AgentRegistry.get_instance()
        registry.register_agent(MyAgent)
        agent = registry.get_agent("my_agent")
    """

    _instance: Optional["AgentRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AgentRegistry":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._agents: Dict[str, AgentRegistration] = {}
        self._frameworks: Dict[str, FrameworkConfig] = {}
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        """Get the singleton registry instance.

        Returns:
            The global AgentRegistry instance
        """
        return cls()

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (primarily for testing).

        This clears all registered agents and frameworks.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._agents.clear()
                cls._instance._frameworks.clear()

    def register_agent(
        self,
        agent_class: Type[AgentBase],
        agent_id: Optional[str] = None,
        framework: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> AgentRegistration:
        """Register an agent class with the registry.

        Args:
            agent_class: The agent class to register
            agent_id: Optional custom ID (defaults to class's AGENT_ID)
            framework: Optional framework name
            description: Optional description
            tags: Optional list of tags
            metadata: Optional metadata
            overwrite: Whether to overwrite existing registration

        Returns:
            The AgentRegistration record

        Raises:
            ValueError: If agent_id already exists and overwrite=False
            TypeError: If agent_class doesn't inherit from AgentBase
        """
        if not issubclass(agent_class, AgentBase):
            raise TypeError(f"{agent_class} must inherit from AgentBase")

        # Get agent ID from class if not provided
        final_id = agent_id or getattr(agent_class, 'AGENT_ID', agent_class.__name__.lower())

        # Check for existing registration
        if final_id in self._agents and not overwrite:
            raise ValueError(f"Agent '{final_id}' is already registered")

        # Create registration
        registration = AgentRegistration(
            agent_id=final_id,
            agent_class=agent_class,
            framework=framework,
            description=description or agent_class.__doc__ or "",
            tags=tags or [],
            metadata=metadata or {}
        )

        self._agents[final_id] = registration
        return registration

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry.

        Args:
            agent_id: The ID of the agent to remove

        Returns:
            True if the agent was removed, False if it wasn't registered
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get_agent(
        self,
        agent_id: str,
        instantiate: bool = True
    ) -> Optional[AgentBase | Type[AgentBase]]:
        """Get an agent by ID.

        Args:
            agent_id: The ID of the agent to retrieve
            instantiate: If True, return an instance; if False, return the class

        Returns:
            The agent instance or class, or None if not found
        """
        registration = self._agents.get(agent_id)
        if registration is None:
            return None

        if instantiate:
            return registration.agent_class()
        return registration.agent_class

    def get_agent_registration(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get the registration record for an agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            The AgentRegistration or None if not found
        """
        return self._agents.get(agent_id)

    def list_agents(
        self,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[AgentRegistration]:
        """List registered agents with optional filtering.

        Args:
            framework: Filter by framework name
            tags: Filter by tags (agents must have all specified tags)

        Returns:
            List of matching AgentRegistration records
        """
        results = list(self._agents.values())

        if framework is not None:
            results = [r for r in results if r.framework == framework]

        if tags:
            results = [r for r in results if all(t in r.tags for t in tags)]

        return results

    def register_framework(
        self,
        name: str,
        config: Optional[FrameworkConfig] = None,
        version: str = "1.0.0",
        description: str = "",
        default_agent: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False
    ) -> FrameworkConfig:
        """Register a framework configuration.

        Args:
            name: Unique framework name
            config: Optional pre-built FrameworkConfig
            version: Framework version
            description: Human-readable description
            default_agent: Default agent ID for this framework
            settings: Framework settings
            metadata: Additional metadata
            overwrite: Whether to overwrite existing registration

        Returns:
            The FrameworkConfig

        Raises:
            ValueError: If framework name already exists and overwrite=False
        """
        if name in self._frameworks and not overwrite:
            raise ValueError(f"Framework '{name}' is already registered")

        if config is not None:
            framework_config = config
        else:
            framework_config = FrameworkConfig(
                name=name,
                version=version,
                description=description,
                default_agent=default_agent,
                settings=settings or {},
                metadata=metadata or {}
            )

        self._frameworks[name] = framework_config
        return framework_config

    def unregister_framework(self, name: str) -> bool:
        """Remove a framework from the registry.

        Args:
            name: The name of the framework to remove

        Returns:
            True if the framework was removed, False if it wasn't registered
        """
        if name in self._frameworks:
            del self._frameworks[name]
            return True
        return False

    def get_framework(self, name: str) -> Optional[FrameworkConfig]:
        """Get a framework configuration by name.

        Args:
            name: The name of the framework

        Returns:
            The FrameworkConfig or None if not found
        """
        return self._frameworks.get(name)

    def list_frameworks(self) -> List[FrameworkConfig]:
        """List all registered frameworks.

        Returns:
            List of all FrameworkConfig records
        """
        return list(self._frameworks.values())

    def get_default_agent(self, framework: str) -> Optional[AgentBase]:
        """Get the default agent for a framework.

        Args:
            framework: The framework name

        Returns:
            The default agent instance or None
        """
        config = self.get_framework(framework)
        if config is None or config.default_agent is None:
            return None
        return self.get_agent(config.default_agent)


# Module-level convenience functions
def get_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    return AgentRegistry.get_instance()


T = TypeVar('T', bound=Type[AgentBase])


def register(
    agent_id: Optional[str] = None,
    framework: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[T], T]:
    """Decorator for auto-registering agent classes.

    Usage:
        @register(framework="my_framework", tags=["planning"])
        class MyAgent(AgentBase):
            AGENT_ID = "my_agent"
            ...

    Args:
        agent_id: Optional custom ID (defaults to class's AGENT_ID)
        framework: Optional framework name
        description: Optional description
        tags: Optional list of tags
        metadata: Optional metadata

    Returns:
        A decorator function that registers the class and returns it unchanged
    """
    def decorator(cls: T) -> T:
        registry = get_registry()
        registry.register_agent(
            agent_class=cls,
            agent_id=agent_id,
            framework=framework,
            description=description,
            tags=tags,
            metadata=metadata
        )
        return cls
    return decorator


# Alias for the decorator
register_agent = register
