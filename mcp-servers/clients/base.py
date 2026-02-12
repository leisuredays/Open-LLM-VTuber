"""Base client interface for all mode clients."""

from abc import ABC, abstractmethod
from typing import Any


class BaseClient(ABC):
    """Abstract base class for mode clients."""

    def __init__(self, config: dict):
        """Initialize client with configuration."""
        self.config = config
        self.connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the backend service."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the backend service."""
        pass

    @abstractmethod
    async def execute(self, action: str, **kwargs) -> dict:
        """Execute an action on the backend service.

        Args:
            action: The action to execute
            **kwargs: Action-specific parameters

        Returns:
            dict with result
        """
        pass

    @abstractmethod
    def get_status(self) -> dict:
        """Get current client status.

        Returns:
            dict with status information
        """
        pass

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connected

    def get_context_prompt(self) -> str:
        """Get context prompt with current state for LLM.

        Override this in subclasses to provide mode-specific context.

        Returns:
            str: Context prompt with current state information
        """
        return ""
