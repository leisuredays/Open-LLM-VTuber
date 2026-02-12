"""Base mode configuration and interface."""

from dataclasses import dataclass, field
from typing import Callable, Any
import json


@dataclass
class ModeConfig:
    """Configuration for a mode."""

    id: str
    name: str
    description: str
    client_type: str  # Which client to use (e.g., "minecraft")
    system_prompt: str
    trigger_keywords: list[str] = field(default_factory=list)
    exit_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "client_type": self.client_type,
            "system_prompt": self.system_prompt,
            "trigger_keywords": self.trigger_keywords,
            "exit_keywords": self.exit_keywords,
        }

    def matches_trigger(self, message: str) -> bool:
        """Check if message matches any trigger keyword."""
        msg_lower = message.lower()
        return any(kw.lower() in msg_lower for kw in self.trigger_keywords)

    def matches_exit(self, message: str) -> bool:
        """Check if message matches any exit keyword."""
        msg_lower = message.lower()
        return any(kw.lower() in msg_lower for kw in self.exit_keywords)


@dataclass
class ModeTools:
    """Tools provided by a mode."""

    # List of (tool_name, tool_function, description) tuples
    tools: list[tuple[str, Callable, str]] = field(default_factory=list)

    def register_tool(self, name: str, func: Callable, description: str) -> None:
        """Register a tool."""
        self.tools.append((name, func, description))

    def get_tools(self) -> list[tuple[str, Callable, str]]:
        """Get all registered tools."""
        return self.tools


class BaseMode:
    """Base class for modes. Subclass this to create new modes."""

    def __init__(self, config: ModeConfig, client: Any):
        """Initialize mode with config and client.

        Args:
            config: Mode configuration
            client: The client instance for this mode
        """
        self.config = config
        self.client = client
        self._tools = ModeTools()
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Override this to set up mode-specific tools."""
        pass

    def get_tools(self) -> list[tuple[str, Callable, str]]:
        """Get mode tools."""
        return self._tools.get_tools()

    async def execute(self, action: str, **kwargs) -> dict:
        """Execute an action via the client."""
        if not self.client:
            return {"success": False, "error": "No client available"}
        return await self.client.execute(action, **kwargs)

    def get_status(self) -> dict:
        """Get mode/client status."""
        if not self.client:
            return {"success": False, "error": "No client available"}
        return self.client.get_status()


def json_response(data: dict) -> str:
    """Helper to format JSON response."""
    return json.dumps(data, indent=2, ensure_ascii=False)
