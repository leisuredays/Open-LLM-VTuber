"""Client registry for Router MCP Server."""

from typing import Type
from .base import BaseClient
from .minecraft import MindcraftClient
from .mindserver_process import MindServerProcessManager, ProcessConfig, LaunchCommand

# Client registry - maps client type to class
CLIENT_REGISTRY: dict[str, Type[BaseClient]] = {
    "minecraft": MindcraftClient,
}


def get_client_class(client_type: str) -> Type[BaseClient] | None:
    """Get client class by type name."""
    return CLIENT_REGISTRY.get(client_type)


def register_client(client_type: str, client_class: Type[BaseClient]) -> None:
    """Register a new client type."""
    CLIENT_REGISTRY[client_type] = client_class


def list_client_types() -> list[str]:
    """List all registered client types."""
    return list(CLIENT_REGISTRY.keys())


__all__ = [
    "BaseClient",
    "MindcraftClient",
    "MindServerProcessManager",
    "ProcessConfig",
    "LaunchCommand",
    "CLIENT_REGISTRY",
    "get_client_class",
    "register_client",
    "list_client_types",
]
