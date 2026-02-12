"""Mode registry for Router MCP Server."""

from typing import Type
from .base import ModeConfig, BaseMode, ModeTools, json_response
from .minecraft import MinecraftMode, MINECRAFT_MODE_CONFIG

# Mode config registry - maps mode_id to ModeConfig
MODE_CONFIGS: dict[str, ModeConfig] = {
    "minecraft": MINECRAFT_MODE_CONFIG,
}

# Mode class registry - maps mode_id to Mode class
MODE_CLASSES: dict[str, Type[BaseMode]] = {
    "minecraft": MinecraftMode,
}


def get_mode_config(mode_id: str) -> ModeConfig | None:
    """Get mode configuration by ID."""
    return MODE_CONFIGS.get(mode_id)


def get_mode_class(mode_id: str) -> Type[BaseMode] | None:
    """Get mode class by ID."""
    return MODE_CLASSES.get(mode_id)


def register_mode(mode_id: str, config: ModeConfig, mode_class: Type[BaseMode]) -> None:
    """Register a new mode."""
    MODE_CONFIGS[mode_id] = config
    MODE_CLASSES[mode_id] = mode_class


def list_modes() -> list[str]:
    """List all registered mode IDs."""
    return list(MODE_CONFIGS.keys())


def get_all_mode_configs() -> dict[str, ModeConfig]:
    """Get all mode configurations."""
    return MODE_CONFIGS.copy()


__all__ = [
    "ModeConfig",
    "BaseMode",
    "ModeTools",
    "json_response",
    "MinecraftMode",
    "MINECRAFT_MODE_CONFIG",
    "MODE_CONFIGS",
    "MODE_CLASSES",
    "get_mode_config",
    "get_mode_class",
    "register_mode",
    "list_modes",
    "get_all_mode_configs",
]
