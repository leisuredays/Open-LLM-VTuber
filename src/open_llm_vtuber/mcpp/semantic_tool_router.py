"""
Semantic Tool Router for Dynamic MCP Tool Filtering

Uses semantic-router library to classify user intent and
dynamically filter MCP tools before passing to LLM.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from loguru import logger

from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import (
    OpenAIEncoder,
    CohereEncoder,
    HuggingFaceEncoder,
    FastEmbedEncoder
)

from .tool_manager import ToolManager


class SemanticToolRouter:
    """
    Filters MCP tools based on semantic understanding of user intent.

    Uses semantic-router to classify queries and return only relevant tools,
    reducing context size for LLM calls.
    """

    def __init__(
        self,
        route_config_path: str | Path,
        encoder_type: str = "openai",
        encoder_model: Optional[str] = None,
    ):
        """
        Initialize semantic tool router.

        Args:
            route_config_path: Path to route configuration YAML
            encoder_type: Type of encoder ("openai", "cohere", "huggingface", "fastembed")
            encoder_model: Optional model name for encoder
        """
        self.config = self._load_config(route_config_path)
        self.encoder = self._create_encoder(encoder_type, encoder_model)
        self.routes = self._create_routes()
        self.router = SemanticRouter(
            encoder=self.encoder,
            routes=self.routes,
            auto_sync="local"
        )

        logger.info(f"SEMTR: Initialized with {len(self.routes)} routes")

    def _load_config(self, config_path: str | Path) -> dict:
        """Load route configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_encoder(self, encoder_type: str, model: Optional[str]):
        """Create semantic encoder based on configuration."""
        if encoder_type == "openai":
            return OpenAIEncoder(name=model or "text-embedding-3-small")
        elif encoder_type == "cohere":
            return CohereEncoder(name=model or "embed-english-v3.0")
        elif encoder_type == "huggingface":
            return HuggingFaceEncoder(name=model or "sentence-transformers/all-MiniLM-L6-v2")
        elif encoder_type == "fastembed":
            return FastEmbedEncoder(name=model or "BAAI/bge-small-en-v1.5")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _create_routes(self) -> List[Route]:
        """Create Route objects from configuration."""
        routes = []
        for route_config in self.config['routes']:
            route = Route(
                name=route_config['name'],
                utterances=route_config['utterances']
            )
            routes.append(route)
        logger.info(f"SEMTR: Created {len(routes)} routes")
        return routes

    def classify_intent(self, user_message: str) -> Optional[str]:
        """
        Classify user message intent.

        Args:
            user_message: The user's input text

        Returns:
            Route name if matched, None otherwise
        """
        result = self.router(user_message)
        if result:
            logger.debug(f"SEMTR: Matched route '{result.name}' for message: {user_message[:50]}...")
            return result.name
        else:
            logger.debug(f"SEMTR: No route matched for message: {user_message[:50]}...")
            return None

    def get_relevant_servers(self, user_message: str) -> List[str]:
        """
        Get list of relevant MCP server names for user message.

        Args:
            user_message: The user's input text

        Returns:
            List of MCP server names that should be enabled
        """
        route_name = self.classify_intent(user_message)

        if route_name is None:
            # No match - use fallback
            return self._get_fallback_servers()

        # Find route config
        for route_config in self.config['routes']:
            if route_config['name'] == route_name:
                servers = route_config['mcp_servers']
                logger.info(f"SEMTR: Selected servers {servers} for route '{route_name}'")
                return servers

        return self._get_fallback_servers()

    def _get_fallback_servers(self) -> List[str]:
        """Get fallback servers based on configuration."""
        fallback_mode = self.config['fallback']['mode']

        if fallback_mode == "no_tools":
            logger.debug("SEMTR: Fallback - no tools")
            return []
        elif fallback_mode == "most_common":
            servers = self.config['fallback']['most_common_servers']
            logger.debug(f"SEMTR: Fallback - most common servers: {servers}")
            return servers
        else:  # "all_tools"
            logger.debug("SEMTR: Fallback - all tools")
            return ["__all__"]  # Special marker for "all servers"

    def filter_tools(
        self,
        user_message: str,
        tool_manager: ToolManager,
        api_format: str = "Claude"
    ) -> List[Dict[str, Any]]:
        """
        Filter tools based on user message intent.

        Args:
            user_message: The user's input text
            tool_manager: ToolManager instance with all available tools
            api_format: API format ("Claude" or "OpenAI")

        Returns:
            Filtered list of formatted tools for the specified API
        """
        relevant_servers = self.get_relevant_servers(user_message)

        # Special case: all servers
        if "__all__" in relevant_servers:
            all_tools = tool_manager.get_formatted_tools(api_format)
            logger.info(f"SEMTR: Returning all {len(all_tools)} tools")
            return all_tools

        # Get all tools
        all_tools = tool_manager.get_formatted_tools(api_format)

        if not relevant_servers:
            logger.info("SEMTR: No servers selected, returning empty tool list")
            return []

        # Filter tools by server
        filtered_tools = []
        for tool in all_tools:
            # Extract tool name from formatted tool
            if api_format == "Claude":
                tool_name = tool.get('name', '')
            else:  # OpenAI
                tool_name = tool.get('function', {}).get('name', '')

            # Check if this tool belongs to a relevant server
            tool_metadata = tool_manager.get_tool(tool_name)
            if tool_metadata and tool_metadata.related_server in relevant_servers:
                filtered_tools.append(tool)

        logger.info(
            f"SEMTR: Filtered {len(all_tools)} tools → {len(filtered_tools)} tools "
            f"(servers: {relevant_servers})"
        )

        return filtered_tools
