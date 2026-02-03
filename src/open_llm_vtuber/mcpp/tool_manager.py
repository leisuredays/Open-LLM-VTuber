from loguru import logger
from typing import Dict, Any, List, Literal

from .types import FormattedTool


class ToolManager:
    """Tool Manager for managing pre-formatted tools for different LLM APIs."""

    def __init__(
        self,
        formatted_tools_openai: List[Dict[str, Any]] = None,
        formatted_tools_claude: List[Dict[str, Any]] = None,
        initial_tools_dict: Dict[str, FormattedTool] = None,
    ) -> None:
        """Initialize the Tool Manager with pre-formatted tool lists."""
        # Store the raw tool data (optional, for get_tool)
        self.tools: Dict[str, FormattedTool] = initial_tools_dict or {}

        # Store the pre-formatted lists
        self._formatted_tools_openai: List[Dict[str, Any]] = (
            formatted_tools_openai or []
        )
        self._formatted_tools_claude: List[Dict[str, Any]] = (
            formatted_tools_claude or []
        )

        logger.info(
            f"ToolManager initialized with {len(self._formatted_tools_openai)} OpenAI tools and {len(self._formatted_tools_claude)} Claude tools."
        )

    def get_tool(self, tool_name: str) -> FormattedTool | None:
        """Get a tool's raw information by its name."""
        tool = self.tools.get(tool_name)
        if isinstance(tool, FormattedTool):
            return tool
        logger.warning(
            f"TM: Raw tool info for '{tool_name}' not found (was initial_tools_dict provided?)."
        )
        return None

    def get_formatted_tools(
        self, mode: Literal["OpenAI", "Claude"]
    ) -> List[Dict[str, Any]] | Any:
        """Get the pre-formatted list of tools for the specified API mode."""

        if mode == "OpenAI":
            return self._formatted_tools_openai
        elif mode == "Claude":
            return self._formatted_tools_claude

    def register_builtin_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        """Register a built-in tool (not backed by an MCP server).

        The tool is added to both OpenAI and Claude formatted tool lists
        and stored in self.tools with related_server='__builtin__'.
        """
        # Store in tools dict
        self.tools[name] = FormattedTool(
            input_schema=input_schema,
            related_server="__builtin__",
            description=description,
        )

        # OpenAI format
        self._formatted_tools_openai.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
                },
            }
        )

        # Claude format
        self._formatted_tools_claude.append(
            {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }
        )

        logger.info(f"Registered built-in tool: {name}")
