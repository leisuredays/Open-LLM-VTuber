"""
Dify LLM implementation for Open-LLM-VTuber.
This module provides integration with Dify's Chat API.
"""

import json
import base64
import httpx
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple
from loguru import logger
from io import BytesIO

from .stateless_llm_interface import StatelessLLMInterface


class DifyLLM(StatelessLLMInterface):
    """Dify API implementation supporting Chat and Workflow apps."""

    def __init__(
        self,
        base_url: str,
        llm_api_key: str,
        user: str = "open-llm-vtuber",
        app_type: str = "workflow",
        input_variable: str = "query",
        system_prompt_variable: str = "",
        router_mcp_url: str = "",
        mode_prompt_variable: str = "activated_mode_prompt",
        image_variable: str = "",
        image_upload_method: str = "direct",
    ):
        """
        Initialize the Dify LLM client.

        Parameters:
        - base_url (str): The base URL for Dify API (e.g., "http://localhost/v1")
        - llm_api_key (str): The Dify app API key (e.g., "app-xxx")
        - user (str): User identifier for Dify conversations
        - app_type (str): Dify app type - "chat", "workflow", or "completion"
        - input_variable (str): Input variable name for workflow (default: "query")
        - system_prompt_variable (str): Input variable name for system prompt (optional)
        - router_mcp_url (str): URL of Router MCP server for mode management (e.g., "http://localhost:8769")
        - mode_prompt_variable (str): Input variable name for mode prompt (default: "activated_mode_prompt")
        - image_variable (str): Input variable name for images in workflow (optional, leave empty to disable vision)
        - image_upload_method (str): Method for sending images - "direct" (base64 data URL) or "upload" (file upload)
        """
        self.base_url = base_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.user = user
        self.app_type = app_type.lower()
        self.input_variable = input_variable
        self.system_prompt_variable = system_prompt_variable
        self.router_mcp_url = router_mcp_url.rstrip("/") if router_mcp_url else ""
        self.mode_prompt_variable = mode_prompt_variable
        self.image_variable = image_variable
        self.image_upload_method = image_upload_method.lower()
        self.conversation_id = None  # Will be set after first message
        self.support_tools = False  # Dify handles tools internally

        logger.info(f"Initialized DifyLLM with base_url: {self.base_url}, app_type: {self.app_type}")
        if self.router_mcp_url:
            logger.info(f"Router MCP integration enabled: {self.router_mcp_url}")
        if self.image_variable:
            logger.info(f"Vision enabled with image_variable: {self.image_variable}, method: {self.image_upload_method}")

    async def _get_mode_prompt(self) -> Dict[str, Any]:
        """
        Fetch the current mode prompt from Router MCP server.

        Returns:
            Dict containing mode information and prompt, or empty dict on failure.
        """
        if not self.router_mcp_url:
            return {}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.router_mcp_url}/api/mode-prompt")
                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"Router MCP mode prompt: {data}")
                    return data
                else:
                    logger.warning(
                        f"Router MCP returned status {response.status_code}"
                    )
                    return {}
        except httpx.ConnectError:
            logger.warning(
                f"Cannot connect to Router MCP at {self.router_mcp_url}"
            )
            return {}
        except Exception as e:
            logger.warning(f"Error fetching mode prompt: {e}")
            return {}

    def _extract_images(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract images from messages.

        Returns:
            List of image data dicts with 'data_url' (full data URL), 'data' (base64), and 'media_type' keys.
        """
        images = []

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")

                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            url = image_url.get("url", "")

                            # Parse data URL: data:image/jpeg;base64,/9j/...
                            if url.startswith("data:image"):
                                try:
                                    header, base64_data = url.split(",", 1)
                                    # Extract media type from 'data:image/jpeg;base64'
                                    media_type = header.split(":")[1].split(";")[0]
                                    images.append({
                                        "data_url": url,  # Keep full data URL for direct use
                                        "data": base64_data,
                                        "media_type": media_type,
                                    })
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Failed to parse image data URL: {e}")
                            else:
                                # Remote URL - store as-is
                                images.append({
                                    "url": url,
                                    "media_type": "image/jpeg",  # Default
                                })
                # Only process the last user message
                break

        logger.debug(f"Extracted {len(images)} images from messages")
        return images

    async def _upload_image(
        self, image_data: Dict[str, Any], client: httpx.AsyncClient
    ) -> Optional[str]:
        """
        Upload a base64 image to Dify and return the file ID.

        Args:
            image_data: Dict with 'data' (base64) and 'media_type' keys
            client: httpx AsyncClient instance

        Returns:
            The uploaded file ID, or None on failure.
        """
        try:
            # Decode base64 to binary
            binary_data = base64.b64decode(image_data["data"])
            media_type = image_data.get("media_type", "image/jpeg")

            # Determine file extension from media type
            ext_map = {
                "image/jpeg": "jpg",
                "image/jpg": "jpg",
                "image/png": "png",
                "image/gif": "gif",
                "image/webp": "webp",
            }
            extension = ext_map.get(media_type, "jpg")
            filename = f"image.{extension}"

            # Prepare multipart form data
            files = {
                "file": (filename, BytesIO(binary_data), media_type),
            }
            data = {
                "user": self.user,
            }

            headers = {
                "Authorization": f"Bearer {self.llm_api_key}",
            }

            response = await client.post(
                f"{self.base_url}/files/upload",
                files=files,
                data=data,
                headers=headers,
            )

            if response.status_code == 201 or response.status_code == 200:
                result = response.json()
                file_id = result.get("id")
                logger.info(f"Successfully uploaded image to Dify, file_id: {file_id}")
                return file_id
            else:
                logger.error(
                    f"Failed to upload image to Dify: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error uploading image to Dify: {e}")
            return None

    def _prepare_image_inputs_direct(
        self, images: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare image inputs using direct base64 data URL (no upload).
        This method tries to send base64 data URL directly via remote_url.

        Args:
            images: List of image data dicts

        Returns:
            List of Dify file input objects.
        """
        file_inputs = []

        for img in images:
            if "url" in img:
                # Remote URL - use directly
                file_inputs.append({
                    "type": "image",
                    "transfer_method": "remote_url",
                    "url": img["url"],
                })
            elif "data_url" in img:
                # Try to send base64 data URL directly via remote_url
                file_inputs.append({
                    "type": "image",
                    "transfer_method": "remote_url",
                    "url": img["data_url"],
                })
                logger.debug("Using base64 data URL directly (no upload)")

        return file_inputs

    async def _prepare_image_inputs_upload(
        self, images: List[Dict[str, Any]], client: httpx.AsyncClient
    ) -> List[Dict[str, Any]]:
        """
        Prepare image inputs by uploading to Dify first.
        Use this as fallback if direct base64 URL doesn't work.

        Args:
            images: List of image data dicts
            client: httpx AsyncClient instance

        Returns:
            List of Dify file input objects.
        """
        file_inputs = []

        for img in images:
            if "url" in img:
                # Remote URL - use directly
                file_inputs.append({
                    "type": "image",
                    "transfer_method": "remote_url",
                    "url": img["url"],
                })
            elif "data" in img:
                # Base64 data - upload first
                file_id = await self._upload_image(img, client)
                if file_id:
                    file_inputs.append({
                        "type": "image",
                        "transfer_method": "local_file",
                        "upload_file_id": file_id,
                    })

        return file_inputs

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate a chat completion using Dify's API.

        Parameters:
        - messages: List of message dictionaries
        - system: System prompt (will be included in query if provided)
        - tools: Not used for Dify (tools are configured in Dify app)

        Yields:
        - str: Response text chunks
        """
        # Extract the latest user message and images
        query = self._extract_query(messages)
        images = self._extract_images(messages) if self.image_variable else []

        # Fetch mode prompt from Router MCP if configured
        mode_data = await self._get_mode_prompt()
        mode_prompt = mode_data.get("prompt", "") if mode_data else ""

        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }

        # Use a client for both image upload and chat completion
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Prepare image inputs if vision is enabled and images are present
            image_inputs = []
            if self.image_variable and images:
                if self.image_upload_method == "upload":
                    # Upload images to Dify first, then use file IDs
                    image_inputs = await self._prepare_image_inputs_upload(images, client)
                    logger.info(f"Prepared {len(image_inputs)} image inputs via upload")
                else:
                    # Default: send base64 data URL directly (no upload)
                    image_inputs = self._prepare_image_inputs_direct(images)
                    logger.info(f"Prepared {len(image_inputs)} image inputs via direct base64 URL")

            # Prepare request payload based on app type
            if self.app_type == "workflow":
                inputs = {self.input_variable: query}
                # Add system prompt as separate input variable if configured
                if self.system_prompt_variable and system:
                    inputs[self.system_prompt_variable] = system
                # Add mode prompt if available
                if self.mode_prompt_variable and mode_prompt:
                    inputs[self.mode_prompt_variable] = mode_prompt
                elif self.mode_prompt_variable:
                    # Provide empty string if variable is expected but no mode is active
                    inputs[self.mode_prompt_variable] = ""
                # Add image inputs if vision is enabled
                if self.image_variable and image_inputs:
                    inputs[self.image_variable] = image_inputs
                payload = {
                    "inputs": inputs,
                    "response_mode": "streaming",
                    "user": self.user,
                }
                endpoint = f"{self.base_url}/workflows/run"
            elif self.app_type == "completion":
                payload = {
                    "inputs": {self.input_variable: query},
                    "response_mode": "streaming",
                    "user": self.user,
                }
                endpoint = f"{self.base_url}/completion-messages"
            else:  # chat (default)
                payload = {
                    "inputs": {},
                    "query": query,
                    "response_mode": "streaming",
                    "user": self.user,
                }
                # Add files for chat mode if images are present
                if image_inputs:
                    payload["files"] = image_inputs
                if self.conversation_id:
                    payload["conversation_id"] = self.conversation_id
                endpoint = f"{self.base_url}/chat-messages"

            logger.debug(f"Dify API endpoint: {endpoint}")
            logger.debug(f"Dify payload: {payload}")

            try:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(
                            f"Dify API error: {response.status_code} - {error_text.decode()}"
                        )
                        yield f"Error: Dify API returned status {response.status_code}"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        # SSE format: "data: {...}"
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix

                            try:
                                data = json.loads(data_str)
                                event = data.get("event", "")

                                # Workflow events
                                if event == "text_chunk":
                                    # Workflow text output chunk
                                    text = data.get("data", {}).get("text", "")
                                    if text:
                                        yield text

                                elif event == "workflow_finished":
                                    # Workflow completed
                                    outputs = data.get("data", {}).get("outputs", {})
                                    logger.debug(f"Dify workflow finished: {outputs}")

                                # Chat/Agent events
                                elif event == "message":
                                    # Regular message chunk
                                    answer = data.get("answer", "")
                                    if answer:
                                        yield answer

                                elif event == "agent_message":
                                    # Agent mode message
                                    answer = data.get("answer", "")
                                    if answer:
                                        yield answer

                                elif event == "message_end":
                                    # Conversation ended, save conversation_id
                                    self.conversation_id = data.get("conversation_id")
                                    logger.debug(
                                        f"Dify conversation_id: {self.conversation_id}"
                                    )

                                elif event == "error":
                                    error_msg = data.get("message", "Unknown error")
                                    logger.error(f"Dify error event: {error_msg}")
                                    yield f"Error: {error_msg}"

                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse Dify SSE data: {data_str}")
                                continue

            except httpx.ConnectError as e:
                logger.error(f"Dify connection error: {e}")
                yield "Error: Failed to connect to Dify API. Please check the server is running."

            except httpx.TimeoutException as e:
                logger.error(f"Dify timeout error: {e}")
                yield "Error: Dify API request timed out."

            except Exception as e:
                logger.error(f"Dify unexpected error: {e}")
                yield f"Error: Unexpected error occurred: {str(e)}"

    def _extract_query(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract the query text from messages.
        Dify expects a single query string, not a message list.
        """
        if not messages:
            return ""

        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")

                # Handle different content formats
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Extract text from content list (multimodal format)
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "")
                    return str(content)
                else:
                    return str(content)

        # Fallback: return last message content
        last_msg = messages[-1]
        content = last_msg.get("content", "")
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
        return str(content)

    def reset_conversation(self):
        """Reset the conversation by clearing the conversation_id."""
        self.conversation_id = None
        logger.info("Dify conversation reset")
