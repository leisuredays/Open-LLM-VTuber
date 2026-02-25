"""OpenClaw LLM provider.

OpenClaw exposes an OpenAI-compatible `/v1/chat/completions` endpoint with
optional custom headers for agent and session routing.
"""

import copy
from typing import AsyncIterator, List, Dict, Any

from loguru import logger
from openai import NotGiven, NOT_GIVEN
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from .openai_compatible_llm import AsyncLLM


class OpenClawLLM(AsyncLLM):
    def __init__(
        self,
        model: str = "openclaw:main",
        base_url: str = "http://127.0.0.1:18789/v1",
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        max_tokens: int = None,
        agent_id: str = "",
        session_key: str = "",
    ):
        super().__init__(
            model=model,
            base_url=base_url,
            llm_api_key=llm_api_key,
            organization_id=organization_id,
            project_id=project_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Apply custom OpenClaw headers
        extra_headers = {}
        if agent_id:
            extra_headers["x-openclaw-agent-id"] = agent_id
        if session_key:
            extra_headers["x-openclaw-session-key"] = session_key

        if extra_headers:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                base_url=base_url,
                organization=organization_id,
                project=project_id,
                api_key=llm_api_key,
                default_headers=extra_headers,
            )

        logger.info(
            f"Initialized OpenClawLLM with agent_id='{agent_id}', "
            f"session_key='{session_key}'"
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
    ) -> AsyncIterator[str | List[ChoiceDeltaToolCall]]:
        tagged_messages = copy.deepcopy(messages)
        for msg in tagged_messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                msg["content"] = "[VTuber] " + msg["content"]
        async for chunk in super().chat_completion(tagged_messages, system, tools):
            yield chunk
