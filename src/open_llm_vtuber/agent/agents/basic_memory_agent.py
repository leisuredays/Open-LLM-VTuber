import asyncio
import json
from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import get_history
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor
from ...debug_monitor import debug_monitor


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        rag_engine=None,
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._memory = []
        self._rag_engine = rag_engine
        self._current_rag_context: Optional[str] = None
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()

        # 모드 기반 도구 필터링을 위한 상태 추적
        self._active_mode: str | None = None
        self._base_system: str | None = None  # 모드 활성화 전 원래 시스템 프롬프트
        self._mode_context_prompt: str = ""  # 모드 컨텍스트 프롬프트 캐시 (매 턴 갱신)
        self._context_refresh_task: asyncio.Task | None = (
            None  # 독립 컨텍스트 갱신 태스크
        )
        # 모드별 도구 접두사 매핑 (해당 모드가 활성화되어야만 사용 가능)
        self._mode_tool_prefixes = {
            "minecraft": ["minecraft_"],
        }

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        self._current_trigger_reason: str | None = None

        logger.info("BasicMemoryAgent initialized.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug(f"Memory Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _filter_tools_by_mode(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """활성화된 모드에 따라 도구 목록을 필터링합니다."""
        if not tools:
            return tools

        filtered = []
        for tool in tools:
            tool_name = tool.get("function", {}).get("name") or tool.get("name", "")

            # activate_mode는 모드가 비활성화 상태일 때만 표시
            if tool_name == "activate_mode":
                if self._active_mode is None:
                    filtered.append(tool)
                else:
                    logger.debug(
                        f"Filtering out 'activate_mode' (mode already active: {self._active_mode})"
                    )
                continue

            # deactivate_mode는 모드가 활성화 상태일 때만 표시
            if tool_name == "deactivate_mode":
                if self._active_mode is not None:
                    filtered.append(tool)
                else:
                    logger.debug("Filtering out 'deactivate_mode' (no active mode)")
                continue

            # 모드별 도구 접두사 확인
            is_mode_specific = False
            required_mode = None
            for mode, prefixes in self._mode_tool_prefixes.items():
                for prefix in prefixes:
                    if tool_name.startswith(prefix):
                        is_mode_specific = True
                        required_mode = mode
                        break
                if is_mode_specific:
                    break

            # 모드별 도구는 해당 모드가 활성화된 경우에만 포함
            if is_mode_specific:
                if self._active_mode == required_mode:
                    filtered.append(tool)
                    logger.debug(f"Including mode-specific tool: {tool_name}")
                else:
                    logger.debug(
                        f"Filtering out tool '{tool_name}' (requires mode: {required_mode}, active: {self._active_mode})"
                    )
            else:
                filtered.append(tool)

        return filtered

    def _update_mode_from_tool_result(self, tool_name: str, result: str):
        """도구 실행 결과에서 모드 상태를 업데이트하고, 모드 시스템 프롬프트를 주입/복원합니다."""
        try:
            if tool_name == "activate_mode" and '"success": true' in result.lower():
                data = json.loads(result)
                if data.get("success") and data.get("activated_mode"):
                    self._active_mode = data["activated_mode"].get("id")
                    logger.info(f"Mode activated: {self._active_mode}")

                    # 모드 시스템 프롬프트를 LLM 시스템 프롬프트에 주입
                    mode_system_prompt = data.get("system_prompt", "")
                    if mode_system_prompt:
                        if self._base_system is None:
                            self._base_system = self._system
                        self._system = f"{self._base_system}\n\n{mode_system_prompt}"
                        logger.info(
                            f"Mode system prompt injected for: {self._active_mode}"
                        )

            elif tool_name == "deactivate_mode" and '"success": true' in result.lower():
                logger.info(f"Mode deactivated (was: {self._active_mode})")
                self._active_mode = None
                self._mode_context_prompt = ""

                # 원래 시스템 프롬프트 복원
                if self._base_system is not None:
                    self._system = self._base_system
                    self._base_system = None
                    logger.info("Restored base system prompt")

        except Exception as e:
            logger.debug(f"Could not parse mode from tool result: {e}")

    async def _refresh_mode_context(self) -> None:
        """모드가 활성화된 경우, 독립 태스크로 컨텍스트를 갱신합니다.
        대화 인터럽트에 의한 CancelledError가 MCP 세션을 오염시키지 않도록
        별도 태스크에서 실행합니다.
        """
        if not self._active_mode or not self._tool_executor:
            self._mode_context_prompt = ""
            return

        # 이전 갱신 태스크가 아직 실행 중이면 완료를 기다림
        if self._context_refresh_task and not self._context_refresh_task.done():
            try:
                await asyncio.wait_for(self._context_refresh_task, timeout=3.0)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise  # 대화 인터럽트 전파

        # 독립 태스크로 실행 — 대화 취소에 영향 안 받음
        self._context_refresh_task = asyncio.create_task(
            self._do_refresh_mode_context()
        )

        # 짧은 시간 내 완료되면 결과 사용, 아니면 캐시 유지하고 진행
        try:
            await asyncio.wait_for(
                asyncio.shield(self._context_refresh_task), timeout=3.0
            )
        except asyncio.TimeoutError:
            logger.debug(
                f"Mode context refresh timed out for {self._active_mode}. "
                "Using cached context."
            )
        except asyncio.CancelledError:
            logger.debug(
                f"Mode context refresh cancelled for {self._active_mode}. "
                "Using cached context. Independent task continues."
            )
            raise  # 대화 인터럽트 전파 — 독립 태스크는 계속 실행됨

    async def _do_refresh_mode_context(self) -> None:
        """실제 MCP 호출로 모드 컨텍스트를 갱신합니다. (독립 태스크에서 실행)"""
        try:
            is_error, text_content, _, _ = await self._tool_executor.run_single_tool(
                "get_mode_context_prompt", "auto-context", {}
            )
            if not is_error and text_content:
                data = json.loads(text_content)
                if data.get("success") and data.get("has_mode"):
                    self._mode_context_prompt = data.get("prompt", "")
                    logger.debug(
                        f"Mode context refreshed for {self._active_mode} "
                        f"(len={len(self._mode_context_prompt)})"
                    )
                    return
        except Exception as e:
            logger.debug(f"Failed to fetch mode context: {e}")
        self._mode_context_prompt = ""

    def _get_effective_system(self) -> str:
        """캐시된 모드 컨텍스트와 RAG 컨텍스트를 포함한 시스템 프롬프트를 반환합니다."""
        system = self._system
        if self._mode_context_prompt:
            system += f"\n\n--- 현재 모드 상태 ---\n{self._mode_context_prompt}"
        if self._current_rag_context:
            system += (
                "\n\n# 배경 지식\n"
                "아래는 대화에 참고할 수 있는 정보입니다. "
                "관련될 때만 자연스럽게 활용하고, 관련 없으면 무시하세요.\n\n"
                f"{self._current_rag_context}"
            )
        # 핵심 규칙 리마인더 (recency effect 활용)
        system += (
            "\n\n# 리마인더\n"
            "위 배경 지식보다 캐릭터 성격과 말투 규칙이 항상 우선합니다. "
            "캐릭터를 벗어나는 응답은 절대 하지 마세요."
        )
        return system

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if not text_content and role == "assistant":
            return

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from chat history."""
        messages = get_history(conf_uid, history_uid)

        self._memory = []
        for msg in messages:
            role = "user" if msg["role"] == "human" else "assistant"
            content = msg["content"]
            if isinstance(content, str) and content:
                self._memory.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )
            else:
                logger.warning(f"Skipping invalid message from history: {msg}")
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            if not self._memory[-1]["content"].endswith("..."):
                self._memory[-1]["content"] = heard_response + "..."
            else:
                self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _extract_latest_user_text(self, input_data: BatchInput) -> str:
        """Extract the latest user text from input data for RAG search."""
        return self._to_text_prompt(input_data) or ""

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        messages = self._memory.copy()
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []

        while True:
            # 모드에 따라 도구 필터링
            filtered_tools = self._filter_tools_by_mode(tools)

            # Debug: 프롬프트 모니터링
            effective_system = self._get_effective_system()
            await debug_monitor.log_prompt(
                system_prompt=effective_system,
                messages=messages,
                tools=filtered_tools,
                model=getattr(self._llm, "model", None),
                trigger_reason=self._current_trigger_reason,
            )

            stream = self._llm.chat_completion(
                messages, effective_system, tools=filtered_tools
            )
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
                    yield text
                    if (
                        not current_assistant_message_content
                        or current_assistant_message_content[-1]["type"] != "text"
                    ):
                        current_assistant_message_content.append(
                            {"type": "text", "text": text}
                        )
                    else:
                        current_assistant_message_content[-1]["text"] += text
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                    )
                    pending_tool_calls.append(tool_call_data)
                    current_assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data["id"],
                            "name": tool_call_data["name"],
                            "input": tool_call_data["input"],
                        }
                    )
                # elif event["type"] == "message_delta":
                #     if event["data"]["delta"].get("stop_reason"):
                #         stop_reason = event["data"]["delta"].get("stop_reason")
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    yield f"[Error from LLM: {event['message']}]"
                    return

            if pending_tool_calls:
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        elif update.get("type") == "tool_call_status":
                            # 모드 상태 업데이트 확인
                            tool_name = update.get("tool_name", "")
                            content = update.get("content", "")
                            self._update_mode_from_tool_result(tool_name, content)
                            yield update
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        effective_system = self._get_effective_system()
        current_system_prompt = effective_system

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    current_system_prompt = (
                        f"{effective_system}\n\n{self._mcp_prompt_string}"
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    current_system_prompt = effective_system
                tools_for_api = None
            else:
                current_system_prompt = effective_system
                # 모드에 따라 도구 필터링
                tools_for_api = self._filter_tools_by_mode(tools)

            # Debug: 프롬프트 모니터링
            await debug_monitor.log_prompt(
                system_prompt=current_system_prompt,
                messages=messages,
                tools=tools_for_api,
                model=getattr(self._llm, "model", None),
                trigger_reason=self._current_trigger_reason,
            )

            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                        yield event
                else:
                    if isinstance(event, str):
                        current_turn_text += event
                        yield event
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": current_turn_text if current_turn_text else None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            elif update.get("type") == "tool_call_status":
                                # 모드 상태 업데이트 확인
                                tool_name = update.get("tool_name", "")
                                content = update.get("content", "")
                                self._update_mode_from_tool_result(tool_name, content)
                                yield update
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        elif update.get("type") == "tool_call_status":
                            # 모드 상태 업데이트 확인
                            tool_name = update.get("tool_name", "")
                            content = update.get("content", "")
                            self._update_mode_from_tool_result(tool_name, content)
                            yield update
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                continue

            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False

            # 트리거 이유 추출
            self._current_trigger_reason = (
                input_data.metadata.get("trigger_reason")
                if input_data.metadata
                else None
            )

            # 모드가 활성화된 경우 컨텍스트 프롬프트를 갱신
            await self._refresh_mode_context()

            messages = self._to_messages(input_data)

            # RAG search injection
            if self._rag_engine:
                user_query = self._extract_latest_user_text(input_data)
                if user_query:
                    self._current_rag_context = self._rag_engine.search(user_query)
                    if self._current_rag_context:
                        logger.info(
                            f"RAG context injected ({len(self._current_rag_context)} chars): "
                            f"{self._current_rag_context[:100]}..."
                        )
                    else:
                        logger.debug("RAG search returned empty results.")
                else:
                    self._current_rag_context = None
            else:
                logger.debug("RAG engine not available.")

            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._formatted_tools_claude
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = self._formatted_tools_openai
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            else:
                logger.info("Starting simple chat completion.")
                effective_system = self._get_effective_system()
                # Debug: 프롬프트 모니터링
                await debug_monitor.log_prompt(
                    system_prompt=effective_system,
                    messages=messages,
                    tools=None,
                    model=getattr(self._llm, "model", None),
                    trigger_reason=self._current_trigger_reason,
                )
                token_stream = self._llm.chat_completion(messages, effective_system)
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response:
                    self._add_message(complete_response, "assistant")

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
