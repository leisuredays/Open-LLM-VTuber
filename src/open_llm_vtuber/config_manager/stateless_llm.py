# config_manager/llm.py
from typing import ClassVar, Literal
from pydantic import BaseModel, Field
from .i18n import I18nMixin, Description


class StatelessLLMBaseConfig(I18nMixin):
    """Base configuration for StatelessLLM."""

    # interrupt_method. If the provider supports inserting system prompt anywhere in the chat memory, use "system". Otherwise, use "user".
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )
    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "interrupt_method": Description(
            en="""The method to use for prompting the interruption signal.
            If the provider supports inserting system prompt anywhere in the chat memory, use "system". 
            Otherwise, use "user". You don't need to change this setting.""",
            zh="""用于表示中断信号的方法(提示词模式)。如果LLM支持在聊天记忆中的任何位置插入系统提示词，请使用“system”。
            否则，请使用“user”。您不需要更改此设置。""",
        ),
    }


class StatelessLLMWithTemplate(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible LLM providers."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    template: str | None = Field(None, alias="template")
    temperature: float = Field(1.0, alias="temperature")

    _OPENAI_COMPATIBLE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(en="Base URL for the API endpoint", zh="API的URL端点"),
        "llm_api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "organization_id": Description(
            en="Organization ID for the API (Optional)", zh="组织 ID (可选)"
        ),
        "project_id": Description(
            en="Project ID for the API (Optional)", zh="项目 ID (可选)"
        ),
        "model": Description(en="Name of the LLM model to use", zh="LLM 模型名称"),
        "temperature": Description(
            en="What sampling temperature to use, between 0 and 2.",
            zh="使用的采样温度，介于 0 和 2 之间。",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENAI_COMPATIBLE_DESCRIPTIONS,
    }


class OpenAICompatibleConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI-compatible LLM providers."""

    base_url: str = Field(..., alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    organization_id: str | None = Field(None, alias="organization_id")
    project_id: str | None = Field(None, alias="project_id")
    temperature: float = Field(1.0, alias="temperature")
    max_tokens: int | None = Field(None, alias="max_tokens")

    _OPENAI_COMPATIBLE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(en="Base URL for the API endpoint", zh="API的URL端点"),
        "llm_api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "organization_id": Description(
            en="Organization ID for the API (Optional)", zh="组织 ID (可选)"
        ),
        "project_id": Description(
            en="Project ID for the API (Optional)", zh="项目 ID (可选)"
        ),
        "model": Description(en="Name of the LLM model to use", zh="LLM 模型名称"),
        "temperature": Description(
            en="What sampling temperature to use, between 0 and 2.",
            zh="使用的采样温度，介于 0 和 2 之间。",
        ),
        "max_tokens": Description(
            en="Maximum number of tokens to generate. Leave empty for no limit.",
            zh="生成的最大令牌数。留空表示无限制。",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_OPENAI_COMPATIBLE_DESCRIPTIONS,
    }


# Ollama config is completely the same as OpenAICompatibleConfig


class OllamaConfig(OpenAICompatibleConfig):
    """Configuration for Ollama API."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    keep_alive: float = Field(-1, alias="keep_alive")
    unload_at_exit: bool = Field(True, alias="unload_at_exit")
    num_ctx: int | None = Field(None, alias="num_ctx")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    # Ollama-specific descriptions
    _OLLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "llm_api_key": Description(
            en="API key for authentication (defaults to 'default_api_key' for Ollama)",
            zh="API 认证密钥 (Ollama 默认为 'default_api_key')",
        ),
        "keep_alive": Description(
            en="Keep the model loaded for this many seconds after the last request. "
            "Set to -1 to keep the model loaded indefinitely.",
            zh="在最后一个请求之后保持模型加载的秒数。设置为 -1 以无限期保持模型加载。",
        ),
        "unload_at_exit": Description(
            en="Unload the model when the program exits.",
            zh="是否在程序退出时卸载模型。",
        ),
        "num_ctx": Description(
            en="Context window size for the model. If not set, Ollama uses its default (2048).",
            zh="模型的上下文窗口大小。如果未设置，Ollama 使用默认值 (2048)。",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **OpenAICompatibleConfig.DESCRIPTIONS,
        **_OLLAMA_DESCRIPTIONS,
    }


class LmStudioConfig(OpenAICompatibleConfig):
    """Configuration for LM Studio."""

    llm_api_key: str = Field("default_api_key", alias="llm_api_key")
    base_url: str = Field("http://localhost:1234/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class OpenAIConfig(OpenAICompatibleConfig):
    """Configuration for Official OpenAI API."""

    base_url: str = Field("https://api.openai.com/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class GeminiConfig(OpenAICompatibleConfig):
    """Configuration for Gemini API."""

    base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/openai/", alias="base_url"
    )
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class MistralConfig(OpenAICompatibleConfig):
    """Configuration for Mistral API."""

    base_url: str = Field("https://api.mistral.ai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )


class ZhipuConfig(OpenAICompatibleConfig):
    """Configuration for Zhipu API."""

    base_url: str = Field("https://open.bigmodel.cn/api/paas/v4/", alias="base_url")


class DeepseekConfig(OpenAICompatibleConfig):
    """Configuration for Deepseek API."""

    base_url: str = Field("https://api.deepseek.com/v1", alias="base_url")


class GroqConfig(OpenAICompatibleConfig):
    """Configuration for Groq API."""

    base_url: str = Field("https://api.groq.com/openai/v1", alias="base_url")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )


class ClaudeConfig(StatelessLLMBaseConfig):
    """Configuration for OpenAI Official API."""

    base_url: str = Field("https://api.anthropic.com", alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    model: str = Field(..., alias="model")
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )

    _CLAUDE_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for Claude API", zh="Claude API 的API端点"
        ),
        "llm_api_key": Description(en="API key for authentication", zh="API 认证密钥"),
        "model": Description(
            en="Name of the Claude model to use", zh="要使用的 Claude 模型名称"
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_CLAUDE_DESCRIPTIONS,
    }


class LlamaCppConfig(StatelessLLMBaseConfig):
    """Configuration for LlamaCpp."""

    model_path: str = Field(..., alias="model_path")
    interrupt_method: Literal["system", "user"] = Field(
        "system", alias="interrupt_method"
    )

    _LLAMA_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "model_path": Description(
            en="Path to the GGUF model file", zh="GGUF 模型文件路径"
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_LLAMA_DESCRIPTIONS,
    }


class DifyConfig(StatelessLLMBaseConfig):
    """Configuration for Dify API (Chat, Workflow, Completion)."""

    base_url: str = Field("http://localhost/v1", alias="base_url")
    llm_api_key: str = Field(..., alias="llm_api_key")
    user: str = Field("open-llm-vtuber", alias="user")
    app_type: Literal["chat", "workflow", "completion"] = Field(
        "workflow", alias="app_type"
    )
    input_variable: str = Field("query", alias="input_variable")
    system_prompt_variable: str = Field("", alias="system_prompt_variable")
    router_mcp_url: str = Field("", alias="router_mcp_url")
    mode_prompt_variable: str = Field(
        "activated_mode_prompt", alias="mode_prompt_variable"
    )
    image_variable: str = Field("", alias="image_variable")
    image_upload_method: Literal["direct", "upload"] = Field(
        "direct", alias="image_upload_method"
    )
    interrupt_method: Literal["system", "user"] = Field(
        "user", alias="interrupt_method"
    )

    _DIFY_DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "base_url": Description(
            en="Base URL for Dify API (e.g., http://localhost/v1)",
            zh="Dify API 基础 URL (例如: http://localhost/v1)",
        ),
        "llm_api_key": Description(
            en="Dify App API key (starts with 'app-')",
            zh="Dify 应用 API 密钥 (以 'app-' 开头)",
        ),
        "user": Description(
            en="User identifier for Dify conversations",
            zh="Dify 对话的用户标识符",
        ),
        "app_type": Description(
            en="Dify app type: 'chat', 'workflow', or 'completion'",
            zh="Dify 应用类型: 'chat', 'workflow' 或 'completion'",
        ),
        "input_variable": Description(
            en="Input variable name for workflow/completion (default: 'query')",
            zh="工作流/补全的输入变량名称 (默认: 'query')",
        ),
        "system_prompt_variable": Description(
            en="Input variable name for system prompt (optional, leave empty to disable)",
            zh="系统提示词的输入变量名称 (可选, 留空则禁用)",
        ),
        "router_mcp_url": Description(
            en="URL of Router MCP server for mode management (e.g., http://localhost:8769). Leave empty to disable.",
            zh="Router MCP 服务器 URL，用于模式管理 (例如: http://localhost:8769)。留空则禁用。",
        ),
        "mode_prompt_variable": Description(
            en="Input variable name for mode prompt (default: 'activated_mode_prompt')",
            zh="模式提示词的输入变量名称 (默认: 'activated_mode_prompt')",
        ),
        "image_variable": Description(
            en="Input variable name for images in workflow (optional, leave empty to disable vision)",
            zh="工作流中图像的输入变量名称 (可选, 留空则禁用视觉功能)",
        ),
        "image_upload_method": Description(
            en="Method for sending images: 'direct' (base64 data URL, faster) or 'upload' (file upload, more compatible)",
            zh="图像发送方式: 'direct' (base64 URL直接发送, 更快) 或 'upload' (先上传文件, 兼容性更好)",
        ),
    }

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        **StatelessLLMBaseConfig.DESCRIPTIONS,
        **_DIFY_DESCRIPTIONS,
    }


class StatelessLLMConfigs(I18nMixin, BaseModel):
    """Pool of LLM provider configurations.
    This class contains configurations for different LLM providers."""

    stateless_llm_with_template: StatelessLLMWithTemplate | None = Field(
        None, alias="stateless_llm_with_template"
    )
    openai_compatible_llm: OpenAICompatibleConfig | None = Field(
        None, alias="openai_compatible_llm"
    )
    ollama_llm: OllamaConfig | None = Field(None, alias="ollama_llm")
    lmstudio_llm: LmStudioConfig | None = Field(None, alias="lmstudio_llm")
    openai_llm: OpenAIConfig | None = Field(None, alias="openai_llm")
    gemini_llm: GeminiConfig | None = Field(None, alias="gemini_llm")
    zhipu_llm: ZhipuConfig | None = Field(None, alias="zhipu_llm")
    deepseek_llm: DeepseekConfig | None = Field(None, alias="deepseek_llm")
    groq_llm: GroqConfig | None = Field(None, alias="groq_llm")
    claude_llm: ClaudeConfig | None = Field(None, alias="claude_llm")
    llama_cpp_llm: LlamaCppConfig | None = Field(None, alias="llama_cpp_llm")
    mistral_llm: MistralConfig | None = Field(None, alias="mistral_llm")
    dify_llm: DifyConfig | None = Field(None, alias="dify_llm")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "stateless_llm_with_template": Description(
            en="Stateless LLM with Template", zh=""
        ),
        "openai_compatible_llm": Description(
            en="Configuration for OpenAI-compatible LLM providers",
            zh="OpenAI兼容的语言模型提供者配置",
        ),
        "ollama_llm": Description(en="Configuration for Ollama", zh="Ollama 配置"),
        "lmstudio_llm": Description(
            en="Configuration for LM Studio", zh="LM Studio 配置"
        ),
        "openai_llm": Description(
            en="Configuration for Official OpenAI API", zh="官方 OpenAI API 配置"
        ),
        "gemini_llm": Description(
            en="Configuration for Gemini API", zh="Gemini API 配置"
        ),
        "mistral_llm": Description(
            en="Configuration for Mistral API", zh="Mistral API 配置"
        ),
        "zhipu_llm": Description(en="Configuration for Zhipu API", zh="Zhipu API 配置"),
        "deepseek_llm": Description(
            en="Configuration for Deepseek API", zh="Deepseek API 配置"
        ),
        "groq_llm": Description(en="Configuration for Groq API", zh="Groq API 配置"),
        "claude_llm": Description(
            en="Configuration for Claude API", zh="Claude API配置"
        ),
        "llama_cpp_llm": Description(
            en="Configuration for local Llama.cpp", zh="本地Llama.cpp配置"
        ),
        "dify_llm": Description(
            en="Configuration for Dify Chat API", zh="Dify Chat API 配置"
        ),
    }
