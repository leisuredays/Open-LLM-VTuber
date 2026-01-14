"""Emotion analysis configuration."""

from typing import ClassVar, Dict, Optional
from pydantic import Field

from .i18n import I18nMixin, Description


class TransformerAnalyzerConfig(I18nMixin):
    """Configuration for transformer-based emotion analyzer."""

    model_name: str = Field(
        default="monologg/koelectra-small-v3-discriminator",
        alias="model_name",
    )
    device: str = Field(default="auto", alias="device")
    cache_dir: Optional[str] = Field(default="models/emotion", alias="cache_dir")
    use_fp16: bool = Field(
        default=True,
        alias="use_fp16",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model_name": Description(
            en="HuggingFace model name for emotion analysis",
            zh="用于情感分析的HuggingFace模型名称",
        ),
        "device": Description(
            en="Device to run model on ('cpu', 'cuda', 'auto')",
            zh="运行模型的设备（'cpu'、'cuda'、'auto'）",
        ),
        "cache_dir": Description(
            en="Directory to cache downloaded models", zh="缓存下载模型的目录"
        ),
        "use_fp16": Description(
            en="Use FP16 (half precision) for GPU inference (2x faster, 50% less memory)",
            zh="在GPU上使用FP16（半精度）推理（速度提升2倍，内存减少50%）",
        ),
    }


class EmotionConfig(I18nMixin):
    """Configuration for emotion analysis system."""

    enabled: bool = Field(default=False, alias="enabled")
    engine: str = Field(default="transformer", alias="engine")
    transformer_analyzer: Optional[TransformerAnalyzerConfig] = Field(
        default=None, alias="transformer_analyzer"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable automatic emotion analysis for Live2D control",
            zh="启用自动情感分析以控制Live2D",
        ),
        "engine": Description(
            en="Emotion analysis engine ('transformer', 'keyword', 'ollama')",
            zh="情感分析引擎（'transformer'、'keyword'、'ollama'）",
        ),
        "transformer_analyzer": Description(
            en="Configuration for transformer-based analyzer", zh="基于transformer的分析器配置"
        ),
    }
