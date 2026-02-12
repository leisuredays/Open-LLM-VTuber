# config_manager/rag.py
from pydantic import Field
from typing import Dict, ClassVar
from .i18n import I18nMixin, Description


class RagConfig(I18nMixin):
    """RAG (Retrieval-Augmented Generation) configuration settings."""

    enabled: bool = Field(default=False, alias="enabled")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="ollama_base_url"
    )
    embedding_model: str = Field(default="bge-m3", alias="embedding_model")
    summarization_model: str = Field(default="", alias="summarization_model")
    summarization_base_url: str = Field(
        default="http://localhost:11434/v1", alias="summarization_base_url"
    )
    summarization_api_key: str = Field(default="", alias="summarization_api_key")
    knowledge_dir: str = Field(default="data/knowledge", alias="knowledge_dir")
    n_results: int = Field(default=3, alias="n_results")
    min_messages_for_summary: int = Field(default=6, alias="min_messages_for_summary")
    chunk_size: int = Field(default=500, alias="chunk_size")
    chunk_overlap: int = Field(default=100, alias="chunk_overlap")
    semantic_weight: float = Field(default=0.7, alias="semantic_weight")
    score_threshold: float = Field(default=0.0, alias="score_threshold")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable or disable RAG system", zh="RAG 시스템 활성화 여부"
        ),
        "ollama_base_url": Description(
            en="Base URL for Ollama API", zh="Ollama API 기본 URL"
        ),
        "embedding_model": Description(
            en="Embedding model name for Ollama", zh="Ollama 임베딩 모델 이름"
        ),
        "summarization_model": Description(
            en="Model for conversation summarization (empty to disable)",
            zh="대화 요약용 모델 (빈값이면 비활성화)",
        ),
        "summarization_base_url": Description(
            en="Base URL for OpenAI-compatible summarization API (default: Ollama)",
            zh="요약용 OpenAI 호환 API 기본 URL (기본: Ollama)",
        ),
        "summarization_api_key": Description(
            en="API key for summarization service (optional for Ollama)",
            zh="요약 서비스 API 키 (Ollama는 불필요)",
        ),
        "knowledge_dir": Description(
            en="Directory for knowledge documents", zh="지식 문서 디렉토리"
        ),
        "n_results": Description(
            en="Number of search results to return", zh="검색 결과 반환 수"
        ),
        "min_messages_for_summary": Description(
            en="Minimum messages before generating summary",
            zh="요약 생성 전 최소 메시지 수",
        ),
        "chunk_size": Description(
            en="Text chunk size for document splitting",
            zh="문서 분할 텍스트 청크 크기",
        ),
        "chunk_overlap": Description(
            en="Overlap between text chunks", zh="텍스트 청크 간 오버랩"
        ),
        "semantic_weight": Description(
            en="Weight for semantic search (0.0-1.0, keyword weight = 1 - semantic)",
            zh="의미론적 검색 가중치 (0.0-1.0, 키워드 가중치 = 1 - semantic)",
        ),
        "score_threshold": Description(
            en="Minimum score to include results (0.0 to disable)",
            zh="결과 포함 최소 점수 (0.0이면 비활성화)",
        ),
    }
