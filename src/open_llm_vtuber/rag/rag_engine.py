"""RAG Engine using Ollama embeddings and ChromaDB vector store."""

import hashlib
import os
from pathlib import Path

import httpx
from loguru import logger

from ..config_manager.rag import RagConfig

try:
    import chromadb
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
except Exception as _import_err:
    logger.warning(f"Failed to import chromadb: {_import_err}")
    chromadb = None
    OllamaEmbeddingFunction = None


class RagEngine:
    """RAG engine that manages knowledge retrieval and conversation summarization."""

    LORE_COLLECTION_PREFIX = "lore_"
    SUMMARY_COLLECTION_PREFIX = "summary_"

    def __init__(self, config: RagConfig, conf_uid: str):
        self.config = config
        self.conf_uid = conf_uid
        self._client: "chromadb.ClientAPI | None" = None
        self._embedding_fn = None
        self._lore_collection = None
        self._summary_collection = None

    async def initialize(self) -> bool:
        """Initialize ChromaDB connection and load knowledge documents.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if not chromadb or not OllamaEmbeddingFunction:
            logger.warning("chromadb is not installed. RAG system will be disabled.")
            self.config.enabled = False
            return False

        try:
            self._embedding_fn = OllamaEmbeddingFunction(
                url=self.config.ollama_base_url,
                model_name=self.config.embedding_model,
            )
            # Test embedding connectivity
            self._embedding_fn(["test"])
        except Exception as e:
            logger.warning(
                f"Failed to connect to Ollama for embeddings: {e}. "
                "RAG system will be disabled."
            )
            self.config.enabled = False
            return False

        try:
            self._client = chromadb.PersistentClient(path="chromadb_data")
        except Exception as e:
            logger.warning(
                f"Failed to initialize ChromaDB: {e}. RAG system will be disabled."
            )
            self.config.enabled = False
            return False

        lore_name = f"{self.LORE_COLLECTION_PREFIX}{self.conf_uid}"
        summary_name = f"{self.SUMMARY_COLLECTION_PREFIX}{self.conf_uid}"
        self._lore_collection = self._client.get_or_create_collection(
            name=lore_name, embedding_function=self._embedding_fn
        )
        self._summary_collection = self._client.get_or_create_collection(
            name=summary_name, embedding_function=self._embedding_fn
        )

        await self._load_knowledge_documents()
        logger.info(
            f"RAG engine initialized for '{self.conf_uid}' "
            f"(lore: {self._lore_collection.count()}, "
            f"summaries: {self._summary_collection.count()})"
        )
        return True

    async def _load_knowledge_documents(self) -> None:
        """Scan knowledge directory and upsert text chunks into ChromaDB.

        Automatically removes stale chunks that no longer match the current
        file contents and gracefully skips chunks whose embeddings fail.
        """
        knowledge_path = Path(self.config.knowledge_dir) / self.conf_uid
        os.makedirs(knowledge_path, exist_ok=True)

        files = list(knowledge_path.glob("*.txt")) + list(knowledge_path.glob("*.md"))
        if not files:
            logger.info(f"No knowledge documents found in {knowledge_path}. Skipping.")
            return

        existing_ids = set()
        if self._lore_collection.count() > 0:
            existing = self._lore_collection.get()
            existing_ids = set(existing["ids"])

        # Build the set of chunk IDs that should exist based on current files
        current_ids: set[str] = set()

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

            chunks = self._split_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = self._make_chunk_id(file_path.name, i, chunk)
                current_ids.add(chunk_id)

                if chunk_id in existing_ids:
                    continue

                try:
                    self._lore_collection.upsert(
                        ids=[chunk_id],
                        documents=[chunk],
                        metadatas=[
                            {"source": file_path.name, "type": "lore", "chunk_index": i}
                        ],
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to embed chunk {chunk_id} "
                        f"('{chunk[:50]}...'): {e}. Skipping."
                    )

        # Remove stale chunks that no longer exist in current files
        stale_ids = [cid for cid in existing_ids if cid not in current_ids]
        if stale_ids:
            self._lore_collection.delete(ids=stale_ids)
            logger.info(f"Removed {len(stale_ids)} stale lore chunks.")

        logger.info(
            f"Knowledge documents loaded. "
            f"Total lore chunks: {self._lore_collection.count()}"
        )

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks by lines first, then by character size.

        Each non-empty line becomes its own chunk. Lines exceeding chunk_size
        are further split with overlap.
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(line) <= chunk_size:
                chunks.append(line)
            else:
                start = 0
                while start < len(line):
                    end = start + chunk_size
                    chunks.append(line[start:end].strip())
                    start += chunk_size - overlap

        return [c for c in chunks if c]

    @staticmethod
    def _make_chunk_id(filename: str, index: int, content: str) -> str:
        """Create a deterministic ID for a chunk based on content hash."""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
        return f"{filename}_{index}_{content_hash}"

    def _hybrid_search(self, collection, query: str, n_results: int) -> list[str]:
        """Hybrid search combining semantic and keyword matching.

        Returns list of documents sorted by combined score.
        """
        if not collection or collection.count() == 0:
            return []

        semantic_w = self.config.semantic_weight
        keyword_w = 1.0 - semantic_w
        threshold = self.config.score_threshold

        # Semantic search
        fetch_n = min(n_results * 3, collection.count())
        results = collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        distances = results["distances"][0]

        # Score each document: combine semantic similarity + keyword match
        scored = []
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        for doc, dist in zip(docs, distances):
            # Semantic score: convert cosine distance to similarity (0~1)
            semantic_score = max(0.0, 1.0 - dist)

            # Keyword score: fraction of query tokens found in document
            doc_lower = doc.lower()
            if query_tokens:
                matched = sum(1 for t in query_tokens if t in doc_lower)
                keyword_score = matched / len(query_tokens)
            else:
                keyword_score = 0.0

            combined = (semantic_w * semantic_score) + (keyword_w * keyword_score)

            if threshold > 0 and combined < threshold:
                continue

            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:n_results]]

    def search(self, query: str) -> str:
        """Search knowledge base and summaries for relevant context.

        Args:
            query: The user's query text.

        Returns:
            Formatted string with relevant context, or empty string if nothing found.
        """
        if not self.config.enabled or not self._client:
            return ""

        parts = []

        # Search lore collection (hybrid)
        try:
            lore_docs = self._hybrid_search(
                self._lore_collection, query, self.config.n_results
            )
            if lore_docs:
                lore_text = "\n".join(f"• {doc}" for doc in lore_docs)
                parts.append(f"[캐릭터 배경]\n{lore_text}")
        except Exception as e:
            logger.warning(f"RAG lore search failed: {e}")

        # Search summary collection (hybrid)
        try:
            summary_docs = self._hybrid_search(
                self._summary_collection, query, self.config.n_results
            )
            if summary_docs:
                summary_text = "\n".join(f"• {doc}" for doc in summary_docs)
                parts.append(f"[이전 대화 요약]\n{summary_text}")
        except Exception as e:
            logger.warning(f"RAG summary search failed: {e}")

        return "\n\n".join(parts)

    def has_summary(self, history_uid: str) -> bool:
        """Check if a summary already exists for the given history."""
        if not self._summary_collection:
            return False
        try:
            results = self._summary_collection.get(where={"history_uid": history_uid})
            return bool(results["ids"])
        except Exception:
            return False

    async def generate_and_store_summary(
        self, history_uid: str, messages: list
    ) -> None:
        """Generate a summary of the conversation and store it in ChromaDB.

        Args:
            history_uid: Unique identifier for the conversation history.
            messages: List of message dicts with 'role' and 'content' keys.
        """
        if not self.config.summarization_model:
            return

        if not self._summary_collection:
            return

        # Extract conversation time range
        timestamps = [
            msg.get("timestamp", "")
            for msg in messages
            if msg.get("role") not in ("system", "metadata") and msg.get("timestamp")
        ]
        if timestamps:
            first_ts = timestamps[0].replace("T", " ")
            last_ts = timestamps[-1].replace("T", " ")
            time_info = f"대화 일시: {first_ts} ~ {last_ts}\n"
        else:
            time_info = ""

        # Format conversation for summarization
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue
            conversation_text += f"{role}: {content}\n"

        if not conversation_text.strip():
            return

        user_prompt = (
            "다음 대화를 간결하게 요약해주세요. "
            "주요 주제, 중요한 정보, 사용자의 관심사를 포함해주세요.\n\n"
            f"{time_info}"
            f"{conversation_text}"
        )

        try:
            base_url = self.config.summarization_base_url.rstrip("/")
            headers = {"Content-Type": "application/json"}
            if self.config.summarization_api_key:
                headers["Authorization"] = (
                    f"Bearer {self.config.summarization_api_key}"
                )

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.config.summarization_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "당신은 대화를 한국어로 간결하게 요약하는 어시스턴트입니다. 반드시 한국어로 요약해주세요.",
                            },
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                )
                response.raise_for_status()
                result = response.json()
                summary = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

            if not summary:
                logger.warning("Empty summary generated, skipping storage.")
                return

            # Prepend conversation time range to the summary
            if time_info:
                summary = f"{time_info.strip()}\n{summary}"

            summary_id = f"summary_{history_uid}"
            self._summary_collection.upsert(
                ids=[summary_id],
                documents=[summary],
                metadatas=[
                    {
                        "type": "summary",
                        "history_uid": history_uid,
                    }
                ],
            )
            logger.info(
                f"Summary stored for history '{history_uid}': {summary[:80]}..."
            )

        except Exception as e:
            logger.warning(f"Failed to generate/store summary: {e}")

    async def close(self) -> None:
        """Clean up resources."""
        self._lore_collection = None
        self._summary_collection = None
        self._client = None
        logger.info("RAG engine closed.")
