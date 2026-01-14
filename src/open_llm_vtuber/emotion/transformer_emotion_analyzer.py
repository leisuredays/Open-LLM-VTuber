"""Transformer-based emotion analyzer using lightweight Korean models."""

import re
from functools import lru_cache
from typing import Optional

from loguru import logger

from .emotion_interface import EmotionAnalyzerInterface


class TransformerEmotionAnalyzer(EmotionAnalyzerInterface):
    """
    Lightweight transformer model for emotion analysis.

    Uses koelectra-small-v3 (54MB, 14M parameters) for fast and accurate
    Korean emotion detection with minimal resource usage.
    """

    def __init__(
        self,
        model_name: str = "monologg/koelectra-small-v3-discriminator",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        emotion_mapping: Optional[dict] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize the transformer emotion analyzer.

        Args:
            model_name: HuggingFace model name (default: koelectra-small-v3)
            device: Device to run on ('cpu', 'cuda', 'auto')
            cache_dir: Directory to cache models
            emotion_mapping: Custom emotion label mapping
            use_fp16: Use FP16 for GPU inference (default: True)
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16
        self._pipeline = None
        self._is_ready = False

        # Default emotion mapping (Korean labels -> Live2D emotions)
        self.emotion_mapping = emotion_mapping or {
            "기쁨": "joy",
            "즐거움": "joy",
            "행복": "joy",
            "설렘": "joy",
            "슬픔": "sadness",
            "우울": "sadness",
            "분노": "anger",
            "화남": "anger",
            "짜증": "anger",
            "공포": "fear",
            "두려움": "fear",
            "불안": "fear",
            "놀람": "surprise",
            "놀라움": "surprise",
            "당황": "surprise",
            "혐오": "disgust",
            "역겨움": "disgust",
            "불쾌함": "disgust",
            "중립": "neutral",
            "평범": "neutral",
            "평범함": "neutral",
            # English labels (for multilingual models)
            "joy": "joy",
            "happiness": "joy",
            "sadness": "sadness",
            "anger": "anger",
            "fear": "fear",
            "surprise": "surprise",
            "disgust": "disgust",
            "neutral": "neutral",
        }

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the transformer model pipeline."""
        try:
            from transformers import pipeline
            import torch

            logger.info(f"Loading emotion analysis model: {self.model_name}")

            # Determine device
            if self.device == "auto":
                device = 0 if torch.cuda.is_available() else -1
            elif self.device == "cuda":
                device = 0
            else:
                device = -1

            device_name = "GPU" if device == 0 else "CPU"

            # Determine dtype (FP16 only on GPU)
            torch_dtype = None
            precision_info = ""
            if device == 0 and self.use_fp16:
                torch_dtype = torch.float16
                precision_info = " (FP16)"

            logger.info(f"Using device: {device_name}{precision_info}")

            # Prepare model_kwargs
            model_kwargs = {}
            if self.cache_dir:
                model_kwargs["cache_dir"] = self.cache_dir

            # Load model with emotion classification
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=device,
                torch_dtype=torch_dtype,
                top_k=None,  # Return all emotion scores
                model_kwargs=model_kwargs,
            )

            self._is_ready = True
            logger.success(
                f"Emotion analyzer ready! Model: {self.model_name} on {device_name}{precision_info}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize emotion analyzer: {e}")
            logger.warning("Emotion analysis will be disabled")
            self._is_ready = False

    def is_ready(self) -> bool:
        """Check if the analyzer is ready."""
        return self._is_ready

    @lru_cache(maxsize=128)
    def analyze(self, text: str) -> str:
        """
        Analyze emotion from text with caching.

        Args:
            text: Text to analyze

        Returns:
            str: Detected emotion ('joy', 'sadness', 'anger', etc.)
        """
        if not self._is_ready or not text or not text.strip():
            return "neutral"

        try:
            # Clean text (remove special characters but keep emoji)
            cleaned_text = self._clean_text(text)

            if not cleaned_text:
                return "neutral"

            # Run emotion classification
            results = self._pipeline(cleaned_text[:512])  # Limit to 512 chars

            # Get the emotion with highest score
            if results and len(results) > 0:
                top_emotion = max(results[0], key=lambda x: x["score"])
                emotion_label = top_emotion["label"]
                confidence = top_emotion["score"]

                # Map to Live2D emotion
                mapped_emotion = self._map_emotion(emotion_label)

                logger.debug(
                    f"🎭 Emotion detected: {emotion_label} -> {mapped_emotion} "
                    f"(confidence: {confidence:.2f})"
                )

                return mapped_emotion

            return "neutral"

        except Exception as e:
            logger.warning(f"Error during emotion analysis: {e}")
            return "neutral"

    def _clean_text(self, text: str) -> str:
        """
        Clean text for emotion analysis.

        Args:
            text: Raw text

        Returns:
            str: Cleaned text
        """
        # Remove markdown-style tags [expression]
        text = re.sub(r"\[.*?\]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _map_emotion(self, emotion_label: str) -> str:
        """
        Map model's emotion label to Live2D emotion.

        Args:
            emotion_label: Raw emotion label from model

        Returns:
            str: Mapped Live2D emotion
        """
        # Normalize label (lowercase)
        normalized = emotion_label.lower().strip()

        # Direct mapping
        if normalized in self.emotion_mapping:
            return self.emotion_mapping[normalized]

        # Try to find partial match
        for key, value in self.emotion_mapping.items():
            if key.lower() in normalized or normalized in key.lower():
                return value

        # Default to neutral
        logger.debug(f"Unknown emotion label: {emotion_label}, defaulting to neutral")
        return "neutral"

    def clear_cache(self):
        """Clear the LRU cache."""
        self.analyze.cache_clear()
        logger.debug("Emotion analysis cache cleared")
