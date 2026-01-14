"""Factory for creating emotion analyzer instances."""

from typing import Type

from .emotion_interface import EmotionAnalyzerInterface


class EmotionAnalyzerFactory:
    """Factory class for creating emotion analyzer engines."""

    @staticmethod
    def get_analyzer(engine_type: str, **kwargs) -> Type[EmotionAnalyzerInterface]:
        """
        Create an emotion analyzer instance based on engine type.

        Args:
            engine_type: Type of analyzer ('transformer', 'keyword', 'ollama')
            **kwargs: Configuration parameters for the analyzer

        Returns:
            EmotionAnalyzerInterface: Configured emotion analyzer instance

        Raises:
            ValueError: If engine_type is unknown
        """
        if engine_type == "transformer":
            from .transformer_emotion_analyzer import TransformerEmotionAnalyzer

            return TransformerEmotionAnalyzer(
                model_name=kwargs.get(
                    "model_name", "monologg/koelectra-small-v3-discriminator"
                ),
                device=kwargs.get("device", "auto"),
                cache_dir=kwargs.get("cache_dir"),
                emotion_mapping=kwargs.get("emotion_mapping"),
                use_fp16=kwargs.get("use_fp16", True),
            )

        # Future implementations:
        # elif engine_type == "keyword":
        #     from .keyword_emotion_analyzer import KeywordEmotionAnalyzer
        #     return KeywordEmotionAnalyzer(**kwargs)
        #
        # elif engine_type == "ollama":
        #     from .ollama_emotion_analyzer import OllamaEmotionAnalyzer
        #     return OllamaEmotionAnalyzer(**kwargs)

        else:
            raise ValueError(f"Unknown emotion analyzer engine type: {engine_type}")


# Example usage:
if __name__ == "__main__":
    # Create a transformer-based analyzer
    analyzer = EmotionAnalyzerFactory.get_analyzer(
        "transformer", device="cpu", model_name="monologg/koelectra-small-v3-discriminator"
    )

    # Test
    test_texts = [
        "정말 기쁘네요! 오늘 좋은 일이 있었어요.",
        "너무 슬퍼요... 힘들어요.",
        "화나네요! 짜증나!",
        "무서워요... 걱정돼요.",
        "헐! 깜짝이야!",
        "네, 알겠습니다.",
    ]

    for text in test_texts:
        emotion = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Emotion: {emotion}\n")
