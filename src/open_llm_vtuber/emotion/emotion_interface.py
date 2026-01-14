"""Interface for emotion analysis engines."""

from abc import ABC, abstractmethod


class EmotionAnalyzerInterface(ABC):
    """Abstract base class for emotion analysis engines."""

    @abstractmethod
    def analyze(self, text: str) -> str:
        """
        Analyze the emotion of the given text.

        Args:
            text: The text to analyze

        Returns:
            str: The detected emotion (e.g., 'joy', 'sadness', 'anger', etc.)
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the analyzer is ready to use.

        Returns:
            bool: True if the analyzer is ready, False otherwise
        """
        pass
