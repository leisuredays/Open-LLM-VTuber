"""Emotion analysis module for automatic Live2D expression control."""

from .emotion_interface import EmotionAnalyzerInterface
from .emotion_factory import EmotionAnalyzerFactory

__all__ = ["EmotionAnalyzerInterface", "EmotionAnalyzerFactory"]
