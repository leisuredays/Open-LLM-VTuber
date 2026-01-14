#!/usr/bin/env python3
"""Test script for emotion analysis system"""

import sys
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_emotion_analyzer():
    """Test the emotion analyzer"""
    logger.info("=" * 60)
    logger.info("Testing Emotion Analysis System")
    logger.info("=" * 60)

    # Test 1: Import modules
    logger.info("\n[Test 1] Importing modules...")
    try:
        from src.open_llm_vtuber.emotion.emotion_factory import EmotionAnalyzerFactory
        logger.success("✓ Successfully imported EmotionAnalyzerFactory")
    except ImportError as e:
        logger.error(f"✗ Failed to import EmotionAnalyzerFactory: {e}")
        return False

    # Test 2: Check transformers installation
    logger.info("\n[Test 2] Checking transformers installation...")
    try:
        import transformers
        logger.success(f"✓ transformers installed: v{transformers.__version__}")
    except ImportError:
        logger.error("✗ transformers not installed! Run: uv sync")
        return False

    try:
        import torch
        logger.success(f"✓ torch installed: v{torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("✗ torch not installed!")
        return False

    # Test 3: Initialize analyzer
    logger.info("\n[Test 3] Initializing emotion analyzer...")
    try:
        analyzer = EmotionAnalyzerFactory.get_analyzer(
            engine_type="transformer",
            model_name="LimYeri/HowRU-KoELECTRA-Emotion-Classifier",
            device="auto",
            cache_dir="models/emotion"
        )
        logger.success("✓ Analyzer created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Check if ready
    logger.info("\n[Test 4] Checking if analyzer is ready...")
    if analyzer.is_ready():
        logger.success("✓ Analyzer is ready!")
    else:
        logger.error("✗ Analyzer not ready")
        return False

    # Test 5: Test emotion detection
    logger.info("\n[Test 5] Testing emotion detection...")
    test_texts = [
        ("정말 기쁘고 행복해요!", "joy"),
        ("너무 슬프고 우울해요", "sadness"),
        ("화가 나서 미치겠어!", "anger"),
        ("무섭고 두려워요", "fear"),
        ("와! 깜짝 놀랐어요!", "surprise"),
        ("역겹고 싫어요", "disgust"),
        ("그냥 평범한 하루예요", "neutral"),
        ("I'm so happy today!", "joy"),
        ("This makes me angry", "anger"),
    ]

    success_count = 0
    for text, expected_emotion in test_texts:
        try:
            detected = analyzer.analyze(text)
            status = "✓" if detected == expected_emotion else "~"
            logger.info(f"{status} '{text[:30]}...' -> {detected} (expected: {expected_emotion})")
            if detected == expected_emotion:
                success_count += 1
        except Exception as e:
            logger.error(f"✗ Error analyzing '{text[:30]}...': {e}")

    accuracy = (success_count / len(test_texts)) * 100
    logger.info(f"\nAccuracy: {success_count}/{len(test_texts)} ({accuracy:.1f}%)")

    # Test 6: Performance test
    logger.info("\n[Test 6] Testing performance...")
    import time
    test_text = "정말 기쁘고 행복한 하루입니다!"

    # First run (cold start)
    start = time.perf_counter()
    result = analyzer.analyze(test_text)
    cold_time = (time.perf_counter() - start) * 1000
    logger.info(f"  Cold start: {cold_time:.2f}ms -> {result}")

    # Cached run
    start = time.perf_counter()
    result = analyzer.analyze(test_text)
    cached_time = (time.perf_counter() - start) * 1000
    logger.info(f"  Cached run: {cached_time:.2f}ms -> {result}")

    # Average over 10 runs
    times = []
    for _ in range(10):
        different_text = f"테스트 문장입니다 {_}"
        start = time.perf_counter()
        analyzer.analyze(different_text)
        times.append((time.perf_counter() - start) * 1000)

    avg_time = sum(times) / len(times)
    logger.info(f"  Average (10 runs): {avg_time:.2f}ms")

    logger.info("\n" + "=" * 60)
    logger.success("All tests completed!")
    logger.info("=" * 60)

    return True

if __name__ == "__main__":
    try:
        success = test_emotion_analyzer()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
