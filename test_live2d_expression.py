#!/usr/bin/env python3
"""Test script for Live2D expression changes"""

import asyncio
import json
import sys
import websockets
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

async def test_live2d_expressions():
    """Test Live2D expression changes through WebSocket"""

    # WebSocket server URL (adjust if needed)
    url = "ws://localhost:12393/client-ws"

    # Test expressions (common Live2D emotions)
    test_expressions = [
        "neutral",
        "joy",
        "anger",
        "sadness",
        "fear",
        "surprise",
        "disgust",
    ]

    try:
        logger.info(f"Connecting to {url}...")
        async with websockets.connect(url) as websocket:
            logger.success("✓ Connected to WebSocket server")

            # Wait for initial messages
            logger.info("Waiting for initial handshake...")
            await asyncio.sleep(2)

            # Try to consume any pending messages
            try:
                while True:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    logger.debug(f"Received: {msg[:100]}...")
            except asyncio.TimeoutError:
                pass

            logger.info("\n" + "="*60)
            logger.info("Starting Live2D Expression Test")
            logger.info("="*60 + "\n")

            for expression in test_expressions:
                logger.info(f"Testing expression: {expression}")

                # Create test payload with expression
                payload = {
                    "type": "audio",
                    "audio": None,  # No audio, just expression
                    "volumes": [],
                    "slice_length": 20,
                    "display_text": {
                        "text": f"Testing {expression} expression",
                        "name": "Test",
                        "avatar": None
                    },
                    "actions": {
                        "expressions": [expression]
                    },
                    "forwarded": False
                }

                # Send the payload
                await websocket.send(json.dumps(payload))
                logger.success(f"✓ Sent {expression} expression")

                # Wait for user to see the change
                await asyncio.sleep(3)

            logger.info("\n" + "="*60)
            logger.success("All expressions tested!")
            logger.info("="*60)

            # Send neutral expression to reset
            reset_payload = {
                "type": "audio",
                "audio": None,
                "volumes": [],
                "slice_length": 20,
                "display_text": {
                    "text": "Test complete - resetting to neutral",
                    "name": "Test",
                    "avatar": None
                },
                "actions": {
                    "expressions": ["neutral"]
                },
                "forwarded": False
            }
            await websocket.send(json.dumps(reset_payload))
            logger.info("Reset to neutral expression")

            await asyncio.sleep(2)

    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket error: {e}")
        logger.error("Make sure the server is running: uv run run_server.py")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

async def test_with_model_info():
    """Test with checking Live2D model emotion map"""
    logger.info("Checking Live2D model configuration...")

    try:
        from src.open_llm_vtuber.live2d_model import Live2dModel
        from src.open_llm_vtuber.config_manager import load_config

        # Load config
        config = load_config("conf.yaml")
        model_name = config.character_config.live2d_model_name

        logger.info(f"Live2D model: {model_name}")

        # Load model
        model = Live2dModel(model_name)
        logger.info(f"Available expressions: {list(model.emo_map.keys())}")
        logger.info(f"Expression map: {model.emo_map}")

        # Now run WebSocket test
        await test_live2d_expressions()

    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        logger.warning("Proceeding with WebSocket test anyway...")
        await test_live2d_expressions()

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Live2D Expression Test")
    logger.info("="*60)
    logger.info("")
    logger.info("This script will:")
    logger.info("1. Connect to the WebSocket server")
    logger.info("2. Send test expressions one by one")
    logger.info("3. Each expression will be displayed for 3 seconds")
    logger.info("")
    logger.info("Make sure:")
    logger.info("- Server is running (uv run run_server.py)")
    logger.info("- Frontend is open in browser")
    logger.info("- Live2D model is loaded")
    logger.info("")
    logger.info("="*60)
    logger.info("")

    try:
        asyncio.run(test_with_model_info())
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
