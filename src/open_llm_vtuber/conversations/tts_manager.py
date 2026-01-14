import asyncio
import json
import re
import uuid
import base64
from datetime import datetime
from typing import List, Optional, Dict
from loguru import logger

from ..agent.output_types import DisplayText, Actions
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload
from .types import WebSocketSend


class TTSTaskManager:
    """Manages TTS tasks and ensures ordered delivery to frontend while allowing parallel TTS generation"""

    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        # Counter for sequence numbering
        self._sequence_counter = 0

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """
        Queue a TTS task while maintaining order of delivery.

        Args:
            tts_text: Text to synthesize
            display_text: Text to display in UI
            actions: Live2D model actions
            live2d_model: Live2D model instance
            tts_engine: TTS engine instance
            websocket_send: WebSocket send function
        """
        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            # Get current sequence number for silent payload
            current_sequence = self._sequence_counter
            self._sequence_counter += 1

            await self._send_silent_payload(display_text, actions, websocket_send, current_sequence)
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Check if TTS engine supports streaming
        supports_streaming = hasattr(tts_engine, 'stream_audio') and callable(getattr(tts_engine, 'stream_audio'))

        # Create and queue the TTS task (streaming or traditional)
        if supports_streaming:
            task = asyncio.create_task(
                self._process_tts_streaming(
                    tts_text=tts_text,
                    display_text=display_text,
                    actions=actions,
                    live2d_model=live2d_model,
                    tts_engine=tts_engine,
                    websocket_send=websocket_send,
                    sequence_number=current_sequence,
                )
            )
        else:
            task = asyncio.create_task(
                self._process_tts(
                    tts_text=tts_text,
                    display_text=display_text,
                    actions=actions,
                    live2d_model=live2d_model,
                    tts_engine=tts_engine,
                    websocket_send=websocket_send,
                    sequence_number=current_sequence,
                )
            )
        self.task_list.append(task)

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        websocket_send: WebSocketSend,
        sequence_number: int,
    ) -> None:
        """Send a silent audio payload directly"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await websocket_send(json.dumps(audio_payload))
        logger.debug(f"📤 [WebSocket] Sent silent audio for sequence {sequence_number}")

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        sequence_number: int,
    ) -> None:
        """Process TTS generation and send result directly to frontend"""
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(tts_engine, tts_text)
            logger.debug(f"👄 [TTS] Calling prepare_audio_payload with audio_path: {audio_file_path}")
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            # Send directly to frontend
            await websocket_send(json.dumps(payload))
            logger.debug(f"📤 [WebSocket] Sent traditional audio for sequence {sequence_number}")

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Send silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await websocket_send(json.dumps(payload))
            logger.debug(f"📤 [WebSocket] Sent error fallback for sequence {sequence_number}")

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text"""
        logger.debug(f"🏃Generating audio for '''{text}'''...")
        audio_path = await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
        )
        logger.debug(f"👄 [TTS] Generated audio file: {audio_path}")
        return audio_path

    async def _process_tts_streaming(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
        sequence_number: int,
    ) -> None:
        """Process TTS generation with real-time streaming - send chunks IMMEDIATELY to frontend"""
        try:
            logger.debug(f"🎵 [Streaming] Starting streaming TTS for: '{tts_text[:50]}...'")

            # Send audio-start payload with metadata
            start_payload = {
                "type": "audio-start",
                "sequence": sequence_number,
                "display_text": display_text.to_dict() if hasattr(display_text, 'to_dict') else display_text,
                "actions": actions.to_dict() if actions else None,
                "slice_length": 20,  # 20ms chunks for lip sync
            }
            await websocket_send(json.dumps(start_payload))
            logger.debug(f"📤 [WebSocket] Sent audio-start for sequence {sequence_number}")

            # Stream audio chunks
            chunk_index = 0
            try:
                async for audio_chunk in tts_engine.stream_audio(tts_text):
                    chunk_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                    chunk_payload = {
                        "type": "audio-chunk",
                        "sequence": sequence_number,
                        "chunk_index": chunk_index,
                        "audio": chunk_base64,
                        "is_header": (chunk_index == 0),  # First chunk is WAV header
                    }
                    await websocket_send(json.dumps(chunk_payload))

                    if chunk_index == 0:
                        logger.info(f"📤 [WebSocket] Sent WAV header chunk (seq {sequence_number}, {len(audio_chunk)} bytes)")
                    elif chunk_index % 10 == 0:
                        logger.debug(f"📤 [WebSocket] Sent audio-chunk {chunk_index} for sequence {sequence_number}")

                    chunk_index += 1

                logger.info(f"🎵 [Streaming] Completed streaming {chunk_index} chunks for sequence {sequence_number}")

            except NotImplementedError as e:
                logger.warning(f"🎵 [Streaming] TTS engine doesn't support streaming, falling back: {e}")
                # Fallback to traditional method
                audio_file_path = await self._generate_audio(tts_engine, tts_text)
                payload = prepare_audio_payload(
                    audio_path=audio_file_path,
                    display_text=display_text,
                    actions=actions,
                )
                await websocket_send(json.dumps(payload))
                logger.debug(f"📤 [WebSocket] Sent traditional audio for sequence {sequence_number}")
                tts_engine.remove_file(audio_file_path)
                return

            # Send audio-complete payload
            complete_payload = {
                "type": "audio-complete",
                "sequence": sequence_number,
                "total_chunks": chunk_index,
            }
            await websocket_send(json.dumps(complete_payload))
            logger.info(f"📤 [WebSocket] Sent audio-complete for sequence {sequence_number}")

        except Exception as e:
            logger.error(f"Error in streaming TTS: {e}")
            # Send silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await websocket_send(json.dumps(payload))
            logger.debug(f"📤 [WebSocket] Sent error fallback for sequence {sequence_number}")

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        self._sequence_counter = 0
