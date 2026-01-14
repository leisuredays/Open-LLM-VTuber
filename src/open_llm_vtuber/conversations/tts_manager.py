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
        # Queue to store ordered payloads
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        # Task to handle sending payloads in order
        self._sender_task: Optional[asyncio.Task] = None
        # Counter for maintaining order
        self._sequence_counter = 0
        self._next_sequence_to_send = 0

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

            # Start sender task if not running
            if not self._sender_task or self._sender_task.done():
                self._sender_task = asyncio.create_task(
                    self._process_payload_queue(websocket_send)
                )

            await self._send_silent_payload(display_text, actions, current_sequence)
            return

        logger.debug(
            f"🏃Queuing TTS task for: '''{tts_text}''' (by {display_text.name})"
        )

        # Get current sequence number
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # Start sender task if not running
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

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
                    sequence_number=current_sequence,
                )
            )
        self.task_list.append(task)

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """
        Process and send payloads in correct order.
        Supports both traditional single-payload mode and streaming multi-chunk mode.
        Runs continuously until all payloads are processed.
        """
        from collections import defaultdict
        buffered_payloads: Dict[int, List[Dict]] = defaultdict(list)
        sent_counts: Dict[int, int] = defaultdict(int)  # Track how many payloads sent per sequence

        while True:
            try:
                # Get payload from queue
                payload, sequence_number = await self._payload_queue.get()

                # Add to buffer
                buffered_payloads[sequence_number].append(payload)

                # Send payloads for current sequence in order
                while self._next_sequence_to_send in buffered_payloads:
                    sequence_payloads = buffered_payloads[self._next_sequence_to_send]
                    sent_count = sent_counts[self._next_sequence_to_send]

                    # Send only NEW payloads (not already sent)
                    for i in range(sent_count, len(sequence_payloads)):
                        await websocket_send(json.dumps(sequence_payloads[i]))
                        sent_counts[self._next_sequence_to_send] += 1

                    # Check if sequence is complete
                    last_payload = sequence_payloads[-1]
                    payload_type = last_payload.get("type")

                    # Sequence is complete if:
                    # 1. It's a traditional "audio" payload (single payload per sequence)
                    # 2. It's an "audio-complete" payload (end of streaming)
                    if payload_type in ("audio", "audio-complete"):
                        buffered_payloads.pop(self._next_sequence_to_send)
                        sent_counts.pop(self._next_sequence_to_send, None)
                        self._next_sequence_to_send += 1
                    else:
                        # Still waiting for more chunks/completion
                        break

                self._payload_queue.task_done()

            except asyncio.CancelledError:
                break

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        """Queue a silent audio payload"""
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
        )
        await self._payload_queue.put((audio_payload, sequence_number))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
    ) -> None:
        """Process TTS generation and queue the result for ordered delivery"""
        audio_file_path = None
        try:
            audio_file_path = await self._generate_audio(tts_engine, tts_text)
            logger.debug(f"👄 [TTS] Calling prepare_audio_payload with audio_path: {audio_file_path}")
            payload = prepare_audio_payload(
                audio_path=audio_file_path,
                display_text=display_text,
                actions=actions,
            )
            # Queue the payload with its sequence number
            await self._payload_queue.put((payload, sequence_number))

        except Exception as e:
            logger.error(f"Error preparing audio payload: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self._payload_queue.put((payload, sequence_number))

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
        sequence_number: int,
    ) -> None:
        """Process TTS generation with real-time streaming and queue chunks for ordered delivery"""
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
            await self._payload_queue.put((start_payload, sequence_number))
            logger.debug(f"🎵 [Streaming] Sent audio-start for sequence {sequence_number}")

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
                    await self._payload_queue.put((chunk_payload, sequence_number))

                    if chunk_index == 0:
                        logger.info(f"🎵 [Streaming] Sent WAV header chunk ({len(audio_chunk)} bytes)")

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
                await self._payload_queue.put((payload, sequence_number))
                tts_engine.remove_file(audio_file_path)
                return

            # Send audio-complete payload
            complete_payload = {
                "type": "audio-complete",
                "sequence": sequence_number,
                "total_chunks": chunk_index,
            }
            await self._payload_queue.put((complete_payload, sequence_number))
            logger.debug(f"🎵 [Streaming] Sent audio-complete for sequence {sequence_number}")

        except Exception as e:
            logger.error(f"Error in streaming TTS: {e}")
            # Queue silent payload for error case
            payload = prepare_audio_payload(
                audio_path=None,
                display_text=display_text,
                actions=actions,
            )
            await self._payload_queue.put((payload, sequence_number))

    def clear(self) -> None:
        """Clear all pending tasks and reset state"""
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        # Create a new queue to clear any pending items
        self._payload_queue = asyncio.Queue()
