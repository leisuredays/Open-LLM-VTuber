"""
Discord voice channel audio utilities for Open-LLM-VTuber.

Provides audio format conversion (VTuber WAV → Discord PCM and vice versa),
a discord.AudioSource for streaming from memory buffers,
a playback queue for sequential TTS segment playback,
and a voice capture sink for receiving audio from Discord users.
"""

import asyncio
import base64
import io
import json
from typing import Optional

import numpy as np
from loguru import logger

try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    from discord.ext import voice_recv

    VOICE_RECV_AVAILABLE = True
except ImportError:
    VOICE_RECV_AVAILABLE = False


def base64_wav_to_pcm_48k_stereo(b64_wav: str) -> bytes:
    """
    Convert a base64-encoded WAV (typically 16kHz mono 16-bit from VTuber TTS)
    to raw PCM suitable for Discord (48kHz stereo 16-bit).

    Args:
        b64_wav: Base64-encoded WAV audio data.

    Returns:
        Raw PCM bytes at 48kHz, stereo, 16-bit signed little-endian.

    Raises:
        ValueError: If the audio data cannot be decoded or converted.
    """
    if not PYDUB_AVAILABLE:
        raise ImportError(
            "pydub is required for audio conversion. "
            "It should already be installed as a project dependency."
        )

    try:
        wav_bytes = base64.b64decode(b64_wav)
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")

        # Convert to Discord-compatible format: 48kHz stereo 16-bit
        audio = audio.set_frame_rate(48000).set_channels(2).set_sample_width(2)

        return audio.raw_data
    except Exception as e:
        raise ValueError(f"Failed to convert audio: {e}") from e


class VTuberAudioSource(discord.AudioSource):
    """
    A discord.AudioSource that reads 20ms PCM frames from an in-memory buffer.

    Discord's voice client expects 20ms frames of PCM audio at 48kHz stereo 16-bit.
    Each frame = 48000 * 2 channels * 2 bytes * 0.020s = 3840 bytes.
    """

    FRAME_SIZE = 3840  # 20ms at 48kHz stereo 16-bit

    def __init__(self, pcm_data: bytes):
        """
        Args:
            pcm_data: Raw PCM bytes (48kHz, stereo, 16-bit signed LE).
        """
        self._pcm_data = pcm_data
        self._offset = 0

    def read(self) -> bytes:
        """Read one 20ms frame of audio data."""
        if self._offset >= len(self._pcm_data):
            return b""

        end = self._offset + self.FRAME_SIZE
        frame = self._pcm_data[self._offset : end]
        self._offset = end

        # Pad with silence if the last frame is shorter than expected
        if len(frame) < self.FRAME_SIZE:
            frame += b"\x00" * (self.FRAME_SIZE - len(frame))

        return frame

    def is_opus(self) -> bool:
        """This source provides raw PCM, not Opus."""
        return False

    def cleanup(self) -> None:
        """Release the PCM buffer."""
        self._pcm_data = b""
        self._offset = 0


class AudioPlaybackQueue:
    """
    Manages sequential playback of multiple TTS audio segments through
    a Discord voice client.

    Audio segments (base64 WAV) are enqueued and played one after another.
    Supports interruption (clearing the queue and stopping current playback).
    """

    def __init__(self, voice_client: "discord.VoiceClient"):
        """
        Args:
            voice_client: The Discord voice client to play audio through.
        """
        self._voice_client = voice_client
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._current_done_event: Optional[asyncio.Event] = None

    def start(self) -> None:
        """Start the playback loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._playback_loop())
        logger.debug("AudioPlaybackQueue started")

    async def stop(self) -> None:
        """Stop the playback loop and clean up."""
        self._running = False

        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send sentinel to unblock the loop
        await self._queue.put(None)

        # Stop current playback
        if self._voice_client and self._voice_client.is_playing():
            self._voice_client.stop()

        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

        logger.debug("AudioPlaybackQueue stopped")

    async def enqueue(self, b64_wav: str) -> None:
        """
        Add a base64-encoded WAV segment to the playback queue.

        Args:
            b64_wav: Base64-encoded WAV audio data.
        """
        try:
            pcm_data = base64_wav_to_pcm_48k_stereo(b64_wav)
            await self._queue.put(pcm_data)
            logger.debug(f"Enqueued audio segment ({len(pcm_data)} bytes PCM)")
        except ValueError as e:
            logger.error(f"Failed to enqueue audio: {e}")

    def interrupt(self) -> None:
        """
        Interrupt current playback and clear the queue.
        Used when a new user message arrives to stop the current response.
        """
        # Clear pending segments
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Stop current playback
        if self._voice_client and self._voice_client.is_playing():
            self._voice_client.stop()

        # Signal done event so the loop moves on
        if self._current_done_event:
            self._current_done_event.set()

        logger.debug("Audio playback interrupted")

    async def _playback_loop(self) -> None:
        """Main loop that dequeues and plays audio segments sequentially."""
        logger.debug("Playback loop started")

        while self._running:
            try:
                pcm_data = await self._queue.get()

                # None is the sentinel to stop the loop
                if pcm_data is None:
                    break

                if not self._voice_client or not self._voice_client.is_connected():
                    logger.warning("Voice client not connected, skipping audio")
                    continue

                # Wait if something is already playing (shouldn't normally happen)
                while self._voice_client.is_playing():
                    await asyncio.sleep(0.05)

                # Play the audio segment
                source = VTuberAudioSource(pcm_data)
                self._current_done_event = asyncio.Event()
                done_event = self._current_done_event

                def after_play(error):
                    if error:
                        logger.error(f"Audio playback error: {error}")
                    done_event.set()

                self._voice_client.play(source, after=after_play)
                logger.debug("Playing audio segment")

                # Wait for playback to finish
                await done_event.wait()
                self._current_done_event = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in playback loop: {e}")
                await asyncio.sleep(0.1)

        logger.debug("Playback loop ended")


def pcm_48k_stereo_to_float32_16k_mono(pcm_data: bytes) -> list:
    """
    Convert Discord PCM audio (48kHz stereo 16-bit) to float32 samples
    at 16kHz mono, suitable for the VTuber server's VAD/ASR pipeline.

    Args:
        pcm_data: Raw PCM bytes (48kHz, stereo, 16-bit signed LE).

    Returns:
        List of float32 values normalized to [-1.0, 1.0].
    """
    # Parse as int16 samples
    samples = np.frombuffer(pcm_data, dtype=np.int16)

    # Stereo to mono: reshape into (N, 2) and average channels
    if len(samples) % 2 == 0:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # 48kHz to 16kHz: take every 3rd sample
    samples = samples[::3]

    # int16 to float32 normalized to [-1.0, 1.0]
    float_samples = samples.astype(np.float32) / 32768.0

    return float_samples.tolist()


class VTuberVoiceSink:
    """
    Audio sink for capturing voice from Discord users and forwarding
    it to the VTuber server via WebSocket.

    When discord-ext-voice-recv is available, this subclasses its AudioSink.
    The write() method is called from a separate audio thread, so we use
    asyncio.run_coroutine_threadsafe() to safely send data to the WebSocket.
    """

    def __init__(
        self,
        websocket,
        bot_user_id: int,
        event_loop: asyncio.AbstractEventLoop,
    ):
        """
        Args:
            websocket: The WebSocket connection to the proxy server.
            bot_user_id: The bot's own user ID (to filter out self-audio).
            event_loop: The asyncio event loop for scheduling coroutines.
        """
        self._websocket = websocket
        self._bot_user_id = bot_user_id
        self._event_loop = event_loop

    def wants_opus(self) -> bool:
        """Request decoded PCM audio, not raw Opus."""
        return False

    def write(self, user, data) -> None:
        """
        Called from the audio receive thread for each 20ms audio frame.

        Args:
            user: The Discord user who produced this audio (or None).
            data: VoiceData object with .pcm property containing raw PCM bytes.
        """
        # Ignore audio from the bot itself (prevent feedback loops)
        if user is not None and getattr(user, "id", None) == self._bot_user_id:
            return

        # Ignore audio with no identified user
        if user is None:
            return

        try:
            pcm_bytes = data.pcm
            float_samples = pcm_48k_stereo_to_float32_16k_mono(pcm_bytes)

            # Log periodically (every ~1s = 50 frames at 20ms)
            if not hasattr(self, "_frame_count"):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 50 == 1:
                max_amp = max(abs(s) for s in float_samples) if float_samples else 0
                logger.debug(
                    f"[Discord→VAD] frame #{self._frame_count} from user={getattr(user, 'name', '?')}, "
                    f"samples={len(float_samples)}, max_amplitude={max_amp:.4f}"
                )

            message = json.dumps({"type": "raw-audio-data", "audio": float_samples})

            asyncio.run_coroutine_threadsafe(
                self._websocket.send(message), self._event_loop
            )
        except Exception as e:
            logger.error(f"VTuberVoiceSink write error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        self._websocket = None
        logger.debug("VTuberVoiceSink cleaned up")


def _create_voice_sink(
    websocket,
    bot_user_id: int,
    event_loop: asyncio.AbstractEventLoop,
):
    """
    Factory that creates a proper AudioSink subclass if discord-ext-voice-recv
    is available, otherwise returns None.

    Returns:
        An AudioSink instance, or None if voice_recv is not available.
    """
    if not VOICE_RECV_AVAILABLE:
        logger.warning(
            "discord-ext-voice-recv not installed. "
            "Voice capture is disabled. Install with: uv sync --extra discord"
        )
        return None

    class _VoiceSink(voice_recv.AudioSink):
        """AudioSink subclass that delegates to VTuberVoiceSink."""

        def __init__(self, inner: VTuberVoiceSink):
            super().__init__()
            self._inner = inner

        def wants_opus(self) -> bool:
            return self._inner.wants_opus()

        def write(self, user, data) -> None:
            self._inner.write(user, data)

        def cleanup(self) -> None:
            self._inner.cleanup()

    inner = VTuberVoiceSink(websocket, bot_user_id, event_loop)
    return _VoiceSink(inner)
