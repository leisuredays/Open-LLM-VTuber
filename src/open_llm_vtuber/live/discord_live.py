"""
Discord integration for Open-LLM-VTuber.

Connects to a Discord server and forwards messages to the VTuber
through the proxy WebSocket, then relays responses back to Discord channels.
Supports optional voice channel playback of TTS audio.
"""

import asyncio
import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional

import websockets
from loguru import logger

try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

from .live_interface import LivePlatformInterface


class DiscordLivePlatform(LivePlatformInterface):
    """
    Implementation of LivePlatformInterface for Discord.
    Connects to Discord via discord.py and forwards messages to the VTuber
    through the proxy WebSocket.

    Supports optional voice channel integration: when voice_enabled=True,
    users can use `join`/`leave` commands to have the bot join a voice channel
    and play TTS responses as audio.
    """

    def __init__(
        self,
        bot_token: str,
        channel_ids: List[int],
        prefix: str = "",
        proxy_url: str = "ws://localhost:12393/proxy-ws",
        voice_enabled: bool = False,
    ):
        """
        Initialize the Discord Live platform client.

        Args:
            bot_token: Discord bot token
            channel_ids: List of Discord channel IDs to listen on
            prefix: Command prefix for triggering the bot (empty string = all messages)
            proxy_url: WebSocket URL for the proxy server
            voice_enabled: Whether to enable voice channel support
        """
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py library is required for Discord functionality. "
                "Install with: uv sync --extra discord"
            )

        self._bot_token = bot_token
        self._channel_ids = channel_ids
        self._prefix = prefix
        self._proxy_url = proxy_url
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._running = False
        self._message_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._bot: Optional[discord.Client] = None
        self._pending_responses: Dict[int, discord.TextChannel] = {}
        self._current_response_channel: Optional[discord.TextChannel] = None
        self._response_buffer: str = ""

        # Voice channel support
        self._voice_enabled = voice_enabled
        self._voice_client: Optional[discord.VoiceClient] = None
        self._playback_queue = None  # AudioPlaybackQueue, lazily imported
        self._voice_sink = None  # VTuberVoiceSink for capturing user audio

    @property
    def is_connected(self) -> bool:
        """Check if connected to the proxy server."""
        try:
            if hasattr(self._websocket, "closed"):
                return (
                    self._connected and self._websocket and not self._websocket.closed
                )
            elif hasattr(self._websocket, "open"):
                return self._connected and self._websocket and self._websocket.open
            else:
                return self._connected and self._websocket is not None
        except Exception:
            return False

    async def connect(self, proxy_url: str) -> bool:
        """
        Connect to the proxy WebSocket server.

        Args:
            proxy_url: The WebSocket URL of the proxy

        Returns:
            bool: True if connection successful
        """
        try:
            self._websocket = await websockets.connect(
                proxy_url, ping_interval=20, ping_timeout=10, close_timeout=5
            )
            self._connected = True
            logger.info(f"Connected to proxy at {proxy_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to proxy: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the proxy server and stop the Discord bot."""
        self._running = False

        # Clean up voice resources
        await self._leave_voice_channel()

        if self._bot and not self._bot.is_closed():
            try:
                await self._bot.close()
                self._bot = None
            except Exception as e:
                logger.warning(f"Error while closing Discord bot: {e}")

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error while closing WebSocket: {e}")

        self._connected = False
        logger.info("Disconnected from Discord and proxy server")

    async def send_message(self, text: str) -> bool:
        """
        Send a text message to the VTuber through the proxy.

        Args:
            text: The message text

        Returns:
            bool: True if sent successfully
        """
        if not self.is_connected:
            logger.error("Cannot send message: Not connected to proxy")
            return False

        try:
            message = {"type": "text-input", "text": text}
            await self._websocket.send(json.dumps(message))
            logger.info(f"Sent Discord message to VTuber: {text}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to proxy: {e}")
            self._connected = False
            return False

    async def register_message_handler(
        self, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback for handling incoming messages.

        Args:
            handler: Function to call when a message is received
        """
        self._message_handlers.append(handler)
        logger.debug("Registered new message handler")

    async def start_receiving(self) -> None:
        """
        Start receiving messages from the proxy WebSocket.
        This runs in the background to receive messages from the VTuber.
        """
        if not self.is_connected:
            logger.error("Cannot start receiving: Not connected to proxy")
            return

        try:
            logger.info("Started receiving messages from proxy")
            while self._running and self.is_connected:
                try:
                    message = await self._websocket.recv()
                    data = json.loads(message)

                    # Log received message (truncate audio data for readability)
                    if "audio" in data:
                        log_data = data.copy()
                        log_data["audio"] = (
                            f"[Audio data, length: {len(data['audio'])}]"
                        )
                        logger.debug(f"Received message from VTuber: {log_data}")
                    else:
                        logger.debug(f"Received message from VTuber: {data}")

                    await self.handle_incoming_messages(data)

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed by server")
                    self._connected = False
                    break
                except Exception as e:
                    logger.error(f"Error receiving message from proxy: {e}")
                    await asyncio.sleep(1)

            logger.info("Stopped receiving messages from proxy")
        except Exception as e:
            logger.error(f"Error in message receiving loop: {e}")

    async def handle_incoming_messages(self, message: Dict[str, Any]) -> None:
        """
        Process messages received from the VTuber and relay to Discord.

        Args:
            message: The message received from the VTuber
        """
        msg_type = message.get("type", "")

        if msg_type == "audio" and self._current_response_channel:
            display_text = message.get("display_text", "")
            if display_text:
                # display_text can be a dict like {'text': '...', 'name': '...'}
                if isinstance(display_text, dict):
                    text = display_text.get("text", "")
                else:
                    text = str(display_text)
                # Strip emotion tags like [joy], [neutral], etc.
                text = re.sub(r"\[[\w]+\]\s*", "", text).strip()
                if text:
                    self._response_buffer += text + "\n"

            # If voice is enabled and we have a playback queue, enqueue the audio
            audio_data = message.get("audio")
            if audio_data and self._playback_queue:
                await self._playback_queue.enqueue(audio_data)

        elif msg_type == "control":
            control_text = message.get("text", "")
            if control_text == "conversation-chain-end":
                await self._flush_response_buffer()
                await self._signal_conversation_end()
            elif control_text == "interrupt":
                # Server VAD detected user speech start → interrupt playback
                if self._playback_queue:
                    self._playback_queue.interrupt()
                    logger.debug("Interrupted playback due to VAD speech start")
            elif control_text == "mic-audio-end":
                # Server VAD detected speech end → send mic-audio-end to trigger
                # the conversation pipeline (ASR → LLM → TTS)
                await self._send_mic_audio_end()

        elif msg_type == "backend-synth-complete":
            # TTS synthesis complete, flush any remaining response
            await self._flush_response_buffer()
            await self._signal_conversation_end()

        elif msg_type == "error":
            # Server error, flush and reset proxy queue
            await self._flush_response_buffer()
            await self._signal_conversation_end()

        elif msg_type == "full-text" and self._current_response_channel:
            text = message.get("text", "")
            # Ignore status messages like "Thinking..."
            if text and text != "Thinking...":
                self._response_buffer += text + "\n"

        # Process with registered handlers
        for handler in self._message_handlers:
            try:
                await asyncio.to_thread(handler, message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")

    async def _flush_response_buffer(self) -> None:
        """Send accumulated response buffer to the Discord channel."""
        if not self._response_buffer or not self._current_response_channel:
            self._response_buffer = ""
            return

        text = self._response_buffer.strip()
        self._response_buffer = ""

        if not text:
            return

        try:
            # Discord has a 2000 character limit per message
            for chunk in _split_message(text, max_length=2000):
                await self._current_response_channel.send(chunk)
            logger.info(f"Sent response to Discord: {text[:100]}")
        except Exception as e:
            logger.error(f"Failed to send response to Discord: {e}")

    async def _signal_conversation_end(self) -> None:
        """Send interrupt signal to proxy to reset the message queue."""
        if not self.is_connected:
            return
        try:
            signal = {"type": "interrupt-signal"}
            await self._websocket.send(json.dumps(signal))
            logger.debug("Sent conversation end signal to proxy")
        except Exception as e:
            logger.warning(f"Failed to send end signal: {e}")

    async def _send_mic_audio_end(self) -> None:
        """
        Send mic-audio-end to the proxy server to trigger the
        ASR → LLM → TTS conversation pipeline after VAD detects speech end.
        """
        if not self.is_connected:
            return
        try:
            message = {"type": "mic-audio-end"}
            await self._websocket.send(json.dumps(message))
            logger.debug("Sent mic-audio-end to proxy")
        except Exception as e:
            logger.warning(f"Failed to send mic-audio-end: {e}")

    # ── Voice channel methods ──────────────────────────────────────────

    async def _join_voice_channel(self, channel: "discord.VoiceChannel") -> bool:
        """
        Join a Discord voice channel, start the audio playback queue,
        and begin voice capture if discord-ext-voice-recv is available.

        Args:
            channel: The voice channel to join.

        Returns:
            True if successfully joined.
        """
        from .discord_voice import AudioPlaybackQueue

        try:
            # Disconnect from any existing voice channel first
            if self._voice_client and self._voice_client.is_connected():
                await self._leave_voice_channel()

            # Try to use VoiceRecvClient for voice capture support
            recv_cls = None
            try:
                from discord.ext import voice_recv

                recv_cls = voice_recv.VoiceRecvClient
            except ImportError:
                logger.info(
                    "discord-ext-voice-recv not available, "
                    "voice capture disabled (playback only)"
                )

            if recv_cls is not None:
                self._voice_client = await channel.connect(cls=recv_cls)
            else:
                self._voice_client = await channel.connect()

            self._playback_queue = AudioPlaybackQueue(self._voice_client)
            self._playback_queue.start()

            # Start voice capture if possible
            await self._start_voice_capture()

            logger.info(f"Joined voice channel: {channel.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to join voice channel: {e}")
            self._voice_client = None
            self._playback_queue = None
            self._voice_sink = None
            return False

    async def _start_voice_capture(self) -> None:
        """
        Start capturing voice audio from the Discord voice channel.
        Requires discord-ext-voice-recv and an active VoiceRecvClient.
        """
        from .discord_voice import _create_voice_sink

        if not self._voice_client or not self.is_connected:
            return

        # Check if the voice client supports listening
        if not hasattr(self._voice_client, "listen"):
            logger.debug("Voice client does not support listen(), skipping capture")
            return

        if not self._bot or not self._bot.user:
            logger.warning("Bot user not available, cannot start voice capture")
            return

        bot_user_id = self._bot.user.id
        event_loop = asyncio.get_event_loop()

        sink = _create_voice_sink(self._websocket, bot_user_id, event_loop)
        if sink is None:
            return

        self._voice_sink = sink
        self._voice_client.listen(sink)
        logger.info("Voice capture started")

    async def _leave_voice_channel(self) -> None:
        """Leave the current voice channel and clean up resources."""
        # Stop voice capture
        if self._voice_sink:
            try:
                self._voice_sink.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up voice sink: {e}")
            self._voice_sink = None

        if self._voice_client and hasattr(self._voice_client, "stop_listening"):
            try:
                self._voice_client.stop_listening()
            except Exception as e:
                logger.warning(f"Error stopping voice listening: {e}")

        if self._playback_queue:
            try:
                await self._playback_queue.stop()
            except Exception as e:
                logger.warning(f"Error stopping playback queue: {e}")
            self._playback_queue = None

        if self._voice_client and self._voice_client.is_connected():
            try:
                await self._voice_client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting voice client: {e}")

        self._voice_client = None
        logger.info("Left voice channel")

    async def run(self) -> None:
        """
        Main entry point for running the Discord Live platform client.
        Connects to Discord and the proxy, and starts monitoring messages.
        """
        receive_task = None

        try:
            self._running = True

            # Connect to the proxy
            if not await self.connect(self._proxy_url):
                logger.error("Failed to connect to proxy, exiting")
                return

            # Start background task for receiving messages from the proxy
            receive_task = asyncio.create_task(self.start_receiving())

            # Set up Discord bot
            intents = discord.Intents.default()
            try:
                intents.message_content = True
            except Exception:
                logger.warning(
                    "Could not enable message_content intent. "
                    "Bot will only respond to @mentions."
                )

            # Enable voice state intent for auto-leave when channel empties
            if self._voice_enabled:
                intents.voice_states = True

            self._bot = discord.Client(intents=intents)
            platform = self  # reference for closures

            @self._bot.event
            async def on_ready():
                logger.info(f"Discord bot logged in as {platform._bot.user}")
                if platform._channel_ids:
                    logger.info(f"Listening on channels: {platform._channel_ids}")
                else:
                    logger.info("Listening on all channels (no filter)")
                if platform._voice_enabled:
                    logger.info("Voice channel support is enabled")

            @self._bot.event
            async def on_message(msg: discord.Message):
                # Ignore messages from the bot itself
                if msg.author == platform._bot.user:
                    return

                # Filter by channel IDs if configured
                if (
                    platform._channel_ids
                    and msg.channel.id not in platform._channel_ids
                ):
                    return

                content = msg.content

                # Check prefix if configured
                if platform._prefix:
                    if not content.startswith(platform._prefix):
                        return
                    content = content[len(platform._prefix) :].strip()

                if not content:
                    return

                # Handle voice commands when voice is enabled
                if platform._voice_enabled:
                    lower_content = content.lower()

                    if lower_content == "join":
                        await platform._handle_join_command(msg)
                        return

                    if lower_content == "leave":
                        await platform._handle_leave_command(msg)
                        return

                logger.info(f"[Discord] {msg.author.display_name}: {content}")

                # Set the channel for responses
                platform._current_response_channel = msg.channel
                platform._response_buffer = ""

                # Interrupt current playback when a new message arrives
                if platform._playback_queue:
                    platform._playback_queue.interrupt()

                # Send typing indicator
                async with msg.channel.typing():
                    await platform.send_message(content)

            if self._voice_enabled:

                @self._bot.event
                async def on_voice_state_update(
                    member: discord.Member,
                    before: discord.VoiceState,
                    after: discord.VoiceState,
                ):
                    """Auto-leave when all users leave the voice channel."""
                    if not platform._voice_client:
                        return
                    if not platform._voice_client.is_connected():
                        return

                    channel = platform._voice_client.channel
                    if channel is None:
                        return

                    # Count non-bot members remaining in the channel
                    members = [m for m in channel.members if not m.bot]
                    if len(members) == 0:
                        logger.info("All users left the voice channel, auto-leaving")
                        if platform._current_response_channel:
                            try:
                                await platform._current_response_channel.send(
                                    "Everyone left the voice channel. Leaving."
                                )
                            except Exception:
                                pass
                        await platform._leave_voice_channel()

            # Start the Discord bot (this blocks until the bot stops)
            logger.info("Starting Discord bot...")
            await self._bot.start(self._bot_token)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        except Exception as e:
            logger.error(f"Error in Discord run loop: {e}")
            logger.debug(traceback.format_exc())
        finally:
            if receive_task and not receive_task.done():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
            await self.disconnect()

    async def _handle_join_command(self, msg: discord.Message) -> None:
        """
        Handle the 'join' voice command. The bot joins the user's voice channel.

        Args:
            msg: The Discord message that triggered this command.
        """
        if not msg.author.voice or not msg.author.voice.channel:
            await msg.channel.send("You need to be in a voice channel first!")
            return

        voice_channel = msg.author.voice.channel
        success = await self._join_voice_channel(voice_channel)

        if success:
            await msg.channel.send(
                f"Joined **{voice_channel.name}**! I'll play TTS responses here."
            )
        else:
            await msg.channel.send(
                "Failed to join the voice channel. Check bot permissions."
            )

    async def _handle_leave_command(self, msg: discord.Message) -> None:
        """
        Handle the 'leave' voice command. The bot leaves the voice channel.

        Args:
            msg: The Discord message that triggered this command.
        """
        if not self._voice_client or not self._voice_client.is_connected():
            await msg.channel.send("I'm not in a voice channel.")
            return

        await self._leave_voice_channel()
        await msg.channel.send("Left the voice channel.")


def _split_message(text: str, max_length: int = 2000) -> List[str]:
    """
    Split a message into chunks that fit within Discord's message limit.

    Args:
        text: The text to split
        max_length: Maximum length per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to split at a newline
        split_idx = text.rfind("\n", 0, max_length)
        if split_idx == -1:
            # Try to split at a space
            split_idx = text.rfind(" ", 0, max_length)
        if split_idx == -1:
            # Hard split
            split_idx = max_length

        chunks.append(text[:split_idx])
        text = text[split_idx:].lstrip("\n")

    return chunks
