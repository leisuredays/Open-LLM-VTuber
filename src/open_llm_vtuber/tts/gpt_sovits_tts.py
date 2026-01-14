####
# change from xTTS.py
####

import re
import requests
from io import BytesIO
from typing import AsyncGenerator
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:9880/tts",
        text_lang: str = "zh",
        ref_audio_path: str = "",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        text_split_method: str = "cut5",
        batch_size: str = "1",
        media_type: str = "wav",
        streaming_mode: int | str = 2,  # Default to medium quality streaming
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type

        # Convert streaming_mode to int if it's a string
        if isinstance(streaming_mode, str):
            if streaming_mode.lower() in ['false', '0']:
                self.streaming_mode = 0
            elif streaming_mode.lower() in ['true', '1']:
                self.streaming_mode = 1
            else:
                try:
                    self.streaming_mode = int(streaming_mode)
                except ValueError:
                    logger.warning(f"Invalid streaming_mode '{streaming_mode}', defaulting to 2 (medium quality)")
                    self.streaming_mode = 2
        else:
            self.streaming_mode = streaming_mode

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        cleaned_text = re.sub(r"\[.*?\]", "", text)

        # Prepare the data for the request
        data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": self.prompt_lang,
            "prompt_text": self.prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,

            # --- WebUI settings ---
            "speed_factor": getattr(self, "speed_factor", 1.0),           # Speech rate
            "fragment_interval": getattr(self, "fragment_interval", 0.3), # Pause Duration
            "top_k": getattr(self, "top_k", 15),                         # top_k
            "top_p": getattr(self, "top_p", 1.0),                        # top_p
            "temperature": getattr(self, "temperature", 1.0),            # temperature
            "seed": getattr(self, "seed", -1),                           # Randomness control
            "repetition_penalty": getattr(self, "repetition_penalty", 1.35) # Repetition penalty
        }

        try:
            # Send GET request with streaming enabled
            # When streaming_mode > 0, GPT-SoVITS returns chunked audio
            if self.streaming_mode > 0:
                logger.debug(f"🎵 Streaming mode {self.streaming_mode} enabled - receiving audio chunks")
                response = requests.get(self.api_url, params=data, timeout=120, stream=True)
            else:
                logger.debug(f"🎵 Streaming disabled - receiving full audio")
                response = requests.get(self.api_url, params=data, timeout=120)

            # Check if the request was successful
            if response.status_code == 200:
                if self.streaming_mode > 0:
                    # Stream mode: receive chunks into memory buffer
                    # GPT-SoVITS sends: 1st chunk = WAV header, rest = audio data
                    logger.debug(f"🎵 Streaming mode {self.streaming_mode} - receiving chunks")

                    audio_buffer = BytesIO()
                    chunk_count = 0
                    total_bytes = 0

                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            audio_buffer.write(chunk)
                            chunk_count += 1
                            total_bytes += len(chunk)

                    logger.debug(f"🎵 Received {chunk_count} chunks, total {total_bytes} bytes")

                    # Validate and fix WAV file if needed
                    audio_buffer.seek(0)
                    try:
                        from pydub import AudioSegment

                        # Load audio from buffer and re-export to ensure valid WAV
                        audio = AudioSegment.from_file(audio_buffer, format="wav")
                        audio.export(file_name, format="wav")

                        logger.debug(f"🎵 WAV file validated and saved: duration={len(audio)}ms")
                    except Exception as e:
                        # If validation fails, save raw data
                        logger.warning(f"🎵 WAV validation failed, saving raw data: {e}")
                        audio_buffer.seek(0)
                        with open(file_name, "wb") as f:
                            f.write(audio_buffer.read())
                else:
                    # Non-streaming: receive all at once
                    with open(file_name, "wb") as audio_file:
                        audio_file.write(response.content)
                    logger.debug(f"🎵 Received {len(response.content)} bytes")

                logger.info(f"🎵 Audio saved to {file_name}")
                return file_name
            else:
                # Handle errors or unsuccessful requests
                logger.critical(
                    f"Error: Failed to generate audio. Status code: {response.status_code}"
                )
                return None

        except requests.exceptions.RequestException as e:
            logger.critical(f"Error: Request failed: {e}")
            return None

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks in real-time from GPT-SoVITS API.

        First chunk contains WAV header, subsequent chunks contain PCM audio data.

        Args:
            text: Text to synthesize

        Yields:
            bytes: Audio chunks (first chunk = WAV header, rest = PCM data)
        """
        if self.streaming_mode == 0:
            raise ValueError("Streaming is disabled (streaming_mode=0). Use generate_audio() instead.")

        cleaned_text = re.sub(r"\[.*?\]", "", text)

        # Prepare the data for the request
        data = {
            "text": cleaned_text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "prompt_lang": self.prompt_lang,
            "prompt_text": self.prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "media_type": self.media_type,
            "streaming_mode": self.streaming_mode,

            # WebUI settings
            "speed_factor": getattr(self, "speed_factor", 1.0),
            "fragment_interval": getattr(self, "fragment_interval", 0.3),
            "top_k": getattr(self, "top_k", 15),
            "top_p": getattr(self, "top_p", 1.0),
            "temperature": getattr(self, "temperature", 1.0),
            "seed": getattr(self, "seed", -1),
            "repetition_penalty": getattr(self, "repetition_penalty", 1.35)
        }

        try:
            logger.debug(f"🎵 Streaming audio chunks for: '{cleaned_text[:50]}...'")
            response = requests.get(self.api_url, params=data, timeout=120, stream=True)

            if response.status_code == 200:
                chunk_count = 0
                total_bytes = 0

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunk_count += 1
                        total_bytes += len(chunk)

                        if chunk_count == 1:
                            logger.debug(f"🎵 Yielding WAV header chunk ({len(chunk)} bytes)")

                        yield chunk

                logger.info(f"🎵 Streamed {chunk_count} chunks, total {total_bytes} bytes")
            else:
                logger.error(f"Error: Failed to stream audio. Status code: {response.status_code}")
                raise RuntimeError(f"GPT-SoVITS API returned status {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error: Stream request failed: {e}")
            raise
