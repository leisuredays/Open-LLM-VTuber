####
# change from xTTS.py
####

import re
import requests
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
                # Save the audio content to a file
                with open(file_name, "wb") as audio_file:
                    if self.streaming_mode > 0:
                        # Stream mode: receive chunks
                        # First chunk contains WAV header, subsequent chunks are raw PCM
                        chunk_count = 0
                        total_bytes = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                audio_file.write(chunk)
                                chunk_count += 1
                                total_bytes += len(chunk)
                        logger.debug(f"🎵 Received {chunk_count} chunks, total {total_bytes} bytes")
                    else:
                        # Non-streaming: receive all at once
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
