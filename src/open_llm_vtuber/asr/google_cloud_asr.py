"""Google Cloud Speech-to-Text ASR module."""

import io
import base64
import logging
import numpy as np
import requests

from .asr_interface import ASRInterface

logger = logging.getLogger(__name__)


class VoiceRecognition(ASRInterface):
    """Google Cloud Speech-to-Text v1 REST API."""

    def __init__(self, api_key: str = "", language: str = "ko-KR", model: str = "latest_short", **kwargs):
        self.api_key = api_key
        self.language = language
        self.model = model
        self.url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"
        logger.info(f"Initialized Google Cloud ASR (lang={language}, model={model})")

    def transcribe_np(self, audio: np.ndarray) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to 16-bit PCM
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

        # Encode as WAV
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        audio_bytes = buf.getvalue()

        payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": self.SAMPLE_RATE,
                "languageCode": self.language,
                "model": self.model,
                "enableAutomaticPunctuation": True,
            },
            "audio": {
                "content": base64.b64encode(audio_bytes).decode("utf-8")
            }
        }

        try:
            resp = requests.post(self.url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                transcript = results[0]["alternatives"][0]["transcript"]
                logger.info(f"Google STT: '{transcript}'")
                return transcript
            return ""
        except Exception as e:
            logger.error(f"Google Cloud STT error: {e}")
            return ""
