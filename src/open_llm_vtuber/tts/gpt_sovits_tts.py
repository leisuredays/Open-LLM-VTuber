"""
GPT-SoVITS TTS engine for Open-LLM-VTuber.
Windows path: uses mashiro-tts package (MashiroTTS + A2F).
Linux/fallback path: direct HTTP POST to GPT-SoVITS API.
"""

import os
import re
import time
from loguru import logger
from .tts_interface import TTSInterface

try:
    from mashiro_tts import MashiroTTS
    _MASHIRO_AVAILABLE = True
except ImportError:
    _MASHIRO_AVAILABLE = False
    logger.warning("mashiro_tts not available — using direct HTTP fallback (no A2F)")


class TTSEngine(TTSInterface):
    # Emotion → reference audio mapping
    EMOTION_REF_MAP = {
        "angry": {
            "ref_audio_path": "C:/Users/leisu/OneDrive/Desktop/yuzuha_angry.wav",
            "prompt_text": "그 초콜릿은 함부로 먹으면 안돼! 니가 먹고 있는 그 초콜릿은... 아니 그걸 대체 어떻게 구한 거야?",
        },
        "annoyed": {
            "ref_audio_path": "C:/Users/leisu/OneDrive/Desktop/yuzuha_angry.wav",
            "prompt_text": "그 초콜릿은 함부로 먹으면 안돼! 니가 먹고 있는 그 초콜릿은... 아니 그걸 대체 어떻게 구한 거야?",
        },
        "shocked": {
            "ref_audio_path": "C:/Users/leisu/OneDrive/Desktop/yuzuha_surprised.wav",
            "prompt_text": "헉! 잠깐... 니 손에 들린 그건...",
        },
        "sleepy": {
            "ref_audio_path": "C:/Users/leisu/OneDrive/Desktop/yuzuha_sleepy.wav",
            "prompt_text": "어휴... 꼬마야... 이제 좀 알겠네.",
        },
        "excited": {
            "ref_audio_path": "C:/Users/leisu/OneDrive/Desktop/yuzuha_curious.wav",
            "prompt_text": "어라 꼬마야, 무슨 일이야? 왜이렇게 괴로워하고 있어?",
        },
    }

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
        streaming_mode: str = "true",
    ):
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode

        if _MASHIRO_AVAILABLE:
            a2f_url = os.environ.get("A2F_SERVER_URL", "http://192.168.0.3:9872")
            self._tts = MashiroTTS(
                api_url=api_url,
                ref_audio_path=ref_audio_path,
                prompt_text=prompt_text,
                text_lang=text_lang,
                prompt_lang=prompt_lang,
                text_split_method=text_split_method,
                batch_size=batch_size,
                media_type=media_type,
                streaming_mode=(streaming_mode == "true"),
                emotion_map=self.EMOTION_REF_MAP,
                normalize=True,
                a2f_enabled=True,
                a2f_url=a2f_url,
            )
        else:
            self._tts = None

    def _get_emotion_refs(self, emotion: str):
        """Return (ref_audio_path, prompt_text) for the given emotion."""
        if emotion and emotion in self.EMOTION_REF_MAP:
            m = self.EMOTION_REF_MAP[emotion]
            return m["ref_audio_path"], m["prompt_text"]
        return self.ref_audio_path, self.prompt_text

    def _fallback_generate(self, text: str, file_name: str, emotion: str = None) -> str:
        """Direct HTTP POST fallback (used when mashiro_tts is not available)."""
        import requests as _requests

        cleaned = re.sub(r"\[.*?\]", "", text)
        ref_audio, prompt_text = self._get_emotion_refs(emotion)

        # streaming_mode can be "0"/"1"/"2"/"3" or "true"/"false"
        try:
            sm_int = int(self.streaming_mode)
        except (ValueError, TypeError):
            sm_int = 0

        data = {
            "text": cleaned,
            "text_lang": self.text_lang,
            "ref_audio_path": ref_audio,
            "prompt_lang": self.prompt_lang,
            "prompt_text": prompt_text,
            "text_split_method": self.text_split_method,
            "batch_size": int(self.batch_size),
            "media_type": self.media_type,
            "streaming_mode": sm_int,
        }

        resp = _requests.post(self.api_url, json=data, timeout=120)
        resp.raise_for_status()

        with open(file_name, "wb") as f:
            f.write(resp.content)

        return file_name

    def generate_audio(self, text, file_name_no_ext=None, emotion: str = None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)
        logger.info(f"TTS input (raw): {text}")

        try:
            t0 = time.perf_counter()

            if not _MASHIRO_AVAILABLE:
                file_path = self._fallback_generate(text, file_name, emotion)
                logger.info(f"⏱ TTS fallback: {(time.perf_counter()-t0)*1000:.0f}ms")
                return file_path

            if self.streaming_mode == "true":
                file_path, a2f_frames = self._tts.generate_with_a2f(
                    text, file_name, emotion=emotion
                )
                logger.info(
                    f"⏱ TTS+A2F pipeline: total={(time.perf_counter()-t0)*1000:.0f}ms"
                    f" | frames={len(a2f_frames)}"
                )
                return file_path
            else:
                file_path = self._tts.generate_to_file(text, file_name, emotion=emotion)
                logger.info(f"⏱ TTS total: {(time.perf_counter()-t0)*1000:.0f}ms")
                return file_path

        except Exception as e:
            logger.error(f"GPT-SoVITS error: {e}")
            return None

    async def async_generate_audio_streaming(self, text: str, emotion: str = None):
        """
        Generate audio in streaming mode, yielding chunks as they arrive.

        Yields tuples of (audio_chunk_bytes, a2f_frames_so_far, metadata).
        First yield includes metadata dict with WAV header info.
        """
        logger.info(f"TTS streaming input: {text}")

        if not _MASHIRO_AVAILABLE:
            # Fallback: generate full audio, yield as single chunk
            import tempfile, asyncio
            tmp = tempfile.mktemp(suffix=f".{self.media_type}")
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._fallback_generate(text, tmp, emotion)
            )
            with open(tmp, "rb") as f:
                audio_bytes = f.read()
            import os as _os
            _os.unlink(tmp)
            yield (audio_bytes, [], {"fallback": True})
            return

        try:
            t0 = time.perf_counter()
            async for audio_chunk, a2f_frames, metadata in self._tts.stream_with_a2f(
                text, emotion=emotion
            ):
                yield (audio_chunk, a2f_frames, metadata)
            logger.info(
                f"⏱ TTS streaming pipeline complete: {(time.perf_counter()-t0)*1000:.0f}ms"
            )
        except Exception as e:
            logger.error(f"GPT-SoVITS streaming error: {e}")
            raise
