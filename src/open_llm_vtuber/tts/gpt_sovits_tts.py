"""
GPT-SoVITS TTS engine for Open-LLM-VTuber.
Powered by mashiro-tts package.
"""

import os
import time
from loguru import logger
from .tts_interface import TTSInterface
from mashiro_tts import MashiroTTS


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
        # A2F config
        a2f_url = os.environ.get("A2F_SERVER_URL", "http://192.168.0.3:9872")

        # Create MashiroTTS instance
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

        # Keep raw config for backward compat
        self.api_url = api_url
        self.text_lang = text_lang
        self.ref_audio_path = ref_audio_path
        self.prompt_lang = prompt_lang
        self.prompt_text = prompt_text
        self.text_split_method = text_split_method
        self.batch_size = batch_size
        self.media_type = media_type
        self.streaming_mode = streaming_mode

    def generate_audio(self, text, file_name_no_ext=None, emotion: str = None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.media_type)

        logger.info(f"TTS input (raw): {text}")

        try:
            t0 = time.perf_counter()

            if self.streaming_mode == "true":
                # Streaming + parallel A2F
                file_path, a2f_frames = self._tts.generate_with_a2f(
                    text, file_name, emotion=emotion
                )
                t_total = time.perf_counter() - t0
                logger.info(
                    f"⏱ TTS+A2F pipeline: total={t_total*1000:.0f}ms | frames={len(a2f_frames)}"
                )
                return file_path
            else:
                # Non-streaming: generate then A2F
                file_path = self._tts.generate_to_file(
                    text, file_name, emotion=emotion
                )
                t_total = time.perf_counter() - t0
                logger.info(f"⏱ TTS total: {t_total*1000:.0f}ms")
                return file_path

        except Exception as e:
            logger.error(f"GPT-SoVITS error: {e}")
            return None

    async def async_generate_audio_streaming(self, text: str, emotion: str = None):
        """
        Generate audio in streaming mode, yielding chunks as they arrive.
        
        Yields tuples of (audio_chunk_bytes, a2f_frames_so_far, metadata).
        First yield includes metadata dict with WAV header info.
        
        Args:
            text: Text to synthesize
            emotion: Optional emotion for reference selection
            
        Yields:
            (audio_chunk_bytes, a2f_frames_so_far, metadata_dict)
        """
        logger.info(f"TTS streaming input: {text}")
        
        try:
            import time
            t0 = time.perf_counter()
            
            async for audio_chunk, a2f_frames, metadata in self._tts.stream_with_a2f(
                text, emotion=emotion
            ):
                yield (audio_chunk, a2f_frames, metadata)
            
            t_total = time.perf_counter() - t0
            logger.info(f"⏱ TTS streaming pipeline complete: {t_total*1000:.0f}ms")
            
        except Exception as e:
            logger.error(f"GPT-SoVITS streaming error: {e}")
            raise
