import abc
import os
import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger


class TTSInterface(metaclass=abc.ABCMeta):
    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Asynchronously generate speech audio file using TTS.

        By default, this runs the synchronous generate_audio in a coroutine.
        Subclasses can override this method to provide true async implementation.

        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file

        """
        return await asyncio.to_thread(self.generate_audio, text, file_name_no_ext)

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks as they're generated (optional, not all TTS engines support this).

        This method enables real-time audio streaming for supported TTS engines.
        If not implemented, the TTS system will fall back to the standard generate_audio method.

        Args:
            text: The text to synthesize

        Yields:
            bytes: Audio chunks as they're generated

        Raises:
            NotImplementedError: If the TTS engine doesn't support streaming
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support audio streaming")
        # Make this a generator to satisfy type hints
        yield  # This line is never reached but satisfies AsyncGenerator type

    @abc.abstractmethod
    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file

        """
        raise NotImplementedError

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        """
        Remove a file from the file system.

        This is a separate method instead of a part of the `play_audio_file_local()` because `play_audio_file_local()` is not the only way to play audio files. This method will be used to remove the audio file after it has been played.

        Parameters:
            filepath (str): The path to the file to remove.
            verbose (bool): If True, print messages to the console.
        """
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist")
            return
        try:
            logger.debug(f"Removing file {filepath}") if verbose else None
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to remove file {filepath}: {e}")

    def generate_cache_file_name(self, file_name_no_ext=None, file_extension="wav"):
        """
        Generate a cross-platform cache file name.

        file_name_no_ext: str
            name of the file without extension
        file_extension: str
            file extension

        Returns:
        str: the path to the generated cache file
        """
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if file_name_no_ext is None:
            file_name_no_ext = "temp"

        file_name = f"{file_name_no_ext}.{file_extension}"
        return os.path.join(cache_dir, file_name)
