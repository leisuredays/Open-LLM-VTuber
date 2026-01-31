import re
import unicodedata
from loguru import logger
from ..translate.translate_interface import TranslateInterface


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting markers while preserving the content text.

    Handles: bold, italic, strikethrough, inline code, code blocks,
    headers, bullet/numbered lists, links, images, blockquotes, horizontal rules.

    Args:
        text: The text with potential markdown formatting.

    Returns:
        The text with markdown markers removed but content preserved.
    """
    # Remove code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove inline code (` ... `)
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove images ![alt](url) → alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Remove links [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Remove bold/italic markers: ***text***, **text**, *text*,
    # ___text___, __text__, _text_ (word-boundary aware for underscores)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    # Remove strikethrough ~~text~~ → text
    text = re.sub(r"~~(.+?)~~", r"\1", text)

    # Remove horizontal rules (---, ***, ___ on their own line)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove headers (# Header → Header)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove blockquote markers (> text → text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # Remove unordered list markers (- item, * item)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)

    # Remove ordered list markers (1. item → item)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def convert_units_to_spoken_korean(text: str) -> str:
    """
    Convert units and symbols to spoken Korean form for TTS.

    Handles temperature (°C, °F, ℃, ℉), percentage (%), distance (km, m, cm, mm),
    speed (km/h), weight (kg, g), volume (ml, mL, L, l), and negative temperatures.

    Args:
        text: The text containing units and symbols.

    Returns:
        The text with units converted to spoken Korean.
    """
    # Negative temperature: -3°C → 영하 3도 (must come before general °C)
    text = re.sub(r"-\s*(\d+(?:\.\d+)?)\s*(?:°C|℃)", r"영하 \1도", text)
    # Negative temperature Fahrenheit
    text = re.sub(r"-\s*(\d+(?:\.\d+)?)\s*(?:°F|℉)", r"영하 화씨 \1도", text)

    # Positive temperature: 25°C → 25도
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:°C|℃)", r"\1도", text)
    # Fahrenheit: 77°F → 화씨 77도
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:°F|℉)", r"화씨 \1도", text)

    # Percentage: 52% → 52퍼센트
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1퍼센트", text)

    # Speed: km/h must come before km
    text = re.sub(r"(\d+(?:\.\d+)?)\s*km/h(?![a-zA-Z])", r"\1킬로미터", text)

    # Distance/length (longer units first to avoid partial matches)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*km(?![a-zA-Z/])", r"\1킬로미터", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*cm(?![a-zA-Z])", r"\1센티미터", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*mm(?![a-zA-Z])", r"\1밀리미터", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*m(?![a-zA-Z])", r"\1미터", text)

    # Weight (kg before g)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*kg(?![a-zA-Z])", r"\1킬로그램", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*g(?![a-zA-Z])", r"\1그램", text)

    # Volume (mL/ml before L/l)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:mL|ml)(?![a-zA-Z])", r"\1밀리리터", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:L|l)(?![a-zA-Z])", r"\1리터", text)

    return text


def tts_filter(
    text: str,
    remove_special_char: bool,
    ignore_brackets: bool,
    ignore_parentheses: bool,
    ignore_asterisks: bool,
    ignore_angle_brackets: bool,
    strip_markdown_formatting: bool = True,
    convert_units_to_spoken: bool = False,
    translator: TranslateInterface | None = None,
) -> str:
    """
    Filter or do anything to the text before TTS generates the audio.
    Changes here do not affect subtitles or LLM's memory. The generated audio is
    the only affected thing.

    Args:
        text (str): The text to filter.
        remove_special_char (bool): Whether to remove special characters.
        ignore_brackets (bool): Whether to ignore text within brackets.
        ignore_parentheses (bool): Whether to ignore text within parentheses.
        ignore_asterisks (bool): Whether to ignore text within asterisks.
        ignore_angle_brackets (bool): Whether to ignore text within angle brackets.
        strip_markdown_formatting (bool): Whether to strip markdown formatting markers.
        convert_units_to_spoken (bool): Whether to convert units to spoken Korean.
        translator (TranslateInterface, optional):
            The translator to use. If None, we'll skip the translation. Defaults to None.

    Returns:
        str: The filtered text.
    """
    # Step 1: Strip markdown formatting (before filter_asterisks to preserve content)
    if strip_markdown_formatting:
        try:
            text = strip_markdown(text)
        except Exception as e:
            logger.warning(f"Error stripping markdown: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")

    # Step 2: Convert units to spoken Korean (before remove_special_characters)
    if convert_units_to_spoken:
        try:
            text = convert_units_to_spoken_korean(text)
        except Exception as e:
            logger.warning(f"Error converting units: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")

    # Step 3: Existing filters
    if ignore_asterisks:
        try:
            text = filter_asterisks(text)
        except Exception as e:
            logger.warning(f"Error ignoring asterisks: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")

    if ignore_brackets:
        try:
            text = filter_brackets(text)
        except Exception as e:
            logger.warning(f"Error ignoring brackets: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if ignore_parentheses:
        try:
            text = filter_parentheses(text)
        except Exception as e:
            logger.warning(f"Error ignoring parentheses: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if ignore_angle_brackets:
        try:
            text = filter_angle_brackets(text)
        except Exception as e:
            logger.warning(f"Error ignoring angle brackets: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if remove_special_char:
        try:
            text = remove_special_characters(text)
        except Exception as e:
            logger.warning(f"Error removing special characters: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if translator:
        try:
            logger.info("Translating...")
            text = translator.translate(text)
            logger.info(f"Translated: {text}")
        except Exception as e:
            logger.critical(f"Error translating: {e}")
            logger.critical(f"Text: {text}")
            logger.warning("Skipping...")

    logger.debug(f"Filtered text: {text}")
    return text


def remove_special_characters(text: str) -> str:
    """
    Filter text to remove all non-letter, non-number, and non-punctuation characters.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    normalized_text = unicodedata.normalize("NFKC", text)

    def is_valid_char(char: str) -> bool:
        category = unicodedata.category(char)
        return (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("P")
            or char.isspace()
        )

    filtered_text = "".join(char for char in normalized_text if is_valid_char(char))
    return filtered_text


def _filter_nested(text: str, left: str, right: str) -> str:
    """
    Generic function to handle nested symbols.

    Args:
        text (str): The text to filter.
        left (str): The left symbol (e.g. '[' or '(').
        right (str): The right symbol (e.g. ']' or ')').

    Returns:
        str: The filtered text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not text:
        return text

    result = []
    depth = 0
    for char in text:
        if char == left:
            depth += 1
        elif char == right:
            if depth > 0:
                depth -= 1
        else:
            if depth == 0:
                result.append(char)
    filtered_text = "".join(result)
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
    return filtered_text


def filter_brackets(text: str) -> str:
    """
    Filter text to remove all text within brackets, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "[", "]")


def filter_parentheses(text: str) -> str:
    """
    Filter text to remove all text within parentheses, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "(", ")")


def filter_angle_brackets(text: str) -> str:
    """
    Filter text to remove all text within angle brackets, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "<", ">")


def filter_asterisks(text: str) -> str:
    """
    Removes text enclosed within asterisks of any length (*, **, ***, etc.) from a string.

    Args:
        text: The input string.

    Returns:
        The string with asterisk-enclosed text removed.
    """
    # Handle asterisks of any length (*, **, ***, etc.)
    filtered_text = re.sub(r"\*{1,}((?!\*).)*?\*{1,}", "", text)

    # Clean up any extra spaces
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()

    return filtered_text
