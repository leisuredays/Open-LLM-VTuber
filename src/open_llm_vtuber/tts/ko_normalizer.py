"""
Korean text normalizer for TTS preprocessing.
Converts numbers, English abbreviations, and symbols to Korean pronunciation.
Extensible — add new rules to CUSTOM_REPLACEMENTS or new handler functions as needed.
"""

import re
from typing import Optional

# ── Digit-to-Korean mappings ──────────────────────────────────────────────

_DIGITS = {
    "0": "영", "1": "일", "2": "이", "3": "삼", "4": "사",
    "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
}

_UNITS = [
    (1_0000_0000, "억"),
    (1_0000, "만"),
    (1000, "천"),
    (100, "백"),
    (10, "십"),
]


def _int_to_korean(n: int) -> str:
    """Convert a non-negative integer to Korean reading."""
    if n == 0:
        return "영"
    if n < 0:
        return "마이너스 " + _int_to_korean(-n)

    result = ""
    for unit_val, unit_name in _UNITS:
        if n >= unit_val:
            digit = n // unit_val
            if digit > 1 or unit_val >= 10000:
                result += _int_to_korean(digit)
            result += unit_name
            n %= unit_val
    if n > 0:
        # remaining single digits
        for ch in str(n):
            result += _DIGITS[ch]
    return result


def _number_to_korean(match: re.Match) -> str:
    """Regex replacement handler for numbers (integers and decimals)."""
    text = match.group(0)

    # Percentage: 99.9% → 구십구점구퍼센트
    if text.endswith("%"):
        num_part = text[:-1]
        return _number_str_to_korean(num_part) + "퍼센트"

    return _number_str_to_korean(text)


def _number_str_to_korean(text: str) -> str:
    """Convert a numeric string (int or decimal) to Korean."""
    if "." in text:
        int_part, dec_part = text.split(".", 1)
        int_korean = _int_to_korean(int(int_part)) if int_part else "영"
        dec_korean = "".join(_DIGITS.get(ch, ch) for ch in dec_part)
        return int_korean + "쩜" + dec_korean
    else:
        return _int_to_korean(int(text))


# ── Version numbers ──────────────────────────────────────────────────────

def _version_to_korean(match: re.Match) -> str:
    """Convert version string like v10.13.0 → 브이 일영점일삼점영"""
    ver = match.group(1)
    parts = ver.split(".")
    korean_parts = []
    for part in parts:
        # Read each digit individually for version components
        korean_parts.append("".join(_DIGITS.get(ch, ch) for ch in part))
    return "브이 " + "쩜".join(korean_parts)


# ── Korean slang / initial-consonant abbreviations ───────────────────────

KOREAN_SLANG: dict[str, str] = {
    "ㅎㅇ": "하잉",
    "ㅂㅇ": "바이",
    "ㄱㄱ": "고고",
    "ㅇㅇ": "응응",
    "ㄴㄴ": "노노",
    "ㅇㅋ": "오케이",
    "ㄱㅅ": "감사",
    "ㅈㅅ": "죄송",
    "ㄷㄷ": "덜덜",
    "ㅎㅎ": "하하",
    "ㅋㅋ": "크크",
    "ㅠㅠ": "흑흑",
    "ㅜㅜ": "흑흑",
    "ㄱㄷ": "기다려",
    "ㅇㄱㄹㅇ": "이거레알",
    "ㄹㅇ": "리얼",
    "ㅈㄹ": "지랄",
    "ㅁㅊ": "미친",
}

_slang_pattern: re.Pattern | None = None


def _get_slang_pattern() -> re.Pattern:
    global _slang_pattern
    if _slang_pattern is None:
        escaped = [re.escape(k) for k in sorted(KOREAN_SLANG, key=len, reverse=True)]
        _slang_pattern = re.compile("|".join(escaped))
    return _slang_pattern


def _replace_slang(m: re.Match) -> str:
    return KOREAN_SLANG.get(m.group(0), m.group(0))


# ── English abbreviation / word pronunciation ────────────────────────────

# Common abbreviations and their Korean readings.
# Add new entries here as needed.
ENGLISH_READINGS: dict[str, str] = {
    # Tech
    "CPU": "씨피유",
    "GPU": "지피유",
    "API": "에이피아이",
    "SDK": "에스디케이",
    "TTS": "티티에스",
    "STT": "에스티티",
    "ASR": "에이에스알",
    "LLM": "엘엘엠",
    "AI": "에이아이",
    "UI": "유아이",
    "URL": "유알엘",
    "USB": "유에스비",
    "SSD": "에스에스디",
    "RAM": "램",
    "ROM": "롬",
    "FPS": "에프피에스",
    "HTTP": "에이치티티피",
    "HTTPS": "에이치티티피에스",
    "HTML": "에이치티엠엘",
    "CSS": "씨에스에스",
    "JSON": "제이슨",
    "XML": "엑스엠엘",
    "SSH": "에스에스에이치",
    "VPN": "브이피엔",
    "DNS": "디엔에스",
    "IP": "아이피",
    "TCP": "티씨피",
    "UDP": "유디피",
    "A2F": "에이투에프",
    "A2E": "에이투이",
    "CUDA": "쿠다",
    "VRAM": "브이램",
    "WDDM": "더블유디디엠",
    "TCC": "티씨씨",
    "WSS": "더블유에스에스",
    "WS": "더블유에스",
    "PC": "피씨",
    "OS": "오에스",
    "VTuber": "브이튜버",
    "RTX": "알티엑스",
    "NVIDIA": "엔비디아",
    "AMD": "에이엠디",
    "ARKit": "에이알킷",
    "TensorRT": "텐서알티",
    # Services
    "GitHub": "깃허브",
    "Discord": "디스코드",
    "Telegram": "텔레그램",
    "YouTube": "유튜브",
    "Twilio": "트윌리오",
    "ngrok": "엔그록",
    # Units
    "GB": "기가바이트",
    "MB": "메가바이트",
    "KB": "킬로바이트",
    "TB": "테라바이트",
    "GHz": "기가헤르츠",
    "MHz": "메가헤르츠",
    "kHz": "킬로헤르츠",
    "Hz": "헤르츠",
    "ms": "밀리초",
    "fps": "에프피에스",
    # Common words
    "OK": "오케이",
    "VPN": "브이피엔",
    "WiFi": "와이파이",
    "Wifi": "와이파이",
    "WIFI": "와이파이",
    "AI": "에이아이",
}

# Build regex: match longest first to avoid partial matches
_eng_pattern: Optional[re.Pattern] = None


def _get_eng_pattern() -> re.Pattern:
    global _eng_pattern
    if _eng_pattern is None:
        # Sort by length (longest first) to match "HTTPS" before "HTTP"
        sorted_keys = sorted(ENGLISH_READINGS.keys(), key=len, reverse=True)
        escaped = [re.escape(k) for k in sorted_keys]
        # Use lookaround instead of \b — works better with Korean characters
        _eng_pattern = re.compile(
            r"(?<![A-Za-z])(" + "|".join(escaped) + r")(?![A-Za-z])"
        )
    return _eng_pattern


def _replace_english(match: re.Match) -> str:
    word = match.group(0)
    # Try exact case match first, then upper, then title
    for variant in [word, word.upper(), word.title(), word.lower()]:
        if variant in ENGLISH_READINGS:
            return ENGLISH_READINGS[variant]
    return word


# ── Custom replacements (symbols, special patterns) ──────────────────────

# Simple string replacements applied in order.
# Add new entries here for quick fixes.
CUSTOM_REPLACEMENTS: list[tuple[str, str]] = [
    ("A2F", "에이투에프"),   # pre-English-pass: catch even when glued (TTSA2F)
    ("A2E", "에이투이"),
    ("%", "퍼센트"),
    ("×", "곱하기"),
    ("÷", "나누기"),
    ("+", "플러스"),
    ("=", "이퀄"),
    ("→", "화살표"),
    ("←", "왼쪽 화살표"),
    ("↔", "양쪽 화살표"),
]


# ── Main normalizer ──────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that TTS shouldn't read."""
    # Remove code blocks (``` ... ```)
    result = re.sub(r"```[^`]*```", "", text)
    # Remove inline code (`...`)
    result = re.sub(r"`([^`]*)`", r"\1", result)
    # Remove bold (**...**)
    result = re.sub(r"\*\*([^*]*)\*\*", r"\1", result)
    # Remove italic (*...*)
    result = re.sub(r"\*([^*]*)\*", r"\1", result)
    # Remove strikethrough (~~...~~)
    result = re.sub(r"~~([^~]*)~~", r"\1", result)
    # Remove headers (# ## ### etc.)
    result = re.sub(r"^#{1,6}\s+", "", result, flags=re.MULTILINE)
    # Remove link syntax [text](url) → text
    result = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", result)
    # Remove bare URLs (http://, https://, www.)
    result = re.sub(r"https?://\S+", "", result)
    result = re.sub(r"www\.\S+", "", result)
    # Remove bullet points (- or *)
    result = re.sub(r"^\s*[-*]\s+", "", result, flags=re.MULTILINE)
    # Remove remaining markdown artifacts
    result = result.replace("**", "").replace("__", "")
    # Remove leftover decorative symbols (standalone *, ~, -, —, ·, •, /, \)
    # Keep ~ between digits (handled later as range), remove others
    result = re.sub(r"(?<!\d)~(?!\d)", "", result)  # ~ not between digits
    result = result.replace("*", "").replace("—", " ").replace("·", " ").replace("•", " ")
    # Remove standalone dashes (not in words like "well-known")
    result = re.sub(r"(?<![A-Za-z가-힣])-(?![A-Za-z가-힣\d])", "", result)
    # Collapse multiple spaces
    result = re.sub(r" {2,}", " ", result)
    return result.strip()


def normalize_korean(text: str) -> str:
    """
    Normalize Korean text for TTS consumption.
    Pipeline:
    0. Strip markdown formatting
    1. English abbreviations/words → Korean reading (dictionary)
    2. Numbers (with decimals, percentages) → Korean reading
    3. Custom symbol replacements
    Unrecognized English words are left as-is for GPT-SoVITS auto language handling.
    """
    if not text or not text.strip():
        return text

    # Step 0: Remove markdown formatting
    result = _strip_markdown(text)

    # Step 0.3: Custom replacements (catches glued patterns like TTSA2F before eng pass)
    for old, new in CUSTOM_REPLACEMENTS:
        result = result.replace(old, new)

    # Step 0.5: Korean slang / initial-consonant abbreviations
    # Normalize Choseong Jamo (U+1100-U+1112) → Compatibility Jamo (U+3131-U+314E)
    # LLMs sometimes output Hangul Jamo block instead of Compatibility Jamo
    _CHOSEONG_TO_COMPAT = str.maketrans(
        "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ",
        "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ",
    )
    result = result.translate(_CHOSEONG_TO_COMPAT)
    # Collapse repeated single jamo (ㅋㅋㅋㅋ → ㅋㅋ, ㅎㅎㅎ → ㅎㅎ)
    result = re.sub(r"([\u3131-\u314E\u314F-\u3163])\1{2,}", r"\1\1", result)
    result = _get_slang_pattern().sub(_replace_slang, result)

    # Step 1: Replace known English abbreviations (before number processing)
    result = _get_eng_pattern().sub(_replace_english, result)

    # Step 2a: Version numbers (v1.2.3) — read digit by digit
    result = re.sub(r"v(\d[\d.]*\d)", _version_to_korean, result)

    # Step 2b: Range with tilde (2~3 → 이에서삼, 10~20 → 십에서이십)
    def _range_to_korean(m: re.Match) -> str:
        return _number_str_to_korean(m.group(1)) + "에서" + _number_str_to_korean(m.group(2))
    result = re.sub(r"(\d+(?:\.\d+)?)~(\d+(?:\.\d+)?)", _range_to_korean, result)

    # Step 2c: Numbers — decimal, percentage, integer
    # Match: optional minus, digits with optional decimal, optional %
    result = re.sub(r"-?\d+(?:\.\d+)?%?", _number_to_korean, result)

    # Step 3: Unknown English words are left as-is
    # GPT-SoVITS with text_lang='auto' will handle them natively

    return result


if __name__ == "__main__":
    tests = [
        "배율을 1.5에서 3.0으로 올려볼까?",
        "정확도가 99.9%야.",
        "TensorRT v10.13.0을 설치했어.",
        "CPU 사용률이 85%고 GPU는 12GB VRAM 중 8.5GB 사용 중.",
        "API 레이턴시가 350ms야.",
        "RTX 5090은 CUDA 코어가 많아.",
        "1234567원 보냈어.",
        "속도가 0.76초야.",
    ]
    for t in tests:
        print(f"IN:  {t}")
        print(f"OUT: {normalize_korean(t)}")
        print()
