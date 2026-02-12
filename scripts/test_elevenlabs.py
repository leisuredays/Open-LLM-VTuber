"""
ElevenLabs Flash v2.5 테스트 스크립트

모델별 비교, 감정 표현 테스트, 지연시간 측정을 수행합니다.
사용법:
    uv run python scripts/test_elevenlabs.py --api-key YOUR_API_KEY
    uv run python scripts/test_elevenlabs.py  # conf.yaml에서 api_key 읽기
"""

import argparse
import os
import time
from pathlib import Path

from elevenlabs import ElevenLabs


# ── 설정 ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("cache/elevenlabs_test")

MODELS = {
    "flash_v2.5": "eleven_flash_v2_5",
    "multilingual_v2": "eleven_multilingual_v2",
    "turbo_v2.5": "eleven_turbo_v2_5",
}

# 감정별 테스트 문장 (한국어)
TEST_SENTENCES = {
    "neutral": "안녕하세요, 오늘 날씨가 좋네요. 산책하러 갈까요?",
    "excited": "와! 정말요?! 이거 진짜 대박이다! 너무 좋아!",
    "sad": "그 소식을 듣고 정말 마음이 아팠어요... 많이 힘들었겠다.",
    "angry": "도대체 왜 이런 일이 생긴 거야! 이건 정말 용납할 수 없어!",
    "whisper": "쉿, 조용히 해. 아무도 모르게 살짝 가자.",
    "laugh": "하하하! 너 진짜 웃기다! 아 배 아파!",
}

# style 파라미터 변화에 따른 비교 (0.0 = 안정적, 1.0 = 과장)
STYLE_VALUES = [0.0, 0.3, 0.6, 1.0]


def load_api_key_from_conf() -> str | None:
    """conf.yaml에서 ElevenLabs API 키를 읽어옵니다."""
    conf_path = Path("conf.yaml")
    if not conf_path.exists():
        return None
    try:
        import yaml

        with open(conf_path) as f:
            conf = yaml.safe_load(f)
        return (
            conf.get("character_config", {})
            .get("tts_config", {})
            .get("elevenlabs_tts", {})
            .get("api_key")
        )
    except Exception:
        return None


def list_voices(client: ElevenLabs):
    """사용 가능한 음성 목록을 출력합니다."""
    print("\n" + "=" * 60)
    print("📋 사용 가능한 음성 목록")
    print("=" * 60)

    response = client.voices.get_all()
    for voice in response.voices:
        labels = ", ".join(f"{k}={v}" for k, v in (voice.labels or {}).items())
        print(f"  {voice.name:20s} | ID: {voice.voice_id} | {labels}")

    return response.voices


def generate_and_measure(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    model_id: str,
    output_path: Path,
    stability: float = 0.5,
    similarity_boost: float = 0.5,
    style: float = 0.0,
) -> float:
    """TTS를 생성하고 소요 시간(초)을 반환합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
        voice_settings={
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True,
        },
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    elapsed = time.time() - start

    size_kb = output_path.stat().st_size / 1024
    print(f"  ✅ {output_path.name:40s} | {elapsed:.2f}s | {size_kb:.1f}KB")
    return elapsed


def test_model_comparison(client: ElevenLabs, voice_id: str):
    """모델별 지연시간 비교 테스트"""
    print("\n" + "=" * 60)
    print("🔬 테스트 1: 모델별 지연시간 비교")
    print("=" * 60)

    text = TEST_SENTENCES["neutral"]
    print(f'  텍스트: "{text}"')
    print(f"  글자 수: {len(text)}")
    print()

    results = {}
    for name, model_id in MODELS.items():
        out = OUTPUT_DIR / f"model_{name}.mp3"
        try:
            elapsed = generate_and_measure(
                client, text, voice_id, model_id, out
            )
            results[name] = elapsed
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    if results:
        print("\n  ── 결과 요약 ──")
        fastest = min(results, key=results.get)
        for name, elapsed in sorted(results.items(), key=lambda x: x[1]):
            marker = " ⚡ 최단" if name == fastest else ""
            print(f"  {name:20s}: {elapsed:.2f}초{marker}")


def test_emotions(client: ElevenLabs, voice_id: str, model_id: str):
    """감정별 문장 생성 테스트"""
    print("\n" + "=" * 60)
    print(f"🎭 테스트 2: 감정별 문장 테스트 (모델: {model_id})")
    print("=" * 60)

    for emotion, text in TEST_SENTENCES.items():
        print(f'\n  [{emotion}] "{text}"')
        out = OUTPUT_DIR / f"emotion_{emotion}.mp3"
        try:
            generate_and_measure(client, text, voice_id, model_id, out)
        except Exception as e:
            print(f"  ❌ {e}")


def test_style_variation(client: ElevenLabs, voice_id: str, model_id: str):
    """style 파라미터 변화에 따른 비교"""
    print("\n" + "=" * 60)
    print(f"🎚️ 테스트 3: style 파라미터 변화 비교 (모델: {model_id})")
    print("  style=0.0(안정적) → style=1.0(과장/감정적)")
    print("=" * 60)

    text = TEST_SENTENCES["excited"]
    print(f'\n  텍스트: "{text}"')

    for style_val in STYLE_VALUES:
        out = OUTPUT_DIR / f"style_{style_val:.1f}.mp3"
        try:
            generate_and_measure(
                client, text, voice_id, model_id, out, style=style_val
            )
        except Exception as e:
            print(f"  ❌ style={style_val}: {e}")


def test_stability_variation(client: ElevenLabs, voice_id: str, model_id: str):
    """stability 파라미터 변화에 따른 비교"""
    print("\n" + "=" * 60)
    print(f"🎛️ 테스트 4: stability 파라미터 변화 비교 (모델: {model_id})")
    print("  stability=0.0(변화많음/감정적) → stability=1.0(안정적/일관됨)")
    print("=" * 60)

    text = TEST_SENTENCES["sad"]
    print(f'\n  텍스트: "{text}"')

    for stab in [0.0, 0.3, 0.5, 0.7, 1.0]:
        out = OUTPUT_DIR / f"stability_{stab:.1f}.mp3"
        try:
            generate_and_measure(
                client, text, voice_id, model_id, out, stability=stab
            )
        except Exception as e:
            print(f"  ❌ stability={stab}: {e}")


def main():
    parser = argparse.ArgumentParser(description="ElevenLabs TTS 테스트")
    parser.add_argument("--api-key", type=str, help="ElevenLabs API 키")
    parser.add_argument("--voice-id", type=str, help="사용할 음성 ID")
    parser.add_argument(
        "--model",
        type=str,
        default="eleven_flash_v2_5",
        help="테스트할 모델 ID (기본: eleven_flash_v2_5)",
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="음성 목록만 출력"
    )
    parser.add_argument(
        "--test",
        nargs="*",
        choices=["models", "emotions", "style", "stability", "all"],
        default=["all"],
        help="실행할 테스트 (기본: all)",
    )
    args = parser.parse_args()

    # API 키 결정
    api_key = args.api_key or os.environ.get("ELEVENLABS_API_KEY") or load_api_key_from_conf()
    if not api_key:
        print("❌ API 키를 지정해주세요:")
        print("   --api-key YOUR_KEY")
        print("   또는 환경변수 ELEVENLABS_API_KEY 설정")
        print("   또는 conf.yaml의 elevenlabs_tts.api_key에 설정")
        return

    client = ElevenLabs(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 음성 목록
    if args.list_voices:
        list_voices(client)
        return

    # voice_id 결정
    voice_id = args.voice_id
    if not voice_id:
        print("음성 ID가 지정되지 않았습니다. 사용 가능한 음성 목록:")
        voices = list_voices(client)
        if voices:
            voice_id = voices[0].voice_id
            print(f"\n  → 첫 번째 음성 사용: {voices[0].name} ({voice_id})")
        else:
            print("❌ 사용 가능한 음성이 없습니다.")
            return

    print(f"\n🔊 음성 ID: {voice_id}")
    print(f"📁 출력 폴더: {OUTPUT_DIR}")

    tests = set(args.test)
    run_all = "all" in tests

    if run_all or "models" in tests:
        test_model_comparison(client, voice_id)

    if run_all or "emotions" in tests:
        test_emotions(client, voice_id, args.model)

    if run_all or "style" in tests:
        test_style_variation(client, voice_id, args.model)

    if run_all or "stability" in tests:
        test_stability_variation(client, voice_id, args.model)

    print("\n" + "=" * 60)
    print(f"✅ 테스트 완료! 생성된 파일: {OUTPUT_DIR}/")
    print("  파일을 재생하여 음질과 감정 표현을 비교해보세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
