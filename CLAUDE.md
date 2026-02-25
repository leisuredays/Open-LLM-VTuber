# Open-LLM-VTuber 프로젝트

## 개요
Live2D VTuber + LLM + ASR + TTS 파이프라인. 음성 대화형 AI VTuber 시스템.

## 구조
- `conf.yaml` — 메인 설정 파일
- `model_dict.json` — Live2D 모델 정의 + emotionMap
- `src/open_llm_vtuber/` — 핵심 소스코드
  - `agent/` — LLM 에이전트 (basic_memory_agent, agent_factory, transformers)
  - `utils/sentence_divider.py` — 문장 분할 + 태그 처리
  - `live2d_model.py` — Live2D 모델 + 감정태그 파싱
  - `tts/` — TTS 엔진들
  - `asr/` — ASR 엔진들
- `live2d-models/` — Live2D 모델 파일들
- `characters/` — 캐릭터 프리셋

## 현재 설정
- LLM: openclaw_llm (OpenClaw 게이트웨이 → Anthropic Claude)
- TTS: elevenlabs_tts
- ASR: sherpa_onnx_asr
- Live2D: mashiro 모델
- 감정태그: [neutral], [angry], [annoyed], [cry], [love], [excited], [shocked], [sad], [sleepy]

## 언어
- 코드 분석/수정 시 한국어로 답변
