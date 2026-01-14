# 🎭 NLP 기반 자동 감정 분석 시스템 설계

## 📋 개요

LLM의 응답 텍스트를 NLP로 분석하여 자동으로 감정을 감지하고 Live2D를 제어하는 시스템

---

## 🏗️ 시스템 아키텍처

### 1. **처리 흐름**

```
LLM 응답 → 감정 분석 → 감정 매핑 → Actions 생성 → Live2D 제어
```

### 2. **통합 지점**

#### 현재 코드 구조:
```
conversations/
├── single_conversation.py    # 개인 대화 처리
├── group_conversation.py     # 그룹 대화 처리
├── conversation_utils.py     # 공통 유틸리티
│   └── handle_sentence_output()  ← 여기에 감정 분석 추가
└── tts_manager.py            # TTS 생성 관리
```

#### 통합 위치:
- **`conversation_utils.py:handle_sentence_output()` (84-113줄)**
  - LLM의 `display_text`를 받아 처리하는 단계
  - TTS 생성 전에 감정 분석 수행
  - 분석된 감정을 `Actions` 객체에 추가

---

## 🧠 감정 분석 방법 (3가지 옵션)

### **옵션 1: 경량 Transformer 모델 (추천)**

#### 장점:
- 정확도가 높음
- 한국어 특화 모델 사용 가능
- 오프라인 동작 가능

#### 단점:
- 초기 모델 다운로드 필요
- GPU 권장 (CPU도 가능하지만 느림)

#### 구현:
```python
from transformers import pipeline

# 한국어 감정 분석 모델
emotion_classifier = pipeline(
    "text-classification",
    model="beomi/KcELECTRA-base-v2022-emotion",
    top_k=None
)

# 사용 예시
text = "오늘 정말 기분이 좋아요!"
emotions = emotion_classifier(text)
# [{'label': 'joy', 'score': 0.95}, {'label': 'neutral', 'score': 0.03}, ...]
```

**지원 감정**: joy, sadness, anger, fear, surprise, disgust, neutral

---

### **옵션 2: 규칙 기반 키워드 매칭 (가장 가볍고 빠름)**

#### 장점:
- 의존성 없음 (순수 Python)
- 매우 빠름 (1ms 미만)
- 쉬운 커스터마이징

#### 단점:
- 정확도가 낮을 수 있음
- 복잡한 문맥 이해 불가

#### 구현:
```python
EMOTION_KEYWORDS = {
    "joy": ["기쁘", "행복", "좋아", "웃음", "즐거", "신나", "만족", "😊", "😄"],
    "sadness": ["슬프", "우울", "눈물", "힘들", "괴로", "안타까", "😢", "😭"],
    "anger": ["화나", "짜증", "분노", "열받", "미워", "😡", "😠"],
    "fear": ["무서", "두려", "걱정", "불안", "공포", "😨", "😰"],
    "surprise": ["놀라", "깜짝", "어머", "헐", "대박", "😲", "😮"],
    "disgust": ["싫어", "역겹", "징그", "더러", "🤢"],
    "neutral": []  # 기본값
}

def analyze_emotion_simple(text: str) -> str:
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[emotion] += 1

    # 가장 높은 점수의 감정 반환
    max_emotion = max(scores, key=scores.get)
    return max_emotion if scores[max_emotion] > 0 else "neutral"
```

---

### **옵션 3: Ollama 로컬 LLM 활용**

#### 장점:
- 문맥 이해 능력이 뛰어남
- 이미 Ollama가 설치되어 있으면 추가 설치 불필요

#### 단점:
- 느림 (100-500ms)
- 리소스 사용량 많음

#### 구현:
```python
import requests

def analyze_emotion_ollama(text: str) -> str:
    prompt = f"""다음 텍스트의 감정을 분석해주세요.
감정은 다음 중 하나만 선택: joy, sadness, anger, fear, surprise, disgust, neutral

텍스트: "{text}"

감정(한 단어만):"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mili-8b",
            "prompt": prompt,
            "stream": False
        }
    )

    emotion = response.json()["response"].strip().lower()
    return emotion
```

---

## 📦 구현 계획

### Phase 1: 감정 분석 엔진 구조 설계

**새 파일 생성**: `src/open_llm_vtuber/emotion/emotion_analyzer.py`

```python
from abc import ABC, abstractmethod
from typing import Optional

class EmotionAnalyzerInterface(ABC):
    """감정 분석 인터페이스"""

    @abstractmethod
    def analyze(self, text: str) -> str:
        """텍스트를 분석하여 감정을 반환"""
        pass

class KeywordEmotionAnalyzer(EmotionAnalyzerInterface):
    """규칙 기반 키워드 매칭"""
    pass

class TransformerEmotionAnalyzer(EmotionAnalyzerInterface):
    """Transformer 모델 기반"""
    pass

class OllamaEmotionAnalyzer(EmotionAnalyzerInterface):
    """Ollama LLM 기반"""
    pass
```

---

### Phase 2: Factory 패턴 구현

**새 파일 생성**: `src/open_llm_vtuber/emotion/emotion_factory.py`

```python
class EmotionAnalyzerFactory:
    @staticmethod
    def get_analyzer(engine_type: str, **kwargs) -> EmotionAnalyzerInterface:
        if engine_type == "keyword":
            return KeywordEmotionAnalyzer(**kwargs)
        elif engine_type == "transformer":
            return TransformerEmotionAnalyzer(**kwargs)
        elif engine_type == "ollama":
            return OllamaEmotionAnalyzer(**kwargs)
        else:
            raise ValueError(f"Unknown emotion analyzer type: {engine_type}")
```

---

### Phase 3: 설정 시스템 통합

**`conf.yaml`에 추가**:

```yaml
# =================== Emotion Analysis (Live2D Auto Control) ===================
emotion_config:
  enabled: true  # 감정 분석 활성화 여부
  engine: 'keyword'  # 'keyword', 'transformer', 'ollama'

  # 규칙 기반 엔진 설정
  keyword_analyzer:
    custom_keywords: {}  # 사용자 정의 키워드 추가 가능

  # Transformer 모델 설정
  transformer_analyzer:
    model_name: 'beomi/KcELECTRA-base-v2022-emotion'
    device: 'auto'  # 'cpu', 'cuda', 'auto'
    cache_dir: 'models/emotion'

  # Ollama LLM 설정
  ollama_analyzer:
    base_url: 'http://localhost:11434'
    model: 'mili-8b'
    timeout: 1.0  # 1초 타임아웃

  # 감정 매핑 (감정 분석 결과 → Live2D emotionMap)
  emotion_mapping:
    joy: 'joy'
    happiness: 'joy'
    sadness: 'sadness'
    anger: 'anger'
    fear: 'fear'
    surprise: 'surprise'
    disgust: 'disgust'
    neutral: 'neutral'
```

---

### Phase 4: conversation_utils.py 수정

**`handle_sentence_output()` 함수 수정**:

```python
async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
    emotion_analyzer: Optional[EmotionAnalyzerInterface] = None,  # 추가
) -> str:
    """Handle sentence output type with optional translation and emotion analysis"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug(f"🏃 Processing output: '''{tts_text}'''...")

        # 감정 분석 (활성화된 경우)
        if emotion_analyzer:
            detected_emotion = emotion_analyzer.analyze(display_text.text)
            logger.info(f"🎭 Detected emotion: {detected_emotion}")

            # Actions 객체가 없으면 생성
            if not actions:
                from ..agent.output_types import Actions
                actions = Actions()

            # 감정을 Live2D 표정으로 설정
            if detected_emotion in live2d_model.emo_map:
                actions.expression = detected_emotion

        # 기존 코드 계속...
        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？\'\"』」）】\s]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info(f"🏃 Text after translation: '''{tts_text}'''...")

        full_response += display_text.text
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )
    return full_response
```

---

### Phase 5: service_context.py 수정

**EmotionAnalyzer를 ServiceContext에 추가**:

```python
class ServiceContext:
    def __init__(self, ...):
        # 기존 코드...

        # Emotion Analyzer 초기화
        emotion_config = character_config.emotion_config
        if emotion_config and emotion_config.enabled:
            from .emotion.emotion_factory import EmotionAnalyzerFactory
            self.emotion_analyzer = EmotionAnalyzerFactory.get_analyzer(
                emotion_config.engine,
                **emotion_config.get_engine_config()
            )
        else:
            self.emotion_analyzer = None
```

---

## 🎯 권장 구현 순서

1. **Phase 1**: 규칙 기반 키워드 분석부터 시작 (가장 간단)
2. **Phase 2**: 설정 시스템 통합
3. **Phase 3**: conversation_utils.py 수정
4. **Phase 4**: 테스트 및 디버깅
5. **Phase 5**: (선택) Transformer 모델 추가

---

## 🔍 테스트 시나리오

### 테스트 케이스:

```python
test_cases = [
    ("정말 기쁘네요! 오늘 좋은 일이 있었어요.", "joy"),
    ("너무 슬퍼요... 힘들어요.", "sadness"),
    ("화나네요! 짜증나!", "anger"),
    ("무서워요... 걱정돼요.", "fear"),
    ("헐! 깜짝이야!", "surprise"),
    ("역겹네요.", "disgust"),
    ("네, 알겠습니다.", "neutral"),
]

for text, expected in test_cases:
    result = emotion_analyzer.analyze(text)
    print(f"Text: {text}")
    print(f"Expected: {expected}, Got: {result}")
    print(f"Match: {'✅' if result == expected else '❌'}")
    print()
```

---

## ⚡ 성능 최적화

### 1. 캐싱
- 동일한 텍스트에 대해 재분석하지 않음
- LRU 캐시 사용 (`functools.lru_cache`)

### 2. 비동기 처리
- 감정 분석을 비동기로 실행하여 지연 최소화

### 3. 배치 처리
- 여러 문장을 한 번에 분석 (Transformer 모델)

---

## 📊 비교표

| 방법 | 정확도 | 속도 | 리소스 | 의존성 | 난이도 |
|------|--------|------|--------|--------|--------|
| **키워드** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | 없음 | ⭐ |
| **Transformer** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | transformers | ⭐⭐⭐ |
| **Ollama** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Ollama | ⭐⭐ |

---

## 🚀 시작하기

### 최소 구현 (규칙 기반):

1. `emotion/` 폴더 생성
2. `keyword_analyzer.py` 구현
3. `conf.yaml`에 설정 추가
4. `conversation_utils.py` 수정
5. 테스트!

**예상 작업 시간**: 2-3시간

---

**Last Updated: 2026-01-14**
