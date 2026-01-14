# 백엔드가 제어하는 Live2D 파라미터

## 1. WebSocket 페이로드 구조

백엔드는 다음 구조의 JSON을 프론트엔드로 전송합니다:

```json
{
  "type": "audio",
  "audio": "base64_encoded_wav_data",
  "volumes": [0.1, 0.5, 0.8, ...],
  "slice_length": 20,
  "display_text": {
    "text": "안녕하세요",
    "name": "Mao",
    "avatar": "mao.png"
  },
  "actions": {
    "expressions": [3],
    "pictures": ["image1.png"],
    "sounds": ["sound1.wav"]
  },
  "forwarded": false
}
```

## 2. 백엔드 제어 요소

### 2.1 표정 (Expressions)
- **필드**: `actions.expressions`
- **타입**: `List[int]` (표정 인덱스)
- **제어 방식**: **간접 제어** (인덱스만 전달)
- **값**: 0-7 범위의 정수 (model3.json의 Expressions 배열 인덱스)
- **프론트엔드 매핑**: 
  - 인덱스 → `model3.json`의 `FileReferences.Expressions[index]`
  - 예: `expressions: [3]` → `expressions/exp_04.exp3.json` 로드

**감정 → 인덱스 매핑 (model_dict.json):**
```python
"emotionMap": {
    "neutral": 0,
    "anger": 2,
    "disgust": 2,
    "fear": 1,
    "joy": 3,
    "smirk": 3,
    "sadness": 1,
    "surprise": 3
}
```

**생성 위치:**
- `src/open_llm_vtuber/conversations/conversation_utils.py:114-120`
```python
if detected_emotion in live2d_model.emo_map:
    emotion_index = live2d_model.emo_map[detected_emotion]
    actions.expressions = [emotion_index]  # 숫자 인덱스 전달
```

---

### 2.2 립싱크 (Lip Sync)
- **필드**: `volumes`
- **타입**: `List[float]` (정규화된 볼륨 값, 0.0~1.0)
- **제어 방식**: **간접 제어** (볼륨 데이터만 전달)
- **값**: 오디오 청크별 RMS 볼륨 (20ms 단위)
- **프론트엔드 매핑**:
  - `model3.json`의 `Groups` → `"Name": "LipSync"` → `Ids` 배열의 파라미터 적용
  - 예시:
    ```json
    {
      "Target": "Parameter",
      "Name": "LipSync",
      "Ids": ["PARAM_MOUTH_OPEN_Y"]
    }
    ```
  - 프론트엔드가 `volumes[i]` 값을 `PARAM_MOUTH_OPEN_Y` 파라미터에 적용

**생성 위치:**
- `src/open_llm_vtuber/utils/stream_audio.py:8-30`
```python
def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    return [volume / max_volume for volume in volumes]  # 0.0~1.0 정규화
```

---

### 2.3 이미지 (Pictures)
- **필드**: `actions.pictures`
- **타입**: `List[str]` (이미지 경로)
- **제어 방식**: **직접 제어 안 함** (현재 미사용)
- **용도**: 향후 확장용 (예: 배경 이미지 변경)

---

### 2.4 사운드 (Sounds)
- **필드**: `actions.sounds`
- **타입**: `List[str]` (사운드 파일 경로)
- **제어 방식**: **직접 제어 안 함** (현재 미사용)
- **용도**: 향후 확장용 (예: 효과음 재생)

---

## 3. Live2D 파라미터 매핑 (프론트엔드)

백엔드는 **파라미터 이름을 직접 지정하지 않습니다**. 프론트엔드가 `model3.json`을 읽어서 매핑합니다.

### 3.1 표정 파라미터 (Expressions)
프론트엔드는 `expressions/exp_XX.exp3.json` 파일을 로드하여 적용합니다.

**exp3.json 구조 예시:**
```json
{
  "Type": "Live2D Expression",
  "FadeInTime": 0.5,
  "FadeOutTime": 0.5,
  "Parameters": [
    {
      "Id": "ParamEyeLSmile",
      "Value": 1.0,
      "Blend": "Add"
    },
    {
      "Id": "ParamEyeRSmile", 
      "Value": 1.0,
      "Blend": "Add"
    }
  ]
}
```

### 3.2 립싱크 파라미터 (LipSync)
프론트엔드는 `model3.json`의 `LipSync` 그룹을 읽어서 적용합니다.

**모델별 립싱크 파라미터:**

| 모델 | LipSync 파라미터 | 표준 여부 |
|------|------------------|-----------|
| **shizuku** | `PARAM_MOUTH_OPEN_Y` | ✅ 표준 |
| **mao_pro** | `ParamA` | ❌ 비표준 |

**표준 파라미터:**
- `ParamMouthOpenY` (Live2D Cubism SDK 기본)
- `PARAM_MOUTH_OPEN_Y` (대문자 버전)

---

## 4. 현재 시스템의 제약

### 4.1 립싱크 호환성 문제
- ❌ **문제**: 프론트엔드가 표준 파라미터만 지원
- ❌ **영향**: `ParamA` 같은 비표준 파라미터를 사용하는 모델은 립싱크 불가
- ✅ **해결책**: 
  1. 표준 파라미터를 사용하는 모델 사용 (shizuku 등)
  2. 프론트엔드 수정: `model3.json`의 `LipSync` 그룹을 동적으로 읽도록 개선

### 4.2 표정 호환성
- ✅ **정상 동작**: 모든 모델에서 작동 (인덱스 기반)
- ⚠️ **주의**: `model_dict.json`의 `emotionMap`과 실제 `expressions/` 파일 개수가 일치해야 함

---

## 5. 코드 경로

### 백엔드
- **표정 생성**: `src/open_llm_vtuber/conversations/conversation_utils.py:100-127`
- **립싱크 생성**: `src/open_llm_vtuber/utils/stream_audio.py:8-30`
- **페이로드 구성**: `src/open_llm_vtuber/utils/stream_audio.py:33-100`
- **WebSocket 전송**: `src/open_llm_vtuber/conversations/tts_manager.py:92-109`

### 모델 설정
- **표정 매핑**: `live2d-models/model_dict.json`
- **모델 구조**: `live2d-models/{model_name}/runtime/{model_name}.model3.json`
- **표정 파일**: `live2d-models/{model_name}/runtime/expressions/exp_*.exp3.json`

---

## 6. 디버깅 로그

대화 중 다음 로그로 파라미터 생성 확인:

```
🎭 Detected emotion: joy
🎭 Setting Live2D expression: joy (index: 3)
👄 [TTS] Generated audio file: cache/20260114_055109_a9c92db9.wav
👄 [Lip Sync] Processing audio file: cache/20260114_055109_a9c92db9.wav
👄 [Lip Sync] Audio loaded: duration=4980ms, size=637484 bytes
👄 [Lip Sync] Calculated 249 volume chunks, max_volume: 682305907.00
👄 [Lip Sync] Payload ready: audio_size=849980 chars, volumes=249 chunks
🎭 [WebSocket Payload] actions: {'expressions': [3]}
```

---

## 7. 요약

| 항목 | 백엔드 역할 | 프론트엔드 역할 |
|------|------------|----------------|
| **표정** | 인덱스 전달 (0-7) | exp3.json 로드 및 파라미터 적용 |
| **립싱크** | 볼륨 배열 전달 | LipSync 그룹의 파라미터에 적용 |
| **파라미터 이름** | 직접 지정 안 함 | model3.json 기반 동적 매핑 |

**결론**: 백엔드는 **간접적으로** Live2D를 제어합니다. 실제 파라미터 이름과 매핑은 프론트엔드와 model3.json이 담당합니다.
