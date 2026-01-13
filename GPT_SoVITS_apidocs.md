네, 모든 내용을 하나의 마크다운(`.md`) 파일로 통합하여 정리해 드립니다. 이 내용을 그대로 복사해서 `API_GUIDE.md`라는 이름으로 저장하시면 됩니다.

---

```markdown
# 🎤 GPT-SoVITS API 서버 가이드 (V2 통합본)

이 문서는 **GPT-SoVITS v2** 모델을 이용한 음성 합성 API 서버의 구축, 최적화 및 호출 방법을 다루는 통합 가이드입니다.

---

## 1. 서버 실행 (Deployment)

서버 실행 시 모델의 경로와 기본 참조 음성(비비안)을 설정합니다. 성능 최적화(FP16, 스트리밍) 옵션을 포함한 권장 명령어입니다.

### 🚀 서버 실행 명령어
```bash
python api.py \
  -s "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth" \
  -g "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" \
  -dr "vivian.wav" \
  -dt "뭐 새로운 괴담 없나요? 어디 갈까요~ 오늘은 학교를 쉬는 날이에요!" \
  -dl "ko" \
  -hp \
  -sm normal \
  -cp ",.!?，。！？"

```

### ⚙️ 실행 파라미터 상세 설명

| 파라미터 | 기능 | 설명 |
| --- | --- | --- |
| `-s` | SoVITS 모델 | `.pth` 파일 경로 (음색/감정 담당) |
| `-g` | GPT 모델 | `.ckpt` 파일 경로 (텍스트 분석 담당) |
| `-dr` | 참조 오디오 | 기본 목소리가 될 `.wav` 파일 경로 |
| `-dt` | 참조 대사 | 참조 오디오의 실제 대사 내용 |
| `-dl` | 참조 언어 | 참조 대사의 언어 (`ko`, `zh`, `en`, `ja`, `yue`) |
| **`-hp`** | **속도 최적화** | **반정밀도(FP16)** 사용으로 생성 속도 대폭 향상 |
| **`-sm`** | **스트리밍** | 음성 생성이 완료되기 전부터 데이터 전송 (`normal` 권장) |
| **`-cp`** | **문장 절단** | 지정된 기호에서 문장을 끊어 처리하여 병목 현상 방지 |

---

## 2. API 호출 방법 (Endpoints)

### 🔊 음성 합성 추론 (`/`)

설정한 텍스트를 음성으로 변환하여 반환합니다.

* **Endpoint:** `http://127.0.0.1:9880/`
* **Method:** `GET` 또는 `POST`

#### [Python 테스트 코드 예시]

```python
import requests

def generate_voice(text, output_path="output.wav"):
    url = "[http://127.0.0.1:9880](http://127.0.0.1:9880)"
    payload = {
        "text": text,
        "text_language": "ko",
        "speed": 1.0,
        "top_k": 10
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 생성 성공: {output_path}")
    else:
        print(f"❌ 생성 실패: {response.json()}")

# 실행
generate_voice("안녕하세요, 비비안입니다. 오늘 학교 쉬는 날이라 너무 좋네요!")

```

#### [Curl 테스트 (터미널)]

```bash
# 주의: URL에 공백이 있을 경우 %20으로 대체하거나 공백 없이 입력
curl "[http://127.0.0.1:9880/?text=반가워요!&text_language=ko](http://127.0.0.1:9880/?text=반가워요!&text_language=ko)" --output test.wav

```

---

## 3. 부가 기능 엔드포인트

### 🔄 기본 목소리 변경 (`/change_refer`)

서버를 끄지 않고 기본 참조 오디오를 교체합니다.

* **URL:** `http://127.0.0.1:9880/change_refer`
* **필수 값:** `refer_wav_path`, `prompt_text`, `prompt_language`

### 🛠️ 서버 제어 (`/control`)

* **재시작:** `GET /control?command=restart`
* **종료:** `GET /control?command=exit`

---

## 4. 에러 및 트러블슈팅

### 🚨 `KeyError: 'default'`

* **원인:** 서버 실행 시 `-s`와 `-g` 경로를 지정하지 않았거나 파일 경로가 틀린 경우.
* **해결:** 위에 제공된 **1. 서버 실행 명령어**를 사용하여 경로를 다시 지정하세요.

### 🐢 생성 속도가 너무 느림

* `-hp` 옵션을 사용 중인지 확인하세요.
* 긴 문장을 한 번에 넣지 말고 문장 기호(`, . ! ?`)를 적절히 섞어주세요.
* GPU를 사용 중인지 확인하세요 (터미널 로그에 `cuda` 확인).

---

**Last Updated: 2026-01-14**

```

---

이 파일을 프로젝트 폴더에 저장해두시면 언제든 참고하여 서버를 실행하고 테스트하실 수 있습니다. 추가로 필요한 기능이 있으시면 말씀해주세요!

```