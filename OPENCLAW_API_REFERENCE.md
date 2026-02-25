# OpenClaw API Reference

> Base URL: `http://127.0.0.1:18789`
> 모든 엔드포인트는 Gateway와 동일한 포트에서 HTTP로 동작한다.

---

## 인증

모든 요청에 Bearer 토큰을 포함한다.

```
Authorization: Bearer <GATEWAY_TOKEN>
```

| auth mode | 값 출처 |
|-----------|---------|
| `token` | `gateway.auth.token` 또는 `OPENCLAW_GATEWAY_TOKEN` 환경변수 |
| `password` | `gateway.auth.password` 또는 `OPENCLAW_GATEWAY_PASSWORD` 환경변수 |

---

## 1. POST /v1/chat/completions

OpenAI Chat Completions 호환 엔드포인트. **기본 비활성 — 설정 필요.**

### 활성화

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

### 에이전트 지정

| 방법 | 예시 |
|------|------|
| `model` 필드 | `"openclaw:main"`, `"agent:main"` |
| 헤더 | `x-openclaw-agent-id: main` |

### 세션

- 기본: 요청마다 새 세션 (stateless)
- `user` 필드를 보내면 동일 세션 키로 대화 유지
- `x-openclaw-session-key` 헤더로 직접 세션 키 지정 가능

### Request

```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "model": "openclaw:main",
  "messages": [
    { "role": "user", "content": "안녕하세요" }
  ],
  "stream": false
}
```

### Response (non-streaming)

표준 OpenAI Chat Completion 응답 형식과 동일.

### Response (streaming)

`"stream": true` 설정 시 SSE(Server-Sent Events) 반환.

```
Content-Type: text/event-stream

data: {"id":"...","choices":[{"delta":{"content":"안녕"},...}]}
data: {"id":"...","choices":[{"delta":{"content":"하세요"},...}]}
data: [DONE]
```

### 프론트엔드 예시 (fetch)

```js
const res = await fetch('http://127.0.0.1:18789/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${GATEWAY_TOKEN}`,
  },
  body: JSON.stringify({
    model: 'openclaw:main',
    messages: [{ role: 'user', content: prompt }],
  }),
});
const data = await res.json();
const reply = data.choices[0].message.content;
```

### 프론트엔드 예시 (streaming)

```js
const res = await fetch('http://127.0.0.1:18789/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${GATEWAY_TOKEN}`,
  },
  body: JSON.stringify({
    model: 'openclaw:main',
    messages: [{ role: 'user', content: prompt }],
    stream: true,
  }),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  for (const line of chunk.split('\n')) {
    if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
    const json = JSON.parse(line.slice(6));
    const delta = json.choices?.[0]?.delta?.content;
    if (delta) process.stdout.write(delta); // 또는 UI 업데이트
  }
}
```

---

## 2. POST /v1/responses

OpenResponses 호환 엔드포인트. **기본 비활성 — 설정 필요.**

### 활성화

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "responses": { "enabled": true }
      }
    }
  }
}
```

### Request

```http
POST /v1/responses
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "model": "openclaw:main",
  "input": "안녕하세요",
  "stream": false
}
```

### 지원 필드

| 필드 | 설명 |
|------|------|
| `input` | `string` 또는 item 객체 배열 |
| `instructions` | 시스템 프롬프트에 병합 |
| `tools` | 클라이언트 function tool 정의 |
| `tool_choice` | 도구 필터/강제 |
| `stream` | SSE 스트리밍 활성화 |
| `max_output_tokens` | 출력 토큰 제한 (best-effort) |
| `user` | 세션 라우팅 키 |

### Input Item 타입

**message** — `role`: `system` | `developer` | `user` | `assistant`

```json
{ "type": "message", "role": "user", "content": "질문 내용" }
```

**function_call_output** — 도구 결과 반환

```json
{ "type": "function_call_output", "call_id": "call_123", "output": "{\"result\":\"ok\"}" }
```

**input_image** — 이미지 첨부 (max 10MB, jpeg/png/gif/webp)

```json
{
  "type": "input_image",
  "source": { "type": "url", "url": "https://example.com/img.png" }
}
```

**input_file** — 파일 첨부 (max 5MB, text/markdown/html/csv/json/pdf)

```json
{
  "type": "input_file",
  "source": { "type": "base64", "media_type": "text/plain", "data": "SGVsbG8=", "filename": "hello.txt" }
}
```

### Streaming SSE 이벤트 타입

```
response.created
response.in_progress
response.output_item.added
response.content_part.added
response.output_text.delta
response.output_text.done
response.content_part.done
response.output_item.done
response.completed
response.failed
```

---

## 3. POST /tools/invoke

단일 도구 직접 호출. **항상 활성** (인증 + 도구 정책으로 제어).

### Request

```http
POST /tools/invoke
Content-Type: application/json
Authorization: Bearer <token>
```

```json
{
  "tool": "sessions_list",
  "action": "json",
  "args": {},
  "sessionKey": "main"
}
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `tool` | O | 호출할 도구 이름 |
| `action` | X | 도구 액션 |
| `args` | X | 도구별 인수 객체 |
| `sessionKey` | X | 대상 세션 키 (기본: `main`) |

### Response

```json
{ "ok": true, "result": { ... } }
```

---

## 에러 응답

모든 엔드포인트 공통.

| HTTP | 의미 |
|------|------|
| `400` | 잘못된 요청 |
| `401` | 인증 실패 |
| `404` | 도구 없음 / 정책에 의해 차단 |
| `405` | 잘못된 HTTP 메서드 |

```json
{ "error": { "message": "...", "type": "invalid_request_error" } }
```

---

## 헤더 요약

| 헤더 | 용도 |
|------|------|
| `Authorization` | `Bearer <token>` — 필수 |
| `Content-Type` | `application/json` — 필수 |
| `x-openclaw-agent-id` | 에이전트 지정 (기본: `main`) |
| `x-openclaw-session-key` | 세션 키 직접 지정 |
| `x-openclaw-message-channel` | 채널 컨텍스트 (`telegram`, `slack` 등) |
| `x-openclaw-account-id` | 멀티 계정 시 계정 ID |

---

## OpenAI SDK 호환 사용

OpenAI 공식 SDK에서 `baseURL`만 바꿔 사용 가능.

```js
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://127.0.0.1:18789/v1',
  apiKey: GATEWAY_TOKEN,
});

const completion = await client.chat.completions.create({
  model: 'openclaw:main',
  messages: [{ role: 'user', content: '오늘 날씨 어때?' }],
});

console.log(completion.choices[0].message.content);
```
