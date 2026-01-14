# Frontend Implementation Guide: Real-Time Audio Streaming

## 📋 Overview

**Backend Commit**: `cd8a790` - feat: Implement true real-time audio streaming for TTS

The backend now supports **real-time audio streaming** for TTS engines (currently GPT-SoVITS). Instead of sending a complete audio file after full generation, the backend now streams audio chunks as they're generated, reducing user-perceived latency from ~2 seconds to ~0.6 seconds.

**Current Status**: ✅ Backend implementation complete | ❌ Frontend implementation required

---

## 🎯 Goal

Implement MediaSource API-based audio streaming in the frontend to:
1. Receive and play audio chunks in real-time as they arrive from the backend
2. Reduce time-to-first-sound from ~2s to ~0.6s
3. Maintain backward compatibility with traditional single-payload audio

---

## 🔄 What Changed in Backend

### Previous Flow (Traditional)
```
Backend:
1. Generate complete audio file (2 seconds)
2. Load entire file into memory
3. Send single WebSocket message with complete audio

Frontend:
1. Receive complete audio Base64
2. Create data URL
3. Play audio with HTML5 Audio element
```

### New Flow (Streaming)
```
Backend:
1. Start TTS streaming request
2. Receive first chunk (WAV header, ~0.6s) → send immediately
3. Receive subsequent chunks (PCM data) → send immediately
4. Send completion signal when done

Frontend (TO BE IMPLEMENTED):
1. Receive "audio-start" → prepare MediaSource
2. Receive "audio-chunk" → append to SourceBuffer
3. Receive "audio-complete" → finalize MediaSource
```

---

## 📡 New WebSocket Message Types

### 1. `audio-start` (Streaming Session Start)

Sent once at the beginning of each audio stream.

```typescript
interface AudioStartMessage {
  type: "audio-start";
  sequence: number;           // Sequence number for ordering
  display_text: {             // Text to display in chat
    text: string;
    name: string;
    avatar: string;
  };
  actions: {                  // Live2D expressions/motions
    expressions?: number[];
    motions?: number[];
  } | null;
  slice_length: number;       // Reserved for future use
}
```

**Example**:
```json
{
  "type": "audio-start",
  "sequence": 0,
  "display_text": {
    "text": "안녕하세요, 유즈하입니다!",
    "name": "유즈하",
    "avatar": "/avatars/uzuha.png"
  },
  "actions": {
    "expressions": [2]
  },
  "slice_length": 20
}
```

### 2. `audio-chunk` (Audio Data Chunk)

Sent multiple times for each audio chunk. First chunk contains WAV header, subsequent chunks contain PCM audio data.

```typescript
interface AudioChunkMessage {
  type: "audio-chunk";
  sequence: number;           // Same sequence as audio-start
  chunk_index: number;        // 0-based index
  audio: string;              // Base64-encoded audio data
  is_header: boolean;         // true for first chunk (WAV header)
}
```

**Example**:
```json
// First chunk (WAV header)
{
  "type": "audio-chunk",
  "sequence": 0,
  "chunk_index": 0,
  "audio": "UklGRiQAAABXQVZFZm10IBAAAA...",
  "is_header": true
}

// Subsequent chunks (PCM data)
{
  "type": "audio-chunk",
  "sequence": 0,
  "chunk_index": 1,
  "audio": "AAECAw...",
  "is_header": false
}
```

### 3. `audio-complete` (Streaming Session End)

Sent once when all audio chunks have been transmitted.

```typescript
interface AudioCompleteMessage {
  type: "audio-complete";
  sequence: number;           // Same sequence as audio-start
  total_chunks: number;       // Total number of chunks sent
}
```

**Example**:
```json
{
  "type": "audio-complete",
  "sequence": 0,
  "total_chunks": 15
}
```

### 4. `audio` (Traditional - Backward Compatibility)

The existing message type for non-streaming TTS engines. **Must still be supported!**

```typescript
interface AudioMessage {
  type: "audio";
  audio: string | null;       // Complete audio Base64 or null
  volumes: number[];          // Volume data for traditional playback
  slice_length: number;       // Chunk duration in ms
  display_text: {...};
  actions: {...} | null;
  forwarded?: boolean;
}
```

---

## 🛠️ Implementation Requirements

### MediaSource API for Audio Streaming

The HTML5 `<audio>` element with data URLs cannot handle streaming. You need to use the **MediaSource API**.

#### Step 1: Detect Streaming vs Traditional Messages

```typescript
// In WebSocket message handler
function handleWebSocketMessage(event: MessageEvent) {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'audio-start':
      handleAudioStart(message);
      break;
    case 'audio-chunk':
      handleAudioChunk(message);
      break;
    case 'audio-complete':
      handleAudioComplete(message);
      break;
    case 'audio':
      // Existing traditional handler
      handleTraditionalAudio(message);
      break;
  }
}
```

#### Step 2: Create MediaSource Manager

```typescript
class AudioStreamManager {
  private mediaSource: MediaSource | null = null;
  private sourceBuffer: SourceBuffer | null = null;
  private audioElement: HTMLAudioElement;
  private pendingChunks: Uint8Array[] = [];
  private isAppending: boolean = false;

  constructor(audioElement: HTMLAudioElement) {
    this.audioElement = audioElement;
  }

  async start(sequence: number, displayText: any, actions: any) {
    console.log(`[AudioStream] Starting sequence ${sequence}`);

    // Create MediaSource
    this.mediaSource = new MediaSource();
    const objectUrl = URL.createObjectURL(this.mediaSource);
    this.audioElement.src = objectUrl;

    // Wait for MediaSource to open
    await new Promise<void>((resolve, reject) => {
      this.mediaSource!.addEventListener('sourceopen', () => {
        try {
          // Create SourceBuffer for WAV audio
          this.sourceBuffer = this.mediaSource!.addSourceBuffer('audio/wav');

          // Set up updateend listener for sequential appending
          this.sourceBuffer.addEventListener('updateend', () => {
            this.isAppending = false;
            this.processNextChunk();
          });

          resolve();
        } catch (error) {
          reject(error);
        }
      });

      this.mediaSource!.addEventListener('error', reject);
    });

    // Display text in chat UI
    this.displayChatMessage(displayText);

    // Apply Live2D expressions/motions
    if (actions) {
      this.applyLive2DActions(actions);
    }
  }

  addChunk(chunkBase64: string, isHeader: boolean) {
    // Decode Base64 to Uint8Array
    const binaryString = atob(chunkBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    if (isHeader) {
      console.log(`[AudioStream] Received WAV header (${bytes.length} bytes)`);
    }

    // Add to pending queue
    this.pendingChunks.push(bytes);

    // Try to process immediately
    this.processNextChunk();
  }

  private processNextChunk() {
    // If already appending or no chunks, wait
    if (this.isAppending || this.pendingChunks.length === 0) {
      return;
    }

    // If SourceBuffer is not ready, wait
    if (!this.sourceBuffer || this.sourceBuffer.updating) {
      return;
    }

    // Get next chunk
    const chunk = this.pendingChunks.shift()!;

    try {
      this.isAppending = true;
      this.sourceBuffer.appendBuffer(chunk);

      // Start playback after first chunk
      if (this.audioElement.paused && this.audioElement.readyState >= 2) {
        console.log('[AudioStream] Starting playback');
        this.audioElement.play().catch(err => {
          console.error('[AudioStream] Playback error:', err);
        });
      }
    } catch (error) {
      console.error('[AudioStream] Error appending chunk:', error);
      this.isAppending = false;
    }
  }

  complete(totalChunks: number) {
    console.log(`[AudioStream] Completed with ${totalChunks} chunks`);

    // Process any remaining chunks
    if (this.pendingChunks.length > 0) {
      console.log(`[AudioStream] Processing ${this.pendingChunks.length} remaining chunks`);
      this.processNextChunk();
    }

    // Close MediaSource when SourceBuffer is done
    const closeMediaSource = () => {
      if (this.sourceBuffer && !this.sourceBuffer.updating && this.pendingChunks.length === 0) {
        try {
          if (this.mediaSource && this.mediaSource.readyState === 'open') {
            this.mediaSource.endOfStream();
            console.log('[AudioStream] MediaSource closed');
          }
        } catch (error) {
          console.error('[AudioStream] Error closing MediaSource:', error);
        }
      } else {
        // Check again after a short delay
        setTimeout(closeMediaSource, 50);
      }
    };

    closeMediaSource();
  }

  cleanup() {
    if (this.mediaSource) {
      const src = this.audioElement.src;
      if (src && src.startsWith('blob:')) {
        URL.revokeObjectURL(src);
      }
    }

    this.mediaSource = null;
    this.sourceBuffer = null;
    this.pendingChunks = [];
    this.isAppending = false;
  }

  private displayChatMessage(displayText: any) {
    // TODO: Implement chat message display
    // Same as traditional audio handling
  }

  private applyLive2DActions(actions: any) {
    // TODO: Implement Live2D expression/motion application
    // Same as traditional audio handling
  }
}
```

#### Step 3: Integrate with WebSocket Handlers

```typescript
// Global state
const activeStreams = new Map<number, AudioStreamManager>();

function handleAudioStart(message: AudioStartMessage) {
  const { sequence, display_text, actions } = message;

  // Create new stream manager
  const audioElement = document.getElementById('main-audio') as HTMLAudioElement;
  const streamManager = new AudioStreamManager(audioElement);

  // Store by sequence
  activeStreams.set(sequence, streamManager);

  // Initialize streaming
  streamManager.start(sequence, display_text, actions)
    .catch(error => {
      console.error('[AudioStream] Failed to start:', error);
      activeStreams.delete(sequence);
    });
}

function handleAudioChunk(message: AudioChunkMessage) {
  const { sequence, chunk_index, audio, is_header } = message;

  const streamManager = activeStreams.get(sequence);
  if (!streamManager) {
    console.warn(`[AudioStream] No active stream for sequence ${sequence}`);
    return;
  }

  streamManager.addChunk(audio, is_header);
}

function handleAudioComplete(message: AudioCompleteMessage) {
  const { sequence, total_chunks } = message;

  const streamManager = activeStreams.get(sequence);
  if (!streamManager) {
    console.warn(`[AudioStream] No active stream for sequence ${sequence}`);
    return;
  }

  streamManager.complete(total_chunks);

  // Cleanup after audio ends
  const audioElement = document.getElementById('main-audio') as HTMLAudioElement;
  audioElement.addEventListener('ended', () => {
    streamManager.cleanup();
    activeStreams.delete(sequence);
  }, { once: true });
}
```

---

## 🔄 Backward Compatibility

**CRITICAL**: The traditional `audio` message type must still work for non-streaming TTS engines (Azure, Edge, Melo, etc.).

```typescript
function handleTraditionalAudio(message: AudioMessage) {
  // Existing implementation - DO NOT REMOVE
  const { audio, volumes, slice_length, display_text, actions } = message;

  if (audio === null) {
    // Silent audio, just display text
    displayChatMessage(display_text);
    if (actions) applyLive2DActions(actions);
    return;
  }

  // Create data URL (existing method)
  const audioElement = document.getElementById('main-audio') as HTMLAudioElement;
  audioElement.src = `data:audio/wav;base64,${audio}`;

  audioElement.play();

  // Display text and actions
  displayChatMessage(display_text);
  if (actions) applyLive2DActions(actions);

  // Note: volumes array can be used for lip sync if needed
  // Implementation depends on your existing Live2D integration
}
```

---

## 🧪 Testing

### Test Case 1: Streaming Audio (GPT-SoVITS)

**Setup**:
- Set `streaming_mode: 2` in `conf.yaml`
- Use GPT-SoVITS TTS engine

**Expected Behavior**:
1. User sends message
2. ~0.6s later: audio starts playing (first chunk received)
3. Audio continues playing as more chunks arrive
4. Chat message appears with audio
5. Expression/motion applied

**Backend Logs** (for verification):
```
🎵 [Streaming] Starting streaming TTS for: '안녕하세요...'
🎵 [Streaming] Sent WAV header chunk (8192 bytes)
🎵 [Streaming] Completed streaming 15 chunks for sequence 0
```

**Frontend Logs** (to implement):
```
[AudioStream] Starting sequence 0
[AudioStream] Received WAV header (8192 bytes)
[AudioStream] Starting playback
[AudioStream] Completed with 15 chunks
[AudioStream] MediaSource closed
```

### Test Case 2: Traditional Audio (Azure TTS)

**Setup**:
- Use Azure TTS or any non-GPT-SoVITS engine

**Expected Behavior**:
1. User sends message
2. ~2s later: complete audio arrives
3. Audio plays using traditional method
4. No MediaSource API involved

**Verification**:
- Check that `message.type === 'audio'` (not `audio-start`)
- Ensure existing audio playback logic still works

### Test Case 3: Multiple Concurrent Messages

**Setup**:
- Send multiple messages in quick succession
- LLM generates multiple sentences simultaneously

**Expected Behavior**:
1. Multiple audio streams can be active (TTS generation in parallel)
2. Audio playback happens in correct order (sequence-based)
3. No audio overlap or mixing
4. Each audio has correct display text and actions

---

## 📊 Performance Metrics

### Before (Traditional)
- Time to first sound: **~2000ms**
- User experience: "AI is slow to respond"

### After (Streaming)
- Time to first sound: **~600ms** (67% reduction)
- User experience: "AI responds immediately"

### Measurement Points

Add performance logging:

```typescript
// In handleAudioStart
const startTime = performance.now();

// In handleAudioChunk (first chunk)
if (chunk_index === 0) {
  const timeToFirstChunk = performance.now() - startTime;
  console.log(`[Performance] Time to first audio chunk: ${timeToFirstChunk.toFixed(0)}ms`);
}

// When audio starts playing
audioElement.addEventListener('play', () => {
  const timeToPlay = performance.now() - startTime;
  console.log(`[Performance] Time to audio playback: ${timeToPlay.toFixed(0)}ms`);
}, { once: true });
```

---

## ⚠️ Known Limitations

### 1. Browser Compatibility
- **MediaSource API** requires modern browsers:
  - ✅ Chrome 23+
  - ✅ Firefox 42+
  - ✅ Safari 8+
  - ✅ Edge 12+
  - ❌ IE 11 (partial support, may have issues)

### 2. Mobile Considerations
- iOS Safari may have autoplay restrictions
- Audio streaming may consume more battery than traditional method
- Consider adding user preference toggle for streaming vs traditional

### 3. Network Conditions
- Poor network: chunks may arrive slowly, causing stuttering
- Consider implementing buffering strategy (wait for N chunks before playing)

---

## 📚 Reference Materials

### MediaSource API
- [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MediaSource)
- [HTML5 Rocks Tutorial](https://www.html5rocks.com/en/tutorials/eme/basics/)

### WAV Format
- [WAV Specification](http://soundfile.sapp.org/doc/WaveFormat/)
- Understanding WAV header structure for SourceBuffer

---

## ✅ Implementation Checklist

Frontend engineer should complete:

- [ ] Add TypeScript interfaces for new message types
- [ ] Implement `AudioStreamManager` class with MediaSource API
- [ ] Add WebSocket message handlers (audio-start, audio-chunk, audio-complete)
- [ ] Ensure backward compatibility with traditional audio
- [ ] Add performance logging
- [ ] Test with GPT-SoVITS (streaming mode)
- [ ] Test with Azure TTS (traditional mode)
- [ ] Test multiple concurrent messages
- [ ] Add error handling for MediaSource API failures
- [ ] Add fallback for unsupported browsers
- [ ] Document user-facing changes (if any)
- [ ] Update any relevant UI/UX elements

---

## 🤝 Questions?

If you need clarification on:
- Backend implementation details
- Message format specifications
- Audio format (WAV header structure)
- Testing procedures
- Integration with existing code

Please refer to:
- Backend commit: `cd8a790`
- Backend files:
  - `src/open_llm_vtuber/tts/tts_interface.py`
  - `src/open_llm_vtuber/tts/gpt_sovits_tts.py`
  - `src/open_llm_vtuber/conversations/tts_manager.py`

Or contact the backend team for assistance.

---

**Priority**: 🔴 **HIGH** - Blocking feature for real-time TTS experience

**Estimated Effort**: 1-2 days (including testing)

**Dependencies**: None (backend complete)

**Deadline**: TBD
