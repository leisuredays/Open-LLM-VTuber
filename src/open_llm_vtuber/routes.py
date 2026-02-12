import os
import json
from typing import Tuple
from uuid import uuid4
import numpy as np
from datetime import datetime
from fastapi import APIRouter, WebSocket, UploadFile, File, Response
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler
from .debug_monitor import debug_monitor


def init_client_ws_route(
    default_context_cache: ServiceContext,
) -> Tuple[APIRouter, WebSocketHandler]:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        Tuple of (APIRouter, WebSocketHandler).
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        await websocket.accept()
        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router, ws_handler


class MinecraftEventRequest(BaseModel):
    event_text: str
    importance: str = "MEDIUM"


def init_event_routes(ws_handler: WebSocketHandler) -> APIRouter:
    """
    Create API routes for external event triggers (e.g. Minecraft events).

    Args:
        ws_handler: The WebSocketHandler instance managing client connections.

    Returns:
        APIRouter with event endpoints.
    """
    router = APIRouter()

    @router.post("/api/minecraft-event")
    async def minecraft_event(req: MinecraftEventRequest):
        """Receive a Minecraft event and trigger VTuber reaction."""
        if not ws_handler.client_connections:
            return JSONResponse(
                {"status": "ignored", "reason": "no connected clients"},
                status_code=200,
            )

        # Pick the first connected client
        client_uid = next(iter(ws_handler.client_connections))
        websocket = ws_handler.client_connections[client_uid]

        # Skip if a conversation is already in progress for this client
        task = ws_handler.current_conversation_tasks.get(client_uid)
        if task and not task.done():
            return JSONResponse(
                {"status": "ignored", "reason": "conversation in progress"},
                status_code=200,
            )

        data = {
            "type": "minecraft-event",
            "text": req.event_text,
            "importance": req.importance,
        }

        try:
            await ws_handler._handle_conversation_trigger(websocket, client_uid, data)
            return JSONResponse({"status": "ok"}, status_code=200)
        except Exception as e:
            logger.error(f"Error handling minecraft event: {e}")
            return JSONResponse({"status": "error", "reason": str(e)}, status_code=500)

    return router


def init_proxy_route(server_url: str) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        server_url: The WebSocket URL of the actual server

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    proxy_handler = ProxyHandler(server_url)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info")
    async def get_live2d_folder_info():
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
                    avatar_file = None
                    for ext in supported_extensions:
                        avatar_path = os.path.join(
                            live2d_dir, folder_name, f"{folder_name}{ext}"
                        )
                        if os.path.isfile(avatar_path):
                            avatar_file = avatar_path.replace("\\", "/")
                            break

                    valid_characters.append(
                        {
                            "name": folder_name,
                            "avatar": avatar_file,
                            "model_path": model3_file,
                        }
                    )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr")
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        try:
            contents = await file.read()

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
            audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        await websocket.accept()
        logger.info("TTS WebSocket connection established")

        try:
            while True:
                data = await websocket.receive_json()
                text = data.get("text")
                if not text:
                    continue

                logger.info(f"Received text for TTS: {text}")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()

    return router


def init_debug_routes() -> APIRouter:
    """
    프롬프트 디버그 모니터링 라우트
    """
    router = APIRouter()

    @router.websocket("/debug-ws")
    async def debug_websocket(websocket: WebSocket):
        """실시간 프롬프트 모니터링 WebSocket"""
        await websocket.accept()
        await debug_monitor.subscribe(websocket)

        try:
            while True:
                # 클라이언트 메시지 대기 (연결 유지)
                data = await websocket.receive_text()
                if data == "clear":
                    debug_monitor.clear_history()
                    await websocket.send_json({"type": "cleared"})
        except WebSocketDisconnect:
            debug_monitor.unsubscribe(websocket)
        except Exception as e:
            logger.error(f"Debug WebSocket error: {e}")
            debug_monitor.unsubscribe(websocket)

    @router.get("/debug")
    async def debug_page():
        """프롬프트 모니터링 페이지"""
        html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>프롬프트 모니터</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status {
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
            background: #333;
        }
        .status.connected { background: #0a3; color: #fff; }
        .status.disconnected { background: #a30; color: #fff; }
        .controls {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        button {
            padding: 8px 16px;
            background: #00d4ff;
            border: none;
            border-radius: 4px;
            color: #000;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #00b8e6; }
        button.danger { background: #ff4757; color: #fff; }
        button.danger:hover { background: #ff3344; }
        #logs {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .log-entry {
            background: #16213e;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #00d4ff;
        }
        .log-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 12px;
            color: #888;
        }
        .log-model { color: #00d4ff; font-weight: bold; }
        .section {
            margin-top: 10px;
            padding: 10px;
            background: #0f0f23;
            border-radius: 4px;
        }
        .section-title {
            font-size: 11px;
            color: #00d4ff;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .system-prompt {
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
            line-height: 1.5;
        }
        .messages {
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
        }
        .message {
            padding: 8px;
            margin: 4px 0;
            border-radius: 4px;
        }
        .message.user { background: #1e3a5f; border-left: 3px solid #4a9eff; }
        .message.assistant { background: #2d1f3d; border-left: 3px solid #a855f7; }
        .message.system { background: #1f2d1f; border-left: 3px solid #22c55e; }
        .message.tool { background: #3d2d1f; border-left: 3px solid #f59e0b; }
        .message-role {
            font-size: 10px;
            color: #888;
            margin-bottom: 4px;
        }
        .message-content {
            white-space: pre-wrap;
            word-break: break-word;
        }
        .tools {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        .tool-tag {
            background: #00d4ff33;
            color: #00d4ff;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
        }
        .empty {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .collapsible {
            cursor: pointer;
        }
        .collapsible::before {
            content: "▼ ";
            font-size: 10px;
        }
        .collapsible.collapsed::before {
            content: "▶ ";
        }
        .collapse-content {
            display: block;
        }
        .collapse-content.hidden {
            display: none;
        }
    </style>
</head>
<body>
    <h1>
        프롬프트 모니터
        <span id="status" class="status disconnected">연결 끊김</span>
    </h1>
    <div class="controls">
        <button onclick="clearLogs()">로그 지우기</button>
        <button onclick="toggleAutoScroll()">자동 스크롤: <span id="autoScrollStatus">ON</span></button>
    </div>
    <div id="logs">
        <div class="empty">대기 중... LLM 호출이 발생하면 여기에 표시됩니다.</div>
    </div>

    <script>
        let ws;
        let autoScroll = true;
        let isFirstLog = true;

        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/debug-ws`);

            ws.onopen = () => {
                document.getElementById('status').textContent = '연결됨';
                document.getElementById('status').className = 'status connected';
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = '연결 끊김';
                document.getElementById('status').className = 'status disconnected';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                if (data.type === 'history') {
                    data.data.forEach(log => addLogEntry(log));
                } else if (data.type === 'prompt') {
                    addLogEntry(data.data);
                } else if (data.type === 'cleared') {
                    document.getElementById('logs').innerHTML = '<div class="empty">로그가 지워졌습니다.</div>';
                    isFirstLog = true;
                }
            };
        }

        function addLogEntry(log) {
            const logsDiv = document.getElementById('logs');
            if (isFirstLog) {
                logsDiv.innerHTML = '';
                isFirstLog = false;
            }

            const entry = document.createElement('div');
            entry.className = 'log-entry';

            const time = new Date(log.timestamp).toLocaleTimeString('ko-KR');
            const tools = log.tools ? log.tools.map(t => `<span class="tool-tag">${t}</span>`).join('') : '';

            let messagesHtml = '';
            if (log.messages) {
                log.messages.forEach(msg => {
                    const role = msg.role || 'unknown';
                    let content = '';
                    if (typeof msg.content === 'string') {
                        content = msg.content;
                    } else if (Array.isArray(msg.content)) {
                        content = msg.content.map(c => c.text || JSON.stringify(c)).join('\\n');
                    } else if (msg.content) {
                        content = JSON.stringify(msg.content, null, 2);
                    }
                    // Tool calls
                    if (msg.tool_calls) {
                        content += '\\n[Tool Calls: ' + msg.tool_calls.map(tc => tc.function?.name || tc.name).join(', ') + ']';
                    }
                    messagesHtml += `<div class="message ${role}">
                        <div class="message-role">${role.toUpperCase()}</div>
                        <div class="message-content">${escapeHtml(content)}</div>
                    </div>`;
                });
            }

            entry.innerHTML = `
                <div class="log-header">
                    <span>${time}</span>
                    <span class="log-model">${log.model || 'unknown'}</span>
                </div>
                ${tools ? `<div class="section"><div class="section-title">사용 가능한 도구</div><div class="tools">${tools}</div></div>` : ''}
                <div class="section">
                    <div class="section-title collapsible" onclick="toggleCollapse(this)">시스템 프롬프트</div>
                    <div class="system-prompt collapse-content">${escapeHtml(log.system_prompt || '')}</div>
                </div>
                <div class="section">
                    <div class="section-title collapsible" onclick="toggleCollapse(this)">메시지 (${log.messages?.length || 0})</div>
                    <div class="messages collapse-content">${messagesHtml}</div>
                </div>
            `;

            logsDiv.insertBefore(entry, logsDiv.firstChild);

            if (autoScroll) {
                window.scrollTo(0, 0);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function toggleCollapse(el) {
            el.classList.toggle('collapsed');
            el.nextElementSibling.classList.toggle('hidden');
        }

        function clearLogs() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('clear');
            }
        }

        function toggleAutoScroll() {
            autoScroll = !autoScroll;
            document.getElementById('autoScrollStatus').textContent = autoScroll ? 'ON' : 'OFF';
        }

        connect();
    </script>
</body>
</html>"""
        return Response(content=html_content, media_type="text/html")

    @router.get("/debug/history")
    async def get_debug_history():
        """프롬프트 히스토리 JSON"""
        return JSONResponse(debug_monitor.get_history())

    return router
