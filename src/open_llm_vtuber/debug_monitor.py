"""
Debug Monitor for Open-LLM-VTuber
실시간으로 LLM에 전송되는 프롬프트를 모니터링합니다.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Set
from dataclasses import dataclass, asdict
from fastapi import WebSocket
from loguru import logger


@dataclass
class PromptLog:
    """프롬프트 로그 항목"""
    timestamp: str
    system_prompt: str
    messages: List[Dict[str, Any]]
    tools: List[str] | None = None
    model: str | None = None
    trigger_reason: str | None = None


class DebugMonitor:
    """실시간 프롬프트 모니터링"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._subscribers: Set[WebSocket] = set()
        self._history: List[PromptLog] = []
        self._max_history = 50
        logger.info("DebugMonitor initialized")

    async def subscribe(self, websocket: WebSocket):
        """WebSocket 구독 추가"""
        self._subscribers.add(websocket)
        logger.info(f"Debug subscriber added. Total: {len(self._subscribers)}")

        # 기존 히스토리 전송
        if self._history:
            await websocket.send_json({
                "type": "history",
                "data": [asdict(log) for log in self._history]
            })

    def unsubscribe(self, websocket: WebSocket):
        """WebSocket 구독 해제"""
        self._subscribers.discard(websocket)
        logger.info(f"Debug subscriber removed. Total: {len(self._subscribers)}")

    async def log_prompt(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        trigger_reason: str | None = None,
    ):
        """프롬프트 로그 및 브로드캐스트"""
        # 도구 이름만 추출
        tool_names = None
        if tools:
            tool_names = [t.get("function", {}).get("name") or t.get("name", "unknown") for t in tools]

        log = PromptLog(
            timestamp=datetime.now().isoformat(),
            system_prompt=system_prompt,
            messages=messages,
            tools=tool_names,
            model=model,
            trigger_reason=trigger_reason,
        )

        # 히스토리 저장
        self._history.append(log)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # 구독자에게 브로드캐스트
        if self._subscribers:
            message = {
                "type": "prompt",
                "data": asdict(log)
            }

            dead_subscribers = set()
            for ws in self._subscribers:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead_subscribers.add(ws)

            # 연결 끊긴 구독자 제거
            for ws in dead_subscribers:
                self._subscribers.discard(ws)

    def get_history(self) -> List[Dict[str, Any]]:
        """히스토리 반환"""
        return [asdict(log) for log in self._history]

    def clear_history(self):
        """히스토리 초기화"""
        self._history.clear()


# 싱글톤 인스턴스
debug_monitor = DebugMonitor()
