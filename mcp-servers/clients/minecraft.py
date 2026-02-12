"""Minecraft client - connects to Mindcraft MindServer via Socket.IO."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import aiohttp
import socketio

from .base import BaseClient

logger = logging.getLogger("router-mcp.minecraft")


@dataclass
class MindcraftClient(BaseClient):
    """Socket.IO client for Mindcraft MindServer."""

    host: str = "localhost"
    port: int = 8080
    bot_name: str = "andy"
    minecraft_host: str = "localhost"
    minecraft_port: int = 25565
    sio: socketio.AsyncClient = field(default_factory=socketio.AsyncClient)
    connected: bool = False
    _bot_outputs: list = field(default_factory=list)
    _max_outputs: int = 100
    _agents: dict = field(default_factory=dict)
    _agent_states: dict = field(default_factory=dict)

    def __init__(self, config: dict):
        """Initialize Mindcraft client from config."""
        self.config = config
        self.host = config.get("mindserver", {}).get("host", "localhost")
        self.port = config.get("mindserver", {}).get("port", 8080)
        self.bot_name = config.get("bot_name", "andy")
        self.minecraft_host = config.get("minecraft", {}).get("host", "localhost")
        self.minecraft_port = config.get("minecraft", {}).get("port", 25565)
        self.sio = socketio.AsyncClient()
        self.connected = False
        self._bot_outputs = []
        self._max_outputs = 100
        self._agents = {}
        self._agent_states = {}

        # VTuber callback settings
        self._vtuber_callback_url = config.get("vtuber_callback_url", "")
        self._callback_cooldown = config.get("callback_cooldown_seconds", 5)
        self._last_callback_time: float = 0.0
        self._http_session: Optional[aiohttp.ClientSession] = None

        # State change detection
        self._prev_health: Optional[float] = None
        self._prev_is_idle: Optional[bool] = None

    async def connect(self) -> bool:
        """Connect to the MindServer."""
        if self.connected:
            return True

        try:
            url = f"http://{self.host}:{self.port}"
            logger.info(f"Connecting to MindServer at {url}...")
            self._setup_event_handlers()
            await self.sio.connect(url, wait_timeout=10)
            self.connected = True
            await self.sio.emit("listen-to-agents")
            logger.info("Connected to MindServer successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MindServer: {e}")
            return False

    def _setup_event_handlers(self):
        """Set up Socket.IO event handlers."""

        @self.sio.on("agents-status")
        async def on_agents_status(agents: list):
            logger.debug(f"Agents status: {[a.get('name') for a in agents]}")
            for agent_data in agents:
                name = agent_data.get("name")
                if name:
                    self._agents[name] = agent_data

        @self.sio.on("bot-output")
        async def on_bot_output(agent_name: str, message: str):
            current_time = asyncio.get_event_loop().time()
            output = {
                "agent": agent_name,
                "message": message,
                "timestamp": current_time,
            }
            self._bot_outputs.append(output)
            if len(self._bot_outputs) > self._max_outputs:
                self._bot_outputs.pop(0)
            logger.debug(f"Bot output: {message[:100]}")

            # Classify and send callback for own bot's outputs
            if agent_name == self.bot_name and self._vtuber_callback_url:
                importance, _ = self._classify_bot_output(agent_name, message)
                if importance in ("HIGH", "MEDIUM"):
                    event_text = f"[마인크래프트 이벤트] {message}"
                    await self._send_vtuber_callback(importance, event_text)

        @self.sio.on("state-update")
        async def on_state_update(states: dict):
            if states:
                # Check for state changes before updating
                if self._vtuber_callback_url:
                    change = self._check_state_changes(states)
                    if change:
                        importance, event_text = change
                        await self._send_vtuber_callback(importance, event_text)
                self._agent_states = states

        @self.sio.on("connect")
        async def on_connect():
            logger.info("Socket.IO connected")

        @self.sio.on("disconnect")
        async def on_disconnect():
            logger.warning("Socket.IO disconnected")
            self.connected = False

    async def disconnect(self) -> None:
        """Disconnect from MindServer."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
        if self.connected:
            await self.sio.disconnect()
            self.connected = False

    async def execute(self, action: str, **kwargs) -> dict:
        """Execute an action on Mindcraft.

        Supported actions:
        - send_message: Send a message to the bot
        - create_bot: Create/start the bot
        - get_outputs: Get recent bot outputs
        """
        if action == "send_message":
            return await self._send_message(kwargs.get("message", ""))
        elif action == "create_bot":
            return await self._create_bot()
        elif action == "get_outputs":
            return {"outputs": self._get_outputs(kwargs.get("limit", 10))}
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    async def _send_message(self, message: str) -> dict:
        """Send message to bot."""
        if not self.connected:
            await self.connect()

        if not self.connected:
            return {"success": False, "error": "Not connected to MindServer"}

        await self.sio.emit(
            "send-message", (self.bot_name, {"from": "HALLOUI", "message": message})
        )
        return {"success": True, "bot": self.bot_name, "message_sent": message}

    async def _create_bot(self) -> dict:
        """Create the bot."""
        if not self.connected:
            await self.connect()

        if not self.connected:
            return {"success": False, "error": "Not connected to MindServer"}

        future = asyncio.get_event_loop().create_future()

        def callback(result):
            if not future.done():
                future.set_result(result)

        settings = {
            "profile": {"name": self.bot_name, "model": "gpt-4o-mini"},
            "host": self.minecraft_host,
            "port": self.minecraft_port,
            "auth": "offline",
            "base_profile": "survival",
        }

        await self.sio.emit("create-agent", settings, callback=callback)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result if result else {"success": True, "name": self.bot_name}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout"}

    def _get_outputs(self, limit: int = 10) -> list:
        """Get recent bot outputs."""
        outputs = [o for o in self._bot_outputs if o["agent"] == self.bot_name]
        return outputs[-limit:]

    def get_status(self) -> dict:
        """Get current client status."""
        bot_state = self._agent_states.get(self.bot_name, {})
        is_active = self.bot_name in self._agents

        status = {
            "connected": self.connected,
            "bot": self.bot_name,
            "is_active": is_active,
        }

        if bot_state:
            if "gameplay" in bot_state:
                gp = bot_state["gameplay"]
                status["gameplay"] = {
                    "health": gp.get("health"),
                    "hunger": gp.get("hunger"),
                    "position": gp.get("position"),
                    "time": gp.get("timeLabel"),
                    "weather": gp.get("weather"),
                }

            if "inventory" in bot_state:
                inv = bot_state["inventory"].get("counts", {})
                status["inventory"] = dict(list(inv.items())[:10])

            if "action" in bot_state:
                status["current_action"] = bot_state["action"].get("current")
                status["is_idle"] = bot_state["action"].get("isIdle", True)

        recent = self._get_outputs(5)
        if recent:
            status["recent_outputs"] = [o["message"][:100] for o in recent]

        return status

    def _classify_bot_output(self, agent_name: str, message: str) -> Tuple[str, str]:
        """Classify a bot output message by importance.

        Returns:
            Tuple of (importance, reason) where importance is
            HIGH, MEDIUM, LOW, or IGNORE.
        """
        msg = message.lower().strip()

        # IGNORE: debug/system messages
        if msg.startswith("full response to"):
            return ("IGNORE", "debug message")
        if len(msg) < 3:
            return ("IGNORE", "too short")

        # HIGH: death, combat, rare items, task completion
        high_patterns = [
            r"\bdied\b",
            r"\bkilled\b",
            r"\battack",
            r"\bzombie\b",
            r"\bskeleton\b",
            r"\bcreeper\b",
            r"\bspider\b",
            r"\benderman\b",
            r"\bdiamond\b",
            r"\bemerald\b",
            r"\bnetherite\b",
            r"\bcompleted\b",
            r"\bfinished\b",
            r"\bdanger\b",
            r"\bfighting\b",
            r"\bhurt\b",
            r"\bdamage\b",
        ]
        for pattern in high_patterns:
            if re.search(pattern, msg):
                return ("HIGH", f"matched: {pattern}")

        # MEDIUM: crafting, discovery, arrival
        medium_patterns = [
            r"\bcraft",
            r"\bfound\b",
            r"\barrived\b",
            r"\bbuild",
            r"\bmine[ds]?\b",
            r"\bpicked up\b",
            r"\bcollect",
        ]
        for pattern in medium_patterns:
            if re.search(pattern, msg):
                return ("MEDIUM", f"matched: {pattern}")

        return ("LOW", "no significant pattern")

    def _check_state_changes(self, states: dict) -> Optional[Tuple[str, str]]:
        """Check for significant state changes. Returns (importance, event_text) or None."""
        bot_state = states.get(self.bot_name, {})
        if not bot_state:
            return None

        result = None

        # Check health drop
        gameplay = bot_state.get("gameplay", {})
        current_health = gameplay.get("health")
        if current_health is not None and self._prev_health is not None:
            health_drop = self._prev_health - current_health
            if health_drop >= 5:
                result = (
                    "HIGH",
                    f"[마인크래프트 이벤트] 체력이 크게 떨어졌습니다! "
                    f"{self._prev_health:.0f} → {current_health:.0f} "
                    f"(-{health_drop:.0f})",
                )
        if current_health is not None:
            self._prev_health = current_health

        # Check idle transition (active → idle)
        action = bot_state.get("action", {})
        current_is_idle = action.get("isIdle")
        if (
            current_is_idle is not None
            and self._prev_is_idle is not None
            and current_is_idle
            and not self._prev_is_idle
        ):
            current_action = action.get("current", "알 수 없음")
            if result is None:
                result = (
                    "MEDIUM",
                    f"[마인크래프트 이벤트] 작업이 완료되어 대기 상태로 전환됨. "
                    f"마지막 작업: {current_action}",
                )
        if current_is_idle is not None:
            self._prev_is_idle = current_is_idle

        return result

    async def _send_vtuber_callback(self, importance: str, event_text: str) -> None:
        """Send event to VTuber server via HTTP POST."""
        now = time.monotonic()
        if now - self._last_callback_time < self._callback_cooldown:
            logger.debug("VTuber callback skipped (cooldown)")
            return

        if not self._http_session or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        payload = {"event_text": event_text, "importance": importance}
        try:
            async with self._http_session.post(
                self._vtuber_callback_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                self._last_callback_time = now
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(
                        f"VTuber callback sent ({importance}): {result.get('status')}"
                    )
                else:
                    logger.warning(f"VTuber callback returned {resp.status}")
        except Exception as e:
            logger.warning(f"VTuber callback failed: {e}")

    def get_context_prompt(self) -> str:
        """Get context prompt with current player state for LLM (first-person)."""
        status = self.get_status()

        if not status.get("connected"):
            return "⚠️ 마인크래프트에 연결되지 않았습니다."

        lines = ["🎮 당신은 마인크래프트 세계에 있습니다."]

        # Gameplay info
        gp = status.get("gameplay", {})
        if gp:
            lines.append("\n📊 당신의 현재 상태:")
            if gp.get("health") is not None:
                lines.append(f"  ❤️ 체력: {gp['health']}/20")
            if gp.get("hunger") is not None:
                lines.append(f"  🍖 배고픔: {gp['hunger']}/20")
            if gp.get("position"):
                pos = gp["position"]
                lines.append(
                    f"  📍 위치: x={pos.get('x', 0):.0f}, y={pos.get('y', 0):.0f}, z={pos.get('z', 0):.0f}"
                )
            if gp.get("time"):
                lines.append(f"  🕐 시간: {gp['time']}")
            if gp.get("weather"):
                lines.append(f"  🌤️ 날씨: {gp['weather']}")

        # Current action
        if status.get("current_action"):
            lines.append(f"\n🎯 지금 하고 있는 일: {status['current_action']}")
        elif status.get("is_idle"):
            lines.append("\n🎯 지금 하고 있는 일: 없음 (대기 중)")

        # Inventory
        inv = status.get("inventory", {})
        if inv:
            lines.append("\n🎒 당신의 인벤토리 (상위 10개):")
            for item, count in list(inv.items())[:10]:
                lines.append(f"  - {item}: {count}")

        # Recent outputs
        outputs = status.get("recent_outputs", [])
        if outputs:
            lines.append("\n💬 최근 활동 로그:")
            for msg in outputs[-3:]:
                lines.append(f"  > {msg}")

        lines.append("\n사용자의 요청을 직접 행동으로 실행하세요.")
        lines.append("모드 종료: '종료' 또는 '모드 종료'")

        return "\n".join(lines)
