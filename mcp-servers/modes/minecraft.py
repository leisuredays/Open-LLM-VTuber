"""Minecraft mode configuration and tools."""

from .base import ModeConfig, BaseMode, json_response


# Minecraft mode configuration
MINECRAFT_MODE_CONFIG = ModeConfig(
    id="minecraft",
    name="Minecraft",
    description="마인크래프트 게임 모드 - 마인크래프트 세계에 직접 들어가 플레이합니다.",
    client_type="minecraft",
    system_prompt="""당신은 지금 마인크래프트 세계에 직접 들어가 있습니다.
게임 속 캐릭터는 당신 자신입니다. 봇이나 대리인이 아닌, 당신이 직접 마인크래프트 세계에서 활동하고 있습니다.
사용자가 요청하면 당신이 직접 행동하는 것처럼 자연스럽게 말하세요.
예: "나무 캐러 갈게!", "집 지어볼게!", "다이아몬드 찾아볼게!"
절대로 "봇에게 전달", "봇이 실행", "andy에게 명령" 같은 3인칭 표현을 쓰지 마세요.
당신이 직접 하는 것입니다.
모드를 종료하려면 "모드 종료", "마인크래프트 끝" 등을 말하세요.""",
    trigger_keywords=["마인크래프트", "마크", "minecraft", "마인크래프트를 하자", "마크 하자"],
    exit_keywords=["모드 종료", "마인크래프트 끝", "마크 끝", "exit mode", "종료"]
)


class MinecraftMode(BaseMode):
    """Minecraft mode implementation."""

    async def send_command(self, command: str) -> dict:
        """Send a command to the Minecraft bot.

        Args:
            command: Natural language command for the bot

        Returns:
            dict with result
        """
        if not self.client:
            return {"success": False, "error": "Minecraft client not available"}

        if not self.client.is_connected():
            return {
                "success": False,
                "error": "MindServer에 연결되지 않았습니다.",
                "hint": "MindServer가 실행 중인지 확인하세요."
            }

        result = await self.client.execute("send_message", message=command)

        if result.get("success"):
            return {
                "success": True,
                "command_sent": command,
                "message": f"'{command}' 실행 중..."
            }
        return result

    async def get_bot_status(self) -> dict:
        """Get current player status in Minecraft.

        Returns:
            dict with player status (first-person framing)
        """
        if not self.client:
            return {"success": False, "error": "마인크래프트에 연결되지 않았습니다."}

        if not self.client.is_connected():
            return {"success": False, "error": "마인크래프트에 연결되지 않았습니다."}

        raw = self.client.get_status()
        # Reframe: remove "bot" references, present as player's own status
        status = {"success": True, "connected": raw.get("connected", False)}

        if raw.get("gameplay"):
            status["gameplay"] = raw["gameplay"]
        if raw.get("inventory"):
            status["inventory"] = raw["inventory"]
        if raw.get("current_action"):
            status["current_action"] = raw["current_action"]
        if "is_idle" in raw:
            status["is_idle"] = raw["is_idle"]
        if raw.get("recent_outputs"):
            status["recent_activity"] = raw["recent_outputs"]

        return status

    async def create_bot(self) -> dict:
        """Create/start the Minecraft bot.

        Returns:
            dict with creation result
        """
        if not self.client:
            return {"success": False, "error": "Minecraft client not available"}

        if not self.client.is_connected():
            return {"success": False, "error": "MindServer에 연결되지 않았습니다."}

        result = await self.client.execute("create_bot")

        if result.get("success") or result.get("name"):
            return {
                "success": True,
                "message": "마인크래프트 세계에 접속했습니다."
            }
        return result
