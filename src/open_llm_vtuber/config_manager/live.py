from pydantic import Field
from typing import Dict, ClassVar, List
from .i18n import I18nMixin, Description


class BiliBiliLiveConfig(I18nMixin):
    """Configuration for BiliBili Live platform."""

    room_ids: List[int] = Field([], alias="room_ids")
    sessdata: str = Field("", alias="sessdata")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "room_ids": Description(
            en="List of BiliBili live room IDs to monitor", zh="要监控的B站直播间ID列表"
        ),
        "sessdata": Description(
            en="SESSDATA cookie value for authenticated requests (optional)",
            zh="用于认证请求的SESSDATA cookie值（可选）",
        ),
    }


class DiscordLiveConfig(I18nMixin):
    """Configuration for Discord Live platform."""

    bot_token: str = Field("", alias="bot_token")
    channel_ids: List[int] = Field([], alias="channel_ids")
    prefix: str = Field("", alias="prefix")
    voice_enabled: bool = Field(False, alias="voice_enabled")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "bot_token": Description(
            en="Discord bot token from Developer Portal",
            zh="Discord 开发者门户的机器人令牌",
        ),
        "channel_ids": Description(
            en="List of Discord channel IDs to listen on (empty = all channels)",
            zh="要监听的 Discord 频道 ID 列表（为空则监听所有频道）",
        ),
        "prefix": Description(
            en="Command prefix to trigger the bot (empty = respond to all messages)",
            zh="触发机器人的命令前缀（为空则响应所有消息）",
        ),
        "voice_enabled": Description(
            en="Enable voice channel support (join/leave commands, TTS audio playback)",
            zh="启用语音频道支持（加入/离开命令，TTS 音频播放）",
        ),
    }


class LiveConfig(I18nMixin):
    """Configuration for live streaming platforms integration."""

    bilibili_live: BiliBiliLiveConfig = Field(
        BiliBiliLiveConfig(), alias="bilibili_live"
    )
    discord_live: DiscordLiveConfig = Field(DiscordLiveConfig(), alias="discord_live")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "bilibili_live": Description(
            en="Configuration for BiliBili Live platform", zh="B站直播平台配置"
        ),
        "discord_live": Description(
            en="Configuration for Discord Live platform",
            zh="Discord 直播平台配置",
        ),
    }
