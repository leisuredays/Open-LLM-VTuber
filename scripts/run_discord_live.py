import os
import sys
import asyncio
from loguru import logger

# Add project root to path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.open_llm_vtuber.live.discord_live import DiscordLivePlatform
from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config


async def main():
    """
    Main function to run the Discord Live platform client.
    Connects to Discord and forwards messages to the VTuber.
    """
    logger.info("Starting Discord Live platform client")

    try:
        # Load configuration
        config_path = os.path.join(project_root, "conf.yaml")
        config_data = read_yaml(config_path)
        config = validate_config(config_data)

        # Extract Discord configuration
        discord_config = config.live_config.discord_live

        # Check if bot token is provided
        if not discord_config.bot_token:
            logger.error(
                "No Discord bot token specified in configuration. "
                "Please add your bot token to conf.yaml under "
                "live_config.discord_live.bot_token"
            )
            return

        logger.info(
            f"Discord bot prefix: '{discord_config.prefix}', "
            f"channels: {discord_config.channel_ids or 'all'}, "
            f"voice: {discord_config.voice_enabled}"
        )

        # Build proxy URL from server config
        host = config.system_config.host
        port = config.system_config.port
        if host == "0.0.0.0":
            host = "localhost"
        proxy_url = f"ws://{host}:{port}/proxy-ws"

        # Initialize and run the Discord platform
        platform = DiscordLivePlatform(
            bot_token=discord_config.bot_token,
            channel_ids=discord_config.channel_ids,
            prefix=discord_config.prefix,
            proxy_url=proxy_url,
            voice_enabled=discord_config.voice_enabled,
        )

        await platform.run()

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error(
            "Make sure you have installed discord.py with: uv sync --extra discord"
        )
    except Exception as e:
        logger.error(f"Error starting Discord Live client: {e}")
        import traceback

        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Discord Live platform")

# Usage: uv run python scripts/run_discord_live.py
