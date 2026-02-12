#!/usr/bin/env python3
"""
Router MCP Server - Modular mode-based routing

Thin main router that:
1. Loads mode configurations and clients from modules
2. Manages global mode state
3. Registers tools dynamically based on available modes
"""

import os
# Disable FastMCP banner before importing fastmcp
os.environ["FASTMCP_DISABLE_ANALYTICS"] = "true"

import asyncio
import json
import logging
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# Import modular components
from clients import get_client_class, list_client_types
from clients.mindserver_process import MindServerProcessManager, ProcessConfig
from modes import (
    get_mode_config, get_mode_class, get_all_mode_configs,
    list_modes, json_response, ModeConfig
)

# ============================================================================
# Logging Setup
# ============================================================================

# Ensure logging goes to stderr (not stdout) to avoid interfering with stdio transport
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("router-mcp")

# ============================================================================
# Configuration
# ============================================================================

def load_config() -> dict:
    """Load configuration from router_config.json."""
    config_path = Path(__file__).parent / "router_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

CONFIG = load_config()
logger.info(f"Config loaded: {list(CONFIG.keys())}")

# ============================================================================
# Global State
# ============================================================================

@dataclass
class GlobalState:
    """Global state shared across all sessions."""
    active_mode: Optional[str] = None
    mode_history: list = field(default_factory=list)

_state = GlobalState()

# Client instances (initialized on startup)
_clients: dict = {}  # mode_id -> client instance
_modes: dict = {}    # mode_id -> mode instance

# Process managers for modes that require external process management.
# Each mode has an ordered list: started in order, stopped in reverse order.
_process_managers: dict[str, list[MindServerProcessManager]] = {}  # mode_id -> [mgr, ...]

# Shutdown management
shutdown_event = threading.Event()
_client_threads: dict = {}

# ============================================================================
# Client Management
# ============================================================================

def init_clients():
    """Initialize all clients based on registered modes.

    For modes with a process manager (e.g. minecraft with mindserver_process config),
    the client and mode objects are created but NOT connected. Connection happens
    on activate_mode().
    """
    # Initialize process managers from config (ordered: MC server first, MindServer second)
    managers: list[MindServerProcessManager] = []

    mc_srv_cfg = CONFIG.get("minecraft_server")
    if mc_srv_cfg and mc_srv_cfg.get("enabled", False):
        managers.append(MindServerProcessManager(
            config=ProcessConfig.from_dict(mc_srv_cfg, name="minecraft-server"),
        ))
        logger.info("Process manager registered: minecraft-server")

    ms_cfg = CONFIG.get("mindserver_process")
    if ms_cfg and ms_cfg.get("enabled", False):
        managers.append(MindServerProcessManager(
            config=ProcessConfig.from_dict(ms_cfg, name="mindserver"),
        ))
        logger.info("Process manager registered: mindserver")

    if managers:
        _process_managers["minecraft"] = managers

    for mode_id in list_modes():
        mode_config = get_mode_config(mode_id)
        if not mode_config:
            continue

        client_type = mode_config.client_type
        client_class = get_client_class(client_type)

        if client_class:
            try:
                # Get client-specific config
                client_config = CONFIG.copy()
                client = client_class(client_config)
                _clients[mode_id] = client

                # Create mode instance
                mode_class = get_mode_class(mode_id)
                if mode_class:
                    _modes[mode_id] = mode_class(mode_config, client)

                logger.info(f"Initialized client for mode: {mode_id}")
            except Exception as e:
                logger.error(f"Failed to initialize client for {mode_id}: {e}")


async def connect_clients():
    """Connect all initialized clients."""
    for mode_id, client in _clients.items():
        try:
            await client.connect()
            logger.info(f"Connected client for mode: {mode_id}")
        except Exception as e:
            logger.error(f"Failed to connect client for {mode_id}: {e}")


MAX_CONNECT_RETRIES = 3

def run_client_loop(mode_id: str):
    """Run client connection loop in a separate thread.
    Retries up to MAX_CONNECT_RETRIES times on initial connection.
    If connected and later disconnected, retries again up to MAX_CONNECT_RETRIES.

    For process-managed modes: does NOT remove client/mode on max retries
    (they can be re-activated), and exits the loop if the managed process has stopped.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    is_managed = mode_id in _process_managers

    async def maintain_connection():
        client = _clients.get(mode_id)
        if not client:
            return

        retries = 0

        while not shutdown_event.is_set():
            # For managed modes: if the primary process (last in list) is no longer running, exit
            if is_managed:
                mgr_list = _process_managers.get(mode_id, [])
                primary = mgr_list[-1] if mgr_list else None
                if primary and not primary.is_running and not primary.is_port_available:
                    logger.info(f"Client {mode_id}: managed process stopped — exiting loop")
                    return

            try:
                if not client.is_connected():
                    connected = await client.connect()
                    if not connected:
                        retries += 1
                        if retries >= MAX_CONNECT_RETRIES:
                            if is_managed:
                                # Keep client/mode for re-activation
                                logger.warning(
                                    f"Client {mode_id}: failed to connect after "
                                    f"{MAX_CONNECT_RETRIES} attempts. "
                                    f"Keeping for re-activation."
                                )
                            else:
                                logger.warning(
                                    f"Client {mode_id}: failed to connect after "
                                    f"{MAX_CONNECT_RETRIES} attempts. Disabling."
                                )
                                _clients.pop(mode_id, None)
                                _modes.pop(mode_id, None)
                            return
                        logger.warning(
                            f"Client {mode_id}: connection attempt "
                            f"{retries}/{MAX_CONNECT_RETRIES} failed, retrying..."
                        )
                        await asyncio.sleep(3)
                        continue

                # Connected successfully — reset retry counter
                retries = 0

                while client.is_connected() and not shutdown_event.is_set():
                    await asyncio.sleep(0.5)

                if shutdown_event.is_set():
                    break

                logger.warning(f"Client {mode_id} disconnected, reconnecting...")
                await asyncio.sleep(3)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not shutdown_event.is_set():
                    retries += 1
                    if retries >= MAX_CONNECT_RETRIES:
                        if is_managed:
                            logger.warning(
                                f"Client {mode_id}: error after "
                                f"{MAX_CONNECT_RETRIES} attempts. "
                                f"Keeping for re-activation. ({e})"
                            )
                        else:
                            logger.warning(
                                f"Client {mode_id}: error after "
                                f"{MAX_CONNECT_RETRIES} attempts. Disabling. ({e})"
                            )
                            _clients.pop(mode_id, None)
                            _modes.pop(mode_id, None)
                        return
                    logger.error(f"Client {mode_id} error ({retries}/{MAX_CONNECT_RETRIES}): {e}")
                    await asyncio.sleep(3)

        # Cleanup
        if client.is_connected():
            try:
                await client.disconnect()
            except Exception:
                pass

    try:
        loop.run_until_complete(maintain_connection())
    finally:
        loop.close()


# ============================================================================
# FastMCP Server
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """MCP server lifecycle."""
    # Wait for non-managed clients to connect (managed ones connect on activate)
    non_managed = {
        mid: c for mid, c in _clients.items() if mid not in _process_managers
    }
    if non_managed:
        for _ in range(20):
            connected = any(c.is_connected() for c in non_managed.values())
            if connected:
                break
            await asyncio.sleep(0.5)

    yield

mcp = FastMCP(
    name="router-mcp",
    instructions="""Router MCP Server - 모드 기반 라우팅

모드 관리:
- list_available_modes(): 사용 가능한 모드 목록
- activate_mode(mode_id): 모드 활성화
- deactivate_mode(): 모드 비활성화

마인크래프트 모드 (활성화 후 사용):
- minecraft_command(command): 마인크래프트에서 행동 실행
- minecraft_status(): 현재 상태 확인
""",
    lifespan=lifespan,
)

# ============================================================================
# REST API Endpoints (for external integration)
# ============================================================================

@mcp.custom_route("/api/mode-prompt", methods=["GET"])
async def api_get_mode_prompt(request: Request) -> JSONResponse:
    """
    REST API endpoint to get the current mode's context prompt.
    Used by external systems (e.g., Dify) to inject mode-specific prompts.

    Returns:
        JSONResponse with mode prompt and metadata
    """
    if not _state.active_mode:
        return JSONResponse({
            "success": True,
            "has_mode": False,
            "mode_id": None,
            "mode_name": None,
            "prompt": "일반 대화 모드입니다. 사용자와 자연스럽게 대화하세요.",
            "exit_keywords": []
        })

    config = get_mode_config(_state.active_mode)
    client = _clients.get(_state.active_mode)

    context_prompt = ""
    if client:
        context_prompt = client.get_context_prompt()

    return JSONResponse({
        "success": True,
        "has_mode": True,
        "mode_id": _state.active_mode,
        "mode_name": config.name if config else _state.active_mode,
        "prompt": context_prompt,
        "system_prompt": config.system_prompt if config else "",
        "exit_keywords": config.exit_keywords if config else []
    })


@mcp.custom_route("/api/mode/activate", methods=["POST"])
async def api_activate_mode(request: Request) -> JSONResponse:
    """
    REST API endpoint to activate a mode.

    Request body:
        {"mode_id": "minecraft"}
    """
    try:
        body = await request.json()
        mode_id = body.get("mode_id")
    except Exception:
        return JSONResponse({"success": False, "error": "Invalid JSON body"}, status_code=400)

    if not mode_id:
        return JSONResponse({"success": False, "error": "mode_id is required"}, status_code=400)

    config = get_mode_config(mode_id)
    if not config:
        return JSONResponse({
            "success": False,
            "error": f"Unknown mode '{mode_id}'",
            "available_modes": list_modes()
        }, status_code=404)

    if _state.active_mode and _state.active_mode != mode_id:
        _state.mode_history.append(_state.active_mode)

    _state.active_mode = mode_id

    client = _clients.get(mode_id)
    context_prompt = client.get_context_prompt() if client else ""

    return JSONResponse({
        "success": True,
        "activated_mode": {
            "id": mode_id,
            "name": config.name,
            "description": config.description
        },
        "prompt": context_prompt,
        "system_prompt": config.system_prompt,
        "exit_keywords": config.exit_keywords,
        "message": f"✅ {config.name} 모드가 활성화되었습니다!"
    })


@mcp.custom_route("/api/mode/deactivate", methods=["POST"])
async def api_deactivate_mode(request: Request) -> JSONResponse:
    """REST API endpoint to deactivate the current mode."""
    if not _state.active_mode:
        return JSONResponse({
            "success": False,
            "error": "No active mode to deactivate"
        }, status_code=400)

    deactivated = _state.active_mode
    config = get_mode_config(deactivated)

    _state.mode_history.append(deactivated)
    _state.active_mode = None

    return JSONResponse({
        "success": True,
        "deactivated_mode": deactivated,
        "message": f"🔚 {config.name if config else deactivated} 모드가 종료되었습니다."
    })


@mcp.custom_route("/api/health", methods=["GET"])
async def api_health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    connected_clients = [
        mode_id for mode_id, client in _clients.items()
        if client.is_connected()
    ]
    return JSONResponse({
        "status": "ok",
        "active_mode": _state.active_mode,
        "connected_clients": connected_clients,
        "available_modes": list_modes()
    })


# ============================================================================
# Mode Management Tools
# ============================================================================

@mcp.tool()
def list_available_modes() -> str:
    """사용 가능한 모드 목록을 반환합니다."""
    modes = []
    for mode_id, config in get_all_mode_configs().items():
        modes.append({
            "id": mode_id,
            "name": config.name,
            "description": config.description,
            "trigger_keywords": config.trigger_keywords
        })

    return json_response({
        "available_modes": modes,
        "total_count": len(modes)
    })


@mcp.tool()
async def activate_mode(mode_id: str) -> str:
    """특정 모드를 활성화합니다.

    프로세스 관리가 설정된 모드의 경우, 외부 프로세스를 자동으로 시작하고
    연결합니다.

    Args:
        mode_id: 활성화할 모드 ID (예: 'minecraft')
    """
    config = get_mode_config(mode_id)
    if not config:
        return json_response({
            "success": False,
            "error": f"Unknown mode '{mode_id}'",
            "available_modes": list_modes()
        })

    if _state.active_mode and _state.active_mode != mode_id:
        _state.mode_history.append(_state.active_mode)

    # Start managed processes in order (e.g. MC server -> MindServer)
    mgr_list = _process_managers.get(mode_id, [])
    started_mgrs: list[MindServerProcessManager] = []
    for mgr in mgr_list:
        logger.info(f"Starting managed process for mode {mode_id}: {mgr._name}")
        started = await mgr.start()
        if not started:
            # Rollback: stop already-started managers in reverse order
            for prev in reversed(started_mgrs):
                await prev.stop()
            return json_response({
                "success": False,
                "error": f"Failed to start {mgr._name} for mode '{mode_id}'",
                "message": f"{mgr._name} 프로세스를 시작하지 못했습니다."
            })
        started_mgrs.append(mgr)

    # Fire-and-forget launch commands from all managers
    for mgr in mgr_list:
        mgr.run_launch_commands()

    # Connect client
    client = _clients.get(mode_id)
    if client and not client.is_connected():
        logger.info(f"Connecting client for mode: {mode_id}")
        connected = await client.connect()
        if not connected:
            # Rollback: stop all started managers in reverse order
            for mgr in reversed(started_mgrs):
                await mgr.stop()
            return json_response({
                "success": False,
                "error": f"Failed to connect client for mode '{mode_id}'",
                "message": "MindServer에 연결하지 못했습니다."
            })

    _state.active_mode = mode_id

    # Start reconnection loop thread for this mode (if not already running)
    existing_thread = _client_threads.get(mode_id)
    if not existing_thread or not existing_thread.is_alive():
        thread = threading.Thread(
            target=run_client_loop,
            args=(mode_id,),
            name=f"Client-{mode_id}",
            daemon=True,
        )
        thread.start()
        _client_threads[mode_id] = thread

    # Get real-time context prompt from client
    context_prompt = ""
    if client:
        context_prompt = client.get_context_prompt()

    return json_response({
        "success": True,
        "activated_mode": {
            "id": mode_id,
            "name": config.name,
            "description": config.description
        },
        "system_prompt": config.system_prompt,
        "context_prompt": context_prompt,
        "exit_keywords": config.exit_keywords,
        "message": f"✅ {config.name} 모드가 활성화되었습니다!"
    })


@mcp.tool()
async def deactivate_mode() -> str:
    """현재 활성화된 모드를 비활성화합니다.

    프로세스 관리가 설정된 모드의 경우, 클라이언트 연결을 해제하고
    외부 프로세스를 자동으로 종료합니다.
    """
    if not _state.active_mode:
        return json_response({
            "success": False,
            "error": "No active mode",
            "message": "현재 활성화된 모드가 없습니다."
        })

    deactivated = _state.active_mode
    config = get_mode_config(deactivated)

    # Disconnect client
    client = _clients.get(deactivated)
    if client and client.is_connected():
        logger.info(f"Disconnecting client for mode: {deactivated}")
        try:
            await client.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting client for {deactivated}: {e}")

    # Stop managed processes in reverse order (MindServer first, then MC server)
    mgr_list = _process_managers.get(deactivated, [])
    for mgr in reversed(mgr_list):
        logger.info(f"Stopping managed process for mode {deactivated}: {mgr._name}")
        await mgr.stop()

    _state.mode_history.append(deactivated)
    _state.active_mode = None

    return json_response({
        "success": True,
        "deactivated_mode": deactivated,
        "message": f"🔚 {config.name if config else deactivated} 모드가 종료되었습니다."
    })


@mcp.tool()
def get_mode_context_prompt() -> str:
    """현재 활성화된 모드의 컨텍스트 프롬프트를 가져옵니다.

    모드가 활성화되어 있으면 해당 모드의 실시간 상태 정보가 포함된
    프롬프트를 반환합니다. LLM 시스템 프롬프트에 사용할 수 있습니다.
    """
    if not _state.active_mode:
        return json_response({
            "success": True,
            "has_mode": False,
            "prompt": "일반 대화 모드입니다. 사용자와 자연스럽게 대화하세요."
        })

    client = _clients.get(_state.active_mode)
    config = get_mode_config(_state.active_mode)

    if not client:
        return json_response({
            "success": False,
            "error": f"Client not found for mode: {_state.active_mode}"
        })

    context_prompt = client.get_context_prompt()

    return json_response({
        "success": True,
        "has_mode": True,
        "mode_id": _state.active_mode,
        "mode_name": config.name if config else _state.active_mode,
        "prompt": context_prompt,
        "exit_keywords": config.exit_keywords if config else []
    })


# ============================================================================
# Minecraft Mode Tools (Dynamic based on mode)
# ============================================================================

@mcp.tool()
async def minecraft_command(command: str) -> str:
    """마인크래프트에서 행동을 실행합니다. (마인크래프트 모드 필요)

    Args:
        command: 실행할 행동을 자연어로 입력 (예: "나무 캐줘", "집 지어줘")
    """
    if _state.active_mode != "minecraft":
        return json_response({
            "success": False,
            "error": "마인크래프트 모드가 활성화되지 않았습니다.",
            "hint": "activate_mode('minecraft')를 먼저 호출하세요.",
            "current_mode": _state.active_mode
        })

    mode = _modes.get("minecraft")
    if not mode:
        return json_response({
            "success": False,
            "error": "Minecraft mode not initialized"
        })

    result = await mode.send_command(command)
    return json_response(result)


@mcp.tool()
async def minecraft_status() -> str:
    """마인크래프트에서 현재 상태를 확인합니다. (마인크래프트 모드 필요)"""
    if _state.active_mode != "minecraft":
        return json_response({
            "success": False,
            "error": "마인크래프트 모드가 활성화되지 않았습니다.",
            "current_mode": _state.active_mode
        })

    mode = _modes.get("minecraft")
    if not mode:
        return json_response({
            "success": False,
            "error": "Minecraft mode not initialized"
        })

    result = await mode.get_bot_status()
    return json_response(result)


@mcp.tool()
async def minecraft_create_bot() -> str:
    """마인크래프트 세계에 접속합니다. (마인크래프트 모드 필요)"""
    if _state.active_mode != "minecraft":
        return json_response({
            "success": False,
            "error": "마인크래프트 모드가 활성화되지 않았습니다."
        })

    mode = _modes.get("minecraft")
    if not mode:
        return json_response({
            "success": False,
            "error": "Minecraft mode not initialized"
        })

    result = await mode.create_bot()
    return json_response(result)


# ============================================================================
# Resources
# ============================================================================

@mcp.resource("router://mode/current")
def current_mode_resource() -> str:
    """현재 활성화된 모드 정보"""
    if not _state.active_mode:
        return "현재 모드: 없음 (일반 대화)"

    config = get_mode_config(_state.active_mode)
    if config:
        return f"현재 모드: {config.name}\n{config.description}"
    return f"현재 모드: {_state.active_mode}"


# ============================================================================
# Graceful Shutdown
# ============================================================================

_shutdown_count = 0

def graceful_shutdown(signum, frame):
    """Signal handler for graceful shutdown."""
    global _shutdown_count
    _shutdown_count += 1

    if _shutdown_count >= 2:
        logger.info("\nForce shutdown!")
        import os
        os._exit(1)

    logger.info(f"\nShutting down... (Ctrl+C again to force)")
    shutdown_event.set()
    raise KeyboardInterrupt


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Router MCP Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"],
                        help="Transport mode: 'stdio' for subprocess, 'sse' for HTTP server")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    logger.info(f"Starting Router MCP Server (transport: {args.transport})")
    logger.info(f"Available modes: {list_modes()}")
    logger.info(f"Available clients: {list_client_types()}")

    # Initialize clients and modes
    init_clients()

    # Start client connection threads only for non-managed modes
    for mode_id in _clients.keys():
        if mode_id in _process_managers:
            logger.info(f"Skipping auto-connect for managed mode: {mode_id}")
            continue
        thread = threading.Thread(
            target=run_client_loop,
            args=(mode_id,),
            name=f"Client-{mode_id}"
        )
        thread.start()
        _client_threads[mode_id] = thread

    # Wait for initial connections (only non-managed clients)
    import time
    non_managed_clients = {
        mid: c for mid, c in _clients.items() if mid not in _process_managers
    }
    if non_managed_clients:
        for _ in range(10):
            if shutdown_event.is_set():
                break
            time.sleep(0.5)
            if any(c.is_connected() for c in non_managed_clients.values()):
                logger.info("Client connected!")
                break
    else:
        logger.info("All modes are process-managed — skipping initial connection wait")

    try:
        if args.transport == "sse":
            # Run MCP server with SSE transport (for Dify MCP connection)
            logger.info(f"MCP SSE endpoint: http://{args.host}:{args.port}/sse")
            logger.info(f"REST API endpoints available at http://{args.host}:{args.port}/api/")
            logger.info(f"  - GET  /api/mode-prompt  : Get current mode prompt")
            logger.info(f"  - POST /api/mode/activate : Activate a mode")
            logger.info(f"  - POST /api/mode/deactivate : Deactivate current mode")
            logger.info(f"  - GET  /api/health : Health check")
            mcp.run(transport="sse", host=args.host, port=args.port)
        else:
            # Run MCP server with stdio transport (for subprocess communication)
            logger.info("Running in stdio mode for subprocess communication")
            mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Cleaning up...")
        shutdown_event.set()

        # Stop all managed processes (reverse order per mode)
        if _process_managers:
            import asyncio as _aio
            _cleanup_loop = _aio.new_event_loop()
            try:
                async def _stop_all_processes():
                    for mid, mgr_list in _process_managers.items():
                        for mgr in reversed(mgr_list):
                            logger.info(f"Stopping managed process: {mgr._name}")
                            await mgr.ensure_stopped()
                _cleanup_loop.run_until_complete(_stop_all_processes())
            finally:
                _cleanup_loop.close()

        # Wait for client threads
        for mode_id, thread in _client_threads.items():
            if thread.is_alive():
                thread.join(timeout=3)

        logger.info("Router MCP Server shutdown complete")
        sys.exit(0)


if __name__ == "__main__":
    main()
