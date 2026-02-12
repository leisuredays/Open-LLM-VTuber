"""Subprocess process manager - starts/stops managed processes for mode activation."""

import asyncio
import logging
import os
import shutil
import signal
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("router-mcp.process-manager")


@dataclass
class LaunchCommand:
    """A command to run on mode activation (fire-and-forget)."""

    type: str  # "windows_shortcut" or "shell"
    path: str = ""
    command: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "LaunchCommand":
        return cls(
            type=data.get("type", "shell"),
            path=data.get("path", ""),
            command=data.get("command", ""),
        )


@dataclass
class ProcessConfig:
    """Configuration for a managed subprocess.

    Supports two styles:
    - Generic: set ``command`` to a list like ["java", "-jar", "server.jar"]
    - Legacy (MindServer): set ``node_binary`` + ``script``
    """

    enabled: bool = True
    name: str = ""
    command: list[str] = field(default_factory=list)
    working_dir: str = ""
    host: str = "localhost"
    port: int = 0
    startup_timeout_seconds: int = 30
    shutdown_timeout_seconds: int = 10
    launch_on_activate: list[LaunchCommand] = field(default_factory=list)

    # Legacy fields (used when ``command`` is empty)
    node_binary: str = ""
    script: str = ""

    def build_command(self) -> list[str]:
        """Return the command to execute."""
        if self.command:
            return list(self.command)
        if self.node_binary and self.script:
            return [self.node_binary, self.script]
        return []

    @classmethod
    def from_dict(cls, data: dict, name: str = "") -> "ProcessConfig":
        launch_cmds = [
            LaunchCommand.from_dict(item)
            for item in data.get("launch_on_activate", [])
        ]
        return cls(
            enabled=data.get("enabled", True),
            name=name or data.get("name", ""),
            command=data.get("command", []),
            working_dir=data.get("working_dir", ""),
            host=data.get("host", "localhost"),
            port=data.get("port", 0),
            startup_timeout_seconds=data.get("startup_timeout_seconds", 30),
            shutdown_timeout_seconds=data.get("shutdown_timeout_seconds", 10),
            launch_on_activate=launch_cmds,
            node_binary=data.get("node_binary", ""),
            script=data.get("script", ""),
        )


def _is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


class MindServerProcessManager:
    """Manages a subprocess lifecycle with port-readiness detection.

    - start(): launches the process and waits for the configured port to open
    - stop(): gracefully terminates the process (SIGTERM -> wait -> SIGKILL)
    - is_running: checks if the managed process is alive
    - Skips launching if the port is already occupied (external process)
    """

    def __init__(self, config: ProcessConfig, host: str = "", port: int = 0):
        self._config = config
        # Constructor args override config values (backward compat)
        self._host = host or config.host or "localhost"
        self._port = port or config.port
        self._name = config.name or "process"
        self._process: Optional[subprocess.Popen] = None
        self._externally_managed = False

    @property
    def is_running(self) -> bool:
        """True if the managed subprocess is still alive."""
        if self._process is not None:
            return self._process.poll() is None
        return False

    @property
    def is_port_available(self) -> bool:
        """True if the configured port is open (managed or external)."""
        if self._port <= 0:
            return False
        return _is_port_open(self._host, self._port)

    async def start(self) -> bool:
        """Start the process and wait for the port to open.

        Returns True on success, False on failure.
        If the port is already open (external process), skips launching.
        """
        if not self._config.enabled:
            logger.info("[%s] Process management is disabled", self._name)
            return False

        # Already running from a previous start()
        if self.is_running:
            logger.info("[%s] Process already running (pid=%d)", self._name, self._process.pid)
            return True

        # Port already occupied by an external process
        if self._port > 0 and self.is_port_available:
            logger.info(
                "[%s] Port %d already open — assuming external process",
                self._name,
                self._port,
            )
            self._externally_managed = True
            return True

        self._externally_managed = False

        working_dir = Path(self._config.working_dir)
        if not working_dir.is_dir():
            logger.error("[%s] Working directory does not exist: %s", self._name, working_dir)
            return False

        cmd = self._config.build_command()
        if not cmd:
            logger.error("[%s] No command configured", self._name)
            return False

        logger.info("[%s] Starting: %s (cwd=%s)", self._name, " ".join(cmd), working_dir)

        try:
            log_path = working_dir / f"{self._name}.log"
            self._log_file = open(log_path, "w")

            # Build env: ensure the node binary's directory is first in PATH
            # so child processes (e.g. AgentProcess spawning 'node') use the
            # same version as the parent.
            env = os.environ.copy()
            cmd_bin = Path(cmd[0])
            if cmd_bin.is_absolute() and cmd_bin.exists():
                env["PATH"] = str(cmd_bin.parent) + os.pathsep + env.get("PATH", "")

            self._process = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=env,
            )
        except Exception as e:
            logger.error("[%s] Failed to start process: %s", self._name, e)
            self._process = None
            return False

        logger.info("[%s] Process started (pid=%d)", self._name, self._process.pid)

        # Wait for port to open (if port is configured)
        if self._port > 0:
            timeout = self._config.startup_timeout_seconds
            ok = await self._wait_for_port(timeout)
            if not ok:
                logger.error(
                    "[%s] Port %d not ready within %ds — killing process",
                    self._name,
                    self._port,
                    timeout,
                )
                await self.stop()
                return False
            logger.info("[%s] Ready on port %d", self._name, self._port)

        return True

    async def stop(self) -> None:
        """Stop the managed process.

        Sends SIGTERM to the process group, waits for shutdown, then SIGKILL if needed.
        Does nothing if the process was externally managed.
        """
        if self._externally_managed:
            logger.info("[%s] Externally managed — skipping stop", self._name)
            self._externally_managed = False
            return

        proc = self._process
        if proc is None or proc.poll() is not None:
            logger.info("[%s] Process is not running — nothing to stop", self._name)
            self._process = None
            return

        pid = proc.pid
        pgid = None
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pass

        logger.info("[%s] Sending SIGTERM (pid=%d, pgid=%s)", self._name, pid, pgid)
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except OSError as e:
            logger.warning("[%s] Error sending SIGTERM: %s", self._name, e)

        timeout = self._config.shutdown_timeout_seconds
        try:
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, proc.wait),
                timeout=timeout,
            )
            logger.info("[%s] Exited gracefully (code=%d)", self._name, proc.returncode)
        except asyncio.TimeoutError:
            logger.warning("[%s] Did not exit within %ds — sending SIGKILL", self._name, timeout)
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=5)
            except Exception as e:
                logger.error("[%s] Failed to kill: %s", self._name, e)

        self._process = None

    async def ensure_stopped(self) -> None:
        """Ensure the process is stopped. Safe to call multiple times."""
        await self.stop()

    def run_launch_commands(self) -> None:
        """Execute fire-and-forget launch commands (e.g. Windows shortcuts).

        These are best-effort — failures are logged but don't block activation.
        """
        for cmd in self._config.launch_on_activate:
            try:
                if cmd.type == "windows_shortcut":
                    self._launch_windows_shortcut(cmd.path)
                elif cmd.type == "shell":
                    self._launch_shell_command(cmd.command)
                else:
                    logger.warning("Unknown launch command type: %s", cmd.type)
            except Exception as e:
                logger.error("Failed to run launch command (%s): %s", cmd.type, e)

    @staticmethod
    def _launch_windows_shortcut(win_path: str) -> None:
        """Launch a Windows shortcut (.lnk) from WSL2 via cmd.exe."""
        cmd_exe = shutil.which("cmd.exe")
        if not cmd_exe:
            logger.warning("cmd.exe not found — cannot launch Windows shortcut")
            return

        logger.info("Launching Windows shortcut: %s", win_path)
        subprocess.Popen(
            [cmd_exe, "/c", "start", "", win_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _launch_shell_command(command: str) -> None:
        """Launch a shell command (fire-and-forget)."""
        if not command:
            return
        logger.info("Launching shell command: %s", command)
        subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def _wait_for_port(self, timeout: int) -> bool:
        """Poll until the port is open or timeout is reached."""
        loop = asyncio.get_event_loop()
        elapsed = 0.0
        interval = 0.5

        while elapsed < timeout:
            if self._process is not None and self._process.poll() is not None:
                logger.error(
                    "[%s] Process exited prematurely (code=%d)",
                    self._name,
                    self._process.returncode,
                )
                return False

            open_ = await loop.run_in_executor(
                None, _is_port_open, self._host, self._port
            )
            if open_:
                return True

            await asyncio.sleep(interval)
            elapsed += interval

        return False
