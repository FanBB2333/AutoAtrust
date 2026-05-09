#!/usr/bin/env python3
"""Periodic watchdog that re-clicks the aTrust Log In button when the session drops.

Pipeline:
  1. Ensure the SSH tunnel `localhost:5901 -> remote:5901` is up.
  2. Pull a frame from the VNC server (TigerVNC, VncAuth password).
  3. Template-match against the captured "Log In" button.
  4. If found above threshold, send a VNC mouse click on its center.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "captures" / "login_template.png"
LOGOUT_OK_TEMPLATE_PATH = ROOT / "captures" / "logout_ok_template.png"

load_dotenv(ROOT / ".env")

SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_PORT = int(os.environ.get("SSH_PORT", "22"))
VNC_PORT = int(os.environ.get("VNC_PORT", "5901"))
VNC_PASSWORD = os.environ.get("VNC_PASSWORD", "")
MATCH_THRESHOLD = 0.85
DEFAULT_INTERVAL = 300  # seconds between checks
SSH_TUNNEL_TIMEOUT = 30  # seconds to wait for ssh -f to authenticate and fork
VNC_CMD_TIMEOUT = 15  # seconds for any single vncdo invocation

log = logging.getLogger("auto_atrust")


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def ensure_tunnel() -> None:
    """Open the SSH tunnel if nothing is already listening on VNC_PORT."""
    if port_listening(VNC_PORT):
        log.info("tunnel: port %d already listening, reusing", VNC_PORT)
        return
    log.info("tunnel: opening SSH tunnel %s -> localhost:%d (timeout %ds)",
             SSH_HOST, VNC_PORT, SSH_TUNNEL_TIMEOUT)
    cmd = [
        "ssh",
        "-p", str(SSH_PORT),
        "-L", f"{VNC_PORT}:localhost:{VNC_PORT}",
        "-N", "-f",
        "-o", "ServerAliveInterval=30",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",  # fail fast instead of prompting for password
        SSH_HOST,
    ]
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=SSH_TUNNEL_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ssh -f did not return within {SSH_TUNNEL_TIMEOUT}s")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise RuntimeError(f"ssh exited {exc.returncode}: {stderr}")
    log.info("tunnel: ssh forked to background after %.1fs, waiting for listener",
             time.monotonic() - t0)
    # ssh -f forks before the tunnel is fully ready, so wait for the listener.
    for _ in range(20):
        if port_listening(VNC_PORT):
            log.info("tunnel: listener up after %.1fs", time.monotonic() - t0)
            return
        time.sleep(0.25)
    raise RuntimeError("SSH tunnel did not come up in time")


def vnc_capture(out_path: Path) -> bool:
    """Capture one VNC frame to ``out_path``. Returns False on failure."""
    cmd = [
        "vncdo",
        "-p", VNC_PASSWORD,
        "-s", f"localhost::{VNC_PORT}",
        "capture", str(out_path),
    ]
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=VNC_CMD_TIMEOUT)
    except subprocess.CalledProcessError as exc:
        log.warning("vnc: capture failed after %.1fs: %s",
                    time.monotonic() - t0,
                    exc.stderr.decode(errors="replace").strip())
        return False
    except subprocess.TimeoutExpired:
        log.warning("vnc: capture timed out after %ds", VNC_CMD_TIMEOUT)
        return False
    size = out_path.stat().st_size if out_path.exists() else 0
    log.info("vnc: captured %d bytes in %.1fs", size, time.monotonic() - t0)
    return size > 0


def vnc_click(x: int, y: int) -> bool:
    cmd = [
        "vncdo",
        "-p", VNC_PASSWORD,
        "-s", f"localhost::{VNC_PORT}",
        "move", str(x), str(y),
        "click", "1",
    ]
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=VNC_CMD_TIMEOUT)
        log.info("vnc: click (%d, %d) sent in %.1fs", x, y, time.monotonic() - t0)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("vnc: click (%d, %d) failed: %s", x, y,
                    exc.stderr.decode(errors="replace").strip())
        return False
    except subprocess.TimeoutExpired:
        log.warning("vnc: click (%d, %d) timed out after %ds", x, y, VNC_CMD_TIMEOUT)
        return False


def match_template(frame_path: Path, template) -> tuple[float, int, int]:
    img = cv2.imread(str(frame_path))
    if img is None:
        return 0.0, -1, -1
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, top_left = cv2.minMaxLoc(result)
    h, w = template.shape[:2]
    return float(score), top_left[0] + w // 2, top_left[1] + h // 2


def tick(login_template, ok_template, scratch: Path) -> None:
    tick_t0 = time.monotonic()
    log.info("tick: start (threshold=%.2f)", MATCH_THRESHOLD)
    try:
        ensure_tunnel()
    except Exception as exc:
        log.error("tick: tunnel setup failed: %s", exc)
        return

    if not vnc_capture(scratch):
        log.info("tick: aborting (no frame), elapsed %.1fs", time.monotonic() - tick_t0)
        return

    if ok_template is not None:
        ok_score, ok_x, ok_y = match_template(scratch, ok_template)
        log.info("match: logout-dialog OK score=%.3f at (%d, %d)", ok_score, ok_x, ok_y)
        if ok_score >= MATCH_THRESHOLD:
            log.warning("match: logout dialog visible (%.3f) — clicking OK (%d, %d)",
                        ok_score, ok_x, ok_y)
            if vnc_click(ok_x, ok_y):
                log.info("tick: dialog dismissed, re-capturing in 1s")
                time.sleep(1.0)
                if not vnc_capture(scratch):
                    log.info("tick: aborting (no frame after OK), elapsed %.1fs",
                             time.monotonic() - tick_t0)
                    return

    score, cx, cy = match_template(scratch, login_template)
    log.info("match: Log In score=%.3f at (%d, %d)", score, cx, cy)
    if score >= MATCH_THRESHOLD:
        log.warning("match: Log In button visible (%.3f) — clicking (%d, %d)", score, cx, cy)
        vnc_click(cx, cy)
    else:
        log.info("tick: no Log In button — session looks healthy")
    log.info("tick: done in %.1fs", time.monotonic() - tick_t0)


def main() -> int:
    global MATCH_THRESHOLD
    parser = argparse.ArgumentParser(description="Auto-click aTrust Log In when session drops")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Seconds between checks (default %(default)s)")
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD,
                        help="Template match threshold (0..1)")
    parser.add_argument("--log-file", type=Path, default=ROOT / "auto_atrust.log",
                        help="Log file path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    MATCH_THRESHOLD = args.threshold

    handlers: list[logging.Handler] = [logging.FileHandler(args.log_file)]
    if sys.stdout.isatty() or args.verbose:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    missing = [k for k, v in (("SSH_HOST", SSH_HOST), ("VNC_PASSWORD", VNC_PASSWORD)) if not v]
    if missing:
        log.error("missing required env vars: %s — copy .env.sample to .env and fill in",
                  ", ".join(missing))
        return 2

    if not TEMPLATE_PATH.exists():
        log.error("template missing: %s", TEMPLATE_PATH)
        return 1
    template = cv2.imread(str(TEMPLATE_PATH))
    ok_template = None
    if LOGOUT_OK_TEMPLATE_PATH.exists():
        ok_template = cv2.imread(str(LOGOUT_OK_TEMPLATE_PATH))
    else:
        log.warning("logout OK template missing: %s (logout-dialog handling disabled)",
                    LOGOUT_OK_TEMPLATE_PATH)

    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td) / "frame.png"
        if args.once:
            tick(template, ok_template, scratch)
            return 0
        log.info("auto_atrust started, interval=%ds (Ctrl+C to stop)", args.interval)
        try:
            while True:
                tick(template, ok_template, scratch)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("stopped by user")
            return 0


if __name__ == "__main__":
    sys.exit(main())
