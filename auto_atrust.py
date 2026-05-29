#!/usr/bin/env python3
"""Periodic watchdog that logs the aTrust session back in when it drops.

Pipeline (per tick):
  1. Ensure the SSH tunnel `localhost:5901 -> remote:5901` is up.
  2. Pull a frame from the VNC server (TigerVNC, VncAuth password).
  3. Decide state from the "上网账号" login-form title, NOT from the Log In
     button: the title is the one element always visible when logged out and
     never hidden by the "you have been logged out" dialog. If it is absent we
     are logged in (healthy) and stop here.
  4. If logged out: dismiss the logout dialog (click OK), click the now-visible
     Log In button, then re-capture to confirm the form is gone.
  5. If we are logged out but cannot complete the login (button not found, or
     the form is still up afterwards), warn loudly and save the frame to
     captures/anomaly.png instead of silently reporting "healthy".
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
LOGIN_BUTTON_TEMPLATE = CAPTURES / "login_template.png"     # blue "Log In" button
LOGOUT_OK_TEMPLATE = CAPTURES / "logout_ok_template.png"    # "OK" on the "logged out" dialog
LOGIN_FORM_TEMPLATE = CAPTURES / "login_form_template.png"  # "上网账号" title; the logged-out anchor
LAST_FRAME_PATH = CAPTURES / "last_frame.png"              # most recent capture, kept for debugging
ANOMALY_FRAME_PATH = CAPTURES / "anomaly.png"             # frame saved when login can't complete

load_dotenv(ROOT / ".env")

SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_PORT = int(os.environ.get("SSH_PORT", "22"))
VNC_PORT = int(os.environ.get("VNC_PORT", "5901"))
VNC_PASSWORD = os.environ.get("VNC_PASSWORD", "")
MATCH_THRESHOLD = 0.85
MATCH_SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)  # template scales tried per match; best score wins
DEFAULT_INTERVAL = 300  # seconds between checks
LOGIN_SETTLE = 5  # seconds to wait after clicking Log In before verifying the result
SSH_TUNNEL_TIMEOUT = 30  # seconds to wait for ssh -f to authenticate and fork
SSH_CMD_TIMEOUT = 15  # seconds for any single ssh remote-command invocation
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


def ssh_remote(remote_cmd: str) -> bool:
    """Run ``remote_cmd`` over SSH on the configured host. Returns False on failure."""
    cmd = [
        "ssh",
        "-p", str(SSH_PORT),
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        SSH_HOST,
        remote_cmd,
    ]
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=SSH_CMD_TIMEOUT)
        log.info("ssh: ran %r in %.1fs", remote_cmd, time.monotonic() - t0)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("ssh: %r failed (rc=%d): %s", remote_cmd, exc.returncode,
                    exc.stderr.decode(errors="replace").strip())
        return False
    except subprocess.TimeoutExpired:
        log.warning("ssh: %r timed out after %ds", remote_cmd, SSH_CMD_TIMEOUT)
        return False


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


def best_match(img, template) -> tuple[float, int, int]:
    """Best multi-scale template match. Returns (score, center_x, center_y).

    The remote framebuffer is normally a fixed 1112x620, but matching across a
    few scales is cheap insurance against minor render/DPI drift between the
    frame a template was cut from and the live one.
    """
    H, W = img.shape[:2]
    best = (0.0, -1, -1)
    for scale in MATCH_SCALES:
        if scale == 1.0:
            t = template
        else:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            t = cv2.resize(template, None, fx=scale, fy=scale, interpolation=interp)
        th, tw = t.shape[:2]
        if th >= H or tw >= W:
            continue
        result = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, top_left = cv2.minMaxLoc(result)
        if score > best[0]:
            best = (float(score), top_left[0] + tw // 2, top_left[1] + th // 2)
    return best


def capture_frame():
    """Capture one VNC frame to LAST_FRAME_PATH; return it as a BGR ndarray or None."""
    if not vnc_capture(LAST_FRAME_PATH):
        return None
    img = cv2.imread(str(LAST_FRAME_PATH))
    if img is None:
        log.warning("vnc: captured frame unreadable at %s", LAST_FRAME_PATH)
    return img


def save_anomaly() -> None:
    """Preserve the frame that triggered a warning so later healthy ticks don't overwrite it."""
    try:
        shutil.copyfile(LAST_FRAME_PATH, ANOMALY_FRAME_PATH)
        log.warning("saved offending frame to %s", ANOMALY_FRAME_PATH)
    except OSError as exc:
        log.warning("could not save anomaly frame: %s", exc)


def tick(templates: dict) -> None:
    tick_t0 = time.monotonic()
    log.info("tick: start (threshold=%.2f)", MATCH_THRESHOLD)
    try:
        ensure_tunnel()
    except Exception as exc:
        log.error("tick: tunnel setup failed: %s", exc)
        return

    # aTrust spawns an `xmessage http://zhfw.zju.edu.cn/` popup after each
    # login. Those stack and cover the window, so clear any leftovers before we
    # look at the screen. Match by process name (not `-f`) so pkill doesn't
    # self-kill via the parent shell's command line.
    ssh_remote("pkill xmessage || true")

    img = capture_frame()
    if img is None:
        log.info("tick: aborting (no frame), elapsed %.1fs", time.monotonic() - tick_t0)
        return

    # The "上网账号" login-form title is the anchor for the whole decision: it is
    # present whenever the session is logged out and is NEVER hidden by the
    # logout dialog (which sits lower on screen). Its ABSENCE is what means
    # "logged in" — NOT the absence of the Log In button, which the dialog can
    # occlude and which a flaky/partial capture can drop below threshold. The
    # old code treated "no Log In button" as healthy and so reported a logged-out
    # session as fine.
    form_score, _, _ = best_match(img, templates["form"])
    if form_score < MATCH_THRESHOLD:
        log.info("tick: login form absent (%.3f) — session is logged in, healthy", form_score)
        log.info("tick: done in %.1fs", time.monotonic() - tick_t0)
        return

    log.warning("tick: login form present (%.3f) — session is LOGGED OUT, attempting login",
                form_score)

    # Step 1: the "You have been logged out" dialog covers the Log In button, so
    # dismiss it first to make the button clickable.
    ok_template = templates.get("ok")
    if ok_template is not None:
        ok_score, ok_x, ok_y = best_match(img, ok_template)
        log.info("match: logout-dialog OK score=%.3f at (%d, %d)", ok_score, ok_x, ok_y)
        if ok_score >= MATCH_THRESHOLD:
            log.info("match: dismissing logout dialog — clicking OK (%d, %d)", ok_x, ok_y)
            if vnc_click(ok_x, ok_y):
                time.sleep(1.0)
                img = capture_frame()
                if img is None:
                    log.info("tick: aborting (no frame after OK), elapsed %.1fs",
                             time.monotonic() - tick_t0)
                    return

    # Step 2: click the now-visible blue Log In button.
    btn_score, btn_x, btn_y = best_match(img, templates["login"])
    log.info("match: Log In score=%.3f at (%d, %d)", btn_score, btn_x, btn_y)
    if btn_score < MATCH_THRESHOLD:
        log.warning("tick: logged out but Log In button not found (%.3f) — cannot log in",
                    btn_score)
        save_anomaly()
        log.info("tick: done in %.1fs", time.monotonic() - tick_t0)
        return
    log.info("match: clicking Log In (%d, %d)", btn_x, btn_y)
    vnc_click(btn_x, btn_y)

    # Step 3: confirm the login took — the form title must disappear.
    time.sleep(LOGIN_SETTLE)
    img = capture_frame()
    if img is None:
        log.warning("tick: clicked Log In but could not re-capture to verify")
        log.info("tick: done in %.1fs", time.monotonic() - tick_t0)
        return
    form_after, _, _ = best_match(img, templates["form"])
    if form_after < MATCH_THRESHOLD:
        log.info("tick: login confirmed (form gone, %.3f) — session is back online", form_after)
    else:
        log.warning("tick: login form still present (%.3f) after clicking — login may have failed",
                    form_after)
        save_anomaly()
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

    # Required templates: the form-title anchor and the Log In button.
    templates: dict = {}
    for key, path in (("form", LOGIN_FORM_TEMPLATE), ("login", LOGIN_BUTTON_TEMPLATE)):
        if not path.exists():
            log.error("required template missing: %s", path)
            return 1
        templates[key] = cv2.imread(str(path))
        if templates[key] is None:
            log.error("required template unreadable: %s", path)
            return 1
    # Optional: the logout-dialog OK button (dialog handling is skipped if absent).
    if LOGOUT_OK_TEMPLATE.exists():
        templates["ok"] = cv2.imread(str(LOGOUT_OK_TEMPLATE))
    else:
        log.warning("logout OK template missing: %s (logout-dialog handling disabled)",
                    LOGOUT_OK_TEMPLATE)

    CAPTURES.mkdir(exist_ok=True)
    if args.once:
        tick(templates)
        return 0
    log.info("auto_atrust started, interval=%ds (Ctrl+C to stop)", args.interval)
    try:
        while True:
            tick(templates)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
