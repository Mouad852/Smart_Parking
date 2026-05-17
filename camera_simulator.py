"""
camera_simulator.py
====================
Simulates multiple IP cameras by reading local MP4 files and streaming
Base64-encoded JPEG frames over WebSocket on ws://127.0.0.1:8765.

Each camera runs as an independent async task at its own FPS.

Usage:
    python camera_simulator.py

Dependencies:
    pip install websockets opencv-python
"""

import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("camera_simulator")

# ── Camera configuration ──────────────────────────────────────────────────────
# Each entry maps a camera ID to its video source and broadcast FPS.
# Place your MP4 files in the input_video/ directory (or adjust paths).
CAMERA_CONFIG: list[dict] = [
    {"camera_id": "cam1", "location": "Zone A — Avenue Hassan II",   "video": "input_video/parking_space.mp4", "fps": 10},
    {"camera_id": "cam2", "location": "Zone B — Rue Ibn Sina",       "video": "input_video/parking_space.mp4", "fps": 10},
    {"camera_id": "cam3", "location": "Zone C — Boulevard Mohammed V","video": "input_video/parking_space.mp4", "fps": 10},
    {"camera_id": "cam4", "location": "Zone D — Rue Patrice Lumumba","video": "input_video/parking_space.mp4", "fps": 10},
    {"camera_id": "cam5", "location": "Zone E — Avenue Annakhil",    "video": "input_video/parking_space.mp4", "fps": 10},
]

HOST = "127.0.0.1"
PORT = 8765
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 75          # 0-100; lower = smaller payload
# ─────────────────────────────────────────────────────────────────────────────

# Shared set of connected WebSocket clients (thread-safe via asyncio event loop)
CONNECTED: set[WebSocketServerProtocol] = set()


def encode_frame(frame) -> str:
    """Resize frame to target resolution and encode as Base64 JPEG string."""
    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def camera_worker(cfg: dict) -> None:
    """
    Continuously reads frames from an MP4 file and broadcasts JSON packets
    to all connected WebSocket clients. Loops the video on EOF.
    """
    cam_id   = cfg["camera_id"]
    location = cfg["location"]
    video    = cfg["video"]
    target_fps = cfg["fps"]
    interval = 1.0 / target_fps

    if not os.path.exists(video):
        log.error("[%s] Video file not found: %s — skipping camera.", cam_id, video)
        return

    log.info("[%s] Starting — source: %s @ %d fps", cam_id, video, target_fps)
    frame_index = 0

    while True:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            log.error("[%s] Cannot open video: %s — retrying in 5 s", cam_id, video)
            await asyncio.sleep(5)
            continue

        while True:
            t0 = time.monotonic()
            ret, frame = cap.read()

            if not ret:
                # End of video — loop back
                log.debug("[%s] EOF — looping video", cam_id)
                break

            if not CONNECTED:
                # No clients connected; skip encoding to save CPU
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
                continue

            try:
                b64_image = encode_frame(frame)
            except Exception as exc:
                log.warning("[%s] Frame encode error: %s", cam_id, exc)
                continue

            payload = json.dumps({
                "camera_id":   cam_id,
                "location":    location,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "frame_index": frame_index,
                "image":       b64_image,
            })

            # Broadcast to all connected clients (fire-and-forget per client)
            dead: set[WebSocketServerProtocol] = set()
            for ws in list(CONNECTED):
                try:
                    await ws.send(payload)
                except websockets.exceptions.ConnectionClosed:
                    dead.add(ws)
            CONNECTED.difference_update(dead)

            frame_index += 1
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

        cap.release()


async def on_connect(ws: WebSocketServerProtocol) -> None:
    """Register / deregister clients and keep the connection alive."""
    CONNECTED.add(ws)
    log.info("Client connected: %s  (total: %d)", ws.remote_address, len(CONNECTED))
    try:
        await ws.wait_closed()
    finally:
        CONNECTED.discard(ws)
        log.info("Client disconnected: %s  (total: %d)", ws.remote_address, len(CONNECTED))


async def main() -> None:
    # Create input_video directory if absent so the error message is friendlier
    Path("input_video").mkdir(exist_ok=True)

    # Launch one camera worker task per configured camera
    tasks = [asyncio.create_task(camera_worker(cfg)) for cfg in CAMERA_CONFIG]

    async with websockets.serve(on_connect, HOST, PORT, max_size=10 * 1024 * 1024):
        log.info("WebSocket server listening on ws://%s:%d", HOST, PORT)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Simulator stopped.")