"""
parking_api.py
===============
FastAPI bridge between camera_simulator.py and the Leaflet.js dashboard.

Startup sequence
----------------
1. Connects (with auto-reconnect) to the WebSocket camera simulator on
   ws://127.0.0.1:8765.
2. Decodes incoming Base64 JPEG frames.
3. Runs YOLOv8 inference + parking-slot polygon mapping (via utilis.py).
4. Stores the latest annotated frame and occupancy stats per camera.
5. Pushes live JSON status updates to all dashboard WebSocket clients (/ws).

HTTP endpoints
--------------
GET  /               — health check
GET  /status         — current occupancy of all zones (JSON)
GET  /status/{cam}   — occupancy of one zone (JSON)
GET  /frame/{cam}    — latest annotated JPEG (binary, browser-cacheable)
WS   /ws             — real-time occupancy push to dashboard clients

Dependencies:
    pip install fastapi uvicorn[standard] websockets ultralytics opencv-python numpy
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from utilis import YOLO_Detection, draw_summary_bar, drawPolygons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("parking_api")

# ── Configuration ─────────────────────────────────────────────────────────────
SIMULATOR_WS_URL = "ws://127.0.0.1:8765"
RECONNECT_DELAY  = 2          # seconds between reconnect attempts
YOLO_MODEL_PATH  = "yolov8n.pt"  # auto-downloaded on first run
YOLO_CONF        = 0.25
ROI_FILE         = "Space_ROIs"  # pickle file with polygon coordinates

# Geographic metadata for each camera (used by the dashboard)
CAMERAS_METADATA: dict[str, dict] = {
    "cam1": {"lat": 33.9911, "lng": -6.8550, "label": "Zone A — Avenue Hassan II"},
    "cam2": {"lat": 33.9880, "lng": -6.8510, "label": "Zone B — Rue Ibn Sina"},
    "cam3": {"lat": 33.9860, "lng": -6.8570, "label": "Zone C — Boulevard Mohammed V"},
    "cam4": {"lat": 33.9930, "lng": -6.8490, "label": "Zone D — Rue Patrice Lumumba"},
    "cam5": {"lat": 33.9845, "lng": -6.8530, "label": "Zone E — Avenue Annakhil"},
}
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared in-memory state ────────────────────────────────────────────────────
# latest annotated JPEG bytes per camera
latest_frames: dict[str, bytes] = {}

# latest occupancy stats per camera
latest_status: dict[str, dict[str, Any]] = {
    cam_id: {
        "cam_id":   cam_id,
        "label":    meta["label"],
        "lat":      meta["lat"],
        "lng":      meta["lng"],
        "free":     0,
        "occupied": 0,
        "total":    0,
        "status":   "unknown",   # "free" | "warning" | "full" | "unknown"
    }
    for cam_id, meta in CAMERAS_METADATA.items()
}

# Connected dashboard WebSocket clients
dashboard_clients: set[WebSocket] = set()
# ─────────────────────────────────────────────────────────────────────────────


# ── YOLO model (loaded once at startup) ───────────────────────────────────────
model = None

def load_model():
    global model
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
        log.info("YOLOv8 model loaded: %s", YOLO_MODEL_PATH)
    except Exception as exc:
        log.error("Failed to load YOLO model: %s", exc)
        model = None


# ── Parking slot ROIs ─────────────────────────────────────────────────────────
slot_polygons: list = []

def load_rois():
    global slot_polygons
    roi_path = Path(ROI_FILE)
    if roi_path.exists():
        try:
            with open(roi_path, "rb") as f:
                slot_polygons = pickle.load(f)
            log.info("Loaded %d parking slot ROIs from %s", len(slot_polygons), ROI_FILE)
        except Exception as exc:
            log.warning("Could not load ROI file (%s): %s — falling back to empty list.", ROI_FILE, exc)
            slot_polygons = []
    else:
        log.warning("ROI file '%s' not found. Run the ROI picker tool to create it.", ROI_FILE)
        # Default: single full-frame polygon so the system still works for demo
        slot_polygons = [[[10, 10], [630, 10], [630, 470], [10, 470]]]


# ── Frame processing ──────────────────────────────────────────────────────────

def _compute_zone_status(free: int, total: int) -> str:
    if total == 0:
        return "unknown"
    if free == 0:
        return "full"
    if free <= 2:
        return "warning"
    return "free"


def process_frame(cam_id: str, b64_image: str) -> None:
    """
    Decode → infer → annotate → store the result for *cam_id*.
    Updates latest_frames and latest_status in-place.
    """
    try:
        raw = base64.b64decode(b64_image)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as exc:
        log.warning("[%s] Frame decode error: %s", cam_id, exc)
        return

    # Run YOLO (skip if model failed to load)
    boxes: list[list[int]] = []
    if model is not None:
        try:
            boxes = YOLO_Detection(model, frame, conf=YOLO_CONF)
        except Exception as exc:
            log.warning("[%s] YOLO inference error: %s", cam_id, exc)

    # Draw polygon overlays and count occupancy
    try:
        annotated, free, occupied = drawPolygons(frame, slot_polygons, boxes)
    except Exception as exc:
        log.warning("[%s] drawPolygons error: %s", cam_id, exc)
        annotated, free, occupied = frame, 0, 0

    total = free + occupied
    annotated = draw_summary_bar(annotated, free, occupied, cam_id)

    # Encode annotated frame as JPEG bytes
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        latest_frames[cam_id] = buf.tobytes()

    # Update status
    latest_status[cam_id].update({
        "free":     free,
        "occupied": occupied,
        "total":    total,
        "status":   _compute_zone_status(free, total),
    })


# ── WebSocket broadcaster ─────────────────────────────────────────────────────

async def broadcast_status() -> None:
    """Send the latest status snapshot to all connected dashboard clients."""
    if not dashboard_clients:
        return
    payload = json.dumps(list(latest_status.values()))
    dead: set[WebSocket] = set()
    for ws in list(dashboard_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    dashboard_clients.difference_update(dead)


# ── Camera simulator consumer ─────────────────────────────────────────────────

async def consume_simulator() -> None:
    """
    Connect to camera_simulator.py WebSocket, receive frames, and
    trigger processing. Reconnects automatically on failure.
    """
    while True:
        try:
            async with websockets.connect(
                SIMULATOR_WS_URL, max_size=20 * 1024 * 1024
            ) as ws:
                log.info("Connected to simulator at %s", SIMULATOR_WS_URL)
                async for raw_msg in ws:
                    try:
                        msg = json.loads(raw_msg)
                        cam_id = msg.get("camera_id", "unknown")
                        image  = msg.get("image", "")
                        if image:
                            process_frame(cam_id, image)
                            await broadcast_status()
                    except json.JSONDecodeError as exc:
                        log.warning("JSON decode error: %s", exc)
        except (websockets.exceptions.ConnectionClosed,
                ConnectionRefusedError,
                OSError) as exc:
            log.warning("Simulator connection lost (%s) — retrying in %ds", exc, RECONNECT_DELAY)
        except Exception as exc:
            log.error("Unexpected error in consume_simulator: %s", exc)

        await asyncio.sleep(RECONNECT_DELAY)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_rois()
    asyncio.create_task(consume_simulator())
    log.info("FastAPI bridge started — listening on http://0.0.0.0:8000")
    yield
    log.info("FastAPI bridge shutting down.")


app = FastAPI(
    title="Smart Parking API",
    description="Real-time parking occupancy bridge",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def health():
    return {"status": "ok", "cameras": list(CAMERAS_METADATA.keys())}


@app.get("/status", tags=["occupancy"])
async def get_all_status():
    """Return occupancy stats for all configured camera zones."""
    return list(latest_status.values())


@app.get("/status/{cam_id}", tags=["occupancy"])
async def get_cam_status(cam_id: str):
    """Return occupancy stats for a specific camera zone."""
    if cam_id not in latest_status:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found.")
    return latest_status[cam_id]


@app.get("/frame/{cam_id}", tags=["frames"])
async def get_frame(cam_id: str):
    """
    Stream the latest annotated JPEG frame for a camera.
    The response includes cache-control headers to allow browser polling.
    """
    if cam_id not in CAMERAS_METADATA:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found.")

    frame_bytes = latest_frames.get(cam_id)
    if not frame_bytes:
        # Return a small placeholder grey JPEG when no frame is available yet
        placeholder = np.full((120, 160, 3), 40, dtype=np.uint8)
        cv2.putText(placeholder, "No feed", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)
        _, buf = cv2.imencode(".jpg", placeholder)
        frame_bytes = buf.tobytes()

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# ─── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Dashboard clients connect here to receive real-time status pushes.
    On connect, immediately send the current state so the UI isn't blank.
    """
    await ws.accept()
    dashboard_clients.add(ws)
    log.info("Dashboard client connected. Total: %d", len(dashboard_clients))

    try:
        # Push current state immediately
        await ws.send_text(json.dumps(list(latest_status.values())))
        # Keep alive — we push from broadcast_status(), so just wait here
        while True:
            await ws.receive_text()   # absorbs keep-alive pings from browser
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_clients.discard(ws)
        log.info("Dashboard client disconnected. Total: %d", len(dashboard_clients))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("parking_api:app", host="0.0.0.0", port=8000, reload=False)