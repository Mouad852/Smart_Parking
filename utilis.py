"""
utilis.py
==========
Core computer-vision pipeline for the Smart Parking system.

Public API
----------
YOLO_Detection(model, frame, conf=0.25)
    Run inference on a BGR frame and return filtered bounding boxes.

drawPolygons(frame, points_list, detection_points)
    Draw transparent slot overlays and label each slot FREE / OCCUPIED.

is_point_in_polygon(point, polygon)
    Pure-Python fallback for point-in-polygon (used when OpenCV unavailable).

Dependencies:
    pip install ultralytics opencv-python numpy
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── COCO class IDs we care about ──────────────────────────────────────────────
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES: set[int] = {2, 3, 5, 7}

# Colour constants (BGR)
COLOR_FREE     = (0, 200, 0)    # green
COLOR_OCCUPIED = (0, 0, 220)    # red
COLOR_TEXT     = (255, 255, 255)
OVERLAY_ALPHA  = 0.35           # transparency of the polygon fill


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

def YOLO_Detection(model, frame: np.ndarray, conf: float = 0.25) -> list[list[int]]:
    """
    Run YOLOv8 inference on *frame* and return a list of bounding boxes
    for vehicles only.

    Parameters
    ----------
    model : ultralytics.YOLO
        Pre-loaded YOLO model instance.
    frame : np.ndarray
        BGR image (H × W × 3).
    conf : float
        Minimum confidence threshold (default 0.25 for high sensitivity).

    Returns
    -------
    list of [x1, y1, x2, y2]
        Integer pixel coordinates for each detected vehicle box.
    """
    results = model(frame, conf=conf, verbose=False)
    boxes: list[list[int]] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append([x1, y1, x2, y2])

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_points_in_polygon(points: list[tuple[int, int]], polygon: np.ndarray) -> bool:
    """
    Return True if *any* of the probe points falls inside (or on the boundary
    of) the polygon, using cv2.pointPolygonTest.
    """
    for pt in points:
        # Returns positive if inside, zero if on boundary, negative if outside
        if cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0:
            return True
    return False


def _get_probe_points(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """
    Build the three strategic probe points for a bounding box.

    Rationale: a single centre-point fails under camera perspective distortion
    because the bounding box drifts above the actual ground position.  Using
    the lower portion of the box (where the tyres meet the tarmac) is robust.

      • centre        → (cx, cy)
      • 75 % down     → (cx, y1 + 0.75 × h)
      • 90 % down     → (cx, y1 + 0.90 × h)   ← most reliable
    """
    cx = (x1 + x2) // 2
    h  = y2 - y1
    return [
        (cx, (y1 + y2) // 2),
        (cx, int(y1 + h * 0.75)),
        (cx, int(y1 + h * 0.90)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def drawPolygons(
    frame: np.ndarray,
    points_list: Sequence[Sequence[Sequence[int]]],
    detection_points: list[list[int]],
) -> tuple[np.ndarray, int, int]:
    """
    Draw transparent parking-slot polygons on *frame* and determine whether
    each slot is FREE or OCCUPIED.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate (modified in-place on a copy).
    points_list : sequence of polygons
        Each element is a list of (x, y) vertices defining one parking slot.
    detection_points : list of [x1, y1, x2, y2]
        Bounding boxes returned by YOLO_Detection.

    Returns
    -------
    annotated_frame : np.ndarray
        A new BGR image with overlays drawn.
    free_count : int
    occupied_count : int
    """
    annotated = frame.copy()
    overlay   = frame.copy()

    free_count     = 0
    occupied_count = 0

    for slot_idx, raw_polygon in enumerate(points_list):
        polygon = np.array(raw_polygon, dtype=np.int32)

        # Determine occupancy
        occupied = False
        for box in detection_points:
            x1, y1, x2, y2 = box
            probes = _get_probe_points(x1, y1, x2, y2)
            if _check_points_in_polygon(probes, polygon):
                occupied = True
                break

        color = COLOR_OCCUPIED if occupied else COLOR_FREE
        if occupied:
            occupied_count += 1
        else:
            free_count += 1

        # Filled semi-transparent polygon
        cv2.fillPoly(overlay, [polygon], color)

        # Merge overlay with transparency
        cv2.addWeighted(overlay, OVERLAY_ALPHA, annotated, 1 - OVERLAY_ALPHA, 0, annotated)

        # Draw polygon outline
        cv2.polylines(annotated, [polygon], isClosed=True, color=color, thickness=2)

        # Slot label (number + status)
        centroid_x = int(polygon[:, 0].mean())
        centroid_y = int(polygon[:, 1].mean())
        status_text = f"#{slot_idx + 1} {'OCC' if occupied else 'FREE'}"

        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            annotated,
            (centroid_x - tw // 2 - 3, centroid_y - th - 4),
            (centroid_x + tw // 2 + 3, centroid_y + 4),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            status_text,
            (centroid_x - tw // 2, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        # Reset overlay for next slot
        overlay = annotated.copy()

    return annotated, free_count, occupied_count


# ─────────────────────────────────────────────────────────────────────────────
# Summary overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_summary_bar(frame: np.ndarray, free: int, occupied: int, cam_id: str) -> np.ndarray:
    """
    Draw a small status bar at the top-left of *frame* showing slot counts.
    Returns the annotated frame (modified in-place copy).
    """
    total = free + occupied
    text  = f"{cam_id}  |  Free: {free}/{total}  Occupied: {occupied}/{total}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pad = 8
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + pad * 2), (30, 30, 30), cv2.FILLED)
    cv2.putText(
        frame, text, (pad, th + pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return frame