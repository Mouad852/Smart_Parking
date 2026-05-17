"""
roi_picker.py
==============
Interactive tool to define parking-slot polygon ROIs on a video frame.

Run this once to create (or update) the Space_ROIs pickle file that
parking_api.py loads at startup.

Controls
--------
  Left-click   — add a vertex to the current polygon
  Right-click  — finish the current polygon (must have ≥ 3 points)
  z            — undo last vertex in the current polygon
  c            — clear all polygons and start over
  s            — save all polygons to Space_ROIs and quit
  q / Esc      — quit without saving

Usage:
    python roi_picker.py
    # or point at a specific video:
    python roi_picker.py --video input_video/parking_space.mp4 --frame 120

Dependencies:
    pip install opencv-python numpy
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

OUTPUT_FILE = "Space_ROIs"

# State
polygons: list[list[list[int]]] = []    # finished polygons
current:  list[list[int]] = []          # points for the polygon being drawn
img_original = None                     # clean reference frame


def redraw(img_orig):
    display = img_orig.copy()
    overlay = display.copy()

    # Draw finished polygons
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.polylines(display, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        cx = int(pts[:, 0].mean()); cy = int(pts[:, 1].mean())
        cv2.putText(display, f"#{polygons.index(poly)+1}", (cx-8, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        overlay = display.copy()

    # Draw current in-progress polygon
    if current:
        for pt in current:
            cv2.circle(display, pt, 4, (0, 120, 255), -1)
        if len(current) > 1:
            cv2.polylines(display, [np.array(current, dtype=np.int32)],
                          isClosed=False, color=(0, 120, 255), thickness=1)

    # HUD
    info = [
        f"Polygons: {len(polygons)}  |  Current pts: {len(current)}",
        "Left-click: add pt | Right-click: finish | Z: undo | C: clear | S: save | Q: quit",
    ]
    for i, line in enumerate(info):
        cv2.putText(display, line, (10, 20 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,80), 1, cv2.LINE_AA)

    return display


def mouse_callback(event, x, y, flags, param):
    global current, polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current) >= 3:
            polygons.append(list(current))
            current = []
        else:
            print("[!] Need at least 3 points to close a polygon.")


def main():
    global current, polygons, img_original

    parser = argparse.ArgumentParser(description="Parking ROI Picker")
    parser.add_argument("--video", default="input_video/parking_space.mp4")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to use as background (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read frame from video.")
        return

    img_original = cv2.resize(frame, (640, 480))

    cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Picker", 960, 720)
    cv2.setMouseCallback("ROI Picker", mouse_callback)

    print(f"ROI Picker ready. Using frame {args.frame} from {args.video}")
    print("Instructions:")
    print("  Left-click  → add vertex")
    print("  Right-click → close polygon (≥3 pts)")
    print("  Z           → undo last vertex")
    print("  C           → clear all")
    print("  S           → save & quit")
    print("  Q / Esc     → quit without saving")

    while True:
        display = redraw(img_original)
        cv2.imshow("ROI Picker", display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):       # q or Esc
            print("Quitting without saving.")
            break
        elif key == ord('z'):           # undo
            if current:
                current.pop()
        elif key == ord('c'):           # clear
            polygons = []
            current  = []
        elif key == ord('s'):           # save
            if current and len(current) >= 3:
                polygons.append(list(current))
                current = []
            if polygons:
                with open(OUTPUT_FILE, "wb") as f:
                    pickle.dump(polygons, f)
                print(f"[OK] Saved {len(polygons)} polygon(s) to '{OUTPUT_FILE}'")
            else:
                print("[!] No polygons to save.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()