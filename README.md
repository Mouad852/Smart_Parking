# Smart Parking — occupancy detection and a live map dashboard

This local computer-vision prototype detects parking-space occupancy with
**YOLOv8** and **OpenCV**, then streams telemetry over **WebSockets** to a
Leaflet-based map dashboard. It is intended for local demonstration and
experimentation, not presented as a deployed parking-management service.

<p align="center">
  <img src="ROI_Reference.png" alt="Parking-space ROI reference image" width="72%">
</p>

---

## 🔍 System Architecture Overview

The system consists of three decoupled processing layers:

```
  [1. VIRTUAL NETWORK CAMERAS] (WebSocket Server — Port 8765)
               │
               │  Streams Base64 JPEG frames + metadata
               ▼
  [2. FASTAPI BRIDGE API ENGINE] (FastAPI Client/Server — Port 8000)
    • Connects to Simulator WebSocket with auto-reconnection
    • Performs YOLOv8 Vehicle Inference (COCO class 2,3,5,7)
    • Map Space_ROIs polygons utilizing 3-Point Centerline Checking
    • Exposes REST endpoints & broadcasts WS updates (/ws)
               │
               │  Pushes JSON telemetry updates & hosts JPEG frames
               ▼
  [3. LEAFLET.JS WEB DASHBOARD] (Single-Page HTML Browser Client)
    • Renders custom dark-filtered OpenStreetMap tiles of Rabat, Morocco
    • Dynamic markers (🟢 Green: Available, 🟡 Yellow: Nearly Full, 🔴 Red: Full)
    • Bottom Live Feed strip with camera panning map-clicks
    • Seamless WebSockets with 2-second HTTP Polling Fallbacks
```

---

## 🛠️ Key System Features

*   **Real-time YOLOv8 Vehicle Classification:** Specifically targets passenger cars, trucks, buses, and motorcycles while filtering out pedestrians or background noise.
*   **3-Point Vertical Centerline Check:** Outperforms single-point checks under heavy camera perspective distortions by scanning coordinates along the tire/ground plane (`y1 + h * 0.75` and `y1 + h * 0.90`).
*   **Decoupled Async WS Streaming:** Simulated camera nodes run concurrently at target FPS levels, streaming raw JPEGs over local WebSockets.
*   **Premium Map UI:** Tailored single-file client interface that applies a native CSS filter to OpenStreetMap tiles, creating a state-of-the-art Dark Mode map entirely offline.
*   **Dynamic ROI Calibration Tool:** Interactive polygon picker that allows you to click, draw, undo, and save parking slots instantly into a serialized database.

---

## 📂 Codebase File Mapping

```
📂 Smart-Parking-Detector/
 ┣ 📂 input_video/               # Local MP4 video source files
 ┣ 📂 dashboard/
 ┃ ┗ 📜 index.html              # Premium dark-theme Leaflet.js Web Dashboard
 ┣ 📜 camera_simulator.py       # Decoupled multiple-camera WebSocket streamer
 ┣ 📜 parking_api.py            # FastAPI ASGI Backend & YOLO/Polygon analyzer
 ┣ 📜 utilis.py                 # Core OpenCV, Bounding Box, and draw utilities
 ┣ 📜 roi_picker.py             # Interactive GUI tool to draw/calibrate slot polygons
 ┣ 📜 Space_ROIs                # Serialized database containing parking coordinates
 ┣ 📜 requirements.txt          # Original dependencies
 ┣ 📜 requirements_api.txt      # Web & FastAPI dependencies
 ┗ 📜 README.md                 # Project documentation
```

---

## 🚀 Installation & Setup Instructions

### 1. Clone this repository and install dependencies
```bash
git clone https://github.com/Mouad852/Smart_Parking.git
cd Smart_Parking
```

### 2️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
```

---

## 🏃 Running the Live Demo

Open **two separate terminals** on your computer:

### Step 1: Start the camera streamer (Terminal 1)
```bash
python camera_simulator.py
```
*(Simulates 5 IP cameras from local videos and listens on `ws://127.0.0.1:8765`)*

### Step 2: Start the FastAPI backend (Terminal 2)
```bash
python parking_api.py
```
*(Starts the bridge on `http://127.0.0.1:8000`, loads YOLOv8, consumes simulated frames, processes boundaries, and hosts endpoints)*

### Step 3: Open the web dashboard
*   Simply open the `/dashboard` folder and **double-click `index.html`** in Chrome, Firefox, or Edge.
*   *Note: Ensure you have active internet connection when opening the HTML page so Leaflet can successfully pull map scripts from the CDN.*

---

## 🎨 Interactive Calibration: Drawing Custom Parking Slots

If you want to re-map or create custom coordinates on the video frame, launch the built-in ROI selector:
```bash
python roi_picker.py
```

### ⌨️ ROI Picker Keyboard Shortcuts:
| Action | Key / Control |
| :--- | :--- |
| **Add Vertex** | `Left-click` on the frame |
| **Finish Slot Polygon** | `Right-click` (requires $\ge$ 3 vertices) |
| **Undo Last Node** | Press `Z` |
| **Clear All Polygons** | Press `C` |
| **Save to Space_ROIs & Exit** | Press `S` |
| **Quit without Saving** | Press `Q` or `Esc` |

---

## Attribution and scope

The earlier setup instructions referenced
[`prajesdas/Smart-Parking-Space-Detector-using-YOLO-and-OpenCV`](https://github.com/prajesdas/Smart-Parking-Space-Detector-using-YOLO-and-OpenCV).
That link is retained as historical attribution; use the clone command above for
this repository. The project has a single authored initialization commit, and no
upstream license file is included here. Do not infer an MIT license or reuse
rights beyond the licenses of the dependencies and assets actually present.

## Known limitations

- Inputs are local simulated camera streams and local video files.
- Detection quality depends on the YOLO model, camera angle, selected ROI
  polygons, and the supplied video material.
- The Leaflet dashboard fetches map assets from a CDN, so it needs network
  access even though the application services run locally.
