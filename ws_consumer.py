# ws_consumer.py
import asyncio, websockets, json, base64, pickle, cv2
import numpy as np
from utilis import YOLO_Detection, drawPolygons
from ultralytics import YOLO

WS_URL    = "ws://127.0.0.1:8765"
MODEL     = YOLO("yolov8n.pt")
CONF      = 0.25

# Load your existing slot polygons
with open("Space_ROIs", "rb") as f:
    slot_polygons = pickle.load(f)

async def consume():
    async with websockets.connect(WS_URL) as ws:
        print("[CONSUMER] Connected to simulator")
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "handshake":
                print(f"[CONSUMER] Cameras: {msg['cameras']}")
                continue

            cam_id = msg["camera_id"]

            # Decode Base64 → numpy frame
            img_bytes = base64.b64decode(msg["image"])
            frame     = cv2.imdecode(
                            np.frombuffer(img_bytes, np.uint8),
                            cv2.IMREAD_COLOR)

            # Your existing pipeline
            boxes, class_ids, names = YOLO_Detection(MODEL, frame, CONF)
            
            # Convert boxes to center points
            detection_points = []
            for box in boxes:
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Use three points along the centerline of the bounding box
                # This is highly robust for perspective distortion and partial visibility
                detection_points.append((center_x, center_y))                       # Center
                detection_points.append((center_x, int(y1 + (y2 - y1) * 0.75)))     # Lower-middle
                detection_points.append((center_x, int(y1 + (y2 - y1) * 0.90)))     # Bottom-middle

            frame, occupied_count = drawPolygons(frame, slot_polygons, detection_points=detection_points)

            # Show with camera label
            cv2.putText(frame, f"Camera: {cam_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow(f"Smart Parking — {cam_id}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

asyncio.run(consume())