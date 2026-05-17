# test_client.py — run this to confirm simulator is working
import asyncio, websockets, json, base64, numpy as np, cv2

async def test():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "handshake":
                print("[OK] Connected. Cameras:", msg["cameras"])
                continue
            img = cv2.imdecode(
                      np.frombuffer(base64.b64decode(msg["image"]), np.uint8),
                      cv2.IMREAD_COLOR)
            print(f"[{msg['camera_id']}] frame={msg['frame_index']} "
                  f"size={img.shape}")
            cv2.imshow(msg["camera_id"], img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

asyncio.run(test())