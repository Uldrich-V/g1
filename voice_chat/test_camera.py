import asyncio
import cv2
from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod

ROBOT_IP = "192.168.123.161"

async def video_callback(track):
    print("Video track received")
    saved = False

    while True:
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            print(f"Frame received: {img.shape}")

            if not saved:
                cv2.imwrite("test_frame.jpg", img)
                print("Saved test_frame.jpg")
                saved = True

        except Exception as e:
            print(f"Video error: {e}")
            break

async def main():
    print(f"Connecting to {ROBOT_IP} ...")

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )

    await conn.connect()
    print("Connected!")

    conn.video.add_track_callback(video_callback)
    print("Video callback registered")

    conn.video.switchVideoChannel(True)
    print("Video channel enabled")

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
