import time
import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient


def main():
    # Ініціалізація каналу (як у тебе в інших прикладах)
    ChannelFactoryInitialize(0)

    front = FrontVideoClient()
    front.SetTimeout(3.0)
    front.Init()

    print("Camera initialized, waiting frame...")

    # чекаємо стабілізації
    for _ in range(10):
        code, data = front.GetImageSample()
        time.sleep(0.05)

    # беремо кілька кадрів і вибираємо найкращий
    best = None
    best_score = -1

    for i in range(10):
        code, data = front.GetImageSample()

        if code != 0:
            print("No frame")
            continue

        if isinstance(data, list):
            data = bytes(data)

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # різкість
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        print(f"Frame {i+1}: sharpness={score:.2f}")

        if score > best_score:
            best_score = score
            best = frame.copy()

        time.sleep(0.03)

    if best is None:
        print("Failed to capture image")
        return

    filename = f"g1_photo_{int(time.time())}.jpg"
    cv2.imwrite(filename, best, [cv2.IMWRITE_JPEG_QUALITY, 98])

    print("\n✅ Photo saved:", filename)
    print("Sharpness:", best_score)


if __name__ == "__main__":
    main()
