import cv2
import numpy as np
import time

cam_ids = [2,4,6]
caps = [cv2.VideoCapture(i) for i in cam_ids]

# Verification: Check if all cameras opened
for i, cap in enumerate(caps):
    time.sleep(1)
    if not cap.isOpened():
        print(f"Warning: Could not open camera {cam_ids[i]}")

print("Press 'q' to exit.")

while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"{fps} frames per second")

        if not ret:
            # Create a black placeholder if a frame is missing
            frame = np.zeros((640, 360, 3), dtype=np.uint8)
        
        cv2.imshow(str(cam_ids[i]), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
