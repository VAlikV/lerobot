import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(4)
if not cam.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

while 1:
    ok, frame = cam.read()
    if ok:
        plt.imshow(frame)
        plt.show()