from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("kuka/models/nano_assemble.pt", task="segment")
model.to("cuda")

cap = cv2.VideoCapture("kuka/test_videos/2_h264.mp4")

COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def draw_pca_axes(image, polygon):
    """Calculate PCA from mask pixels and draw its two principal axes."""
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(binary_mask, [polygon], 255)

    y, x = np.nonzero(binary_mask)
    if len(x) < 3:
        return

    mask_points = np.column_stack((x, y)).astype(np.float32)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(mask_points, mean=None)
    center = mean[0]

    # Axis length is two standard deviations of the mask pixels.
    axis_1 = eigenvectors[0] * (2.0 * np.sqrt(eigenvalues[0, 0]))
    axis_2 = eigenvectors[1] * (2.0 * np.sqrt(eigenvalues[1, 0]))

    center_point = np.rint(center).astype(int)/image.shape[:2]
    axis_1_start = np.rint(center - axis_1).astype(int)
    axis_1_end = np.rint(center + axis_1).astype(int)
    axis_2_start = np.rint(center - axis_2).astype(int)
    axis_2_end = np.rint(center + axis_2).astype(int)


    cv2.line(image, axis_1_start, axis_1_end, (0, 0, 255), 3)
    # cv2.line(image, axis_2_start, axis_2_end, (255, 255, 0), 2)
    # cv2.circle(image, center_point*224, 5, (255, 255, 255), -1)

if not cap.isOpened():
    print("Jopa")
    raise NameError("AAA")

try:
    while True:
        state, frame = cap.read()

        if not state:
            break

        result = model.predict(frame, verbose=False)[0]
        overlay = frame.copy()
        pca_polygons = []

        if result.masks is not None:
            classes = result.boxes.cls.int().cpu().tolist()

            for polygon, class_id in zip(result.masks.xy, classes):
                points = np.asarray(polygon, dtype=np.int32)
                if len(points) < 3:
                    continue

                color = COLORS[class_id % len(COLORS)]
                cv2.fillPoly(overlay, [points], color)
                pca_polygons.append(points)
                # cv2.polylines(frame, [points], True, color, 2)

            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            for points in pca_polygons:
                draw_pca_axes(frame, points)

        frame = cv2.resize(frame, (448, 448))
        cv2.imshow("YOLO masks", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
