"""Example of using DataExtractor before feeding geometry to ACT."""

import cv2
import numpy as np

try:
    from .data_extractor import DataExtractor, feature_aggregation
except ImportError:
    # Allows running the example directly: python extended_act/extractor_test.py
    from data_extractor import DataExtractor, feature_aggregation


MODEL_PATH = "extended_act/models/nano_assemble.pt"
VIDEO_PATH = "kuka/test_videos/1_h264.mp4"
REQUIRED_CLASSES = ["Down", "Target"]


def draw_features(
    frame: np.ndarray,
    features: dict[str, np.ndarray],
) -> np.ndarray:
    """Draw masks, centers and principal axes for visual inspection."""
    output = frame.copy()
    overlay = frame.copy()
    height, width = frame.shape[:2]
    id_to_name = {
        class_id: class_name
        for class_name, class_id in DataExtractor.class_dict.items()
    }

    for index, mask in enumerate(features["masks"]):
        class_id = int(features["class_ids"][index])
        confidence = float(features["confidences"][index])
        color = (
            int((37 * class_id + 80) % 255),
            int((97 * class_id + 120) % 255),
            int((157 * class_id + 160) % 255),
        )
        overlay[mask] = color

        center_normalized = features["centers"][index]
        axis = features["principal_axes"][index]
        eigenvalue = features["eigenvalues"][index, 0]

        center = np.array(
            [
                center_normalized[0] * width,
                center_normalized[1] * height,
            ],
            dtype=np.float32,
        )
        half_length = 2.0 * np.sqrt(max(float(eigenvalue), 0.0))
        start = tuple(np.rint(center - axis * half_length).astype(int))
        end = tuple(np.rint(center + axis * half_length).astype(int))
        center_point = tuple(np.rint(center).astype(int))

        cv2.line(output, start, end, (0, 0, 255), 3)
        cv2.circle(output, center_point, 5, (255, 255, 255), -1)
        cv2.putText(
            output,
            f"{id_to_name.get(class_id, class_id)} {confidence:.2f}",
            (center_point[0] + 7, center_point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, 0.35, output, 0.65, 0.0)


def main() -> None:
    extractor = DataExtractor(
        model_path=MODEL_PATH,
        classes=REQUIRED_CLASSES,
        # DataExtractor automatically selects CUDA when it is available.
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Variable number of detections: geometry has shape [N, 6].
            features = extractor(frame)

            # One slot per requested class. Missing objects become zero vectors.
            fixed_geometry = feature_aggregation(features, REQUIRED_CLASSES)

            # This vector always has len(REQUIRED_CLASSES) * 6 elements and can
            # later be concatenated with other fixed-size ACT inputs.
            act_geometry = fixed_geometry.reshape(-1)

            print(act_geometry)

            visualization = draw_features(frame, features)
            cv2.putText(
                visualization,
                f"ACT geometry shape: {act_geometry.shape}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("DataExtractor example", visualization)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
