from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class DataExtractor:
    """Extract segmentation masks and simple 2D geometry from an RGB image.

    Coordinate convention:
        centers[:, 0] is x / image_width;
        centers[:, 1] is y / image_height;
        principal_axes are unit vectors in image (x, y) coordinates.

    The sign of a PCA axis is mathematically ambiguous. It is canonicalized here
    so that x is positive (or y is positive for a vertical axis), preventing
    otherwise arbitrary 180-degree flips between frames.
    """

    class_dict = {
        "Target": 0,
        "Helper": 1,
        "Down": 2,
        "PCB": 3,
        "Up": 4,
        "Gripper": 5,
    }

    def __init__(
        self,
        model_path: str | Path,
        classes: Sequence[str | int] | None = None,
        *,
        device: str | None = None,
        confidence: float = 0.25,
        image_size: int | tuple[int, int] | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in the [0, 1] range")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.image_size = image_size
        self.classes = self._resolve_classes(classes)

        self.model = YOLO(str(model_path), task="segment")
        self.model.to(self.device)

    def forward(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Run segmentation and return training-friendly NumPy arrays.

        Args:
            image: BGR or RGB uint8 image with shape ``[H, W, 3]``. Ultralytics
                interprets NumPy input as BGR, as returned by OpenCV.

        Returns:
            A dictionary containing:
                masks: bool array ``[N, H, W]``;
                class_ids: int64 array ``[N]``;
                confidences: float32 array ``[N]``;
                centers: normalized float32 array ``[N, 2]``;
                principal_axes: unit float32 array ``[N, 2]``;
                eigenvalues: PCA variances in pixel units, ``[N, 2]``;
                geometry: ``[center_x, center_y, axis_x, axis_y,
                    confidence, visibility]``, shape ``[N, 6]``.

            All arrays have a valid zero-length first dimension when no object
            is detected, which makes the output safe for preprocessing code.
        """
        self._validate_image(image)
        height, width = image.shape[:2]

        predict_kwargs: dict[str, Any] = {
            "source": image,
            "conf": self.confidence,
            "classes": self.classes or None,
            "device": self.device,
            "verbose": False,
        }
        if self.image_size is not None:
            predict_kwargs["imgsz"] = self.image_size

        result = self.model.predict(**predict_kwargs)[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return self._empty_result(height, width)

        masks = result.masks.data.detach().cpu().numpy()
        masks = np.stack(
            [
                cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                >= 0.5
                for mask in masks
            ],
            axis=0,
        )
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

        centers = np.zeros((len(masks), 2), dtype=np.float32)
        principal_axes = np.zeros((len(masks), 2), dtype=np.float32)
        eigenvalues = np.zeros((len(masks), 2), dtype=np.float32)
        visibility = np.zeros((len(masks), 1), dtype=np.float32)

        for index, mask in enumerate(masks):
            pca = self._pca_axes(mask)
            if pca is None:
                continue

            center, axis, values = pca
            centers[index] = (center[0] / width, center[1] / height)
            principal_axes[index] = axis
            eigenvalues[index] = values
            visibility[index, 0] = 1.0

        geometry = np.concatenate(
            (
                centers,
                principal_axes,
                confidences[:, None],
                visibility,
            ),
            axis=1,
        ).astype(np.float32, copy=False)

        return {
            "masks": masks,
            "class_ids": class_ids,
            "confidences": confidences,
            "centers": centers,
            "principal_axes": principal_axes,
            "eigenvalues": eigenvalues,
            "geometry": geometry,
        }

    __call__ = forward

    @classmethod
    def _resolve_classes(
        cls, classes: Sequence[str | int] | None
    ) -> list[int]:
        if classes is None:
            return []

        resolved: list[int] = []
        for item in classes:
            if isinstance(item, str):
                if item not in cls.class_dict:
                    valid = ", ".join(cls.class_dict)
                    raise ValueError(f"Unknown class {item!r}. Valid classes: {valid}")
                class_id = cls.class_dict[item]
            elif isinstance(item, (int, np.integer)):
                class_id = int(item)
                if class_id < 0:
                    raise ValueError("Class ids must be non-negative")
            else:
                raise TypeError("classes must contain only class names or integer ids")

            if class_id not in resolved:
                resolved.append(class_id)

        return resolved

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape [H, W, 3]")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image dimensions must be non-zero")

    @staticmethod
    def _pca_axes(
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return center, canonical principal axis and eigenvalues of a mask."""
        y, x = np.nonzero(mask)
        if len(x) < 3:
            return None

        mask_points = np.column_stack((x, y)).astype(np.float32)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(mask_points, mean=None)

        center = mean[0].astype(np.float32)
        principal_axis = eigenvectors[0].astype(np.float32)
        values = eigenvalues.reshape(-1).astype(np.float32)

        norm = float(np.linalg.norm(principal_axis))
        if norm == 0.0:
            return None
        principal_axis /= norm

        if principal_axis[0] < 0.0 or (
            np.isclose(principal_axis[0], 0.0) and principal_axis[1] < 0.0
        ):
            principal_axis *= -1.0

        return center, principal_axis, values[:2]

    @staticmethod
    def _empty_result(height: int, width: int) -> dict[str, np.ndarray]:
        return {
            "masks": np.empty((0, height, width), dtype=bool),
            "class_ids": np.empty((0,), dtype=np.int64),
            "confidences": np.empty((0,), dtype=np.float32),
            "centers": np.empty((0, 2), dtype=np.float32),
            "principal_axes": np.empty((0, 2), dtype=np.float32),
            "eigenvalues": np.empty((0, 2), dtype=np.float32),
            "geometry": np.empty((0, 6), dtype=np.float32),
        }

def feature_aggregation(
    features: dict[str, np.ndarray],
    classes: Sequence[str | int] | None = None,
) -> np.ndarray:
    """Keep the highest-confidence geometry vector for each requested class.

    Args:
        features: Output of :meth:`DataExtractor.forward`.
        classes: Class names or ids that define the output slots and their
            order. If omitted, all classes from ``DataExtractor.class_dict``
            are used.

    Returns:
        Array with shape ``[len(classes), geometry_dim]``. A class that has no
        detection is represented by a zero vector. If several objects of the
        same class are detected, only the one with maximum confidence remains.

        Before concatenating the result with ACT inputs, it can be flattened
        into a fixed-size vector with ``fixed_geometry.reshape(-1)``.
    """
    required_classes: Sequence[str | int]
    if classes is None:
        required_classes = list(DataExtractor.class_dict.values())
    else:
        required_classes = classes
    class_ids_to_keep = DataExtractor._resolve_classes(required_classes)

    try:
        geometry = np.asarray(features["geometry"])
        detected_class_ids = np.asarray(features["class_ids"])
        confidences = np.asarray(features["confidences"])
    except KeyError as error:
        raise KeyError(
            "features must contain 'geometry', 'class_ids' and 'confidences'"
        ) from error

    if geometry.ndim != 2:
        raise ValueError("features['geometry'] must have shape [N, geometry_dim]")
    if detected_class_ids.ndim != 1 or confidences.ndim != 1:
        raise ValueError("'class_ids' and 'confidences' must have shape [N]")
    if not (
        len(geometry) == len(detected_class_ids) == len(confidences)
    ):
        raise ValueError(
            "'geometry', 'class_ids' and 'confidences' must have the same length"
        )

    fixed_geometry = np.zeros(
        (len(class_ids_to_keep), geometry.shape[1]),
        dtype=geometry.dtype,
    )

    for slot, class_id in enumerate(class_ids_to_keep):
        indices = np.flatnonzero(detected_class_ids == class_id)
        if len(indices) == 0:
            continue

        best_index = indices[np.argmax(confidences[indices])]
        fixed_geometry[slot] = geometry[best_index]

    return fixed_geometry
