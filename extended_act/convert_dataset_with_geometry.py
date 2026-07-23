"""Add offline segmentation geometry to an existing LeRobot dataset.

The converter does not re-encode videos. It creates a new dataset directory
using hard links when possible, computes geometry only for ``CAMERA_KEYS``,
adds ``observation.geometry`` to parquet files, and updates dataset statistics.

Run from the repository root:

    python extended_act/convert_dataset_with_geometry.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Array3D, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

try:
    from .data_extractor import DataExtractor, feature_aggregation
except ImportError:
    from data_extractor import DataExtractor, feature_aggregation


SOURCE_DATASET_ID = "local/kuka_stage_1_gray"
TARGET_DATASET_ID = "local/kuka_stage_1_gray_geometry"
SOURCE_ROOT: Path | None = None
TARGET_ROOT: Path | None = None

SEGMENTATION_MODEL_PATH = Path("extended_act/models/nano_assemble.pt")

# Only these cameras are decoded and passed through YOLO. Their order becomes
# the first dimension of observation.geometry.
CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.side",
)
GEOMETRY_CLASSES = ("Down", "Target")
GEOMETRY_FEATURE_NAMES = (
    "center_x",
    "center_y",
    "axis_x",
    "axis_y",
    "confidence",
    "visibility",
)

PERCEPTION_DEVICE: str | None = None
LOG_EVERY = 100
OBS_GEOMETRY = "observation.geometry"


def tensor_rgb_to_bgr_image(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3, H, W], got {tuple(image.shape)}.")

    image = image.detach().to(device="cpu", dtype=torch.float32)
    image = image.clamp(0.0, 1.0).permute(1, 2, 0)
    rgb = (image.numpy() * 255.0).round().astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def resolve_root(repo_id: str, root: Path | None) -> Path:
    return root.expanduser().resolve() if root is not None else HF_LEROBOT_HOME / repo_id


def link_or_copy(source: str, destination: str) -> str:
    """Hard-link large unchanged files and fall back to a regular copy."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def validate_configuration(
    source: LeRobotDataset,
    target_root: Path,
    camera_keys: tuple[str, ...],
    classes: tuple[str, ...],
    model_path: Path,
) -> None:
    if target_root.exists():
        raise FileExistsError(
            f"Target dataset already exists: {target_root}. "
            "Change TARGET_DATASET_ID or TARGET_ROOT."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")
    if not camera_keys:
        raise ValueError("At least one camera must be selected.")
    if len(set(camera_keys)) != len(camera_keys):
        raise ValueError(f"Camera keys must be unique: {camera_keys}")
    if not classes:
        raise ValueError("At least one object class must be selected.")

    missing = [key for key in camera_keys if key not in source.meta.video_keys]
    if missing:
        raise KeyError(
            f"Selected cameras are absent or are not video features: {missing}. "
            f"Available video keys: {source.meta.video_keys}"
        )
    if OBS_GEOMETRY in source.features:
        raise ValueError(f"Source dataset already contains `{OBS_GEOMETRY}`.")


def restrict_decoded_cameras(
    source: LeRobotDataset,
    camera_keys: tuple[str, ...],
) -> None:
    """Keep all scalar fields but decode only selected video features."""
    selected = set(camera_keys)
    source.meta.info["features"] = {
        key: feature
        for key, feature in source.meta.info["features"].items()
        if feature["dtype"] != "video" or key in selected
    }


@torch.no_grad()
def extract_geometry(
    source: LeRobotDataset,
    extractor: DataExtractor,
    camera_keys: tuple[str, ...],
    classes: tuple[str, ...],
    log_every: int,
) -> np.ndarray:
    shape = (
        len(source),
        len(camera_keys),
        len(classes),
        len(GEOMETRY_FEATURE_NAMES),
    )
    geometry = np.zeros(shape, dtype=np.float32)

    for frame_index in range(len(source)):
        frame = source[frame_index]
        for camera_index, camera_key in enumerate(camera_keys):
            image = tensor_rgb_to_bgr_image(frame[camera_key])
            detections = extractor(image)
            geometry[frame_index, camera_index] = feature_aggregation(
                detections,
                classes,
            )

        if frame_index == 0 or (frame_index + 1) % log_every == 0:
            print(f"Processed {frame_index + 1}/{len(source)} frames")

    return geometry


def calculate_stats(values: np.ndarray) -> dict[str, Any]:
    values64 = values.astype(np.float64)
    return {
        "min": values64.min(axis=0).tolist(),
        "max": values64.max(axis=0).tolist(),
        "mean": values64.mean(axis=0).tolist(),
        "std": values64.std(axis=0).tolist(),
        "count": [len(values64)],
        "q01": np.quantile(values64, 0.01, axis=0).tolist(),
        "q10": np.quantile(values64, 0.10, axis=0).tolist(),
        "q50": np.quantile(values64, 0.50, axis=0).tolist(),
        "q90": np.quantile(values64, 0.90, axis=0).tolist(),
        "q99": np.quantile(values64, 0.99, axis=0).tolist(),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as file:
        json.dump(value, file, indent=4)
    temporary_path.replace(path)


def add_geometry_to_data_parquets(target_root: Path, geometry: np.ndarray) -> None:
    offset = 0
    parquet_files = sorted((target_root / "data").glob("*/*.parquet"))

    for parquet_path in parquet_files:
        dataset = Dataset.from_parquet(str(parquet_path))
        next_offset = offset + len(dataset)
        file_geometry = geometry[offset:next_offset]
        if len(file_geometry) != len(dataset):
            raise RuntimeError("Geometry length does not match parquet row count.")

        dataset = dataset.add_column(OBS_GEOMETRY, file_geometry.tolist())
        updated_features = dataset.features.copy()
        updated_features[OBS_GEOMETRY] = Array3D(
            shape=geometry.shape[1:],
            dtype="float32",
        )
        dataset = dataset.cast(updated_features)

        temporary_path = parquet_path.with_suffix(".parquet.tmp")
        dataset.to_parquet(str(temporary_path))
        temporary_path.replace(parquet_path)
        offset = next_offset

    if offset != len(geometry):
        raise RuntimeError(
            f"Wrote {offset} geometry rows, but extracted {len(geometry)}."
        )


def add_geometry_to_episode_metadata(
    target_root: Path,
    geometry: np.ndarray,
) -> None:
    episode_files = sorted((target_root / "meta" / "episodes").glob("*/*.parquet"))

    for episode_path in episode_files:
        table = pq.read_table(episode_path)
        rows = table.to_pylist()
        new_columns: dict[str, list[Any]] = {}

        for row in rows:
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            stats = calculate_stats(geometry[start:end])
            for statistic_name, value in stats.items():
                column_name = f"stats/{OBS_GEOMETRY}/{statistic_name}"
                new_columns.setdefault(column_name, []).append(value)

        for column_name, values in new_columns.items():
            table = table.append_column(column_name, pa.array(values))

        temporary_path = episode_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path)
        temporary_path.replace(episode_path)


def update_metadata(
    target_root: Path,
    geometry: np.ndarray,
    camera_keys: tuple[str, ...],
    classes: tuple[str, ...],
    source_id: str,
) -> None:
    info_path = target_root / "meta" / "info.json"
    with info_path.open() as file:
        info = json.load(file)

    info["features"][OBS_GEOMETRY] = {
        "dtype": "float32",
        "shape": list(geometry.shape[1:]),
        "names": None,
    }
    info["geometry_extraction"] = {
        "source_dataset": source_id,
        "camera_keys": list(camera_keys),
        "classes": list(classes),
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
    }
    write_json_atomic(info_path, info)

    stats_path = target_root / "meta" / "stats.json"
    with stats_path.open() as file:
        stats = json.load(file)
    stats[OBS_GEOMETRY] = calculate_stats(geometry)
    write_json_atomic(stats_path, stats)


def convert_dataset() -> Path:
    source_root = resolve_root(SOURCE_DATASET_ID, SOURCE_ROOT)
    target_root = resolve_root(TARGET_DATASET_ID, TARGET_ROOT)
    camera_keys = tuple(CAMERA_KEYS)
    classes = tuple(GEOMETRY_CLASSES)

    source = LeRobotDataset(SOURCE_DATASET_ID, root=source_root)
    validate_configuration(
        source,
        target_root,
        camera_keys,
        classes,
        SEGMENTATION_MODEL_PATH,
    )
    restrict_decoded_cameras(source, camera_keys)

    extractor = DataExtractor(
        model_path=SEGMENTATION_MODEL_PATH,
        classes=classes,
        device=PERCEPTION_DEVICE,
    )
    geometry = extract_geometry(
        source,
        extractor,
        camera_keys,
        classes,
        LOG_EVERY,
    )

    print(f"Copying dataset to {target_root}")
    shutil.copytree(source_root, target_root, copy_function=link_or_copy)
    add_geometry_to_data_parquets(target_root, geometry)
    add_geometry_to_episode_metadata(target_root, geometry)
    update_metadata(
        target_root,
        geometry,
        camera_keys,
        classes,
        SOURCE_DATASET_ID,
    )

    converted = LeRobotDataset(TARGET_DATASET_ID, root=target_root)
    if len(converted) != len(source):
        raise RuntimeError("Converted dataset length differs from the source dataset.")
    if tuple(converted.features[OBS_GEOMETRY]["shape"]) != geometry.shape[1:]:
        raise RuntimeError("Converted geometry feature has an unexpected shape.")

    print(f"Converted dataset: {target_root}")
    print(f"{OBS_GEOMETRY} shape: {geometry.shape[1:]}")
    return target_root


def main() -> None:
    convert_dataset()


if __name__ == "__main__":
    main()
