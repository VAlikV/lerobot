"""Training entrypoint must fail locally before downloading model weights."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script():
    path = Path(__file__).resolve().parents[3] / "kuka/act/train_dinov2.py"
    assert path.exists(), "DINOv2 training entrypoint is missing"
    spec = importlib.util.spec_from_file_location("train_dinov2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_videos(tmp_path):
    module = load_script()
    metadata = SimpleNamespace(
        root=tmp_path,
        total_episodes=2,
        features={key: {"shape": [224, 224, 3]} for key in module.CAMERAS},
        get_video_file_path=lambda ep, key: Path("videos") / key / f"{ep}.mp4",
    )
    with pytest.raises(FileNotFoundError, match="Missing camera videos"):
        module.validate_videos(metadata)
    for key in module.CAMERAS:
        for ep in range(2):
            path = tmp_path / metadata.get_video_file_path(ep, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    module.validate_videos(metadata)


def test_cli_rejects_empty_training():
    module = load_script()
    for args in (["--steps", "0"], ["--batch-size", "0"]):
        with pytest.raises(SystemExit):
            module.parse_args(args)
