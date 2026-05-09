"""Generate tiny synthetic LeRobotDatasets from the sim_assembling env for tests.

Scripted policies:
- ``success`` — lower the ee, close gripper, lift. final frame reward=1, done=True.
- ``failure`` — random zero-mean walk. reward stays 0.

Output is written to a local root (no hub push). Keep episodes short
(~20 frames) so tests stay fast.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SUCCESS_PHASES = [
    # (n_steps, [dx, dy, dz, gripper_discrete])
    (5, [0.0, 0.0, -1.0, 0.0]),  # descend
    (3, [0.0, 0.0, 0.0, 2.0]),  # close gripper
    (7, [0.0, 0.0, 1.0, 0.0]),  # lift
    (5, [0.0, 0.0, 0.0, 0.0]),  # hold
]


def _success_action(t: int) -> np.ndarray:
    cum = 0
    for n, a in _SUCCESS_PHASES:
        if t < cum + n:
            return np.asarray(a, dtype=np.float32)
        cum += n
    return np.zeros(4, dtype=np.float32)


def _failure_action(_t: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
    a[3] = 0.0  # keep gripper open
    return a


def _build_features(obs_sample: dict, action_dim: int, fps: int) -> dict:
    agent_pos = obs_sample["agent_pos"]
    front = obs_sample["pixels"]["front"]
    wrist = obs_sample["pixels"]["wrist"]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(agent_pos.shape),
            "names": [f"state_{i}" for i in range(agent_pos.shape[0])],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": tuple(front.shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": tuple(wrist.shape),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"a_{i}" for i in range(action_dim)],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def generate_sim_synth_dataset(
    root: str | Path,
    repo_id: str,
    kind: str = "success",
    num_episodes: int = 3,
    ep_len: int = 20,
    seed: int = 0,
    fps: int = 20,
) -> "LeRobotDataset":  # type: ignore[name-defined]
    """Record ``num_episodes`` scripted trajectories into a fresh LeRobotDataset.

    Args:
        root: local directory (not a hub push).
        repo_id: placeholder repo id string.
        kind: ``"success"`` or ``"failure"``.
        num_episodes: how many episodes to record.
        ep_len: steps per episode.
        seed: rng seed for failure policy.
        fps: dataset fps to record at.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym

    import lerobot.envs.sim_assembling  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(seed)

    env = gym.make(
        "sim_assembling/AssembleBase-v0",
        control_hz=float(fps),
        mode="fast",
        max_episode_steps=ep_len + 5,
        render_mode="rgb_array",
    )

    obs, _ = env.reset()
    features = _build_features(obs, env.action_space.shape[0], fps)

    root = Path(root)
    if root.exists():
        import shutil

        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(root),
        use_videos=False,
    )

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            for t in range(ep_len):
                if kind == "success":
                    a = _success_action(t)
                else:
                    a = _failure_action(t, rng)
                obs_next, r_env, term, trunc, info = env.step(a)
                is_terminal = (t == ep_len - 1)
                reward = 1.0 if (kind == "success" and is_terminal) else 0.0
                done = is_terminal and (kind == "success")

                frame = {
                    "observation.state": torch.from_numpy(obs["agent_pos"].copy()).float(),
                    "observation.images.front": obs["pixels"]["front"].copy(),
                    "observation.images.wrist": obs["pixels"]["wrist"].copy(),
                    "action": torch.from_numpy(a.copy()).float(),
                    "next.reward": torch.tensor([reward], dtype=torch.float32),
                    "next.done": torch.tensor([done], dtype=torch.bool),
                    "task": f"assemble_{kind}",
                }
                ds.add_frame(frame)
                obs = obs_next
                if term or trunc:
                    break
            ds.save_episode()
        ds.finalize()
    finally:
        env.close()

    return ds
