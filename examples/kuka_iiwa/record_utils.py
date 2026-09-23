from lerobot.datasets import LeRobotDataset


def _finish_episode_buffer(dataset: LeRobotDataset, *, rerecord: bool) -> dict | None:
    """Detach an accepted episode, or discard a take while preserving its index.
 
    Actual encoding/writing is deferred to `save_episode(episode_data=...)`,
    called later (after the robot is disconnected). Accepted episodes must
    keep their temporary camera frame files on disk until that later save,
    so `delete_images` is only set for a discarded (rerecord) take.
    """
    if dataset.writer is None:
        raise RuntimeError("Recording requires a dataset opened in write mode.")
    buffer = dataset.writer.episode_buffer
    episode_index = int(buffer["episode_index"])
    dataset.clear_episode_buffer(delete_images=rerecord)
    dataset.writer.episode_buffer["episode_index"] = episode_index if rerecord else episode_index + 1
    return None if rerecord else buffer
 
 
def _assert_pending_episode_indices(dataset: LeRobotDataset, pending_episode_buffers: list[dict]) -> None:
    for expected_idx, episode_buffer in enumerate(pending_episode_buffers, start=dataset.num_episodes):
        actual_idx = int(episode_buffer["episode_index"])
        if actual_idx != expected_idx:
            raise RuntimeError(
                "Pending episode buffer indices are inconsistent: "
                f"buffer {expected_idx} has episode_index={actual_idx}. "
                "Discard this recording run and record again."
            )
 
