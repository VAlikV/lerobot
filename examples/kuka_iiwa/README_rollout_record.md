# Recording ACT rollouts

Run from the repository root with the environment containing this checkout.
This script supports ResNet ACT and the frozen DINOv2 ACT added on this branch.
The invocation connects the robot and starts motion only after Enter.

```bash
python examples/kuka_iiwa/rollout_record.py \
  --config examples/kuka_iiwa/configs/record_config.json \
  --policy /absolute/path/to/checkpoint \
  --source-dataset /absolute/path/to/kuka_put_cap \
  --repo-id local/kuka_put_cap_dinov2_rollouts \
  --root /absolute/path/to/new_rollout_dataset \
  --device cuda --num-episodes 5 --episode-time-s 30 \
  --n-action-steps 1 --temporal-ensemble-coeff 0.01
```

`--source-dataset` must be the training dataset (metadata and feature names/order).
Its video files are not read. `--root` is a **new** output directory, never the
training dataset. Resume and Hub upload are not performed. Policy normalization
comes from the checkpoint's saved processors, not the output dataset statistics.
The robot/camera settings and task description come from `--config`; the leader
is not connected. Run from the repository root for relative URDF paths.

Omit execution overrides to retain checkpoint settings. For open-loop chunks:
`--n-action-steps 60 --temporal-ensemble-coeff none` (chunk_size must be at least 60).
Relative ACT supports an external queue with chunk-wise absolute conversion;
relative actions combined with temporal ensembling are rejected.

- Enter: start next episode after arranging objects. By default no automatic reset movement.
- `--reset-to-pose`: after Enter, move to configured reset_pose over 3 seconds,
  outside recording; then start the episode.
- `n` / right: finish and save current episode.
- `r` / left: discard current episode and retry after Enter.
- `q` / Esc: stop and save collected frames of the current episode.
- Ctrl+C or an exception: discard the partial episode; earlier saved episodes remain.

Keyboard controls require the listener backend available on the host. During the
Enter prompt the listener is suspended; Ctrl+C cancels without starting motion. The teleoperation
`s` scaling control has no effect on autonomous rollout.

Recorded frames contain raw camera images and measured state from **before** the
command, plus the absolute action returned by `send_action` after safety limits.
Policy and processor state are reset each episode. Images use streaming encoding
with one encoder thread and a queue of four frames per camera. Episodes are saved
immediately, rather than retained for the whole session. Control overruns are logged;
dataset timestamps represent the configured nominal FPS, not measured wall time.

`rollout_config.json` records the checkpoint/source paths and execution settings.
Success/failure labels are not inferred: retain a separate episode outcome log.
No full-size model or robot is needed for the mock tests in
`tests/policies/act/test_rollout_record.py`.
