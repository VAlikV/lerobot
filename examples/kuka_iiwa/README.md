# Kuka iiwa IL with LeRobot

## Teleoperation

Set the leader port and ID, then calibrate:

```bash
lerobot-calibrate --teleop.type=kuka_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_kuka_leader
```

Configure ports and FPS in the selected script, then run:

```bash
python examples/kuka_iiwa/joint_teleop.py
# Or Cartesian teleoperation:
python examples/kuka_iiwa/cartesian_teleop.py
```

## Dataset recording

Set robot, cameras, FPS, dataset ID and episode settings in
`examples/kuka_iiwa/configs/record_config.json`, then run:

```bash
python examples/kuka_iiwa/record.py
```

Keyboard controls support advancing, re-recording and quitting episodes, and
switching action scaling.

## Action Chunking with TransformersACT

### Training

Set `DATASET_ID`, `PATH`, `DEVICE` and `USE_RELATIVE_ACTIONS` in
`kuka/act/simple_training.py`.

For relative actions, first compute relative statistics using the training chunk size:

```bash
lerobot-edit-dataset \
  --repo_id your_dataset \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 100 \
  --operation.relative_exclude_joints "['gripper']"
```

Train:

```bash
python kuka/act/simple_training.py
```

### Inference

Set `CONFIG_PATH`, `MODEL_ID`, `DATASET_ID`, `DEVICE` and `N_ACTION_STEPS` in
`examples/kuka_iiwa/rollout.py`.

```bash
python examples/kuka_iiwa/rollout.py
```

## Streaming Flow Matching (SFP)

### Train and resume

Set the defaults at the top of `train_streaming_flow_matching.py` or override them on the command line.

```bash
# Deterministic:
python examples/kuka_iiwa/train_streaming_flow_matching.py

# Stochastic (separate output directory):
python examples/kuka_iiwa/train_streaming_flow_matching.py \
  --policy.variant=stochastic --output_dir=outputs/kuka_sfp_stochastic

# Resume:
python examples/kuka_iiwa/train_streaming_flow_matching.py \
  --config_path=outputs/kuka_streaming_flow_matching/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=100000
```

### Replay and Inference

```bash
# Offline replay; no hardware connection:
python examples/kuka_iiwa/rollout_streaming_flow_matching.py \
  --dry-run --device=cpu --max-steps=100

# Real robot, refresh observations every 8 commands:
python examples/kuka_iiwa/rollout_streaming_flow_matching.py \
  --device=cuda --n-action-steps=8
```

Use `--n-action-steps=1` to refresh observations every tick, and
`--integration-steps-per-action` to change integration resolution (default: `1`).

## DAgger

Set `MODEL_ID`, `POLICY_DATASET_ID` and `CONFIG_PATH` in `examples/kuka_iiwa/dagger.py`:

```bash
python examples/kuka_iiwa/dagger.py
```

The session starts paused. Keyboard controls:

- `p`: pause/resume the policy.
- `c`: start/finish a correction; follow the rewind and leader-sync prompts.
- `r`: discard the current correction.
- `s`: toggle action scaling.
- `Esc`: stop; a correction in progress is kept.
