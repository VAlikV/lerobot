# Kuka iiwa IL with LeRobot

Imitation learning pipeline for Kuka iiwa with a kinematic clone as the leader arm.

## Teleoperation

Activate the `lerobot` environment.

### 1. Calibrate the leader arm

```bash
lerobot-calibrate --teleop.type=kuka_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_kuka_leader
```

### 2. Run teleoperation

Configure ports, FPS, and other parameters in the selected script:

```bash
python examples/kuka_iiwa/joint_teleop.py
```

Cartesian teleoperation is also available:

```bash
python examples/kuka_iiwa/cartesian_teleop.py
```

## Dataset recording

Dataset recording is configured in:

```text
examples/kuka_iiwa/configs/record_config.json
```

The recording script supports:

* keyboard recording control (next, re-record, quit);
* action scaling.

Run:

```bash
python examples/kuka_iiwa/record.py
```

## ACT training

Chunk-wise delta actions are supported in ACT. Enable relative actions in:

> Note: ACT applies relative conversion before normalization (relative → normalize), so the normalizer always sees delta (relative) values. This means relative action stats are required for all of them when training with use_relative_actions=true.

**Step 1:** Precompute relative action statistics for dataset

```sh
lerobot-edit-dataset \
    --repo_id your_dataset \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.chunk_size 100 \
    --operation.relative_exclude_joints "['gripper']"
```

**Step 2:** Train with relative actions enabled

Set `USE_RELATIVE_ACTIONS = True` in

```text
kuka/act/simple_training.py
```

## Inference

Simple inference can be run with:

```bash
python examples/kuka_iiwa/rollout.py
```