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
examples/kuka_iiwa/configs/record_rollout_config.json
```

The recording script supports:

* keyboard recording control (next, re-record, quit);
* action scaling.

Run:

```bash
python examples/kuka_iiwa/record.py \
    --config examples/kuka_iiwa/configs/record_rollout_config.json \
    --dataset-id local/kuka_test_2
```

## ACT training

Chunk-wise delta actions are supported in ACT.

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
python examples/kuka_iiwa/rollout.py \
    --config examples/kuka_iiwa/configs/record_rollout_config.json \
    --dataset-id local/kuka_test_2 \
    --model-id outputs/device_assemble
```

### Record policy rollouts

```bash
python examples/kuka_iiwa/rollout_record.py \
    --config examples/kuka_iiwa/configs/record_rollout_config.json \
    --dataset-id local/kuka_test_2 \
    --model-id outputs/device_assemble
```

## Dagger

Adapted from the official LeRobot DAgger strategy.

### State machine:
```text
PAUSED --> AUTONOMOUS --> PAUSED --> CORRECTING --> PAUSED
The session starts in PAUSED, so nothing moves until you press `p`.
```

### Keyboard:
- p   - pause / resume the policy
- c   - start / finish a correction (start: rewind -> sync -> recording / finish: the correction is moved to the pending buffer as one episode)
- r   - discard the correction currently being recorded (and go back to PAUSED)
- s   - toggle action scaling (same as in record script)
- esc - stop the session (a correction in progress is kept)


### PAUSED -> CORRECTING sequence:
1. "Rewind seconds?" prompt (0 / Enter = no rewind). The follower walks back along the
    trajectory it has just executed (reverse playback of sent actions).
2. "Press Enter to sync leader..." prompt.
3. Leader is driven to the follower pose; scaling and leader filter are reset.
4. Recording starts; frames are written into the episode buffer.


```bash
python examples/kuka_iiwa/dagger.py \
    --config examples/kuka_iiwa/configs/record_rollout_config.json \
    --dataset-id local/kuka_corrections_1 \
    --policy-dataset-id local/kuka_test_2 \
    --model-id outputs/device_assemble
```
