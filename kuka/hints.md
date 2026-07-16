## Record

```bash
python -m lerobot.rl.gym_manipulator --config_path kuka/configs/kuka_iiwa_env_3cams_yaw_record.json
```

## Train

By `simple_training.py`

## Eval

By `simple_using_example.py`

## Concatenate 

```bash
lerobot-edit-dataset   --new_repo_id local/kuka_device_assemble2_abs_stage3   --operation.type merge   --operation.repo_ids "['local/kuka_device_assemble2_abs_stage3_part1', 'local/kuka_device_assemble2_abs_stage3_part2', 'local/kuka_device_assemble2_abs_stage3_part3']"
```