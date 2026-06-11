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
lerobot-edit-dataset   --new_repo_id local/kuka_iiwa_2cams_abs_packing   --operation.type merge   --operation.repo_ids "['local/kuka_iiwa_2cams_abs_packing_park1', 'local/kuka_iiwa_2cams_abs_packing_park2']"
```