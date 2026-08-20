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
lerobot-edit-dataset   --new_repo_id local/kuka_multi_policy_stage_2_finetune   --operation.type merge   --operation.repo_ids "['local/kuka_multi_policy_stage_2', 'local/kuka_multi_policy_stage_2_fine_tune']"
```