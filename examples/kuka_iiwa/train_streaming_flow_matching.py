"""Train SFP with the standard LeRobot trainer; CLI flags override these defaults.

Run from the lerobot repository in the conda lerobot environment. Resume with
--config_path=<checkpoint>/pretrained_model/train_config.json --resume=true.
"""

import sys

from lerobot.scripts.lerobot_train import train

DATASET_ID = "local/kuka_sfp"
OUTPUT_DIR = "outputs/kuka_streaming_flow_matching"
DEVICE = "cuda"
VARIANT = "deterministic"  # or stochastic
BATCH_SIZE = 32
TRAINING_STEPS = 70_000
SAVE_FREQ = 5_000


def main() -> None:
    # A saved training config owns resume settings; do not replace them with defaults.
    supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    if "--config_path" not in supplied:
        defaults = {
            "dataset.repo_id": DATASET_ID,
            "policy.type": "streaming_flow_matching",
            "policy.device": DEVICE,
            "policy.variant": VARIANT,
            "policy.push_to_hub": "false",
            "output_dir": OUTPUT_DIR,
            "batch_size": BATCH_SIZE,
            "steps": TRAINING_STEPS,
            "save_freq": SAVE_FREQ,
            "log_freq": 100,
            "env_eval_freq": 0,
            "ema.enable": "true",
        }
        sys.argv.extend(f"--{key}={value}" for key, value in defaults.items() if f"--{key}" not in supplied)
    train()


if __name__ == "__main__":
    main()
