#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to find end effector (EE) workspace bounds for RC10 robot via PS4 joystick teleoperation
Adapted from src/lerobot/scripts/lerobot_find_joint_limits.py

Example:
```shell
    python rc10/rc10_find_ee_limits.py \
        --robot.type=rc10_follower \
        --robot.ip=10.10.10.10 \
        --robot.gripper_port=/dev/ttyUSB0 \
        --robot.id=rc10 \
        --teleop.type=ps4_joystick \
        --teleop.id=ps4 \
        --teleop_time_s=60 \
        --warmup_time_s=5 \
        --control_loop_fps=30
```
"""

import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.robots import RobotConfig, rc10_follower  # noqa: F401
from lerobot.robots.rc10_follower import RC10FollowerCut
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    make_teleoperator_from_config,
    ps4_joystick,
)
from lerobot.utils.robot_utils import precise_sleep


@dataclass
class FindEEBoundsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig

    # Duration of the recording phase in seconds
    teleop_time_s: float = 60
    # Duration of the warmup phase in seconds
    warmup_time_s: float = 5
    # Control loop frequency
    control_loop_fps: int = 30


@draccus.wrap()
def find_ee_bounds(cfg: FindEEBoundsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = RC10FollowerCut(cfg.robot)
    print(f"Connecting to robot: {cfg.robot.type}...")
    teleop.connect()
    robot.connect()
    print("Devices connected.")

    max_tcp = None
    min_tcp = None
    first_tcp = None

    start_t = time.perf_counter()
    warmup_done = False

    print("\n" + "=" * 50)
    print(f"  WARMUP PHASE ({cfg.warmup_time_s}s)")
    print("  Move the robot freely to ensure control works")
    print("  Data is NOT being recorded yet.")
    print("=" * 50 + "\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Teleoperation - move robot with PS4 joystick
            action = teleop.get_action()
            robot.send_action(action)

            # 2. Read current TCP position directly from controller
            tcp = np.array(robot._controller.get_current_tcp(), dtype=np.float32)
            # tcp = [x, y, z, roll, pitch, yaw]

            current_time = time.perf_counter()
            elapsed = current_time - start_t

            # 3. Handle Phases
            if elapsed < cfg.warmup_time_s:
                # Still in warmup
                pass
            else:
                # Phase Transition: Warmup -> Recording
                if not warmup_done:
                    print("\n" + "=" * 50)
                    print("  RECORDING STARTED")
                    print("  Move robot to ALL edges of our task workspace")
                    print("  Cover the full x, y, z, yaw range you need.")
                    print("  Press Ctrl+C to stop early and save results.")
                    print("=" * 50 + "\n")

                    max_tcp = tcp.copy()
                    min_tcp = tcp.copy()
                    first_tcp = tcp.copy()
                    warmup_done = True

                # Update limits
                max_tcp = np.maximum(max_tcp, tcp)
                min_tcp = np.minimum(min_tcp, tcp)

                # Time check
                recording_time = elapsed - cfg.warmup_time_s
                remaining = cfg.teleop_time_s - recording_time

                if int(recording_time * 2) % 1 == 0:
                    print(
                        f"  TCP: x={tcp[0]:.4f} y={tcp[1]:.4f} z={tcp[2]:.4f} "
                        f"yaw={tcp[5]:.4f}  |  Remaining: {remaining:.0f}s",
                        end="\r",
                    )

                if recording_time > cfg.teleop_time_s:
                    print("\n\nTime limit reached.")
                    break

            precise_sleep(max(1.0 / cfg.control_loop_fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping safely...")

    except Exception as e:
        import traceback
        print(f"\n\nError: {e}")
        traceback.print_exc()

    finally:
        print("\nDisconnecting devices...")
        robot.disconnect()
        teleop.disconnect()

    # Results Output
    if max_tcp is not None:
        r_max = np.round(max_tcp, 4).tolist()
        r_min = np.round(min_tcp, 4).tolist()
        r_first = np.round(first_tcp, 4).tolist()

        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)

        print("\n# EE Bounds (from our workspace sweep):")
        print(f"#   x:    [{r_min[0]:.4f}, {r_max[0]:.4f}]")
        print(f"#   y:    [{r_min[1]:.4f}, {r_max[1]:.4f}]")
        print(f"#   z:    [{r_min[2]:.4f}, {r_max[2]:.4f}]")
        print(f"#   roll: [{r_min[3]:.4f}, {r_max[3]:.4f}]")
        print(f"#   pitch:[{r_min[4]:.4f}, {r_max[4]:.4f}]")
        print(f"#   yaw:  [{r_min[5]:.4f}, {r_max[5]:.4f}]")

        print("\n# Paste into our config JSON (env.processor.inverse_kinematics):")
        print('"end_effector_bounds": {')
        print(f'    "min": [{r_min[0]}, {r_min[1]}, {r_min[2]}],')
        print(f'    "max": [{r_max[0]}, {r_max[1]}, {r_max[2]}]')
        print('}')

        print("\n# Starting TCP (first position after warmup):")
        print("#   Use this as reset_tcp if it's a good home position.")
        print(f'"fixed_reset_joint_positions": {r_first}')

    else:
        print("No data recorded (exited during warmup).")


def main():
    find_ee_bounds()


if __name__ == "__main__":
    main()
