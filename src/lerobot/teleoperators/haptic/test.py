import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.utils.robot_utils import precise_sleep

FPS = 30

def main(config: dict[str, Any] | None = None) -> None:

    from lerobot.teleoperators.haptic import HapticTeleop, HapticTeleopConfig
    from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig

    robot_cfg = KukaIiwaConfig(urdf_path="src/lerobot/robots/kuka_iiwa/iiwa.urdf",
                                gripper_port = "/dev/ttyUSB0",
                                gripper_baudrate = 115200,
                                cameras = {})
    
    robot = KukaIiwa(robot_cfg)

    print("Connecting to KUKA iiwa ...")
    robot.connect()
    obs = robot.get_observation()
    
    dt = 1.0 / float(FPS)

    teleop_cfg = HapticTeleopConfig(ip="127.0.0.1",
                                    port=8081,
                                    delta_mode=False,
                                    init_values=[obs["x.pos"],
                                                 obs["y.pos"],
                                                 obs["z.pos"],
                                                 obs["roll.pos"], 
                                                 obs["pitch.pos"], 
                                                 obs["yaw.pos"]])
    

    print("Connecting haptic ...")
    teleop = HapticTeleop(teleop_cfg)
    teleop.connect()

    try:
        while True:
            t0 = time.perf_counter()

            teleop_action = teleop.get_action()
            x = float(teleop_action.get("x.pos", 0.0))
            y = float(teleop_action.get("y.pos", 0.0))
            z = float(teleop_action.get("z.pos", 0.0))
            roll = float(teleop_action.get("roll.pos", 0.0))
            pitch = float(teleop_action.get("pitch.pos", 0.0))
            yaw = float(teleop_action.get("yaw.pos", 0.0))
            grip = int(teleop_action.get("gripper", 1))

            print(x, y, z, roll, pitch, yaw)

            robot.send_action(
                {
                    "x.pos": x,
                    "y.pos": y,
                    "z.pos": z,
                    "roll.pos": roll,
                    "pitch.pos": pitch,
                    "yaw.pos": yaw,
                    "gripper.pos": float(grip),
                }
            )

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n\nInterrupted. Stopping safely ...")

    finally:
        robot.disconnect()
        teleop.disconnect()

if __name__ == "__main__":
    main()
