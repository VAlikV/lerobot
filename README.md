# Kuka iiwa IL with LeRobot

## Teleoperation with kinematic clone (leader arm)

Activate `lerobot` environment.

First, calibrate the leader arm:
```
lerobot-calibrate --teleop.type=kuka_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_kuka_leader
```

**Cartesian Space:**
Leader arm returns joint angles but follower arm accepts EE position and orientation and gripper state (open/close). So, to work together action proccessor is needed. In `examples/сcartesian_teleop.py`:
- define `URDF_PATH` to follower's .urdf file
- define ports in `follower_config` and `leader_config`
- define `FOLLOWER_JOINT_NAMES` according to according to follower's urdf

Run:
```
python examples/kuka_iiwa/cartesian_teleop.py
```

**Joint Space:**
```
python examples/kuka_iiwa/joint_teleop.py
```

### Architecture

- **KukaLeader**

Responsible for reading data from stm32 and software-only callibration, implements Teleoperator interface.

Path: src/lerobot/teleoperators/kuka_leader

- **LeaderJointDeltaToFollowerEE** and **KukaEEBoundsAndSafety**

Responsible for data transition from leader to follower format, implements RobotActionProcessorStep interface.

Path: src/lerobot/robots/kuka_iiwa/robot_kinematic_processor.py