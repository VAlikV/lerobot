# Kuka iiwa IL with LeRobot

## Teleoperation with kinematic clone (leader arm)

Activate `lerobot` environment.

First, calibrate the leader arm:
```
lerobot-calibrate --teleop.type=kuka_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_kuka_leader
```
Second, set user configurations (ports, fps, etc.) inside choosen teleoperation script and run it:

```
python examples/kuka_iiwa/joint_teleop.py   # or examples/kuka_iiwa/cartesian_teleop.py
```

### Architecture

- **KukaLeader**

Responsible for reading data from stm32, sending command to stm32, and software-only callibration; implements Teleoperator interface.

Path: src/lerobot/teleoperators/kuka_leader

- **LeaderJointDeltaToFollowerEE**, **KukaEEBoundsAndSafety**, and **GripperPositionToDiscrete**

Responsible for data transition from leader to follower format, implements RobotActionProcessorStep interface.

Path: src/lerobot/robots/kuka_iiwa/robot_kinematic_processor.py