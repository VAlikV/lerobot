# Direct joint control

Enable with `KukaIiwaConfig(use_task_space=False, use_direct_joint_control=True)`.
Robot actions use `joint_1.pos` through `joint_7.pos` (or configured `joint_names`),
followed by `gripper.pos`. Joint targets and observations are absolute **degrees**;
TCP orientation observations remain in radians. Gripper position is `1` for open
and `-1` for closed.

`reset_pose` accepts seven angles in degrees and an optional gripper position.
Without it, the robot captures its current pose at connection as home. Without
the optional gripper value, the current gripper state is retained.

`KukaIiwaRobotEnv` selects the mode from the robot configuration. Its actions are
`[q1, q2, q3, q4, q5, q6, q7, gripper_cmd]`, omitting the final value when
`use_gripper=False`. Angles are absolute degrees; `gripper_cmd` is `0` (close),
`1` (stay), or `2` (open). The environment does not normalize joint actions.
Its state retains the existing TCP/force/gripper values and appends seven joint
angles. Absolute recording stores seven target angles and measured gripper state.

Set `KukaIiwaRobotEnvConfig.home_joints` to override the robot home pose, using
the same layout as `reset_pose`. Reset interpolates joint targets over up to three
seconds and holds them for the remaining reset duration. TCP bounds and TCP
randomization do not apply in joint mode. The environment sends joint targets without clipping them.

The current `kuka_leader` produces radians. Its future direct-control adapter must
convert arm angles to degrees and map the gripper to the appropriate command.
The existing task-space FK example and HIL intervention processor are not joint
teleoperation adapters.
