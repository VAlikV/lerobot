from unittest.mock import Mock

import numpy as np
import pytest

from lerobot.robots.kuka_iiwa import KukaIiwa, KukaIiwaConfig, KukaIiwaRobotEnv, KukaIiwaRobotEnvConfig


def make_robot(joint_mode=True, reset_pose=None):
    robot = KukaIiwa(KukaIiwaConfig(
        use_task_space=not joint_mode, use_direct_joint_control=joint_mode, reset_pose=reset_pose,
    ))
    raw = np.zeros(25)
    raw[:7] = np.deg2rad(np.arange(7) * 10)
    raw[7:10] = [0.2, 0.1, 0.3]
    raw[10:19] = np.eye(3).flatten()
    robot._controller = Mock()
    robot._controller.get_observation.return_value = raw
    robot._gripper = Mock(is_open=True)
    robot._home_action = robot._make_home_action()
    return robot


@pytest.mark.parametrize("joint_mode", [False, True])
def test_home_from_observation(joint_mode):
    robot = make_robot(joint_mode)
    assert set(robot._home_action) == set(robot.action_features)
    assert set(robot.get_observation()) == set(robot.observation_features)
    robot._set_target_action(robot._home_action)
    if joint_mode:
        np.testing.assert_allclose(robot._target_joints_position, np.arange(7) * 10)


@pytest.mark.parametrize("joint_mode,count", [(False, 6), (True, 7)])
@pytest.mark.parametrize("gripper", [[], [-1.0]])
def test_explicit_home(joint_mode, count, gripper):
    robot = make_robot(joint_mode, list(range(count)) + gripper)
    assert list(robot._home_action.values()) == list(range(count)) + (gripper or [1.0])
    robot.config.reset_pose = [0.0] * (count - 1)
    with pytest.raises(ValueError):
        robot._make_home_action()


@pytest.mark.parametrize("use_gripper", [False, True])
def test_joint_env_step_reset_recording(use_gripper):
    robot = make_robot()
    env = KukaIiwaRobotEnv(robot, KukaIiwaRobotEnvConfig(use_gripper=use_gripper, reset_time_s=0))
    assert env.action_space.shape == (7 + int(use_gripper),)
    obs, _ = env.reset()
    assert obs["agent_pos"].shape == (17,)
    np.testing.assert_allclose(robot._target_joints_position, np.arange(7) * 10)
    target = [90.0, -30.0, 0.0, 45.0, 10.0, 20.0, 30.0]
    env.step(target + ([0.0] if use_gripper else []))
    np.testing.assert_allclose(robot._target_joints_position, target)
    np.testing.assert_allclose(env.get_recording_action("absolute")[:7], target)
    assert env.get_recording_action_features("absolute")["shape"] == (8,)
    assert env.get_recording_action_features()["shape"] == env.action_space.shape
    if use_gripper:
        robot._gripper.send.assert_called_with(-1.0)
    else:
        robot._gripper.send.assert_not_called()
    with pytest.raises(ValueError):
        env.step([0.0] * 3)
    with pytest.raises(ValueError):
        env.step([np.nan] * env.action_space.shape[0])


def test_joint_control_loop_uses_degrees():
    robot = make_robot()
    robot._set_target_action(robot._home_action)
    robot._controller.set_target_joints_degrees.side_effect = lambda _: robot._control_stop_event.set()
    robot._control_loop()
    np.testing.assert_allclose(robot._controller.set_target_joints_degrees.call_args.args[0], np.arange(7) * 10)
    robot._controller.set_target.assert_not_called()


def test_task_env_unchanged():
    robot = make_robot(False)
    env = KukaIiwaRobotEnv(robot, KukaIiwaRobotEnvConfig(reset_time_s=0))
    assert env.action_space.shape == (4,)
    obs, _ = env.reset()
    assert obs["agent_pos"].shape == (10,)
    env.step([0.0, 0.0, 0.0, 1.0])
    assert env.get_recording_action("absolute").shape == (7,)
