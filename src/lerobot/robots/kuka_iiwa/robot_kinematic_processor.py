from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from scipy.spatial.transform import Rotation

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0


def _pose_dict_to_transform(pose: dict, euler_order: str = "xyz") -> np.ndarray:
    # Builds a 4x4 transform from a {'x.pos', 'y.pos', 'z.pos', 'roll.pos', 'pitch.pos', 'yaw.pos'} dict.
    t = np.eye(4, dtype=float)
    t[:3, 3] = [float(pose["x.pos"]), float(pose["y.pos"]), float(pose["z.pos"])]
    t[:3, :3] = Rotation.from_euler(
        euler_order, [float(pose["roll.pos"]), float(pose["pitch.pos"]), float(pose["yaw.pos"])]
    ).as_matrix()
    return t


@ProcessorStepRegistry.register("kuka_leader_joint_delta_to_follower_ee")
@dataclass
class LeaderJointDeltaToFollowerEE(RobotActionProcessorStep):
    """
    Converts joint-position teleop commands from a kinematic-clone leader arm into a
    relative Cartesian target for a Cartesian-controlled follower.
    """

    kinematics: RobotKinematics
    leader_motor_names: list[str]
    euler_order: str = "xyz"
    gripper_open_threshold: float = 0.0
    use_latched_reference: bool = True

    _q_leader_ref: np.ndarray | None = field(default=None, init=False, repr=False)
    _t_leader_ref: np.ndarray | None = field(default=None, init=False, repr=False)
    _t_follower_ref: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Follower observation is required to anchor the relative EE target.")

        q_leader = np.array(
            [float(action.pop(f"{name}.pos")) for name in self.leader_motor_names], dtype=float
        )
        gripper_raw = float(action.pop("gripper.pos"))

        needs_latch = (
            self._q_leader_ref is None or self._t_follower_ref is None or not self.use_latched_reference
        )
        if needs_latch:
            self._q_leader_ref = q_leader.copy()
            self._t_leader_ref = self.kinematics.forward_kinematics(self._q_leader_ref)
            self._t_follower_ref = _pose_dict_to_transform(observation, self.euler_order)

        t_leader_curr = self.kinematics.forward_kinematics(q_leader)
        # Delta pose expressed in the (follower-scaled) leader-reference-local frame.
        t_delta = np.linalg.inv(self._t_leader_ref) @ t_leader_curr
        t_target = self._t_follower_ref @ t_delta

        pos = t_target[:3, 3]
        rpy = Rotation.from_matrix(t_target[:3, :3]).as_euler(self.euler_order)

        action["x.pos"] = float(pos[0])
        action["y.pos"] = float(pos[1])
        action["z.pos"] = float(pos[2])
        action["roll.pos"] = float(rpy[0])
        action["pitch.pos"] = float(rpy[1])
        action["yaw.pos"] = float(rpy[2])
        action["gripper.pos"] = GRIPPER_OPEN if gripper_raw > self.gripper_open_threshold else GRIPPER_CLOSED

        return action

    def reset(self):
        self._q_leader_ref = None
        self._t_leader_ref = None
        self._t_follower_ref = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for name in self.leader_motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{name}.pos", None)
        features[PipelineFeatureType.ACTION].pop("gripper.pos", None)

        for feat in ["x.pos", "y.pos", "z.pos", "roll.pos", "pitch.pos", "yaw.pos", "gripper.pos"]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features


@ProcessorStepRegistry.register("kuka_ee_bounds_and_safety")
@dataclass
class KukaEEBoundsAndSafety(RobotActionProcessorStep):

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        pos = np.array([action["x.pos"], action["y.pos"], action["z.pos"]], dtype=float)
        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        # Check for jumps in position
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        self._last_pos = pos
        action["x.pos"], action["y.pos"], action["z.pos"] = (float(p) for p in pos)
        return action

    def reset(self):
        self._last_pos = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
