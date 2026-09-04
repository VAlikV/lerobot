"""
Standalone UR10e robot for relative-action ACT — rc10_follower style.

A standard lerobot `Robot` subclass backed by the process-based OSC torque
controller (`ur10_osc_controller.UR10OSCController`). It plugs straight into the
standard lerobot ACT path (LeRobotDataset record + vanilla ACT train + simple
get_observation→policy→send_action inference) — NO gym_manipulator / UR10RobotEnv.

ACTION SPACE = RELATIVE to the per-episode home pose:
    action = [x.pos, y.pos, z.pos, yaw.pos, gripper.pos]
      x/y/z  : target position - home position (metres)
      yaw    : target yaw offset from the home wrist orientation (radians)
      gripper: 1.0 = open, 0.0 = closed

The relative<->absolute round-trip lives ENTIRELY in this robot:
  - capture_home()      : snapshot home_xyz / home_rot (call at each episode start)
  - get_observation()   : returns state RELATIVE to home (tcp_xyz - home, yaw offset)
  - send_action(rel)    : abs = home + rel  -> controller.set_target (absolute)

So record and eval scripts only ever deal in relative targets — no manual
+initial_tcp anywhere.

Observation.state (11-D): [joint_pos(6), tcp_xyz_rel(3), yaw_offset(1), gripper(1)]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_not_connected

from ur10_osc_controller import UR10OSCController
from ur10_servol_backend import UR10ServoLBackend


@RobotConfig.register_subclass("ur10_follower")
@dataclass
class UR10FollowerConfig(RobotConfig):
    """Config for the standalone UR10e ACT robot (OSC backend)."""

    ip: str = "192.168.0.100"
    frequency: int = 500

    # Control backend: "osc" = compliant task-space torque (UR10OSCController), used for
    # the contact-rich press stage; "servol" = UR's stiff position controller
    # (UR10ServoLBackend), used for free-space pick/place/transport. Both expose the same
    # send_action/get_observation interface, so datasets + ACT training are identical
    # across backends — pick per stage. The two control modes are mutually exclusive on
    # the robot; only the selected backend is instantiated.
    control_backend: str = "osc"

    # OSC gains (finalized on hardware: pure PD, no integral/DOB). Used when control_backend="osc".
    kp_pos: float = 5000.0
    kp_rot: float = 100.0
    damping_ratio_pos: float = 1.0
    damping_ratio_rot: float = 1.0
    error_delta_pos: float = 0.05
    error_delta_rot: float = 0.3

    # servoL streaming params. Used when control_backend="servol". Conservative defaults
    # (gain at floor, lookahead > stream dt) — lower gain first when tuning, don't raise it.
    stream_frequency_hz: int = 200
    servo_lookahead_time: float = 0.15
    servo_gain: float = 100.0
    reset_speed: float = 0.1          # m/s for the blocking moveL reset
    reset_acceleration: float = 0.1   # m/s^2 for the blocking moveL reset

    # Tool / payload. Default: trust the pendant payload (Payload->Measure).
    tcp_offset: list[float] | None = None
    set_payload: bool = False
    payload_mass: float = 1.3
    payload_cog: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Gripper (our custom serial gripper, driven inside the controller process).
    use_gripper: bool = True
    gripper_port: str = "/dev/ttyACM0"
    gripper_baudrate: int = 115200

    # Wrist yaw DOF.
    use_yaw: bool = True

    # Cameras + image shaping. crop_boxes: name -> (top, left, height, width); applied
    # in get_observation BEFORE resize to `resolution`. Empty box / missing key = no crop.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    resolution: tuple[int, int] = (224, 224)
    crop_boxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    # Absolute workspace clip (safety) applied to the resolved absolute xyz target.
    ee_bounds_min: tuple[float, float, float] = (-0.6, -0.7, 0.05)
    ee_bounds_max: tuple[float, float, float] = (0.6, 0.7, 0.7)
    # Relative-yaw clip (radians) around the home orientation.
    yaw_min: float = -1.5708
    yaw_max: float = 1.5708

    # Optional moveJ to a known joint config on connect (radians).
    joints_init: list[float] | None = None
    soft_real_time: bool = True

    # Programmatic reset home: absolute TCP [x, y, z, rx, ry, rz]. When set, go_to_home()
    # drives the OSC target here (+ optional randomization) and settles, then anchors.
    # None -> go_to_home() just anchors at the current pose (manual reposition).
    home_tcp: list[float] | None = None
    reset_settle_s: float = 3.0
    randomization_xy: float = 0.0   # m, uniform [-r, r] on x and y
    randomization_z: float = 0.0    # m, uniform [-r, r] on z
    randomization_yaw: float = 0.0  # rad, uniform [-r, r] composed on home orientation

    # Grip-at-start: open the gripper at reset so the operator re-grips the object during
    # the reset window (matches the old PCB flow). False keeps the current gripper state.
    open_gripper_on_reset: bool = True


class UR10Follower(Robot):
    config_class = UR10FollowerConfig
    name = "ur10_follower"

    def __init__(self, config: UR10FollowerConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._controller: UR10OSCController | UR10ServoLBackend | None = None

        # Per-episode home (captured by capture_home()).
        self.home_xyz = np.zeros(3)
        self.home_rot = Rot.identity()
        # Commanded gripper state (binary); the controller owns the serial gripper, so
        # we report the last commanded state in the observation.
        self.gripper_is_open = True

        self.ee_min = np.array(config.ee_bounds_min, dtype=float)
        self.ee_max = np.array(config.ee_bounds_max, dtype=float)
        self._joint_keys = [f"joint_{i}.pos" for i in range(6)]

    # -- features -----------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict:
        feats: dict = {k: float for k in self._joint_keys}     # joint_0..5 .pos
        feats |= {"x.pos": float, "y.pos": float, "z.pos": float}  # tcp_xyz_rel
        feats["yaw.pos"] = float                                # yaw_offset
        feats["gripper.pos"] = float
        for cam_name in self.cameras:
            feats[cam_name] = (self.config.resolution[0], self.config.resolution[1], 3)
        return feats

    @cached_property
    def action_features(self) -> dict:
        # RELATIVE target. Same keys as RC10FollowerCut but relative-to-home semantics.
        return {
            "x.pos": float, "y.pos": float, "z.pos": float,
            "yaw.pos": float, "gripper.pos": float,
        }

    # -- lifecycle ----------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return (
            self._controller is not None
            and self._controller.is_alive()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("UR10Follower is already connected.")

        backend = self.config.control_backend.lower()
        if backend == "osc":
            self._controller = UR10OSCController(
                self.config.ip, frequency=self.config.frequency,
                kp_pos=self.config.kp_pos, kp_rot=self.config.kp_rot,
                damping_ratio_pos=self.config.damping_ratio_pos,
                damping_ratio_rot=self.config.damping_ratio_rot,
                error_delta_pos=self.config.error_delta_pos,
                error_delta_rot=self.config.error_delta_rot,
                tcp_offset=self.config.tcp_offset,
                set_payload=self.config.set_payload,
                payload_mass=self.config.payload_mass, payload_cog=self.config.payload_cog,
                use_gripper=self.config.use_gripper, gripper_port=self.config.gripper_port,
                gripper_baudrate=self.config.gripper_baudrate,
                joints_init=self.config.joints_init,
                soft_real_time=self.config.soft_real_time,
            )
        elif backend == "servol":
            self._controller = UR10ServoLBackend(
                self.config.ip, frequency=self.config.frequency,
                tcp_offset=self.config.tcp_offset,
                set_payload=self.config.set_payload,
                payload_mass=self.config.payload_mass, payload_cog=self.config.payload_cog,
                use_gripper=self.config.use_gripper, gripper_port=self.config.gripper_port,
                gripper_baudrate=self.config.gripper_baudrate,
                joints_init=self.config.joints_init,
                stream_frequency_hz=self.config.stream_frequency_hz,
                servo_lookahead_time=self.config.servo_lookahead_time,
                servo_gain=self.config.servo_gain,
                reset_speed=self.config.reset_speed,
                reset_acceleration=self.config.reset_acceleration,
            )
        else:
            raise ValueError(
                f"Unknown control_backend {self.config.control_backend!r}; "
                "expected 'osc' or 'servol'."
            )
        self._controller.start(wait=True)

        self._reset_realsense()
        for cam in self.cameras.values():
            cam.connect()

        self.capture_home()

    def _reset_realsense(self) -> None:
        """Hardware-reset configured RealSense devices before opening them. D405s often
        return zero/black frames until reset (same preflight the old ur10_robot ran)."""
        serials = [
            getattr(cfg, "serial_number_or_name", None)
            for cfg in self.config.cameras.values()
            if getattr(cfg, "type", None) == "intelrealsense"
        ]
        serials = [s for s in serials if s]
        if not serials:
            return
        try:
            import pyrealsense2 as rs
        except Exception:
            return
        ctx = rs.context()
        reset_any = False
        for dev in ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) in serials:
                dev.hardware_reset()
                reset_any = True
        if reset_any:
            time.sleep(5.0)

    def disconnect(self) -> None:
        if self._controller is not None:
            self._controller.stop(wait=True)
            self._controller = None
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

    # -- relative-home helpers ---------------------------------------------
    def capture_home(self) -> None:
        """Snapshot the current TCP as this episode's home (relative-action anchor)."""
        tcp = np.array(self._controller.get_current_tcp(), dtype=float)
        self.home_xyz = tcp[:3].copy()
        self.home_rot = Rot.from_rotvec(tcp[3:6])

    def get_current_tcp(self) -> np.ndarray:
        return np.array(self._controller.get_current_tcp(), dtype=float)

    @property
    def safety_status(self) -> int:
        """0=ok, 1=protective stop, 2=emergency stop (from the controller process)."""
        if self._controller is None:
            return 0
        return int(self._controller.get_state().get("safety", 0))

    @property
    def controller_alive(self) -> bool:
        return self._controller is not None and self._controller.is_alive()

    @check_if_not_connected
    def go_to_home(self, rng: np.random.Generator | None = None) -> None:
        """Reset: drive the OSC target to config.home_tcp (+ optional randomization),
        settle, then capture_home() so relative actions/obs anchor at the start pose.

        If home_tcp is None, just anchor at the current pose (manual reposition mode).
        The home target is driven by the impedance controller in task space — no
        moveJ/servo mode switch (avoids the old wedge path).
        """
        if self.config.home_tcp is None:
            self.capture_home()
            return

        rng = rng if rng is not None else np.random.default_rng()
        home = np.array(self.config.home_tcp, dtype=float)  # [x,y,z,rx,ry,rz]
        if self.config.randomization_xy > 0:
            r = self.config.randomization_xy
            home[0] += float(rng.uniform(-r, r))
            home[1] += float(rng.uniform(-r, r))
        if self.config.randomization_z > 0:
            home[2] += float(rng.uniform(-self.config.randomization_z, self.config.randomization_z))
        home[:3] = np.clip(home[:3], self.ee_min, self.ee_max)

        R_home = Rot.from_rotvec(home[3:6])
        if self.config.use_yaw and self.config.randomization_yaw > 0:
            r = self.config.randomization_yaw
            R_home = R_home * Rot.from_euler("z", float(rng.uniform(-r, r)))
        rotvec = R_home.as_rotvec()

        # Grip-at-start: open the gripper on reset (release the object) so the operator
        # re-grips during the reset window; else preserve the current gripper state.
        close = False if self.config.open_gripper_on_reset else (not self.gripper_is_open)
        # move_to_pose: OSC drives the target via task-space PD (gentle, non-blocking);
        # servoL runs a speed/accel-limited blocking moveL. Either way the settle sleep
        # below is the operator's reposition/grip window.
        self._controller.move_to_pose(
            [home[0], home[1], home[2], rotvec[0], rotvec[1], rotvec[2]], close_gripper=close
        )
        self.gripper_is_open = not close
        time.sleep(self.config.reset_settle_s)  # let the impedance drive + settle
        self.capture_home()

    # -- observation / action ----------------------------------------------
    @check_if_not_connected
    def get_observation(self) -> dict:
        st = self._controller.get_state()
        q = st["q"]
        tcp = st["tcp"]
        tcp_xyz_rel = tcp[:3] - self.home_xyz
        yaw_offset = float((self.home_rot.inv() * Rot.from_rotvec(tcp[3:6])).as_euler("zyx")[0])

        obs: dict = {self._joint_keys[i]: float(q[i]) for i in range(6)}
        obs["x.pos"] = float(tcp_xyz_rel[0])
        obs["y.pos"] = float(tcp_xyz_rel[1])
        obs["z.pos"] = float(tcp_xyz_rel[2])
        obs["yaw.pos"] = yaw_offset
        obs["gripper.pos"] = 1.0 if self.gripper_is_open else 0.0

        for cam_name, cam in self.cameras.items():
            image = cam.async_read()
            box = self.config.crop_boxes.get(cam_name)
            if box:
                t, l, h, w = box
                image = image[t:t + h, l:l + w]
            image = cv2.resize(image, (self.config.resolution[1], self.config.resolution[0]))
            obs[cam_name] = image
        return obs

    @check_if_not_connected
    def send_action(self, action: dict) -> dict:
        """Apply a RELATIVE target. Converts to absolute (home + rel) and commands
        the OSC controller. `action` keys: x.pos, y.pos, z.pos, yaw.pos, gripper.pos."""
        rel = np.array([action["x.pos"], action["y.pos"], action["z.pos"]], dtype=float)
        abs_xyz = np.clip(self.home_xyz + rel, self.ee_min, self.ee_max)

        if self.config.use_yaw:
            yaw = float(np.clip(action.get("yaw.pos", 0.0), self.config.yaw_min, self.config.yaw_max))
            R_target = self.home_rot * Rot.from_euler("z", yaw)
        else:
            R_target = self.home_rot
        rotvec = R_target.as_rotvec()

        grip_open = float(action.get("gripper.pos", 1.0)) >= 0.5
        self._controller.set_target(
            [abs_xyz[0], abs_xyz[1], abs_xyz[2], rotvec[0], rotvec[1], rotvec[2]],
            close_gripper=not grip_open,
        )
        self.gripper_is_open = grip_open
        return action
