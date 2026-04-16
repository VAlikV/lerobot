from dataclasses import dataclass, field
from ..config import RobotConfig


@RobotConfig.register_subclass("assembling_sim")
@dataclass
class AssemblingSimConfig(RobotConfig):
    """Configuration class for simulator."""

    xml_path: str = "scene.xml",
    sim_timestep: float = 0.001,
    control_hz: int = 20,
    mode: str = "fast",   # "realtime" | "fast"
    max_episode_steps: int = 1000,
    use_task_space: bool = True,
    render_mode: str = "rgb_array",   # None | "human" | "rgb_array" | "all"
    camera_names: list[str] = field(default_factory=lambda: ["cam_front", "cam_side", "cam_gripper"])
    resolution: tuple = (224, 224)

    action_pos_scale: int = 1000,
    action_angle_scale: int = 100