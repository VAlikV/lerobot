import numpy as np
import kuka_fri_py as fri
import time

controller = fri.KukaController(
    fri.ControlMode.JOINT_POSITION,
    "src/lerobot/robots/kuka_iiwa/iiwa.urdf",
    False,
)

controller.start()

obs = controller.get_observation()

pos = np.array([obs[7], obs[8], obs[9]], dtype=np.float64)

rot = np.array([obs[10:13],
                obs[13:16],
                obs[16:19]])

controller.set_target(pos, rot)
print(pos)
print(rot)

while 1:
    obs = controller.get_observation()
    controller.set_target(pos, rot)

    time.sleep(0.1)



controller.stop()