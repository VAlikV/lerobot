import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.robots.so_follower.config_so100_follower import SO100FollowerConfig
# from lerobot.robots.so_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.robots.assembling_sim import AssemblingSim, AssemblingSimCut, AssemblingSimConfig

import matplotlib.pyplot as plt

dataset = LeRobotDataset("local/kuka_device_assemble2_stage1_finetune")
actions = dataset.hf_dataset.select_columns("action")

x = []
y = []
z = []
r = []
p = []
yaw = []

for a in actions['action']:
    x.append(a[0].numpy())
    y.append(a[1].numpy())
    z.append(a[2].numpy())
    r.append(a[3].numpy())
    p.append(a[4].numpy())
    yaw.append(a[5].numpy())

print(min(x), max(x))
print(min(y), max(y))
print(min(z), max(z))
print(min(r), max(r))
print(min(p), max(p))
print(min(yaw), max(yaw))


plt.figure(1)
plt.hist(x, 300)

plt.figure(2)
plt.hist(y, 300)

plt.figure(3)
plt.hist(z, 300)

plt.figure(4)
plt.hist(r, 300)

plt.figure(5)
plt.hist(p, 300)

plt.figure(6)
plt.hist(yaw, 300)

plt.show()
# print(actions['action'][0][0].numpy())