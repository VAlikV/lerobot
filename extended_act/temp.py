
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/kuka_stage_1_gray_geometry")

print(dataset[0]['observation.geometry'])

