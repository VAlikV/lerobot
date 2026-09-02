from lerobot.teleoperators.quest3 import (
    QuestRos2,
    QuestRos2Config,
)
import time

HZ = 30

config = QuestRos2Config(
    id="quest3_right",

    position_scale=1.0,
    rotation_scale=1.0,
)

teleoperator = QuestRos2(config)
teleoperator.connect()

t = time.time()

try:
    while True:
        action = teleoperator.get_action()
        print(action)

        delta = time.time() - t
        if 1/HZ - delta > 0:
            time.sleep(1/HZ - delta)
        t = time.time()

finally:
    teleoperator.disconnect()