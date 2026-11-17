import time
import numpy as np
import mujoco.viewer
from fr3_push_her_env import FR3PushHEREnv

env = FR3PushHEREnv()
env.reset()
moves = {"+x": [1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "down": [0, 0, -1], "up": [0, 0, 1]}

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    for name, a in moves.items():
        print(name, "start:", np.round(env.data.site_xpos[env.ee_id], 3))
        for _ in range(20):
            env.step(np.array(a, dtype=np.float32))
            viewer.sync()
            time.sleep(0.05)
        print(name, "end:  ", np.round(env.data.site_xpos[env.ee_id], 3))