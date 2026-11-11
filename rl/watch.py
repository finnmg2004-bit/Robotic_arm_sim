import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from fr3_reach_env import FR3ReachEnv

env = FR3ReachEnv()
model = SAC.load("fr3_reach_sac")      # the file train.py saves at the end
obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        # Ask the trained policy for an action (no randomness when testing)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Draw the target as a red sphere (visual only, nothing can hit it)
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.03, 0, 0]), env.target, np.eye(3).flatten(),
            np.array([1, 0, 0, 1], dtype=np.float32))

        viewer.sync()          # update the window
        time.sleep(0.05)       # one RL step = 0.05 s, so this plays in real time

        if terminated or truncated:
            print("reached" if terminated else "timed out", f"(distance {info['distance']:.3f} m)")
            obs, _ = env.reset()