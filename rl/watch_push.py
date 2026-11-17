import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from fr3_push_env import FR3PushEnv

env = FR3PushEnv()
model = SAC.load("fr3_push_sac")
obs, _ = env.reset()
successes, episodes = 0, 0

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        time.sleep(0.05)

        if terminated or truncated:
            episodes += 1
            successes += info["success"]
            print(f"episode {episodes}: cube {info['d_push']*100:.1f} cm from goal, "
                  f"{'SUCCESS' if info['success'] else 'miss'}   ({successes}/{episodes} so far)")
            obs, _ = env.reset()