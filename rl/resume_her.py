from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from fr3_push_her_env import FR3PushHEREnv

env = FR3PushHEREnv()
model = SAC.load("checkpoints_her/rl_model_200000_steps.zip", env=env)   # your latest
ckpt = CheckpointCallback(save_freq=50_000, save_path="checkpoints_her/",
                          save_replay_buffer=True)    # buffer saved from now on
model.learn(total_timesteps=300_000, callback=ckpt, reset_num_timesteps=False)
model.save("fr3_push_her_sac")