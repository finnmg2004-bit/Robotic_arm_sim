from stable_baselines3 import SAC
from fr3_push_env import FR3PushEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = FR3PushEnv()

model = SAC(
    "MlpPolicy",       # a plain neural network: observation in, action out
    env,
    verbose=1,         # print progress every few episodes
)

checkpoint = CheckpointCallback(save_freq=50_000, save_path="checkpoints_push/", name_prefix="fr3_push")
model.learn(total_timesteps=500_000, callback=checkpoint)
model.save("fr3_push_sac")           # saves fr3_reach_sac.zip
print("saved")