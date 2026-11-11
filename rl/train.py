from stable_baselines3 import SAC
from fr3_reach_env import FR3ReachEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = FR3ReachEnv()

model = SAC(
    "MlpPolicy",       # a plain neural network: observation in, action out
    env,
    verbose=1,         # print progress every few episodes
)

checkpoint = CheckpointCallback(save_freq=20_000, save_path="checkpoints/", name_prefix="fr3_reach")
model.learn(total_timesteps=200_000, callback=checkpoint)
model.save("fr3_reach_sac")           # saves fr3_reach_sac.zip
print("saved")