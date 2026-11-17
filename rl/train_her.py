from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from fr3_push_her_env import FR3PushHEREnv

env = FR3PushHEREnv()

model = SAC(
    "MultiInputPolicy",                 # handles the dict observation
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,               # 4 fake goals per real one
        goal_selection_strategy="future",   # fake goal = where the cube ended up later
    ),
    learning_starts=1000,
    batch_size=512,
    gamma=0.95,                         # short episodes, so look less far ahead
    tau=0.05,
    learning_rate=1e-3,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    verbose=1,
)

ckpt = CheckpointCallback(save_freq=50_000, save_path="checkpoints_her/")
model.learn(total_timesteps=500_000, callback=ckpt)
model.save("fr3_push_her_sac")