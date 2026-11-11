import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class FR3ReachEnv(gym.Env):
    def __init__(self, max_steps=200):
        path = os.path.expanduser("~/Projects/mujoco_menagerie/franka_fr3/scene.xml")
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.site_id = self.model.site("attachment_site").id
        self.joint_low = self.model.jnt_range[:7, 0]
        self.joint_high = self.model.jnt_range[:7, 1]
        self.frame_skip = 25          # 25 x 0.002 s = 0.05 s per RL step
        self.max_delta = 0.05         # max joint change per step (rad)
        self.max_steps = max_steps

        # Action: 7 numbers in [-1, 1], scaled to small joint changes
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        # Observation: 7 joint angles + gripper xyz + target xyz + (target - gripper) xyz = 16
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

    def _get_obs(self):
        q = self.data.qpos[:7]
        ee = self.data.site_xpos[self.site_id]
        return np.concatenate([q, ee, self.target, self.target - ee]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # go to "home"
        self.data.ctrl[:7] = self.data.qpos[:7]                 # motors hold home
        mujoco.mj_forward(self.model, self.data)                # update positions
        # Random target in a box in front of the robot (metres)
        self.target = self.np_random.uniform([0.3, -0.3, 0.2], [0.6, 0.3, 0.6])
        self.steps = 0
        self.prev_action = np.zeros(7)
        return self._get_obs(), {}

    def step(self, action):
        # turn the action into new joint targets.
        new_target = self.data.ctrl[:7] + action * self.max_delta
        new_target = np.clip(new_target, self.joint_low, self.joint_high)
        #   clip it between self.joint_low and self.joint_high (np.clip)
        self.data.ctrl[:7] = new_target
        #   write it into self.data.ctrl[:7]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        obs = self._get_obs()
        #   reward. distance = np.linalg.norm(target - gripper position)
        #   start simple: reward = -distance
        ee_pos = self.data.site_xpos[self.site_id]
        distance = np.linalg.norm(self.target - ee_pos)
        reward = (-distance
                  - 0.01 * np.sum(action ** 2)                       # prefer small moves
                  - 0.05 * np.sum((action - self.prev_action) ** 2))  # prefer smooth moves
        self.prev_action = np.array(action)

        terminated = False                      # keep going: it must learn to STAY at the target
        truncated = bool(self.steps >= self.max_steps)

        return obs, reward, terminated, truncated, {"distance": distance}