import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class FR3PushHEREnv(gym.Env):
    def __init__(self, max_steps=300):
        path = os.path.expanduser("~/Projects/mujoco_menagerie/franka_fr3/push_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.ee_id = self.model.site("attachment_site").id
        self.goal_id = self.model.site("goal").id
        self.cube_id = self.model.body("cube").id
        self.joint_low = self.model.jnt_range[:7, 0]
        self.joint_high = self.model.jnt_range[:7, 1]
        self.frame_skip = 25
        self.max_delta = 0.05
        self.max_steps = max_steps

        self.observation_space = spaces.Dict({
        "observation":   spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(2,),  dtype=np.float64),
        "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(2,),  dtype=np.float64),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.max_dx = 0.02
        self.max_steps = 100
        
    def _positions(self):
        ee = self.data.site_xpos[self.ee_id].copy()
        cube = self.data.xpos[self.cube_id].copy()
        goal = self.model.site_pos[self.goal_id].copy()
        return ee, cube, goal

    def _get_obs(self):
        q = self.data.qpos[:7]
        ee, cube, goal = self._positions()
        return {
            "observation":   np.concatenate([q, ee, cube, cube - ee]),
            "achieved_goal": cube[:2].copy(),
            "desired_goal":  goal[:2].copy(),
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # home

        # random cube position, flat on the table
        self.data.qpos[7] = self.np_random.uniform(0.40, 0.55)
        self.data.qpos[8] = self.np_random.uniform(-0.15, 0.15)
        self.data.qpos[9] = 0.225
        self.data.qpos[10:14] = [1, 0, 0, 0]

        # random goal, at least 10 cm from the cube
        cube_xy = self.data.qpos[7:9]
        while True:
            goal_xy = np.array([self.np_random.uniform(0.40, 0.65),
                                self.np_random.uniform(-0.20, 0.20)])
            if np.linalg.norm(goal_xy - cube_xy) > 0.10:
                break
        self.model.site_pos[self.goal_id][:2] = goal_xy

        mujoco.mj_forward(self.model, self.data)   # update positions after editing qpos

        # NEW: remember "hand pointing down" so the controller can hold it
        self.R_des = self.data.site_xmat[self.ee_id].reshape(3, 3).copy()

        self.prev_action = np.zeros(3)             # NEW: 3, not 7
        self.steps = 0
        cube = self.data.xpos[self.cube_id]
        self.prev_d_push = np.linalg.norm((self.model.site_pos[self.goal_id] - cube)[:2])
        self._move_hand_to(np.array([0.32, 0.0, 0.625]))   # pull back at height
        self._move_hand_to(np.array([0.32, 0.0, 0.24]))    # then drop to push height
        return self._get_obs(), {}

    def _move_hand_to(self, target, n=25):
        lo, hi = self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1]
        for _ in range(n):
            ee = self.data.site_xpos[self.ee_id]
            dx = np.clip(target - ee, -self.max_dx, self.max_dx)
            self.data.ctrl[:7] = np.clip(self._cartesian_to_joints(dx), lo, hi)
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)

    def _cartesian_to_joints(self, dx):
        site = self.ee_id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])        # 6x7, arm joints only

        R = self.data.site_xmat[site].reshape(3, 3)
        rot_err = 0.5 * sum(np.cross(R[:, i], self.R_des[:, i]) for i in range(3))

        err = np.concatenate([dx, 1.0 * rot_err])        # move + hold orientation
        lam = 0.05                                        # damping, avoids blow-ups
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
        return self.data.qpos[:7] + dq

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        dx = action * self.max_dx
        ee_now = self.data.site_xpos[self.ee_id]
        dx[2] = max(dx[2], 0.215 - ee_now[2])

        q_target = self._cartesian_to_joints(dx)
        lo, hi = self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1]
        self.data.ctrl[:7] = np.clip(q_target, lo, hi)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        obs = self._get_obs()
        d = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        info = {"is_success": float(d < 0.03), "d_push": d}
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))

        terminated = False                       # never end early, see below
        truncated = bool(self.steps >= self.max_steps)
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > 0.03).astype(np.float32)     # 0 = success, -1 = not yet