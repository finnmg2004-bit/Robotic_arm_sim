import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class FR3PushEnv(gym.Env):
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

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(22,), dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.max_dx = 0.02
        
    def _positions(self):
        ee = self.data.site_xpos[self.ee_id].copy()
        cube = self.data.xpos[self.cube_id].copy()
        goal = self.model.site_pos[self.goal_id].copy()
        return ee, cube, goal

    def _get_obs(self):
        ee, cube, goal = self._positions()
        q = self.data.qpos[:7]
        return np.concatenate([q, ee, cube, goal, cube - ee, goal - cube]).astype(np.float32)

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

        return self._get_obs(), {}

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
        dx = action * self.max_dx                       # desired hand move (m)

        # don't let the hand dive into the table
        ee_now = self.data.site_xpos[self.ee_id]
        dx[2] = max(dx[2], 0.215 - ee_now[2])

        # classical controller: hand move -> joint targets
        q_target = self._cartesian_to_joints(dx)
        lo, hi = self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1]
        self.data.ctrl[:7] = np.clip(q_target, lo, hi)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        # ---- reward: same as before ----
        ee, cube, goal = self._positions()
        push_dir = (goal - cube)[:2] / (np.linalg.norm((goal - cube)[:2]) + 1e-6)
        push_point = cube.copy()
        push_point[:2] -= 0.06 * push_dir
        d_reach = np.linalg.norm(push_point - ee)
        d_push = np.linalg.norm((goal - cube)[:2])
        progress = self.prev_d_push - d_push
        self.prev_d_push = d_push

        reward = (-0.5 * d_reach + 50.0 * progress
                - 0.01 * np.sum(action ** 2)
                - 0.05 * np.sum((action - self.prev_action) ** 2))
        if d_push < 0.03:
            reward += 1.0

        fell_off = cube[2] < 0.15
        if fell_off:
            reward -= 2.0 * (self.max_steps - self.steps)
        terminated = bool(fell_off)
        truncated = bool(self.steps >= self.max_steps)

        self.prev_action = action.copy()
        info = {"d_push": d_push, "success": d_push < 0.03}
        return self._get_obs(), float(reward), terminated, truncated, info