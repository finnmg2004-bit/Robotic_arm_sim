# RoboArmSim: Reinforcement Learning for a Franka FR3 Arm

<!-- GIF HERE -->

Training a Franka Research 3 arm with reinforcement learning in **MuJoCo**, then running the learned policy in **Gazebo Harmonic** through **ROS 2 Jazzy** and **ros2_control**.

| Task | Method | Result |
|---|---|---|
| Reach a 3D target | SAC, joint-space actions | Smooth reaching in MuJoCo and Gazebo |
| Push a cube to a goal | SAC + Hindsight Experience Replay, Cartesian actions | **91% success** (training), <!-- TODO: deterministic eval % --> |

## How it fits together

```
        Training (MuJoCo)                          Deployment (ROS 2 + Gazebo)
 ┌──────────────────────────────┐        ┌───────────────────────────────────────────┐
 │ Gymnasium env  ◄──►  SAC     │  .zip  │ /joint_states ─► policy node ─► /arm_controller/commands
 │ (fr3 menagerie model)        │ ─────► │  (20 Hz)            ▲               │
 │                              │        │  TF fr3_link0→link8 ┘               ▼
 └──────────────────────────────┘        │            ros2_control (position) ─► Gazebo FR3
                                         └───────────────────────────────────────────┘
```

- **MuJoCo** is used for training because it is fast and deterministic.
- **Gazebo + ros2_control** is the deployment target. Because the policy talks to standard ROS 2 topics through the hardware abstraction layer, swapping Gazebo for the real robot's hardware interface would not require changing the policy node.
- The policy runs at **20 Hz** and outputs small position deltas. The low-level joint controllers run much faster and handle actually tracking those targets.

## What I learned (the interesting part)

### 1. Reach: smoothness matters for sim-to-sim transfer
The first reach policy worked in MuJoCo but **trembled in Gazebo**. MuJoCo's position actuators behave like springs and dampers, so they smooth out jittery commands. Gazebo's position interface doesn't. Two changes fixed it:
- adding action-magnitude and action-change penalties to the reward (`-0.01|a|² - 0.05|a - a_prev|²`), and removing early termination
- a low-pass filter on commands in the ROS 2 node

### 2. Push: reward hacking, three times
Joint-space actions with a hand-shaped reward kept finding shortcuts:
- **stopping just short of the cube**, a local optimum from the distance term
- **knocking the cube off the table** to end the episode early, because the termination penalty was cheaper than many steps of negative reward
- **lying the whole arm across the table** to reach a low target point, because nothing in the reward said "stay upright"

### 3. Fix: classical control underneath, RL on top
Instead of more reward tuning, I changed the **action space**. The policy now outputs a Cartesian hand motion `(dx, dy, dz)`. A damped least-squares Jacobian controller (`mujoco.mj_jacSite`) converts that into joint targets while holding the gripper pointing down:

```
Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ [Δx ; k·e_rot]
```

The arm physically can't sprawl anymore, and the policy only has to learn *where to move the hand*.

### 4. Sparse reward + Hindsight Experience Replay
With a dense reward the arm learned to sit behind the cube without pushing it. Switching to a **sparse reward** (0 at the goal, −1 otherwise) with **HER** fixed this. HER relabels failed episodes as if the goal had been wherever the cube actually ended up, so every accidental push becomes a training example.

One more catch: at first success stayed at **0%**, because the hand started 40 cm above the cube and random exploration never touched it. Starting the hand low near the table took it to **~65% by 160k steps and 91% by 500k**.

## Repository layout

```
rl/
  fr3_reach_env.py        # reach task (joint-space actions)
  fr3_push_her_env.py     # push task (Cartesian actions, goal-conditioned, HER)
  train.py / train_her.py / resume_her.py
  watch.py / watch_push.py    # MuJoCo viewer + success rate
  test_controller.py      # sanity check for the Jacobian controller
ros2_ws/src/
  arm_bringup/            # Gazebo launch, FR3 + ros2_control xacro, controller config
  arm_policy/             # ROS 2 node that runs the trained reach policy
```
<!-- TODO: check this matches your actual files; add push_scene.xml location -->

## Setup

Uses pixi (https://pixi.sh) with RoboStack (ROS 2 Jazzy, Gazebo Harmonic, Python 3.12).

```bash
git clone <this repo> && cd RoboArmSim
pixi install
pixi shell
```

The MuJoCo model comes from mujoco_menagerie (https://github.com/google-deepmind/mujoco_menagerie), `franka_fr3`. The push scene adds a table, a cube and a goal marker.
<!-- TODO: say where push_scene.xml lives / how to get it -->

## Usage

**Train and watch in MuJoCo**
```bash
cd rl
python train_her.py        # push task, ~500k steps
python watch_push.py       # viewer + success rate
```

**Run the reach policy in Gazebo**
```bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch arm_bringup spawn_franka.launch.py
# new terminal
ros2 run arm_policy reach_policy --ros-args -p use_sim_time:=true
# new terminal
ros2 topic pub --once /reach_target geometry_msgs/msg/Point "{x: 0.4, y: 0.2, z: 0.4}"
```

## Tech
ROS 2 Jazzy · Gazebo Harmonic · ros2_control · MuJoCo · Gymnasium · Stable-Baselines3 (SAC, HER) · pixi
