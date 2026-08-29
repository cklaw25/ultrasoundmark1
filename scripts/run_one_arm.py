# One-Arm Robot RL Training
# Ported from progressnew_1-.ipynb with fixed URDF paths.
# Run from C:/ultrasoundmark1/ with the venv active.

import os
import numpy as np
import math
import random
import json
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

URDF_DIR = os.path.join(os.path.dirname(__file__),
    "ImprovedUltrasoundRLRobot-main",
    "ImprovedUltrasoundRLRobot-main",
    "One_Arm_Robot",
    "urdf"
)

# ---------------------------------------------------------------------------
# Utility functions (from notebook cell 2)
# ---------------------------------------------------------------------------

def EulerToRMat(Euler):
    sx, sy, sz = math.sin(Euler[0]), math.sin(Euler[1]), math.sin(Euler[2])
    cx, cy, cz = math.cos(Euler[0]), math.cos(Euler[1]), math.cos(Euler[2])
    R = np.eye(3)
    R[0,0]=cy*cz; R[0,1]=sx*sy*cz-cx*sz; R[0,2]=cx*sy*cz+sx*sz
    R[1,0]=cy*sz; R[1,1]=sx*sy*sz+cx*cz; R[1,2]=cx*sy*sz-sx*cz
    R[2,0]=-sy;   R[2,1]=sx*cy;           R[2,2]=cx*cy
    return R

def RMatToQuat(R):
    q = np.zeros((4,1))
    q[0] = 0.5 * math.sqrt(abs(1 + R[0,0] + R[1,1] + R[2,2]))
    q[1] = 0.5 * math.sqrt(abs(1 + R[0,0] - R[1,1] - R[2,2])) * (1 if R[2,1]-R[1,2]>0 else -1)
    q[2] = 0.5 * math.sqrt(abs(1 - R[0,0] + R[1,1] - R[2,2])) * (1 if R[0,2]-R[2,0]>0 else -1)
    q[3] = 0.5 * math.sqrt(abs(1 - R[0,0] - R[1,1] + R[2,2])) * (1 if R[1,0]-R[0,1]>0 else -1)
    return q

def normalise(x, min_val, max_val):
    return (2 * (x - min_val) / (max_val - min_val)) - 1

def quatDiff(q1, q2):
    q1 = np.array(q1).flatten()
    q2 = np.array(q2).flatten()
    qd = np.zeros((4, 1))
    qd[0] =  q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    qd[1] = -q1[1]*q2[0] + q1[0]*q2[1] - q1[3]*q2[2] + q1[2]*q2[3]
    qd[2] = -q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3]
    qd[3] = -q1[3]*q2[0] - q1[2]*q2[1] + q1[1]*q2[2] + q1[0]*q2[3]
    a = abs(2 * np.arccos(np.float32(qd[0])))
    return a, qd

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RightArmRobot(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "simulation_fps": 50}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self._physics_client_id = -1
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(19,), dtype=np.float32)
        self.pos_tol = 0.02
        self.orient_tol = 0.2
        self.step_number = 0
        self.episode_number = 0
        self.max_step_size = 300
        self.reset()

    def reset(self, seed=None, options=None):
        self.step_number = 0
        self.episode_number += 1
        super().reset(seed=seed, options=options)

        self.x_target = random.random() * (0.15 - -0.10) + (-0.10)
        self.y_target = random.random() * (0.65 - 0.3) + 0.3
        self.z_target = random.random() * (0.35 - 0.15) + 0.15
        self.pos_target = np.array([self.x_target, self.y_target, self.z_target], dtype=np.float32)

        self.x_euler_t = random.random() * (-0.785398 - -2.35619) + (-2.35619)
        self.y_euler_t = random.random() * (0.785398 - -0.785398) + (-0.785398)
        self.z_euler_t = random.random() * (math.pi - -math.pi) + (-math.pi)
        self.euler_t = np.array([self.x_euler_t, self.y_euler_t, self.z_euler_t])
        self.Quat_t = RMatToQuat(EulerToRMat(self.euler_t))

        if self._physics_client_id < 0:
            mode = p.DIRECT if self.render_mode == "human" else p.GUI
            self._bullet_client = bullet_client.BulletClient(connection_mode=mode)
            self._physics_client_id = self._bullet_client._client

        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.metadata["simulation_fps"])
        self._bullet_client.setAdditionalSearchPath(URDF_DIR)

        self.RightArmRobot = self._bullet_client.loadURDF(
            os.path.join(URDF_DIR, "RightArm.urdf"), [0, 0, 0], useFixedBase=True
        )
        if self.RightArmRobot < 0:
            raise FileNotFoundError(f"Failed to load RightArm.urdf from {URDF_DIR}")

        self.num_joints = self._bullet_client.getNumJoints(self.RightArmRobot)
        link_state = self._bullet_client.getLinkState(self.RightArmRobot, self.num_joints - 1)
        self.position = np.array(link_state[0], dtype=np.float32)
        self.orientation = np.array(link_state[1], dtype=np.float32)

        self.Euclid_dist = float(math.dist(self.position, self.pos_target))
        self.prev_Euclid_dist = self.Euclid_dist
        self.Quaternion_diff = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(self.Quaternion_diff[0]).item())
        self.Quat_diff_Q = self.Quaternion_diff[1]
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        self.pos_target_n = np.array([
            normalise(self.x_target, -0.10, 0.15),
            normalise(self.y_target, 0.3, 0.65),
            normalise(self.z_target, 0.15, 0.35)
        ], dtype=np.float32)
        self.pos_n = np.array([
            normalise(self.position[0], -0.10, 0.15),
            normalise(self.position[1], 0.3, 0.65),
            normalise(self.position[2], 0.15, 0.35)
        ], dtype=np.float32)

        obs = np.concatenate([
            [self.Euclid_dist],
            np.array([self.Quat_diff_Q], dtype=np.float32).flatten(),
            self.pos_target_n, self.pos_n,
            self.orientation, self.Quat_t.flatten()
        ]).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.step_number += 1
        scaled = action.copy()
        scaled[0] = action[0] / 4
        for i in range(1, 10):
            scaled[i] = action[i] * np.pi
        scaled[2] = -scaled[1]

        for i in range(10):
            self._bullet_client.setJointMotorControl2(
                self.RightArmRobot, i, p.POSITION_CONTROL, targetPosition=scaled[i]
            )
        for _ in range(3):
            self._bullet_client.stepSimulation()

        link_state = self._bullet_client.getLinkState(self.RightArmRobot, self.num_joints - 1)
        self.position = np.array(link_state[0], dtype=np.float32)
        self.orientation = np.array(link_state[1], dtype=np.float32)
        self.pos_n = np.array([
            normalise(self.position[0], -0.10, 0.15),
            normalise(self.position[1], 0.3, 0.65),
            normalise(self.position[2], 0.15, 0.35)
        ], dtype=np.float32)

        self.Euclid_dist = float(math.dist(self.position, self.pos_target))
        self.Quaternion_diff = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(self.Quaternion_diff[0]).item())
        self.Quat_diff_Q = self.Quaternion_diff[1]

        obs = np.concatenate([
            [self.Euclid_dist],
            np.array([self.Quat_diff_Q], dtype=np.float32).flatten(),
            self.pos_target_n, self.pos_n,
            self.orientation, self.Quat_t.flatten()
        ]).astype(np.float32)

        dn = float(np.clip(self.Euclid_dist / 0.7533, 0, 1))
        reward = -dn + 5.0 * float(self.prev_Euclid_dist - self.Euclid_dist)
        self.prev_Euclid_dist = self.Euclid_dist
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        terminated = self.Euclid_dist < self.pos_tol
        if terminated:
            reward += 100.0
        truncated = self.step_number >= self.max_step_size

        return obs, float(reward), bool(terminated), bool(truncated), {}

# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class CustomCallback(BaseCallback):
    def __init__(self, run_name="run", total_timesteps=500000, verbose=0):
        super().__init__(verbose)
        self.run_name = run_name
        self.total_timesteps_target = total_timesteps
        self.timesteps = 0
        self.episode_rewards = []
        self.terminations = 0
        self.final_Euclid_dists = []
        self.mean_rewards = []
        self.std_rewards = []
        self.num_term_per_interval = []
        self._ep_reward = 0.0
        self._ep_final_dist = 0.0

    def _on_step(self):
        self.timesteps += 1
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            info = self.locals["infos"][0]
            env = self.training_env.envs[0].unwrapped
            self.final_Euclid_dists.append(env.Euclid_dist)
            if env.Euclid_dist < env.pos_tol:
                self.terminations += 1
            self._ep_reward = 0.0
            if len(self.episode_rewards) % 10 == 0:
                recent = self.episode_rewards[-10:]
                self.mean_rewards.append(float(np.mean(recent)))
                self.std_rewards.append(float(np.std(recent)))
                print(f"Step {self.timesteps}/{self.total_timesteps_target} | "
                      f"Episodes: {len(self.episode_rewards)} | "
                      f"Mean reward (last 10): {self.mean_rewards[-1]:.2f} | "
                      f"Successes: {self.terminations}")
        return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up environment...")
    env = Monitor(RightArmRobot())
    check_env(env, warn=True)
    print("Environment OK")

    TOTAL_TIMESTEPS = 500000
    CHECKPOINT = "models/checkpoints/phase1_ckpt_200000_steps"

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/checkpoints/",
        name_prefix="phase1_ckpt",
        verbose=1,
    )

    if os.path.exists(CHECKPOINT + ".zip"):
        STEPS_DONE = 200000
        REMAINING = TOTAL_TIMESTEPS - STEPS_DONE
        print(f"Resuming from {CHECKPOINT} ({STEPS_DONE} steps done, {REMAINING} remaining)...")
        model = SAC.load(CHECKPOINT, env=env)
        cb = CustomCallback(run_name="run1", total_timesteps=TOTAL_TIMESTEPS)
        model.learn(
            total_timesteps=REMAINING,
            callback=CallbackList([cb, checkpoint_cb]),
            log_interval=1,
            reset_num_timesteps=False,
        )
    else:
        print(f"No checkpoint found, starting fresh for {TOTAL_TIMESTEPS} timesteps...")
        cb = CustomCallback(run_name="run1", total_timesteps=TOTAL_TIMESTEPS)
        model = SAC("MlpPolicy", env, verbose=0).learn(
            total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([cb, checkpoint_cb]), log_interval=1
        )

    model.save("models/run1_model")
    print("Training complete. Model saved to models/run1_model")
    p.disconnect()
