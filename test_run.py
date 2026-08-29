# Quick 2000-step smoke test — confirms the full training loop works
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
os.chdir(r"C:\ultrasoundmark1")

import numpy as np
import math
import random
import json

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

URDF_DIR = os.path.join(
    r"C:\ultrasoundmark1",
    "ImprovedUltrasoundRLRobot-main",
    "ImprovedUltrasoundRLRobot-main",
    "One_Arm_Robot",
    "urdf"
)

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
    qd = np.zeros((4,1))
    qd[0] =  q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
    qd[1] = -q1[1]*q2[0] + q1[0]*q2[1] - q1[3]*q2[2] + q1[2]*q2[3]
    qd[2] = -q1[2]*q2[0] + q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3]
    qd[3] = -q1[3]*q2[0] - q1[2]*q2[1] + q1[1]*q2[2] + q1[0]*q2[3]
    a = abs(2 * np.arccos(np.float32(qd[0])))
    return a, qd

class RightArmRobot(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "simulation_fps": 50}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self._physics_client_id = -1
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(19,), dtype=np.float32)
        self.pos_tol = 0.02
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
        self.Quat_t = RMatToQuat(EulerToRMat(
            np.array([self.x_euler_t, self.y_euler_t, self.z_euler_t])
        ))

        if self._physics_client_id < 0:
            self._bullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
            self._physics_client_id = self._bullet_client._client

        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.metadata["simulation_fps"])
        self._bullet_client.setAdditionalSearchPath(URDF_DIR)

        self.RightArmRobot = self._bullet_client.loadURDF(
            os.path.join(URDF_DIR, "RightArm.urdf"), [0, 0, 0], useFixedBase=True
        )

        self.num_joints = self._bullet_client.getNumJoints(self.RightArmRobot)
        link_state = self._bullet_client.getLinkState(self.RightArmRobot, self.num_joints - 1)
        self.position = np.array(link_state[0], dtype=np.float32)
        self.orientation = np.array(link_state[1], dtype=np.float32)

        self.Euclid_dist = float(math.dist(self.position, self.pos_target))
        self.prev_Euclid_dist = self.Euclid_dist
        self.Quaternion_diff = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(self.Quaternion_diff[0]).item())
        self.Quat_diff_Q = self.Quaternion_diff[1]

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

        terminated = self.Euclid_dist < self.pos_tol
        if terminated:
            reward += 100.0
        truncated = self.step_number >= self.max_step_size

        return obs, float(reward), bool(terminated), bool(truncated), {}

class SimpleCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = 0
        self.ep_reward = 0.0

    def _on_step(self):
        self.ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episodes += 1
            env = self.training_env.envs[0].unwrapped
            print(f"Episode {self.episodes:3d} | steps={self.num_timesteps:5d} | reward={self.ep_reward:7.2f} | dist={env.Euclid_dist:.4f}m", flush=True)
            self.ep_reward = 0.0
        return True

if __name__ == "__main__":
    print("Setting up environment...", flush=True)
    env = Monitor(RightArmRobot())
    check_env(env, warn=True)
    print("Environment OK — starting 2000-step test...", flush=True)

    model = SAC("MlpPolicy", env, verbose=0).learn(
        total_timesteps=2000, callback=SimpleCallback(), log_interval=1
    )
    print("Test complete — training loop works!", flush=True)
    p.disconnect()
