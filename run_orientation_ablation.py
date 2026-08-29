# Orientation-only ablation.
#
# Question: is orientation control itself hard to learn, or is the real
# bottleneck position convergence (which is what session 7's eval numbers
# suggest)? Phase 2 never really tests this because ORIENT reward only turns
# on inside a 3cm position gate that's rarely reached.
#
# Design: each episode, sample a fully reachable target config via FK (as in
# Phase 2). Start the arm at that SAME config but with the 4 wrist joints
# (URDF joints 6,7,8,9 -- confirmed orientation-critical, see JOINT_LIMITS_FK
# comments) re-randomised within their limits. That keeps the arm's starting
# end-effector position close to the target "for free" (only wrist joints
# moved, short lever arm) while orientation starts wrong. Reward is orientation
# error only -- no position gate, no position reward. If this converges fast,
# orientation control is learnable and position is confirmed as the real
# blocker. If it also plateaus, something is wrong with orientation reward
# shaping / action space itself, independent of position.

import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from run_phase2 import URDF_DIR, JOINT_LIMITS_FK, POS_X, POS_Y, POS_Z, quatDiff

WRIST_ANGLE_IDX = [6, 7, 8, 9]  # indices into the 10-long "angles" list (URDF joint numbers)


class OrientationAblationEnv(gym.Env):
    metadata = {"simulation_fps": 50}

    ORIENT_TOL = 1.2  # rad, same success threshold as Phase 2 (post double-cover fix)
    max_step_size = 150

    def __init__(self):
        super().__init__()
        self._physics_client_id = -1
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(10,), dtype=np.float32)
        self.step_number = 0
        self.reset()

    def _sample_reachable_config(self):
        attempts = 0
        while True:
            attempts += 1
            j = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS_FK]
            angles = [j[0], j[1], -j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8]]
            for i, a in enumerate(angles):
                self._bullet_client.resetJointState(self.robot, i, a)
            ls = self._bullet_client.getLinkState(self.robot, self.num_joints - 1)
            pos = np.array(ls[0], dtype=np.float32)
            orn = np.array(ls[1], dtype=np.float32)
            in_ws = POS_X[0] <= pos[0] <= POS_X[1] and POS_Y[0] <= pos[1] <= POS_Y[1] and POS_Z[0] <= pos[2] <= POS_Z[1]
            if in_ws or attempts >= 500:
                return angles, pos, orn

    def reset(self, seed=None, options=None):
        self.step_number = 0
        super().reset(seed=seed, options=options)

        if self._physics_client_id < 0:
            self._bullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
            self._physics_client_id = self._bullet_client._client

        self._bullet_client.resetSimulation()
        self._bullet_client.setGravity(0, 0, -9.8)
        self._bullet_client.setTimeStep(1.0 / self.metadata["simulation_fps"])
        self._bullet_client.setAdditionalSearchPath(URDF_DIR)
        self.robot = self._bullet_client.loadURDF(
            os.path.join(URDF_DIR, "RightArm.urdf"), [0, 0, 0], useFixedBase=True
        )
        self.num_joints = self._bullet_client.getNumJoints(self.robot)

        target_angles, pos_target, quat_target = self._sample_reachable_config()
        self.pos_target = pos_target
        self.Quat_t = quat_target.reshape(4, 1)

        # Start config: same as target except wrist joints randomised -> position
        # stays close "for free", orientation starts wrong.
        start_angles = list(target_angles)
        # map URDF joint idx -> JOINT_LIMITS_FK row: angles idx 6,7,8,9 -> j[5],j[6],j[7],j[8]
        wrist_limit_rows = [5, 6, 7, 8]
        for angle_idx, limit_row in zip(WRIST_ANGLE_IDX, wrist_limit_rows):
            lo, hi = JOINT_LIMITS_FK[limit_row]
            start_angles[angle_idx] = random.uniform(lo, hi)

        for i, a in enumerate(start_angles):
            self._bullet_client.resetJointState(self.robot, i, a)

        ls = self._bullet_client.getLinkState(self.robot, self.num_joints - 1)
        self.position    = np.array(ls[0], dtype=np.float32)
        self.orientation = np.array(ls[1], dtype=np.float32)
        self.Euclid_dist = float(np.linalg.norm(self.position - self.pos_target))

        qd = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(qd[0]).item())
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        return self._obs(), {}

    def _obs(self):
        qd = quatDiff(self.orientation, self.Quat_t)
        quat_diff_q = np.asarray(qd[1], dtype=np.float32).flatten()
        return np.concatenate([
            [self.Euclid_dist], quat_diff_q, self.orientation, [self.Quat_diff_angle]
        ]).astype(np.float32)

    def step(self, action):
        self.step_number += 1
        scaled = action.copy()
        scaled[0] = action[0] / 4
        for i in range(1, 10):
            scaled[i] = action[i] * np.pi
        scaled[2] = -scaled[1]

        for i in range(10):
            self._bullet_client.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL, targetPosition=scaled[i]
            )
        for _ in range(3):
            self._bullet_client.stepSimulation()

        ls = self._bullet_client.getLinkState(self.robot, self.num_joints - 1)
        self.position    = np.array(ls[0], dtype=np.float32)
        self.orientation = np.array(ls[1], dtype=np.float32)
        self.Euclid_dist = float(np.linalg.norm(self.position - self.pos_target))

        qd = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(qd[0]).item())

        # Orientation-only reward: pure progress + small absolute penalty.
        # No position term at all -- position is a free starting condition here,
        # not something the policy is being scored on.
        qn = float(np.clip(self.Quat_diff_angle / np.pi, 0, 1))
        reward = -0.05 * qn + 2.0 * float(self.prev_Quat_diff_angle - self.Quat_diff_angle)
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        terminated = self.Quat_diff_angle < self.ORIENT_TOL
        if terminated:
            reward += 20.0

        truncated = self.step_number >= self.max_step_size
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}


class AblationCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps
        self.timesteps = 0
        self.terminations = 0
        self.episode_count = 0
        self.final_angles = []
        self._ep_reward = 0.0

    def _on_step(self):
        self.timesteps += 1
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_count += 1
            env = self.training_env.envs[0].unwrapped
            self.final_angles.append(env.Quat_diff_angle)
            if env.Quat_diff_angle < env.ORIENT_TOL:
                self.terminations += 1
            if self.episode_count % 20 == 0:
                recent = self.final_angles[-20:]
                print(f"Step {self.timesteps:>7}/{self.total_timesteps_target} | "
                      f"Ep {self.episode_count:>5} | "
                      f"MeanAngle(last20) {np.mean(recent):.3f}rad | "
                      f"Successes {self.terminations}/{self.episode_count} "
                      f"({100*self.terminations/self.episode_count:.1f}%)")
            self._ep_reward = 0.0
        return True


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up orientation-only ablation environment...")
    env = Monitor(OrientationAblationEnv())
    check_env(env, warn=True)
    print("Environment OK\n")

    TOTAL_TIMESTEPS = 60000
    model = SAC("MlpPolicy", env, verbose=0)
    cb = AblationCallback(total_timesteps=TOTAL_TIMESTEPS)
    print(f"Training orientation-only ablation for {TOTAL_TIMESTEPS} timesteps...\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, log_interval=1)

    model.save("models/orientation_ablation_model")
    print(f"\nDone: {cb.terminations} successes / {cb.episode_count} episodes "
          f"({100*cb.terminations/max(cb.episode_count,1):.1f}%)")
    print("Model saved to models/orientation_ablation_model")
