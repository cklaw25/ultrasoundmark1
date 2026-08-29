# Phase 2: Combined position + orientation training
# Key fix: FK-based target sampling — targets are always physically reachable.
# Original failure: Euler-space target sampling hit unreachable orientations due to
# joint 6 ([1.57, 3.58] rad) and joint 8 ([-2.43, -0.70] rad) having asymmetric limits.
# This script uses PyBullet FK to sample valid configs and derive targets from them.

import os
import numpy as np
import math
import random

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

# URDF joint limits (joint 2 excluded — always = -joint 1 due to mechanical coupling)
JOINT_LIMITS_FK = [
    (-0.5000,  0.5000),   # joint 0: prismatic (gantry height)
    (-0.3142,  0.7400),   # joint 1
    #                       joint 2 = -joint 1 (derived)
    (-1.8326,  1.0123),   # joint 3
    (-1.8151,  2.7576),   # joint 4
    (-3.0107,  2.9671),   # joint 5
    ( 1.5708,  3.5779),   # joint 6  ← positive-only: original sampling ignored this
    (-1.5708,  1.5882),   # joint 7
    (-2.4260, -0.6981),   # joint 8  ← negative-only: original sampling ignored this
    (-3.1416,  3.1416),   # joint 9
]

# Clinical workspace position bounds (same as Phase 1)
POS_X = (-0.10, 0.15)
POS_Y = ( 0.30, 0.65)
POS_Z = ( 0.15, 0.35)


# ---------------------------------------------------------------------------
# Utility functions (identical to Phase 1)
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
    # Quaternions have double cover: q and -q represent the same rotation, so the
    # dot product can come back negative for two orientations that are actually
    # aligned. Take abs() BEFORE arccos (not after) to collapse both signs onto
    # the correct true angular distance, and clip first to avoid NaN from
    # floating-point rounding pushing the value fractionally outside [-1,1].
    a = 2 * np.arccos(float(np.clip(abs(qd[0]), -1.0, 1.0)))
    return a, qd


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RightArmRobotPhase2(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "simulation_fps": 50}

    # Phase 2 reward hyperparameters
    POS_GATE       = 0.03   # orientation reward only kicks in once within this distance
    ORIENT_TOL     = 1.2    # success: angle error must be below this (rad)
    ORIENT_W       = 0.05   # weight on orientation error penalty
    ORIENT_PROG_W  = 0.50   # weight on orientation progress bonus
    DRIFT_PENALTY  = 10.0   # penalty if arm drifts back out of position gate

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

    def _sample_reachable_pose(self):
        """
        Sample a random joint config (all within URDF limits), compute FK via PyBullet,
        return end-effector (pos, orn). Rejection-sample until pos is in clinical workspace.
        This guarantees both position and orientation targets are physically reachable.
        """
        attempts = 0
        while True:
            attempts += 1
            j = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS_FK]
            # Build full 10-joint list; joint 2 = -joint 1
            angles = [j[0], j[1], -j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8]]
            for i, a in enumerate(angles):
                self._bullet_client.resetJointState(self.RightArmRobot, i, a)
            ls = self._bullet_client.getLinkState(self.RightArmRobot, self.num_joints - 1)
            pos = np.array(ls[0], dtype=np.float32)
            orn = np.array(ls[1], dtype=np.float32)
            in_workspace = (
                POS_X[0] <= pos[0] <= POS_X[1] and
                POS_Y[0] <= pos[1] <= POS_Y[1] and
                POS_Z[0] <= pos[2] <= POS_Z[1]
            )
            if in_workspace or attempts >= 500:
                return pos, orn, attempts

    def reset(self, seed=None, options=None):
        self.step_number = 0
        self.episode_number += 1
        super().reset(seed=seed, options=options)

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

        # Sample reachable target via FK, then reset joints to zero for episode start
        pos_target, quat_target, _ = self._sample_reachable_pose()
        self.pos_target = pos_target
        self.x_target, self.y_target, self.z_target = pos_target
        self.Quat_t = quat_target.reshape(4, 1)
        for i in range(self.num_joints):
            self._bullet_client.resetJointState(self.RightArmRobot, i, 0.0)

        link_state = self._bullet_client.getLinkState(self.RightArmRobot, self.num_joints - 1)
        self.position    = np.array(link_state[0], dtype=np.float32)
        self.orientation = np.array(link_state[1], dtype=np.float32)

        self.Euclid_dist = float(math.dist(self.position, self.pos_target))
        self.prev_Euclid_dist = self.Euclid_dist
        qd = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(qd[0]).item())
        self.Quat_diff_Q     = qd[1]
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        self.pos_target_n = np.array([
            normalise(self.x_target, *POS_X),
            normalise(self.y_target, *POS_Y),
            normalise(self.z_target, *POS_Z),
        ], dtype=np.float32)
        self.pos_n = np.array([
            normalise(self.position[0], *POS_X),
            normalise(self.position[1], *POS_Y),
            normalise(self.position[2], *POS_Z),
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
        # /2 (not /4) so the full [-1,1] action range covers the joint's full
        # physical +-0.5m travel -- /4 silently capped it to +-0.25m (half the
        # rail permanently unreachable by any policy). See conversation log.
        scaled[0] = action[0] / 2
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
        self.position    = np.array(link_state[0], dtype=np.float32)
        self.orientation = np.array(link_state[1], dtype=np.float32)
        self.pos_n = np.array([
            normalise(self.position[0], *POS_X),
            normalise(self.position[1], *POS_Y),
            normalise(self.position[2], *POS_Z),
        ], dtype=np.float32)

        self.Euclid_dist = float(math.dist(self.position, self.pos_target))
        qd = quatDiff(self.orientation, self.Quat_t)
        self.Quat_diff_angle = float(np.asarray(qd[0]).item())
        self.Quat_diff_Q     = qd[1]

        obs = np.concatenate([
            [self.Euclid_dist],
            np.array([self.Quat_diff_Q], dtype=np.float32).flatten(),
            self.pos_target_n, self.pos_n,
            self.orientation, self.Quat_t.flatten()
        ]).astype(np.float32)

        # Position reward (same as Phase 1)
        dn = float(np.clip(self.Euclid_dist / 0.7533, 0, 1))
        reward = -dn + 8.0 * float(self.prev_Euclid_dist - self.Euclid_dist)

        # Orientation reward: only active once inside the position gate
        in_gate = self.Euclid_dist < self.POS_GATE
        if in_gate:
            # angle is now bounded to [0, pi] post double-cover fix (was normalised
            # against 2*pi to match the old buggy metric's inflated range)
            qn = float(np.clip(self.Quat_diff_angle / np.pi, 0, 1))
            reward += -self.ORIENT_W * qn
            reward += self.ORIENT_PROG_W * float(self.prev_Quat_diff_angle - self.Quat_diff_angle)

        # Penalise drifting back outside the position gate while tuning orientation
        if self.prev_Euclid_dist < self.POS_GATE and not in_gate:
            reward -= self.DRIFT_PENALTY

        self.prev_Euclid_dist     = self.Euclid_dist
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        terminated = (self.Euclid_dist < self.pos_tol and
                      self.Quat_diff_angle < self.ORIENT_TOL)
        if terminated:
            reward += 100.0

        truncated = self.step_number >= self.max_step_size

        return obs, float(reward), bool(terminated), bool(truncated), {}


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class Phase2Callback(BaseCallback):
    def __init__(self, run_name="phase2", total_timesteps=500000, verbose=0):
        super().__init__(verbose)
        self.run_name = run_name
        self.total_timesteps_target = total_timesteps
        self.timesteps   = 0
        self.terminations = 0
        self.episode_rewards = []
        self.final_dists  = []
        self.final_angles = []
        self._ep_reward   = 0.0

    def _on_step(self):
        self.timesteps += 1
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            env = self.training_env.envs[0].unwrapped
            self.final_dists.append(env.Euclid_dist)
            self.final_angles.append(env.Quat_diff_angle)
            if env.Euclid_dist < env.pos_tol and env.Quat_diff_angle < env.ORIENT_TOL:
                self.terminations += 1
            self._ep_reward = 0.0
            if len(self.episode_rewards) % 10 == 0:
                recent = self.episode_rewards[-10:]
                print(f"Step {self.timesteps:>7}/{self.total_timesteps_target} | "
                      f"Ep {len(self.episode_rewards):>5} | "
                      f"Reward {np.mean(recent):>8.2f} | "
                      f"Dist {self.final_dists[-1]:.4f}m | "
                      f"Angle {self.final_angles[-1]:.3f}rad | "
                      f"Successes {self.terminations}")
        return True

    def _on_training_end(self):
        os.makedirs("results", exist_ok=True)
        np.savetxt(f"results/{self.run_name}_rewards.txt", np.array(self.episode_rewards))
        np.savetxt(f"results/{self.run_name}_dists.txt",   np.array(self.final_dists))
        np.savetxt(f"results/{self.run_name}_angles.txt",  np.array(self.final_angles))
        n_ep = len(self.episode_rewards)
        print(f"\nPhase 2 done: {self.terminations} successes / {n_ep} episodes "
              f"({100*self.terminations/max(n_ep,1):.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up Phase 2 environment (FK target sampling)...")
    env = Monitor(RightArmRobotPhase2())
    check_env(env, warn=True)
    print("Environment OK\n")

    # Quick sampling efficiency check: how many FK attempts does it take on average?
    raw_env = env.unwrapped
    attempts_list = []
    for _ in range(50):
        raw_env._bullet_client.resetSimulation()
        raw_env._bullet_client.setGravity(0, 0, -9.8)
        raw_env._bullet_client.setAdditionalSearchPath(URDF_DIR)
        raw_env.RightArmRobot = raw_env._bullet_client.loadURDF(
            os.path.join(URDF_DIR, "RightArm.urdf"), [0, 0, 0], useFixedBase=True
        )
        raw_env.num_joints = raw_env._bullet_client.getNumJoints(raw_env.RightArmRobot)
        _, _, attempts = raw_env._sample_reachable_pose()
        attempts_list.append(attempts)
    print(f"FK sampling efficiency: avg {np.mean(attempts_list):.1f} attempts per target "
          f"(max {max(attempts_list)}, fallback threshold: 500)\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/checkpoints_phase2_extended/",
        name_prefix="phase2_ext_ckpt",
        verbose=1,
    )

    RESUME_CKPT = "models/phase2_fk_targets_model"
    if os.path.exists(RESUME_CKPT + ".zip"):
        REMAINING = 200000
        print(f"Resuming Phase 2 (extended run, pos progress mult 8.0) from {RESUME_CKPT} ...")
        model = SAC.load(RESUME_CKPT, env=env)
        cb = Phase2Callback(run_name="phase2_extended", total_timesteps=REMAINING)
        print(f"Training remaining {REMAINING} timesteps...\n")
        model.learn(total_timesteps=REMAINING, callback=CallbackList([cb, checkpoint_cb]),
                    log_interval=1, reset_num_timesteps=False)
    else:
        TOTAL_TIMESTEPS = 500000
        cb = Phase2Callback(run_name="phase2_fk_targets", total_timesteps=TOTAL_TIMESTEPS)
        phase1_path = "models/run1_model"
        if os.path.exists(phase1_path + ".zip"):
            print(f"Loading Phase 1 model from {phase1_path} ...")
            model = SAC.load(phase1_path, env=env)
            print("Phase 1 weights loaded — continuing into Phase 2.\n")
        else:
            print("No Phase 1 model found — training Phase 2 from scratch.\n")
            model = SAC("MlpPolicy", env, verbose=0)
        print(f"Training for {TOTAL_TIMESTEPS} timesteps...\n")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([cb, checkpoint_cb]), log_interval=1)

    model.save("models/phase2_extended_model")
    print("Model saved to models/phase2_extended_model")
