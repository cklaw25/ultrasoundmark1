# Orientation-first phase (standalone duplicate — does NOT modify run_phase2.py).
#
# Tests the user's proposed curriculum: train orientation first, then resume
# into the full position+orientation task afterward.
#
# Unlike run_orientation_ablation.py (which used a smaller, incompatible
# observation space purely to get a fast yes/no diagnostic), this script uses
# the EXACT SAME 19-dim observation space and 10-dim action space as
# RightArmRobotPhase2 in run_phase2.py, so the resulting model can be loaded
# directly with SAC.load(...) and continued into the real Phase 2 environment
# for the "then train position on top" half of the experiment.
#
# Design: each episode, sample a fully reachable target config via FK (same
# method as Phase 2). Start the arm at that same config but with the 4 wrist
# joints (URDF joints 6,7,8,9 -- confirmed orientation-critical) re-randomised
# within their limits. Position starts close to target "for free"; orientation
# starts wrong. Reward is orientation-only (no position term, no position
# gate) -- the position-reaching subproblem is deliberately not trained here.
#
# quatDiff() below already includes the double-cover fix (abs() taken BEFORE
# arccos) discovered this session -- this script never had the bug.

import os
import random
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

URDF_DIR = os.path.join(os.path.dirname(__file__),
    "ImprovedUltrasoundRLRobot-main",
    "ImprovedUltrasoundRLRobot-main",
    "One_Arm_Robot",
    "urdf"
)

JOINT_LIMITS_FK = [
    (-0.5000,  0.5000),   # joint 0: prismatic (gantry height)
    (-0.3142,  0.7400),   # joint 1
    (-1.8326,  1.0123),   # joint 3
    (-1.8151,  2.7576),   # joint 4
    (-3.0107,  2.9671),   # joint 5
    ( 1.5708,  3.5779),   # joint 6  (wrist / orientation-critical)
    (-1.5708,  1.5882),   # joint 7  (wrist / orientation-critical)
    (-2.4260, -0.6981),   # joint 8  (wrist / orientation-critical)
    (-3.1416,  3.1416),   # joint 9  (wrist / orientation-critical)
]
WRIST_ANGLE_IDX = [6, 7, 8, 9]        # indices into the 10-long "angles" list
WRIST_LIMIT_ROWS = [5, 6, 7, 8]       # corresponding rows in JOINT_LIMITS_FK

POS_X = (-0.10, 0.15)
POS_Y = ( 0.30, 0.65)
POS_Z = ( 0.15, 0.35)


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
    # abs() BEFORE arccos to collapse quaternion double-cover (q and -q are the
    # same rotation) onto the correct true angular distance in [0, pi].
    a = 2 * np.arccos(float(np.clip(abs(qd[0]), -1.0, 1.0)))
    return a, qd


class RightArmRobotOrientationFirst(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "simulation_fps": 50}

    ORIENT_TOL = 1.2   # rad, same success threshold as Phase 2
    max_step_size = 150

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self._physics_client_id = -1
        # Same shapes as RightArmRobotPhase2 -> resulting model is directly
        # loadable into that environment for the "position on top" phase.
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(19,), dtype=np.float32)
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
            mode = p.DIRECT if self.render_mode == "human" else p.GUI
            self._bullet_client = bullet_client.BulletClient(connection_mode=mode)
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
        self.x_target, self.y_target, self.z_target = pos_target
        self.Quat_t = quat_target.reshape(4, 1)

        # Start config: same as target except wrist joints randomised ->
        # position starts close "for free", orientation starts wrong.
        start_angles = list(target_angles)
        for angle_idx, limit_row in zip(WRIST_ANGLE_IDX, WRIST_LIMIT_ROWS):
            lo, hi = JOINT_LIMITS_FK[limit_row]
            start_angles[angle_idx] = random.uniform(lo, hi)
        for i, a in enumerate(start_angles):
            self._bullet_client.resetJointState(self.robot, i, a)

        ls = self._bullet_client.getLinkState(self.robot, self.num_joints - 1)
        self.position    = np.array(ls[0], dtype=np.float32)
        self.orientation = np.array(ls[1], dtype=np.float32)

        self.Euclid_dist = float(math.dist(self.position, self.pos_target))
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
        # physical +-0.5m travel -- kept in sync with the same fix in
        # run_phase2.py so Phase A and Phase B use matching action scaling.
        scaled[0] = action[0] / 2
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

        # Orientation-only reward -- no position term, no position gate.
        # Position is a free starting condition here, not something scored.
        qn = float(np.clip(self.Quat_diff_angle / np.pi, 0, 1))
        reward = -0.05 * qn + 2.0 * float(self.prev_Quat_diff_angle - self.Quat_diff_angle)
        self.prev_Quat_diff_angle = self.Quat_diff_angle

        terminated = self.Quat_diff_angle < self.ORIENT_TOL
        if terminated:
            reward += 20.0

        truncated = self.step_number >= self.max_step_size
        return obs, float(reward), bool(terminated), bool(truncated), {}


class OrientationFirstCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps
        self.timesteps = 0
        self.terminations = 0
        self.episode_count = 0
        self.final_angles = []

    def _on_step(self):
        self.timesteps += 1
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
        return True

    def _on_training_end(self):
        n_ep = self.episode_count
        print(f"\nOrientation-first phase done: {self.terminations} successes / {n_ep} episodes "
              f"({100*self.terminations/max(n_ep,1):.1f}%)")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up orientation-first phase environment...")
    env = Monitor(RightArmRobotOrientationFirst())
    check_env(env, warn=True)
    print("Environment OK\n")

    TOTAL_TIMESTEPS = 200000
    checkpoint_cb = CheckpointCallback(
        save_freq=25000,
        save_path="models/checkpoints_orientation_first/",
        name_prefix="orientation_first_ckpt",
        verbose=1,
    )
    model = SAC("MlpPolicy", env, verbose=0)
    cb = OrientationFirstCallback(total_timesteps=TOTAL_TIMESTEPS)
    print(f"Training orientation-first phase for {TOTAL_TIMESTEPS} timesteps...\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[cb, checkpoint_cb], log_interval=1)

    model.save("models/orientation_first_model")
    print("Model saved to models/orientation_first_model")
    print("\nNext step (not run by this script): resume this model into "
          "run_phase2.py's RightArmRobotPhase2 env for the 'position on top' phase, "
          "e.g. SAC.load('models/orientation_first_model', env=phase2_env).")
