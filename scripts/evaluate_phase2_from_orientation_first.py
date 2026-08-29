# Deterministic evaluation of models/phase2_from_orientation_first_model.zip
# Same pattern as evaluate_phase2.py, pointed at the orientation-first-then-position
# curriculum result for direct comparison against the baseline/extended models.

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from run_phase2 import RightArmRobotPhase2

N_EPISODES = 100
MODEL_PATH = "models/phase2_from_orientation_first_model"

env = Monitor(RightArmRobotPhase2())
model = SAC.load(MODEL_PATH, env=env)

final_dists = []
final_angles = []
successes = 0

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    raw_env = env.unwrapped
    final_dists.append(raw_env.Euclid_dist)
    final_angles.append(raw_env.Quat_diff_angle)
    if terminated:
        successes += 1

final_dists = np.array(final_dists)
final_angles = np.array(final_angles)

print("\n========== ORIENTATION-FIRST -> POSITION EVALUATION SUMMARY ==========")
print(f"Episodes:                {N_EPISODES}")
print(f"Mean final distance:     {final_dists.mean():.4f}m  (std {final_dists.std():.4f})")
print(f"Min final distance:      {final_dists.min():.4f}m")
print(f"Mean final angle error:  {final_angles.mean():.3f}rad (std {final_angles.std():.3f})")
print(f"Success rate:            {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.1f}%)")
print("=========================================================================")
