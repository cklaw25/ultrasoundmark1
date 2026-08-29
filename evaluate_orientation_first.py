# Deterministic evaluation of models/orientation_first_model.zip
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from run_orientation_first_phase import RightArmRobotOrientationFirst

N_EPISODES = 100
env = Monitor(RightArmRobotOrientationFirst())
model = SAC.load("models/orientation_first_model", env=env)

final_angles = []
final_dists = []
successes = 0
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    raw_env = env.unwrapped
    final_angles.append(raw_env.Quat_diff_angle)
    final_dists.append(raw_env.Euclid_dist)
    if terminated:
        successes += 1

final_angles = np.array(final_angles)
final_dists = np.array(final_dists)
print("\n==================== ORIENTATION-FIRST PHASE EVAL ====================")
print(f"Episodes:               {N_EPISODES}")
print(f"Mean final angle error: {final_angles.mean():.3f}rad (std {final_angles.std():.3f})")
print(f"Min final angle error:  {final_angles.min():.3f}rad")
print(f"Mean final distance:    {final_dists.mean():.4f}m (std {final_dists.std():.4f}) [not trained, informational only]")
print(f"Success rate:           {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.1f}%)")
print("=========================================================================")
