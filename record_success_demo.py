# Records a real trained-policy rollout that ends in a genuine success
# (position AND orientation both under threshold, i.e. terminated=True),
# by running deterministic evaluation episodes until one actually
# succeeds, capturing a frame every step along the way.
#
# Unlike visualize_success.py (which just poses the arm directly at a
# target's own joint angles -- illustrative, not an actual policy demo),
# this captures the policy really driving the arm there step by step.

import os
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from run_phase2 import RightArmRobotPhase2

OUT_DIR = "demo_frames"
os.makedirs(OUT_DIR, exist_ok=True)

# Try the model most likely to produce a success quickly first (higher
# recorded success rate), fall back to the other if needed.
CANDIDATES = [
    ("models/phase2_from_orientation_first_model_v1_oldscaling_3pct", 150),
    ("models/ik_warmstart_model", 150),
]

env = Monitor(RightArmRobotPhase2())


def render_frame(raw_env):
    bc = raw_env._bullet_client
    target = raw_env.pos_target
    eye = (target + np.array([0.6, 0.6, 0.4])).tolist()
    view = bc.computeViewMatrix(eye, target.tolist(), [0, 0, 1])
    proj = bc.computeProjectionMatrixFOV(60, 1.0, 0.05, 3.0)
    w, h, rgb, _, _ = bc.getCameraImage(480, 480, view, proj, renderer=p.ER_TINY_RENDERER)
    return np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)


found = False
for model_path, max_episodes in CANDIDATES:
    print(f"Searching for a real success with {model_path} ...")
    model = SAC.load(model_path, env=env)

    for ep in range(max_episodes):
        obs, _ = env.reset()
        raw = env.unwrapped
        bc = raw._bullet_client
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 1])
        bc.createMultiBody(baseVisualShapeIndex=vis, basePosition=raw.pos_target.tolist())

        frames = [render_frame(raw)]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            raw = env.unwrapped
            frames.append(render_frame(raw))
            done = terminated or truncated

        if terminated:
            print(f"  SUCCESS at episode {ep + 1}/{max_episodes}: "
                  f"dist={raw.Euclid_dist:.4f}m angle={raw.Quat_diff_angle:.3f}rad, "
                  f"{len(frames)} frames")
            for i, f in enumerate(frames):
                plt.imsave(os.path.join(OUT_DIR, f"frame_{i:04d}.png"), f)
            with open("demo_success_info.txt", "w") as fh:
                fh.write(f"model: {model_path}\n")
                fh.write(f"episode: {ep + 1}/{max_episodes}\n")
                fh.write(f"final dist: {raw.Euclid_dist:.4f}m\n")
                fh.write(f"final angle: {raw.Quat_diff_angle:.4f}rad\n")
                fh.write(f"frames: {len(frames)}\n")
            found = True
            break
        elif (ep + 1) % 25 == 0:
            print(f"  ... {ep + 1}/{max_episodes} tried, no success yet")

    if found:
        break

if not found:
    print("No success found within the search budget on any candidate model.")
