import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_one_arm import RightArmRobot
import numpy as np

if __name__ == "__main__":
    print("Opening PyBullet GUI window (Option B — live RL training view)...")
    print("The arm will move randomly — this is an UNTRAINED agent.")
    print("Watch the 3D window. Use mouse to rotate/zoom.\n")

    env = RightArmRobot(render_mode="rgb_array")  # rgb_array triggers GUI mode

    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0.0
        for step in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            time.sleep(1 / 30)  # slow down so it's visible
            if terminated or truncated:
                break
        status = "SUCCESS" if env.Euclid_dist < env.pos_tol else "timeout"
        print(f"Episode {ep+1} | steps={step+1:3d} | reward={ep_reward:8.2f} | "
              f"dist={env.Euclid_dist:.4f}m | {status}")

    print("\nDone. Close the PyBullet window to exit, or press Enter here.")
    input()
    env.close()
