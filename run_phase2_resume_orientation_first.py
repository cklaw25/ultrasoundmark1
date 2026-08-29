# Phase B: resume Phase A's orientation-first model into the real Phase 2
# task (position + orientation together). Tests the user's proposed
# curriculum end to end: orientation first, then position on top.
#
# Uses the real RightArmRobotPhase2 environment from run_phase2.py unchanged
# -- this is the actual task we care about, not a diagnostic duplicate.

import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from run_phase2 import RightArmRobotPhase2, Phase2Callback

RESUME_CKPT = "models/orientation_first_model"
TOTAL_TIMESTEPS = 200000  # matches session 7's phase2_extended budget for a fair comparison

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up Phase 2 environment (resuming from orientation-first model)...")
    env = Monitor(RightArmRobotPhase2())
    check_env(env, warn=True)
    print("Environment OK\n")

    print(f"Loading orientation-first weights from {RESUME_CKPT} ...")
    model = SAC.load(RESUME_CKPT, env=env)

    checkpoint_cb = CheckpointCallback(
        save_freq=25000,
        save_path="models/checkpoints_phase2_from_orientation_first/",
        name_prefix="phase2_from_orient_first_ckpt",
        verbose=1,
    )
    cb = Phase2Callback(run_name="phase2_from_orientation_first", total_timesteps=TOTAL_TIMESTEPS)

    print(f"Training {TOTAL_TIMESTEPS} timesteps on the full position+orientation task...\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([cb, checkpoint_cb]),
                log_interval=1, reset_num_timesteps=False)

    model.save("models/phase2_from_orientation_first_model")
    print("Model saved to models/phase2_from_orientation_first_model")
