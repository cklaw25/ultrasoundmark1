# Second resume of the IK warm-start RL fine-tune, continuing from the
# step-150,000 checkpoint saved during Session 10's resume run (paused by
# the user at relative step ~122,012/150,000, absolute ~172,000, before
# reaching its own next checkpoint -- so ik_warmstart_resume_ckpt_150000
# is the latest valid save). Same RESUME_CKPT pattern as
# run_ik_warmstart_resume.py.

import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from run_phase2 import RightArmRobotPhase2, Phase2Callback

RESUME_CKPT = "models/checkpoints_ik_warmstart/ik_warmstart_resume_ckpt_150000_steps"
ALREADY_DONE = 150000
ORIGINAL_TARGET = 200000
REMAINING = ORIGINAL_TARGET - ALREADY_DONE  # 50000

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up Phase 2 environment (IK warm-start resume #2)...")
    env = Monitor(RightArmRobotPhase2())
    check_env(env, warn=True)
    print("Environment OK\n")

    print(f"Resuming IK warm-start RL fine-tune from {RESUME_CKPT} "
          f"(step {ALREADY_DONE}/{ORIGINAL_TARGET})...")
    model = SAC.load(RESUME_CKPT, env=env)

    checkpoint_cb = CheckpointCallback(
        save_freq=25000,
        save_path="models/checkpoints_ik_warmstart/",
        name_prefix="ik_warmstart_resume2_ckpt",
        verbose=1,
    )
    cb = Phase2Callback(run_name="ik_warmstart_resume2", total_timesteps=REMAINING)

    print(f"Training remaining {REMAINING} timesteps "
          f"(to reach the original {ORIGINAL_TARGET} target)...\n")
    model.learn(total_timesteps=REMAINING, callback=CallbackList([cb, checkpoint_cb]),
                log_interval=1, reset_num_timesteps=False)

    model.save("models/ik_warmstart_model")
    print("Model saved to models/ik_warmstart_model")
