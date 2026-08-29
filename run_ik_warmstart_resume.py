# Resume the IK warm-start RL fine-tune from the step-50,000 checkpoint saved
# in Session 9 (paused at step 82,729/200,000 before that; the 50k checkpoint
# is the last on-disk save). Mirrors the RESUME_CKPT pattern already used in
# run_phase2.py's __main__ (SAC.load + reset_num_timesteps=False).
#
# Does NOT rebuild the IK dataset or redo BC pretraining -- both already
# happened in the paused run and are baked into the checkpoint's weights.

import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from run_phase2 import RightArmRobotPhase2, Phase2Callback

RESUME_CKPT = "models/checkpoints_ik_warmstart/ik_warmstart_ckpt_50000_steps"
ALREADY_DONE = 50000
ORIGINAL_TARGET = 200000
REMAINING = ORIGINAL_TARGET - ALREADY_DONE  # 150000

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up Phase 2 environment (IK warm-start resume)...")
    env = Monitor(RightArmRobotPhase2())
    check_env(env, warn=True)
    print("Environment OK\n")

    print(f"Resuming IK warm-start RL fine-tune from {RESUME_CKPT} "
          f"(step {ALREADY_DONE}/{ORIGINAL_TARGET})...")
    model = SAC.load(RESUME_CKPT, env=env)

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/checkpoints_ik_warmstart/",
        name_prefix="ik_warmstart_resume_ckpt",
        verbose=1,
    )
    cb = Phase2Callback(run_name="ik_warmstart_resume", total_timesteps=REMAINING)

    print(f"Training remaining {REMAINING} timesteps "
          f"(to reach the original {ORIGINAL_TARGET} target)...\n")
    model.learn(total_timesteps=REMAINING, callback=CallbackList([cb, checkpoint_cb]),
                log_interval=1, reset_num_timesteps=False)

    model.save("models/ik_warmstart_model")
    print("Model saved to models/ik_warmstart_model")
