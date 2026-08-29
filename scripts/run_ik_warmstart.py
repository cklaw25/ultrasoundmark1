# IK warm-start: pretrain the SAC actor via behaviour cloning on PyBullet's
# analytical IK solver before RL fine-tuning, instead of starting from a
# random network. Targets the position bottleneck directly (see Session 8
# diagnosis), independent of the orientation-first curriculum trick.
#
# Uses the real RightArmRobotPhase2 environment from run_phase2.py unchanged
# for both target sampling and the RL fine-tune phase -- this is the actual
# task we care about, not a diagnostic duplicate.
#
# Every episode starts at the all-zero joint config (see reset() in
# run_phase2.py), so the IK label for a given target is also the correct
# action at any later step in that episode -- Phase 2's joint motors are
# POSITION_CONTROL, so the "right" command is the same static target
# throughout, not something that needs to change as the arm approaches it.
# That's what makes a one-shot (obs -> IK action) BC dataset a reasonable
# warm-start for this task, without needing per-step trajectory labels.
#
# Known limitation (not engineered around in this first pass -- see it as an
# empirical question, not a known bug): only the actor is warm-started here.
# The critic (Q-network) starts fresh, so early RL steps may push against the
# pretrained actor before the critic has learned to value it correctly. If
# the run underperforms in a way that looks like the actor "forgetting" its
# warm-start early on, this is the first place to look.

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from run_phase2 import RightArmRobotPhase2, Phase2Callback, JOINT_LIMITS_FK, quatDiff

N_IK_SAMPLES    = 4000
N_IK_RESTARTS   = 3     # best-of-N random restarts per target, since this is a
                         # redundant 9-DOF-for-6D task with local minima
BC_EPOCHS       = 100
BC_BATCH_SIZE   = 256
TOTAL_TIMESTEPS = 200000  # matches Phase B's budget for a fair comparison

# PyBullet's calculateInverseKinematics doesn't know about the joint2 = -joint1
# mechanical coupling (it's not a URDF constraint, just enforced in code -- see
# run_phase2.py's step()/​_sample_reachable_pose()). Overriding joint2 AFTER an
# unconstrained IK solve invalidates the rest of the chain (joint2 is early in
# the kinematic chain, so most of the arm's pose depends on it) -- confirmed
# empirically: that approach gave mean 0.16m / 1.97rad error, worse than
# useless. Instead, solve directly over the 9 free joints (JOINT_LIMITS_FK)
# with the coupling baked into forward kinematics, via least_squares -- same
# FK approach _sample_reachable_pose() already uses (resetJointState +
# getLinkState), just wrapped in an optimizer instead of rejection sampling.


def _fk(bullet_client, robot, num_joints, free_vars):
    angles = [free_vars[0], free_vars[1], -free_vars[1], free_vars[2], free_vars[3],
              free_vars[4], free_vars[5], free_vars[6], free_vars[7], free_vars[8]]
    for i, a in enumerate(angles):
        bullet_client.resetJointState(robot, i, a)
    ls = bullet_client.getLinkState(robot, num_joints - 1)
    return np.array(ls[0]), np.array(ls[1])


def _ik_residual(free_vars, bullet_client, robot, num_joints, target_pos, target_quat):
    pos, orn = _fk(bullet_client, robot, num_joints, free_vars)
    pos_err = pos - target_pos
    _, qd = quatDiff(orn, target_quat)
    qd = qd.flatten()
    if qd[0] < 0:
        qd = -qd  # double-cover: pick the representative closer to target_quat
    orient_err = qd[1:4]  # -> 0 as orientations align (sin(angle/2) scale)
    return np.concatenate([pos_err, orient_err])


def solve_ik(env, target_pos, target_quat, n_restarts=N_IK_RESTARTS):
    raw_env = env.unwrapped
    lower = np.array([lo for lo, hi in JOINT_LIMITS_FK])
    upper = np.array([hi for lo, hi in JOINT_LIMITS_FK])

    best_x, best_cost = None, None
    for r in range(n_restarts):
        # joints 6/8 have asymmetric limits that exclude 0 (see JOINT_LIMITS_FK),
        # so the "matches episode start" restart must be the closest in-bounds
        # point to zero, not raw zero.
        x0 = np.clip(np.zeros(9), lower, upper) if r == 0 else np.random.uniform(lower, upper)
        result = least_squares(
            _ik_residual, x0,
            args=(raw_env._bullet_client, raw_env.RightArmRobot, raw_env.num_joints, target_pos, target_quat),
            bounds=(lower, upper), method="trf", max_nfev=300,
        )
        cost = float(np.sum(result.fun ** 2))
        if best_cost is None or cost < best_cost:
            best_cost, best_x = cost, result.x
    return best_x, best_cost


def build_ik_dataset(env, n_samples, verbose=True):
    """
    For n_samples reachable targets (env.reset() already does FK-based
    rejection sampling), solve IK (respecting the joint2 coupling) and
    convert the result into the env's normalised action space.
    Returns (obs_array, action_array, target_pos_array, target_quat_array) --
    targets are kept alongside their action so quality can be checked against
    the SAME target each action was actually solved for.
    """
    raw_env = env.unwrapped
    obs_list = []
    action_list = []
    target_pos_list = []
    target_quat_list = []
    costs = []

    for n in range(n_samples):
        obs, _ = env.reset()
        target_pos = raw_env.pos_target
        target_quat = raw_env.Quat_t

        free_vars, cost = solve_ik(env, target_pos, target_quat)
        costs.append(cost)

        jp = [free_vars[0], free_vars[1], -free_vars[1], free_vars[2], free_vars[3],
              free_vars[4], free_vars[5], free_vars[6], free_vars[7], free_vars[8]]

        action = np.zeros(10, dtype=np.float32)
        action[0] = np.clip(jp[0] * 2.0, -1.0, 1.0)  # inverse of run_phase2.py's scaled[0] = action[0]/2
        for i in range(1, 10):
            action[i] = np.clip(jp[i] / np.pi, -1.0, 1.0)

        obs_list.append(obs)
        action_list.append(action)
        target_pos_list.append(np.array(target_pos, dtype=np.float32))
        target_quat_list.append(np.array(target_quat, dtype=np.float32).flatten())

        if verbose and (n + 1) % 500 == 0:
            print(f"  IK dataset: {n + 1}/{n_samples} samples built "
                  f"(recent mean IK residual cost {np.mean(costs[-500:]):.5f})")

    return (np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.float32),
            np.array(target_pos_list, dtype=np.float32), np.array(target_quat_list, dtype=np.float32))


def check_ik_label_quality(env, actions, target_pos_arr, target_quat_arr, n_check=200):
    """
    Sanity check: apply a subset of the IK-derived actions directly (bypassing
    the policy entirely) and measure the resulting position/orientation error
    via FK AGAINST THE TARGET EACH ACTION WAS ACTUALLY SOLVED FOR, to confirm
    the labels are good before spending time pretraining on them. Does not
    call env.reset() -- doing so would sample a new, unrelated target and
    silently compare each action against the wrong goal.
    """
    from run_phase2 import quatDiff
    raw_env = env.unwrapped
    dists, angles = [], []
    idx = np.random.choice(len(actions), size=min(n_check, len(actions)), replace=False)
    for i in idx:
        action = actions[i]
        target_pos = target_pos_arr[i]
        target_quat = target_quat_arr[i].reshape(4, 1)
        scaled = action.copy()
        scaled[0] = action[0] / 2
        for j in range(1, 10):
            scaled[j] = action[j] * np.pi
        scaled[2] = -scaled[1]
        for j in range(10):
            raw_env._bullet_client.resetJointState(raw_env.RightArmRobot, j, float(scaled[j]))
        ls = raw_env._bullet_client.getLinkState(raw_env.RightArmRobot, raw_env.num_joints - 1)
        pos = np.array(ls[0])
        angle, _ = quatDiff(np.array(ls[1]), target_quat)
        dists.append(float(np.linalg.norm(pos - target_pos)))
        angles.append(float(angle))
    dists = np.array(dists)
    angles = np.array(angles)
    print(f"IK label quality ({len(idx)} samples, applied directly via FK, no RL involved):")
    print(f"  mean dist  {dists.mean():.4f}m (max {dists.max():.4f}m)")
    print(f"  mean angle {angles.mean():.3f}rad (max {angles.max():.3f}rad)")


def pretrain_actor_bc(model, obs_arr, action_arr, epochs, batch_size, verbose=True):
    actor = model.policy.actor
    device = model.device
    obs_t = torch.as_tensor(obs_arr, device=device)
    act_t = torch.as_tensor(action_arr, device=device)
    n = len(obs_arr)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = actor(obs_t[idx], deterministic=True)
            loss = F.mse_loss(pred, act_t[idx])
            actor.optimizer.zero_grad()
            loss.backward()
            actor.optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  BC epoch {epoch + 1}/{epochs} | mean MSE loss {epoch_loss / n_batches:.5f}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Setting up Phase 2 environment (IK warm-start)...")
    env = Monitor(RightArmRobotPhase2())
    check_env(env, warn=True)
    print("Environment OK\n")

    print(f"Building IK dataset ({N_IK_SAMPLES} samples)...")
    obs_arr, action_arr, target_pos_arr, target_quat_arr = build_ik_dataset(env, N_IK_SAMPLES)
    print("Dataset built.\n")

    check_ik_label_quality(env, action_arr, target_pos_arr, target_quat_arr)
    print()

    print("Creating fresh SAC model...")
    model = SAC("MlpPolicy", env, verbose=0)

    print(f"Pretraining actor via behaviour cloning ({BC_EPOCHS} epochs)...")
    pretrain_actor_bc(model, obs_arr, action_arr, BC_EPOCHS, BC_BATCH_SIZE)
    print("BC pretraining done.\n")

    model.save("models/ik_warmstart_pretrained")
    print("Pretrained (BC-only, no RL yet) model saved to models/ik_warmstart_pretrained\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="models/checkpoints_ik_warmstart/",
        name_prefix="ik_warmstart_ckpt",
        verbose=1,
    )
    cb = Phase2Callback(run_name="ik_warmstart", total_timesteps=TOTAL_TIMESTEPS)

    print(f"RL fine-tuning for {TOTAL_TIMESTEPS} timesteps on the full position+orientation task...\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([cb, checkpoint_cb]), log_interval=1)

    model.save("models/ik_warmstart_model")
    print("Model saved to models/ik_warmstart_model")
