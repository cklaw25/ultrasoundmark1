# One-shot visualization: what does a *successful* trial's end state look like?
# Renders two screenshots (headless, PyBullet DIRECT + getCameraImage):
#   1) viz_1_start_vs_target.png — every episode starts here (all joints = 0),
#      red sphere marks where the sampled target pose is
#   2) viz_2_success_pose.png    — arm physically posed at the target config
#      (this is what "success" = pos_tol 0.02m + ORIENT_TOL 1.2rad actually means)

import os, random
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bullet_client
from run_phase2 import URDF_DIR, JOINT_LIMITS_FK, POS_X, POS_Y, POS_Z

bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
bc.setAdditionalSearchPath(URDF_DIR)
robot = bc.loadURDF(os.path.join(URDF_DIR, "RightArm.urdf"), [0, 0, 0], useFixedBase=True)
num_joints = bc.getNumJoints(robot)


def sample_reachable():
    attempts = 0
    while True:
        attempts += 1
        j = [random.uniform(lo, hi) for lo, hi in JOINT_LIMITS_FK]
        angles = [j[0], j[1], -j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8]]
        for i, a in enumerate(angles):
            bc.resetJointState(robot, i, a)
        ls = bc.getLinkState(robot, num_joints - 1)
        pos = np.array(ls[0], dtype=np.float32)
        orn = np.array(ls[1], dtype=np.float32)
        in_ws = POS_X[0] <= pos[0] <= POS_X[1] and POS_Y[0] <= pos[1] <= POS_Y[1] and POS_Z[0] <= pos[2] <= POS_Z[1]
        if in_ws or attempts >= 500:
            return angles, pos, orn, attempts


random.seed(3)
target_angles, target_pos, target_orn, attempts = sample_reachable()
print(f"Sampled target after {attempts} attempts: pos={target_pos}, orn(quat)={target_orn}")


def snapshot(path, eye, target, marker_pos=None):
    ids = []
    if marker_pos is not None:
        vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 1])
        ids.append(bc.createMultiBody(baseVisualShapeIndex=vis, basePosition=marker_pos.tolist()))
    view = bc.computeViewMatrix(eye, target, [0, 0, 1])
    proj = bc.computeProjectionMatrixFOV(60, 1.0, 0.05, 3.0)
    w, h, rgb, _, _ = bc.getCameraImage(900, 900, view, proj, renderer=p.ER_TINY_RENDERER)
    arr = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
    import matplotlib.pyplot as plt
    plt.imsave(path, arr)
    print("saved", path)
    for i in ids:
        bc.removeBody(i)


cam_eye = (target_pos + np.array([0.7, 0.7, 0.5])).tolist()
cam_target = target_pos.tolist()

# 1) start pose: every episode begins with all joints = 0
for i in range(num_joints):
    bc.resetJointState(robot, i, 0.0)
snapshot("viz_1_start_vs_target.png", cam_eye, cam_target, marker_pos=target_pos)

# 2) success pose: arm actually driven to the target joint config
for i, a in enumerate(target_angles):
    bc.resetJointState(robot, i, a)
snapshot("viz_2_success_pose.png", cam_eye, cam_target)
