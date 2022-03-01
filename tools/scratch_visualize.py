# %%
import os
import sys
import subprocess
import signal
import vedo
import torch
import time
import numpy as np
import random
import re
from scipy.spatial.transform import Rotation as R
from scipy import linalg
from docopt import docopt
import glob
import tqdm
from smplx import SMPL
import soundfile as sf

# %%
# settings

npy_path = sys.argv[1]
num_frames = 60
out_vid_path = os.path.splitext(npy_path)[0] + ".mp4"
out_fps = 10

# %%

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest
    

def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def visualize(motion, smpl_model, vedo_video: vedo.io.Video = None):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()   # (seq_len, 24, 3)

    bbox_center = (
        keypoints3d.reshape(-1, 3).max(axis=0)
        + keypoints3d.reshape(-1, 3).min(axis=0)
    ) / 2.0
    bbox_size = (
        keypoints3d.reshape(-1, 3).max(axis=0) 
        - keypoints3d.reshape(-1, 3).min(axis=0)
    )
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    plotter = vedo.show(world, axes=True, viewup="y", interactive=False)
    for kpts in keypoints3d:
        pts = vedo.Points(kpts).c("red")
        plotter = vedo.show(world, pts)

        if vedo_video is not None:
            vedo_video.addFrame()

        if hasattr(plotter, 'escaped') and plotter.escaped:
            break  # if ESC
        time.sleep(0.01)
    # vedo.interactive().close()
    plotter.close()


# %%
smpl_dir = '/home/aravind/work/hays-8903/store/smpl_models'

smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

# %%
result_motion = np.load(npy_path)
if result_motion.ndim == 2:
    result_motion = result_motion[None, ...]
assert result_motion.shape[0] == 1

result_motion = result_motion[:, :num_frames]
result_motion.shape, result_motion

# %%
# for some reason, output window is not shown, so saving directly
out_vid = vedo.io.Video(out_vid_path, fps=out_fps, duration=None, backend='ffmpeg')
visualize(result_motion, smpl, out_vid)
out_vid.close()

