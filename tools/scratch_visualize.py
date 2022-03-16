# %%
import os
import sys
import vedo
import torch
import time
import numpy as np
from smplx import SMPL
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from tqdm import tqdm

from conversion_util import rotation_6d_to_matrix


# %%
# settings

npy_path = sys.argv[1]
num_frames = 60
out_vid_path = os.path.splitext(npy_path)[0] + ".mp4"
out_fps = 10

# %%

def recover_to_axis_angles(motion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, seq_len, dim = motion.shape
    transl = motion[:, :, 6:9]
    rot_6D = motion[:, :, 9:].reshape(batch_size, seq_len, -1, 6)
    rotmats = rotation_6d_to_matrix(rot_6D)
    axis_angles = R.from_matrix(rotmats.reshape(-1,3,3)).as_rotvec().reshape(batch_size, seq_len, -1, 3)
    return axis_angles, transl


def visualize(motion: np.ndarray, smpl_model: SMPL, vedo_video: vedo.io.Video) -> None:
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
    for kpts in tqdm(keypoints3d):
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
# for some reason, output window is not shown sometimes
out_vid = vedo.io.Video(out_vid_path, fps=out_fps, duration=None, backend='ffmpeg')
visualize(result_motion, smpl, out_vid)
out_vid.close()


