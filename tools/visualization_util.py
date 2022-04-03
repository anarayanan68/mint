import vedo
import trimesh
import torch
import time
import numpy as np
from smplx import SMPL
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from tqdm import tqdm

from conversion_util import rotation_6d_to_matrix


def load_motion(motion_npy_path: str, num_frames: int = None) -> np.ndarray:
    result_motion = np.load(motion_npy_path)
    if result_motion.ndim == 2:
        result_motion = result_motion[None, ...]
    result_motion = result_motion[:, :num_frames, ...]
    return result_motion


def axis_angles_from_6D(motions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rest_of_shape = motions.shape[:-2]
    seq_len = motions.shape[-2]

    transl = motions[..., :, :3]
    rot_6D = motions[..., :, 3:].reshape(*rest_of_shape, seq_len, -1, 6)

    rotmats = rotation_6d_to_matrix(rot_6D)
    axis_angles = (R.from_matrix(rotmats.reshape(-1,3,3))
                    .as_rotvec()
                    .reshape(*rest_of_shape, seq_len, -1, 3))
    return axis_angles, transl


def smpl_output_on_axis_angles(poses_1seq: np.ndarray, transl_1seq: np.ndarray, smpl_model: SMPL):
    poses_1seq, transl_1seq = np.squeeze(poses_1seq), np.squeeze(transl_1seq)
    assert poses_1seq.ndim == 3     # 1 sequence only, shape (seq_len, 24, 3)
    assert transl_1seq.ndim == 2    # 1 sequence only, shape (seq_len, 3)

    return smpl_model.forward(
        global_orient=torch.from_numpy(poses_1seq[:, 0:1]).float(),
        body_pose=torch.from_numpy(poses_1seq[:, 1:]).float(),
        transl=torch.from_numpy(transl_1seq).float(),
    )


def compute_bbox(keypoints3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bbox_center = (
        keypoints3d.reshape(-1, 3).max(axis=0)
        + keypoints3d.reshape(-1, 3).min(axis=0)
    ) / 2.0
    bbox_size = (
        keypoints3d.reshape(-1, 3).max(axis=0) 
        - keypoints3d.reshape(-1, 3).min(axis=0)
    )

    return bbox_center, bbox_size


def show_SMPL_output(smpl_output, smpl_model: SMPL,
                     vedo_video: vedo.Video = None, mode: str = 'mesh', bbox: Tuple[np.ndarray, np.ndarray] = None) -> None:
    keypoints3d = smpl_output.joints.detach().numpy()
    if bbox is None:
        bbox = compute_bbox(keypoints3d)
    bbox_center, bbox_size = bbox
    world = vedo.Box(bbox_center, bbox_size[0], bbox_size[1], bbox_size[2]).wireframe()
    plotter = vedo.show(world, axes=True, viewup="y", interactive=False)

    if mode == 'mesh':
        vertices = smpl_output.vertices.detach().numpy()
        faces = smpl_model.faces

    num_frames = len(keypoints3d)
    for i in range(num_frames):
        pts = vedo.Points(keypoints3d[i]).c("red")
        actors = [pts]
        if mode == 'mesh':
            mesh = trimesh.Trimesh(vertices[i], faces)
            mesh.visual.face_colors = [200, 200, 250, 100]
            actors.append(mesh)

        plotter.show(world, *actors)

        if vedo_video is not None:
            vedo_video.addFrame()

        if hasattr(plotter, 'escaped') and plotter.escaped:
            break  # if ESC
        time.sleep(0.01)

    # vedo.interactive().close()
    plotter.close()
    del world, plotter


def visualize_motion(   motion: np.ndarray,
                        smpl_model: SMPL,
                        mode: str = 'mesh',
                        bbox: Tuple[np.ndarray, np.ndarray] = None,
                        out_video_path: str = None,
                        out_video_fps: int = 60) -> None:
    # TODO: handle multiple motions later if needed
    axis_angles, transl = axis_angles_from_6D(motion)
    smpl_output = smpl_output_on_axis_angles(axis_angles, transl, smpl_model)

    vedo_video = (  vedo.Video(out_video_path, duration=None, fps=out_video_fps, backend='ffmpeg')
                    if out_video_path is not None
                    else None)
    show_SMPL_output(smpl_output, smpl_model, vedo_video=vedo_video, mode=mode, bbox=bbox)
    vedo_video.close()
