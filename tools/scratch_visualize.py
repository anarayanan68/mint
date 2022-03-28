'''
Scratchpad for visualization - modify as requirements change.
'''
import os
import sys
import glob
import numpy as np
from smplx import SMPL

from visualization_util import load_motion, visualize_motion, compute_bbox, keypoints3D_from_axis_angles, axis_angles_from_6D


# settings
npy_path = sys.argv[1]
num_frames = 60
out_fps = 10
bbox = ( np.array([0,2,0], dtype=np.float32), np.array([2.5,2.5,2.5], dtype=np.float32) )

smpl_dir = '/home/aravind/work/hays-8903/store/smpl_models'
gender = 'MALE'
batch_size = 1

smpl = SMPL(model_path=smpl_dir, gender=gender, batch_size=batch_size)

if os.path.isfile(npy_path) and npy_path.endswith('.npy'):
    ## Uncomment to compute a bbox for reference
    # motion = load_motion(npy_path, num_frames)
    # bb = compute_bbox(keypoints3D_from_axis_angles(*axis_angles_from_6D(motion), smpl))
    # print(f'BBox computed: {bb}')
    out_vid_path = os.path.splitext(npy_path)[0] + ".mp4"
    visualize_motion(
        load_motion(npy_path, num_frames),
        smpl,
        bbox=bbox,
        out_video_path=out_vid_path,
        out_video_fps=out_fps
    )
elif os.path.isdir(npy_path):
    for file_path in glob.glob(os.path.join(npy_path, 'OUTPUT--*.npy'), recursive=True):
        out_vid_path = os.path.splitext(file_path)[0] + ".mp4"
        visualize_motion(
            load_motion(file_path, num_frames),
            smpl,
            bbox=bbox,
            out_video_path=out_vid_path,
            out_video_fps=out_fps
        )
else:
    print(f"INVALID PATH: {npy_path}")