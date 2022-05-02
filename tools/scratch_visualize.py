'''
Scratchpad for visualization - modify as requirements change.
'''
import os
import sys
import glob
import numpy as np
from smplx import SMPL

from visualization_util import load_motion, visualize_motion


# settings
npy_path = sys.argv[1]
fname_pattern = sys.argv[2] # e.g. 'OUTPUT--*'
viz_mode = sys.argv[3] if len(sys.argv) >= 4 else 'mesh'    # either 'mesh' or 'keypoints'
num_frames = 60
out_fps = 10
bbox = ( np.array([0,2,0], dtype=np.float32), np.array([2.5,2.5,2.5], dtype=np.float32) )   # computed before

smpl_dir = '/home/aravind/work/hays-8903/store/smpl_models'
gender = 'MALE'
batch_size = 1

smpl = SMPL(model_path=smpl_dir, gender=gender, batch_size=batch_size)

if os.path.isfile(npy_path) and npy_path.endswith('.npy'):
    ## Uncomment to compute a bbox for reference
    # from visualization_util import smpl_output_on_axis_angles, compute_bbox, axis_angles_from_6D
    # motion = load_motion(npy_path, num_frames)
    # smpl_output = smpl_output_on_axis_angles(*axis_angles_from_6D(motion), smpl)
    # bb = compute_bbox(smpl_output.joints.detach().numpy())
    # print(f'BBox computed: {bb}')

    out_vid_path = os.path.splitext(npy_path)[0] + ".mp4"
    visualize_motion(
        load_motion(npy_path, num_frames),
        smpl,
        mode=viz_mode,
        bbox=bbox,
        out_video_path=out_vid_path,
        out_video_fps=out_fps
    )
elif os.path.isdir(npy_path):
    for file_path in sorted(glob.glob(os.path.join(npy_path, f'{fname_pattern}.npy'), recursive=True)):
        out_vid_path = os.path.splitext(file_path)[0] + ".mp4"
        if not os.path.exists(out_vid_path):
            visualize_motion(
                load_motion(file_path, num_frames),
                smpl,
                mode=viz_mode,
                bbox=bbox,
                out_video_path=out_vid_path,
                out_video_fps=out_fps
            )
        else:
            print(f"Skipping pre-existing out file: {out_vid_path}")
else:
    print(f"INVALID PATH: {npy_path}")