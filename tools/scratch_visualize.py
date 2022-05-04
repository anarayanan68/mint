"""Scratchpad for visualization - modify as requirements change. Doc for current form of usage below.

Usage:
    scratch_visualize.py VIZ_PATH [-p PATTERN -v VIZ_MODE --overwrite_existing --compute_bbox]

Options:
    VIZ_PATH                    Path to motion npy file, or folder containing motion npy files
    -p, --pattern=PATTERN       Glob pattern for filenames (rooted at VIZ_PATH, without the extension). [default: *]
                                For example, passing '**/*' will select all npy files in all subfolders of VIZ_PATH.
                                Use quotes while passing this argument to avoid shell wildcard expansion.
                                Only used if VIZ_PATH is a folder.
    -v, --viz_mode=VIZ_MODE     Visualization mode: either 'mesh' or 'keypoints'. [default: mesh]
    --overwrite_existing        Overwrite all pre-existing outputs. Default behavior is to skip all such files.
                                Only used if VIZ_PATH is a folder.
    --compute_bbox              Compute a tight bbox on the motion and show its values. Default behavior is to use the hard-coded bbox.
                                Only used if VIZ_PATH is a file.
"""
import os
import sys
import glob
import numpy as np
from smplx import SMPL

from visualization_util import load_motion, visualize_motion
from docopt import docopt


if __name__ == '__main__':
    args = docopt(__doc__)

    # settings
    viz_path = args['VIZ_PATH']
    fname_pattern = args['--pattern']
    viz_mode = args['--viz_mode']
    overwrite_existing = args['--overwrite_existing']
    to_compute_bbox = args['--compute_bbox']

    num_frames = 60
    out_fps = 10
    bbox = ( np.array([0,2,0], dtype=np.float32), np.array([2.5,2.5,2.5], dtype=np.float32) )   # computed before

    smpl_dir = '/home/aravind/work/hays-8903/store/smpl_models'
    gender = 'MALE'
    batch_size = 1

    smpl = SMPL(model_path=smpl_dir, gender=gender, batch_size=batch_size)

    if os.path.isfile(viz_path) and viz_path.endswith('.npy'):
        if to_compute_bbox:
            from visualization_util import smpl_output_on_axis_angles, compute_bbox, axis_angles_from_6D
            motion = load_motion(viz_path, num_frames)
            smpl_output = smpl_output_on_axis_angles(*axis_angles_from_6D(motion), smpl)
            bb = compute_bbox(smpl_output.joints.detach().numpy())
            print(f'BBox computed: {bb}')

        out_vid_path = os.path.splitext(viz_path)[0] + ".mp4"
        visualize_motion(
            load_motion(viz_path, num_frames),
            smpl,
            mode=viz_mode,
            bbox=bbox,
            out_video_path=out_vid_path,
            out_video_fps=out_fps
        )
    elif os.path.isdir(viz_path):
        for file_path in sorted(glob.glob(os.path.join(viz_path, f'{fname_pattern}.npy'), recursive=True)):
            out_vid_path = os.path.splitext(file_path)[0] + ".mp4"
            if not os.path.exists(out_vid_path) or overwrite_existing:
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
        print(f"INVALID PATH: {viz_path}")