"""Visualize stored result motions from `evaluator.py`.

Usage:
    visualize.py -s STORE_DIR -e EXPT_DIR [-r SEED -n NUM_FILES -p PATTERN] [--skip_existing | --overwrite_existing]

Options:
    -s, --store_dir=STORE_DIR   Path to store folder housing mp4/ (seed motions), wav/ (music), smpl_models/ (SMPL) subfolders
    -e, --expt_dir=EXPT_DIR     Path to expt folder housing preds/ (.npy, predicted motions) subfolder, and will contain the outputs of this script in viz/
    -r, --random=SEED           Randomly select files with the given random seed. Pass -1 for time seed, don't pass to avoid randomness. Done before using NUM_FILES.
    -n, --num_files=NUM_FILES   Number of files to visualize. Will visualize all files if not provided.
    -p, --pattern=PATTERN       Glob pattern for filenames (only basename, without the extension). [default: *]
    --skip_existing             Skip visualizations and outputs if the output file already exists. Default behavior is to prompt for every such file.
                                Can't be used with --overwrite_existing.
    --overwrite_existing        Overwrite all pre-existing outputs. Default behavior is to prompt for every such file.
                                Can't be used with --skip_existing.
"""

import os
import subprocess
import signal
from typing import Tuple
import vedo
import torch
import time
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy import linalg
from docopt import docopt
import glob
import tqdm
from smplx import SMPL
import soundfile as sf

from conversion_util import rotation_6d_to_matrix


def recover_to_axis_angles(motion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rot_6D = motion[:, :, 9:].reshape(batch_size, seq_len, -1, 6)
    rotmats = rotation_6d_to_matrix(rot_6D)
    axis_angles = R.from_matrix(rotmats.reshape(-1,3,3)).as_rotvec().reshape(batch_size, seq_len, -1, 3)
    return axis_angles, transl


_interrupted = False


def visualize(motion: np.ndarray, smpl_model: SMPL, vedo_video: vedo.io.Video) -> None:
    global _interrupted

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

        vedo_video.addFrame()

        if plotter.escaped or _interrupted:
            _interrupted = False
            break  # if ESC or C-C
        time.sleep(0.01)
    # vedo.interactive().close()
    plotter.close()

def handler_sigint(signum, frame):
    global _interrupted

    ch = input('\nDetected KB interrupt. Move on to next file? [y]/n:\t')
    if ch == 'n':
        print('Quitting.')
        exit(0)
    else:
        _interrupted = True


if __name__ == "__main__":
    args = docopt(__doc__)
    smpl_dir = os.path.join(args['--store_dir'], 'smpl_models')
    video_dir = os.path.join(args['--store_dir'], 'mp4')
    audio_dir = os.path.join(args['--store_dir'], 'wav')

    pred_dir = os.path.join(args['--expt_dir'], 'preds')
    out_dir = os.path.join(args['--expt_dir'], 'viz')
    os.makedirs(out_dir, exist_ok=True)

    # set smpl
    smpl = SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

    result_files = glob.glob(os.path.join(pred_dir, f"{args['--pattern']}.npy"))
    if args['--random'] is not None:
        random.seed(time.time() if args['--random'] == '-1' else int(args['--random']))
        random.shuffle(result_files)
    if args['--num_files'] is not None:
        result_files = result_files[:int(args['--num_files'])]

    signal.signal(signal.SIGINT, handler_sigint)

    for result_file in tqdm.tqdm(result_files):
        result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
        FPS = 60
        HOP_LENGTH = 512    # samples between two frames
        SR = FPS * HOP_LENGTH
        
        base_name = os.path.splitext(os.path.basename(result_file))[0]
        split_idx = base_name.rfind('_')
        vid_name, aud_name = base_name[:split_idx], base_name[split_idx+1:]
        vid_name = vid_name.replace('_cAll', '_c01')
        vid_file = vid_name + ".mp4"
        aud_file = aud_name + ".wav"
        aud_file_matched = aud_name + "_matched.wav"

        if os.path.exists(os.path.join(video_dir, vid_file)):
            print(f"Video {vid_file} already synced.")
        else:
            print(f"Syncing video: {vid_file}")
            subprocess.call(["rsync",
                f"anarayanan68@sky1.cc.gatech.edu:/srv/share/datasets/AIST/aist_plusplus_media/{vid_file}",
                video_dir])
            print('Done.')
        vid_file = os.path.join(video_dir, vid_file)

        if os.path.exists(os.path.join(audio_dir, aud_file)):
            print(f"Audio {aud_file} already synced.")
        else:
            print(f"Syncing audio: {aud_file}")
            subprocess.call(["rsync",
                f"anarayanan68@sky1.cc.gatech.edu:/srv/share/datasets/AIST/aist_plusplus_media/audio/wav/{aud_file}",
                audio_dir])

        aud_file = os.path.join(audio_dir, aud_file)
        aud_file_matched = os.path.join(audio_dir, aud_file_matched)

        if os.path.exists(aud_file_matched):
            print(f"Audio {aud_file_matched} already processed.")
        else:
            print(f"Matching audio to fit video, saving to {aud_file_matched}")
            audio_data, _ = sf.read(aud_file)       # `audio_data` has length = no. of total samples
            trunc_nframes = result_motion.shape[1]
            audio_data_trunc = audio_data[:trunc_nframes * HOP_LENGTH]  # keep only `trunc_nframes` frames, at `HOP_LENGTH` samples/frame
            sf.write(aud_file_matched, audio_data_trunc, samplerate=SR)
            print('Done.')

        print(f"Visualizing result file {result_file}")
        print(f"-- Audio used: {aud_file_matched}")
        print(f"-- Seed video used: {vid_file}")

        out_combined_file = os.path.join(out_dir, f"out__{base_name}.mp4")
        if os.path.exists(out_combined_file):
            if args['--skip_existing']:
                print('\nOut file already exists, skipping.\n')
                continue
            elif args['--overwrite_existing']:
                pass
            else:
                ch = input('\nWARNING: this out file already exists. Overwrite? [y]/n:\t')
                if ch == 'n':
                    print('Skipping this file.\n')
                    continue

        interm_vid_file = os.path.join(out_dir, f"no-audio__{base_name}.mp4")
        interm_video = vedo.io.Video(interm_vid_file, duration=None, fps=FPS, backend='ffmpeg')
        visualize(result_motion, smpl, interm_video)
        interm_video.close()

        print(f"Creating video with synchronized audio at {out_combined_file}...")

        # Command is like: `ffmpeg -i ...__out-no-audio.mp4 -i ..._matched.wav
        #                   -c:a libmp3lame -c:v copy -map 0:v:0 -map 1:a:0
        #                   ..._out.mp4 -y -hide_banner -loglevel error`
        subprocess.call([
            'ffmpeg', '-i', interm_vid_file, '-i', aud_file_matched,
            '-c:a', 'libmp3lame', '-c:v', 'copy',
            '-map', '0:v:0', '-map', '1:a:0',
            out_combined_file,
            '-y', '-hide_banner', '-loglevel', 'error'
        ])

        os.remove(interm_vid_file)
        print(f"Done, and deleted intermediate {interm_vid_file}\n")

    print('\nAll done!')