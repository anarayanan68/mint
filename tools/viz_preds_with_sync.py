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
import time
import numpy as np
import random
import re
from docopt import docopt
import glob
import tqdm
from smplx import SMPL
import soundfile as sf

from visualization_util import load_motion, visualize_motion


if __name__ == "__main__":
    args = docopt(__doc__)
    smpl_dir = os.path.join(args['--store_dir'], 'smpl_models')
    video_dir = os.path.join(args['--store_dir'], 'mp4')
    audio_dir = os.path.join(args['--store_dir'], 'wav')

    pred_dir = os.path.join(args['--expt_dir'], 'preds')
    out_dir = os.path.join(args['--expt_dir'], 'viz')
    os.makedirs(out_dir, exist_ok=True)

    # load smpl model
    gender = 'MALE'
    batch_size = 1
    smpl = SMPL(model_path=smpl_dir, gender=gender, batch_size=batch_size)

    # load result filenames according to settings
    result_files = glob.glob(os.path.join(pred_dir, f"{args['--pattern']}.npy"))
    if args['--random'] is not None:
        random.seed(time.time() if args['--random'] == '-1' else int(args['--random']))
        random.shuffle(result_files)
    if args['--num_files'] is not None:
        result_files = result_files[:int(args['--num_files'])]

    # regex to separate out video and audio names
    match_obj = re.compile(r'.*(g\w+_s\w+_cAll_d\w+_m\w+_ch\w+)_(m\w+).*')

    # processing and visualization parameters
    BBOX = None
    FPS = 60
    HOP_LENGTH = 512    # samples between two frames
    SR = FPS * HOP_LENGTH

    for result_file in tqdm.tqdm(result_files):
        result_motion = load_motion(result_file)

        # extract video & audio names to sync, from motion file name
        base_name = os.path.splitext(os.path.basename(result_file))[0]
        match_res = match_obj.match(base_name)
        vid_name, aud_name = match_res.group(1), match_res.group(2)

        vid_name = vid_name.replace('_cAll', '_c01') + ".mp4"
        aud_file = aud_name + ".wav"

        # sync video
        if os.path.exists(os.path.join(video_dir, vid_file)):
            print(f"Video {vid_file} already synced.")
        else:
            print(f"Syncing video: {vid_file}")
            subprocess.call(["rsync",
                f"anarayanan68@sky1.cc.gatech.edu:/srv/share/datasets/AIST/aist_plusplus_media/{vid_file}",
                video_dir])
        vid_file = os.path.join(video_dir, vid_file)

        # sync audio
        if os.path.exists(os.path.join(audio_dir, aud_file)):
            print(f"Audio {aud_file} already synced.")
        else:
            print(f"Syncing audio: {aud_file}")
            subprocess.call(["rsync",
                f"anarayanan68@sky1.cc.gatech.edu:/srv/share/datasets/AIST/aist_plusplus_media/audio/wav/{aud_file}",
                audio_dir])
        aud_file = os.path.join(audio_dir, aud_file)

        # truncate audio to length of video, save as new audio
        trunc_nframes = result_motion.shape[1]
        aud_file_matched = f"{aud_name}__{trunc_nframes}_frames.wav"
        aud_file_matched = os.path.join(audio_dir, aud_file_matched)
        if os.path.exists(aud_file_matched):
            print(f"Audio {aud_file_matched} already processed.")
        else:
            print(f"Matching audio to fit video, saving to {aud_file_matched}")
            audio_data, _ = sf.read(aud_file)       # `audio_data` has length = no. of total samples
            audio_data_trunc = audio_data[:trunc_nframes * HOP_LENGTH]  # keep only `trunc_nframes` frames, at `HOP_LENGTH` samples/frame
            sf.write(aud_file_matched, audio_data_trunc, samplerate=SR)
            print('Done.')

        print(f"Visualizing result file {result_file}")
        print(f"-- Audio used: {aud_file_matched}")
        print(f"-- Seed video used: {vid_file}")

        # handle pre-existing output video
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

        # visualize and generate intermediate video (no audio)
        interm_vid_file = os.path.join(out_dir, f"no-audio__{base_name}.mp4")
        visualize_motion(result_motion, smpl, bbox=BBOX, out_video_path=interm_vid_file, out_video_fps=FPS)

        # add audio track using ffmpeg 
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

        # delete intermediate video and finish loop iteration
        os.remove(interm_vid_file)
        print(f"Done, and deleted intermediate {interm_vid_file}\n")

    print('\nAll done!')