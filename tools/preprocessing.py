from absl import app
from absl import flags
from absl import logging

import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import librosa
import pickle
from aist_plusplus.loader import AISTDataset

import tensorflow as tf

from conversion_util import matrix_to_rotation_6d


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '/mnt/data/aist_plusplus_final/', 
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    'audio_dir', '/mnt/data/AIST/music/', 
    'Path to the AIST wav files.')
flags.DEFINE_string(
    'audio_cache_dir', './data/aist_audio_feats/', 
    'Path to cache dictionary for audio features.')
flags.DEFINE_bool(
    'overwrite_audio_cache', False,
    'Whether to overwite the audio cache if the directory already exists.')
flags.DEFINE_enum(
    'split', 'train', ['train', 'testval'],
    'Whether do training set or testval set.')
flags.DEFINE_string(
    'tfrecord_path', './data/aist_tfrecord', 
    'Output path for the tfrecord files.')
flags.DEFINE_integer(
    'random_audio_seed', None,
    'Random seed int >= 0, to create random audio features. Uses normal audio if not passed.',
    lower_bound=0)
flags.DEFINE_string(
    'enc_pkl_path', None,
    'Path to pkl file to load/save motion name encoding data (weights etc). Skips encoding if not passed.')
flags.DEFINE_integer(
    'subset_size', None,
    'Int >= 1. If passed, sets the number of sequences (from the start) to actually use for tfrecords. '
    'All videos are chosen if this is not passed or is higher than the dataset size.',
    lower_bound=1)

RNG = np.random.RandomState(42)


def create_tfrecord_writers(output_file, n_shards):
    writers = []
    for i in range(n_shards):
        writers.append(tf.io.TFRecordWriter(
            "{}-{:0>5d}-of-{:0>5d}".format(output_file, i, n_shards)
        ))
    return writers


def close_tfrecord_writers(writers):
    for w in writers:
        w.close()


def write_tfexample(writers, tf_example):
    random_writer_idx = RNG.randint(0, len(writers))
    writers[random_writer_idx].write(tf_example.SerializeToString())


def to_tfexample(motion_sequence, audio_sequence, motion_name, motion_name_enc, audio_name):
    features = dict()
    features['motion_name'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[motion_name.encode('utf-8')]))

    if motion_name_enc is not None:
        features['motion_name_enc'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=motion_name_enc.flatten()))

    features['motion_sequence'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=motion_sequence.flatten()))
    features['motion_sequence_shape'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=motion_sequence.shape))
    features['audio_name'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[audio_name.encode('utf-8')]))
    features['audio_sequence'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=audio_sequence.flatten()))
    features['audio_sequence_shape'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=audio_sequence.shape))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def load_cached_audio_features(seq_name):
    audio_name = seq_name.split("_")[-2]
    return np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")), audio_name


def cache_audio_features(seq_names):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    def _get_tempo(audio_name):
        """Get tempo (BPM) for a music by parsing music name."""
        assert len(audio_name) == 4
        if audio_name[0:3] in ['mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB']:
            return int(audio_name[3]) * 10 + 80
        elif audio_name[0:3] == 'mHO':
            return int(audio_name[3]) * 5 + 110
        else: assert False, audio_name

    audio_names = list(set([seq_name.split("_")[-2] for seq_name in seq_names]))

    for audio_name in audio_names:
        save_path = os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy")
        if os.path.exists(save_path):
            continue
        data, _ = librosa.load(os.path.join(FLAGS.audio_dir, f"{audio_name}.wav"), sr=SR)
        envelope = librosa.onset.onset_strength(data, sr=SR)  # (seq_len,)
        mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
            start_bpm=_get_tempo(audio_name), tightness=100)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        audio_feature = np.concatenate([
            envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]
        ], axis=-1)

        if FLAGS.random_audio_seed is not None:
            audio_rng = np.random.default_rng(seed=FLAGS.random_audio_seed)
            audio_feature = audio_rng.uniform(np.min(audio_feature) * 5, np.max(audio_feature) * 5, audio_feature.shape)
        np.save(save_path, audio_feature)


def compute_SMPL_motion(seq_name, motion_dir):
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(motion_dir, seq_name)
    smpl_trans /= smpl_scaling

    num_frames = smpl_poses.shape[0]
    smpl_poses = R.from_rotvec(smpl_poses.reshape((-1, 3))).as_matrix().reshape((num_frames, -1, 3, 3))
    smpl_poses = matrix_to_rotation_6d(smpl_poses).reshape((num_frames, -1))

    smpl_motion = np.concatenate([smpl_trans, smpl_poses], axis=-1)
    return smpl_motion


def get_onehot_enc_map(seq_names: list):
    onehot_enc_map = { 'len': len(seq_names) }
    for i, seq_name in enumerate(seq_names):
        enc = np.zeros(len(seq_names))
        enc[i] = 1

        onehot_enc_map[seq_name] = { 'index': i, 'enc': enc }

    return onehot_enc_map


def load_enc_pkl(pkl_path):
    res = None
    if pkl_path is not None and os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)
    return res


def cache_enc_pkl(pkl_path: str, seq_names: list):
    assert len(seq_names) > 0

    onehot_enc_map = get_onehot_enc_map(seq_names)

    pkl_data = {
        'onehot_enc_map': onehot_enc_map,
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_data, f)


def get_latent_from_seq_name(seq_name, enc_pkl_data):
    return enc_pkl_data['onehot_enc_map'][seq_name]['enc']


def get_encoded_input(seq_name, enc_pkl_data):
    if enc_pkl_data is None:
        return None

    return get_latent_from_seq_name(seq_name, enc_pkl_data)


def main(_):
    os.makedirs(os.path.dirname(FLAGS.tfrecord_path), exist_ok=True)
    tfrecord_writers = create_tfrecord_writers(
        "%s-%s" % (FLAGS.tfrecord_path, FLAGS.split), n_shards=20)

    # create list
    seq_names = []
    if "train" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_train.txt"), dtype=str
        ).tolist()
    if "val" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_val.txt"), dtype=str
        ).tolist()
    if "test" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
        ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]
    seq_names = seq_names[:FLAGS.subset_size]

    # create audio features
    if FLAGS.overwrite_audio_cache or not os.path.isdir(FLAGS.audio_cache_dir):
        print("Pre-compute audio features ...")
        os.makedirs(FLAGS.audio_cache_dir, exist_ok=True)
        cache_audio_features(seq_names)
    else:
        print("Using existing audio cache.")
    
    # load data
    dataset = AISTDataset(FLAGS.anno_dir)
    n_samples = len(seq_names)

    enc_pkl_data = None
    if FLAGS.enc_pkl_path is not None:
        if not os.path.exists(FLAGS.enc_pkl_path):
            cache_enc_pkl(FLAGS.enc_pkl_path, seq_names)
            print(f"Cached encoding data at {FLAGS.enc_pkl_path}")
        else:
            print(f"Used existing encoding data at {FLAGS.enc_pkl_path}")
        enc_pkl_data = load_enc_pkl(FLAGS.enc_pkl_path)

    for i, seq_name in enumerate(seq_names):
        logging.info("processing %d / %d" % (i + 1, n_samples))

        motion_seq = compute_SMPL_motion(seq_name, dataset.motion_dir)
        seq_name_enc = get_encoded_input(seq_name, enc_pkl_data)
        audio_seq, audio_name = load_cached_audio_features(seq_name)

        tfexample = to_tfexample(motion_seq, audio_seq, seq_name, seq_name_enc, audio_name)
        write_tfexample(tfrecord_writers, tfexample)

    # If testval, also test on un-paired data
    if FLAGS.split == "testval":
        logging.info("Also add un-paired motion-music data for testing.")
        for i, seq_name in enumerate(seq_names * 10):
            logging.info("processing %d / %d" % (i + 1, n_samples * 10))

            motion_seq = compute_SMPL_motion(seq_name, dataset.motion_dir)
            seq_name_enc = get_encoded_input(seq_name, enc_pkl_data)
            audio_seq, audio_name = load_cached_audio_features(random.choice(seq_names))

            tfexample = to_tfexample(motion_seq, audio_seq, seq_name, seq_name_enc, audio_name)
            write_tfexample(tfrecord_writers, tfexample)
    
    close_tfrecord_writers(tfrecord_writers)

if __name__ == '__main__':
  app.run(main)