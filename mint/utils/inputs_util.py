# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Util functions for creating inputs."""
import tensorflow as tf

import numpy as np
from collections import namedtuple


def get_modality_to_param_dict(dataset_config):
  """Creates a map from modality name to modality parameters."""

  modality_to_param_dict = {}
  for modality in dataset_config.modality:
    modality_type = modality.WhichOneof("modality")
    if modality_type == "general_modality":
      modality = modality.general_modality
      modality_to_param_dict[modality.feature_name] = {}
      modality_to_param_dict[
          modality.feature_name]["feature_dim"] = modality.dimension
      modality_to_param_dict[modality.feature_name]["input_length"] = int(
          dataset_config.input_length_sec * modality.sample_rate)
      modality_to_param_dict[modality.feature_name]["target_length"] = int(
          dataset_config.target_length_sec * modality.sample_rate)
      modality_to_param_dict[modality.feature_name]["target_shift"] = int(
          dataset_config.target_shift_sec * modality.sample_rate)
      modality_to_param_dict[
          modality.feature_name]["sample_rate"] = modality.sample_rate
      # Raw image specific parameters.
      modality_to_param_dict[modality.feature_name]["resize"] = modality.resize
      modality_to_param_dict[
          modality.feature_name]["crop_size"] = modality.crop_size
    elif modality_type == "raw_text":
      modality_to_param_dict[modality.feature_name] = {}
    else:
      raise ValueError("Unknown modality type:", modality_type)
  return modality_to_param_dict


def preprocess_labels(example, dataset_config):
  """Preprocess labels to one_hot encoding."""
  target = example.pop(dataset_config.data_target_field)
  example["target"] = tf.reduce_max(
      tf.one_hot(
          tf.sparse.to_dense(target),
          depth=dataset_config.target_num_categories),
      axis=0)
  return example


def fact_preprocessing(example, modality_to_params, is_training):
  """Preprocess data for FACT model."""
  motion_seq_length = tf.shape(example["motion_sequence"])[0]
  motion_input_length = modality_to_params["motion"]["input_length"]
  motion_target_length = modality_to_params["motion"]["target_length"]
  motion_target_shift = modality_to_params["motion"]["target_shift"]
  audio_input_length = modality_to_params["audio"]["input_length"]

  motion_dim = modality_to_params["motion"]["feature_dim"]
  audio_dim = modality_to_params["audio"]["feature_dim"]

  # Pad the input motion translation from 3-dim to 9-dim.
  motion_dim += 6
  example["motion_sequence"] = tf.pad(example["motion_sequence"],
                                      [[0, 0], [6, 0]])
  if is_training:
    windows_size = tf.maximum(motion_input_length,
                              motion_target_shift + motion_target_length)
    windows_size = tf.maximum(windows_size, audio_input_length)
    # the start frame id for this window.
    start = tf.random.uniform([],
                              0,
                              motion_seq_length - windows_size + 1,
                              dtype=tf.int32)
  else:
    start = 0

  # motion input: [start, start + motion_input_length)
  example["motion_input"] = example["motion_sequence"][start:start +
                                                       motion_input_length, :]
  example["motion_input"].set_shape([motion_input_length, motion_dim])
  if is_training:
    # motion target: [start + shift, start + shift + motion_target_length)
    example["target"] = example["motion_sequence"][start +
                                                   motion_target_shift:start +
                                                   motion_target_shift +
                                                   motion_target_length, :]
    example["target"].set_shape([motion_target_length, motion_dim])
  del example["motion_sequence"]

  if is_training:
    # audio input: [start, start + audio_input_length)
    example["audio_input"] = example["audio_sequence"][start:start +
                                                      audio_input_length, :]
    example["audio_input"].set_shape([audio_input_length, audio_dim])
  else:
    example["audio_input"] = example["audio_sequence"]
  del example["audio_sequence"]
  return example


def fact_preprocessing_overfit(example, modality_to_params, is_training):
  motion_seq_length = tf.shape(example["motion_sequence"])[0]
  motion_input_length = modality_to_params["motion"]["input_length"]
  motion_target_length = modality_to_params["motion"]["target_length"]
  motion_target_shift = modality_to_params["motion"]["target_shift"]
  audio_input_length = modality_to_params["audio"]["input_length"]

  motion_dim = modality_to_params["motion"]["feature_dim"]
  audio_dim = modality_to_params["audio"]["feature_dim"]

  start = 0
  # motion target: [start + shift, start + shift + motion_target_length) derived from the actual motion seq
  example["target"] = example["motion_sequence"][start +
                                                  motion_target_shift:start +
                                                  motion_target_shift +
                                                  motion_target_length, :]
  example["target"].set_shape([motion_target_length, motion_dim])

  del example["motion_sequence"]

  # audio input: [start, start + audio_input_length)
  example["audio_input"] = example["audio_sequence"][start:start +
                                                    audio_input_length, :]
  example["audio_input"].set_shape([audio_input_length, audio_dim])

  del example["audio_sequence"]

  return example


def compute_encoding_based_dataset(clip_based_ds: tf.data.Dataset,
                                   random_encoding_seed: int,
                                   is_training: bool) -> tf.data.Dataset:
  # Extract relevant data from clip based dataset
  sample_data = next(iter(clip_based_ds))
  audio_input_shape = sample_data["audio_input"].shape

  num_clips = sample_data["motion_name_enc"].shape[-1]
  targets = [None] * num_clips
  audios = [None] * num_clips
  audio_names = [None] * num_clips
  num_found = 0

  for example in clip_based_ds:
    enc = example["motion_name_enc"]
    idx = tf.argmax(enc)  # find the single "hot" index

    if targets[idx] is None:
      targets[idx] = example["target"]
      audios[idx] = example["audio_input"]
      audio_names[idx] = example["audio_name"]
      num_found += 1
      if num_found == num_clips:
        break
  targets = tf.stack(targets)
  audios = tf.stack(audios)

  # Build generator
  def encoding_generator(num_primitives, targets, audios, audio_names, seed, is_training):
    ScheduleKeys = namedtuple('ScheduleKeys', ['blend_num', 'alpha_max'])

    if is_training:
      schedule = {
        ScheduleKeys(blend_num=1, alpha_max=0.0): {'num_encodings': num_primitives * 800}
      }
      for alpha_max in np.arange(0.05, 0.55, 0.05):
        schedule.update({
          ScheduleKeys(blend_num=2, alpha_max=alpha_max): {'num_encodings': num_primitives * 320}
        })
    else:
      schedule = {
        ScheduleKeys(blend_num=1, alpha_max=0.0): {'num_encodings': num_primitives}
      }
      for alpha_max in np.arange(0.1, 0.55, 0.1):
        schedule.update({
          ScheduleKeys(blend_num=2, alpha_max=alpha_max): {'num_encodings': 10}
        })
      schedule.update({
        ScheduleKeys(blend_num=3, alpha_max=0.3): {'num_encodings': 10},
        ScheduleKeys(blend_num=4, alpha_max=0.25): {'num_encodings': 10},
        ScheduleKeys(blend_num=5, alpha_max=0.2): {'num_encodings': 10},
      })

    position_gen = np.random.RandomState(seed)
    alpha_gen = np.random.RandomState(seed)
    audio_choice_gen = np.random.RandomState(seed)

    out_dict = dict.fromkeys([ "motion_name_enc", "target", "audio_input", "audio_name" ])
    out_dict["target"] = targets # retaining the key verbatim for compatibility with rest of pipeline (e.g. loss fns)

    for keys, vdict in schedule.items():
      blend_num = keys.blend_num
      num_encodings = vdict['num_encodings']

      if blend_num == 1:
        # pure primitives - cycle through in order to learn them first
        for ctr in range(num_encodings):
          pos = ctr % num_primitives
          encoding_vec = tf.one_hot(pos, depth=num_primitives)

          out_dict["motion_name_enc"] = encoding_vec
          out_dict["audio_input"] = audios[pos]
          out_dict["audio_name"] = audio_names[pos]
          yield out_dict

      else:
        alpha_max = keys.alpha_max

        for ctr in range(num_encodings):
          alphas = alpha_gen.uniform(low=0.0, high=alpha_max, size=(blend_num-1,))
          alphas = np.concatenate([alphas, 1.0 - np.sum(alphas,keepdims=True)], axis=-1)

          positions = position_gen.choice(num_primitives, size=(blend_num,), replace=False)

          encoding_vec = np.zeros(shape=(num_primitives,))
          encoding_vec[positions] = alphas
          encoding_vec = tf.convert_to_tensor(encoding_vec)

          audio_choice = audio_choice_gen.choice(positions)
          audio_input = audios[audio_choice]
          audio_name = audio_names[audio_choice]

          out_dict["motion_name_enc"] = encoding_vec
          out_dict["audio_input"] = audio_input
          out_dict["audio_name"] = audio_name
          yield out_dict


  encoding_based_ds = tf.data.Dataset.from_generator(
    encoding_generator,
    args=[num_clips, targets, audios, audio_names, random_encoding_seed, is_training],
    output_signature={
      "motion_name_enc": tf.TensorSpec(shape=(num_clips,), dtype=tf.float32),
      "target": tf.TensorSpec(shape=targets.shape, dtype=tf.float32),
      "audio_input": tf.TensorSpec(shape=audio_input_shape, dtype=tf.float32),
      "audio_name": tf.TensorSpec(shape=(), dtype=tf.string),
    })

  del clip_based_ds
  return encoding_based_ds