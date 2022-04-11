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
"""Module to evaluate a conditional flow prediction model."""

import os
import pickle

from absl import app
from absl import flags
from mint.core import inputs
from mint.core import model_builder
from mint.ctl import single_task_evaluator
from mint.utils import config_util
from third_party.tf_models import orbit
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None,
                    'Directory to write training checkpoints and logs')
flags.DEFINE_string('config_path', None, 'Path to the config file.')
flags.DEFINE_string('eval_prefix', 'valid', 'Prefix for evaluation summaries.')
flags.DEFINE_string('output_dir', 'outputs', 'Where to save the results.')

# Unused flags to play nice with xm hyperparameter sweep. Add all flags under
# hyperparameter sweep in trainer.py here.
flags.DEFINE_float('initial_learning_rate', 0.1, 'UNUSED FLAG.')
flags.DEFINE_float('weight_decay', None, 'UNUSED FLAG.')
flags.DEFINE_string('head_initializer', 'he_normal',
                    'Initializer for prediction head.')
flags.DEFINE_bool('overfit_expt', False, 'Whether running the overfit experiment or not (which controls a few important settings).')
flags.DEFINE_float('timeout_sec', 70000, 'The timeout to wait for the next checkpoint, as per tf.train.checkpoints_iterator. Pass a small value to run once and exit.')
flags.DEFINE_string('name_enc_cfg_yaml_path', None,
                    'Path to YAML config file for the name encoder network.')
flags.DEFINE_integer(
    'random_encoding_seed', None,
    'Random seed int >= 0, to create random encodings. No such encodings generated if not passed.',
    lower_bound=0)

def evaluate():
  """Evaluates the given model."""
  configs = config_util.get_configs_from_pipeline_file(FLAGS.config_path)
  name_enc_config_yaml = config_util.read_yaml_config(FLAGS.name_enc_cfg_yaml_path) # None case handled here

  model_config = configs['model']
  eval_config = configs['eval_config']
  eval_dataset_config = configs['eval_dataset']
  dataset = inputs.create_input(
      train_eval_config=eval_config,
      dataset_config=eval_dataset_config,
      is_training=False,
      use_tpu=False,
      overfit_expt=FLAGS.overfit_expt,
      random_encoding_seed=FLAGS.random_encoding_seed)

  model_ = model_builder.build(model_config, True, name_encoder_config_yaml=name_enc_config_yaml, dataset_config=eval_dataset_config)
  model_.global_step = tf.Variable(initial_value=0, dtype=tf.int64)
  metrics_ = model_.get_metrics(eval_config)
  evaluator = single_task_evaluator.SingleTaskEvaluator(
      dataset,
      model=model_,
      metrics=metrics_,
      output_dir=FLAGS.output_dir,
      overfit_expt=FLAGS.overfit_expt)

  controller = orbit.Controller(
      evaluator=evaluator,
      checkpoint_manager=tf.train.CheckpointManager(
          tf.train.Checkpoint(model=model_, global_step=model_.global_step),
          directory=FLAGS.model_dir,
          max_to_keep=5),
      eval_summary_dir=os.path.join(FLAGS.model_dir, FLAGS.eval_prefix),
      global_step=model_.global_step)

  controller.evaluate_continuously(timeout=FLAGS.timeout_sec)


def main(_):
  evaluate()


if __name__ == '__main__':
  # run Keras in eager mode as well.
  tf.config.experimental_run_functions_eagerly(True)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  app.run(main)
