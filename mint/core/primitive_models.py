"""Models used for motion primitive experiments."""

import imp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mint.core import fact_model, base_model_util
from mint.utils import inputs_util


class Vec2SeqEncoder(keras.Model):
    def __init__(self, input_dim, hidden_size, target_shape, wt_seed, name="Vec2SeqEncoder", **kwargs):
        super(Vec2SeqEncoder, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.target_shape = target_shape
        self.wt_seed = wt_seed

        initializer = keras.initializers.RandomNormal(mean=0., stddev=1., seed=wt_seed)
        self.encoder = keras.Sequential(
            [
                layers.Input((input_dim,)),
                layers.Dense(hidden_size, activation='tanh', use_bias=True, kernel_initializer=initializer, bias_initializer=initializer),
                layers.Dense(np.prod(target_shape), use_bias=True, kernel_initializer=initializer, bias_initializer=initializer),
                layers.Reshape(target_shape)
            ],
            name="encoder"
        )

        self.build((None,input_dim))    # needed before calling summary() etc. No intention to build on the fly

    def call(self, vec):
        return self.encoder(vec)


class NameFACTJointModel(keras.Model):
    def __init__(self, fact_config, is_training, encoder_config_yaml, dataset_config, name="NameFACTJointModel", **kwargs):
        super(NameFACTJointModel, self).__init__(name=name, **kwargs)

        self.fact_stage = fact_model.FACTModel(fact_config, is_training)
        self.name_enc_stage = Vec2SeqEncoder(
            input_dim=encoder_config_yaml['input_dim'],
            hidden_size=encoder_config_yaml['hidden_size'],
            target_shape=tuple((int(x) for x in encoder_config_yaml['target_shape'].split(','))),
            wt_seed=encoder_config_yaml['wt_seed'],
        )

        self.modality_to_params = inputs_util.get_modality_to_param_dict(dataset_config)

        self.get_metrics = self.fact_stage.get_metrics


    def middle_processing(self, inputs):
        motion_input_length = self.modality_to_params["motion"]["input_length"]
        motion_dim = self.modality_to_params["motion"]["feature_dim"]

        start = 0
        # so-called "motion input": [start, start + motion_input_length) but derived from encoding
        # key left unchanged for compatibility with model code
        inputs["motion_input"] = inputs["motion_name_enc_seq"][:,
                                                                start:start +
                                                                motion_input_length, :]
        inputs["motion_input"].set_shape([inputs["motion_input"].shape[0], motion_input_length, motion_dim])

        del inputs["motion_name_enc_seq"]


    def call(self, inputs):
        vec = inputs['motion_name_enc']
        seq = self.name_enc_stage(vec)
        inputs['motion_name_enc_seq'] = seq

        self.middle_processing(inputs)
        return self.fact_stage(inputs)


    def loss(self, target_tensors, pred_tensors, inputs):
        _, _, target_seq_len, _ = base_model_util.get_shape_list(target_tensors)

        ## Pseudo-Huber (smooth) loss per latent, *blended over all clips*
        ##    -> Pseudo-Huber loss from https://en.wikipedia.org/wiki/Huber_loss#Pseudo-Huber_loss_function
        #
        # targets shape (batch_size, latent_dim, target_seq_len, feature_dim)
        # preds shape   (batch_size, pred_seq_len, feature_dim)
        #     -> will be sliced to match sequence lengths, and broadcasted to align with targets
        #
        # also use input latents of shape (batch_size, latent_dim) to weigh the losses w.r.t each target
        in_latents = inputs["motion_name_enc"]
        diff = target_tensors - pred_tensors[:, None, :target_seq_len]

        delta = 1.0
        # Loss is first averaged over the sequence and feature dimensions
        smooth_huber_loss = delta**2 * tf.reduce_mean(tf.sqrt(1.0 + tf.square(diff/delta)) - 1, axis=[-1,-2])
        # -> now of shape (batch_size, latent_dim), need to weight and sum over latent dim and then average over batch dim
        blended_loss = tf.reduce_mean(tf.reduce_sum(smooth_huber_loss * in_latents, axis=-1))
        return blended_loss
