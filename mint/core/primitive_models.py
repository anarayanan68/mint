"""Models used for motion primitive experiments."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers

from mint.core import fact_model, base_model_util
from mint.utils import inputs_util


class Vec2SeqEncoder(keras.Model):
    def __init__(self, num_primitives, target_shape, wt_seed, name="Vec2SeqEncoder", **kwargs):
        super(Vec2SeqEncoder, self).__init__(name=name, **kwargs)

        self.num_primitives = num_primitives
        self.target_shape = target_shape
        self.wt_seed = wt_seed

        prod_target_shape = np.prod(target_shape)
        initializer = initializers.Constant(1.0 / np.sqrt(prod_target_shape)) # for unit output vector norm
        regularizer = regularizers.L2(l2=1e-3)
        self.embedding = layers.Embedding(num_primitives, prod_target_shape,
            input_length=num_primitives, embeddings_initializer=initializer, embeddings_regularizer=None, name='embedding')

        self.emb_input = tf.convert_to_tensor(np.arange(num_primitives))
        
    def call(self, vec):
        # vec shape: (batch_size, num_primitives), will be expanded to (batch_size, 1, num_primitives) here
        # embedding should be of shape: (batch_size, num_primitives, prod(target_shape)), collecting embeddings for all indices
        # then batch matrix mult of vec with embedding will yield weighted average, which can be reshaped and returned

        batch_size = vec.shape[0]
        # collect all possible indices, repeated per batch
        emb_input = tf.broadcast_to(self.emb_input, (batch_size, self.num_primitives))
        # get embeddings for all indices in correct dtype
        embeddings = tf.cast(self.embedding(emb_input), vec.dtype)
        # get weighted average according to input vector
        res = tf.matmul(tf.expand_dims(vec, 1), embeddings)
        # reshape to target shape and return
        return tf.reshape(res, (batch_size,) + self.target_shape)


class NameFACTJointModel(keras.Model):
    def __init__(self, fact_config, is_training, encoder_config_yaml, dataset_config, name="NameFACTJointModel", **kwargs):
        super(NameFACTJointModel, self).__init__(name=name, **kwargs)

        self.fact_stage = fact_model.FACTModel(fact_config, is_training)
        self.enc_stage = Vec2SeqEncoder(
            num_primitives=encoder_config_yaml['num_primitives'],
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
        inputs["motion_input"] = inputs["motion_enc_seq"][:,
                                                                start:start +
                                                                motion_input_length, :]
        inputs["motion_input"].set_shape([inputs["motion_input"].shape[0], motion_input_length, motion_dim])

        del inputs["motion_enc_seq"]


    def call(self, inputs):
        vec = inputs['motion_enc']
        seq = self.enc_stage(vec)
        inputs['motion_enc_seq'] = seq

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
        in_latents = inputs["motion_enc"]
        diff = target_tensors - pred_tensors[:, None, :target_seq_len]

        # Loss is first averaged over the sequence and feature dimensions
        # # IF using Smooth Huber loss:
        # delta = 1.0
        # direct_loss = delta**2 * tf.reduce_mean(tf.sqrt(1.0 + tf.square(diff/delta)) - 1, axis=[-1,-2])

        # IF using MSE loss:
        direct_loss = tf.reduce_mean(tf.square(diff), axis=[-1,-2])

        # -> now of shape (batch_size, latent_dim), need to weight and sum over latent dim and then average over batch dim
        blended_loss = tf.reduce_mean(tf.reduce_sum(direct_loss * in_latents, axis=-1))
        return blended_loss
