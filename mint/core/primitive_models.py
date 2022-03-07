"""Models used for motion primitive experiments."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from mint.core import fact_model


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
    def __init__(self, fact_config, encoder_config_yaml, name="NameFACTJointModel", **kwargs):
        super(NameFACTJointModel, self).__init__(name=name, **kwargs)

        self.fact_stage = fact_model.FACTModel(fact_config, kwargs.pop('is_training', False))
        self.name_enc_stage = Vec2SeqEncoder(
            input_dim=encoder_config_yaml['input_dim'],
            hidden_size=encoder_config_yaml['hidden_size'],
            target_shape=encoder_config_yaml['target_shape'],
            wt_seed=encoder_config_yaml['wt_seed'],
        )

    def call(self, inputs):
        vec = inputs['motion_name_enc']
        seq = self.name_enc_stage(vec)

        inputs['motion_name_enc_seq'] = seq
        return self.fact_stage(inputs)
