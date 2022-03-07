"""Models used for motion primitive experiments."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

    def call(self, inputs):
        return self.encoder(inputs)