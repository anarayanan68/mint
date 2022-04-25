"""Models used for motion primitive experiments."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers

from mint.core import fact_model, base_model_util, base_models
from mint.utils import inputs_util


class BlendController(keras.Model):
    def __init__(self, num_primitives, cond_vocab_size, config_dict, name="BlendController", **kwargs):
        super(BlendController, self).__init__(name=name, **kwargs)

        self.num_primitives = num_primitives
        
        transformer_config_yaml = config_dict['transformer']
        self.audio_linear_embedding = base_models.LinearEmbedding(
            transformer_config_yaml['hidden_size'])
        self.audio_pos_embedding = base_models.PositionEmbedding(
            transformer_config_yaml['sequence_length'],
            transformer_config_yaml['hidden_size'])
        self.transformer = base_models.Transformer(
            hidden_size=transformer_config_yaml['hidden_size'],
            num_hidden_layers=transformer_config_yaml['num_hidden_layers'],
            num_attention_heads=transformer_config_yaml['num_attention_heads'],
            intermediate_size=transformer_config_yaml['intermediate_size']
        )

        initializer = initializers.RandomNormal()   # so that, hopefully, each embedding is different from the start
        self.conditioning_block = layers.Embedding(cond_vocab_size, transformer_config_yaml['hidden_size'],
            input_length=1, embeddings_initializer=initializer, embeddings_regularizer=None, name='cond_input_embedding')

        output_block_config_yaml = config_dict['output_block']
        self.output_block = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            base_models.MLP(out_dim=num_primitives, hidden_dim=output_block_config_yaml['hidden_dim']),
            layers.Softmax()
        ])


    def call(self, inputs):
        audio_seq = inputs['audio_input']                               # (batch_size, seq_len, audio_feature_dim)
        audio_features = self.audio_linear_embedding(audio_seq)         # (batch_size, seq_len, transformer_hidden_size)
        audio_features = self.audio_pos_embedding(audio_features)       # (batch_size, seq_len, transformer_hidden_size)
        audio_features = self.transformer(audio_features)               # (batch_size, seq_len, transformer_hidden_size)

        conditioning = inputs['conditioning_input']                     # (batch_size, conditioning_input_dim)
        conditioning_features = self.conditioning_block(conditioning)   # (batch_size, 1, transformer_hidden_size)

        combined_features = audio_features + conditioning_features      # (batch_size, seq_len, transformer_hidden_size)
        out_vec = self.output_block(combined_features)                  # (batch_size, num_primitives)
        return out_vec


class BlendVecToSeq(keras.Model):
    def __init__(self, num_primitives, config_dict, name="BlendVecToSeq", **kwargs):
        super(BlendVecToSeq, self).__init__(name=name, **kwargs)

        self.num_primitives = num_primitives
        
        self.target_shape = tuple((int(x) for x in config_dict['target_shape'].split(',')))

        prod_target_shape = np.prod(self.target_shape)
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


class EncFACTJointModel(keras.Model):
    def __init__(self, fact_config, is_training, num_primitives, encoder_config_yaml, dataset_config, name="EncFACTJointModel", **kwargs):
        super(EncFACTJointModel, self).__init__(name=name, **kwargs)

        self.blend_controller = BlendController(
            num_primitives=num_primitives,
            cond_vocab_size=encoder_config_yaml['conditioning_vocab_size'],
            config_dict=encoder_config_yaml['audio_to_blend_vec'],
        )
        self.blend_vec_to_seq = BlendVecToSeq(
            num_primitives=num_primitives,
            config_dict=encoder_config_yaml['blend_vec_to_seq'],
        )
        self.fact_stage = fact_model.FACTModel(fact_config, is_training)

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
        blend_vec = self.blend_controller(inputs)
        seq = self.blend_vec_to_seq(blend_vec)
        inputs['motion_enc_seq'] = seq

        self.middle_processing(inputs)
        return self.fact_stage(inputs)


    def loss(self, target_tensors, pred_tensors, inputs):
        return self.fact_stage.loss(target_tensors, pred_tensors, inputs)
