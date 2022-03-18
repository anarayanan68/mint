"""Additional losses."""
import tensorflow as tf
from mint.core import base_model_util

class TVLoss:
    def __init__(self, weight) -> None:
        self.weight = weight

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        shape = base_model_util.get_shape_list(y_true)
        batch_size = shape[0]
        target_seq_len, feature_dim = shape[-2], shape[-1]

        y_pred_relevant = y_pred[:, :target_seq_len]
        count = (target_seq_len - 1) * feature_dim

        tv = tf.reduce_sum(tf.pow(y_pred_relevant[:,1:] - y_pred_relevant[:,:target_seq_len-1], 2))
        return self.weight * (tv / count) / batch_size