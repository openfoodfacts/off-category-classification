from typing import List

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class OutputMapperLayer(tf.keras.layers.Layer):
    """
    The OutputMapperLayer converts the label indices produced by the model to
    the taxonomy category ids and limits them to top N labels.
    """

    def __init__(self, labels: List[str], top_n: int, **kwargs):
        self.labels = labels
        self.top_n = top_n

        super(OutputMapperLayer, self).__init__(**kwargs)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        tf_labels = tf.constant([self.labels], dtype="string")
        tf_labels = tf.tile(tf_labels, [batch_size, 1])

        top_n = tf.nn.top_k(x, k=self.top_n, sorted=True, name="top_k").indices

        top_conf = tf.gather(x, top_n, batch_dims=1)
        top_labels = tf.gather(tf_labels, top_n, batch_dims=1)

        return (top_conf, top_labels)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        top_shape = (batch_size, self.top_n)
        return [top_shape, top_shape]

    def get_config(self):
        config = {"labels": self.labels, "top_n": self.top_n}
        base_config = super(OutputMapperLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def to_serving_model(base_model: tf.keras.Model, categories: List[str]) -> tf.keras.Model:
    mapper_layer = OutputMapperLayer(categories, 50)(base_model.output)
    return tf.keras.Model(base_model.input, mapper_layer)
