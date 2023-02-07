from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf


@tf.function
def top_labeled_predictions(
    predictions: Union[tf.Tensor, np.array], labels: List[str], k: int = 10
):
    """
    Top labeled predictions.

    This `@tf.function` can be used as a custom serving function.

    Parameters
    ----------
    predictions: tf.Tensor or np.array
        Predictions, as returned by `model.predict` or equivalent.

    labels: List[str]
        Label vocabulary.

    k: int, optional
        Number of top predictions to return.

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        Top predicted labels with their scores, as (scores, labels).
        Returned tensors will have shape `(predictions.shape[0], k)`.
    """
    tf_labels = tf.constant([labels], dtype="string")

    top_indices = tf.nn.top_k(predictions, k=k, sorted=True, name="top_k").indices

    top_labels = tf.experimental.numpy.take(tf_labels, top_indices)
    top_scores = tf.gather(predictions, top_indices, batch_dims=1)

    return top_scores, top_labels


def top_predictions_table(labeled_predictions) -> pd.DataFrame:
    """
    Format the top labeled predictions into a pretty table.

    Parameters
    ----------
    labeled_predictions: (tf.Tensor, tf.Tensor)
        Labeled predictions, as returned by `top_labeled_predictions`.

    Returns
    -------
    pd.DataFrame
    """
    labels = labeled_predictions[1].numpy()
    scores = labeled_predictions[0].numpy()

    cells = np.vectorize(lambda l, s: f"{l.decode()}: {s:.2%}")(labels, scores)
    columns = [f"top prediction {i+1}" for i in range(labels.shape[1])]

    return pd.DataFrame(cells, columns=columns)


def generate_mask_matrix(inputs):
    """Generate mask matrix for multi-head self-attention layer from a mask
    tensor of (batch_size, sequence_dim).

    Return a mask matrix of shape (batch_size, sequence_dim, sequence_dim)."""
    shape = tf.shape(inputs)
    batch_size = shape[0]
    sequence_dim = shape[1]
    mask_matrix = tf.ones((batch_size, sequence_dim, sequence_dim))
    indices = tf.where(inputs == 0.0)
    output = tf.tensor_scatter_nd_update(
        mask_matrix, indices, tf.zeros((tf.shape(indices)[0], sequence_dim))
    )
    return tf.tensor_scatter_nd_update(
        tf.transpose(output, perm=[0, 2, 1]),
        indices,
        tf.zeros((tf.shape(indices)[0], sequence_dim)),
    )


def replace_nan_by_zero(x):
    """Replace all NaN values by zero.

    GlobalAveragePooling1D produces NaN values when all steps are masked, this
    occurs when they there are only padding tokens as input."""
    indices = tf.where(tf.math.is_nan(x))
    batch_size = tf.shape(indices)[0]
    return tf.tensor_scatter_nd_update(x, indices, tf.zeros(batch_size))


def build_attention_over_sequence_layer(
    embedding_dim: int, input_name: str, num_heads: int, key_dim: int
):
    image_embedding_input = tf.keras.Input(shape=[None, embedding_dim], name=input_name)
    image_embedding_mask_input = tf.keras.Input(shape=[None], name=f"{input_name}_mask")
    image_embedding_full_mask = tf.keras.layers.Lambda(
        generate_mask_matrix, name="attention_mask_builder"
    )(image_embedding_mask_input)
    attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )
    attention_output, attention_scores = attention_layer(
        query=image_embedding_input,
        value=image_embedding_input,
        attention_mask=image_embedding_full_mask,
        return_attention_scores=True,
    )
    average_output = tf.keras.layers.GlobalAveragePooling1D()(
        attention_output, image_embedding_mask_input
    )
    average_output = tf.keras.layers.Lambda(replace_nan_by_zero, name="nan_remover")(
        average_output
    )
    return [image_embedding_input, image_embedding_mask_input], [
        attention_output,
        attention_scores,
        average_output,
    ]
