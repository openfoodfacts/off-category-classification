from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf


@tf.function
def top_labeled_predictions(
        predictions: Union[tf.Tensor, np.array],
        labels: List[str],
        k: int = 10):
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
    tf_labels = tf.constant([labels], dtype='string')

    top_indices = tf.nn.top_k(predictions, k=k, sorted=True, name='top_k').indices

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

    cells =  np.vectorize(lambda l, s: f"{l.decode()}: {s:.2%}")(labels, scores)
    columns = [f"top prediction {i+1}" for i in range(labels.shape[1])]

    return pd.DataFrame(cells, columns=columns)
