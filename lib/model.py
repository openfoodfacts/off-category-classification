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
    {'labels': tf.Tensor, 'scores': tf.Tensor}
        Top predicted labels with their scores.
        Returned tensors will have shape `(predictions.shape[0], k)`.
    """
    batch_size = tf.shape(predictions)[0]

    tiled_labels = tf.tile(tf.constant([labels], dtype='string'), (batch_size, 1))

    top_indices = tf.nn.top_k(predictions, k=k, sorted=True, name='top_k').indices

    top_labels = tf.gather(tiled_labels, top_indices, batch_dims=1)
    top_scores = tf.gather(predictions, top_indices, batch_dims=1)

    return {'labels': top_labels, 'scores': top_scores}


def top_predictions_table(labeled_predictions) -> pd.DataFrame:
    """
    Format the top labeled predictions into a pretty table.

    Parameters
    ----------
    labeled_predictions: {'labels': tf.Tensor, 'scores': tf.Tensor}
        Labeled predictions, as returned by `top_labeled_predictions`.

    Returns
    -------
    pd.DataFrame
    """
    labels = pd.DataFrame(labeled_predictions['labels'].numpy()).stack()
    scores = pd.DataFrame(labeled_predictions['scores'].numpy()).stack()

    df = (
        pd.DataFrame.from_dict({
            'label': labels.apply(lambda x: x.decode()),
            'score': (scores * 100).round(2).astype(str) + '%'
        })
        .agg(lambda x: f"{x['label']}: {x['score']}", axis=1)
        .unstack()
    )

    df.columns = [f"top prediction {i+1}" for i in range(len(df.columns))]

    return df
