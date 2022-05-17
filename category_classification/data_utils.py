from collections import Counter
import itertools
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import dense_to_ragged_batch


def flat_batch(ds: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    """
    Combine consecutive elements of this dataset into flattened batches.

    Parameters
    ----------
    ds : tf.data.Dataset
        Single-feature dataset.

    batch_size : int
        Batch size for the returned dataset.

    Returns
    -------
    tf.data.Dataset
        Flattened, batched dataset.

    Examples
    --------
    With a dataset containing the following elements:
    ds = [[1, 2], [3], [4, 5, 6], [7, 8], [9]]

    `flat_batch(ds, batch_size=2)` will return:
    [[1, 2, 3], [4, 5, 6, 7, 8], [9]]

    whereas `ds.batch(batch_size=2)` would return:
    [[[1, 2], [3]], [[4, 5, 6], [7]], [[9]]]
    """

    return (
        ds
        .apply(dense_to_ragged_batch(batch_size))
        .map(lambda x: x.merge_dims(0, 1))
    )


def get_vocabulary(ds: tf.data.Dataset, min_freq: int = 1) -> List[str]:
    """
    Get the feature vocabulary.

    Parameters
    ----------
    ds : tf.data.Dataset
        Single-feature dataset.

    min_freq : int
        Minimum token frequency to be included in the vocabulary.
        Tokens strictly below `min_freq` won't be listed.

    Returns
    -------
    list
        List of tokens in the vocabulary
    """
    counter = Counter()
    for batch in ds:
        counter.update(batch.numpy())
    return [
        x[0].decode()
        for x in itertools.takewhile(lambda x: x[1] >= min_freq, counter.most_common())
    ]


def get_feature(ds: tf.data.Dataset, feature_name: str) -> tf.data.Dataset:
    """
    Parameters
    ----------
    ds : tf.data.Dataset
        Dict-based dataset. Nested features are not supported.

    feature_name : str
        Name of the feature to select.

    Returns
    -------
    tf.data.Dataset
        Dataset containing only the feature `feature_name`.
    """
    return ds.map(lambda x: x[feature_name])


def get_features(ds: tf.data.Dataset, feature_names: List[str]) -> tf.data.Dataset:
    """
    Parameters
    ----------
    ds : tf.data.Dataset
        Dict-based dataset. Nested features are not supported.

    feature_names : List[str]
        Names of the features to select.

    Returns
    -------
    tf.data.Dataset
        Dataset containing only the features in `feature_names`.
    """
    return ds.map(lambda x: {k: x[k] for k in feature_names})


def get_labels(ds: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([y for x, y in ds], axis=0)
