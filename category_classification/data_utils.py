from collections import Counter
import itertools
import pathlib
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import dense_to_ragged_batch
import tensorflow_datasets as tfds


def load_dataset(
        name: str,
        split: str,
        features: List[str],
        labels: str,
        labels_vocab: List[str],
        batch_size: int):

    label_lookup = tf.keras.layers.StringLookup(
            vocabulary = labels_vocab,
            output_mode = 'multi_hot',
            num_oov_indices = 1)

    @tf.function
    def _to_xy(x):
        return ({f: x[f] for f in features}, x[labels])

    @tf.function
    def _label_multi_hot(x, y):
        y = label_lookup(y)
        y = y[1:]  # drop OOV inserted by StringLookup
        return (x, y)

    @tf.function
    def _has_label(x, y):
        return tf.math.reduce_max(y, 0) > 0

    ds = (
        tfds.load(name, split=split)
        .map(_to_xy, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .map(_label_multi_hot, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .filter(_has_label)
    )

    if batch_size > 1:
        ds = ds.padded_batch(batch_size)

    return ds


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
