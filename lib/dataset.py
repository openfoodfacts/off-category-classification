from collections import Counter
from functools import partial
import itertools
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data.experimental import dense_to_ragged_batch


def load_dataset(name: str, features: List[str] = None, **kwargs) -> tf.data.Dataset:
    """
    Thin wrapper around `tfds.load`.

    Works around the unsupported case combining `features` selection
    (partial decoding) and `as_supervised`=True.

    See https://www.tensorflow.org/datasets/api_docs/python/tfds/load
    for the full documentation on `tfds.load`.

    Parameters
    ----------
    name : str
        Registered name of the dataset.

    features : List[str], optional
        List of features to include (partial decoding).
        Include all features if None.

        Note that partial decoding is applied to the 'x' part of the
        '(x, y)' dataset when `as_supervised=True`, so only actual
        features should be listed in `features`. The labels will
        always be returned.
        However, when `as_supervised=False`, only features explicitly
        included in `features` will be returned.

    **kwargs : dict, optional
        Additional keyword arguments passed to `tfds.load`.

    Returns
    -------
    Same as `tfds.load`. Usually a `tf.data.Dataset`.
    """
    wants_partial = features is not None
    wants_supervised = kwargs.get('as_supervised', False)

    if wants_partial:
        if wants_supervised:
            # tfds.load(..., decoders=...) doesn't work in that case
            # this is slower but at least it works...
            feature_subset = partial(select_features, feature_names=features, supervised=True)
            ds = tfds.load(name, **kwargs).apply(feature_subset)
        else:
            decoders = tfds.decode.PartialDecoding({f: True for f in features})
            ds = tfds.load(name, decoders=decoders, **kwargs)
    else:
        ds = tfds.load(name, **kwargs)

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


def select_feature(ds: tf.data.Dataset, feature_name: str, supervised=False) -> tf.data.Dataset:
    """
    Parameters
    ----------
    ds : tf.data.Dataset
        Dict-based dataset. Nested features are not supported.

    feature_name : str
        Name of the feature to select.

    supervised : bool
        True if dataset was built using `tfds.load(..., as_supervised=True)`, False otherwise.

    Returns
    -------
    tf.data.Dataset
        Dataset containing only the feature `feature_name`.
        If supervised=True, y is discarded.
    """
    if supervised:
        return ds.map(lambda x, _: x[feature_name])
    else:
        return ds.map(lambda x: x[feature_name])


def select_features(ds: tf.data.Dataset, feature_names: List[str], supervised=False) -> tf.data.Dataset:
    """
    Parameters
    ----------
    ds : tf.data.Dataset
        Dict-based dataset. Nested features are not supported.

    feature_names : List[str]
        Names of the features to select.

    supervised : bool
        True if dataset was built using `tfds.load(..., as_supervised=True)`, False otherwise.

    Returns
    -------
    tf.data.Dataset
        Dict-based dataset containing only the features in `feature_names`.
    """
    if supervised:
        return ds.map(lambda x, y: ({k: x[k] for k in feature_names}, y))
    else:
        return ds.map(lambda x: {k: x[k] for k in feature_names})


def get_labels(ds: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([y for x, y in ds], axis=0)


def get_vocabulary(ds: tf.data.Dataset, min_freq: int = 1, max_tokens: int = None) -> List[str]:
    """
    Get the feature vocabulary.

    Parameters
    ----------
    ds : tf.data.Dataset
        Single-feature dataset.

    min_freq : int, optional
        Minimum token frequency to be included in the vocabulary.
        Tokens strictly below `min_freq` won't be listed.

    max_tokens : int, optional
        Maximum size of the vocabulary.
        If there are more unique values in the input than the maximum
        vocabulary size, the most frequent terms will be used to
        create the vocabulary.

    Returns
    -------
    list
        List of tokens in the vocabulary
    """
    counter = Counter()
    for batch in ds:
        counter.update(batch.numpy())

    voc = (
        x[0].decode()
        for x in itertools.takewhile(lambda x: x[1] >= min_freq, counter.most_common())
    )

    if max_tokens:
        voc = itertools.islice(voc, max_tokens)

    return list(voc)


def filter_empty_labels(ds: tf.data.Dataset) -> tf.data.Dataset:
    """
    Drop elements with empty labels from a supervised dataset.
    """
    @tf.function
    def _has_labels(x, y):
        return tf.math.reduce_max(y, 0) > 0

    return ds.filter(_has_labels)


def as_dataframe(ds: tf.data.Dataset) -> pd.DataFrame:
    """
    Return the dataset as a pandas dataframe.

    Same as `tfds.as_dataframe`, but with properly decoded string tensors.
    """
    def _maybe_decode(x):
        try:
            return x.decode()
        except (UnicodeDecodeError, AttributeError):
            return x

    return tfds.as_dataframe(ds).applymap(_maybe_decode)
