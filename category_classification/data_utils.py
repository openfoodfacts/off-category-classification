import pathlib
from typing import List

import numpy as np
import tensorflow as tf
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


def get_labels(ds: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([y for x, y in ds], axis=0)
