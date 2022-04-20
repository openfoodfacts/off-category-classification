import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from robotoff.utils import gzip_jsonl_iter

import settings


def _data_path(split: str) -> pathlib.Path:
    return settings.DATA_DIR / f"category_xx.{split}.jsonl.gz"


def load_dataframe(split: str) -> pd.DataFrame:
    return pd.DataFrame(_iter_product(_data_path(split)))


@tf.function
def _to_xy(*args):
    """
    Convert flat dataset to ([features], labels) format expected by `model.fit`
    """
    return (args[:-1], args[-1])


def _label_multi_hot(category_vocab):
    """
    Multi-hot encode the labels.
    """
    category_multihot = tf.keras.layers.StringLookup(
        vocabulary=category_vocab,
        output_mode="multi_hot",
        num_oov_indices=1)

    @tf.function
    def _label_multi_hot(x, y):
        y = category_multihot(y)
        y = y[1:]  # drop OOV inserted by StringLookup
        return (x, y)

    return _label_multi_hot


@tf.function
def _has_label(x, y):
    """
    Drop tf.Dataset entries that do not have any label.
    """
    return tf.math.reduce_max(y, 0) > 0


def create_tf_dataset(
    split: str,
    category_vocab: List[str],
    batch_size: int
) -> tf.data.Dataset:
    def generator():
        for p in _iter_product(_data_path(split)):
            yield tuple(p.values())

    return (
        tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None,), dtype=tf.string),
            )
        )
        .map(_to_xy, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .map(_label_multi_hot(category_vocab), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .filter(_has_label)
        .padded_batch(batch_size)
    )


def get_labels(ds: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([y for x, y in ds], axis=0)


def _iter_product(data_path: pathlib.Path):
    # feature order matters for `create_tf_dataset`
    features = ["known_ingredient_tags", "product_name"]
    labels = "categories_tags"
    columns = features + [labels]

    for product in gzip_jsonl_iter(data_path):
        yield {key: product[key] for key in columns}
