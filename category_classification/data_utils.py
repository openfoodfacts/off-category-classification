from collections import defaultdict
import dataclasses
import pathlib
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from robotoff.utils import gzip_jsonl_iter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental import preprocessing

import settings
import functools
from .constants import NUTRIMENTS, NUTRIMENTS_TO_IDX

def _data_path(type: str) -> pathlib.Path:
    return settings.DATA_DIR / f"category_xx.{type}.jsonl.gz"


def load_dataframe(type: str) -> pd.DataFrame:
    return pd.DataFrame(_iter_product(_data_path(type)))

class TFTransformer():
    def __init__(self, category_vocab: List[str]):
        self.category_to_ind = {name: idx for idx, name in enumerate(category_vocab)}
        self.category_size = len(category_vocab)
        # TODO(kulizhsy): consider using SparseTensors?
        self.encoder = tf.keras.layers.CategoryEncoding(num_tokens=self.category_size, output_mode="multi_hot")

    def transform(self, product: Dict):
        category_int = [self.category_to_ind[cat] for cat in product["categories_tags"] if cat in self.category_to_ind]
        # Try to return ragged tensors here?
        return (product["known_ingredient_tags"], tf.constant(product["product_name"], dtype=tf.string)), self.encoder(category_int)

def create_tf_dataset(type: str, batch_size: int, transformer: TFTransformer) -> tf.data.Dataset:
    generator = functools.partial(
        _iter_product,
        data_path = _data_path(type),
        tf_transformer= transformer.transform)
    return tf.data.Dataset.from_generator(generator,
        output_signature=(
            (tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string)),
            tf.TensorSpec(shape=(transformer.category_size,), dtype=tf.int32)
            )
        ).padded_batch(batch_size)

counter = 0

def _iter_product(data_path: pathlib.Path, tf_transformer: Callable = None):
    global counter
    for product in gzip_jsonl_iter(data_path): 
        if counter == 3000:
            counter = 0
            break
            counter = counter + 1
        if tf_transformer:
            tf = tf_transformer(product)
            yield tf
        else:
            # Filter out fields we don't need.
            filtered_product = {key: product[key] for key in {"product_name", "categories_tags","known_ingredient_tags"}}       
            yield filtered_product
