from collections import defaultdict
import dataclasses
import pathlib
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from robotoff.utils import gzip_jsonl_iter
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

from category_classification.models import TextPreprocessingConfig
import settings
from utils.constant import UNK_TOKEN
from .constants import NUTRIMENTS, NUTRIMENTS_TO_IDX


def _generate_y(categories_tags: Iterable[Iterable[str]], category_vocab: List[str]):
    category_to_ind = {name: idx for idx, name in enumerate(category_vocab)}
    category_count = len(category_to_ind)
    cat_binarizer = MultiLabelBinarizer(classes=list(range(category_count)))
    category_int = [
        [category_to_ind[cat] for cat in product_categories if cat in category_to_ind]
        for product_categories in categories_tags
    ]
    return cat_binarizer.fit_transform(category_int)

# Load the dataset => build the model using this dataset.

def load_dataframe(type: str) -> pd.DataFrame:
    full_path = settings.DATA_DIR / f"category_xx.{type}.jsonl.gz"
    return pd.DataFrame(_iter_product(full_path))

def convert_to_tf_dataset(df: pd.DataFrame, category_vocab: List[str]) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensors(
        ((tf.ragged.constant(df.ingredient_tags, dtype=tf.string), df.product_name), _generate_y(df["categories_tags"], category_vocab)))


def _iter_product(data_path: pathlib.Path):
    for product in gzip_jsonl_iter(data_path): 
        # Filter out fields we don't need.
        filtered_product = {key: product[key] for key in {"product_name", "categories_tags","ingredient_tags"}}        

        yield filtered_product
