from collections import defaultdict
import dataclasses
import pathlib
from typing import Any, Callable, Dict, Iterable, Optional, List

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
from utils.preprocess import (
    generate_y
)
from .constants import NUTRIMENTS, NUTRIMENTS_TO_IDX


def create_dataframes() -> Dict[str, pd.DataFrame]:
    dfs = {}
    for split in ("train", "test", "val"):
        file_name = "category_xx.{}.jsonl.gz".format(split)
        full_path = settings.DATA_DIR / file_name
        dfs[split] = pd.DataFrame(iter_product(full_path))
    return dfs


def iter_product(data_path: pathlib.Path):
    for product in gzip_jsonl_iter(data_path): 
        # Filter out fields we don't need.
        filtered_product = {key: product[key] for key in {"product_name", "categories_tags","ingredient_tags", "nutriments"}}

        # Filter for only supported nutriments.
        if "nutriments" in filtered_product:
            nutriments = filtered_product["nutriments"] or {}
            for key in list(nutriments.keys()):
                if key not in NUTRIMENTS:
                    nutriments.pop(key)
        yield filtered_product
        break

def generate_data_from_df(
    df: pd.DataFrame,
    category_to_id: Dict,
    categories: List[str],
    nutriment_input: bool,
):

    inputs = [tf.ragged.constant(df.ingredient_tags, dtype=tf.string), tf.constant(df.product_name, dtype=tf.string)]
    if nutriment_input:
        nutriments_matrix = process_nutriments(df.nutriments)
        inputs.append(nutriments_matrix)


    # categoriser = preprocessing.StringLookup(vocabulary=categories)

    # print(f"Vocabulary is {categoriser.get_vocabulary()}")

    # print(f"Category tags are {df.categories_tags}")
    # y = categoriser(tf.constant(df.categories_tags))
    # print(y)
    # print("\n\n")
    y = generate_y(df.categories_tags, category_to_id)
    return inputs, y


def get_nutriment_value(nutriments: Dict[str, Any], nutriment_name: str) -> float:
    if nutriment_name in nutriments:
        value = nutriments[nutriment_name]

        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return 0.0

        if isinstance(value, float):
            if value == float("nan"):
                return 0.0

            return value

    return 0.0


def process_nutriments(nutriments_iter: Iterable[Optional[Dict]]) -> np.ndarray:
    nutriments_list = list(nutriments_iter)

    array = np.zeros((len(nutriments_list), len(NUTRIMENTS)), type=np.float32)

    for i, product_nutriments in enumerate(nutriments_list):
        if product_nutriments is not None:
            for nutriment in NUTRIMENTS:
                array[i, NUTRIMENTS_TO_IDX[nutriment]] = get_nutriment_value(
                    product_nutriments, nutriment
                )
