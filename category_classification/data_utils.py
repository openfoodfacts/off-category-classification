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

from category_classification.models import TextPreprocessingConfig
import settings
from utils.constant import UNK_TOKEN
from utils.preprocess import (
    extract_vocabulary_from_counter,
    preprocess_product_name,
    tokenize,
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

    inputs = [df.ingredients, df.product_name]

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


def process_ingredients(
    ingredients: Iterable[Iterable[str]], ingredient_to_id: Dict[str, int]
) -> np.ndarray:
    ingredient_count = len(ingredient_to_id)
    ingredient_binarizer = MultiLabelBinarizer(classes=list(range(ingredient_count)))
    ingredient_int = [
        [
            ingredient_to_id[ing]
            for ing in product_ingredients
            if ing in ingredient_to_id
        ]
        for product_ingredients in ingredients
    ]
    return ingredient_binarizer.fit_transform(ingredient_int)


def process_product_name(
    product_names: Iterable[str],
    nlp,
    token_to_int: Dict,
    max_length: int,
    preprocessing_config: TextPreprocessingConfig,
) -> np.ndarray:
    tokens_all = [
        tokenize(
            preprocess_product_name(text, **dataclasses.asdict(preprocessing_config)),
            nlp,
        )
        for text in product_names
    ]
    tokens_int = [
        [token_to_int[t if t in token_to_int else UNK_TOKEN] for t in tokens]
        for tokens in tokens_all
    ]
    return pad_sequences(tokens_int, max_length)


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
