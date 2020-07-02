from collections import defaultdict
import dataclasses
import pathlib
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from robotoff.utils import gzip_jsonl_iter
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from category_classification.models import TextPreprocessingConfig
import settings
from utils.constant import UNK_TOKEN
from utils.preprocess import (
    extract_vocabulary_from_counter,
    generate_y,
    preprocess_product_name,
    tokenize,
)
from .constants import NUTRIMENTS, NUTRIMENTS_TO_IDX


def create_dataframe(split: str, lang: str) -> pd.DataFrame:
    if split not in ("train", "test", "val"):
        raise ValueError("split must be either 'train', 'test' or 'val'")

    file_name = "category_{}.{}.jsonl.gz".format(lang, split)
    full_path = settings.DATA_DIR / file_name
    return pd.DataFrame(iter_product(full_path))


def iter_product(data_path: pathlib.Path):
    for product in gzip_jsonl_iter(data_path):
        product.pop("images", None)

        if "nutriments" in product:
            nutriments = product["nutriments"] or {}
            for key in list(nutriments.keys()):
                if key not in NUTRIMENTS:
                    nutriments.pop(key)

        yield product


def analyze_dataset(
    data_path: pathlib.Path,
    min_category_count: int,
    min_ingredient_count: int,
    product_name_min_count: int,
    product_name_process_fn: Callable,
):
    category_count: Dict[str, int] = defaultdict(int)
    ingredient_count: Dict[str, int] = defaultdict(int)
    vocabulary: Dict[str, int] = defaultdict(int)

    for product in iter_product(data_path):
        product_name = product.get("product_name", "") or ""
        tokens = product_name_process_fn(product_name)

        for token in tokens:
            vocabulary[token] += 1

        for category in product["categories_tags"]:
            category_count[category] += 1

        for ingredient_tag in product["ingredients_tags"]:
            ingredient_count[ingredient_tag] += 1

    category_to_id = filter_min_count(category_count, min_category_count)
    ingredient_to_id = filter_min_count(ingredient_count, min_ingredient_count)
    token_to_id = extract_vocabulary_from_counter(vocabulary, product_name_min_count)

    return category_to_id, ingredient_to_id, token_to_id


def filter_min_count(counter: Dict[str, int], min_count: int):
    selected = sorted(
        set((cat for cat, count in counter.items() if count >= min_count))
    )
    return {name: idx for idx, name in enumerate(selected)}


def generate_data_from_df(
    df: pd.DataFrame,
    ingredient_to_id: Dict,
    category_to_id: Dict,
    product_name_token_to_int: Dict[str, int],
    nlp,
    product_name_max_length: int,
    product_name_preprocessing_config: TextPreprocessingConfig,
    nutriment_input: bool,
):
    ingredient_matrix = process_ingredients(
        df.known_ingredient_tags, ingredient_to_id
    ).astype(np.float32)
    product_name_matrix = process_product_name(
        df.product_name,
        nlp=nlp,
        token_to_int=product_name_token_to_int,
        max_length=product_name_max_length,
        preprocessing_config=product_name_preprocessing_config,
    )

    inputs = [ingredient_matrix, product_name_matrix]

    if nutriment_input:
        nutriments_matrix = process_nutriments(df.nutriments)
        inputs.append(nutriments_matrix)

    y = generate_y(df.categories_tags, category_to_id)
    return inputs, y


def generate_data(
    product: Dict[str, Any],
    ingredient_to_id: Dict,
    product_name_token_to_int: Dict[str, int],
    nlp,
    product_name_max_length: int,
    product_name_preprocessing_config: TextPreprocessingConfig,
    nutriment_input: bool,
):
    ingredient_tags = product.get("ingredients_tags", []) or []
    product_name = product.get("product_name", "") or ""
    nutriments = product.get("nutriments") or None
    ingredient_matrix = process_ingredients([list(ingredient_tags)], ingredient_to_id)
    product_name_matrix = process_product_name(
        [product_name],
        nlp=nlp,
        token_to_int=product_name_token_to_int,
        max_length=product_name_max_length,
        preprocessing_config=product_name_preprocessing_config,
    )

    inputs = [ingredient_matrix, product_name_matrix]
    if nutriment_input:
        nutriments_matrix = process_nutriments([nutriments])
        inputs.append(nutriments_matrix)

    return inputs


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
