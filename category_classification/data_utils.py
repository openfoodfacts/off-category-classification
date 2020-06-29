import dataclasses
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from robotoff.utils import gzip_jsonl_iter
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from category_classification.models import TextPreprocessingConfig
import settings
from utils.constant import UNK_TOKEN
from utils.preprocess import generate_y, preprocess_product_name, tokenize


NUTRIMENTS = [
    "energy-kcal_100g",
    "proteins_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fat_100g",
    "saturated-fat_100g",
    "fiber_100g",
    "sodium_100g",
    "alcohol_100g",
    "fruits-vegetables-nuts_100g",
]

NUTRIMENT_TO_IDX = {nutrient: idx for idx, nutrient in enumerate(NUTRIMENTS)}


def create_dataframe(split: str, lang: str) -> pd.DataFrame:
    if split not in ("train", "test", "val"):
        raise ValueError("split must be either 'train', 'test' or 'val'")

    file_name = "category_{}.{}.jsonl.gz".format(lang, split)
    full_path = settings.DATA_DIR / file_name
    return pd.DataFrame(gzip_jsonl_iter(full_path))


def generate_data_from_df(
    df: pd.DataFrame,
    ingredient_to_id: Dict,
    category_to_id: Dict,
    product_name_token_to_int: Dict[str, int],
    nlp,
    product_name_max_length: int,
    product_name_preprocessing_config: TextPreprocessingConfig,
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
    y = generate_y(df.categories_tags, category_to_id)
    return [ingredient_matrix, product_name_matrix], y


def generate_data(
    ingredient_tags: Iterable[str],
    product_name: str,
    ingredient_to_id: Dict,
    product_name_token_to_int: Dict[str, int],
    nlp,
    product_name_max_length: int,
    product_name_preprocessing_config: TextPreprocessingConfig,
):
    ingredient_matrix = process_ingredients([list(ingredient_tags)], ingredient_to_id)
    product_name_matrix = process_product_name(
        [product_name],
        nlp=nlp,
        token_to_int=product_name_token_to_int,
        max_length=product_name_max_length,
        preprocessing_config=product_name_preprocessing_config,
    )
    return [ingredient_matrix, product_name_matrix]


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


def process_nutriments(nutriments_iter: Iterable[Dict]) -> np.ndarray:
    nutriments_list = list(nutriments_iter)

    array = np.zeros((len(nutriments_list), len(NUTRIMENTS)), type=np.float32)

    for i, product_nutriments in enumerate(nutriments_list):
        for nutriment in NUTRIMENTS:
            if nutriment in product_nutriments:
                array[i, NUTRIMENT_TO_IDX[nutriment]] = product_nutriments[nutriment]
