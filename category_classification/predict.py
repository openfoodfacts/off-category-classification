import argparse
import operator
import pathlib

from robotoff.off import get_product
from robotoff.taxonomy import Taxonomy
from tensorflow import keras

import settings
from utils.io import (
    load_config,
    load_product_name_vocabulary,
    load_category_vocabulary,
    load_ingredient_vocabulary,
)
from utils.metrics import fill_ancestors
from utils.preprocess import get_nlp
from category_classification.data_utils import generate_data

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    return parser.parse_args()


args = parse_args()
model_path = args.model_path.resolve()
model_dir = model_path.parent

config = load_config(model_dir)

category_to_id = load_category_vocabulary(model_dir)
ingredient_to_id = load_ingredient_vocabulary(model_dir)
category_names = [
    category
    for category, _ in sorted(category_to_id.items(), key=operator.itemgetter(1))
]

nlp = get_nlp(config.lang)

product_name_vocabulary = load_product_name_vocabulary(model_dir)
model = keras.models.load_model(str(model_path))

while True:
    barcode = input("barcode: ").strip()
    product = get_product(barcode, fields=["product_name", "ingredients_tags"])

    if product is None:
        print("Product {} not found".format(barcode))
        continue

    ingredient_tags = product.get("ingredients_tags", []) or []
    product_name = product.get("product_name", "") or ""

    X = generate_data(
        ingredient_tags=ingredient_tags,
        product_name=product_name,
        ingredient_to_id=ingredient_to_id,
        product_name_token_to_int=product_name_vocabulary,
        nlp=nlp,
        product_name_max_length=config.model_config.product_name_max_length,
        product_name_preprocessing_config=config.product_name_preprocessing_config,
    )

    y_pred = model.predict(X)
    y_pred_int = (y_pred > 0.5).astype(y_pred.dtype)
    taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)
    y_pred_int_filled = fill_ancestors(
        y_pred_int, taxonomy=taxonomy, category_names=category_names
    )

    predicted_categories_ids = y_pred_int_filled[0].nonzero()[0]
    predicted_categories = [category_names[id_] for id_ in predicted_categories_ids]

    predicted = []
    for predicted_category_id, predicted_category in zip(
        predicted_categories_ids, predicted_categories
    ):
        confidence = y_pred[0, predicted_category_id]
        predicted.append((predicted_category, confidence))

    if not predicted:
        print("No category predicted")
        continue

    sorted(predicted, key=operator.itemgetter(1), reverse=True)

    for cat, confidence in predicted:
        print("{}: {}".format(cat, confidence))
