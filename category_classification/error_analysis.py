## TODO(kulizhsy): check if this still works
import argparse
import functools
import operator
import pathlib
from typing import Dict

import pandas as pd
from bokeh.plotting import show
from robotoff.taxonomy import Taxonomy
from robotoff.utils import gzip_jsonl_iter
from tensorflow import keras

import settings
from category_classification.data_utils import generate_data_from_df
from utils.error_analysis import (
    generate_analysis_model,
    get_deepest_categories,
    get_error_category,
    get_interactive_embedding_plot,
)
from utils.io import (
    load_category_vocabulary,
    load_config,
    load_ingredient_vocabulary,
    load_product_name_vocabulary,
)
from utils.metrics import evaluation_report
from utils.preprocess import get_nlp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "weights/saved_model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_path.parent

    config = load_config(model_dir)

    category_to_id = load_category_vocabulary(model_dir)
    ingredient_to_id = load_ingredient_vocabulary(model_dir)
    category_names = [
        category
        for category, _ in sorted(category_to_id.items(), key=operator.itemgetter(1))
    ]

    nlp = get_nlp(config.lang)

    product_name_vocabulary = load_product_name_vocabulary(model_dir)
    model = keras.models.load_model(str(args.model_path))

    analysis_model = generate_analysis_model(model, "dense")

    generate_data_partial = functools.partial(
        generate_data_from_df,
        ingredient_to_id=ingredient_to_id,
        category_to_id=category_to_id,
        product_name_max_length=config.model_config.product_name_max_length,
        product_name_token_to_int=product_name_vocabulary,
        nlp=nlp,
        product_name_preprocessing_config=config.product_name_preprocessing_config,
        nutriment_input=config.model_config.nutriment_input,
    )

    val_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_XX_VAL_PATH))

    category_taxonomy: Taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)

    val_df["deepest_categories"] = get_deepest_categories(
        category_taxonomy, val_df.categories_tags
    )

    X_val, y_val = generate_data_partial(val_df)

    y_pred_val = model.predict(X_val)

    val_df["predicted_deepest_categories"] = get_deepest_categories(
        category_taxonomy,
        [
            [category_names[i] for i, conf in enumerate(y) if conf >= 0.5]
            for y in y_pred_val
        ],
    )

    val_df["is_correct"] = [
        source == predicted
        for source, predicted in zip(
            val_df.deepest_categories, val_df.predicted_deepest_categories
        )
    ]

    error_categories_val = [
        get_error_category(predicted, true, category_taxonomy)
        for predicted, true in zip(
            val_df.predicted_deepest_categories, val_df.deepest_categories
        )
    ]

    (
        val_df["missing_cat_error"],
        val_df["additional_cat_error"],
        val_df["over_pred_cat_error"],
        val_df["under_pred_cat_error"],
    ) = zip(*error_categories_val)

    val_df["url"] = [
        "https://world.openfoodfacts.org/product/{}".format(c) for c in val_df.code
    ]
    val_df.to_csv(str(model_dir / "error_analysis.tsv"), sep="\t")

    emb_val = analysis_model.predict(X_val)

    p = get_interactive_embedding_plot(emb_val, val_df)
    show(p)


if __name__ == "__main__":
    main()
