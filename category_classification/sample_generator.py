import argparse
import functools
import operator
import pathlib
from typing import Dict

from bokeh.plotting import save
import pandas as pd
from robotoff.taxonomy import Taxonomy
from robotoff.utils import gzip_jsonl_iter
from tensorflow import keras

import settings
from category_classification.data_utils import generate_data_from_df
from utils.error_analysis import (
    generate_analysis_model,
    get_deepest_categories,
    get_interactive_embedding_plot,
    get_error_category,
)

from utils.io import (
    load_config,
    load_category_vocabulary,
    load_ingredient_vocabulary,
    load_product_name_vocabulary,
)
from utils.metrics import evaluation_report
from utils.preprocess import get_nlp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path, default= pathlib.Path(__file__).parent / "weights/0/saved_model")
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

    val_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_XX_TEST_PATH))

    category_taxonomy: Taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)

    X_val, y_val = generate_data_partial(val_df)

    y_pred_val = model.predict(X_val)

    predicted = [
            [{category_names[i]: conf} for i, conf in sorted(enumerate(y)) if conf >= 0.5]
            for y in y_pred_val
        ]

    val_df["predicted categories"] = [[p for p in preds if next(iter(p)) in categories]
        for preds, categories in zip(predicted, val_df.categories_tags)]


    val_df["wrong prediction"] = [[p for p in preds if next(iter(p)) not in categories]
            for preds, categories in zip(predicted, val_df.categories_tags)]

    val_df["missed prediction"] = [[category for category in categories if category not in [next(iter(d)) for d in preds]]
        for preds, categories in zip(predicted, val_df.categories_tags)]

    val_df = val_df[(val_df["wrong prediction"].map(len) > 0) | (val_df["missed prediction"].map(len) > 0)]

    val_df.drop(["nutriments", "images", "product_name", "lang", "categories_tags", "ingredient_tags", "ingredients_text", "known_ingredient_tags"], axis=1, inplace=True)

    val_df.rename(columns={"code": "barcode"}, inplace=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    val_df.head(n=100).to_csv('misprediction_sample.csv')



    #
    # report_val, clf_report_val = evaluation_report(y_val, y_pred_val,
    #                                                taxonomy=category_taxonomy,
    #                                                category_names=category_names)
    #
    #
    # def low_perf_categories_gen(clf_report: Dict,
    #                             min_support: int,
    #                             max_f1_score: float):
    #     for category, metrics in clf_report.items():
    #         f1_score = metrics['f1-score']
    #         support = metrics['support']
    #
    #         if support >= min_support:
    #             if f1_score < max_f1_score:
    #                 yield category
    #
    # # train_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_TRAIN_PATH))
    # # train_df['deepest_categories'] = get_deepest_categories(category_taxonomy, train_df.categories_tags)
    # # X_train, y_train = generate_data_partial(train_df)
    #
    #
    # gen = low_perf_categories_gen(clf_report_val, min_support=10, max_f1_score=0.5)
    # cat = next(gen)
    # val_metrics = clf_report_val[cat]
    # cat_id = category_to_id[cat]
    # # train_samples_idx = y_train[:, cat_id].nonzero()[0]
    # val_samples_idx = y_val[:, cat_id].nonzero()[0]
    # # train_samples = train_df.iloc[train_samples_idx, :]
    # val_samples = val_df.iloc[val_samples_idx, :]

if __name__ == "__main__":
    main()
