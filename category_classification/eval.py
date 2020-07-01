import argparse
import operator
import pathlib

from robotoff.taxonomy import Taxonomy
from tensorflow import keras

import settings
from utils.io import (
    load_config,
    load_product_name_vocabulary,
    load_category_vocabulary,
    load_ingredient_vocabulary,
    save_json,
)
from utils.metrics import evaluation_report
from utils.preprocess import get_nlp
from category_classification.data_utils import generate_data_from_df, create_dataframe

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    parser.add_argument("--type", default="val", choices=["test", "val"])
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--lang")
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


eval_type = args.type

df = create_dataframe(eval_type, args.lang if args.lang else config.lang)

X, y = generate_data_from_df(
    df,
    ingredient_to_id,
    category_to_id,
    product_name_vocabulary,
    nlp=nlp,
    product_name_max_length=config.model_config.product_name_max_length,
    product_name_preprocessing_config=config.product_name_preprocessing_config,
    nutriments_input=config.model_config.nutriment_input,
)


y_pred = model.predict(X)

category_taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)
report, clf_report = evaluation_report(
    y, y_pred, taxonomy=category_taxonomy, category_names=category_names
)

output_prefix = args.output_prefix
if output_prefix:
    output_prefix += "_"

save_json(report, model_dir / "{}metrics_{}.json".format(output_prefix, eval_type))
save_json(
    clf_report,
    model_dir / "{}classification_report_{}.json".format(output_prefix, eval_type),
)
