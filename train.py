import argparse
import datetime
import functools
import json
import pathlib
import shutil
import tempfile
from typing import List

import dacite
from robotoff.taxonomy import Taxonomy
from tensorflow import keras
from tensorflow.keras import callbacks

from category_classification.data_utils import create_dataframe, generate_data_from_df
from category_classification.models import build_model, Config
import settings
from utils import update_dict_dot
from utils.io import (
    copy_category_taxonomy,
    save_category_vocabulary,
    save_config,
    save_ingredient_vocabulary,
    save_json,
    save_product_name_vocabulary,
)
from utils.metrics import evaluation_report
from utils.preprocess import (
    count_categories,
    count_ingredients,
    extract_vocabulary,
    get_nlp,
    preprocess_product_name,
    tokenize_batch,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument(
        "--extra-params", help="extra parameters updating the base configuration"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="number of replicates to run"
    )
    parser.add_argument("--lang", type=str, default="xx")
    return parser.parse_args()


def create_model(config: Config) -> keras.Model:
    model = build_model(config.model_config)
    loss_fn = keras.losses.BinaryCrossentropy(
        label_smoothing=config.train_config.label_smoothing
    )
    optimizer = keras.optimizers.Adam(learning_rate=config.train_config.lr)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["binary_accuracy", "Precision", "Recall"],
    )
    return model


def get_config(args) -> Config:
    with args.config.open("r") as f:
        config_dict = json.load(f)

    config_dict["lang"] = args.lang

    if args.extra_params:
        print("Extra parameters: {}".format(args.extra_params))
        update_dict_dot(config_dict, args.extra_params)

    print("Full configuration:\n{}".format(json.dumps(config_dict, indent=4)))
    return dacite.from_dict(Config, config_dict)


def train(
    train_data,
    val_data,
    test_data,
    model: keras.Model,
    save_dir: pathlib.Path,
    config: Config,
    category_taxonomy: Taxonomy,
    category_names: List[str],
):
    print("Starting training...")
    temporary_log_dir = pathlib.Path(tempfile.mkdtemp())
    print("Temporary log directory: {}".format(temporary_log_dir))

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    model.fit(
        X_train,
        y_train,
        batch_size=config.train_config.batch_size,
        epochs=config.train_config.epochs,
        validation_data=(X_val, y_val),
        callbacks=[
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(
                filepath=str(save_dir / "weights.{epoch:02d}-{val_loss:.4f}.hdf5"),
                monitor="val_loss",
                save_best_only=True,
            ),
            callbacks.TensorBoard(log_dir=str(temporary_log_dir), histogram_freq=2, profile_batch = '500, 510'),
            callbacks.EarlyStopping(monitor="val_loss", patience=4),
            callbacks.CSVLogger(str(save_dir / "training.csv")),
        ],
    )
    print("Training ended")

    log_dir = save_dir / "logs"
    print("Moving log directory from {} to {}".format(temporary_log_dir, log_dir))
    shutil.move(str(temporary_log_dir), str(log_dir))

    model.save(str(save_dir / "last_checkpoint.hdf5"))

    last_checkpoint_path = sorted(save_dir.glob("weights.*.hdf5"))[-1]

    print("Restoring last checkpoint {}".format(last_checkpoint_path))
    model = keras.models.load_model(str(last_checkpoint_path))

    print("Evaluating on validation dataset")
    y_pred_val = model.predict(X_val)
    report, clf_report = evaluation_report(
        y_val, y_pred_val, taxonomy=category_taxonomy, category_names=category_names
    )

    save_json(report, save_dir / "metrics_val.json")
    save_json(clf_report, save_dir / "classification_report_val.json")

    print("Evaluating on test dataset")
    y_pred_test = model.predict(X_test)
    report, clf_report = evaluation_report(
        y_test, y_pred_test, taxonomy=category_taxonomy, category_names=category_names
    )

    save_json(report, save_dir / "metrics_test.json")
    save_json(clf_report, save_dir / "classification_report_test.json")


def main():
    args = parse_args()
    config: Config = get_config(args)
    model_config = config.model_config

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    category_taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)
    ingredient_taxonomy = Taxonomy.from_json(settings.INGREDIENTS_TAXONOMY_PATH)

    train_df = create_dataframe("train", args.lang)
    test_df = create_dataframe("test", args.lang)
    val_df = create_dataframe("val", args.lang)

    categories_count = count_categories(train_df)
    ingredients_count = count_ingredients(train_df)

    selected_categories = set(
        (
            cat
            for cat, count in categories_count.items()
            if count >= config.category_min_count
        )
    )
    selected_ingredients = set(
        (
            ingredient
            for ingredient, count in ingredients_count.items()
            if count >= config.ingredient_min_count
        )
    )
    print("{} categories selected".format(len(selected_categories)))
    print("{} ingredients selected".format(len(selected_ingredients)))

    category_names = [
        x for x in sorted(category_taxonomy.keys()) if x in selected_categories
    ]

    ingredient_names = [
        x for x in sorted(ingredient_taxonomy.keys()) if x in selected_ingredients
    ]

    category_to_id = {name: idx for idx, name in enumerate(category_names)}
    ingredient_to_id = {name: idx for idx, name in enumerate(ingredient_names)}

    nlp = get_nlp(lang=config.lang)

    preprocess_product_name_func = functools.partial(
        preprocess_product_name,
        lower=config.product_name_preprocessing_config.lower,
        strip_accent=config.product_name_preprocessing_config.strip_accent,
        remove_punct=config.product_name_preprocessing_config.remove_punct,
        remove_digit=config.product_name_preprocessing_config.remove_digit,
    )
    preprocessed_product_names_iter = (
        preprocess_product_name_func(product_name)
        for product_name in train_df.product_name
    )
    train_tokens_iter = tokenize_batch(preprocessed_product_names_iter, nlp)
    product_name_to_int = extract_vocabulary(
        train_tokens_iter, config.product_name_min_count
    )

    model_config.ingredient_voc_size = len(ingredient_to_id)
    model_config.output_dim = len(category_to_id)
    model_config.product_name_voc_size = len(product_name_to_int)

    print("Selected vocabulary: {}".format(len(product_name_to_int)))

    generate_data_partial = functools.partial(
        generate_data_from_df,
        ingredient_to_id=ingredient_to_id,
        category_to_id=category_to_id,
        product_name_max_length=model_config.product_name_max_length,
        product_name_token_to_int=product_name_to_int,
        nlp=nlp,
        product_name_preprocessing_config=config.product_name_preprocessing_config,
        nutriment_input=config.model_config.nutriment_input,
    )

    replicates = args.repeat
    if replicates == 1:
        save_dirs = [output_dir]
    else:
        save_dirs = [output_dir / str(i) for i in range(replicates)]

    for i, save_dir in enumerate(save_dirs):
        model = create_model(config)
        save_dir.mkdir(exist_ok=True)
        config.train_config.start_datetime = str(datetime.datetime.utcnow())
        print("Starting training repeat {}".format(i))
        save_product_name_vocabulary(product_name_to_int, save_dir)
        save_config(config, save_dir)
        copy_category_taxonomy(settings.CATEGORY_TAXONOMY_PATH, save_dir)
        save_category_vocabulary(category_to_id, save_dir)
        save_ingredient_vocabulary(ingredient_to_id, save_dir)

        X_train, y_train = generate_data_partial(train_df)
        X_val, y_val = generate_data_partial(val_df)
        X_test, y_test = generate_data_partial(test_df)

        train(
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            model,
            save_dir,
            config,
            category_taxonomy,
            category_names,
        )

        config.train_config.end_datetime = str(datetime.datetime.utcnow())
        save_config(config, save_dir)
        config.train_config.start_datetime = None
        config.train_config.end_datetime = None


if __name__ == "__main__":
    main()
