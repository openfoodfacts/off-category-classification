import argparse
import datetime
import functools
import json
import pathlib
import shutil
import tempfile
from typing import Dict, List

import numpy as np

import dacite
import pandas as pd
import tensorflow as tf
from robotoff.taxonomy import Taxonomy
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import callbacks
from tensorflow.python.ops import summary_ops_v2

import settings
from category_classification.data_utils import (
    load_dataframe,
    create_tf_dataset,
    get_labels,
)
from category_classification.models import (
    KerasPreprocessing,
    build_model,
    construct_preprocessing,
    to_serving_model,
)

from category_classification.config import Config

from utils.io import (
    copy_category_taxonomy,
    save_category_vocabulary,
    save_config,
    save_json,
)
from utils.metrics import evaluation_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument(
        "--repeat", type=int, default=1, help="number of replicates to run"
    )
    return parser.parse_args()


def create_model(config: Config, preprocess: KerasPreprocessing) -> keras.Model:
    model = build_model(config.model_config, preprocess)
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

    print("Full configuration:\n{}".format(json.dumps(config_dict, indent=4)))
    return dacite.from_dict(Config, config_dict)


class TBCallback(callbacks.TensorBoard):
    """Get around a bug where you cannot use the TensorBoard callback with the StringLookup layers
    - https://github.com/tensorflow/tensorboard/issues/4530#issuecomment-783318292"""

    def _log_weights(self, epoch):
        with self._train_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                for layer in self.model.layers:
                    for weight in layer.weights:
                        if hasattr(weight, "name"):
                            weight_name = weight.name.replace(":", "_")
                            summary_ops_v2.histogram(weight_name, weight, step=epoch)
                            if self.write_images:
                                self._log_weight_as_image(weight, weight_name, epoch)
                self._train_writer.flush()


def train(
    model: keras.Model,
    save_dir: pathlib.Path,
    config: Config,
    category_vocab: List[str],
):
    print("Starting training...")
    temporary_log_dir = pathlib.Path(tempfile.mkdtemp())
    print("Temporary log directory: {}".format(temporary_log_dir))

    # for now, clear cache on each call to train(), to be safe
    cache_dir = pathlib.Path("tf_cache")
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir()
    print("Cache directory: {}".format(cache_dir))

    train = (
        create_tf_dataset("train", category_vocab, config.train_config.batch_size)
        .cache(str(cache_dir / "train"))
    )
    val = (
        create_tf_dataset("val", category_vocab, config.train_config.batch_size)
        .cache(str(cache_dir / "val"))
    )

    model.fit(train,
        epochs= config.train_config.epochs,
        validation_data=val,
        callbacks=[
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(
                filepath=str(save_dir / "weights.{epoch:02d}-{val_loss:.4f}"),
                monitor="val_loss",
                save_best_only=True,
                save_format='tf',
            ),
            TBCallback(log_dir=str(temporary_log_dir), histogram_freq=1),
            callbacks.EarlyStopping(monitor="val_loss", patience=4),
            callbacks.CSVLogger(str(save_dir / "training.csv")),
        ],
    )
    print("Training ended")

    log_dir = save_dir / "logs"
    print("Moving log directory from {} to {}".format(temporary_log_dir, log_dir))
    shutil.move(str(temporary_log_dir), str(log_dir))

    print("Saving the base and the serving model {}".format(save_dir))
    model.save(str(save_dir / "base/saved_model"))
    to_serving_model(model, category_vocab).save(str(save_dir / "serving/saved_model"))

    category_taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)

    print("Evaluating on validation dataset")
    y_pred_val = model.predict(val)
    report, clf_report = evaluation_report(
        get_labels(val), y_pred_val, taxonomy=category_taxonomy, category_names=category_vocab
    )

    save_json(report, save_dir / "metrics_val.json")
    save_json(clf_report, save_dir / "classification_report_val.json")

    print("Evaluating on test dataset")
    test = create_tf_dataset("test", config.train_config.batch_size, tf_transformer)
    y_pred_test = model.predict(test)
    report, clf_report = evaluation_report(
        get_labels(test), y_pred_test, taxonomy=category_taxonomy, category_names=category_vocab
    )

    save_json(report, save_dir / "metrics_test.json")
    save_json(clf_report, save_dir / "classification_report_test.json")


def main():
    args = parse_args()
    config: Config = get_config(args)
    model_config = config.model_config

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Pre-processing of training data...")
    keras_preprocess = construct_preprocessing(
        model_config.category_min_count,
        model_config.ingredient_min_count,
        model_config.product_name_max_tokens,
        model_config.product_name_max_length,
        load_dataframe("train"),
    )
    print("Pre-processing of training data complete.")

    replicates = args.repeat
    if replicates == 1:
        save_dirs = [output_dir]
    else:
        save_dirs = [output_dir / str(i) for i in range(replicates)]

    for i, save_dir in enumerate(save_dirs):
        model = create_model(config, keras_preprocess)

        save_dir.mkdir(exist_ok=True)
        config.train_config.start_datetime = str(datetime.datetime.utcnow())
        print(f"Starting training repeat {i}")

        save_config(config, save_dir)
        copy_category_taxonomy(settings.CATEGORY_TAXONOMY_PATH, save_dir)
        save_category_vocabulary(keras_preprocess.category_vocab, save_dir)

        train(
            model,
            save_dir,
            config,
            keras_preprocess.category_vocab,
        )

        config.train_config.end_datetime = str(datetime.datetime.utcnow())
        save_config(config, save_dir)
        config.train_config.start_datetime = None
        config.train_config.end_datetime = None


if __name__ == "__main__":
    main()
