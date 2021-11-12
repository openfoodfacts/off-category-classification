import argparse
import datetime
import functools
import json
import pathlib
import shutil
import tempfile
from typing import List, Dict

import os, psutil # Remove this?

import dacite
from robotoff.taxonomy import Taxonomy
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.data import Dataset
import pandas as pd

from tensorflow.keras import callbacks
import pandas as pd

from category_classification.data_utils import convert_to_tf_dataset, load_dataframe
from category_classification.models import build_model, to_serving_model, Config
import settings
from utils import update_dict_dot
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


def create_model(config: Config, train_data: pd.DataFrame) -> keras.Model:
    model = build_model(config.model_config, train_data)
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
    ''' Get around a bug in using the StringLookup layers - https://github.com/tensorflow/tensorboard/issues/4530#issuecomment-783318292
    '''
    def _log_weights(self, epoch):
        with self._train_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                for layer in self.model.layers:
                    for weight in layer.weights:
                        if hasattr(weight, "name"):
                            weight_name = weight.name.replace(':', '_')
                            summary_ops_v2.histogram(weight_name, weight, step=epoch)
                            if self.write_images:
                                self._log_weight_as_image(weight, weight_name, epoch)
                self._train_writer.flush()

def train(
    train_df: pd.DataFrame,
    model: keras.Model,
    save_dir: pathlib.Path,
    config: Config,
    category_vocab: List[str],
):
    print("Starting training...")
    temporary_log_dir = pathlib.Path(tempfile.mkdtemp())
    print("Temporary log directory: {}".format(temporary_log_dir))

    process = psutil.Process(os.getpid())

    train = convert_to_tf_dataset(train_df, category_vocab)
    val = convert_to_tf_dataset(load_dataframe("val"), category_vocab)

    print(f"Memory usage on after dataset formatting: {process.memory_info().rss}")

    model.fit(train,
        batch_size=config.train_config.batch_size,
        epochs=config.train_config.epochs,
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
    y_pred_val = model.predict(X_val) # fix this later.
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

    process = psutil.Process(os.getpid())
    print(f"Memory usage on start: {process.memory_info().rss}")

    train_df = load_dataframe("train")

    print(f"Memory usage on after training dataframe loaded: {process.memory_info().rss}")

    category_lookup = tf.keras.layers.StringLookup(max_tokens=3969, output_mode="multi_hot", num_oov_indices=0)
    category_lookup.adapt(tf.ragged.constant(train_df.categories_tags))

    category_vocab = category_lookup.get_vocabulary()
    model_config.output_dim = len(category_vocab)

    replicates = args.repeat
    if replicates == 1:
        save_dirs = [output_dir]
    else:
        save_dirs = [output_dir / str(i) for i in range(replicates)]

    for i, save_dir in enumerate(save_dirs):
        model = create_model(config, train_df)

        save_dir.mkdir(exist_ok=True)
        config.train_config.start_datetime = str(datetime.datetime.utcnow())
        print("Starting training repeat {}".format(i))

        save_config(config, save_dir)
        copy_category_taxonomy(settings.CATEGORY_TAXONOMY_PATH, save_dir)
        save_category_vocabulary(category_vocab, save_dir)

        print(f"Memory usage on training start: {process.memory_info().rss}")
        train(
            train_df,
            model,
            save_dir,
            config,
            category_vocab,
        )

        config.train_config.end_datetime = str(datetime.datetime.utcnow())
        save_config(config, save_dir)
        config.train_config.start_datetime = None
        config.train_config.end_datetime = None


if __name__ == "__main__":
    main()
