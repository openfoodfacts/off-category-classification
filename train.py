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
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.data import Dataset
import pandas as pd

from tensorflow.keras import callbacks
import pandas as pd

from category_classification.data_utils import create_dataframes, generate_data_from_df
from category_classification.models import build_model, to_serving_model, Config
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
)


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
                filepath=str(save_dir / "weights.{epoch:02d}-{val_loss:.4f}"),
                monitor="val_loss",
                save_best_only=True,
                save_format='tf',
            ),
            TBCallback(log_dir=str(temporary_log_dir), histogram_freq=1, profile_batch = '500, 510'),
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

    to_serving_model(model, category_names).save(str(save_dir / "serving/saved_model"))

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

    dfs = create_dataframes()
    train_df, test_df, val_df = dfs["train"], dfs["test"], dfs["val"]

    categories_count = count_categories(train_df)

    selected_categories = set(
        (
            cat
            for cat, count in categories_count.items()
            if count >= config.category_min_count
        )
    )

    print("{} categories selected".format(len(selected_categories)))

    sorted_categories = [
        x for x in sorted(category_taxonomy.keys()) if x in selected_categories
    ]

    category_to_id = {name: idx for idx, name in enumerate(sorted_categories)}

    model_config.output_dim = len(category_to_id)

    generate_data_partial = functools.partial(
        generate_data_from_df,
        category_to_id=category_to_id,
        categories=sorted_categories,
        nutriment_input=config.model_config.nutriment_input,
    )

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
        save_category_vocabulary(category_to_id, save_dir)

        print("Processing training data")
        X_train, y_train = generate_data_partial(train_df)
        print("Processing validation data")
        X_val, y_val = generate_data_partial(val_df)
        print("Processing test data")
        X_test, y_test = generate_data_partial(test_df)

        train(
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            model,
            save_dir,
            config,
            category_taxonomy,
            sorted_categories,
        )

        config.train_config.end_datetime = str(datetime.datetime.utcnow())
        save_config(config, save_dir)
        config.train_config.start_datetime = None
        config.train_config.end_datetime = None


if __name__ == "__main__":
    main()
