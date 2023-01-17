import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path("..").absolute().resolve()
sys.path.append(
    str(PROJECT_DIR)
)  # append a relative path to the top package to the search path


# codecarbon - start tracking
from codecarbon import EmissionsTracker
import dataclasses
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras import callbacks, layers
from tensorflow.keras.utils import plot_model
import typer

import datasets.off_categories
from lib.constant import NUTRIMENT_NAMES
from lib.dataset import (
    as_dataframe,
    flat_batch,
    get_vocabulary,
    load_dataset,
    select_feature,
    select_features,
)
from lib.directories import init_model_dir
from lib.io import load_model, save_model
from lib.metrics import PrecisionWithAverage, RecallWithAverage
from lib.model import top_labeled_predictions, top_predictions_table
from lib.taxonomy import get_taxonomy

PREPROC_BATCH_SIZE = 25_000  # some large value, only affects execution time


@dataclasses.dataclass
class Config:
    epochs: int
    batch_size: int
    learning_rate: float
    mixed_precision: bool
    random_seed: int
    category_min_count: int
    add_product_name_input: bool
    add_ingredient_input: bool
    add_nutriment_input: bool
    nutriment_num_bins: int
    ingredient_min_freq: int
    ingredient_embedding_size: int
    ingredient_output_size: int
    ingredient_recurrent: bool
    ingredient_lstm_dropout: float
    product_name_max_tokens: int
    product_name_output_sequence_length: int
    product_name_split_method: str
    product_name_embedding_size: int
    product_name_output_size: int
    product_name_lstm_dropout: float
    nutriment_num_bins: int
    dense_layer_dropout: float
    dense_layer_output_size: int
    label_smoothing: float


def get_metrics(threshold: float, num_classes: int):
    return [
        PrecisionWithAverage(
            average="micro",
            threshold=threshold,
            num_classes=num_classes,
            name="precision_micro",
        ),
        PrecisionWithAverage(
            average="macro",
            threshold=threshold,
            num_classes=num_classes,
            name="precision_macro",
        ),
        RecallWithAverage(
            average="micro",
            threshold=threshold,
            num_classes=num_classes,
            name="recall_micro",
        ),
        RecallWithAverage(
            average="macro",
            threshold=threshold,
            num_classes=num_classes,
            name="recall_macro",
        ),
        tfa.metrics.F1Score(
            average="micro",
            threshold=threshold,
            num_classes=num_classes,
            name="f1_score_micro",
        ),
        tfa.metrics.F1Score(
            average="macro",
            threshold=threshold,
            num_classes=num_classes,
            name="f1_score_macro",
        ),
    ]


def add_product_name_feature(dataset, inputs: dict, graph: dict, config: Config):
    feature_name = "product_name"
    product_name_input = tf.keras.Input(shape=(1,), dtype=tf.string, name=feature_name)
    product_name_vectorizer = layers.TextVectorization(
        split=config.product_name_split_method,
        max_tokens=config.product_name_max_tokens,
        output_sequence_length=config.product_name_output_sequence_length,
    )
    product_name_vectorizer.adapt(
        select_feature(dataset, feature_name).batch(PREPROC_BATCH_SIZE)
    )
    x = product_name_vectorizer(product_name_input)
    x = layers.Embedding(
        input_dim=product_name_vectorizer.vocabulary_size(),
        output_dim=config.product_name_embedding_size,
        mask_zero=False,
    )(x)
    product_name_graph = layers.Bidirectional(
        layers.LSTM(
            units=config.product_name_output_size,
            recurrent_dropout=0.0,
            dropout=config.product_name_lstm_dropout,
        )
    )(x)
    vocabulary_size = len(product_name_vectorizer.get_vocabulary())
    print(f"{feature_name}: {vocabulary_size=}")
    inputs[feature_name] = product_name_input
    graph[feature_name] = product_name_graph


def add_ingredient_feature(dataset, inputs: dict, graph: dict, config: Config):
    feature_name = "ingredients_tags"
    ingredients_input = tf.keras.Input(
        shape=(None,), dtype=tf.string, name=feature_name
    )

    if config.ingredient_recurrent:
        ingredients_vocab = get_vocabulary(
            flat_batch(
                select_feature(dataset, feature_name), batch_size=PREPROC_BATCH_SIZE
            ),
            min_freq=config.ingredient_min_freq,
            add_pad_token=True,
            add_oov_token=True,
        )
        print(f"ingredients vocabulary size: {len(ingredients_vocab)}")
        ingredients_lookup_layer = layers.StringLookup(
            vocabulary=ingredients_vocab,
            num_oov_indices=1,
            output_mode="int",
            mask_token="",
        )
        x = ingredients_lookup_layer(ingredients_input)
        x = layers.Embedding(
            input_dim=ingredients_lookup_layer.vocabulary_size(),
            output_dim=config.ingredient_embedding_size,
            mask_zero=True,
        )(x)
        ingredients_graph = layers.Bidirectional(
            layers.LSTM(
                units=config.ingredient_output_size,
                recurrent_dropout=0.0,
                dropout=config.ingredient_lstm_dropout,
            )
        )(x)
        vocabulary_size = len(ingredients_vocab)
        print(f"{feature_name}: {vocabulary_size=}")
    else:
        ingredients_vocab = get_vocabulary(
            flat_batch(
                select_feature(dataset, feature_name), batch_size=PREPROC_BATCH_SIZE
            ),
            min_freq=config.ingredient_min_freq,
            add_pad_token=False,
            add_oov_token=True,
        )
        print(f"ingredients vocabulary size: {len(ingredients_vocab)}")
        ingredients_lookup_layer = layers.StringLookup(
            vocabulary=ingredients_vocab,
            num_oov_indices=1,
            output_mode="multi_hot",
            mask_token="",
        )
        ingredients_graph = ingredients_lookup_layer(ingredients_input)

    inputs[feature_name] = ingredients_input
    graph[feature_name] = ingredients_graph


def add_nutriment_features(dataset, inputs: dict, graph: dict, config: Config):
    for feature_name in NUTRIMENT_NAMES:
        inputs[feature_name] = tf.keras.Input(
            shape=(1,), dtype=tf.float32, name=feature_name
        )

    for nutriment_name in NUTRIMENT_NAMES:
        print(f"{nutriment_name=}")
        discretization_layer = layers.Discretization(
            output_mode="one_hot", num_bins=config.nutriment_num_bins
        )
        discretization_layer.adapt(
            select_feature(dataset, nutriment_name)
            .filter(lambda x: x >= 0)
            .batch(PREPROC_BATCH_SIZE)
        )
        input_ = inputs[nutriment_name]
        graph[nutriment_name] = discretization_layer(input_)


def set_random_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(
    mixed_precision: bool = typer.Option(
        False, help="If True, used mixed precision training"
    ),
    random_seed: int = typer.Option(42, help="Random seed to use"),
    category_min_count: int = typer.Option(
        10, help="Minimum number of ocurrences for category to be considered"
    ),
    nutriment_num_bins: int = typer.Option(
        100, help="Number of bins to use for discretization of nutriment values"
    ),
    ingredient_min_freq: int = typer.Option(
        3,
        help="Minimum number of occurences in train dataset for an ingredient to be considered as input",
    ),
    ingredient_embedding_size: int = typer.Option(
        64, help="Size of the output embedding for ingredient input"
    ),
    ingredient_output_size: int = typer.Option(
        64,
        help="Size of the embedding after ingredient LSTM (*2 as the LSTM is bidirectionnal). Ignored if ingredient-recurrent is False.",
    ),
    ingredient_recurrent: bool = typer.Option(
        False,
        help="If True, take ingredient order into account, otherwise use a multi-hot encoding",
    ),
    ingredient_lstm_dropout: float = typer.Option(
        0.2,
        help="Dropout value for ingredient LSTM. Ignored if ingredient-recurrent is False.",
    ),
    product_name_max_tokens: int = typer.Option(
        93_000, help="Maximum number of tokens to keep in vocabulary for product name"
    ),
    product_name_output_sequence_length: int = typer.Option(
        30, help="Maximum sequence length (in tokens) for product name output"
    ),
    product_name_split_method: str = typer.Option(
        "whitespace", help="Token split method for TextVectorization layer"
    ),
    product_name_embedding_size: int = typer.Option(
        64, help="Size of the output embedding for product name input"
    ),
    product_name_output_size: int = typer.Option(
        64,
        help="Size of the embedding after product name LSTM (*2 as the LSTM is bidirectionnal).",
    ),
    product_name_lstm_dropout: float = typer.Option(
        0.2, help="Dropout value for product name LSTM."
    ),
    add_product_name_input: bool = typer.Option(
        True, help="If True, add product name as input feature."
    ),
    add_ingredient_input: bool = typer.Option(
        True, help="If True, add ingredients as input feature."
    ),
    add_nutriment_input: bool = typer.Option(
        True, help="If True, add nutriments as input feature."
    ),
    epochs: int = typer.Option(50, help="Number of epochs to train"),
    batch_size: int = typer.Option(50, help="Batch size during training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate (Adam optimizer)"),
    dense_layer_dropout: float = typer.Option(
        0.2, help="Dropout value of final dense layers"
    ),
    dense_layer_output_size: int = typer.Option(
        64, help="Output size of final dense layer"
    ),
    label_smoothing: float = typer.Option(0.0, help="Value of label smoothing"),
):
    MODEL_BASE_DIR = PROJECT_DIR / "experiments" / "trainings"
    MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("Setting up carbon emission tracking")
    tracker = EmissionsTracker(
        log_level="WARNING",
        save_to_api=True,
        experiment_id="6d2c8401-afba-42de-9600-6e95bea5fd80",
        output_file=str(MODEL_BASE_DIR / "emissions.csv")
    )
    tracker.start()

    # splits are handled by `tfds.load`, see doc for more elaborate ways to sample
    TRAIN_SPLIT = "train[:80%]"
    VAL_SPLIT = "train[80%:90%]"
    TEST_SPLIT = "train[90%:]"

    config = Config(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mixed_precision=mixed_precision,
        category_min_count=category_min_count,
        random_seed=random_seed,
        add_product_name_input=add_product_name_input,
        add_ingredient_input=add_ingredient_input,
        add_nutriment_input=add_nutriment_input,
        nutriment_num_bins=nutriment_num_bins,
        ingredient_min_freq=ingredient_min_freq,
        ingredient_embedding_size=ingredient_embedding_size,
        ingredient_output_size=ingredient_output_size,
        ingredient_recurrent=ingredient_recurrent,
        ingredient_lstm_dropout=ingredient_lstm_dropout,
        product_name_max_tokens=product_name_max_tokens,
        product_name_output_sequence_length=product_name_output_sequence_length,
        product_name_split_method=product_name_split_method,
        product_name_embedding_size=product_name_embedding_size,
        product_name_output_size=product_name_output_size,
        product_name_lstm_dropout=product_name_lstm_dropout,
        dense_layer_dropout=dense_layer_dropout,
        dense_layer_output_size=dense_layer_output_size,
        label_smoothing=label_smoothing,
    )

    MODEL_DIR = init_model_dir(MODEL_BASE_DIR / "model")

    with (MODEL_DIR / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    print("Fetching taxonomies")
    category_taxonomy = get_taxonomy("category", offline=True)
    ingredient_taxonomy = get_taxonomy("ingredient", offline=True)
    print(f"{len(ingredient_taxonomy)=}, {len(category_taxonomy)=}")

    print("Downloading and preparing dataset...")
    builder = tfds.builder("off_categories")
    builder.download_and_prepare()

    set_random_seed(config.random_seed)

    if config.mixed_precision:
        print("Using mixed precision")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # we use dicts so rerunning individual model cells is idempotent
    inputs = {}
    input_graphs = {}

    ds = load_dataset("off_categories", split=TRAIN_SPLIT)

    if add_nutriment_input:
        print("Adding nutriment inputs")
        add_nutriment_features(ds, inputs, input_graphs, config)
    if add_product_name_input:
        print("Adding product name input")
        add_product_name_feature(ds, inputs, input_graphs, config)
    if add_nutriment_input:
        print("Adding ingredient input")
        add_ingredient_feature(ds, inputs, input_graphs, config)

    labels = "categories_tags"
    print("Generating category vocabulary")
    categories_vocab = get_vocabulary(
        flat_batch(select_feature(ds, labels), batch_size=PREPROC_BATCH_SIZE),
        min_freq=config.category_min_count,
    )

    # StringLookup(output_mode='multi_hot') mode requires num_oov_indices >= 1.
    # We don't want OOVs in the categories_tags output layer, since it wouldn't make
    # sense to predict OOV. So we'll drop the OOV in _transform below.
    # Be careful when using StringLookup methods, some of them will return values
    # based on a vocabulary with OOV (e.g. vocabulary_size()). Keep this in mind when
    # mapping predictions back to the original vocabulary.
    categories_multihot = layers.StringLookup(
        vocabulary=categories_vocab, output_mode="multi_hot", num_oov_indices=1
    )

    def categories_encode(ds: tf.data.Dataset):
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def _transform(x, y):
            y = categories_multihot(y)
            y = y[1:]  # drop OOV
            return (x, y)

        # applies to non-batched dataset
        return ds.map(
            _transform, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )

    print(f"{len(categories_vocab)=}")

    # ensure final order is independent of cell execution/insertion order
    features = sorted(inputs.keys())

    x = layers.Concatenate()([input_graphs[k] for k in features])
    x = layers.Dropout(config.dense_layer_dropout)(x)
    x = layers.Dense(config.dense_layer_output_size)(x)
    x = layers.Dropout(config.dense_layer_dropout)(x)
    x = layers.Activation("relu")(x)
    output = layers.Dense(len(categories_vocab), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=[inputs[k] for k in features], outputs=[output])
    print("Compiling model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing),
        metrics=get_metrics(threshold=0.5, num_classes=len(categories_vocab)),
    )
    model.summary()

    plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        to_file=str(MODEL_DIR / "model.png"),
    )

    ds_train = (
        load_dataset(
            "off_categories", split=TRAIN_SPLIT, features=features, as_supervised=True
        )
        .apply(categories_encode)
        .padded_batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_val = (
        load_dataset(
            "off_categories", split=VAL_SPLIT, features=features, as_supervised=True
        )
        .apply(categories_encode)
        .padded_batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.fit(
        ds_train,
        epochs=config.epochs,
        validation_data=ds_val,
        callbacks=[
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(
                filepath=str(MODEL_DIR / "weights.{epoch:02d}-{val_loss:.4f}"),
                monitor="f1_score_micro",
                save_best_only=True,
                save_format="tf",
            ),
            callbacks.CSVLogger(str(MODEL_DIR / "training.log")),
            callbacks.History(),
            callbacks.TensorBoard(
                log_dir="{}/logs".format(MODEL_DIR),
                write_graph=False,
            ),
        ],
    )

    SAVED_MODEL_DIR = MODEL_DIR / "saved_model"

    @tf.function
    def serving_func(*args, **kwargs):
        preds = model(*args, **kwargs)
        return top_labeled_predictions(preds, categories_vocab, k=len(categories_vocab))

    save_model(SAVED_MODEL_DIR, model, categories_vocab, serving_func)

    m, labels = load_model(SAVED_MODEL_DIR)

    ds_test = load_dataset("off_categories", split=TEST_SPLIT)
    preds_test = m.predict(ds_test.padded_batch(config.config))

    # This is the function exported as the default serving function in our saved model
    top_preds_test = top_labeled_predictions(preds_test, labels, k=10)
    # Same data, but pretty
    pred_table_test = top_predictions_table(top_preds_test)

    # Add some interpretable features to the final table
    # Table must be row-aligned with predictions above (= taken from same data sample)
    extra_cols_test = as_dataframe(select_features(ds_test, ["code", "product_name"]))

    output_df = pd.concat([extra_cols_test, pred_table_test], axis=1)
    output_df.to_csv(MODEL_DIR / "test_predictions.tsv", sep="\t", index=False)

    # codecarbon - stop tracking
    tracker.stop()


if __name__ == "__main__":
    typer.run(main)