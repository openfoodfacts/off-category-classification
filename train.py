import dataclasses
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import typer
from codecarbon import EmissionsTracker
from tensorflow.keras import callbacks, layers
from tensorflow.keras.utils import plot_model
from wandb.keras import WandbMetricsLogger

import datasets.off_categories
import wandb
from lib.constant import IMAGE_EMBEDDING_DIM, NUTRIMENT_NAMES
from lib.dataset import (
    as_dataframe,
    flat_batch,
    get_vocabulary,
    load_dataset,
    select_feature,
    select_features,
)
from lib.directories import get_best_checkpoint, init_model_dir
from lib.io import load_model, save_model
from lib.metrics import PrecisionWithAverage, RecallWithAverage
from lib.model import (
    build_attention_over_sequence_layer,
    top_labeled_predictions,
    top_predictions_table,
)
from lib.taxonomy import get_taxonomy

PREPROC_BATCH_SIZE = 25_000  # some large value, only affects execution time
PROJECT_DIR = Path(__file__).parent.absolute().resolve()


@dataclasses.dataclass
class Config:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    mixed_precision: bool
    random_seed: int
    category_min_count: int
    add_product_name_input: bool
    add_ingredient_ocr_input: bool
    add_ingredient_input: bool
    add_nutriment_input: bool
    add_image_embedding_input: bool
    nutriment_num_bins: int
    ingredient_min_freq: int
    ingredient_ocr_min_freq: int
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
    image_embedding_key_dim: int
    image_embedding_num_heads: int
    dense_layer_dropout: float
    dense_layer_output_size: int
    label_smoothing: float
    cosine_scheduler: bool


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
    print(f"{feature_name}: vocabulary_size={vocabulary_size}")
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
        print(f"{feature_name}: vocabulary_size={vocabulary_size}")
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


def add_ingredient_ocr_feature(dataset, inputs: dict, graph: dict, config: Config):
    feature_name = "ingredients_ocr_tags"
    ingredients_input = tf.keras.Input(
        shape=(None,), dtype=tf.string, name=feature_name
    )

    ingredients_vocab = get_vocabulary(
        flat_batch(
            select_feature(dataset, feature_name), batch_size=PREPROC_BATCH_SIZE
        ),
        min_freq=config.ingredient_ocr_min_freq,
        add_pad_token=False,
        add_oov_token=True,
    )
    print(f"ingredients ocr vocabulary size: {len(ingredients_vocab)}")
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
        print(f"nutriment_name={nutriment_name}")
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


def add_image_embedding_feature(inputs: dict, graph: dict, config: Config):
    feature_name = "image_embeddings"
    [image_embedding_input, image_embedding_mask_input], [
        _,
        attention_scores,
        average_output,
    ] = build_attention_over_sequence_layer(
        # embedding dim of CLIP model
        IMAGE_EMBEDDING_DIM,
        feature_name,
        num_heads=config.image_embedding_num_heads,
        key_dim=config.image_embedding_key_dim,
    )
    inputs[image_embedding_input.name] = image_embedding_input
    inputs[image_embedding_mask_input.name] = image_embedding_mask_input
    graph[feature_name] = average_output
    graph[f"{feature_name}_attention_scores"] = attention_scores


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
    ingredient_ocr_min_freq: int = typer.Option(
        3,
        help="Minimum number of occurences in train dataset for an ingredient from OCR text to be "
        "considered as input",
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
    image_embedding_num_heads: int = typer.Option(
        1, help="Number of heads of the image embedding MultiHeadAttention layer"
    ),
    image_embedding_key_dim: int = typer.Option(
        64,
        help="Dimension of the key vector in image embedding MultiHeadAttention layer",
    ),
    add_product_name_input: bool = typer.Option(
        True, help="If True, add product name as input feature."
    ),
    add_ingredient_input: bool = typer.Option(
        True, help="If True, add ingredients as input feature."
    ),
    add_ingredient_ocr_input: bool = typer.Option(
        True, help="If True, add OCR ingredients as input feature."
    ),
    add_nutriment_input: bool = typer.Option(
        True, help="If True, add nutriments as input feature."
    ),
    add_image_embedding_input: bool = typer.Option(
        False,
        help="If True, add image embeddings (generated by clip-base-patch32) as input feature.",
    ),
    epochs: int = typer.Option(50, help="Number of epochs to train"),
    batch_size: int = typer.Option(128, help="Batch size during training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate (Adam optimizer)"),
    dense_layer_dropout: float = typer.Option(
        0.2, help="Dropout value of final dense layers"
    ),
    dense_layer_output_size: int = typer.Option(
        64, help="Output size of final dense layer"
    ),
    label_smoothing: float = typer.Option(0.0, help="Value of label smoothing"),
    cosine_scheduler: bool = typer.Option(
        False,
        help="If True, uses a cosine scheduler, use a constant learning rate otherwise.",
    ),
    name: Optional[str] = typer.Option(None, help="Name of the experiment."),
    notes: Optional[str] = typer.Option(
        None, help="Description of the experiment (for W&B tracking)."
    ),
    tags: Optional[List[str]] = typer.Option(
        None, help="Tags of the experiment (for W&B tracking)."
    ),
):
    MODEL_BASE_DIR = PROJECT_DIR / "experiments" / "trainings"
    MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("Setting up carbon emission tracking")
    tracker = EmissionsTracker(
        log_level="WARNING",
        save_to_api=True,
        experiment_id="6d2c8401-afba-42de-9600-6e95bea5fd80",
        output_file=str(PROJECT_DIR / "experiments/emissions.csv"),
    )
    tracker.start()

    # splits are handled by `tfds.load`, see doc for more elaborate ways to sample
    TRAIN_SPLIT = "train[:80%]"
    VAL_SPLIT = "train[80%:90%]"
    TEST_SPLIT = "train[90%:]"

    MODEL_DIR = init_model_dir(MODEL_BASE_DIR / name if name else "model")

    config = Config(
        name=name or MODEL_DIR.name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mixed_precision=mixed_precision,
        category_min_count=category_min_count,
        random_seed=random_seed,
        add_product_name_input=add_product_name_input,
        add_ingredient_input=add_ingredient_input,
        add_nutriment_input=add_nutriment_input,
        add_ingredient_ocr_input=add_ingredient_ocr_input,
        add_image_embedding_input=add_image_embedding_input,
        nutriment_num_bins=nutriment_num_bins,
        ingredient_min_freq=ingredient_min_freq,
        ingredient_ocr_min_freq=ingredient_ocr_min_freq,
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
        image_embedding_key_dim=image_embedding_key_dim,
        image_embedding_num_heads=image_embedding_num_heads,
        dense_layer_dropout=dense_layer_dropout,
        dense_layer_output_size=dense_layer_output_size,
        label_smoothing=label_smoothing,
        cosine_scheduler=cosine_scheduler,
    )

    wandb_run = wandb.init(
        project="product-categorization",
        name=config.name,
        config=dataclasses.asdict(config),
        notes=notes,
        tags=tags,
    )

    with (MODEL_DIR / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    print("Fetching taxonomies")
    category_taxonomy = get_taxonomy("category", offline=True)
    ingredient_taxonomy = get_taxonomy("ingredient", offline=True)
    print(
        f"ingredient_taxonomy_count={len(ingredient_taxonomy)}, category_taxonomy_count={len(category_taxonomy)}"
    )

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
    if add_ingredient_input:
        print("Adding ingredient input")
        add_ingredient_feature(ds, inputs, input_graphs, config)
    if add_ingredient_ocr_input:
        print("Adding ingredient OCR input")
        add_ingredient_ocr_feature(ds, inputs, input_graphs, config)
    if add_image_embedding_feature:
        print("Adding image embeddings input")
        add_image_embedding_feature(inputs, input_graphs, config)

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

    print(f"categories_vocab_count={len(categories_vocab)}")

    # ensure final order is independent of cell execution/insertion order
    features = sorted(inputs.keys())
    if len(features) > 1:
        x = layers.Concatenate()(
            [input_graphs[k] for k in features if k != "image_embeddings_mask"]
        )
    else:
        x = input_graphs[features[0]]
    x = layers.Dropout(config.dense_layer_dropout)(x)
    x = layers.Dense(config.dense_layer_output_size)(x)
    x = layers.Dropout(config.dense_layer_dropout)(x)
    x = layers.Activation("relu")(x)
    output = layers.Dense(len(categories_vocab), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[inputs[k] for k in features], outputs=[output])

    ds_train = (
        load_dataset(
            "off_categories", split=TRAIN_SPLIT, features=features, as_supervised=True
        )
        .apply(categories_encode)
        .padded_batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Compiling model")
    if config.cosine_scheduler:
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            config.learning_rate, decay_steps=len(ds_train) * config.epochs
        )
    else:
        learning_rate = config.learning_rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

    ds_val = (
        load_dataset(
            "off_categories", split=VAL_SPLIT, features=features, as_supervised=True
        )
        .apply(categories_encode)
        .padded_batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    WEIGHTS_DIR = MODEL_DIR / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model.fit(
        ds_train,
        epochs=config.epochs,
        validation_data=ds_val,
        callbacks=[
            callbacks.TerminateOnNaN(),
            callbacks.ModelCheckpoint(
                filepath=str(
                    WEIGHTS_DIR / "weights.{epoch:02d}-{val_f1_score_micro:.4f}"
                ),
                monitor="val_f1_score_micro",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
            ),
            callbacks.CSVLogger(str(MODEL_DIR / "training.log")),
            callbacks.History(),
            callbacks.TensorBoard(
                log_dir="{}/logs".format(MODEL_DIR),
                write_graph=False,
            ),
            WandbMetricsLogger(),
        ],
    )

    SAVED_MODEL_DIR = MODEL_DIR / "saved_model"

    @tf.function
    def serving_func(model, model_spec, categories_vocab):
        model_args, model_kwargs = model_spec
        preds = model(*model_args, **model_kwargs)
        return top_labeled_predictions(preds, categories_vocab, k=len(categories_vocab))

    checkpoint_path = get_best_checkpoint(WEIGHTS_DIR)
    if checkpoint_path is not None:
        print(f"Loading best checkpoint file {checkpoint_path}")
        model.load_weights(str(checkpoint_path))
    else:
        print("No checkpoint file found!")

    save_model(
        SAVED_MODEL_DIR,
        model,
        categories_vocab,
        serving_func,
        serving_func_kwargs={"categories_vocab": categories_vocab},
        include_optimizer=False,
    )

    m, labels = load_model(SAVED_MODEL_DIR)

    ds_test = load_dataset("off_categories", split=TEST_SPLIT)
    preds_test = m.predict(ds_test.padded_batch(config.batch_size))

    # This is the function exported as the default serving function in our saved model
    top_preds_test = top_labeled_predictions(preds_test, labels, k=10)
    # Same data, but pretty
    pred_table_test = top_predictions_table(top_preds_test)

    # Add some interpretable features to the final table
    # Table must be row-aligned with predictions above (= taken from same data sample)
    extra_cols_test = as_dataframe(
        select_features(
            ds_test,
            ["code", "product_name"] + list(NUTRIMENT_NAMES),
        )
    )

    output_df = pd.concat([extra_cols_test, pred_table_test], axis=1)
    output_df.to_csv(MODEL_DIR / "test_predictions.tsv", sep="\t", index=False)
    wandb_run.log({"predictions_test": wandb.Table(dataframe=output_df)})

    # codecarbon - stop tracking
    tracker.stop()


if __name__ == "__main__":
    typer.run(main)
