import copy
import json
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import typer
from rich import print
from sklearn.metrics import classification_report

import datasets.off_categories
from lib.constant import NUTRIMENT_NAMES
from lib.dataset import load_dataset, select_feature
from lib.io import load_model

PREPROC_BATCH_SIZE = 25_000  # some large value, only affects execution time


def extract_barcodes(ds) -> List[str]:
    "Extract all barcodes in sequential order from a dataset."
    barcodes = []
    for batch in tfds.as_numpy(select_feature(ds, "code").batch(PREPROC_BATCH_SIZE)):
        barcodes += [x.decode("utf-8") for x in batch.tolist()]
    return barcodes


def generate_y_true(shape, ds, labels: List[str]):
    y_true = np.zeros(shape, dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    categories_tags_all = (
        [cat.decode("utf-8") for cat in x["categories_tags"].numpy().tolist()]
        for x in ds
    )

    for i, categories_tags in enumerate(categories_tags_all):
        for category_tag in categories_tags:
            if category_tag in label_to_idx:
                y_true[i, label_to_idx[category_tag]] = 1
    return y_true


def remove_ingredients_ocr_func(x):
    x = copy.copy(x)
    x["ingredients_ocr_tags"] = np.array([], dtype=np.string_)
    return x


def remove_ingredients_func(x):
    x = copy.copy(x)
    x["ingredients_tags"] = np.array([], dtype=np.string_)
    return x


def remove_nutriments_func(x):
    x = copy.copy(x)
    for nutriment_name in NUTRIMENT_NAMES:
        x[nutriment_name] = np.array(-1, dtype=np.float32)
    return x


def remove_product_name_func(x):
    x = copy.copy(x)
    x["product_name"] = np.array("", dtype=np.string_)
    return x


def main(
    model_dir: Path = typer.Option(..., help="name of the model"),
    remove_ingredient_ocr_tags: bool = typer.Option(
        False, help="Remove OCR ingredient tags input from dataset (ablation)"
    ),
    remove_ingredients_tags: bool = typer.Option(
        False, help="Remove ingredient tags input from dataset (ablation)"
    ),
    remove_nutriments: bool = typer.Option(
        False, help="Remove nutriment inputs from dataset (ablation)"
    ),
    remove_product_name: bool = typer.Option(
        False, help="Remove product name inputs from dataset (ablation)"
    ),
):
    MODEL_BASE_DIR = model_dir.parent
    TRAIN_SPLIT = "train[:80%]"
    VAL_SPLIT = "train[80%:90%]"
    TEST_SPLIT = "train[90%:]"

    print("checking training splits...")
    split_barcodes = {}
    SPLIT_DIR = MODEL_BASE_DIR / "splits"
    missing_splits = not SPLIT_DIR.exists()
    for split_name, split_command in (
        ("train", TRAIN_SPLIT),
        ("val", VAL_SPLIT),
        ("test", TEST_SPLIT),
    ):
        print(f"checking split {split_name}")
        barcodes = extract_barcodes(load_dataset("off_categories", split=split_command))
        split_barcodes[split_name] = set(barcodes)

        if len(split_barcodes[split_name]) != len(barcodes):
            raise ValueError("duplicate products in %s split", split_name)

        if missing_splits:
            SPLIT_DIR.mkdir(exist_ok=True)
            (SPLIT_DIR / f"{split_name}.txt").write_text("\n".join(barcodes))
        else:
            expected_barcodes = (
                (SPLIT_DIR / f"{split_name}.txt").read_text().splitlines()
            )
            if barcodes != expected_barcodes:
                raise ValueError(
                    "barcodes for split %s did not match reference", split_name
                )

    for split_1, split_2 in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_barcodes[split_1].intersection(split_barcodes[split_2]):
            raise ValueError("splits %s and %s intersect", split_1, split_2)

    print("Downloading and preparing dataset...")
    builder = tfds.builder("off_categories")
    builder.download_and_prepare()

    SAVED_MODEL_DIR = model_dir / "saved_model"
    m, labels = load_model(SAVED_MODEL_DIR)

    for split_name, split_command in (("val", VAL_SPLIT), ("test", TEST_SPLIT)):
        split_ds = load_dataset("off_categories", split=split_command)

        suffixes = []
        if remove_ingredient_ocr_tags:
            suffixes.append("ingredients_ocr_tags")
            split_ds = split_ds.map(remove_ingredients_ocr_func)
        if remove_ingredients_tags:
            suffixes.append("ingredients_tags")
            split_ds = split_ds.map(remove_ingredients_func)
        if remove_nutriments:
            suffixes.append("nutriments")
            split_ds = split_ds.map(remove_nutriments_func)
        if remove_product_name:
            suffixes.append("product_name")
            split_ds = split_ds.map(remove_product_name_func)

        y_pred = m.predict(split_ds.padded_batch(32))
        y_pred_binary = np.zeros(y_pred.shape, dtype=int)
        y_pred_binary[y_pred >= 0.5] = 1
        y_true = generate_y_true(y_pred.shape, split_ds, labels)
        ablation_dir = model_dir / "ablations"
        ablation_dir.mkdir(exist_ok=True)

        suffixes.append(split_name)
        suffix = "_".join(suffixes)
        output_report_path = ablation_dir / f"classification_report_{suffix}.json"
        if not output_report_path.exists():
            print("generating classification report...")
            metrics = classification_report(
                y_true, y_pred_binary, target_names=labels, output_dict=True
            )
            metrics = dict(
                sorted(metrics.items(), key=lambda x: x[1]["support"], reverse=True)
            )
            with output_report_path.open("w") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
