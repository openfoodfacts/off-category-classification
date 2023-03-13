import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support


def generate_df(
    labels: List[str], predictions: np.ndarray, df: pd.DataFrame, output_path: Path
):
    mask = predictions < 0.1
    predictions[mask] = 0
    sort_indices = np.argsort(-predictions, axis=-1)
    positives = np.sum(np.logical_not(mask.astype(int)), axis=-1)

    predict_scores = []
    predict_categories_tags = []
    predict_full = []
    for i in range(len(predictions)):
        item_indices = sort_indices[i, : positives[i]]
        item_full_scores = list(
            zip(
                [labels[idx] for idx in item_indices],
                predictions[i, item_indices].tolist(),
            )
        )
        selected_predictions = [
            (label, score) for label, score in item_full_scores if score >= 0.5
        ]
        item_labels, item_scores = (
            zip(*selected_predictions) if selected_predictions else ([], [])
        )
        predict_scores.append(list(item_scores))
        predict_categories_tags.append(list(item_labels))
        predict_full.append(item_full_scores)

    df["predict_scores"] = predict_scores
    df["predict_categories_tags"] = predict_categories_tags
    df["predict_full"] = predict_full

    prediction_exact = []
    prediction_missing = []
    prediction_extra = []

    for i in range(len(df)):
        gold_set = set(df.categories_tags.iat[i])
        predict_set = set(df.predict_categories_tags.iat[i])
        prediction_exact.append(gold_set == predict_set)
        prediction_missing.append(bool(gold_set.difference(predict_set)))
        prediction_extra.append(bool(predict_set.difference(gold_set)))

    df["prediction_exact"] = prediction_exact
    df["prediction_missing"] = prediction_missing
    df["prediction_extra"] = prediction_extra
    df.to_pickle(output_path)


def generate_y_true(shape, categories_tags_all: list[list[str]], labels: list[str]):
    y_true = np.zeros(shape, dtype=int)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for i, categories_tags in enumerate(categories_tags_all):
        for category_tag in categories_tags:
            if category_tag in label_to_idx:
                y_true[i, label_to_idx[category_tag]] = 1
    return y_true


def find_start_threshold(
    y_true, y_pred, min_precision: float, start_threshold: float = 0.5
):
    below_min_precision = True
    threshold = start_threshold
    y_pred_sorted = np.sort(y_pred)
    indices = np.where(y_pred_sorted >= threshold)[0]

    if not len(indices):
        return threshold

    index = int(indices[0])
    step = 50 if len(indices) >= 500 else 10
    while below_min_precision:
        candidate_threshold = y_pred_sorted[index]
        threshold_y_pred = np.zeros_like(y_pred)
        threshold_y_pred[y_pred >= candidate_threshold] = 1
        below_min_precision = (
            precision_recall_fscore_support(y_true, threshold_y_pred, average="binary")[
                0
            ]
            < min_precision
        )

        if not below_min_precision:
            return threshold

        else:
            threshold = candidate_threshold
            previous_index = index
            index += step

            if index == previous_index or index >= len(y_pred_sorted):
                return threshold


def compute_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    min_precision: float,
    positive_threshold: float = 0.5,
):
    thresholds = {}
    for i, label in tqdm.tqdm(enumerate(labels), total=len(labels)):
        cat_y_true = y_true[:, i]
        cat_y_pred = y_pred[:, i]
        cat_y_pred_sorted = np.sort(cat_y_pred)

        previous_prob = None
        start_threshold = find_start_threshold(cat_y_true, cat_y_pred, min_precision)
        for prob in cat_y_pred_sorted[cat_y_pred_sorted >= start_threshold]:
            if prob == previous_prob:
                continue
            previous_prob = prob
            threshold_cat_y_pred = np.zeros_like(cat_y_pred)
            threshold_cat_y_pred[cat_y_pred >= prob] = 1
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                cat_y_true, threshold_cat_y_pred, average="binary"
            )
            fraction = (cat_y_pred >= prob).astype(int).sum() / (
                cat_y_pred >= positive_threshold
            ).astype(int).sum()
            support = int(cat_y_true.sum())
            if precision >= min_precision:
                thresholds[label] = {
                    "threshold": float(prob),
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "fraction": fraction,
                    "support": support,
                }
                break
        else:
            thresholds[label] = {
                "threshold": None,
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "fraction": fraction,
                "support": support,
            }

    return thresholds


TRAINING_DIR = Path(__file__).parent / "trainings"
SPLIT_DIR = TRAINING_DIR / "splits"

product_df = pd.read_pickle(TRAINING_DIR / "products_df.pkl")
print(f"num of products in df (total): {len(product_df)}")
y_true_by_split = {}

for split_name in ("val", "test"):
    split_barcodes = (SPLIT_DIR / f"{split_name}.txt").read_text().splitlines()
    print(f"num of split products in split {split_name}: {len(split_barcodes)}")

    split_df = product_df.loc[split_barcodes]
    print(f"num of products in df (split): {len(split_df)}")

    previous_labels = None
    for model_dir in (
        p
        for p in TRAINING_DIR.glob("*")
        if p.is_dir() and p.name not in ("splits", "previous_releases")
    ):
        print(f"model: {model_dir}")
        labels = (
            (model_dir / "saved_model/assets/labels_vocab.txt").read_text().splitlines()
        )
        print(f"num of labels: {len(labels)}")

        if previous_labels is None:
            previous_labels = labels
        elif previous_labels != labels:
            raise ValueError("labels changed between models!")

        predictions_dir = model_dir / "predictions"
        pred_file_path = predictions_dir / f"{split_name}.npy"

        if not pred_file_path.exists():
            print(f"{pred_file_path} not found, skipping")
            continue

        y_pred = np.load(predictions_dir / f"{split_name}.npy")
        print(f"shape of predictions: {y_pred.shape}")
        output_df_path = predictions_dir / f"{split_name}_df.pkl"

        if not output_df_path.exists():
            print("generating prediction df...")
            generate_df(
                labels, y_pred, split_df, predictions_dir / f"{split_name}_df.pkl"
            )

        y_pred_binary = np.zeros(y_pred.shape, dtype=int)
        y_pred_binary[y_pred >= 0.5] = 1
        y_true = generate_y_true(y_pred.shape, split_df.categories_tags, labels)
        y_true_by_split[split_name] = y_true
        output_report_path = (
            predictions_dir / f"classification_report_{split_name}.json"
        )
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


for model_dir in (p for p in TRAINING_DIR.glob("*") if p.is_dir()):
    y_pred = np.concatenate(
        (
            np.load(predictions_dir / "val.npy"),
            np.load(predictions_dir / "test.npy"),
        )
    )
    y_true = np.concatenate((y_true_by_split["val"], y_true_by_split["test"]))
    min_precision = 0.99
    threshold_report_path = predictions_dir / f"threshold_report_{min_precision}.json"
    if not threshold_report_path.exists():
        thresholds = compute_thresholds(
            y_true, y_pred, labels, min_precision=min_precision
        )
        thresholds = dict(
            sorted(
                thresholds.items(),
                key=lambda x: x[1]["support"],
                reverse=True,
            )
        )
        with threshold_report_path.open("w") as f:
            json.dump(thresholds, f, indent=4)

        automatically_processable = sum(
            data["support"] * data["fraction"]
            for data in thresholds.values()
            if data["threshold"] is not None and data["support"] >= 30
        )
        total = sum(data["support"] for data in thresholds.values())
        print(
            f"{automatically_processable} automatically processable "
            f"(/{total}, {automatically_processable * 100 / total}%)"
        )
