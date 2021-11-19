import operator
from typing import Dict, List, Optional

import numpy as np
from robotoff.taxonomy import Taxonomy
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def fill_ancestors(
    y: np.ndarray,
    taxonomy: Taxonomy,
    category_to_id: Optional[Dict[str, int]] = None,
    category_names: Optional[List[str]] = None,
):
    if category_to_id is None and category_names is None:
        raise ValueError("one of category_to_id, category_names must be provided")

    if category_names is None:
        category_names = [
            cat for cat, _ in sorted(category_to_id.items(), key=operator.itemgetter(1))
        ]
    elif category_to_id is None:
        category_to_id = {cat: i for i, cat in enumerate(category_names)}

    y_ = y.copy()
    for i in range(y_.shape[1]):
        cat_mask = y_[:, i].nonzero()[0]

        if len(cat_mask):
            category_name = category_names[i]
            parents = taxonomy[category_name].get_parents_hierarchy()
            parent_ids = [category_to_id[parent.id] for parent in parents]
            for parent_id in parent_ids:
                y_[cat_mask, parent_id] = 1

    return y_


def evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taxonomy: Taxonomy,
    category_names: Optional[List[str]],
):
    y_pred_int = (y_pred > 0.5).astype(y_pred.dtype)
    y_pred_int_filled = fill_ancestors(
        y_pred_int, taxonomy=taxonomy, category_names=category_names
    )

    clf_report = None

    report = {}

    for metric, metric_func in (
        ("precision", precision_score),
        ("recall", recall_score),
        ("f1", f1_score),
    ):
        for average in ("micro", "macro"):
            metric_value = metric_func(y_true, y_pred_int, average=average)
            report["{}-{}".format(average, metric)] = metric_value

    return report, clf_report
