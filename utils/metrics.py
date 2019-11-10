import operator
from typing import Optional, List, Dict

import numpy as np
from robotoff.taxonomy import Taxonomy

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def hierarchical_precision_recall_f1(y_true: np.ndarray,
                                     y_pred: np.ndarray):
    ix = np.where((y_true != 0) & (y_pred != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred)
    all_positives = np.count_nonzero(y_true)

    h_precision = true_positives / all_results
    h_recall = true_positives / all_positives
    beta = 1
    h_f_1 = (1. + beta ** 2.) * h_precision * h_recall / (beta ** 2. * h_precision + h_recall)

    return {
        'h_precision': h_precision,
        'h_recall': h_recall,
        'h_f1': h_f_1,
    }


def fill_ancestors(y: np.ndarray, taxonomy: Taxonomy,
                   category_to_id: Optional[Dict[str, int]] = None,
                   category_names: Optional[List[str]] = None):
    if category_to_id is None and category_names is None:
        raise ValueError("one of category_to_id, category_names must be provided")

    if category_names is None:
        category_names = [cat for cat, _ in sorted(category_to_id.items(),
                                                   key=operator.itemgetter(1))]
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


def evaluation_report(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      taxonomy: Taxonomy,
                      category_names: Optional[List[str]]):
    y_pred_int = (y_pred > 0.5).astype(y_pred.dtype)
    y_pred_int_filled = fill_ancestors(y_pred_int,
                                       taxonomy=taxonomy,
                                       category_names=category_names)

    clf_report = classification_report(y_true, y_pred_int,
                                       target_names=category_names,
                                       output_dict=True)

    report = {}

    for metric, metric_func in (
            ('precision', precision_score),
            ('recall', recall_score),
            ('f1', f1_score),
    ):
        for average in ('micro', 'macro'):
            metric_value = metric_func(y_true, y_pred_int, average=average)
            report["{}-{}".format(average, metric)] = metric_value

            metric_value_filled = metric_func(y_true, y_pred_int_filled, average=average)
            report["ancestor-{}-{}".format(average, metric)] = metric_value_filled

    hierarchical_metrics = hierarchical_precision_recall_f1(y_true, y_pred_int_filled)
    report.update(hierarchical_metrics)

    return report, clf_report
