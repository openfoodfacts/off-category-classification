from typing import List, Iterable, Tuple

import numpy as np
import pandas as pd
from robotoff.taxonomy import Taxonomy, TaxonomyNode
from sklearn.decomposition import PCA
from tensorflow import keras

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Set2


def generate_analysis_model(model: keras.Model, embedding_layer_name: str):
    inputs = model.inputs[:]
    embedding_layer = model.get_layer(embedding_layer_name)
    return keras.Model(inputs=inputs, outputs=[embedding_layer.output])


def get_interactive_embedding_plot(embeddings: np.ndarray, df: pd.DataFrame):
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    df_copy = df.copy()
    df_copy["pca_x"] = pca_embeddings[:, 0]
    df_copy["pca_y"] = pca_embeddings[:, 1]
    hover = HoverTool(
        tooltips=[
            ("product name", "@product_name"),
            ("barcode", "@code"),
            ("categories", "@deepest_categories"),
            ("predicted categories", "@predicted_deepest_categories"),
        ]
    )
    hover.attachment = "right"

    p = figure(
        title="Embedding projection",
        plot_width=1200,
        plot_height=800,
        tools=("pan,wheel_zoom,reset", "box_zoom", "undo"),
    )
    p.add_tools(hover)

    for filter_, name, color in zip(
        (
            "is_correct",
            "missing_cat_error",
            "additional_cat_error",
            "over_pred_cat_error",
            "under_pred_cat_error",
        ),
        ["correct", "missing", "addition", "over-prediction", "under-prediction"],
        Set2[5],
    ):
        data_source = ColumnDataSource(df_copy[df_copy[filter_] == True])
        p.circle(
            "pca_x",
            "pca_y",
            source=data_source,
            line_alpha=0,
            fill_alpha=0.4,
            size=5,
            legend=name,
            color=color,
        )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    return p


def get_deepest_categories(
    taxonomy: Taxonomy, categories_tags: Iterable[List[str]]
) -> List[List[str]]:
    return [
        sorted(
            (
                x.id
                for x in taxonomy.find_deepest_nodes([taxonomy[c] for c in categories])
            )
        )
        for categories in categories_tags
    ]


def get_error_category(
    predicted_categories: List[str], true_categories: List[str], taxonomy: Taxonomy
) -> Tuple[bool, bool, bool, bool]:
    (
        missing_nodes,
        additional_nodes,
        over_predicted_nodes,
        under_predicted_nodes,
    ) = get_classification_errors(predicted_categories, true_categories, taxonomy)

    return (
        bool(len(missing_nodes)),
        bool(len(additional_nodes)),
        bool(len(over_predicted_nodes)),
        bool(len(under_predicted_nodes)),
    )


def get_classification_errors(
    predicted_categories: List[str], true_categories: List[str], taxonomy: Taxonomy
) -> Tuple[
    List[TaxonomyNode], List[TaxonomyNode], List[TaxonomyNode], List[TaxonomyNode]
]:
    predicted_categories_nodes = [taxonomy[c] for c in predicted_categories]
    true_categories_nodes = [taxonomy[c] for c in true_categories]

    missing_nodes = []
    additional_nodes = []
    over_predicted_nodes = []
    under_predicted_nodes = []

    for true_category in list(true_categories_nodes):
        if true_category in list(predicted_categories_nodes):
            true_categories_nodes.remove(true_category)
            predicted_categories_nodes.remove(true_category)

    for true_category in list(true_categories_nodes):
        if true_category.is_parent_of_any(predicted_categories_nodes):
            true_categories_nodes.remove(true_category)

            for p in predicted_categories_nodes:
                if p.is_child_of(true_category):
                    over_predicted_nodes.append(p)
                    predicted_categories_nodes.remove(p)

        elif any(true_category.is_child_of(p) for p in predicted_categories_nodes):
            true_categories_nodes.remove(true_category)

            for p in predicted_categories_nodes:
                if true_category.is_child_of(p):
                    under_predicted_nodes.append(p)
                    predicted_categories_nodes.remove(p)

        else:
            missing_nodes.append(true_category)
            true_categories_nodes.remove(true_category)

    for predicted_category in list(predicted_categories_nodes):
        additional_nodes.append(predicted_category)
        predicted_categories_nodes.remove(predicted_category)

    assert len(predicted_categories_nodes) == 0
    assert len(true_categories_nodes) == 0

    return missing_nodes, additional_nodes, over_predicted_nodes, under_predicted_nodes
