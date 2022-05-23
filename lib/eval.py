from typing import Tuple, List

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.dataset import select_features


def top_predictions(
        ds: tf.data.Dataset,
        preds: Tuple[tf.Tensor, tf.Tensor],
        top_k: int = 5,
        features: List[str] = ['code', 'product_name']
    ) -> pd.DataFrame:
    """
    Parameters
    ----------
    ds : tf.data.Dataset
        Dict-based dataset. Nested features are not supported.

    preds : Tuple[tf.Tensor, tf.Tensor]
        Predictions for `ds` in the format output by `lib.model.OutputMapperLayer`.

    top_k : int, optional
        Output `top_k` predictions.

    features : List[str], optional
        Features to output along with the top predictions.

    Returns
    -------
    pd.Dataframe
        Table with top-k predictions for each element of `ds`.
    """
    top_conf = pd.DataFrame(preds[0]).iloc[:, :top_k].stack()
    top_cat = pd.DataFrame(preds[1]).iloc[:, :top_k].stack()

    top_pred = (
        pd.DataFrame.from_dict({
            'category': top_cat.apply(lambda x: x.decode()),
            'confidence': (top_conf * 100).round(2).astype(str) + '%'
        })
        .agg(lambda x: f"{x['category']}: {x['confidence']}", axis=1)
        .unstack()
    )

    item = (
        tfds.as_dataframe(select_features(ds, features))
        .applymap(lambda x: x.decode())
    )

    return pd.concat([item, top_pred], axis=1)
