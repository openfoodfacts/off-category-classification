import pathlib
from typing import List

import tensorflow as tf


def export_model(
        model: tf.keras.Model,
        path: pathlib.Path,
        # label_vocab: List,
        serving_func: tf.function = None,
        **kwargs):
    """
    Save the model with an optional custom serving function.

    Parameters
    ----------
    model: tf.keras.Model
        Keras model instance to be saved.

    path: pathlib.Path
        Path where the model will be saved.

    label_vocab: List[str]
        Label vocabulary

    serving_func: tf.function, optional
        Custom serving function.
        If passed, `serving_func` will be the default endpoint in tensorflow serving.

    **kwargs: dict, optional
        Additional keyword arguments passed to `tf.keras.Model.save`.
    """
    signatures = None
    if serving_func:
        arg_specs, kwarg_specs = model.save_spec()
        concrete_func = serving_func.get_concrete_function(*arg_specs, **kwarg_specs)
        signatures = {'serving_default': concrete_func}

    model.save(path, signatures=signatures, **kwargs)
