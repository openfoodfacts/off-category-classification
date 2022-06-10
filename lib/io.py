import pathlib
import tempfile
from typing import List

import tensorflow as tf


def save_model(
        path: pathlib.Path,
        model: tf.keras.Model,
        labels_vocab: List[str],
        serving_func: tf.function = None,
        **kwargs):
    """
    Save the model and labels, with an optional custom serving function.

    Parameters
    ----------
    path: pathlib.Path
        Path where the model will be saved.

    model: tf.keras.Model
        Keras model instance to be saved.

    labels_vocab: List[str]
        Label vocabulary.

    serving_func: tf.function, optional
        Custom serving function.
        If passed, `serving_func` will be the default endpoint in tensorflow serving.

    **kwargs: dict, optional
        Additional keyword arguments passed to `tf.keras.Model.save`.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    labels_path = pathlib.Path(tmp_dir.name).joinpath('labels_vocab.txt')
    with labels_path.open('w') as w:
        w.writelines([f"{label}\n" for label in labels_vocab])
    model.labels_file = tf.saved_model.Asset(str(labels_path))

    signatures = None
    if serving_func:
        arg_specs, kwarg_specs = model.save_spec()
        concrete_func = serving_func.get_concrete_function(*arg_specs, **kwarg_specs)
        signatures = {'serving_default': concrete_func}

    model.save(str(path), signatures=signatures, **kwargs)

    # must occur after model.save, so Asset source is still around for save
    tmp_dir.cleanup()


def load_model(path: pathlib.Path, **kwargs):
    """
    Load the model and labels.

    Parameters
    ----------
    path: pathlib.Path
        Path to the saved model.

    **kwargs: dict, optional
        Additional keyword arguments passed to `tf.keras.models.load_model`.

    Returns
    -------
    (tf.keras.Model, List[str])
        Model and labels.
    """
    model = tf.keras.models.load_model(str(path))
    labels_file = model.labels_file.asset_path.numpy()
    labels = open(labels_file).read().splitlines()
    return model, labels
