import json
import pathlib
import shutil
from typing import Dict

import tensorflow as tf

from lib import settings
from lib.model import to_serving_model


TRAINING_MODEL_SUBDIR = 'training_model'
SERVING_MODEL_SUBDIR = 'serving_model'


def save_model_bundle(
        model_dir: pathlib.Path,
        model: tf.keras.Model,
        categories_vocab: Dict[str, int]):
    save_category_vocabulary(categories_vocab, model_dir)
    model.save(model_dir/TRAINING_MODEL_SUBDIR)
    to_serving_model(model, categories_vocab).save(model_dir/SERVING_MODEL_SUBDIR)


def load_training_model(model_dir: pathlib.Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_dir/TRAINING_MODEL_SUBDIR)


def load_serving_model(model_dir: pathlib.Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_dir/SERVING_MODEL_SUBDIR)


def save_category_vocabulary(category_vocab: Dict[str, int], model_dir: pathlib.Path):
    category_to_ind = {name: idx for idx, name in enumerate(category_vocab)}
    return save_json(category_to_ind, model_dir / settings.CATEGORY_VOC_NAME)


def load_category_vocabulary(model_dir: pathlib.Path):
    return load_json(model_dir / settings.CATEGORY_VOC_NAME)


def copy_category_taxonomy(taxonomy_path: pathlib.Path, model_dir: pathlib.Path):
    shutil.copy(str(taxonomy_path), str(model_dir / settings.CATEGORY_TAXONOMY_NAME))


def save_json(obj: object, path: pathlib.Path):
    with path.open("w") as f:
        return json.dump(obj, f)


def load_json(path: pathlib.Path):
    with path.open("r") as f:
        return json.load(f)
