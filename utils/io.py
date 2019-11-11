import dataclasses
import json
import pathlib
import shutil
from typing import Dict

import dacite

import settings
from category_classification.models import Config


def save_product_name_vocabulary(token_to_int: Dict[str, int],
                                 model_dir: pathlib.Path):
    return save_json(token_to_int, model_dir / settings.PRODUCT_NAME_VOC_NAME)


def load_product_name_vocabulary(model_dir: pathlib.Path):
    return load_json(model_dir / settings.PRODUCT_NAME_VOC_NAME)


def save_category_vocabulary(category_to_int: Dict[str, int],
                             model_dir: pathlib.Path):
    return save_json(category_to_int, model_dir / settings.CATEGORY_VOC_NAME)


def load_category_vocabulary(model_dir: pathlib.Path):
    return load_json(model_dir / settings.CATEGORY_VOC_NAME)


def copy_category_taxonomy(taxonomy_path: pathlib.Path,
                           model_dir: pathlib.Path):
    shutil.copy(str(taxonomy_path), str(model_dir / settings.CATEGORY_TAXONOMY_NAME))


def save_ingredient_vocabulary(ingredient_to_int: Dict[str, int],
                               model_dir: pathlib.Path):
    return save_json(ingredient_to_int, model_dir / settings.INGREDIENT_VOC_NAME)


def load_ingredient_vocabulary(model_dir: pathlib.Path):
    return load_json(model_dir / settings.INGREDIENT_VOC_NAME)


def save_config(model_config: Config, model_dir: pathlib.Path):
    config_dict = dataclasses.asdict(model_config)
    save_json(config_dict, model_dir / settings.CONFIG_NAME)


def load_config(model_dir: pathlib.Path) -> Config:
    config = load_json(model_dir / settings.CONFIG_NAME)
    return dacite.from_dict(Config, config)


def save_json(obj: object, path: pathlib.Path):
    with path.open('w') as f:
        return json.dump(obj, f)


def load_json(path: pathlib.Path):
    with path.open('r') as f:
        return json.load(f)
