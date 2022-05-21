import dataclasses
import json
import pathlib
import shutil
from typing import Dict

import dacite

from lib import settings
from lib.config import Config


def save_category_vocabulary(category_vocab: Dict[str, int], model_dir: pathlib.Path):
    category_to_ind = {name: idx for idx, name in enumerate(category_vocab)}
    return save_json(category_to_ind, model_dir / settings.CATEGORY_VOC_NAME)


def load_category_vocabulary(model_dir: pathlib.Path):
    return load_json(model_dir / settings.CATEGORY_VOC_NAME)


def copy_category_taxonomy(taxonomy_path: pathlib.Path, model_dir: pathlib.Path):
    shutil.copy(str(taxonomy_path), str(model_dir / settings.CATEGORY_TAXONOMY_NAME))


def save_config(model_config: Config, model_dir: pathlib.Path):
    config_dict = dataclasses.asdict(model_config)
    save_json(config_dict, model_dir / settings.CONFIG_NAME)


def load_config(model_dir: pathlib.Path) -> Config:
    config = load_json(model_dir / settings.CONFIG_NAME)
    return dacite.from_dict(Config, config)


def save_json(obj: object, path: pathlib.Path):
    with path.open("w") as f:
        return json.dump(obj, f)


def load_json(path: pathlib.Path):
    with path.open("r") as f:
        return json.load(f)
