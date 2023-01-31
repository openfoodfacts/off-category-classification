import functools
import gzip
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import orjson
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.constant import NUTRIMENT_NAMES
from lib.preprocessing import (
    build_ingredient_processor,
    transform_category_input,
    transform_ingredients_input,
    transform_nutrition_input,
    transform_ocr_ingredients_input,
)
from lib.taxonomy import get_taxonomy


@dataclass
class Feature:
    spec: tfds.features.FeatureConnector
    default_value: Any
    input_field: str = None  # use feature name if None
    transform: Optional[Callable] = None
    # products or ocr
    source: str = "product"


_DESCRIPTION = """
Open Food Facts product categories classification dataset.

More info at:
https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_documentation.txt
"""

_RELEASE_NOTES = {
    "3.0.0": "Deduped dataset",
    "2.0.0": "DataForGood 2022 dataset",
    "1.0.0": "Initial release",
}

# Don't forget to run `tfds build --register_checksums` when changing the data source
_DATA_URLS = {
    "product_data": "https://openfoodfacts.org/data/dataforgood2022/big/v3/predict_categories_dataset_products.jsonl.gz",
    "image_ocr": "https://openfoodfacts.org/data/dataforgood2022/big/v3/predict_categories_dataset_ocrs.jsonl.gz",
}

TEXT_EMBEDDING_DIM = 768
PRODUCT_NAME_MAX_LENGTH = 40


category_taxonomy = get_taxonomy("category", offline=True)
ingredient_taxonomy = get_taxonomy("ingredient", offline=True)
ingredient_processor = build_ingredient_processor(
    ingredient_taxonomy, add_synonym_combinations=True
)


_FEATURES = {
    "code": Feature(tfds.features.Tensor(shape=(), dtype=tf.string), default_value=""),
    "product_name": Feature(
        tfds.features.Tensor(shape=(), dtype=tf.string), default_value=""
    ),
    "ingredients_tags": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        default_value=[],
        input_field="ingredients",
        transform=functools.partial(
            transform_ingredients_input, taxonomy=ingredient_taxonomy
        ),
    ),
    "categories_tags": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        default_value=[],
        transform=functools.partial(
            transform_category_input, taxonomy=category_taxonomy
        ),
    ),
    "ingredients_ocr_tags": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        default_value=None,
        input_field="ocrs",
        transform=functools.partial(
            transform_ocr_ingredients_input, processor=ingredient_processor
        ),
        source="ocr",
    ),
    # Debug field with lang information
    "ingredients_ocr_tags_debug": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        default_value=None,
        input_field="ocrs",
        transform=functools.partial(
            transform_ocr_ingredients_input, processor=ingredient_processor, debug=True
        ),
        source="ocr",
    ),
}

for nutriment_name in NUTRIMENT_NAMES:
    _FEATURES[nutriment_name] = Feature(
        tfds.features.Tensor(shape=(), dtype=tf.float32),
        default_value=None,
        input_field=f"nutriments.{nutriment_name.replace('_', '-')}_100g",
        transform=functools.partial(
            transform_nutrition_input, nutriment_name=nutriment_name
        ),
    )

_LABEL = "categories_tags"


class OffCategories(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = _RELEASE_NOTES

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {k: f.spec for k, f in _FEATURES.items()}
            ),
            supervised_keys=({k: k for k in _FEATURES.keys() if k != _LABEL}, _LABEL),
            # Shuffle is deterministic as long as the split names stay the same
            # (split_name is used internally as hashing salt in the shuffler)
            disable_shuffling=False,
            homepage="https://github.com/openfoodfacts/off-category-classification",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Downloads the data and defines the splits
        paths = dl_manager.download(_DATA_URLS)
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths: List[str]):
        product_data_path = paths["product_data"]
        image_ocr_path = paths["image_ocr"]
        # Yields (key, example) tuples from the dataset
        for i, (product, ocr) in enumerate(
            zip(
                OffCategories._read_json(product_data_path),
                OffCategories._read_json(image_ocr_path),
            )
        ):
            if product["code"] != ocr["code"]:
                raise ValueError(
                    "product and ocr data files are not synchronized: "
                    f"{product['code']} != {ocr['code']}"
                )
            if _LABEL not in product:
                continue
            features = {
                k: OffCategories._get_feature(
                    product if f.source == "product" else ocr, k, f
                )
                for k, f in _FEATURES.items()
            }
            if not features[_LABEL]:
                # Don't keep products without categories
                continue
            yield (i, features)

    @staticmethod
    def _get_feature(item: Dict, name: str, feature: Feature):
        field = feature.input_field if feature.input_field else name
        # Allow to specify subfield using dot notation: nutriments.sugars_100g
        splitted_field = field.split(".")
        for subfield in splitted_field[:-1]:
            item = item[subfield]
        field = splitted_field[-1]
        value = item.get(field, feature.default_value)
        if feature.transform is not None:
            value = feature.transform(value)
        return value

    @staticmethod
    def _read_json(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if l:
                    yield orjson.loads(l)
