from dataclasses import dataclass
import functools
import gzip
import orjson
from typing import Any, Callable, Dict, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from lib.taxonomy import Taxonomy, get_taxonomy

from .constants import EXCLUDE_LIST_CATEGORIES


@dataclass
class Feature:
    spec: tfds.features.FeatureConnector
    default_value: Any
    input_field: str = None  # use feature name if None
    transform: Optional[Callable] = None


_DESCRIPTION = """
Open Food Facts product categories classification dataset.

More info at:
https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_documentation.txt
"""

_RELEASE_NOTES = {"2.0.0": "DataForGood 2022 dataset", "1.0.0": "Initial release"}

# Don't forget to run `tfds build --register_checksums` when changing the data source
_DATA_URL = "https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_products.jsonl.gz"


def remove_untaxonomized_values(value_tags: list[str], taxonomy: Taxonomy) -> list[str]:
    return [value_tag for value_tag in value_tags if value_tag in taxonomy]


def infer_missing_category_tags(
    category_tags: list[str], taxonomy: Taxonomy
) -> set[str]:
    all_categories = set()
    for category_tag in category_tags:
        category_node = taxonomy[category_tag]
        if category_node:
            all_categories.add(category_node.id)
            all_categories |= set(x.id for x in category_node.get_parents_hierarchy())
    return all_categories


def transform_category_input(category_tags: list[str], taxonomy: Taxonomy) -> list[str]:
    category_tags = remove_untaxonomized_values(category_taxonomy, taxonomy)
    # first get deepest nodes, as we're removing some excluded categories below,
    # we don't want parent categories of excluded categories to be kept in the list
    category_tags = taxonomy.find_deepest_nodes(
        [taxonomy[category_tag] for category_tag in category_tags]
    )
    # Remove excluded categories
    category_tags = [
        category_tag
        for category_tag in category_tags
        if category_tag not in EXCLUDE_LIST_CATEGORIES
    ]
    # Generate the full parent hierarchy
    return list(infer_missing_category_tags(category_tags))


category_taxonomy = get_taxonomy("category")
ingredient_taxonomy = get_taxonomy("ingredient")

_FEATURES = {
    "code": Feature(tfds.features.Tensor(shape=(), dtype=tf.string), default_value=""),
    "product_name": Feature(
        tfds.features.Tensor(shape=(), dtype=tf.string), default_value=""
    ),
    "ingredients_tags": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        default_value=[],
        transform=functools.partial(
            remove_untaxonomized_values, taxonomy=ingredient_taxonomy
        ),
    ),
    "categories_tags": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        default_value=[],
        transform=functools.partial(
            transform_category_input, taxonomy=category_taxonomy
        ),
    ),
}

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
        paths = dl_manager.download({"train": _DATA_URL})
        return {s: self._generate_examples(p) for s, p in paths.items()}

    def _generate_examples(self, path):
        # Yields (key, example) tuples from the dataset
        for i, item in enumerate(OffCategories._read_json(path)):
            if _LABEL not in item:
                continue
            features = {
                k: OffCategories._get_feature(item, k, f) for k, f in _FEATURES.items()
            }
            yield i, features

    @staticmethod
    def _get_feature(item: Dict, name: str, feature: Feature):
        field = feature.input_field if feature.input_field else name
        return item.get(field, feature.default_value)

    @staticmethod
    def _read_json(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                l = line.strip()
                if l:
                    yield orjson.loads(l)
