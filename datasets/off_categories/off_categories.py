from collections import namedtuple
from dataclasses import dataclass
import gzip
import orjson
from typing import Any, Dict

import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class Feature:
  spec: tfds.features.FeatureConnector
  default_value: Any
  input_field: str = None  # use feature name if None


_DESCRIPTION = """
Open Food Facts product categories classification dataset.

More info at:
https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_documentation.txt
"""

_RELEASE_NOTES = {
  '2.0.0': 'DataForGood 2022 dataset',
  '1.0.0': 'Initial release'
}

# Don't forget to run `tfds build --register_checksums` when changing the data source
_DATA_URL = 'https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_products.jsonl.gz'

_FEATURES = {
  'code': Feature(tfds.features.Tensor(shape=(), dtype=tf.string), ''),
  'product_name': Feature(tfds.features.Tensor(shape=(), dtype=tf.string), ''),
  'ingredients_tags': Feature(tfds.features.Tensor(shape=(None,), dtype=tf.string), []),
  'categories_tags': Feature(tfds.features.Tensor(shape=(None,), dtype=tf.string), [])
}

_LABEL = 'categories_tags'


class OffCategories(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = _RELEASE_NOTES

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({k: f.spec for k, f in _FEATURES.items()}),
        supervised_keys=({k: k for k in _FEATURES.keys() if k != _LABEL}, _LABEL),
        # Shuffle is deterministic as long as the split names stay the same
        # (split_name is used internally as hashing salt in the shuffler)
        disable_shuffling=False,
        homepage='https://github.com/openfoodfacts/off-category-classification'
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    # Downloads the data and defines the splits
    paths = dl_manager.download({
      'train': _DATA_URL
    })
    return {s: self._generate_examples(p) for s, p in paths.items()}

  def _generate_examples(self, path):
    # Yields (key, example) tuples from the dataset
    for i, item in enumerate(OffCategories._read_json(path)):
      if _LABEL not in item:
        continue
      features = {k: OffCategories._get_feature(item, k, f) for k, f in _FEATURES.items()}
      yield i, features

  @staticmethod
  def _get_feature(item: Dict, name: str, feature: Feature):
    field = feature.input_field if feature.input_field else name
    return item.get(field, feature.default_value)

  @staticmethod
  def _read_json(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
      for line in f:
        l = line.strip()
        if l:
          yield orjson.loads(l)
