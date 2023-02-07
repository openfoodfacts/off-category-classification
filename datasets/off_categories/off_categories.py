import functools
import gzip
from collections import defaultdict
from contextlib import ContextDecorator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import numpy as np
import orjson
import tensorflow as tf
import tensorflow_datasets as tfds

from lib.constant import IMAGE_EMBEDDING_DIM, MAX_IMAGE_EMBEDDING, NUTRIMENT_NAMES
from lib.preprocessing import (
    build_ingredient_processor,
    extract_ocr_ingredients,
    transform_category_input,
    transform_image_embeddings,
    transform_image_embeddings_mask,
    transform_ingredients_input,
    transform_ingredients_ocr_tags,
    transform_nutrition_input,
)
from lib.taxonomy import get_taxonomy


@dataclass
class Feature:
    spec: tfds.features.FeatureConnector
    default_value: Optional[Any] = None
    input_field: Optional[str] = None  # use feature name if None
    transform: Optional[Callable] = None
    # products or ocr
    source: str = "product"


_DESCRIPTION = """
Open Food Facts product categories classification dataset.

More info at:
https://openfoodfacts.org/data/dataforgood2022/big/predict_categories_dataset_documentation.txt
"""

_RELEASE_NOTES = {
    "4.0.0": "Dataset with pre-shuffling",
    "3.0.0": "Deduped dataset",
    "2.0.0": "DataForGood 2022 dataset",
    "1.0.0": "Initial release",
}

# Don't forget to run `tfds build --register_checksums` when changing the data source
_DATA_URLS = {
    "product_data": "https://openfoodfacts.org/data/dataforgood2022/big/v4/predict_categories_dataset_products.jsonl.gz",
    "image_ocr": "https://openfoodfacts.org/data/dataforgood2022/big/v4/predict_categories_dataset_ocrs.jsonl.gz",
    "image_embedding": "https://openfoodfacts.org/data/dataforgood2022/big/v4/predict_categories_dataset_image_embeddings.hdf5",
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
        input_field="ocr_ingredients",
        transform=transform_ingredients_ocr_tags,
    ),
    # Debug field with lang information
    "ingredients_ocr_tags_debug": Feature(
        tfds.features.Tensor(shape=(None,), dtype=tf.string),
        input_field="ocr_ingredients",
    ),
    "image_embeddings": Feature(
        tfds.features.Tensor(
            shape=(
                MAX_IMAGE_EMBEDDING,
                IMAGE_EMBEDDING_DIM,
            ),
            dtype=tf.float32,
        ),
        input_field="image_embeddings",
        transform=functools.partial(
            transform_image_embeddings,
            max_images=MAX_IMAGE_EMBEDDING,
            embedding_dim=IMAGE_EMBEDDING_DIM,
        ),
    ),
    "image_embeddings_mask": Feature(
        tfds.features.Tensor(shape=(MAX_IMAGE_EMBEDDING,), dtype=tf.int64),
        input_field="image_embeddings",
        transform=functools.partial(
            transform_image_embeddings_mask, max_images=MAX_IMAGE_EMBEDDING
        ),
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


class ImageEmbeddingDataset(ContextDecorator):
    def __init__(
        self,
        data_path: Path,
        id_field_name: str = "external_id",
        embedding_field_name: str = "embedding",
    ):
        self.data_path = data_path
        if not self.data_path.is_file():
            raise ValueError("file not found: %s", self.data_path)
        self.id_field_name = id_field_name
        self.embedding_field_name = embedding_field_name
        self.index_mapping = defaultdict(dict)

        with h5py.File(self.data_path, "r") as f:
            id_dset = f[self.id_field_name]
            non_zeros_indexes = np.nonzero(id_dset[:])[0]
            external_ids = id_dset[non_zeros_indexes]
            for i in range(len(external_ids)):
                external_id = external_ids[i]
                index = non_zeros_indexes[i]
                barcode, image_id = external_id.decode().split("_")
                self.index_mapping[barcode][int(image_id)] = int(index)

        self.opened_file = None

    def __enter__(self):
        self.opened_file = h5py.File(self.data_path, "r")
        return self

    def __exit__(self, *exc):
        if self.opened_file is not None:
            self.opened_file.close()
        self.opened_file = None

    def __getitem__(self, barcode: str):
        if self.opened_file is None:
            raise ValueError("ImageEmbeddingDataset must be used as a context manager")
        items = self.index_mapping[barcode]
        indexes = list(items.values())
        embeddings = self.opened_file["embedding"][indexes]
        return {
            image_id: embedding
            for (image_id, embedding) in zip(list(items.keys()), embeddings)
        }


class OffCategories(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("4.0.0")
    RELEASE_NOTES = _RELEASE_NOTES

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {k: f.spec for k, f in _FEATURES.items()}
            ),
            supervised_keys=({k: k for k in _FEATURES.keys() if k != _LABEL}, _LABEL),
            # As adding new files
            disable_shuffling=True,
            homepage="https://github.com/openfoodfacts/off-category-classification",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Downloads the data and defines the splits
        paths = dl_manager.download(_DATA_URLS)
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths: List[str]):
        product_data_path = paths["product_data"]
        image_ocr_path = paths["image_ocr"]
        image_embedding_path = paths["image_embedding"]
        embedding_ds = ImageEmbeddingDataset(image_embedding_path)

        # Yields (key, example) tuples from the dataset
        with embedding_ds:
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
                if _LABEL not in product or not product.get("product_name"):
                    # Remove items without product name or categories
                    continue
                barcode = product["code"]
                image_embeddings = embedding_ds[barcode]
                ocr_ingredients = extract_ocr_ingredients(
                    ocr["ocrs"], processor=ingredient_processor, debug=True
                )
                input_data = {
                    **product,
                    "ocr_ingredients": ocr_ingredients,
                    "image_embeddings": image_embeddings,
                }
                features = {
                    k: OffCategories._get_feature(input_data, k, f)
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
