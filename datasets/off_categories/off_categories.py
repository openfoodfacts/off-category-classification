from dataclasses import dataclass
import functools
import gzip
import orjson
from typing import Any, Callable, Dict, List, Optional


from more_itertools import chunked
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFAutoModel, AutoTokenizer
from lib.preprocessing import (
    transform_category_input,
    transform_ingredients_input,
    transform_nutrition_input,
)

from lib.taxonomy import Taxonomy, get_taxonomy
from lib.constant import NUTRIMENT_NAMES


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

TEXT_EMBEDDING_DIM = 768
PRODUCT_NAME_MAX_LENGTH = 40


def get_ingredient_embeddings(ingredient_taxonomy: Taxonomy) -> Dict[str, str]:
    mapping = {}
    for node in ingredient_taxonomy.iter_nodes():
        if "en" in node.names:
            mapping[node.id] = node.names["en"]

        if "xx" in node.names:
            # Return international name if it exists
            mapping[node.id] = node.names["xx"]

        raise ValueError(f"no en or xx translation for ingredient {node.id}")

    text_to_embedding = {}
    for batch in chunked(mapping.items(), 64):
        embeddings = generate_embeddings([x[1] for x in batch])
        for node_id, name in batch:
            text_to_embedding[node_id] = embeddings[name]
    return text_to_embedding


@functools.lru_cache()
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = TFAutoModel.from_pretrained("microsoft/deberta-v3-base")
    return tokenizer, model


def generate_embeddings(texts: List[str], batch_size: int = 64):
    tokenizer, model = load_resources()
    text_to_embeddings = {}
    for text_batch in chunked(texts, batch_size):
        inputs = tokenizer(
            text_batch,
            return_tensors="tf",
            padding="max_length",
            truncation=True,
            max_length=PRODUCT_NAME_MAX_LENGTH,
        )
        outputs = model(inputs)
        embeddings = outputs.last_hidden_state.numpy()
        for i in enumerate(text_batch):
            embedding = embeddings[i]
            # Remove padding tokens
            text_to_embeddings[text_batch[i]] = embedding[
                inputs.input_ids[i] != tokenizer.pad_token_id
            ]
    return text_to_embeddings


category_taxonomy = get_taxonomy("category", offline=True)
ingredient_taxonomy = get_taxonomy("ingredient", offline=True)

_FEATURES = {
    "code": Feature(tfds.features.Tensor(shape=(), dtype=tf.string), default_value=""),
    "product_name": Feature(
        tfds.features.Tensor(shape=(), dtype=tf.string), default_value=""
    ),
    # "product_name_embed": Feature(
    #     tfds.features.Tensor(
    #         shape=(PRODUCT_NAME_MAX_LENGTH, TEXT_EMBEDDING_DIM), dtype=tf.float32
    #     ),
    #     default_value=None,
    # ),
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
}

for nutriment_name in NUTRIMENT_NAMES:
    _FEATURES[nutriment_name] = Feature(
        tfds.features.Tensor(shape=(), dtype=tf.float32),
        default_value=None,
        input_field=f"nutriments.{nutriment_name}_100g",
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
        paths = dl_manager.download({"train": _DATA_URL})
        return {s: self._generate_examples(p) for s, p in paths.items()}

    def _generate_examples(self, path):
        # Yields (key, example) tuples from the dataset
        data_gen = (
            x for x in enumerate(OffCategories._read_json(path)) if _LABEL in x[1]
        )
        # null_string_embed = generate_embeddings([""])[""]
        for batch in chunked(data_gen, 256):
            features_batch = []
            for i, item in batch:
                features = {
                    k: OffCategories._get_feature(item, k, f)
                    for k, f in _FEATURES.items()
                    if k not in ("product_name_embed",)
                }
                if not features[_LABEL]:
                    # Don't keep products without categories
                    continue
                features_batch.append((i, features))

            # text_to_embedding = generate_embeddings(
            #     [
            #         features["product_name"]
            #         for _, features in features_batch
            #         if features["product_name"]
            #     ],
            #     batch_size=256,
            # )
            # for i, features in features_batch:
            #     features["product_name_embed"] = (
            #         text_to_embedding[features["product_name"]]
            #         if features["product_name"]
            #         else null_string_embed
            #     )

            yield from features_batch

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
