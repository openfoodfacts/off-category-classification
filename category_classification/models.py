import contextlib
import dataclasses
import datetime
from typing import List, Optional, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .constants import NUTRIMENTS


@dataclasses.dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    label_smoothing: float = 0
    start_datetime: Union[datetime.datetime, None, str] = None
    end_datetime: Union[datetime.datetime, None, str] = None


@dataclasses.dataclass
class TextPreprocessingConfig:
    lower: bool
    strip_accent: bool
    remove_punct: bool
    remove_digit: bool


@dataclasses.dataclass
class ModelConfig:
    product_name_lstm_recurrent_dropout: float
    product_name_lstm_dropout: float
    product_name_embedding_size: int
    product_name_lstm_units: int
    product_name_max_length: int
    hidden_dim: int
    hidden_dropout: float
    output_dim: Optional[int] = None
    product_name_voc_size: Optional[int] = None
    ingredient_voc_size: Optional[int] = None
    nutriment_input: bool = False


@dataclasses.dataclass
class Config:
    product_name_preprocessing_config: TextPreprocessingConfig
    train_config: TrainConfig
    model_config: ModelConfig
    lang: str
    product_name_min_count: int
    category_min_count: int = 0
    ingredient_min_count: int = 0



@tf.keras.utils.register_keras_serializable()
class OutputMapperLayer(layers.Layer):
    '''
        The OutputMapperLayer converts the label indices produced by the model to
        the taxonomy category ids and limits them to top N labels.
    '''
    def __init__(self, labels: List[str], top_n: int, **kwargs):
        self.labels = labels
        self.top_n = top_n

        super(OutputMapperLayer, self).__init__(**kwargs)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        tf_labels = tf.constant([self.labels], dtype="string")
        tf_labels = tf.tile(tf_labels, [batch_size, 1])

        top_n = tf.nn.top_k(x, k=self.top_n, sorted=True, name="top_k").indices

        top_conf = tf.gather(x, top_n, batch_dims=1)
        top_labels = tf.gather(tf_labels, top_n, batch_dims=1)

        return [top_conf, top_labels]


def build_model(config: ModelConfig) -> keras.Model:
    ingredient_input = keras.Input(shape=(config.ingredient_voc_size), dtype=tf.float32, name="ingredients")
    product_name_input = keras.Input(shape=(1,), dtype=tf.string, name="product_name")

    # max_tokens is chosen somewhat arbitrarily. TODO(kulizhsy): investigate if it could be better.
    product_name_preprocessing = tf.keras.layers.TextVectorization(split='whitespace', max_tokens=93000, output_sequence_length=config.product_name_max_length)
    product_name_preprocessing.adapt(train_data.product_name)

    product_name_layer = product_name_preprocessing(product_name_input)

    product_name_embedding = layers.Embedding(
        input_dim=93000,
        output_dim=config.product_name_embedding_size,
        mask_zero=False,
    )(product_name_layer)
    product_name_lstm = layers.Bidirectional(
        layers.LSTM(
            units=config.product_name_lstm_units,
            recurrent_dropout=config.product_name_lstm_recurrent_dropout,
            dropout=config.product_name_lstm_dropout,
        )
    )(product_name_embedding)

    ingredient_preprocessing = tf.keras.layers.StringLookup(max_tokens=335, output_mode="multi_hot")
    ingredient_preprocessing.adapt(tf.ragged.constant(train_data.ingredient_tags))

    ingredient_layer = ingredient_preprocessing(ingredient_input)

    inputs = [ingredient_input, product_name_input]
    concat_input = [ingredient_layer, product_name_lstm]

    if config.nutriment_input:
        nutriment_input = layers.Input(shape=len(NUTRIMENTS))
        inputs.append(nutriment_input)
        concat_input.append(nutriment_input)

    concat = layers.Concatenate()(concat_input)
    concat = layers.Dropout(config.hidden_dropout)(concat)
    hidden = layers.Dense(config.hidden_dim)(concat)
    hidden = layers.Dropout(config.hidden_dropout)(hidden)
    hidden = layers.Activation("relu")(hidden)
    output = layers.Dense(config.output_dim, activation="sigmoid")(hidden)
    return keras.Model(inputs=inputs, outputs=[output])

def to_serving_model(base_model: keras.Model, categories: List[str]) -> keras.Model:
    mapper_layer = OutputMapperLayer(categories, 50)(base_model.output)
    return keras.Model(base_model.input, mapper_layer)
