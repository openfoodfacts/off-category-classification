import contextlib
import dataclasses
from typing import List, Optional

from tensorflow import keras
from tensorflow.keras import layers


@dataclasses.dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    save_dirname: str


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


@dataclasses.dataclass
class Config:
    product_name_preprocessing_config: TextPreprocessingConfig
    train_config: TrainConfig
    model_config: ModelConfig
    lang: str
    product_name_min_count: int
    category_min_count: int = 0
    ingredient_min_count: int = 0


def build_model(config: ModelConfig) -> keras.Model:
    ingredient_input = layers.Input(shape=(config.ingredient_voc_size, ))
    product_name_input = layers.Input(shape=(config.product_name_max_length, ))
    product_name_embedding = layers.Embedding(input_dim=config.product_name_voc_size+1,
                                              output_dim=config.product_name_embedding_size,
                                              mask_zero=False)(product_name_input)
    product_name_lstm = layers.Bidirectional(
        layers.LSTM(
            units=config.product_name_lstm_units,
            recurrent_dropout=config.product_name_lstm_recurrent_dropout,
            dropout=config.product_name_lstm_dropout,
        )
    )(product_name_embedding)
    concat = layers.Concatenate()([ingredient_input, product_name_lstm])

    hidden = layers.Dense(config.hidden_dim)(concat)
    hidden = layers.Dropout(config.hidden_dropout)(hidden)
    hidden = layers.Activation('relu')(hidden)

    output = layers.Dense(config.output_dim, activation='sigmoid')(hidden)
    return keras.Model(inputs=[ingredient_input, product_name_input], outputs=[output])


@contextlib.contextmanager
def use_params(model: keras.Model, weights: List):
    old_weights = model.get_weights()
    model.set_weights(weights)
    try:
        yield
    finally:
        model.set_weights(old_weights)
