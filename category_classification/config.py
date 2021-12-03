import dataclasses
from typing import Optional


@dataclasses.dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    label_smoothing: float = 0

    # Should be left unset in the provided config.
    start_datetime: str = ""
    end_datetime: str = ""


@dataclasses.dataclass
class ModelConfig:
    product_name_lstm_recurrent_dropout: float
    product_name_lstm_dropout: float
    product_name_embedding_size: int
    product_name_lstm_units: int
    product_name_max_length: int
    product_name_max_tokens: int
    hidden_dim: int
    hidden_dropout: float

    category_min_count: int
    ingredient_min_count: int


@dataclasses.dataclass
class Config:
    train_config: TrainConfig
    model_config: ModelConfig
