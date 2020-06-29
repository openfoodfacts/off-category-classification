from typing import List

import numpy as np

from tensorflow.keras.callbacks import Callback


def update_averages(
    ema_list: List[np.ndarray], weights_list: List[np.ndarray], t: int, max_decay=0.9999
):
    decay = (1.0 + t) / (10.0 + t)
    if decay > max_decay:
        decay = max_decay

    for weight, ema in zip(weights_list, ema_list):
        ema -= (1 - decay) * (ema - weight)


class MovingWeightAveraging(Callback):
    def __init__(self):
        super().__init__()
        self.averages = []
        self.train_weights = None

    def on_train_batch_begin(self, batch, logs=None):
        if not self.averages:
            self.averages = self.model.get_weights()

    def on_train_batch_end(self, batch, logs=None):
        update_averages(self.averages, self.model.get_weights(), batch)

    def on_test_begin(self, logs=None):
        self.train_weights = self.model.get_weights()
        self.model.set_weights(self.averages)

    def on_test_end(self, logs=None):
        self.model.set_weights(self.train_weights)
        self.train_weights = None
