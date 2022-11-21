# INDEX
* [Imports and functions](#Imports-and-functions)
* [Configuration](#Configuration)
* [Prepare dataset](#Prepare-dataset)
* [Build model](#Build-model)
    * [Model inputs](#Model-inputs)
    * [Model output](#Model-output)
    * [Model](#Model)
* [Train model](#Train-model)
    * [Save model and resources](#Save-model-and-resources)
    * [Training stats](#Training-stats)
* [Test model](#Test-model)
    * [Predict with training model](#Predict-with-training-model)
    * [Predict with serving model](#Predict-with-serving-model)


```python
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False
# eventual initialization for colab notebooks
if IN_COLAB:
  # we try hard to be re-entrant,
  # that is to be able to rerun this without cloning repository more than once
  COLAB_BRANCH = "master"
  !curl https://raw.githubusercontent.com/openfoodfacts/off-category-classification/$COLAB_BRANCH/lib/colab.py --output /content/colab.py
  !cd /content && python /content/colab.py $COLAB_BRANCH
  %cd /content/off-category-classification/experiments
```


```python
# codecarbon - start tracking
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(log_level = "WARNING", save_to_api = True, experiment_id = "6d2c8401-afba-42de-9600-6e95bea5fd80")
tracker.start()
```

# Imports


```python
import sys
sys.path.append('../') #append a relative path to the top package to the search path
```


```python
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras import callbacks, layers
from tensorflow.keras.utils import plot_model

from lib.dataset import *
from lib.directories import init_cache_dir, init_model_dir
from lib.io import load_model, save_model
from lib.model import top_labeled_predictions, top_predictions_table
from lib.plot import plot_training_stats
```

# Configuration


```python
MODEL_BASE_DIR = pathlib.Path('../model')
CACHE_DIR = pathlib.Path('../tensorflow_cache')

PREPROC_BATCH_SIZE = 10_000  # some large value, only affects execution time

# splits are handled by `tfds.load`, see doc for more elaborate ways to sample
TRAIN_SPLIT = 'train[0:80%]'
VAL_SPLIT = 'train[80%:90%]'
TEST_SPLIT = 'train[90%:]'
MAX_EPOCHS = 50
```

# Prepare dataset

Run this once to fetch, build and cache the dataset.
Further runs will be no-ops, unless you force operations (see TFDS doc).

Once this is done, `load_dataset('off_categories', ...)` to access the dataset.


```python
import datasets.off_categories

builder = tfds.builder('off_categories')
builder.download_and_prepare()

# Or run via command line (if `tfds` is in the path):
# !cd ../datasets && tfds build off_categories
```

# Build model


```python
tf.random.set_seed(42)
```

# Taxonomy information


```python
import json
from lib.taxonomy import Taxonomy
! ls category_taxonomy.json || wget https://github.com/openfoodfacts/robotoff-models/releases/download/keras-category-classifier-xx-2.0/category_taxonomy.json

taxo = Taxonomy.from_data(json.load(open('category_taxonomy.json')))
```

## Model inputs


```python
# we use dicts so rerunning individual model cells is idempotent
inputs = {}
input_graphs = {}
```


```python
ds = load_dataset('off_categories', split=TRAIN_SPLIT)
```


```python
%%time

feature_name = 'product_name'

product_name_input = tf.keras.Input(shape=(1,), dtype=tf.string, name=feature_name)

product_name_vectorizer = layers.TextVectorization(
    split = 'whitespace',
    max_tokens = 93_000,
    output_sequence_length = 30)

product_name_vectorizer.adapt(
    select_feature(ds, feature_name).batch(PREPROC_BATCH_SIZE))

x = product_name_vectorizer(product_name_input)

x = layers.Embedding(
    input_dim = product_name_vectorizer.vocabulary_size(),
    output_dim = 64,
    mask_zero = False)(x)

product_name_graph = layers.Bidirectional(layers.LSTM(
    units = 64,
    recurrent_dropout = 0.2,
    dropout = 0.0))(x)

inputs[feature_name] = product_name_input
input_graphs[feature_name] = product_name_graph

len(product_name_vectorizer.get_vocabulary())
```


```python
%%time

feature_name = 'ingredients_tags'

ingredients_input = tf.keras.Input(shape=(None,), dtype=tf.string, name=feature_name)

ingredients_vocab = get_vocabulary(
    flat_batch(select_feature(ds, feature_name), batch_size=PREPROC_BATCH_SIZE),
    min_freq = 3,
    max_tokens = 5_000)

ingredients_graph = layers.StringLookup(
    vocabulary = ingredients_vocab,
    output_mode = 'multi_hot')(ingredients_input)

inputs[feature_name] = ingredients_input
input_graphs[feature_name] = ingredients_graph

len(ingredients_vocab)
```

## Model output


```python
%%time

labels = 'categories_tags'

categories_vocab = get_vocabulary(
    flat_batch(select_feature(ds, labels), batch_size=PREPROC_BATCH_SIZE),
    min_freq = 10)

# StringLookup(output_mode='multi_hot') mode requires num_oov_indices >= 1.
# We don't want OOVs in the categories_tags output layer, since it wouldn't make
# sense to predict OOV. So we'll drop the OOV in _transform below.
# Be careful when using StringLookup methods, some of them will return values
# based on a vocabulary with OOV (e.g. vocabulary_size()). Keep this in mind when
# mapping predictions back to the original vocabulary.
categories_multihot = layers.StringLookup(
    vocabulary = categories_vocab,
    output_mode = 'multi_hot',
    num_oov_indices = 1)

len(categories_vocab)
```

## Model


```python
# a specific model that do not penalize on certain categories
from lib.taxonomy_mask import MaskingModel
```


```python
# ensure final order is independent of cell execution/insertion order
features = sorted(inputs.keys())

x = layers.Concatenate()([input_graphs[k] for k in features])
x = layers.Dropout(0.2)(x)
x = layers.Dense(64)(x)
x = layers.Dropout(0.2)(x)
x = layers.Activation('relu')(x)
output = layers.Dense(len(categories_vocab), activation='sigmoid')(x)

model = MaskingModel(inputs=[inputs[k] for k in features], outputs=[output])

threshold = 0.5
num_labels = len(categories_vocab)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
    metrics=[
        tf.metrics.Precision(thresholds=threshold, name='precision'),
        tf.metrics.Recall(thresholds=threshold, name='recall'),
        tfa.metrics.F1Score(average='micro', threshold=threshold, num_classes=num_labels, name='f1_score_micro'),
        tfa.metrics.F1Score(average='macro', threshold=threshold, num_classes=num_labels, name='f1_score_macro'),
    ]
)
```


```python
model.summary()
```


```python
plot_model(model, show_shapes=True, show_layer_names=True)
```

# Train model


```python
# helpers to add features and encode
from lib.taxonomy_mask import TaxonomyTransformer, binarize_compat 

add_compatible_categories = TaxonomyTransformer(taxo).add_compatible_categories
```


```python
def categories_encode(ds: tf.data.Dataset):
    """encode categories

    - as multi-hot for y
    - as a mask for "compat" feature
    """
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _transform(x, y):
        y = categories_multihot(y)
        y = y[1:]  # drop OOV
        # we also binarize compatibility feature
        x = binarize_compat(x, categories_multihot, "compat")
        return (x, y)

    # applies to non-batched dataset
    return (
        ds
        .map(_transform, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .apply(filter_empty_labels)
    )
```


```python
# Remember to clean obsolete dirs once in a while
MODEL_DIR = init_model_dir(MODEL_BASE_DIR)
CACHE_DIR = init_cache_dir(CACHE_DIR)

batch_size = 128

ds_train = (
    load_dataset('off_categories', split=TRAIN_SPLIT, features=features, as_supervised=True)
    .apply(add_compatible_categories)
    .apply(categories_encode)
    .padded_batch(batch_size)
    .cache(str(CACHE_DIR / 'train'))
)

ds_val = (
    load_dataset('off_categories', split=VAL_SPLIT, features=features, as_supervised=True)
    .apply(add_compatible_categories)
    .apply(categories_encode)
    .padded_batch(batch_size)
    .cache(str(CACHE_DIR / 'val'))
)
```


```python
%%time

history = model.fit(
    ds_train,
    epochs = MAX_EPOCHS,
    validation_data = ds_val,
    callbacks = [
        callbacks.TerminateOnNaN(),
        callbacks.ModelCheckpoint(
            filepath = str(MODEL_DIR / "weights.{epoch:02d}-{val_loss:.4f}"),
            monitor = 'val_loss',
            save_best_only = True,
            save_format = 'tf',
        ),
        # callbacks.EarlyStopping(monitor='f1_score_macro', patience=4),
        callbacks.CSVLogger(str(MODEL_DIR / 'training.log')),
        callbacks.History()
    ]
)
```

## Training stats


```python
stats = pd.read_csv(MODEL_DIR / 'training.log')
stats
```


```python
%matplotlib inline
```


```python
plot_training_stats(stats)
```

## Save model and resources


```python
SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'

@tf.function
def serving_func(*args, **kwargs):
    preds = model(*args, **kwargs)
    return top_labeled_predictions(preds, categories_vocab, k=50)

save_model(SAVED_MODEL_DIR, model, categories_vocab, serving_func)
```

# Test model


```python
m, labels = load_model(SAVED_MODEL_DIR)
```


```python
ds_test = load_dataset('off_categories', split=TEST_SPLIT)
```

## Predict with serving model


```python
%%time

preds_test = m.predict(ds_test.padded_batch(128))
preds_test
```


```python
# This is the function exported as the default serving function in our saved model
top_preds_test = top_labeled_predictions(preds_test, labels, k=7)
top_preds_test
```


```python
%%time

# Same data, but pretty
pred_table_test = top_predictions_table(top_preds_test)

# Add some interpretable features to the final table
# Table must be row-aligned with predictions above (= taken from same data sample)
extra_cols_test = as_dataframe(select_features(ds_test, ['code', 'product_name']))

pd.concat([extra_cols_test, pred_table_test], axis=1)
```


```python
# codecarbon - stop tracking
tracker.stop()
```


```python

```
