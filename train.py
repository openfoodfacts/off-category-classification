import argparse
import functools
import json
import pathlib

import dacite
import pandas as pd
from robotoff.taxonomy import Taxonomy
from robotoff.utils import gzip_jsonl_iter
from tensorflow.keras import callbacks
from tensorflow import keras

import settings
from category_classification.data_utils import generate_data_from_df
from category_classification.models import build_model, Config
from utils.io import save_product_name_vocabulary, save_config, save_category_vocabulary, save_ingredient_vocabulary, \
    save_json
from utils.metrics import evaluation_report
from utils.preprocess import count_categories, count_ingredients, get_nlp, tokenize_batch, extract_vocabulary, \
    preprocess_product_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)
    return parser.parse_args()


args = parse_args()

with args.config.open('r') as f:
    config_dict = json.load(f)
    config = dacite.from_dict(Config, config_dict)

model_config = config.model_config

save_dir = args.output_dir / config.train_config.save_dirname
save_dir.mkdir(parents=True, exist_ok=True)

category_taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)
ingredient_taxonomy = Taxonomy.from_json(settings.INGREDIENTS_TAXONOMY_PATH)

train_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_TRAIN_PATH))
test_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_TEST_PATH))
val_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_VAL_PATH))

categories_count = count_categories(train_df)
ingredients_count = count_ingredients(train_df)

selected_categories = set((cat
                           for cat, count in categories_count.items()
                           if count >= config.category_min_count))
selected_ingredients = set((ingredient
                            for ingredient, count in ingredients_count.items()
                            if count >= config.ingredient_min_count))
print("{} categories selected".format(len(selected_categories)))
print("{} ingredients selected".format(len(selected_ingredients)))

category_names = [x for x in sorted(category_taxonomy.keys())
                  if x in selected_categories]

ingredient_names = [x for x in sorted(ingredient_taxonomy.keys())
                    if x in selected_ingredients]

category_to_id = {name: idx for idx, name in enumerate(category_names)}
ingredient_to_id = {name: idx for idx, name in enumerate(ingredient_names)}

save_category_vocabulary(category_to_id, save_dir)
save_ingredient_vocabulary(ingredient_to_id, save_dir)


nlp = get_nlp(lang=config.lang)

preprocess_product_name_func = functools.partial(preprocess_product_name,
                                                 lower=config.product_name_preprocessing_config.lower,
                                                 strip_accent=config.product_name_preprocessing_config.strip_accent,
                                                 remove_punct=config.product_name_preprocessing_config.remove_punct,
                                                 remove_digit=config.product_name_preprocessing_config.remove_digit)
preprocessed_product_names_iter = (preprocess_product_name_func(product_name)
                                   for product_name in train_df.product_name)
train_tokens_iter = tokenize_batch(preprocessed_product_names_iter, nlp)
product_name_to_int = extract_vocabulary(train_tokens_iter,
                                         config.product_name_min_count)

model_config.ingredient_voc_size = len(ingredient_to_id)
model_config.output_dim = len(category_to_id)
model_config.product_name_voc_size = len(product_name_to_int)

save_product_name_vocabulary(product_name_to_int,
                             save_dir)
save_config(config, save_dir)

print("Selected vocabulary: {}".format(len(product_name_to_int)))

model = build_model(model_config)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy', 'Precision', 'Recall'])


generate_data_partial = functools.partial(generate_data_from_df,
                                          ingredient_to_id=ingredient_to_id,
                                          category_to_id=category_to_id,
                                          product_name_max_length=model_config.product_name_max_length,
                                          product_name_token_to_int=product_name_to_int,
                                          nlp=nlp,
                                          product_name_preprocessing_config=config.product_name_preprocessing_config)

X_train, y_train = generate_data_partial(train_df)
X_val, y_val = generate_data_partial(val_df)

model.fit(X_train, y_train,
          batch_size=config.train_config.batch_size,
          epochs=config.train_config.epochs,
          validation_data=(X_val, y_val),
          callbacks=[
              callbacks.TerminateOnNaN(),
              callbacks.ModelCheckpoint(
                  filepath=str(save_dir / "weights.{epoch:02d}-{val_loss:.4f}.hdf5"),
                  monitor='val_loss',
                  save_best_only=True,
              ),
              callbacks.TensorBoard(
                  log_dir=str(save_dir / 'logs'),
                  histogram_freq=2,
                  write_grads=True,
              ),
              callbacks.EarlyStopping(
                  monitor='val_loss',
                  patience=4,
              ),
          ])

model.save(str(save_dir / 'last_checkpoint.hdf5'))

last_checkpoint_path = sorted(save_dir.glob('weights.*.hdf5'))[-1]

print("Restoring last checkpoint {}".format(last_checkpoint_path))
model = keras.models.load_model(str(last_checkpoint_path))

y_pred_val = model.predict(X_val)
report, clf_report = evaluation_report(y_val, y_pred_val,
                                       taxonomy=category_taxonomy,
                                       category_names=category_names)

save_json(report, save_dir / 'metrics_val.json')
save_json(clf_report, save_dir / 'classification_report_val.json')


X_test, y_test = generate_data_partial(test_df)

y_pred_test = model.predict(X_test)
report, clf_report = evaluation_report(y_test, y_pred_test,
                                       taxonomy=category_taxonomy,
                                       category_names=category_names)

save_json(report, save_dir / 'metrics_test.json')
save_json(clf_report, save_dir / 'classification_report_test.json')
