import re
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

from robotoff.utils import gzip_jsonl_iter
from robotoff.taxonomy import Taxonomy

import settings
from utils.preprocess import generate_y
from utils.preprocess import preprocess_product_name
from tf.data_utils import process_ingredients


def process_df(
    df: pd.DataFrame,
    category_to_id: Dict,
    ingredient_to_id: Dict,
    vectorizer: CountVectorizer,
):
    y = generate_y(df.categories_tags, category_to_id)
    X = generate_X(df, ingredient_to_id, vectorizer)
    return X, y


def generate_X(df: pd.DataFrame, ingredient_to_id: Dict, vectorizer: CountVectorizer):
    product_name_matrix = vectorizer.transform(df.product_name)
    ingredient_matrix = process_ingredients(df.known_ingredient_tags, ingredient_to_id)
    return np.concatenate((product_name_matrix.toarray(), ingredient_matrix), axis=1)


category_taxonomy = Taxonomy.from_json(settings.CATEGORY_TAXONOMY_PATH)
ingredient_taxonomy = Taxonomy.from_json(settings.INGREDIENTS_TAXONOMY_PATH)

CATEGORY_NAMES = sorted(category_taxonomy.keys())
INGREDIENT_NAMES = sorted(ingredient_taxonomy.keys())

CATEGORY_TO_ID = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
INGREDIENT_TO_ID = {name: idx for idx, name in enumerate(INGREDIENT_NAMES)}

train_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_TRAIN_PATH)).head(1000)
test_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_TEST_PATH)).head(100)
val_df = pd.DataFrame(gzip_jsonl_iter(settings.CATEGORY_FR_VAL_PATH)).head(100)

count_vectorizer = CountVectorizer(min_df=5, preprocessor=preprocess_product_name)
count_vectorizer.fit(train_df.product_name)

X_train, y_train = process_df(
    train_df, CATEGORY_TO_ID, INGREDIENT_TO_ID, count_vectorizer
)
X_test, y_test = process_df(test_df, CATEGORY_TO_ID, INGREDIENT_TO_ID, count_vectorizer)
X_val, y_val = process_df(val_df, CATEGORY_TO_ID, INGREDIENT_TO_ID, count_vectorizer)

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred, target_names=CATEGORY_NAMES))
accuracy = accuracy_score(y_val, y_pred)

print(
    classification_report(
        y_val, y_pred, target_names=CATEGORY_NAMES, labels=[CATEGORY_TO_ID["en:snacks"]]
    )
)

