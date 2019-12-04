from collections import defaultdict
import operator
import re
from typing import Dict, Iterable
from typing import Set

import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
from sklearn.preprocessing import MultiLabelBinarizer
from spacy.lang.en import English
from spacy.lang.fr import French

from utils.constant import PAD_TOKEN, UNK_TOKEN

PUNCTUATION_REGEX = re.compile(r"""[:,;.&~"'|`_\\={}%()\[\]]+""")
DIGIT_REGEX = re.compile(r"[0-9]+")
MULTIPLE_SPACES_REGEX = re.compile(r" +")


def count_categories(df: pd.DataFrame) -> Dict:
    categories_count = defaultdict(int)

    for categories in df.categories_tags:
        for category in categories:
            categories_count[category] += 1

    return categories_count


def count_ingredients(df: pd.DataFrame) -> Dict:
    ingredients_count = defaultdict(int)

    for ingredients in df.known_ingredient_tags:
        for ingredient in ingredients:
            ingredients_count[ingredient] += 1

    return ingredients_count


def get_categories(df: pd.DataFrame) -> Set[str]:
    categories_all = set()

    for categories in df.categories_tags:
        for category in categories:
            categories_all.add(category)

    return categories_all


def get_ingredients(df: pd.DataFrame) -> Set[str]:
    ingredients_all = set()

    for ingredients in df.known_ingredient_tags:
        for ingredient in ingredients:
            ingredients_all.add(ingredient)

    return ingredients_all


def preprocess_product_name(text: str,
                            lower: bool,
                            strip_accent: bool,
                            remove_punct: bool,
                            remove_digit: bool) -> str:
    if strip_accent:
        text = strip_accents_ascii(text)

    if lower:
        text = text.lower()

    if remove_punct:
        text = PUNCTUATION_REGEX.sub(' ', text)

    if remove_digit:
        text = DIGIT_REGEX.sub(' ', text)

    return MULTIPLE_SPACES_REGEX.sub(' ', text)


def tokenize_batch(texts: Iterable[str], nlp) -> Iterable[Iterable[str]]:
    for doc in nlp.pipe(texts):
        yield [t.orth_ for t in doc]


def get_nlp(lang: str):
    lang = 'en' if lang == 'xx' else lang

    if lang == 'fr':
        return French()
    elif lang == 'en':
        return English()
    else:
        raise ValueError("unknown lang: {}".format(lang))


def tokenize(text: str, nlp):
    return [token.orth_ for token in nlp(text)]


def extract_vocabulary(tokens: Iterable[Iterable[str]],
                       min_count: int = 0) -> Dict[str, int]:
    vocabulary = defaultdict(int)

    for doc_tokens in tokens:
        for token in doc_tokens:
            vocabulary[token] += 1

    to_delete: Set[str] = set()
    for token, count in vocabulary.items():
        if count < min_count:
            to_delete.add(token)

    for token in to_delete:
        vocabulary.pop(token)

    token_to_int: Dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    offset = 2
    for token, _ in sorted(vocabulary.items(),
                           key=operator.itemgetter(1),
                           reverse=True):
        token_to_int[token] = offset
        offset += 1

    return token_to_int


def generate_y(categories_tags: Iterable[Iterable[str]], category_to_id: Dict):
    category_count = len(category_to_id)
    cat_binarizer = MultiLabelBinarizer(classes=list(range(category_count)))
    category_int = [[category_to_id[cat]
                     for cat in product_categories
                     if cat in category_to_id]
                    for product_categories in categories_tags]
    return cat_binarizer.fit_transform(category_int)
