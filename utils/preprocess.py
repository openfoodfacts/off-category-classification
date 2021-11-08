from collections import defaultdict
import operator
import re
from typing import Dict, Iterable
from typing import Set

import pandas as pd
from sklearn.feature_extraction.text import strip_accents_ascii
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

def generate_y(categories_tags: Iterable[Iterable[str]], category_to_id: Dict):
    category_count = len(category_to_id)
    cat_binarizer = MultiLabelBinarizer(classes=list(range(category_count)))
    category_int = [
        [category_to_id[cat] for cat in product_categories if cat in category_to_id]
        for product_categories in categories_tags
    ]
    return cat_binarizer.fit_transform(category_int)
