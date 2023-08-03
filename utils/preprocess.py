# rescued from
# https://github.com/raphael0202/off-category-classification/blob/ea24b763c465a7b8e281a7aad7cb0ee9c14303f4/utils/preprocess.py

import pandas as pd
from collections import defaultdict
from typing import Dict

def count_categories(df: pd.DataFrame) -> Dict:
    categories_count = defaultdict(int)

    for categories in df.categories_tags:
        for category in categories:
            categories_count[category] += 1

    return categories_count
