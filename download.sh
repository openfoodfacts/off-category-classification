#!/bin/bash

mkdir -p data
cd data
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-category-2019-09-16/ingredients.full.json
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-category-2019-09-16/categories.full.json
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-category-2019-09-16/category_fr.test.jsonl.gz
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-category-2019-09-16/category_fr.train.jsonl.gz
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-category-2019-09-16/category_fr.val.jsonl.gz
